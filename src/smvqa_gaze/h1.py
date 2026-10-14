from __future__ import annotations

import csv
import gc
import json
import math
import os
import random
import re
import time
import warnings
from bisect import bisect_left
from pathlib import Path
from typing import Any

from .qa import read_jsonl, write_jsonl
from .projection import export_rgb_calibration


GAZE_LABEL_PROMPT = (
    "Name the single object or piece of text at the center of this image in at "
    "most 6 words. If nothing specific, answer 'none'."
)
MAX_TARGET_AREA_FRACTION = 0.25
PROJECTION_CHECK_PROMPT = (
    "Is the red dot on a meaningful object the person is likely interacting with? "
    "Answer yes or no."
)


def require_video_time_fixations(fixations: Any, path: Path) -> None:
    """Reject fixation tables written before the shared video time axis."""
    required = {"fix_id", "t_on", "t_off", "t_on_us", "t_off_us", "dur_ms", "u", "v"}
    missing = required.difference(fixations.columns)
    if missing:
        raise ValueError(
            f"{path} is missing {sorted(missing)}; regenerate it with "
            "`smvqa-gaze prepare-selected-sessions` so t_on/t_off use video time"
        )


def crop_fixation_frame(
    frame: Any, u: float, v: float, size: int = 256
) -> tuple[Any, list[float]]:
    import cv2

    height, width = frame.shape[:2]
    if not math.isfinite(u) or not math.isfinite(v):
        raise ValueError("Fixation coordinates must be finite")
    if not 0 <= u < width or not 0 <= v < height:
        raise ValueError(f"Fixation point ({u}, {v}) is outside {width}x{height}")
    center_x, center_y = round(u), round(v)
    before = size // 2
    after = size - before
    padded = cv2.copyMakeBorder(
        frame,
        before,
        after,
        before,
        after,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    left, top = center_x, center_y
    crop = padded[top : top + size, left : left + size]
    if crop.shape[:2] != (size, size):
        raise RuntimeError("Could not extract a full fixation-centered crop")
    return crop, [u - before, v - before, u + after, v + after]


class Qwen3VLLabeler:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        cuda_visible_devices: str = "1",
    ) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Install the local GPU extra to run H1 Qwen annotations"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "H1 Qwen annotations require CUDA; verify the local PyTorch build"
            )
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < 18 * 1024**3:
            raise RuntimeError(
                "The selected GPU needs at least 18 GiB free for Qwen3-VL-8B; "
                f"only {free_bytes / 1024**3:.1f} GiB is free"
            )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.model.eval()

    def label_image(self, image: Any, prompt: str = GAZE_LABEL_PROMPT) -> str:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError("Install qwen-vl-utils to run Qwen3-VL") from exc
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        import torch

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
            )
        answer = self.processor.batch_decode(
            generated[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        label = " ".join(answer.strip().split())
        if not label:
            raise RuntimeError("Qwen3-VL returned an empty fixation label")
        return label

    def generate_text(self, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        import torch

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )
        answer = self.processor.batch_decode(
            generated[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        answer = " ".join(answer.strip().split())
        if not answer:
            raise RuntimeError("Qwen3-VL returned an empty text response")
        return answer


def parse_projection_judgment(answer: str) -> str:
    judgments = re.findall(r"\b(yes|no)\b", answer.lower())
    if not judgments or len(set(judgments)) != 1:
        raise ValueError(f"Expected an unambiguous yes/no response, got {answer!r}")
    return judgments[0]


def sample_unique_projection_frames(
    rows: list[dict[str, Any]], sample_count: int, seed: int
) -> list[dict[str, Any]]:
    unique: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        unique.setdefault((str(row["video_id"]), int(row["frame_index"])), row)
    candidates = list(unique.values())
    return random.Random(seed).sample(candidates, min(sample_count, len(candidates)))


def classify_fixation_state(
    target_detected: bool, fixated_dwell_ms: float, glance_dwell_ms: float
) -> str:
    if fixated_dwell_ms < 0 or glance_dwell_ms < 0:
        raise ValueError("Gaze dwell durations must be non-negative")
    if not target_detected:
        return "undetected"
    if fixated_dwell_ms >= 200:
        return "fixated"
    if glance_dwell_ms > 0:
        return "glanced"
    return "never"


VISUAL_MODALITIES = {"video", "ocr", "gaze", "depth", "visual", "image"}
FE_SCHEMA_VERSION = 2


def _is_visual_evidence(span: dict[str, Any]) -> bool:
    """Released modalities are Video/Audio/OCR/Trajectory/Gaze/Depth; OCR, gaze
    and depth evidence is visual (text or objects the user saw)."""
    modalities = span.get("modalities") or []
    if isinstance(modalities, str):
        modalities = [modalities]
    return any(str(modality).casefold() in VISUAL_MODALITIES for modality in modalities)


def evidence_status(row: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Pre-localization F(E) category of a question.

    - ``unanswerable``: no correct answer to look for (separate from NA)
    - ``NA``: answerable but no visual evidence (e.g. audio-only conversation)
    - ``visual``: needs target extraction and localization
    """
    visual = [span for span in row.get("answer_evidence", []) if _is_visual_evidence(span)]
    if row.get("is_answerable") is False:
        return "unanswerable", visual
    if not visual:
        return "NA", visual
    return "visual", visual


def extract_qa_visual_targets(
    qa_path: Path,
    output_path: Path,
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    cuda_visible_devices: str = "1",
    checkpoint_every: int = 25,
) -> dict[str, Any]:
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1")
    rows = read_jsonl(qa_path)
    previous = {
        str(row["question_id"]): row
        for row in (read_jsonl(output_path) if output_path.is_file() else [])
    }
    completed: dict[str, dict[str, Any]] = {}
    pending = []
    for row in rows:
        question_id = str(row["question_id"])
        category, visual_evidence = evidence_status(row)
        old = previous.get(question_id)
        reusable_targets = (
            old is not None
            and old.get("status") in {"pending_localization", "no_target"}
            and "targets" in old
        )
        if category != "visual":
            completed[question_id] = {
                "question_id": question_id,
                "category_key": row["category_key"],
                "primary_video_id": row["primary_video_id"],
                "targets": [],
                "status": category,
                "evidence": visual_evidence,
                "schema": FE_SCHEMA_VERSION,
            }
        elif reusable_targets:
            # Greedy decoding is deterministic: earlier targets are reused.
            completed[question_id] = {
                **old,
                "status": "pending_localization" if old["targets"] else "no_target",
                "evidence": visual_evidence,
                "schema": FE_SCHEMA_VERSION,
            }
        else:
            pending.append((row, visual_evidence))
    labeler = Qwen3VLLabeler(model_id, cuda_visible_devices) if pending else None
    for index, (row, visual_evidence) in enumerate(pending, start=1):
        assert labeler is not None
        targets = extract_visual_targets(
            str(row["question"]), str(row.get("correct_answer") or ""), labeler
        )
        completed[str(row["question_id"])] = {
            "question_id": str(row["question_id"]),
            "category_key": row["category_key"],
            "primary_video_id": row["primary_video_id"],
            "targets": targets,
            "status": "pending_localization" if targets else "no_target",
            "evidence": visual_evidence,
            "schema": FE_SCHEMA_VERSION,
        }
        if index % checkpoint_every == 0:
            _write_jsonl_atomic(
                [completed[str(item["question_id"])] for item in rows if str(item["question_id"]) in completed],
                output_path,
            )
    output_rows = [completed[str(row["question_id"])] for row in rows]
    _write_jsonl_atomic(output_rows, output_path)
    return {
        "questions": len(output_rows),
        "newly_extracted": len(pending),
        "status_counts": {
            status: sum(row["status"] == status for row in output_rows)
            for status in sorted({row["status"] for row in output_rows})
        },
        "output": str(output_path),
        "model": model_id,
    }


def check_projection_sessions(
    sessions_path: Path,
    data_root: Path,
    output_dir: Path,
    report_path: Path,
    sample_count: int = 50,
    seed: int = 2026,
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    cuda_visible_devices: str = "1",
) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    try:
        import cv2
        import numpy as np
        import pandas as pd
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install the local and local-gpu extras for H1.2") from exc
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")
    sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    candidates: dict[int, list[dict[str, Any]]] = {}
    for session in sessions:
        participant = int(session["participant"])
        video_id = str(session["video_id"])
        gaze_path = (
            data_root
            / "proc/gaze"
            / video_id
            / "gaze_px.parquet"
        )
        if not gaze_path.is_file():
            continue
        projected = pd.read_parquet(gaze_path)
        valid = projected[
            projected["valid"].astype(bool)
            & np.isfinite(projected["u"])
            & np.isfinite(projected["v"])
            & projected["u"].between(0, 1408, inclusive="left")
            & projected["v"].between(0, 1408, inclusive="left")
        ]
        video_path = (
            data_root / "raw/video" / f"Person_{participant}" / f"{video_id}.mp4"
        )
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing video for projection QC: {video_path}")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video for projection QC: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
            raise RuntimeError(f"Invalid video frame rate for projection QC: {video_path}")
        first_t_us = int(projected.iloc[0]["t_us"])
        for row in valid.to_dict("records"):
            frame_index = round((int(row["t_us"]) - first_t_us) / 1_000_000 * fps)
            if not 0 <= frame_index < frame_count:
                continue
            candidates.setdefault(participant, []).append(
                {
                    "video_id": video_id,
                    "video_path": video_path,
                    "participant": participant,
                    "t_us": int(row["t_us"]),
                    "u": float(row["u"]),
                    "v": float(row["v"]),
                    "frame_index": frame_index,
                    "fps": fps,
                }
            )
    chosen = []
    for participant, rows in sorted(candidates.items()):
        chosen.extend(
            sample_unique_projection_frames(
                rows, sample_count, seed + participant
            )
        )
    if not chosen:
        raise ValueError("No valid projected gaze samples are available for QC")

    labeler = Qwen3VLLabeler(model_id, cuda_visible_devices)
    results = []
    for sample_index, sample in enumerate(chosen):
        capture = cv2.VideoCapture(str(sample["video_path"]))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video {sample['video_path']}")
        fps = sample["fps"]
        frame_index = sample["frame_index"]
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(
                f"Could not decode projection QC frame {frame_index} "
                f"from {sample['video_path']}"
            )
        point = (
            min(1407, max(0, round(sample["u"]))),
            min(1407, max(0, round(sample["v"]))),
        )
        cv2.circle(frame, point, 12, (0, 0, 255), thickness=-1)
        cv2.circle(frame, point, 20, (255, 255, 255), thickness=2)
        answer = labeler.label_image(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
            PROJECTION_CHECK_PROMPT,
        )
        judgment = parse_projection_judgment(answer)
        participant = sample["participant"]
        image_path = (
            output_dir
            / f"person_{participant}"
            / f"{sample['video_id']}_frame_{frame_index}.jpg"
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError(f"Could not save projection QC image {image_path}")
        results.append(
            {
                "participant": participant,
                "video_id": sample["video_id"],
                "frame_index": frame_index,
                "video_time_s": frame_index / fps,
                "u": sample["u"],
                "v": sample["v"],
                "answer": answer,
                "judgment": judgment,
                "image": str(image_path),
                "sample_index": sample_index,
            }
        )
    report = {
        "samples": len(results),
        "manual_visual_review": "pending",
        "participants": {
            str(participant): {
                "samples": sum(row["participant"] == participant for row in results),
                "yes_fraction": sum(
                    row["participant"] == participant and row["judgment"] == "yes"
                    for row in results
                )
                / sum(row["participant"] == participant for row in results),
            }
            for participant in sorted(candidates)
        },
        "model": model_id,
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _decode_visual_target_response(response: str) -> list[str]:
    start, end = response.find("["), response.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"Qwen3-VL did not return a JSON array: {response!r}")
    try:
        targets = json.loads(response[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON target list from Qwen3-VL: {response!r}") from exc
    if (
        not isinstance(targets, list)
        or any(not isinstance(target, str) or not target.strip() for target in targets)
    ):
        raise ValueError(f"Invalid visual target list from Qwen3-VL: {targets!r}")
    normalized = []
    for target in targets:
        value = " ".join(target.split())
        if value.casefold() not in {item.casefold() for item in normalized}:
            normalized.append(value)
    return normalized


def _parse_visual_target_response(response: str) -> list[str]:
    targets = _decode_visual_target_response(response)
    if len(targets) > 3:
        raise ValueError(
            f"Qwen3-VL returned {len(targets)} visual targets; expected at most 3"
        )
    return targets


def extract_visual_targets(
    question: str, correct_answer: str, labeler: Qwen3VLLabeler
) -> list[str]:
    prompt = (
        "Extract 1 to 3 concrete visual object or text target phrases needed to "
        "verify the correct answer. Return only a JSON array of short strings. "
        "Do not include audio-only facts, explanations, or full sentences.\n"
        f"Question: {question}\nCorrect answer: {correct_answer}"
    )
    response = labeler.generate_text(prompt)
    try:
        return _parse_visual_target_response(response)
    except ValueError as initial_error:
        retry_prompt = (
            "Return only a valid JSON array containing at most 3 short visual "
            "object or text phrases. Each array item must be a phrase, not one "
            "word from a phrase. Use [] if no concrete visual target exists. "
            'Example: ["car keys", "recipe on tablet"].\n'
            f"Question: {question}\nCorrect answer: {correct_answer}"
        )
        retry_response = labeler.generate_text(retry_prompt)
        try:
            return _parse_visual_target_response(retry_response)
        except ValueError as retry_error:
            try:
                retry_targets = _decode_visual_target_response(retry_response)
            except ValueError:
                raise ValueError(
                    "Qwen3-VL returned invalid visual targets after one constrained "
                    f"retry; initial error: {initial_error}; retry response: "
                    f"{retry_response!r}"
                ) from retry_error
            if len(retry_targets) > 3:
                warnings.warn(
                    "Qwen3-VL still returned more than 3 targets after a constrained "
                    "retry; retaining the first 3 in model rank order.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return retry_targets[:3]
            raise ValueError(
                "Qwen3-VL returned invalid visual targets after one constrained "
                f"retry; initial error: {initial_error}; retry response: "
                f"{retry_response!r}"
            ) from retry_error


def expand_detection_box(
    box: list[float],
    focal_lengths_px: list[float],
    image_size: tuple[int, int] = (1408, 1408),
    padding_degrees: float = 2.0,
) -> list[float]:
    if len(box) != 4 or len(focal_lengths_px) != 2:
        raise ValueError("A box needs four coordinates and two focal lengths")
    if padding_degrees < 0 or min(focal_lengths_px) <= 0:
        raise ValueError("Padding must be non-negative and focal lengths positive")
    width, height = image_size
    pad_x = focal_lengths_px[0] * math.tan(math.radians(padding_degrees))
    pad_y = focal_lengths_px[1] * math.tan(math.radians(padding_degrees))
    return [
        max(0.0, float(box[0]) - pad_x),
        max(0.0, float(box[1]) - pad_y),
        min(float(width), float(box[2]) + pad_x),
        min(float(height), float(box[3]) + pad_y),
    ]


class GroundingDinoLocalizer:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        cuda_visible_devices: str = "1",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Install the local-gpu extra to run GroundingDINO localization"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("GroundingDINO localization requires CUDA")
        if not 0 < box_threshold <= 1 or not 0 < text_threshold <= 1:
            raise ValueError("Detection thresholds must be in (0, 1]")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to("cuda:0")
        self.model.eval()
        self.device = "cuda:0"
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model_id = model_id

    def detect(self, image: Any, targets: list[str]) -> list[dict[str, Any]]:
        import torch

        prompt = " . ".join(target.strip().rstrip(".") for target in targets) + " ."
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device
        )
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.model.dtype)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        labels = result.get("text_labels", result.get("labels", []))
        boxes = result["boxes"].detach().cpu().tolist()
        scores = result["scores"].detach().cpu().tolist()
        if len(scores) != len(boxes):
            raise RuntimeError(
                "GroundingDINO returned mismatched scored boxes: "
                f"{len(scores)} scores for {len(boxes)} boxes"
            )
        if len(labels) != len(boxes):
            warnings.warn(
                "GroundingDINO returned "
                f"{len(labels)} labels for {len(boxes)} scored boxes; "
                "unmatched labels are omitted.",
                RuntimeWarning,
                stacklevel=2,
            )
        return [
            {
                "label": str(labels[index]) if index < len(labels) else None,
                "score": float(score),
                "box": [float(value) for value in box],
            }
            for index, (score, box) in enumerate(zip(scores, boxes, strict=True))
        ]


def _nearest_frame_boxes(
    timestamp_s: float,
    frame_times: list[float],
    frame_boxes: list[list[list[float]]],
    tolerance_s: float = 0.251,
) -> list[list[float]]:
    index = bisect_left(frame_times, timestamp_s)
    candidates = [
        candidate
        for candidate in (index - 1, index)
        if 0 <= candidate < len(frame_times)
    ]
    if not candidates:
        return []
    closest = min(candidates, key=lambda candidate: abs(frame_times[candidate] - timestamp_s))
    if abs(frame_times[closest] - timestamp_s) > tolerance_s:
        return []
    return frame_boxes[closest]


def _point_in_any_box(u: float, v: float, boxes: list[list[float]]) -> bool:
    return any(x1 <= u <= x2 and y1 <= v <= y2 for x1, y1, x2, y2 in boxes)


def _match_target(label: Any, targets: list[str]) -> str | None:
    """GroundingDINO labels are spans of the prompt; map one back to a target."""
    text = " ".join(str(label or "").casefold().split())
    if not text:
        return None
    for target in targets:
        candidate = " ".join(target.casefold().split())
        if text == candidate or text in candidate or candidate in text:
            return target
    words = set(text.split())
    overlaps = [(len(words & set(t.casefold().split())), t) for t in targets]
    best = max(overlaps, default=(0, None))
    return best[1] if best[0] > 0 else None


def filter_target_detections(
    detections: list[dict[str, Any]],
    targets: list[str],
    image_size: tuple[int, int] = (1408, 1408),
    max_area_fraction: float = MAX_TARGET_AREA_FRACTION,
) -> list[dict[str, Any]]:
    """Keep, per target, the single highest-scoring box that is not a
    scene-sized region (> max_area_fraction of the frame before expansion).

    Unfiltered GroundingDINO output often contains frame-sized boxes
    ("table", "floor"), which made every gaze sample count as on-target."""
    width, height = image_size
    best: dict[str, dict[str, Any]] = {}
    for detection in detections:
        target = _match_target(detection.get("label"), targets)
        if target is None:
            continue
        x1, y1, x2, y2 = detection["box"]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height)
        if area > max_area_fraction:
            continue
        if target not in best or detection["score"] > best[target]["score"]:
            best[target] = {**detection, "target": target, "area_fraction": area}
    return list(best.values())


def localize_qa_fixation_states(
    qa_path: Path,
    targets_path: Path,
    data_root: Path,
    output_path: Path,
    distribution_path: Path,
    model_id: str = "IDEA-Research/grounding-dino-tiny",
    cuda_visible_devices: str = "1",
    checkpoint_every: int = 10,
    max_area_fraction: float = MAX_TARGET_AREA_FRACTION,
) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
        import pandas as pd
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install local and local-gpu extras for H1.3") from exc
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1")

    qa_rows = read_jsonl(qa_path)
    target_rows = read_jsonl(targets_path)
    targets_by_id = {str(row["question_id"]): row for row in target_rows}
    if len(targets_by_id) != len(target_rows):
        raise ValueError("Visual target file has duplicate question IDs")
    missing_ids = {str(row["question_id"]) for row in qa_rows}.difference(targets_by_id)
    if missing_ids:
        raise ValueError(f"Visual targets missing {len(missing_ids)} QA questions")
    if any(row.get("schema") != FE_SCHEMA_VERSION for row in target_rows):
        raise ValueError("Visual targets predate the current F(E) schema; rerun extract-visual-targets")
    existing_rows = read_jsonl(output_path) if output_path.is_file() else []
    completed = {
        str(row["question_id"]): row
        for row in existing_rows
        if row.get("schema") == FE_SCHEMA_VERSION
    }
    pending = [row for row in qa_rows if str(row["question_id"]) not in completed]
    needs_localization = any(
        targets_by_id[str(row["question_id"])]["status"] == "pending_localization"
        for row in pending
    )
    localizer = (
        GroundingDinoLocalizer(model_id, cuda_visible_devices) if needs_localization else None
    )
    video_cache: dict[str, tuple[Path, float, float, Any]] = {}
    gaze_cache: dict[str, tuple[Any, Any, float]] = {}
    calibration_cache: dict[str, list[float]] = {}

    def video_info(video_id: str, participant: int) -> tuple[Path, float, float, Any]:
        if video_id not in video_cache:
            video_path = data_root / "raw/video" / f"Person_{participant}" / f"{video_id}.mp4"
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise FileNotFoundError(f"Could not open evidence video {video_path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or frame_count <= 0:
                capture.release()
                raise ValueError(f"Invalid video metadata for {video_path}")
            video_cache[video_id] = (video_path, frame_count / fps, fps, capture)
        return video_cache[video_id]

    def gaze_info(video_id: str) -> tuple[Any, Any, float]:
        if video_id not in gaze_cache:
            directory = data_root / "proc/gaze" / video_id
            gaze = pd.read_parquet(directory / "gaze_px.parquet")
            fixations = pd.read_parquet(directory / "fixations.parquet")
            require_video_time_fixations(fixations, directory / "fixations.parquet")
            if "t_video_s" not in gaze.columns:
                raise ValueError(f"{directory}/gaze_px.parquet lacks t_video_s; rerun prepare-selected-sessions")
            differences = np.diff(gaze["t_video_s"].to_numpy(dtype=float)) * 1000
            positive = differences[differences > 0]
            sample_period_ms = float(np.median(positive)) if len(positive) else 1000 / 60
            gaze_cache[video_id] = (gaze, fixations, sample_period_ms)
        return gaze_cache[video_id]

    def focal_lengths(video_id: str, participant: int) -> list[float]:
        if video_id not in calibration_cache:
            calibration_path = data_root / "proc/calib" / f"{video_id}.json"
            if not calibration_path.is_file():
                stem = video_id.replace(f"Person_{participant}_", f"person_{participant}_", 1)
                export_rgb_calibration(
                    data_root / "raw/mps" / f"person_{participant}" / f"{stem}_vrs/eye_gaze/general_eye_gaze.csv",
                    data_root / "raw" / f"person_{participant}" / f"{stem}.vrs",
                    calibration_path,
                )
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            calibration_cache[video_id] = calibration["focal_lengths_px"]
        return calibration_cache[video_id]

    try:
        for index, qa in enumerate(pending, start=1):
            question_id = str(qa["question_id"])
            target_row = targets_by_id[question_id]
            status = str(target_row["status"])
            targets = list(target_row.get("targets", []))
            span_results = []
            if status == "pending_localization":
                assert localizer is not None
                for span in target_row.get("evidence", []):
                    video_id = str(span.get("video_id") or qa["primary_video_id"])
                    start, end = span.get("t_start"), span.get("t_end")
                    if start is None or end is None or float(end) < float(start):
                        span_results.append(
                            {"video_id": video_id, "status": "undetected", "reason": "missing_or_invalid_evidence_time"}
                        )
                        continue
                    start_s, end_s = float(start), float(end)
                    video_path, video_duration, fps, capture = video_info(video_id, int(qa["participant"]))
                    focal = focal_lengths(video_id, int(qa["participant"]))
                    frame_times: list[float] = []
                    frame_boxes: list[list[list[float]]] = []
                    detections = []
                    raw_detection_count = 0
                    frame_time = max(0.0, start_s)
                    # Seek once per span, then decode forward (seeking every
                    # 0.5 s re-decodes from the previous keyframe).
                    position = round(frame_time * fps)
                    capture.set(cv2.CAP_PROP_POS_FRAMES, position)
                    while frame_time <= min(end_s, video_duration - 1e-6):
                        frame_index = round(frame_time * fps)
                        while position < frame_index:
                            if not capture.grab():
                                break
                            position += 1
                        ok, frame = capture.read()
                        position += 1
                        if not ok:
                            raise RuntimeError(f"Could not decode evidence frame {frame_index} from {video_path}")
                        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        raw = localizer.detect(image, targets)
                        raw_detection_count += len(raw)
                        kept = filter_target_detections(raw, targets, (image.width, image.height), max_area_fraction)
                        boxes = [expand_detection_box(item["box"], focal) for item in kept]
                        frame_times.append(frame_time)
                        frame_boxes.append(boxes)
                        detections.extend(
                            {**item, "box_expanded_px": expanded, "video_time_s": frame_time}
                            for item, expanded in zip(kept, boxes, strict=True)
                        )
                        frame_time += 0.5
                    gaze, fixations, sample_period_ms = gaze_info(video_id)
                    window_start = max(0.0, start_s - 2)
                    window_end = min(video_duration, end_s + 2)
                    window = gaze[
                        gaze["t_video_s"].between(window_start, window_end)
                        & gaze["valid"].astype(bool)
                        & gaze["u"].notna()
                        & gaze["v"].notna()
                    ]
                    glance_samples = sum(
                        _point_in_any_box(
                            float(sample.u),
                            float(sample.v),
                            _nearest_frame_boxes(float(sample.t_video_s), frame_times, frame_boxes),
                        )
                        for sample in window.itertuples(index=False)
                    )
                    # README H1.3: cumulative dwell of fixations inside the box.
                    fixated_dwell_ms = 0.0
                    for fixation in fixations.itertuples(index=False):
                        if pd.isna(fixation.u) or pd.isna(fixation.v):
                            continue
                        midpoint = (float(fixation.t_on) + float(fixation.t_off)) / 2
                        if not window_start <= midpoint <= window_end:
                            continue
                        boxes = _nearest_frame_boxes(midpoint, frame_times, frame_boxes)
                        if _point_in_any_box(float(fixation.u), float(fixation.v), boxes):
                            fixated_dwell_ms += float(fixation.dur_ms)
                    span_status = classify_fixation_state(
                        bool(detections), fixated_dwell_ms, glance_samples * sample_period_ms
                    )
                    span_results.append(
                        {
                            "video_id": video_id,
                            "t_start": start_s,
                            "t_end": end_s,
                            "status": span_status,
                            "sampled_frames": len(frame_times),
                            "detected_frames": len({item["video_time_s"] for item in detections}),
                            "raw_detections": raw_detection_count,
                            "detections": detections,
                            "gaze_samples_in_boxes": glance_samples,
                            "glance_dwell_ms": glance_samples * sample_period_ms,
                            "fixated_dwell_ms": fixated_dwell_ms,
                        }
                    )
                statuses = {result["status"] for result in span_results}
                status = next(
                    (name for name in ("fixated", "glanced", "never") if name in statuses),
                    "undetected",
                )
            completed[question_id] = {
                "question_id": question_id,
                "category_key": qa["category_key"],
                "targets": targets,
                "status": status,
                "evidence": span_results,
                "schema": FE_SCHEMA_VERSION,
            }
            if index % checkpoint_every == 0:
                _write_jsonl_atomic(
                    [completed[str(row["question_id"])] for row in qa_rows if str(row["question_id"]) in completed],
                    output_path,
                )
    finally:
        for _, _, _, capture in video_cache.values():
            capture.release()

    output_rows = [completed[str(row["question_id"])] for row in qa_rows]
    _write_jsonl_atomic(output_rows, output_path)
    counts: dict[tuple[str, str], int] = {}
    category_totals: dict[str, int] = {}
    for row in output_rows:
        key = (row["category_key"], row["status"])
        counts[key] = counts.get(key, 0) + 1
        category_totals[row["category_key"]] = category_totals.get(row["category_key"], 0) + 1
    distribution_path.parent.mkdir(parents=True, exist_ok=True)
    with distribution_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["category_key", "fixation_status", "count", "proportion"])
        for category, fixation_status in sorted(counts):
            count = counts[(category, fixation_status)]
            writer.writerow([category, fixation_status, count, f"{count / category_totals[category]:.6f}"])
    areas = [
        detection["area_fraction"]
        for row in output_rows
        for span in row["evidence"]
        for detection in span.get("detections", [])
    ]
    return {
        "questions": len(output_rows),
        "status_counts": {
            status: sum(row["status"] == status for row in output_rows)
            for status in sorted({row["status"] for row in output_rows})
        },
        "output": str(output_path),
        "distribution": str(distribution_path),
        "model": model_id,
        "box_threshold": localizer.box_threshold if localizer else None,
        "text_threshold": localizer.text_threshold if localizer else None,
        "max_target_area_fraction": max_area_fraction,
        "kept_box_area_fraction_quantiles": (
            [float(q) for q in np.quantile(areas, [0.25, 0.5, 0.75, 0.9])] if areas else []
        ),
        "fixated_rule": "cumulative dwell of fixations (>=100 ms I-VT) inside the expanded box >= 200 ms",
    }


FE_QC_PROMPT = (
    "These frames come from one time window of a first-person video. Green "
    "boxes mark the target ({targets}). Red dots are where the person was "
    "looking at that moment; blue circles are steady fixations. Did the person "
    "look at the target? Answer with exactly one word: 'fixated' (looked at it "
    "steadily for at least a moment), 'glanced' (gaze only touched it briefly) "
    "or 'never' (gaze never on it)."
)
FE_QC_STATES = ("fixated", "glanced", "never")


def cohen_kappa(first: list[str], second: list[str]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("Kappa needs two equally long, non-empty label lists")
    labels = sorted(set(first) | set(second))
    total = len(first)
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / total
    expected = sum(
        (first.count(label) / total) * (second.count(label) / total) for label in labels
    )
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def parse_fe_judgment(answer: str) -> str | None:
    found = [state for state in FE_QC_STATES if re.search(rf"\b{state}\b", answer.casefold())]
    return found[0] if len(found) == 1 else None


def fe_label_qc(
    labels_path: Path,
    data_root: Path,
    output_dir: Path,
    sample_count: int = 150,
    seed: int = 2026,
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    cuda_visible_devices: str = "1",
) -> dict[str, Any]:
    """README H1.3 step 4: mosaics for human spot checks plus a Qwen second
    opinion and Cohen's kappa against the automatic F(E) labels."""
    import cv2
    import numpy as np
    import pandas as pd
    from PIL import Image

    rows = [row for row in read_jsonl(labels_path) if row["status"] in FE_QC_STATES]
    chosen = random.Random(seed).sample(rows, min(sample_count, len(rows)))
    output_dir.mkdir(parents=True, exist_ok=True)
    labeler = Qwen3VLLabeler(model_id, cuda_visible_devices)
    participant_of = {
        str(row["question_id"]): int(row["participant"])
        for row in read_jsonl(data_root / "proc/qa/qa_subset.jsonl")
    }
    results = []
    for row in chosen:
        span = next(span for span in row["evidence"] if span.get("status") == row["status"])
        video_id = span["video_id"]
        participant = participant_of[str(row["question_id"])]
        capture = cv2.VideoCapture(str(data_root / "raw/video" / f"Person_{participant}" / f"{video_id}.mp4"))
        fps = capture.get(cv2.CAP_PROP_FPS)
        gaze = pd.read_parquet(data_root / "proc/gaze" / video_id / "gaze_px.parquet")
        fixations = pd.read_parquet(data_root / "proc/gaze" / video_id / "fixations.parquet")
        detected_times = sorted({d["video_time_s"] for d in span.get("detections", [])})
        times = detected_times or [span["t_start"] + i * 0.5 for i in range(int((span["t_end"] - span["t_start"]) / 0.5) + 1)]
        picks = [times[int(i)] for i in np.linspace(0, len(times) - 1, min(6, len(times)))]
        tiles = []
        for t in picks:
            capture.set(cv2.CAP_PROP_POS_FRAMES, round(t * fps))
            ok, frame = capture.read()
            if not ok:
                continue
            for detection in span.get("detections", []):
                if abs(detection["video_time_s"] - t) < 1e-6:
                    x1, y1, x2, y2 = (int(v) for v in detection["box_expanded_px"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 6)
            near = gaze[(gaze["t_video_s"] - t).abs() <= 0.25]
            for sample in near.itertuples(index=False):
                if sample.valid and not pd.isna(sample.u):
                    cv2.circle(frame, (int(sample.u), int(sample.v)), 9, (0, 0, 255), -1)
            mid = (fixations["t_on"] + fixations["t_off"]) / 2
            for fixation in fixations[(mid - t).abs() <= 0.5].itertuples(index=False):
                if not pd.isna(fixation.u):
                    cv2.circle(frame, (int(fixation.u), int(fixation.v)), 28, (255, 0, 0), 5)
            tiles.append(cv2.resize(frame, (448, 448), interpolation=cv2.INTER_AREA))
        capture.release()
        if not tiles:
            continue
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        mosaic = np.vstack([np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)])
        image_path = output_dir / f"{row['question_id']}_{row['status']}.jpg"
        cv2.imwrite(str(image_path), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 90])
        answer = labeler.label_image(
            Image.fromarray(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)),
            FE_QC_PROMPT.format(targets=", ".join(row["targets"])),
        )
        results.append(
            {
                "question_id": row["question_id"],
                "automatic": row["status"],
                "qwen": parse_fe_judgment(answer),
                "qwen_raw": answer,
                "image": str(image_path),
            }
        )
    judged = [row for row in results if row["qwen"] is not None]
    kappa = cohen_kappa([r["automatic"] for r in judged], [r["qwen"] for r in judged]) if judged else None
    report = {
        "sampled": len(results),
        "qwen_parsed": len(judged),
        "agreement": sum(r["automatic"] == r["qwen"] for r in judged) / len(judged) if judged else None,
        "cohen_kappa": kappa,
        "confusion": {
            f"{a}->{b}": sum(r["automatic"] == a and r["qwen"] == b for r in judged)
            for a in FE_QC_STATES
            for b in FE_QC_STATES
        },
        "human_review": "pending: inspect the mosaics in this directory",
        "results": results,
    }
    (output_dir / "agreement.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def merge_gaze_episodes(
    labeled_fixations: list[dict[str, Any]],
    embeddings: list[list[float]],
    similarity_threshold: float = 0.85,
) -> list[dict[str, Any]]:
    if len(labeled_fixations) != len(embeddings):
        raise ValueError("Each fixation label must have one text embedding")
    episodes: list[dict[str, Any]] = []
    previous_embedding: list[float] | None = None
    previous_fix_id: int | None = None
    for fixation, embedding in zip(labeled_fixations, embeddings, strict=True):
        normalized = math.sqrt(sum(value * value for value in embedding))
        if normalized == 0:
            raise ValueError("Text embeddings must be non-zero")
        vector = [value / normalized for value in embedding]
        similarity = (
            sum(left * right for left, right in zip(previous_embedding, vector, strict=True))
            if previous_embedding is not None
            else -1.0
        )
        box = fixation["box_px"]
        is_adjacent = (
            previous_fix_id is not None
            and int(fixation["fix_id"]) == previous_fix_id + 1
        )
        if episodes and is_adjacent and similarity > similarity_threshold:
            episode = episodes[-1]
            episode["t_off"] = fixation["t_off"]
            episode["total_dwell_ms"] += fixation["dur_ms"]
            episode["n_fixations"] += 1
            old_box = episode["box_px"]
            episode["box_px"] = [
                min(old_box[0], box[0]),
                min(old_box[1], box[1]),
                max(old_box[2], box[2]),
                max(old_box[3], box[3]),
            ]
        else:
            episodes.append(
                {
                    "ep_id": len(episodes),
                    "t_on": fixation["t_on"],
                    "t_off": fixation["t_off"],
                    "total_dwell_ms": fixation["dur_ms"],
                    "n_fixations": 1,
                    "label": fixation["label"],
                    "box_px": list(box),
                }
            )
        previous_embedding = vector
        previous_fix_id = int(fixation["fix_id"])
    return episodes


def annotate_gaze_episodes(
    video_path: Path,
    gaze_path: Path,
    fixations_path: Path,
    labels_path: Path,
    episodes_path: Path,
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    cuda_visible_devices: str = "1",
    checkpoint_every: int = 32,
    max_fixations: int | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    peak_gpu_allocated_bytes = None
    peak_gpu_reserved_bytes = None
    try:
        import cv2
        import pandas as pd
        from PIL import Image
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Install the local and local-gpu extras to run H1 gaze episodes"
        ) from exc
    fixations = pd.read_parquet(fixations_path)
    require_video_time_fixations(fixations, fixations_path)
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1")
    if max_fixations is not None and max_fixations < 1:
        raise ValueError("max_fixations must be at least 1")
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Could not open RGB video {video_path}")
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        video.release()
        raise ValueError(f"Invalid video metadata for {video_path}")
    if (width, height) != (1408, 1408):
        video.release()
        raise ValueError(f"Expected 1408x1408 RGB video, got {width}x{height}")

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    existing = (
        pd.read_parquet(labels_path).to_dict("records")
        if labels_path.is_file()
        else []
    )
    completed = {int(row["fix_id"]) for row in existing}
    pending = []
    out_of_frame_fixations = 0
    for row in fixations.to_dict("records"):
        if (
            int(row["fix_id"]) in completed
            or float(row["dur_ms"]) < 200
            or pd.isna(row["u"])
            or pd.isna(row["v"])
        ):
            continue
        u, v = float(row["u"]), float(row["v"])
        if not 0 <= u < width or not 0 <= v < height:
            out_of_frame_fixations += 1
            continue
        frame_index = round(float(row["t_on"]) * fps)
        if frame_index < 0 or frame_index >= frame_count:
            continue
        pending.append((frame_index, row))
    pending.sort(key=lambda item: (item[0], int(item[1]["fix_id"])))
    eligible_count = len(pending) + len(completed)
    if max_fixations is not None:
        pending = pending[:max_fixations]

    if pending:
        labeler = Qwen3VLLabeler(model_id, cuda_visible_devices)
        import torch

        torch.cuda.reset_peak_memory_stats()
        current_index = -1
        current_frame = None
        for frame_index, fixation in pending:
            while current_index < frame_index:
                ok, current_frame = video.read()
                current_index += 1
                if not ok:
                    video.release()
                    raise RuntimeError(
                        f"Could not decode frame {current_index} from {video_path}"
                    )
            crop, box = crop_fixation_frame(
                current_frame, float(fixation["u"]), float(fixation["v"])
            )
            label = labeler.label_image(
                Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            )
            existing.append(
                {
                    "fix_id": int(fixation["fix_id"]),
                    "t_on": float(fixation["t_on"]),
                    "t_off": float(fixation["t_off"]),
                    "dur_ms": float(fixation["dur_ms"]),
                    "u": float(fixation["u"]),
                    "v": float(fixation["v"]),
                    "box_px": box,
                    "label": label,
                }
            )
            if len(existing) % checkpoint_every == 0:
                _write_parquet_atomic(
                    existing,
                    labels_path,
                    ["fix_id", "t_on", "t_off", "dur_ms", "u", "v", "box_px", "label"],
                )
            if len(existing) % (checkpoint_every * 8) == 0:
                print(
                    f"H1.1 {video_path.stem}: "
                    f"{len(existing)}/{eligible_count} fixations labeled",
                    flush=True,
                )
        peak_gpu_allocated_bytes = torch.cuda.max_memory_allocated()
        peak_gpu_reserved_bytes = torch.cuda.max_memory_reserved()
        del labeler
        gc.collect()
        torch.cuda.empty_cache()
    video.release()
    _write_parquet_atomic(
        existing,
        labels_path,
        ["fix_id", "t_on", "t_off", "dur_ms", "u", "v", "box_px", "label"],
    )

    labeled = sorted(existing, key=lambda row: (row["t_on"], row["fix_id"]))
    if labeled:
        embedder = SentenceTransformer(embedding_model_id, device="cpu")
        vectors = embedder.encode(
            [row["label"] for row in labeled],
            normalize_embeddings=False,
            show_progress_bar=True,
        ).tolist()
    else:
        vectors = []
    episodes = merge_gaze_episodes(labeled, vectors)
    _write_parquet_atomic(
        episodes,
        episodes_path,
        ["ep_id", "t_on", "t_off", "total_dwell_ms", "n_fixations", "label", "box_px"],
    )
    return {
        "eligible_fixations": eligible_count,
        "skipped_out_of_frame_fixations": out_of_frame_fixations,
        "processed_this_run": len(pending),
        "labeled_fixations": len(existing),
        "episodes": len(episodes),
        "labels": str(labels_path),
        "episodes_output": str(episodes_path),
        "model": model_id,
        "embedding_model": embedding_model_id,
        "time_axis": "video seconds; frame 0 is the first gaze sample",
        "elapsed_seconds": time.perf_counter() - started_at,
        "peak_gpu_allocated_gib": (
            peak_gpu_allocated_bytes / 1024**3
            if peak_gpu_allocated_bytes is not None
            else None
        ),
        "peak_gpu_reserved_gib": (
            peak_gpu_reserved_bytes / 1024**3
            if peak_gpu_reserved_bytes is not None
            else None
        ),
    }


CONTEXT_LABEL_PROMPT = (
    "The red dot marks where the person was looking; eye tracking can be off by "
    "a few pixels. Name the single object or piece of text closest to the red "
    "dot in at most 6 words. If the dot is on a plain surface with nothing "
    "nearby, answer 'none'."
)


def context_crop_with_marker(frame: Any, u: float, v: float, size: int = 512) -> Any:
    """512 px context crop centred on the fixation with a red dot (the visual
    prompt used by H1.2 and C3-V)."""
    import cv2
    import numpy as np

    crop, _ = crop_fixation_frame(frame, u, v, size)
    crop = np.ascontiguousarray(crop.copy())
    center = (size // 2, size // 2)
    cv2.circle(crop, center, 9, (0, 0, 255), -1)
    cv2.circle(crop, center, 14, (255, 255, 255), 2)
    return crop


def relabel_none_fixations(
    sessions_path: Path,
    data_root: Path,
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    cuda_visible_devices: str = "1",
    checkpoint_every: int = 64,
) -> dict[str, Any]:
    """Optional H1.1 supplement for fixations labeled 'none'.

    The README prompt asks for the object exactly at the crop centre; with
    1-2 degrees of gaze error the point often falls between objects. This pass
    re-asks only those fixations with a 512 px context crop and a red dot.
    The README label is kept as ``label_strict``; ``label`` holds the final
    label and ``label_source`` says which pass produced it. Episodes are then
    rebuilt with the same merging rule.
    """
    import cv2
    import pandas as pd
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    labeler = None
    report = []
    columns = ["fix_id", "t_on", "t_off", "dur_ms", "u", "v", "box_px", "label", "label_strict", "label_source"]
    for session in sessions:
        participant = int(session["participant"])
        video_id = str(session["video_id"])
        gaze_dir = data_root / "proc/gaze" / video_id
        labels_path = gaze_dir / "fixation_labels.parquet"
        frame = pd.read_parquet(labels_path)
        rows = frame.to_dict("records")
        for row in rows:
            row.setdefault("label_strict", row["label"])
            if not isinstance(row.get("label_source"), str):
                row["label_source"] = "strict"
        pending = [
            row for row in rows
            if row["label_source"] == "strict" and _is_none_label(row["label_strict"])
        ]
        pending.sort(key=lambda row: row["t_on"])
        if pending:
            if labeler is None:
                labeler = Qwen3VLLabeler(model_id, cuda_visible_devices)
            capture = cv2.VideoCapture(str(data_root / "raw/video" / f"Person_{participant}" / f"{video_id}.mp4"))
            fps = capture.get(cv2.CAP_PROP_FPS)
            position = -1
            current = None
            for done, row in enumerate(pending, start=1):
                index = round(float(row["t_on"]) * fps)
                while position < index:
                    ok, current = capture.read()
                    position += 1
                    if not ok:
                        raise RuntimeError(f"Could not decode frame {position} of {video_id}")
                crop = context_crop_with_marker(current, float(row["u"]), float(row["v"]))
                row["label"] = labeler.label_image(
                    Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), CONTEXT_LABEL_PROMPT
                )
                row["label_source"] = "context"
                if done % checkpoint_every == 0:
                    _write_parquet_atomic(rows, labels_path, columns)
            capture.release()
        _write_parquet_atomic(rows, labels_path, columns)
        labeled = sorted(rows, key=lambda row: (row["t_on"], row["fix_id"]))
        encoder = SentenceTransformer(embedding_model_id, device="cpu")
        vectors = encoder.encode([row["label"] for row in labeled], show_progress_bar=False).tolist()
        episodes = merge_gaze_episodes(labeled, vectors)
        _write_parquet_atomic(
            episodes,
            gaze_dir / "episodes.parquet",
            ["ep_id", "t_on", "t_off", "total_dwell_ms", "n_fixations", "label", "box_px"],
        )
        report.append(
            {
                "video_id": video_id,
                "fixations": len(rows),
                "none_strict": sum(_is_none_label(row["label_strict"]) for row in rows),
                "none_final": sum(_is_none_label(row["label"]) for row in rows),
                "episodes": len(episodes),
            }
        )
        print(f"relabel {video_id}: {report[-1]}", flush=True)
    summary = {
        "prompt": CONTEXT_LABEL_PROMPT,
        "sessions": report,
        "none_strict_fraction": sum(r["none_strict"] for r in report) / max(1, sum(r["fixations"] for r in report)),
        "none_final_fraction": sum(r["none_final"] for r in report) / max(1, sum(r["fixations"] for r in report)),
    }
    path = data_root / "proc/reports/h1_1_relabel_report.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _is_none_label(value: Any) -> bool:
    return " ".join(str(value).casefold().strip(" .'\"").split()) in {"", "none", "nothing", "n/a"}


def annotate_selected_gaze_episodes(
    sessions_path: Path,
    data_root: Path,
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    cuda_visible_devices: str = "1",
    checkpoint_every: int = 32,
    max_fixations: int | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    if not sessions:
        raise ValueError(f"No selected sessions found in {sessions_path}")
    reports = []
    report_dir = data_root / "proc/reports/h1_1"
    report_dir.mkdir(parents=True, exist_ok=True)
    for session in sessions:
        participant = int(session["participant"])
        video_id = str(session["video_id"])
        stem = video_id.replace(
            f"Person_{participant}_", f"person_{participant}_", 1
        )
        video_path = (
            data_root / "raw/video" / f"Person_{participant}" / f"{video_id}.mp4"
        )
        gaze_dir = data_root / "proc/gaze" / video_id
        output_dir = report_dir / f"{video_id}.json"
        result = annotate_gaze_episodes(
            video_path,
            gaze_dir / "gaze_px.parquet",
            gaze_dir / "fixations.parquet",
            gaze_dir / "fixation_labels.parquet",
            gaze_dir / "episodes.parquet",
            model_id=model_id,
            embedding_model_id=embedding_model_id,
            cuda_visible_devices=cuda_visible_devices,
            checkpoint_every=checkpoint_every,
            max_fixations=max_fixations,
        )
        result["video_id"] = video_id
        _write_json_atomic(result, output_dir)
        reports.append(result)
        _write_json_atomic(
            {
                "completed_sessions": len(reports),
                "total_sessions": len(sessions),
                "labeled_fixations": sum(
                    row["labeled_fixations"] for row in reports
                ),
                "episodes": sum(row["episodes"] for row in reports),
                "sessions": reports,
            },
            data_root / "proc/reports/h1_1_report.json",
        )
    return {
        "sessions": len(reports),
        "labeled_fixations": sum(row["labeled_fixations"] for row in reports),
        "episodes": sum(row["episodes"] for row in reports),
        "elapsed_seconds": time.perf_counter() - started_at,
        "report": str(data_root / "proc/reports/h1_1_report.json"),
    }


def _write_parquet_atomic(
    rows: list[dict[str, Any]], path: Path, columns: list[str]
) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    pd.DataFrame(rows, columns=columns).to_parquet(temporary, index=False)
    temporary.replace(path)


def _write_jsonl_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    write_jsonl(rows, temporary)
    temporary.replace(path)


def _write_json_atomic(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_h1_report(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
