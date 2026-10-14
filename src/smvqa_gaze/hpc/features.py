"""Video-RAG visual features: EasyOCR text, CLIP keyframe embeddings and
open-vocabulary detection (GroundingDINO instead of the paper's APE)."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import CLIP_ID, DETECTOR_ID, TIER_SIDE, Condition, RunConfig, atomic_save_npy, atomic_write_text
from .data import Dataset, SessionInfo
from .frames import FrameStore
from .gaze_source import GazeSources
from .transforms import crop

_GPU_LOCK = threading.Lock()
# Lazy model loads happen from worker threads; concurrent from_pretrained calls
# race on meta-device initialisation in transformers 5, so loads are serialised.
_LOAD_LOCK = threading.Lock()


def visual_variant(condition: Condition, tier: str) -> str:
    """OCR/detection see crops only in fovea conditions (README 7.4)."""
    if condition.visual == "fovea":
        return f"fovea-{condition.gaze_source}-{tier}"
    return "full"


def variant_image(
    variant: str, frames: FrameStore, sources: GazeSources, video_id: str, t: float
) -> np.ndarray:
    image = frames.get(video_id, t)
    if variant == "full":
        return image
    _, source, tier = variant.split("-")
    box, _ = sources.box(source, video_id, t, TIER_SIDE[tier])
    return crop(image, box)


class OcrIndex:
    """EasyOCR over frames sampled at config.ocr_fps (query-independent DB)."""

    def __init__(self, dataset: Dataset, config: RunConfig, frames: FrameStore, sources: GazeSources) -> None:
        self.dataset = dataset
        self.config = config
        self.frames = frames
        self.sources = sources
        self._reader = None

    def reader(self):
        with _LOAD_LOCK:
            if self._reader is None:
                import easyocr
                import torch

                self._reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
        return self._reader

    def path(self, variant: str, video_id: str) -> Path:
        return self.dataset.paths.feature_cache / "ocr" / variant / f"{video_id}.json"

    def entries(
        self, variant: str, session: SessionInfo, keep: Callable[[float], bool] | None = None
    ) -> list[dict[str, Any]]:
        """``keep`` restricts a smoke run to some times (cached separately)."""
        tag = variant if keep is None else f"{variant}-smoke"
        path = self.path(tag, session.video_id)
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        step = 1.0 / self.config.ocr_fps
        rows: list[dict[str, Any]] = []
        previous = None
        for index in range(int(math.floor(session.duration_s / step)) + 1):
            t = float(round(index * step))
            if t >= session.duration_s:
                break
            if keep is not None and not keep(t):
                continue
            image = variant_image(variant, self.frames, self.sources, session.video_id, t)
            with _GPU_LOCK:
                results = self.reader().readtext(image)
            # Drop single-character fragments (dark scenes yield stray digits).
            words = [
                str(text).strip()
                for _, text, confidence in results
                if confidence >= self.config.ocr_min_confidence
                and sum(char.isalnum() for char in str(text)) >= 2
            ]
            text = " | ".join(words)
            if not text or text == previous:
                previous = text or previous
                continue
            previous = text
            rows.append({"video_id": session.video_id, "start": t, "end": t + step, "text": f"On-screen text: {text}"})
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(rows) + "\n")
        return rows


class ClipIndex:
    """CLIP image embeddings at 1 fps, used to select detection keyframes."""

    def __init__(self, dataset: Dataset, frames: FrameStore, sources: GazeSources) -> None:
        self.dataset = dataset
        self.frames = frames
        self.sources = sources
        self._model = None

    def model(self):
        with _LOAD_LOCK:
            if self._model is None:
                import torch
                from transformers import CLIPModel, CLIPProcessor

                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = (
                    CLIPModel.from_pretrained(CLIP_ID, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval(),
                    CLIPProcessor.from_pretrained(CLIP_ID),
                    device,
                )
        return self._model

    def image_vectors(
        self, variant: str, session: SessionInfo, keep: Callable[[float], bool] | None = None
    ) -> np.ndarray:
        """One CLIP vector per second; with ``keep`` (smoke runs) only those
        seconds are encoded and the rest stay zero."""
        import torch
        from PIL import Image

        tag = variant if keep is None else f"{variant}-smoke"
        path = self.dataset.paths.feature_cache / "clip" / tag / f"{session.video_id}.npy"
        if path.is_file():
            return np.load(path)
        model, processor, device = self.model()
        seconds = int(math.ceil(session.duration_s))
        wanted = [t for t in range(seconds) if keep is None or keep(float(t))]
        dimension = model.config.projection_dim
        output = np.zeros((seconds, dimension), dtype=np.float16)
        for start in range(0, len(wanted), 64):
            batch = wanted[start : start + 64]
            images = [
                Image.fromarray(variant_image(variant, self.frames, self.sources, session.video_id, float(t)))
                for t in batch
            ]
            inputs = processor(images=images, return_tensors="pt").to(device)
            with _GPU_LOCK, torch.inference_mode():
                features = model.get_image_features(pixel_values=inputs["pixel_values"].to(model.dtype))
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = torch.nn.functional.normalize(features.float(), dim=-1)
            output[batch] = features.cpu().numpy().astype(np.float16)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_save_npy(path, output)
        return output

    def text_vector(self, text: str) -> np.ndarray:
        import torch

        model, processor, device = self.model()
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
        with _GPU_LOCK, torch.inference_mode():
            features = model.get_text_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        return torch.nn.functional.normalize(features.float(), dim=-1)[0].cpu().numpy()


class DetectionCache:
    """GroundingDINO on keyframes; results cached by (variant, frame, objects)."""

    def __init__(self, dataset: Dataset, config: RunConfig, frames: FrameStore, sources: GazeSources) -> None:
        self.dataset = dataset
        self.config = config
        self.frames = frames
        self.sources = sources
        self.path = dataset.paths.feature_cache / "detections.jsonl"
        self.cache: dict[str, list[dict[str, Any]]] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.cache[row["key"]] = row["detections"]
        self._model = None
        self._write_lock = threading.Lock()

    def model(self):
        with _LOAD_LOCK:
            if self._model is None:
                import torch
                from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = (
                    AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR_ID).to(device).eval(),
                    AutoProcessor.from_pretrained(DETECTOR_ID),
                    device,
                )
        return self._model

    def detect(self, variant: str, video_id: str, t: float, objects: list[str]) -> list[dict[str, Any]]:
        import torch
        from PIL import Image

        objects = [" ".join(obj.split()).strip(" .") for obj in objects if obj.strip()]
        if not objects:
            return []
        digest = hashlib.sha1("|".join(sorted(o.casefold() for o in objects)).encode()).hexdigest()[:12]
        key = f"{variant}:{video_id}:{t:.2f}:{digest}"
        if key in self.cache:
            return self.cache[key]
        image = variant_image(variant, self.frames, self.sources, video_id, t)
        model, processor, device = self.model()
        prompt = " . ".join(objects) + " ."
        inputs = processor(images=Image.fromarray(image), text=prompt, return_tensors="pt").to(device)
        with _GPU_LOCK, torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.config.detector_box_threshold,
            text_threshold=self.config.detector_text_threshold,
            target_sizes=[image.shape[:2]],
        )[0]
        labels = result.get("text_labels", result.get("labels", []))
        detections = [
            {"label": str(label), "score": round(float(score), 3), "box": [round(float(v), 1) for v in box]}
            for label, score, box in zip(labels, result["scores"].tolist(), result["boxes"].tolist())
            if str(label).strip()
        ]
        with self._write_lock:
            self.cache[key] = detections
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"key": key, "detections": detections}) + "\n")
        return detections


def describe_detections(detections: list[dict[str, Any]], image_side: int) -> str:
    """Video-RAG style object text: counts plus coarse positions."""
    if not detections:
        return "no requested objects detected"
    groups: dict[str, list[str]] = {}
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        cx, cy = (x1 + x2) / 2 / image_side, (y1 + y2) / 2 / image_side
        horizontal = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
        vertical = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle"
        groups.setdefault(detection["label"], []).append(f"{vertical}-{horizontal}")
    return "; ".join(f"{len(places)} x {label} ({', '.join(places)})" for label, places in sorted(groups.items()))


QUERY_OBJECT_PROMPT = (
    "To answer the question below from a first-person video, which physical "
    "objects should be located by an object detector? Return only a JSON array "
    "of at most 4 short noun phrases (use [] if none).\nQuestion: {question}"
)


def parse_object_list(text: str) -> list[str]:
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        return []
    try:
        values = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [str(value).strip() for value in values if isinstance(value, str) and value.strip()][:4]
