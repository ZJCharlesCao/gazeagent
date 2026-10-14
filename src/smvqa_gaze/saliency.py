from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .manifest import sha256_file


def normalize_saliency_map(values: Any) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("UNISAL output must be a finite 2-D saliency map")
    maximum = float(array.max())
    if maximum <= 0:
        raise ValueError("UNISAL saliency map has no positive values")
    return np.clip(array / maximum * 255, 0, 255).astype(np.uint8)


def _extract_one_fps_frames(video_path: Path, images_dir: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open UNISAL input video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or source_frame_count <= 0:
            raise RuntimeError(f"Invalid video metadata for {video_path}")
        sample_count = math.ceil(source_frame_count / fps)
        written_count = 0
        for sample_index in range(sample_count):
            source_index = round(sample_index * fps)
            if source_index >= source_frame_count:
                break
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Could not decode frame {source_index} from {video_path}"
                )
            output_path = images_dir / f"frame_{sample_index:08d}.jpg"
            if not cv2.imwrite(
                str(output_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                raise RuntimeError(f"Could not write UNISAL input frame {output_path}")
            written_count += 1
        return written_count
    finally:
        capture.release()


def generate_unisal_saliency(
    sessions_path: Path,
    data_root: Path,
    unisal_root: Path,
    cuda_visible_devices: str = "1",
) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ["TRAIN_DIR"] = str(unisal_root / "training_runs")
    import cv2
    import numpy as np
    import torch

    started_at = time.perf_counter()
    checkpoint_dir = unisal_root / "training_runs" / "pretrained_unisal"
    checkpoint = checkpoint_dir / "weights_best.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Official UNISAL checkpoint is missing: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("UNISAL H1.4 inference requires CUDA")

    sys.path.insert(0, str(unisal_root))
    try:
        from unisal import data as unisal_data
        from unisal.train import Trainer
    except ImportError as exc:
        raise RuntimeError(
            "Install the local-gpu extra (including scipy and tensorboardX) "
            "before importing the official UNISAL checkout"
        ) from exc

    sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    if not sessions:
        raise ValueError(f"No selected sessions found in {sessions_path}")
    trainer = Trainer.init_from_cfg_dir(checkpoint_dir)
    trainer.model.load_best_weights(trainer.train_dir)
    trainer.model.to(trainer.device)
    trainer.model.eval()
    repository_revision = subprocess.run(
        ["git", "-C", str(unisal_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    reports = []
    temporary_root = data_root / "proc/unisal_work"
    temporary_root.mkdir(parents=True, exist_ok=True)
    for session in sessions:
        participant = int(session["participant"])
        video_id = str(session["video_id"])
        video_path = (
            data_root / "raw/video" / f"Person_{participant}" / f"{video_id}.mp4"
        )
        if not video_path.is_file():
            raise FileNotFoundError(f"UNISAL input video is missing: {video_path}")
        output_dir = data_root / "proc/saliency" / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(
            prefix=f"{video_id}_", dir=temporary_root
        ) as temporary_dir:
            session_dir = Path(temporary_dir)
            images_dir = session_dir / "images"
            images_dir.mkdir()
            extracted_frames = _extract_one_fps_frames(video_path, images_dir)
            dataset = unisal_data.FolderVideoDataset(
                images_dir, frame_modulo=1, source="DHF1K"
            )
            frame_count = dataset.n_images_dict[0]
            if frame_count < 1 or frame_count != extracted_frames:
                raise RuntimeError(
                    f"UNISAL found {frame_count} of {extracted_frames} extracted "
                    f"frames for {video_path}"
                )
            sequence_length = int(12 * 0.5)
            hidden = [None]
            target_size = dataset.target_size_dict[0]
            frame_index = 0
            for start in range(0, frame_count, sequence_length):
                end = min(frame_count, start + sequence_length)
                images = [
                    dataset.load_frame(index + 1)
                    for index in range(start, end)
                ]
                frame_sequence = (
                    dataset.preprocess_sequence(images)
                    .unsqueeze(0)
                    .float()
                    .to(trainer.device)
                )
                with torch.inference_mode():
                    predictions, hidden = trainer.model(
                        frame_sequence,
                        h0=hidden,
                        return_hidden=True,
                        source="DHF1K",
                        target_size=target_size,
                        static=False,
                    )
                for prediction in torch.unbind(predictions[0], dim=0):
                    values = prediction.squeeze().exp().detach().cpu().numpy()
                    saliency_map = normalize_saliency_map(values)
                    output_path = output_dir / f"frame_{frame_index:08d}.jpg"
                    temporary_output = output_path.with_name(
                        output_path.name + ".part.jpg"
                    )
                    if not cv2.imwrite(
                        str(temporary_output),
                        saliency_map,
                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                    ):
                        raise RuntimeError(
                            f"Could not write saliency map {temporary_output}"
                        )
                    temporary_output.replace(output_path)
                    frame_index += 1
                del frame_sequence, predictions
            if frame_index != frame_count:
                raise RuntimeError(
                    f"UNISAL produced {frame_index} maps for {frame_count} "
                    f"input frames in {video_id}"
                )

        manifest = {
            "video_id": video_id,
            "source_video": str(video_path),
            "model": "rdroste/unisal pretrained_unisal",
            "model_repository": "https://github.com/rdroste/unisal",
            "model_repository_revision": repository_revision,
            "checkpoint_sha256": sha256_file(checkpoint),
            "sampling_fps": 1,
            "frames": frame_count,
            "map_format": "uint8 JPEG quality 95; independently normalized per frame",
        }
        manifest_path = output_dir / "manifest.json"
        temporary_manifest = manifest_path.with_name("manifest.json.part")
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary_manifest.replace(manifest_path)
        reports.append(manifest)
        print(
            f"H1.4 {video_id}: generated {frame_count} 1-fps saliency maps",
            flush=True,
        )

    report_path = data_root / "proc/reports/h1_4_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "sessions": len(reports),
        "frames": sum(row["frames"] for row in reports),
        "elapsed_seconds": time.perf_counter() - started_at,
        "checkpoint": str(checkpoint),
        "sessions_report": reports,
    }
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {**result, "report": str(report_path)}
