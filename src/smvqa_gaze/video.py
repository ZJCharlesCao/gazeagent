from __future__ import annotations

import csv
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        diagnostic = (result.stderr or result.stdout).strip()
        diagnostic = "\n".join(diagnostic.splitlines()[-20:])
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: "
            f"{' '.join(command)}\n{diagnostic}"
        )
    return result


def export_vrs_to_mp4(
    vrs_path: Path, output_path: Path, log_dir: Path
) -> dict[str, Any]:
    if shutil.which("vrs_to_mp4") is None:
        raise RuntimeError(
            "The local Project Aria environment must provide the vrs_to_mp4 command"
        )
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required for RGB export")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    intermediate = log_dir / "rgb_intermediate.mp4"
    _run(
        [
            "vrs_to_mp4",
            "--vrs",
            str(vrs_path),
            "--stream_id",
            "214-1",
            "--output_video",
            str(intermediate),
            "--log_folder",
            str(log_dir),
        ]
    )
    probe = json.loads(
        _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames",
                "-of",
                "json",
                str(intermediate),
            ]
        ).stdout
    )
    streams = probe.get("streams", [])
    if not streams:
        raise RuntimeError(f"No RGB video stream exported from {vrs_path}")
    source = streams[0]
    if (source.get("width"), source.get("height")) != (1408, 1408):
        raise ValueError(
            f"Expected 1408x1408 RGB frames, got "
            f"{source.get('width')}x{source.get('height')} from {vrs_path}"
        )
    timestamp_csv = log_dir / "mp4_to_vrs_time_ns.csv"
    if not timestamp_csv.is_file():
        raise RuntimeError(f"Project Aria did not produce {timestamp_csv}")

    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(intermediate),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )
    output_probe = json.loads(
        _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames",
                "-of",
                "json",
                str(output_path),
            ]
        ).stdout
    )
    exported = output_probe["streams"][0]
    if (exported.get("width"), exported.get("height")) != (1408, 1408):
        raise RuntimeError("H.264 re-encode changed the RGB frame dimensions")
    timestamp_count = sum(1 for _ in timestamp_csv.open(encoding="utf-8")) - 1
    try:
        frame_count = int(exported["nb_frames"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("ffprobe could not determine the exported frame count") from exc
    if timestamp_count != frame_count:
        raise RuntimeError(
            f"Frame timestamp mismatch: {timestamp_count} timestamps for {frame_count} frames"
        )
    return {
        "vrs": str(vrs_path),
        "video": str(output_path),
        "timestamp_csv": str(timestamp_csv),
        "width": exported["width"],
        "height": exported["height"],
        "frame_rate": exported.get("avg_frame_rate"),
        "frames": frame_count,
        "codec": "libx264",
        "crf": 18,
    }


def write_frame_timestamp_map(
    video_path: Path, gaze_csv: Path, output_csv: Path
) -> dict[str, Any]:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is required to inspect the released RGB video")
    probe = json.loads(
        _run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(video_path),
            ]
        ).stdout
    )
    streams = probe.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")
    video_stream = streams[0]
    if (video_stream.get("width"), video_stream.get("height")) != (1408, 1408):
        raise ValueError(
            f"Expected 1408x1408 RGB video, got "
            f"{video_stream.get('width')}x{video_stream.get('height')}"
        )
    try:
        frame_rate = Fraction(video_stream["avg_frame_rate"])
        frame_count = int(video_stream["nb_frames"])
        duration_s = float(probe["format"]["duration"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(f"Incomplete video metadata for {video_path}") from exc
    fps = float(frame_rate)
    if frame_rate <= 0 or frame_count <= 0 or duration_s <= 0:
        raise ValueError(f"Invalid video timing metadata for {video_path}")
    if abs(frame_count / fps - duration_s) > 1.0:
        raise ValueError(
            f"Frame count and duration disagree for {video_path}: "
            f"{frame_count} frames at {fps} fps vs {duration_s}s"
        )
    with gaze_csv.open(encoding="utf-8", newline="") as gaze_stream:
        first_gaze = next(csv.DictReader(gaze_stream), None)
    if first_gaze is None:
        raise ValueError(f"No gaze samples in {gaze_csv}")
    first_gaze_us = int(first_gaze["tracking_timestamp_us"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "video_time_s", "gaze_timestamp_us"])
        for frame_index in range(frame_count):
            video_time_s = frame_index / fps
            gaze_timestamp_us = first_gaze_us + round(video_time_s * 1_000_000)
            writer.writerow([frame_index, f"{video_time_s:.9f}", gaze_timestamp_us])
    return {
        "video": str(video_path),
        "gaze_csv": str(gaze_csv),
        "timestamp_csv": str(output_csv),
        "frames": frame_count,
        "frame_rate": str(frame_rate),
        "width": video_stream["width"],
        "height": video_stream["height"],
        "alignment": "video frame 0 aligned to first gaze sample",
        "first_gaze_timestamp_us": first_gaze_us,
    }
