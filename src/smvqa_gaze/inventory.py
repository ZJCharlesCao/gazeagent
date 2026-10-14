from __future__ import annotations

import concurrent.futures
import json
import re
import shutil
import subprocess
import urllib.parse
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

from .download import DATASET_ID, DATASET_REVISION, _download_file, dataset_tree
from .gaze import gaze_coverage, read_gaze_csv


def _video_id(remote_path: str) -> str:
    return Path(remote_path).stem


def _session_stem(participant: int, video_id: str) -> str:
    prefix = f"Person_{participant}_"
    if not video_id.startswith(prefix):
        raise ValueError(f"Unexpected video ID {video_id!r}")
    return video_id.replace(prefix, f"person_{participant}_", 1)


def _probe_remote_video(remote_path: str) -> dict[str, Any]:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is required to inventory released RGB videos")
    encoded_path = urllib.parse.quote(remote_path, safe="/")
    url = (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
        f"{DATASET_REVISION}/{encoded_path}?download=true"
    )
    result = subprocess.run(
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
            url,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode:
        diagnostic = "\n".join((result.stderr or result.stdout).splitlines()[-10:])
        raise RuntimeError(f"ffprobe failed for {remote_path}: {diagnostic}")
    probe = json.loads(result.stdout)
    streams = probe.get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found in {remote_path}")
    stream = streams[0]
    try:
        fps = float(Fraction(stream["avg_frame_rate"]))
        duration_s = float(probe["format"]["duration"])
        frame_count = int(stream["nb_frames"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Incomplete video metadata for {remote_path}") from exc
    if (stream.get("width"), stream.get("height")) != (1408, 1408):
        raise ValueError(f"Unexpected RGB dimensions in {remote_path}")
    if fps <= 0 or frame_count <= 0 or duration_s <= 0:
        raise ValueError(f"Invalid timing metadata in {remote_path}")
    return {
        "duration_s": duration_s,
        "video_width": stream["width"],
        "video_height": stream["height"],
        "video_frame_rate": stream["avg_frame_rate"],
        "video_frames": frame_count,
    }


def _gaze_metrics(path: Path) -> dict[str, Any]:
    samples = read_gaze_csv(path)
    if not samples:
        raise ValueError(f"No gaze samples in {path}")
    return {
        "gaze_coverage": gaze_coverage(samples),
        "gaze_valid_fraction": sum(sample.valid for sample in samples) / len(samples),
        "gaze_samples": len(samples),
        "first_gaze_timestamp_us": samples[0].timestamp_us,
        "last_gaze_timestamp_us": samples[-1].timestamp_us,
    }


def build_dataset_inventory(
    qa_rows: list[dict[str, Any]],
    data_root: Path,
    workers: int = 6,
) -> list[dict[str, Any]]:
    """Inventory released video sessions and gaze quality without downloading video."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    files: dict[str, dict[str, Any]] = {}
    for participant in range(1, 11):
        for prefix in (
            f"data/video/Person_{participant}",
            f"data/raw/person_{participant}",
            f"data/mps/person_{participant}",
        ):
            for row in dataset_tree(prefix):
                if row.get("type") == "file":
                    files[str(row["path"])] = row

    video_rows = [
        row
        for path, row in files.items()
        if path.startswith("data/video/") and path.endswith(".mp4")
    ]
    if not video_rows:
        raise FileNotFoundError("No released MP4 videos found in the dataset")
    video_metadata: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_probe_remote_video, str(row["path"])): str(row["path"])
            for row in video_rows
        }
        for future in concurrent.futures.as_completed(futures):
            remote_path = futures[future]
            video_metadata[remote_path] = future.result()

    question_counts = Counter(
        (
            int(row["participant"]),
            int(row["primary_session"]),
            str(row["category_key"]),
        )
        for row in qa_rows
    )
    metadata_by_video: dict[str, dict[str, Any]] = {}
    for video_row in video_rows:
        video_path = str(video_row["path"])
        video_id = _video_id(video_path)
        match = re.match(r"^Person_(\d+)_session_(\d+)_", video_id)
        date_match = re.search(r"_(\d{8})_", video_id)
        if not match or not date_match:
            raise ValueError(f"Could not parse released video ID {video_id!r}")
        participant, session = int(match.group(1)), int(match.group(2))
        stem = _session_stem(participant, video_id)
        gaze_remote = (
            f"data/mps/person_{participant}/{stem}_vrs/"
            "eye_gaze/general_eye_gaze.csv"
        )
        vrs_remote = f"data/raw/person_{participant}/{stem}.vrs"
        metadata_by_video[video_id] = {
            "participant": participant,
            "session": session,
            "date": (
                f"{date_match.group(1)[4:]}-{date_match.group(1)[:2]}-"
                f"{date_match.group(1)[2:4]}"
            ),
            "video_id": video_id,
            "video_remote_path": video_path,
            "video_size_bytes": int(video_row.get("size") or 0),
            "video_available": True,
            "gaze_remote_path": gaze_remote,
            "gaze_available": gaze_remote in files,
            "vrs_remote_path": vrs_remote,
            "vrs_available": vrs_remote in files,
            "category_counts": {
                category: question_counts[participant, session, category]
                for category in (
                    "object_location_memory",
                    "visual_recall",
                    "intent_recall",
                    "conversational_memory",
                )
            },
            **video_metadata[video_path],
        }

    gaze_jobs = []
    for video_id, row in metadata_by_video.items():
        if not row["gaze_available"]:
            continue
        participant = row["participant"]
        stem = _session_stem(participant, video_id)
        local_raw = (
            data_root
            / "raw"
            / "mps"
            / f"person_{participant}"
            / f"{stem}_vrs"
            / "eye_gaze"
            / "general_eye_gaze.csv"
        )
        if local_raw.is_file():
            local_path = local_raw
        else:
            local_path = (
                data_root
                / "scouting"
                / "mps"
                / f"person_{participant}"
                / f"{stem}_vrs"
                / "eye_gaze"
                / "general_eye_gaze.csv"
            )
            expected_size = int(files[row["gaze_remote_path"]].get("size") or 0)
            gaze_jobs.append(
                (row["gaze_remote_path"], local_path, expected_size)
            )
        row["gaze_path"] = str(local_path)

    def fetch_gaze(job: tuple[str, Path, int]) -> None:
        _download_file(job[0], job[1], job[2])

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(fetch_gaze, gaze_jobs))

    for row in metadata_by_video.values():
        if row["gaze_available"]:
            row.update(_gaze_metrics(Path(row["gaze_path"])))
        else:
            row.update(
                {
                    "gaze_coverage": 0.0,
                    "gaze_valid_fraction": 0.0,
                    "gaze_samples": 0,
                }
            )
    inventory = sorted(
        metadata_by_video.values(),
        key=lambda row: (row["participant"], row["session"]),
    )
    output_path = data_root / "proc" / "qa" / "session_inventory.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    return inventory
