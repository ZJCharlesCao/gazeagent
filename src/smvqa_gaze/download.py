from __future__ import annotations

import concurrent.futures
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DATASET_ID = "OSU-AIoT-MLSys-Lab/SuperMemory-VQA"
DATASET_REVISION = "main"
USER_AGENT = "smvqa-gaze/0.1"


def dataset_tree(prefix: str) -> list[dict[str, Any]]:
    path = urllib.parse.quote(prefix.strip("/"), safe="/")
    url = (
        f"https://huggingface.co/api/datasets/{DATASET_ID}/tree/"
        f"{DATASET_REVISION}/{path}?recursive=true&expand=false&limit=1000"
    )
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        rows = json.load(response)
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected Hugging Face tree response for {prefix}")
    return rows


def _session_stem(participant: int, video_id: str) -> str:
    prefix = f"Person_{participant}_"
    if not video_id.startswith(prefix):
        raise ValueError(
            f"video_id {video_id!r} does not belong to participant {participant}"
        )
    return video_id.replace(prefix, f"person_{participant}_", 1)


def _required_paths(session: dict[str, Any]) -> tuple[list[str], list[str]]:
    participant = int(session["participant"])
    video_id = str(session["video_id"])
    stem = _session_stem(participant, video_id)
    mps_stem = f"{stem}_vrs"
    required = [
        f"data/video/Person_{participant}/{video_id}.mp4",
        f"data/raw/person_{participant}/{stem}.vrs",
        f"data/mps/person_{participant}/{mps_stem}/eye_gaze/general_eye_gaze.csv",
    ]
    optional = [
        f"data/mps/person_{participant}/{mps_stem}/eye_gaze/summary.json",
        f"data/mps/person_{participant}/{mps_stem}/slam/online_calibration.jsonl",
        f"data/transcripts/person_{participant}/{stem}_whisper_transcript.json",
        f"data/transcripts/person_{participant}/{stem}_gemini_aligned_transcript.json",
    ]
    return required, optional


def _download_file(remote_path: str, destination: Path, expected_size: int | None) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (
        expected_size is None or destination.stat().st_size == expected_size
    ):
        return "cached"
    encoded_path = urllib.parse.quote(remote_path, safe="/")
    url = (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
        f"{DATASET_REVISION}/{encoded_path}"
    )
    temporary = destination.with_name(destination.name + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=180) as response:
                content_length = response.headers.get("Content-Length")
                declared_size = int(content_length) if content_length else expected_size
                with temporary.open("wb") as output:
                    while chunk := response.read(4 * 1024 * 1024):
                        output.write(chunk)
            actual_size = temporary.stat().st_size
            if declared_size is not None and actual_size != declared_size:
                raise IOError(
                    f"Size mismatch for {remote_path}: {actual_size} != {declared_size}"
                )
            temporary.replace(destination)
            return "downloaded"
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {remote_path}")


def download_sessions(
    sessions: list[dict[str, Any]], data_root: Path, workers: int = 4
) -> list[dict[str, Any]]:
    """Download only selected RGB video, VRS, gaze, calibration, and transcript assets."""
    people = sorted({int(session["participant"]) for session in sessions})
    available: dict[str, dict[str, Any]] = {}
    for participant in people:
        for prefix in (
            f"data/video/Person_{participant}",
            f"data/raw/person_{participant}",
            f"data/mps/person_{participant}",
            f"data/transcripts/person_{participant}",
        ):
            for row in dataset_tree(prefix):
                if row.get("type") == "file":
                    available[str(row["path"])] = row

    requests: dict[str, bool] = {}
    for session in sessions:
        required, optional = _required_paths(session)
        for remote_path in required:
            if remote_path not in available:
                raise FileNotFoundError(f"Required dataset asset is missing: {remote_path}")
            requests[remote_path] = True
        for remote_path in optional:
            if remote_path in available:
                requests[remote_path] = False

    data_root.mkdir(parents=True, exist_ok=True)
    items = sorted(requests.items())

    def transfer(item: tuple[str, bool]) -> dict[str, Any]:
        remote_path, required = item
        relative = (
            remote_path.removeprefix("data/raw/")
            if remote_path.startswith("data/raw/")
            else remote_path.removeprefix("data/")
        )
        destination = data_root / "raw" / relative
        row = available[remote_path]
        status = _download_file(remote_path, destination, row.get("size"))
        return {
            "path": str(destination),
            "remote_path": remote_path,
            "bytes": destination.stat().st_size,
            "status": status,
            "required": required,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(transfer, items))
