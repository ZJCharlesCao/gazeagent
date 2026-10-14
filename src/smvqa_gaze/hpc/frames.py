from __future__ import annotations

import concurrent.futures
import math
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from .config import FRAME_SIZE, Paths
from .data import Dataset, SessionInfo


def cached_frame_path(paths: Paths, video_id: str, second: int) -> Path:
    return paths.frame_cache / video_id / f"{second:06d}.jpg"


def extract_session_frames(video_path: Path, output_dir: Path) -> dict[str, Any]:
    """Decode a session once and cache one full-resolution JPEG per second.

    Frame i is the source frame round(i * fps), matching the 1-fps sampling of
    the UNISAL saliency maps.
    """
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    done_marker = output_dir / "_complete"
    if done_marker.is_file():
        return {"video": video_path.name, "status": "cached"}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = {round(second * fps): second for second in range(math.ceil(frame_count / fps))}
    index = 0
    written = 0
    try:
        while True:
            ok = capture.grab()
            if not ok:
                break
            if index in wanted:
                ok, frame = capture.retrieve()
                if not ok:
                    raise RuntimeError(f"Cannot decode frame {index} of {video_path}")
                path = output_dir / f"{wanted[index]:06d}.jpg"
                if not path.is_file():
                    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise RuntimeError(f"Cannot write {path}")
                written += 1
            index += 1
    finally:
        capture.release()
    done_marker.write_text(f"{written}\n", encoding="utf-8")
    return {"video": video_path.name, "frames": written, "fps": fps}


def extract_all_frames(dataset: Dataset, workers: int = 4) -> list[dict[str, Any]]:
    paths = dataset.paths
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                extract_session_frames,
                paths.video(session.video_id),
                paths.frame_cache / session.video_id,
            )
            for session in dataset.sessions
        ]
        return [future.result() for future in futures]


class FrameStore:
    """RGB 1408x1408 frames by video time; integer seconds come from the cache."""

    def __init__(self, paths: Paths, max_cached: int = 256) -> None:
        self.paths = paths
        self._lru: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._max = max_cached
        self._lock = threading.Lock()
        self._local = threading.local()

    @property
    def _captures(self) -> dict[str, Any]:
        # cv2.VideoCapture is not thread-safe: one set of handles per thread.
        if not hasattr(self._local, "captures"):
            self._local.captures = {}
        return self._local.captures

    def get(self, video_id: str, t: float) -> np.ndarray:
        import cv2

        second = int(round(t))
        if abs(t - second) < 1e-3:
            key = (video_id, second)
            with self._lock:
                if key in self._lru:
                    self._lru.move_to_end(key)
                    return self._lru[key]
            path = cached_frame_path(self.paths, video_id, second)
            image = cv2.imread(str(path)) if path.is_file() else None
            if image is None:
                image = self._decode(video_id, t)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._lru[key] = image
                if len(self._lru) > self._max:
                    self._lru.popitem(last=False)
            return image
        return self._decode(video_id, t)

    def _decode(self, video_id: str, t: float) -> np.ndarray:
        import cv2

        capture = self._captures.get(video_id)
        if capture is None:
            capture = cv2.VideoCapture(str(self.paths.video(video_id)))
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video for {video_id}")
            self._captures[video_id] = capture
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        index = min(max(int(round(t * fps)), 0), frame_count - 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot decode {video_id} at {t:.3f}s")
        if frame.shape[:2] != (FRAME_SIZE, FRAME_SIZE):
            raise ValueError(f"Unexpected frame shape {frame.shape} for {video_id}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()


def clip_times(start_s: float, end_s: float, fps: float = 1.0) -> list[float]:
    """1-fps sample times inside [start, end) on integer seconds."""
    first = math.ceil(start_s - 1e-9)
    return [float(t) for t in range(first, int(math.ceil(end_s - 1e-9))) if t < end_s]


def uniform_times(start_s: float, end_s: float, count: int) -> list[float]:
    """Uniform centres of `count` equal bins, snapped to integer seconds when
    that keeps them distinct (they then come from the 1-fps cache)."""
    if count <= 0 or end_s <= start_s:
        return []
    step = (end_s - start_s) / count
    times = [start_s + step * (index + 0.5) for index in range(count)]
    snapped = [float(min(max(round(t), math.ceil(start_s)), math.floor(end_s - 1e-6))) for t in times]
    return snapped if len(set(snapped)) == len(snapped) else times


def session_of(dataset: Dataset, video_id: str) -> SessionInfo:
    return dataset.session_by_video[video_id]
