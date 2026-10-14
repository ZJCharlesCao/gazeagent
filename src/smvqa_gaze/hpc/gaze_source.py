from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import FRAME_SIZE, Paths, RunConfig, atomic_save_npy
from .data import Dataset, SessionInfo

# README L3.4: invalid samples inherit the previous valid value for <= 300 ms.
HOLD_PREVIOUS_S = 0.300
# README section 7.4: a missing gaze box reuses valid gaze within 1 s, else centre.
BOX_FALLBACK_S = 1.0


@dataclass(frozen=True)
class GazePoint:
    u: float
    v: float
    status: str  # valid | held | held_1s | center | saliency


def square_box(u: float, v: float, side: int, frame: int = FRAME_SIZE) -> tuple[int, int, int, int]:
    """S x S box centred on the gaze point, translated (not shrunk) into frame."""
    if side > frame:
        raise ValueError(f"Box side {side} exceeds frame size {frame}")
    x1 = int(round(u - side / 2))
    y1 = int(round(v - side / 2))
    x1 = min(max(x1, 0), frame - side)
    y1 = min(max(y1, 0), frame - side)
    return (x1, y1, x1 + side, y1 + side)


class GazeTrack:
    """Per-session projected gaze on the shared video time axis."""

    def __init__(self, t_s: np.ndarray, u: np.ndarray, v: np.ndarray, valid: np.ndarray) -> None:
        order = np.argsort(t_s)
        self.t = np.asarray(t_s, dtype=float)[order]
        self.u = np.asarray(u, dtype=float)[order]
        self.v = np.asarray(v, dtype=float)[order]
        self.valid = (
            np.asarray(valid, dtype=bool)[order]
            & np.isfinite(self.u)
            & np.isfinite(self.v)
        )
        if len(self.t) < 2:
            raise ValueError("A gaze track needs at least two samples")
        differences = np.diff(self.t)
        self.period = float(np.median(differences[differences > 0]))
        self.valid_index = np.flatnonzero(self.valid)
        self.duration = float(self.t[-1])

    @classmethod
    def load(cls, path: Path) -> "GazeTrack":
        import pandas as pd

        frame = pd.read_parquet(path)
        if "t_video_s" not in frame.columns:
            raise ValueError(f"{path} lacks t_video_s; regenerate gaze_px.parquet")
        return cls(
            frame["t_video_s"].to_numpy(),
            frame["u"].to_numpy(dtype=float),
            frame["v"].to_numpy(dtype=float),
            frame["valid"].to_numpy(dtype=bool),
        )

    def sample(self, t: float) -> GazePoint | None:
        """README L3.4 per-frame sample: nearest sample within half a period,
        or the previous valid sample within 300 ms."""
        index = int(np.searchsorted(self.t, t))
        candidates = [i for i in (index - 1, index) if 0 <= i < len(self.t)]
        nearest = min(candidates, key=lambda i: abs(self.t[i] - t))
        if abs(self.t[nearest] - t) <= self.period / 2 + 1e-9 and self.valid[nearest]:
            return GazePoint(self.u[nearest], self.v[nearest], "valid")
        position = int(np.searchsorted(self.valid_index, index, side="left")) - 1
        while position >= 0 and self.t[self.valid_index[position]] > t:
            position -= 1
        if position >= 0:
            previous = self.valid_index[position]
            if t - self.t[previous] <= HOLD_PREVIOUS_S:
                return GazePoint(self.u[previous], self.v[previous], "held")
        return None

    def box_point(self, t: float) -> GazePoint:
        """Gaze point for a gaze box (README section 7.4 fallbacks)."""
        point = self.sample(t)
        if point is not None:
            return point
        if len(self.valid_index):
            position = int(np.searchsorted(self.t[self.valid_index], t))
            near = [
                self.valid_index[i]
                for i in (position - 1, position)
                if 0 <= i < len(self.valid_index)
            ]
            best = min(near, key=lambda i: abs(self.t[i] - t))
            if abs(self.t[best] - t) <= BOX_FALLBACK_S:
                return GazePoint(self.u[best], self.v[best], "held_1s")
        return GazePoint(FRAME_SIZE / 2, FRAME_SIZE / 2, "center")


class SaliencyBoxes:
    """README section 7.9: S x S window with maximal smoothed saliency mass."""

    def __init__(self, directory: Path, cache_dir: Path, sigma_px: float = 24.0) -> None:
        self.directory = directory
        self.cache_dir = cache_dir
        self.sigma_px = sigma_px
        self._boxes: dict[int, np.ndarray] = {}

    def _compute(self, side: int) -> np.ndarray:
        import cv2

        cache = self.cache_dir / f"{self.directory.name}_side{side}.npy"
        if cache.is_file():
            return np.load(cache)
        files = sorted(self.directory.glob("frame_*.jpg"))
        if not files:
            raise FileNotFoundError(f"No saliency maps in {self.directory}")
        scale = 8
        window = side // scale
        boxes = np.zeros((len(files), 4), dtype=np.int32)
        for index, path in enumerate(files):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"Cannot read saliency map {path}")
            # Smooth at 1/8 resolution (the window search grid) for speed.
            small = cv2.resize(
                image.astype(np.float32),
                (FRAME_SIZE // scale, FRAME_SIZE // scale),
                interpolation=cv2.INTER_AREA,
            )
            small = cv2.GaussianBlur(small, (0, 0), self.sigma_px / scale)
            integral = cv2.integral(small)
            mass = (
                integral[window:, window:]
                - integral[:-window, window:]
                - integral[window:, :-window]
                + integral[:-window, :-window]
            )
            y, x = np.unravel_index(int(np.argmax(mass)), mass.shape)
            x1 = min(int(x) * scale, FRAME_SIZE - side)
            y1 = min(int(y) * scale, FRAME_SIZE - side)
            boxes[index] = (x1, y1, x1 + side, y1 + side)
        cache.parent.mkdir(parents=True, exist_ok=True)
        atomic_save_npy(cache, boxes)
        return boxes

    def box(self, t: float, side: int) -> tuple[int, int, int, int]:
        if side not in self._boxes:
            self._boxes[side] = self._compute(side)
        boxes = self._boxes[side]
        index = min(max(int(round(t)), 0), len(boxes) - 1)
        return tuple(int(value) for value in boxes[index])  # type: ignore[return-value]


def cross_pairing(dataset: Dataset) -> dict[str, str]:
    """README section 7.8 C8-cross: another participant's session, preferring a
    similar activity. Persons 8 and 10 recorded the same sessions together, so
    they are paired by session index; other participants use the person with
    the most similar session index range."""
    by_participant: dict[int, dict[int, SessionInfo]] = {}
    for session in dataset.sessions:
        by_participant.setdefault(session.participant, {})[session.session] = session
    pairing: dict[str, str] = {}
    participants = sorted(by_participant)
    for participant in participants:
        others = [p for p in participants if p != participant]
        for index, session in by_participant[participant].items():
            same_day = [
                by_participant[p][index]
                for p in others
                if index in by_participant[p] and by_participant[p][index].date == session.date
            ]
            same_index = [by_participant[p][index] for p in others if index in by_participant[p]]
            pool = same_day or same_index or [
                item for p in others for item in by_participant[p].values()
            ]
            best = min(pool, key=lambda item: (abs(item.duration_s - session.duration_s), item.video_id))
            pairing[session.video_id] = best.video_id
    return pairing


class GazeSources:
    """Resolve a gaze box for any condition source at a video time."""

    def __init__(self, dataset: Dataset, config: RunConfig) -> None:
        self.dataset = dataset
        self.config = config
        self._tracks: dict[str, GazeTrack] = {}
        self._saliency: dict[str, SaliencyBoxes] = {}
        self._pairing: dict[str, str] | None = None
        self._shift: dict[str, float] | None = None

    def track(self, video_id: str) -> GazeTrack:
        if video_id not in self._tracks:
            self._tracks[video_id] = GazeTrack.load(
                self.dataset.paths.gaze_dir(video_id) / "gaze_px.parquet"
            )
        return self._tracks[video_id]

    @property
    def pairing(self) -> dict[str, str]:
        if self._pairing is None:
            self._pairing = cross_pairing(self.dataset)
        return self._pairing

    def shift_seconds(self, video_id: str) -> float:
        if self._shift is None:
            path = self.dataset.paths.work / "cache/shift_check.json"
            self._shift = (
                {row["video_id"]: row["shift_s"] for row in json.loads(path.read_text())["sessions"]}
                if path.is_file()
                else {}
            )
        return float(self._shift.get(video_id, self.config.shift_seconds))

    def point(self, source: str, video_id: str, t: float) -> GazePoint:
        session = self.dataset.session_by_video[video_id]
        if source == "own":
            return self.track(video_id).box_point(t)
        if source in {"shift", "shift2s"}:
            delta = (
                self.config.shift2s_seconds
                if source == "shift2s"
                else self.shift_seconds(video_id)
            )
            return self.track(video_id).box_point((t - delta) % session.duration_s)
        if source == "cross":
            other = self.dataset.session_by_video[self.pairing[video_id]]
            return self.track(other.video_id).box_point(t % other.duration_s)
        raise ValueError(f"Point source {source!r} is not a gaze track")

    def box(self, source: str, video_id: str, t: float, side: int) -> tuple[tuple[int, int, int, int], GazePoint]:
        if source == "saliency":
            if video_id not in self._saliency:
                self._saliency[video_id] = SaliencyBoxes(
                    self.dataset.paths.saliency_dir(video_id),
                    self.dataset.paths.work / "cache/saliency_boxes",
                )
            box = self._saliency[video_id].box(t, side)
            center = GazePoint((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, "saliency")
            return box, center
        point = self.point(source, video_id, t)
        return square_box(point.u, point.v, side), point


def fixation_label_series(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import pandas as pd

    frame = pd.read_parquet(path).sort_values("t_on")
    labels = [normalize_label(value) for value in frame["label"].tolist()]
    return frame["t_on"].to_numpy(dtype=float), frame["t_off"].to_numpy(dtype=float), labels


def normalize_label(value: Any) -> str:
    return " ".join(str(value).casefold().strip(" .'\"").split())


def is_none_label(value: Any) -> bool:
    return normalize_label(value) in {"", "none", "nothing", "n/a"}


def label_at(t_on: np.ndarray, t_off: np.ndarray, labels: list[str], t: float) -> str | None:
    index = int(np.searchsorted(t_on, t, side="right")) - 1
    if index >= 0 and t <= t_off[index] and not is_none_label(labels[index]):
        return labels[index]
    return None


def check_shift_validity(dataset: Dataset, config: RunConfig, output: Path) -> dict[str, Any]:
    """README section 7.8: labels under the shifted gaze should agree with the
    original labels in < 20% of frames; otherwise use the 7-minute shift."""
    sessions = []
    for session in dataset.sessions:
        path = dataset.paths.gaze_dir(session.video_id) / "fixation_labels.parquet"
        t_on, t_off, labels = fixation_label_series(path)
        results = {}
        for delta in (config.shift_seconds, config.shift_fallback_seconds):
            same = total = 0
            for t in np.arange(0, session.duration_s, 1.0):
                original = label_at(t_on, t_off, labels, float(t))
                shifted = label_at(t_on, t_off, labels, float((t - delta) % session.duration_s))
                if original is None or shifted is None:
                    continue
                total += 1
                same += original == shifted
            results[delta] = same / total if total else math.nan
        chosen = (
            config.shift_seconds
            if not results[config.shift_seconds] >= 0.2
            else config.shift_fallback_seconds
        )
        sessions.append(
            {
                "video_id": session.video_id,
                "same_label_fraction": {str(key): value for key, value in results.items()},
                "shift_s": chosen,
            }
        )
    report = {"rule": "same-label fraction < 0.20 at 5 min, else 7 min", "sessions": sessions}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def gaze_fallback_stats(dataset: Dataset, times_by_video: dict[str, list[float]]) -> dict[str, float]:
    """Fraction of frames whose gaze box used each fallback (README 7.4)."""
    counts: dict[str, int] = {}
    total = 0
    sources = GazeSources(dataset, RunConfig())
    for video_id, times in times_by_video.items():
        track = sources.track(video_id)
        for t in times:
            status = track.box_point(t).status
            counts[status] = counts.get(status, 0) + 1
            total += 1
    return {key: value / total for key, value in sorted(counts.items())} if total else {}
