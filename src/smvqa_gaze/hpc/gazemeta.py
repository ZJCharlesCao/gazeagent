"""Gaze-derived memory structure: C5 metadata, GazeDB and C6 segmentation.

Fixation labels come from H1.1. Labels judged 'none' are treated as
*unnamed gaze*: they never enter C5 focus lists, never create C6 boundaries,
but their dwell still counts towards C6 frame weights (README 7.7 defines the
weight by fixation dwell, not by labels).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import COVERAGE_ENCODER_ID, FRAME_SIZE, atomic_write_text
from .data import Dataset, SessionInfo, format_clock
from .gaze_source import is_none_label, normalize_label

FIRST_SEEN_WINDOW_S = 600.0
NOVEL_WEIGHT = 1.0
REPEAT_WEIGHT = 0.3
MIN_FOCUS_DWELL_S = 0.5
MAX_FOCUS = 5
BOUNDARY_PERSIST_S = 2.0
MIN_SEGMENT_S = 5.0
MAX_SEGMENT_S = 60.0
MAX_FRAMES_PER_UNIT = 60
LABEL_MERGE_SIMILARITY = 0.85


class LabelIndex:
    """Canonical label ids: labels with embedding cosine > 0.85 are the same
    target (the H1.1 episode-merging rule)."""

    def __init__(self, mapping: dict[str, int]) -> None:
        self.mapping = mapping

    def canonical(self, label: Any) -> int | None:
        if is_none_label(label):
            return None
        return self.mapping.get(normalize_label(label))

    @classmethod
    def build(cls, labels: list[str], cache: Path, encoder_id: str = COVERAGE_ENCODER_ID) -> "LabelIndex":
        unique = sorted({normalize_label(label) for label in labels if not is_none_label(label)})
        if cache.is_file():
            stored = json.loads(cache.read_text(encoding="utf-8"))
            if set(stored) == set(unique):
                return cls({key: int(value) for key, value in stored.items()})
        mapping: dict[str, int] = {}
        if unique:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(encoder_id, device="cpu")
            vectors = encoder.encode(unique, normalize_embeddings=True, batch_size=256)
            centroids: list[np.ndarray] = []
            for label, vector in zip(unique, vectors, strict=True):
                if centroids:
                    similarities = np.stack(centroids) @ vector
                    best = int(np.argmax(similarities))
                    if similarities[best] > LABEL_MERGE_SIMILARITY:
                        mapping[label] = best
                        continue
                centroids.append(vector)
                mapping[label] = len(centroids) - 1
        cache.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(cache, json.dumps(mapping, indent=0) + "\n")
        return cls(mapping)


@dataclass
class SessionGaze:
    session: SessionInfo
    labeled: Any  # DataFrame: fix_id, t_on, t_off, dur_ms, label, box_px, canon
    fixations: Any  # DataFrame: all I-VT fixations on the video time axis
    episodes: Any


def load_session_gaze(dataset: Dataset, session: SessionInfo, labels: LabelIndex | None) -> SessionGaze:
    import pandas as pd

    directory = dataset.paths.gaze_dir(session.video_id)
    labeled = pd.read_parquet(directory / "fixation_labels.parquet").sort_values("t_on")
    fixations = pd.read_parquet(directory / "fixations.parquet").sort_values("t_on")
    if "t_on_us" not in fixations.columns:
        raise ValueError(f"{directory}/fixations.parquet predates the shared video time axis")
    episodes = pd.read_parquet(directory / "episodes.parquet").sort_values("t_on")
    if labels is not None:
        labeled = labeled.assign(canon=[labels.canonical(value) for value in labeled["label"]])
    return SessionGaze(session, labeled.reset_index(drop=True), fixations.reset_index(drop=True), episodes)


def build_label_index(dataset: Dataset) -> LabelIndex:
    import pandas as pd

    labels: list[str] = []
    for session in dataset.sessions:
        frame = pd.read_parquet(dataset.paths.gaze_dir(session.video_id) / "fixation_labels.parquet", columns=["label"])
        labels.extend(str(value) for value in frame["label"])
    return LabelIndex.build(labels, dataset.paths.work / "cache/label_index.json")


def _first_seen_flags(gaze: SessionGaze) -> np.ndarray:
    """True when the canonical label did not occur in the previous 10 minutes."""
    last_seen: dict[int, float] = {}
    flags = np.zeros(len(gaze.labeled), dtype=bool)
    for index, row in enumerate(gaze.labeled.itertuples(index=False)):
        canon = row.canon
        if canon is None or (isinstance(canon, float) and math.isnan(canon)):
            continue
        previous = last_seen.get(int(canon))
        flags[index] = previous is None or row.t_on - previous > FIRST_SEEN_WINDOW_S
        last_seen[int(canon)] = float(row.t_off)
    return flags


def clip_gaze_meta(gaze: SessionGaze, start_s: float, end_s: float) -> dict[str, Any]:
    """README section 7.6 gaze_meta for one memory unit."""
    labeled = gaze.labeled
    first_seen = gaze.__dict__.setdefault("_first_seen", _first_seen_flags(gaze))
    midpoints = (labeled["t_on"].to_numpy() + labeled["t_off"].to_numpy()) / 2
    inside = np.flatnonzero((midpoints >= start_s) & (midpoints < end_s))
    groups: dict[int, dict[str, Any]] = {}
    for index in inside:
        row = labeled.iloc[index]
        canon = row["canon"]
        if canon is None or (isinstance(canon, float) and math.isnan(canon)):
            continue
        group = groups.setdefault(
            int(canon),
            {"labels": {}, "dwell_ms": 0.0, "n_fix": 0, "first_seen": False, "box": None, "first_t": row["t_on"]},
        )
        group["labels"][row["label"]] = group["labels"].get(row["label"], 0.0) + float(row["dur_ms"])
        group["dwell_ms"] += float(row["dur_ms"])
        group["n_fix"] += 1
        group["first_seen"] = group["first_seen"] or bool(first_seen[index])
        box = [float(value) for value in row["box_px"]]
        group["box"] = box if group["box"] is None else [
            min(group["box"][0], box[0]),
            min(group["box"][1], box[1]),
            max(group["box"][2], box[2]),
            max(group["box"][3], box[3]),
        ]
    focus = []
    for group in groups.values():
        dwell_s = group["dwell_ms"] / 1000
        if dwell_s < MIN_FOCUS_DWELL_S:
            continue
        label = max(group["labels"].items(), key=lambda item: item[1])[0]
        box = [min(max(value / FRAME_SIZE, 0.0), 1.0) for value in group["box"]]
        focus.append(
            {
                "label": label,
                "dwell_s": round(dwell_s, 1),
                "n_fix": group["n_fix"],
                "first_seen": group["first_seen"],
                "box_norm": [round(value, 2) for value in box],
            }
        )
    focus.sort(key=lambda item: -item["dwell_s"])
    focus = focus[:MAX_FOCUS]
    all_fix = gaze.fixations
    fix_mid = (all_fix["t_on"].to_numpy() + all_fix["t_off"].to_numpy()) / 2
    n_fix = int(np.count_nonzero((fix_mid >= start_s) & (fix_mid < end_s)))
    summary = (
        "User mainly looked at: "
        + "; ".join(f"{item['label']} ({item['dwell_s']}s)" for item in focus)
        + "."
        if focus
        else "User's gaze did not settle on a nameable object."
    )
    return {
        "clip": [format_clock(start_s), format_clock(end_s)],
        "focus": focus,
        "gaze_shift_count": max(0, n_fix - 1),
        "summary": summary,
    }


def gaze_db_entries(gaze: SessionGaze, labels: LabelIndex) -> list[dict[str, Any]]:
    """Video-RAG GazeDB: one entry per named gaze episode (label + time)."""
    entries = []
    for row in gaze.episodes.itertuples(index=False):
        if is_none_label(row.label):
            continue
        entries.append(
            {
                "video_id": gaze.session.video_id,
                "start": float(row.t_on),
                "end": float(row.t_off),
                "text": f"User looked at {row.label} at {format_clock(row.t_on)} "
                f"for {row.total_dwell_ms / 1000:.1f}s.",
            }
        )
    return entries


def frame_labels(gaze: SessionGaze, times: list[float]) -> list[str | None]:
    """H1.1 target label under the gaze at each frame time (C3 archive)."""
    t_on = gaze.labeled["t_on"].to_numpy()
    t_off = gaze.labeled["t_off"].to_numpy()
    output = []
    for t in times:
        index = int(np.searchsorted(t_on, t, side="right")) - 1
        if index >= 0 and t <= t_off[index] + 0.5:
            label = gaze.labeled.iloc[index]["label"]
            output.append(None if is_none_label(label) else str(label))
        else:
            output.append(None)
    return output


# --------------------------------------------------------------------- C6


def candidate_boundaries(gaze: SessionGaze) -> list[float]:
    """A boundary starts each run of a new (named) label lasting >= 2 s."""
    runs: list[tuple[int, float, float]] = []
    for row in gaze.labeled.itertuples(index=False):
        canon = row.canon
        if canon is None or (isinstance(canon, float) and math.isnan(canon)):
            continue  # 'none' is transparent
        canon = int(canon)
        if runs and runs[-1][0] == canon:
            runs[-1] = (canon, runs[-1][1], float(row.t_off))
        else:
            runs.append((canon, float(row.t_on), float(row.t_off)))
    return [
        start
        for index, (canon, start, end) in enumerate(runs)
        if index > 0 and end - start >= BOUNDARY_PERSIST_S
    ]


def constrain_segments(boundaries: list[float], duration_s: float) -> list[tuple[float, float]]:
    """Merge boundaries closer than 5 s; split segments longer than 60 s at midpoints."""
    kept = [0.0]
    for boundary in sorted(boundaries):
        if boundary - kept[-1] >= MIN_SEGMENT_S and duration_s - boundary >= MIN_SEGMENT_S:
            kept.append(boundary)
    kept.append(duration_s)
    segments: list[tuple[float, float]] = []
    pending = [(kept[i], kept[i + 1]) for i in range(len(kept) - 1)]
    while pending:
        start, end = pending.pop(0)
        if end - start > MAX_SEGMENT_S:
            middle = (start + end) / 2
            pending[:0] = [(start, middle), (middle, end)]
        else:
            segments.append((start, end))
    return segments


def fixation_weights(gaze: SessionGaze) -> tuple[np.ndarray, np.ndarray]:
    """Per I-VT fixation: midpoint time and dwell x novelty (1 new, 0.3 repeat).

    Fixations without a named label (shorter than 200 ms or 'none') count as
    repeats, so unnamed gaze cannot dominate the allocation.
    """
    first_seen = gaze.__dict__.setdefault("_first_seen", _first_seen_flags(gaze))
    novel_by_fix = {
        int(fix_id): bool(flag) and canon is not None and not (isinstance(canon, float) and math.isnan(canon))
        for fix_id, flag, canon in zip(gaze.labeled["fix_id"], first_seen, gaze.labeled["canon"], strict=True)
    }
    fix = gaze.fixations
    mid = (fix["t_on"].to_numpy() + fix["t_off"].to_numpy()) / 2
    weights = np.array(
        [
            dur / 1000 * (NOVEL_WEIGHT if novel_by_fix.get(int(fix_id), False) else REPEAT_WEIGHT)
            for fix_id, dur in zip(fix["fix_id"], fix["dur_ms"], strict=True)
        ]
    )
    return mid, weights


def allocate_frames(
    segments: list[tuple[float, float]],
    weights: list[float],
    total: int,
    alpha: float,
    minimum: int = 1,
    maximum: int | None = MAX_FRAMES_PER_UNIT,
) -> list[int]:
    """n_i = N [(1-a) d_i/D + a w_i/W], >= 1 each, rounded to sum to N."""
    durations = np.array([end - start for start, end in segments], dtype=float)
    weight = np.array(weights, dtype=float)
    share_time = durations / durations.sum()
    share_gaze = weight / weight.sum() if weight.sum() > 0 else share_time
    share = (1 - alpha) * share_time + alpha * share_gaze
    if total < minimum * len(segments):
        raise ValueError("Frame budget is smaller than one frame per segment")
    raw = total * share
    counts = np.maximum(np.floor(raw).astype(int), minimum)
    if maximum is not None:
        counts = np.minimum(counts, maximum)
    # Largest-remainder adjustment towards the exact total.
    while counts.sum() != total:
        if counts.sum() < total:
            room = raw - counts
            if maximum is not None:
                room[counts >= maximum] = -np.inf
            index = int(np.argmax(room))
            if not math.isfinite(room[index]):
                break
            counts[index] += 1
        else:
            surplus = counts - raw
            surplus[counts <= minimum] = -np.inf
            index = int(np.argmax(surplus))
            if not math.isfinite(surplus[index]):
                break
            counts[index] -= 1
    return counts.tolist()


@dataclass(frozen=True)
class Unit:
    """A memory-writing unit: a 30 s clip (C0) or a gaze event segment (C6)."""

    video_id: str
    index: int
    start: float
    end: float
    frame_times: tuple[float, ...]
    max_tokens: int

    @property
    def key(self) -> str:
        return f"{self.video_id}:{self.index:05d}"


def c6_units(
    gaze: SessionGaze,
    alpha: float,
    caption_tokens: int,
    clip_seconds: int,
    c0_frames: int,
    max_frames: int = MAX_FRAMES_PER_UNIT,
) -> list[Unit]:
    from .frames import uniform_times

    session = gaze.session
    segments = constrain_segments(candidate_boundaries(gaze), session.duration_s)
    mid, weights = fixation_weights(gaze)
    segment_weights = [
        float(weights[(mid >= start) & (mid < end)].sum()) for start, end in segments
    ]
    counts = allocate_frames(segments, segment_weights, c0_frames, alpha, maximum=max_frames)
    units = []
    for index, ((start, end), count) in enumerate(zip(segments, counts, strict=True)):
        units.append(
            Unit(
                session.video_id,
                index,
                start,
                end,
                tuple(uniform_times(start, end, count)),
                max(16, round(caption_tokens * count / clip_seconds)),
            )
        )
    return units


def weighted_times(
    gaze: SessionGaze, start_s: float, end_s: float, count: int, alpha: float
) -> list[float]:
    """Video-RAG C6: place `count` frames inside a shard window by the C6
    density (time share mixed with gaze-weight share), at CDF quantiles."""
    segments = [
        (max(start, start_s), min(end, end_s))
        for start, end in constrain_segments(candidate_boundaries(gaze), gaze.session.duration_s)
        if end > start_s and start < end_s
    ]
    if not segments or count <= 0:
        return []
    mid, weights = fixation_weights(gaze)
    durations = np.array([end - start for start, end in segments])
    gaze_weight = np.array([weights[(mid >= s) & (mid < e)].sum() for s, e in segments])
    share = (1 - alpha) * durations / durations.sum() + alpha * (
        gaze_weight / gaze_weight.sum() if gaze_weight.sum() > 0 else durations / durations.sum()
    )
    cdf = np.concatenate([[0.0], np.cumsum(share)])
    times = []
    for k in range(count):
        q = (k + 0.5) / count
        index = min(int(np.searchsorted(cdf, q, side="right")) - 1, len(segments) - 1)
        fraction = (q - cdf[index]) / max(share[index], 1e-12)
        start, end = segments[index]
        times.append(start + fraction * (end - start))
    return times

