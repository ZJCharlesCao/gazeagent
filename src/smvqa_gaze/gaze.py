from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable


@dataclass(frozen=True)
class GazeSample:
    timestamp_us: int
    yaw_rad: float
    pitch_rad: float
    depth_m: float
    uncertainty_deg: float
    valid: bool
    u: float | None = None
    v: float | None = None


@dataclass(frozen=True)
class Fixation:
    fix_id: int
    t_on_us: int
    t_off_us: int
    dur_ms: float
    yaw_rad: float
    pitch_rad: float
    sample_count: int
    u: float | None = None
    v: float | None = None


def read_gaze_csv(path: Path) -> list[GazeSample]:
    samples = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "tracking_timestamp_us",
            "left_yaw_rads_cpf",
            "right_yaw_rads_cpf",
            "pitch_rads_cpf",
            "depth_m",
            "left_yaw_low_rads_cpf",
            "right_yaw_low_rads_cpf",
            "pitch_low_rads_cpf",
            "left_yaw_high_rads_cpf",
            "right_yaw_high_rads_cpf",
            "pitch_high_rads_cpf",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected gaze CSV columns in {path}")
        for row in reader:
            yaw_l, yaw_r = float(row["left_yaw_rads_cpf"]), float(
                row["right_yaw_rads_cpf"]
            )
            yaw_low = (float(row["left_yaw_low_rads_cpf"]) + float(
                row["right_yaw_low_rads_cpf"]
            )) / 2
            yaw_high = (float(row["left_yaw_high_rads_cpf"]) + float(
                row["right_yaw_high_rads_cpf"]
            )) / 2
            pitch = float(row["pitch_rads_cpf"])
            pitch_low, pitch_high = (
                float(row["pitch_low_rads_cpf"]),
                float(row["pitch_high_rads_cpf"]),
            )
            values = (yaw_l, yaw_r, pitch, yaw_low, yaw_high, pitch_low, pitch_high)
            depth = float(row["depth_m"])
            samples.append(
                GazeSample(
                    timestamp_us=int(row["tracking_timestamp_us"]),
                    yaw_rad=(yaw_l + yaw_r) / 2,
                    pitch_rad=pitch,
                    depth_m=depth,
                    uncertainty_deg=math.degrees(
                        math.hypot(yaw_high - yaw_low, pitch_high - pitch_low) / 2
                    ),
                    valid=all(math.isfinite(value) for value in values)
                    and math.isfinite(depth)
                    and depth > 0,
                )
            )
    return samples


def gaze_coverage(samples: Iterable[GazeSample], expected_hz: float = 60.0) -> float:
    rows = list(samples)
    if len(rows) < 2:
        return 0.0
    valid = sum(sample.valid for sample in rows)
    duration_s = (rows[-1].timestamp_us - rows[0].timestamp_us) / 1_000_000
    if duration_s <= 0:
        return 0.0
    return min(1.0, valid / (duration_s * expected_hz))


def _median_filter_three(values: list[float]) -> list[float]:
    return [
        median(values[max(0, index - 1) : min(len(values), index + 2)])
        for index in range(len(values))
    ]


def detect_ivt_fixations(
    samples: list[GazeSample],
    velocity_threshold_deg_s: float = 30.0,
    merge_gap_ms: float = 75.0,
    merge_distance_deg: float = 0.5,
    minimum_duration_ms: float = 100.0,
) -> list[Fixation]:
    if len(samples) < 2:
        return []
    yaw = _median_filter_three([math.degrees(s.yaw_rad) for s in samples])
    pitch = _median_filter_three([math.degrees(s.pitch_rad) for s in samples])
    stable = [False] * len(samples)
    for index in range(1, len(samples)):
        previous, current = samples[index - 1], samples[index]
        delta_s = (current.timestamp_us - previous.timestamp_us) / 1_000_000
        if not previous.valid or not current.valid or delta_s <= 0:
            continue
        velocity = math.hypot(
            yaw[index] - yaw[index - 1], pitch[index] - pitch[index - 1]
        ) / delta_s
        stable[index] = velocity < velocity_threshold_deg_s

    ranges: list[tuple[int, int]] = []
    start = None
    for index, is_stable in enumerate(stable):
        if is_stable and start is None:
            start = max(0, index - 1)
        elif not is_stable and start is not None:
            ranges.append((start, index - 1))
            start = None
    if start is not None:
        ranges.append((start, len(samples) - 1))

    merged: list[list[int]] = []
    for start, end in ranges:
        if merged:
            prev_start, prev_end = merged[-1]
            gap_ms = (samples[start].timestamp_us - samples[prev_end].timestamp_us) / 1000
            distance_deg = math.hypot(
                yaw[start] - yaw[prev_end], pitch[start] - pitch[prev_end]
            )
            if gap_ms < merge_gap_ms and distance_deg < merge_distance_deg:
                merged[-1][1] = end
                continue
        merged.append([start, end])

    fixations = []
    for start, end in merged:
        duration_ms = (samples[end].timestamp_us - samples[start].timestamp_us) / 1000
        if duration_ms < minimum_duration_ms:
            continue
        fixations.append(
            Fixation(
                fix_id=len(fixations),
                t_on_us=samples[start].timestamp_us,
                t_off_us=samples[end].timestamp_us,
                dur_ms=duration_ms,
                yaw_rad=math.radians(median(yaw[start : end + 1])),
                pitch_rad=math.radians(median(pitch[start : end + 1])),
                sample_count=end - start + 1,
                u=_median_projected_coordinate(samples, start, end, "u"),
                v=_median_projected_coordinate(samples, start, end, "v"),
            )
        )
    return fixations


def _median_projected_coordinate(
    samples: list[GazeSample], start: int, end: int, coordinate: str
) -> float | None:
    values = [
        getattr(samples[index], coordinate)
        for index in range(start, end + 1)
        if getattr(samples[index], coordinate) is not None
    ]
    return median(values) if values else None


def detect_idt_fixations(
    samples: list[GazeSample],
    dispersion_threshold_deg: float = 1.0,
    minimum_duration_ms: float = 100.0,
) -> list[Fixation]:
    """I-DT (Salvucci & Goldberg 2000) robustness check, README L3.5.

    Dispersion is (max - min yaw) + (max - min pitch) in degrees over a window
    of at least ``minimum_duration_ms``; invalid samples split windows.
    """
    yaw = [math.degrees(sample.yaw_rad) for sample in samples]
    pitch = [math.degrees(sample.pitch_rad) for sample in samples]
    fixations: list[Fixation] = []
    start = 0
    count = len(samples)

    def dispersion(first: int, last: int) -> float:
        window_yaw = yaw[first : last + 1]
        window_pitch = pitch[first : last + 1]
        return (max(window_yaw) - min(window_yaw)) + (max(window_pitch) - min(window_pitch))

    while start < count:
        if not samples[start].valid:
            start += 1
            continue
        end = start
        while (
            end + 1 < count
            and samples[end + 1].valid
            and (samples[end].timestamp_us - samples[start].timestamp_us) / 1000 < minimum_duration_ms
        ):
            end += 1
        duration_ms = (samples[end].timestamp_us - samples[start].timestamp_us) / 1000
        if duration_ms < minimum_duration_ms or dispersion(start, end) > dispersion_threshold_deg:
            start += 1
            continue
        while end + 1 < count and samples[end + 1].valid and dispersion(start, end + 1) <= dispersion_threshold_deg:
            end += 1
        fixations.append(
            Fixation(
                fix_id=len(fixations),
                t_on_us=samples[start].timestamp_us,
                t_off_us=samples[end].timestamp_us,
                dur_ms=(samples[end].timestamp_us - samples[start].timestamp_us) / 1000,
                yaw_rad=math.radians(median(yaw[start : end + 1])),
                pitch_rad=math.radians(median(pitch[start : end + 1])),
                sample_count=end - start + 1,
                u=_median_projected_coordinate(samples, start, end, "u"),
                v=_median_projected_coordinate(samples, start, end, "v"),
            )
        )
        start = end + 1
    return fixations
