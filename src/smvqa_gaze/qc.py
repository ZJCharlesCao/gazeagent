"""README L3.4-L3.6 checks: I-DT robustness fixations and gaze overlays."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def write_idt_fixations(data_root: Path, sessions_path: Path) -> dict[str, Any]:
    """I-DT fixations per selected session plus a comparison with I-VT."""
    import pandas as pd

    from .gaze import GazeSample, detect_idt_fixations, read_gaze_csv
    from .projection import FIXATION_COLUMNS

    sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    report = []
    for session in sessions:
        participant = int(session["participant"])
        video_id = str(session["video_id"])
        stem = video_id.replace(f"Person_{participant}_", f"person_{participant}_", 1)
        gaze_csv = data_root / "raw/mps" / f"person_{participant}" / f"{stem}_vrs/eye_gaze/general_eye_gaze.csv"
        gaze_dir = data_root / "proc/gaze" / video_id
        projected = pd.read_parquet(gaze_dir / "gaze_px.parquet")
        samples = read_gaze_csv(gaze_csv)
        if len(samples) != len(projected):
            raise ValueError(f"Projected gaze rows differ from CSV for {video_id}")
        u = projected["u"].to_numpy()
        v = projected["v"].to_numpy()
        valid = projected["valid"].to_numpy()
        merged = [
            GazeSample(s.timestamp_us, s.yaw_rad, s.pitch_rad, s.depth_m, s.uncertainty_deg, bool(s.valid and valid[i]), u[i], v[i])
            for i, s in enumerate(samples)
        ]
        fixations = detect_idt_fixations(merged)
        first = samples[0].timestamp_us
        rows = [
            {
                "fix_id": f.fix_id,
                "t_on": (f.t_on_us - first) / 1e6,
                "t_off": (f.t_off_us - first) / 1e6,
                "t_on_us": f.t_on_us,
                "t_off_us": f.t_off_us,
                "dur_ms": f.dur_ms,
                "u": f.u,
                "v": f.v,
            }
            for f in fixations
        ]
        pd.DataFrame(rows, columns=FIXATION_COLUMNS).to_parquet(gaze_dir / "fixations_idt.parquet", index=False)
        ivt = pd.read_parquet(gaze_dir / "fixations.parquet")
        idt = pd.DataFrame(rows)
        report.append(
            {
                "video_id": video_id,
                "ivt_count": len(ivt),
                "idt_count": len(idt),
                "ivt_median_ms": float(ivt["dur_ms"].median()),
                "idt_median_ms": float(idt["dur_ms"].median()) if len(idt) else None,
                "ivt_ge200": int((ivt["dur_ms"] >= 200).sum()),
                "idt_ge200": int((idt["dur_ms"] >= 200).sum()) if len(idt) else 0,
            }
        )
    output = {
        "sessions": report,
        "world_frame_version": "not computed: closed_loop_trajectory.csv was not downloaded (optional in README L2)",
    }
    path = data_root / "proc/reports/fixation_robustness.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def gaze_overlays(
    data_root: Path,
    sessions_path: Path,
    output_dir: Path,
    frames_per_participant: int = 20,
    clips: int = 5,
    clip_seconds: float = 10.0,
    seed: int = 2026,
) -> dict[str, Any]:
    """README L3.6: 20 random frames per participant with the per-frame gaze
    point (L3.4 rule) and 5 random 10 s clips with a gaze trail."""
    import cv2

    from .hpc.gaze_source import GazeTrack

    sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {"frames": [], "clips": []}

    def open_session(session: dict[str, Any]):
        participant = int(session["participant"])
        video_id = str(session["video_id"])
        track = GazeTrack.load(data_root / "proc/gaze" / video_id / "gaze_px.parquet")
        capture = cv2.VideoCapture(str(data_root / "raw/video" / f"Person_{participant}" / f"{video_id}.mp4"))
        return participant, video_id, track, capture

    def draw(frame, point, status):
        if point is None:
            cv2.putText(frame, "gaze missing", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
            return
        center = (int(round(point.u)), int(round(point.v)))
        cv2.circle(frame, center, 14, (0, 0, 255), -1)
        cv2.circle(frame, center, 24, (255, 255, 255), 3)
        cv2.putText(frame, status, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

    by_participant: dict[int, list[dict[str, Any]]] = {}
    for session in sessions:
        by_participant.setdefault(int(session["participant"]), []).append(session)
    for participant, rows in sorted(by_participant.items()):
        for index in range(frames_per_participant):
            session = rng.choice(rows)
            _, video_id, track, capture = open_session(session)
            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_index = rng.randrange(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            capture.release()
            if not ok:
                continue
            point = track.sample(frame_index / fps)
            draw(frame, point, point.status if point else "missing")
            path = output_dir / f"person_{participant}" / f"{video_id}_frame_{frame_index}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), cv2.resize(frame, (704, 704)), [cv2.IMWRITE_JPEG_QUALITY, 90])
            written["frames"].append(str(path))
    for index in range(clips):
        session = rng.choice(sessions)
        _, video_id, track, capture = open_session(session)
        fps = capture.get(cv2.CAP_PROP_FPS)
        start = rng.uniform(0, float(session["duration_s"]) - clip_seconds)
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(start * fps))
        path = output_dir / "clips" / f"{video_id}_{start:.0f}s.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (704, 704))
        trail: list[tuple[int, int]] = []
        for frame_number in range(round(clip_seconds * fps)):
            ok, frame = capture.read()
            if not ok:
                break
            t = start + frame_number / fps
            point = track.sample(t)
            if point is not None:
                trail = (trail + [(int(point.u), int(point.v))])[-15:]
            for a, b in zip(trail, trail[1:]):
                cv2.line(frame, a, b, (0, 255, 255), 4)
            draw(frame, point, f"{t:.2f}s")
            writer.write(cv2.resize(frame, (704, 704)))
        writer.release()
        capture.release()
        written["clips"].append(str(path))
    (output_dir / "index.json").write_text(json.dumps(written, indent=2) + "\n", encoding="utf-8")
    return {"frames": len(written["frames"]), "clips": len(written["clips"])}
