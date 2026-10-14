from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .manifest import write_manifest


def assemble_transfer(
    data_root: Path,
    project_root: Path,
    output_dir: Path,
    hardlink_media: bool = False,
) -> dict[str, Any]:
    """Copy the selected-session package. With ``hardlink_media`` the large,
    immutable media (MP4s and saliency JPEGs) are hard-linked instead of copied
    (same filesystem only); every other file is copied."""
    data_root = data_root.resolve()
    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Transfer directory already exists; refusing to overwrite it: {output_dir}"
        )

    selected_path = data_root / "proc/qa/selected_sessions.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if not selected:
        raise ValueError(f"No selected sessions in {selected_path}")
    output_dir.mkdir(parents=True)
    copied: list[str] = []

    def copy_file(
        source: Path, relative: Path, required: bool = False, media: bool = False
    ) -> None:
        if not source.is_file():
            if required:
                raise FileNotFoundError(f"Required transfer input is missing: {source}")
            return
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if media and hardlink_media:
            os.link(source, destination)
        else:
            shutil.copy2(source, destination)
        copied.append(relative.as_posix())

    for session in selected:
        participant = int(session["participant"])
        video_id = str(session["video_id"])
        stem = video_id.replace(
            f"Person_{participant}_", f"person_{participant}_", 1
        )
        copy_file(
            data_root
            / "raw/video"
            / f"Person_{participant}"
            / f"{video_id}.mp4",
            Path("videos") / f"{video_id}.mp4",
            required=True,
            media=True,
        )
        copy_file(
            data_root / "proc/video" / f"{video_id}_frame_timestamps.csv",
            Path("videos") / f"{video_id}_frame_timestamps.csv",
            required=True,
        )
        for filename in ("gaze_px.parquet", "fixations.parquet"):
            copy_file(
                data_root / "proc/gaze" / video_id / filename,
                Path("gaze") / video_id / filename,
                required=True,
            )
        for filename in ("fixation_labels.parquet", "episodes.parquet", "fixations_idt.parquet"):
            copy_file(
                data_root / "proc/gaze" / video_id / filename,
                Path("gaze") / video_id / filename,
            )
        copy_file(
            data_root / "proc/calib" / f"{video_id}.json",
            Path("calib") / f"{video_id}.json",
            required=True,
        )

        transcript_dir = data_root / "raw/transcripts" / f"person_{participant}"
        if transcript_dir.is_dir():
            for transcript in sorted(transcript_dir.glob(f"{stem}_*transcript.json")):
                copy_file(
                    transcript,
                    Path("transcripts") / f"person_{participant}" / transcript.name,
                )

        saliency_dir = data_root / "proc/saliency" / video_id
        if saliency_dir.is_dir():
            for saliency_file in sorted(saliency_dir.rglob("*")):
                if saliency_file.is_file():
                    copy_file(
                        saliency_file,
                        Path("saliency") / video_id / saliency_file.relative_to(saliency_dir),
                        media=True,
                    )

    for filename in (
        "qa_clean.jsonl",
        "qa_subset.jsonl",
        "visual_targets.jsonl",
        "qa_fixation_labels.jsonl",
    ):
        copy_file(
            data_root / "proc/qa" / filename,
            Path("qa") / filename,
            required=filename in {"qa_clean.jsonl", "qa_subset.jsonl"},
        )

    for filename in ("selected_sessions.json", "session_inventory.json"):
        copy_file(
            data_root / "proc/qa" / filename,
            Path("docs") / filename,
            required=filename == "selected_sessions.json",
        )
    for filename in ("README.md", "SUBSET.md", "INVENTORY.md", "HPC_RUNBOOK.md"):
        copy_file(project_root / filename, Path("docs") / filename, required=filename != "HPC_RUNBOOK.md")
    copy_file(
        project_root / "env/ENV_REPORT.md",
        Path("docs") / "ENV_REPORT.md",
        required=True,
    )
    copy_file(
        data_root / "proc/reports/gaze_processing_report.json",
        Path("docs/gaze_processing_report.json"),
    )
    for filename in (
        "h1_1_report.json",
        "h1_2_report.json",
        "h1_3_report.json",
        "h1_3_qc_report.json",
        "h1_4_report.json",
        "fixation_robustness.json",
    ):
        copy_file(
            data_root / "proc/reports" / filename,
            Path("docs") / filename,
        )
    copy_file(
        project_root / "results/T0_fe_distribution.csv",
        Path("qa/T0_fe_distribution.csv"),
    )

    for relative_dir in (
        Path("qc/gaze_overlay"),
        Path("qc/fe_labels"),
        Path("qc/l3_gaze_overlay"),
        Path("qc/h1_none_examples"),
    ):
        source_dir = project_root / "results/qc" / relative_dir.name
        if source_dir.is_dir():
            for source in sorted(source_dir.rglob("*")):
                if source.is_file():
                    copy_file(
                        source,
                        Path("docs") / relative_dir / source.relative_to(source_dir),
                    )

    manifest_path = write_manifest(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "output": str(output_dir),
        "file_count": manifest["file_count"],
        "total_size_bytes": manifest["total_size_bytes"],
        "copied_file_count": len(copied),
        "manifest": str(manifest_path),
    }
