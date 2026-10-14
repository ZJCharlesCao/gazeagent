import json

import pytest

from smvqa_gaze.manifest import verify_manifest
from smvqa_gaze.transfer import assemble_transfer


def _write(path, content="sample"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_transfer_contains_only_selected_processed_assets(tmp_path):
    data_root = tmp_path / "data"
    project_root = tmp_path / "project"
    output = data_root / "transfer"
    video_id = "Person_3_session_1_example"
    _write(
        data_root / "proc/qa/selected_sessions.json",
        json.dumps([{"participant": 3, "video_id": video_id}]),
    )
    stem = video_id.replace("Person_3_", "person_3_", 1)
    for path in (
        data_root / "raw/video/Person_3" / f"{video_id}.mp4",
        data_root / "proc/video" / f"{video_id}_frame_timestamps.csv",
        data_root / "proc/gaze" / video_id / "gaze_px.parquet",
        data_root / "proc/gaze" / video_id / "fixations.parquet",
        data_root / "proc/calib" / f"{video_id}.json",
        data_root / "raw/person_3" / f"{stem}.vrs",
    ):
        _write(path)
    _write(data_root / "proc/qa/qa_clean.jsonl", "{}")
    _write(data_root / "proc/qa/qa_subset.jsonl", "{}")
    for path in (
        project_root / "README.md",
        project_root / "SUBSET.md",
        project_root / "INVENTORY.md",
        project_root / "env/ENV_REPORT.md",
    ):
        _write(path)

    result = assemble_transfer(data_root, project_root, output)

    assert result["file_count"] == 12
    assert (output / "videos" / f"{video_id}.mp4").is_file()
    assert (output / "docs/selected_sessions.json").is_file()
    assert not list(output.rglob("*.vrs"))
    assert verify_manifest(output) == []


def test_transfer_includes_completed_h1_reports_and_distribution(tmp_path):
    data_root = tmp_path / "data"
    project_root = tmp_path / "project"
    output = data_root / "transfer"
    video_id = "Person_3_session_1_example"
    _write(
        data_root / "proc/qa/selected_sessions.json",
        json.dumps([{"participant": 3, "video_id": video_id}]),
    )
    _write(data_root / "proc/qa/qa_clean.jsonl", "{}")
    _write(data_root / "proc/qa/qa_subset.jsonl", "{}")
    for path in (
        data_root / "raw/video/Person_3" / f"{video_id}.mp4",
        data_root / "proc/video" / f"{video_id}_frame_timestamps.csv",
        data_root / "proc/gaze" / video_id / "gaze_px.parquet",
        data_root / "proc/gaze" / video_id / "fixations.parquet",
        data_root / "proc/calib" / f"{video_id}.json",
        project_root / "README.md",
        project_root / "SUBSET.md",
        project_root / "INVENTORY.md",
        project_root / "env/ENV_REPORT.md",
        project_root / "results/T0_fe_distribution.csv",
        data_root / "proc/reports/h1_1_report.json",
        data_root / "proc/reports/h1_2_report.json",
        data_root / "proc/reports/h1_3_report.json",
        data_root / "proc/reports/h1_4_report.json",
    ):
        _write(path)

    assemble_transfer(data_root, project_root, output)

    assert (output / "docs/h1_1_report.json").is_file()
    assert (output / "docs/h1_4_report.json").is_file()
    assert (output / "qa/T0_fe_distribution.csv").is_file()
    assert verify_manifest(output) == []


def test_transfer_refuses_to_overwrite_existing_destination(tmp_path):
    data_root = tmp_path / "data"
    project_root = tmp_path / "project"
    output = data_root / "transfer"
    output.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        assemble_transfer(data_root, project_root, output)
