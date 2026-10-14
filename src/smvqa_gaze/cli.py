from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .download import download_sessions
from .gaze import detect_ivt_fixations, gaze_coverage, read_gaze_csv
from .h1 import (
    annotate_gaze_episodes,
    annotate_selected_gaze_episodes,
    check_projection_sessions,
    extract_qa_visual_targets,
    fe_label_qc,
    relabel_none_fixations,
    localize_qa_fixation_states,
    write_h1_report,
)
from .inventory import build_dataset_inventory
from .manifest import create_tar_archive, verify_manifest, write_manifest
from .projection import (
    export_rgb_calibration,
    project_gaze_to_rgb,
    write_fixations_parquet,
)
from .qa import load_json_rows, normalize_qa, read_jsonl, write_jsonl
from .saliency import generate_unisal_saliency
from .subset import (
    Session,
    TARGET_CATEGORIES,
    assign_splits,
    choose_prefix_subset,
    filter_subset,
)
from .transfer import assemble_transfer
from .video import export_vrs_to_mp4, write_frame_timestamp_map


def _data_root(argument: str | None) -> Path:
    return Path(argument or os.environ.get("SMVQA_DATA_ROOT", "data")).expanduser()


def _normalize_qa(args: argparse.Namespace) -> None:
    raw_path = Path(args.input)
    output_path = _data_root(args.data_root) / "proc" / "qa" / "qa_clean.jsonl"
    rows = [normalize_qa(row) for row in load_json_rows(raw_path)]
    count = write_jsonl(rows, output_path)
    print(json.dumps({"rows": count, "output": str(output_path)}, ensure_ascii=False))


def _build_subset(args: argparse.Namespace) -> None:
    data_root = _data_root(args.data_root)
    clean_path = Path(args.input) if args.input else data_root / "proc" / "qa" / "qa_clean.jsonl"
    with clean_path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    cutoffs = {
        int(participant): int(session)
        for participant, session in (item.split(":", 1) for item in args.cutoff)
    }
    selected = filter_subset(rows, cutoffs)
    split_rows = assign_splits(selected, dev_fraction=args.dev_fraction, seed=args.seed)
    output_path = data_root / "proc" / "qa" / "qa_subset.jsonl"
    count = write_jsonl(split_rows, output_path)
    counts = {}
    for row in split_rows:
        key = f"{row['category_key']}:{row['split']}"
        counts[key] = counts.get(key, 0) + 1
    print(
        json.dumps(
            {"rows": count, "cutoffs": cutoffs, "category_split_counts": counts},
            ensure_ascii=False,
            indent=2,
        )
    )


def _select_subset(args: argparse.Namespace) -> None:
    data_root = _data_root(args.data_root)
    qa_rows = read_jsonl(Path(args.input) if args.input else data_root / "proc/qa/qa_clean.jsonl")
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    session_fields = Session.__dataclass_fields__
    sessions = [
        Session(
            **{
                field: row.get(field, False)
                if field in {"video_available", "vrs_available"}
                else row[field]
                for field in session_fields
            }
        )
        for row in inventory
    ]
    candidate = choose_prefix_subset(
        qa_rows, sessions, minimum_per_category=args.minimum_per_category
    )
    cutoffs = dict(candidate.cutoffs)
    selected_rows = filter_subset(qa_rows, cutoffs)
    split_rows = assign_splits(
        selected_rows, dev_fraction=args.dev_fraction, seed=args.seed
    )
    qa_path = data_root / "proc/qa/qa_subset.jsonl"
    count = write_jsonl(split_rows, qa_path)
    selected_sessions = [
        row
        for row in inventory
        if int(row["participant"]) in cutoffs
        and int(row["session"]) <= cutoffs[int(row["participant"])]
    ]
    session_path = data_root / "proc/qa/selected_sessions.json"
    session_path.parent.mkdir(parents=True, exist_ok=True)
    session_path.write_text(
        json.dumps(selected_sessions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "qa_count": count,
                "cutoffs": cutoffs,
                "duration_hours": candidate.duration_s / 3600,
                "category_counts": dict(
                    zip(TARGET_CATEGORIES, candidate.counts, strict=True)
                ),
                "qa_output": str(qa_path),
                "session_output": str(session_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _inventory_dataset(args: argparse.Namespace) -> None:
    data_root = _data_root(args.data_root)
    qa_path = Path(args.qa) if args.qa else data_root / "proc/qa/qa_clean.jsonl"
    rows = read_jsonl(qa_path)
    inventory = build_dataset_inventory(rows, data_root, workers=args.workers)
    print(
        json.dumps(
            {
                "sessions": len(inventory),
                "video_sessions": sum(row["video_available"] for row in inventory),
                "gaze_sessions": sum(row["gaze_available"] for row in inventory),
                "inventory": str(data_root / "proc/qa/session_inventory.json"),
                "scouting_gaze_files": sum(
                    str(row.get("gaze_path", "")).startswith(str(data_root / "scouting"))
                    for row in inventory
                ),
            },
            indent=2,
        )
    )


def _download(args: argparse.Namespace) -> None:
    data_root = _data_root(args.data_root)
    sessions = json.loads(Path(args.sessions).read_text(encoding="utf-8"))
    results = download_sessions(sessions, data_root, workers=args.workers)
    manifest_path = data_root / "proc" / "download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"files": len(results), "manifest": str(manifest_path)}, indent=2))


def _prepare_selected_sessions(args: argparse.Namespace) -> None:
    data_root = _data_root(args.data_root)
    sessions_path = (
        Path(args.sessions)
        if args.sessions
        else data_root / "proc/qa/selected_sessions.json"
    )
    sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    if not sessions:
        raise ValueError(f"No selected sessions found in {sessions_path}")
    reports = []
    for session in sessions:
        participant = int(session["participant"])
        video_id = str(session["video_id"])
        stem = video_id.replace(
            f"Person_{participant}_", f"person_{participant}_", 1
        )
        video_path = (
            data_root / "raw/video" / f"Person_{participant}" / f"{video_id}.mp4"
        )
        vrs_path = data_root / "raw" / f"person_{participant}" / f"{stem}.vrs"
        gaze_csv = (
            data_root
            / "raw/mps"
            / f"person_{participant}"
            / f"{stem}_vrs/eye_gaze/general_eye_gaze.csv"
        )
        for input_path in (video_path, vrs_path, gaze_csv):
            if not input_path.is_file():
                raise FileNotFoundError(f"Required selected-session file is missing: {input_path}")

        output_dir = data_root / "proc/gaze" / video_id
        projected_path = output_dir / "gaze_px.parquet"
        fixations_path = output_dir / "fixations.parquet"
        projection_result = project_gaze_to_rgb(
            gaze_csv, vrs_path, projected_path
        )
        fixation_result = write_fixations_parquet(
            gaze_csv, projected_path, fixations_path
        )
        timestamp_result = write_frame_timestamp_map(
            video_path,
            gaze_csv,
            data_root / "proc/video" / f"{video_id}_frame_timestamps.csv",
        )
        calibration = export_rgb_calibration(
            gaze_csv,
            vrs_path,
            data_root / "proc/calib" / f"{video_id}.json",
        )
        reports.append(
            {
                "participant": participant,
                "session": int(session["session"]),
                "video_id": video_id,
                "projection": projection_result,
                "fixations": fixation_result,
                "video_timing": timestamp_result,
                "calibration_source": calibration["source"],
                "focal_lengths_px": calibration["focal_lengths_px"],
            }
        )
    report_path = data_root / "proc/reports/gaze_processing_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "sessions": len(reports),
                "report": str(report_path),
                "projected_fraction_min": min(
                    row["projection"]["projected_fraction"] for row in reports
                ),
                "fixation_count_total": sum(
                    row["fixations"]["fixation_count"] for row in reports
                ),
            },
            indent=2,
        )
    )


def _analyze_gaze(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    samples = read_gaze_csv(input_path)
    fixations = detect_ivt_fixations(samples)
    data_root = _data_root(args.data_root)
    session_id = input_path.parents[1].name
    output_path = data_root / "proc" / "gaze" / session_id / "fixations.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install the local environment to write Parquet outputs") from exc
    pd.DataFrame([fixation.__dict__ for fixation in fixations]).to_parquet(
        output_path, index=False
    )
    print(
        json.dumps(
            {
                "sample_count": len(samples),
                "valid_fraction": sum(sample.valid for sample in samples) / max(1, len(samples)),
                "coverage_at_60hz": gaze_coverage(samples),
                "fixation_count": len(fixations),
                "fixations_output": str(output_path),
                "note": "pixel projection requires the session camera calibration and VRS",
            },
            indent=2,
        )
    )


def _project_gaze(args: argparse.Namespace) -> None:
    result = project_gaze_to_rgb(
        Path(args.gaze_csv),
        Path(args.vrs),
        Path(args.output),
        make_upright=not args.keep_raw_orientation,
    )
    print(json.dumps(result, indent=2))


def _fixations(args: argparse.Namespace) -> None:
    result = write_fixations_parquet(
        Path(args.gaze_csv), Path(args.projected), Path(args.output)
    )
    print(json.dumps(result, indent=2))


def _export_calibration(args: argparse.Namespace) -> None:
    result = export_rgb_calibration(
        Path(args.gaze_csv),
        Path(args.vrs),
        Path(args.output),
    )
    print(json.dumps(result, indent=2))


def _annotate_gaze_episodes(args: argparse.Namespace) -> None:
    result = annotate_gaze_episodes(
        Path(args.video),
        Path(args.gaze),
        Path(args.fixations),
        Path(args.labels),
        Path(args.episodes),
        model_id=args.model,
        embedding_model_id=args.embedding_model,
        cuda_visible_devices=args.cuda_visible_devices,
        checkpoint_every=args.checkpoint_every,
        max_fixations=args.limit,
    )
    report_path = Path(args.report) if args.report else Path(args.episodes).with_suffix(".report.json")
    write_h1_report(result, report_path)
    print(json.dumps({**result, "report": str(report_path)}, indent=2))


def _annotate_selected_gaze_episodes(args: argparse.Namespace) -> None:
    data_root = _data_root(args.data_root)
    result = annotate_selected_gaze_episodes(
        Path(args.sessions)
        if args.sessions
        else data_root / "proc/qa/selected_sessions.json",
        data_root,
        model_id=args.model,
        embedding_model_id=args.embedding_model,
        cuda_visible_devices=args.cuda_visible_devices,
        checkpoint_every=args.checkpoint_every,
        max_fixations=args.limit,
    )
    print(json.dumps(result, indent=2))


def _check_projection(args: argparse.Namespace) -> None:
    result = check_projection_sessions(
        Path(args.sessions),
        _data_root(args.data_root),
        Path(args.output_dir),
        Path(args.report),
        sample_count=args.sample_count,
        seed=args.seed,
        model_id=args.model,
        cuda_visible_devices=args.cuda_visible_devices,
    )
    print(json.dumps(result, indent=2))


def _extract_visual_targets(args: argparse.Namespace) -> None:
    result = extract_qa_visual_targets(
        Path(args.input),
        Path(args.output),
        model_id=args.model,
        cuda_visible_devices=args.cuda_visible_devices,
        checkpoint_every=args.checkpoint_every,
    )
    print(json.dumps(result, indent=2))


def _label_fixation_states(args: argparse.Namespace) -> None:
    data_root = _data_root(args.data_root)
    result = localize_qa_fixation_states(
        Path(args.qa),
        Path(args.targets),
        data_root,
        Path(args.output),
        Path(args.distribution),
        model_id=args.model,
        cuda_visible_devices=args.cuda_visible_devices,
        checkpoint_every=args.checkpoint_every,
    )
    report_path = data_root / "proc/reports/h1_3_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["report"] = str(report_path)
    print(json.dumps(result, indent=2))


def _qc_fixation_labels(args: argparse.Namespace) -> None:
    data_root = _data_root(args.data_root)
    result = fe_label_qc(
        Path(args.labels),
        data_root,
        Path(args.output_dir),
        sample_count=args.sample_count,
        seed=args.seed,
        cuda_visible_devices=args.cuda_visible_devices,
    )
    report_path = data_root / "proc/reports/h1_3_qc_report.json"
    report_path.write_text(
        json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))


def _relabel_none(args: argparse.Namespace) -> None:
    data_root = _data_root(args.data_root)
    result = relabel_none_fixations(
        Path(args.sessions) if args.sessions else data_root / "proc/qa/selected_sessions.json",
        data_root,
        cuda_visible_devices=args.cuda_visible_devices,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "sessions"}, indent=2))


def _gaze_qc(args: argparse.Namespace) -> None:
    from .qc import gaze_overlays, write_idt_fixations

    data_root = _data_root(args.data_root)
    sessions = Path(args.sessions) if args.sessions else data_root / "proc/qa/selected_sessions.json"
    result = {
        "overlays": gaze_overlays(data_root, sessions, Path(args.output_dir)),
        "idt": [
            {k: row[k] for k in ("video_id", "ivt_count", "idt_count")}
            for row in write_idt_fixations(data_root, sessions)["sessions"]
        ],
    }
    print(json.dumps(result, indent=2))


def _generate_saliency(args: argparse.Namespace) -> None:
    data_root = _data_root(args.data_root)
    result = generate_unisal_saliency(
        Path(args.sessions)
        if args.sessions
        else data_root / "proc/qa/selected_sessions.json",
        data_root,
        Path(args.unisal_root),
        cuda_visible_devices=args.cuda_visible_devices,
    )
    print(json.dumps(result, indent=2))


def _export_video(args: argparse.Namespace) -> None:
    result = export_vrs_to_mp4(
        Path(args.vrs), Path(args.output), Path(args.log_dir)
    )
    print(json.dumps(result, indent=2))


def _write_frame_timestamps(args: argparse.Namespace) -> None:
    result = write_frame_timestamp_map(
        Path(args.video), Path(args.gaze_csv), Path(args.output)
    )
    print(json.dumps(result, indent=2))


def _manifest(args: argparse.Namespace) -> None:
    print(write_manifest(Path(args.transfer_dir)))


def _verify_manifest(args: argparse.Namespace) -> None:
    errors = verify_manifest(Path(args.transfer_dir))
    if errors:
        raise SystemExit("\n".join(errors))
    print("Manifest verified")


def _archive(args: argparse.Namespace) -> None:
    create_tar_archive(Path(args.transfer_dir), Path(args.output))
    print(args.output)


def _assemble_transfer(args: argparse.Namespace) -> None:
    data_root = _data_root(args.data_root)
    result = assemble_transfer(
        data_root,
        Path(args.project_root),
        Path(args.output) if args.output else data_root / "transfer",
        hardlink_media=args.hardlink_media,
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="smvqa-gaze")
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize = subparsers.add_parser("normalize-qa")
    normalize.add_argument("--input", required=True)
    normalize.add_argument("--data-root")
    normalize.set_defaults(func=_normalize_qa)

    subset = subparsers.add_parser("build-subset")
    subset.add_argument("--input")
    subset.add_argument("--data-root")
    subset.add_argument("--cutoff", action="append", required=True, metavar="PERSON:SESSION")
    subset.add_argument("--dev-fraction", type=float, default=0.15)
    subset.add_argument("--seed", type=int, default=2026)
    subset.set_defaults(func=_build_subset)

    select = subparsers.add_parser("select-subset")
    select.add_argument("--input")
    select.add_argument("--inventory", required=True)
    select.add_argument("--data-root")
    select.add_argument("--minimum-per-category", type=int, default=120)
    select.add_argument("--dev-fraction", type=float, default=0.15)
    select.add_argument("--seed", type=int, default=2026)
    select.set_defaults(func=_select_subset)

    inventory = subparsers.add_parser("inventory-dataset")
    inventory.add_argument("--qa")
    inventory.add_argument("--data-root")
    inventory.add_argument("--workers", type=int, default=6)
    inventory.set_defaults(func=_inventory_dataset)

    download = subparsers.add_parser("download-sessions")
    download.add_argument("--sessions", required=True)
    download.add_argument("--data-root")
    download.add_argument("--workers", type=int, default=4)
    download.set_defaults(func=_download)

    prepare = subparsers.add_parser("prepare-selected-sessions")
    prepare.add_argument("--sessions")
    prepare.add_argument("--data-root")
    prepare.set_defaults(func=_prepare_selected_sessions)

    gaze = subparsers.add_parser("analyze-gaze")
    gaze.add_argument("--input", required=True)
    gaze.add_argument("--data-root")
    gaze.set_defaults(func=_analyze_gaze)

    project = subparsers.add_parser("project-gaze")
    project.add_argument("--gaze-csv", required=True)
    project.add_argument("--vrs", required=True)
    project.add_argument("--output", required=True)
    project.add_argument("--keep-raw-orientation", action="store_true")
    project.set_defaults(func=_project_gaze)

    fixations = subparsers.add_parser("detect-fixations")
    fixations.add_argument("--gaze-csv", required=True)
    fixations.add_argument("--projected", required=True)
    fixations.add_argument("--output", required=True)
    fixations.set_defaults(func=_fixations)

    calibration = subparsers.add_parser("export-calibration")
    calibration.add_argument("--gaze-csv", required=True)
    calibration.add_argument("--vrs", required=True)
    calibration.add_argument("--output", required=True)
    calibration.set_defaults(func=_export_calibration)

    annotate = subparsers.add_parser("annotate-gaze-episodes")
    annotate.add_argument("--video", required=True)
    annotate.add_argument("--gaze", required=True)
    annotate.add_argument("--fixations", required=True)
    annotate.add_argument("--labels", required=True)
    annotate.add_argument("--episodes", required=True)
    annotate.add_argument("--report")
    annotate.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    annotate.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    annotate.add_argument("--cuda-visible-devices", default="1")
    annotate.add_argument("--checkpoint-every", type=int, default=32)
    annotate.add_argument("--limit", type=int)
    annotate.set_defaults(func=_annotate_gaze_episodes)

    annotate_selected = subparsers.add_parser("annotate-selected-gaze-episodes")
    annotate_selected.add_argument("--sessions")
    annotate_selected.add_argument("--data-root")
    annotate_selected.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    annotate_selected.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    annotate_selected.add_argument("--cuda-visible-devices", default="1")
    annotate_selected.add_argument("--checkpoint-every", type=int, default=32)
    annotate_selected.add_argument("--limit", type=int)
    annotate_selected.set_defaults(func=_annotate_selected_gaze_episodes)

    projection_qc = subparsers.add_parser("check-projection")
    projection_qc.add_argument("--sessions", required=True)
    projection_qc.add_argument("--data-root")
    projection_qc.add_argument("--output-dir", required=True)
    projection_qc.add_argument("--report", required=True)
    projection_qc.add_argument("--sample-count", type=int, default=50)
    projection_qc.add_argument("--seed", type=int, default=2026)
    projection_qc.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    projection_qc.add_argument("--cuda-visible-devices", default="1")
    projection_qc.set_defaults(func=_check_projection)

    extract_targets = subparsers.add_parser("extract-visual-targets")
    extract_targets.add_argument("--input", required=True)
    extract_targets.add_argument("--output", required=True)
    extract_targets.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    extract_targets.add_argument("--cuda-visible-devices", default="1")
    extract_targets.add_argument("--checkpoint-every", type=int, default=25)
    extract_targets.set_defaults(func=_extract_visual_targets)

    fixation_states = subparsers.add_parser("label-fixation-states")
    fixation_states.add_argument("--qa", required=True)
    fixation_states.add_argument("--targets", required=True)
    fixation_states.add_argument("--data-root")
    fixation_states.add_argument("--output", required=True)
    fixation_states.add_argument("--distribution", required=True)
    fixation_states.add_argument(
        "--model", default="IDEA-Research/grounding-dino-tiny"
    )
    fixation_states.add_argument("--cuda-visible-devices", default="1")
    fixation_states.add_argument("--checkpoint-every", type=int, default=10)
    fixation_states.set_defaults(func=_label_fixation_states)

    fe_qc = subparsers.add_parser("qc-fixation-labels")
    fe_qc.add_argument("--labels", required=True)
    fe_qc.add_argument("--data-root")
    fe_qc.add_argument("--output-dir", required=True)
    fe_qc.add_argument("--sample-count", type=int, default=150)
    fe_qc.add_argument("--seed", type=int, default=2026)
    fe_qc.add_argument("--cuda-visible-devices", default="1")
    fe_qc.set_defaults(func=_qc_fixation_labels)

    relabel = subparsers.add_parser(
        "relabel-none-fixations",
        help="optional H1.1 supplement: context crop + red dot for 'none' labels",
    )
    relabel.add_argument("--data-root")
    relabel.add_argument("--sessions")
    relabel.add_argument("--cuda-visible-devices", default="1")
    relabel.set_defaults(func=_relabel_none)

    gaze_qc = subparsers.add_parser(
        "gaze-qc", help="L3.6 overlays/clips and L3.5 I-DT robustness fixations"
    )
    gaze_qc.add_argument("--data-root")
    gaze_qc.add_argument("--sessions")
    gaze_qc.add_argument("--output-dir", default="results/qc/l3_gaze_overlay")
    gaze_qc.set_defaults(func=_gaze_qc)

    saliency = subparsers.add_parser("generate-saliency")
    saliency.add_argument("--sessions")
    saliency.add_argument("--data-root")
    saliency.add_argument("--unisal-root", required=True)
    saliency.add_argument("--cuda-visible-devices", default="1")
    saliency.set_defaults(func=_generate_saliency)

    export = subparsers.add_parser("export-video")
    export.add_argument("--vrs", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--log-dir", required=True)
    export.set_defaults(func=_export_video)

    timestamps = subparsers.add_parser("make-frame-timestamps")
    timestamps.add_argument("--video", required=True)
    timestamps.add_argument("--gaze-csv", required=True)
    timestamps.add_argument("--output", required=True)
    timestamps.set_defaults(func=_write_frame_timestamps)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--transfer-dir", required=True)
    manifest.set_defaults(func=_manifest)

    verify = subparsers.add_parser("verify-manifest")
    verify.add_argument("--transfer-dir", required=True)
    verify.set_defaults(func=_verify_manifest)

    archive = subparsers.add_parser("archive")
    archive.add_argument("--transfer-dir", required=True)
    archive.add_argument("--output", required=True)
    archive.set_defaults(func=_archive)

    assemble = subparsers.add_parser("assemble-transfer")
    assemble.add_argument("--data-root")
    assemble.add_argument("--project-root", default=".")
    assemble.add_argument("--output")
    assemble.add_argument(
        "--hardlink-media",
        action="store_true",
        help="hard-link MP4s and saliency maps instead of copying (same disk)",
    )
    assemble.set_defaults(func=_assemble_transfer)

    from .hpc.cli_hpc import register as register_hpc

    register_hpc(subparsers)

    args = parser.parse_args()
    args.func(args)
