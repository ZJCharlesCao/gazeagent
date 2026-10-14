"""Command-line entry points for the HPC stages (H0-H4)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from .config import CONDITIONS, PIPELINES, TIER_SIDE, Paths, RunConfig

BETA_GRID = (0.0, 0.25, 0.5, 1.0)


def _paths(args: argparse.Namespace) -> Paths:
    data = Path(args.data or os.environ.get("SMVQA_TRANSFER", "transfer")).resolve()
    work = Path(args.work or os.environ.get("SMVQA_WORK", "work")).resolve()
    return Paths(data, work)


def _config(args: argparse.Namespace) -> RunConfig:
    config = RunConfig.load(Path(args.config) if getattr(args, "config", None) else None)
    if getattr(args, "base_url", None):
        config.base_url = args.base_url
    if getattr(args, "concurrency", None):
        config.request_concurrency = args.concurrency
    return config


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


# ------------------------------------------------------------------- H0/prep


def fetch_models(args: argparse.Namespace) -> None:
    """Download every model on a node with internet (compute nodes may lack it)."""
    from huggingface_hub import snapshot_download

    from .config import CLIP_ID, COVERAGE_ENCODER_ID, DETECTOR_ID, MODEL_ID, RETRIEVER_ID

    for model in (MODEL_ID, RETRIEVER_ID, COVERAGE_ENCODER_ID, CLIP_ID, DETECTOR_ID):
        print(f"downloading {model}", flush=True)
        snapshot_download(model)
    import easyocr

    easyocr.Reader(["en"], gpu=False, verbose=False)
    print("All models cached under HF_HOME and ~/.EasyOCR", flush=True)


def prepare(args: argparse.Namespace) -> None:
    from ..manifest import verify_manifest
    from .config import write_stage_manifest
    from .data import Dataset
    from .frames import extract_all_frames
    from .gaze_source import SaliencyBoxes, check_shift_validity
    from .gazemeta import build_label_index

    paths = _paths(args)
    config = _config(args)
    if not args.skip_verify:
        errors = verify_manifest(paths.data)
        if errors:
            raise SystemExit("Transfer package failed SHA-256 verification:\n" + "\n".join(errors[:20]))
        print("MANIFEST.json verified", flush=True)
    dataset = Dataset(paths)
    frames = extract_all_frames(dataset, workers=args.workers)
    shift = check_shift_validity(dataset, config, paths.work / "cache/shift_check.json")
    labels = build_label_index(dataset)
    for session in dataset.sessions:
        boxes = SaliencyBoxes(paths.saliency_dir(session.video_id), paths.work / "cache/saliency_boxes")
        for side in sorted(set(TIER_SIDE.values())):
            boxes.box(0.0, side)
    write_stage_manifest(
        paths.work / "cache",
        "hpc-prepare",
        inputs=[paths.data / "MANIFEST.json", paths.sessions],
        parameters={"workers": args.workers},
        outputs={
            "frame_sessions": len(frames),
            "shift_check": [row["shift_s"] for row in shift["sessions"]],
            "label_clusters": len(set(labels.mapping.values())),
        },
    )
    _print({"sessions": len(frames), "shift": shift["sessions"][:3], "labels": len(labels.mapping)})


def features(args: argparse.Namespace) -> None:
    """Precompute Video-RAG OCR and CLIP features (run with vLLM stopped or at
    gpu-memory-utilization 0.70)."""
    from .data import Dataset
    from .features import ClipIndex, OcrIndex
    from .frames import FrameStore
    from .gaze_source import GazeSources

    paths = _paths(args)
    config = _config(args)
    dataset = Dataset(paths)
    frames = FrameStore(paths)
    sources = GazeSources(dataset, config)
    ocr = OcrIndex(dataset, config, frames, sources)
    clip = ClipIndex(dataset, frames, sources)
    for variant in args.variants.split(","):
        for session in dataset.sessions:
            ocr.entries(variant, session)
            clip.image_vectors(variant, session)
            print(f"features {variant} {session.video_id} done", flush=True)


def token_audit_all(args: argparse.Namespace) -> None:
    """README 7.2: processor-measured visual tokens for every condition x tier."""
    import csv

    from .common import RunContext, token_audit
    from .data import Dataset

    paths = _paths(args)
    config = _config(args)
    dataset = Dataset(paths)
    session = dataset.sessions[0]
    rows = []
    for name, condition in CONDITIONS.items():
        for tier in condition.tiers:
            ctx = RunContext(dataset, config, "egobutler", tier, condition)
            rendered = ctx.render(session.video_id, [10.0, 11.0], CONDITIONS[condition.caption_from] if condition.caption_from else condition)
            audit = token_audit(ctx, rendered)
            rows.append({"condition": name, "tier": tier, **{k: v for k, v in audit.items() if k != "image_sizes"}, "image_sizes": audit["image_sizes"]})
            ctx.frames.close()
    output = paths.results / "token_audit.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _print(rows)


# ------------------------------------------------------------------- H3


def _tuned_beta(paths: Paths, pipeline: str) -> float | None:
    path = paths.work / f"cache/beta_{pipeline}.json"
    return json.loads(path.read_text())["beta"] if path.is_file() else None


def run(args: argparse.Namespace) -> None:
    from .runner import build_context, evaluate_run, run_one

    paths = _paths(args)
    config = _config(args)
    stages = set(args.stages.split(","))
    condition = CONDITIONS[args.condition]
    if condition.gaze_meta and "answer" in stages and args.split == "test" and args.beta is None:
        beta = _tuned_beta(paths, args.pipeline)
        if beta is None:
            beta = tune_beta(paths, config, args.pipeline, args.tier, args.condition)
        config.beta = beta
    if args.beta is not None:
        config.beta = args.beta
    ctx = build_context(paths, config, args.pipeline, args.tier, args.condition, args.split, args.limit, args.smoke_window_s)
    result = run_one(ctx, stages)
    _print(result)
    failed = (result.get("answer") or {}).get("failed", 0)
    if failed:
        raise SystemExit(f"{failed} questions failed; rerun the same command to retry them")


def tune_beta(paths: Paths, config: RunConfig, pipeline: str, tier: str, condition: str) -> float:
    """README 7.6: sweep beta on dev only, then fix it for the test run."""
    from .runner import build_context, run_one

    scores = {}
    for beta in BETA_GRID:
        tuned = dataclasses.replace(config, beta=beta)
        ctx = _beta_context(build_context(paths, tuned, pipeline, tier, condition, "dev", None, None), beta)
        result = run_one(ctx, {"memory", "answer"})
        scores[beta] = result["metrics"]["qa_acc"]
        print(f"[beta] {pipeline}/{tier}/{condition} beta={beta}: dev Acc {scores[beta]:.3f}", flush=True)
    best = max(BETA_GRID, key=lambda b: (scores[b], -abs(b - 0.5)))
    path = paths.work / f"cache/beta_{pipeline}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"beta": best, "dev_acc": {str(k): v for k, v in scores.items()}, "tier": tier}, indent=2) + "\n")
    return best


def _beta_context(ctx, beta: float):
    """Write each beta's dev predictions to its own directory."""
    from .common import RunContext

    class BetaContext(RunContext):
        @property
        def pred_dir(self):  # type: ignore[override]
            return self.paths.pred_dir(self.pipeline, self.tier, f"{self.condition.name}_dev_beta{beta}")

    return BetaContext(ctx.dataset, ctx.config, ctx.pipeline, ctx.tier, ctx.condition, split="dev")


def select_variant(args: argparse.Namespace) -> None:
    """Budget cut 3: choose the best C3 variant on dev (answering only)."""
    from .runner import build_context, run_one

    paths = _paths(args)
    config = _config(args)
    scores: dict[str, list[float]] = {}
    for hint in ("T", "V", "M"):
        for pipeline in PIPELINES:
            ctx = build_context(paths, config, pipeline, "B1", f"C3-{hint}", "dev", None, None)
            result = run_one(ctx, {"answer"} if pipeline == "egobutler" else {"memory", "answer"})
            scores.setdefault(hint, []).append(result["metrics"]["qa_acc"])
    best = max(scores, key=lambda h: sum(scores[h]) / len(scores[h]))
    path = paths.work / "cache/best_c3_variant.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"variant": best, "dev_acc": scores}, indent=2) + "\n")
    _print({"best": best, "dev_acc": scores})


def group(args: argparse.Namespace) -> None:
    from .plan import load_plan_tasks
    from .runner import run_group

    paths = _paths(args)
    tasks = load_plan_tasks(Path(args.plan))
    if args.priorities:
        wanted = set(args.priorities.split(","))
        tasks = [task for task in tasks if task.priority in wanted]
    summary = run_group(
        paths,
        tasks,
        Path(args.config) if args.config else None,
        args.max_parallel,
        args.split,
        base_url=_config(args).base_url,
        extra_args=(["--limit", str(args.limit)] if args.limit else [])
        + (["--smoke-window-s", str(args.smoke_window_s)] if args.smoke_window_s else []),
    )
    _print(summary)


def evaluate(args: argparse.Namespace) -> None:
    from .runner import evaluate_run

    paths = _paths(args)
    config = _config(args)
    results = {}
    for metrics_dir in sorted((paths.work / "preds").glob("*/*/*")):
        if (metrics_dir / "predictions.jsonl").is_file():
            results[str(metrics_dir.relative_to(paths.work / "preds"))] = evaluate_run(metrics_dir, config)["overall"]
    _print(results)


# ------------------------------------------------------------------- H2/H4


def bench(args: argparse.Namespace) -> None:
    from .bench import run_benchmark

    paths = _paths(args)
    grid = json.loads(args.grid) if args.grid else None
    summary = run_benchmark(paths, _config(args), grid, args.launch_server, args.n_clips, args.n_questions, args.port)
    _print({"best_server": summary["best_server"], "h2_gpu_hours": summary["h2_gpu_hours"]})


def plan(args: argparse.Namespace) -> None:
    from .plan import make_plan, write_plan

    paths = _paths(args)
    bench_summary = json.loads((paths.results / "bench/bench_summary.json").read_text())
    result = make_plan(paths, _config(args), bench_summary, include_p3=args.include_p3)
    path = write_plan(paths, result, bench_summary)
    _print({"plan": str(path), "total_with_margin_h": result["total_with_margin_h"], "cuts": result["cuts"]})


def analyze(args: argparse.Namespace) -> None:
    from .analysis import run_analysis

    paths = _paths(args)
    _print(run_analysis(paths, _config(args), split=args.split, make_figures=not args.no_figures))


def wait_server(args: argparse.Namespace) -> None:
    from .llm import wait_for_server

    config = _config(args)
    wait_for_server(config.base_url, args.timeout)
    print(f"vLLM ready at {config.base_url}", flush=True)


def budget(args: argparse.Namespace) -> None:
    from .runner import HARD_LIMIT_HOURS, PLANNED_HOURS, BudgetLedger

    paths = _paths(args)
    # Labels starting with "job-" always record (they are the job-level account).
    ledger = BudgetLedger(paths.logs / "budget.json", f"job-{args.label}")
    if args.heartbeat_until_pid:
        # Account a job's GPU time (e.g. dev check or features) until it ends.
        while Path(f"/proc/{args.heartbeat_until_pid}").exists():
            ledger.heartbeat()
            import time

            time.sleep(60)
    used = ledger.heartbeat() if args.heartbeat_until_pid else ledger.used()
    _print({"used_gpu_hours": round(used, 3), "planned": PLANNED_HOURS, "hard_limit": HARD_LIMIT_HOURS})


def register(subparsers: argparse._SubParsersAction) -> None:
    def common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--data", help="unpacked transfer directory (or $SMVQA_TRANSFER)")
        parser.add_argument("--work", help="work directory for caches/memory/preds/results (or $SMVQA_WORK)")
        parser.add_argument("--config", help="JSON overrides for RunConfig")
        parser.add_argument("--base-url", help="vLLM OpenAI endpoint, default http://127.0.0.1:8000/v1")
        return parser

    parser = subparsers.add_parser("hpc-fetch-models", help="download all models (login node)")
    parser.set_defaults(func=fetch_models)

    parser = common(subparsers.add_parser("hpc-prepare", help="verify transfer, cache frames, gaze checks"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-verify", action="store_true")
    parser.set_defaults(func=prepare)

    parser = common(subparsers.add_parser("hpc-features", help="Video-RAG OCR/CLIP features"))
    parser.add_argument("--variants", default="full,fovea-own-B1,fovea-own-B2")
    parser.set_defaults(func=features)

    parser = common(subparsers.add_parser("hpc-token-audit", help="processor-measured tokens per condition"))
    parser.set_defaults(func=token_audit_all)

    parser = common(subparsers.add_parser("hpc-run", help="one pipeline x tier x condition run"))
    parser.add_argument("--pipeline", choices=PIPELINES, required=True)
    parser.add_argument("--tier", choices=sorted(TIER_SIDE), required=True)
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--limit", type=int, help="dev/smoke: number of questions")
    parser.add_argument("--stages", default="memory,answer")
    parser.add_argument("--smoke-window-s", type=float, help="smoke: only build memory near the questions")
    parser.add_argument("--beta", type=float, help="override C5 beta (skips tuning)")
    parser.add_argument("--concurrency", type=int)
    parser.set_defaults(func=run)

    parser = common(subparsers.add_parser("hpc-select-variant", help="best C3 variant on dev (budget cut 3)"))
    parser.set_defaults(func=select_variant)

    parser = common(subparsers.add_parser("hpc-group", help="run plan tasks 4 at a time under the budget"))
    parser.add_argument("--plan", required=True)
    parser.add_argument("--priorities", help="e.g. P0 or P0,P1")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, help="dev rehearsal: questions per task")
    parser.add_argument("--smoke-window-s", type=float, help="dev rehearsal: memory window")
    parser.set_defaults(func=group)

    parser = common(subparsers.add_parser("hpc-evaluate", help="recompute metrics.json for all runs"))
    parser.set_defaults(func=evaluate)

    parser = common(subparsers.add_parser("hpc-bench", help="H2 throughput benchmark"))
    parser.add_argument("--launch-server", action="store_true", help="start/stop vLLM per grid point")
    parser.add_argument("--grid", help='JSON list, e.g. [{"max_num_seqs":16,"max_num_batched_tokens":32768}]')
    parser.add_argument("--n-clips", type=int, default=100)
    parser.add_argument("--n-questions", type=int, default=100)
    parser.add_argument("--port", type=int, default=8000)
    parser.set_defaults(func=bench)

    parser = common(subparsers.add_parser("hpc-plan", help="H2 compute plan from the benchmark"))
    parser.add_argument("--include-p3", action="store_true")
    parser.set_defaults(func=plan)

    parser = common(subparsers.add_parser("hpc-wait-server", help="block until vLLM answers"))
    parser.add_argument("--timeout", type=float, default=1800)
    parser.set_defaults(func=wait_server)

    parser = common(subparsers.add_parser("hpc-budget", help="show or account GPU hours"))
    parser.add_argument("--label", default="job")
    parser.add_argument("--heartbeat-until-pid", type=int)
    parser.set_defaults(func=budget)

    parser = common(subparsers.add_parser("hpc-analyze", help="H4 tables, figures, statistics, REPORT.md"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--no-figures", action="store_true")
    parser.set_defaults(func=analyze)
