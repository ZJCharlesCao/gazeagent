from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import RunContext, condition_by_name, pred_suffix
from .config import Paths, RunConfig
from .data import Dataset
from .llm import JsonlStore
from .metrics import compute_metrics

PLANNED_HOURS = 22.0
HARD_LIMIT_HOURS = 23.5


# --------------------------------------------------------------------- budget


class BudgetLedger:
    """GPU-hour ledger shared by every HPC job (README sections 1.7 and H2.6).

    One GPU is held by each job for its whole wall time, so GPU hours equal the
    summed wall time of jobs. Each job heartbeats its own entry.
    """

    def __init__(self, path: Path, label: str) -> None:
        self.path = path
        self.job_id = f"{label}-{os.environ.get('SLURM_JOB_ID', 'local')}-{uuid.uuid4().hex[:6]}"
        self.started = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.is_file():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def heartbeat(self) -> float:
        """Record this job's wall time unless the job script already accounts
        for it (SMVQA_JOB_ACCOUNTED=1), then return total GPU hours used."""
        entries = self._load()
        if os.environ.get("SMVQA_JOB_ACCOUNTED") == "1" and not self.job_id.startswith("job-"):
            return self.used_hours(entries)
        entries[self.job_id] = {"start": self.started, "last": time.time()}
        temporary = self.path.with_name(self.path.name + f".{os.getpid()}.part")
        temporary.write_text(json.dumps(entries, indent=1) + "\n", encoding="utf-8")
        temporary.replace(self.path)
        return self.used_hours(entries)

    @staticmethod
    def used_hours(entries: dict[str, dict[str, Any]]) -> float:
        return sum(max(0.0, row["last"] - row["start"]) for row in entries.values()) / 3600

    def used(self) -> float:
        return self.used_hours(self._load())


# --------------------------------------------------------------------- one run


def build_context(
    paths: Paths,
    config: RunConfig,
    pipeline: str,
    tier: str,
    condition: str,
    split: str,
    limit: int | None,
    smoke_window_s: float | None,
) -> RunContext:
    condition_obj = condition_by_name(condition)
    if tier not in condition_obj.tiers:
        raise ValueError(f"{condition} is only defined for tiers {condition_obj.tiers}")
    return RunContext(
        Dataset(paths),
        config,
        pipeline,
        tier,
        condition_obj,
        split=split,
        limit=limit,
        smoke_window_s=smoke_window_s,
    )


def run_one(ctx: RunContext, stages: set[str]) -> dict[str, Any]:
    from .egobutler import run_egobutler
    from .videorag import run_videorag

    started = time.perf_counter()
    runner = {"egobutler": run_egobutler, "videorag": run_videorag}.get(ctx.pipeline)
    if runner is None:
        raise ValueError(f"Unknown pipeline {ctx.pipeline}")
    result, pipeline = asyncio.run(runner(ctx, stages))
    if "answer" in stages:
        result["coverage"] = write_coverage(ctx, pipeline)
        result["metrics"] = evaluate_run(ctx.pred_dir, ctx.config)["overall"]
    result["elapsed_s"] = time.perf_counter() - started
    ctx.frames.close()
    return result


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?;])\s+|\n", text) if part.strip()]


def write_coverage(ctx: RunContext, pipeline: Any) -> dict[str, Any]:
    """README 7.6 target coverage: does memory overlapping the evidence spans
    mention an evidence target (string hit or embedding similarity > 0.7)?"""
    from sentence_transformers import SentenceTransformer

    from .config import COVERAGE_ENCODER_ID

    store = JsonlStore(ctx.pred_dir / "coverage.jsonl")
    questions = [q for q in ctx.questions() if q["question_id"] not in store]
    if not questions:
        return {"questions": len(store.records)}
    encoder = SentenceTransformer(COVERAGE_ENCODER_ID, device="cpu")
    threshold = ctx.config.coverage_threshold
    for question in questions:
        targets = [t for t in question.get("visual_targets") or [] if t.strip()]
        texts = pipeline.memory_texts(question)
        if not targets or not any(span.get("t_start") is not None for span in question.get("answer_evidence", [])):
            covered = None
        else:
            lowered = " ".join(texts).casefold()
            covered = any(target.casefold() in lowered for target in targets)
            if not covered:
                sentences = [s for text in texts for s in _sentences(text)]
                if sentences:
                    target_vectors = encoder.encode(targets, normalize_embeddings=True)
                    sentence_vectors = encoder.encode(sentences, normalize_embeddings=True, batch_size=128)
                    covered = bool((target_vectors @ sentence_vectors.T).max() > threshold)
                else:
                    covered = False
        store.append_sync({"key": question["question_id"], "target_coverage": covered, "n_memory_texts": len(texts)})
    return {"questions": len(store.records)}


def evaluate_run(pred_dir: Path, config: RunConfig) -> dict[str, Any]:
    predictions = JsonlStore(pred_dir / "predictions.jsonl").values()
    coverage = JsonlStore(pred_dir / "coverage.jsonl")
    for row in predictions:
        extra = coverage.get(row["key"])
        row["target_coverage"] = extra["target_coverage"] if extra else None
    metrics = compute_metrics(predictions, config.ans_f1_positive)
    (pred_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


# --------------------------------------------------------------------- groups


@dataclass
class Task:
    pipeline: str
    tier: str
    condition: str
    after: list[str] = field(default_factory=list)  # hard dependencies
    sanity_after: str | None = None  # the condition's B1 run (README 8.2)
    variant_gate: bool = False  # budget cut 3: run only the best C3 variant
    estimate_h: float = 0.0
    priority: str = "P0"
    group: int = 0

    @property
    def name(self) -> str:
        return f"{self.pipeline}/{self.tier}/{self.condition}"

    def done_marker(self, paths: Paths, suffix: str = "") -> Path:
        return paths.pred_dir(self.pipeline, self.tier, self.condition + suffix) / "DONE"

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Task":
        return cls(**{key: row[key] for key in cls.__dataclass_fields__ if key in row})


def sanity_ok(paths: Paths, name: str, suffix: str = "") -> tuple[bool, str]:
    """README 8.2: a condition's B1 output must look sane before its B2 runs."""
    pipeline, tier, condition = name.split("/")
    metrics_path = paths.pred_dir(pipeline, tier, condition + suffix) / "metrics.json"
    if not metrics_path.is_file():
        return False, "metrics missing"
    overall = json.loads(metrics_path.read_text())["overall"]
    if overall.get("n", 0) == 0:
        return False, "no predictions"
    if (overall.get("parse_failure_rate") or 0) > 0.10:
        return False, f"parse failure rate {overall['parse_failure_rate']:.2f}"
    return True, "ok"


def run_group(
    paths: Paths,
    tasks: list[Task],
    config_path: Path | None,
    max_parallel: int = 4,
    split: str = "test",
    poll_s: float = 30.0,
    planned_hours: float = PLANNED_HOURS,
    hard_limit_hours: float = HARD_LIMIT_HOURS,
    base_url: str | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run tasks as subprocesses (each one vLLM client), respecting
    dependencies, B1-before-B2 sanity checks and the GPU-hour budget: after the
    planned hours no new task starts; at the hard limit everything stops.
    Predictions are appended per question, so stopped work is not lost."""
    limit = None
    if extra_args and "--limit" in extra_args:
        limit = int(extra_args[extra_args.index("--limit") + 1])
    suffix = pred_suffix(split, limit)
    ledger = BudgetLedger(paths.logs / "budget.json", "group")
    log_dir = paths.logs / "tasks"
    log_dir.mkdir(parents=True, exist_ok=True)
    best_path = paths.work / "cache/best_c3_variant.json"
    best = json.loads(best_path.read_text())["variant"] if best_path.is_file() else None
    status = {task.name: ("done" if task.done_marker(paths, suffix).is_file() else "pending") for task in tasks}
    for task in tasks:
        if task.variant_gate and status[task.name] == "pending":
            hint = task.condition.rsplit("-", 1)[-1]
            if best is None:
                status[task.name] = "waiting: run `smvqa-gaze hpc-select-variant` after P0"
            elif hint != best:
                status[task.name] = f"skipped: budget cut keeps C3-{best} only"
    running: dict[str, subprocess.Popen] = {}
    events = paths.logs / "group_events.jsonl"

    def log(event: str, **fields: Any) -> None:
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": time.time(), "event": event, **fields}) + "\n")
        print(f"[group] {event} {fields}", flush=True)

    def ready(task: Task) -> bool:
        if any(status.get(dependency, "done") != "done" for dependency in task.after):
            return False
        if task.sanity_after:
            if status.get(task.sanity_after, "done") != "done":
                return False
            ok, reason = sanity_ok(paths, task.sanity_after, suffix)
            if not ok:
                status[task.name] = f"blocked ({task.sanity_after}: {reason})"
                log("blocked", task=task.name, reason=reason)
                return False
        return True

    stop_new = False
    while True:
        used = ledger.heartbeat()
        if used >= hard_limit_hours:
            log("hard_limit", used_h=used)
            for name, process in running.items():
                process.send_signal(signal.SIGTERM)
                status[name] = "stopped_at_hard_limit"
            for process in running.values():
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
            running.clear()
            break
        if used >= planned_hours and not stop_new:
            stop_new = True
            log("planned_budget_reached", used_h=used)
        for name, process in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            del running[name]
            task = next(t for t in tasks if t.name == name)
            if code == 0:
                task.done_marker(paths, suffix).parent.mkdir(parents=True, exist_ok=True)
                task.done_marker(paths, suffix).write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
                status[name] = "done"
            else:
                status[name] = f"failed({code})"
            log("finished", task=name, code=code, used_h=used)
        launchable = [] if stop_new else [t for t in tasks if status[t.name] == "pending" and ready(t)]
        for task in launchable[: max(0, max_parallel - len(running))]:
            command = [
                sys.executable, "-m", "smvqa_gaze", "hpc-run",
                "--data", str(paths.data), "--work", str(paths.work),
                "--pipeline", task.pipeline, "--tier", task.tier,
                "--condition", task.condition, "--split", split,
            ]
            if config_path is not None:
                command += ["--config", str(config_path)]
            if base_url:
                command += ["--base-url", base_url]
            command += extra_args or []
            log_path = log_dir / f"{task.name.replace('/', '_')}.log"
            running[task.name] = subprocess.Popen(command, stdout=log_path.open("a"), stderr=subprocess.STDOUT)
            status[task.name] = "running"
            log("started", task=task.name, used_h=used)
        if not running and not launchable:
            break
        time.sleep(poll_s)
    ledger.heartbeat()
    summary = {"status": status, "used_hours": ledger.used()}
    (paths.logs / "group_status.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
