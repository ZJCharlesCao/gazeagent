"""H2 steps 2-5: per-task GPU-hour estimates, budget cuts and run order."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import CLIP_SECONDS, CONDITIONS, PIPELINES, PRIORITY_ORDER, Paths, RunConfig
from .data import Dataset
from .runner import PLANNED_HOURS, Task

MARGIN = 1.1
TWO_IMAGE = {"C4"}
MASK_FACTOR = 1.06  # C3-M/C8-M/C9-M: +16 mask tokens per frame on B1 (~+1.6% on B2)


def _units(dataset: Dataset) -> int:
    return sum(math.ceil(session.duration_s / CLIP_SECONDS) for session in dataset.sessions)


def _hours(dataset: Dataset) -> int:
    return sum(math.ceil(session.duration_s / 3600) for session in dataset.sessions) + len(
        {(s.participant, s.date) for s in dataset.sessions}
    )


def task_estimate(
    pipeline: str, tier: str, condition: str, bench: dict[str, Any], dataset: Dataset, n_questions: int
) -> dict[str, float]:
    best = bench["best"]
    spec = CONDITIONS[condition]
    factor = MASK_FACTOR if spec.hint == "M" else 1.0
    caption_key = f"caption_{'C4' if condition in TWO_IMAGE else 'C0'}_{tier}"
    caption_s = best[caption_key]["seconds_per_item"] * factor
    text_s = best["text_summary"]["seconds_per_item"]
    answer_s = best[f"videorag_answer_C0_{tier}"]["seconds_per_item"] * factor
    if pipeline == "egobutler":
        captions = 0.0 if spec.caption_from else _units(dataset) * caption_s
        summaries = _hours(dataset) * text_s
        answers = n_questions * text_s * 0.25  # short text prompts, 32-token output
        parts = {"captions": captions, "summaries": summaries, "answers": answers}
    else:
        parts = {"answers": n_questions * answer_s, "query_objects": n_questions * text_s * 0.1}
    return {key: value / 3600 for key, value in parts.items()}


def feature_estimate(dataset: Dataset, bench: dict[str, Any], config: RunConfig, variants: int) -> float:
    features = bench.get("features") or {"ocr_s_per_frame": 0.3, "clip_s_per_frame": 0.01}
    seconds = sum(session.duration_s for session in dataset.sessions)
    ocr = seconds * config.ocr_fps * features["ocr_s_per_frame"]
    clip = seconds * features["clip_s_per_frame"]
    return variants * (ocr + clip) / 3600


def candidate_tasks(selected_conditions: list[str], tiers_by_condition: dict[str, tuple[str, ...]]) -> list[Task]:
    tasks: list[Task] = []
    for condition in selected_conditions:
        spec = CONDITIONS[condition]
        for pipeline in PIPELINES:
            for tier in tiers_by_condition.get(condition, spec.tiers):
                after = []
                if spec.caption_from and pipeline == "egobutler":
                    after.append(f"{pipeline}/{tier}/{spec.caption_from}")
                sanity = f"{pipeline}/B1/{condition}" if tier == "B2" and "B1" in tiers_by_condition.get(condition, spec.tiers) else None
                tasks.append(Task(pipeline, tier, condition, after=after, sanity_after=sanity, priority=spec.priority))
    return tasks


def make_plan(paths: Paths, config: RunConfig, bench: dict[str, Any], include_p3: bool = False) -> dict[str, Any]:
    dataset = Dataset(paths)
    n_test = len(dataset.select_questions("test"))
    n_dev = len(dataset.select_questions("dev"))
    h2_used = float(bench.get("h2_gpu_hours", 0.0))
    conditions = [name for name, spec in CONDITIONS.items() if spec.priority != "P3" or include_p3]
    tiers = {name: CONDITIONS[name].tiers for name in conditions}
    cuts: list[dict[str, Any]] = []
    variant_gate = False

    def total(tasks: list[Task]) -> float:
        fovea_variants = len({(t.tier, t.condition) for t in tasks if CONDITIONS[t.condition].visual == "fovea" and t.pipeline == "videorag"})
        features = feature_estimate(dataset, bench, config, 1 + fovea_variants)
        tuning = sum(
            sum(task_estimate(p, "B1", "C5", bench, dataset, n_dev).values()) for p in PIPELINES
        ) * 4 * 0.2  # beta sweep: answering only, 4 values on dev
        # With variant gating only one of the three hint variants runs.
        h3 = sum(t.estimate_h / (3 if t.variant_gate else 1) for t in tasks)
        return h2_used + MARGIN * (h3 + features + tuning)

    def build() -> list[Task]:
        tasks = candidate_tasks(conditions, tiers)
        for task in tasks:
            spec = CONDITIONS[task.condition]
            task.variant_gate = variant_gate and spec.priority == "P1" and spec.hint is not None
            task.estimate_h = sum(task_estimate(task.pipeline, task.tier, task.condition, bench, dataset, n_test).values())
        return tasks

    tasks = build()
    steps = [
        ("Remove P3 (all robustness runs)", lambda: [c for c in conditions if CONDITIONS[c].priority != "P3"], "P3 robustness checks are not run."),
        ("P2 keeps only C6(alpha=0) and C6(alpha=1); C4 removed", lambda: [c for c in conditions if c not in {"C4", "C6a0.5"}], "C4 (fovea+periphery) untested; alpha dose curve has two points."),
        ("C8/C9 only for the best C3 variant on dev", None, "H4 is tested for one prompt variant only."),
        ("P1 and P2 only on B1; B2 only for P0", None, "H5 (budget) is tested on P0 conditions only."),
        ("Remove P2", lambda: [c for c in conditions if CONDITIONS[c].priority != "P2"], "C4/C6 (H1 via segmentation) untested."),
    ]
    for description, change, consequence in steps:
        if total(tasks) <= PLANNED_HOURS:
            break
        if description.startswith("C8/C9"):
            variant_gate = True
        elif description.startswith("P1 and P2 only on B1"):
            tiers = {c: (("B1",) if CONDITIONS[c].priority in {"P1", "P2"} else CONDITIONS[c].tiers) for c in conditions}
        else:
            conditions = change()
            tiers = {c: tiers.get(c, CONDITIONS[c].tiers) for c in conditions}
        before = {(t.name, t.variant_gate) for t in tasks}
        tasks = build()
        if {(t.name, t.variant_gate) for t in tasks} == before:
            continue  # e.g. P3 was not planned in the first place
        cuts.append({"step": description, "consequence": consequence, "total_after_h": round(total(tasks), 2)})
    if total(tasks) > PLANNED_HOURS:
        cuts.append(
            {
                "step": "Shrink the subset (README H2 4.6): drop the participant with the largest video share per question",
                "consequence": "Requires regenerating SUBSET.md locally; plan exceeds budget until done.",
                "total_after_h": round(total(tasks), 2),
            }
        )
    order = {priority: index for index, priority in enumerate(PRIORITY_ORDER)}
    tasks.sort(key=lambda t: (t.condition != "C0", order[t.priority], t.tier, t.condition, t.pipeline))
    for index, task in enumerate(tasks):
        task.group = index // 4
    plan = {
        "server": bench["best_server"],
        "h2_gpu_hours": h2_used,
        "margin": MARGIN,
        "planned_limit_h": PLANNED_HOURS,
        "total_with_margin_h": round(total(tasks), 2),
        "n_test_questions": n_test,
        "n_units_per_condition": _units(dataset),
        "cuts": cuts,
        "variant_gate": variant_gate,
        "tasks": [asdict(task) for task in tasks],
    }
    return plan


def write_plan(paths: Paths, plan: dict[str, Any], bench: dict[str, Any]) -> Path:
    results = paths.results
    results.mkdir(parents=True, exist_ok=True)
    (results / "compute_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    server = plan["server"]
    lines = [
        "# COMPUTE_PLAN",
        "",
        "Generated by `smvqa-gaze hpc-plan` from measured H2 throughput. **Confirm with the user before running H3.**",
        "",
        "## Server",
        "",
        f"`max_num_seqs={server['max_num_seqs']}`, `max_num_batched_tokens={server['max_num_batched_tokens']}`; "
        "other flags as in scripts/start_vllm.sh (bf16, max-model-len 49152, image limit 70, prefix caching, seed 0).",
        "",
        "## Measured throughput (best configuration)",
        "",
        "| workload | s/item | prompt tok/s | errors | OOM | peak GPU MiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in bench["best"].items():
        lines.append(
            f"| {name} | {row['seconds_per_item']:.2f} | {row['prompt_tokens_per_s']:.0f} | {row['errors']} | {row['oom']} | {row.get('peak_gpu_mib', '')} |"
        )
    lines += [
        "",
        "## Budget",
        "",
        f"- H2 used: {plan['h2_gpu_hours']:.2f} GPU h",
        f"- H3 tasks + features + beta tuning, x{plan['margin']} margin: total **{plan['total_with_margin_h']:.2f} GPU h** (plan limit {plan['planned_limit_h']} h, hard limit 23.5 h)",
        "- Local H1 GPU time is reported separately in docs/ENV_REPORT.md.",
        "",
        "## Cuts",
        "",
    ]
    lines += [f"- {cut['step']} -> {cut['total_after_h']} h. Consequence: {cut['consequence']}" for cut in plan["cuts"]] or ["- None."]
    lines += ["", "## Tasks (execution order; groups of 4 run in parallel)", "", "| group | priority | task | depends on | B1 sanity gate | est. GPU h |", "|---:|---|---|---|---|---:|"]
    for task in plan["tasks"]:
        lines.append(
            f"| {task['group']} | {task['priority']} | {task['pipeline']}/{task['tier']}/{task['condition']} | "
            f"{', '.join(task['after']) or '-'} | {task['sanity_after'] or '-'} | "
            f"{task['estimate_h'] / (3 if task['variant_gate'] else 1):.2f}{' (best variant only)' if task['variant_gate'] else ''} |"
        )
    lines += [
        "",
        "## Job script",
        "",
        "Submit `scripts/hpc_job.sh` (see HPC_RUNBOOK.md). It starts vLLM with the server values above, waits for the health check, tunes beta on dev, then runs `smvqa-gaze hpc-group` over this plan. The group runner stops launching tasks at 22 GPU h and stops everything at 23.5 GPU h.",
        "",
    ]
    path = results / "COMPUTE_PLAN.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def load_plan_tasks(path: Path) -> list[Task]:
    return [Task.from_dict(row) for row in json.loads(path.read_text())["tasks"]]
