"""H4: evaluation, statistics and the final report (README sections 9 and 10).

Everything here reads finished runs from ``paths.work`` and writes under
``paths.results``:

- ``metrics_all.csv`` and tables T0-T7 (CSV + Markdown, README 9.2);
- figures 1-6 in ``figs/`` (README 9.3);
- statistics in ``stats/`` (README 9.4): cluster-bootstrap CIs, paired
  McNemar tests with Holm correction, the main and personalization logistic
  models (mixed model with a fixed-effect fallback), leave-one-participant-out
  re-estimates, NSS consistency groups (README 7.9) and the H1-H6 verdicts;
- failure-case mosaics in ``failure_cases/``;
- ``REPORT.md`` (README section 10).

Only exact condition names from ``CONDITIONS`` are analysed; dev/smoke runs
(``C0_dev_n20`` etc.) are ignored. Missing runs never raise: they show up as
empty cells and are listed under "Known limitations".
"""

from __future__ import annotations

import json
import math
import textwrap
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..qa import read_jsonl  # noqa: E402
from .config import (  # noqa: E402
    CONDITIONS,
    FRAME_SIZE,
    PIPELINES,
    TIER_SIDE,
    Paths,
    RunConfig,
    write_stage_manifest,
)
from .frames import cached_frame_path  # noqa: E402
from .metrics import summarize  # noqa: E402

SEED = 0
N_BOOT = 1000
ALPHA = 0.05
TIERS = tuple(TIER_SIDE)
CATEGORIES = (
    "object_location_memory",
    "visual_recall",
    "intent_recall",
    "conversational_memory",
)
CATEGORY_SHORT = {
    "object_location_memory": "OLM",
    "visual_recall": "VR",
    "intent_recall": "IR",
    "conversational_memory": "CM",
}
FE_GROUPS = ("fixated", "glanced", "never")
FE_ORDER = ("fixated", "glanced", "never", "undetected", "NA", "no_target", "unanswerable", "unlabeled")
HINTS = ("T", "V", "M")
SOURCE_PREFIX = {"own": "C3", "shift": "C8shift", "cross": "C8cross", "saliency": "C9"}
ALPHA_CONDITIONS = {0.0: "C6a0", 0.5: "C6a0.5", 1.0: "C6a1"}
MAIN_MODEL_CONDITIONS = ("C0", "C1", "C3-T", "C3-V", "C3-M", "C5", "C4", "C6a0", "C6a0.5", "C6a1")
FIG2_CONDITIONS = ("C0", "C1", "C3-T", "C3-V", "C3-M", "C5", "C4")
FAILURE_CONDITIONS = ("C1", "C3-V", "C5", "C8shift-V", "C9-V", "C4")
FULL_FRAME_CONDITIONS = ("C0", "C3-T", "C3-V", "C3-M", "C5")
PAPER_BASELINE = {  # README section 8: Qwen-3-VL-8B on the full benchmark.
    "videorag": {"ans_f1": 75.0, "qa_acc": 41.8, "qa_mrr": 63.8},
    "egobutler": {"ans_f1": 44.5, "qa_acc": 38.8, "qa_mrr": 61.0},
}
CONDITION_ORDER = {name: index for index, name in enumerate(CONDITIONS)}
METRIC_KEYS = (
    "n",
    "ans_f1",
    "ans_f1_answerable",
    "ans_f1_unanswerable",
    "ans_f1_macro",
    "qa_acc",
    "qa_mrr",
    "false_abstention_rate",
    "vague_rate",
    "recall_at_k",
    "target_coverage",
    "parse_failure_rate",
    "parse_partial_rate",
)
RUN_KEYS = ["pipeline", "tier", "condition"]
PRED_COLUMNS = [
    "pipeline", "tier", "condition", "question_id", "participant", "session", "session_id",
    "category", "split", "fe", "is_answerable", "correct", "rank_of_correct", "rr", "abstained",
    "top1", "top1_type", "has_vague", "parse_status", "retrieval_hit", "target_coverage",
    "raw_output", "retrieved", "frame_meta", "frame_times", "k", "prompt_tokens",
]
HEAVY_COLUMNS = ("raw_output", "retrieved", "frame_meta", "frame_times")
QUESTION_COLUMNS = [
    "question_id", "participant", "session", "category", "split", "fe_label", "question",
    "choices", "correct_option_index", "answer_evidence",
]
# Categorical palette (fixed order; README figures are static PNGs).
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
NEUTRAL = "#8c8b86"
INK = "#2b2b2a"


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(isinstance(value, (float, np.floating)) and not np.isfinite(value))
    except TypeError:
        return False


def _pct(value: Any, digits: int = 1) -> str:
    return "" if _missing(value) else f"{100 * float(value):.{digits}f}"


def _signed_pct(value: Any, digits: int = 1) -> str:
    return "" if _missing(value) else f"{100 * float(value):+.{digits}f}"


def _fmt_p(value: Any) -> str:
    if _missing(value):
        return ""
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def condition_sort_key(name: str) -> tuple[int, str]:
    return (CONDITION_ORDER.get(name, len(CONDITION_ORDER)), name)


def _ordered_conditions(names: Any) -> list[str]:
    return sorted({str(name) for name in names}, key=condition_sort_key)


def _sort_runs(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    keys = frame.assign(
        _p=frame["pipeline"].map(lambda p: PIPELINES.index(p) if p in PIPELINES else 99),
        _t=frame["tier"].map(lambda t: TIERS.index(t) if t in TIERS else 99),
        _c=frame["condition"].map(lambda c: condition_sort_key(c)[0]),
    )
    return keys.sort_values(["_p", "_t", "_c", "condition"]).drop(columns=["_p", "_t", "_c"]).reset_index(drop=True)


def to_markdown(frame: pd.DataFrame, index: bool = False) -> str:
    """Plain GitHub Markdown table (no ``tabulate`` dependency)."""
    if index:
        frame = frame.reset_index()
    if frame.empty and not len(frame.columns):
        return "_(no data)_\n"
    columns = [str(column) for column in frame.columns]

    def cell(value: Any) -> str:
        if _missing(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{value:.4g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    if frame.empty:
        lines.append("| " + " | ".join("" for _ in columns) + " |")
    return "\n".join(lines) + "\n"


@dataclass
class Table:
    name: str
    data: pd.DataFrame  # numeric, written as CSV
    markdown: str
    display: pd.DataFrame | None = None


def write_table(table: Table, results_dir: Path) -> dict[str, str]:
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{table.name}.csv"
    md_path = results_dir / f"{table.name}.md"
    table.data.to_csv(csv_path, index=False)
    md_path.write_text(table.markdown, encoding="utf-8")
    return {"csv": str(csv_path), "md": str(md_path)}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return str(value)


def _clean_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame -> JSON-safe records (NaN -> None)."""
    return [
        {key: (None if _missing(value) else value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_questions(paths: Paths) -> pd.DataFrame:
    """qa_subset joined with the F(E) labels (one row per question)."""
    if not paths.qa_subset.is_file():
        return pd.DataFrame(columns=QUESTION_COLUMNS)
    labels = (
        {str(row["question_id"]): row.get("status") for row in read_jsonl(paths.fe_labels)}
        if paths.fe_labels.is_file()
        else {}
    )
    records = []
    for row in read_jsonl(paths.qa_subset):
        question_id = str(row["question_id"])
        records.append(
            {
                "question_id": question_id,
                "participant": int(row["participant"]),
                "session": int(row.get("primary_session", -1)),
                "category": row.get("category_key"),
                "split": row.get("split"),
                "fe_label": labels.get(question_id),
                "question": row.get("question", ""),
                "choices": row.get("choices") or [],
                "correct_option_index": row.get("correct_option_index"),
                "answer_evidence": row.get("answer_evidence") or [],
            }
        )
    return pd.DataFrame(records, columns=QUESTION_COLUMNS)


def discover_runs(paths: Paths) -> list[tuple[str, str, str]]:
    """(pipeline, tier, condition) with a predictions file; dev runs ignored."""
    runs = []
    for pipeline in PIPELINES:
        for tier in TIERS:
            directory = paths.work / "preds" / pipeline / tier
            if not directory.is_dir():
                continue
            for child in directory.iterdir():
                if child.name in CONDITIONS and (child / "predictions.jsonl").is_file():
                    runs.append((pipeline, tier, child.name))
    return sorted(
        runs,
        key=lambda run: (PIPELINES.index(run[0]), TIERS.index(run[1]), condition_sort_key(run[2])),
    )


def _prediction_row(pipeline: str, tier: str, condition: str, row: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
    question_id = str(row.get("question_id", row.get("key")))
    participant = int(row.get("participant", -1))
    session = int(row.get("session", -1))
    label = labels.get(question_id)
    fe = label if isinstance(label, str) and label else (row.get("fe_status") or "unlabeled")
    rank = row.get("rank_of_correct")
    rank = int(rank) if rank is not None else 4
    return {
        "pipeline": pipeline,
        "tier": tier,
        "condition": condition,
        "question_id": question_id,
        "participant": participant,
        "session": session,
        "session_id": f"p{participant}_s{session}",
        "category": row.get("category"),
        "split": row.get("split"),
        "fe": str(fe),
        "is_answerable": bool(row.get("is_answerable", True)),
        "correct": bool(row.get("correct", False)),
        "rank_of_correct": rank,
        "rr": 1.0 / rank,
        "abstained": bool(row.get("abstained", False)),
        "top1": row.get("top1"),
        "top1_type": row.get("top1_type"),
        "has_vague": bool(row.get("has_vague", True)),
        "parse_status": row.get("parse_status", "ok"),
        "retrieval_hit": _float_or_nan(row.get("retrieval_hit")),
        "target_coverage": _float_or_nan(row.get("target_coverage")),
        "raw_output": row.get("raw_output", ""),
        "retrieved": row.get("retrieved") or [],
        "frame_meta": row.get("frame_meta"),
        "frame_times": row.get("frame_times"),
        "k": row.get("k"),
        "prompt_tokens": row.get("prompt_tokens"),
    }


def load_runs(paths: Paths, split: str | None = "test", questions: pd.DataFrame | None = None) -> pd.DataFrame:
    """One long frame: pipeline x tier x condition x question, restricted to
    ``split``. F(E) comes from qa_fixation_labels when available (it is the
    authoritative label file), else from the prediction row. Duplicate rows of
    a question (restarted jobs) keep the last one."""
    if questions is None:
        questions = load_questions(paths)
    labels = dict(zip(questions["question_id"], questions["fe_label"]))
    rows: list[dict[str, Any]] = []
    for pipeline, tier, condition in discover_runs(paths):
        latest: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(paths.pred_dir(pipeline, tier, condition) / "predictions.jsonl"):
            latest[str(row.get("question_id", row.get("key")))] = row
        for row in latest.values():
            if split not in (None, "all") and row.get("split") != split:
                continue
            rows.append(_prediction_row(pipeline, tier, condition, row, labels))
    frame = pd.DataFrame(rows, columns=PRED_COLUMNS)
    for column in ("correct", "abstained", "is_answerable", "has_vague"):
        frame[column] = frame[column].astype(bool)
    for column in ("retrieval_hit", "target_coverage", "rr"):
        frame[column] = frame[column].astype(float)
    return frame


def run_frame(df: pd.DataFrame, pipeline: str, tier: str, condition: str) -> pd.DataFrame:
    return df[(df["pipeline"] == pipeline) & (df["tier"] == tier) & (df["condition"] == condition)]


def subset_rows(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "all":
        return frame
    if subset == "answerable":
        return frame[frame["is_answerable"]]
    return frame[frame["fe"] == subset]


def metric_values(frame: pd.DataFrame, metric: str) -> pd.Series:
    """Per-question binary value of a metric (NaN when undefined)."""
    if metric == "correct":
        return frame["correct"].astype(float)
    if metric == "false_abstention":
        return frame["abstained"].astype(float).where(frame["is_answerable"])
    if metric in ("retrieval_hit", "target_coverage"):
        return frame[metric].astype(float)
    if metric == "rr":
        return frame["rr"].astype(float)
    raise ValueError(f"Unknown metric {metric!r}")


# --------------------------------------------------------------------------
# Metrics per run
# --------------------------------------------------------------------------


def _summary_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "is_answerable": bool(row.is_answerable),
            "abstained": bool(row.abstained),
            "correct": bool(row.correct),
            "rank_of_correct": int(row.rank_of_correct),
            "has_vague": bool(row.has_vague),
            "top1_type": row.top1_type,
            "retrieval_hit": None if _missing(row.retrieval_hit) else bool(row.retrieval_hit),
            "target_coverage": None if _missing(row.target_coverage) else bool(row.target_coverage),
            "parse_status": row.parse_status,
        }
        for row in frame.itertuples(index=False)
    ]


def run_metrics(df: pd.DataFrame, ans_f1_positive: str = "answerable") -> pd.DataFrame:
    """Recompute README 9.1 metrics for every run (restricted to the split)."""
    rows = []
    for (pipeline, tier, condition), frame in df.groupby(RUN_KEYS, sort=False):
        summary = summarize(_summary_records(frame), ans_f1_positive)
        row: dict[str, Any] = {"pipeline": pipeline, "tier": tier, "condition": condition}
        row.update({key: summary.get(key) for key in METRIC_KEYS})
        for fe in ("fixated", "glanced", "never"):
            part = frame[frame["fe"] == fe]
            row[f"n_{fe}"] = len(part)
            row[f"qa_acc_{fe}"] = _mean(part["correct"].astype(float))
            row[f"recall_at_k_{fe}"] = _mean(part["retrieval_hit"])
            row[f"target_coverage_{fe}"] = _mean(part["target_coverage"])
        rows.append(row)
    columns = RUN_KEYS + list(METRIC_KEYS) + [
        f"{prefix}_{fe}" for fe in ("fixated", "glanced", "never")
        for prefix in ("n", "qa_acc", "recall_at_k", "target_coverage")
    ]
    frame = pd.DataFrame(rows, columns=columns)
    for column in columns[3:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return _sort_runs(frame)


def attach_stored_metrics(metrics: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    """Add n / QA-Acc from each run's metrics.json (consistency check only)."""
    stored_n, stored_acc = [], []
    for row in metrics.itertuples(index=False):
        path = paths.pred_dir(row.pipeline, row.tier, row.condition) / "metrics.json"
        overall: dict[str, Any] = {}
        if path.is_file():
            try:
                overall = json.loads(path.read_text(encoding="utf-8")).get("overall", {})
            except (OSError, json.JSONDecodeError):
                overall = {}
        stored_n.append(_float_or_nan(overall.get("n")))
        stored_acc.append(_float_or_nan(overall.get("qa_acc")))
    return metrics.assign(metrics_json_n=stored_n, metrics_json_qa_acc=stored_acc)


def expected_runs(include_optional: bool = False) -> list[tuple[str, str, str]]:
    runs = []
    for condition in CONDITIONS.values():
        if condition.priority == "P3" and not include_optional:
            continue
        for pipeline in PIPELINES:
            for tier in condition.tiers:
                runs.append((pipeline, tier, condition.name))
    return runs


def run_inventory(metrics: pd.DataFrame, questions: pd.DataFrame, split: str) -> dict[str, Any]:
    present = {(row.pipeline, row.tier, row.condition): int(row.n) for row in metrics.itertuples(index=False)}
    n_expected = int((questions["split"] == split).sum()) if split not in (None, "all") else len(questions)
    missing = [
        {"pipeline": p, "tier": t, "condition": c, "priority": CONDITIONS[c].priority}
        for p, t, c in expected_runs()
        if (p, t, c) not in present
    ]
    optional_missing = [
        {"pipeline": p, "tier": t, "condition": c, "priority": "P3"}
        for p, t, c in expected_runs(include_optional=True)
        if CONDITIONS[c].priority == "P3" and (p, t, c) not in present
    ]
    incomplete = [
        {"pipeline": p, "tier": t, "condition": c, "n": n, "n_expected": n_expected}
        for (p, t, c), n in present.items()
        if n < n_expected
    ]
    return {
        "present": [list(run) for run in present],
        "missing": missing,
        "optional_missing": optional_missing,
        "incomplete": incomplete,
        "n_expected_questions": n_expected,
    }


# --------------------------------------------------------------------------
# Tables T0-T7
# --------------------------------------------------------------------------


def _run_columns() -> list[tuple[str, str]]:
    return [(pipeline, tier) for pipeline in PIPELINES for tier in TIERS]


def _pivot_runs(metrics: pd.DataFrame, cell: Callable[[Any], str]) -> pd.DataFrame:
    """Rows: conditions present; columns: every pipeline x tier (empty if missing)."""
    conditions = _ordered_conditions(metrics["condition"]) if not metrics.empty else []
    lookup = {(row.pipeline, row.tier, row.condition): row for row in metrics.itertuples(index=False)}
    data = {
        f"{pipeline} {tier}": [
            cell(lookup[(pipeline, tier, condition)]) if (pipeline, tier, condition) in lookup else ""
            for condition in conditions
        ]
        for pipeline, tier in _run_columns()
    }
    return pd.DataFrame(data, index=pd.Index(conditions, name="condition"))


def table_t0(questions: pd.DataFrame, split: str = "test") -> Table:
    rows = questions if split in (None, "all") else questions[questions["split"] == split]
    fe = rows["fe_label"].fillna("unlabeled").astype(str)
    counts = pd.crosstab(rows["category"], fe) if len(rows) else pd.DataFrame()
    statuses = [s for s in FE_ORDER if s in counts.columns] + sorted(s for s in counts.columns if s not in FE_ORDER)
    categories = [c for c in CATEGORIES if c in counts.index] + sorted(c for c in counts.index if c not in CATEGORIES)
    long_rows = []
    display_rows = []
    for category in categories + (["All"] if categories else []):
        line = counts.sum(axis=0) if category == "All" else counts.loc[category]
        total = int(line.sum())
        display = {"category": category}
        for status in statuses:
            count = int(line.get(status, 0))
            proportion = count / total if total else float("nan")
            long_rows.append({"category": category, "fe_status": status, "count": count, "proportion": proportion})
            display[status] = f"{count} ({_pct(proportion)}%)"
        display["total"] = total
        display_rows.append(display)
    data = pd.DataFrame(long_rows, columns=["category", "fe_status", "count", "proportion"])
    display = pd.DataFrame(display_rows, columns=["category", *statuses, "total"])
    markdown = f"### T0: question category x F(E) ({split} split; n (row %))\n\n" + to_markdown(display)
    return Table("T0_fe_by_category", data, markdown, display)


def table_t1(metrics: pd.DataFrame) -> Table:
    display = _pivot_runs(
        metrics, lambda row: f"{_pct(row.ans_f1)} / {_pct(row.qa_acc)} / {_pct(row.qa_mrr)}"
    )
    data = metrics[RUN_KEYS + ["n", "ans_f1", "qa_acc", "qa_mrr"]].copy()
    markdown = (
        "### T1: main results (cells: Ans-F1 / QA-Acc / QA-MRR, x100)\n\n" + to_markdown(display, index=True)
    )
    return Table("T1_main", data, markdown, display)


def _acc_n(frame: pd.DataFrame) -> tuple[float, int]:
    return _mean(frame["correct"].astype(float)), int(len(frame))


def _cell(acc: float, n: int) -> str:
    return "" if n == 0 else f"{_pct(acc)} ({n})"


def table_t2(df: pd.DataFrame) -> Table:
    rows, sections = [], []
    for (pipeline, tier), group in _groups_by_run_column(df):
        display_rows = []
        for condition in _ordered_conditions(group["condition"]):
            frame = group[group["condition"] == condition]
            row: dict[str, Any] = {"pipeline": pipeline, "tier": tier, "condition": condition}
            display: dict[str, Any] = {"condition": condition}
            for category in CATEGORIES:
                acc, n = _acc_n(frame[frame["category"] == category])
                row[f"acc_{category}"], row[f"n_{category}"] = acc, n
                display[CATEGORY_SHORT[category]] = _cell(acc, n)
            acc, n = _acc_n(frame)
            row["acc_overall"], row["n_overall"] = acc, n
            display["Overall"] = _cell(acc, n)
            rows.append(row)
            display_rows.append(display)
        sections.append(f"#### {pipeline} {tier}\n\n" + to_markdown(pd.DataFrame(display_rows)))
    columns = RUN_KEYS + [f"{p}_{c}" for c in CATEGORIES for p in ("acc", "n")] + ["acc_overall", "n_overall"]
    legend = ", ".join(f"{short} = {name}" for name, short in CATEGORY_SHORT.items())
    markdown = (
        "### T2: QA-Acc by category (x100, n in parentheses)\n\n"
        + f"{legend}.\n\n"
        + ("\n".join(sections) if sections else "_(no runs)_\n")
    )
    return Table("T2_by_category", pd.DataFrame(rows, columns=columns), markdown)


def _groups_by_run_column(df: pd.DataFrame):
    for pipeline, tier in _run_columns():
        group = df[(df["pipeline"] == pipeline) & (df["tier"] == tier)]
        if len(group):
            yield (pipeline, tier), group


def table_t3(df: pd.DataFrame) -> Table:
    rows, sections = [], []
    for (pipeline, tier), group in _groups_by_run_column(df):
        display_rows = []
        for condition in _ordered_conditions(group["condition"]):
            frame = group[group["condition"] == condition]
            row: dict[str, Any] = {"pipeline": pipeline, "tier": tier, "condition": condition}
            display: dict[str, Any] = {"condition": condition}
            for category in CATEGORIES:
                for fe in FE_GROUPS:
                    acc, n = _acc_n(frame[(frame["category"] == category) & (frame["fe"] == fe)])
                    row[f"acc_{category}_{fe}"], row[f"n_{category}_{fe}"] = acc, n
                    display[f"{CATEGORY_SHORT[category]} {fe[:3]}"] = _cell(acc, n)
            fix_acc, fix_n = _acc_n(frame[frame["fe"] == "fixated"])
            nev_acc, nev_n = _acc_n(frame[frame["fe"] == "never"])
            diff = fix_acc - nev_acc if fix_n and nev_n else float("nan")
            row.update({"acc_fixated": fix_acc, "n_fixated": fix_n, "acc_never": nev_acc, "n_never": nev_n, "fixated_minus_never": diff})
            display.update({"fixated": _cell(fix_acc, fix_n), "never": _cell(nev_acc, nev_n), "fix-never": _signed_pct(diff)})
            rows.append(row)
            display_rows.append(display)
        sections.append(f"#### {pipeline} {tier}\n\n" + to_markdown(pd.DataFrame(display_rows)))
    markdown = (
        "### T3: QA-Acc by category x F(E) (x100, n in parentheses; fix/gla/nev = fixated/glanced/never)\n\n"
        + ("\n".join(sections) if sections else "_(no runs)_\n")
    )
    return Table("T3_category_by_fe", pd.DataFrame(rows), markdown)


def _paired_tiers(df: pd.DataFrame, pipeline: str, condition: str) -> pd.DataFrame:
    b1 = run_frame(df, pipeline, "B1", condition)
    b2 = run_frame(df, pipeline, "B2", condition)
    if b1.empty or b2.empty:
        return pd.DataFrame()
    return b1[["question_id", "participant", "category", "fe", "correct"]].merge(
        b2[["question_id", "correct"]], on="question_id", suffixes=("_b1", "_b2")
    )


def table_t4(df: pd.DataFrame) -> Table:
    rows, display_rows = [], []
    groups = [("overall", None)] + [(c, ("category", c)) for c in CATEGORIES] + [(f, ("fe", f)) for f in FE_GROUPS]
    for pipeline in PIPELINES:
        for condition in _ordered_conditions(df.loc[df["pipeline"] == pipeline, "condition"]):
            paired = _paired_tiers(df, pipeline, condition)
            if paired.empty:
                continue
            row: dict[str, Any] = {"pipeline": pipeline, "condition": condition}
            display: dict[str, Any] = {"pipeline": pipeline, "condition": condition}
            for label, selector in groups:
                part = paired if selector is None else paired[paired[selector[0]] == selector[1]]
                diff = _mean(part["correct_b2"].astype(float) - part["correct_b1"].astype(float)) if len(part) else float("nan")
                row[f"diff_{label}"], row[f"n_{label}"] = diff, len(part)
                display[CATEGORY_SHORT.get(label, label)] = _signed_pct(diff)
            rows.append(row)
            display_rows.append(display)
    markdown = (
        "### T4: budget effect, QA-Acc(B2) - QA-Acc(B1) on the same questions (x100)\n\n"
        + to_markdown(pd.DataFrame(display_rows, columns=["pipeline", "condition", "overall", *CATEGORY_SHORT.values(), *FE_GROUPS]))
    )
    return Table("T4_budget", pd.DataFrame(rows), markdown)


def personalization_conditions(hint: str) -> dict[str, str]:
    return {source: f"{prefix}-{hint}" for source, prefix in SOURCE_PREFIX.items()}


def table_t5(df: pd.DataFrame) -> Table:
    rows = []
    for (pipeline, tier), group in _groups_by_run_column(df):
        for hint in HINTS:
            runs = {source: group[group["condition"] == name] for source, name in personalization_conditions(hint).items()}
            if all(frame.empty for frame in runs.values()):
                continue
            selectors: list[tuple[str, str, Callable[[pd.DataFrame], pd.DataFrame]]] = [("overall", "all", lambda f: f)]
            selectors += [("category", c, lambda f, c=c: f[f["category"] == c]) for c in CATEGORIES]
            selectors += [("fe", fe, lambda f, fe=fe: f[f["fe"] == fe]) for fe in FE_GROUPS]
            participants = sorted(set(group["participant"]))
            selectors += [("participant", str(p), lambda f, p=p: f[f["participant"] == p]) for p in participants]
            for group_type, label, pick in selectors:
                row: dict[str, Any] = {"pipeline": pipeline, "tier": tier, "hint": hint, "group_type": group_type, "group": label}
                for source, frame in runs.items():
                    acc, n = _acc_n(pick(frame))
                    row[f"acc_{source}"], row[f"n_{source}"] = acc, n
                row["pg_shift"] = row["acc_own"] - row["acc_shift"]
                row["pg_cross"] = row["acc_own"] - row["acc_cross"]
                row["pg_saliency"] = row["acc_own"] - row["acc_saliency"]
                rows.append(row)
    data = pd.DataFrame(rows)
    if data.empty:
        return Table("T5_personalization", data, "### T5: personalization\n\n_(no C3/C8/C9 runs)_\n")
    display = pd.DataFrame(
        {
            "pipeline": data["pipeline"],
            "tier": data["tier"],
            "hint": data["hint"],
            "group": data["group_type"] + ":" + data["group"],
            "own (C3)": data["acc_own"].map(_pct),
            "shift (C8shift)": data["acc_shift"].map(_pct),
            "cross (C8cross)": data["acc_cross"].map(_pct),
            "saliency (C9)": data["acc_saliency"].map(_pct),
            "PG shift": data["pg_shift"].map(_signed_pct),
            "PG saliency": data["pg_saliency"].map(_signed_pct),
            "n own": data["n_own"],
        }
    )
    markdown = (
        "### T5: personalization (QA-Acc x100; PG = Acc(C3) - Acc(C8shift) and Acc(C3) - Acc(C9))\n\n"
        + to_markdown(display)
    )
    return Table("T5_personalization", data, markdown, display)


def table_t6(metrics: pd.DataFrame) -> Table:
    display = _pivot_runs(
        metrics,
        lambda row: f"{_pct(row.false_abstention_rate)} / {_pct(row.vague_rate)} / {_pct(row.ans_f1)}",
    )
    data = metrics[RUN_KEYS + ["false_abstention_rate", "vague_rate", "ans_f1", "ans_f1_macro"]].copy()
    markdown = (
        "### T6: answerability (cells: false abstention / vague rate / Ans-F1, x100)\n\n"
        + to_markdown(display, index=True)
    )
    return Table("T6_answerability", data, markdown, display)


def table_t7(metrics: pd.DataFrame) -> Table:
    columns = [
        "recall_at_k", "recall_at_k_fixated", "recall_at_k_never",
        "target_coverage", "target_coverage_fixated", "target_coverage_never",
    ]
    data = metrics[RUN_KEYS + columns].copy()
    display = data[RUN_KEYS].copy()
    for column in columns:
        display[column] = data[column].map(_pct)
    markdown = "### T7: retrieval and memory (x100; Recall@k and target coverage split fixated/never)\n\n" + to_markdown(display)
    return Table("T7_retrieval_memory", data, markdown, display)


def build_tables(df: pd.DataFrame, metrics: pd.DataFrame, questions: pd.DataFrame, split: str) -> list[Table]:
    return [
        table_t0(questions, split),
        table_t1(metrics),
        table_t2(df),
        table_t3(df),
        table_t4(df),
        table_t5(df),
        table_t6(metrics),
        table_t7(metrics),
    ]


# --------------------------------------------------------------------------
# Statistics: bootstrap, McNemar, Holm
# --------------------------------------------------------------------------


def cluster_bootstrap(
    values: np.ndarray | pd.Series,
    clusters: np.ndarray | pd.Series,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, Any]:
    """Two-stage cluster bootstrap of a mean (README 9.4): resample
    participants with replacement, then questions with replacement within
    each sampled participant. Returns the point estimate, a percentile 95% CI
    and the bootstrap distribution."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    keep = np.isfinite(values)
    values, clusters = values[keep], clusters[keep]
    if values.size == 0:
        return {"estimate": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "boot": np.array([]), "n": 0}
    groups = [values[clusters == label] for label in pd.unique(clusters)]
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    sums = np.zeros(n_boot)
    counts = np.zeros(n_boot)
    for slot in range(len(groups)):
        for index, group in enumerate(groups):
            reps = np.flatnonzero(picks[:, slot] == index)
            if reps.size == 0:
                continue
            draws = rng.integers(0, group.size, size=(reps.size, group.size))
            sums[reps] += group[draws].sum(axis=1)
            counts[reps] += group.size
    boot = sums / counts
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"estimate": float(values.mean()), "ci_lo": float(low), "ci_hi": float(high), "boot": boot, "n": int(values.size)}


def bootstrap_p(boot: np.ndarray) -> float:
    """Two-sided bootstrap p-value for 'mean differs from 0'."""
    if boot.size == 0:
        return float("nan")
    return float(min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))))


def mcnemar_test(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> dict[str, Any]:
    """Paired McNemar test of binary outcomes; exact binomial when the number
    of discordant pairs is < 25, chi-square with continuity correction otherwise."""
    from statsmodels.stats.contingency_tables import mcnemar

    a = np.asarray(a, dtype=float) > 0.5
    b = np.asarray(b, dtype=float) > 0.5
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    table = [[int(np.sum(a & b)), n10], [n01, int(np.sum(~a & ~b))]]
    discordant = n10 + n01
    exact = discordant < 25
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(mcnemar(table, exact=exact, correction=True).pvalue)
    return {"n10": n10, "n01": n01, "discordant": discordant, "exact": exact, "p": min(1.0, max(0.0, p_value))}


def holm(p_values: list[float] | np.ndarray | pd.Series) -> np.ndarray:
    """Holm step-down adjustment; NaN entries stay NaN and are not counted.

    Same result as ``statsmodels.stats.multitest.multipletests(method="holm")``,
    which is avoided because it runs ``gc.collect()`` on every call."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(p_values))
    if finite.size:
        order = np.argsort(p_values[finite], kind="stable")
        ranked = p_values[finite][order] * np.arange(finite.size, 0, -1)
        adjusted[finite[order]] = np.minimum(1.0, np.maximum.accumulate(ranked))
    return adjusted


def paired_values(
    df: pd.DataFrame,
    pipeline: str,
    run_a: tuple[str, str],
    run_b: tuple[str, str],
    subset: str = "all",
    metric: str = "correct",
) -> pd.DataFrame:
    """Per-question metric for two runs (tier, condition) on common questions."""
    frame_a = subset_rows(run_frame(df, pipeline, *run_a), subset)
    frame_b = subset_rows(run_frame(df, pipeline, *run_b), subset)
    values_a = metric_values(frame_a, metric).to_numpy(dtype=float)
    lookup = dict(zip(frame_b["question_id"], metric_values(frame_b, metric).to_numpy(dtype=float)))
    questions = frame_a["question_id"].to_numpy()
    values_b = np.array([lookup.get(q, np.nan) for q in questions], dtype=float)
    keep = np.isfinite(values_a) & np.isfinite(values_b)
    return pd.DataFrame({
        "question_id": questions[keep],
        "participant": frame_a["participant"].to_numpy()[keep],
        "a": values_a[keep],
        "b": values_b[keep],
    })


def compare_runs(
    df: pd.DataFrame,
    pipeline: str,
    tier: str,
    cond_a: str,
    cond_b: str,
    subset: str = "all",
    metric: str = "correct",
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, Any]:
    """Paired contrast mean(a) - mean(b): cluster-bootstrap CI and McNemar p."""
    pairs = paired_values(df, pipeline, (tier, cond_a), (tier, cond_b), subset, metric)
    result: dict[str, Any] = {
        "pipeline": pipeline, "tier": tier, "cond_a": cond_a, "cond_b": cond_b,
        "subset": subset, "metric": metric, "n": len(pairs),
    }
    if pairs.empty:
        return {**result, "mean_a": np.nan, "mean_b": np.nan, "diff": np.nan, "ci_lo": np.nan,
                "ci_hi": np.nan, "p": np.nan, "p_method": "", "n10": 0, "n01": 0}
    boot = cluster_bootstrap(pairs["a"] - pairs["b"], pairs["participant"], n_boot, seed)
    test = mcnemar_test(pairs["a"], pairs["b"])
    return {
        **result,
        "mean_a": float(pairs["a"].mean()),
        "mean_b": float(pairs["b"].mean()),
        "diff": boot["estimate"],
        "ci_lo": boot["ci_lo"],
        "ci_hi": boot["ci_hi"],
        "p": test["p"],
        "p_method": "mcnemar_exact" if test["exact"] else "mcnemar_chi2",
        "n10": test["n10"],
        "n01": test["n01"],
    }


def compare_budget_gap(
    df: pd.DataFrame,
    pipeline: str,
    full: str,
    crop: str,
    subset: str = "all",
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, Any]:
    """H5 difference-in-differences on questions present in all four runs:
    [Acc(full) - Acc(crop)]_B1 - [Acc(full) - Acc(crop)]_B2 (positive = gap shrinks)."""
    pieces = []
    for tier in ("B1", "B2"):
        for condition, name in ((full, "full"), (crop, "crop")):
            frame = subset_rows(run_frame(df, pipeline, tier, condition), subset)
            pieces.append(frame[["question_id", "participant"]].assign(**{f"{name}_{tier}": frame["correct"].astype(float).to_numpy()}))
    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece.drop(columns="participant"), on="question_id")
    result = {"pipeline": pipeline, "tier": "B1->B2", "cond_a": full, "cond_b": crop, "subset": subset,
              "metric": "gap_shrink", "n": len(merged)}
    if merged.empty:
        return {**result, "mean_a": np.nan, "mean_b": np.nan, "diff": np.nan, "ci_lo": np.nan,
                "ci_hi": np.nan, "p": np.nan, "p_method": "", "n10": 0, "n01": 0}
    gap_b1 = merged["full_B1"] - merged["crop_B1"]
    gap_b2 = merged["full_B2"] - merged["crop_B2"]
    boot = cluster_bootstrap(gap_b1 - gap_b2, merged["participant"], n_boot, seed)
    return {
        **result,
        "mean_a": float(gap_b1.mean()),
        "mean_b": float(gap_b2.mean()),
        "diff": boot["estimate"],
        "ci_lo": boot["ci_lo"],
        "ci_hi": boot["ci_hi"],
        "p": bootstrap_p(boot["boot"]),
        "p_method": "cluster_bootstrap",
        "n10": 0,
        "n01": 0,
    }


def bootstrap_accuracy(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Cluster-bootstrap 95% CI of QA-Acc for every run (overall and per F(E))."""
    rows = []
    for (pipeline, tier, condition), frame in df.groupby(RUN_KEYS, sort=False):
        for subset in ("all", *FE_GROUPS):
            part = subset_rows(frame, subset)
            boot = cluster_bootstrap(part["correct"].astype(float), part["participant"], n_boot, seed)
            rows.append({"pipeline": pipeline, "tier": tier, "condition": condition, "subset": subset,
                         "n": boot["n"], "acc": boot["estimate"], "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"]})
    return _sort_runs(pd.DataFrame(rows, columns=RUN_KEYS + ["subset", "n", "acc", "ci_lo", "ci_hi"]))


def pairwise_vs_c0(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Every condition vs C0 within pipeline x tier: paired bootstrap CI of the
    accuracy difference, McNemar p and Holm-adjusted p (family = all
    conditions of one pipeline x tier x subset)."""
    rows = []
    for (pipeline, tier), group in _groups_by_run_column(df):
        if "C0" not in set(group["condition"]):
            continue
        for condition in _ordered_conditions(group["condition"]):
            if condition == "C0":
                continue
            for subset in ("all", *FE_GROUPS):
                rows.append(compare_runs(df, pipeline, tier, condition, "C0", subset, "correct", n_boot, seed))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["pipeline", "tier", "cond_a", "cond_b", "subset", "n", "diff", "ci_lo", "ci_hi", "p", "p_holm"])
    frame["p_holm"] = np.nan
    for _, index in frame.groupby(["pipeline", "tier", "subset"]).groups.items():
        frame.loc[index, "p_holm"] = holm(frame.loc[index, "p"])
    return frame


# --------------------------------------------------------------------------
# Hypotheses H1-H6
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Component:
    group: str
    cond_a: str
    cond_b: str
    subset: str
    metric: str  # correct | retrieval_hit | false_abstention | gap_shrink
    direction: str  # greater | less | not_less | not_greater


@dataclass(frozen=True)
class Hypothesis:
    name: str
    statement: str
    groups: dict[str, str]  # group -> "any" | "all" | "report"
    components: tuple[Component, ...]
    per_pipeline: bool = False  # H5: one cell per pipeline (B1->B2)


def _hint_components(group: str, prefix_a: str, prefix_b: str, subset: str, direction: str) -> list[Component]:
    return [Component(group, f"{prefix_a}-{h}", f"{prefix_b}-{h}" if prefix_b != "C0" else "C0", subset, "correct", direction) for h in HINTS]


HYPOTHESES: tuple[Hypothesis, ...] = (
    Hypothesis(
        "H1",
        "Gaze as metadata/index (C5, C6) improves evidence retrieval and accuracy on fixated-target questions.",
        {"fixated accuracy up (any of C5/C6)": "any", "Recall@k up (any of C5/C6)": "any"},
        tuple(
            [Component("fixated accuracy up (any of C5/C6)", c, "C0", "fixated", "correct", "greater") for c in ("C5", "C6a0", "C6a0.5", "C6a1")]
            + [Component("Recall@k up (any of C5/C6)", c, "C0", "all", "retrieval_hit", "greater") for c in ("C5", "C6a0", "C6a0.5", "C6a1")]
        ),
    ),
    Hypothesis(
        "H2",
        "Fovea-only filtering (C1) hurts never-fixated questions and keeps fixated questions level or better.",
        {"never drops": "all", "fixated not worse": "all"},
        (
            Component("never drops", "C1", "C0", "never", "correct", "less"),
            Component("fixated not worse", "C1", "C0", "fixated", "correct", "not_less"),
        ),
    ),
    Hypothesis(
        "H3",
        "Gaze prompting (C3-T/V/M) biases memory toward fixated targets, mildly hurting never-fixated evidence.",
        {"fixated up (any hint)": "any", "never not improved (all hints)": "all", "never slightly down": "report"},
        tuple(
            _hint_components("fixated up (any hint)", "C3", "C0", "fixated", "greater")
            + _hint_components("never not improved (all hints)", "C3", "C0", "never", "not_greater")
            + _hint_components("never slightly down", "C3", "C0", "never", "less")
        ),
    ),
    Hypothesis(
        "H4",
        "Own, time-aligned gaze beats shifted gaze, another person's gaze and generic saliency, mainly on fixated questions.",
        {
            "own > shift on fixated (any hint)": "any",
            "own > cross on fixated (any hint)": "any",
            "own > saliency on fixated (any hint)": "any",
            "all questions": "report",
        },
        tuple(
            _hint_components("own > shift on fixated (any hint)", "C3", "C8shift", "fixated", "greater")
            + _hint_components("own > cross on fixated (any hint)", "C3", "C8cross", "fixated", "greater")
            + _hint_components("own > saliency on fixated (any hint)", "C3", "C9", "fixated", "greater")
            + _hint_components("all questions", "C3", "C8shift", "all", "greater")
            + _hint_components("all questions", "C3", "C9", "all", "greater")
        ),
    ),
    Hypothesis(
        "H5",
        "The accuracy gap between full-frame conditions and C1 shrinks from B1 to B2.",
        {"C0-C1 gap shrinks": "all", "other full-frame conditions": "report"},
        tuple(
            [Component("C0-C1 gap shrinks", "C0", "C1", "all", "gap_shrink", "greater")]
            + [Component("other full-frame conditions", c, "C1", "all", "gap_shrink", "greater") for c in FULL_FRAME_CONDITIONS[1:]]
        ),
        per_pipeline=True,
    ),
    Hypothesis(
        "H6",
        "Filtering conditions (C1, C4) raise the false-abstention rate relative to C0.",
        {"false abstention up (C1 and C4 where run)": "all"},
        (
            Component("false abstention up (C1 and C4 where run)", "C1", "C0", "answerable", "false_abstention", "greater"),
            Component("false abstention up (C1 and C4 where run)", "C4", "C0", "answerable", "false_abstention", "greater"),
        ),
    ),
)

DECISION_RULES = """\
Decision rules (fixed before looking at results):
1. Each component is a paired contrast cond_a - cond_b on the same questions within one
   pipeline x tier (H5: per pipeline, B1 vs B2 difference-in-differences). Estimate =
   mean difference; 95% CI = two-stage cluster bootstrap (participants, then questions;
   1000 reps by default, seed 0); p = McNemar (exact if < 25 discordant pairs; H5: bootstrap p).
   Holm correction runs over all non-"report" components of the hypothesis.
2. Component outcome: 'greater' passes iff CI_lo > 0 and p_Holm < 0.05; 'less' passes iff
   CI_hi < 0 and p_Holm < 0.05; 'not_less' (non-inferiority) passes unless CI_hi < 0 and
   p_Holm < 0.05; 'not_greater' passes unless CI_lo > 0 and p_Holm < 0.05. A component
   without paired data is 'missing'.
3. Group outcome: mode 'any' passes if any available component passes; 'all' passes if
   every available component passes; 'report' groups are listed but never decide.
4. Cell (pipeline x tier) verdict: 'inconclusive' if any deciding group has no data;
   otherwise 'supported' iff every deciding group passes, else 'not supported'.
5. Hypothesis verdict over cells: 'inconclusive' if no cell has data; 'supported' if more
   than half of the cells with data are supported; 'not supported' if none is;
   otherwise 'inconclusive (mixed across pipeline x tier)'.
"""


def _component_outcome(row: dict[str, Any]) -> str:
    if row["n"] == 0 or _missing(row["diff"]):
        return "missing"
    significant = not _missing(row["p_holm"]) and row["p_holm"] < ALPHA
    above = not _missing(row["ci_lo"]) and row["ci_lo"] > 0
    below = not _missing(row["ci_hi"]) and row["ci_hi"] < 0
    direction = row["direction"]
    if direction == "greater":
        return "pass" if above and significant else "fail"
    if direction == "less":
        return "pass" if below and significant else "fail"
    if direction == "not_less":
        return "fail" if below and significant else "pass"
    if direction == "not_greater":
        return "fail" if above and significant else "pass"
    raise ValueError(direction)


def hypothesis_tests(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    rows = []
    for hypothesis in HYPOTHESES:
        cells = (
            [(pipeline, "B1->B2") for pipeline in PIPELINES]
            if hypothesis.per_pipeline
            else _run_columns()
        )
        block = []
        for pipeline, tier in cells:
            for component in hypothesis.components:
                if component.metric == "gap_shrink":
                    result = compare_budget_gap(df, pipeline, component.cond_a, component.cond_b, component.subset, n_boot, seed)
                else:
                    result = compare_runs(df, pipeline, tier, component.cond_a, component.cond_b,
                                          component.subset, component.metric, n_boot, seed)
                block.append({"hypothesis": hypothesis.name, "group": component.group,
                              "mode": hypothesis.groups[component.group], "direction": component.direction, **result})
        frame = pd.DataFrame(block)
        deciding = frame["mode"] != "report"
        frame["p_holm"] = np.nan
        frame.loc[deciding, "p_holm"] = holm(frame.loc[deciding, "p"])
        if (~deciding).any():
            frame.loc[~deciding, "p_holm"] = holm(frame.loc[~deciding, "p"])
        rows.append(frame)
    tests = pd.concat(rows, ignore_index=True)
    tests["outcome"] = [_component_outcome(row) for row in tests.to_dict(orient="records")]
    return tests


def _group_outcome(frame: pd.DataFrame, mode: str) -> str:
    available = frame[frame["outcome"] != "missing"]
    if available.empty:
        return "missing"
    passed = available["outcome"] == "pass"
    return "pass" if (passed.any() if mode == "any" else passed.all()) else "fail"


def hypothesis_verdicts(tests: pd.DataFrame) -> list[dict[str, Any]]:
    verdicts = []
    for hypothesis in HYPOTHESES:
        block = tests[tests["hypothesis"] == hypothesis.name]
        cells = []
        for (pipeline, tier), cell in block.groupby(["pipeline", "tier"], sort=False):
            groups = {
                group: _group_outcome(cell[cell["group"] == group], mode)
                for group, mode in hypothesis.groups.items()
                if mode != "report"
            }
            if any(outcome == "missing" for outcome in groups.values()):
                verdict = "inconclusive"
            elif all(outcome == "pass" for outcome in groups.values()):
                verdict = "supported"
            else:
                verdict = "not supported"
            cells.append({"pipeline": pipeline, "tier": tier, "verdict": verdict, "groups": groups})
        decided = [cell for cell in cells if cell["verdict"] != "inconclusive"]
        supported = sum(cell["verdict"] == "supported" for cell in decided)
        if not decided:
            overall = "inconclusive"
            reason = "no pipeline x tier has data for every deciding contrast"
        elif supported > len(decided) / 2:
            overall = "supported"
            reason = f"supported in {supported}/{len(decided)} cells with data"
        elif supported == 0:
            overall = "not supported"
            reason = f"supported in 0/{len(decided)} cells with data"
        else:
            overall = "inconclusive"
            reason = f"mixed across pipeline x tier ({supported}/{len(decided)} supported)"
        verdicts.append({"hypothesis": hypothesis.name, "statement": hypothesis.statement,
                         "verdict": overall, "reason": reason, "cells": cells})
    return verdicts


# --------------------------------------------------------------------------
# Logistic models (mixed with fixed-effect fallback)
# --------------------------------------------------------------------------


def _model_frame(frame: pd.DataFrame, levels: dict[str, list[str]]) -> pd.DataFrame:
    data = frame.copy()
    data["correct_i"] = data["correct"].astype(int)
    data["participant"] = data["participant"].astype(str)
    data["session_id"] = data["session_id"].astype(str)
    for column, order in levels.items():
        present = [level for level in order if level in set(data[column])]
        data[column] = pd.Categorical(data[column].astype(str), categories=present)
    return data


def _terms_from_estimates(names: list[str], coef: np.ndarray, se: np.ndarray) -> pd.DataFrame:
    from scipy.stats import norm

    coef = np.asarray(coef, dtype=float)
    se = np.asarray(se, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = coef / se
    p = 2 * norm.sf(np.abs(z))
    return pd.DataFrame({"term": names, "coef": coef, "se": se, "z": z, "p": p})


def fit_logit_model(
    data: pd.DataFrame,
    formula: str,
    vc_formulas: dict[str, str],
    prefer_mixed: bool = True,
) -> dict[str, Any]:
    """Fit ``formula`` (response ``correct_i``) as a Bayesian binomial mixed
    model with variance components ``vc_formulas`` (statsmodels
    BinomialBayesMixedGLM, variational Bayes). If that raises or does not
    converge, fall back to a fixed-effect binomial GLM with participant as a
    fixed effect (README 9.4) and record why."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    reason = "mixed model not requested"
    if prefer_mixed:
        try:
            from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = BinomialBayesMixedGLM.from_formula(formula, vc_formulas, data)
                result = model.fit_vb()
            not_converged = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
            retvals = getattr(result, "optim_retvals", None)
            if not_converged or (retvals is not None and not getattr(retvals, "success", True)):
                raise RuntimeError("variational Bayes optimisation did not converge")
            if not (np.all(np.isfinite(result.fe_mean)) and np.all(np.isfinite(result.fe_sd))):
                raise RuntimeError("non-finite posterior estimates")
            terms = _terms_from_estimates(list(model.fep_names), result.fe_mean, result.fe_sd)
            variance = {
                name: float(np.exp(mean))
                for name, mean in zip(model.vcp_names, result.vcp_mean)
            }
            return {
                "model_used": "BinomialBayesMixedGLM (variational Bayes)",
                "reason": "converged",
                "formula": formula,
                "random_effects": vc_formulas,
                "random_effect_sd": variance,
                "n": int(len(data)),
                "terms": terms,
                "p_note": "p from posterior mean / posterior sd (normal approximation)",
            }
        except Exception as exc:  # noqa: BLE001 - any failure triggers the documented fallback
            reason = f"mixed model failed: {type(exc).__name__}: {exc}"
    fallback = formula
    if data["participant"].nunique() > 1:
        fallback += " + C(participant)"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = smf.glm(fallback, data, family=sm.families.Binomial()).fit()
        notes = sorted({str(w.message).splitlines()[0] for w in caught})
        params = result.params
        terms = _terms_from_estimates(list(params.index), params.to_numpy(), result.bse.to_numpy())
        return {
            "model_used": "GLM Binomial with participant as fixed effect",
            "reason": reason,
            "formula": fallback,
            "random_effects": {},
            "n": int(len(data)),
            "terms": terms,
            "converged": bool(getattr(result, "converged", True)),
            "warnings": notes,
            "p_note": "Wald p-values",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "model_used": "failed",
            "reason": f"{reason}; fallback failed: {type(exc).__name__}: {exc}",
            "formula": fallback,
            "n": int(len(data)),
            "terms": pd.DataFrame(columns=["term", "coef", "se", "z", "p"]),
        }


def _formula(response: str, crossed: list[str], extra: list[str], data: pd.DataFrame) -> str | None:
    crossed = [term for term in crossed if data[term].nunique() > 1]
    extra = [term for term in extra if data[term.split(" ")[0]].nunique() > 1]
    if not crossed:
        return None
    return f"{response} ~ " + " + ".join([" * ".join(crossed), *extra])


def main_model(df: pd.DataFrame, pipeline: str, include_glanced: bool = False, prefer_mixed: bool = True) -> dict[str, Any]:
    """correct ~ condition * fe * tier + category + (1|participant) + (1|session)."""
    fes = ["fixated", "never"] + (["glanced"] if include_glanced else [])
    frame = df[(df["pipeline"] == pipeline) & df["condition"].isin(MAIN_MODEL_CONDITIONS) & df["fe"].isin(fes)]
    if frame.empty or "C0" not in set(frame["condition"]):
        return {"model_used": "skipped", "reason": "no C0 / no fixated-never rows", "n": int(len(frame)), "terms": pd.DataFrame()}
    data = _model_frame(frame, {"condition": list(MAIN_MODEL_CONDITIONS), "fe": fes, "tier": list(TIERS), "category": list(CATEGORIES)})
    formula = _formula("correct_i", ["condition", "fe", "tier"], ["category"], data)
    if formula is None or data["condition"].nunique() < 2:
        return {"model_used": "skipped", "reason": "fewer than two conditions", "n": int(len(data)), "terms": pd.DataFrame()}
    vc = {"participant": "0 + C(participant)"}
    if data["session_id"].nunique() > data["participant"].nunique():
        vc["session"] = "0 + C(session_id)"
    result = fit_logit_model(data, formula, vc, prefer_mixed)
    result["key_terms"] = _key_terms(result["terms"], ["fe[T.never]", "tier[T.B2]"])
    return result


def _key_terms(terms: pd.DataFrame, markers: list[str]) -> pd.DataFrame:
    if terms is None or terms.empty:
        return pd.DataFrame(columns=["term", "coef", "se", "z", "p"])
    mask = terms["term"].map(lambda name: ":" in name and any(marker in name for marker in markers))
    return terms[mask].reset_index(drop=True)


def personalization_frame(df: pd.DataFrame) -> pd.DataFrame:
    source_of = {f"{prefix}-{hint}": (source, hint) for source, prefix in SOURCE_PREFIX.items() for hint in HINTS}
    frame = df[df["condition"].isin(source_of)].copy()
    frame["gaze_source"] = frame["condition"].map(lambda c: source_of[c][0])
    frame["hint"] = frame["condition"].map(lambda c: source_of[c][1])
    return frame


def personalization_models(
    df: pd.DataFrame, pipeline: str, nss: pd.DataFrame | None = None, prefer_mixed: bool = True
) -> dict[str, Any]:
    """correct ~ gaze_source * fe + category + (1|participant) (+ hint + tier,
    because variants and tiers are pooled), and the same with an NSS-group
    interaction (README 7.9)."""
    frame = personalization_frame(df[df["pipeline"] == pipeline])
    frame = frame[frame["fe"].isin(["fixated", "never"])]
    output: dict[str, Any] = {}
    if frame.empty or frame["gaze_source"].nunique() < 2 or "own" not in set(frame["gaze_source"]):
        skipped = {"model_used": "skipped", "reason": "needs C3 plus at least one of C8/C9", "n": int(len(frame)), "terms": pd.DataFrame()}
        return {"base": skipped, "nss": dict(skipped)}
    levels = {"gaze_source": list(SOURCE_PREFIX), "fe": ["fixated", "never"], "category": list(CATEGORIES),
              "hint": list(HINTS), "tier": list(TIERS)}
    data = _model_frame(frame, levels)
    vc = {"participant": "0 + C(participant)"}
    formula = _formula("correct_i", ["gaze_source", "fe"], ["category", "hint", "tier"], data)
    output["base"] = fit_logit_model(data, formula, vc, prefer_mixed) if formula else {"model_used": "skipped", "reason": "single level", "terms": pd.DataFrame()}
    output["base"]["key_terms"] = _key_terms(output["base"].get("terms"), ["fe[T.never]"])
    if nss is not None and not nss.empty and "nss_group" in nss:
        groups = dict(zip(nss["question_id"].astype(str), nss["nss_group"]))
        with_group = frame.assign(nss_group=frame["question_id"].map(groups)).dropna(subset=["nss_group"])
        if with_group["nss_group"].nunique() == 2:
            data = _model_frame(with_group, {**levels, "nss_group": ["conventional", "idiosyncratic"]})
            formula = _formula("correct_i", ["gaze_source", "fe"], ["category", "hint", "tier"], data)
            if formula:
                formula += " + gaze_source * nss_group"
                output["nss"] = fit_logit_model(data, formula, vc, prefer_mixed)
                output["nss"]["key_terms"] = _key_terms(output["nss"].get("terms"), ["nss_group[T."])
    output.setdefault("nss", {"model_used": "skipped", "reason": "no NSS groups", "terms": pd.DataFrame()})
    return output


# --------------------------------------------------------------------------
# Leave-one-participant-out
# --------------------------------------------------------------------------


LOPO_CONTRASTS: tuple[tuple[str, str, str, str], ...] = (
    ("C1-C0 | never", "C1", "C0", "never"),
    ("C5-C0 | fixated", "C5", "C0", "fixated"),
    *[(f"C3-{h}-C8shift-{h} | all", f"C3-{h}", f"C8shift-{h}", "all") for h in HINTS],
    *[(f"C3-{h}-C8shift-{h} | fixated", f"C3-{h}", f"C8shift-{h}", "fixated") for h in HINTS],
)


def leave_one_participant_out(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pipeline, tier), _ in _groups_by_run_column(df):
        for label, cond_a, cond_b, subset in LOPO_CONTRASTS:
            pairs = paired_values(df, pipeline, (tier, cond_a), (tier, cond_b), subset)
            if pairs.empty:
                continue
            full = float((pairs["a"] - pairs["b"]).mean())
            rows.append({"pipeline": pipeline, "tier": tier, "contrast": label, "left_out": "none",
                         "n": len(pairs), "diff": full})
            for participant in sorted(pairs["participant"].unique()):
                rest = pairs[pairs["participant"] != participant]
                diff = float((rest["a"] - rest["b"]).mean()) if len(rest) else float("nan")
                rows.append({"pipeline": pipeline, "tier": tier, "contrast": label,
                             "left_out": str(participant), "n": len(rest), "diff": diff})
    frame = pd.DataFrame(rows, columns=["pipeline", "tier", "contrast", "left_out", "n", "diff"])
    if frame.empty:
        return frame.assign(sign_consistent=pd.Series(dtype=bool))
    consistent = {}
    for key, block in frame.groupby(["pipeline", "tier", "contrast"]):
        signs = np.sign(block["diff"].dropna().to_numpy())
        consistent[key] = bool(len(signs) and (np.all(signs > 0) or np.all(signs < 0)))
    frame["sign_consistent"] = [consistent[(r.pipeline, r.tier, r.contrast)] for r in frame.itertuples(index=False)]
    return frame


# --------------------------------------------------------------------------
# NSS consistency (README 7.9) and participant gaze statistics
# --------------------------------------------------------------------------


def _load_gaze_samples(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    frame = pd.read_parquet(path, columns=["t_video_s", "u", "v", "valid"])
    keep = frame["valid"].astype(bool) & np.isfinite(frame["u"]) & np.isfinite(frame["v"]) & np.isfinite(frame["t_video_s"])
    return frame[keep].sort_values("t_video_s").reset_index(drop=True)


def _standardized_map(path: Path) -> np.ndarray | None:
    import cv2

    if not path.is_file():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = image.astype(np.float32)
    std = float(image.std())
    if std <= 0:
        return None
    return (image - float(image.mean())) / std


def question_nss(
    paths: Paths,
    evidence: list[dict[str, Any]],
    gaze_cache: dict[str, pd.DataFrame | None],
    max_hz: float = 10.0,
) -> tuple[float, int, int]:
    """NSS of one question: mean standardized saliency at valid gaze samples
    inside the evidence spans (gaze subsampled to <= max_hz; each saliency map
    loaded at most once per question). Returns (nss, n_samples, n_maps)."""
    maps: dict[tuple[str, int], np.ndarray | None] = {}
    values: list[np.ndarray] = []
    for span in evidence:
        video_id = span.get("video_id")
        if not video_id or span.get("t_start") is None or span.get("t_end") is None:
            continue
        if video_id not in gaze_cache:
            gaze_cache[video_id] = _load_gaze_samples(paths.gaze_dir(video_id) / "gaze_px.parquet")
        gaze = gaze_cache[video_id]
        if gaze is None or gaze.empty:
            continue
        t = gaze["t_video_s"].to_numpy()
        lo, hi = np.searchsorted(t, float(span["t_start"]), "left"), np.searchsorted(t, float(span["t_end"]), "right")
        if hi <= lo:
            continue
        times = t[lo:hi]
        _, first = np.unique(np.floor(times * max_hz).astype(np.int64), return_index=True)
        index = lo + first
        times = t[index]
        u = gaze["u"].to_numpy()[index]
        v = gaze["v"].to_numpy()[index]
        seconds = np.rint(times).astype(int)
        for second in np.unique(seconds):
            key = (video_id, int(second))
            if key not in maps:
                maps[key] = _standardized_map(paths.saliency_dir(video_id) / f"frame_{int(second):08d}.jpg")
            saliency = maps[key]
            if saliency is None:
                continue
            pick = seconds == second
            height, width = saliency.shape
            x = np.clip((u[pick] * width / FRAME_SIZE).astype(int), 0, width - 1)
            y = np.clip((v[pick] * height / FRAME_SIZE).astype(int), 0, height - 1)
            values.append(saliency[y, x])
    if not values:
        return float("nan"), 0, sum(m is not None for m in maps.values())
    stacked = np.concatenate(values)
    return float(stacked.mean()), int(stacked.size), sum(m is not None for m in maps.values())


def compute_nss(paths: Paths, questions: pd.DataFrame, max_hz: float = 10.0) -> pd.DataFrame:
    """NSS per question with timed evidence plus the median split into
    'conventional' (NSS >= median) and 'idiosyncratic' gaze."""
    gaze_cache: dict[str, pd.DataFrame | None] = {}
    rows = []
    for row in questions.itertuples(index=False):
        evidence = [s for s in (row.answer_evidence or []) if s.get("t_start") is not None and s.get("video_id")]
        if not evidence:
            continue
        nss, n_samples, n_maps = question_nss(paths, evidence, gaze_cache, max_hz)
        rows.append({"question_id": str(row.question_id), "participant": row.participant, "category": row.category,
                     "split": row.split, "fe": row.fe_label, "n_samples": n_samples, "n_maps": n_maps, "nss": nss})
    frame = pd.DataFrame(rows, columns=["question_id", "participant", "category", "split", "fe", "n_samples", "n_maps", "nss"])
    return assign_nss_groups(frame)


def assign_nss_groups(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    finite = np.isfinite(frame["nss"].astype(float)) if len(frame) else pd.Series(dtype=bool)
    median = float(frame.loc[finite, "nss"].median()) if len(frame) and finite.any() else float("nan")
    frame["nss_median"] = median
    frame["nss_group"] = [
        None if not math.isfinite(value) else ("conventional" if value >= median else "idiosyncratic")
        for value in frame["nss"].astype(float)
    ]
    return frame


def load_or_compute_nss(paths: Paths, questions: pd.DataFrame, recompute: bool = False) -> pd.DataFrame:
    """Cached in results/stats/nss_by_question.csv; recomputed when the cache
    misses any question that has timed evidence."""
    cache = paths.results / "stats" / "nss_by_question.csv"
    wanted = {
        str(row.question_id)
        for row in questions.itertuples(index=False)
        if any(s.get("t_start") is not None for s in (row.answer_evidence or []))
    }
    if cache.is_file() and not recompute:
        cached = pd.read_csv(cache, dtype={"question_id": str})
        if wanted <= set(cached["question_id"]):
            return assign_nss_groups(cached[cached["question_id"].isin(wanted)].drop(columns=["nss_group", "nss_median"], errors="ignore"))
    frame = compute_nss(paths, questions)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame


def participant_gaze_stats(paths: Paths) -> pd.DataFrame:
    """Per participant: fixations per minute, dispersion (mean distance of
    fixations from their centroid, px) and centre bias (mean distance from
    the frame centre, px). Descriptive only (README 9.3 figure 5)."""
    columns = ["participant", "n_fixations", "minutes", "fixation_rate_per_min", "dispersion_px", "center_distance_px"]
    if not paths.sessions.is_file():
        return pd.DataFrame(columns=columns)
    sessions = json.loads(paths.sessions.read_text(encoding="utf-8"))
    points: dict[int, list[np.ndarray]] = {}
    minutes: dict[int, float] = {}
    for session in sessions:
        path = paths.gaze_dir(str(session["video_id"])) / "fixations.parquet"
        if not path.is_file():
            continue
        fixations = pd.read_parquet(path, columns=["u", "v"])
        xy = fixations[["u", "v"]].to_numpy(dtype=float)
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        participant = int(session["participant"])
        points.setdefault(participant, []).append(xy)
        minutes[participant] = minutes.get(participant, 0.0) + float(session["duration_s"]) / 60
    rows = []
    for participant in sorted(points):
        xy = np.concatenate(points[participant]) if points[participant] else np.zeros((0, 2))
        if not len(xy):
            continue
        centroid = xy.mean(axis=0)
        rows.append({
            "participant": participant,
            "n_fixations": int(len(xy)),
            "minutes": minutes[participant],
            "fixation_rate_per_min": len(xy) / minutes[participant] if minutes[participant] else float("nan"),
            "dispersion_px": float(np.linalg.norm(xy - centroid, axis=1).mean()),
            "center_distance_px": float(np.linalg.norm(xy - FRAME_SIZE / 2, axis=1).mean()),
        })
    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------------------------------
# Figures 1-6
# --------------------------------------------------------------------------


def _new_figure(n_rows: int, n_cols: int, width: float, height: float):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height), squeeze=False)
    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK, labelsize=8)
    return fig, axes


def _save(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def figure_fe_by_category(t0: pd.DataFrame, path: Path) -> Path | None:
    data = t0[t0["category"] != "All"]
    if data.empty:
        return None
    pivot = data.pivot(index="category", columns="fe_status", values="proportion").fillna(0.0)
    main = [s for s in FE_GROUPS if s in pivot.columns]
    other = [s for s in pivot.columns if s not in FE_GROUPS]
    if other:
        pivot["other"] = pivot[other].sum(axis=1)
    stack = main + (["other"] if other else [])
    fig, axes = _new_figure(1, 1, 7, 0.6 * len(pivot) + 1.6)
    ax = axes[0, 0]
    left = np.zeros(len(pivot))
    labels = [CATEGORY_SHORT.get(c, c) for c in pivot.index]
    colors = {"fixated": PALETTE[0], "glanced": PALETTE[1], "never": PALETTE[2], "other": NEUTRAL}
    for status in stack:
        ax.barh(labels, pivot[status], left=left, color=colors[status], label=status, edgecolor="white", linewidth=2)
        left += pivot[status].to_numpy()
    ax.set_xlim(0, 1)
    ax.set_xlabel("proportion of questions")
    ax.set_title("F(E) by question category", fontsize=10, color=INK)
    ax.legend(ncol=len(stack), fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def figure_condition_by_fe(df: pd.DataFrame, figs_dir: Path) -> list[Path]:
    paths = []
    for (pipeline, tier), group in _groups_by_run_column(df):
        conditions = [c for c in FIG2_CONDITIONS if c in set(group["condition"])]
        if not conditions:
            continue
        fig, axes = _new_figure(1, 1, 6, 4)
        ax = axes[0, 0]
        x = np.arange(len(FE_GROUPS))
        for index, condition in enumerate(conditions):
            frame = group[group["condition"] == condition]
            acc = [_mean(frame.loc[frame["fe"] == fe, "correct"].astype(float)) for fe in FE_GROUPS]
            ax.plot(x, acc, marker="o", markersize=6, linewidth=2, color=PALETTE[index % len(PALETTE)], label=condition)
        ax.set_xticks(x, FE_GROUPS)
        ax.set_ylim(0, 1)
        ax.set_ylabel("QA-Acc")
        ax.set_title(f"Condition x F(E): {pipeline} {tier}", fontsize=10, color=INK)
        ax.legend(fontsize=8, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        paths.append(_save(fig, figs_dir / f"fig2_condition_x_fe_{pipeline}_{tier}.png"))
    return paths


def figure_budget_slopes(metrics: pd.DataFrame, path: Path) -> Path | None:
    pipelines = [p for p in PIPELINES if p in set(metrics["pipeline"])]
    if not pipelines:
        return None
    fig, axes = _new_figure(1, len(pipelines), 4.5 * len(pipelines), 5)
    highlight = {"C0": PALETTE[0], "C1": PALETTE[1]}
    for ax, pipeline in zip(axes[0], pipelines):
        block = metrics[metrics["pipeline"] == pipeline]
        for condition in _ordered_conditions(block["condition"]):
            values = [block.loc[(block["condition"] == condition) & (block["tier"] == t), "qa_acc"] for t in TIERS]
            if any(v.empty for v in values):
                continue
            y = [float(v.iloc[0]) for v in values]
            color = highlight.get(condition, NEUTRAL)
            ax.plot([0, 1], y, marker="o", markersize=5, linewidth=2 if condition in highlight else 1, color=color)
            ax.annotate(condition, (1, y[1]), xytext=(4, 0), textcoords="offset points", fontsize=7, va="center", color=INK)
        ax.set_xticks([0, 1], list(TIERS))
        ax.set_xlim(-0.2, 1.4)
        ax.set_ylabel("QA-Acc")
        ax.set_title(f"B1 -> B2: {pipeline}", fontsize=10, color=INK)
    return _save(fig, path)


def figure_personalization(df: pd.DataFrame, path: Path) -> Path | None:
    frame = personalization_frame(df)
    frame = frame[frame["fe"].isin(["fixated", "never"])]
    cells = [cell for cell in _run_columns() if not frame[(frame["pipeline"] == cell[0]) & (frame["tier"] == cell[1])].empty]
    if not cells:
        return None
    fig, axes = _new_figure(len(cells), 2, 9, 2.8 * len(cells))
    sources = list(SOURCE_PREFIX)
    width = 0.2
    for row, (pipeline, tier) in enumerate(cells):
        block = frame[(frame["pipeline"] == pipeline) & (frame["tier"] == tier)]
        for col, fe in enumerate(("fixated", "never")):
            ax = axes[row, col]
            for s_index, source in enumerate(sources):
                acc = [_mean(block.loc[(block["hint"] == h) & (block["gaze_source"] == source) & (block["fe"] == fe), "correct"].astype(float)) for h in HINTS]
                ax.bar(np.arange(len(HINTS)) + (s_index - 1.5) * width, np.nan_to_num(acc), width * 0.9,
                       color=PALETTE[s_index], label=f"{source} ({SOURCE_PREFIX[source]})")
            ax.set_xticks(np.arange(len(HINTS)), [f"-{h}" for h in HINTS])
            ax.set_ylim(0, 1)
            ax.set_title(f"{pipeline} {tier}: {fe}", fontsize=9, color=INK)
            if col == 0:
                ax.set_ylabel("QA-Acc")
    axes[0, 1].legend(fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    return _save(fig, path)


def figure_pg_vs_gaze(t5: pd.DataFrame, gaze_stats: pd.DataFrame, path: Path) -> Path | None:
    if t5.empty or gaze_stats.empty:
        return None
    per = t5[t5["group_type"] == "participant"]
    pg = per.groupby("group")["pg_shift"].mean()
    merged = gaze_stats.assign(group=gaze_stats["participant"].astype(str)).merge(pg.rename("pg").reset_index(), on="group")
    merged = merged[np.isfinite(merged["pg"])]
    if merged.empty:
        return None
    stats = [("fixation_rate_per_min", "fixations / min"), ("dispersion_px", "dispersion (px)"), ("center_distance_px", "distance from centre (px)")]
    fig, axes = _new_figure(1, 3, 11, 3.6)
    for ax, (column, label) in zip(axes[0], stats):
        ax.scatter(merged[column], merged["pg"], s=64, color=PALETTE[0], edgecolor="white", linewidth=2, zorder=3)
        for row in merged.itertuples(index=False):
            ax.annotate(f"P{row.participant}", (getattr(row, column), row.pg), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK)
        ax.axhline(0, color=NEUTRAL, linewidth=1)
        ax.set_xlabel(label)
    axes[0, 0].set_ylabel("PG = Acc(C3) - Acc(C8shift)")
    fig.suptitle("Per-participant personalization gain vs gaze behaviour (descriptive)", fontsize=10, color=INK)
    return _save(fig, path)


def figure_alpha_dose(metrics: pd.DataFrame, path: Path) -> Path | None:
    alpha_names = set(ALPHA_CONDITIONS.values())
    pipelines = [p for p in PIPELINES if not metrics[(metrics["pipeline"] == p) & metrics["condition"].isin(alpha_names)].empty]
    if not pipelines:
        return None
    fig, axes = _new_figure(1, len(pipelines), 4.5 * len(pipelines), 3.8)
    for ax, pipeline in zip(axes[0], pipelines):
        block = metrics[metrics["pipeline"] == pipeline]
        for t_index, tier in enumerate(TIERS):
            color = PALETTE[t_index]
            points = [
                (alpha, float(block.loc[(block["tier"] == tier) & (block["condition"] == name), "qa_acc"].iloc[0]))
                for alpha, name in ALPHA_CONDITIONS.items()
                if not block[(block["tier"] == tier) & (block["condition"] == name)].empty
            ]
            if points:
                ax.plot(*zip(*points), marker="o", markersize=6, linewidth=2, color=color, label=f"C6(alpha) {tier}")
            c0 = block[(block["tier"] == tier) & (block["condition"] == "C0")]
            if not c0.empty:
                ax.axhline(float(c0["qa_acc"].iloc[0]), color=color, linestyle="--", linewidth=1, label=f"C0 {tier}")
        ax.set_xticks(list(ALPHA_CONDITIONS))
        ax.set_xlabel("alpha")
        ax.set_ylabel("QA-Acc")
        ax.set_title(f"C6 alpha dose: {pipeline}", fontsize=10, color=INK)
        ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    return _save(fig, path)


def build_figures(
    df: pd.DataFrame, metrics: pd.DataFrame, t0: pd.DataFrame, t5: pd.DataFrame,
    gaze_stats: pd.DataFrame, figs_dir: Path,
) -> list[Path]:
    figures: list[Path | None] = [figure_fe_by_category(t0, figs_dir / "fig1_fe_by_category.png")]
    figures += figure_condition_by_fe(df, figs_dir)
    figures.append(figure_budget_slopes(metrics, figs_dir / "fig3_budget_slopes.png"))
    figures.append(figure_personalization(df, figs_dir / "fig4_personalization.png"))
    figures.append(figure_pg_vs_gaze(t5, gaze_stats, figs_dir / "fig5_pg_vs_gaze_behaviour.png"))
    figures.append(figure_alpha_dose(metrics, figs_dir / "fig6_c6_alpha_dose.png"))
    return [figure for figure in figures if figure is not None]


# --------------------------------------------------------------------------
# Failure cases
# --------------------------------------------------------------------------


def select_failure_cases(df: pd.DataFrame, pipeline: str, tier: str, condition: str, n: int = 5) -> pd.DataFrame:
    """Deterministic pick of wrong answers: round-robin over categories, each
    category ordered by question_id (parsed outputs first)."""
    wrong = run_frame(df, pipeline, tier, condition)
    wrong = wrong[~wrong["correct"]]
    if wrong.empty:
        return wrong
    wrong = wrong.assign(_failed=(wrong["parse_status"] == "failed").astype(int)).sort_values(["_failed", "question_id"])
    queues = {c: list(block.index) for c, block in wrong.groupby("category", sort=True)}
    picked: list[Any] = []
    while len(picked) < n and any(queues.values()):
        for category in sorted(queues):
            if queues[category] and len(picked) < n:
                picked.append(queues[category].pop(0))
    return wrong.loc[picked].drop(columns="_failed")


def load_unit_index(paths: Paths, tier: str, condition: str) -> dict[tuple[str, float, float], dict[str, Any]]:
    path = paths.memory_dir("egobutler", tier, condition) / "units.jsonl"
    if not path.is_file():
        return {}
    index = {}
    for unit in read_jsonl(path):
        if unit.get("video_id") is None or unit.get("start") is None:
            continue
        index[(str(unit["video_id"]), round(float(unit["start"]), 3), round(float(unit["end"]), 3))] = unit
    return index


def case_frames(row: Any, units: dict[tuple[str, float, float], dict[str, Any]], max_frames: int = 8) -> list[dict[str, Any]]:
    """Frames to show for one prediction: its own frame_meta / frame_times
    (Video-RAG) or the frames of the retrieved memory units (EgoButler)."""
    retrieved = list(row.retrieved or [])
    shard = next((r for r in retrieved if r.get("db") == "SHARD"), retrieved[0] if retrieved else {})
    video_id = shard.get("video_id")
    frames: list[dict[str, Any]] = []
    if row.frame_meta:
        frames = [{"video_id": video_id, "t": m.get("t"), "box": m.get("gaze_box")} for m in row.frame_meta if isinstance(m, dict)]
    elif row.frame_times:
        frames = [{"video_id": video_id, "t": t, "box": None} for t in row.frame_times]
    else:
        for item in retrieved:
            if item.get("video_id") is None or item.get("start") is None or item.get("end") is None:
                continue
            unit = units.get((str(item["video_id"]), round(float(item["start"]), 3), round(float(item["end"]), 3)))
            if unit is None:
                continue
            if unit.get("frame_meta"):
                frames += [{"video_id": unit["video_id"], "t": m.get("t"), "box": m.get("gaze_box")} for m in unit["frame_meta"]]
            else:
                frames += [{"video_id": unit["video_id"], "t": t, "box": None} for t in unit.get("frame_times") or []]
    frames = [f for f in frames if f["video_id"] is not None and f["t"] is not None]
    if len(frames) > max_frames:
        picks = np.unique(np.linspace(0, len(frames) - 1, max_frames).round().astype(int))
        frames = [frames[i] for i in picks]
    return frames


def _ascii(text: Any) -> str:
    return str(text).encode("ascii", "replace").decode("ascii")


def render_mosaic(paths: Paths, frames: list[dict[str, Any]], lines: list[str], output: Path, tile: int = 320) -> Path:
    """Up to 8 frames (4 per row) with gaze boxes, plus a text panel."""
    import cv2

    columns = 4
    tiles = []
    for frame in frames:
        image = None
        path = cached_frame_path(paths, str(frame["video_id"]), int(round(float(frame["t"]))))
        if path.is_file():
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            image = np.full((tile, tile, 3), 200, np.uint8)
            cv2.putText(image, "frame missing", (10, tile // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
        else:
            image = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)
        box = frame.get("box")
        if box is not None and len(box) == 4:
            scale = tile / FRAME_SIZE
            x1, y1, x2, y2 = (int(round(float(value) * scale)) for value in box)
            cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 2)
        label = f"t={float(frame['t']):.0f}s  {frame['video_id'].split('_glasses')[0]}"
        cv2.putText(image, _ascii(label)[:44], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(image)
    width = columns * tile
    grid = None
    if tiles:
        while len(tiles) % columns:
            tiles.append(np.full((tile, tile, 3), 255, np.uint8))
        grid = np.vstack([np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)])
    wrapped: list[str] = []
    for line in lines:
        wrapped += textwrap.wrap(_ascii(line), width=width // 8) or [""]
    wrapped = wrapped[:40]
    panel = np.full((22 * len(wrapped) + 16, width, 3), 255, np.uint8)
    for index, line in enumerate(wrapped):
        cv2.putText(panel, line, (8, 22 * (index + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    mosaic = panel if grid is None else np.vstack([grid, panel])
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), mosaic):
        raise RuntimeError(f"Cannot write {output}")
    return output


def _option_text(question: dict[str, Any] | None, index: Any) -> str:
    if question is None or index is None:
        return "?"
    choices = question.get("choices") or []
    try:
        choice = choices[int(index)]
    except (IndexError, TypeError, ValueError):
        return "?"
    return f"{choice.get('text', '?')} [{choice.get('type', '?')}]"


def write_failure_cases(paths: Paths, df: pd.DataFrame, questions: pd.DataFrame, n: int = 5) -> list[dict[str, Any]]:
    root = paths.results / "failure_cases"
    by_id = {row["question_id"]: row for row in questions.to_dict(orient="records")}
    cases: list[dict[str, Any]] = []
    for (pipeline, tier), group in _groups_by_run_column(df):
        for condition in FAILURE_CONDITIONS:
            if condition not in set(group["condition"]):
                continue
            picked = select_failure_cases(df, pipeline, tier, condition, n)
            units = load_unit_index(paths, tier, condition) if pipeline == "egobutler" else {}
            for row in picked.itertuples(index=False):
                question = by_id.get(row.question_id)
                chosen = _option_text(question, row.top1)
                correct = _option_text(question, (question or {}).get("correct_option_index"))
                lines = [
                    f"{pipeline} {tier} {condition} | Q{row.question_id} | P{row.participant} | {row.category} | F(E)={row.fe}",
                    f"Question: {(question or {}).get('question', '?')}",
                    f"Chosen: {chosen}",
                    f"Correct: {correct}",
                    f"Raw output: {row.raw_output}",
                ]
                frames = case_frames(row, units)
                output = root / f"{pipeline}_{tier}_{condition}" / f"{row.question_id}.png"
                render_mosaic(paths, frames, lines, output)
                cases.append({
                    "pipeline": pipeline, "tier": tier, "condition": condition, "question_id": row.question_id,
                    "category": row.category, "fe": row.fe, "question": (question or {}).get("question", ""),
                    "chosen": chosen, "correct": correct, "raw_output": row.raw_output, "n_frames": len(frames),
                    "image": str(output.relative_to(paths.results)),
                })
    lines = ["# Failure cases", "", "Five wrong answers per key condition (deterministic: round-robin over categories, "
             "ordered by question id). Red boxes are the gaze boxes used by the condition.", ""]
    for key, block in pd.DataFrame(cases).groupby(["pipeline", "tier", "condition"], sort=False) if cases else []:
        lines += [f"## {' '.join(key)}", ""]
        for case in block.to_dict(orient="records"):
            lines += [
                f"- **Q{case['question_id']}** ({case['category']}, F(E)={case['fe']}): {case['question']}",
                f"  - chosen: {case['chosen']}; correct: {case['correct']}",
                f"  - raw output: `{_ascii(case['raw_output']).replace('`', '')}`; frames: {case['n_frames']}",
                f"  - ![Q{case['question_id']}](../{case['image']})",
            ]
        lines.append("")
    if not cases:
        lines.append("_(no key-condition runs with wrong answers)_")
    root.mkdir(parents=True, exist_ok=True)
    (root / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cases


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _read_optional(path: Path, limit: int = 4000) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if len(text) <= limit else text[:limit] + "\n... (truncated)"


def collect_token_audits(paths: Paths) -> pd.DataFrame:
    """Token audits recorded in run manifests (README 7.2)."""
    rows = []
    for kind in ("preds", "memory"):
        root = paths.work / kind
        if not root.is_dir():
            continue
        for manifest in sorted(root.glob("*/*/*/manifest.json")):
            pipeline, tier, condition = manifest.parts[-4:-1]
            if condition not in CONDITIONS:
                continue
            try:
                audit = json.loads(manifest.read_text(encoding="utf-8")).get("parameters", {}).get("token_audit")
            except (OSError, json.JSONDecodeError):
                continue
            if not audit:
                continue
            rows.append({"source": kind, "pipeline": pipeline, "tier": tier, "condition": condition,
                         "target": audit.get("target_tokens_per_frame"),
                         "measured": audit.get("measured_tokens_per_frame"),
                         "deviation": audit.get("deviation"), "within_5pct": audit.get("within_5pct")})
    return pd.DataFrame(rows, columns=["source", "pipeline", "tier", "condition", "target", "measured", "deviation", "within_5pct"])


def _model_markdown(name: str, result: dict[str, Any]) -> str:
    lines = [f"- **{name}**: {result.get('model_used')} (n = {result.get('n', 0)}); {result.get('reason', '')}"]
    if result.get("formula"):
        lines.append(f"  - formula: `{result['formula']}`" + (f"; random effects: {result.get('random_effects')}" if result.get("random_effects") else ""))
    key = result.get("key_terms")
    if isinstance(key, pd.DataFrame) and not key.empty:
        shown = key.assign(coef=key["coef"].map(lambda v: f"{v:+.3f}"), se=key["se"].map(lambda v: f"{v:.3f}"),
                           p=key["p"].map(_fmt_p))[["term", "coef", "se", "p"]]
        lines.append("")
        lines.append(textwrap.indent(to_markdown(shown), "  "))
    return "\n".join(lines)


def _verdict_markdown(verdicts: list[dict[str, Any]], tests: pd.DataFrame) -> str:
    out = []
    for verdict in verdicts:
        out += [f"#### {verdict['hypothesis']}: **{verdict['verdict'].upper()}** ({verdict['reason']})", "",
                f"_{verdict['statement']}_", ""]
        block = tests[(tests["hypothesis"] == verdict["hypothesis"]) & (tests["n"] > 0)]
        if block.empty:
            out += ["No paired data for any contrast of this hypothesis.", ""]
        else:
            shown = pd.DataFrame({
                "cell": block["pipeline"] + " " + block["tier"],
                "group": block["group"] + block["mode"].map(lambda m: " [report]" if m == "report" else ""),
                "contrast": block["cond_a"] + " - " + block["cond_b"],
                "subset": block["subset"],
                "metric": block["metric"],
                "n": block["n"],
                "diff x100": block["diff"].map(_signed_pct),
                "95% CI": [f"[{_signed_pct(lo)}, {_signed_pct(hi)}]" for lo, hi in zip(block["ci_lo"], block["ci_hi"])],
                "p": block["p"].map(_fmt_p),
                "p Holm": block["p_holm"].map(_fmt_p),
                "outcome": block["outcome"],
            })
            out += [to_markdown(shown)]
        cells = ", ".join(f"{c['pipeline']} {c['tier']}: {c['verdict']}" for c in verdict["cells"])
        out += [f"Per cell: {cells}", "", "Interpretation: TODO(human)", ""]
    return "\n".join(out)


def write_report(paths: Paths, context: dict[str, Any]) -> Path:
    """Assemble results/REPORT.md (README section 10) from analysis outputs."""
    results = paths.results
    inventory = context.get("inventory", {})
    metrics: pd.DataFrame = context.get("metrics", pd.DataFrame())
    tables: list[Table] = context.get("tables", [])
    tests: pd.DataFrame = context.get("tests", pd.DataFrame())
    lines: list[str] = [
        "# Gaze x egocentric memory QA: results report",
        "",
        f"Split analysed: **{context.get('split')}**; runs analysed: {len(inventory.get('present', []))}; "
        f"questions per complete run: {inventory.get('n_expected_questions')}; "
        f"bootstrap replicates: {context.get('n_boot', N_BOOT)} (seed {SEED}).",
        "This report is generated by `smvqa_gaze.hpc.analysis`; every `TODO(human)` marks text that needs a human author.",
        "",
        "## 1. Environment, data and differences from the paper setup",
        "",
    ]
    env = None
    for candidate in (paths.data / "docs" / "ENV_REPORT.md", paths.work / "env" / "ENV_REPORT.md", paths.work / "ENV_REPORT.md"):
        env = _read_optional(candidate)
        if env is not None:
            lines += [f"Environment report (`{candidate}`):", "", "<details><summary>ENV_REPORT.md</summary>", "", env, "", "</details>", ""]
            break
    if env is None:
        lines += ["- ENV_REPORT.md not found (looked under data/docs and work/env). TODO(human)", ""]
    plan = _read_optional(results / "COMPUTE_PLAN.md")
    lines += ["- Compute plan: " + ("[COMPUTE_PLAN.md](COMPUTE_PLAN.md)" if plan is not None else "COMPUTE_PLAN.md not found."), ""]
    lines += [
        "Known differences from the SuperMemory-VQA paper setup (fixed by the protocol, README 1 and 7):",
        "",
        "- Answer/caption model Qwen3-VL-8B-Instruct (bf16) everywhere; EgoButler hour/day summaries use the same Qwen model instead of Gemini-3-Flash.",
        "- Video-RAG object detection uses GroundingDINO-tiny (not APE); retrieval uses Contriever + FAISS; OCR uses EasyOCR.",
        "- Frames are given as image lists at fixed per-frame token budgets (B1 = 512 px, B2 = 1024 px) instead of video mode.",
        "- Only the 4 selected question categories and the selected sessions (README 3) are used.",
        "- Video orientation / gaze projection handling: TODO(human) (see the local L3 projection checks).",
        "",
        "## 2. Quality checks",
        "",
    ]
    lines += _qc_section(paths, context, metrics)
    lines += ["## 3. Tables and figures", ""]
    for table in tables:
        lines += [table.markdown, f"(CSV: [{table.name}.csv]({table.name}.csv))", ""]
    figures = context.get("figures", [])
    if figures:
        lines += ["Figures:", ""]
        for figure in figures:
            relative = Path(figure).relative_to(results)
            lines += [f"![{relative.stem}]({relative.as_posix()})", ""]
    else:
        lines += ["No figures were produced (figures disabled or no data).", ""]
    lines += ["Figure 2 shows the P0 conditions plus C4; all conditions are in T3. Figure 5 is descriptive only (3-4 participants, no correlation test).", ""]
    lines += ["## 4. Hypotheses H1-H6", "", "```text", DECISION_RULES.rstrip(), "```", ""]
    verdicts = context.get("verdicts", [])
    summary_table = pd.DataFrame([{"hypothesis": v["hypothesis"], "verdict": v["verdict"], "reason": v["reason"]} for v in verdicts])
    lines += [to_markdown(summary_table), ""]
    lines += [_verdict_markdown(verdicts, tests) if not tests.empty else "_(no tests)_", ""]
    lines += ["### Logistic models", ""]
    for name, result in context.get("models", {}).items():
        lines += [_model_markdown(name, result), ""]
    lopo: pd.DataFrame = context.get("lopo", pd.DataFrame())
    lines += ["### Leave-one-participant-out", ""]
    if lopo.empty:
        lines += ["No LOPO contrasts could be computed (missing runs).", ""]
    else:
        summary = lopo[lopo["left_out"] == "none"][["pipeline", "tier", "contrast", "n", "diff", "sign_consistent"]].copy()
        ranges = lopo[lopo["left_out"] != "none"].groupby(["pipeline", "tier", "contrast"])["diff"].agg(["min", "max"]).reset_index()
        summary = summary.merge(ranges, on=["pipeline", "tier", "contrast"], how="left")
        summary["diff"] = summary["diff"].map(_signed_pct)
        summary["LOPO range"] = [f"[{_signed_pct(a)}, {_signed_pct(b)}]" for a, b in zip(summary.pop("min"), summary.pop("max"))]
        lines += [to_markdown(summary), "Full table: [stats/lopo.csv](stats/lopo.csv)", ""]
    nss: pd.DataFrame = context.get("nss", pd.DataFrame())
    if not nss.empty:
        finite = nss["nss"].astype(float)
        lines += [f"NSS consistency (README 7.9): {int(np.isfinite(finite).sum())} questions with NSS, median = {np.nanmedian(finite):.3f}; "
                  "see [stats/nss_by_question.csv](stats/nss_by_question.csv).", ""]
    lines += ["Overall interpretation: TODO(human)", "", "## 5. Failure cases", ""]
    cases = context.get("failure_cases", [])
    if cases:
        lines += [f"{len(cases)} cases; full index with mosaics: [failure_cases/INDEX.md](failure_cases/INDEX.md)", ""]
        frame = pd.DataFrame(cases)
        for key, block in frame.groupby(["pipeline", "tier", "condition"], sort=False):
            lines += [f"- {' '.join(key)}: " + ", ".join(f"[Q{c}]({img})" for c, img in zip(block["question_id"], block["image"]))]
        lines += ["", "Typical failure patterns: TODO(human)", ""]
    else:
        lines += ["No failure cases (no key-condition runs).", ""]
    lines += ["## 6. Known limitations / not completed", ""]
    lines += _limitations(context)
    report = results / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _qc_section(paths: Paths, context: dict[str, Any], metrics: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    inventory = context.get("inventory", {})
    lines += [f"- Runs present: {len(inventory.get('present', []))}; missing P0-P2 runs: {len(inventory.get('missing', []))}; "
              f"incomplete runs: {len(inventory.get('incomplete', []))} (details in section 6).", ""]
    if not metrics.empty:
        c0 = metrics[metrics["condition"] == "C0"]
        rows = []
        for row in c0.itertuples(index=False):
            paper = PAPER_BASELINE.get(row.pipeline, {})
            rows.append({"pipeline": row.pipeline, "tier": row.tier,
                         "Ans-F1 / Acc / MRR (ours)": f"{_pct(row.ans_f1)} / {_pct(row.qa_acc)} / {_pct(row.qa_mrr)}",
                         "paper (full benchmark)": f"{paper.get('ans_f1', '')} / {paper.get('qa_acc', '')} / {paper.get('qa_mrr', '')}"})
        lines += ["Baseline reproduction (C0 vs. paper; numbers need not match, README 8):", "", to_markdown(pd.DataFrame(rows)), ""]
        parse = metrics[(metrics["parse_failure_rate"].fillna(0) > 0) | (metrics["parse_partial_rate"].fillna(0) > 0)]
        if parse.empty:
            lines += ["- Answer parsing: no failed or partial rankings.", ""]
        else:
            shown = parse[RUN_KEYS].assign(failed=parse["parse_failure_rate"].map(_pct), partial=parse["parse_partial_rate"].map(_pct))
            lines += ["Answer parsing (x100, failed outputs count as wrong):", "", to_markdown(shown), ""]
        stored = metrics.dropna(subset=["metrics_json_qa_acc"]) if "metrics_json_qa_acc" in metrics else pd.DataFrame()
        if not stored.empty:
            mismatch = stored[(stored["metrics_json_qa_acc"] - stored["qa_acc"]).abs() > 1e-9]
            lines += [f"- metrics.json consistency: {len(stored) - len(mismatch)}/{len(stored)} runs match the recomputed QA-Acc"
                      + ("" if mismatch.empty else " (mismatch: " + ", ".join(f"{r.pipeline}/{r.tier}/{r.condition}" for r in mismatch.itertuples()) + "; metrics.json may include other splits or be stale)")
                      + ".", ""]
    audits = collect_token_audits(paths)
    if not audits.empty:
        audits = audits.assign(deviation=audits["deviation"].map(lambda v: "" if _missing(v) else f"{100 * v:+.1f}%"))
        lines += ["Token audit from run manifests (README 7.2; allowed exceptions: C4 at B1 about -5.5%, C3-M/C9-M at B1 about +6%):", "", to_markdown(audits), ""]
    token_csv = paths.results / "token_audit.csv"
    if token_csv.is_file():
        try:
            lines += ["Token audit (`results/token_audit.csv`):", "", to_markdown(pd.read_csv(token_csv).head(80)), ""]
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            lines += ["- `results/token_audit.csv` exists but could not be parsed.", ""]
    if audits.empty and not token_csv.is_file():
        lines += ["- Token audit: no manifest audits and no results/token_audit.csv found. TODO(human)", ""]
    shift = _read_optional(paths.work / "cache" / "shift_check.json", 3000)
    lines += (["C8-shift validity check (`cache/shift_check.json`, README 7.8):", "", "```json", shift.strip(), "```", ""]
              if shift is not None else ["- C8-shift validity check: cache/shift_check.json not found.", ""])
    lines += [
        "- Gaze overlay / projection checks and F(E) label agreement: produced locally (H1); summarize here. TODO(human)",
        "",
    ]
    return lines


def _limitations(context: dict[str, Any]) -> list[str]:
    inventory = context.get("inventory", {})
    lines: list[str] = []
    missing = inventory.get("missing", [])
    if missing:
        lines += [f"Missing runs (P0-P2, {len(missing)}):", ""]
        frame = pd.DataFrame(missing)
        for (priority, condition), block in frame.groupby(["priority", "condition"], sort=False):
            cells = ", ".join(f"{r.pipeline} {r.tier}" for r in block.itertuples(index=False))
            lines.append(f"- {priority} {condition}: {cells}")
        lines.append("")
    else:
        lines += ["- All P0-P2 runs are present.", ""]
    optional = inventory.get("optional_missing", [])
    if optional:
        names = sorted({item["condition"] for item in optional}, key=condition_sort_key)
        lines += [f"- Optional P3 runs not present: {', '.join(names)}.", ""]
    for item in inventory.get("incomplete", []):
        lines.append(f"- Incomplete run {item['pipeline']} {item['tier']} {item['condition']}: {item['n']}/{item['n_expected']} questions.")
    if inventory.get("incomplete"):
        lines.append("")
    for name, result in context.get("models", {}).items():
        if str(result.get("model_used", "")).startswith("GLM"):
            lines.append(f"- {name}: mixed model not used ({result.get('reason')}); fixed-effect logistic regression with participant as fixed effect reported instead.")
        elif result.get("model_used") in ("skipped", "failed"):
            lines.append(f"- {name}: {result.get('model_used')} ({result.get('reason')}).")
    for error in context.get("errors", []):
        lines.append(f"- Analysis step failed: {error['step']}: {error['error']}")
    inconclusive = [v["hypothesis"] for v in context.get("verdicts", []) if v["verdict"] == "inconclusive"]
    if inconclusive:
        lines.append(f"- Inconclusive hypotheses: {', '.join(inconclusive)}.")
    lines += [
        "- Only 3-4 participants: random-effect variances are poorly identified and participant-level analyses are descriptive.",
        "- Further limitations and deviations: TODO(human)",
        "",
    ]
    return lines


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def _terms_to_csv(result: dict[str, Any], path: Path) -> None:
    terms = result.get("terms")
    if isinstance(terms, pd.DataFrame) and not terms.empty:
        terms.to_csv(path, index=False)


def _model_json(result: dict[str, Any]) -> dict[str, Any]:
    output = {key: value for key, value in result.items() if not isinstance(value, pd.DataFrame)}
    key = result.get("key_terms")
    if isinstance(key, pd.DataFrame):
        output["key_terms"] = _clean_records(key)
    return output


def run_analysis(
    paths: Paths,
    config: RunConfig,
    split: str = "test",
    make_figures: bool = True,
    n_boot: int | None = None,
    prefer_mixed: bool = True,
) -> dict[str, Any]:
    """README H4: tables, figures, statistics, failure cases and REPORT.md."""
    n_boot = int(n_boot if n_boot is not None else config.extra.get("analysis_n_boot", N_BOOT))
    results = paths.results
    stats_dir = results / "stats"
    figs_dir = results / "figs"
    stats_dir.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, str]] = []

    def guarded(step: str, function: Callable[[], Any], default: Any) -> Any:
        try:
            return function()
        except Exception as exc:  # noqa: BLE001 - recorded in the report, never silent
            errors.append({"step": step, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
            return default

    questions = load_questions(paths)
    full = load_runs(paths, split, questions)
    # Statistics only need scalar columns; dropping the list-valued ones makes
    # the many per-run selections much cheaper. Failure cases use ``full``.
    df = full.drop(columns=list(HEAVY_COLUMNS))
    metrics = attach_stored_metrics(run_metrics(df, config.ans_f1_positive), paths)
    metrics.to_csv(results / "metrics_all.csv", index=False)
    inventory = run_inventory(metrics, questions, split)

    tables = build_tables(df, metrics, questions, split)
    table_files = {table.name: write_table(table, results) for table in tables}

    accuracy_ci = bootstrap_accuracy(df, n_boot)
    accuracy_ci.to_csv(stats_dir / "bootstrap_acc.csv", index=False)
    pairwise = pairwise_vs_c0(df, n_boot)
    pairwise.to_csv(stats_dir / "pairwise_vs_c0.csv", index=False)

    split_questions = questions if split in (None, "all") else questions[questions["split"] == split]
    nss = guarded("NSS consistency", lambda: load_or_compute_nss(paths, split_questions), pd.DataFrame())

    models: dict[str, Any] = {}
    for pipeline in PIPELINES:
        if df[df["pipeline"] == pipeline].empty:
            continue
        models[f"main model ({pipeline})"] = guarded(
            f"main model {pipeline}", lambda p=pipeline: main_model(df, p, prefer_mixed=prefer_mixed),
            {"model_used": "failed", "reason": "exception", "terms": pd.DataFrame()},
        )
        personal = guarded(
            f"personalization model {pipeline}",
            lambda p=pipeline: personalization_models(df, p, nss, prefer_mixed),
            {"base": {"model_used": "failed", "reason": "exception"}, "nss": {"model_used": "failed", "reason": "exception"}},
        )
        models[f"personalization model ({pipeline})"] = personal["base"]
        models[f"personalization x NSS group ({pipeline})"] = personal["nss"]
    for name, result in models.items():
        slug = name.replace(" ", "_").replace("(", "").replace(")", "")
        _terms_to_csv(result, stats_dir / f"{slug}.csv")
    _write_json(stats_dir / "models.json", {name: _model_json(result) for name, result in models.items()})

    lopo = leave_one_participant_out(df)
    lopo.to_csv(stats_dir / "lopo.csv", index=False)
    tests = hypothesis_tests(df, n_boot)
    tests.to_csv(stats_dir / "hypothesis_tests.csv", index=False)
    verdicts = hypothesis_verdicts(tests)
    _write_json(stats_dir / "verdicts.json", {"decision_rules": DECISION_RULES, "verdicts": verdicts})

    gaze_stats = guarded("participant gaze statistics", lambda: participant_gaze_stats(paths), pd.DataFrame())
    if isinstance(gaze_stats, pd.DataFrame) and not gaze_stats.empty:
        gaze_stats.to_csv(stats_dir / "participant_gaze_stats.csv", index=False)
    figures: list[Path] = []
    if make_figures:
        t0 = tables[0].data
        t5 = next(table.data for table in tables if table.name == "T5_personalization")
        stats_frame = gaze_stats if isinstance(gaze_stats, pd.DataFrame) else pd.DataFrame()
        figures = guarded("figures", lambda: build_figures(df, metrics, t0, t5, stats_frame, figs_dir), [])
    cases = guarded("failure cases", lambda: write_failure_cases(paths, full, questions), [])

    context = {
        "split": split, "n_boot": n_boot, "inventory": inventory, "metrics": metrics, "tables": tables, "tests": tests,
        "verdicts": verdicts, "models": models, "lopo": lopo, "nss": nss if isinstance(nss, pd.DataFrame) else pd.DataFrame(),
        "figures": figures, "failure_cases": cases, "errors": errors,
    }
    report = write_report(paths, context)
    summary = {
        "split": split,
        "n_runs": len(inventory["present"]),
        "runs": inventory["present"],
        "missing_runs": inventory["missing"],
        "incomplete_runs": inventory["incomplete"],
        "n_rows": int(len(df)),
        "n_boot": n_boot,
        "tables": table_files,
        "figures": [str(path) for path in figures],
        "verdicts": {v["hypothesis"]: v["verdict"] for v in verdicts},
        "models": {name: result.get("model_used") for name, result in models.items()},
        "n_failure_cases": len(cases),
        "errors": [{"step": e["step"], "error": e["error"]} for e in errors],
        "report": str(report),
    }
    _write_json(results / "analysis_summary.json", summary)
    write_stage_manifest(
        results, "analysis",
        inputs=[paths.qa_subset, paths.fe_labels, paths.sessions],
        parameters={"split": split, "n_boot": n_boot, "seed": SEED, "prefer_mixed": prefer_mixed,
                    "ans_f1_positive": config.ans_f1_positive},
        outputs={"report": str(report), "n_runs": summary["n_runs"]},
    )
    return summary

