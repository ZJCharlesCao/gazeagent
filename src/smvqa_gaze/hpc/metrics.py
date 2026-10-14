from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def answerability_f1(records: list[dict[str, Any]]) -> dict[str, float]:
    """Ans-F1. The paper does not publish its positive class, so both class F1s
    and their macro mean are reported; ``ans_f1`` uses RunConfig.ans_f1_positive."""
    counts = {"answerable": [0, 0, 0], "unanswerable": [0, 0, 0]}
    for row in records:
        truth = "answerable" if row["is_answerable"] else "unanswerable"
        predicted = "unanswerable" if row["abstained"] else "answerable"
        for label in counts:
            tp, fp, fn = counts[label]
            if predicted == label and truth == label:
                tp += 1
            elif predicted == label:
                fp += 1
            elif truth == label:
                fn += 1
            counts[label] = [tp, fp, fn]
    answerable = _f1(*counts["answerable"])
    unanswerable = _f1(*counts["unanswerable"])
    return {
        "ans_f1_answerable": answerable,
        "ans_f1_unanswerable": unanswerable,
        "ans_f1_macro": (answerable + unanswerable) / 2,
    }


def _mean(values: Iterable[float]) -> float | None:
    values = [value for value in values if value is not None and not (isinstance(value, float) and math.isnan(value))]
    return sum(values) / len(values) if values else None


def summarize(records: list[dict[str, Any]], ans_f1_positive: str = "answerable") -> dict[str, Any]:
    if not records:
        return {"n": 0}
    f1 = answerability_f1(records)
    answerable = [row for row in records if row["is_answerable"]]
    with_vague = [row for row in records if row.get("has_vague", True) and row["is_answerable"]]
    with_evidence = [row for row in records if row.get("retrieval_hit") is not None]
    return {
        "n": len(records),
        "ans_f1": f1[f"ans_f1_{ans_f1_positive}"] if ans_f1_positive != "macro" else f1["ans_f1_macro"],
        **f1,
        "qa_acc": _mean(float(row["correct"]) for row in records),
        "qa_mrr": _mean(1.0 / row["rank_of_correct"] for row in records),
        "false_abstention_rate": _mean(float(row["abstained"]) for row in answerable),
        "vague_rate": _mean(float(row["top1_type"] == "vague") for row in with_vague),
        "recall_at_k": _mean(float(row["retrieval_hit"]) for row in with_evidence),
        "target_coverage": _mean(
            float(row["target_coverage"]) for row in records if row.get("target_coverage") is not None
        ),
        "parse_failure_rate": _mean(float(row["parse_status"] == "failed") for row in records),
        "parse_partial_rate": _mean(float(row["parse_status"] == "partial") for row in records),
    }


def compute_metrics(records: list[dict[str, Any]], ans_f1_positive: str = "answerable") -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_fe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_category[row["category"]].append(row)
        by_fe[str(row.get("fe_status"))].append(row)
    return {
        "overall": summarize(records, ans_f1_positive),
        "by_category": {key: summarize(rows, ans_f1_positive) for key, rows in sorted(by_category.items())},
        "by_fe_status": {key: summarize(rows, ans_f1_positive) for key, rows in sorted(by_fe.items())},
        "ans_f1_positive_class": ans_f1_positive,
        "definitions": {
            "qa_acc": "top-1 option equals the correct option",
            "qa_mrr": "mean of 1 / rank of the correct option in the model's full ranking",
            "ans_f1": "F1 of answerable-vs-unanswerable; a prediction is 'unanswerable' "
            "when the top-1 option is the canonical abstention option",
            "false_abstention_rate": "answerable questions whose top-1 is the abstention option",
            "vague_rate": "answerable questions with a vague option whose top-1 is that option",
            "recall_at_k": "retrieved memory units overlapping any answer-evidence span "
            "(questions with timed evidence only)",
            "target_coverage": "memory text overlapping the evidence spans mentions an "
            "evidence target (string hit or embedding similarity > threshold)",
        },
    }
