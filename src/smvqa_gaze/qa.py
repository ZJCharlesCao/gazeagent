from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


CATEGORY_NAMES = {
    "object_location_memory": "Object & Location Memory",
    "visual_recall": "Visual Scene Recall",
    "intent_recall": "Intent Recall",
    "conversational_memory": "Conversational Memory",
}
SESSION_RE = re.compile(r"^Person_(?P<person>\d+)_session_(?P<session>\d+)_", re.IGNORECASE)


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a JSON array of QA objects in {path}")
    return rows


ABSTAIN_OPTION_TEXT = "this question can not be answered."


def is_abstain_option(text: str) -> bool:
    """Match the benchmark's canonical abstention choice exactly.

    Distractors such as "This question cannot be answered because ..." are
    ordinary wrong answers, so only the canonical sentence is an abstention.
    """
    normalized = " ".join(str(text).split()).casefold().replace("cannot", "can not")
    return normalized == ABSTAIN_OPTION_TEXT


def _normalized_option_type(value: Any, text: str) -> str:
    """Return the content role of an option: correct/vague/wrong/unanswerable.

    The released ``choice_types`` mark the abstention option as ``incorrect`` for
    answerable questions and as ``correct`` for unanswerable ones. The role is
    therefore taken from the canonical text first; correctness is stored
    separately in ``is_correct``.
    """
    if is_abstain_option(text):
        return "unanswerable"
    option_type = str(value or "").strip().lower()
    if option_type in {"correct", "vague"}:
        return option_type
    if option_type in {"incorrect", "wrong", "distractor"}:
        return "wrong"
    raise ValueError(f"Unknown choice type {value!r}")


def _evidence_spans(answer_evidence: Any) -> list[dict[str, Any]]:
    if not isinstance(answer_evidence, dict):
        return []
    evidence = answer_evidence.get("evidence_list", [])
    if not isinstance(evidence, list):
        return []
    spans = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        time_span = item.get("time_span") or {}
        if not isinstance(time_span, dict):
            time_span = {}
        spans.append(
            {
                "video_id": item.get("video_id"),
                "t_start": time_span.get("start_time"),
                "t_end": time_span.get("end_time"),
                "video_start_time_unix": item.get("start_time"),
                "modalities": item.get("modalities", []),
                "room": item.get("room"),
                "text": item.get("text"),
            }
        )
    return spans


def normalize_qa(row: dict[str, Any]) -> dict[str, Any]:
    required = {
        "question_id",
        "question",
        "choices",
        "choice_types",
        "subject",
        "metadata",
        "video_ids",
        "start_time",
        "answer_evidence",
    }
    missing = required.difference(row)
    if missing:
        raise ValueError(f"QA row missing required fields: {sorted(missing)}")

    metadata = row["metadata"] if isinstance(row["metadata"], dict) else {}
    primary_video_id = metadata.get("primary_video_id")
    match = SESSION_RE.match(str(primary_video_id or ""))
    if not match:
        raise ValueError(f"Cannot parse primary_video_id: {primary_video_id!r}")

    choices = row["choices"]
    choice_types = row["choice_types"]
    if not isinstance(choices, list) or len(choices) != 4:
        raise ValueError(f"Question {row['question_id']} must have four choices")
    if not isinstance(choice_types, list) or len(choice_types) != len(choices):
        raise ValueError(f"Question {row['question_id']} has mismatched choice types")

    correct_index = row.get("correct_option_index")
    correct_flags = [str(kind).strip().lower() == "correct" for kind in choice_types]
    if sum(correct_flags) != 1 or correct_flags.index(True) != correct_index:
        raise ValueError(
            f"Question {row['question_id']} has inconsistent correct-option fields"
        )
    abstain_indices = [
        index for index, text in enumerate(choices) if is_abstain_option(str(text))
    ]
    if len(abstain_indices) != 1:
        raise ValueError(
            f"Question {row['question_id']} must have exactly one abstention option"
        )
    is_answerable = bool(row.get("is_answerable"))
    if is_answerable == (abstain_indices[0] == correct_index):
        raise ValueError(
            f"Question {row['question_id']} answerability disagrees with its "
            "correct option"
        )

    skill = str(metadata.get("skill", ""))
    question_evidence = row.get("question_evidence")
    question_spans = (
        question_evidence.get("time_spans", [])
        if isinstance(question_evidence, dict)
        else []
    )
    matching_question_spans = [
        span
        for span in question_spans
        if isinstance(span, dict) and span.get("video_id") == primary_video_id
    ]
    question_time_s = (
        max(float(span["end_time"]) for span in matching_question_spans)
        if matching_question_spans
        and all(span.get("end_time") is not None for span in matching_question_spans)
        else None
    )
    return {
        "question_id": str(row["question_id"]),
        "question": row["question"],
        "category": CATEGORY_NAMES.get(skill, skill),
        "category_key": skill,
        "participant": int(row["subject"]),
        "primary_video_id": primary_video_id,
        "primary_session": int(match.group("session")),
        "recording_start_unix": row["start_time"],
        "question_time_s": question_time_s,
        "question_time_unix": (
            row["start_time"] + question_time_s
            if question_time_s is not None and isinstance(row["start_time"], (int, float))
            else None
        ),
        "video_ids": row["video_ids"],
        "choices": [
            {
                "text": str(text),
                "type": _normalized_option_type(kind, str(text)),
                "is_correct": index == correct_index,
                "is_abstain": index == abstain_indices[0],
            }
            for index, (text, kind) in enumerate(
                zip(choices, choice_types, strict=True)
            )
        ],
        "correct_answer": row.get("correct_answer"),
        "correct_option_index": correct_index,
        "abstain_option_index": abstain_indices[0],
        "is_answerable": is_answerable,
        "question_evidence": question_evidence,
        "answer_evidence": _evidence_spans(row["answer_evidence"]),
    }


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def category_counts(rows: Iterable[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("category_key", "unknown")) for row in rows)
