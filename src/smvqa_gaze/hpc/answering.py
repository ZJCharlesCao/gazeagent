from __future__ import annotations

import random
import re
from typing import Any

LETTERS = "ABCD"

ANSWER_INSTRUCTION = (
    "Rank all four options from most likely to least likely to be correct. "
    "Reply with only the four letters separated by '>' (for example: C>A>D>B)."
)


def option_order(question: dict[str, Any], seed: int = 0) -> list[int]:
    """Fixed per-question shuffle shared by every condition (README 7.1)."""
    order = list(range(len(question["choices"])))
    random.Random(f"{seed}:{question['question_id']}").shuffle(order)
    return order


def question_block(question: dict[str, Any], seed: int = 0) -> tuple[str, list[int]]:
    order = option_order(question, seed)
    lines = [
        f"Question (asked at {question.get('question_clock', 'the end of the memory')}): "
        f"{question['question']}",
        "Options:",
    ]
    for letter, original in zip(LETTERS, order, strict=True):
        lines.append(f"{letter}. {question['choices'][original]['text']}")
    lines.append(ANSWER_INSTRUCTION)
    return "\n".join(lines), order


def parse_ranking(text: str) -> tuple[list[int], str]:
    """Return letter positions (0-3) in ranked order and a parse status."""
    # Prefer an explicit "X>Y>..." chain; otherwise take standalone capital
    # letters in order of appearance (lower-case "a" is an article, not A).
    chain = re.search(r"\b[ABCD](?:\s*>\s*[ABCD]\b)+", text)
    letters = re.findall(r"[ABCD]", chain.group(0)) if chain else re.findall(r"\b([ABCD])\b", text)
    ranking: list[int] = []
    for letter in letters:
        index = LETTERS.index(letter)
        if index not in ranking:
            ranking.append(index)
    if not ranking:
        return [], "failed"
    status = "ok" if len(ranking) == 4 else "partial"
    ranking.extend(index for index in range(4) if index not in ranking)
    return ranking, status


def score_ranking(
    question: dict[str, Any], letter_ranking: list[int], order: list[int]
) -> dict[str, Any]:
    """Map a letter ranking back to original options and score it."""
    ranking = [order[position] for position in letter_ranking]
    correct = int(question["correct_option_index"])
    abstain = int(question["abstain_option_index"])
    top1 = ranking[0]
    return {
        "ranking": ranking,
        "top1": top1,
        "top1_type": question["choices"][top1]["type"],
        "correct": top1 == correct,
        "rank_of_correct": ranking.index(correct) + 1,
        "abstained": top1 == abstain,
    }


def answer_record(
    question: dict[str, Any],
    response: dict[str, Any] | None,
    order: list[int],
    retrieved: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = (response or {}).get("text", "")
    letter_ranking, status = parse_ranking(text)
    if status == "failed":
        # Unparseable output counts as wrong with the original order and is
        # reported separately (parse_status), never silently dropped.
        letter_ranking = sorted(range(4), key=lambda position: order[position])
        scored = score_ranking(question, letter_ranking, order)
        scored.update({"correct": False, "abstained": False})
    else:
        scored = score_ranking(question, letter_ranking, order)
    return {
        "key": str(question["question_id"]),
        "question_id": str(question["question_id"]),
        "participant": question["participant"],
        "session": question["primary_session"],
        "category": question["category_key"],
        "split": question["split"],
        "fe_status": question.get("fe_status"),
        "is_answerable": bool(question["is_answerable"]),
        "option_order": order,
        "raw_output": text,
        "parse_status": status,
        **scored,
        "retrieved": retrieved,
        "prompt_tokens": (response or {}).get("prompt_tokens"),
        "latency_s": (response or {}).get("latency_s"),
        **(extra or {}),
    }
