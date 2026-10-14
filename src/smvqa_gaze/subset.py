from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from .qa import CATEGORY_NAMES


TARGET_CATEGORIES = tuple(CATEGORY_NAMES)


@dataclass(frozen=True)
class Session:
    participant: int
    session: int
    date: str
    duration_s: float
    gaze_coverage: float
    video_available: bool = True
    vrs_available: bool = True


@dataclass(frozen=True)
class Candidate:
    duration_s: float
    cutoffs: tuple[tuple[int, int], ...]
    counts: tuple[int, ...]


def choose_prefix_subset(
    qa_rows: list[dict[str, Any]],
    sessions: Iterable[Session],
    minimum_per_category: int = 120,
) -> Candidate:
    """Find the minimum-duration valid participant prefixes with a multi-day user."""
    session_rows = sorted(sessions, key=lambda item: (item.participant, item.session))
    grouped: dict[int, list[Session]] = {}
    for session in session_rows:
        grouped.setdefault(session.participant, []).append(session)

    question_counts = Counter(
        (int(row["participant"]), int(row["primary_session"]), str(row["category_key"]))
        for row in qa_rows
    )
    category_index = {category: index for index, category in enumerate(TARGET_CATEGORIES)}
    options: dict[int, list[tuple[int, tuple[int, ...], float, bool]]] = {}
    for participant, participant_sessions in grouped.items():
        cumulative = [0] * len(TARGET_CATEGORIES)
        elapsed = 0.0
        dates: set[str] = set()
        prefixes = []
        expected_session = 1
        for session in participant_sessions:
            if (
                session.session != expected_session
                or session.gaze_coverage < 0.8
                or not session.video_available
                or not session.vrs_available
            ):
                break
            expected_session += 1
            elapsed += session.duration_s
            dates.add(session.date)
            for category, index in category_index.items():
                cumulative[index] += question_counts[
                    (participant, session.session, category)
                ]
            prefixes.append(
                (session.session, tuple(cumulative), elapsed, len(dates) > 1)
            )
        options[participant] = prefixes

    # Dynamic programming over the ten participants avoids enumerating every
    # Cartesian product of their possible recording cutoffs.
    states: dict[tuple[tuple[int, ...], bool], Candidate] = {
        ((0,) * len(TARGET_CATEGORIES), False): Candidate(
            duration_s=0.0, cutoffs=(), counts=(0,) * len(TARGET_CATEGORIES)
        )
    }
    for participant in sorted(grouped):
        next_states = dict(states)
        for (counts, is_multiday), candidate in states.items():
            for cutoff, prefix_counts, duration_s, prefix_multiday in options[participant]:
                combined = tuple(
                    min(minimum_per_category, counts[index] + prefix_counts[index])
                    for index in range(len(TARGET_CATEGORIES))
                )
                key = (combined, is_multiday or prefix_multiday)
                proposal = Candidate(
                    duration_s=candidate.duration_s + duration_s,
                    cutoffs=candidate.cutoffs + ((participant, cutoff),),
                    counts=tuple(
                        candidate.counts[index] + prefix_counts[index]
                        for index in range(len(TARGET_CATEGORIES))
                    ),
                )
                incumbent = next_states.get(key)
                if incumbent is None or proposal.duration_s < incumbent.duration_s:
                    next_states[key] = proposal
        states = next_states

    target = (minimum_per_category,) * len(TARGET_CATEGORIES)
    result = states.get((target, True))
    if result is None:
        best = max(
            (sum(counts), counts) for counts, _ in states
        )[1]
        raise ValueError(
            "No valid subset meets the category and multi-day constraints; "
            f"best observed category counts were {dict(zip(TARGET_CATEGORIES, best))}"
        )
    return result


def assign_splits(
    rows: list[dict[str, Any]], dev_fraction: float = 0.15, seed: int = 2026
) -> list[dict[str, Any]]:
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between 0 and 1")
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(str(row["category_key"]), []).append(row)

    split_by_id: dict[str, str] = {}
    for category, category_rows in sorted(by_category.items()):
        ids = [str(row["question_id"]) for row in category_rows]
        random.Random(f"{seed}:{category}").shuffle(ids)
        dev_count = round(len(ids) * dev_fraction)
        split_by_id.update(
            {question_id: "dev" for question_id in ids[:dev_count]}
        )
        split_by_id.update(
            {question_id: "test" for question_id in ids[dev_count:]}
        )
    return [
        {**row, "split": split_by_id[str(row["question_id"])]}
        for row in rows
    ]


def filter_subset(
    qa_rows: list[dict[str, Any]], cutoffs: dict[int, int]
) -> list[dict[str, Any]]:
    return [
        row
        for row in qa_rows
        if row["category_key"] in TARGET_CATEGORIES
        and row["participant"] in cutoffs
        and row["primary_session"] <= cutoffs[row["participant"]]
    ]


def selected_session_ids(cutoffs: dict[int, int]) -> list[tuple[int, int]]:
    return [
        (participant, session)
        for participant, cutoff in sorted(cutoffs.items())
        for session in range(1, cutoff + 1)
    ]
