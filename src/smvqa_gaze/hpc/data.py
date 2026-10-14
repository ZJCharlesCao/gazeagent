from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from ..qa import read_jsonl
from .config import Paths


@dataclass(frozen=True)
class SessionInfo:
    participant: int
    session: int
    date: str
    video_id: str
    duration_s: float

    @property
    def key(self) -> tuple[int, int]:
        return (self.participant, self.session)


@dataclass(frozen=True)
class Cutoff:
    """Memory visible to a question: earlier sessions plus the current prefix."""

    participant: int
    session: int
    time_s: float

    def allows(self, session: SessionInfo, end_s: float) -> bool:
        if session.participant != self.participant:
            return False
        if session.session < self.session:
            return True
        return session.session == self.session and end_s <= self.time_s + 1e-6


def load_sessions(path: Path) -> list[SessionInfo]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    sessions = [
        SessionInfo(
            participant=int(row["participant"]),
            session=int(row["session"]),
            date=str(row["date"]),
            video_id=str(row["video_id"]),
            duration_s=float(row["duration_s"]),
        )
        for row in rows
    ]
    return sorted(sessions, key=lambda item: item.key)


class Dataset:
    """Read-only view of the transfer package used by every HPC stage."""

    def __init__(self, paths: Paths) -> None:
        self.paths = paths

    @cached_property
    def sessions(self) -> list[SessionInfo]:
        return load_sessions(self.paths.sessions)

    @cached_property
    def session_by_video(self) -> dict[str, SessionInfo]:
        return {session.video_id: session for session in self.sessions}

    @cached_property
    def session_by_key(self) -> dict[tuple[int, int], SessionInfo]:
        return {session.key: session for session in self.sessions}

    def participant_sessions(self, participant: int) -> list[SessionInfo]:
        return [item for item in self.sessions if item.participant == participant]

    @cached_property
    def questions(self) -> list[dict[str, Any]]:
        rows = read_jsonl(self.paths.qa_subset)
        labels = (
            {str(row["question_id"]): row for row in read_jsonl(self.paths.fe_labels)}
            if self.paths.fe_labels.is_file()
            else {}
        )
        targets = (
            {str(row["question_id"]): row for row in read_jsonl(self.paths.visual_targets)}
            if self.paths.visual_targets.is_file()
            else {}
        )
        output = []
        for row in rows:
            question_id = str(row["question_id"])
            if row.get("question_time_s") is None:
                raise ValueError(f"Question {question_id} has no question time")
            label = labels.get(question_id, {})
            output.append(
                {
                    **row,
                    "question_id": question_id,
                    "fe_status": label.get("status", "unlabeled"),
                    "visual_targets": label.get(
                        "targets", targets.get(question_id, {}).get("targets", [])
                    ),
                }
            )
        return output

    def select_questions(
        self, split: str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        rows = [
            row for row in self.questions if split in (None, "all") or row["split"] == split
        ]
        rows.sort(key=lambda row: (row["participant"], row["primary_session"], row["question_time_s"], row["question_id"]))
        if limit is not None:
            # Spread a limited smoke/dev run over categories deterministically.
            by_category: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                by_category.setdefault(row["category_key"], []).append(row)
            picked: list[dict[str, Any]] = []
            while len(picked) < limit and any(by_category.values()):
                for category in sorted(by_category):
                    if by_category[category] and len(picked) < limit:
                        picked.append(by_category[category].pop(0))
            rows = picked
        return rows

    def cutoff(self, question: dict[str, Any]) -> Cutoff:
        return Cutoff(
            participant=int(question["participant"]),
            session=int(question["primary_session"]),
            time_s=float(question["question_time_s"]),
        )

    def transcript(self, session: SessionInfo) -> list[dict[str, Any]]:
        path = self.paths.transcript(session.participant, session.video_id)
        if not path.is_file():
            return []
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = rows.get("transcript", [])
        return [
            {
                "start": float(row["start"]),
                "end": float(row["end"]),
                "speaker": row.get("speaker") or row.get("person"),
                "text": " ".join(str(row.get("text", "")).split()),
            }
            for row in rows
            if str(row.get("text", "")).strip()
        ]

    def transcript_text(self, session: SessionInfo, start_s: float, end_s: float) -> str:
        lines = [
            f"[{format_clock(row['start'])}] {row['speaker'] or 'speaker'}: {row['text']}"
            for row in self.transcript_cached(session)
            if row["end"] > start_s and row["start"] < end_s
        ]
        return "\n".join(lines)

    def transcript_cached(self, session: SessionInfo) -> list[dict[str, Any]]:
        cache = self.__dict__.setdefault("_transcripts", {})
        if session.video_id not in cache:
            cache[session.video_id] = self.transcript(session)
        return cache[session.video_id]


def format_clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def unit_label(session: SessionInfo, start_s: float, end_s: float) -> str:
    return (
        f"day {session.date}, session {session.session}, "
        f"{format_clock(start_s)}-{format_clock(end_s)}"
    )


def evidence_overlaps(
    question: dict[str, Any], video_id: str, start_s: float, end_s: float
) -> bool:
    for span in question.get("answer_evidence", []):
        if span.get("video_id") != video_id:
            continue
        if span.get("t_start") is None or span.get("t_end") is None:
            continue
        if float(span["t_start"]) < end_s and float(span["t_end"]) > start_s:
            return True
    return False
