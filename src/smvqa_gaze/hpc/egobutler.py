"""EgoButler pipeline (information loss at write time), README section 7.1.

Write: every 30 s clip (or C6 gaze segment) is captioned from 1-fps frames plus
the transcript; captions are summarised per hour and per day.
Read: EgoRAG retrieves day -> hour -> clip and answers from the top-k captions.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

import numpy as np

from .common import RunContext, answer_all, token_audit, write_run_manifest
from .config import CLIP_SECONDS, CONDITIONS, TIER_TARGET_TOKENS
from .data import SessionInfo, evidence_overlaps, format_clock, unit_label
from .frames import clip_times
from .gazemeta import MAX_FRAMES_PER_UNIT, Unit, c6_units, clip_gaze_meta, frame_labels, load_session_gaze
from .llm import JsonlStore, run_tasks
from .transforms import frames_to_content

CAPTION_PROMPT = (
    "You are writing the long-term memory of the person wearing smart glasses "
    "(refer to them as 'the user'). The frames above were sampled at 1 fps from "
    "{span} of their first-person recording. Speech transcript of this interval:\n"
    "{transcript}\n\n"
    "Write a factual caption of at most 150 words: what the user does, objects "
    "and where they are put, readable text (screens, labels, documents), people "
    "present, and what is said. Do not speculate."
)
GAZE_CAPTION_SUFFIX = (
    " Describe the objects and text inside the gaze region first, but do not "
    "omit other important content."
)
HOUR_PROMPT = (
    "Summarise these time-stamped memory captions from one hour of a "
    "smart-glasses recording in at most 200 words. Keep objects and their "
    "locations, readable text, people, conversations, plans and intentions, "
    "with approximate times.{gaze}\n\n{captions}"
)
DAY_PROMPT = (
    "Summarise these hourly summaries of one day of smart-glasses recordings in "
    "at most 250 words. Keep the key events, objects and locations, people, "
    "conversations, plans and intentions, in time order.{gaze}\n\n{summaries}"
)
GAZE_SUMMARY_SUFFIX = " Preserve the objects and text the user repeatedly looked at."


def hour_of(unit_start: float) -> int:
    return int(unit_start // 3600)


class EgoButler:
    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.condition = ctx.condition
        self.caption_condition = CONDITIONS[self.condition.caption_from] if self.condition.caption_from else self.condition
        self.unit_store = JsonlStore(ctx.memory_dir / "units.jsonl")
        self.summary_store = JsonlStore(ctx.memory_dir / "summaries.jsonl")
        self._gaze: dict[str, Any] = {}
        if self.condition.needs_gaze:
            # Build lazily created shared state before worker threads start.
            _ = ctx.label_index

    # ------------------------------------------------------------------ units
    def gaze(self, session: SessionInfo):
        if session.video_id not in self._gaze:
            self._gaze[session.video_id] = load_session_gaze(self.ctx.dataset, session, self.ctx.label_index)
        return self._gaze[session.video_id]

    def units(self, session: SessionInfo) -> list[Unit]:
        config = self.ctx.config
        c0_units = []
        start = 0.0
        index = 0
        while start < session.duration_s - 1e-6:
            end = min(start + CLIP_SECONDS, session.duration_s)
            times = clip_times(start, end)
            if times:
                c0_units.append(Unit(session.video_id, index, start, end, tuple(times), config.caption_max_tokens))
                index += 1
            start = end
        if self.condition.alpha is None:
            return c0_units
        total_frames = sum(len(unit.frame_times) for unit in c0_units)
        # Frames per unit are capped by the context window and the per-request
        # image limit (B2: 44 frames x 1024 tokens fit in 49,152 tokens).
        max_frames = min(
            MAX_FRAMES_PER_UNIT,
            (config.max_model_len - config.text_margin_tokens) // TIER_TARGET_TOKENS[self.ctx.tier],
        )
        return c6_units(
            self.gaze(session),
            self.condition.alpha,
            config.caption_max_tokens,
            CLIP_SECONDS,
            total_frames,
            max_frames,
        )

    def all_units(self) -> list[tuple[SessionInfo, Unit]]:
        rows = []
        for session in self.ctx.memory_sessions():
            for unit in self.units(session):
                if self.ctx.in_smoke_window(session, unit.start, unit.end):
                    rows.append((session, unit))
        return rows

    # ------------------------------------------------------------------ write
    def caption_content(self, session: SessionInfo, unit: Unit) -> tuple[list[dict[str, Any]], list[Any]]:
        rendered = self.ctx.render(unit.video_id, list(unit.frame_times), self.caption_condition)
        labels = [format_clock(t) for t in unit.frame_times]
        transcript = self.ctx.dataset.transcript_text(session, unit.start, unit.end) or "(no speech)"
        instruction = ""
        if self.caption_condition.hint is not None:
            instruction = self.ctx.gaze_instruction() + GAZE_CAPTION_SUFFIX
        elif self.caption_condition.visual == "dual":
            instruction = self.ctx.gaze_instruction()
        prompt = CAPTION_PROMPT.format(span=f"{format_clock(unit.start)}-{format_clock(unit.end)}", transcript=transcript)
        content = frames_to_content(rendered, labels)
        content.append({"type": "text", "text": (instruction + "\n" if instruction else "") + prompt})
        return content, rendered

    async def build_captions(self) -> dict[str, Any]:
        units = self.all_units()
        audit = None
        if units and self.condition.caption_from is None:
            session, unit = units[0]
            rendered = await asyncio.to_thread(self.ctx.render, unit.video_id, list(unit.frame_times)[:2], self.caption_condition)
            audit = await asyncio.to_thread(token_audit, self.ctx, rendered)
        source_store = None
        if self.condition.caption_from is not None:
            source_dir = self.ctx.paths.memory_dir(self.ctx.pipeline, self.ctx.tier, self.condition.caption_from)
            source_store = JsonlStore(source_dir / "units.jsonl")
        client = self.ctx.llm() if source_store is None else None
        pending = [(s, u) for s, u in units if u.key not in self.unit_store]

        async def work(item: tuple[SessionInfo, Unit]) -> None:
            session, unit = item
            record: dict[str, Any] = {
                "key": unit.key,
                "video_id": unit.video_id,
                "participant": session.participant,
                "session": session.session,
                "date": session.date,
                "start": unit.start,
                "end": unit.end,
                "frame_times": list(unit.frame_times),
                "max_tokens": unit.max_tokens,
            }
            if source_store is not None:
                source = source_store.get(unit.key)
                if source is None:
                    raise RuntimeError(f"{self.condition.caption_from} caption missing for {unit.key}; run it first")
                record["caption"] = source["caption"]
                record["caption_source"] = self.condition.caption_from
            else:
                content, rendered = await asyncio.to_thread(self.caption_content, session, unit)
                response = await client.chat(content, unit.max_tokens)
                record.update(
                    {
                        "caption": response["text"],
                        "finish_reason": response["finish_reason"],
                        "prompt_tokens": response["prompt_tokens"],
                        "completion_tokens": response["completion_tokens"],
                        "latency_s": response["latency_s"],
                        "frame_meta": [frame.meta for frame in rendered] if self.caption_condition.needs_gaze else None,
                    }
                )
            if self.condition.needs_gaze:
                gaze = self.gaze(session)
                if self.condition.gaze_meta:
                    record["gaze_meta"] = clip_gaze_meta(gaze, unit.start, unit.end)
                if self.condition.hint is not None and record.get("frame_meta"):
                    # README 7.5: archive per-frame gaze box/point and H1.1 label.
                    for meta, label in zip(record["frame_meta"], frame_labels(gaze, list(unit.frame_times)), strict=True):
                        meta["fixation_label"] = label
            await self.unit_store.append(record)

        stats = await run_tasks(
            pending, work, f"egobutler/{self.ctx.tier}/{self.condition.name} caption",
            max_in_flight=2 * self.ctx.config.request_concurrency,
        )
        write_run_manifest(
            self.ctx,
            "memory:egobutler:captions",
            {"units": len(self.unit_store.records), **stats},
            audit,
        )
        return stats

    def unit_text(self, record: dict[str, Any]) -> str:
        text = record["caption"]
        if self.condition.gaze_meta and record.get("gaze_meta"):
            text += " Gaze: " + record["gaze_meta"]["summary"]
        return text

    async def build_summaries(self) -> dict[str, Any]:
        records = sorted(self.unit_store.values(), key=lambda r: (r["participant"], r["session"], r["start"]))
        hours: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            hours[(record["video_id"], hour_of(record["start"]))].append(record)
        gaze = GAZE_SUMMARY_SUFFIX if self.condition.gaze_meta else ""
        client = self.ctx.llm()
        config = self.ctx.config

        async def hour_work(item: tuple[tuple[str, int], list[dict[str, Any]]]) -> None:
            (video_id, hour), rows = item
            key = f"hour:{video_id}:{hour}"
            unit_keys = [row["key"] for row in rows]
            existing = self.summary_store.get(key)
            if existing is not None and existing.get("unit_keys") == unit_keys:
                return
            captions = "\n".join(
                f"[{format_clock(row['start'])}-{format_clock(row['end'])}] {self.unit_text(row)}" for row in rows
            )
            response = await client.chat(HOUR_PROMPT.format(gaze=gaze, captions=captions), config.summary_max_tokens)
            first = rows[0]
            await self.summary_store.append(
                {
                    "key": key,
                    "level": "hour",
                    "video_id": video_id,
                    "participant": first["participant"],
                    "session": first["session"],
                    "date": first["date"],
                    "start": min(row["start"] for row in rows),
                    "end": max(row["end"] for row in rows),
                    "unit_keys": unit_keys,
                    "text": response["text"],
                }
            )

        stats_hour = await run_tasks(
            list(hours.items()), hour_work, f"egobutler/{self.ctx.tier}/{self.condition.name} hour",
            max_in_flight=2 * self.ctx.config.request_concurrency,
        )
        days: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in self.summary_store.values():
            if row["level"] == "hour":
                days[(row["participant"], row["date"])].append(row)

        async def day_work(item: tuple[tuple[int, str], list[dict[str, Any]]]) -> None:
            (participant, date), rows = item
            key = f"day:{participant}:{date}"
            rows = sorted(rows, key=lambda row: (row["session"], row["start"]))
            hour_keys = [row["key"] for row in rows]
            existing = self.summary_store.get(key)
            if existing is not None and existing.get("hour_keys") == hour_keys:
                return
            summaries = "\n".join(
                f"[session {row['session']}, {format_clock(row['start'])}-{format_clock(row['end'])}] {row['text']}"
                for row in rows
            )
            response = await client.chat(DAY_PROMPT.format(gaze=gaze, summaries=summaries), config.summary_max_tokens)
            await self.summary_store.append(
                {
                    "key": key,
                    "level": "day",
                    "participant": participant,
                    "date": date,
                    "hour_keys": hour_keys,
                    "text": response["text"],
                }
            )

        stats_day = await run_tasks(
            list(days.items()), day_work, f"egobutler/{self.ctx.tier}/{self.condition.name} day",
            max_in_flight=2 * self.ctx.config.request_concurrency,
        )
        write_run_manifest(
            self.ctx,
            "memory:egobutler:summaries",
            {"units": len(self.unit_store.records), "summaries": len(self.summary_store.records), "hour": stats_hour, "day": stats_day},
        )
        return {"hour": stats_hour, "day": stats_day}

    # ------------------------------------------------------------------ read
    def _prepare_index(self) -> None:
        if hasattr(self, "_records"):
            return
        retriever = self.ctx.retriever
        self._records = sorted(self.unit_store.values(), key=lambda r: (r["participant"], r["session"], r["start"]))
        tag = f"egobutler_{self.ctx.tier}_{self.condition.name}"
        self._unit_vectors = retriever.encode([r["caption"] for r in self._records], f"{tag}_units")
        self._gaze_vectors = (
            retriever.encode([r.get("gaze_meta", {}).get("summary", "") for r in self._records], f"{tag}_gaze")
            if self.condition.gaze_meta
            else None
        )
        self._summaries = list(self.summary_store.values())
        self._summary_vectors = retriever.encode([r["text"] for r in self._summaries], f"{tag}_summaries")

    def retrieve(self, question: dict[str, Any], query_vector: np.ndarray) -> list[dict[str, Any]]:
        """EgoRAG day -> hour -> clip retrieval restricted to pre-question memory.

        Summaries covering time after the question are never used: such hours
        and days are scored by their eligible children instead (causality)."""
        self._prepare_index()
        config = self.ctx.config
        cutoff = self.ctx.dataset.cutoff(question)
        sessions = self.ctx.dataset.session_by_video
        unit_scores = self._unit_vectors @ query_vector
        if self._gaze_vectors is not None:
            unit_scores = unit_scores + config.beta * (self._gaze_vectors @ query_vector)
        eligible = [
            index
            for index, record in enumerate(self._records)
            if cutoff.allows(sessions[record["video_id"]], record["end"])
        ]
        if not eligible:
            return []
        hour_units: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index, record in enumerate(self._records):
            if record["participant"] == cutoff.participant:
                hour_units[(record["video_id"], hour_of(record["start"]))].append(index)
        eligible_set = set(eligible)
        summary_score = {row["key"]: float(score) for row, score in zip(self._summaries, self._summary_vectors @ query_vector, strict=True)}
        hour_score: dict[tuple[str, int], float] = {}
        day_hours: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for hour, members in hour_units.items():
            usable = [index for index in members if index in eligible_set]
            if not usable:
                continue
            complete = len(usable) == len(members)
            key = f"hour:{hour[0]}:{hour[1]}"
            hour_score[hour] = summary_score[key] if complete and key in summary_score else max(unit_scores[usable])
            day_hours[sessions[hour[0]].date].append(hour)
        day_score: dict[str, float] = {}
        for date, hours in day_hours.items():
            key = f"day:{cutoff.participant}:{date}"
            all_hours = {h for h in hour_units if sessions[h[0]].date == date}
            complete = all(
                all(index in eligible_set for index in hour_units[h]) for h in all_hours
            )
            day_score[date] = summary_score[key] if complete and key in summary_score else max(hour_score[h] for h in hours)
        top_days = sorted(day_score, key=lambda d: -day_score[d])[: config.egorag_days_k]
        candidate_hours = [h for d in top_days for h in day_hours[d]]
        top_hours = sorted(candidate_hours, key=lambda h: -hour_score[h])[: config.egorag_hours_k]
        candidates = [index for h in top_hours for index in hour_units[h] if index in eligible_set]
        top_units = sorted(candidates, key=lambda index: -unit_scores[index])[: config.egorag_clips_k]
        top_units.sort(key=lambda index: (self._records[index]["session"], self._records[index]["start"]))
        return [
            {**self._records[index], "score": float(unit_scores[index])} for index in top_units
        ]

    async def answer(self) -> dict[str, Any]:
        retriever = self.ctx.retriever
        dataset = self.ctx.dataset
        questions = self.ctx.questions()
        query_vectors = retriever.encode([q["question"] for q in questions], None)
        query_by_id = {q["question_id"]: vector for q, vector in zip(questions, query_vectors, strict=True)}

        async def build(question: dict[str, Any]):
            units = self.retrieve(question, query_by_id[question["question_id"]])
            lines = []
            for record in units:
                session = dataset.session_by_video[record["video_id"]]
                line = f"[{unit_label(session, record['start'], record['end'])}] {record['caption']}"
                if self.condition.gaze_meta and record.get("gaze_meta"):
                    line += f"\n  Gaze: {record['gaze_meta']['summary']}"
                lines.append(line)
            memory = "\n".join(lines) if lines else "(no memory before the question)"
            content = [
                {
                    "type": "text",
                    "text": "You are the personal memory assistant of a smart-glasses wearer "
                    "(the user). Answer using only these memory entries retrieved from "
                    f"the user's earlier recordings:\n{memory}\n",
                }
            ]
            has_timed = any(span.get("t_start") is not None for span in question.get("answer_evidence", []))
            hit = (
                any(evidence_overlaps(question, r["video_id"], r["start"], r["end"]) for r in units)
                if has_timed
                else None
            )
            retrieved = [
                {"video_id": r["video_id"], "start": r["start"], "end": r["end"], "score": r["score"]} for r in units
            ]
            return content, retrieved, {"retrieval_hit": hit, "k": self.ctx.config.egorag_clips_k}

        stats = await answer_all(self.ctx, build)
        write_run_manifest(self.ctx, "answer:egobutler", stats)
        return stats

    def memory_texts(self, question: dict[str, Any]) -> list[str]:
        """Memory text overlapping the evidence spans (target coverage, 7.6)."""
        texts = []
        for record in self.unit_store.values():
            if evidence_overlaps(question, record["video_id"], record["start"], record["end"]):
                texts.append(self.unit_text(record))
        for row in self.summary_store.values():
            if row["level"] == "hour" and evidence_overlaps(question, row["video_id"], row["start"], row["end"]):
                texts.append(row["text"])
        return texts


async def run_egobutler(ctx: RunContext, stages: set[str]) -> dict[str, Any]:
    pipeline = EgoButler(ctx)
    result: dict[str, Any] = {}
    if "memory" in stages:
        result["captions"] = await pipeline.build_captions()
        if result["captions"]["failed"]:
            raise RuntimeError(
                f"{result['captions']['failed']} captions failed; rerun the same command "
                "to retry them before summaries are built"
            )
        result["summaries"] = await pipeline.build_summaries()
    if "answer" in stages:
        result["answer"] = await pipeline.answer()
    return result, pipeline
