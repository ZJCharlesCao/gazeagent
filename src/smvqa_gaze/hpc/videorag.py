"""Video-RAG pipeline (information loss at read time), README section 7.1.

Sessions are split into 30-minute shards with three retrieval databases: ASR
(released WhisperX transcripts), OCR (EasyOCR on sampled frames) and object
detection (GroundingDINO on CLIP-selected keyframes, query dependent). C5 adds
a fourth GazeDB. Answering retrieves the most relevant shard, samples 32 frames
from it and merges the retrieved text.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .common import RunContext, answer_all, token_audit, write_run_manifest
from .config import SHARD_SECONDS, TIER_SIDE, VIDEORAG_ANSWER_FRAMES
from .data import SessionInfo, evidence_overlaps, format_clock, unit_label
from .features import (
    QUERY_OBJECT_PROMPT,
    ClipIndex,
    DetectionCache,
    OcrIndex,
    describe_detections,
    parse_object_list,
    visual_variant,
)
from .frames import uniform_times
from .gazemeta import clip_gaze_meta, gaze_db_entries, load_session_gaze, weighted_times
from .llm import JsonlStore
from .retrieval import TextIndex
from .transforms import frames_to_content


@dataclass(frozen=True)
class Shard:
    session: SessionInfo
    index: int
    start: float
    end: float


def session_shards(session: SessionInfo) -> list[Shard]:
    count = max(1, math.ceil(session.duration_s / SHARD_SECONDS - 1e-9))
    return [
        Shard(session, index, index * SHARD_SECONDS, min((index + 1) * SHARD_SECONDS, session.duration_s))
        for index in range(count)
    ]


class VideoRAG:
    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.condition = ctx.condition
        self.variant = visual_variant(ctx.condition, ctx.tier)
        self.ocr = OcrIndex(ctx.dataset, ctx.config, ctx.frames, ctx.sources)
        self.clip = ClipIndex(ctx.dataset, ctx.frames, ctx.sources)
        self.detector = DetectionCache(ctx.dataset, ctx.config, ctx.frames, ctx.sources)
        self.objects = JsonlStore(ctx.paths.work / "cache/videorag_query_objects.jsonl")
        self._gaze: dict[str, Any] = {}
        if self.condition.needs_gaze:
            # Build lazily created shared state before worker threads start.
            _ = ctx.label_index

    def gaze(self, session: SessionInfo):
        if session.video_id not in self._gaze:
            self._gaze[session.video_id] = load_session_gaze(self.ctx.dataset, session, self.ctx.label_index)
        return self._gaze[session.video_id]

    # ------------------------------------------------------------------ memory
    def build_databases(self) -> dict[str, Any]:
        """Query-independent DBs (ASR, OCR, GazeDB, CLIP) for the needed sessions."""
        dataset = self.ctx.dataset
        entries: list[dict[str, Any]] = []
        sessions = self.ctx.memory_sessions()
        for session in sessions:
            keep = None
            if self.ctx.smoke_window_s is not None:
                keep = lambda t, session=session: self.ctx.in_smoke_window(session, t, t + 1.0)  # noqa: E731
            for row in dataset.transcript_cached(session):
                entries.append(
                    {
                        "db": "ASR",
                        "video_id": session.video_id,
                        "start": row["start"],
                        "end": row["end"],
                        "text": f"{row['speaker'] or 'speaker'}: {row['text']}",
                    }
                )
            for row in self.ocr.entries(self.variant, session, keep):
                entries.append({"db": "OCR", **row})
            if self.condition.gaze_meta:
                for row in gaze_db_entries(self.gaze(session), self.ctx.label_index):
                    entries.append({"db": "GAZE", **row})
            self.clip.image_vectors(self.variant, session, keep)
        self._keep = self.ctx.smoke_window_s is not None
        self.entries = entries
        tag = f"videorag_{self.variant}_{'gaze' if self.condition.gaze_meta else 'base'}"
        vectors = self.ctx.retriever.encode([row["text"] for row in entries], tag)
        self.index = TextIndex(entries, vectors)
        self.entry_session = [dataset.session_by_video[row["video_id"]] for row in entries]
        self.entry_db = np.array([row["db"] for row in entries])
        self.entry_video = np.array([row["video_id"] for row in entries])
        self.entry_start = np.array([row["start"] for row in entries], dtype=float)
        if self.condition.gaze_meta:
            # gaze_meta summaries of 30 s clips, used by the C5 shard score.
            self.clip_meta: list[dict[str, Any]] = []
            for session in sessions:
                gaze = self.gaze(session)
                start = 0.0
                while start < session.duration_s:
                    end = min(start + 30.0, session.duration_s)
                    meta = clip_gaze_meta(gaze, start, end)
                    self.clip_meta.append({"video_id": session.video_id, "start": start, "end": end, "summary": meta["summary"]})
                    start = end
            self.clip_meta_vectors = self.ctx.retriever.encode([row["summary"] for row in self.clip_meta], f"{tag}_clipmeta")
        counts = {db: int((self.entry_db == db).sum()) for db in ("ASR", "OCR", "GAZE")}
        return {"entries": len(entries), **counts, "sessions": len(sessions)}

    # ------------------------------------------------------------------ read
    def in_window(self, video_id: str, start: float, end: float) -> np.ndarray:
        return (self.entry_video == video_id) & (self.entry_start >= start) & (self.entry_start < end)

    def eligible_mask(self, question: dict[str, Any]) -> np.ndarray:
        cutoff = self.ctx.dataset.cutoff(question)
        return np.array(
            [cutoff.allows(session, row["end"]) for session, row in zip(self.entry_session, self.entries, strict=True)],
            dtype=bool,
        )

    def select_shard(self, question: dict[str, Any], query: np.ndarray, eligible: np.ndarray) -> tuple[Shard, float]:
        cutoff = self.ctx.dataset.cutoff(question)
        shards = [
            shard
            for session in self.ctx.dataset.participant_sessions(cutoff.participant)
            if session.session <= cutoff.session
            for shard in session_shards(session)
            if session.session < cutoff.session or shard.start < cutoff.time_s
        ]
        scores = self.index.scores(query)
        content_dbs = np.isin(self.entry_db, ["ASR", "OCR"])
        best: tuple[float, Shard] | None = None
        for shard in shards:
            members = self.in_window(shard.session.video_id, shard.start, shard.end)
            usable = members & eligible & content_dbs
            score = float(scores[usable].max()) if usable.any() else -math.inf
            if self.condition.gaze_meta:
                end = min(shard.end, cutoff.time_s) if shard.session.session == cutoff.session else shard.end
                meta_mask = np.array(
                    [
                        row["video_id"] == shard.session.video_id and shard.start <= row["start"] and row["end"] <= end
                        for row in self.clip_meta
                    ],
                    dtype=bool,
                )
                if meta_mask.any():
                    score = score + self.ctx.config.beta * float((self.clip_meta_vectors[meta_mask] @ query).max())
            if best is None or score > best[0]:
                best = (score, shard)
        assert best is not None, "a question always has at least its own session prefix"
        if not math.isfinite(best[0]):
            # Nothing retrievable yet: fall back to the most recent shard.
            return shards[-1], best[0]
        return best[1], best[0]

    async def query_objects(self, question: dict[str, Any]) -> list[str]:
        key = question["question_id"]
        cached = self.objects.get(key)
        if cached is None:
            response = await self.ctx.llm().chat(QUERY_OBJECT_PROMPT.format(question=question["question"]), 64)
            cached = {"key": key, "objects": parse_object_list(response["text"]), "raw": response["text"]}
            await self.objects.append(cached)
        return cached["objects"]

    def keyframes(self, question: dict[str, Any], window: tuple[float, float], shard: Shard) -> list[float]:
        count = self.ctx.config.videorag_keyframes
        start, end = window
        if self.condition.alpha is not None:
            return [float(round(t)) for t in weighted_times(self.gaze(shard.session), start, end, count, self.condition.alpha)]
        keep = (lambda t: self.ctx.in_smoke_window(shard.session, t, t + 1.0)) if self._keep else None
        vectors = self.clip.image_vectors(self.variant, shard.session, keep)
        seconds = np.arange(int(math.ceil(start)), int(math.floor(end - 1e-6)) + 1)
        seconds = seconds[seconds < len(vectors)]
        if not len(seconds):
            return []
        similarity = vectors[seconds].astype(np.float32) @ self.clip.text_vector(question["question"])
        chosen = seconds[np.argsort(-similarity)[:count]]
        return sorted(float(t) for t in chosen)

    async def answer(self) -> dict[str, Any]:
        questions = self.ctx.questions()
        queries = self.ctx.retriever.encode([q["question"] for q in questions], None)
        query_by_id = {q["question_id"]: v for q, v in zip(questions, queries, strict=True)}
        config = self.ctx.config
        audit_done = False

        async def build(question: dict[str, Any]):
            nonlocal audit_done
            query = query_by_id[question["question_id"]]
            eligible = self.eligible_mask(question)
            shard, shard_score = self.select_shard(question, query, eligible)
            cutoff = self.ctx.dataset.cutoff(question)
            window_end = min(shard.end, cutoff.time_s) if shard.session.session == cutoff.session else shard.end
            window = (shard.start, max(window_end, shard.start + 1.0))
            in_shard = self.in_window(shard.session.video_id, window[0], window[1])
            texts: list[tuple[float, str]] = []
            retrieved_entries = []
            for db, k in (("ASR", config.videorag_asr_k), ("OCR", config.videorag_ocr_k), ("GAZE", config.videorag_gaze_k)):
                hits = self.index.search(query, k, eligible & in_shard & (self.entry_db == db))
                for entry, score in hits:
                    texts.append((entry["start"], f"[{format_clock(entry['start'])}] {db}: {entry['text']}"))
                    retrieved_entries.append({"db": db, "video_id": entry["video_id"], "start": entry["start"], "end": entry["end"], "score": score})
            objects = await self.query_objects(question)
            keyframes = await asyncio.to_thread(self.keyframes, question, window, shard)
            side = TIER_SIDE[self.ctx.tier] if self.condition.visual == "fovea" else 1408
            for t in keyframes:
                detections = await asyncio.to_thread(self.detector.detect, self.variant, shard.session.video_id, t, objects)
                texts.append((t, f"[{format_clock(t)}] DET: {describe_detections(detections, side)}"))
            texts.sort(key=lambda item: item[0])

            if self.condition.alpha is not None:
                times = weighted_times(self.gaze(shard.session), window[0], window[1], VIDEORAG_ANSWER_FRAMES, self.condition.alpha)
            else:
                times = uniform_times(window[0], window[1], VIDEORAG_ANSWER_FRAMES)
            rendered = await asyncio.to_thread(self.ctx.render, shard.session.video_id, times)
            if not audit_done:
                audit_done = True
                audit = await asyncio.to_thread(token_audit, self.ctx, rendered[:2])
                write_run_manifest(self.ctx, "answer:videorag", {"token_audit_only": True}, audit)
            content = [
                {
                    "type": "text",
                    "text": "You are the personal memory assistant of a smart-glasses wearer "
                    "(the user). Below are frames from the most relevant earlier "
                    f"recording segment ({unit_label(shard.session, window[0], window[1])}).",
                }
            ]
            content += frames_to_content(rendered, [format_clock(t) for t in times])
            instruction = self.ctx.gaze_instruction()
            retrieved_text = "\n".join(text for _, text in texts) or "(nothing retrieved)"
            content.append(
                {
                    "type": "text",
                    "text": (instruction + "\n" if instruction else "")
                    + "Retrieved transcript (ASR), on-screen text (OCR), detected objects (DET)"
                    + (" and gaze records (GAZE)" if self.condition.gaze_meta else "")
                    + f" from this segment:\n{retrieved_text}\n",
                }
            )
            has_timed = any(span.get("t_start") is not None for span in question.get("answer_evidence", []))
            hit = evidence_overlaps(question, shard.session.video_id, window[0], window[1]) if has_timed else None
            retrieved = [
                {"video_id": shard.session.video_id, "start": window[0], "end": window[1], "score": shard_score, "db": "SHARD"},
                *retrieved_entries,
            ]
            extra = {
                "retrieval_hit": hit,
                "k": 1,
                "frame_times": times,
                "keyframes": keyframes,
                "objects": objects,
                "frame_meta": [frame.meta for frame in rendered] if self.condition.needs_gaze else None,
            }
            return content, retrieved, extra

        stats = await answer_all(self.ctx, build)
        write_run_manifest(self.ctx, "answer:videorag", stats)
        return stats

    def memory_texts(self, question: dict[str, Any]) -> list[str]:
        return [
            row["text"]
            for row in self.entries
            if evidence_overlaps(question, row["video_id"], row["start"], row["end"])
        ]


async def run_videorag(ctx: RunContext, stages: set[str]) -> dict[str, Any]:
    pipeline = VideoRAG(ctx)
    result: dict[str, Any] = {}
    result["databases"] = await asyncio.to_thread(pipeline.build_databases)
    write_run_manifest(ctx, "memory:videorag", result["databases"])
    if "answer" in stages:
        result["answer"] = await pipeline.answer()
    return result, pipeline

