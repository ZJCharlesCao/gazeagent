from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Awaitable, Callable

from .answering import answer_record, question_block
from .config import (
    CONDITIONS,
    MM_PROCESSOR_KWARGS,
    TIER_TARGET_TOKENS,
    Condition,
    Paths,
    RunConfig,
    write_stage_manifest,
)
from .data import Dataset, SessionInfo, format_clock
from .frames import FrameStore
from .gaze_source import GazeSources
from .llm import JsonlStore, LLMClient, run_tasks
from .transforms import RenderedFrame, condition_prompt, render_frame


def pred_suffix(split: str, limit: int | None) -> str:
    """Final test runs use the bare condition name; dev/limited runs get a
    suffix so they can never be mistaken for (or mark) a finished test run."""
    suffix = "" if split == "test" and limit is None else f"_{split}"
    if limit is not None:
        suffix += f"_n{limit}"
    return suffix


@dataclass
class RunContext:
    """Everything one (pipeline, tier, condition) run needs."""

    dataset: Dataset
    config: RunConfig
    pipeline: str
    tier: str
    condition: Condition
    split: str = "test"
    limit: int | None = None
    smoke_window_s: float | None = None
    client: LLMClient | None = None
    frames: FrameStore = field(init=False)

    def __post_init__(self) -> None:
        self.frames = FrameStore(self.dataset.paths)

    @property
    def paths(self) -> Paths:
        return self.dataset.paths

    @property
    def memory_dir(self) -> Path:
        return self.paths.memory_dir(self.pipeline, self.tier, self.condition.name)

    @property
    def pred_dir(self) -> Path:
        return self.paths.pred_dir(
            self.pipeline, self.tier, self.condition.name + pred_suffix(self.split, self.limit)
        )

    @cached_property
    def sources(self) -> GazeSources:
        return GazeSources(self.dataset, self.config)

    @cached_property
    def label_index(self):
        from .gazemeta import build_label_index

        return build_label_index(self.dataset)

    @cached_property
    def retriever(self):
        from .retrieval import Contriever

        return Contriever(self.paths.feature_cache / "contriever")

    def llm(self) -> LLMClient:
        if self.client is None:
            self.client = LLMClient(self.config)
        return self.client

    def questions(self) -> list[dict[str, Any]]:
        cached = self.__dict__.get("_questions")
        if cached is not None:
            return cached
        rows = self.dataset.select_questions(self.split, self.limit)
        for row in rows:
            session = self.dataset.session_by_key[(row["participant"], row["primary_session"])]
            row["question_clock"] = (
                f"day {session.date}, session {session.session}, {format_clock(row['question_time_s'])}"
            )
        self.__dict__["_questions"] = rows
        return rows

    def memory_sessions(self) -> list[SessionInfo]:
        """Sessions whose memory is needed by the selected questions."""
        questions = self.questions()
        needed: set[tuple[int, int]] = set()
        for question in questions:
            for session in self.dataset.participant_sessions(question["participant"]):
                if session.session <= question["primary_session"]:
                    needed.add(session.key)
        return [session for session in self.dataset.sessions if session.key in needed]

    def in_smoke_window(self, session: SessionInfo, start_s: float, end_s: float) -> bool:
        """Smoke mode builds memory only shortly before each selected question."""
        if self.smoke_window_s is None:
            return True
        for question in self.questions():
            if (question["participant"], question["primary_session"]) != session.key:
                continue
            q_time = float(question["question_time_s"])
            if end_s <= q_time and end_s >= q_time - self.smoke_window_s:
                return True
            for span in question.get("answer_evidence", []):
                if span.get("video_id") == session.video_id and span.get("t_start") is not None:
                    if float(span["t_start"]) < end_s and float(span["t_end"]) > start_s:
                        return True
        return False

    def render(self, video_id: str, times: list[float], condition: Condition | None = None) -> list[RenderedFrame]:
        condition = condition or self.condition
        return [
            render_frame(condition, self.tier, self.frames.get(video_id, t), video_id, t, self.sources)
            for t in times
        ]

    def gaze_instruction(self) -> str:
        return condition_prompt(self.condition)


def token_audit(ctx: RunContext, rendered: list[RenderedFrame]) -> dict[str, Any]:
    """README 7.2: count visual tokens with the Qwen3-VL processor itself."""
    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(ctx.config.model)
    per_frame = []
    for frame in rendered:
        tokens = 0
        for image in frame.images:
            output = processor.image_processor(
                images=[Image.fromarray(image)], return_tensors="np", **MM_PROCESSOR_KWARGS
            )
            grid = output["image_grid_thw"][0]
            merge = getattr(processor.image_processor, "merge_size", 2)
            tokens += int(grid[0] * grid[1] * grid[2]) // (merge * merge)
        per_frame.append(tokens)
    measured = sum(per_frame) / max(len(per_frame), 1)
    target = TIER_TARGET_TOKENS[ctx.tier]
    return {
        "target_tokens_per_frame": target,
        "measured_tokens_per_frame": measured,
        "deviation": measured / target - 1,
        "within_5pct": abs(measured / target - 1) <= 0.05,
        "images_per_frame": [len(frame.images) for frame in rendered[:1]],
        "image_sizes": [list(image.shape[:2]) for image in rendered[0].images] if rendered else [],
    }


def write_run_manifest(ctx: RunContext, stage: str, outputs: dict[str, Any], audit: dict[str, Any] | None = None) -> None:
    directory = ctx.memory_dir if stage.startswith("memory") else ctx.pred_dir
    existing_path = directory / "manifest.json"
    previous = json.loads(existing_path.read_text()) if existing_path.is_file() else {}
    token_info = audit or previous.get("parameters", {}).get("token_audit")
    write_stage_manifest(
        directory,
        stage,
        inputs=[ctx.paths.qa_subset, ctx.paths.sessions],
        parameters={
            "pipeline": ctx.pipeline,
            "tier": ctx.tier,
            "condition": ctx.condition.__dict__,
            "split": ctx.split,
            "limit": ctx.limit,
            "smoke_window_s": ctx.smoke_window_s,
            "config": ctx.config.to_dict(),
            "token_audit": token_info,
        },
        outputs=outputs,
    )


async def answer_all(
    ctx: RunContext,
    build: Callable[[dict[str, Any]], Awaitable[tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]]],
) -> dict[str, Any]:
    """Shared answering loop: prompt -> ranking -> per-question append."""
    store = JsonlStore(ctx.pred_dir / "predictions.jsonl")
    pending = [q for q in ctx.questions() if q["question_id"] not in store]
    client = ctx.llm()

    async def work(question: dict[str, Any]) -> None:
        content, retrieved, extra = await build(question)
        block, order = question_block(question, ctx.config.seed)
        content = content + [{"type": "text", "text": block}]
        response = await client.chat(content, ctx.config.answer_max_tokens)
        record = answer_record(question, response, order, retrieved, extra)
        record["has_vague"] = any(choice["type"] == "vague" for choice in question["choices"])
        await store.append(record)

    stats = await run_tasks(
        pending, work, f"{ctx.pipeline}/{ctx.tier}/{ctx.condition.name} answer",
        max_in_flight=2 * ctx.config.request_concurrency,
    )
    return {**stats, "completed": len(store.records)}


def run_async(coroutine: Awaitable[Any]) -> Any:
    return asyncio.run(coroutine)


def condition_by_name(name: str) -> Condition:
    if name not in CONDITIONS:
        raise KeyError(f"Unknown condition {name!r}; choose from {sorted(CONDITIONS)}")
    return CONDITIONS[name]
