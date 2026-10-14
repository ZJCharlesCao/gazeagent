"""H2 step 1: measure throughput with the section 5.2 server configuration."""

from __future__ import annotations

import asyncio
import json
import os
import random
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .answering import question_block
from .common import RunContext
from .config import CONDITIONS, VIDEORAG_ANSWER_FRAMES, Paths, RunConfig
from .data import Dataset, format_clock
from .egobutler import CAPTION_PROMPT
from .frames import clip_times, uniform_times
from .llm import LLMClient, wait_for_server
from .runner import BudgetLedger
from .transforms import frames_to_content

DEFAULT_GRID = [
    {"max_num_seqs": 16, "max_num_batched_tokens": 32768},
    {"max_num_seqs": 32, "max_num_batched_tokens": 32768},
    {"max_num_seqs": 32, "max_num_batched_tokens": 65536},
]


class GpuMonitor:
    """Poll nvidia-smi for peak memory while a workload runs."""

    def __init__(self, interval_s: float = 2.0) -> None:
        self.interval = interval_s
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10, check=True,
                ).stdout
                device = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or 0)
                values = [int(v) for v in output.split()]
                self.peak_mib = max(self.peak_mib, values[min(device, len(values) - 1)])
            except Exception:  # noqa: BLE001 - monitoring is best effort
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "GpuMonitor":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def launch_server(server: dict[str, Any], log_path: Path, port: int) -> subprocess.Popen:
    env = dict(os.environ)
    env.update(
        {
            "VLLM_PORT": str(port),
            "VLLM_MAX_NUM_SEQS": str(server["max_num_seqs"]),
            "VLLM_MAX_NUM_BATCHED_TOKENS": str(server["max_num_batched_tokens"]),
        }
    )
    for key in ("gpu_memory_utilization", "max_model_len"):
        if key in server:
            env[f"VLLM_{key.upper()}"] = str(server[key])
    script = Path(__file__).resolve().parents[3] / "scripts/start_vllm.sh"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(["bash", str(script)], env=env, stdout=log_path.open("w"), stderr=subprocess.STDOUT)


def sample_clips(dataset: Dataset, count: int, seed: int = 0) -> list[tuple[Any, float, float]]:
    clips = []
    for session in dataset.sessions:
        start = 0.0
        while start + 30 <= session.duration_s:
            clips.append((session, start, start + 30))
            start += 30
    random.Random(seed).shuffle(clips)
    return sorted(clips[:count], key=lambda item: (item[0].key, item[1]))


async def _measure(client: LLMClient, payloads: list[tuple[list[dict[str, Any]], int]]) -> dict[str, Any]:
    started = time.perf_counter()
    results = await asyncio.gather(
        *(client.chat(content, max_tokens, retries=1) for content, max_tokens in payloads), return_exceptions=True
    )
    wall = time.perf_counter() - started
    ok = [row for row in results if isinstance(row, dict)]
    errors = [str(row) for row in results if not isinstance(row, dict)]
    prompt_tokens = sum(row.get("prompt_tokens") or 0 for row in ok)
    completion_tokens = sum(row.get("completion_tokens") or 0 for row in ok)
    return {
        "requests": len(payloads),
        "ok": len(ok),
        "errors": len(errors),
        "oom": sum("out of memory" in error.lower() for error in errors),
        "error_examples": errors[:3],
        "wall_s": wall,
        "seconds_per_item": wall / max(len(ok), 1),
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_per_s": prompt_tokens / wall,
        "completion_tokens_per_s": completion_tokens / wall,
        "mean_latency_s": sum(row["latency_s"] for row in ok) / max(len(ok), 1),
    }


def caption_payloads(ctx: RunContext, clips: list[tuple[Any, float, float]]) -> list[tuple[list[dict[str, Any]], int]]:
    payloads = []
    for session, start, end in clips:
        times = clip_times(start, end)
        rendered = ctx.render(session.video_id, times)
        content = frames_to_content(rendered, [format_clock(t) for t in times])
        transcript = ctx.dataset.transcript_text(session, start, end) or "(no speech)"
        prompt = CAPTION_PROMPT.format(span=f"{format_clock(start)}-{format_clock(end)}", transcript=transcript)
        instruction = ctx.gaze_instruction()
        content.append({"type": "text", "text": (instruction + "\n" if instruction else "") + prompt})
        payloads.append((content, ctx.config.caption_max_tokens))
    return payloads


def answer_payloads(ctx: RunContext, questions: list[dict[str, Any]]) -> list[tuple[list[dict[str, Any]], int]]:
    """Video-RAG answering cost: 32 frames from the question's session prefix
    plus a retrieval-sized text block (retrieval itself is CPU-side)."""
    payloads = []
    for question in questions:
        session = ctx.dataset.session_by_key[(question["participant"], question["primary_session"])]
        end = max(30.0, min(float(question["question_time_s"]), session.duration_s))
        start = max(0.0, end - 1800)
        times = uniform_times(start, end, VIDEORAG_ANSWER_FRAMES)
        rendered = ctx.render(session.video_id, times)
        content = frames_to_content(rendered, [format_clock(t) for t in times])
        filler = ctx.dataset.transcript_text(session, start, end)[:6000] or "(nothing retrieved)"
        block, _ = question_block(question, ctx.config.seed)
        content.append({"type": "text", "text": f"Retrieved text:\n{filler}\n{block}"})
        payloads.append((content, ctx.config.answer_max_tokens))
    return payloads


def text_payloads(ctx: RunContext, count: int) -> list[tuple[list[dict[str, Any]], int]]:
    session = ctx.dataset.sessions[0]
    text = ctx.dataset.transcript_text(session, 0, session.duration_s)[:12000]
    return [([{"type": "text", "text": f"Summarise in 200 words:\n{text}"}], ctx.config.summary_max_tokens)] * count


def feature_timing(ctx: RunContext) -> dict[str, Any]:
    """Per-frame cost of the non-LLM GPU models (run with vLLM stopped)."""
    from .features import ClipIndex, OcrIndex

    session = ctx.dataset.sessions[0]
    ocr = OcrIndex(ctx.dataset, ctx.config, ctx.frames, ctx.sources)
    started = time.perf_counter()
    for t in range(0, 150, 5):
        ocr.reader().readtext(ctx.frames.get(session.video_id, float(t)))
    ocr_s = (time.perf_counter() - started) / 30
    clip = ClipIndex(ctx.dataset, ctx.frames, ctx.sources)
    import torch
    from PIL import Image

    model, processor, device = clip.model()
    images = [Image.fromarray(ctx.frames.get(session.video_id, float(t))) for t in range(64)]
    started = time.perf_counter()
    with torch.inference_mode():
        model.get_image_features(pixel_values=processor(images=images, return_tensors="pt")["pixel_values"].to(device, model.dtype))
    clip_s = (time.perf_counter() - started) / 64
    return {"ocr_s_per_frame": ocr_s, "clip_s_per_frame": clip_s}


def run_benchmark(
    paths: Paths,
    config: RunConfig,
    grid: list[dict[str, Any]] | None,
    launch: bool,
    n_clips: int = 100,
    n_questions: int = 100,
    port: int = 8000,
) -> dict[str, Any]:
    dataset = Dataset(paths)
    ledger = BudgetLedger(paths.logs / "budget.json", "h2-bench")
    out_dir = paths.results / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = sample_clips(dataset, n_clips)
    dev = dataset.select_questions("dev")
    questions = (dev + dataset.select_questions("test"))[:n_questions] if len(dev) < n_questions else dev[:n_questions]
    grid = grid or ([{"max_num_seqs": 16, "max_num_batched_tokens": 32768}] if not launch else DEFAULT_GRID)
    runs = []
    for server in grid:
        process = launch_server(server, paths.logs / f"bench_vllm_{server['max_num_seqs']}_{server['max_num_batched_tokens']}.log", port) if launch else None
        try:
            wait_for_server(config.base_url, timeout_s=1800)
            ledger.heartbeat()
            run_config = RunConfig(**{**config.to_dict(), "request_concurrency": 4 * config.request_concurrency})
            client = LLMClient(run_config)
            row: dict[str, Any] = {"server": server, "workloads": {}}
            for tier in ("B1", "B2"):
                for condition in ("C0", "C4"):
                    ctx = RunContext(dataset, run_config, "egobutler", tier, CONDITIONS[condition], split="dev")
                    payloads = caption_payloads(ctx, clips)
                    with GpuMonitor() as monitor:
                        measured = asyncio.run(_measure(client, payloads))
                    measured["peak_gpu_mib"] = monitor.peak_mib
                    row["workloads"][f"caption_{condition}_{tier}"] = measured
                    print(f"[bench] {server} caption {condition} {tier}: {measured['seconds_per_item']:.2f}s/clip, errors {measured['errors']}", flush=True)
                    ledger.heartbeat()
                ctx = RunContext(dataset, run_config, "videorag", tier, CONDITIONS["C0"], split="dev")
                payloads = answer_payloads(ctx, questions)
                with GpuMonitor() as monitor:
                    measured = asyncio.run(_measure(client, payloads))
                measured["peak_gpu_mib"] = monitor.peak_mib
                row["workloads"][f"videorag_answer_C0_{tier}"] = measured
                print(f"[bench] {server} answer {tier}: {measured['seconds_per_item']:.2f}s/q, errors {measured['errors']}", flush=True)
                ledger.heartbeat()
            ctx = RunContext(dataset, run_config, "egobutler", "B1", CONDITIONS["C0"], split="dev")
            row["workloads"]["text_summary"] = asyncio.run(_measure(client, text_payloads(ctx, 20)))
            row["errors_total"] = sum(w["errors"] for w in row["workloads"].values())
            row["score"] = sum(1 / w["seconds_per_item"] for w in row["workloads"].values() if w["ok"])
            runs.append(row)
            (out_dir / f"bench_{server['max_num_seqs']}_{server['max_num_batched_tokens']}.json").write_text(json.dumps(row, indent=2))
        finally:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    process.kill()
                time.sleep(10)
    stable = [row for row in runs if row["errors_total"] == 0] or runs
    best = max(stable, key=lambda row: row["score"])
    features = None
    if launch:
        ctx = RunContext(dataset, config, "videorag", "B1", CONDITIONS["C0"], split="dev")
        features = feature_timing(ctx)
    summary = {
        "runs": runs,
        "best_server": best["server"],
        "best": best["workloads"],
        "features": features,
        "n_clips": len(clips),
        "n_questions": len(questions),
        "h2_gpu_hours": ledger.heartbeat(),
        "command": " ".join(shlex.quote(part) for part in ["smvqa-gaze", "hpc-bench"]),
    }
    (out_dir / "bench_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
