from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from .config import RunConfig


class JsonlStore:
    """Append-only, crash-safe result log keyed by ``key`` (README section 1.5)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[str, dict[str, Any]] = {}
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        # A torn final line from an interrupted job is ignored
                        # and its item is recomputed.
                        continue
                    self.records[str(row["key"])] = row
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def __contains__(self, key: str) -> bool:
        return key in self.records

    def get(self, key: str) -> dict[str, Any] | None:
        return self.records.get(key)

    def values(self) -> list[dict[str, Any]]:
        return list(self.records.values())

    def append_sync(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.records[str(row["key"])] = row

    async def append(self, row: dict[str, Any]) -> None:
        async with self._lock:
            self.append_sync(row)


def wait_for_server(base_url: str, timeout_s: float = 1800.0) -> None:
    """Block until the vLLM OpenAI server answers /v1/models."""
    deadline = time.time() + timeout_s
    url = base_url.rstrip("/") + "/models"
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001 - polling until ready
            last_error = exc
        time.sleep(5)
    raise TimeoutError(f"vLLM server at {base_url} not ready: {last_error}")


class LLMClient:
    """Async OpenAI-compatible client with bounded concurrency and retries."""

    def __init__(self, config: RunConfig) -> None:
        from openai import AsyncOpenAI

        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            timeout=config.request_timeout_s,
            max_retries=0,
        )
        self.semaphore = asyncio.Semaphore(config.request_concurrency)

    async def chat(
        self, content: list[dict[str, Any]] | str, max_tokens: int, retries: int = 4
    ) -> dict[str, Any]:
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        delay = 5.0
        async with self.semaphore:
            for attempt in range(retries + 1):
                started = time.perf_counter()
                try:
                    response = await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[{"role": "user", "content": content}],
                        max_tokens=max_tokens,
                        temperature=self.config.temperature,
                        seed=self.config.seed,
                    )
                    choice = response.choices[0]
                    usage = response.usage
                    return {
                        "text": (choice.message.content or "").strip(),
                        "finish_reason": choice.finish_reason,
                        "prompt_tokens": getattr(usage, "prompt_tokens", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "latency_s": time.perf_counter() - started,
                    }
                except Exception as exc:  # noqa: BLE001 - classify below
                    message = str(exc)
                    permanent = any(
                        token in message
                        for token in ("maximum context length", "too many images", "At most")
                    )
                    if permanent or attempt == retries:
                        raise
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 120)
        raise RuntimeError("unreachable")


async def run_tasks(
    items: Iterable[Any],
    worker: Callable[[Any], Awaitable[None]],
    progress_label: str,
    report_every: int = 25,
) -> dict[str, int]:
    """Run per-item coroutines; failures are logged and counted, not fatal."""
    items = list(items)
    done = failed = 0
    started = time.perf_counter()

    async def guarded(item: Any) -> None:
        nonlocal done, failed
        try:
            await worker(item)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[{progress_label}] FAILED {_item_label(item)}: {type(exc).__name__}: {exc}", flush=True)
        else:
            done += 1
        finished = done + failed
        if finished % report_every == 0 or finished == len(items):
            rate = finished / max(time.perf_counter() - started, 1e-6)
            print(
                f"[{progress_label}] {finished}/{len(items)} done ({failed} failed, {rate:.2f}/s)",
                flush=True,
            )

    await asyncio.gather(*(guarded(item) for item in items))
    return {"done": done, "failed": failed, "total": len(items)}


def _item_label(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("question_id", item.get("key", "?")))
    if isinstance(item, tuple):
        return " ".join(_item_label(part) if isinstance(part, dict) else str(getattr(part, "key", part)) for part in item)
    return str(getattr(item, "key", item))
