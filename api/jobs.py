from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncGenerator, Callable

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="training")
_jobs: dict[str, "_Job"] = {}
_lock = threading.Lock()


@dataclass
class _Job:
    run_id: str
    mode: str
    started_at: datetime
    pct: int = 0
    done: bool = False
    error: str | None = None
    _events: list[tuple[str, str]] = field(default_factory=list)


def _run(run_id: str, fn: Callable, *args) -> None:
    try:
        fn(*args)
    except Exception as exc:
        emit(run_id, "error", f"{type(exc).__name__}: {exc}")


def is_training() -> bool:
    with _lock:
        return any(not j.done for j in _jobs.values())


def launch(run_id: str, mode: str, fn: Callable, *args) -> None:
    job = _Job(run_id=run_id, mode=mode, started_at=datetime.now(timezone.utc))
    with _lock:
        _jobs[run_id] = job
    _executor.submit(_run, run_id, fn, *args)


def emit(run_id: str, event: str, data: str) -> None:
    with _lock:
        job = _jobs.get(run_id)
        if job is None:
            return
        job._events.append((event, data))
        if event == "step":
            try:
                job.pct = int(data)
            except ValueError:
                pass
        elif event in ("done", "error"):
            job.done = True
            if event == "error":
                job.error = data


async def subscribe(run_id: str) -> AsyncGenerator[str, None]:
    """SSE generator — replays buffered history then streams new events."""
    if run_id not in _jobs:
        return
    cursor = 0
    last_hb = time.monotonic()
    while True:
        with _lock:
            chunk = _jobs[run_id]._events[cursor:]
        cursor += len(chunk)
        for ev, data in chunk:
            yield f"event: {ev}\ndata: {data}\n\n"
            if ev in ("done", "error"):
                return
        await asyncio.sleep(0.1)
        now = time.monotonic()
        if now - last_hb >= 30:
            yield "event: heartbeat\ndata: ping\n\n"
            last_hb = now


def get_status(run_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(run_id)
    if job is None:
        return None
    return {
        "run_id": job.run_id,
        "pct": job.pct,
        "mode": job.mode,
        "started_at": job.started_at.isoformat(),
    }
