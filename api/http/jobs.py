from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable

_training_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="training")
_predict_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="predict")
_active_jobs: dict[str, "_Job"] = {}
_predict_jobs: dict[str, "_PredictJob"] = {}
_jobs_lock = threading.Lock()


@dataclass
class _Job:
    run_id: str
    mode: str
    started_at: datetime
    pct: int = 0
    done: bool = False
    error: str | None = None
    event_log: list[tuple[str, str]] = field(default_factory=list)


def _execute_training_job(run_id: str, fn: Callable, *args) -> None:
    try:
        fn(*args)
    except Exception as exc:
        emit(run_id, "error", f"{type(exc).__name__}: {exc}")


def is_training() -> bool:
    with _jobs_lock:
        return any(not j.done for j in _active_jobs.values())


def launch_training_job(run_id: str, mode: str, fn: Callable, *args) -> None:
    job = _Job(run_id=run_id, mode=mode, started_at=datetime.now(timezone.utc))
    with _jobs_lock:
        _active_jobs[run_id] = job
    _training_executor.submit(_execute_training_job, run_id, fn, *args)


def emit(run_id: str, event: str, data: str) -> None:
    with _jobs_lock:
        job = _active_jobs.get(run_id)
        if job is None:
            return
        job.event_log.append((event, data))
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
    if run_id not in _active_jobs:
        return
    cursor = 0
    last_heartbeat_at = time.monotonic()
    while True:
        with _jobs_lock:
            new_events = _active_jobs[run_id].event_log[cursor:]
        cursor += len(new_events)
        for event_type, event_data in new_events:
            yield f"event: {event_type}\ndata: {event_data}\n\n"
            if event_type in ("done", "error"):
                return
        await asyncio.sleep(0.1)
        now = time.monotonic()
        if now - last_heartbeat_at >= 30:
            yield "event: heartbeat\ndata: ping\n\n"
            last_heartbeat_at = now


def get_status(run_id: str) -> dict | None:
    with _jobs_lock:
        job = _active_jobs.get(run_id)
    if job is None:
        return None
    return {
        "run_id": job.run_id,
        "pct": job.pct,
        "mode": job.mode,
        "started_at": job.started_at.isoformat(),
    }


# ── Async predict jobs ─────────────────────────────────────────────────────────

@dataclass
class _PredictJob:
    job_id: str
    pct: int = 0
    done: bool = False
    error: str | None = None
    result: Any = None


def _execute_predict_job(job_id: str, fn: Callable, *args) -> None:
    try:
        result = fn(*args)
        with _jobs_lock:
            if job_id in _predict_jobs:
                _predict_jobs[job_id].pct = 100
                _predict_jobs[job_id].done = True
                _predict_jobs[job_id].result = result
    except Exception as exc:
        with _jobs_lock:
            if job_id in _predict_jobs:
                _predict_jobs[job_id].done = True
                _predict_jobs[job_id].error = f"{type(exc).__name__}: {exc}"


def launch_predict_job(job_id: str, fn: Callable, *args) -> None:
    job = _PredictJob(job_id=job_id)
    with _jobs_lock:
        _predict_jobs[job_id] = job
    _predict_executor.submit(_execute_predict_job, job_id, fn, *args)


def get_predict_status(job_id: str) -> dict | None:
    with _jobs_lock:
        job = _predict_jobs.get(job_id)
    if job is None:
        return None
    return {"job_id": job.job_id, "pct": job.pct, "done": job.done}


def get_predict_result(job_id: str) -> tuple[bool, Any]:
    """Returns (done, result). result is None if still running or if errored."""
    with _jobs_lock:
        job = _predict_jobs.get(job_id)
    if job is None:
        return False, None
    return job.done, job.result
