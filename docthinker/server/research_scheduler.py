"""Optional local context work: one actual worker, no queue or model retries.

A timeout/cancelled request cannot kill a Python thread. Keep its slot occupied
until it really exits instead of allowing later requests to accumulate runaway
work. Callers fall back to normal evidence retrieval while this helper is busy.
Only deterministic, bounded local algorithms belong here, never provider calls.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Literal


@dataclass(frozen=True)
class ContextWorkResult:
    status: Literal["completed", "busy", "timed_out", "failed", "closed"]
    value: Any = None
    trace: dict[str, Any] | None = None


class _WorkTiming:
    """Content-free, per-call timing with immutable-at-return snapshots.

    The event loop reads while the actual pool thread marks start/finish. A
    queue duration is final only after start, and worker duration only after
    finish; None must not be interpreted as a measured zero. No late callback
    mutates a trace already returned to a caller.
    """

    def __init__(self, requested_at: float, budget: float, *, submitted: bool = False):
        self._requested_at = requested_at
        self._budget = budget
        self._submitted = submitted
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._lock = threading.Lock()

    def call(self, work: Callable[[], Any]) -> Any:
        with self._lock:
            self._started_at = time.perf_counter()
        try:
            return work()
        finally:
            with self._lock:
                self._finished_at = time.perf_counter()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.perf_counter()
            start, finish = self._started_at, self._finished_at
            return {
                "wait_ms": round((now - self._requested_at) * 1000, 3),
                "wait_budget_ms": round(self._budget * 1000, 3),
                "queue_ms": None if start is None else round((start - self._requested_at) * 1000, 3),
                "worker_ms": None if finish is None or start is None else round((finish - start) * 1000, 3),
                "worker_elapsed_ms": None if start is None else round(((finish if finish is not None else now) - start) * 1000, 3),
                "work_submitted": self._submitted,
                "worker_started": start is not None,
                "worker_finished": finish is not None,
                "worker_running": start is not None and finish is None,
                "timing_scope": "this_call_snapshot_not_cpu_time",
            }


class BackgroundContextScheduler:
    """A per-service optional lane with no per-service request backlog.

    Uses asyncio's existing worker pool, not another executor per service. The
    deadline limits foreground waiting only; it is not a CPU preemption claim.
    The admitted job may still wait for the shared pool; queue_ms reports that
    delay separately. Algorithms need finite work/memory limits of their own.
    """

    def __init__(self, *, timeout_seconds: float = 1.0):
        if (type(timeout_seconds) not in (int, float)
                or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 30):
            raise ValueError("Invalid background context wait budget")
        self.timeout_seconds = float(timeout_seconds)
        self._task: asyncio.Task | None = None
        self._closed = False

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def _finished(self, task: asyncio.Task) -> None:
        # Retrieve failures even when the waiting request has gone away; never
        # log exception text, which can contain uploaded/private source content.
        if not task.cancelled():
            task.exception()
        if self._task is task:
            self._task = None

    async def run(self, work: Callable[[], Any]) -> ContextWorkResult:
        requested_at = time.perf_counter()
        if self._closed:
            return ContextWorkResult("closed", trace=_WorkTiming(requested_at, self.timeout_seconds).snapshot())
        if not callable(work):
            raise ValueError("Background context work must be callable")
        if self.running:
            # These are this rejected call's timings, not the active request's.
            return ContextWorkResult("busy", trace=_WorkTiming(requested_at, self.timeout_seconds).snapshot())
        # No await between checking and reserving the slot on this event loop.
        timing = _WorkTiming(requested_at, self.timeout_seconds, submitted=True)
        task = asyncio.create_task(asyncio.to_thread(timing.call, work))
        self._task = task
        task.add_done_callback(self._finished)
        try:
            value = await asyncio.wait_for(asyncio.shield(task), self.timeout_seconds)
            return ContextWorkResult("completed", value, timing.snapshot())
        except asyncio.TimeoutError:
            return ContextWorkResult("timed_out", trace=timing.snapshot())
        except asyncio.CancelledError:
            # Leave shielded work and its slot alive until genuine completion.
            raise
        except Exception:
            return ContextWorkResult("failed", trace=timing.snapshot())

    async def close(self) -> None:
        """Stop accepting new work; never wait forever on an unkillable thread.

        The bounded worker may finish normally and releases its one retained
        reference through _finished. Its late result is not used by any answer.
        Application/process shutdown still owns the shared executor lifetime.
        """
        self._closed = True
