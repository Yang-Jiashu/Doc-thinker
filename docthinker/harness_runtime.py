"""One request-local execution owner shared by transports and capabilities.

This is an in-process lifecycle, not a durable job queue. A snapshot contains
bounded operational metadata only. Cancelling the local coroutine never claims
to roll back a provider request or stop an already running worker thread.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import json
import math
import re
import time
from typing import Any, AsyncIterator, Awaitable, Callable, TypeVar
import uuid

T = TypeVar("T")
MAX_EVENTS = 64
MAX_EVENT_BYTES = 1024
_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")
_CURRENT: ContextVar[HarnessRun | None] = ContextVar("docthinker_harness_run", default=None)
# Textual content is deliberately not an event field. Callers may only expose
# identifiers, counts, timings and bounded enum-like operational values.
_PAYLOAD_KEYS = frozenset({
    "status", "reason", "elapsed_ms", "source_count", "selected_count", "candidate_count",
    "model_calls", "input_bytes", "output_bytes", "budget_bytes", "version", "schema_id",
    "activation_count", "omitted_count", "external_outcome", "code", "count", "enabled",
})


class HarnessRun:
    def __init__(self, session_id: str | None, *, run_id: str | None = None,
                 timeout_seconds: float | None = None):
        if session_id is not None and (not isinstance(session_id, str) or len(session_id) > 128):
            raise ValueError("Invalid run session")
        if run_id is not None and (not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id)):
            raise ValueError("Invalid run identifier")
        if timeout_seconds is not None and (type(timeout_seconds) not in (int, float)
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("Invalid run deadline")
        self.run_id = run_id or str(uuid.uuid4())
        self.session_id = session_id
        self.status = "accepted"
        self.stage_name = None
        self.started_at = time.monotonic()
        self.deadline = self.started_at + timeout_seconds if timeout_seconds is not None else None
        self._events: list[dict[str, Any]] = []
        self._seq = 0
        self._cancel_requested = False
        self._task: asyncio.Task | None = None

    def emit(self, event_type: str, stage: str | None = None, payload: dict | None = None):
        if not isinstance(event_type, str) or not _NAME.fullmatch(event_type):
            raise ValueError("Invalid run event")
        if stage is not None and (not isinstance(stage, str) or not _NAME.fullmatch(stage)):
            raise ValueError("Invalid run stage")
        clean = {}
        for key, value in (payload or {}).items():
            if key not in _PAYLOAD_KEYS:
                continue
            if isinstance(value, str):
                # Enum-like fields cannot contain a prompt, source excerpt or URL.
                if re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", value):
                    clean[key] = value
            elif value is None or type(value) is bool:
                clean[key] = value
            elif type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1e12:
                clean[key] = value
        self._seq += 1
        event = {"seq": self._seq, "type": event_type, "stage": stage or self.stage_name,
                 "elapsed_ms": round((time.monotonic() - self.started_at) * 1000, 3)}
        if clean:
            event["payload"] = clean
        if len(json.dumps(event, separators=(",", ":")).encode()) > MAX_EVENT_BYTES:
            event.pop("payload", None)
        self._events.append(event)
        self._events = self._events[-MAX_EVENTS:]
        return dict(event)

    def checkpoint(self):
        if self._cancel_requested:
            raise asyncio.CancelledError()
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise asyncio.TimeoutError()

    def request_cancel(self):
        if self.status != "running" or self._cancel_requested:
            return False
        self._cancel_requested = True
        self.emit("run.cancel_requested", payload={"external_outcome": "unknown"})
        if self._task is not None:
            self._task.cancel()
        return True

    @contextmanager
    def stage(self, name: str):
        self.checkpoint()
        previous = self.stage_name
        self.emit("stage.started", stage=name)
        self.stage_name = name
        try:
            yield self
            self.checkpoint()
        except BaseException:
            self.emit("stage.interrupted", stage=name)
            raise
        else:
            self.emit("stage.completed", stage=name)
        finally:
            self.stage_name = previous

    def snapshot(self):
        return {"version": 1, "run_id": self.run_id, "session_id": self.session_id,
                "stage": self.stage_name, "status": self.status, "seq": self._seq,
                "events": json.loads(json.dumps(self._events)),
                "omitted_events": max(0, self._seq - len(self._events)),
                "durable": False, "external_outcome_verified": False}

    async def execute(self, operation: Callable[[], Awaitable[T]]) -> T:
        if self.status != "accepted":
            raise RuntimeError("A run may execute only once")
        if not callable(operation):
            raise TypeError("Run operation must be callable")
        self.status = "running"
        self._task = asyncio.current_task()
        token = _CURRENT.set(self)
        self.emit("run.started")
        try:
            self.checkpoint()
            if self.deadline is None:
                result = await operation()
            else:
                result = await asyncio.wait_for(operation(), max(0, self.deadline - time.monotonic()))
            self.checkpoint()
        except asyncio.CancelledError:
            self.status = "cancelled"
            self.emit("run.cancelled", payload={"external_outcome": "unknown"})
            raise
        except asyncio.TimeoutError:
            self.status = "unknown"
            self.emit("run.unknown", payload={"reason": "timeout", "external_outcome": "unknown"})
            raise
        except BaseException:
            self.status = "failed"
            self.emit("run.failed", payload={"external_outcome": "unknown"})
            raise
        else:
            self.status = "completed"
            self.emit("run.completed")
            return result
        finally:
            self._task = None
            _CURRENT.reset(token)


def current_run() -> HarnessRun | None:
    run = _CURRENT.get()
    return run if run is not None and run.status == "running" else None


async def execute_run(operation: Callable[[], Awaitable[T]], *, session_id: str | None,
                      run_id: str | None = None, timeout_seconds: float | None = None) -> T:
    run = current_run()
    if run is not None:
        if session_id != run.session_id or (run_id is not None and run_id != run.run_id):
            raise ValueError("Nested execution belongs to a different run")
        run.checkpoint()
        # The caller already owns this lifetime. An adapter must not produce
        # another task, completion, retry or model call.
        return await operation()
    return await HarnessRun(session_id, run_id=run_id, timeout_seconds=timeout_seconds).execute(operation)


async def stream_run(factory: Callable[[], AsyncIterator[T]], *, session_id: str | None,
                     run_id: str | None = None, timeout_seconds: float | None = None) -> AsyncIterator[T]:
    """Keep one owner alive for consumption, including disconnect/early close."""
    inherited = current_run()
    if inherited is not None:
        if inherited.session_id != session_id or (run_id is not None and inherited.run_id != run_id):
            raise ValueError("Nested execution belongs to a different run")
        iterator = factory()
        try:
            async for item in iterator:
                inherited.checkpoint()
                yield item
        finally:
            await iterator.aclose()
        return
    run = HarnessRun(session_id, run_id=run_id, timeout_seconds=timeout_seconds)
    # Execute the producer through the same lifecycle without buffering the
    # answer. A one-slot queue propagates backpressure; closing the transport
    # cancels and observes its producer instead of leaving generation running.
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    done = object()

    async def produce():
        iterator = factory()
        try:
            async for item in iterator:
                run.checkpoint()
                await queue.put((True, item))
        finally:
            await iterator.aclose()

    async def drive():
        try:
            await run.execute(produce)
        except asyncio.CancelledError as error:
            # Cleanup must never block trying to publish into a full queue
            # after the consumer has closed its iterator.
            try:
                queue.put_nowait((False, error))
            except asyncio.QueueFull:
                pass
            raise
        except BaseException as error:
            await queue.put((False, error))
        else:
            await queue.put((True, done))

    task = asyncio.create_task(drive())
    try:
        while True:
            if task.done() and queue.empty():
                task.result()
                break
            ok, value = await queue.get()
            if not ok:
                raise value
            if value is done:
                break
            yield value
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
