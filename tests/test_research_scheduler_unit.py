"""No real model/clock-cost tests for the bounded local assistance lane."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest

from docthinker.server.research_scheduler import BackgroundContextScheduler, ContextWorkResult


async def _started(event):
    for _ in range(100):
        if event.is_set():
            return
        await asyncio.sleep(0.005)
    pytest.fail("Synthetic worker did not start")


async def _idle(scheduler):
    for _ in range(100):
        if not scheduler.running:
            return
        await asyncio.sleep(0.005)
    pytest.fail("Synthetic worker did not release its slot")


@pytest.mark.parametrize("budget", [0, -1, True, "1", float("inf"), float("nan"), 31])
def test_invalid_wait_budgets_are_rejected(budget):
    with pytest.raises(ValueError):
        BackgroundContextScheduler(timeout_seconds=budget)


async def test_completed_local_work_returns_value_without_cache_or_queue():
    scheduler = BackgroundContextScheduler()
    calls = []
    def work():
        calls.append(1)
        return {"candidate": len(calls)}
    first = await scheduler.run(work)
    second = await scheduler.run(work)
    assert first.status == second.status == "completed"
    assert first.value == {"candidate": 1} and second.value == {"candidate": 2}
    assert not scheduler.running
    for result in (first, second):
        trace = result.trace
        assert trace["work_submitted"] and trace["worker_started"] and trace["worker_finished"]
        assert not trace["worker_running"]
        assert trace["worker_elapsed_ms"] == trace["worker_ms"]
        assert trace["queue_ms"] >= 0 and trace["worker_ms"] >= 0
        assert trace["wait_ms"] + .002 >= trace["queue_ms"] + trace["worker_ms"]
        assert trace["wait_budget_ms"] == 1000
        assert "candidate" not in json.dumps(trace)


async def test_timeout_keeps_actual_worker_slot_and_new_requests_never_queue():
    scheduler = BackgroundContextScheduler(timeout_seconds=0.03)
    started, finish = threading.Event(), threading.Event()
    calls = []
    def blocked():
        calls.append("first")
        started.set()
        finish.wait(3)
        return "late private result"
    try:
        first = await scheduler.run(blocked)
        assert started.is_set() and first.status == "timed_out" and first.value is None
        assert first.trace["worker_started"] and first.trace["worker_running"]
        assert first.trace["queue_ms"] is not None
        assert first.trace["worker_ms"] is None
        assert first.trace["worker_elapsed_ms"] >= 0
        snapshot = dict(first.trace)
        assert scheduler.running
        results = await asyncio.gather(*[scheduler.run(lambda: calls.append("must not run")) for _ in range(50)])
        assert {result.status for result in results} == {"busy"}
        assert calls == ["first"]
        for result in results:
            assert not result.trace["work_submitted"]
            assert result.trace["queue_ms"] is result.trace["worker_ms"] is None
            assert not result.trace["worker_started"]
    finally:
        finish.set()
        await _idle(scheduler)
    assert first.trace == snapshot  # Actual late completion cannot rewrite history.
    assert "private" not in json.dumps(first.trace)
    current = await scheduler.run(lambda: "fresh request")
    assert current.status == "completed" and current.value == "fresh request"


async def test_cancellation_does_not_clear_slot_before_thread_really_finishes():
    scheduler = BackgroundContextScheduler(timeout_seconds=1)
    started, finish = threading.Event(), threading.Event()
    def blocked():
        started.set()
        finish.wait(3)
    task = asyncio.create_task(scheduler.run(blocked))
    try:
        await _started(started)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert scheduler.running
        busy = await scheduler.run(lambda: pytest.fail("overlapping CPU work"))
        assert busy.status == "busy" and not busy.trace["work_submitted"]
        assert busy.trace["queue_ms"] is busy.trace["worker_ms"] is None
    finally:
        finish.set()
        await _idle(scheduler)
    assert (await scheduler.run(lambda: True)).value is True


async def test_failures_are_sanitized_and_release_the_slot():
    scheduler = BackgroundContextScheduler()
    def fail():
        raise RuntimeError("synthetic private document content")
    result = await scheduler.run(fail)
    assert result.status == "failed" and result.value is None
    assert "private" not in repr(result)
    assert result.trace["worker_finished"] and result.trace["worker_ms"] is not None
    assert not result.trace["worker_running"]
    assert (await scheduler.run(lambda: "next")).value == "next"


async def test_shutdown_returns_without_waiting_or_cancelling_a_running_thread():
    scheduler = BackgroundContextScheduler(timeout_seconds=0.03)
    started, finish = threading.Event(), threading.Event()
    def blocked():
        started.set()
        finish.wait(3)
        raise RuntimeError("late private failure must be consumed")
    try:
        assert (await scheduler.run(blocked)).status == "timed_out"
        await asyncio.wait_for(scheduler.close(), 0.1)
        assert scheduler.running
        result = await scheduler.run(lambda: pytest.fail("closed scheduler accepted work"))
        assert result.status == "closed"
        assert not result.trace["work_submitted"]
        assert result.trace["queue_ms"] is result.trace["worker_ms"] is None
    finally:
        finish.set()
        await _idle(scheduler)
    assert (await scheduler.run(lambda: "never")).status == "closed"


async def test_separate_services_do_not_share_results_or_mutual_exclusion():
    first = BackgroundContextScheduler(timeout_seconds=0.03)
    second = BackgroundContextScheduler()
    started, finish = threading.Event(), threading.Event()
    def blocked():
        started.set()
        finish.wait(3)
    try:
        assert (await first.run(blocked)).status == "timed_out"
        result = await second.run(lambda: "other private user")
        assert result.status == "completed" and result.value == "other private user"
    finally:
        finish.set()
        await _idle(first)


async def test_waiting_does_not_block_event_loop_or_require_another_executor():
    scheduler = BackgroundContextScheduler(timeout_seconds=0.03)
    started, finish = threading.Event(), threading.Event()
    def blocked():
        started.set()
        finish.wait(3)
    task = asyncio.create_task(scheduler.run(blocked))
    try:
        await _started(started)
        ticks = []
        for _ in range(3):
            await asyncio.sleep(0.003)
            ticks.append(True)
        assert ticks == [True, True, True]
        assert (await task).status == "timed_out"
    finally:
        finish.set()
        await _idle(scheduler)


def test_context_result_trace_is_optional_for_older_callers_and_test_doubles():
    assert ContextWorkResult("busy").trace is None
    assert ContextWorkResult("completed", "value").value == "value"


def test_shared_pool_starvation_is_distinct_from_worker_execution():
    """Mechanism reproduction only, not attribution of a real model timeout."""
    async def scenario():
        loop = asyncio.get_running_loop()
        # Isolated loop: never replace the pytest application's executor.
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        occupied, release, late_started = threading.Event(), threading.Event(), threading.Event()

        def occupy_pool():
            occupied.set()
            release.wait(3)

        blocker = loop.run_in_executor(None, occupy_pool)
        scheduler = BackgroundContextScheduler(timeout_seconds=.03)
        try:
            await _started(occupied)
            result = await scheduler.run(late_started.set)
            assert result.status == "timed_out" and not late_started.is_set()
            trace = dict(result.trace)
            assert trace["work_submitted"] and not trace["worker_started"]
            assert not trace["worker_running"] and not trace["worker_finished"]
            assert trace["queue_ms"] is trace["worker_ms"] is trace["worker_elapsed_ms"] is None
            assert trace["wait_ms"] >= trace["wait_budget_ms"]
            assert scheduler.running  # A queued task still retains its lane.
            assert (await scheduler.run(lambda: pytest.fail("queued twice"))).status == "busy"
        finally:
            release.set()
            await blocker
            await _idle(scheduler)
            await scheduler.close()
        assert late_started.is_set()
        assert result.trace == trace
    asyncio.run(scenario())
