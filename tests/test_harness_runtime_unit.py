"""Execution ownership, cancellation and transport adapter contracts."""
import asyncio
import json

import pytest

from docthinker.harness_runtime import HarnessRun, MAX_EVENTS, current_run, execute_run, stream_run


async def test_nested_adapters_share_owner_and_operation_is_called_once():
    runs, calls = [], []

    async def provider():
        runs.append(current_run())
        calls.append("provider")
        with current_run().stage("generation"):
            await asyncio.sleep(0)
        return "answer"

    async def service():
        runs.append(current_run())
        return await execute_run(provider, session_id="#00001")

    result = await execute_run(lambda: execute_run(service, session_id="#00001"),
                               session_id="#00001", run_id="accepted-request")
    assert result == "answer"
    assert calls == ["provider"]
    assert runs[0] is runs[1]
    receipt = runs[0].snapshot()
    assert receipt["run_id"] == "accepted-request"
    assert receipt["status"] == "completed"
    assert [item["type"] for item in receipt["events"]].count("run.started") == 1
    assert [item["type"] for item in receipt["events"]].count("run.completed") == 1
    assert current_run() is None


async def test_scope_mismatch_never_dispatches_nested_operation():
    called = []

    async def forbidden():
        called.append(True)

    async def operation():
        with pytest.raises(ValueError, match="different run"):
            await execute_run(forbidden, session_id="#00002")
        with pytest.raises(ValueError, match="different run"):
            await execute_run(forbidden, session_id="#00001", run_id="other")

    await execute_run(operation, session_id="#00001")
    assert not called


async def test_deadline_cancels_once_and_reports_unknown_external_outcome():
    stopped = asyncio.Event()
    run = HarnessRun("#00001", timeout_seconds=0.01)
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    with pytest.raises(asyncio.TimeoutError):
        await run.execute(operation)
    assert stopped.is_set() and calls == 1
    assert run.snapshot()["status"] == "unknown"
    assert run.snapshot()["events"][-1]["payload"]["external_outcome"] == "unknown"
    with pytest.raises(RuntimeError, match="only once"):
        await run.execute(operation)


async def test_cancel_running_operation_is_observed_without_retry_or_rollback_claim():
    started = asyncio.Event()
    stopped = asyncio.Event()
    run = HarnessRun("#00001")

    async def operation():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    task = asyncio.create_task(run.execute(operation))
    await started.wait()
    assert run.request_cancel()
    assert not run.request_cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()
    assert run.snapshot()["status"] == "cancelled"
    assert run.snapshot()["external_outcome_verified"] is False


async def test_concurrent_sessions_do_not_inherit_each_others_run():
    async def operation(sid):
        first = current_run()
        await asyncio.sleep(0)
        assert current_run() is first
        return first.session_id, first.run_id

    results = await asyncio.gather(*[
        execute_run(lambda sid=sid: operation(sid), session_id=sid)
        for sid in ("#00001", "#00002")
    ])
    assert results[0][0] != results[1][0]
    assert results[0][1] != results[1][1]


def test_receipt_is_bounded_detached_and_excludes_content_fields():
    run = HarnessRun("#00001")
    for _ in range(MAX_EVENTS + 20):
        run.emit("insight.selected", payload={"count": 2, "question": "private question",
                 "source": "private source", "api_key": "secret", "status": "ready"})
    receipt = run.snapshot()
    assert len(receipt["events"]) == MAX_EVENTS
    assert receipt["omitted_events"] == 20
    assert "private" not in json.dumps(receipt)
    assert "secret" not in json.dumps(receipt)
    receipt["events"][0]["payload"]["count"] = 99
    assert run.snapshot()["events"][0]["payload"]["count"] == 2


async def test_stream_uses_same_owner_and_cleans_up_on_early_close():
    runs = []
    stopped = asyncio.Event()

    async def generate():
        runs.append(current_run())
        try:
            for item in range(100):
                yield item
        finally:
            stopped.set()

    stream = stream_run(generate, session_id="#00001")
    assert await anext(stream) == 0
    assert current_run() is None
    await asyncio.wait_for(stream.aclose(), 1)
    assert stopped.is_set()
    assert runs[0].status == "cancelled"


async def test_stream_exception_is_not_swallowed_or_retried():
    calls = []

    async def generate():
        calls.append(current_run())
        yield "first"
        raise ValueError("provider failed")

    stream = stream_run(generate, session_id="#00001")
    assert await anext(stream) == "first"
    with pytest.raises(ValueError, match="provider failed"):
        await anext(stream)
    assert len(calls) == 1 and calls[0].status == "failed"
