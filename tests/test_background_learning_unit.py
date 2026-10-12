"""Bounded session scheduling and lifecycle cleanup, with no model API calls."""

import asyncio
import importlib
import sys
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from docthinker.server.background_learning import SessionLearningRunner


async def wait(event):
    await asyncio.wait_for(event.wait(), timeout=2)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, float("inf"), float("nan"), "2"])
def test_concurrency_limit_rejects_non_integer_or_unbounded_values(value):
    with pytest.raises(ValueError, match="positive integer"):
        SessionLearningRunner(AsyncMock(), max_concurrent_sessions=value)


async def test_same_session_coalesces_queued_and_running_requests():
    started = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    calls = 0
    active = 0

    async def work(sid):
        nonlocal calls, active
        assert sid == "A"
        index = calls
        calls += 1
        active += 1
        assert active == 1
        started[index].set()
        try:
            await release[index].wait()
        finally:
            active -= 1

    runner = SessionLearningRunner(work)
    try:
        for _ in range(20):
            assert runner.submit("A")
        await wait(started[0])
        assert calls == 1
        for _ in range(20):
            assert runner.submit("A")
        assert runner.snapshot() == {"A": {"running": True, "pending": True}}
        release[0].set()
        await wait(started[1])
        release[1].set()
        await asyncio.wait_for(runner.wait_idle(), 2)
        assert calls == 2
        assert runner.snapshot() == {}
    finally:
        await runner.shutdown()


async def test_sessions_are_isolated_and_global_concurrency_is_bounded():
    releases = {sid: asyncio.Event() for sid in "ABC"}
    starts = asyncio.Queue()
    counts = Counter()
    active = set()
    peak = 0

    async def work(sid):
        nonlocal peak
        assert sid not in active
        active.add(sid)
        peak = max(peak, len(active))
        counts[sid] += 1
        starts.put_nowait(sid)
        try:
            await releases[sid].wait()
        finally:
            active.remove(sid)

    runner = SessionLearningRunner(work, max_concurrent_sessions=2)
    try:
        for sid in "ABC":
            assert runner.submit(sid)
        assert await asyncio.wait_for(starts.get(), 2) == "A"
        assert await asyncio.wait_for(starts.get(), 2) == "B"
        assert not runner.snapshot()["C"]["running"]
        for _ in range(10):
            assert runner.submit("A")
            assert runner.submit("C")
        releases["A"].set()
        assert await asyncio.wait_for(starts.get(), 2) == "C"
        for event in releases.values():
            event.set()
        await asyncio.wait_for(runner.wait_idle(), 2)
        assert counts == {"A": 2, "B": 1, "C": 1}
        assert peak == 2
    finally:
        await runner.shutdown()


async def test_failure_is_observed_pending_run_survives_and_session_is_reusable(caplog):
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def work(sid):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await release.wait()
            raise ValueError("intentional model failure")

    runner = SessionLearningRunner(work)
    try:
        runner.submit("A")
        await wait(started)
        runner.submit("A")
        release.set()
        await asyncio.wait_for(runner.wait_idle(), 2)
        assert calls == 2
        assert runner.snapshot() == {}
        assert "intentional model failure" in caplog.text
        assert runner.submit("A")
        await asyncio.wait_for(runner.wait_idle(), 2)
        assert calls == 3
    finally:
        await runner.shutdown()


async def test_cancel_drops_only_target_session_and_queued_work():
    starts = asyncio.Queue()
    cancelled = []

    async def work(sid):
        starts.put_nowait(sid)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(sid)

    runner = SessionLearningRunner(work, max_concurrent_sessions=1)
    try:
        runner.submit("A")
        assert await asyncio.wait_for(starts.get(), 2) == "A"
        runner.submit("A")
        runner.submit("B")
        assert await runner.cancel("B")
        assert runner.snapshot() == {"A": {"running": True, "pending": True}}
        assert not await runner.cancel("missing")
        assert await runner.cancel("A")
        assert cancelled == ["A"]
        assert runner.snapshot() == {}
        assert runner.submit("B")
        assert await asyncio.wait_for(starts.get(), 2) == "B"
    finally:
        await runner.shutdown()
    assert cancelled == ["A", "B"]


async def test_shutdown_cancels_running_queued_and_pending_requests():
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def work(sid):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    runner = SessionLearningRunner(work, max_concurrent_sessions=1)
    runner.submit("A")
    await wait(started)
    runner.submit("A")
    runner.submit("B")
    await asyncio.wait_for(runner.shutdown(), 2)
    assert stopped.is_set()
    assert runner.closed
    assert runner.snapshot() == {}
    assert not runner.submit("A")
    await runner.shutdown()


async def test_cancel_before_task_starts_releases_session_reference():
    work = AsyncMock()
    runner = SessionLearningRunner(work)
    runner.submit("A")
    assert await runner.cancel("A")
    assert runner.snapshot() == {}
    work.assert_not_awaited()
    await runner.shutdown()


async def test_post_upload_stages_run_serially_even_after_one_failure(monkeypatch):
    ingest = importlib.import_module("docthinker.server.routers.ingest")
    order = []

    async def first(sid):
        order.append(("eclrr", sid))
        raise ValueError("ECLRR failed")

    async def second(sid):
        order.append(("self_study", sid))

    monkeypatch.setattr(ingest, "_background_path_edge_discovery", first)
    monkeypatch.setattr(ingest, "_background_self_study", second)
    await ingest._run_post_upload_learning("A")
    assert order == [("eclrr", "A"), ("self_study", "A")]


async def test_app_lifespan_stops_learning_before_finalizing_graph_on_error(
    tmp_path, monkeypatch
):
    app_module = importlib.import_module("docthinker.server.app")
    ingest = importlib.import_module("docthinker.server.routers.ingest")
    fake_state = SimpleNamespace(vision_vlm_client=None)
    monkeypatch.setattr(app_module, "state", fake_state)
    monkeypatch.setattr(
        app_module, "load_settings", lambda: SimpleNamespace(workdir=str(tmp_path))
    )
    for name in (
        "APIConfig",
        "SessionManager",
        "CognitiveProcessor",
        "IngestionService",
        "_cleanup_global_graphcore_artifacts",
        "save_all_memory_engines",
    ):
        monkeypatch.setattr(app_module, name, Mock())
    monkeypatch.setattr(app_module, "_warmup_llm_connection", AsyncMock())
    # These optional integrations are outside this lifecycle regression.
    for name in ("neuro_memory", "claw", "docthinker.auto_thinking.classifier"):
        monkeypatch.setitem(sys.modules, name, SimpleNamespace())

    order = []
    started = asyncio.Event()

    async def work(sid):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            order.append("learning_cancelled")

    async def finalize():
        order.append("graph_finalized")

    rag = SimpleNamespace(
        llm_model_func=AsyncMock(),
        embedding_func=AsyncMock(),
        finalize_storages=AsyncMock(side_effect=finalize),
    )
    monkeypatch.setattr(app_module, "_initialize_rag", AsyncMock(return_value=rag))
    monkeypatch.setattr(ingest, "_run_post_upload_learning", work)
    with pytest.raises(RuntimeError, match="application error"):
        async with app_module.lifespan(SimpleNamespace()):
            runner = fake_state.post_upload_learning
            assert runner.submit("A")
            await wait(started)
            raise RuntimeError("application error")
    assert order == ["learning_cancelled", "graph_finalized"]
    assert runner.closed and runner.snapshot() == {}
    assert fake_state.post_upload_learning is None
    rag.llm_model_func.assert_not_awaited()
    rag.embedding_func.assert_not_awaited()
