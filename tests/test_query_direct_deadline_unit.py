"""Non-streaming requests share the bounded transport lifetime, without retry."""
import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks, HTTPException

from docthinker.harness_runtime import current_run, execute_run
from docthinker.server.schemas import QueryRequest

api = importlib.import_module('docthinker.server.routers.query')


def research_environment(monkeypatch, operation):
    service = SimpleNamespace(answer=AsyncMock(side_effect=operation))
    monkeypatch.setattr(api.state, 'mobile_safe_mode', True)
    monkeypatch.setattr(api.state, 'research_service', service)
    monkeypatch.setattr(api.state, 'session_manager', SimpleNamespace(get_session=lambda _: {'id': 'one'}))
    monkeypatch.setattr(api.state, 'mobile_chat_func', None)
    return service


async def test_direct_research_deadline_cancels_once_without_committing_or_retry(monkeypatch):
    seen, committed, closed = [], [], asyncio.Event()

    async def operation(*args):
        seen.append(current_run())
        try:
            await asyncio.Event().wait()
            committed.append('late answer')
        finally:
            closed.set()

    service = research_environment(monkeypatch, operation)
    monkeypatch.setattr(api, 'SESSION_QUERY_TIMEOUT_SECONDS', 0.02)
    with pytest.raises(HTTPException) as error:
        await asyncio.wait_for(api.query(QueryRequest(question='q', session_id='#00001'), BackgroundTasks()), 1)
    assert error.value.status_code == 504
    assert 'unknown' in error.value.detail and 'No automatic retry' in error.value.detail
    assert closed.is_set() and committed == []
    service.answer.assert_awaited_once()
    assert seen[0].status == 'unknown'
    assert all(row['type'] != 'run.completed' for row in seen[0].snapshot()['events'])
    assert current_run() is None


async def test_inherited_request_keeps_existing_deadline_and_propagates_unknown(monkeypatch):
    seen = []

    async def operation(*args):
        seen.append(current_run())
        await asyncio.Event().wait()

    service = research_environment(monkeypatch, operation)
    monkeypatch.setattr(api, 'SESSION_QUERY_TIMEOUT_SECONDS', 100)
    with pytest.raises(asyncio.TimeoutError):
        await execute_run(lambda: api.query(QueryRequest(question='q', session_id='#00001'), BackgroundTasks()),
                          session_id='#00001', run_id='journal-owner', timeout_seconds=0.02)
    service.answer.assert_awaited_once()
    assert seen[0].run_id == 'journal-owner' and seen[0].status == 'unknown'
    assert sum(row['type'] == 'run.started' for row in seen[0].snapshot()['events']) == 1


async def test_success_preserves_answer_and_has_one_owner(monkeypatch):
    seen = []
    expected = {'answer': 'original answer', 'sources': [], 'research_core': True}

    async def operation(*args):
        seen.append(current_run())
        return expected

    service = research_environment(monkeypatch, operation)
    result = await api.query(QueryRequest(question='q', session_id='#00001'), BackgroundTasks())
    assert result is expected
    service.answer.assert_awaited_once()
    assert seen[0].deadline is not None and seen[0].status == 'completed'


async def test_disconnect_propagates_cancellation_and_never_turns_into_timeout(monkeypatch):
    seen, entered, closed = [], asyncio.Event(), asyncio.Event()

    async def operation(*args):
        seen.append(current_run())
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    service = research_environment(monkeypatch, operation)
    task = asyncio.create_task(api.query(QueryRequest(question='q', session_id='#00001'), BackgroundTasks()))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    service.answer.assert_awaited_once()
    assert closed.is_set() and seen[0].status == 'cancelled'
