"""Fair, bounded local parse ordering with synthetic queues and no providers."""
from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from docthinker.server.research_service import ResearchService

A, B, C = "#00001", "#00002", "#00003"


def document(session, number):
    return {"session_id": session, "id": f"{session}-{number}"}


async def wait_for(predicate):
    async def reached():
        while not predicate():
            await asyncio.sleep(0.005)
    await asyncio.wait_for(reached(), 1)


async def test_backlog_is_round_robin_across_three_projects_and_fifo_within_each(tmp_path, monkeypatch):
    service = ResearchService(tmp_path)
    queue = [document(sid, index) for sid in (A, B, C) for index in range(3)]
    calls, snapshots = [], []
    def pending(limit):
        snapshots.append(limit)
        return list(queue)
    async def process(sid, identifier):
        calls.append((sid, identifier))
        queue[:] = [row for row in queue if row["id"] != identifier]
        return True
    monkeypatch.setattr(service.store, "pending_documents", pending)
    monkeypatch.setattr(service, "process", process)
    service.start()
    try:
        await wait_for(lambda: not queue)
        assert calls == [(sid, f"{sid}-{index}") for index in range(3) for sid in (A, B, C)]
        assert len(snapshots) >= len(calls)
        assert set(snapshots) == {100}
    finally:
        await service.stop()


async def test_new_project_arriving_between_jobs_is_seen_before_existing_backlog(tmp_path, monkeypatch):
    service = ResearchService(tmp_path)
    queue = [document(A, index) for index in range(4)]
    calls = []
    monkeypatch.setattr(service.store, "pending_documents", lambda limit: list(queue))
    async def process(sid, identifier):
        calls.append((sid, identifier))
        queue[:] = [row for row in queue if row["id"] != identifier]
        if len(calls) == 1:
            queue.append(document(B, 0))
            service.notify()
        return True
    monkeypatch.setattr(service, "process", process)
    service.start()
    try:
        await wait_for(lambda: not queue)
        assert calls[:3] == [(A, f"{A}-0"), (B, f"{B}-0"), (A, f"{A}-1")]
        assert len(calls) == 5
    finally:
        await service.stop()


async def test_failed_claim_does_not_loop_or_starve_other_projects_in_same_wake(tmp_path, monkeypatch):
    service = ResearchService(tmp_path)
    queue = [document(A, 0), document(A, 1), document(B, 0), document(C, 0)]
    calls = []
    pending = Mock(side_effect=lambda limit: list(queue))
    monkeypatch.setattr(service.store, "pending_documents", pending)
    async def process(sid, identifier):
        calls.append((sid, identifier))
        if identifier == f"{A}-0":
            return False  # Stays queued after a synthetic durable-claim outage.
        queue[:] = [row for row in queue if row["id"] != identifier]
        return True
    monkeypatch.setattr(service, "process", process)
    service.start()
    try:
        await wait_for(lambda: len(calls) == 4)
        await asyncio.sleep(0.03)
        assert calls == [(A, f"{A}-0"), (B, f"{B}-0"), (C, f"{C}-0"), (A, f"{A}-1")]
        count = pending.call_count
        await asyncio.sleep(0.03)
        assert pending.call_count == count
        service.notify()  # Only an explicit wake permits another claim attempt.
        await wait_for(lambda: len(calls) == 5)
        await asyncio.sleep(0.03)
        assert calls[-1] == (A, f"{A}-0")
        assert len(calls) == 5
    finally:
        await service.stop()


async def test_cycle_cursor_is_preserved_across_empty_queue_and_new_wake(tmp_path, monkeypatch):
    service = ResearchService(tmp_path)
    queue = [document(B, 0)]
    calls = []
    monkeypatch.setattr(service.store, "pending_documents", lambda limit: list(queue))
    async def process(sid, identifier):
        calls.append((sid, identifier))
        queue[:] = [row for row in queue if row["id"] != identifier]
        return True
    monkeypatch.setattr(service, "process", process)
    service.start()
    try:
        await wait_for(lambda: len(calls) == 1)
        await asyncio.sleep(0.02)
        queue.extend([document(A, 1), document(B, 1), document(C, 1)])
        service.notify()
        await wait_for(lambda: len(calls) == 4)
        assert [sid for sid, _ in calls] == [B, C, A, B]
    finally:
        await service.stop()


async def test_all_claims_failing_are_attempted_once_then_runner_sleeps(tmp_path, monkeypatch):
    service = ResearchService(tmp_path)
    queue = [document(A, 0), document(B, 0), document(C, 0)]
    calls = []
    pending = Mock(return_value=queue)
    monkeypatch.setattr(service.store, "pending_documents", pending)
    async def process(sid, identifier):
        calls.append((sid, identifier))
        return False
    monkeypatch.setattr(service, "process", process)
    service.start()
    try:
        await wait_for(lambda: len(calls) == 3)
        await asyncio.sleep(0.03)
        assert [sid for sid, _ in calls] == [A, B, C]
        assert pending.call_count == 4  # One finite scan per attempt + exhaustion.
        assert not service._runner.done()
    finally:
        await service.stop()
