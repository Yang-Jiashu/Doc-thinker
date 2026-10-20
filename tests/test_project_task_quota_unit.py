"""Long runs reserve finite normal-query units atomically, never one free batch."""
import hashlib
import json
import sqlite3
import uuid

import httpx
import pytest

from tests.test_invite_store_unit import registry, enroll
from docthinker.server.invite_store import InviteError
from docthinker.server.invite_app import create_invite_app

ORIGIN = 'http://127.0.0.1:8120'


def count(store):
    with sqlite3.connect(store.path) as db:
        return db.execute("SELECT COALESCE(SUM(count),0) FROM usage WHERE kind='query'").fetchone()[0]


def test_six_call_approval_consumes_six_units_once_and_changed_budget_conflicts(registry):
    store, _ = registry
    _, auth = enroll(store)
    uid, rid, digest = auth['user']['id'], str(uuid.uuid4()), '1' * 64
    assert store.reserve_query(uid, '#00001', rid, digest, units=6)
    assert count(store) == 6
    assert not store.reserve_query(uid, '#00001', rid, digest, units=6)
    assert count(store) == 6
    for units in (1, 2, 5):
        with pytest.raises(InviteError) as error:
            store.reserve_query(uid, '#00001', rid, digest, units=units)
        assert error.value.status == 409
    assert count(store) == 6


def test_multi_unit_admission_is_all_or_nothing_at_personal_quota(registry):
    store, _ = registry
    _, auth = enroll(store)
    uid = auth['user']['id']
    for index in range(8):
        assert store.reserve_query(uid, '#00001', str(uuid.uuid4()), hashlib.sha256(str(index).encode()).hexdigest(), units=6)
    assert count(store) == 48
    rid = str(uuid.uuid4())
    with pytest.raises(InviteError) as error:
        store.reserve_query(uid, '#00001', rid, 'f' * 64, units=3)
    assert error.value.status == 429 and count(store) == 48
    with sqlite3.connect(store.path) as db:
        assert db.execute('SELECT count(*) FROM query_reservations WHERE request_id=?', (rid,)).fetchone()[0] == 0
    assert store.reserve_query(uid, '#00001', rid, 'f' * 64, units=2)
    assert count(store) == 50


@pytest.mark.parametrize('units', [0, 7, True, 1.0, '2'])
def test_invalid_batch_units_have_no_ledger_side_effect(registry, units):
    store, _ = registry
    _, auth = enroll(store)
    with pytest.raises(InviteError) as error:
        store.reserve_query(auth['user']['id'], '#00001', str(uuid.uuid4()), 'f' * 64, units=units)
    assert error.value.status == 422 and count(store) == 0


async def test_gateway_routes_batch_as_queries_not_metadata_and_replay_is_free(registry, monkeypatch):
    store, _ = registry
    _, auth = enroll(store)
    monkeypatch.setenv('DOCTHINKER_RESEARCH_ENABLED', 'true')
    calls = []
    async def upstream(request):
        calls.append(request)
        return httpx.Response(200, json={'task': {'status': 'ready'}})
    app = create_invite_app(store=store, origin=ORIGIN, transport=httpx.MockTransport(upstream))
    approval = {'session_id': '#00001', 'task_id': str(uuid.uuid4()), 'expected_revision': 1,
                'request_id': str(uuid.uuid4()), 'max_calls': 3}
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=ORIGIN,
            headers={'Origin': ORIGIN, 'Authorization': 'Bearer ' + auth['access_token']}) as browser:
            for _ in range(2):
                response = await browser.post('/api/v1/mobile/workbench/tasks/run', json=approval)
                assert response.status_code == 200, response.text
            assert count(store) == 3 and len(calls) == 2
            altered = await browser.post('/api/v1/mobile/workbench/tasks/run', json={**approval, 'max_calls': 4})
            assert altered.status_code == 409 and count(store) == 3 and len(calls) == 2
            for changed in ({'max_calls': True}, {'max_calls': 7}, {'user_id': 'other'}):
                denied = await browser.post('/api/v1/mobile/workbench/tasks/run', json={**approval, **changed})
                assert denied.status_code in {400, 422} and count(store) == 3
            assert len(calls) == 2


async def test_disabled_research_blocks_task_admission_before_quota_or_upstream(registry, monkeypatch):
    store, _ = registry
    _, auth = enroll(store)
    monkeypatch.setenv('DOCTHINKER_RESEARCH_ENABLED', 'false')
    calls = []
    async def upstream(request):
        calls.append(request)
        return httpx.Response(200, json={})
    app = create_invite_app(store=store, origin=ORIGIN, transport=httpx.MockTransport(upstream))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=ORIGIN,
            headers={'Origin': ORIGIN, 'Authorization': 'Bearer ' + auth['access_token']}) as browser:
            response = await browser.post('/api/v1/mobile/workbench/tasks/run', json={
                'session_id': '#00001', 'task_id': str(uuid.uuid4()), 'expected_revision': 1,
                'request_id': str(uuid.uuid4()), 'max_calls': 6})
            assert response.status_code == 403 and count(store) == 0 and not calls
