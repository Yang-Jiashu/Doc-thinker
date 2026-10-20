"""Memory revocation through actual mobile HTTP replay and persisted history.

The ASGI router, journal, service, memory store and SessionManager are real.
Only text generation is synthetic; these tests make no model-quality claim.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
import uuid

import pytest

from docthinker.server.research_service import ResearchService
from tests.test_research_mobile_unit import (
    PREFIX, mobile as base_mobile, mobile_client, upload_ready,
)
from tests.test_research_service_unit import SessionManager, text_extractor


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('PYTHON_DOTENV_DISABLED', '1')
    monkeypatch.setenv('DOCTHINKER_RESEARCH_SEMANTIC', 'off')


def new_service(root):
    return ResearchService(root, extractor=text_extractor,
        context_assistance_enabled=False, semantic_retriever=None,
        evidence_reranker=None, query_translator=None)


def new_manager(root):
    return SessionManager(str(root / '_system'), str(root), migrate_legacy=False)


@pytest.fixture
def transport(base_mobile, monkeypatch):
    # The general mobile fixture supplies a real app/auth/router but a fake
    # session manager. Replace it to exercise marker persistence after restart.
    manager = new_manager(base_mobile.root)
    sid = manager.create_session('家庭预算与私人安排')['id']
    service = new_service(base_mobile.root)
    monkeypatch.setattr(base_mobile.api, '_query_boot_id', 'offline-personal-memory-initial-process')
    base_mobile.manager = base_mobile.state.session_manager = manager
    base_mobile.service = base_mobile.state.research_service = service
    return SimpleNamespace(mobile=base_mobile, sid=sid, service=service, manager=manager)


async def restart(transport):
    await transport.service.stop()
    mobile = transport.mobile
    transport.manager = new_manager(mobile.root)
    transport.service = new_service(mobile.root)
    mobile.manager = mobile.state.session_manager = transport.manager
    mobile.service = mobile.state.research_service = transport.service
    mobile.api._query_boot_id = 'offline-personal-memory-restarted-process'


async def confirmed_answer(transport, browser, *, secret='私密预算4827元'):
    model = AsyncMock(return_value=secret + '仅供规划；没有执行付款。')
    transport.mobile.state.mobile_chat_func = model
    saved = await browser.post(PREFIX + '/research/memory', json={
        'session_id': transport.sid, 'kind': 'decision', 'text': secret + '，不得自动付款。'})
    assert saved.status_code == 200, saved.text
    memory = saved.json()['memory']
    payload = {'session_id': transport.sid, 'question': '我已确认的决定是什么？',
        'request_id': str(uuid.uuid4()), 'use_memory': True, 'use_conversation_context': False,
        'use_llm_cache': False, 'use_self_evolution': False}
    response = await browser.post(PREFIX + '/query', json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['request_status'] == 'completed' and secret in result['answer']
    assert result['research_memory_required'] is True and model.await_count == 1
    return memory, payload, result, model


def assert_hidden(response, secret):
    assert response.status_code == 200, response.text
    assert secret not in json.dumps(response.json(), ensure_ascii=False), 'Private text escaped in body or a related UI card'
    assert response.headers.get('cache-control') == 'no-store'


@pytest.mark.asyncio
@pytest.mark.parametrize('mutation', ['delete', 'correct'])
async def test_completed_mobile_request_replay_rechecks_memory_after_mutation_and_restart(transport, mutation):
    mobile, sid = transport.mobile, transport.sid
    secret = '私密预算4827元'
    try:
        async with mobile_client(mobile) as browser:
            memory, payload, result, model = await confirmed_answer(transport, browser, secret=secret)
            if mutation == 'delete':
                changed = await browser.post(PREFIX + '/research/memory/delete', json={
                    'session_id': sid, 'memory_id': memory['id'], 'expected_revision': memory['revision']})
            else:
                changed = await browser.post(PREFIX + '/research/memory', json={
                    'session_id': sid, 'memory_id': memory['id'], 'expected_revision': memory['revision'],
                    'kind': 'decision', 'text': '目前预算更正为4300元，不得自动付款。'})
            assert changed.status_code == 200, changed.text
            await restart(transport)
            persisted = next(row for row in transport.manager.get_history(sid) if row['id'] == result['answer_id'])
            assert persisted['research_memory_required'] is True
            blocked = AsyncMock(side_effect=AssertionError('Replay must not regenerate'))
            mobile.state.mobile_chat_func = blocked
            replay = await browser.post(PREFIX + '/query', json=payload)
            status = await browser.get(PREFIX + '/query/status', params={
                'session_id': sid, 'request_id': payload['request_id']})
            history = await browser.get(PREFIX + '/history', params={'session_id': sid})
            assert_hidden(replay, secret)
            assert_hidden(status, secret)
            assert secret not in json.dumps(history.json(), ensure_ascii=False)
            assert replay.json()['request_status'] == status.json()['request_status'] == 'completed'
            assert replay.json()['answer_id'] == status.json()['answer_id'] == result['answer_id']
            assert replay.json()['sources'] == status.json()['sources'] == []
            assert model.await_count == 1
            blocked.assert_not_awaited()
    finally:
        await transport.service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize('outage', ['disabled_service', 'failed_hydration'])
async def test_mobile_replay_and_history_fail_closed_on_current_memory_service_outage(transport, monkeypatch, outage):
    mobile, sid = transport.mobile, transport.sid
    secret = '私人送货地址虹桥路777号'
    try:
        async with mobile_client(mobile) as browser:
            _, payload, result, model = await confirmed_answer(transport, browser, secret=secret)
            await restart(transport)
            if outage == 'disabled_service':
                monkeypatch.setattr(mobile.state, 'research_service', None)
            else:
                def unavailable(*args, **kwargs):
                    raise OSError('offline metadata read unavailable')
                monkeypatch.setattr(transport.service, 'hydrate_history', unavailable)
            replay = await browser.post(PREFIX + '/query', json=payload)
            status = await browser.get(PREFIX + '/query/status', params={
                'session_id': sid, 'request_id': payload['request_id']})
            history = await browser.get(PREFIX + '/history', params={'session_id': sid})
            assert_hidden(replay, secret)
            assert_hidden(status, secret)
            assert secret not in json.dumps(history.json(), ensure_ascii=False)
            for packet in (replay.json(), status.json()):
                assert packet['answer_id'] == result['answer_id'] and packet['request_status'] == 'completed'
                assert all(key not in packet for key in ('reasoning', 'claim_audit', 'knowledge_skill', 'trial_delivery'))
            assert model.await_count == 1
    finally:
        await transport.service.stop()


@pytest.mark.asyncio
async def test_memory_marker_alone_survives_session_restart_and_hides_history_during_outage(transport, monkeypatch):
    mobile, sid = transport.mobile, transport.sid
    secret = '只有记忆标记的私人记录token4827'
    identifier = transport.manager.add_message(sid, 'assistant', secret, research_memory_required=True)
    try:
        await restart(transport)
        row = next(item for item in transport.manager.get_history(sid) if item['id'] == identifier)
        assert row['research_memory_required'] is True and not row.get('research_provenance_required')
        monkeypatch.setattr(mobile.state, 'research_service', None)
        async with mobile_client(mobile) as browser:
            history = await browser.get(PREFIX + '/history', params={'session_id': sid})
        assert history.status_code == 200, history.text
        assert secret not in json.dumps(history.json(), ensure_ascii=False)
    finally:
        await transport.service.stop()


@pytest.mark.asyncio
async def test_memory_dependency_write_outage_is_not_retried_and_cannot_replay_raw_reply(transport, monkeypatch):
    mobile, sid = transport.mobile, transport.sid
    secret = '不可回放的私人预算4827元'
    def unavailable(*args, **kwargs):
        raise OSError('offline memory dependency write unavailable')
    monkeypatch.setattr(transport.service.memory_dependencies, 'record', unavailable)
    model = AsyncMock(return_value=secret)
    mobile.state.mobile_chat_func = model
    try:
        async with mobile_client(mobile) as browser:
            saved = await browser.post(PREFIX + '/research/memory', json={
                'session_id': sid, 'kind': 'decision', 'text': secret + '，不得自动付款。'})
            assert saved.status_code == 200
            payload = {'session_id': sid, 'question': '我已确认的决定是什么？',
                'request_id': str(uuid.uuid4()), 'use_memory': True}
            first = await browser.post(PREFIX + '/query', json=payload)
            assert_hidden(first, secret)
            result = first.json()
            assert result['request_status'] == 'completed' and result['research_memory_required'] is True
            await restart(transport)
            row = next(item for item in transport.manager.get_history(sid) if item['id'] == result['answer_id'])
            assert row['research_memory_required'] is True
            replay = await browser.post(PREFIX + '/query', json=payload)
            history = await browser.get(PREFIX + '/history', params={'session_id': sid})
            assert_hidden(replay, secret)
            assert secret not in json.dumps(history.json(), ensure_ascii=False)
            assert model.await_count == 1
    finally:
        await transport.service.stop()


async def source_answer(transport, browser):
    text = '客户专属延期通知silverfox：交付截止改为2026-11-06；原日期作废。'
    document_id = await upload_ready(transport.mobile, browser, sid=transport.sid,
        text=text, filename='client-deadline.txt')
    model = AsyncMock(return_value='客户通知交付截止改为2026-11-06 [S1]。')
    transport.mobile.state.mobile_chat_func = model
    result = await browser.post(PREFIX + '/query', json={
        'session_id': transport.sid, 'question': '客户通知交付截止是什么？',
        'request_id': str(uuid.uuid4()), 'use_memory': False, 'use_conversation_context': False})
    assert result.status_code == 200, result.text
    assert result.json()['request_status'] == 'completed' and result.json()['sources']
    assert model.await_count == 1
    return document_id, result.json(), model


@pytest.mark.asyncio
async def test_actual_memory_api_keeps_valid_answer_origin_and_distinguishes_omission_from_unbinding(transport):
    mobile, sid = transport.mobile, transport.sid
    try:
        async with mobile_client(mobile) as browser:
            _, origin, model = await source_answer(transport, browser)
            response = await browser.post(PREFIX + '/research/memory', json={
                'session_id': sid, 'kind': 'decision', 'text': '交付时间按客户通知安排，执行前另行批准。',
                'source_answer_id': origin['answer_id']})
            assert response.status_code == 200, response.text
            note = response.json()['memory']
            assert note['source_answer_id'] == origin['answer_id']
            assert {row['id'] for row in origin['sources']} <= set(note['source_ids'])
            page = await browser.get(PREFIX + '/research/memory', params={'session_id': sid})
            assert next(row for row in page.json()['items'] if row['id'] == note['id'])['source_answer_id'] == origin['answer_id']
            updated = await browser.post(PREFIX + '/research/memory', json={
                'session_id': sid, 'memory_id': note['id'], 'expected_revision': note['revision'],
                'kind': 'decision', 'text': '更正表达：按客户通知安排，仍须另行批准。'})
            assert updated.status_code == 200, updated.text
            revised = updated.json()['memory']
            assert revised['source_answer_id'] == origin['answer_id']
            cleared = await browser.post(PREFIX + '/research/memory', json={
                'session_id': sid, 'memory_id': note['id'], 'expected_revision': revised['revision'],
                'kind': 'decision', 'text': '用户独立输入的新安排，不再以之前答案为依据。',
                'source_answer_id': None, 'source_ids': []})
            assert cleared.status_code == 200, cleared.text
            assert cleared.json()['memory']['source_answer_id'] is None
            assert cleared.json()['memory']['source_ids'] == []
            assert model.await_count == 1
    finally:
        await transport.service.stop()


@pytest.mark.asyncio
async def test_actual_memory_api_rejects_forged_answer_origins_and_redacts_after_origin_document_deletion(transport):
    mobile, sid = transport.mobile, transport.sid
    secret = '客户专属确认记录silverfox'
    try:
        async with mobile_client(mobile) as browser:
            document_id, origin, model = await source_answer(transport, browser)
            other = transport.manager.create_session('另一隔离客户项目')['id']
            for destination, answer_id in ((other, origin['answer_id']), (sid, str(uuid.uuid4()))):
                forged = await browser.post(PREFIX + '/research/memory', json={
                    'session_id': destination, 'kind': 'finding', 'text': '没有合法来源的确认。',
                    'source_answer_id': answer_id})
                assert forged.status_code == 422, forged.text
            saved = await browser.post(PREFIX + '/research/memory', json={
                'session_id': sid, 'kind': 'finding', 'text': secret,
                'source_answer_id': origin['answer_id']})
            assert saved.status_code == 200, saved.text
            note = saved.json()['memory']
            deleted = await browser.post(PREFIX + '/research/documents/delete', json={
                'session_id': sid, 'document_id': document_id})
            assert deleted.status_code == 200, deleted.text
            await restart(transport)
            page = await browser.get(PREFIX + '/research/memory', params={'session_id': sid})
            versions = await browser.get(PREFIX + '/research/memory/history', params={
                'session_id': sid, 'memory_id': note['id']})
            assert page.status_code == versions.status_code == 200
            current = next(row for row in page.json()['items'] if row['id'] == note['id'])
            assert current['redacted'] is True and current['text'] == ''
            assert current['source_answer_id'] is None and current['source_ids'] == []
            assert versions.json()['items']
            assert all(row['redacted'] and row['text'] == '' and row['source_answer_id'] is None
                and row['source_ids'] == [] for row in versions.json()['items'])
            assert secret not in json.dumps([page.json(), versions.json()], ensure_ascii=False)
            source = await browser.get(PREFIX + '/research/source', params={
                'session_id': sid, 'chunk_id': origin['sources'][0]['id']})
            assert source.status_code == 404
            assert model.await_count == 1
    finally:
        await transport.service.stop()
