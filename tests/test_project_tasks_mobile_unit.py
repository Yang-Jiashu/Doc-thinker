"""Durable multi-step HTTP workflow with actual stores; no paid model calls."""
import asyncio
import json
import sqlite3
import uuid

import pytest

from tests.test_research_mobile_unit import (mobile as research_mobile, mobile_client,
    upload_ready, SID, OTHER_SID, PREFIX)
from docthinker.server.routers import project_tasks


@pytest.fixture
def mobile(research_mobile):
    research_mobile.manager.get_session.side_effect = lambda sid: {
        'id': sid, 'instance_id': 'instance-' + sid[1:]} if sid in research_mobile.manager.histories else None
    return research_mobile


def plan(documents=None):
    return {'session_id': SID, 'goal': 'Compare renewable supply and imports with evidence',
            'constraints': 'Preserve conditions and original evidence', 'questions': [
        {'id': 'q1', 'title': 'Supply', 'question': 'How does renewable supply reduce imports?',
         'document_ids': documents},
        {'id': 'q2', 'title': 'Conditions', 'question': 'What conditions apply to renewable supply and imports?',
         'document_ids': documents, 'depends_on': ['q1']}]}


async def settled(browser, task_id):
    for _ in range(200):
        response = await browser.get(PREFIX + '/workbench/task', params={'session_id': SID, 'task_id': task_id})
        assert response.status_code == 200, response.text
        task = response.json()['task']
        if task['status'] in {'completed', 'paused', 'blocked', 'stale', 'cancelled'}:
            return task
        await asyncio.sleep(.01)
    raise AssertionError('Task did not make durable progress')


async def start(browser, task, max_calls=6, request_id=None):
    response = await browser.post(PREFIX + '/workbench/tasks/run', json={
        'session_id': SID, 'task_id': task['id'], 'expected_revision': task['revision'],
        'request_id': request_id or str(uuid.uuid4()), 'max_calls': max_calls})
    assert response.status_code == 200, response.text
    return response.json()['task']


async def test_creation_and_reads_do_not_generate_or_expose_other_instances(mobile):
    try:
        async with mobile_client(mobile) as browser:
            caps = (await browser.get(PREFIX + '/capabilities')).json()
            assert caps['workbench']['tasks'] is True
            result = await browser.post(PREFIX + '/workbench/tasks', json=plan())
            assert result.status_code == 200, result.text
            task = result.json()['task']
            assert task['status'] == 'ready' and len(task['steps']) == 4
            assert len((await browser.get(PREFIX + '/workbench/tasks', params={'session_id': SID})).json()['items']) == 1
            denied = await browser.get(PREFIX + '/workbench/task', params={'session_id': OTHER_SID, 'task_id': task['id']})
            assert denied.status_code == 404
            assert (await browser.get(PREFIX + '/workbench/tasks', params={'session_id': OTHER_SID})).json()['items'] == []
        mobile.model.assert_not_awaited()
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_http_staged_plan_keeps_twelve_nodes_but_not_twelve_call_grant(mobile):
    rows = []
    for index in range(6):
        key = f'q{index}'
        base = {'question': f'Draft private checklist {index}; do not execute it.',
                'input_kind': 'personal', 'memory_ids': []}
        rows.extend([{**base, 'id': key}, {**base, 'id': f'v{index}',
            'purpose': 'verify', 'review_of': key, 'depends_on': [key]}])
    try:
        async with mobile_client(mobile) as browser:
            response = await browser.post(PREFIX + '/workbench/tasks',
                json={**plan(), 'questions': rows})
            assert response.status_code == 200, response.text
            task = response.json()['task']
            analyses = [row for row in task['steps'] if row['kind'] == 'analysis']
            assert len(analyses) == 12
            assert len([row for row in analyses if row.get('purpose') == 'verify']) == 6
            denied = await browser.post(PREFIX + '/workbench/tasks/run', json={
                'session_id': SID, 'task_id': task['id'], 'expected_revision': task['revision'],
                'request_id': str(uuid.uuid4()), 'max_calls': 12})
            assert denied.status_code == 422
        mobile.model.assert_not_awaited()
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_one_approval_executes_dependency_steps_and_real_report(mobile):
    try:
        async with mobile_client(mobile) as browser:
            doc = await upload_ready(mobile, browser)
            task = (await browser.post(PREFIX + '/workbench/tasks', json=plan([doc]))).json()['task']
            approved_id = str(uuid.uuid4())
            await start(browser, task, max_calls=2, request_id=approved_id)
            done = await settled(browser, task['id'])
            assert done['status'] == 'completed'
            assert all(step['status'] == 'completed' for step in done['steps'])
            assert done['run_remaining'] == 0 and mobile.model.await_count == 2
            detail = await browser.get(PREFIX + '/workbench/task/artifact', params={
                'session_id': SID, 'task_id': task['id'], 'version': 1})
            assert detail.status_code == 200, detail.text
            artifact = detail.json()['artifact']
            assert artifact['available'] and 'Supply' in artifact['content'] and 'Conditions' in artifact['content']
            assert done['artifact_versions'][0]['version'] == 1
            assert '已完成步骤的分析线索' in mobile.model.await_args_list[1].args[0]
            # A duplicated approved submission cannot consume a fresh attempt.
            replay = await browser.post(PREFIX + '/workbench/tasks/run', json={
                'session_id': SID, 'task_id': task['id'], 'expected_revision': task['revision'],
                'request_id': approved_id, 'max_calls': 2})
            assert replay.status_code == 200, replay.text
            await asyncio.sleep(.01)
            assert mobile.model.await_count == 2
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_budget_pause_then_restart_continue_without_reasking_completed_work(mobile, monkeypatch):
    try:
        async with mobile_client(mobile) as browser:
            doc = await upload_ready(mobile, browser)
            task = (await browser.post(PREFIX + '/workbench/tasks', json=plan([doc]))).json()['task']
            await start(browser, task, max_calls=1)
            paused = await settled(browser, task['id'])
            assert paused['status'] == 'paused' and mobile.model.await_count == 1
            first = next(step for step in paused['steps'] if step['id'] == 'q1')
            await project_tasks.shutdown_project_tasks()
            monkeypatch.setattr(mobile.api, '_query_boot_id', str(uuid.uuid4()))
            current = (await browser.get(PREFIX + '/workbench/task', params={
                'session_id': SID, 'task_id': task['id']})).json()['task']
            ready = (await browser.post(PREFIX + '/workbench/tasks/control', json={
                'session_id': SID, 'task_id': task['id'], 'expected_revision': current['revision'],
                'action': 'resume'})).json()['task']
            await start(browser, ready, max_calls=1)
            done = await settled(browser, task['id'])
            assert done['status'] == 'completed' and mobile.model.await_count == 2
            kept = next(step for step in done['steps'] if step['id'] == 'q1')
            assert kept['artifact_version'] == first['artifact_version'] == 1
            assert kept['request_id'] == first['request_id']
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_unknown_attempt_is_not_automatically_replayed(mobile):
    try:
        mobile.model.side_effect = asyncio.TimeoutError
        async with mobile_client(mobile) as browser:
            doc = await upload_ready(mobile, browser)
            task = (await browser.post(PREFIX + '/workbench/tasks', json=plan([doc]))).json()['task']
            await start(browser, task, max_calls=2)
            blocked = await settled(browser, task['id'])
            assert blocked['status'] == 'blocked'
            assert next(step for step in blocked['steps'] if step['id'] == 'q1')['reason'] == 'provider_outcome_unknown'
            for _ in range(3):
                await browser.get(PREFIX + '/workbench/task', params={'session_id': SID, 'task_id': task['id']})
            refused = await browser.post(PREFIX + '/workbench/tasks/run', json={
                'session_id': SID, 'task_id': task['id'], 'expected_revision': blocked['revision'],
                'request_id': str(uuid.uuid4()), 'max_calls': 2})
            assert refused.status_code == 409
            assert mobile.model.await_count == 1
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_source_withdrawal_hides_current_and_historical_report(mobile):
    try:
        async with mobile_client(mobile) as browser:
            doc = await upload_ready(mobile, browser)
            task = (await browser.post(PREFIX + '/workbench/tasks', json=plan([doc]))).json()['task']
            await start(browser, task, max_calls=2)
            done = await settled(browser, task['id'])
            assert done['status'] == 'completed'
            await browser.post(PREFIX + '/research/documents/delete', json={'session_id': SID, 'document_id': doc})
            response = await browser.get(PREFIX + '/workbench/task/artifact', params={
                'session_id': SID, 'task_id': task['id'], 'version': 1})
            assert response.status_code == 200, response.text
            result = response.json()['artifact']
            assert not result['available'] and result['content'] is None and result['references'] is None
            assert 'Renewable supply reduced' not in response.text
            assert mobile.model.await_count == 2
            database = mobile.manager.base_storage_path / 'project_tasks.db'
            assert b'PRIVATE_SOURCE_QUOTE_TOKEN_ONLY' not in database.read_bytes()
    finally:
        await project_tasks.shutdown_project_tasks()


@pytest.mark.parametrize('changes', [{'max_calls': True}, {'max_calls': 7}, {'expected_revision': True},
                                    {'request_id': 'bad'}, {'user_id': 'other'}])
async def test_execution_controls_are_strict_and_invalid_requests_never_call_model(mobile, changes):
    try:
        async with mobile_client(mobile) as browser:
            result = await browser.post(PREFIX + '/workbench/tasks/run', json={
                'session_id': SID, 'task_id': str(uuid.uuid4()), 'expected_revision': 1,
                'request_id': str(uuid.uuid4()), 'max_calls': 1, **changes})
            assert result.status_code == 422
        mobile.model.assert_not_awaited()
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_client_omitted_title_uses_server_default_and_budget_rejects_before_acceptance(mobile):
    try:
        async with mobile_client(mobile) as browser:
            body = plan(); body['questions'][0].pop('title')
            result = await browser.post(PREFIX + '/workbench/tasks', json=body)
            assert result.status_code == 200, result.text
            assert result.json()['task']['steps'][1]['title']
            body['goal'] = '😀' * 1200; body['constraints'] = '😀' * 1200
            too_big = await browser.post(PREFIX + '/workbench/tasks', json=body)
            assert too_big.status_code == 422
        mobile.model.assert_not_awaited()
    finally:
        await project_tasks.shutdown_project_tasks()
