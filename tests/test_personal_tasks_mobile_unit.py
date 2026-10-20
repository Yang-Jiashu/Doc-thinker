"""Actual ASGI -> task bridge -> ResearchService -> durable stores, no provider."""
import asyncio
import uuid

import pytest

from tests.test_research_mobile_unit import (
    mobile as research_mobile, mobile_client, upload_ready, SID, OTHER_SID, PREFIX,
)
from docthinker.server.routers import project_tasks


@pytest.fixture
def mobile(research_mobile, monkeypatch):
    research_mobile.manager.get_session.side_effect = lambda sid: {
        'id': sid, 'instance_id': 'instance-' + sid[1:]} if sid in research_mobile.manager.histories else None
    monkeypatch.setenv('DOCTHINKER_RESEARCH_SEMANTIC', 'off')
    research_mobile.service.context_assistance_enabled = False
    research_mobile.model.return_value = '可直接核对的私人草稿；缺少信息待确认，未发送、未付款、未修改文件。'
    return research_mobile


def plan(memory_ids=(), *, domain='家庭预约跟进', chain=False):
    rows = [{'id': 'draft', 'title': '待核对草稿', 'question': domain + '，给我一个不执行外部操作的安排。',
             'input_kind': 'personal', 'memory_ids': list(memory_ids)}]
    if chain:
        rows.append({'id': 'followup', 'title': '后续安排', 'question': '根据前步草稿列出下一步待确认事项。',
                     'input_kind': 'personal', 'memory_ids': list(memory_ids), 'depends_on': ['draft']})
    return {'session_id': SID, 'goal': domain, 'constraints': '不发送、不付款、不修改文件；操作须逐次确认。', 'questions': rows}


async def create(browser, payload):
    result = await browser.post(PREFIX + '/workbench/tasks', json=payload)
    assert result.status_code == 200, result.text
    return result.json()['task']


async def start(browser, task, *, calls=6, rid=None):
    response = await browser.post(PREFIX + '/workbench/tasks/run', json={
        'session_id': SID, 'task_id': task['id'], 'expected_revision': task['revision'],
        'request_id': rid or str(uuid.uuid4()), 'max_calls': calls})
    assert response.status_code == 200, response.text
    return response.json()['task']


async def settled(browser, task_id):
    for _ in range(200):
        response = await browser.get(PREFIX + '/workbench/task', params={'session_id': SID, 'task_id': task_id})
        assert response.status_code == 200, response.text
        task = response.json()['task']
        if task['status'] in {'completed', 'blocked', 'stale', 'paused', 'cancelled'}:
            return task
        await asyncio.sleep(.01)
    raise AssertionError('Personal task made no durable progress')


async def memory(mobile, browser, text, *, sid=SID):
    response = await browser.post(PREFIX + '/research/memory', json={
        'session_id': sid, 'kind': 'decision', 'text': text})
    assert response.status_code == 200, response.text
    return response.json()['memory']


async def artifact(browser, task, *, step_id='report'):
    result = await browser.get(PREFIX + '/workbench/task/artifact', params={
        'session_id': SID, 'task_id': task['id'], 'step_id': step_id, 'version': 1})
    assert result.status_code == 200, result.text
    return result.json()['artifact']


@pytest.mark.parametrize('domain', ['家庭预约跟进', '客户交接安排'])
async def test_native_personal_instruction_only_task_needs_no_upload_and_delivers_draft(mobile, domain):
    try:
        async with mobile_client(mobile) as browser:
            task = await create(browser, plan(domain=domain))
            mobile.model.assert_not_awaited()
            assert mobile.service.store.list_documents(SID) == []
            await start(browser, task, calls=1)
            done = await settled(browser, task['id'])
            assert done['status'] == 'completed'
            assert mobile.model.await_count == 1
            result = await artifact(browser, task)
            assert result['available'] and '可直接核对的私人草稿' in result['content']
            assert result.get('acceptance_status') == 'review_required'
            assert result.get('goal_verified') is False
            assert '不是文档证据' in result['content']
            assert '个人任务' in mobile.model.await_args.args[0]
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_native_personal_task_only_injects_selected_note_not_other_memory_or_document(mobile):
    try:
        async with mobile_client(mobile) as browser:
            selected = await memory(mobile, browser, '已确认家庭预算4827元，不得自动付款。')
            await memory(mobile, browser, '未选择的敏感私事 UNSELECTED_SECRET_749。')
            await upload_ready(mobile, browser, text='Unrelated client document UNSELECTED_DOC_158.')
            task = await create(browser, plan([selected['id']]))
            await start(browser, task, calls=1)
            done = await settled(browser, task['id'])
            assert done['status'] == 'completed' and mobile.model.await_count == 1
            prompt = mobile.model.await_args.args[0]
            assert selected['text'] in prompt
            assert 'UNSELECTED_SECRET_749' not in prompt and 'UNSELECTED_DOC_158' not in prompt
            step = next(row for row in done['steps'] if row['id'] == 'draft')
            assert step['input_snapshot'] == {} and step['context_snapshot'] == {selected['id']: 1}
            detail = await artifact(browser, task, step_id='draft')
            assert detail['references']['context_refs'] == {selected['id']: 1}
            assert detail['sources'] == [] and detail['references']['source_ids'] == []
    finally:
        await project_tasks.shutdown_project_tasks()


@pytest.mark.parametrize('origin', ['foreign', 'deleted'])
async def test_unavailable_selected_note_cannot_authorize_a_native_personal_call(mobile, origin):
    try:
        async with mobile_client(mobile) as browser:
            note = await memory(mobile, browser, 'Private selected client deadline.', sid=OTHER_SID if origin == 'foreign' else SID)
            if origin == 'deleted':
                mobile.service.store.delete_memory(SID, note['id'], expected_revision=1)
            task = await create(browser, plan([note['id']]))
            response = await browser.post(PREFIX + '/workbench/tasks/run', json={
                'session_id': SID, 'task_id': task['id'], 'expected_revision': task['revision'],
                'request_id': str(uuid.uuid4()), 'max_calls': 1})
            assert response.status_code in {404, 409, 422}, response.text
            mobile.model.assert_not_awaited()
    finally:
        await project_tasks.shutdown_project_tasks()


@pytest.mark.parametrize('mutation', ['delete', 'correct'])
async def test_changed_selected_note_hides_native_report_and_step_after_completion(mobile, mutation):
    try:
        async with mobile_client(mobile) as browser:
            note = await memory(mobile, browser, '家庭预算4827元，私人信息不得外发。')
            mobile.model.return_value = '私人草稿采用4827元预算；未付款。'
            task = await create(browser, plan([note['id']]))
            await start(browser, task, calls=1)
            assert (await settled(browser, task['id']))['status'] == 'completed'
            assert (await artifact(browser, task))['available']
            if mutation == 'delete':
                response = await browser.post(PREFIX + '/research/memory/delete', json={
                    'session_id': SID, 'memory_id': note['id'], 'expected_revision': 1})
            else:
                response = await browser.post(PREFIX + '/research/memory', json={
                    'session_id': SID, 'memory_id': note['id'], 'expected_revision': 1,
                    'kind': 'decision', 'text': '家庭预算更正为4300元，仍不得外发。'})
            assert response.status_code == 200
            for key in ('report', 'draft'):
                value = await artifact(browser, task, step_id=key)
                assert not value['available'] and value['content'] is None and value['references'] is None
                assert '4827' not in str(value)
            assert mobile.model.await_count == 1
    finally:
        await project_tasks.shutdown_project_tasks()


@pytest.mark.parametrize('mutation', ['correct', 'delete'])
async def test_native_inflight_memory_change_cannot_publish_or_repeat_old_answer(mobile, mutation):
    try:
        async with mobile_client(mobile) as browser:
            note = await memory(mobile, browser, '客户确认截止周五；尚未发送。')
            async def mutate(*args, **kwargs):
                if mutation == 'correct':
                    mobile.service.store.update_memory(SID, note['id'], expected_revision=1, text='客户更正截止周四；尚未发送。')
                else:
                    mobile.service.store.delete_memory(SID, note['id'], expected_revision=1)
                return '旧截止周五的草稿，不能发布。'
            mobile.model.side_effect = mutate
            task = await create(browser, plan([note['id']], domain='客户截止跟进'))
            await start(browser, task, calls=1)
            done = await settled(browser, task['id'])
            assert done['status'] == 'stale'
            assert next(row for row in done['steps'] if row['id'] == 'draft')['artifact_version'] == 0
            assert mobile.model.await_count == 1
            for _ in range(2):
                await browser.get(PREFIX + '/workbench/task', params={'session_id': SID, 'task_id': task['id']})
            assert mobile.model.await_count == 1
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_native_personal_pause_restart_keeps_completed_step_and_explicit_scope(mobile, monkeypatch):
    try:
        async with mobile_client(mobile) as browser:
            note = await memory(mobile, browser, '交接资料只能给内部同事看，外发需要另行同意。')
            task = await create(browser, plan([note['id']], domain='同事交接', chain=True))
            await start(browser, task, calls=1)
            paused = await settled(browser, task['id'])
            assert paused['status'] == 'paused' and mobile.model.await_count == 1
            old = next(row for row in paused['steps'] if row['id'] == 'draft')
            await project_tasks.shutdown_project_tasks()
            monkeypatch.setattr(mobile.api, '_query_boot_id', str(uuid.uuid4()))
            current = (await browser.get(PREFIX + '/workbench/task', params={'session_id': SID, 'task_id': task['id']})).json()['task']
            ready = await browser.post(PREFIX + '/workbench/tasks/control', json={
                'session_id': SID, 'task_id': task['id'], 'expected_revision': current['revision'], 'action': 'resume'})
            assert ready.status_code == 200, ready.text
            await start(browser, ready.json()['task'], calls=1)
            done = await settled(browser, task['id'])
            assert done['status'] == 'completed' and mobile.model.await_count == 2
            kept = next(row for row in done['steps'] if row['id'] == 'draft')
            assert kept['request_id'] == old['request_id'] and kept['artifact_version'] == 1
            assert note['text'] in mobile.model.await_args.args[0]
            assert '已完成步骤' in mobile.model.await_args.args[0]
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_native_unknown_personal_outcome_is_not_repeated_by_status_or_new_approval(mobile):
    try:
        mobile.model.side_effect = asyncio.TimeoutError
        async with mobile_client(mobile) as browser:
            task = await create(browser, plan())
            await start(browser, task, calls=1)
            blocked = await settled(browser, task['id'])
            assert blocked['status'] == 'blocked'
            for _ in range(3):
                await browser.get(PREFIX + '/workbench/task', params={'session_id': SID, 'task_id': task['id']})
            refused = await browser.post(PREFIX + '/workbench/tasks/run', json={
                'session_id': SID, 'task_id': task['id'], 'expected_revision': blocked['revision'],
                'request_id': str(uuid.uuid4()), 'max_calls': 1})
            assert refused.status_code == 409 and mobile.model.await_count == 1
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_native_personal_scope_cannot_be_mixed_with_document_permission(mobile):
    try:
        async with mobile_client(mobile) as browser:
            payload = plan()
            payload['questions'][0]['document_ids'] = [str(uuid.uuid4())]
            response = await browser.post(PREFIX + '/workbench/tasks', json=payload)
            assert response.status_code == 422
            mobile.model.assert_not_awaited()
    finally:
        await project_tasks.shutdown_project_tasks()
