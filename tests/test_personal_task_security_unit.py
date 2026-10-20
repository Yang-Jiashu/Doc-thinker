"""Personal task TOCTOU checks through real ASGI, service and durable stores.

Only generation is synthetic; context readers and memory withdrawal remain the
actual production boundary. Timing hooks schedule a real concurrent mutation,
never manufacture an authorization receipt or replace a reader's result.
"""
import pytest
import uuid

from tests.test_personal_tasks_mobile_unit import (
    mobile, research_mobile, mobile_client, memory, create, start, settled, artifact, plan, SID, PREFIX,
)
from docthinker.server.routers import project_tasks


async def test_report_revalidates_all_personal_contexts_after_last_await(mobile):
    secret = 'WITHDRAWN_FIRST_SECTION_PRIVATE_8421'
    try:
        async with mobile_client(mobile) as browser:
            first = await memory(mobile, browser, '家庭预算已确认4827元。')
            second = await memory(mobile, browser, '客户交接只发内部人员。')
            mobile.model.side_effect = [secret, '第二节的内部交接草稿。']
            task = await create(browser, {
                'session_id': SID, 'goal': '家庭和工作安排', 'constraints': '不执行外部操作。',
                'questions': [
                    {'id': 'family', 'question': '起草家庭预算安排。',
                     'input_kind': 'personal', 'memory_ids': [first['id']]},
                    {'id': 'work', 'question': '起草内部客户交接。',
                     'input_kind': 'personal', 'memory_ids': [second['id']]},
                ],
            })
            await start(browser, task, calls=2)
            assert (await settled(browser, task['id']))['status'] == 'completed'
            assert mobile.model.await_count == 2
            assert secret in (await artifact(browser, task))['content']

            _, runtime, _ = await project_tasks._context(SID)
            original_render = runtime.render_report
            original_read = runtime.read_context
            state = {'rendering': False, 'second_reads': 0, 'withdrawn': False}

            async def rendering(*args, **kwargs):
                state['rendering'] = True
                try:
                    return await original_render(*args, **kwargs)
                finally:
                    state['rendering'] = False

            async def read_context(memory_ids=None):
                value = await original_read(memory_ids)
                if state['rendering'] and memory_ids == [second['id']]:
                    state['second_reads'] += 1
                    # Each section has a pre-hydrate and post-hydrate check;
                    # this third read is the last section's final check. The
                    # first section has already passed its final check.
                    if state['second_reads'] == 3:
                        mobile.service.store.delete_memory(
                            SID, first['id'], expected_revision=1)
                        state['withdrawn'] = True
                return value

            runtime.render_report = rendering
            runtime.read_context = read_context
            result = await artifact(browser, task)
            assert state['withdrawn'] and state['second_reads'] == 3
            assert mobile.service.store.get_memory(SID, first['id'])['redacted']
            assert not result['available'] or secret not in (result['content'] or '')
            assert mobile.model.await_count == 2
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_oversized_explicit_memory_is_not_an_unknown_accepted_model_outcome(mobile):
    try:
        async with mobile_client(mobile) as browser:
            # Valid saved memory (1540 < 2000 characters), but above the
            # existing 3000-byte whole-memory retrieval packet allowance.
            note = await memory(mobile, browser, '预算不得外发。' * 220)
            assert len(note['text']) < 2000
            assert len(note['text'].encode()) > 3000
            task = await create(browser, plan([note['id']]))
            await start(browser, task, calls=1)
            blocked = await settled(browser, task['id'])
            mobile.model.assert_not_awaited()
            draft = next(row for row in blocked['steps'] if row['id'] == 'draft')
            # This is a known deterministic host rejection, not an uncertain
            # provider attempt requiring impossible paid-outcome reconciliation.
            assert draft['reason'] != 'provider_outcome_unknown'
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_accepted_journal_pre_model_fit_error_is_known_failure(mobile, monkeypatch):
    from docthinker.server import research_service
    original_fit = research_service.fit_research_prompt
    fitted = []

    def fail_after_real_fit(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        assert result['messages']
        fitted.append(True)
        raise ValueError('Synthetic deterministic pre-dispatch input rejection')

    monkeypatch.setattr(research_service, 'fit_research_prompt', fail_after_real_fit)
    try:
        async with mobile_client(mobile) as browser:
            note = await memory(mobile, browser, '客户交接只给内部同事。')
            task = await create(browser, plan([note['id']], domain='客户交接'))
            await start(browser, task, calls=1)
            blocked = await settled(browser, task['id'])
            draft = next(row for row in blocked['steps'] if row['id'] == 'draft')
            receipt = mobile.api._query_journal().read(SID, draft['request_id'])
            assert fitted == [True]
            mobile.model.assert_not_awaited()
            assert draft['reason'] == 'provider_failed'
            assert receipt['request_status'] == 'failed'
            assert receipt['request_error']['code'] == 'inputs_unavailable'
            # Status reconciliation may settle a known local failure; it must
            # never reinterpret it as a paid result or dispatch a repair call.
            for _ in range(2):
                response = await browser.get(PREFIX + '/workbench/task', params={
                    'session_id': SID, 'task_id': task['id']})
                assert response.status_code == 200
                current = next(row for row in response.json()['task']['steps'] if row['id'] == 'draft')
                assert current['reason'] != 'provider_outcome_unknown'
            mobile.model.assert_not_awaited()
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_configured_model_value_error_remains_unknown_without_repeat(mobile):
    reached = []

    async def configured_model(*args, **kwargs):
        reached.append(True)
        raise ValueError('Synthetic configured-model error after actual dispatch')

    mobile.model.side_effect = configured_model
    try:
        async with mobile_client(mobile) as browser:
            note = await memory(mobile, browser, '家庭预算4827元；付款另行确认。')
            task = await create(browser, plan([note['id']]))
            await start(browser, task, calls=1)
            blocked = await settled(browser, task['id'])
            draft = next(row for row in blocked['steps'] if row['id'] == 'draft')
            assert reached == [True] and mobile.model.await_count == 1
            assert draft['reason'] == 'provider_outcome_unknown'
            assert mobile.api._query_journal().read(SID, draft['request_id'])['request_status'] == 'unknown'
            # Even a now-healthy model cannot be called by status/replay or by
            # a fresh approval while the old accepted outcome is unresolved.
            mobile.model.side_effect = None
            mobile.model.return_value = '不得被自动生成的替代草稿。'
            for _ in range(2):
                response = await browser.get(PREFIX + '/workbench/task', params={
                    'session_id': SID, 'task_id': task['id']})
                assert response.status_code == 200
                blocked = response.json()['task']
            refused = await browser.post(PREFIX + '/workbench/tasks/run', json={
                'session_id': SID, 'task_id': task['id'], 'expected_revision': blocked['revision'],
                'request_id': str(uuid.uuid4()), 'max_calls': 1})
            assert refused.status_code == 409
            assert reached == [True] and mobile.model.await_count == 1
            assert mobile.api._query_journal().read(SID, draft['request_id'])['request_status'] == 'unknown'
    finally:
        await project_tasks.shutdown_project_tasks()
