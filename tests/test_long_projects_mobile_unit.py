"""Native long-project HTTP -> existing task executor -> real service/stores.

Only the answer writer is deterministic. These are execution, persistence,
scope, handoff and withdrawal checks, not claims about model answer quality.
"""
import asyncio
import json
import sqlite3
import uuid

import pytest

from tests.test_personal_memory_transport_unit import (
    transport, base_mobile, mobile_client, restart, PREFIX,
)
from tests.test_research_mobile_unit import upload_ready
from docthinker.server.routers import project_tasks


def uid():
    return str(uuid.uuid4())


def personal(key='draft', *, memories=(), question='Draft a private next action, do not send it.'):
    return {'id': key, 'question': question, 'input_kind': 'personal', 'memory_ids': list(memories)}


async def new_project(browser, sid, *, request_id=None):
    result = await browser.post(PREFIX + '/workbench/long-projects', json={
        'session_id': sid, 'request_id': request_id or uid(), 'goal': 'Keep household and work decisions coherent over many stages.',
        'acceptance_criteria': ['Deliver a usable draft.', 'Disclose missing evidence and do not execute external actions.']})
    assert result.status_code == 200, result.text
    return result.json()['project']


async def current_project(browser, sid, project):
    response = await browser.get(PREFIX + '/workbench/long-project', params={'session_id': sid, 'project_id': project['id']})
    assert response.status_code == 200, response.text
    return response.json()['project']


async def phase(browser, sid, project, *, rows=None, parents=None, title='Next phase'):
    current = await current_project(browser, sid, project)
    response = await browser.post(PREFIX + '/workbench/long-project/phases', json={
        'session_id': sid, 'project_id': project['id'], 'expected_revision': current['revision'],
        'request_id': uid(), 'title': title, 'acceptance_criteria': ['Draft is ready for user review.'],
        'questions': [personal()] if rows is None else rows, 'parent_refs': parents or [],
        'constraints': 'No sending, payment or file changes without confirmation.'})
    assert response.status_code == 200, response.text
    return response.json()


async def execute(browser, sid, task, calls=6):
    started = await browser.post(PREFIX + '/workbench/tasks/run', json={
        'session_id': sid, 'task_id': task['id'], 'expected_revision': task['revision'],
        'request_id': uid(), 'max_calls': calls})
    assert started.status_code == 200, started.text
    for _ in range(250):
        response = await browser.get(PREFIX + '/workbench/task', params={'session_id': sid, 'task_id': task['id']})
        assert response.status_code == 200, response.text
        current = response.json()['task']
        if current['status'] in {'completed', 'blocked', 'stale', 'paused', 'cancelled'}:
            return current
        await asyncio.sleep(.01)
    raise AssertionError('Native phase did not settle')


async def artifact(browser, sid, task_id, step_id='report'):
    response = await browser.get(PREFIX + '/workbench/task/artifact', params={
        'session_id': sid, 'task_id': task_id, 'step_id': step_id, 'version': 1})
    assert response.status_code == 200, response.text
    return response.json()['artifact']


async def save_memory(browser, sid, text):
    response = await browser.post(PREFIX + '/research/memory', json={'session_id': sid, 'kind': 'decision', 'text': text})
    assert response.status_code == 200, response.text
    return response.json()['memory']


async def test_native_thirty_phases_survive_three_restarts_and_keyset_directory(transport):
    mobile, sid = transport.mobile, transport.sid
    mobile.model.return_value = 'A usable next-action draft. No external operation was executed.'
    try:
        async with mobile_client(mobile) as browser:
            project = await new_project(browser, sid)
            ids, previous = [], None
            for index in range(30):
                if index in {8, 17, 25}:
                    await project_tasks.shutdown_project_tasks()
                    await restart(transport)
                    mobile.api._query_boot_id = 'native-phase-restart-' + str(index)
                refs = [{'phase_id': previous['id'], 'report_version': 1, 'step_ids': ['draft']}] if previous else []
                added = await phase(browser, sid, project, parents=refs, title='Phase ' + str(index + 1))
                done = await execute(browser, sid, added['task'], calls=1)
                assert done['status'] == 'completed', {'phase_index': index, 'task': json.dumps(done, sort_keys=True)}
                ids.append(added['phase']['id'])
                previous = added['phase']
            assert mobile.model.await_count == 30
            seen, cursor = [], None
            while True:
                response = await browser.get(PREFIX + '/workbench/long-project/phases', params={
                    'session_id': sid, 'project_id': project['id'], 'limit': 7, **({'cursor': cursor} if cursor else {})})
                assert response.status_code == 200, response.text
                page = response.json()
                seen.extend(row['id'] for row in page['items'])
                assert all(row['task_status'] == 'completed' and row['report_version'] == 1 for row in page['items'])
                cursor = page['next_cursor']
                if cursor is None:
                    break
            assert seen == ids and (await current_project(browser, sid, project))['phase_count'] == 30
            assert (await artifact(browser, sid, added['task']['id']))['available']
            assert mobile.model.await_count == 30
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_native_phase_read_under_real_writer_lock_denies_export_without_withdrawing_work(transport):
    mobile, sid = transport.mobile, transport.sid
    mobile.model.return_value = 'A private current draft. No external operation was executed.'
    try:
        async with mobile_client(mobile) as browser:
            project = await new_project(browser, sid)
            rows = [personal()]
            first = await phase(browser, sid, project, rows=rows)
            assert (await execute(browser, sid, first['task'], calls=1))['status'] == 'completed'
            second = await phase(browser, sid, project, rows=rows,
                parents=[{'phase_id': first['phase']['id'], 'report_version': 1, 'step_ids': ['draft']}])
            done = await execute(browser, sid, second['task'], calls=1)
            assert done['status'] == 'completed'
            _, runtime, _ = await project_tasks._context(sid)
            before = runtime.store.get_task(second['task']['id'])
            with sqlite3.connect(mobile.service.store.path) as writer:
                writer.execute('BEGIN EXCLUSIVE')
                # Read-only status/export cannot publish a positive result or
                # permanently withdraw artifacts on an unavailable snapshot.
                for route, params in [('/workbench/task', {'task_id': second['task']['id']}),
                                      ('/workbench/task/artifact', {'task_id': second['task']['id'],
                                                                  'step_id': 'report', 'version': 1})]:
                    response = await browser.get(PREFIX + route, params={'session_id': sid, **params})
                    assert response.status_code == 409, response.text
                    assert 'private current draft' not in response.text
                assert mobile.model.await_count == 2
                assert runtime.store.get_task(second['task']['id']) == before
                writer.rollback()
            current = await browser.get(PREFIX + '/workbench/task', params={'session_id': sid, 'task_id': second['task']['id']})
            assert current.status_code == 200 and current.json()['task']['status'] == 'completed', current.text
            assert (await artifact(browser, sid, second['task']['id']))['available']
            assert mobile.model.await_count == 2
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_native_selected_parent_is_really_handed_off_with_existing_byte_budgets(transport):
    mobile, sid = transport.mobile, transport.sid
    selected = 'SELECTED_PARENT_4827 budget deadline requires approval.'
    excluded = 'UNSELECTED_PARENT_PRIVATE_749'
    mobile.model.side_effect = [selected + ' A budget review is needed before execution.' * 150, excluded, 'A usable follow-up draft.']
    try:
        async with mobile_client(mobile) as browser:
            project = await new_project(browser, sid)
            first = await phase(browser, sid, project, rows=[personal('selected'), personal('excluded')])
            assert (await execute(browser, sid, first['task'], calls=2))['status'] == 'completed'
            second = await phase(browser, sid, project, rows=[personal(question='Use SELECTED_PARENT_4827 budget deadline to draft a follow-up.')],
                parents=[{'phase_id': first['phase']['id'], 'report_version': 1, 'step_ids': ['selected']}])
            done = await execute(browser, sid, second['task'], calls=1)
            assert done['status'] == 'completed', done
            prompt = mobile.model.await_args.args[0]
            assert selected in prompt and excluded not in prompt
            assert '不是原文证据' in prompt
            project_data = json.JSONDecoder().raw_decode(prompt.split('PROJECT_TASK_DATA_JSON:\n', 1)[1])[0]['context']
            start = project_data.index('[{"step_id"')
            packet = json.JSONDecoder().raw_decode(project_data[start:])[0]
            assert len(json.dumps(packet, ensure_ascii=False, separators=(',', ':')).encode()) <= 2000
            assert len(prompt.encode()) + len(mobile.model.await_args.kwargs['system_prompt'].encode()) + len('systemuser') <= 24576
            step = next(row for row in done['steps'] if row['id'] == 'draft')
            assert step['phase_handoff']['required'] == step['phase_handoff']['used'] == 1
            assert step['phase_handoff']['omitted'] == 0
            assert mobile.model.await_count == 3
    finally:
        await project_tasks.shutdown_project_tasks()


@pytest.mark.parametrize('origin', ['memory', 'document'])
async def test_native_parent_withdrawal_hides_child_history_step_and_report_without_more_calls(transport, origin):
    mobile, sid = transport.mobile, transport.sid
    secret = 'PRIVATE_DERIVED_CHILD_3826'
    # The pre-existing ordinary source-answer history contract can retain an
    # old parent body with a withdrawal warning. Track the phase-derived child
    # separately: its parent lineage must fail closed, including UI cards.
    mobile.model.side_effect = ['PARENT_ONLY_1749 parent draft.', secret + ' derived child draft.']
    try:
        async with mobile_client(mobile) as browser:
            project = await new_project(browser, sid)
            if origin == 'memory':
                note = await save_memory(browser, sid, 'Budget decision 4827; never pay automatically.')
                rows = [personal(memories=[note['id']])]
            else:
                doc = await upload_ready(mobile, browser, sid=sid,
                    text='Budget deadline is Friday only after client approval.', filename='client-note.txt')
                rows = [{'id': 'draft', 'question': 'Explain the budget deadline and its approval condition.', 'document_ids': [doc]}]
            first = await phase(browser, sid, project, rows=rows)
            assert (await execute(browser, sid, first['task'], calls=1))['status'] == 'completed'
            second = await phase(browser, sid, project, rows=rows,
                parents=[{'phase_id': first['phase']['id'], 'report_version': 1, 'step_ids': ['draft']}])
            assert (await execute(browser, sid, second['task'], calls=1))['status'] == 'completed'
            assert secret in (await artifact(browser, sid, second['task']['id']))['content']
            if origin == 'memory':
                removed = await browser.post(PREFIX + '/research/memory/delete', json={
                    'session_id': sid, 'memory_id': note['id'], 'expected_revision': 1})
            else:
                removed = await browser.post(PREFIX + '/research/documents/delete', json={'session_id': sid, 'document_id': doc})
            assert removed.status_code == 200
            await project_tasks.shutdown_project_tasks()
            await restart(transport)
            for key in ['draft', 'report']:
                current = await artifact(browser, sid, second['task']['id'], key)
                assert not current['available'] and secret not in json.dumps(current)
            history = await browser.get(PREFIX + '/history', params={'session_id': sid})
            assert history.status_code == 200 and secret not in history.text
            assert mobile.model.await_count == 2
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_native_unselected_parent_edit_keeps_child_but_selected_edit_withdraws_it(transport):
    mobile, sid = transport.mobile, transport.sid
    child_secret = 'CHILD_DERIVED_PRIVATE_2913'
    mobile.model.side_effect = ['Selected schedule draft.', 'Other branch draft.', child_secret]
    try:
        async with mobile_client(mobile) as browser:
            project = await new_project(browser, sid)
            rows = [personal('selected', question='Draft the household schedule.'), personal('other', question='Draft a work agenda.')]
            first = await phase(browser, sid, project, rows=rows)
            assert (await execute(browser, sid, first['task'], calls=2))['status'] == 'completed'
            second = await phase(browser, sid, project,
                parents=[{'phase_id': first['phase']['id'], 'report_version': 1, 'step_ids': ['selected']}])
            assert (await execute(browser, sid, second['task'], calls=1))['status'] == 'completed'
            for changed_key, should_live in [('other', True), ('selected', False)]:
                prior = await browser.get(PREFIX + '/workbench/task', params={'session_id': sid, 'task_id': first['task']['id']})
                prior = prior.json()['task']
                rows = [{**row, 'question': row['question'] + ' Revised.'} if row['id'] == changed_key else row for row in rows]
                updated = await browser.post(PREFIX + '/workbench/tasks/update', json={
                    'session_id': sid, 'task_id': prior['id'], 'expected_revision': prior['revision'],
                    'goal': prior['goal'], 'constraints': prior['constraints'], 'questions': rows})
                assert updated.status_code == 200, updated.text
                result = await artifact(browser, sid, second['task']['id'])
                assert result['available'] is should_live
                if not should_live:
                    assert child_secret not in json.dumps(result)
                    history = await browser.get(PREFIX + '/history', params={'session_id': sid})
                    assert history.status_code == 200 and child_secret not in history.text
            assert mobile.model.await_count == 3
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_native_customer_review_binds_report_and_new_phase_stales_root_acceptance(transport):
    mobile, sid = transport.mobile, transport.sid
    mobile.model.return_value = 'Usable draft, still requires user review.'
    try:
        async with mobile_client(mobile) as browser:
            project = await new_project(browser, sid)
            first = await phase(browser, sid, project)
            assert (await execute(browser, sid, first['task'], calls=1))['status'] == 'completed'
            current = await current_project(browser, sid, project)
            reviewed = await browser.post(PREFIX + '/workbench/long-project/phases/review', json={
                'session_id': sid, 'project_id': project['id'], 'phase_id': first['phase']['id'],
                'expected_revision': current['revision'], 'request_id': uid(), 'report_version': 1, 'decision': 'accepted'})
            assert reviewed.status_code == 200, reviewed.text
            assert reviewed.json()['phase']['review_status'] == 'accepted'
            current = await current_project(browser, sid, project)
            result = await browser.post(PREFIX + '/workbench/long-project/review', json={
                'session_id': sid, 'project_id': project['id'], 'expected_revision': current['revision'],
                'request_id': uid(), 'phase_reports': [{'phase_id': first['phase']['id'], 'report_version': 1}], 'decision': 'accepted'})
            assert result.status_code == 200, result.text
            assert result.json()['project']['review_status'] == 'accepted'
            assert result.json()['project']['goal_verified'] is False
            await phase(browser, sid, project)
            assert (await current_project(browser, sid, project))['review_status'] == 'stale'
            assert mobile.model.await_count == 1
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_native_phase_request_cas_and_session_instance_scope_are_preserved(transport):
    mobile, sid = transport.mobile, transport.sid
    try:
        async with mobile_client(mobile) as browser:
            project = await new_project(browser, sid)
            body = {'session_id': sid, 'project_id': project['id'], 'expected_revision': project['revision'],
                'request_id': uid(), 'title': 'Explicit phase', 'acceptance_criteria': ['Needs user review.'],
                'questions': [personal()], 'parent_refs': []}
            first = await browser.post(PREFIX + '/workbench/long-project/phases', json=body)
            repeated = await browser.post(PREFIX + '/workbench/long-project/phases', json=body)
            assert first.status_code == repeated.status_code == 200
            assert first.json()['phase']['id'] == repeated.json()['phase']['id']
            assert first.json()['task']['id'] == repeated.json()['task']['id']
            for changed in [{'title': 'Changed request'}, {'request_id': uid()}]:
                rejected = await browser.post(PREFIX + '/workbench/long-project/phases', json={**body, **changed})
                assert rejected.status_code == 409
            other = transport.manager.create_session('Another private context')['id']
            denied = await browser.get(PREFIX + '/workbench/long-project', params={'session_id': other, 'project_id': project['id']})
            assert denied.status_code == 404
            assert transport.manager.delete_session(sid)
            replacement = transport.manager.create_session('Replacement instance')['id']
            assert replacement == sid
            denied = await browser.get(PREFIX + '/workbench/long-project', params={'session_id': sid, 'project_id': project['id']})
            assert denied.status_code == 404
            mobile.model.assert_not_awaited()
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_native_unknown_phase_cannot_be_bypassed_with_new_phase_or_restart(transport):
    mobile, sid = transport.mobile, transport.sid
    mobile.model.side_effect = asyncio.TimeoutError
    try:
        async with mobile_client(mobile) as browser:
            project = await new_project(browser, sid)
            first = await phase(browser, sid, project)
            blocked = await execute(browser, sid, first['task'], calls=1)
            assert blocked['status'] == 'blocked'
            assert next(row for row in blocked['steps'] if row['id'] == 'draft')['reason'] == 'provider_outcome_unknown'
            assert mobile.model.await_count == 1
            await project_tasks.shutdown_project_tasks()
            await restart(transport)
            mobile.model.side_effect = None
            current = await current_project(browser, sid, project)
            denied = await browser.post(PREFIX + '/workbench/long-project/phases', json={
                'session_id': sid, 'project_id': project['id'], 'expected_revision': current['revision'],
                'request_id': uid(), 'title': 'Cannot disguise a retry as a new phase',
                'acceptance_criteria': ['Review first.'], 'questions': [personal()], 'parent_refs': []})
            assert denied.status_code == 409, denied.text
            for _ in range(2):
                response = await browser.get(PREFIX + '/workbench/task', params={'session_id': sid, 'task_id': first['task']['id']})
                assert response.status_code == 200
            assert mobile.model.await_count == 1
            assert (await current_project(browser, sid, project))['phase_count'] == 1
    finally:
        await project_tasks.shutdown_project_tasks()
