"""Independent cross-domain software lifecycle acceptance, no real model/API.

The controlled model only exercises the actual ASGI/Harness/SQLite evidence
feedback path. These tests do not establish semantic or causal model quality.
"""
import asyncio
import json
import time

import pytest

from docthinker.server.routers import project_tasks
from tests.test_research_mobile_unit import (
    mobile as research_mobile, mobile_client, upload_ready, SID, PREFIX,
)
from tests.test_personal_tasks_mobile_unit import create, start, memory


DOMAINS = [
    pytest.param({
        'goal': 'Assess Orion handover delay and prepare a private internal draft.',
        'questions': ['Check Orion workload and critical path timestamps against staffing absences.',
                      'Draft Orion update with evidence and unresolved alternatives.'],
        'missing': 'Orion approved critical path timestamps and staffing absence roster original evidence',
        'initial': 'Orion added work is recorded. Critical path timestamps and staffing absence roster remain unavailable.',
        'fresh': 'Orion approved critical path timestamps show extra work delayed handover by two days. '
                 'The staffing absence roster records no absences and supplier outage overlapped no critical task.',
        'memory': 'The internal draft must stay private; the confirmed review date is Friday.',
    }, id='work-delivery'),
    pytest.param({
        'goal': 'Compare Maple family appointment options and prepare a private confirmation draft.',
        'questions': ['Check Maple booking timestamps against household availability and appointment confirmation.',
                      'Draft Maple appointment follow-up with confirmed and missing booking information.'],
        'missing': 'Maple original booking timestamps and household availability calendar confirmation evidence',
        'initial': 'Maple appointment was moved. Original booking timestamps and household availability calendar are unavailable.',
        'fresh': 'Maple original booking timestamps confirm the appointment moved from Friday to Saturday. '
                 'The household availability calendar confirms Saturday is free; no appointment has been booked by this agent.',
        'memory': 'The appointment follow-up must stay private; the confirmed family review date is Saturday.',
    }, id='family-appointment'),
]


async def terminal(browser, task_id):
    deadline = time.monotonic() + 15
    latest = None
    while time.monotonic() < deadline:
        response = await browser.get(PREFIX + '/workbench/task', params={'session_id': SID, 'task_id': task_id})
        assert response.status_code == 200, response.text
        latest = response.json()['task']
        if latest['status'] in {'completed', 'blocked', 'stale', 'paused', 'cancelled'}:
            return latest
        await asyncio.sleep(.025)
    summary = {'status': latest['status'], 'steps': [(row['id'], row['status']) for row in latest['steps']]}
    raise AssertionError(f'Task did not reach a durable terminal checkpoint: {summary}')


def private_manager(mobile):
    from docthinker.session_manager import SessionManager
    manager = SessionManager(str(mobile.root / '_system'), str(mobile.root), migrate_legacy=False)
    assert manager.create_session('Independent evidence lifecycle')['id'] == SID
    mobile.manager = manager
    mobile.state.session_manager = manager


def packet(prompt, marker):
    return json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1].lstrip())[0]


async def current_report(browser, task_id, version):
    response = await browser.get(PREFIX + '/workbench/task/artifact', params={
        'session_id': SID, 'task_id': task_id, 'step_id': 'report', 'version': version})
    assert response.status_code == 200, response.text
    return response.json()['artifact']


@pytest.mark.asyncio
@pytest.mark.parametrize('domain', DOMAINS)
async def test_missing_parent_work_repair_new_material_and_explicit_reapproval_can_finish(research_mobile, domain):
    mobile = research_mobile
    private_manager(mobile)
    prompts = []
    evidence_arrived = False

    async def controlled_response(prompt, **kwargs):
        prompts.append(prompt)
        if 'FROZEN_REVIEW_DATA_JSON:\n' in prompt:
            frozen = packet(prompt, 'FROZEN_REVIEW_DATA_JSON:\n')
            fresh = next(row for row in frozen['source_quotes'] if row['filename'] == 'new-original-record.txt')
            return json.dumps({'answer': 'Review metadata, not new prose.', 'task_assessment': {'requirements': [
                {'id': row['id'], 'verdict': 'addressed', 'answer_witness': {'whole_answer': True},
                 'evidence': [{'label': fresh['label']}]}
                for row in frozen['task_requirements'] if row['id'] in frozen['active_task_ids']]}})
        req = packet(prompt, 'TASK_REQUIREMENTS_DATA_JSON:\n')
        if not evidence_arrived:
            if len(prompts) == 2:
                assert '前序证据缺口的待查队列' in prompt and domain['missing'] in prompt
            return json.dumps({'answer': 'A private draft is possible, but original records remain missing [S1].',
                'task_assessment': {'requirements': [{'id': row['id'], 'verdict': 'unknown',
                    'missing_query': domain['missing']} for row in req['requirements']]}})
        quotes = packet(prompt, 'UNTRUSTED_RESEARCH_DATA_JSON:\n')['source_quotes']
        fresh = next(row for row in quotes if row['filename'] == 'new-original-record.txt')
        answer = 'Original records are now available; the private draft still needs user approval [' + fresh['label'] + '].'
        return json.dumps({'answer': answer, 'task_assessment': {'requirements': [
            {'id': row['id'], 'verdict': 'addressed', 'answer_witness': {'quote': answer, 'start': 0},
             'evidence': [{'label': fresh['label'], 'quote': fresh['content']}]}
            for row in req['requirements']]}})

    mobile.model.side_effect = controlled_response
    try:
        async with mobile_client(mobile) as browser:
            await upload_ready(mobile, browser, filename='initial-record.txt', text=domain['initial'])
            task = await create(browser, {'session_id': SID, 'goal': domain['goal'],
                'constraints': 'No sending, booking, payment or other external action; retain missing evidence.',
                'questions': [{'id': 'facts', 'question': domain['questions'][0]},
                              {'id': 'brief', 'question': domain['questions'][1], 'depends_on': ['facts']}]})
            await start(browser, task, calls=2)
            first = await terminal(browser, task['id'])
            assert first['status'] == 'completed' and mobile.model.await_count == 2
            assert first['evidence_progress']['coverage']['unresolved'] > 0
            preview = await browser.post(PREFIX + '/workbench/tasks/evidence/preview', json={
                'session_id': SID, 'task_id': task['id'], 'expected_revision': first['revision']})
            assert preview.status_code == 200, preview.text
            repair = preview.json()['repair']
            assert repair['changed_steps'] and repair['plan']['goal'] == domain['goal']
            additions = [row for row in repair['plan']['questions'] if row['id'] in repair['changed_steps']]
            refinements = [row for row in additions if row.get('refine_of')]
            assert len(refinements) == 2 and len(additions) == 4
            assert {row['refine_of'] for row in refinements} == {'facts', 'brief'}
            for row in refinements:
                original = next(item for item in repair['plan']['questions'] if item['id'] == row['refine_of'])
                assert row['question'] == original['question'] and row['depends_on'] == [original['id']]
                assert any(item.get('review_of') == row['id'] for item in additions)
            assert repair['execution_authorized'] is repair['automatic_execution'] is False
            assert mobile.model.await_count == 2  # preparing/declining a draft is not another grant
            applied = await browser.post(PREFIX + '/workbench/tasks/replan', json={
                'session_id': SID, 'task_id': task['id'], 'expected_revision': first['revision'],
                'proposal_sha256': repair['proposal']['proposal_sha256'], **repair['plan']})
            assert applied.status_code == 200, applied.text
            assert mobile.model.await_count == 2
            old = await current_report(browser, task['id'], 1)
            assert old['available'] is False
            fresh_document = await upload_ready(mobile, browser, filename='new-original-record.txt', text=domain['fresh'])
            evidence_arrived = True
            changed = applied.json()['task']
            refreshed = await browser.post(PREFIX + '/workbench/tasks/control', json={
                'session_id': SID, 'task_id': task['id'], 'expected_revision': changed['revision'], 'action': 'refresh'})
            assert refreshed.status_code == 200, refreshed.text
            assert mobile.model.await_count == 2  # source refresh and CAS save never dispatch a model
            # The reviewed six-node plan is visible. The refreshed originals
            # and two compose/refine + verify pairs need six explicit calls.
            await start(browser, refreshed.json()['task'], calls=6)
            done = await terminal(browser, task['id'])
            assert done['status'] == 'completed' and mobile.model.await_count == 8
            progress = done['evidence_progress']
            assert progress['coverage']['ready_for_review'] == progress['coverage']['total'] == 3
            assert progress['next_action'] == 'review' and progress['goal_verified'] is False
            assert progress['semantic_verified'] is False
            report = await current_report(browser, task['id'], 2)
            assert report['available'] and report['delivery_status'] == 'ready_for_user_review'
            assert report['acceptance_status'] == 'review_required' and report['goal_verified'] is False
            # A newly supported receipt cannot survive withdrawal of the new
            # evidence, even if a prior report/version exists on disk.
            mobile.service.store.delete_document(SID, fresh_document)
            withdrawn = await current_report(browser, task['id'], 2)
            assert withdrawn['available'] is False
            assert mobile.model.await_count == 8
    finally:
        await project_tasks.shutdown_project_tasks()
        await mobile.service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize('domain', DOMAINS)
async def test_current_personal_draft_review_cannot_survive_selected_memory_revision(research_mobile, domain):
    mobile = research_mobile
    private_manager(mobile)

    async def controlled_response(prompt, **kwargs):
        req = packet(prompt, 'TASK_REQUIREMENTS_DATA_JSON:\n')
        answer = 'Private draft prepared from your selected instruction; nothing was sent or booked.'
        return json.dumps({'answer': answer, 'task_assessment': {'requirements': [
            {'id': row['id'], 'verdict': 'addressed', 'evidence': [],
             'answer_witness': {'quote': answer, 'start': 0}} for row in req['requirements']]}})

    mobile.model.side_effect = controlled_response
    try:
        async with mobile_client(mobile) as browser:
            note = await memory(mobile, browser, domain['memory'])
            task = await create(browser, {'session_id': SID, 'goal': 'Prepare a private draft using only my selected requirements.',
                'constraints': 'No external action or claims of externally verified facts.', 'questions': [
                    {'id': 'draft', 'question': domain['questions'][1],
                     'input_kind': 'personal', 'memory_ids': [note['id']]}]})
            await start(browser, task, calls=1)
            done = await terminal(browser, task['id'])
            assert done['status'] == 'completed' and mobile.model.await_count == 1
            assert done['evidence_progress']['next_action'] == 'review'
            before = await current_report(browser, task['id'], 1)
            assert before['available'] and before['goal_verified'] is False
            changed = await browser.post(PREFIX + '/research/memory', json={
                'session_id': SID, 'kind': 'decision', 'text': domain['memory'] + ' The earlier review date is withdrawn.',
                'memory_id': note['id'], 'expected_revision': 1})
            assert changed.status_code == 200, changed.text
            after = await current_report(browser, task['id'], 1)
            assert after['available'] is False and mobile.model.await_count == 1
            response = await browser.get(PREFIX + '/workbench/task', params={'session_id': SID, 'task_id': task['id']})
            assert response.status_code == 200
            progress = response.json()['task']['evidence_progress']
            assert progress.get('status') == 'unavailable' or progress['coverage']['ready_for_review'] == 0
    finally:
        await project_tasks.shutdown_project_tasks()
        await mobile.service.stop()
