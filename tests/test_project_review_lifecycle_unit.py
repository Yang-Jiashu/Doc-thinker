"""Independent ASGI -> Harness -> bridge -> service -> SQLite review lifecycle.

Every model response is deterministic local test code. These checks establish
authorization, durable lineage and revocation behavior, NOT answer quality,
semantic entailment, real provider latency or reliable causal understanding.
"""
import asyncio
import json
import time
import uuid

import pytest

from docthinker.server.routers import project_tasks
from tests.test_personal_memory_transport_unit import (
    base_mobile, transport, mobile_client, restart, PREFIX,
)
from tests.test_research_mobile_unit import upload_ready


def uid():
    return str(uuid.uuid4())


def packet(prompt, marker):
    return json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1].lstrip())[0]


async def current(browser, sid, task_id):
    response = await browser.get(PREFIX + '/workbench/task', params={'session_id': sid, 'task_id': task_id})
    assert response.status_code == 200, response.text
    return response.json()['task']


async def checkpoint(browser, sid, task_id):
    deadline, latest = time.monotonic() + 15, None
    while time.monotonic() < deadline:
        latest = await current(browser, sid, task_id)
        if latest['status'] in {'completed', 'blocked', 'paused', 'stale', 'cancelled'}:
            return latest
        await asyncio.sleep(.025)
    raise AssertionError('Review task did not settle: ' + str({
        'status': latest['status'], 'steps': [(row['id'], row['status']) for row in latest['steps']]}))


async def run(browser, sid, task, *, calls=1):
    response = await browser.post(PREFIX + '/workbench/tasks/run', json={
        'session_id': sid, 'task_id': task['id'], 'expected_revision': task['revision'],
        'request_id': uid(), 'max_calls': calls})
    assert response.status_code == 200, response.text
    return await checkpoint(browser, sid, task['id'])


async def control(browser, sid, task, action):
    response = await browser.post(PREFIX + '/workbench/tasks/control', json={
        'session_id': sid, 'task_id': task['id'], 'expected_revision': task['revision'], 'action': action})
    assert response.status_code == 200, response.text
    return response.json()['task']


async def artifact(browser, sid, task_id, step_id='report', version=1):
    response = await browser.get(PREFIX + '/workbench/task/artifact', params={
        'session_id': sid, 'task_id': task_id, 'step_id': step_id, 'version': version})
    assert response.status_code == 200, response.text
    return response.json()['artifact']


def rows(scope, count=1):
    result = []
    for index in range(1, count + 1):
        writer = {'id': 'draft' + str(index), 'title': 'Private draft ' + str(index),
            'purpose': 'compose', 'question': 'Prepare private branch ' + str(index) + '; disclose uncertainty and do not act externally.',
            **scope}
        result.extend([writer, {**writer, 'id': 'review' + str(index), 'title': 'Check evidence ' + str(index),
                               'purpose': 'verify', 'review_of': writer['id'], 'depends_on': [writer['id']]}])
    return result


class ScriptedModel:
    """Protocol-only test fixture. Its labels do not establish semantic truth."""

    def __init__(self, personal):
        self.personal = personal
        self.prompts = []
        self.fail_review = False
        self.written = []
        self.reviewed = []

    async def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if 'FROZEN_REVIEW_DATA_JSON:\n' not in prompt:
            body = 'PRIVATE_REVIEW_DRAFT_' + str(len(self.written) + 1) + ': Draft only; no external action.'
            if not self.personal:
                body += ' [S1]'
            self.written.append(body)
            return body
        data = packet(prompt, 'FROZEN_REVIEW_DATA_JSON:\n')
        self.reviewed.append(data)
        if self.fail_review:
            # A nonempty provider response with a known invalid protocol,
            # not a network timeout or an unknown paid execution outcome.
            return 'This response is not the required assessment envelope.'
        evidence = [] if self.personal else [{'label': data['source_quotes'][0]['label']}]
        return json.dumps({'answer': 'REVIEW_ENVELOPE_IS_NOT_A_REPLACEMENT_DRAFT',
            'task_assessment': {'requirements': [{'id': row['id'], 'verdict': 'addressed',
                'evidence': evidence, 'answer_witness': {'whole_answer': True}}
                for row in data['task_requirements'] if row['id'] in data['active_task_ids']]}})


@pytest.mark.asyncio
@pytest.mark.parametrize('input_kind', ['documents', 'personal'])
async def test_draft_survives_review_failure_restart_and_explicit_retry_then_revokes(transport, input_kind):
    mobile, sid = transport.mobile, transport.sid
    script = ScriptedModel(input_kind == 'personal')
    mobile.model.side_effect = script.__call__
    try:
        async with mobile_client(mobile) as browser:
            if input_kind == 'personal':
                response = await browser.post(PREFIX + '/research/memory', json={
                    'session_id': sid, 'kind': 'decision', 'text': 'Private budget is 4827; never execute payment.'})
                assert response.status_code == 200, response.text
                note = response.json()['memory']
                scope = {'input_kind': 'personal', 'memory_ids': [note['id']]}
            else:
                document = await upload_ready(mobile, browser, sid=sid, filename='approved-record.txt',
                    text='The approved private record states a budget of 4827; sending and payment require separate approval.')
                scope = {'document_ids': [document]}
            response = await browser.post(PREFIX + '/workbench/tasks', json={
                'session_id': sid, 'goal': 'Produce a usable private draft and separately check its retained evidence.',
                'constraints': 'Keep uncertainty explicit; no sending, booking or payment.', 'questions': rows(scope)})
            assert response.status_code == 200, response.text
            task = response.json()['task']
            assert mobile.model.await_count == 0
            paused = await run(browser, sid, task, calls=1)
            assert paused['status'] == 'paused' and mobile.model.await_count == 1
            first_writer = next(row for row in paused['steps'] if row['id'] == 'draft1')
            assert first_writer['status'] == 'completed'
            draft = await artifact(browser, sid, task['id'], 'draft1')
            assert draft['available'] and draft['content'] == script.written[0]
            assert draft['references']['analysis_review']['purpose'] == 'compose'
            assert paused['evidence_progress']['goal_verified'] is False
            assert paused['evidence_progress']['coverage']['ready_for_review'] == 0

            await project_tasks.shutdown_project_tasks()
            await restart(transport)
            paused = await current(browser, sid, task['id'])
            assert (await artifact(browser, sid, task['id'], 'draft1'))['content'] == script.written[0]
            assert mobile.model.await_count == 1  # status/restart cannot dispatch the audit
            script.fail_review = True
            failed = await run(browser, sid, paused, calls=1)
            reviewer = next(row for row in failed['steps'] if row['id'] == 'review1')
            assert reviewer['status'] == 'blocked' and reviewer['reason'] == 'provider_failed'
            assert mobile.model.await_count == 2 and len(script.written) == 1
            assert (await artifact(browser, sid, task['id'], 'draft1'))['content'] == script.written[0]
            for _ in range(2):
                failed = await current(browser, sid, task['id'])
            assert mobile.model.await_count == 2  # no hidden retry or fallback generation
            reset = await control(browser, sid, failed, 'retry_failed')
            assert mobile.model.await_count == 2
            kept = next(row for row in reset['steps'] if row['id'] == 'draft1')
            assert (kept['request_id'], kept['artifact_version']) == (first_writer['request_id'], first_writer['artifact_version'])
            script.fail_review = False
            done = await run(browser, sid, reset, calls=1)
            assert done['status'] == 'completed' and mobile.model.await_count == 3
            checked = await artifact(browser, sid, task['id'], 'review1')
            assert checked['available'] and checked['content'] == draft['content']
            lineage = checked['references']['analysis_review']
            assert lineage['purpose'] == 'verify' and lineage['root_answer_id'] == draft['references']['answer_id']
            assert lineage['semantic_verified'] is False
            report = await artifact(browser, sid, task['id'])
            assert report['available'] and script.written[0] in report['content']
            assert report['content'].count(script.written[0]) == 1  # not one body per writer/reviewer copy
            assert 'REVIEW_ENVELOPE_IS_NOT_A_REPLACEMENT_DRAFT' not in report['content']
            assert report['acceptance_status'] == 'review_required' and report['goal_verified'] is False

            await project_tasks.shutdown_project_tasks()
            await restart(transport)
            assert (await artifact(browser, sid, task['id'], 'review1'))['available']
            if input_kind == 'personal':
                changed = await browser.post(PREFIX + '/research/memory', json={
                    'session_id': sid, 'memory_id': note['id'], 'expected_revision': 1,
                    'kind': 'decision', 'text': 'Corrected private budget is 4300, not 4827; payment still prohibited.'})
            else:
                changed = await browser.post(PREFIX + '/research/documents/delete', json={'session_id': sid, 'document_id': document})
            assert changed.status_code == 200, changed.text
            await project_tasks.shutdown_project_tasks()
            await restart(transport)
            for step_id in ('draft1', 'review1', 'report'):
                withdrawn = await artifact(browser, sid, task['id'], step_id)
                assert withdrawn['available'] is False and script.written[0] not in str(withdrawn)
            assert mobile.model.await_count == 3
    finally:
        await project_tasks.shutdown_project_tasks()
        await transport.service.stop()


@pytest.mark.asyncio
async def test_long_project_default_parent_carries_six_reviewed_leaves_not_twelve_duplicate_drafts(transport):
    mobile, sid = transport.mobile, transport.sid
    script = ScriptedModel(personal=True)
    mobile.model.side_effect = script.__call__
    try:
        async with mobile_client(mobile) as browser:
            response = await browser.post(PREFIX + '/research/memory', json={
                'session_id': sid, 'kind': 'decision', 'text': 'All six branches share the confirmed private deadline Friday; do not send.'})
            assert response.status_code == 200, response.text
            note = response.json()['memory']
            response = await browser.post(PREFIX + '/workbench/long-projects', json={
                'session_id': sid, 'request_id': uid(), 'goal': 'Maintain a private long-running household and work plan.',
                'acceptance_criteria': ['Produce usable drafts.', 'Disclose uncertainty and never act externally.']})
            assert response.status_code == 200, response.text
            project = response.json()['project']
            response = await browser.post(PREFIX + '/workbench/long-project/phases', json={
                'session_id': sid, 'project_id': project['id'], 'expected_revision': project['revision'],
                'request_id': uid(), 'title': 'Six separately checked drafts',
                'acceptance_criteria': ['Each frozen draft has an explicit evidence review.'],
                'questions': rows({'input_kind': 'personal', 'memory_ids': [note['id']]}, count=6)})
            assert response.status_code == 200, response.text
            first = response.json()
            paused = await run(browser, sid, first['task'], calls=6)
            assert paused['status'] == 'paused' and mobile.model.await_count == 6
            done = await run(browser, sid, paused, calls=6)
            assert done['status'] == 'completed' and mobile.model.await_count == 12
            assert len(script.written) == len(script.reviewed) == 6
            response = await browser.get(PREFIX + '/workbench/long-project', params={'session_id': sid, 'project_id': project['id']})
            assert response.status_code == 200, response.text
            project = response.json()['project']
            response = await browser.post(PREFIX + '/workbench/long-project/phases', json={
                'session_id': sid, 'project_id': project['id'], 'expected_revision': project['revision'],
                'request_id': uid(), 'title': 'Continue from checked leaves',
                'acceptance_criteria': ['Retain all six explicitly selected predecessor results.'],
                'questions': [{'id': 'continue', 'question': 'Continue the six private branches; retain uncertainty and do not act externally.',
                               'input_kind': 'personal', 'memory_ids': [note['id']]}],
                # Deliberately omit step_ids: default must choose review leaves,
                # not both copies, and must remain within the existing six cap.
                'parent_refs': [{'phase_id': first['phase']['id'], 'report_version': 1}]})
            assert response.status_code == 200, response.text
            second = response.json()
            assert second['phase']['parent_refs'][0]['step_ids'] == ['review' + str(index) for index in range(1, 7)]
            assert mobile.model.await_count == 12  # creating the next phase is not a grant
            continued = await run(browser, sid, second['task'], calls=1)
            assert continued['status'] == 'completed' and mobile.model.await_count == 13
            step = next(row for row in continued['steps'] if row['id'] == 'continue')
            assert step['phase_handoff']['required'] == step['phase_handoff']['used'] == 6
            assert step['phase_handoff']['omitted'] == 0
            data = packet(script.prompts[-1], 'PROJECT_TASK_DATA_JSON:\n')['context']
            handoff = json.JSONDecoder().raw_decode(data[data.index('[{"step_id"'):])[0]
            assert [row['step_id'] for row in handoff] == [
                'p:' + first['phase']['id'] + ':review' + str(index) for index in range(1, 7)]
            assert all(row['coverage'] == 'full' for row in handoff)
            assert continued['evidence_progress']['goal_verified'] is False
            child_body = script.written[-1]
            assert child_body in (await artifact(browser, sid, second['task']['id']))['content']
            changed = await browser.post(PREFIX + '/research/memory', json={
                'session_id': sid, 'memory_id': note['id'], 'expected_revision': 1,
                'kind': 'decision', 'text': 'Confirmed private deadline corrected to Thursday, not Friday; do not send.'})
            assert changed.status_code == 200, changed.text
            await project_tasks.shutdown_project_tasks()
            await restart(transport)
            for task_id, step_id in ((first['task']['id'], 'review1'), (second['task']['id'], 'continue'),
                                     (second['task']['id'], 'report')):
                withdrawn = await artifact(browser, sid, task_id, step_id)
                assert withdrawn['available'] is False and child_body not in str(withdrawn)
            history = await browser.get(PREFIX + '/history', params={'session_id': sid})
            assert history.status_code == 200 and child_body not in history.text
            assert mobile.model.await_count == 13  # invalidation is local, never regeneration
    finally:
        await project_tasks.shutdown_project_tasks()
        await transport.service.stop()


@pytest.mark.asyncio
async def test_counterevidence_repair_retrieves_omitted_record_and_replaces_body_without_old_goal_fallback(transport):
    """Real retrieval/control path; scripted text is not semantic evaluation.

    The first controlled writer deliberately contradicts the no-contact
    constraint; the explicit reviewer rejects it against the retained record.
    The relevant execution record already exists in the SAME approved corpus but is
    outside the initial retained source windows. Rechecking frozen windows
    cannot recover it. A typed, explicitly approved compose+verify pair must.
    """
    mobile, sid = transport.mobile, transport.sid
    question = 'Prepare a short Orion status report. Distinguish confirmed details from missing records and do not contact anyone.'
    missing = 'Signed execution log RELEASE_ACCOUNT_RECORD_804 two full days release account unavailable final critical path'
    prompts, retained, compose_count, review_count = [], [], 0, 0

    async def controlled_model(prompt, **kwargs):
        nonlocal compose_count, review_count
        prompts.append(prompt)
        if 'FROZEN_REVIEW_DATA_JSON:\n' in prompt:
            data = packet(prompt, 'FROZEN_REVIEW_DATA_JSON:\n')
            review_count += 1
            evidence = [{'label': data['source_quotes'][0]['label']}] if review_count == 1 else [
                {'label': next(row['label'] for row in data['source_quotes'] if 'RELEASE_ACCOUNT_RECORD_804' in row['content'])}]
            return json.dumps({'answer': 'This assessment is not replacement report text.',
                'task_assessment': {'requirements': [{'id': row['id'],
                    'verdict': 'counterevidence' if review_count == 1 else 'addressed',
                    'evidence': evidence,
                    **({'missing_query': missing} if review_count == 1 else {'answer_witness': {'whole_answer': True}})}
                    for row in data['task_requirements'] if row['id'] in data['active_task_ids']]}})
        compose_count += 1
        quotes = packet(prompt, 'UNTRUSTED_RESEARCH_DATA_JSON:\n')['source_quotes']
        retained.append(quotes)
        target = next((row for row in quotes if 'RELEASE_ACCOUNT_RECORD_804' in row['content']), None)
        if compose_count == 1:
            assert target is None, 'Fixture must genuinely omit the critical retained window before repair'
            return 'Initial private Orion draft: contacting the client is authorized [' + quotes[0]['label'] + ']. The signed execution details remain missing.'
        assert target is not None, 'Repair must genuinely recover the already-authorized original record'
        assert missing in prompt and '前序证据缺口的待查队列' in prompt
        return 'Updated private Orion report: the signed execution log records two full days of release-account unavailability on the final critical path [' + target['label'] + ']. Other causal contributions remain unknown; nobody was contacted.'

    mobile.model.side_effect = controlled_model
    try:
        async with mobile_client(mobile) as browser:
            documents = []
            topics = ['finance budget invoice receipts', 'design typography icon review', 'supply warehouse inventory forecast',
                      'sales contract proposal review', 'staff travel accommodation schedule', 'software testing branch coverage',
                      'customer interview survey feedback', 'hardware device battery calibration', 'marketing campaign audience geography',
                      'legal license attribution notices', 'operations weekly dashboard review', 'accessibility keyboard screenreader audit']
            for index, topic in enumerate(topics):
                documents.append(await upload_ready(mobile, browser, sid=sid, filename='status-' + str(index) + '.txt',
                    text='Orion status report: confirmed details concern ' + topic + '. '
                         'The short private report distinguishes this preliminary record from missing details and prohibits contacting anyone. '
                         'This department record describes only ' + topic + ', not the project execution cause. DECOY_' + str(index)))
            target_document = await upload_ready(mobile, browser, sid=sid, filename='signed-execution-log.txt',
                text='RELEASE_ACCOUNT_RECORD_804. Signed execution log: the release account was unavailable for two full days on the final critical path. '
                     'The record does not establish a universally unique cause and no external communication was performed.')
            documents.append(target_document)
            writer = {'id': 'original', 'title': 'Original brief', 'question': question, 'document_ids': documents}
            response = await browser.post(PREFIX + '/workbench/tasks', json={
                'session_id': sid, 'goal': question, 'constraints': 'Retain uncertainty; private draft only.',
                'questions': [writer, {**writer, 'id': 'original_review', 'title': 'Original evidence review',
                                      'purpose': 'verify', 'review_of': 'original', 'depends_on': ['original']}]})
            assert response.status_code == 200, response.text
            original = response.json()['task']
            first = await run(browser, sid, original, calls=2)
            assert first['status'] == 'completed' and mobile.model.await_count == 2, (
                f"state={first['status']}; calls={mobile.model.await_count}; packets={len(retained)}; "
                f"initial_target_retained={any('RELEASE_ACCOUNT_RECORD_804' in row.get('content', '') for group in retained for row in group)}")
            assert first['evidence_progress']['coverage']['contradicted'] > 0
            old_body = (await artifact(browser, sid, first['id'], 'original'))['content']
            response = await browser.post(PREFIX + '/workbench/tasks/evidence/preview', json={
                'session_id': sid, 'task_id': first['id'], 'expected_revision': first['revision']})
            assert response.status_code == 200, response.text
            repair = response.json()['repair']
            assert len(repair['changed_steps']) == len(repair['proposal']['added']) == 2
            assert repair['proposal']['changed'] == [] and mobile.model.await_count == 2
            added = [row for row in repair['plan']['questions'] if row['id'] in repair['proposal']['added']]
            refinement = next(row for row in added if row.get('purpose') != 'verify')
            audit = next(row for row in added if row.get('purpose') == 'verify')
            assert refinement['refine_of'] == 'original_review' and refinement['depends_on'] == ['original_review']
            assert refinement['question'] == audit['question'] == question
            assert refinement['document_ids'] == audit['document_ids'] == documents
            assert audit['review_of'] == refinement['id']
            response = await browser.post(PREFIX + '/workbench/tasks/replan', json={
                'session_id': sid, 'task_id': first['id'], 'expected_revision': first['revision'],
                'proposal_sha256': repair['proposal']['proposal_sha256'], **repair['plan']})
            assert response.status_code == 200, response.text
            prepared = await control(browser, sid, response.json()['task'], 'refresh')
            pending = await current(browser, sid, first['id'])
            assert pending['evidence_progress']['coverage']['ready_for_review'] == 0
            assert mobile.model.await_count == 2  # save, refresh and read confer no execution authority
            paused = await run(browser, sid, prepared, calls=1)
            assert paused['status'] == 'paused' and mobile.model.await_count == 3
            updated = await artifact(browser, sid, first['id'], refinement['id'])
            assert updated['available'] and updated['content'] != old_body
            assert 'two full days' in updated['content']
            assert any(row['filename'] == 'signed-execution-log.txt' for row in retained[-1])
            assert mobile.model.await_count == 3  # verification is another explicit grant
            done = await run(browser, sid, paused, calls=1)
            assert done['status'] == 'completed' and mobile.model.await_count == 4
            progress = done['evidence_progress']
            assert progress['coverage']['ready_for_review'] == progress['coverage']['total']
            assert progress['goal_verified'] is progress['semantic_verified'] is False
            assert all(row['owner_step_id'] not in {'original', 'original_review'} for row in progress['obligations'])
            report = await artifact(browser, sid, first['id'], version=2)
            assert report['available'] and updated['content'] in report['content'] and old_body not in report['content']
            # Original private history remains available as HISTORY, not as a
            # fallback supported answer for the replacement's active goal.
            assert (await artifact(browser, sid, first['id'], 'original'))['content'] == old_body
            mobile.service.store.delete_document(sid, target_document)
            withdrawn = await current(browser, sid, first['id'])
            assert withdrawn['evidence_progress'].get('status') == 'unavailable' or withdrawn['evidence_progress']['coverage']['ready_for_review'] == 0
            assert (await artifact(browser, sid, first['id'], audit['id']))['available'] is False
            assert mobile.model.await_count == 4
    finally:
        await project_tasks.shutdown_project_tasks()
        await transport.service.stop()
