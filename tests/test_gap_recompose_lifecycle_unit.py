"""A missed retained window must trigger retrieval and rewriting, not audit loops.

Real ASGI/Harness/SQLite/lexical retrieval with a local protocol-only model.
The corpus does not change between the first draft and approved repair.
This establishes control flow, not general semantic or causal model quality.
"""
import json

import pytest

from docthinker.server.routers import project_tasks
from tests.test_evidence_goal_lifecycle_unit import (
    research_mobile, private_manager, packet, terminal,
)
from tests.test_research_mobile_unit import mobile_client, upload_ready, SID, PREFIX
from tests.test_personal_tasks_mobile_unit import create, start


@pytest.mark.asyncio
@pytest.mark.parametrize('topic,hidden', [('Orion delivery timeline', 'zircon'),
                                         ('Maple appointment options', 'amber')])
async def test_unchanged_corpus_missed_window_is_retrieved_by_explicit_recompose(research_mobile, topic, hidden):
    mobile = research_mobile
    private_manager(mobile)
    golden = (hidden + ' audit: the actual account outage lasted two full days. '
              'The measured additional work was outside the final critical path; '
              'the earlier allocation was an estimate, not measured execution.')
    # The initial literal query ranks many real topic windows, but not the
    # low-overlap audit. The explicit missing query can locate it locally.
    background = '\n\n'.join((topic + ' planning note ' + str(index) + '. ') * 24
                              for index in range(28))
    separator = '\n\n'.join(('Unrelated atlas catalog constellation coordinates ' + str(index) + '. ') * 30
                               for index in range(4))
    text = background + '\n\n' + separator + '\n\n' + golden
    calls, drafts, observations = [], [], []

    async def model(prompt, **_kwargs):
        calls.append(prompt)
        if 'FROZEN_REVIEW_DATA_JSON:\n' in prompt:
            frozen = packet(prompt, 'FROZEN_REVIEW_DATA_JSON:\n')
            assert golden in ''.join(row['content'] for row in frozen['source_quotes'])
            return json.dumps({'answer': 'Review summary is not replacement prose.',
                'task_assessment': {'requirements': [
                    {'id': row['id'], 'verdict': 'addressed',
                     'evidence': [{'label': next(source['label'] for source in frozen['source_quotes']
                                                if golden in source['content'])}],
                     'answer_witness': {'whole_answer': True}}
                    for row in frozen['task_requirements'] if row['id'] in frozen['active_task_ids']]}})
        requirements = packet(prompt, 'TASK_REQUIREMENTS_DATA_JSON:\n')
        quotes = packet(prompt, 'UNTRUSTED_RESEARCH_DATA_JSON:\n')['source_quotes']
        observations.append({'golden_retained': any(golden in row['content'] for row in quotes),
                             'source_count': len(quotes)})
        if not drafts:
            assert not any(golden in row['content'] for row in quotes)
            body = 'Private draft: execution evidence is still missing [S1].'
            assessments = [{'id': row['id'], 'verdict': 'unknown',
                            'missing_query': hidden + ' actual account outage audit critical path measured execution'}
                           for row in requirements['requirements']]
        else:
            # A verifier alone cannot reach this branch. The newly approved
            # compose must actually consume the parent's missing-query packet.
            assert hidden in prompt and '前序证据缺口' in prompt
            source = next(row for row in quotes if golden in row['content'])
            body = 'Corrected private draft: ' + golden + ' [' + source['label'] + ']'
            assessments = [{'id': row['id'], 'verdict': 'addressed',
                            'evidence': [{'label': source['label']}],
                            'answer_witness': {'whole_answer': True}}
                           for row in requirements['requirements']]
        drafts.append(body)
        return json.dumps({'answer': body, 'task_assessment': {'requirements': assessments}})

    mobile.model.side_effect = model
    try:
        async with mobile_client(mobile) as browser:
            document = await upload_ready(mobile, browser, filename='unchanged-corpus.txt', text=text)
            task = await create(browser, {'session_id': SID, 'goal': 'Prepare a private agenda.',
                'constraints': 'No external action.',
                'questions': [{'id': 'draft', 'question': 'Prepare ' + topic + ' from the original records.',
                               'document_ids': [document]}]})
            await start(browser, task, calls=1)
            first = await terminal(browser, task['id'])
            assert first['status'] == 'completed' and len(calls) == 1, {
                'observations': observations, 'calls': len(calls),
                'steps': [(row['id'], row['status'], row.get('reason')) for row in first['steps']]}
            original = next(row for row in first['steps'] if row['id'] == 'draft')
            preview = await browser.post(PREFIX + '/workbench/tasks/evidence/preview', json={
                'session_id': SID, 'task_id': task['id'], 'expected_revision': first['revision']})
            assert preview.status_code == 200, preview.text
            repair = preview.json()['repair']
            additions = [row for row in repair['plan']['questions'] if row['id'] != 'draft']
            writers = [row for row in additions if row.get('purpose', 'compose') == 'compose']
            assert len(writers) == 1, 'A frozen audit cannot fetch a missed window or update prose'
            writer = writers[0]
            assert writer['question'] == original['question']
            assert writer['document_ids'] == [document] and writer['depends_on'] == ['draft']
            assert any(row.get('review_of') == writer['id'] for row in additions)
            assert len(calls) == 1  # preview is not execution or a new input snapshot
            applied = await browser.post(PREFIX + '/workbench/tasks/replan', json={
                'session_id': SID, 'task_id': task['id'], 'expected_revision': first['revision'],
                'proposal_sha256': repair['proposal']['proposal_sha256'], **repair['plan']})
            assert applied.status_code == 200, applied.text
            assert len(calls) == 1
            # Replanning deliberately leaves a stale report. Explicitly pin
            # the still-unchanged sources before asking for another grant.
            refreshed = await browser.post(PREFIX + '/workbench/tasks/control', json={
                'session_id': SID, 'task_id': task['id'],
                'expected_revision': applied.json()['task']['revision'], 'action': 'refresh'})
            assert refreshed.status_code == 200, refreshed.text
            assert len(calls) == 1
            await start(browser, refreshed.json()['task'], calls=2)
            done = await terminal(browser, task['id'])
            assert done['status'] == 'completed' and len(calls) == 3 and len(drafts) == 2
            current_writer = next(row for row in done['steps'] if row['id'] == writer['id'])
            assert current_writer['input_snapshot'] == original['input_snapshot']
            assert done['evidence_progress']['coverage']['unresolved'] == 0
            assert done['evidence_progress']['goal_verified'] is False
            artifact = await browser.get(PREFIX + '/workbench/task/artifact', params={
                'session_id': SID, 'task_id': task['id'], 'step_id': writer['id'], 'version': 1})
            assert artifact.status_code == 200, artifact.text
            assert golden in artifact.json()['artifact']['content']
            historical = await browser.get(PREFIX + '/workbench/task/artifact', params={
                'session_id': SID, 'task_id': task['id'], 'step_id': 'draft', 'version': 1})
            assert historical.status_code == 200 and historical.json()['artifact']['available']
            assert historical.json()['artifact']['content'] == drafts[0]
            report = await browser.get(PREFIX + '/workbench/task/artifact', params={
                'session_id': SID, 'task_id': task['id'], 'step_id': 'report', 'version': 2})
            assert report.status_code == 200 and report.json()['artifact']['available']
            assert golden in report.json()['artifact']['content']
            assert drafts[0] not in report.json()['artifact']['content']
            mobile.service.store.delete_document(SID, document)
            withdrawn = await browser.get(PREFIX + '/workbench/task/artifact', params={
                'session_id': SID, 'task_id': task['id'], 'step_id': writer['id'], 'version': 1})
            assert withdrawn.status_code == 200 and withdrawn.json()['artifact']['available'] is False
            assert len(calls) == 3
    finally:
        await project_tasks.shutdown_project_tasks()
        await mobile.service.stop()
