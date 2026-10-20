"""Restart boundaries in workplace and personal-event tasks; never providers."""
import asyncio
import sqlite3
import uuid

import pytest

from docthinker.server.project_task_runtime import ProjectTaskRuntime
from docthinker.server.project_task_store import (
    ProjectTaskStore, ProjectTaskConflictError, ProjectTaskNotFoundError,
)


def uid():
    return str(uuid.uuid4())


def opened(path, boot='before', session='#00001', instance='private-agent'):
    return ProjectTaskStore(path.resolve(), session_id=session, instance_id=instance, boot_id=boot)


def created(store, domain):
    return store.create_task(goal=domain, constraints='Do not send, pay or change files.', questions=[
        {'id': 'review', 'question': 'Review the recorded event.', 'document_ids': ['event']},
        {'id': 'followup', 'question': 'Draft the next action.', 'document_ids': ['event'], 'depends_on': ['review']},
    ])


SOURCES = {'event': 'a' * 64}


def begin(store, task):
    return store.begin_run(task['id'], expected_revision=task['revision'], max_calls=2,
                           source_snapshots=SOURCES, request_id=uid())


def finish(store, task, step):
    artifact = ({'documents': [{'id': 'event', 'digest': SOURCES['event']}]} if step['kind'] == 'inventory'
                else {'answer_id': uid(), 'source_ids': ['record'], 'document_ids': ['event']})
    return store.complete_step(task['id'], step['id'], step['claim_token'], artifact)


@pytest.mark.parametrize('domain', ['Prepare a client handover', 'Follow up a family appointment'])
@pytest.mark.parametrize('completed_count', [0, 1, 2])
def test_restart_between_claims_requires_new_approval_and_keeps_saved_stage(tmp_path, domain, completed_count):
    old = opened(tmp_path)
    task = begin(old, created(old, domain))
    old_run = task['run_id']
    for _ in range(completed_count):
        task = finish(old, task, old.claim_next(task['id'], old_run))
    saved = [(row['id'], row['artifact_version'], row['request_id']) for row in task['steps'] if row['status'] == 'completed']
    fresh = opened(tmp_path, 'after')
    recovered = fresh.recover()
    assert len(recovered) == 1
    task = recovered[0]
    assert task['status'] == 'paused' and task['reason'] == 'process_restarted'
    assert fresh.claim_next(task['id'], old_run) is None
    assert fresh.recover() == []
    assert saved == [(row['id'], row['artifact_version'], row['request_id']) for row in task['steps'] if row['status'] == 'completed']
    task = begin(fresh, task)
    next_step = fresh.claim_next(task['id'], task['run_id'])
    assert next_step['id'] == ['inventory', 'review', 'followup'][completed_count]


def test_same_boot_recovery_does_not_cancel_an_approved_queued_job(tmp_path):
    store = opened(tmp_path)
    task = begin(store, created(store, 'Client handover'))
    assert store.recover() == []
    assert store.claim_next(task['id'], task['run_id'])['id'] == 'inventory'


def test_other_boot_cannot_drive_old_grant_even_before_explicit_recovery(tmp_path):
    old = opened(tmp_path)
    task = begin(old, created(old, 'Appointment follow-up'))
    assert opened(tmp_path, 'after').claim_next(task['id'], task['run_id']) is None


@pytest.mark.parametrize('change', ['source', 'goal'])
def test_invalidating_unknown_work_never_authorizes_another_external_attempt(tmp_path, change):
    store = opened(tmp_path)
    task = begin(store, created(store, 'Client handover'))
    task = finish(store, task, store.claim_next(task['id'], task['run_id']))
    claim = store.claim_next(task['id'], task['run_id'])
    task = store.block_step(task['id'], claim['id'], claim['claim_token'], reason='provider_outcome_unknown')
    if change == 'source':
        task = store.invalidate_sources(['event'])[0]
    else:
        questions = [{'id': row['id'], 'title': row['title'], 'question': row['question'],
                      'document_ids': row['document_ids'], 'depends_on': [x for x in row['depends_on'] if x != 'inventory']}
                     for row in task['steps'] if row['kind'] == 'analysis']
        task = store.revise_task(task['id'], expected_revision=task['revision'], goal='Revised client handover',
                                 constraints=task['constraints'], questions=questions)
    with pytest.raises(ProjectTaskConflictError):
        store.refresh_task(task['id'], expected_revision=task['revision'], source_snapshots={'event': 'b' * 64})
    assert store.get_task(task['id'])['steps'][1]['request_id'] == claim['request_id']


def test_recovery_is_scoped_to_exact_session_and_instance(tmp_path):
    store = opened(tmp_path)
    task = begin(store, created(store, 'Family appointment'))
    for other in [opened(tmp_path, 'after', '#00002'), opened(tmp_path, 'after', instance='other-agent')]:
        assert other.recover() == []
        with pytest.raises(ProjectTaskNotFoundError):
            other.get_task(task['id'])
    assert store.get_task(task['id'])['status'] == 'ready'


def test_runtime_restart_with_changed_source_will_not_dispatch_from_saved_checkpoint(tmp_path):
    async def scenario():
        old = opened(tmp_path)
        task = begin(old, created(old, 'Workplace handover'))
        for _ in range(2):
            task = finish(old, task, old.claim_next(task['id'], task['run_id']))
        fresh = opened(tmp_path, 'after')
        fresh.recover()
        calls = []
        async def execute(**kwargs):
            calls.append(kwargs)
        async def sources(document_ids=None):
            return {'event': 'b' * 64}
        runtime = ProjectTaskRuntime(fresh, execute_question=execute, read_sources=sources)
        task = fresh.get_task(task['id'])
        with pytest.raises(ProjectTaskConflictError):
            await runtime.run(task['id'], expected_revision=task['revision'], run_request_id=uid())
        assert calls == []
    asyncio.run(scenario())


def test_legacy_run_ownership_migrates_without_losing_saved_artifacts(tmp_path):
    old = opened(tmp_path)
    task = begin(old, created(old, 'Client handover'))
    task = finish(old, task, old.claim_next(task['id'], task['run_id']))
    with sqlite3.connect(old.path) as db:
        db.execute('ALTER TABLE project_task_runs DROP COLUMN owner_boot')
    fresh = opened(tmp_path, 'after')
    task = fresh.recover()[0]
    assert task['status'] == 'paused' and task['steps'][0]['artifact_version'] == 1
    assert fresh.get_artifact(task['id'], 'inventory')['available']
    assert fresh.claim_next(task['id'], task['run_id']) is None


@pytest.mark.parametrize('outcome', ['completed', 'failed'])
@pytest.mark.parametrize('domain', ['Client handover', 'Family appointment follow-up'])
def test_terminal_old_journal_resolves_uncertainty_without_publishing_or_dispatch(tmp_path, outcome, domain):
    async def scenario():
        old = opened(tmp_path)
        task = begin(old, created(old, domain))
        task = finish(old, task, old.claim_next(task['id'], task['run_id']))
        claim = old.claim_next(task['id'], task['run_id'])
        old.block_step(task['id'], claim['id'], claim['claim_token'], reason='provider_outcome_unknown')
        old.invalidate_sources(['event'])
        fresh = opened(tmp_path, 'after')
        calls, journal_reads = [], []
        async def execute(**kwargs):
            calls.append(kwargs)
        async def read_journal(**kwargs):
            journal_reads.append(kwargs)
            return {'request_status': outcome, 'answer_id': uid(), 'answer': 'OLD ANSWER NEVER PUBLISH'}
        async def sources(document_ids=None):
            return {'event': 'b' * 64}
        runtime = ProjectTaskRuntime(fresh, execute_question=execute, read_sources=sources, reconcile_request=read_journal)
        task = await runtime.reconcile(task['id'])
        assert calls == [] and len(journal_reads) == 1
        assert journal_reads[0] == {'session_id': '#00001', 'request_id': claim['request_id']}
        assert task['status'] == 'stale' and task['steps'][1]['artifact_version'] == 0
        assert fresh.unresolved_requests(task['id']) == set()
        assert fresh.claim_next(task['id'], task['run_id']) is None
        assert b'OLD ANSWER NEVER PUBLISH' not in fresh.path.read_bytes()
        with pytest.raises(ProjectTaskConflictError):
            fresh.settle_invalidated_request(task['id'], 'review', request_id=uid(), outcome=outcome)
        refreshed = fresh.refresh_task(task['id'], expected_revision=task['revision'], source_snapshots={'event': 'b' * 64})
        assert refreshed['steps'][1]['request_id'] != claim['request_id']
        assert calls == []  # Explicit refresh prepares; it never dispatches.
    asyncio.run(scenario())


def test_unknown_journal_does_not_clear_invalidated_request(tmp_path):
    async def scenario():
        store = opened(tmp_path)
        task = begin(store, created(store, 'Family appointment'))
        task = finish(store, task, store.claim_next(task['id'], task['run_id']))
        claim = store.claim_next(task['id'], task['run_id'])
        store.block_step(task['id'], claim['id'], claim['claim_token'], reason='provider_outcome_unknown')
        store.invalidate_sources(['event'])
        async def journal(**kwargs):
            return {'request_status': 'unknown'}
        async def forbidden(**kwargs):
            pytest.fail('No source or external work is needed to read an unknown journal result')
        runtime = ProjectTaskRuntime(store, execute_question=forbidden, read_sources=forbidden, reconcile_request=journal)
        task = await runtime.reconcile(task['id'])
        assert store.unresolved_requests(task['id']) == {claim['request_id']}
        with pytest.raises(ProjectTaskConflictError):
            store.refresh_task(task['id'], expected_revision=task['revision'], source_snapshots=SOURCES)
    asyncio.run(scenario())
