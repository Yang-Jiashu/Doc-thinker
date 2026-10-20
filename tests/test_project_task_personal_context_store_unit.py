"""Durable no-document work binds explicit memory revisions, never fake S refs."""
import json
import sqlite3
import uuid

import pytest

from docthinker.server.project_task_store import (
    ProjectTaskStore, ProjectTaskConflictError, ProjectTaskError, ProjectTaskNotFoundError,
)


def uid():
    return str(uuid.uuid4())


def opened(path, boot='first', session='#00001'):
    return ProjectTaskStore(path.resolve(), session_id=session, instance_id='personal-worker', boot_id=boot)


def question(key='work', memories=()):
    return {'id': key, 'question': 'Draft the next action, do not execute it.',
            'input_kind': 'personal', 'memory_ids': list(memories)}


def create(store, questions):
    return store.create_task(goal='Prepare a private work/family follow-up', constraints='No sending or payment.', questions=questions)


def begin(store, task, contexts=None, request_id=None):
    return store.begin_run(task['id'], expected_revision=task['revision'], max_calls=6,
                           source_snapshots={}, context_snapshots=contexts, request_id=request_id or uid())


def finish(store, task, step, contexts=None):
    if step['kind'] == 'inventory':
        artifact = {'documents': []}
    elif step['kind'] == 'report':
        artifact = {'step_refs': [{'step_id': row['id'], 'version': row['artifact_version']}
                                 for row in store.get_task(task['id'])['steps'] if row['kind'] == 'analysis']}
    else:
        artifact = {'answer_id': uid(), 'source_ids': [], 'document_ids': [], 'input_kind': 'personal',
                    'context_refs': dict(step['context_snapshot'])}
    return store.complete_step(task['id'], step['id'], step['claim_token'], artifact, context_snapshots=contexts)


def through_inventory(store, task, contexts=None):
    task = begin(store, task, contexts)
    return finish(store, task, store.claim_next(task['id'], task['run_id'], context_snapshots=contexts), contexts)


def test_personal_scope_and_revision_are_separate_from_documents_and_survive_restart(tmp_path):
    store = opened(tmp_path)
    memory = uid()
    task = through_inventory(store, create(store, [question(memories=[memory])]), {memory: 3})
    claim = store.claim_next(task['id'], task['run_id'], context_snapshots={memory: 3})
    assert claim['input_snapshot'] == {} and claim['context_snapshot'] == {memory: 3}
    assert claim['input_kind'] == 'personal' and claim['memory_ids'] == [memory]
    task = finish(store, task, claim, {memory: 3})
    artifact = store.get_artifact(task['id'], 'work')
    assert artifact['references']['context_refs'] == {memory: 3}
    assert artifact['references']['source_ids'] == artifact['references']['document_ids'] == []
    restarted = opened(tmp_path, 'second')
    restarted.recover()
    assert restarted.get_task(task['id'])['steps'][1]['context_snapshot'] == {memory: 3}
    assert restarted.definition_history(task['id'])[0]['questions'][0]['memory_ids'] == [memory]
    assert b'Draft the next action' in store.path.read_bytes() and b'private memory text' not in store.path.read_bytes()


def test_user_instruction_only_task_can_finish_with_zero_documents_and_zero_memories(tmp_path):
    store = opened(tmp_path)
    task = begin(store, create(store, [question()]))
    while (claim := store.claim_next(task['id'], task['run_id'])) is not None:
        task = finish(store, task, claim)
    assert task['status'] == 'completed'
    assert store.get_artifact(task['id'], 'work')['references']['context_refs'] == {}


@pytest.mark.parametrize('extra', [
    {'input_kind': 'other'}, {'input_kind': []}, {'memory_ids': [False]}, {'memory_ids': [{}]},
    {'memory_ids': ['not-a-uuid']}, {'memory_ids': [str(uuid.uuid4())] * 2},
    {'memory_ids': [str(uuid.uuid4()) for _ in range(9)]}, {'document_ids': ['document']},
])
def test_invalid_or_mixed_personal_scope_is_rejected_before_persistence(tmp_path, extra):
    store = opened(tmp_path)
    with pytest.raises(ProjectTaskError):
        create(store, [{**question(), **extra}])
    assert store.list_tasks() == []


def test_document_steps_cannot_enable_untracked_personal_context(tmp_path):
    store = opened(tmp_path)
    with pytest.raises(ProjectTaskError):
        create(store, [{'question': 'Review document', 'memory_ids': [uid()]}])
    task = create(store, [{'id': 'doc', 'question': 'Review document'}])
    assert not {'input_kind', 'memory_ids', 'context_snapshot'}.intersection(task['steps'][1])
    assert not {'input_kind', 'memory_ids'}.intersection(store.definition_history(task['id'])[0]['questions'][0])


@pytest.mark.parametrize('revision', [True, 0, -1, 1.5, '1', 2147483647])
def test_personal_revision_must_be_a_bounded_positive_integer(tmp_path, revision):
    store = opened(tmp_path)
    memory = uid()
    task = create(store, [question(memories=[memory])])
    with pytest.raises(ProjectTaskError):
        begin(store, task, {memory: revision})


def test_missing_selected_memory_cannot_silently_be_removed_from_approved_input(tmp_path):
    store = opened(tmp_path)
    task = create(store, [question(memories=[uid()])])
    with pytest.raises(ProjectTaskConflictError):
        begin(store, task)
    assert store.get_task(task['id'])['run_id'] is None


def test_approval_replay_is_bound_to_exact_personal_revision(tmp_path):
    store = opened(tmp_path)
    memory, rid = uid(), uid()
    original = create(store, [question(memories=[memory])])
    begin(store, original, {memory: 1}, rid)
    assert begin(store, original, {memory: 1}, rid)['run_replayed']
    with pytest.raises(ProjectTaskConflictError):
        begin(store, original, {memory: 2}, rid)


def test_changed_revision_before_claim_invalidates_without_spending_a_call(tmp_path):
    store = opened(tmp_path)
    memory = uid()
    task = through_inventory(store, create(store, [question(memories=[memory])]), {memory: 1})
    assert store.claim_next(task['id'], task['run_id'], context_snapshots={memory: 2}) is None
    task = store.get_task(task['id'])
    assert task['status'] == 'stale' and task['run_remaining'] == 6
    assert task['steps'][1]['request_id'] and task['steps'][1]['artifact_version'] == 0


def test_changed_revision_cannot_publish_late_completion(tmp_path):
    store = opened(tmp_path)
    memory = uid()
    task = through_inventory(store, create(store, [question(memories=[memory])]), {memory: 1})
    claim = store.claim_next(task['id'], task['run_id'], context_snapshots={memory: 1})
    with pytest.raises(ProjectTaskConflictError):
        finish(store, task, claim, {memory: 2})
    assert store.get_task(task['id'])['steps'][1]['artifact_version'] == 0


@pytest.mark.parametrize('tamper', ['missing', 'wrong', 'source', 'document', 'kind'])
def test_personal_artifact_requires_exact_host_context_and_no_fake_document_sources(tmp_path, tamper):
    store = opened(tmp_path)
    memory = uid()
    task = through_inventory(store, create(store, [question(memories=[memory])]), {memory: 1})
    claim = store.claim_next(task['id'], task['run_id'], context_snapshots={memory: 1})
    artifact = {'answer_id': uid(), 'source_ids': [], 'document_ids': [],
                'input_kind': 'personal', 'context_refs': {memory: 1}}
    if tamper == 'missing':
        artifact['context_refs'] = None
    elif tamper == 'wrong':
        artifact['context_refs'] = {memory: 2}
    elif tamper == 'source':
        artifact['source_ids'] = ['fake-S1']
    elif tamper == 'document':
        artifact['document_ids'] = ['fake-document']
    else:
        artifact['input_kind'] = 'documents'
    with pytest.raises(ProjectTaskError):
        store.complete_step(task['id'], 'work', claim['claim_token'], artifact, context_snapshots={memory: 1})


def test_memory_invalidation_is_targeted_and_propagates_only_real_dependents(tmp_path):
    store = opened(tmp_path)
    one, two = uid(), uid()
    rows = [question('client', [one]), question('family', [two]),
            {**question('followup', []), 'depends_on': ['client']}]
    task = begin(store, create(store, rows), {one: 1, two: 1})
    while (claim := store.claim_next(task['id'], task['run_id'], context_snapshots={one: 1, two: 1})) is not None:
        task = finish(store, task, claim, {one: 1, two: 1})
    changed = store.invalidate_memories([one], reason='source_revoked')[0]
    assert {row['id']: row['status'] for row in changed['steps']} == {
        'inventory': 'completed', 'client': 'stale', 'family': 'completed', 'followup': 'stale', 'report': 'stale'}
    assert store.get_artifact(task['id'], 'client')['available'] is False
    assert store.get_artifact(task['id'], 'family')['available'] is True
    assert store.invalidate_sources(['unrelated-document']) == []
    assert opened(tmp_path, session='#00002').invalidate_memories([two]) == []


def test_context_change_cannot_reset_unknown_paid_request(tmp_path):
    store = opened(tmp_path)
    memory = uid()
    task = through_inventory(store, create(store, [question(memories=[memory])]), {memory: 1})
    claim = store.claim_next(task['id'], task['run_id'], context_snapshots={memory: 1})
    store.block_step(task['id'], 'work', claim['claim_token'], reason='provider_outcome_unknown')
    task = store.invalidate_memories([memory])[0]
    with pytest.raises(ProjectTaskConflictError):
        store.refresh_task(task['id'], expected_revision=task['revision'], source_snapshots={}, context_snapshots={memory: 2})
    store.settle_invalidated_request(task['id'], 'work', request_id=claim['request_id'], outcome='completed')
    task = store.get_task(task['id'])
    refreshed = store.refresh_task(task['id'], expected_revision=task['revision'], source_snapshots={}, context_snapshots={memory: 2})
    assert refreshed['steps'][1]['context_snapshot'] == {memory: 2}
    assert refreshed['steps'][1]['request_id'] != claim['request_id']


def test_reconciliation_requires_original_personal_snapshot(tmp_path):
    store = opened(tmp_path)
    memory = uid()
    task = through_inventory(store, create(store, [question(memories=[memory])]), {memory: 1})
    claim = store.claim_next(task['id'], task['run_id'], context_snapshots={memory: 1})
    store.block_step(task['id'], 'work', claim['claim_token'], reason='provider_outcome_unknown')
    payload = {'answer_id': uid(), 'source_ids': [], 'document_ids': [], 'input_kind': 'personal', 'context_refs': {memory: 1}}
    with pytest.raises(ProjectTaskConflictError):
        store.reconcile_step(task['id'], 'work', request_id=claim['request_id'], artifact=payload,
                             source_snapshots={}, context_snapshots={memory: 2})
    done = store.reconcile_step(task['id'], 'work', request_id=claim['request_id'], artifact=payload,
                                source_snapshots={}, context_snapshots={memory: 1})
    assert done['steps'][1]['status'] == 'completed' and done['status'] == 'paused'


def test_old_step_schema_migrates_without_reinterpreting_document_scope(tmp_path):
    store = opened(tmp_path)
    task = create(store, [{'question': 'Review a client document'}])
    with sqlite3.connect(store.path) as db:
        for column in ('input_kind', 'memory_ids', 'context_snapshot'):
            db.execute(f'ALTER TABLE project_task_steps DROP COLUMN {column}')
    task = store.begin_run(task['id'], expected_revision=task['revision'], max_calls=1,
                           source_snapshots={}, request_id=uid())
    assert not {'input_kind', 'memory_ids', 'context_snapshot'}.intersection(task['steps'][1])


def test_all_personal_wrappers_bind_union_but_unrelated_documents_are_not_inputs(tmp_path):
    store = opened(tmp_path)
    one, two = uid(), uid()
    task = create(store, [question('client', [one]), question('family', [two])])
    for wrapper in (task['steps'][0], task['steps'][-1]):
        assert wrapper['input_kind'] == 'personal' and wrapper['memory_ids'] == [one, two]
    task = store.begin_run(task['id'], expected_revision=task['revision'], max_calls=2,
                           source_snapshots={'unrelated': 'a' * 64}, context_snapshots={one: 1, two: 1})
    assert all(row['input_snapshot'] == {} for row in task['steps'])
    while (claim := store.claim_next(task['id'], task['run_id'], context_snapshots={one: 1, two: 1})) is not None:
        task = finish(store, task, claim, {one: 1, two: 1})
    assert task['status'] == 'completed' and store.invalidate_sources(['unrelated']) == []
    refreshed = store.refresh_task(task['id'], expected_revision=task['revision'],
        source_snapshots={'unrelated': 'b' * 64}, context_snapshots={one: 1, two: 1})
    assert refreshed == task


def test_changed_one_personal_memory_refreshes_local_wrapper_not_other_paid_checkpoint(tmp_path):
    store = opened(tmp_path)
    one, two = uid(), uid()
    task = begin(store, create(store, [question('client', [one]), question('family', [two])]), {one: 1, two: 1})
    while (claim := store.claim_next(task['id'], task['run_id'], context_snapshots={one: 1, two: 1})) is not None:
        task = finish(store, task, claim, {one: 1, two: 1})
    family = task['steps'][2]
    task = store.invalidate_memories([one])[0]
    task = store.refresh_task(task['id'], expected_revision=task['revision'], source_snapshots={},
                              context_snapshots={one: 2, two: 1})
    assert task['steps'][2]['status'] == 'completed'
    assert task['steps'][2]['request_id'] == family['request_id']
    assert task['steps'][2]['artifact_version'] == family['artifact_version']
    assert task['steps'][0]['context_snapshot'] == {one: 2, two: 1}


def test_mixed_task_keeps_document_wrapper_and_targets_personal_and_document_inputs_separately(tmp_path):
    store = opened(tmp_path)
    memory = uid()
    task = create(store, [question('family', [memory]), {'id': 'document', 'question': 'Review client material.'}])
    assert 'input_kind' not in task['steps'][0] and 'input_kind' not in task['steps'][-1]
    changed = store.invalidate_sources(['document'])[0]
    assert changed['steps'][1]['status'] == 'ready' and changed['steps'][2]['status'] == 'stale'


def test_explicit_selection_revision_updates_wrapper_without_requiring_removed_memory(tmp_path):
    store = opened(tmp_path)
    old, new = uid(), uid()
    task = create(store, [question(memories=[old])])
    task = store.revise_task(task['id'], expected_revision=task['revision'], goal=task['goal'],
        constraints=task['constraints'], questions=[question(memories=[new])])
    assert task['steps'][0]['memory_ids'] == task['steps'][-1]['memory_ids'] == [new]
    task = store.refresh_task(task['id'], expected_revision=task['revision'], source_snapshots={},
                              context_snapshots={new: 1})
    assert task['steps'][0]['context_snapshot'] == {new: 1}
