"""Offline bounded topology revisions preserve durable work and payment identity."""
from copy import deepcopy
import hashlib
import sqlite3
import uuid

import pytest

import docthinker.server.project_task_store as module
from docthinker.server.project_task_store import (
    ProjectTaskStore, ProjectTaskError, ProjectTaskConflictError,
    ProjectTaskLimitError, ProjectTaskNotFoundError,
)


DOCS = {'work': 'a' * 64, 'family': 'b' * 64}


def uid():
    return str(uuid.uuid4())


def opened(path, **kwargs):
    return ProjectTaskStore(path.resolve(), session_id=kwargs.pop('session_id', '#00001'),
                            instance_id=kwargs.pop('instance_id', 'replan-worker'),
                            boot_id=kwargs.pop('boot_id', 'boot-one'), **kwargs)


def questions():
    return [
        {'id': 'agenda', 'title': 'Agenda', 'question': 'Summarize the work agenda.', 'document_ids': ['work']},
        {'id': 'reply', 'title': 'Reply', 'question': 'Draft the reply, do not send.', 'document_ids': ['work'], 'depends_on': ['agenda']},
        {'id': 'home', 'title': 'Home', 'question': 'Plan the family follow-up.', 'document_ids': ['family']},
    ]


def make(store, rows=None):
    return store.create_task(goal='Prepare work and family follow-ups', constraints='Draft only; do not send.',
                             questions=questions() if rows is None else rows)


def finish(store, task, claim, contexts=None, extra=None):
    if claim['kind'] == 'inventory':
        data = {'documents': [{'id': key, 'digest': digest} for key, digest in claim['input_snapshot'].items()]}
    elif claim['kind'] == 'report':
        data = {'step_refs': [{'step_id': row['id'], 'version': row['artifact_version']}
                             for row in store.get_task(task['id'])['steps'] if row['kind'] == 'analysis']}
    else:
        data = {'answer_id': uid(), 'source_ids': [], 'document_ids': list(claim['input_snapshot'])}
        if claim.get('input_kind') == 'personal':
            data.update(input_kind='personal', context_refs=claim['context_snapshot'])
        data.update(extra or {})
    return store.complete_step(task['id'], claim['id'], claim['claim_token'], data, context_snapshots=contexts)


def done(store, task, contexts=None):
    task = store.begin_run(task['id'], expected_revision=task['revision'], max_calls=6,
                           source_snapshots=DOCS, context_snapshots=contexts)
    for _ in range(8):
        claim = store.claim_next(task['id'], task['run_id'], context_snapshots=contexts)
        if claim is None:
            break
        task = finish(store, task, claim, contexts)
    assert task['status'] == 'completed'
    return task


def proposal(store, task, rows, **changes):
    data = {'expected_revision': task['revision'], 'goal': task['goal'],
            'constraints': task['constraints'], 'questions': rows, **changes}
    return data, store.preview_replan(task['id'], **data)


def apply(store, task, rows, **changes):
    data, preview = proposal(store, task, rows, **changes)
    return store.replan_task(task['id'], proposal_sha256=preview['proposal_sha256'], **data)


def by_id(task):
    return {row['id']: row for row in task['steps']}


def test_preview_is_read_only_and_shows_union_of_old_and_new_downstream(tmp_path):
    store = opened(tmp_path)
    task = done(store, make(store))
    before = hashlib.sha256(store.path.read_bytes()).hexdigest()
    rows = questions()
    rows[0]['question'] = 'Describe the newly agreed work agenda.'
    rows[1]['depends_on'] = []  # Removing the old edge must not preserve its old result.
    rows.append({'id': 'check', 'question': 'Check the reply.', 'depends_on': ['reply']})
    _, view = proposal(store, task, rows)
    assert view['affected'] == ['agenda', 'check', 'reply', 'report']
    assert view['added'] == ['check'] and view['unaffected_completed'] == ['home']
    assert view['execution_authorized'] is False and view['blocked'] is False
    assert hashlib.sha256(store.path.read_bytes()).hexdigest() == before
    assert store.get_task(task['id']) == task and len(store.definition_history(task['id'])) == 1


def test_old_database_reads_and_previews_without_writing_new_migration_columns(tmp_path):
    store = opened(tmp_path)
    task = make(store)
    with sqlite3.connect(store.path) as db:
        for field in ['retired', 'input_kind', 'memory_ids', 'context_snapshot']:
            db.execute('ALTER TABLE project_task_steps DROP COLUMN ' + field)
    before = store.path.read_bytes()
    assert opened(tmp_path).get_task(task['id']) == task
    data, view = proposal(store, task, questions())
    assert view['modified'] is False and store.path.read_bytes() == before
    assert store.replan_task(task['id'], proposal_sha256=view['proposal_sha256'], **data) == task


def test_replan_preserves_independent_completed_answer_and_request_exactly(tmp_path):
    store = opened(tmp_path)
    task = done(store, make(store))
    old = by_id(task)
    home = store.get_artifact(task['id'], 'home')
    rows = questions()
    rows[0]['question'] = 'Use the revised work agenda.'
    changed = apply(store, task, rows)
    assert by_id(changed)['home'] == old['home']
    assert store.get_artifact(task['id'], 'home') == home
    assert by_id(changed)['agenda']['request_id'] == old['agenda']['request_id']
    assert by_id(changed)['reply']['request_id'] == old['reply']['request_id']
    assert {key: row['status'] for key, row in by_id(changed).items()} == {
        'inventory': 'completed', 'agenda': 'stale', 'reply': 'stale', 'home': 'completed', 'report': 'stale'}
    assert changed['run_id'] is None and changed['run_remaining'] == 0
    assert not store.get_artifact(task['id'], 'agenda')['available']
    assert not store.get_artifact(task['id'], 'report')['available']
    assert store.claim_next(task['id'], task['run_id']) is None


def test_explicit_refresh_and_new_grant_execute_only_affected_steps(tmp_path):
    store = opened(tmp_path)
    task = done(store, make(store))
    old = by_id(task)
    rows = questions()
    rows[0]['question'] = 'Use the revised work agenda.'
    changed = apply(store, task, rows)
    ready = store.refresh_task(task['id'], expected_revision=changed['revision'], source_snapshots=DOCS)
    assert by_id(ready)['home'] == old['home']
    assert by_id(ready)['agenda']['request_id'] != old['agenda']['request_id']
    ready = store.begin_run(task['id'], expected_revision=ready['revision'], max_calls=2, source_snapshots=DOCS)
    seen = []
    for _ in range(4):
        claim = store.claim_next(task['id'], ready['run_id'])
        if claim is None:
            break
        seen.append(claim['id'])
        ready = finish(store, ready, claim)
    assert seen == ['agenda', 'reply', 'report'] and ready['status'] == 'completed'
    assert store.get_artifact(task['id'], 'home')['version'] == 1
    assert store.get_artifact(task['id'], 'agenda')['version'] == 2


def test_retiring_completed_node_invalidates_real_descendants_and_never_reuses_id(tmp_path):
    store = opened(tmp_path)
    task = done(store, make(store))
    rows = questions()[1:]
    rows[0]['depends_on'] = []
    changed = apply(store, task, rows)
    assert set(by_id(changed)) == {'inventory', 'reply', 'home', 'report'}
    assert by_id(changed)['reply']['status'] == 'stale'
    assert by_id(changed)['home'] == by_id(task)['home']
    assert not store.get_artifact(task['id'], 'agenda', 1)['available']
    assert store.artifact_history(task['id'], 'agenda')[0]['status'] == 'stale'
    reopened = opened(tmp_path, boot_id='new-boot')
    with pytest.raises(ProjectTaskConflictError, match='Retired'):
        proposal(reopened, changed, questions())
    assert len(reopened.definition_history(task['id'])) == 2


def test_new_node_has_new_request_and_no_artifact_but_no_paid_grant(tmp_path):
    store = opened(tmp_path)
    task = done(store, make(store))
    rows = questions() + [{'id': 'approval', 'question': 'Prepare a draft approval note.', 'depends_on': ['reply']}]
    changed = apply(store, task, rows)
    assert by_id(changed)['approval']['request_id'] not in {row['request_id'] for row in task['steps']}
    assert by_id(changed)['approval']['status'] == 'ready' and changed['status'] == 'stale'
    assert all(by_id(changed)[key] == by_id(task)[key] for key in ['agenda', 'reply', 'home'])
    with pytest.raises(ProjectTaskNotFoundError):
        store.get_artifact(task['id'], 'approval')


def test_diamond_downstream_closure_is_bounded_and_preserves_disconnected_node(tmp_path):
    store = opened(tmp_path)
    rows = [{'id': 'a', 'question': 'Work root'},
            {'id': 'b', 'question': 'Work branch one', 'depends_on': ['a']},
            {'id': 'c', 'question': 'Work branch two', 'depends_on': ['a']},
            {'id': 'd', 'question': 'Join branches', 'depends_on': ['b', 'c']},
            {'id': 'e', 'question': 'Final work draft', 'depends_on': ['d']},
            {'id': 'f', 'question': 'Independent family draft'}]
    task = done(store, make(store, rows))
    revised = deepcopy(rows)
    revised[1]['question'] = 'Revised branch one'
    _, view = proposal(store, task, revised)
    assert view['affected'] == ['b', 'd', 'e', 'report']
    changed = apply(store, task, revised)
    assert all(by_id(changed)[key] == by_id(task)[key] for key in ['a', 'c', 'f'])


def test_title_and_order_only_change_report_not_analysis_identity(tmp_path):
    store = opened(tmp_path)
    task = done(store, make(store))
    rows = questions()[::-1]
    rows[0]['title'] = 'Private family follow-up'
    _, view = proposal(store, task, rows)
    assert view['changed'] == [] and view['affected'] == ['report']
    changed = apply(store, task, rows)
    for key in ['agenda', 'reply', 'home']:
        old, new = by_id(task)[key], by_id(changed)[key]
        assert {name: old[name] for name in ['request_id', 'artifact_version', 'generation', 'status']} == {
            name: new[name] for name in ['request_id', 'artifact_version', 'generation', 'status']}
    assert [row['id'] for row in changed['steps']] == ['inventory', 'home', 'reply', 'agenda', 'report']


@pytest.mark.parametrize('field,value', [('goal', 'Prepare a different long-term objective'), ('constraints', 'Draft for private review only.')])
def test_goal_or_constraint_change_invalidates_all_interpretations(tmp_path, field, value):
    store = opened(tmp_path)
    task = done(store, make(store))
    _, view = proposal(store, task, questions(), **{field: value})
    assert view['changed'] == ['agenda', 'home', 'reply']
    assert view['affected'] == ['agenda', 'home', 'reply', 'report']


@pytest.mark.parametrize('state,reason', [('running', 'active_step'), ('unknown', 'unresolved_outcome'), ('queued', 'active_run'), ('cancelled', 'task_cancelled')])
def test_active_or_uncertain_work_blocks_topology_change_without_rotating_identity(tmp_path, state, reason):
    store = opened(tmp_path)
    task = make(store)
    if state == 'cancelled':
        task = store.cancel_task(task['id'], expected_revision=task['revision'])
    else:
        task = store.begin_run(task['id'], expected_revision=task['revision'], max_calls=3, source_snapshots=DOCS)
        if state != 'queued':
            task = finish(store, task, store.claim_next(task['id'], task['run_id']))
            claim = store.claim_next(task['id'], task['run_id'])
            if state == 'unknown':
                task = store.block_step(task['id'], claim['id'], claim['claim_token'], reason='unknown')
                task = store.pause_task(task['id'], expected_revision=task['revision'])
            else:
                task = store.get_task(task['id'])
    rows = questions()
    rows[0]['question'] += ' Update.'
    data, view = proposal(store, task, rows)
    assert view['blocked_reason'] == reason and view['blocked'] is True
    with pytest.raises(ProjectTaskConflictError):
        store.replan_task(task['id'], proposal_sha256=view['proposal_sha256'], **data)
    assert store.get_task(task['id']) == task
    assert len(store.definition_history(task['id'])) == 1


def test_paused_known_safe_checkpoint_may_replan_but_does_not_resume(tmp_path):
    store = opened(tmp_path)
    task = make(store)
    task = store.begin_run(task['id'], expected_revision=task['revision'], max_calls=1, source_snapshots=DOCS)
    task = finish(store, task, store.claim_next(task['id'], task['run_id']))
    task = store.pause_task(task['id'], expected_revision=task['revision'])
    changed = apply(store, task, questions() + [{'id': 'extra', 'question': 'Plan an extra check.'}])
    assert changed['status'] == 'stale' and changed['run_id'] is None
    assert by_id(changed)['inventory']['status'] == 'completed'


def test_proposal_digest_and_revision_are_both_required_and_atomic(tmp_path):
    store = opened(tmp_path)
    task = make(store)
    rows = questions()
    rows[0]['question'] += ' Revised.'
    data, view = proposal(store, task, rows)
    for kwargs in ({**data, 'goal': 'Different goal'}, {**data, 'expected_revision': task['revision'] + 1}):
        with pytest.raises(ProjectTaskConflictError):
            store.replan_task(task['id'], proposal_sha256=view['proposal_sha256'], **kwargs)
    with pytest.raises(ProjectTaskError):
        store.replan_task(task['id'], proposal_sha256=True, **data)
    assert store.get_task(task['id']) == task
    assert len(store.definition_history(task['id'])) == 1


def test_noop_is_exact_without_revision_or_definition_change(tmp_path):
    store = opened(tmp_path)
    task = done(store, make(store))
    data, view = proposal(store, task, questions())
    assert not view['modified'] and view['affected'] == []
    assert store.replan_task(task['id'], proposal_sha256=view['proposal_sha256'], **data) == task
    assert len(store.definition_history(task['id'])) == 1


@pytest.mark.parametrize('rows', [
    [{'id': 'a', 'question': 'One', 'depends_on': ['missing']}],
    [{'id': 'a', 'question': 'One', 'depends_on': ['b']}, {'id': 'b', 'question': 'Two', 'depends_on': ['a']}],
    [{'id': 'q' + str(index), 'question': 'Work'} for index in range(7)],
    [{'id': 'report', 'question': 'Work'}],
])
def test_invalid_topology_has_no_side_effect(tmp_path, rows):
    store = opened(tmp_path)
    task = make(store)
    with pytest.raises(ProjectTaskError):
        proposal(store, task, rows)
    assert store.get_task(task['id']) == task


def test_definition_capacity_fails_closed_and_never_evicts_history(tmp_path, monkeypatch):
    store = opened(tmp_path)
    task = make(store)
    monkeypatch.setattr(module, 'MAX_ARTIFACT_VERSIONS', 2)
    rows = questions()
    rows[0]['title'] = 'First revision'
    task = apply(store, task, rows)
    rows[0]['title'] = 'Second revision'
    data, view = proposal(store, task, rows)
    assert view['blocked_reason'] == 'definition_capacity'
    with pytest.raises(ProjectTaskConflictError):
        store.replan_task(task['id'], proposal_sha256=view['proposal_sha256'], **data)
    assert len(store.definition_history(task['id'])) == 2


def test_personal_union_changes_inventory_without_invalidating_independent_paid_node(tmp_path):
    store = opened(tmp_path)
    first, second, third = uid(), uid(), uid()
    rows = [{'id': 'work', 'question': 'Draft my work follow-up.', 'input_kind': 'personal', 'memory_ids': [first]},
            {'id': 'home', 'question': 'Draft my family follow-up.', 'input_kind': 'personal', 'memory_ids': [second]}]
    task = done(store, make(store, rows), {first: 1, second: 2})
    revised = deepcopy(rows)
    revised[0]['memory_ids'] = [third]
    _, view = proposal(store, task, revised)
    assert view['affected'] == ['inventory', 'report', 'work']
    changed = apply(store, task, revised)
    assert by_id(changed)['home'] == by_id(task)['home']
    assert by_id(changed)['inventory']['memory_ids'] == [third, second]
    assert by_id(changed)['work']['input_snapshot'] == {}
    assert by_id(changed)['home']['context_snapshot'] == {second: 2}


def test_tenant_or_instance_cannot_preview_or_apply_foreign_task(tmp_path):
    store = opened(tmp_path)
    task = make(store)
    for other in [opened(tmp_path, session_id='#00002'), opened(tmp_path, instance_id='different')]:
        with pytest.raises(ProjectTaskNotFoundError):
            proposal(other, task, questions())


def test_shared_insert_is_in_same_transaction_and_rolls_back_with_parent(tmp_path):
    store = opened(tmp_path)
    task_id = uid()
    with pytest.raises(RuntimeError):
        with store._db(write=True) as db:
            db.execute('CREATE TABLE test_parent_manifest (id TEXT PRIMARY KEY)')
            db.execute('INSERT INTO test_parent_manifest VALUES (?)', ('parent',))
            result = store._insert_task(db, task_id=task_id, goal='Prepare a work decision', questions=questions(),
                                        created_at=1234, enforce_task_limits=False)
            assert result['id'] == task_id and result['created_at'] == 1234 and db.in_transaction
            raise RuntimeError('Abort parent transaction')
    assert store.list_tasks() == []
    with store._db(write=True) as db:
        db.execute('CREATE TABLE test_parent_manifest (id TEXT PRIMARY KEY)')
        db.execute('INSERT INTO test_parent_manifest VALUES (?)', ('parent',))
        store._insert_task(db, task_id=task_id, goal='Prepare a work decision', questions=questions())
    assert store.get_task(task_id)['status'] == 'ready'


def test_shared_insert_default_capacity_preserved_and_explicit_long_scope_bypass_only_counts(tmp_path, monkeypatch):
    store = opened(tmp_path)
    monkeypatch.setattr(module, 'MAX_PROJECT_TASKS', 1)
    make(store)
    with pytest.raises(ProjectTaskLimitError):
        make(store)
    with store._db(write=True) as db:
        second = store._insert_task(db, goal='Phase two', questions=questions(), enforce_task_limits=False)
    assert store.get_task(second['id'])['goal'] == 'Phase two'
    monkeypatch.setattr(module, 'MAX_STORAGE_BYTES', 1)
    with pytest.raises(ProjectTaskLimitError):
        with store._db(write=True) as db:
            store._insert_task(db, goal='Must roll back', questions=questions(), enforce_task_limits=False)
    with store._db() as db:
        assert db.execute('SELECT count(*) FROM project_tasks').fetchone()[0] == 2


def test_thirty_long_phases_do_not_consume_standalone_task_quota(tmp_path, monkeypatch):
    from docthinker.server.long_project_store import LongProjectStore
    store = opened(tmp_path)
    long = LongProjectStore(store)
    root = long.create_project(goal='Long work and family project', acceptance_criteria=['Draft only.'], request_id=uid())
    for index in range(30):
        current = long.get_project(root['id'])
        long.add_phase(root['id'], expected_revision=current['revision'], request_id=uid(),
                       title='Phase ' + str(index), acceptance_criteria=['Prepare a draft.'], questions=questions())
    monkeypatch.setattr(module, 'MAX_PROJECT_TASKS', 1)
    monkeypatch.setattr(module, 'MAX_TASKS', 1)
    assert make(store)['status'] == 'ready'
    with pytest.raises(ProjectTaskLimitError):
        make(store)


def test_session_cleanup_removes_long_metadata_all_instances_not_other_session(tmp_path):
    from docthinker.server.long_project_store import LongProjectStore
    owners = [opened(tmp_path), opened(tmp_path, instance_id='second'), opened(tmp_path, session_id='#00002')]
    roots, phases = [], []
    for store in owners:
        long = LongProjectStore(store)
        root = long.create_project(goal='Private personal follow-up', acceptance_criteria=['Draft only.'], request_id=uid())
        phase = long.add_phase(root['id'], expected_revision=root['revision'], request_id=uid(), title='First phase',
                               acceptance_criteria=['Prepare draft.'], questions=questions())
        roots.append(root)
        phases.append(phase)
        with store._db(write=True) as db:
            db.execute('INSERT INTO long_project_reviews VALUES(?,?,?,?,?,?,?)',
                       (uid(), root['id'], phase['id'], 'accepted', '{}', 'a' * 64, 1234))
            db.execute('INSERT INTO long_project_phase_parents VALUES(?,?,?,?,?,?)',
                       (phase['id'], phase['id'], phase['task_id'], 1, 'agenda', 1))
    owners[0].delete_session_metadata('#00001')
    for owner, root in zip(owners[:2], roots[:2]):
        with pytest.raises(ProjectTaskNotFoundError):
            LongProjectStore(owner).get_project(root['id'])
    assert LongProjectStore(owners[2]).get_project(roots[2]['id'])['phase_count'] == 1
    with owners[2]._db() as db:
        for table in ['long_projects', 'long_project_phases', 'long_project_phase_parents', 'long_project_reviews', 'long_project_requests']:
            expected = 2 if table == 'long_project_requests' else 1
            assert db.execute('SELECT count(*) FROM ' + table).fetchone()[0] == expected
        assert db.execute('SELECT count(*) FROM project_tasks').fetchone()[0] == 1


@pytest.mark.parametrize('options', [{'enforce_task_limits': 0}, {'created_at': True}, {'created_at': float('inf')}, {'created_at': -1}, {'task_id': 'bad'}])
def test_shared_insert_validates_caller_boundary(tmp_path, options):
    store = opened(tmp_path)
    with pytest.raises(ProjectTaskError):
        with store._db(write=True) as db:
            store._insert_task(db, goal='Work', questions=questions(), **options)
    assert store.list_tasks() == []


def test_shared_insert_rejects_foreign_connection_and_no_transaction(tmp_path):
    store = opened(tmp_path)
    with sqlite3.connect(':memory:') as db:
        with pytest.raises(ProjectTaskError):
            store._insert_task(db, goal='Work', questions=questions())
        db.execute('BEGIN')
        with pytest.raises(ProjectTaskError):
            store._insert_task(db, goal='Work', questions=questions())


def phase_receipt():
    return {'required': 1, 'used': 1, 'full': 1, 'excerpted': 0, 'omitted': 0,
            'refs': [{'phase_id': uid(), 'task_id': uid(), 'report_version': 1, 'step_id': 'work', 'version': 2}]}


def test_phase_receipt_is_reference_only_separate_from_local_dependency_counts(tmp_path):
    store = opened(tmp_path)
    task = store.begin_run(make(store)['id'], expected_revision=1, max_calls=3, source_snapshots=DOCS)
    task = finish(store, task, store.claim_next(task['id'], task['run_id']))
    claim = store.claim_next(task['id'], task['run_id'])
    receipt = phase_receipt()
    local = {'required': 0, 'used': 0, 'full': 0, 'excerpted': 0, 'omitted': 0}
    task = finish(store, task, claim, extra={'phase_handoff': receipt, 'handoff': local})
    assert by_id(task)['agenda']['phase_handoff'] == {key: value for key, value in receipt.items() if key != 'refs'}
    assert by_id(task)['agenda']['handoff'] == local
    assert store.get_artifact(task['id'], 'agenda')['references']['phase_handoff'] == receipt


@pytest.mark.parametrize('mutate', [
    lambda value: value.update(omitted=1, used=0, full=0),
    lambda value: value.update(required=True),
    lambda value: value['refs'][0].update(step_id='report'),
    lambda value: value['refs'][0].update(phase_id='foreign'),
    lambda value: value['refs'][0].update(version=True),
    lambda value: value['refs'][0].update(report_version=33),
    lambda value: value['refs'][0].update(body='Source copy'),
    lambda value: value.update(refs=value['refs'] * 2, required=2, used=2, full=2),
])
def test_phase_receipt_rejects_omission_unbounded_or_body_metadata(tmp_path, mutate):
    store = opened(tmp_path)
    task = store.begin_run(make(store)['id'], expected_revision=1, max_calls=3, source_snapshots=DOCS)
    task = finish(store, task, store.claim_next(task['id'], task['run_id']))
    claim = store.claim_next(task['id'], task['run_id'])
    receipt = phase_receipt()
    mutate(receipt)
    with pytest.raises(ProjectTaskError):
        finish(store, task, claim, extra={'phase_handoff': receipt})
    assert by_id(store.get_task(task['id']))['agenda']['status'] == 'running'
