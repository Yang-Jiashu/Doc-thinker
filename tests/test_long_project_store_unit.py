"""Real shared SQLite phase manifests; no providers, copied answers or new jobs."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import uuid

import pytest

from docthinker.server.long_project_store import LongProjectStore
from docthinker.server.project_task_store import (
    ProjectTaskStore, ProjectTaskConflictError, ProjectTaskError,
    ProjectTaskLimitError, ProjectTaskNotFoundError,
)


def uid():
    return str(uuid.uuid4())


def opened(path, *, boot='first', session='#00001', instance='owner-one'):
    tasks = ProjectTaskStore(path.resolve(), session_id=session, instance_id=instance, boot_id=boot)
    return tasks, LongProjectStore(tasks)


def project(store, request_id=None):
    return store.create_project(goal='Keep a long household and work project coherent.',
        acceptance_criteria=['Show sources and uncertainties.', 'No external action without confirmation.'],
        request_id=request_id or uid())


def questions():
    return [{'id': 'draft', 'question': 'Draft a usable next action; do not send it.',
             'input_kind': 'personal', 'memory_ids': []}]


def add(store, root, *, parents=None, rows=None, rid=None, title='Next phase'):
    current = store.get_project(root['id'])
    return store.add_phase(root['id'], expected_revision=current['revision'], request_id=rid or uid(),
        title=title, acceptance_criteria=['Deliver the agreed draft and disclose missing inputs.'],
        questions=questions() if rows is None else rows, parent_refs=parents or [])


def complete(tasks, task_id, sources=None, contexts=None):
    current = tasks.get_task(task_id)
    current = tasks.begin_run(task_id, expected_revision=current['revision'], request_id=uid(),
        max_calls=6, source_snapshots=sources or {}, context_snapshots=contexts or {})
    while (step := tasks.claim_next(task_id, current['run_id'], context_snapshots=contexts or {})) is not None:
        if step['kind'] == 'inventory':
            data = {'documents': [{'id': key, 'digest': digest} for key, digest in step['input_snapshot'].items()]}
        elif step['kind'] == 'report':
            data = {'step_refs': [{'step_id': row['id'], 'version': row['artifact_version']}
                                  for row in tasks.get_task(task_id)['steps'] if row['kind'] == 'analysis']}
        elif step.get('input_kind') == 'personal':
            data = {'answer_id': uid(), 'source_ids': [], 'document_ids': [],
                    'input_kind': 'personal', 'context_refs': step['context_snapshot']}
        else:
            data = {'answer_id': uid(), 'source_ids': ['source-' + step['id']],
                    'document_ids': list(step['input_snapshot'])}
        current = tasks.complete_step(task_id, step['id'], step['claim_token'], data,
                                      context_snapshots=contexts or {})
    assert current['status'] == 'completed'
    return current


def task_count(tasks):
    with sqlite3.connect(tasks.path) as db:
        return db.execute('SELECT count(*) FROM project_tasks').fetchone()[0]


def test_empty_directory_read_does_not_create_database_and_default_tasks_stay_compatible(tmp_path):
    tasks, store = opened(tmp_path)
    assert store.list_projects() == {'items': [], 'next_cursor': None}
    assert not tasks.path.exists()
    ordinary = tasks.create_task(goal='Ordinary source work', questions=[{'question': 'Read the source.'}])
    assert store.phase_for_task(ordinary['id']) is None
    assert store.parent_dependencies(ordinary['id']) == []
    assert store.dependent_tasks(ordinary['id']) == {'items': [], 'next_cursor': None}
    assert store.path == tasks.path


def test_thirty_completed_phases_reopen_and_paginate_without_lifetime_task_cap(tmp_path):
    tasks, store = opened(tmp_path)
    root, prior, ids = project(store), None, []
    for index in range(30):
        if index and index % 10 == 0:
            tasks, store = opened(tmp_path, boot='restart-' + str(index))
            tasks.recover()
        refs = [{'phase_id': prior['id'], 'report_version': 1, 'step_ids': ['draft']}] if prior else []
        phase = add(store, root, parents=refs, title='Phase ' + str(index + 1))
        complete(tasks, phase['task_id'])
        prior = store.get_phase(root['id'], phase['id'])
        assert prior['report_version'] == 1 and prior['task_status'] == 'completed'
        assert not prior['goal_verified'] and prior['handoff_status'] == 'references_only_not_delivered'
        ids.append(phase['id'])
    tasks, store = opened(tmp_path, boot='final-restart')
    seen, cursor = [], None
    while True:
        page = store.list_phases(root['id'], limit=7, cursor=cursor)
        seen.extend(row['id'] for row in page['items'])
        cursor = page['next_cursor']
        if cursor is None:
            break
    assert seen == ids and task_count(tasks) == 30
    assert store.get_project(root['id'])['phase_count'] == 30
    assert store.parent_dependencies(prior['task_id'])[0]['phase_id'] == ids[-2]
    with sqlite3.connect(tasks.path) as db:
        assert db.execute('SELECT count(*) FROM long_project_phase_parents').fetchone()[0] == 29
    assert not list(tmp_path.glob('*long*.db'))


def test_project_and_phase_request_replay_are_immutable_and_atomic(tmp_path):
    tasks, store = opened(tmp_path)
    rid = uid()
    root = project(store, rid)
    assert project(store, rid)['id'] == root['id']
    phase_request = uid()
    body = dict(expected_revision=root['revision'], request_id=phase_request, title='Private phase',
        acceptance_criteria=['Do not send anything.'], questions=questions())
    def create(_):
        _, current = opened(tmp_path)
        return current.add_phase(root['id'], **body)
    with ThreadPoolExecutor(max_workers=2) as pool:
        phases = list(pool.map(create, range(2)))
    assert phases[0]['id'] == phases[1]['id'] and task_count(tasks) == 1
    assert sum(row.get('request_replayed') is True for row in phases) == 1
    with pytest.raises(ProjectTaskConflictError):
        store.add_phase(root['id'], **{**body, 'title': 'Changed approval'})
    with pytest.raises(ProjectTaskConflictError):
        store.add_phase(root['id'], **{**body, 'request_id': uid()})
    assert task_count(tasks) == 1


@pytest.mark.parametrize('other', [{'session': '#00002'}, {'instance': 'other-instance'}])
def test_manifest_phase_cursor_and_request_do_not_cross_owner_scope(tmp_path, other):
    _, store = opened(tmp_path)
    rid = uid()
    root = project(store, rid)
    first = add(store, root)
    add(store, root)
    page = store.list_phases(root['id'], limit=1)
    _, foreign = opened(tmp_path, **other)
    assert foreign.list_projects()['items'] == []
    with pytest.raises(ProjectTaskNotFoundError):
        foreign.get_project(root['id'])
    with pytest.raises(ProjectTaskNotFoundError):
        foreign.get_phase(root['id'], first['id'])
    with pytest.raises(ProjectTaskConflictError):
        project(foreign, rid)
    with pytest.raises(ProjectTaskError):
        foreign.list_phases(root['id'], cursor=page['next_cursor'])


def test_parent_must_be_current_completed_report_in_same_root(tmp_path):
    tasks, store = opened(tmp_path)
    root = project(store)
    parent = add(store, root)
    with pytest.raises(ProjectTaskConflictError):
        add(store, root, parents=[{'phase_id': parent['id'], 'report_version': 1}])
    complete(tasks, parent['task_id'])
    other = project(store)
    with pytest.raises(ProjectTaskNotFoundError):
        add(store, other, parents=[{'phase_id': parent['id'], 'report_version': 1}])
    with pytest.raises(ProjectTaskConflictError):
        add(store, root, parents=[{'phase_id': parent['id'], 'report_version': 2}])
    with pytest.raises(ProjectTaskError):
        add(store, root, parents=[{'phase_id': parent['id'], 'report_version': 1, 'step_ids': ['missing']}])
    assert task_count(tasks) == 1


def test_unknown_accepted_phase_blocks_new_phase_even_after_restart(tmp_path):
    tasks, store = opened(tmp_path)
    root, phase = project(store), None
    phase = add(store, root)
    task = tasks.get_task(phase['task_id'])
    task = tasks.begin_run(task['id'], expected_revision=task['revision'], request_id=uid(),
                           max_calls=1, source_snapshots={})
    inventory = tasks.claim_next(task['id'], task['run_id'])
    tasks.complete_step(task['id'], 'inventory', inventory['claim_token'], {'documents': []})
    tasks.claim_next(task['id'], task['run_id'])
    tasks, store = opened(tmp_path, boot='restarted')
    tasks.recover()
    assert tasks.get_task(task['id'])['steps'][1]['reason'] == 'provider_outcome_unknown'
    with pytest.raises(ProjectTaskConflictError):
        add(store, root)
    assert task_count(tasks) == 1


def test_selected_parent_branch_survives_unselected_change_but_not_selected_change(tmp_path):
    tasks, store = opened(tmp_path)
    root = project(store)
    parent = add(store, root, rows=[
        {'id': 'a', 'question': 'Review alpha.', 'document_ids': ['alpha']},
        {'id': 'b', 'question': 'Review beta.', 'document_ids': ['beta']}])
    complete(tasks, parent['task_id'], {'alpha': 'a' * 64, 'beta': 'b' * 64})
    child = add(store, root, parents=[{'phase_id': parent['id'], 'report_version': 1, 'step_ids': ['a']}])
    exact = store.parent_dependencies(child['task_id'])
    assert exact[0]['step_versions'] == {'a': 1}
    assert store.dependent_tasks(parent['task_id'])['items'][0]['task_id'] == child['task_id']
    tasks.invalidate_sources(['beta'])
    assert store.parent_dependencies(child['task_id']) == exact
    tasks.invalidate_sources(['alpha'])
    with pytest.raises(ProjectTaskConflictError):
        store.parent_dependencies(child['task_id'])


def test_phase_customer_review_binds_exact_report_and_sources_then_stales(tmp_path):
    tasks, store = opened(tmp_path)
    root = project(store)
    note = uid()
    phase = add(store, root, rows=[{**questions()[0], 'memory_ids': [note]}])
    complete(tasks, phase['task_id'], contexts={note: 1})
    root = store.get_project(root['id'])
    rid = uid()
    reviewed = store.review_phase(root['id'], phase['id'], expected_revision=root['revision'],
        request_id=rid, report_version=1, decision='accepted')
    assert reviewed['review_status'] == 'accepted'
    assert not reviewed['review']['goal_verified'] and not reviewed['review']['semantic_verified']
    assert reviewed['review']['binding']['inputs'][0]['context_snapshot'] == {note: 1}
    repeated = store.review_phase(root['id'], phase['id'], expected_revision=root['revision'],
        request_id=rid, report_version=1, decision='accepted')
    assert repeated['request_replayed'] and repeated['review']['id'] == reviewed['review']['id']
    with pytest.raises(ProjectTaskConflictError):
        store.review_phase(root['id'], phase['id'], expected_revision=root['revision'],
            request_id=rid, report_version=1, decision='rejected')
    tasks.invalidate_memories([note])
    stale = store.get_phase(root['id'], phase['id'])
    assert stale['review_status'] == 'stale' and stale['report_version'] is None


def test_latest_manual_review_is_insertion_order_even_at_identical_clock(tmp_path):
    tasks, store = opened(tmp_path)
    tasks.clock = lambda: 1700000000.0
    root = project(store)
    phase = add(store, root)
    complete(tasks, phase['task_id'])
    for decision in ['accepted', 'rejected', 'accepted', 'rejected']:
        root = store.get_project(root['id'])
        result = store.review_phase(root['id'], phase['id'], expected_revision=root['revision'],
            request_id=uid(), report_version=1, decision=decision)
        assert result['review_status'] == decision


def test_project_acceptance_is_version_bound_manual_and_new_phase_invalidates_it(tmp_path):
    tasks, store = opened(tmp_path)
    root = project(store)
    phase = add(store, root)
    complete(tasks, phase['task_id'])
    root = store.get_project(root['id'])
    with pytest.raises(ProjectTaskConflictError):
        store.review_project(root['id'], expected_revision=root['revision'], request_id=uid(),
                             phase_reports=[], decision='accepted')
    accepted = store.review_project(root['id'], expected_revision=root['revision'], request_id=uid(),
        phase_reports=[{'phase_id': phase['id'], 'report_version': 1}], decision='accepted')
    assert accepted['status'] == accepted['review_status'] == 'accepted'
    assert accepted['review']['reviewer'] == 'user' and not accepted['goal_verified']
    add(store, root)
    current = store.get_project(root['id'])
    assert current['status'] == 'active' and current['review_status'] == 'stale'


def test_same_database_byte_quota_rolls_back_task_and_phase_together(tmp_path, monkeypatch):
    tasks, store = opened(tmp_path)
    root = project(store)
    from docthinker.server import project_task_store
    monkeypatch.setattr(project_task_store, 'MAX_STORAGE_BYTES', 1)
    with pytest.raises(ProjectTaskLimitError):
        add(store, root)
    assert task_count(tasks) == 0
    assert store.get_project(root['id'])['phase_count'] == 0


@pytest.mark.parametrize('criteria', [[], [''], ['same', 'same'], [True], ['x' * 1001]])
def test_acceptance_criteria_are_explicit_bounded_data_not_a_claim_of_success(tmp_path, criteria):
    _, store = opened(tmp_path)
    with pytest.raises(ProjectTaskError):
        store.create_project(goal='Explicit goal', acceptance_criteria=criteria, request_id=uid())
