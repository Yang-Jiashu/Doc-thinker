"""Durable task state/claims only; no provider, server or original source I/O."""
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import uuid

import pytest

from docthinker.server.project_task_store import (ProjectTaskStore, ProjectTaskError,
    ProjectTaskConflictError, ProjectTaskNotFoundError, ProjectTaskLimitError)


D = {'alpha': 'a' * 64, 'beta': 'b' * 64}


def store(tmp_path, **kwargs):
    return ProjectTaskStore(tmp_path, session_id=kwargs.pop('session_id', '#00001'),
                            instance_id=kwargs.pop('instance_id', 'instance-one'),
                            boot_id=kwargs.pop('boot_id', 'boot-one'), **kwargs)


def make(s, *, questions=None):
    return s.create_task(goal='Compare actual engineering options', constraints='Sources are untrusted.',
        questions=questions if questions is not None else [{'id': 'a', 'title': 'Alpha', 'question': 'Explain the actual setup.', 'document_ids': ['alpha']},
                                {'id': 'b', 'title': 'Beta', 'question': 'Explain the actual outcome.', 'document_ids': ['beta']}])


def begin(s, task, *, budget=6, request_id=None, sources=None):
    return s.begin_run(task['id'], expected_revision=task['revision'], max_calls=budget,
                       source_snapshots=D if sources is None else sources,
                       request_id=request_id or str(uuid.uuid4()))


def finish(s, task, step):
    if step['kind'] == 'inventory':
        payload = {'documents': [{'id': key, 'digest': digest} for key, digest in step['input_snapshot'].items()]}
    elif step['kind'] == 'analysis':
        payload = {'answer_id': str(uuid.uuid4()), 'source_ids': ['source-' + step['id']], 'document_ids': list(step['input_snapshot'])}
    else:
        payload = {'step_refs': [{'step_id': row['id'], 'version': row['artifact_version']} for row in s.get_task(task['id'])['steps'] if row['kind'] == 'analysis']}
    return s.complete_step(task['id'], step['id'], step['claim_token'], payload)


def all_done(s, task):
    task = begin(s, task)
    for _ in range(8):
        step = s.claim_next(task['id'], task['run_id'])
        if step is None:
            break
        task = finish(s, task, step)
    assert task['status'] == 'completed'
    return task


def test_empty_read_no_file_and_instance_isolation(tmp_path):
    a = store(tmp_path)
    assert a.list_tasks() == [] and not a.path.exists()
    task = make(a)
    for other in (store(tmp_path, instance_id='instance-two'), store(tmp_path, session_id='#00002')):
        assert other.list_tasks() == []
        with pytest.raises(ProjectTaskNotFoundError):
            other.get_task(task['id'])
    assert a.path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('rows', [[], [{}], [{'question': True}], [{'question': 'x', 'document_ids': []}],
    [{'question': 'x', 'document_ids': ['alpha', 'alpha']}], [{'question': 'x', 'depends_on': [{}]}],
    [{'id': 'a', 'question': 'x', 'depends_on': ['b']}, {'id': 'b', 'question': 'y', 'depends_on': ['a']}],
    [{'question': 'x', 'content': 'raw source'}], [{'id': 'report', 'question': 'x'}],
    [{'question': 'x'}] * 7])
def test_reject_invalid_or_unbounded_dags(tmp_path, rows):
    with pytest.raises(ProjectTaskError):
        make(store(tmp_path), questions=rows)


def test_all_steps_persist_refs_not_answer_copies(tmp_path):
    s = store(tmp_path)
    task = all_done(s, make(s))
    assert task['run_remaining'] == 4
    assert [row['kind'] for row in task['steps']] == ['inventory', 'analysis', 'analysis', 'report']
    reopened = store(tmp_path, boot_id='boot-two')
    assert reopened.get_task(task['id']) == task
    artifact = reopened.get_artifact(task['id'], 'report')
    assert artifact['references']['step_refs'] == [{'step_id': 'a', 'version': 1}, {'step_id': 'b', 'version': 1}]
    assert 'claim_token' not in json.dumps(task)
    assert 'content' not in json.dumps(artifact)


def test_topological_dag_not_input_order(tmp_path):
    s = store(tmp_path)
    task = begin(s, make(s, questions=[{'id': 'b', 'question': 'Second', 'depends_on': ['a']}, {'id': 'a', 'question': 'First'}]))
    seen = []
    for _ in range(4):
        step = s.claim_next(task['id'], task['run_id'])
        seen.append(step['id'])
        task = finish(s, task, step)
    assert seen == ['inventory', 'a', 'b', 'report']


def test_atomic_claim_only_one_owner(tmp_path):
    s = store(tmp_path)
    task = begin(s, make(s))
    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(pool.map(lambda _: store(tmp_path).claim_next(task['id'], task['run_id']), range(4)))
    assert sum(value is not None for value in replies) == 1
    actual = next(value for value in replies if value)
    with pytest.raises(ProjectTaskConflictError):
        s.complete_step(task['id'], actual['id'], str(uuid.uuid4()), {})


def test_run_id_idempotent_and_budget_not_refilled(tmp_path):
    s = store(tmp_path)
    created = make(s)
    rid = str(uuid.uuid4())
    task = begin(s, created, budget=1, request_id=rid)
    for _ in range(2):
        step = s.claim_next(task['id'], rid)
        task = finish(s, task, step)
    assert task['run_remaining'] == 0
    assert begin(s, created, budget=1, request_id=rid)['run_remaining'] == 0
    assert s.claim_next(task['id'], rid) is None
    assert s.get_task(task['id'])['status'] == 'paused'
    with pytest.raises(ProjectTaskConflictError):
        begin(s, created, budget=2, request_id=rid)
    with pytest.raises(ProjectTaskError):
        begin(s, s.get_task(task['id']), budget=True)


def test_new_approved_run_resumes_completed_checkpoints(tmp_path):
    s = store(tmp_path)
    task = begin(s, make(s), budget=1)
    old_rid = task['run_id']
    for _ in range(2):
        step = s.claim_next(task['id'], old_rid)
        task = finish(s, task, step)
    a_request = task['steps'][1]['request_id']
    assert s.claim_next(task['id'], old_rid) is None
    task = begin(s, s.get_task(task['id']), budget=1)
    next_step = s.claim_next(task['id'], task['run_id'])
    assert next_step['id'] == 'b'
    assert task['steps'][1]['request_id'] == a_request


def test_paid_process_loss_unknown_no_resume_or_replay(tmp_path):
    s = store(tmp_path)
    task = begin(s, make(s))
    task = finish(s, task, s.claim_next(task['id'], task['run_id']))
    paid = s.claim_next(task['id'], task['run_id'])
    other = store(tmp_path, boot_id='boot-two')
    recovered = other.recover()[0]
    assert recovered['status'] == 'paused'
    a = next(item for item in recovered['steps'] if item['id'] == paid['id'])
    assert (a['status'], a['reason'], a['request_id']) == ('blocked', 'provider_outcome_unknown', paid['request_id'])
    with pytest.raises(ProjectTaskConflictError):
        begin(other, recovered)
    with pytest.raises(ProjectTaskConflictError):
        finish(s, task, paid)
    assert other.recover() == []
    reconciled = other.reconcile_step(task['id'], paid['id'], request_id=paid['request_id'],
        artifact={'answer_id': str(uuid.uuid4()), 'source_ids': ['s-a'], 'document_ids': ['alpha']}, source_snapshots=D)
    assert reconciled['status'] == 'paused'
    assert next(row for row in reconciled['steps'] if row['id'] == 'a')['status'] == 'completed'


def test_local_process_loss_requires_user_run_but_no_paid_unknown(tmp_path):
    s = store(tmp_path)
    task = begin(s, make(s))
    s.claim_next(task['id'], task['run_id'])
    other = store(tmp_path, boot_id='boot-two')
    recovered = other.recover()[0]
    assert recovered['steps'][0]['status'] == 'ready'
    assert other.claim_next(task['id'], task['run_id']) is None
    task = begin(other, recovered)
    assert other.claim_next(task['id'], task['run_id'])['id'] == 'inventory'


def test_source_mutation_targeted_and_report_withheld(tmp_path):
    s = store(tmp_path)
    task = all_done(s, make(s))
    old_a_request = task['steps'][1]['request_id']
    b_request = task['steps'][2]['request_id']
    changed = s.invalidate_sources(['alpha'])[0]
    states = {row['id']: row['status'] for row in changed['steps']}
    assert states == {'inventory': 'stale', 'a': 'stale', 'b': 'completed', 'report': 'stale'}
    assert s.get_artifact(task['id'], 'a')['references'] is None
    assert s.get_artifact(task['id'], 'report')['references'] is None
    assert s.get_artifact(task['id'], 'b')['available']
    new_docs = {**D, 'alpha': 'c' * 64}
    task = s.refresh_task(task['id'], expected_revision=changed['revision'], source_snapshots=new_docs)
    assert task['generation'] == 2
    assert task['steps'][1]['request_id'] != old_a_request
    assert task['steps'][2]['request_id'] == b_request
    task = begin(s, task, sources=new_docs)
    for key in ('inventory', 'a', 'report'):
        step = s.claim_next(task['id'], task['run_id'])
        assert step['id'] == key
        task = finish(s, task, step)
    assert s.get_artifact(task['id'], 'a')['version'] == 2
    assert s.get_artifact(task['id'], 'a', 1)['references'] is None
    assert s.artifact_history(task['id'], 'a')[1]['status'] == 'stale'


def test_explicit_dependent_propagates_even_other_document(tmp_path):
    s = store(tmp_path)
    task = all_done(s, make(s, questions=[{'id': 'a', 'question': 'First', 'document_ids': ['alpha']},
        {'id': 'b', 'question': 'Build on first', 'document_ids': ['beta'], 'depends_on': ['a']}]))
    changed = s.invalidate_sources(['alpha'], reason='source_revoked')[0]
    assert all(row['status'] == 'stale' for row in changed['steps'])
    assert s.get_artifact(task['id'], 'b')['status'] == 'withdrawn'
    with pytest.raises(ProjectTaskConflictError):
        s.refresh_task(task['id'], expected_revision=changed['revision'], source_snapshots={'beta': D['beta']})


def test_projectwide_new_document_invalidates_but_other_instance_does_not(tmp_path):
    s = store(tmp_path)
    task = all_done(s, make(s, questions=[{'question': 'Whole project'}]))
    assert store(tmp_path, instance_id='different').invalidate_sources(['newdoc']) == []
    assert s.get_task(task['id'])['status'] == 'completed'
    assert s.invalidate_sources(['newdoc'])[0]['status'] == 'stale'


def test_refresh_updates_inventory_after_unrelated_doc_added_without_redoing_scoped_answer(tmp_path):
    s = store(tmp_path)
    task = all_done(s, make(s, questions=[
        {'id': 'a', 'question': 'Explain alpha.', 'document_ids': ['alpha']},
        {'id': 'b', 'question': 'Explain beta.', 'document_ids': ['beta']}]))
    old_b = next(row for row in task['steps'] if row['id'] == 'b')
    new_docs = {**D, 'unrelated': 'c' * 64}
    s.invalidate_sources(['unrelated'])
    current = s.get_task(task['id'])
    task = s.revise_task(task['id'], expected_revision=current['revision'],
        goal=current['goal'], constraints=current['constraints'], questions=[
            {'id': 'a', 'question': 'Explain alpha with conditions.', 'document_ids': ['alpha']},
            {'id': 'b', 'question': 'Explain beta.', 'document_ids': ['beta']}])
    task = s.refresh_task(task['id'], expected_revision=task['revision'], source_snapshots=new_docs)
    untouched = next(row for row in task['steps'] if row['id'] == 'b')
    assert untouched['status'] == 'completed'
    assert untouched['request_id'] == old_b['request_id']
    task = begin(s, task, sources=new_docs)
    for wanted in ('inventory', 'a', 'report'):
        claim = s.claim_next(task['id'], task['run_id'])
        assert claim['id'] == wanted
        task = finish(s, task, claim)
    assert task['status'] == 'completed'


def _pending_dependent(s):
    task = begin(s, make(s, questions=[
        {'id': 'a', 'question': 'Establish material.', 'document_ids': ['alpha']},
        {'id': 'b', 'question': 'Use the prior material.', 'document_ids': ['alpha'], 'depends_on': ['a']}]))
    for _ in range(2):
        task = finish(s, task, s.claim_next(task['id'], task['run_id']))
    return task, s.claim_next(task['id'], task['run_id'])


def test_handoff_counts_are_reference_only_current_artifact_metadata(tmp_path):
    s = store(tmp_path)
    task, claim = _pending_dependent(s)
    handoff = {'required': 1, 'used': 1, 'full': 0, 'excerpted': 1, 'omitted': 0}
    task = s.complete_step(task['id'], claim['id'], claim['claim_token'], {
        'answer_id': str(uuid.uuid4()), 'source_ids': ['s-alpha'], 'document_ids': ['alpha'],
        'handoff': handoff})
    handoff['used'] = 0
    current = s.get_artifact(task['id'], 'b')
    assert current['references']['handoff'] == {
        'required': 1, 'used': 1, 'full': 0, 'excerpted': 1, 'omitted': 0}
    assert next(row for row in task['steps'] if row['id'] == 'b')['handoff'] == current['references']['handoff']
    assert 'handoff' not in next(row for row in task['steps'] if row['id'] == 'a')
    task = s.invalidate_sources(['alpha'], reason='source_revoked')[0]
    assert 'handoff' not in next(row for row in task['steps'] if row['id'] == 'b')
    assert s.get_artifact(task['id'], 'b', 1)['references'] is None


@pytest.mark.parametrize('handoff', [None, [], True,
    {'required': 1, 'used': 1, 'full': 1, 'excerpted': 0},
    {'required': 1, 'used': 1, 'full': 1, 'excerpted': 0, 'omitted': 0, 'answer': 'raw text'},
    {'required': True, 'used': 1, 'full': 1, 'excerpted': 0, 'omitted': 0},
    {'required': 1, 'used': 1.0, 'full': 1, 'excerpted': 0, 'omitted': 0},
    {'required': 1, 'used': 1, 'full': '1', 'excerpted': 0, 'omitted': 0},
    {'required': 1, 'used': 1, 'full': 1, 'excerpted': False, 'omitted': 0},
    {'required': 1, 'used': 1, 'full': 1, 'excerpted': 0, 'omitted': -1},
    {'required': 7, 'used': 7, 'full': 7, 'excerpted': 0, 'omitted': 0},
    {'required': 1, 'used': 0, 'full': 1, 'excerpted': 0, 'omitted': 1},
    {'required': 1, 'used': 1, 'full': 1, 'excerpted': 0, 'omitted': 1},
    {'required': 0, 'used': 0, 'full': 0, 'excerpted': 0, 'omitted': 0},
])
def test_handoff_rejects_wrong_types_extra_fields_inconsistent_or_missing_dependencies(tmp_path, handoff):
    s = store(tmp_path)
    task, claim = _pending_dependent(s)
    with pytest.raises(ProjectTaskError):
        s.complete_step(task['id'], claim['id'], claim['claim_token'], {
            'answer_id': str(uuid.uuid4()), 'source_ids': ['s-alpha'], 'document_ids': ['alpha'],
            'handoff': handoff})
    assert next(row for row in s.get_task(task['id'])['steps'] if row['id'] == 'b')['status'] == 'running'


def test_handoff_empty_counts_for_independent_analysis_do_not_invent_dependency(tmp_path):
    s = store(tmp_path)
    task = begin(s, make(s))
    task = finish(s, task, s.claim_next(task['id'], task['run_id']))
    claim = s.claim_next(task['id'], task['run_id'])
    counts = dict.fromkeys(('required', 'used', 'full', 'excerpted', 'omitted'), 0)
    task = s.complete_step(task['id'], claim['id'], claim['claim_token'], {
        'answer_id': str(uuid.uuid4()), 'source_ids': ['s-alpha'], 'document_ids': ['alpha'],
        'handoff': counts})
    assert next(row for row in task['steps'] if row['id'] == claim['id'])['handoff'] == counts


def test_source_change_rejects_late_paid_commit(tmp_path):
    s = store(tmp_path)
    task = begin(s, make(s))
    task = finish(s, task, s.claim_next(task['id'], task['run_id']))
    claim = s.claim_next(task['id'], task['run_id'])
    s.invalidate_sources(['alpha'])
    with pytest.raises(ProjectTaskConflictError):
        finish(s, task, claim)
    with pytest.raises(ProjectTaskNotFoundError):
        s.get_artifact(task['id'], 'a')


def test_immutable_reference_payload_rejects_source_or_answer_body(tmp_path):
    s = store(tmp_path)
    task = begin(s, make(s))
    task = finish(s, task, s.claim_next(task['id'], task['run_id']))
    claim = s.claim_next(task['id'], task['run_id'])
    for payload in ({'answer_id': str(uuid.uuid4()), 'answer': 'private raw answer'},
                    {'answer_id': str(uuid.uuid4()), 'document_ids': ['beta']},
                    {'answer_id': str(uuid.uuid4()), 'source_ids': ['s'], 'document_ids': ['alpha'],
                     'source_refs': [{'id': 's', 'content': 'raw copy'}]}):
        with pytest.raises(ProjectTaskError):
            s.complete_step(task['id'], claim['id'], claim['claim_token'], payload)
    task = finish(s, task, claim)
    with pytest.raises(ProjectTaskConflictError):
        finish(s, task, claim)


def test_revision_preserves_identity_history_and_unrelated_work(tmp_path):
    s = store(tmp_path)
    task = all_done(s, make(s))
    original = s.definition_history(task['id'])[0]
    questions = json.loads(json.dumps(original['questions']))
    questions[0]['question'] = 'Now include failure conditions.'
    changed = s.revise_task(task['id'], expected_revision=task['revision'], goal=task['goal'], constraints=task['constraints'], questions=questions)
    assert changed['id'] == task['id']
    assert {row['id']: row['status'] for row in changed['steps']} == {'inventory': 'completed', 'a': 'stale', 'b': 'completed', 'report': 'stale'}
    assert len(s.definition_history(task['id'])) == 2
    assert s.definition_history(task['id'])[1] == original
    assert s.get_artifact(task['id'], 'a')['references'] is None
    assert s.get_artifact(task['id'], 'b')['available']
    assert s.revise_task(task['id'], expected_revision=changed['revision'], goal=task['goal'], constraints=task['constraints'], questions=questions)['revision'] == changed['revision']


def test_goal_revision_invalidates_every_analysis(tmp_path):
    s = store(tmp_path)
    task = all_done(s, make(s))
    questions = s.definition_history(task['id'])[0]['questions']
    changed = s.revise_task(task['id'], expected_revision=task['revision'], goal='Different project goal', constraints=task['constraints'], questions=questions)
    assert all(row['status'] == 'stale' for row in changed['steps'] if row['kind'] != 'inventory')
    with pytest.raises(ProjectTaskConflictError):
        s.revise_task(task['id'], expected_revision=task['revision'], goal='Old stale write', constraints='', questions=questions)


def test_pause_and_cancel_never_duplicate_inflight_paid_step(tmp_path):
    s = store(tmp_path)
    task = begin(s, make(s))
    task = finish(s, task, s.claim_next(task['id'], task['run_id']))
    paid = s.claim_next(task['id'], task['run_id'])
    current = s.get_task(task['id'])
    paused = s.pause_task(task['id'], expected_revision=current['revision'])
    assert s.claim_next(task['id'], task['run_id']) is None
    done = finish(s, task, paid)
    assert done['status'] == 'paused'
    cancelled = s.cancel_task(task['id'], expected_revision=done['revision'])
    assert cancelled['status'] == 'cancelled'
    with pytest.raises(ProjectTaskConflictError):
        s.resume_task(task['id'], expected_revision=cancelled['revision'])


def test_unsafe_storage_link_or_permissions_refused(tmp_path):
    target = tmp_path / 'other'
    target.write_text('not a database')
    (tmp_path / 'project_tasks.db').symlink_to(target)
    with pytest.raises(ProjectTaskError):
        store(tmp_path).list_tasks()
    (tmp_path / 'project_tasks.db').unlink()
    s = store(tmp_path)
    make(s)
    os.chmod(s.path, 0o644)
    with pytest.raises(ProjectTaskError):
        s.list_tasks()


def test_task_caps_and_no_hidden_raw_copies(tmp_path, monkeypatch):
    import docthinker.server.project_task_store as module
    monkeypatch.setattr(module, 'MAX_PROJECT_TASKS', 1)
    s = store(tmp_path)
    task = all_done(s, make(s))
    with pytest.raises(ProjectTaskLimitError):
        make(s)
    with sqlite3.connect(s.path) as db:
        attempts = db.execute('SELECT request_id,status FROM project_task_attempts').fetchall()
        assert len(attempts) == 2 and {row[1] for row in attempts} == {'completed'}
        assert all(uuid.UUID(row[0]).version == 4 for row in attempts)


def test_old_run_replay_cannot_claim_new_approved_run(tmp_path):
    s = store(tmp_path)
    original = make(s)
    first_id = str(uuid.uuid4())
    first = begin(s, original, budget=1, request_id=first_id)
    for _ in range(2):
        first = finish(s, first, s.claim_next(first['id'], first_id))
    assert s.claim_next(first['id'], first_id) is None
    second = begin(s, s.get_task(first['id']), budget=1)
    replay = begin(s, original, budget=1, request_id=first_id)
    assert replay['run_id'] == first_id
    assert replay['run_replayed'] is True
    assert s.claim_next(replay['id'], replay['run_id']) is None
    assert s.get_task(first['id'])['run_id'] == second['run_id']


def test_explicit_refresh_detects_changes_without_prior_status_read(tmp_path):
    s = store(tmp_path)
    task = all_done(s, make(s))
    refreshed = s.refresh_task(task['id'], expected_revision=task['revision'],
                              source_snapshots={**D, 'alpha': 'c' * 64})
    assert refreshed['generation'] == 2
    assert {row['id']: row['status'] for row in refreshed['steps']} == {
        'inventory': 'ready', 'a': 'ready', 'b': 'completed', 'report': 'ready'}
    assert s.get_artifact(task['id'], 'a')['references'] is None


def test_cancelled_task_still_withholds_withdrawn_artifact_references(tmp_path):
    s = store(tmp_path)
    task = all_done(s, make(s))
    cancelled = s.cancel_task(task['id'], expected_revision=task['revision'])
    s.invalidate_sources(['alpha'], reason='source_revoked')
    assert s.get_task(task['id'])['status'] == 'cancelled'
    assert s.get_artifact(task['id'], 'a')['references'] is None


def test_refresh_unknown_is_not_authority_to_repeat_unchanged_paid_input(tmp_path):
    s = store(tmp_path)
    task = begin(s, make(s))
    task = finish(s, task, s.claim_next(task['id'], task['run_id']))
    claim = s.claim_next(task['id'], task['run_id'])
    task = s.block_step(task['id'], claim['id'], claim['claim_token'], reason='unknown')
    with pytest.raises(ProjectTaskConflictError):
        s.refresh_task(task['id'], expected_revision=task['revision'], source_snapshots=D)


def test_session_cleanup_removes_all_instances_and_rejects_late_claim(tmp_path):
    s = store(tmp_path)
    task = begin(s, make(s))
    claim = s.claim_next(task['id'], task['run_id'])
    second = store(tmp_path, instance_id='instance-two')
    another = all_done(second, make(second))
    unrelated = store(tmp_path, session_id='#00002')
    kept = make(unrelated)
    s.delete_session_metadata('#00001')
    assert s.list_tasks() == second.list_tasks() == []
    assert unrelated.get_task(kept['id'])['id'] == kept['id']
    with pytest.raises(ProjectTaskNotFoundError):
        finish(s, task, claim)
    for table in ('project_task_steps', 'project_task_artifacts', 'project_task_runs',
                  'project_task_definitions', 'project_task_attempts'):
        with sqlite3.connect(s.path) as db:
            assert db.execute('SELECT count(*) FROM ' + table + ' WHERE task_id IN (?,?)', (task['id'], another['id'])).fetchone()[0] == 0


def test_cleanup_absent_does_not_create_db(tmp_path):
    s = store(tmp_path)
    s.delete_session_metadata('#00001')
    assert not s.path.exists()


def test_explicit_known_failure_reset_new_identity_and_new_approval(tmp_path):
    s = store(tmp_path)
    task = begin(s, make(s))
    task = finish(s, task, s.claim_next(task['id'], task['run_id']))
    claim = s.claim_next(task['id'], task['run_id'])
    failed = s.block_step(task['id'], claim['id'], claim['claim_token'], reason='provider_failed')
    reset = s.reset_failed_task(task['id'], expected_revision=failed['revision'])
    assert reset['status'] == 'ready' and reset['run_id'] is None
    a = next(row for row in reset['steps'] if row['id'] == 'a')
    assert a['request_id'] != claim['request_id'] and a['generation'] == 2
    assert s.claim_next(task['id'], task['run_id']) is None
    approved = begin(s, reset, budget=1)
    assert s.claim_next(task['id'], approved['run_id'])['id'] == 'a'


def test_unknown_cannot_be_reset_by_retry_failed(tmp_path):
    s = store(tmp_path)
    task = begin(s, make(s))
    task = finish(s, task, s.claim_next(task['id'], task['run_id']))
    claim = s.claim_next(task['id'], task['run_id'])
    blocked = s.block_step(task['id'], claim['id'], claim['claim_token'], reason='unknown')
    with pytest.raises(ProjectTaskConflictError):
        s.reset_failed_task(task['id'], expected_revision=blocked['revision'])


def test_resume_is_ready_not_fictional_running_without_a_fresh_approval(tmp_path):
    s = store(tmp_path)
    task = begin(s, make(s))
    paused = s.pause_task(task['id'], expected_revision=task['revision'])
    resumed = s.resume_task(task['id'], expected_revision=paused['revision'])
    assert resumed['status'] == 'ready'
    assert s.claim_next(task['id'], task['run_id']) is None


def test_title_revision_preserves_analysis_but_versions_composed_report(tmp_path):
    s = store(tmp_path)
    task = all_done(s, make(s))
    questions = s.definition_history(task['id'])[0]['questions']
    questions[0]['title'] = 'New report section title'
    changed = s.revise_task(task['id'], expected_revision=task['revision'], goal=task['goal'], constraints=task['constraints'], questions=questions)
    assert {row['id']: row['status'] for row in changed['steps']} == {
        'inventory': 'completed', 'a': 'completed', 'b': 'completed', 'report': 'stale'}
    assert not s.get_artifact(task['id'], 'report', 1)['available']


def test_failed_first_write_does_not_make_empty_reads_crash(tmp_path, monkeypatch):
    import docthinker.server.project_task_store as module
    s = store(tmp_path)
    monkeypatch.setattr(module, 'MAX_STORAGE_BYTES', 1)
    with pytest.raises(ProjectTaskLimitError):
        make(s)
    assert s.list_tasks() == []


@pytest.mark.parametrize('bad', [True, 0, 1.0, '1'])
def test_revision_requires_exact_positive_integer(tmp_path, bad):
    s = store(tmp_path)
    task = make(s)
    with pytest.raises(ProjectTaskError):
        s.pause_task(task['id'], expected_revision=bad)


def test_snapshot_type_and_scoped_missing_document_are_rejected(tmp_path):
    s = store(tmp_path)
    task = make(s)
    for snapshot in ({'alpha': True}, {'alpha': 'private text'}, {'alpha': D['alpha']}):
        with pytest.raises(ProjectTaskError):
            begin(s, task, sources=snapshot)
    assert s.get_task(task['id'])['revision'] == task['revision']


def test_read_only_known_failed_reconciliation_keeps_identity_and_no_reset(tmp_path):
    s = store(tmp_path)
    task = begin(s, make(s))
    task = finish(s, task, s.claim_next(task['id'], task['run_id']))
    claim = s.claim_next(task['id'], task['run_id'])
    task = s.block_step(task['id'], claim['id'], claim['claim_token'], reason='unknown')
    old_remaining = task['run_remaining']
    for kwargs in ({'request_id': str(uuid.uuid4()), 'source_snapshots': D},
                   {'request_id': claim['request_id'], 'source_snapshots': {**D, 'alpha': 'c' * 64}}):
        with pytest.raises(ProjectTaskConflictError):
            s.reconcile_failed_step(task['id'], claim['id'], **kwargs)
    failed = s.reconcile_failed_step(task['id'], claim['id'], request_id=claim['request_id'], source_snapshots=D)
    row = next(row for row in failed['steps'] if row['id'] == 'a')
    assert row['status'] == 'blocked' and row['reason'] == 'provider_failed'
    assert row['request_id'] == claim['request_id'] and row['generation'] == claim['generation']
    assert failed['run_remaining'] == old_remaining
    assert s.claim_next(task['id'], task['run_id']) is None
    with pytest.raises(ProjectTaskConflictError):
        s.reconcile_failed_step(task['id'], claim['id'], request_id=claim['request_id'], source_snapshots=D)
