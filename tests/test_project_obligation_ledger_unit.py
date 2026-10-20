"""Current artifact obligations survive restarts without promoting retrieval to truth."""
from copy import deepcopy
import json
import sqlite3
import uuid

import pytest

from docthinker.server.project_evidence_feedback import bind_feedback
from docthinker.server.project_obligation_ledger import (
    ProjectObligationLedger, build_obligation_progress,
)
from docthinker.server.project_task_store import (
    ProjectTaskStore, ProjectTaskError, ProjectTaskConflictError, ProjectTaskNotFoundError,
)
from docthinker.server.research_task_review import (
    build_task_requirements, task_requirement_rows, parse_task_response,
)


GOAL = 'Determine why the internal handover was delayed; do not send anything.'
QUESTION = 'Check the workload change and alternatives against original records.'
BODY = 'The log records additional work; the absence roster is missing.'
SOURCE = {'id': 'source-one', 'label': 'S1', 'document_id': 'doc-one',
          'content': BODY, 'excerpt_start': 0, 'excerpt_end': len(BODY)}


def scoped(tmp_path, **kwargs):
    return ProjectTaskStore(tmp_path, session_id=kwargs.pop('session_id', '#00001'),
        instance_id=kwargs.pop('instance_id', 'instance-one'), boot_id='boot-one', **kwargs)


def task_packet():
    return {'id': str(uuid.uuid4()), 'revision': 5, 'goal': GOAL, 'status': 'completed',
            'steps': [{'id': 'a', 'kind': 'analysis', 'question': QUESTION,
                       'status': 'completed', 'artifact_version': 1, 'generation': 1,
                       'input_snapshot': {'doc-one': 'a' * 64}}]}


def receipt(task, *, version=2, phase=None, project=None, verdict='addressed', evidence=True):
    req = build_task_requirements(QUESTION, phase or [],
        **({'project_goal': task['goal'], 'project_acceptance_criteria': project or []} if version == 2 else {}))
    rows = [{'id': row['id'], 'verdict': verdict,
             'answer_witness': {'quote': BODY, 'start': 0},
             'evidence': [{'label': 'S1', 'quote': BODY, 'start': 0}] if evidence else [],
             'missing_query': 'Retrieve the original absence roster and critical path records.'}
            for row in task_requirement_rows(req)]
    _, value = parse_task_response(json.dumps({'answer': BODY,
        'task_assessment': {'requirements': rows}}), req, [SOURCE])
    return value


def artifact(task, value=None, *, event=None):
    result = {'answer_id': str(uuid.uuid4()), 'source_ids': ['source-one'], 'document_ids': ['doc-one']}
    if value is not None:
        result['task_review'] = value
        result['task_review_binding'] = bind_feedback(value, task_id=task['id'], task_revision=4,
                                                      step_id='a', artifact_version=1)
    if event is not None:
        result['event_reasoning'] = event
        result['evidence_feedback_binding'] = bind_feedback(event, task_id=task['id'], task_revision=4,
                                                            step_id='a', artifact_version=1)
    return result


def progress(task, payload, **kwargs):
    return build_obligation_progress(task, {'a': payload}, eligible_step_ids=['a'], **kwargs)


def event_receipt(identifier='relation-workload', *, status='model_assessed_support', local_id='L1'):
    row = {'id': local_id, 'kind': 'relation', 'status': status, 'evidence': [{
        'label': 'S1', 'source_id': 'source-one', 'span': [0, len(BODY)], 'quote_sha256': 'a' * 64}],
        'gaps': [] if status == 'model_assessed_support' else ['semantic_support_unassessed'],
        'condition_count': 0}
    if identifier is not None:
        row['relation_id'] = identifier
    return {'status': 'assessed_with_caveats' if status == 'model_assessed_support' else 'evidence_gaps',
            'obligations': [row], 'supported': int(status == 'model_assessed_support'),
            'counterevidence': int(status == 'model_assessed_counterevidence'),
            'unknown': int(status == 'unknown'),
            'gaps': [{'obligation_id': local_id, 'code': code} for code in row['gaps']],
            'suggested_queries': ['Retrieve change log and alternative cause records.'],
            'causality_verified': False, 'model_assessed': True, 'condition_schema': 2}


def completed_store(tmp_path, *, review=True, verdict='unknown'):
    store = scoped(tmp_path)
    task = store.create_task(goal=GOAL, questions=[{'id': 'a', 'question': QUESTION,
                                                  'document_ids': ['doc-one']}])
    task = store.begin_run(task['id'], expected_revision=task['revision'], max_calls=1,
                           source_snapshots={'doc-one': 'a' * 64}, request_id=str(uuid.uuid4()))
    inventory = store.claim_next(task['id'], task['run_id'])
    task = store.complete_step(task['id'], inventory['id'], inventory['claim_token'],
                              {'documents': [{'id': 'doc-one', 'digest': 'a' * 64}]})
    step = store.claim_next(task['id'], task['run_id'])
    payload = {'answer_id': str(uuid.uuid4()), 'source_ids': ['source-one'], 'document_ids': ['doc-one']}
    if review:
        payload['task_review'] = receipt(task, verdict=verdict)
    task = store.complete_step(task['id'], step['id'], step['claim_token'], payload)
    return store, task


def test_completed_legacy_step_does_not_mean_goal_or_question_assessed():
    task = task_packet()
    result = progress(task, artifact(task))
    assert result['coverage'] == {'total': 2, 'ready_for_review': 0, 'unresolved': 0,
                                 'contradicted': 0, 'not_assessed': 2, 'goal_verified': False}
    assert {row['requirement_id'] for row in result['obligations']} == {'Q1', 'G1'}
    assert result['next_action'] == 'await_analysis'


def test_live_current_receipt_can_only_be_ready_for_review_not_verified_truth():
    task = task_packet()
    result = progress(task, artifact(task, receipt(task)))
    assert result['coverage']['ready_for_review'] == 2
    assert result['work_items'] == [] and result['next_action'] == 'review'
    assert result['semantic_verified'] is result['goal_verified'] is False


def test_eligibility_is_explicit_not_a_persisted_completed_badge():
    task = task_packet()
    result = build_obligation_progress(task, {'a': artifact(task, receipt(task))})
    assert result['coverage']['not_assessed'] == 2


@pytest.mark.parametrize('verdict,expected', [('counterevidence', 'contradicted'), ('unknown', 'unresolved')])
def test_counterevidence_and_missing_proof_become_work_not_success(verdict, expected):
    task = task_packet()
    result = progress(task, artifact(task, receipt(task, verdict=verdict)))
    assert result['coverage'][expected] == 2 and result['work_items']
    assert all(row['status'] == 'unresolved' for row in result['work_items'])


def test_writing_an_answer_without_original_evidence_is_not_factual_completion():
    task = task_packet()
    result = progress(task, artifact(task, receipt(task, evidence=False)))
    assert result['coverage']['unresolved'] == 2 and not result['coverage']['ready_for_review']


def test_personal_draft_can_be_delivered_without_claiming_external_factual_support():
    task = task_packet()
    task['steps'][0]['input_kind'] = 'personal'
    req = build_task_requirements(QUESTION, input_kind='personal', project_goal=GOAL)
    raw = json.dumps({'answer': BODY, 'task_assessment': {'requirements': [
        {'id': key, 'verdict': 'addressed', 'evidence': [],
         'answer_witness': {'quote': BODY}} for key in ['Q1', 'G1']]}})
    _, assessed = parse_task_response(raw, req, [])
    result = progress(task, artifact(task, assessed))
    assert result['coverage']['ready_for_review'] == 2
    assert assessed['supported'] == 0 and result['goal_verified'] is False


def test_global_criterion_is_shared_not_required_independently_from_every_subanalysis():
    task = task_packet()
    second = {**task['steps'][0], 'id': 'b'}
    task['steps'].append(second)
    first_payload = artifact(task, receipt(task, verdict='unknown'))
    second_payload = artifact(task, receipt(task))
    second_payload['task_review_binding']['step_id'] = 'b'
    result = build_obligation_progress(task, {'a': first_payload, 'b': second_payload},
                                       eligible_step_ids=['a', 'b'])
    assert result['coverage']['total'] == 3  # two original questions + one goal
    goals = [row for row in result['obligations'] if row['requirement_id'] == 'G1']
    assert len(goals) == 1 and goals[0]['status'] == 'ready_for_review'
    assert 'review_group_sha256' in goals[0]['binding']
    assert sum(row['requirement_id'] == 'Q1' for row in result['obligations']) == 2


def test_conflicting_global_reviews_cannot_be_averaged_into_success():
    task = task_packet()
    task['steps'].append({**task['steps'][0], 'id': 'b'})
    first_payload = artifact(task, receipt(task, verdict='counterevidence'))
    second_payload = artifact(task, receipt(task))
    second_payload['task_review_binding']['step_id'] = 'b'
    result = build_obligation_progress(task, {'a': first_payload, 'b': second_payload},
                                       eligible_step_ids=['a', 'b'])
    goal = next(row for row in result['obligations'] if row['requirement_id'] == 'G1')
    assert goal['status'] == 'contradicted' and goal['owner_step_id'] == 'a'
    assert 'conflicting_reviews' in goal['gaps']


def test_legacy_v1_question_support_cannot_fill_missing_global_goal_review():
    task = task_packet()
    result = progress(task, artifact(task, receipt(task, version=1)))
    by_id = {row['requirement_id']: row for row in result['obligations']}
    assert by_id['Q1']['status'] == 'ready_for_review'
    assert by_id['G1']['status'] == 'not_assessed'


@pytest.mark.parametrize('change', ['receipt', 'binding', 'question', 'goal', 'artifact_version'])
def test_modified_or_wrong_artifact_contract_never_inherits_old_support(change):
    task = task_packet()
    payload = artifact(task, receipt(task))
    if change == 'receipt':
        payload['task_review']['requirements_sha256'] = '0' * 64
    elif change == 'binding':
        payload['task_review_binding']['receipt_sha256'] = '0' * 64
    elif change == 'question':
        task['steps'][0]['question'] += ' Also analyze the new deadline.'
    elif change == 'goal':
        task['goal'] += ' Include the revised agreement.'
    else:
        task['steps'][0]['artifact_version'] = 2
    result = progress(task, payload)
    assert result['coverage']['not_assessed'] == 2
    assert result['coverage']['ready_for_review'] == 0


def test_all_twelve_project_and_phase_requirements_are_kept_without_model_selection():
    task = task_packet()
    project = [f'Project acceptance requirement {i} must remain complete.' for i in range(12)]
    phase = [f'Phase acceptance requirement {i} must remain complete.' for i in range(12)]
    assessed = receipt(task, phase=phase, project=project)
    payload = artifact(task, assessed)
    result = progress(task, payload, execution_context={'project_goal': GOAL,
        'project_acceptance_criteria': project, 'phase_acceptance_criteria': phase})
    assert result['coverage']['total'] == 26
    # The parser's receipt byte budget may conservatively downgrade a complete
    # need, but neither the parser nor the ledger may silently omit that need.
    assert result['coverage']['ready_for_review'] == assessed['supported']
    assert {row['requirement_id'] for row in result['obligations']} == {
        'Q1', 'G1', *[f'P{i}' for i in range(1, 13)], *[f'C{i}' for i in range(1, 13)]}


def test_event_identity_is_stable_relation_id_not_local_number():
    task = task_packet()
    result = progress(task, artifact(task, receipt(task), event=event_receipt()))
    row = next(row for row in result['obligations'] if row['kind'] == 'event_relation')
    assert row['semantic_id'] == 'relation-workload' and row['status'] == 'ready_for_review'
    changed = progress(task, artifact(task, receipt(task), event=event_receipt('relation-absence')))
    other = next(row for row in changed['obligations'] if row['kind'] == 'event_relation')
    assert other['semantic_id'] != row['semantic_id'] and other['id'] != row['id']


def test_event_support_with_shed_stable_identity_stays_unresolved():
    task = task_packet()
    result = progress(task, artifact(task, receipt(task), event=event_receipt(None)))
    row = next(row for row in result['obligations'] if row['kind'] == 'event_relation')
    assert row['status'] == 'unresolved' and 'stable_event_identity_missing' in row['gaps']


@pytest.mark.parametrize('verdict,expected', [
    ('model_assessed_counterevidence', 'ready_for_review'),
    ('model_assessed_support', 'contradicted')])
def test_explicit_conflict_uses_typed_polarity_instead_of_raw_support_badge(verdict, expected):
    task = task_packet()
    assessment = event_receipt(status=verdict)
    assessment['obligations'][0]['relation'] = 'contradicts'
    assessment['obligations'][0]['gaps'] = []
    assessment['gaps'] = []
    assessment['causal_review_schema'] = 2
    result = progress(task, artifact(task, receipt(task), event=assessment))
    conflict = next(row for row in result['obligations'] if row['kind'] == 'event_relation')
    assert conflict['status'] == expected


@pytest.mark.parametrize('kind', ['relation', 'event', 'path', 'alternatives'])
def test_event_kinds_are_distinct_work_items_with_receipt_binding(kind):
    task = task_packet()
    event = event_receipt(status='unknown')
    event['obligations'][0]['kind'] = kind
    if kind == 'event':
        event['obligations'][0]['event_id'] = event['obligations'][0].pop('relation_id')
    if kind in {'path', 'alternatives'}:
        event['obligations'][0].pop('relation_id')
    result = progress(task, artifact(task, receipt(task), event=event))
    assert len(result['work_items']) == 1
    item = result['work_items'][0]
    assert item['kind'] == 'event_' + kind and item['binding']['receipt_sha256']
    assert item['binding']['answer_id']


def test_sync_idempotent_scoped_restart_and_no_evidence_body_copy(tmp_path):
    store, task = completed_store(tmp_path)
    ledger = ProjectObligationLedger(store)
    one = ledger.synchronize(task['id'], expected_revision=task['revision'], eligible_step_ids=['a'])
    two = ledger.synchronize(task['id'], task=task, eligible_step_ids=['a'])
    assert one == two
    with sqlite3.connect(store.path) as db:
        assert db.execute('SELECT count(*) FROM project_obligation_ledgers').fetchone()[0] == 1
        raw = db.execute('SELECT payload FROM project_obligation_ledgers').fetchone()[0]
    assert BODY not in raw and 'quote_sha256' not in raw and 'content' not in raw
    restarted = ProjectObligationLedger(scoped(tmp_path))
    assert restarted.read(task['id'], eligible_step_ids=['a']) == one
    assert restarted.read(task['id'])['coverage']['not_assessed'] == 2


@pytest.mark.parametrize('scope', [{'session_id': '#00002'}, {'instance_id': 'different-instance'}])
def test_ledger_cannot_read_another_scope(tmp_path, scope):
    store, task = completed_store(tmp_path)
    ProjectObligationLedger(store).synchronize(task['id'], eligible_step_ids=['a'])
    with pytest.raises(ProjectTaskNotFoundError):
        ProjectObligationLedger(scoped(tmp_path, **scope)).read(task['id'], eligible_step_ids=['a'])


def test_sync_rejects_stale_live_checked_task_revision(tmp_path):
    store, task = completed_store(tmp_path)
    ledger = ProjectObligationLedger(store)
    with pytest.raises(ProjectTaskConflictError):
        ledger.synchronize(task['id'], expected_revision=task['revision'] - 1)
    old = {**task, 'revision': task['revision'] - 1}
    with pytest.raises(ProjectTaskConflictError):
        ledger.synchronize(task['id'], task=old, eligible_step_ids=['a'])


def test_sync_rebuilds_host_question_from_database_not_caller_copy(tmp_path):
    store, task = completed_store(tmp_path, review=False)
    copied = deepcopy(task)
    copied['goal'] = 'An easier substituted goal'
    result = ProjectObligationLedger(store).synchronize(task['id'], task=copied)
    assert any(row['query'] == GOAL for row in result['obligations'])
    assert 'An easier substituted goal' not in json.dumps(result)


@pytest.mark.parametrize('outcome', ['partially_searched', 'exhausted', 'candidates_found'])
def test_lookup_attempt_is_idempotent_survives_restart_and_never_fulfils_goal(tmp_path, outcome):
    store, task = completed_store(tmp_path)
    ledger = ProjectObligationLedger(store)
    before = ledger.synchronize(task['id'], eligible_step_ids=['a'])
    item = before['work_items'][0]
    attempt = ledger.record_attempt(task['id'], item['id'], expected_revision=task['revision'], outcome=outcome)
    assert attempt == ledger.record_attempt(task['id'], item['id'], expected_revision=task['revision'], outcome=outcome)
    after = ProjectObligationLedger(scoped(tmp_path)).read(task['id'], eligible_step_ids=['a'])
    assert after['coverage'] == before['coverage']
    assert after['work_items'][0]['status'] == outcome and after['work_items'][0]['attempts'] == 1
    upgraded = ledger.record_attempt(task['id'], item['id'], expected_revision=task['revision'],
                                     outcome='candidates_found')
    assert upgraded['status'] == 'candidates_found' and upgraded['attempts'] == upgraded['round'] == 1
    delayed = ledger.record_attempt(task['id'], item['id'], expected_revision=task['revision'],
                                    outcome='partially_searched')
    assert delayed['status'] == 'candidates_found' and delayed['attempts'] == 1


@pytest.mark.parametrize('outcome', ['verified', 'ready_for_review', True, [], None])
def test_candidate_lookup_cannot_self_certify_or_execute(tmp_path, outcome):
    store, task = completed_store(tmp_path)
    ledger = ProjectObligationLedger(store)
    item = ledger.synchronize(task['id'], eligible_step_ids=['a'])['work_items'][0]
    with pytest.raises(ProjectTaskError):
        ledger.record_attempt(task['id'], item['id'], expected_revision=task['revision'], outcome=outcome)


@pytest.mark.parametrize('status', ['archived', 'stale'])
def test_noncurrent_receipt_is_not_loaded_from_artifact_archive(tmp_path, status):
    store, task = completed_store(tmp_path)
    with store._db(write=True) as db:
        db.execute('UPDATE project_task_artifacts SET status=? WHERE task_id=? AND step_id=?',
                   (status, task['id'], 'a'))
    result = ProjectObligationLedger(store).synchronize(task['id'], eligible_step_ids=['a'])
    assert result['coverage']['not_assessed'] == 2 and not result['coverage']['ready_for_review']


def test_source_change_invalidates_ledger_and_old_lookup_attempt_cannot_apply(tmp_path):
    store, task = completed_store(tmp_path)
    ledger = ProjectObligationLedger(store)
    one = ledger.synchronize(task['id'], eligible_step_ids=['a'])
    old = one['work_items'][0]
    ledger.record_attempt(task['id'], old['id'], expected_revision=task['revision'], outcome='exhausted')
    store.invalidate_sources(['doc-one'])
    changed = store.get_task(task['id'])
    two = ledger.synchronize(task['id'], task=changed, eligible_step_ids=[])
    assert two['coverage']['not_assessed'] == 2
    assert all(row['attempts'] == 0 for row in two['work_items'])
    assert old['id'] not in {row['id'] for row in two['work_items']}
    with pytest.raises(ProjectTaskConflictError):
        ledger.record_attempt(task['id'], old['id'], expected_revision=changed['revision'], outcome='exhausted')


def test_previously_supported_source_correction_cannot_keep_review_ready_status(tmp_path):
    store, task = completed_store(tmp_path, verdict='addressed')
    ledger = ProjectObligationLedger(store)
    ready = ledger.synchronize(task['id'], eligible_step_ids=['a'])
    assert ready['coverage']['ready_for_review'] == 2
    store.invalidate_sources(['doc-one'])
    changed = ledger.synchronize(task['id'], eligible_step_ids=['a'])
    assert changed['coverage']['ready_for_review'] == 0
    assert changed['coverage']['not_assessed'] == 2


def test_source_snapshot_changes_work_binding_even_if_local_question_alias_is_same():
    task = task_packet()
    payload = artifact(task, receipt(task, verdict='unknown'))
    before = progress(task, payload)
    newer = deepcopy(task)
    newer['steps'][0]['input_snapshot']['doc-one'] = 'b' * 64
    after = progress(newer, payload)
    assert {row['id'] for row in before['work_items']}.isdisjoint(
        {row['id'] for row in after['work_items']})
    assert before['fingerprint'] != after['fingerprint']


def test_alias_reordering_cannot_transfer_a_completed_work_identity_to_another_relation():
    task = task_packet()
    first = event_receipt('relation-workload', status='unknown', local_id='L1')
    second = event_receipt('relation-absence', status='unknown', local_id='L1')
    left = progress(task, artifact(task, receipt(task), event=first))
    right = progress(task, artifact(task, receipt(task), event=second))
    assert left['work_items'][0]['id'] != right['work_items'][0]['id']
    assert next(row for row in left['obligations'] if row['kind'] == 'event_relation')['semantic_id'] == 'relation-workload'
    assert next(row for row in right['obligations'] if row['kind'] == 'event_relation')['semantic_id'] == 'relation-absence'


def test_maximum_original_question_and_query_hint_are_preserved_without_truncation():
    task = task_packet()
    task['steps'][0]['question'] = 'q' * 2000
    requirement = build_task_requirements(task['steps'][0]['question'], project_goal=task['goal'])
    _, review = parse_task_response(json.dumps({'answer': BODY, 'task_assessment': {'requirements': [
        {'id': row['id'], 'verdict': 'unknown', 'missing_query': 'x' * 400}
        for row in task_requirement_rows(requirement)]}}), requirement, [SOURCE])
    payload = artifact(task, review)
    result = progress(task, payload)
    question = next(row for row in result['obligations'] if row['requirement_id'] == 'Q1')
    assert task['steps'][0]['question'] in question['query'] and 'x' * 400 in question['query']
    assert question['lookup_query'] == 'x' * 400
    assert result['work_items'][0]['lookup_query'] == 'x' * 400


def test_long_unicode_host_requirement_has_only_a_bounded_search_hint_not_a_shortened_goal():
    task = task_packet()
    task['steps'][0]['question'] = '核查证据与替代解释并保留原始验收要求。' * 70
    assert len(task['steps'][0]['question']) < 2000
    result = progress(task, artifact(task))
    question = next(row for row in result['obligations'] if row['requirement_id'] == 'Q1')
    assert question['query'] == task['steps'][0]['question']
    assert len(question['lookup_query'].encode('utf-8')) <= 400
    assert question['lookup_query'].endswith('…')
    assert result['work_items'][0]['query'] == task['steps'][0]['question']


def test_unassessed_current_receipt_is_not_treated_as_a_finished_assessment():
    task = task_packet()
    requirement = build_task_requirements(QUESTION, project_goal=task['goal'])
    _, review = parse_task_response(BODY, requirement, [SOURCE])
    assert review['model_assessed'] is False
    result = progress(task, artifact(task, review))
    assert result['coverage']['not_assessed'] == 2


def test_attempt_journal_cannot_hide_a_changed_task_revision(tmp_path):
    store, task = completed_store(tmp_path)
    ledger = ProjectObligationLedger(store)
    item = ledger.synchronize(task['id'], eligible_step_ids=['a'])['work_items'][0]
    paused = store.pause_task(task['id'], expected_revision=task['revision'])
    with pytest.raises(ProjectTaskConflictError):
        ledger.record_attempt(task['id'], item['id'], expected_revision=task['revision'], outcome='exhausted')
    with pytest.raises(ProjectTaskConflictError):
        ledger.record_attempt(task['id'], item['id'], expected_revision=paused['revision'], outcome='exhausted')


def test_long_project_and_phase_criteria_are_read_in_same_scoped_transaction(tmp_path):
    from docthinker.server.long_project_store import LongProjectStore
    store = scoped(tmp_path)
    projects = LongProjectStore(store)
    project = projects.create_project(goal=GOAL, acceptance_criteria=['Explain alternatives.'], request_id=str(uuid.uuid4()))
    phase = projects.add_phase(project['id'], expected_revision=project['revision'], request_id=str(uuid.uuid4()),
        title='Check handover', acceptance_criteria=['Keep absent evidence explicit.'],
        questions=[{'id': 'a', 'question': QUESTION, 'document_ids': ['doc-one']}])
    task_id = phase['task_id']
    result = ProjectObligationLedger(store).synchronize(task_id)
    assert {row['requirement_id'] for row in result['obligations']} == {'Q1', 'G1', 'P1', 'C1'}
    assert {'Explain alternatives.', 'Keep absent evidence explicit.'} <= {row['query'] for row in result['obligations']}


@pytest.mark.parametrize('eligible', [['wrong-step'], ['a', 'a'], [True], 'a'])
def test_live_scope_cannot_expand_or_coerce(eligible):
    task = task_packet()
    with pytest.raises(ProjectTaskError):
        build_obligation_progress(task, {'a': artifact(task)}, eligible_step_ids=eligible)
