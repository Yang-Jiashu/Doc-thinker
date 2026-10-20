"""Bounded partial scans are visited work, not proof or permanent absence."""
from copy import deepcopy
import hashlib
import json
import uuid

from docthinker.server import project_obligation_ledger as module
from docthinker.server.long_project_store import LongProjectStore
from docthinker.server.project_evidence_controller import record_parent_lookups, prioritize_parent_queries, repair_plan
from docthinker.server.research_task_review import (
    build_task_requirements, task_requirement_rows, parse_task_response,
    build_task_followup, task_followup_queries,
)
from tests.test_project_task_followup_unit import lookup_fixture
from tests.test_project_obligation_ledger_unit import scoped, GOAL, QUESTION, BODY, SOURCE


def uid():
    return str(uuid.uuid4())


def current_work(ledger, task, work):
    return next(row for row in ledger.read(task['id'], eligible_step_ids=['a'])['work_items'] if row['id'] == work['id'])


def test_partial_scan_can_upgrade_once_without_becoming_a_goal_or_losing_newer_information(tmp_path):
    bridge, ledger, task, work, trace = lookup_fixture(tmp_path)
    partial = {**deepcopy(trace), 'candidate_query_sha256': [], 'corpus_scan_limited': True,
               'corpus_attempted_query_sha256': list(trace['attempted_query_sha256'])}
    before = ledger.read(task['id'], eligible_step_ids=['a'])
    assert record_parent_lookups(bridge, partial)['recorded_count'] >= 1
    assert current_work(ledger, task, work)['status'] == 'partially_searched'
    assert current_work(ledger, task, work)['attempts'] == 1
    full = {**deepcopy(partial), 'corpus_scan_limited': False}
    record_parent_lookups(bridge, full)
    assert current_work(ledger, task, work)['status'] == 'exhausted'
    record_parent_lookups(bridge, trace)
    candidate = current_work(ledger, task, work)
    assert candidate['status'] == 'candidates_found' and candidate['attempts'] == candidate['round'] == 1
    record_parent_lookups(bridge, partial)
    record_parent_lookups(bridge, full)
    after = ledger.read(task['id'], eligible_step_ids=['a'])
    assert current_work(ledger, task, work) == candidate
    assert after['coverage'] == before['coverage'] and after['goal_verified'] is False
    repair = repair_plan(task, after)
    assert repair['changed_steps'] and repair['plan']['questions'][-1]['purpose'] == 'verify'


def test_full_scan_of_narrower_or_unknown_scope_is_only_partial(tmp_path):
    bridge, ledger, task, work, trace = lookup_fixture(tmp_path)
    trace.update(candidate_query_sha256=[], corpus_scan_limited=False,
                 corpus_attempted_query_sha256=list(trace['attempted_query_sha256']))
    trace['lookup_scope']['document_snapshot'] = {}
    record_parent_lookups(bridge, trace)
    assert current_work(ledger, task, work)['status'] == 'partially_searched'
    trace['lookup_scope'] = None
    record_parent_lookups(bridge, trace)
    assert current_work(ledger, task, work)['status'] == 'partially_searched'
    assert current_work(ledger, task, work)['attempts'] == 1


def test_new_lookup_policy_does_not_inherit_old_exhaustion_or_recreate_it_from_an_old_answer(tmp_path, monkeypatch):
    bridge, ledger, task, work, trace = lookup_fixture(tmp_path)
    trace.update(candidate_query_sha256=[], corpus_scan_limited=False,
                 corpus_attempted_query_sha256=list(trace['attempted_query_sha256']))
    record_parent_lookups(bridge, trace)
    assert current_work(ledger, task, work)['status'] == 'exhausted'
    monkeypatch.setattr(module, 'LOOKUP_POLICY_VERSION', 'goal_evidence_lookup_v3')
    fresh = ledger.read(task['id'], eligible_step_ids=['a'])
    assert all(row['attempts'] == 0 and row['status'] == 'unresolved' for row in fresh['work_items'])
    assert work['id'] not in {row['id'] for row in fresh['work_items']}
    assert record_parent_lookups(bridge, trace) == {'status': 'lookup_policy_unavailable', 'recorded_count': 0}
    assert ledger.read(task['id'], eligible_step_ids=['a']) == fresh


def test_all_twenty_six_original_needs_get_their_turn_after_partial_misses(tmp_path):
    """Actual approved phase/store/receipts, not an injected fake goal list."""
    from types import SimpleNamespace
    owner = scoped(tmp_path)
    projects = LongProjectStore(owner)
    project_rules = [f'Original project condition {i}' for i in range(12)]
    phase_rules = [f'Original stage condition {i}' for i in range(12)]
    project = projects.create_project(goal=GOAL, acceptance_criteria=project_rules, request_id=uid())
    phase = projects.add_phase(project['id'], expected_revision=project['revision'], request_id=uid(),
        title='Review original requirements', acceptance_criteria=phase_rules,
        questions=[{'id': 'a', 'question': QUESTION, 'document_ids': ['doc-one']}])
    task = owner.get_task(phase['task_id'])
    task = owner.begin_run(task['id'], expected_revision=task['revision'], max_calls=1,
        source_snapshots={'doc-one': 'a' * 64}, request_id=uid())
    inventory = owner.claim_next(task['id'], task['run_id'])
    owner.complete_step(task['id'], 'inventory', inventory['claim_token'],
        {'documents': [{'id': 'doc-one', 'digest': 'a' * 64}]})
    step = owner.claim_next(task['id'], task['run_id'])
    required = build_task_requirements(QUESTION, phase_rules, project_goal=GOAL,
                                      project_acceptance_criteria=project_rules)
    hints = [f'Retrieve original approved record R{i} and its missing condition.' for i in range(26)]
    _, review = parse_task_response(json.dumps({'answer': BODY, 'task_assessment': {'requirements': [
        {'id': row['id'], 'verdict': 'unknown', 'missing_query': hint}
        for row, hint in zip(task_requirement_rows(required), hints)]}}), required, [SOURCE])
    owner.complete_step(task['id'], 'a', step['claim_token'],
        {'answer_id': uid(), 'document_ids': ['doc-one'], 'source_ids': ['source-one'], 'task_review': review})
    task = owner.get_task(task['id'])
    artifact = owner.get_artifact(task['id'], 'a')['references']
    ledger = module.ProjectObligationLedger(owner)
    before = ledger.synchronize(task['id'], eligible_step_ids=['a'])
    assert before['coverage']['total'] == len(before['work_items']) == 26
    bridge = SimpleNamespace(session_id=owner.session_id, instance_id=owner.instance_id, phase_guard=None,
        manager=SimpleNamespace(base_storage_path=tmp_path, get_session=lambda _: {'instance_id': owner.instance_id}),
        validate_inputs=lambda _: True)
    metadata = {'instance_id': owner.instance_id, 'task_id': task['id'], 'step_id': 'a', 'artifact_version': 1,
        'answer_id': artifact['answer_id'], 'review_kind': 'task',
        'receipt_sha256': artifact['task_review_binding']['receipt_sha256']}
    dependency = {**metadata, 'priority': 'missing_evidence', 'queries': hints, 'gaps': [],
                  'omitted_query_count': 0, 'omitted_gap_count': 0}
    seen = set()
    for _ in range(4):
        ordered = prioritize_parent_queries(bridge, [dependency])
        selected = task_followup_queries(build_task_followup(ordered))
        # All not-yet-visited queries lead the bounded frontier, even though a
        # previous search had no candidate and could not claim exhaustion.
        unseen = set(hints) - seen
        assert set(selected[:min(8, len(unseen))]) <= unseen
        hashes = [hashlib.sha256(query.encode()).hexdigest() for query in selected]
        trace = {'lookup_policy': module.LOOKUP_POLICY_VERSION, 'lookup_scope': None,
            'attempted_query_sha256': hashes, 'candidate_query_sha256': [],
            'corpus_attempted_query_sha256': [], 'corpus_scan_limited': True, 'dependencies': [metadata]}
        assert record_parent_lookups(bridge, trace)['recorded_count'] >= len(set(selected))
        seen.update(selected)
    after = ledger.read(task['id'], eligible_step_ids=['a'])
    assert seen == set(hints)
    assert all(row['status'] == 'partially_searched' and row['attempts'] == 1 for row in after['work_items'])
    assert after['coverage'] == before['coverage'] and after['goal_verified'] is False
    assert dependency['queries'] == hints  # Complete original hints survive every round.
