"""Explicit visible review nodes, immutable drafts and current proof lineages.

All records and callbacks are invented; no provider or external operation.
"""
import asyncio
from copy import deepcopy
import hashlib
import json
import uuid

import pytest

from docthinker.server.project_task_store import ProjectTaskStore, ProjectTaskError, ProjectTaskConflictError
from docthinker.server.project_task_runtime import ProjectTaskRuntime
from docthinker.server.project_evidence_feedback import bind_feedback
from docthinker.server.project_obligation_ledger import build_obligation_progress, ProjectObligationLedger
from tests.test_project_obligation_ledger_unit import task_packet, receipt, event_receipt, BODY, QUESTION


def uid():
    return str(uuid.uuid4())


def pair(index=0):
    key = 'draft' + str(index)
    question = 'Explain the current work records ' + str(index)
    return [{'id': key, 'question': question, 'document_ids': ['alpha']},
            {'id': 'audit' + str(index), 'purpose': 'verify', 'review_of': key,
             'question': question, 'document_ids': ['alpha'], 'depends_on': [key]}]


def store(tmp_path):
    return ProjectTaskStore(tmp_path, session_id='#00001', instance_id='review-instance', boot_id='review-boot')


def lineage(root, target=None, **changes):
    value = {'version': 1, 'purpose': 'verify' if target else 'compose', 'root_answer_id': root,
        'target_answer_id': target, 'body_sha256': hashlib.sha256(BODY.encode()).hexdigest(),
        'root_body_sha256': hashlib.sha256(BODY.encode()).hexdigest(), 'context_sha256': 'c' * 64,
        'source_scope_sha256': 'd' * 64, 'assessed_need_ids': [], 'semantic_verified': False}
    return {**value, **changes}


def test_twelve_nodes_require_explicit_reviews_and_do_not_expand_six_call_grants(tmp_path):
    owner = store(tmp_path)
    rows = [row for index in range(6) for row in pair(index)]
    task = owner.create_task(goal='Prepare grounded private drafts.', questions=rows)
    assert len([row for row in task['steps'] if row['kind'] == 'analysis']) == 12
    assert sum(row.get('purpose') == 'verify' for row in task['steps']) == 6
    assert owner.definition_history(task['id'])[0]['questions'][1]['review_of'] == 'draft0'
    assert store(tmp_path).get_task(task['id']) == task
    with pytest.raises(ProjectTaskError):
        owner.begin_run(task['id'], expected_revision=task['revision'], max_calls=7,
                        source_snapshots={'alpha': 'a' * 64})
    with pytest.raises(ProjectTaskError):
        owner.create_task(goal='No hidden expansion.', questions=[{'id': f'q{i}', 'question': 'Write.'} for i in range(7)])


@pytest.mark.parametrize('mutation', ['foreign', 'no_dependency', 'different_question', 'wider_scope', 'all_documents',
    'kind', 'compose_target', 'invalid_purpose', 'cycle'])
def test_review_cannot_change_the_question_expand_scope_or_hide_a_dependency(tmp_path, mutation):
    rows = pair()
    audit = rows[1]
    if mutation == 'foreign':
        audit['review_of'] = 'foreign'
    elif mutation == 'no_dependency':
        audit['depends_on'] = []
    elif mutation == 'different_question':
        audit['question'] = 'An easier substitute.'
    elif mutation == 'wider_scope':
        audit['document_ids'] = ['alpha', 'private']
    elif mutation == 'all_documents':
        audit.pop('document_ids')
    elif mutation == 'kind':
        audit.pop('document_ids'); audit.update(input_kind='personal', memory_ids=[])
    elif mutation == 'compose_target':
        audit['purpose'] = 'compose'
    elif mutation == 'invalid_purpose':
        audit['purpose'] = []
    else:
        rows[0]['depends_on'] = [audit['id']]
    with pytest.raises(ProjectTaskError):
        store(tmp_path).create_task(goal='Original task.', questions=rows)


def test_review_can_be_listed_before_its_target_and_cannot_borrow_more_private_memories(tmp_path):
    rows = pair()[::-1]
    task = store(tmp_path).create_task(goal='Topology, not display order.', questions=rows)
    assert task['steps'][1]['review_of'] == 'draft0'
    memory = uid()
    rows = pair()
    for row in rows:
        row.pop('document_ids'); row.update(input_kind='personal', memory_ids=[memory])
    rows[1]['memory_ids'].append(uid())
    with pytest.raises(ProjectTaskError):
        store(tmp_path).create_task(goal='Keep private scope.', questions=rows)


@pytest.mark.parametrize('mutation', ['foreign', 'no_dependency', 'question', 'wider_scope',
                                    'narrower_scope', 'all_documents', 'kind', 'verify'])
def test_typed_refinement_cannot_replace_another_question_or_change_source_authority(tmp_path, mutation):
    rows = pair()
    rows[0]['document_ids'] = rows[1]['document_ids'] = ['alpha', 'beta']
    refinement = {**deepcopy(rows[0]), 'id': 'refined', 'refine_of': 'audit0',
                  'depends_on': ['audit0']}
    if mutation == 'foreign':
        refinement['refine_of'] = 'foreign'
    elif mutation == 'no_dependency':
        refinement['depends_on'] = []
    elif mutation == 'question':
        refinement['question'] += ' A different and easier target.'
    elif mutation == 'wider_scope':
        refinement['document_ids'].append('private')
    elif mutation == 'narrower_scope':
        refinement['document_ids'] = ['alpha']
    elif mutation == 'all_documents':
        refinement.pop('document_ids')
    elif mutation == 'kind':
        refinement.pop('document_ids'); refinement.update(input_kind='personal', memory_ids=[])
    else:
        refinement.update(purpose='verify', review_of='audit0')
    with pytest.raises(ProjectTaskError):
        store(tmp_path).create_task(goal='Preserve the approved requirement.', questions=[*rows, refinement])


def test_typed_refinement_survives_restart_and_can_target_a_review_without_expanding_memories(tmp_path):
    rows = pair()
    memory = uid()
    for row in rows:
        row.pop('document_ids'); row.update(input_kind='personal', memory_ids=[memory])
    refined = {**deepcopy(rows[0]), 'id': 'refined', 'refine_of': 'audit0', 'depends_on': ['audit0']}
    task = store(tmp_path).create_task(goal='Keep the complete original personal goal.', questions=[*rows, refined])
    assert next(row for row in task['steps'] if row['id'] == 'refined')['refine_of'] == 'audit0'
    assert store(tmp_path).get_task(task['id']) == task
    refined['memory_ids'] = []
    with pytest.raises(ProjectTaskError):
        store(tmp_path).create_task(goal='Do not discard selected requirements.', questions=[*rows, refined])


@pytest.mark.parametrize('replacement_status', ['ready', 'blocked', 'stale'])
def test_pending_or_revoked_refinement_never_reactivates_replaced_positive_proof(replacement_status):
    from tests.test_project_obligation_ledger_unit import artifact
    task = task_packet()
    original = task['steps'][0]
    original.update(document_ids=['doc-one'], depends_on=[])
    replacement = {**deepcopy(original), 'id': 'refined', 'refine_of': original['id'],
                   'depends_on': [original['id']], 'status': replacement_status, 'artifact_version': 0}
    task['steps'].append(replacement)
    positive = artifact(task, receipt(task))
    progress = build_obligation_progress(task, {'a': positive}, eligible_step_ids=['a'])
    assert progress['coverage']['ready_for_review'] == 0
    assert progress['coverage']['not_assessed'] == progress['coverage']['total'] == 2
    assert all(row['owner_step_id'] == 'refined' for row in progress['obligations'])
    assert progress['goal_verified'] is progress['semantic_verified'] is False


@pytest.mark.parametrize('fail_review', [False, True])
def test_runtime_needs_a_second_explicit_approval_and_review_failure_keeps_the_draft(tmp_path, fail_review):
    async def scenario():
        owner = store(tmp_path)
        task = owner.create_task(goal='Explain current records.', questions=pair())
        calls, answers = [], {}
        root = None

        async def execute(**kwargs):
            nonlocal root
            calls.append(kwargs)
            if kwargs.get('purpose') == 'verify' and fail_review:
                return {'request_status': 'failed'}
            answer_id = uid()
            if root is None:
                root = answer_id
            parent = (kwargs['dependency_artifacts'][0]['references']['answer_id']
                      if kwargs.get('purpose') == 'verify' else None)
            answers[answer_id] = {'content': BODY, 'sources': []}
            return {'request_status': 'completed', 'answer_id': answer_id,
                'source_ids': ['source-alpha'], 'document_ids': ['alpha'],
                'analysis_review': lineage(root, parent)}

        runtime = ProjectTaskRuntime(owner, execute_question=execute,
            read_sources=lambda **_: {'alpha': 'a' * 64}, load_answer=lambda **kwargs: answers[kwargs['answer_id']],
            validate_inputs=lambda _: True)
        first = await runtime.run(task['id'], expected_revision=task['revision'], run_request_id=uid(), max_calls=1)
        assert len(calls) == 1 and first['status'] == 'paused'
        draft = owner.get_artifact(task['id'], 'draft0')
        assert draft['available'] and draft['references']['answer_id'] == root
        assert next(row for row in first['steps'] if row['id'] == 'audit0')['status'] == 'ready'
        # A status/review projection does not purchase the audit.
        runtime.review_current_feedback(owner.get_task(task['id']))
        assert len(calls) == 1
        resumed = owner.resume_task(task['id'], expected_revision=first['revision'])
        result = await runtime.run(task['id'], expected_revision=resumed['revision'], run_request_id=uid(), max_calls=1)
        assert len(calls) == 2 and calls[-1]['purpose'] == 'verify' and calls[-1]['review_of'] == 'draft0'
        assert calls[-1]['question'] == calls[0]['question']
        assert calls[-1]['origin_task_id'] == task['id']
        assert owner.get_artifact(task['id'], 'draft0') == draft
        audit = next(row for row in result['steps'] if row['id'] == 'audit0')
        assert audit['status'] == ('blocked' if fail_review else 'completed')
        assert result['status'] == ('blocked' if fail_review else 'completed')
        assert result['run_remaining'] == 0
    asyncio.run(scenario())


def lineage_progress(*, broken=None, adverse=False, live=('a', 'audit')):
    task = task_packet()
    task['steps'][0].update(title='Writer', depends_on=[])
    audit = {**deepcopy(task['steps'][0]), 'id': 'audit', 'purpose': 'verify', 'review_of': 'a', 'depends_on': ['a']}
    task['steps'].append(audit)
    root, answer = uid(), uid()
    unknown = event_receipt(status='unknown')
    supported = event_receipt(status='model_assessed_counterevidence' if adverse else 'model_assessed_support')
    # Include a whole-path obligation to test its stable context identity too.
    for event in (unknown, supported):
        event['obligations'].append({**deepcopy(event['obligations'][0]), 'id': 'P1', 'kind': 'path'})
        event['obligations'][-1].pop('relation_id', None)
        key = 'counterevidence' if adverse and event is supported else 'supported' if event is supported else 'unknown'
        event[key] += 1
        if event is unknown:
            event['gaps'].append({'obligation_id': 'P1', 'code': 'semantic_support_unassessed'})
    artifacts = {}
    for step, identifier, event, task_value, marker in [
        (task['steps'][0], root, unknown, receipt(task, verdict='unknown'), lineage(root)),
        (audit, answer, supported, receipt(task), lineage(root, root))]:
        artifacts[step['id']] = {'answer_id': identifier, 'analysis_review': marker,
            'event_reasoning': event, 'task_review': task_value,
            'evidence_feedback_binding': bind_feedback(event, task_id=task['id'], task_revision=4,
                step_id=step['id'], artifact_version=1),
            'task_review_binding': bind_feedback(task_value, task_id=task['id'], task_revision=4,
                step_id=step['id'], artifact_version=1)}
    if broken == 'context':
        artifacts['audit']['analysis_review']['context_sha256'] = 'f' * 64
    elif broken == 'body':
        artifacts['audit']['analysis_review']['root_body_sha256'] = 'f' * 64
    elif broken == 'target':
        artifacts['audit']['analysis_review']['target_answer_id'] = uid()
    elif broken == 'dependency':
        audit['depends_on'] = []
    elif broken == 'question':
        audit['question'] += ' Change.'
    return build_obligation_progress(task, artifacts, eligible_step_ids=list(live))


def test_current_explicit_same_body_context_lineage_merges_old_unknown_without_certifying_causality():
    progress = lineage_progress()
    events = [row for row in progress['obligations'] if row['kind'].startswith('event_')]
    questions = [row for row in progress['obligations'] if row['requirement_id'] == 'Q1']
    assert len(events) == 2 and len(questions) == 1
    assert all(row['status'] == 'ready_for_review' for row in events + questions if row['kind'] != 'event_path')
    # This deliberately old path witness has not checked occurrences; merging
    # its lineage must not bypass the existing conservative schema migration.
    assert next(row for row in events if row['kind'] == 'event_path')['status'] == 'unresolved'
    assert progress['next_action'] != 'review'
    assert progress['goal_verified'] is progress['semantic_verified'] is False


@pytest.mark.parametrize('broken', ['context', 'body', 'target', 'dependency', 'question'])
def test_another_draft_scope_or_unbound_review_cannot_fill_the_writer_proof(broken):
    progress = lineage_progress(broken=broken)
    assert progress['next_action'] != 'review'
    assert progress['coverage']['ready_for_review'] < progress['coverage']['total']


def test_retracted_review_cannot_fill_writer_unknown_and_counterevidence_wins():
    missing = lineage_progress(live=('a',))
    assert missing['next_action'] != 'review'
    adverse = lineage_progress(adverse=True)
    assert adverse['coverage']['contradicted'] >= 2 and adverse['next_action'] != 'review'
