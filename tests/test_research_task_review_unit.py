"""Generic work/personal task review, with only local JSON/coordinate checks."""
from copy import deepcopy
import hashlib
import json

import pytest

from docthinker.server.research_task_review import (
    MAX_REVIEW_BYTES, build_task_requirements, validate_task_requirements,
    render_task_requirements, task_review_contract, parse_task_response,
    hydrate_task_review, normalize_task_review, task_response_without_assessment,
)


def source(label='S1', text='The internal handover is Friday. Sending still needs approval.', start=20):
    return {'id': 'source-' + label, 'document_id': 'document-1', 'label': label,
            'content': text, 'excerpt_start': start, 'excerpt_end': start + len(text)}


def requirements(*, personal=False, criteria=None):
    return build_task_requirements('Draft a private handover note; do not send it.',
        ['Preserve the handover date.', 'Explicitly disclose the missing approval.'] if criteria is None else criteria,
        input_kind='personal' if personal else 'documents')


def envelope(req=None, sources=None, *, evidence=True):
    req = req or requirements()
    sources = sources or [source()]
    answer = 'Draft: the handover is Friday; approval is still required [S1].'
    rows = []
    for identifier in ['Q1', *[f'C{i + 1}' for i in range(len(req['acceptance_criteria']))]]:
        rows.append({'id': identifier, 'verdict': 'addressed',
            'evidence': [{'label': row['label'], 'quote': row['content'], 'start': row['excerpt_start']}
                         for row in sources] if evidence else [],
            'answer_witness': {'quote': answer, 'start': 0}, 'missing_query': 'Which approval is missing?'})
    return {'answer': answer, 'task_assessment': {'requirements': rows}}


def parsed(value=None, req=None, sources=None, **kwargs):
    req = req or requirements()
    return parse_task_response(json.dumps(value or envelope(req)), req, sources or [source()], **kwargs)


@pytest.mark.parametrize('goal', ['工作交接：分析延期并写出内部草稿', '家庭搬家：拟定两天安排但不得付款', '语言学习：比较旧计划和下周练习'])
def test_original_cross_domain_question_and_ordered_criteria_are_host_fixed(goal):
    req = build_task_requirements(goal, ['原始约束甲', '原始约束乙'])
    assert req['question'] == goal and req['acceptance_criteria'] == ['原始约束甲', '原始约束乙']
    packet = json.loads(render_task_requirements(req))
    assert [row['id'] for row in packet['requirements']] == ['Q1', 'C1', 'C2']
    assert packet['requirements'][2]['slot'] == 'phase_acceptance_criteria[1]'
    assert packet['requirements_sha256'] == req['requirements_sha256']
    assert len(render_task_requirements(req).encode()) + len(task_review_contract().encode()) < 4096
    assert build_task_requirements(goal + ' changed', ['原始约束甲', '原始约束乙'])['requirements_sha256'] != req['requirements_sha256']


@pytest.mark.parametrize('field,value', [('question', ''), ('question', True), ('question', 'x' * 2001),
    ('acceptance_criteria', ['x'] * 13), ('acceptance_criteria', ['x', 'x']),
    ('acceptance_criteria', ['x' * 1001]), ('input_kind', 'all_session')])
def test_invalid_or_widened_host_requirements_are_rejected(field, value):
    args = {'question': 'Original step', 'acceptance_criteria': [], 'input_kind': 'documents'}
    args[field] = value
    with pytest.raises(ValueError):
        build_task_requirements(**args)


def test_requirements_hash_prevents_model_or_caller_replacing_hard_goal():
    req = requirements()
    req['question'] = 'Say hello instead'
    with pytest.raises(ValueError, match='changed'):
        validate_task_requirements(req)


def test_exact_source_and_answer_witnesses_remain_model_assessed_not_verified():
    answer, review = parsed()
    assert review['addressed'] == review['supported'] == 3 and review['unknown'] == 0
    assert all(row['basis'] == 'source_and_answer' for row in review['obligations'])
    assert all(review[key] is False for key in ('goal_verified', 'semantic_verified', 'causality_verified'))
    assert hydrate_task_review(review, [source()], answer) == review
    assert normalize_task_review(review) == review
    assert 'quote' not in review['obligations'][0]['answer_witness']
    assert 'quote' not in review['obligations'][0]['evidence'][0]


@pytest.mark.parametrize('personal', [True, False])
def test_answer_only_draft_exists_but_never_becomes_source_support(personal):
    req = requirements(personal=personal)
    answer, review = parsed(envelope(req, evidence=False), req)
    assert review['addressed'] == 3 and review['supported'] == 0
    assert all(row['basis'] == 'answer_fulfilment_only' for row in review['obligations'])
    assert review['goal_verified'] is False and review['semantic_verified'] is False
    assert hydrate_task_review(review, [], answer) == review
    if not personal:
        assert review['gaps'] and all(row['gaps'] == ['factual_support_unassessed'] for row in review['obligations'])


def test_joint_three_source_witnesses_are_kept_whole_not_first_quote_only():
    req = requirements(criteria=[])
    sources = [source('S1'), source('S2', 'Only the internal draft is approved.'),
               source('S3', 'External sending is not authorized.')]
    answer, review = parsed(envelope(req, sources), req, sources)
    assert len(review['obligations'][0]['evidence']) == 3
    assert hydrate_task_review(review, sources, answer) == review
    changed = deepcopy(sources)
    changed[2]['content'] = 'X' + changed[2]['content'][1:]
    assert hydrate_task_review(review, changed, answer) is None


@pytest.mark.parametrize('mutation', ['omit', 'duplicate', 'unknown_id', 'different_answer', 'wrong_source', 'wrong_offset', 'false_offset', 'ambiguous', 'too_many'])
def test_missing_or_invalid_requirement_proof_is_unknown_not_repaired(mutation):
    value = envelope()
    row = value['task_assessment']['requirements'][0]
    sources = [source()]
    if mutation == 'omit':
        value['task_assessment']['requirements'].pop(0)
    elif mutation == 'duplicate':
        value['task_assessment']['requirements'].append(deepcopy(row))
    elif mutation == 'unknown_id':
        row['id'] = 'MODEL_EASY_GOAL'
    elif mutation == 'different_answer':
        row['answer_witness']['quote'] = 'This easier draft was not delivered'
    elif mutation == 'wrong_source':
        row['evidence'][0]['label'] = 'S9'
    elif mutation == 'wrong_offset':
        row['evidence'][0]['start'] = 21
    elif mutation == 'false_offset':
        row['evidence'][0]['start'] = True
    elif mutation == 'too_many':
        row['evidence'] *= 4
    else:
        row['answer_witness'] = {'quote': 'Draft'}
        value['answer'] += ' Draft'
    _, review = parsed(value, sources=sources)
    assert review['obligations'][0]['id'] == 'Q1'
    assert review['obligations'][0]['status'] == 'unknown'
    assert review['unknown'] >= 1 and review['gaps']


def test_counterevidence_requires_source_witness_and_never_satisfies_requirement():
    value = envelope()
    value['task_assessment']['requirements'][0]['verdict'] = 'counterevidence'
    _, review = parsed(value)
    assert review['counterevidence'] == 1 and review['addressed'] == 2
    assert review['obligations'][0]['gaps'] == ['requirement_counterevidence']
    assert review['status'] == 'evidence_gaps'


def test_final_delivery_body_is_authority_after_trial_composition():
    value = envelope()
    final = value['answer'] + '\nA host-rendered trial table (not executed).'
    answer, review = parsed(value, final_answer=final)
    assert answer == final and review['answer_sha256'] == hashlib.sha256(final.encode()).hexdigest()
    assert hydrate_task_review(review, [source()], final) == review
    assert hydrate_task_review(review, [source()], value['answer']) is None
    _, replaced = parsed(value, final_answer='Different actual final delivery')
    assert replaced['unknown'] == 3 and replaced['addressed'] == 0


def test_unstructured_answer_still_has_all_fixed_requirements_as_unknown():
    req = requirements()
    answer, review = parse_task_response('A normal unstructured draft', req, [])
    assert answer == 'A normal unstructured draft' and review['model_assessed'] is False
    assert review['unknown'] == 3 and review['addressed'] == 0
    assert review['suggested_queries'] and review['status'] == 'not_assessed'
    assert hydrate_task_review(review, [], answer) == review


def test_other_single_call_envelope_fields_are_not_reinterpreted_or_dropped():
    value = envelope()
    value.update(event_assessment={'links': []}, trial_spec={'an': 'opaque separate contract'})
    extracted = json.loads(task_response_without_assessment(json.dumps(value)))
    assert extracted == {k: v for k, v in value.items() if k != 'task_assessment'}
    _, review = parsed(value)
    assert review['addressed'] == 3


def test_13_joint_proofs_downgrade_whole_needs_on_capacity_without_losing_answer():
    req = requirements(criteria=[f'Explicit criterion {i}' for i in range(12)])
    sources = [source(f'S{i}', 'A joint source witness for the same original delivery.') for i in range(1, 4)]
    value = envelope(req, sources)
    answer, review = parsed(value, req, sources)
    assert answer == value['answer'] and len(review['obligations']) == 13
    assert len(json.dumps(review, ensure_ascii=False, separators=(',', ':')).encode()) <= MAX_REVIEW_BYTES
    downgraded = [row for row in review['obligations'] if row['gaps'] == ['receipt_budget_exceeded']]
    assert downgraded and all(row['status'] == 'unknown' and row['evidence'] == [] and 'answer_witness' not in row for row in downgraded)
    assert all(len(row['evidence']) == 3 for row in review['obligations'] if row['status'] == 'model_assessed_addressed')
    assert hydrate_task_review(review, sources, answer) == review


@pytest.mark.parametrize('mutation', ['source_id', 'quote_hash', 'answer_hash', 'witness_span', 'counts', 'basis', 'verified', 'extra_body'])
def test_rehydrate_rejects_corrupt_or_retagged_durable_proof(mutation):
    answer, review = parsed()
    row = review['obligations'][0]
    if mutation == 'source_id':
        row['evidence'][0]['source_id'] = 'foreign-source'
    elif mutation == 'quote_hash':
        row['evidence'][0]['quote_sha256'] = 'f' * 64
    elif mutation == 'answer_hash':
        review['answer_sha256'] = 'f' * 64
    elif mutation == 'witness_span':
        row['answer_witness']['span'][0] = True
    elif mutation == 'counts':
        review['addressed'] = True
    elif mutation == 'basis':
        row['basis'] = 'answer_fulfilment_only'
    elif mutation == 'verified':
        review['goal_verified'] = True
    else:
        row['answer'] = 'Do not store a second answer'
    assert hydrate_task_review(review, [source()], answer) is None


def test_sparse_review_keeps_every_requirement_unknown_without_delaying_answer():
    req = build_task_requirements('Draft the requested summary', ['Check A', 'Check B'])
    answer, receipt = parse_task_response(json.dumps({'answer': 'A usable draft.',
        'task_assessment': {'requirements': []}}), req, [])
    assert answer == 'A usable draft.'
    assert [row['id'] for row in receipt['obligations']] == ['Q1', 'C1', 'C2']
    assert receipt['unknown'] == 3 and receipt['addressed'] == 0
    assert not receipt['goal_verified']
    policy = task_review_contract()
    assert 'answer FIRST' in policy and 'at most TWO' in policy
    assert 'omitted items stay unknown' in policy and 'Assess EVERY' not in policy


@pytest.mark.parametrize('fenced', [True, False])
def test_unique_trailing_envelope_does_not_leak_internal_task_metadata(fenced):
    req = build_task_requirements('Draft a summary', ['Check A'])
    blob = json.dumps({'answer': 'The clean draft.', 'task_assessment': {'requirements': []}})
    raw = 'The clean draft.\n\n' + ('```json\n' + blob + '\n```' if fenced else blob)
    answer, review = parse_task_response(raw, req, [])
    assert answer == 'The clean draft.' and review['unknown'] == 2
    assert 'task_assessment' not in answer


def test_unsafe_internal_metadata_is_not_returned_as_normal_task_answer():
    req = build_task_requirements('Draft a summary')
    with pytest.raises(ValueError, match='Invalid research response envelope'):
        parse_task_response('Draft\n{"answer":"draft","task_assessment":', req, [])
