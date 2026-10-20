"""Generic work/personal task review, with only local JSON/coordinate checks."""
from copy import deepcopy
import hashlib
import json

import pytest

from docthinker.server.research_task_review import (
    MAX_REVIEW_BYTES, build_task_requirements, validate_task_requirements,
    render_task_requirements, task_review_contract, parse_task_response,
    hydrate_task_review, normalize_task_review, task_response_without_assessment,
    task_requirement_rows, build_task_followup, validate_task_followup, task_followup_queries,
    MAX_FOLLOWUP_BYTES, MAX_FOLLOWUP_MODEL_BYTES, render_task_followup,
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
    for identifier in [row['id'] for row in task_requirement_rows(req)]:
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


def compact_envelope(req=None, sources=None):
    req, sources = req or requirements(), sources or [source()]
    answer = 'Draft: the handover is Friday; approval is still required [S1].'
    return {'answer': answer, 'task_assessment': {'requirements': [
        {'id': row['id'], 'verdict': 'addressed',
         'evidence': [{'label': item['label']} for item in sources],
         'answer_witness': {'whole_answer': True}}
        for row in task_requirement_rows(req)]}}


def test_compact_source_catalog_and_whole_answer_are_host_bound_locations_not_semantic_proof():
    original = source(text='内部交接在周五。未经批准，不得外发。', start=31)
    value = compact_envelope(sources=[original])
    value['answer'] = '周五交接；审批未取得，不得外发。[S1]'
    answer, receipt = parsed(value, sources=[original])
    assert receipt['addressed'] == receipt['supported'] == 3
    for row in receipt['obligations']:
        assert row['evidence'] == [{'label': 'S1', 'source_id': original['id'],
            'span': [31, 31 + len(original['content'])],
            'quote_sha256': hashlib.sha256(original['content'].encode()).hexdigest()}]
        assert row['answer_witness'] == {'span': [0, len(answer)],
            'quote_sha256': hashlib.sha256(answer.encode()).hexdigest()}
        assert row['status'] == 'model_assessed_addressed' and row['basis'] == 'source_and_answer'
    assert all(receipt[key] is False for key in ('semantic_verified', 'causality_verified', 'goal_verified'))
    assert hydrate_task_review(receipt, [original], answer) == receipt
    assert all('quote' not in row['answer_witness'] and 'whole_answer' not in row['answer_witness']
               for row in receipt['obligations'])


@pytest.mark.parametrize('composition', ['prefix', 'trial_append', 'trial_reorder'])
def test_whole_answer_binds_the_exact_final_delivery_after_host_and_trial_composition(composition):
    value = compact_envelope()
    draft = value['answer']
    table = '\n| Check | Result |\n|---|---|\n| Approval | Missing |\n'
    final = {'prefix': 'Host delivery notice: draft only.\n\n' + draft,
        'trial_append': draft + table,
        'trial_reorder': table + '\n' + draft}[composition]
    answer, receipt = parsed(value, final_answer=final)
    assert answer == final and receipt['answer_sha256'] == hashlib.sha256(final.encode()).hexdigest()
    assert all(row['answer_witness'] == {'span': [0, len(final)],
        'quote_sha256': hashlib.sha256(final.encode()).hexdigest()} for row in receipt['obligations'])
    assert hydrate_task_review(receipt, [source()], final) == receipt
    assert hydrate_task_review(receipt, [source()], draft) is None
    assert receipt['goal_verified'] is False and receipt['semantic_verified'] is False


def test_whole_answer_locator_does_not_require_repeating_a_long_visible_draft():
    value = compact_envelope()
    value['answer'] = 'A directly usable draft with conditions retained. ' * 100
    answer, receipt = parsed(value)
    assert len(answer) > 400 and receipt['addressed'] == 3
    assert all(row['answer_witness']['span'] == [0, len(answer)] for row in receipt['obligations'])
    assert hydrate_task_review(receipt, [source()], answer) == receipt


@pytest.mark.parametrize('witness', [{'whole_answer': False}, {'whole_answer': 1}, {'whole_answer': 'true'},
    {'whole_answer': True, 'quote': 'Model-selected body'}, {'whole_answer': True, 'start': 0},
    {'whole_answer': True, 'quote_sha256': 'a' * 64}, None])
def test_whole_answer_is_a_strict_host_directive_not_model_supplied_coordinates(witness):
    value = compact_envelope()
    value['task_assessment']['requirements'][0]['answer_witness'] = witness
    answer, receipt = parsed(value)
    assert answer == value['answer']
    assert receipt['obligations'][0]['status'] == 'unknown'
    assert receipt['obligations'][0]['gaps'] == ['answer_witness_missing_or_changed']


@pytest.mark.parametrize('final', ['', ' \n\t'])
def test_empty_final_delivery_cannot_receive_a_whole_answer_locator(final):
    with pytest.raises(ValueError, match='Final task answer is unavailable'):
        parsed(compact_envelope(), final_answer=final)


@pytest.mark.parametrize('change', ['missing_label', 'empty', 'wrong_end', 'negative_start', 'boolean_start'])
def test_label_only_witness_does_not_bypass_current_source_window_geometry(change):
    value, original = compact_envelope(), source()
    if change == 'missing_label':
        value['task_assessment']['requirements'][0]['evidence'] = [{'label': 'S9'}]
    elif change == 'empty':
        original['content'] = ''
    elif change == 'wrong_end':
        original['excerpt_end'] += 1
    elif change == 'negative_start':
        original['excerpt_start'] = -1
    else:
        original['excerpt_start'] = True
    answer, receipt = parsed(value, sources=[original])
    assert answer == value['answer'] and receipt['obligations'][0]['status'] == 'unknown'
    assert receipt['obligations'][0]['gaps'] == ['source_witness_invalid']


def test_compact_protocol_retains_all_26_original_goals_without_a_two_item_limit():
    req = build_task_requirements('Deliver the original draft', [f'Original phase rule {i}' for i in range(12)],
        project_goal='Original project goal', project_acceptance_criteria=[f'Original project rule {i}' for i in range(12)])
    _, receipt = parsed(compact_envelope(req), req)
    assert len(receipt['obligations']) == receipt['addressed'] == 26
    assert [row['id'] for row in receipt['obligations']] == [row['id'] for row in task_requirement_rows(req)]
    policy = task_review_contract()
    assert '"answer_witness":{"whole_answer":true}' in policy
    assert '"evidence":[{"label":"S1"}]' in policy
    assert 'Do not repeat source passages' in policy and 'Do not exhaustively' in policy
    assert 'at most TWO' not in policy and 'Assess EVERY' not in policy


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


def test_26_joint_proofs_downgrade_whole_needs_on_capacity_without_losing_answer():
    req = build_task_requirements('Deliver the requested summary',
        [f'Explicit phase criterion {i}' for i in range(12)], project_goal='Original whole project',
        project_acceptance_criteria=[f'Explicit project criterion {i}' for i in range(12)])
    sources = [source(f'S{i}', 'A joint source witness for the same original delivery.') for i in range(1, 4)]
    for item in sources:
        item['id'] = 'source-' + 'x' * 240 + item['label']
    value = envelope(req, sources)
    answer, review = parsed(value, req, sources)
    assert answer == value['answer'] and len(review['obligations']) == 26
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


def test_missing_review_keeps_all_requirements_unknown_without_blocking_normal_delivery():
    req = build_task_requirements('Draft the requested summary', ['Check A', 'Check B'])
    answer, receipt = parse_task_response(json.dumps({'answer': 'A usable draft.',
        'task_assessment': {'requirements': []}}), req, [])
    assert answer == 'A usable draft.'
    assert [row['id'] for row in receipt['obligations']] == ['Q1', 'C1', 'C2']
    assert receipt['unknown'] == 3 and receipt['addressed'] == 0
    assert not receipt['goal_verified']
    policy = task_review_contract()
    assert 'answer FIRST' in policy and 'at most TWO' not in policy
    assert 'omitted items stay unknown' in policy and 'host keeps EVERY' in policy
    assert 'without a fixed item cap' in policy and 'Do not exhaustively' in policy


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


def test_project_goal_and_project_and_phase_criteria_have_distinct_original_hashes():
    req = build_task_requirements('Prepare a draft, never send it.', ['Keep the pending approval visible.'],
        project_goal='Complete the two-week handover without unauthorized sending.',
        project_acceptance_criteria=['Account for all owners.', 'Do not send externally.'])
    assert req['version'] == 2
    rows = task_requirement_rows(req)
    assert [row['id'] for row in rows] == ['Q1', 'G1', 'P1', 'P2', 'C1']
    assert [row['kind'] for row in rows] == ['task', 'project_goal', 'project_criterion', 'project_criterion', 'criterion']
    assert rows[1]['text'] == req['project_goal'] and rows[3]['text'] == req['project_acceptance_criteria'][1]
    slots = json.loads(render_task_requirements(req))['requirements']
    assert slots[1]['slot'] == 'project_goal' and slots[3]['slot'] == 'project_acceptance_criteria[1]'
    _, review = parsed(envelope(req), req)
    assert review['requirements_version'] == 2 and review['unknown'] == 0
    assert [row['requirement_sha256'] for row in review['obligations']] == [row['requirement_sha256'] for row in rows]
    assert hydrate_task_review(review, [source()], envelope(req)['answer']) == review
    assert validate_task_requirements(req) == req
    changed = build_task_requirements(req['question'], req['acceptance_criteria'],
        project_goal=req['project_goal'], project_acceptance_criteria=['Account for all owners.', 'Now send it.'])
    assert changed['requirements_sha256'] != req['requirements_sha256']
    assert task_requirement_rows(changed)[3]['requirement_sha256'] != rows[3]['requirement_sha256']
    assert task_requirement_rows(changed)[4]['requirement_sha256'] == rows[4]['requirement_sha256']


def test_all_26_original_requirements_survive_an_empty_assessment_and_are_not_renamed():
    req = build_task_requirements('Original question', [f'Phase requirement {i}' for i in range(12)],
        project_goal='Original project', project_acceptance_criteria=[f'Project requirement {i}' for i in range(12)])
    answer, review = parse_task_response(json.dumps({'answer': 'Draft only.',
        'task_assessment': {'requirements': [{'id': 'EASIER', 'verdict': 'addressed'}]}}), req, [])
    assert len(review['obligations']) == review['unknown'] == 26 and review['addressed'] == 0
    assert [row['id'] for row in review['obligations']] == [row['id'] for row in task_requirement_rows(req)]
    assert normalize_task_review(review) == hydrate_task_review(review, [], answer)
    assert len(json.dumps(review, ensure_ascii=False).encode()) <= MAX_REVIEW_BYTES


def test_v2_query_hints_bind_to_their_own_requirement_not_the_first_unrelated_gap():
    req = build_task_requirements('Original step', ['Check staffing'], project_goal='Original goal',
                                  project_acceptance_criteria=['Check critical path'])
    value = envelope(req)
    for index, row in enumerate(value['task_assessment']['requirements']):
        row.update(verdict='unknown', missing_query=f'Exact missing record number {index}')
    answer, receipt = parsed(value, req)
    for index, row in enumerate(receipt['obligations']):
        assert row['lookup_query'] == f'Exact missing record number {index}'
        assert row['lookup_query_sha256'] == hashlib.sha256(f'Exact missing record number {index}'.encode()).hexdigest()
    assert hydrate_task_review(receipt, [source()], answer) == receipt
    receipt['obligations'][0]['lookup_query_sha256'] = 'a' * 64
    assert hydrate_task_review(receipt, [source()], answer) is None


def test_all_26_missing_queries_remain_on_their_original_rows_beyond_top8_frontier():
    req = build_task_requirements('Original question', [f'Phase requirement {i}' for i in range(12)],
        project_goal='Original project', project_acceptance_criteria=[f'Project requirement {i}' for i in range(12)])
    rows = [{'id': row['id'], 'verdict': 'unknown',
             'missing_query': f"Specific original missing evidence {row['id']} " + 'x' * 300}
            for row in task_requirement_rows(req)]
    answer, receipt = parse_task_response(json.dumps({'answer': 'Unresolved draft.',
        'task_assessment': {'requirements': rows}}), req, [])
    assert len(receipt['suggested_queries']) == 8
    assert len(receipt['obligations']) == 26
    assert [row['lookup_query'] for row in receipt['obligations']] == [row['missing_query'] for row in rows]
    assert hydrate_task_review(receipt, [], answer) == receipt


@pytest.mark.parametrize('mutation', ['version', 'order', 'kind', 'digest', 'extra'])
def test_v2_receipt_cannot_drop_reorder_or_retag_project_goals(mutation):
    req = build_task_requirements('Original question', ['Phase rule'], project_goal='Project goal',
                                  project_acceptance_criteria=['Project rule'])
    answer, receipt = parsed(envelope(req), req)
    if mutation == 'version':
        receipt['requirements_version'] = True
    elif mutation == 'order':
        receipt['obligations'][1], receipt['obligations'][2] = receipt['obligations'][2], receipt['obligations'][1]
    elif mutation == 'kind':
        receipt['obligations'][1]['kind'] = 'criterion'
    elif mutation == 'digest':
        receipt['obligations'][1]['requirement_sha256'] = 'invalid'
    else:
        receipt['obligations'][1]['text'] = 'Another goal'
    assert hydrate_task_review(receipt, [source()], answer) is None


def followup_row(index=0, *, priority='missing_evidence', query_count=2, large=False):
    return {'instance_id': 'private-instance', 'task_id': f'00000000-0000-4000-8000-{index:012d}',
        'step_id': f'q{index}', 'artifact_version': 1,
        'answer_id': f'10000000-0000-4000-8000-{index:012d}', 'review_kind': 'task',
        'receipt_sha256': hashlib.sha256(str(index).encode()).hexdigest(), 'priority': priority,
        'queries': [f'dependency {index} query {i} ' + ('x' * 350 if large else '') for i in range(query_count)],
        'gaps': [{'obligation_id': f'C{i + 1}', 'code': 'requirement_unassessed'} for i in range(12 if large else 1)],
        'omitted_query_count': 0, 'omitted_gap_count': 0}


def test_followup_prioritizes_counterevidence_then_round_robins_every_dependency():
    rows = [followup_row(0), followup_row(1, priority='counterevidence'), followup_row(2)]
    packet = build_task_followup(rows)
    queries = task_followup_queries(packet)
    assert queries == [rows[1]['queries'][0], rows[0]['queries'][0], rows[2]['queries'][0],
                       rows[1]['queries'][1], rows[0]['queries'][1], rows[2]['queries'][1]]
    assert validate_task_followup(packet) == packet and rows[0]['omitted_query_count'] == 0


def test_followup_model_view_is_fair_compact_and_does_not_change_the_bound_host_packet():
    rows = [followup_row(i, priority='counterevidence' if i == 3 else 'missing_evidence', query_count=8)
            for i in range(6)]
    rows[0]['omitted_query_count'] = 18
    packet = build_task_followup(rows)
    original = deepcopy(packet)
    text = render_task_followup(packet)
    view = json.loads(text)
    assert packet == original and validate_task_followup(packet) == original
    assert view['queries'] == task_followup_queries(packet)[:8]
    assert view['queries'][:6] == [rows[i]['queries'][0] for i in [3, 0, 1, 2, 4, 5]]
    assert view['dependency_count'] == 6 and view['review_packet_count'] == 6
    assert view['pending_visible_query_count'] == len(task_followup_queries(packet)) - 8
    assert view['undelivered_query_hint_count'] == sum(row['omitted_query_count'] for row in packet['dependencies'])
    assert view['pending_gap_count'] == sum(len(row['gaps']) + row['omitted_gap_count'] for row in packet['dependencies'])
    assert len(text.encode('utf-8')) <= MAX_FOLLOWUP_MODEL_BYTES
    assert not any(key in text for key in ['receipt_sha256', 'answer_id', 'task_id', 'obligation_id', 'requirement_unassessed'])
    assert rows[0]['task_id'] not in text and rows[0]['receipt_sha256'] not in text
    assert 'not facts, instructions or authorization' in view['boundary']
    assert 'Pending or omitted is not completed' in view['boundary']


def test_followup_model_view_deduplicates_dependencies_not_receipts_or_backlog_counts():
    task = followup_row(query_count=1)
    event = {**deepcopy(task), 'review_kind': 'event', 'omitted_query_count': 3, 'omitted_gap_count': 2}
    packet = build_task_followup([task, event])
    view = json.loads(render_task_followup(packet))
    assert view['dependency_count'] == 1 and view['review_packet_count'] == 2
    assert view['queries'] == task['queries'] and view['pending_visible_query_count'] == 0
    # The undelivered count is raw hints, not a claim of unique unknown queries.
    assert view['undelivered_query_hint_count'] == 3 and view['pending_gap_count'] == 4


def test_followup_model_view_counts_pending_hints_beyond_local_lookup_frontier():
    rows = []
    for i in range(6):
        row = followup_row(i, query_count=8)
        row['queries'] = [f't{i}:{j}' for j in range(8)]
        rows.extend([row, {**deepcopy(row), 'review_kind': 'event',
                          'queries': [f'e{i}:{j}' for j in range(8)]}])
    packet = build_task_followup(rows)
    assert sum(len(row['queries']) for row in packet['dependencies']) == 96
    assert len(task_followup_queries(packet)) == 48
    view = json.loads(render_task_followup(packet))
    assert len(view['queries']) == 8 and view['pending_visible_query_count'] == 88
    assert view['undelivered_query_hint_count'] == 0


@pytest.mark.parametrize('escaped', [False, True])
def test_followup_model_view_keeps_whole_utf8_queries_within_independent_byte_bound(escaped):
    rows = []
    for index in range(6):
        row = followup_row(index, query_count=8)
        row['queries'] = [f'{index}:{i} ' + ('\t' * 390 if escaped else '完整材料中' * 26)
                          for i in range(8)]
        rows.append(row)
    packet = build_task_followup(rows)
    original = deepcopy(packet)
    text = render_task_followup(packet)
    view = json.loads(text)
    frontier = task_followup_queries(packet)
    assert len(text.encode('utf-8')) <= MAX_FOLLOWUP_MODEL_BYTES
    assert 0 < len(view['queries']) <= 8
    assert view['queries'] == frontier[:len(view['queries'])]
    assert all(query in frontier for query in view['queries'])
    assert view['pending_visible_query_count'] + len(view['queries']) == len(frontier)
    assert packet == original


def test_followup_model_view_validates_the_entire_original_packet_before_rendering():
    packet = build_task_followup([followup_row()])
    packet['dependencies'][0]['receipt_sha256'] = 'invalid'
    with pytest.raises(ValueError):
        render_task_followup(packet)


def test_full_followup_backlog_is_not_claimed_complete_when_wire_frontier_is_bounded():
    rows = []
    for index in range(6):
        row = followup_row(index, query_count=8, large=True)
        rows.extend([row, {**deepcopy(row), 'review_kind': 'event'}])
    packet = build_task_followup(rows)
    assert len(json.dumps(packet, ensure_ascii=False, separators=(',', ':')).encode()) <= MAX_FOLLOWUP_BYTES
    assert len(packet['dependencies']) == 12
    for original, delivered in zip(rows, packet['dependencies']):
        assert len(delivered['queries']) + delivered['omitted_query_count'] == len(original['queries'])
        assert len(delivered['gaps']) + delivered['omitted_gap_count'] == len(original['gaps'])
        assert delivered['receipt_sha256'] == original['receipt_sha256']
    assert any(row['omitted_query_count'] for row in packet['dependencies'])
    assert len(task_followup_queries(packet)) <= 48


def test_event_followup_supports_all_48_gap_codes_without_silently_claiming_completion():
    row = followup_row()
    row.update(review_kind='event', gaps=[{'obligation_id': f'E{i % 24 + 1}',
        'code': f'gap_{i}'} for i in range(48)])
    packet = build_task_followup([row])
    delivered = packet['dependencies'][0]
    assert len(delivered['gaps']) + delivered['omitted_gap_count'] == 48
    assert validate_task_followup(packet) == packet
    row['review_kind'] = 'task'
    with pytest.raises(ValueError):
        build_task_followup([row])


def test_repacking_a_bounded_frontier_does_not_erase_previously_deferred_queries():
    rows = [followup_row(i, query_count=8, large=True) for i in range(6)]
    for row in rows:
        row['omitted_query_count'] = 18
    packet = build_task_followup(rows)
    assert all(len(row['queries']) + row['omitted_query_count'] == 26 for row in packet['dependencies'])


@pytest.mark.parametrize('mutation', ['extra', 'version', 'task_id', 'step_id', 'artifact_version', 'answer_id',
    'instance_id', 'receipt_sha256', 'review_kind', 'priority', 'queries', 'duplicate_query', 'gaps', 'omitted', 'repeat', 'wide'])
def test_followup_rejects_foreign_shape_unbounded_or_retagged_queues(mutation):
    packet = build_task_followup([followup_row()])
    row = packet['dependencies'][0]
    if mutation == 'extra':
        row['source_text'] = 'Must not be copied'
    elif mutation == 'version':
        packet['version'] = True
    elif mutation == 'task_id':
        row['task_id'] = 'other task'
    elif mutation == 'step_id':
        row['step_id'] = '../another'
    elif mutation == 'artifact_version':
        row['artifact_version'] = True
    elif mutation == 'answer_id':
        row['answer_id'] = 'not an answer'
    elif mutation == 'instance_id':
        row['instance_id'] = '../owner'
    elif mutation == 'receipt_sha256':
        row['receipt_sha256'] = 'bad'
    elif mutation == 'review_kind':
        row['review_kind'] = {}
    elif mutation == 'priority':
        row['priority'] = []
    elif mutation == 'queries':
        row['queries'] = ['中' * 134]
    elif mutation == 'duplicate_query':
        row['queries'] *= 2
    elif mutation == 'gaps':
        row['gaps'][0]['content'] = 'Must not be copied'
    elif mutation == 'omitted':
        row['omitted_query_count'] = 27
    elif mutation == 'repeat':
        packet['dependencies'].append(deepcopy(row))
    else:
        packet['dependencies'] = [followup_row(i, query_count=0) for i in range(7)]
    with pytest.raises(ValueError):
        validate_task_followup(packet)


def test_oversize_fitter_still_validates_every_deferred_query_not_only_selected_frontier():
    rows = [followup_row(i, query_count=8, large=True) for i in range(6)]
    rows[-1]['queries'][-1] = {'not': 'a query'}
    with pytest.raises(ValueError):
        build_task_followup(rows)
