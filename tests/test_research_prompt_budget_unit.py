"""Whole-block prompt budgeting is pure formatting, never repeat retrieval."""
from copy import deepcopy
import json

import pytest

from docthinker.server.research_prompt_budget import empty_assistance, fit_research_prompt, retrieval_question, requires_document_evidence
from docthinker.server.research_retrieval import build_retrieval_plan, grounded_prompt
from docthinker.server.research_service import TRUSTED_RESEARCH_POLICY


def plan():
    return build_retrieval_plan('Energy imports', [
        {'id': 'one', 'document_id': 'doc-one', 'content': 'Energy reduces imports.'},
        {'id': 'two', 'document_id': 'doc-two', 'content': 'Energy increases imports under low output.'},
    ])


def fit(value, support, **kwargs):
    return fit_research_prompt('Energy imports', value, [], support,
                               system_policy=TRUSTED_RESEARCH_POLICY, **kwargs)


def test_measurement_is_complete_adapter_bytes_not_claimed_vendor_tokens():
    value = plan()
    result = fit(value, empty_assistance(value['sources']))
    measured = len(result['prompt'].encode()) + len(TRUSTED_RESEARCH_POLICY.encode()) + len('systemuser')
    assert result['budget']['input_bytes'] == measured <= 24576
    assert result['budget']['measurement'] == 'utf8_bytes_not_billed_tokens'
    assert result['budget']['retrieval_passes'] == 1
    assert result['budget']['assist_bytes'] == 0
    assert result['plan']['sources'] == value['sources']


def test_experimental_condition_policy_is_trusted_and_fully_budgeted():
    # This tests the input contract, not the model's semantic faithfulness.
    # Whether real answers keep the conditions requires independent A/B review.
    assert '任务、对象、指标、对照、阶段及适用范围' in TRUSTED_RESEARCH_POLICY
    assert '改变项和固定项' in TRUSTED_RESEARCH_POLICY
    assert '指标定义、单位和原文不确定性' in TRUSTED_RESEARCH_POLICY
    assert '不同对象或阶段不可混比' in TRUSTED_RESEARCH_POLICY
    assert '未见控制说明不等于没有控制' in TRUSTED_RESEARCH_POLICY
    assert '回指前一实验的条件' in TRUSTED_RESEARCH_POLICY
    value = plan()
    original = deepcopy(value)
    result = fit(value, empty_assistance(value['sources']))
    assert result['plan']['sources'] == original['sources']
    assert result['budget']['input_bytes'] == (
        len(result['prompt'].encode()) + len(TRUSTED_RESEARCH_POLICY.encode()) + len('systemuser'))
    assert result['reasoning']['trace']['semantic_entailment_verified'] is False
    assert result['messages'] == [
        {'role': 'system', 'content': TRUSTED_RESEARCH_POLICY},
        {'role': 'user', 'content': result['prompt']},
    ]


def test_oversized_optional_analysis_and_its_extra_sources_leave_together():
    value = plan()
    support = empty_assistance([*value['sources'], {**value['sources'][0], 'id': 'extra', 'label': 'S3'}])
    support['prompt'] = 'never split this analysis ' * 1500
    support['reasoning']['conclusions'] = [{'text': 'conditional auxiliary item'}]
    support['reasoning']['claims'] = [{'id': 'do not keep orphan premise'}]
    result = fit(value, support)
    assert result['plan']['sources'] == value['sources']
    assert result['reasoning']['conclusions'] == result['reasoning']['claims'] == []
    assert result['reasoning']['trace']['skipped_reason'] == 'prompt_budget'
    assert 'never split this analysis' not in result['prompt']
    assert 'S3' not in result['prompt']
    assert result['budget']['assist_bytes'] == 0


def test_source_fallback_omits_whole_excerpts_never_cuts_qualifier():
    value = plan()
    for source in value['sources']:
        source['content'] += ' very long condition' * 45
    value['selected_chunks'] = value['sources']
    first = {**value, 'selected_chunks': value['sources'][:1]}
    # Leave enough for precisely one whole excerpt and the omission warning.
    policy = 's' * (24576 - len(grounded_prompt('Energy imports', first).encode()) - 500)
    result = fit_research_prompt('Energy imports', value, [], empty_assistance(value['sources']), system_policy=policy)
    assert len(result['plan']['sources']) == 1
    assert result['plan']['sources'][0] == value['sources'][0]
    assert result['plan']['sources'][0]['content'].endswith('very long condition')
    assert any(row['code'] == 'context_budget_omitted' for row in result['plan']['gaps'])


def test_oversized_current_question_is_not_silently_truncated():
    value = plan()
    with pytest.raises(ValueError, match='budget'):
        fit_research_prompt('x' * 9000, value, [], empty_assistance(value['sources']), system_policy=TRUSTED_RESEARCH_POLICY)


def test_formatter_does_not_mutate_original_plan_or_assistance():
    value = plan()
    support = empty_assistance(value['sources'])
    original, old_support = deepcopy(value), deepcopy(support)
    fit(value, support)
    assert value == original and support == old_support


def test_concrete_new_question_does_not_expand_with_old_topic():
    history = [{'role': 'user', 'content': 'irrelevant oldtopic'}, {'role': 'assistant', 'content': 'never a query axiom'}]
    assert retrieval_question('How do solar cells work?', history) == 'How do solar cells work?'
    assert 'irrelevant oldtopic' in retrieval_question('Please expand', history)
    assert 'never a query axiom' not in retrieval_question('Please expand', history)
    assert retrieval_question('请再解释一下', []) == '请再解释一下'


@pytest.mark.parametrize('question,expected', [
    ('帮我起草一份使用文档', False), ('Write a document template.', False),
    ('根据上传的论文写一段论证', True), ('What does the paper say?', True),
    ('论文对 unrelated protein 有什么证据？', True),
    ('Write a paragraph according to the paper.', True), ('How to make a paperplane?', False),
])
def test_creating_documents_is_different_from_claiming_source_support(question, expected):
    assert requires_document_evidence(question) is expected


def test_model_sees_offsets_to_resolve_background_absolute_premise_references():
    value = build_retrieval_plan('renewable energy', [{
        'id': 'late', 'document_id': 'long',
        'content': 'irrelevant words. ' * 120 + 'Renewable energy reduces imports.' * 30,
    }])
    result = fit_research_prompt('renewable energy', value, [], empty_assistance(value['sources']), system_policy=TRUSTED_RESEARCH_POLICY)
    payload, _ = json.JSONDecoder().raw_decode(result['prompt'].split('UNTRUSTED_RESEARCH_DATA_JSON:\n', 1)[1])
    source = payload['source_quotes'][0]
    assert source['excerpt_start'] > 0
    assert source['excerpt_end'] - source['excerpt_start'] == len(source['content'])
    assert source['offset_unit'] == 'characters_in_source_chunk'
