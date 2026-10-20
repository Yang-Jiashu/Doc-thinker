"""Goal/source/budget structure, not simulated model quality or scientific proof."""
from copy import deepcopy
import hashlib
import json

import pytest

from docthinker.server import research_decision_brief as brief


def fixture():
    data = {'findings': '当前资料只说明接口设计观察，未证明本地收益。[S1]',
        'do_now': '制定本地编程及多模态资料方案，保留当前约束。',
        'not_now': '不恢复已撤回的六小时方案，不执行代码。',
        'hypothesis': '在已确认适用条件下，简化接口可能降低同任务的遗漏。',
        'conditions_and_alternative': '只适用于当前任务；最强替代解释是材料难度不同。',
        'validation_target': '真实任务产物中的条件遗漏，而不是人工想象模型表现。',
        'validation_plan': '保留实际输入与产物，按明确条件逐项核对；资源未知部分待确认。',
        'refuter': '相同输入条件下遗漏不降或新增关键错误，将反驳这一预测。',
        'missing_policy': '缺失不当成功，不删除失败，无法判断则保留待核。',
        'day_one': '核对当前材料、资源与任务条件。', 'day_one_minutes': 30,
        'day_two': '整理检验设计及可讨论备忘录。', 'day_two_minutes': 40,
        'privacy': '每次执行、写入或截图导出分别征求同意；沙盒不豁免。'}
    sources = [{'id': 'source-a', 'label': 'S1', 'content': '当前原文，非实验观测。'}]
    return data, sources


def parse(data=None, sources=None, **kwargs):
    original, rows = fixture()
    return brief.parse(json.dumps(original if data is None else data, ensure_ascii=False),
        rows if sources is None else sources, trusted_budget_minutes=kwargs.pop('budget', 120),
        question=kwargs.pop('question', '两天内交付私人助手决策备忘录。'),
        goal=kwargs.pop('goal', '本地编程与多模态资料方案，不缩成论文审阅。'),
        constraints=kwargs.pop('constraints', '设计研究，不执行；总工时120分钟。'), **kwargs)


def test_flat_fourteen_fields_bounded_syntax_not_provider_business_claim():
    s = brief.schema()
    assert len(s['properties']) == 14 and set(s['required']) == set(s['properties'])
    assert s['additionalProperties'] is False
    assert all(s['properties'][key]['type'] == 'string' for key in brief.TEXT_LIMITS)
    assert s['properties']['day_one_minutes'] == {'type': ['integer', 'null']}
    c = brief.contract()
    assert c['version'] == 11 and len((c['prompt'] + c['policy']).encode()) <= 1600
    assert 'paper-review substitute' in c['policy'] and '600-900' in c['policy']
    s['properties']['hypothesis']['maxLength'] = 0
    assert brief.schema()['properties']['hypothesis']['maxLength'] == 240


def test_original_goal_hypothesis_target_refuter_and_all_fields_are_owned_not_reconstructed():
    data, sources = fixture()
    result = parse(data, sources)
    assert result['validation']['status'] == 'review_required'
    assert result['brief'] == data and result['brief'] is not data
    for key in brief.TEXT_LIMITS:
        assert data[key] in brief.render(result)
    assert not any(key in result for key in ('samples', 'compiled', 'levels', 'prediction'))
    assert 'paired-bundle-observation' not in brief.render(result)
    assert result['task']['goal'] == '本地编程与多模态资料方案，不缩成论文审阅。'
    assert all(result['validation'][key] is False for key in ('semantic_verified', 'measurement_verified',
        'observations_present', 'execution_authorized', 'executed'))


@pytest.mark.parametrize('key', ['question', 'goal', 'constraints'])
def test_changing_original_task_changes_binding_without_rewriting_nomination(key):
    a = parse(); b = parse(**{key: '新的原始要求，保留逐字内容。'})
    assert a['brief'] == b['brief'] and a['task_sha256'] != b['task_sha256']
    assert a['task'][key] != b['task'][key]
    receipt = brief.receipt(b, ['source-a'], requested_contract=brief.contract())
    assert receipt[key + '_sha256'] == hashlib.sha256(b['task'][key].encode()).hexdigest()


def test_task_whitespace_and_optional_missing_goal_preserved_exactly():
    result = parse(question='  原始问题\n', goal=None, constraints='  不执行\n')
    assert result['task'] == {'question': '  原始问题\n', 'goal': None, 'constraints': '  不执行\n'}


def test_unused_budget_is_not_mechanically_spent():
    result = parse()
    receipt = brief.receipt(result, ['source-a'], requested_contract=brief.contract())
    assert receipt['planned_minutes'] == 70 and receipt['remaining_minutes'] == 50
    assert '50' in brief.render(result) and result['brief']['day_one_minutes'] == 30


def test_unknown_minutes_not_zero_and_not_a_complete_valid_schedule():
    data, _ = fixture(); data['day_one_minutes'] = None
    result = parse(data)
    assert result['allocation']['planned_minutes'] is None
    assert result['allocation']['remaining_minutes'] is None
    assert result['allocation']['known_minutes'] == 40
    assert result['validation']['status'] == 'draft'
    assert result['validation']['violations'] == ['minutes_unconfirmed']
    assert '未按0计算' in brief.render(result)


def test_unknown_budget_never_fabricates_available_resources():
    result = parse(budget=None)
    assert result['validation']['violations'] == ['budget_unconfirmed']
    assert result['allocation']['remaining_minutes'] is None
    assert result['allocation']['budget_verified'] is False


@pytest.mark.parametrize('budget', [True, -1, 120.0, '120', 10081, 10 ** 1000])
def test_invalid_trusted_budget_is_bounded_and_not_coerced(budget):
    result = parse(budget=budget)
    assert 'trusted_budget_invalid' in result['validation']['violations']
    assert result['allocation']['trusted_budget_minutes'] is None


@pytest.mark.parametrize('minutes', [True, -1, 1.0, '30', 10081, 10 ** 1000])
def test_minutes_nonnegative_bounded_integer_or_none_without_coercion(minutes):
    data, _ = fixture(); data['day_one_minutes'] = minutes
    result = parse(data)
    assert 'minutes_invalid' in result['validation']['violations']
    assert result['brief'] is None


def test_budget_excess_fails_without_repairing_schedule():
    data, _ = fixture(); data['day_one_minutes'] = 100
    result = parse(data)
    assert result['brief'] == data and result['allocation']['planned_minutes'] == 140
    assert result['validation']['violations'] == ['budget_exceeded']
    assert result['allocation']['remaining_minutes'] == -20
    assert '不构成有效检验方案认证' in brief.render(result)


@pytest.mark.parametrize('field,cap', list(brief.TEXT_LIMITS.items()))
def test_every_text_limit_counts_unicode_characters_not_encoded_bytes(field, cap):
    data, _ = fixture(); data[field] = '汉' * (cap + 1)
    result = parse(data)
    assert 'field_limit' in result['validation']['violations'] and result['brief'] is None


def test_emoji_and_chinese_at_field_cap_are_not_rejected_as_multibyte():
    data, _ = fixture(); data['findings'] = '🌏' * 600
    assert parse(data)['validation']['status'] == 'review_required'


def test_total_text_cap_applies_even_when_individual_fields_fit():
    data, _ = fixture()
    for key, cap in brief.TEXT_LIMITS.items(): data[key] = '汉' * cap
    remove = sum(len(data[key]) for key in brief.TEXT_LIMITS) - 2401
    for key in brief.TEXT_LIMITS:
        n = min(remove, len(data[key]) - 1); data[key] = data[key][n:]; remove -= n
    assert sum(len(data[key]) for key in brief.TEXT_LIMITS) == 2401
    result = parse(data)
    assert result['validation']['violations'] == ['total_text_limit']


@pytest.mark.parametrize('citation', ['[S99]', '[K1]', '[P1]', '[J1]', '[S1 S99]', '[S01]', '[S0]', '[S99'])
def test_unavailable_legacy_or_malformed_citation_cannot_hide_in_other_fields(citation):
    data, _ = fixture(); data['hypothesis'] += citation
    result = parse(data)
    assert result['validation']['status'] == 'draft' and result['brief'] is None
    assert result['answer'] == data['findings']


def test_current_citation_groups_preserved_and_packet_hash_binds_content_span_and_id():
    data, sources = fixture()
    sources[0].update(excerpt_start=100, excerpt_end=100 + len(sources[0]['content']))
    sources.append({'id': 'source-b', 'label': 'S2', 'content': '另一个原始来源。'})
    data['findings'] = '两个当前来源的材料。[S1, S2]'
    result = parse(data, sources)
    assert result['validation']['status'] == 'review_required'
    assert result['source_packet'][0]['span'] == [100, 100 + len(sources[0]['content'])]
    assert '[S1, S2]' in brief.render(result)
    old = result['evidence_packet_sha256']; sources[0]['content'] += '新'
    sources[0]['excerpt_end'] += 1
    assert parse(data, sources)['evidence_packet_sha256'] != old


@pytest.mark.parametrize('mutate', [
    lambda s: s[0].update(revoked=True),
    lambda s: s[0].update(status='withdrawn'),
    lambda s: s[0].update(sha256='0' * 64),
    lambda s: s[0].update(span=[0, 1]),
    lambda s: s[0].update(span=[True, len(s[0]['content'])]),
    lambda s: s[0].update(excerpt_start=2_000_000, excerpt_end=2_000_000 + len(s[0]['content'])),
    lambda s: s.append(deepcopy(s[0])),
])
def test_invalid_source_binding_or_withdrawal_hides_entire_brief_even_without_S(mutate):
    data, sources = fixture(); data['findings'] = '没有挂S也不能绕开撤回。'
    mutate(sources); result = parse(data, sources)
    assert result['validation']['status'] == 'draft'
    assert '已收起' in brief.render(result)
    assert data['findings'] not in brief.render(result) and data['hypothesis'] not in brief.render(result)


def test_valid_declared_source_hash_is_owned_not_an_alias():
    data, sources = fixture()
    sources[0]['sha256'] = hashlib.sha256(sources[0]['content'].encode()).hexdigest()
    result = parse(data, sources); before = brief.render(result)
    sources[0]['content'] = '后改'; data['hypothesis'] = '后改'
    assert brief.render(result) == before


def test_wrong_fields_fail_but_bounded_current_findings_remain_identifiable():
    data, sources = fixture(); del data['hypothesis']
    result = parse(data, sources)
    assert result['brief'] is None and result['answer'] == data['findings']
    assert '未通过' in brief.render(result) and data['findings'] in brief.render(result)


@pytest.mark.parametrize('raw', ['[]', 'null', '{', '{"findings":"x","findings":"y"}', '{"findings":NaN}', 'x' * 32769])
def test_invalid_json_duplicate_keys_nonfinite_or_oversized_response_is_not_repaired(raw):
    _, sources = fixture()
    result = brief.parse(raw, sources, trusted_budget_minutes=120, question='原问题')
    assert result['validation']['status'] == 'draft' and result['brief'] is None


def test_complete_json_fence_only_is_normalized_not_partial_response_repaired():
    data, sources = fixture(); raw = json.dumps(data, ensure_ascii=False)
    result = brief.parse('```json\n' + raw + '\n```', sources, trusted_budget_minutes=120, question='原问题')
    assert result['brief'] == data and result['validation']['status'] == 'review_required'


def test_input_hash_locks_entire_actual_pair_and_contract_without_goal_rewriting():
    c = brief.contract(); messages = [{'role': 'system', 'content': '安全政策' + c['policy']},
        {'role': 'user', 'content': '完整原始目标\n[S1]原文' + c['prompt']}]
    before = brief.input_sha256(messages); messages[1]['content'] = messages[1]['content'].replace('原始', '新的')
    assert brief.input_sha256(messages) != before
    messages[1]['content'] += c['prompt']
    with pytest.raises(ValueError, match='input_changed'): brief.input_sha256(messages)


def test_parse_binding_is_checked_again_before_render_or_receipt():
    result = parse(); result['brief']['hypothesis'] = '不能偷偷替换原假设'
    with pytest.raises(ValueError, match='binding_changed'): brief.render(result)
    with pytest.raises(ValueError, match='binding_changed'): brief.receipt(result, ['source-a'], requested_contract=brief.contract())


def test_receipt_rejects_changed_source_owner_and_contract_and_unavailable_clears_all_old_text():
    result = parse()
    with pytest.raises(ValueError, match='source_ids_changed'): brief.receipt(result, ['different'], requested_contract=brief.contract())
    with pytest.raises(ValueError, match='invalid_decision'): brief.receipt(result, ['source-a'], requested_contract={'version': 10})
    fresh = brief.unavailable(brief.receipt(result, ['source-a'], requested_contract=brief.contract()))
    assert fresh['status'] == 'source_unavailable' and fresh['source_ids'] == []
    assert 'task' not in fresh and 'brief' not in fresh and fresh['planned_minutes'] is None
