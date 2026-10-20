"""Explicit nominated protocol, not automatic scientific/goal certification."""
from copy import deepcopy
import hashlib
import json

import pytest

from docthinker.server import research_decision_brief as previous
from docthinker.server import research_decision_protocol as protocol


def fixture():
    data = {'findings': '原文只支持动作简单与文档简明，未给出新数值限制。[S1]',
        'do_now': '拟定本地流程，拟议限制是我们的设计选择，不是论文结论。',
        'not_now': '不执行、不恢复已撤回方案。',
        'hypothesis': '在相同任务下，候选流程可能降低人工轮数的平均值。',
        'conditions_and_alternative': '只限当前任务；学习效应是最强替代解释，尚未排除。',
        'validation_target': '人工实际完成固定任务所用的交互轮次，不是模型解决率。',
        'validation_plan': '每次用户发起请求计一轮；保留两流程实际记录，拟议交叉顺序并检查学习效应。',
        'refuter': '若同任务平均轮次不降或错引增多，则反驳该预测。',
        'missing_policy': '保留失败和缺失；记录不足不能判支持。',
        'day_one': '核对原文和两流程、固定任务与记录口径。', 'day_one_minutes': 35,
        'day_two': '形成可讨论的检验方案和待批准事项。', 'day_two_minutes': 45,
        'privacy': '执行代码、写入、截图导出均须逐次批准，沙盒不豁免。',
        'baseline': '当前原流程：先定位同一资料，再查看内容并给出带出处回答。',
        'fixed_tasks': '双方使用同一段原文：识别其全部操作条件并给出带出处短答。[S1]',
        'decision_rule': '以实际平均轮次之差和错误记录判断原假设；不降或错引增加则反驳；缺失则待定。'}
    sources = [{'id': 'source-a', 'label': 'S1', 'content': '动作应简单，文档应简明；这是材料不是本地实验结果。'}]
    return data, sources


def parse(data=None, sources=None, **kwargs):
    value, rows = fixture()
    return protocol.parse(json.dumps(value if data is None else data, ensure_ascii=False), rows if sources is None else sources,
        trusted_budget_minutes=kwargs.pop('budget', 120), question=kwargs.pop('question', '两天内交付原私人助手决策备忘录'),
        goal=kwargs.pop('goal', '本地编程、多模态与长期记忆方案，不改成固定论文审阅'), constraints='不执行；120分钟', **kwargs)


def test_schema_seventeen_flat_fields_and_caps_without_unsupported_numeric_keywords():
    value = protocol.schema()
    assert len(value['properties']) == len(value['required']) == 17
    assert value['additionalProperties'] is False
    for key, cap in protocol.PROTOCOL_LIMITS.items():
        assert value['properties'][key] == {'type': 'string', 'minLength': 1, 'maxLength': cap}
    assert value['properties']['day_one_minutes'] == {'type': ['integer', 'null']}
    value['properties']['baseline']['maxLength'] = 0
    assert protocol.schema()['properties']['baseline']['maxLength'] == 300


def test_contract_is_bounded_substantive_protocol_not_only_a_marker_change():
    value = protocol.contract()
    assert value['version'] == 12 and len((value['prompt'] + value['policy']).encode()) <= 1400
    assert value['contract_sha256'] != previous.contract()['contract_sha256']
    for requirement in ('same-unit', 'our proposed choice', 'not paper fact', 'actual baseline',
                        'fixed concrete tasks', 'counting unit', 'H-linked', 'learning', 'order/control', 'mean/all'):
        assert requirement in value['policy']


def test_all_original_fields_and_three_nominations_survive_without_new_control_or_quantifier():
    data, sources = fixture(); result = parse(data, sources)
    assert result['validation']['status'] == 'review_required' and result['brief'] == data
    assert result['protocol'] == {key: data[key] for key in protocol.PROTOCOL_LIMITS}
    rendered = protocol.render(result)
    assert all(data[key] in rendered for key in protocol.TEXT_LIMITS)
    assert result['brief']['hypothesis'] == data['hypothesis']
    assert '所有样本严格改善' not in rendered and 'baseline/high' not in rendered
    assert not any(key in result for key in ('samples', 'prediction', 'levels'))


def test_old_fourteen_projection_task_sources_and_budget_are_identical_to_version_eleven():
    data, sources = fixture(); result = parse(data, sources)
    projected = {key: value for key, value in data.items() if key not in protocol.PROTOCOL_LIMITS}
    old = previous.parse(json.dumps(projected, ensure_ascii=False), sources, trusted_budget_minutes=120,
        question=result['task']['question'], goal=result['task']['goal'], constraints=result['task']['constraints'])
    for key in ('task', 'task_sha256', 'source_packet', 'evidence_packet_sha256', 'allocation'):
        assert result[key] == old[key]
    assert old['brief'] == projected and result['allocation']['remaining_minutes'] == 40


@pytest.mark.parametrize('key,unknown', [('baseline', '待确认'), ('fixed_tasks', 'TBD'), ('decision_rule', '未知。')])
def test_explicitly_unknown_protocol_is_retained_but_not_declared_complete(key, unknown):
    data, _ = fixture(); data[key] = unknown; result = parse(data)
    assert result['brief'][key] == unknown and result['validation']['status'] == 'draft'
    assert result['validation']['protocol_unconfirmed_fields'] == [key]
    assert 'protocol_unconfirmed' in result['validation']['violations']
    assert '未通过或待确认' in protocol.render(result)


@pytest.mark.parametrize('key', list(protocol.PROTOCOL_LIMITS))
def test_missing_new_field_cannot_pass_the_legacy_projection_alone(key):
    data, _ = fixture(); del data[key]; result = parse(data)
    assert result['brief'] is None and result['answer'] == data['findings']
    assert result['validation']['status'] == 'draft'
    assert '未通过' in protocol.render(result)


@pytest.mark.parametrize('key,cap', list(protocol.PROTOCOL_LIMITS.items()))
def test_new_field_character_limit_does_not_silently_truncate(key, cap):
    data, _ = fixture(); data[key] = '汉' * (cap + 1); result = parse(data)
    assert 'protocol_field_limit' in result['validation']['violations'] and result['brief'] is None


def test_new_unicode_exact_caps_are_not_confused_with_bytes():
    data, _ = fixture()
    for key, cap in protocol.PROTOCOL_LIMITS.items(): data[key] = '🌏' * cap
    result = parse(data)
    assert result['validation']['status'] == 'review_required'
    assert result['brief']['fixed_tasks'] == '🌏' * 360


def test_total_text_ceiling_includes_new_three_fields_not_only_legacy_fourteen():
    data, _ = fixture()
    for key, cap in protocol.TEXT_LIMITS.items(): data[key] = '字' * cap
    remove = sum(len(data[key]) for key in protocol.TEXT_LIMITS) - 2401
    for key in previous.TEXT_LIMITS:
        count = min(remove, len(data[key]) - 1); data[key] = data[key][count:]; remove -= count
    assert sum(len(data[key]) for key in previous.TEXT_LIMITS) < 2400
    assert sum(len(data[key]) for key in protocol.TEXT_LIMITS) == 2401
    result = parse(data)
    assert 'total_text_limit' in result['validation']['violations'] and result['brief'] is None


@pytest.mark.parametrize('key', list(protocol.PROTOCOL_LIMITS))
def test_new_fields_cannot_cite_unknown_or_prior_sources(key):
    data, _ = fixture(); data[key] += '[S99]'; result = parse(data)
    assert 'protocol_citation_invalid' in result['validation']['violations']
    assert result['brief'] is None and result['answer'] == data['findings']


def test_revoked_source_hides_old_and_new_protocol_even_without_visible_citations():
    data, sources = fixture()
    for key in protocol.TEXT_LIMITS: data[key] = data[key].replace('[S1]', '')
    sources[0]['revoked'] = True; result = parse(data, sources); text = protocol.render(result)
    assert '已收起' in text
    assert data['hypothesis'] not in text and data['baseline'] not in text


@pytest.mark.parametrize('where', ['brief', 'protocol', '_base'])
def test_binding_protects_new_nominations_and_old_projection_before_render_receipt(where):
    result = parse()
    if where == '_base': result['_base']['brief']['hypothesis'] = '改掉原主张'
    else: result[where]['baseline'] = '偷偷改掉原基线'
    with pytest.raises(ValueError, match='binding_changed'): protocol.render(result)
    with pytest.raises(ValueError, match='binding_changed'): protocol.receipt(result, ['source-a'], requested_contract=protocol.contract())


def test_receipt_exposes_exact_nominations_hash_not_semantic_or_control_certificate():
    result = parse(); value = protocol.receipt(result, ['source-a'], requested_contract=protocol.contract())
    assert value['version'] == 12 and value['protocol_fields'] == result['protocol']
    assert value['protocol_sha256'] == hashlib.sha256(previous._json(result['protocol']).encode()).hexdigest()
    assert value['planned_minutes'] == 80 and value['remaining_minutes'] == 40
    assert all(value[key] is False for key in ('semantic_verified', 'measurement_verified', 'execution_authorized', 'executed'))
    assert value['task'] == result['task'] and value['evidence_packet_sha256'] == result['evidence_packet_sha256']
    value['protocol_fields']['baseline'] = '不能改回原结果'
    assert result['protocol']['baseline'] != value['protocol_fields']['baseline']


def test_unavailable_clears_new_nominations_and_old_goal_and_budget():
    result = parse(); old = protocol.receipt(result, ['source-a'], requested_contract=protocol.contract())
    value = protocol.unavailable(old)
    assert value['version'] == 12 and value['protocol_fields'] is None and value['protocol_sha256'] is None
    assert 'task' not in value and value['source_ids'] == [] and value['remaining_minutes'] is None


@pytest.mark.parametrize('raw', ['[]', '{', '{"baseline":"a","baseline":"b"}', 'null'])
def test_invalid_framing_is_not_repaired_into_a_protocol(raw):
    _, sources = fixture()
    result = protocol.parse(raw, sources, trusted_budget_minutes=120, question='原任务')
    assert result['validation']['status'] == 'draft' and result['brief'] is None


def test_noncomparative_hypothesis_is_not_forced_into_ab_or_an_invented_control():
    data, _ = fixture()
    data.update(hypothesis='在条件X满足时会出现现象Y。', baseline='不适用：这是存在性主张；先记录当前状态。',
        fixed_tasks='核对同一原始材料中的条件X与现象Y。[S1]', decision_rule='观察到X且没有Y则反驳；材料不足则不能判断。')
    result = parse(data)
    assert result['validation']['status'] == 'review_required' and result['brief'] == data


def test_unknown_word_inside_concrete_proposal_is_not_a_global_keyword_ban():
    data, _ = fixture(); data['baseline'] += '资源未知部分待确认，但这里保留当前流程。'
    assert parse(data)['validation']['status'] == 'review_required'


def test_input_hash_covers_whole_actual_pair_and_rejects_wrong_contract():
    current = protocol.contract(); pair = [{'role': 'system', 'content': current['policy']},
        {'role': 'user', 'content': '原任务与全部资料' + current['prompt']}]
    old = protocol.input_sha256(pair); pair[1]['content'] = '改变目标' + pair[1]['content']
    assert protocol.input_sha256(pair) != old
    pair[0]['content'] = previous.contract()['policy']
    with pytest.raises(ValueError, match='input_changed'): protocol.input_sha256(pair)
