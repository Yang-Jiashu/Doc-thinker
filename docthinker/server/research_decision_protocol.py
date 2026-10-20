"""Original decision brief plus explicit model-nominated test protocol.

Version11 remains unchanged. The host never creates a baseline, task, target,
rule, control or observation; named fields and references are not semantic proof.
"""
from __future__ import annotations

from copy import deepcopy
import html
import json
import re

from . import research_decision_brief as base
from .research_quantity_grounding import build_quantity_index, check_quantity_grounding

VERSION = 12
MARKER = '\nHOST_DECISION_PROTOCOL_V1:\n'
PROTOCOL_LIMITS = {'baseline': 300, 'fixed_tasks': 360, 'decision_rule': 300}
TEXT_LIMITS = {**base.TEXT_LIMITS, **PROTOCOL_LIMITS}
_POLICY = (
    '\nFlat JSON;complete ORIGINAL goal,not a substitute.600-900 Chinese chars,total text<=2400. '
    'findings=facts/context;do_now/not_now=actions/deferrals;hypothesis=original untested claim;'
    'conditions_and_alternative=scope+strongest rival;validation_target=real predicted observable;'
    'validation_plan=concrete method;refuter=incompatible observation;missing_policy=missing/failures;'
    'day_one/day_two=work,minutes=int|null;privacy=per-action consent. '
    'Facts/quantities need same-unit current S evidence;invented limits must say our proposed choice,not paper fact. '
    'Comparative H needs actual baseline,fixed concrete tasks,counting unit and H-linked acceptance,not format/privacy proxy. '
    'baseline=original workflow;fixed_tasks=identical concrete inputs;decision_rule=H-linked support/refute/unknown. '
    'For learning rivals propose order/control,never claim eliminated. No forced AB or changed mean/all quantifier. '
    'Human actual observations may test human interaction;do not substitute imagined results for model effects. '
    'Unknown resources/methods/minutes stay unconfirmed,not0. Current[S#] only;no prior-analysis proof. Not executed.'
)
_UNKNOWN = re.compile(r'(?:未知|待确认|待补充|尚未确定|不清楚|unknown|unconfirmed|tbd|not\s+yet\s+known)[\s。.!！?？]*', re.I)
_SOURCE_FAILURES = {'sources_invalid', 'source_revoked', 'source_span_invalid', 'source_hash_mismatch'}


def schema():
    value = base.schema()
    value['properties'].update({key: {'type': 'string', 'minLength': 1, 'maxLength': cap}
                                for key, cap in PROTOCOL_LIMITS.items()})
    value['required'] = list(value['properties'])
    return value


def contract():
    prompt = MARKER + ','.join(TEXT_LIMITS) + ':strings;day_one_minutes,day_two_minutes:int|null;flat17fields.'
    value = prompt + _POLICY
    if len(value.encode()) > 1400:
        raise ValueError('decision_protocol_contract_limit')
    return {'version': VERSION, 'prompt': prompt, 'policy': _POLICY, 'contract_sha256': base._sha(value)}


def input_sha256(messages):
    current = contract()
    if (type(messages) is not list or len(messages) != 2
            or any(type(row) is not dict or set(row) != {'role', 'content'}
                   or type(row['content']) is not str for row in messages)
            or [row['role'] for row in messages] != ['system', 'user']
            or messages[0]['content'].count(current['policy']) != 1
            or messages[1]['content'].count(current['prompt']) != 1):
        raise ValueError('decision_protocol_input_changed')
    return base._sha(base._json(messages))


def _binding(parsed):
    return base._sha(base._json({key: value for key, value in parsed.items() if key != '_binding_sha256'}))


def parse(raw, sources, *, trusted_budget_minutes, question, goal=None, constraints=None):
    data, protocol, codes, unconfirmed = None, None, [], []
    try:
        if type(raw) is not str or len(raw.encode()) > 32768:
            raise ValueError('response_limit')
        text = raw.strip()
        fence = re.fullmatch(r'```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```', text, re.I)
        data = json.loads(fence[1] if fence else text, object_pairs_hook=base._unique,
                          parse_constant=base._constant)
        if type(data) is not dict:
            raise ValueError('schema_invalid')
    except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
        codes.append('invalid_decision_protocol')
    projection = ({key: value for key, value in data.items()
                   if key in set(base.TEXT_LIMITS) | set(base.MINUTE_FIELDS)} if type(data) is dict else {})
    original = base.parse(base._json(projection), sources, trusted_budget_minutes=trusted_budget_minutes,
                          question=question, goal=goal, constraints=constraints)
    if type(data) is dict:
        if set(data) != set(TEXT_LIMITS) | set(base.MINUTE_FIELDS):
            codes.append('schema_invalid')
        if any(not base._text(data.get(key), cap) for key, cap in PROTOCOL_LIMITS.items()):
            codes.append('protocol_field_limit')
        else:
            protocol = {key: data[key] for key in PROTOCOL_LIMITS}
            labels = {row['label'] for row in original['source_packet']}
            try:
                for text in protocol.values():
                    base._citations(text, labels)
            except ValueError:
                codes.append('protocol_citation_invalid')
            unconfirmed = [key for key, text in protocol.items() if _UNKNOWN.fullmatch(text.strip())]
            if unconfirmed:
                codes.append('protocol_unconfirmed')
        if all(type(data.get(key)) is str for key in TEXT_LIMITS):
            if sum(len(data[key]) for key in TEXT_LIMITS) > base.MAX_TEXT_CHARS:
                codes.append('total_text_limit')
    codes = sorted(set(codes) | set(original['validation']['violations']))
    full = ({**deepcopy(original['brief']), **deepcopy(protocol)}
            if original['brief'] is not None and protocol is not None
            and not any(code in codes for code in ('schema_invalid', 'protocol_field_limit',
                'protocol_citation_invalid', 'total_text_limit', 'invalid_decision_protocol')) else None)
    # One packet-local numeric index, reused across all fields. This is a cheap
    # attribution alarm, not semantic entailment or a replacement for review.
    quantity_checks, withheld = {}, []
    candidate_fields = full or original['brief'] or {'findings': original['answer']}
    if candidate_fields:
        index = build_quantity_index(sources)
        for key, text in candidate_fields.items():
            if key not in TEXT_LIMITS:
                continue
            check = check_quantity_grounding(text, index, field_role='fact')
            if check['checked_quantities'] or check['status'] == 'needs_review':
                quantity_checks[key] = check
            if check['status'] == 'needs_review':
                withheld.append(key)
        if withheld:
            codes = sorted(set(codes) | {'quantity_attribution_requires_review'})
    quantity_grounding = {'scope': 'bounded_exact_quantity_attribution_not_entailment',
        'checks': quantity_checks, 'withheld_fields': withheld,
        'semantic_verified': False, 'extra_model_calls': 0}
    result = {'answer': original['answer'], 'brief': full, 'protocol': protocol,
        'quantity_grounding': quantity_grounding,
        'task': deepcopy(original['task']), 'task_sha256': original['task_sha256'],
        'source_packet': deepcopy(original['source_packet']),
        'evidence_packet_sha256': original['evidence_packet_sha256'],
        'allocation': deepcopy(original['allocation']), '_base': original,
        'validation': {**deepcopy(original['validation']), 'version': VERSION,
            'status': 'draft' if codes else 'review_required', 'violations': codes,
            'structural_validation_passed': not codes, 'projection_complete': not codes,
            'protocol_unconfirmed_fields': unconfirmed,
            'scope': 'original_goal_and_nominated_protocol_structure_not_semantic_verification'}}
    result['_binding_sha256'] = _binding(result)
    return result


def _checked(parsed):
    try:
        if type(parsed) is not dict or parsed.get('_binding_sha256') != _binding(parsed):
            raise ValueError('decision_protocol_binding_changed')
        base._checked(parsed['_base'])
    except (KeyError, TypeError, UnicodeError, OverflowError, RecursionError):
        raise ValueError('decision_protocol_binding_changed') from None
    return parsed


def render(parsed):
    parsed = _checked(parsed)
    if any(code in _SOURCE_FAILURES for code in parsed['validation']['violations']):
        return base.render(parsed['_base'])
    data = parsed['brief']
    if data is None:
        body = ('本段含待核对的数值归因，已收起；请核对原文后再使用。'
                if 'findings' in parsed['quantity_grounding']['withheld_fields']
                else html.escape(parsed['answer'], quote=False) or '未取得可展示正文。')
        return body + '\n\n决策与检验草稿未通过字段、当前来源或预算检查；未执行、未自动补造对照或判据。'
    shown = {key: html.escape(data[key], quote=False) for key in TEXT_LIMITS}
    for key in parsed['quantity_grounding']['withheld_fields']:
        shown[key] = '本段含待核对的数值归因，已收起；请核对原文后再使用。'
    lines = ['当前依据与现在做', shown['findings'], shown['do_now'], '', '暂不做', shown['not_now'], '',
        '假设、条件与最强替代解释', shown['hypothesis'], shown['conditions_and_alternative'], '',
        '观察目标、检验与反证', '目标：' + shown['validation_target'], '原流程基线：' + shown['baseline'],
        '固定任务：' + shown['fixed_tasks'], '方案：' + shown['validation_plan'],
        '对应假设的判据：' + shown['decision_rule'], '反证：' + shown['refuter'],
        '缺失与失败：' + shown['missing_policy'], '', '两天安排与预算余量']
    for key, title in (('day_one', '第1天'), ('day_two', '第2天')):
        minutes = str(data[key + '_minutes']) + '分钟' if data[key + '_minutes'] is not None else '工时待确认（未按0计算）'
        lines.append(title + '（' + minutes + '）：' + shown[key])
    allocation = parsed['allocation']
    lines.append('拟定总工时：' + (str(allocation['planned_minutes']) + '分钟。'
        if allocation['planned_minutes'] is not None else '待确认；已知小计' + str(allocation['known_minutes']) + '分钟。'))
    if allocation['remaining_minutes'] is not None:
        lines.append('相对宿主预算的剩余分钟数：' + str(allocation['remaining_minutes']) + '（负值表示超限）。')
    else:
        lines.append('完整工时或宿主预算未确认，余量未知。')
    lines.extend(('', '隐私与逐次同意边界', shown['privacy'], ''))
    if parsed['validation']['status'] != 'review_required':
        lines.append('草稿存在未通过或待确认项，不构成有效检验方案认证。')
    lines.append('仅检查字段、来源定位和工时算术；事实支持、目标与测法仍待复核。方案未执行，不授权写入或导出。')
    return '\n'.join(lines)


def receipt(parsed, source_ids, *, requested_contract=None):
    if requested_contract != contract():
        raise ValueError('invalid_decision_protocol_contract')
    parsed = _checked(parsed)
    original = base.receipt(parsed['_base'], source_ids, requested_contract=base.contract())
    return {**original, **deepcopy(parsed['validation']), 'contract_sha256': contract()['contract_sha256'],
        'quantity_grounding': deepcopy(parsed['quantity_grounding']),
        'manifest_sha256': base._sha(base._json(parsed['brief'])) if parsed['brief'] is not None else None,
        'protocol_fields': deepcopy(parsed['protocol']),
        'protocol_sha256': base._sha(base._json(parsed['protocol'])) if parsed['protocol'] is not None else None}


def unavailable(previous=None, *, requested_contract=None):
    if requested_contract is not None and requested_contract != contract():
        raise ValueError('invalid_decision_protocol_contract')
    return {**base.unavailable(), 'version': VERSION, 'contract_sha256': contract()['contract_sha256'],
        'quantity_grounding': None,
        'protocol_fields': None, 'protocol_sha256': None,
        'scope': 'original_goal_and_nominated_protocol_structure_not_semantic_verification'}
