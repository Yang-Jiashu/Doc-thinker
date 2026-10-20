"""One-call validation drafts with host-bound samples, never executed results.

New explicit profile; old delivery contracts and saved answers are unchanged.
The model proposes operations and observations. The host binds current source
labels, checks paired structure/budget and renders an empty recording sheet.
"""
from __future__ import annotations

import hashlib
import html
import json
import re

from .research_validation_design import compile_design

VERSION = 6
MARKER = '\nHOST_VALIDATION_DESIGN_V1:\n'
_SHAPE = (
    '{answer:s,validation_design:{design_type:paired|three_level,target:{id:s,description:s},'
    'measurement:{id:s,target_id:s,quantity:s,kind:count|number|ordinal|qualitative,unit:s,method:s,'
    'observation_source:s,ordered_values:null|[s]},factor:{id:s,description:s},'
    'levels:[{id:baseline|intermediate|high,operation:s,settings:{factor_id:s,fixed_key:s}}],'
    'fixed_conditions:{fixed_key:s},samples:[{id:s,task:s,source:S_label}],'
    'prediction:{pairs:[{from:level_id,to:level_id,direction:lower|higher}],description:s},'
    'falsifier:s,missing_policy:retain_inconclusive,activity_minutes:{setup:n,per_sample_level:n,review:n}}}'
)
_POLICY = (
    '\nJSON only;all keys;s=string;n=minutes. Cited answer,no plan repetition. '
    '1 factor;2 paired or3 nonmonotonic levels;1-3 fixed tasks from distinct current S labels. '
    'settings=factor.id plus identical fixed_conditions. Give baseline/operations/rubric now. '
    'target=actual observable;paper/manual proxy cannot test model performance. '
    'pairs=to relative to from,cover all levels. No invented scale/threshold/data;unknown ordered_values=null. '
    'Retain failures/missing. setup+samples*levels*per_sample_level+review<=budget. '
    'Honor resources/privacy/approvals;unexecuted,not causal proof.'
)


def contract():
    prompt = MARKER + _SHAPE
    if len((prompt + _POLICY).encode()) > 1600:
        raise ValueError('validation_contract_limit')
    return {'version': VERSION, 'prompt': prompt, 'policy': _POLICY,
            'contract_sha256': hashlib.sha256((prompt + _POLICY).encode()).hexdigest()}


def input_sha256(messages):
    """Bind the complete final packet, not a mutable formatter receipt."""
    current = contract()
    if (type(messages) is not list or len(messages) != 2
            or any(type(row) is not dict or set(row) != {'role', 'content'}
                   or type(row['content']) is not str for row in messages)
            or [row['role'] for row in messages] != ['system', 'user']
            or messages[0]['content'].count(current['policy']) != 1
            or messages[1]['content'].count(current['prompt']) != 1):
        raise ValueError('validation_input_changed')
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_key')
        result[key] = value
    return result


def _invalid_constant(_):
    raise ValueError('nonfinite_value')


def _sources(rows):
    if type(rows) is not list or len(rows) > 64:
        raise ValueError('sources_invalid')
    result, labels, identifiers = [], set(), set()
    for row in rows:
        if type(row) is not dict:
            raise ValueError('sources_invalid')
        identifier, label, content = row.get('id'), row.get('label'), row.get('content')
        if (type(identifier) is not str or not identifier or len(identifier) > 256
                or type(label) is not str or re.fullmatch(r'S[1-9][0-9]?', label) is None
                or type(content) is not str or not content or len(content) > 4096
                or label in labels or identifier in identifiers):
            raise ValueError('sources_invalid')
        content.encode()
        labels.add(label); identifiers.add(identifier)
        result.append({'source_id': identifier, 'label': label, 'content': content})
    return result


def parse(raw, sources, *, trusted_budget_minutes):
    answer, compiled, codes = '', None, []
    try:
        if type(raw) is not str or len(raw.encode()) > 65536:
            raise ValueError('response_limit')
        envelope = json.loads(raw, object_pairs_hook=_unique, parse_constant=_invalid_constant)
        if type(envelope) is not dict or set(envelope) != {'answer', 'validation_design'}:
            raise ValueError('schema_invalid')
        if type(envelope['answer']) is not str or not envelope['answer'].strip() or len(envelope['answer'].encode()) > 40000:
            raise ValueError('answer_unavailable')
        answer = envelope['answer'].strip()
        spec = envelope['validation_design']
        # Own one bounded JSON copy before replacing labels. Never mutate model
        # nominations or let model source IDs/hashes masquerade as host bindings.
        if len(json.dumps(spec, ensure_ascii=False).encode()) > 32768:
            raise ValueError('design_limit')
        bound = json.loads(json.dumps(spec, ensure_ascii=False, allow_nan=False))
        current = _sources(sources)
        by_label = {row['label']: row for row in current}
        if type(bound) is not dict or type(bound.get('samples')) is not list or not 1 <= len(bound['samples']) <= 3:
            raise ValueError('samples_invalid')
        for sample in bound['samples']:
            if type(sample) is not dict or type(sample.get('source')) is not str:
                raise ValueError('sample_label_invalid')
            row = by_label.get(sample['source'])
            if row is None:
                raise ValueError('source_unavailable')
            sample['source'] = {'source_id': row['source_id'], 'label': row['label'],
                'span': [0, len(row['content'])], 'sha256': hashlib.sha256(row['content'].encode()).hexdigest()}
        if type(trusted_budget_minutes) is not int or not 1 <= trusted_budget_minutes <= 10080:
            raise ValueError('trusted_budget_unavailable')
        compiled = compile_design(bound, current, trusted_budget_minutes=trusted_budget_minutes)
        codes = list(compiled['violations'])
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError, UnicodeError):
        codes = ['invalid_validation_design']
    status = 'draft' if codes else 'review_required'
    return {'answer': answer, 'compiled': compiled,
        'validation': {'version': VERSION, 'status': status, 'violations': codes,
            'structural_validation_passed': not codes, 'projection_complete': not codes,
            'scope': 'paired_design_structure_not_observation_truth', 'semantic_verified': False,
            'measurement_verified': False, 'observations_present': False,
            'execution_authorized': False, 'executed': False, 'extra_model_calls': 0}}


def receipt(parsed, source_ids, *, requested_contract=None):
    if requested_contract != contract():
        raise ValueError('invalid_validation_contract')
    if (type(source_ids) is not list or len(source_ids) > 64
            or any(type(key) is not str or not key or len(key) > 256 for key in source_ids)
            or len(set(source_ids)) != len(source_ids)):
        raise ValueError('invalid_validation_sources')
    compiled = parsed.get('compiled') or {}
    plan = compiled.get('plan')
    return {**parsed['validation'], 'contract_sha256': contract()['contract_sha256'],
        'source_ids': list(source_ids), 'planned_minutes': compiled.get('activity_minutes'),
        'sample_count': len(plan['samples']) if plan else 0,
        'level_count': len(plan['levels']) if plan else 0,
        'manifest_sha256': hashlib.sha256(json.dumps(plan, ensure_ascii=False, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest() if plan else None}


def unavailable(previous=None, *, requested_contract=None):
    return {'version': VERSION, 'status': 'source_unavailable', 'source_ids': [],
        'contract_sha256': contract()['contract_sha256'], 'manifest_sha256': None,
        'scope': 'paired_design_structure_not_observation_truth',
        'structural_validation_passed': False, 'projection_complete': False,
        'semantic_verified': False, 'measurement_verified': False,
        'observations_present': False, 'execution_authorized': False, 'executed': False,
        'extra_model_calls': 0}


def _inline(value):
    return re.sub(r'([\\`*_\[\](){}#!|])', r'\\\1', html.escape(str(value).replace('\n', ' ').replace('\r', ' '), quote=False))


def render(parsed):
    lines = [parsed['answer'] or '未取得可展示正文。', '', '---']
    compiled = parsed.get('compiled') or {}
    plan = compiled.get('plan')
    if not plan or parsed['validation']['status'] != 'review_required':
        return '\n'.join(lines + ['验证方案草稿未通过结构、来源或预算检查；未执行，未自动修补或再次调用模型。'])
    lines += ['可复核的配对验证草稿（尚未执行）', '',
        '待检验：' + _inline(plan['target']['description']),
        '唯一改变项：' + _inline(plan['factor']['description']),
        '固定条件：' + '；'.join(_inline(k) + '＝' + _inline(v) for k, v in plan['fixed_conditions'].items())]
    for level in plan['levels']:
        lines.append('- ' + level['id'] + '：' + _inline(level['operation']) + '；改变项取值：'
                     + _inline(level['settings'][plan['factor']['id']]))
    lines += ['', '固定样本与空白记录表：', '', '| 样本与任务 | ' + ' | '.join(row['id'] for row in plan['levels']) + ' |',
        '|---|' + '---|' * len(plan['levels'])]
    for sample in plan['samples']:
        lines.append('| ' + _inline(sample['id']) + ' · ' + _inline(sample['task']) + ' [' + sample['source']['label'] + '] | '
                     + ' | '.join('待观测' for _ in plan['levels']) + ' |')
    measurement = plan['measurement']
    lines += ['', '观测口径：' + _inline(measurement['quantity']) + '（' + _inline(measurement['unit']) + '）；'
              + _inline(measurement['method']), '实际记录来源：' + _inline(measurement['observation_source']),
              '拟议预测：' + _inline(plan['prediction']['description'])]
    for pair in plan['prediction']['pairs']:
        lines.append('- ' + pair['to'] + ' 应' + ('低于' if pair['direction'] == 'lower' else '高于') + ' ' + pair['from'] + '，逐个相同样本比较。')
    lines += ['反证：' + _inline(plan['falsifier']),
        '缺失或失败保留在表中，不能删除后判通过；无有效观测目前不能判断。',
        '工时：准备 ' + str(plan['activity_minutes']['setup']) + ' + ' + str(len(plan['samples'])) + ' 样本 × '
        + str(len(plan['levels'])) + ' 档 × ' + str(plan['activity_minutes']['per_sample_level']) + ' + 复核 '
        + str(plan['activity_minutes']['review']) + ' ＝ ' + str(compiled['activity_minutes']) + ' 分钟。',
        '仅来源坐标、配对结构和预算算术已检查；任务、测法、资源与科学有效性仍需人工确认。'
        '材料不是实验观测，纸面比较不证明模型表现；未执行，不授权调用模型、写入或导出。']
    return '\n'.join(lines)
