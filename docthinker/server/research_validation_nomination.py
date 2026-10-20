"""Model nominations, host-owned matched protocol comparison and budget.

This is a prospective design contract, not repair of an old generated plan.
Only the host defines pairing, comparison rules, missing-data handling and
arithmetic. Comparing a whole protocol bundle cannot identify component causes.
"""
from __future__ import annotations

import hashlib
import json
import re

from . import research_validation_delivery as base
from .research_validation_design import compile_design

VERSION = 8
MARKER = '\nHOST_PROTOCOL_NOMINATIONS_V1:\n'
_FIELDS = {'evidence_analysis': 1800, 'baseline_operation': 700, 'candidate_operation': 700,
    'measure_name': 160, 'measure_kind': 16, 'measure_unit': 80,
    'measurement_method': 700, 'observation_source': 300, 'direction': 8, 'samples_json': 2048}
_POLICY = (
    '\nJSON only;all fields strings. evidence_analysis=source findings,confirmed user context and rationale only;'
    'host writes hypothesis,plan,budget,rules:do not repeat these in analysis. '
    'Nominate two complete paper-review protocols,not execution/future implementation. '
    'Compare whole bundles,not causes of individual mechanisms;manual proxy cannot test model performance. '
    'Give exact operations and enumerate identical observable rubric now. No invented data/scale/threshold. '
    '1-3 fixed concrete tasks from distinct current S material labels,not resource notes. '
    'Honor actual resources,privacy and approvals. Proposed only,not executed.'
)


def schema():
    properties = {key: {'type': 'string', 'minLength': 1, 'maxLength': cap}
                  for key, cap in _FIELDS.items()}
    properties['measure_kind']['enum'] = ['count', 'qualitative']
    properties['direction']['enum'] = ['lower', 'higher']
    return {'type': 'object', 'properties': properties, 'required': list(_FIELDS),
            'additionalProperties': False}


def contract():
    prompt = MARKER + '{' + ','.join(key + ':string' for key in _FIELDS) + '}'
    prompt += ';measure_kind=count|qualitative;direction=lower|higher;samples_json encodes [{label:S1,task:string}].'
    if len((prompt + _POLICY).encode()) > 1600:
        raise ValueError('validation_contract_limit')
    return {'version': VERSION, 'prompt': prompt, 'policy': _POLICY,
        'contract_sha256': hashlib.sha256((prompt + _POLICY).encode()).hexdigest()}


def input_sha256(messages):
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


def parse(raw, sources, *, trusted_budget_minutes, question=None):
    answer, plan, compiled, allocation, failure = '', None, None, None, 'response_bound'
    try:
        if type(raw) is not str or len(raw.encode()) > 32768:
            raise ValueError('response_limit')
        failure = 'nomination_json'
        # Only complete standard framing is normalized, never plan contents.
        text = raw.strip()
        fence = re.fullmatch(r'```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```', text, re.I)
        if fence:
            text = fence[1]
        data = json.loads(text, object_pairs_hook=base._unique, parse_constant=base._invalid_constant)
        failure = 'nomination_schema'
        if type(data) is not dict:
            raise ValueError('schema_invalid')
        analysis = data.get('evidence_analysis')
        if type(analysis) is str and analysis.strip() and len(analysis) <= _FIELDS['evidence_analysis']:
            answer = analysis.strip()
        if (set(data) != set(_FIELDS) or any(type(data[k]) is not str or not data[k].strip()
                or len(data[k]) > cap for k, cap in _FIELDS.items())
                or data['measure_kind'] not in ('count', 'qualitative')
                or data['direction'] not in ('lower', 'higher')
                or data['baseline_operation'] == data['candidate_operation']):
            raise ValueError('schema_invalid')
        failure = 'sample_binding'
        current = base._sources(sources)
        by_label = {row['label']: row for row in current}
        samples = json.loads(data['samples_json'], object_pairs_hook=base._unique,
                             parse_constant=base._invalid_constant)
        if type(samples) is not list or not 1 <= len(samples) <= 3:
            raise ValueError('samples_invalid')
        bound = []
        for ordinal, sample in enumerate(samples, 1):
            if (type(sample) is not dict or set(sample) != {'label', 'task'}
                    or type(sample['label']) is not str or sample['label'] not in by_label
                    or type(sample['task']) is not str or not sample['task'].strip()
                    or len(sample['task']) > 512):
                raise ValueError('samples_invalid')
            source = by_label[sample['label']]
            bound.append({'id': 'case-' + str(ordinal), 'task': sample['task'], 'source': {
                'source_id': source['source_id'], 'label': source['label'],
                'span': [0, len(source['content'])], 'sha256': hashlib.sha256(source['content'].encode()).hexdigest()}})
        failure = 'trusted_budget'
        if type(trusted_budget_minutes) is not int or not 4 <= trusted_budget_minutes <= 10080:
            raise ValueError('trusted_budget_unavailable')
        setup = review = max(1, trusted_budget_minutes // 6)
        per_cell = (trusted_budget_minutes - setup - review) // (2 * len(bound))
        if per_cell < 1:
            raise ValueError('trusted_budget_too_small')
        planned = setup + review + 2 * len(bound) * per_cell
        allocation = {'setup': setup, 'review': review, 'per_sample_level': per_cell,
            'sample_count': len(bound), 'planned': planned, 'reserve': trusted_budget_minutes - planned,
            'total': trusted_budget_minutes, 'first': setup + len(bound) * per_cell,
            'second': review + len(bound) * per_cell}
        fixed = {'inputs': '同表固定任务及其完整材料，两组逐项相同',
                 'reviewer': '同一人工审查者，不替代模型表现测量',
                 'rubric': data['measurement_method'], 'window': '同一任务范围和观察口径，不删失败或缺失'}
        factor = 'protocol_bundle'
        direction = data['direction']
        relation = '低于' if direction == 'lower' else '高于'
        prediction = f"仅在这{len(bound)}个固定任务上，候选整包的{data['measure_name']}均{relation}基线"
        plan = {'design_type': 'paired', 'target': {'id': 'paired-bundle-observation', 'description': prediction},
            'measurement': {'id': 'paired-measure', 'target_id': 'paired-bundle-observation',
                'quantity': data['measure_name'], 'kind': data['measure_kind'], 'unit': data['measure_unit'],
                'method': data['measurement_method'], 'observation_source': data['observation_source'],
                'ordered_values': None},
            'factor': {'id': factor, 'description': '整套流程版本；这是整包比较，不识别单个组件或机制的因果作用'},
            'levels': [{'id': key, 'operation': operation, 'settings': {**fixed, factor: key}}
                for key, operation in [('baseline', data['baseline_operation']), ('high', data['candidate_operation'])]],
            'fixed_conditions': fixed, 'samples': bound,
            'prediction': {'pairs': [{'from': 'baseline', 'to': 'high', 'direction': direction}], 'description': prediction},
            'falsifier': '任一有效配对等值或方向相反，即反驳“所有固定任务均按预测改善”；其余缺失也不抹去该反例',
            'missing_policy': 'retain_inconclusive',
            'activity_minutes': {'setup': setup, 'per_sample_level': per_cell, 'review': review}}
        compiled = compile_design(plan, current, trusted_budget_minutes=trusted_budget_minutes)
        failure = None
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError):
        pass
    codes = list(compiled['violations']) if compiled is not None else ['invalid_nomination']
    return {'answer': answer, 'compiled': compiled, 'allocation': allocation,
        'two_days': type(question) is str and bool(re.search(r'两天|二天|2\s*天|two\s+days', question, re.I)),
        'validation': {'version': VERSION, 'status': 'draft' if codes else 'review_required',
            'violations': codes, 'framing_failure': failure,
            'structural_validation_passed': not codes, 'projection_complete': not codes,
            'host_owns_pairing_rules_and_budget': True, 'observations_present': False,
            'scope': 'paired_protocol_bundle_comparison_not_mechanism_causality',
            'semantic_verified': False, 'measurement_verified': False,
            'execution_authorized': False, 'executed': False, 'extra_model_calls': 0}}


def receipt(parsed, source_ids, *, requested_contract=None):
    if requested_contract != contract():
        raise ValueError('invalid_validation_contract')
    result = base.receipt(parsed, source_ids, requested_contract=base.contract())
    return {**result, 'version': VERSION, 'contract_sha256': contract()['contract_sha256'],
            'budget_reserve_minutes': (parsed.get('allocation') or {}).get('reserve')}


def unavailable(previous=None, *, requested_contract=None):
    return {**base.unavailable(), 'version': VERSION, 'contract_sha256': contract()['contract_sha256'],
        'scope': 'paired_protocol_bundle_comparison_not_mechanism_causality'}


def render(parsed):
    rendered = base.render(parsed)
    allocation = parsed.get('allocation')
    if parsed['validation']['status'] != 'review_required' or not allocation:
        return rendered
    first, second = ('第1天', '第2天') if parsed.get('two_days') else ('时段1', '时段2')
    lines = [rendered, '', f"{first}：准备及基线审查，共{allocation['first']}分钟；"
        f"{second}：候选审查及复核，共{allocation['second']}分钟。",
        f"以上合计{allocation['planned']}分钟；预算内余量{allocation['reserve']}分钟，"
        f"总计不超过{allocation['total']}分钟。工时仅拟定分配，实际资源可行性未核验。",
        '固定判据：全部固定样本均按所声明方向严格改善才支持本批预测；任一等值或反向的有效配对即反驳。'
        '没有反例但有缺失则无法判断。定性、无已知排序的记录不自动转成分数或判通过。',
        '整包结果不证明其中某个记忆、推理或界面机制导致提升；不外推到全部任务，不构成因果或模型性能结论。',
        '当前仅提供纸面方案；任何代码执行、文件写入、导出或上传仍须另行明确授权。']
    return '\n'.join(lines)
