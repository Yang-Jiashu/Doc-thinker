"""Finite trial-plan structure checks, not experiment or semantic validation.

No model, filesystem, executable expressions, inferred thresholds, or repairs.
The two arms share the fixed inputs; only their declared tested value may vary.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from itertools import product
import json
import math

VERSION = 1
MAX_RESPONSE_BYTES = 65536
MAX_GRID_CHECKS = 256
VALIDATION_SCOPE = 'internal_structure_not_semantic_or_execution_authorization'
TRIAL_RESPONSE_INSTRUCTION = (
    '本次仅输出一个JSON对象{answer,trial_spec}，不加代码围栏。answer保留正常回答及资料引用；'
    'trial_spec遵守给定schema，不在answer重复试点表。未知填null并保留草稿。'
    '两臂使用同一模型/配置、固定题集与历史，只有待测项不同。最多两个计数率指标，'
    '分母均为全部计划问题，失败和缺失计失败。规则针对candidate-baseline的率差[-1,1]，'
    '不重叠且覆盖全域；充足样本的灰区用collect_evidence并给具体下一步，不能称样本不足。'
    '通过、停止和真实样本不足均给操作。阈值未给时可提待确认建议，不能只写显著/可接受，'
    '不补造已测结果，不把规则当执行授权。费用/延迟不是本schema校验指标。'
)

_TEXT = {'type': 'string', 'minLength': 1, 'maxLength': 300}
_MAYBE_TEXT = {'type': ['string', 'null'], 'minLength': 1, 'maxLength': 300}
_NUMBER = {'type': ['number', 'null'], 'minimum': -1, 'maximum': 1}


def _object(properties, **extra):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False, **extra}


_SCHEMA = _object({
    'answer': {'type': 'string', 'minLength': 1, 'maxLength': 24000},
    'trial_spec': _object({
        'version': {'const': VERSION},
        'hypothesis': _TEXT,
        'test_variable': {'type': 'string', 'minLength': 1, 'maxLength': 80},
        'arms': {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': _object({
            'id': {'enum': ['baseline', 'candidate']}, 'model': _MAYBE_TEXT,
            'config': {'type': 'object', 'maxProperties': 8, 'propertyNames': {'maxLength': 60},
                       'additionalProperties': {'type': ['string', 'number', 'boolean', 'null']}},
            'tested_value': _MAYBE_TEXT,
        })},
        'inputs': _object({
            'question_set': _MAYBE_TEXT, 'history_set': _MAYBE_TEXT,
            'planned_questions': {'type': ['integer', 'null'], 'minimum': 1, 'maximum': 1000000},
            'minimum_paired_questions': {'type': ['integer', 'null'], 'minimum': 1, 'maximum': 1000000},
            'insufficient_data_step': _MAYBE_TEXT,
        }),
        'metrics': {'type': 'array', 'minItems': 1, 'maxItems': 2, 'items': _object({
            'id': {'enum': ['m1', 'm2']},
            'numerator': {'enum': ['correct_answers', 'grounded_answers', 'completed_answers', 'failed_answers']},
            'denominator': {'const': 'all_planned_questions'}, 'unit': {'const': 'fraction'},
            'direction': {'enum': ['higher', 'lower']},
            'failure_and_missing': {'const': 'count_as_failure'}, 'scoring': _MAYBE_TEXT,
        })},
        'rules': {'type': 'array', 'minItems': 2, 'maxItems': 8, 'items': _object({
            'outcome': {'enum': ['pass', 'stop', 'collect_evidence']},
            'ranges': {'type': 'array', 'minItems': 1, 'maxItems': 2, 'items': _object({
                'metric': {'enum': ['m1', 'm2']}, 'lower': _NUMBER, 'upper': _NUMBER,
                'lower_inclusive': {'type': 'boolean'}, 'upper_inclusive': {'type': 'boolean'},
            })},
            'next_step': _MAYBE_TEXT,
        })},
        'threshold_origin': {'enum': ['proposed', 'user_provided']},
    }),
})

# Fixed codes only: never include unknown keys, exceptions, model fragments, or
# source instructions in the receipt/error messages.
VIOLATIONS = frozenset({
    'response_limit', 'json_invalid', 'schema_invalid', 'unknown_value',
    'arm_identity', 'fixed_setup_mismatch', 'tested_value_not_distinct',
    'metric_identity', 'metric_direction', 'sample_gate_unreachable',
    'rule_metric_scope', 'interval_invalid', 'rule_overlap', 'rule_gap',
    'decision_outcome_missing', 'validation_limit', 'next_step_unspecified',
})
_UNKNOWN = frozenset({'unknown', 'unspecified', 'tbd', 'pending', 'null',
                       '未知', '待定', '待确认', '未确定', '未指定'})


def trial_response_schema():
    """Return a fresh bounded schema; callers cannot mutate the shared contract."""
    return deepcopy(_SCHEMA)


def _schema_valid(value, schema, depth=0):
    if depth > 12:
        return False
    if 'const' in schema and (type(value) is not type(schema['const']) or value != schema['const']):
        return False
    if 'enum' in schema and (not isinstance(value, str) or value not in schema['enum']):
        return False
    kind = schema.get('type')
    if kind is None:
        return True
    kinds = [kind] if isinstance(kind, str) else kind
    if value is None:
        return 'null' in kinds
    actual = ('boolean' if type(value) is bool else 'integer' if type(value) is int else
              'number' if type(value) is float else 'string' if type(value) is str else
              'object' if type(value) is dict else 'array' if type(value) is list else '')
    if actual not in kinds and not (actual == 'integer' and 'number' in kinds):
        return False
    if actual in ('integer', 'number'):
        return (-1000000000 <= value <= 1000000000 and math.isfinite(value)
                and value >= schema.get('minimum', -1000000000)
                and value <= schema.get('maximum', 1000000000))
    if actual == 'string':
        return schema.get('minLength', 0) <= len(value) <= schema.get('maxLength', 300)
    if actual == 'array':
        return (schema.get('minItems', 0) <= len(value) <= schema.get('maxItems', 8)
                and all(_schema_valid(item, schema['items'], depth + 1) for item in value))
    if actual == 'object':
        fields = schema.get('properties', {})
        extra = schema.get('additionalProperties', False)
        if len(value) > schema.get('maxProperties', len(fields)) or not set(schema.get('required', ())) <= value.keys():
            return False
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > schema.get('propertyNames', {}).get('maxLength', 80):
                return False
            if key in fields:
                if not _schema_valid(item, fields[key], depth + 1):
                    return False
            elif not isinstance(extra, dict) or not _schema_valid(item, extra, depth + 1):
                return False
        return True
    return True


def _unknown(value):
    if value is None or (isinstance(value, str) and (not value.strip() or value.strip().casefold() in _UNKNOWN)):
        return True
    if isinstance(value, dict):
        return any(_unknown(item) for item in value.values())
    if isinstance(value, list):
        return any(_unknown(item) for item in value)
    return False


def _same_json(left, right):
    # Python considers False == 0; fixed model parameters must preserve JSON
    # types, not merely compare equal after Python's numeric coercion.
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_json(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_json(a, b) for a, b in zip(left, right))
    return left == right


def _concrete_step(value):
    # Surface-level minimum only; cannot establish that the proposed step is
    # useful, affordable, safe, authorized, or executable in the real world.
    if value is None or _unknown(value):
        return False
    stripped = value.strip()
    return len(stripped) >= 8 and stripped.casefold() not in {
        'collect more evidence', 'collect evidence', 'continue collecting data',
        '收集更多证据', '继续收集数据', '待进一步验证',
    }


def _verdict(codes, checks=0):
    codes = sorted(set(codes))
    invalid = set(codes) - {'unknown_value', 'next_step_unspecified'}
    return {'status': 'invalid' if invalid else 'draft' if codes else 'valid',
            'violations': codes, 'validation_scope': VALIDATION_SCOPE,
            'schema_version': VERSION, 'grid_checks': checks,
            'execution_authorized': False, 'semantic_verified': False}


def validate_trial_spec(spec):
    """Validate finite internal consistency only; never infer missing thresholds."""
    if not _schema_valid(spec, _SCHEMA['properties']['trial_spec']):
        return _verdict(['schema_invalid'])
    codes = ['unknown_value'] if _unknown(spec) else []
    arms = {item['id']: item for item in spec['arms']}
    if set(arms) != {'baseline', 'candidate'}:
        codes.append('arm_identity')
    else:
        left, right = arms['baseline'], arms['candidate']
        if left['model'] != right['model'] or not _same_json(left['config'], right['config']):
            codes.append('fixed_setup_mismatch')
        if not _unknown(left['tested_value']) and left['tested_value'] == right['tested_value']:
            codes.append('tested_value_not_distinct')
    inputs = spec['inputs']
    if (inputs['planned_questions'] is not None and inputs['minimum_paired_questions'] is not None
            and inputs['minimum_paired_questions'] > inputs['planned_questions']):
        codes.append('sample_gate_unreachable')
    if not _concrete_step(inputs['insufficient_data_step']):
        codes.append('next_step_unspecified')
    metrics = {item['id']: item for item in spec['metrics']}
    if len(metrics) != len(spec['metrics']):
        codes.append('metric_identity')
    for item in metrics.values():
        if item['direction'] != ('lower' if item['numerator'] == 'failed_answers' else 'higher'):
            codes.append('metric_direction')
    outcomes = {rule['outcome'] for rule in spec['rules']}
    if not {'pass', 'stop'} <= outcomes:
        codes.append('decision_outcome_missing')
    rectangles, endpoints = [], {key: {Fraction(-1), Fraction(1)} for key in metrics}
    unresolved_range = False
    for rule in spec['rules']:
        if not _concrete_step(rule['next_step']):
            codes.append('next_step_unspecified')
        ranges = {item['metric']: item for item in rule['ranges']}
        if set(ranges) != set(metrics) or len(ranges) != len(rule['ranges']):
            codes.append('rule_metric_scope')
            continue
        rectangle = {}
        for key, item in ranges.items():
            if item['lower'] is None or item['upper'] is None:
                unresolved_range = True
                continue
            lo, hi = Fraction(str(item['lower'])), Fraction(str(item['upper']))
            if lo > hi or (lo == hi and not (item['lower_inclusive'] and item['upper_inclusive'])):
                codes.append('interval_invalid')
            endpoints[key].update((lo, hi))
            rectangle[key] = (lo, hi, item['lower_inclusive'], item['upper_inclusive'])
        rectangles.append(rectangle)
    if unresolved_range or any(code in codes for code in ('interval_invalid', 'rule_metric_scope')):
        return _verdict(codes)
    # Every endpoint and one exact rational interior representative of each
    # partition suffice for axis-aligned finite intervals, including boundaries.
    keys, axes = sorted(metrics), []
    for key in keys:
        points = sorted(endpoints[key])
        axes.append(sorted(set(points + [(a + b) / 2 for a, b in zip(points, points[1:])])))
    count = math.prod(map(len, axes))
    if count > MAX_GRID_CHECKS:
        return _verdict([*codes, 'validation_limit'])
    for point in product(*axes):
        matching = 0
        for rectangle in rectangles:
            if all((value > rectangle[key][0] or value == rectangle[key][0] and rectangle[key][2])
                   and (value < rectangle[key][1] or value == rectangle[key][1] and rectangle[key][3])
                   for key, value in zip(keys, point)):
                matching += 1
        if not matching:
            codes.append('rule_gap')
        elif matching > 1:
            codes.append('rule_overlap')
    return _verdict(codes, count)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_key')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('non_finite')


def parse_trial_response(raw):
    """Never retry or repair a paid response. Invalid output remains a draft."""
    if not isinstance(raw, str):
        return {'answer': '', 'trial_spec': None, 'raw_draft': '', **_verdict(['json_invalid'])}
    encoded = raw.encode('utf-8', errors='replace')
    if len(encoded) > MAX_RESPONSE_BYTES:
        retained = encoded[:MAX_RESPONSE_BYTES].decode('utf-8', errors='ignore')
        return {'answer': retained, 'trial_spec': None, 'raw_draft': retained,
                'raw_truncated': True, **_verdict(['response_limit'])}
    try:
        envelope = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, TypeError, RecursionError):
        return {'answer': raw, 'trial_spec': None, 'raw_draft': raw, **_verdict(['json_invalid'])}
    if not _schema_valid(envelope, _SCHEMA):
        return {'answer': raw, 'trial_spec': None, 'raw_draft': raw, **_verdict(['schema_invalid'])}
    verdict = validate_trial_spec(envelope['trial_spec'])
    return {'answer': envelope['answer'], 'trial_spec': envelope['trial_spec'],
            'raw_draft': raw if verdict['status'] != 'valid' else None, **verdict}
