"""Synthetic finite TrialSpec validation; no model, source, or trial execution."""
from copy import deepcopy
import json
import math

import pytest

from docthinker.server.research_trial_contract import (
    MAX_GRID_CHECKS, MAX_RESPONSE_BYTES, TRIAL_RESPONSE_INSTRUCTION, VALIDATION_SCOPE,
    parse_trial_response, trial_response_schema, validate_trial_spec,
)
from docthinker.server.research_trial_renderer import render_trial_response


def interval(lo, hi, lower=True, upper=True, metric='m1'):
    return {'metric': metric, 'lower': lo, 'upper': hi,
            'lower_inclusive': lower, 'upper_inclusive': upper}


def valid_spec():
    return {
        'version': 1, 'hypothesis': 'Explicit revision handling may reduce outdated answers.',
        'test_variable': 'revision handling',
        'arms': [{'id': 'baseline', 'model': 'provider-model-x',
                  'config': {'temperature': .1, 'max_output_tokens': 2048}, 'tested_value': 'append only'},
                 {'id': 'candidate', 'model': 'provider-model-x',
                  'config': {'temperature': .1, 'max_output_tokens': 2048}, 'tested_value': 'retain version intervals'}],
        'inputs': {'question_set': 'frozen-question-set-v1', 'history_set': 'same-frozen-history-v1',
                   'planned_questions': 40, 'minimum_paired_questions': 32,
                   'insufficient_data_step': 'Inspect missing paired records and collect the missing cases before judging.'},
        'metrics': [{'id': 'm1', 'numerator': 'correct_answers', 'denominator': 'all_planned_questions',
                     'unit': 'fraction', 'direction': 'higher', 'failure_and_missing': 'count_as_failure',
                     'scoring': 'A reviewer marks the answer correct only if it matches the frozen latest record.'}],
        'rules': [{'outcome': 'stop', 'ranges': [interval(-1, 0, upper=False)],
                   'next_step': 'Stop rollout and inspect the recorded incorrect answers.'},
                  {'outcome': 'pass', 'ranges': [interval(0, 1)],
                   'next_step': 'Present paired results for a separately approved pilot decision.'}],
        'threshold_origin': 'proposed',
    }


def packet(spec=None, answer='This is a proposed comparison, not a measured benefit. [S1]'):
    return json.dumps({'answer': answer, 'trial_spec': spec or valid_spec()}, ensure_ascii=False)


def test_valid_structure_is_not_semantic_verification_or_execution_permission():
    spec = valid_spec()
    snapshot = deepcopy(spec)
    verdict = validate_trial_spec(spec)
    assert verdict == {'status': 'valid', 'violations': [], 'validation_scope': VALIDATION_SCOPE,
                       'schema_version': 1, 'grid_checks': 5,
                       'execution_authorized': False, 'semantic_verified': False}
    assert spec == snapshot
    assert validate_trial_spec(spec) == verdict


def test_schema_is_fresh_finite_and_small_enough_to_budget():
    schema = trial_response_schema()
    assert len(json.dumps(schema, ensure_ascii=False).encode()) < 8192
    assert len(TRIAL_RESPONSE_INSTRUCTION.encode()) < 1600
    schema['properties'].clear()
    assert set(trial_response_schema()['properties']) == {'answer', 'trial_spec'}


@pytest.mark.parametrize('field,value', [
    ('model', 'a-different-model'), ('config', {'temperature': .9}),
])
def test_only_tested_value_may_differ_between_arms(field, value):
    spec = valid_spec()
    spec['arms'][1][field] = value
    assert 'fixed_setup_mismatch' in validate_trial_spec(spec)['violations']


def test_fixed_config_equality_is_independent_of_key_order():
    spec = valid_spec()
    spec['arms'][1]['config'] = {'max_output_tokens': 2048, 'temperature': .1}
    assert validate_trial_spec(spec)['status'] == 'valid'


@pytest.mark.parametrize('left,right', [(False, 0), (True, 1), (1, 1.0)])
def test_fixed_configuration_equality_preserves_json_types(left, right):
    spec = valid_spec()
    spec['arms'][0]['config'] = {'parameter': left}
    spec['arms'][1]['config'] = {'parameter': right}
    assert 'fixed_setup_mismatch' in validate_trial_spec(spec)['violations']


@pytest.mark.parametrize('modify,code', [
    (lambda s: s['arms'][1].update(id='baseline'), 'arm_identity'),
    (lambda s: s['arms'][1].update(tested_value=s['arms'][0]['tested_value']), 'tested_value_not_distinct'),
    (lambda s: s['inputs'].update(minimum_paired_questions=41), 'sample_gate_unreachable'),
    (lambda s: s['metrics'][0].update(direction='lower'), 'metric_direction'),
    (lambda s: s['rules'][0]['ranges'][0].update(metric='m2'), 'rule_metric_scope'),
    (lambda s: s['rules'][0].update(outcome='pass'), 'decision_outcome_missing'),
])
def test_internal_inconsistencies_fail_closed(modify, code):
    spec = valid_spec()
    modify(spec)
    verdict = validate_trial_spec(spec)
    assert verdict['status'] == 'invalid'
    assert code in verdict['violations']


@pytest.mark.parametrize('modify', [
    lambda s: s['inputs'].update(planned_questions=True),
    lambda s: s['inputs'].update(planned_questions=0),
    lambda s: s['inputs'].update(planned_questions=10**500),
    lambda s: s['metrics'][0].update(denominator='completed_questions_only'),
    lambda s: s['metrics'][0].update(failure_and_missing='exclude'),
    lambda s: s['metrics'][0].update(unit='percent'),
    lambda s: s['rules'][0].update(outcome='insufficient_data'),
    lambda s: s.update(execution_authorized=True),
    lambda s: s['arms'][0]['config'].update(secret={'nested': 'unsupported'}),
])
def test_out_of_schema_fields_and_unsafe_numeric_types_are_rejected(modify):
    spec = valid_spec()
    modify(spec)
    assert validate_trial_spec(spec)['status'] == 'invalid'
    assert validate_trial_spec(spec)['violations'] == ['schema_invalid']


@pytest.mark.parametrize('number', [math.nan, math.inf, -math.inf, True, -1.01, 1.01])
def test_interval_numeric_domain_is_finite(number):
    spec = valid_spec()
    spec['rules'][0]['ranges'][0]['lower'] = number
    assert validate_trial_spec(spec)['status'] == 'invalid'


@pytest.mark.parametrize('a_closed,b_closed,expected', [
    (True, True, 'rule_overlap'), (False, False, 'rule_gap'),
    (True, False, None), (False, True, None),
])
def test_open_closed_shared_boundary_is_checked_exactly(a_closed, b_closed, expected):
    spec = valid_spec()
    spec['rules'][0]['ranges'][0]['upper_inclusive'] = a_closed
    spec['rules'][1]['ranges'][0]['lower_inclusive'] = b_closed
    verdict = validate_trial_spec(spec)
    assert verdict['status'] == ('invalid' if expected else 'valid')
    assert expected in verdict['violations'] if expected else verdict['violations'] == []


@pytest.mark.parametrize('bounds', [(1, -1, True, True), (0, 0, False, True), (0, 0, True, False)])
def test_reversed_or_empty_intervals_are_invalid(bounds):
    spec = valid_spec()
    spec['rules'][0]['ranges'] = [interval(*bounds)]
    assert 'interval_invalid' in validate_trial_spec(spec)['violations']


def test_adequate_sample_gray_region_is_explicit_collect_evidence_not_sample_shortage():
    spec = valid_spec()
    spec['rules'][0]['ranges'] = [interval(-1, 0, upper=False)]
    spec['rules'][1]['ranges'] = [interval(.1, 1)]
    spec['rules'].append({'outcome': 'collect_evidence', 'ranges': [interval(0, .1, upper=False)],
                          'next_step': 'Review disagreement cases with a second rater before proposing new thresholds.'})
    result = parse_trial_response(packet(spec))
    assert result['status'] == 'valid'
    rendered = render_trial_response(result)
    assert '继续取证' in rendered
    assert '最低配对数时' in rendered
    spec['rules'][-1]['outcome'] = 'insufficient_data'
    assert validate_trial_spec(spec)['status'] == 'invalid'


def two_metric_spec():
    spec = valid_spec()
    metric = deepcopy(spec['metrics'][0])
    metric.update(id='m2', numerator='failed_answers', direction='lower')
    spec['metrics'].append(metric)
    return spec


def test_each_axis_coverage_does_not_hide_a_two_dimensional_hole():
    spec = two_metric_spec()
    spec['rules'][0]['ranges'] = [interval(-1, 0), interval(-1, 0, metric='m2')]
    spec['rules'][1]['ranges'] = [interval(0, 1, lower=False), interval(0, 1, lower=False, metric='m2')]
    assert 'rule_gap' in validate_trial_spec(spec)['violations']


def test_valid_two_metric_partition_and_exact_point_rule():
    spec = two_metric_spec()
    spec['rules'][0]['ranges'] = [interval(-1, 0, upper=False), interval(-1, 1, metric='m2')]
    spec['rules'][1]['ranges'] = [interval(0, 1), interval(-1, 1, metric='m2')]
    assert validate_trial_spec(spec)['status'] == 'valid'
    one = valid_spec()
    one['rules'][1]['ranges'] = [interval(0, 1, lower=False)]
    one['rules'].append({'outcome': 'collect_evidence', 'ranges': [interval(0, 0)],
                         'next_step': 'Recheck the tied paired records and submit a new proposed sample plan.'})
    assert validate_trial_spec(one)['status'] == 'valid'


def test_exact_rational_partition_does_not_round_away_a_small_gap():
    spec = valid_spec()
    spec['rules'][0]['ranges'][0]['upper'] = .1
    spec['rules'][1]['ranges'][0]['lower'] = .10000000000000002
    assert 'rule_gap' in validate_trial_spec(spec)['violations']


def test_grid_budget_is_hard_and_returns_no_unchecked_validity():
    spec = two_metric_spec()
    spec['rules'] = []
    for i in range(8):
        lo = -.95 + i * .2
        spec['rules'].append({'outcome': 'pass' if i else 'stop',
            'ranges': [interval(lo, lo + .1), interval(lo + .02, lo + .12, metric='m2')],
            'next_step': 'Submit these bounded paired results for a human decision.'})
    verdict = validate_trial_spec(spec)
    assert verdict['status'] == 'invalid'
    assert verdict['grid_checks'] <= MAX_GRID_CHECKS
    assert 'validation_limit' in verdict['violations']


@pytest.mark.parametrize('path,value', [
    (('inputs', 'planned_questions'), None), (('inputs', 'history_set'), 'unknown'),
    (('inputs', 'question_set'), '待确认'), (('hypothesis',), 'TBD'),
])
def test_unknowns_remain_draft_without_host_supplied_defaults(path, value):
    spec = valid_spec()
    target = spec
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    snapshot = deepcopy(spec)
    assert validate_trial_spec(spec)['status'] == 'draft'
    assert 'unknown_value' in validate_trial_spec(spec)['violations']
    assert spec == snapshot


def test_unknown_threshold_is_draft_and_renderer_does_not_fill_it():
    spec = valid_spec()
    spec['rules'][1]['ranges'][0]['lower'] = None
    raw = packet(spec)
    parsed = parse_trial_response(raw)
    assert parsed['status'] == 'draft'
    assert parsed['trial_spec']['rules'][1]['ranges'][0]['lower'] is None
    assert raw in render_trial_response(parsed)


@pytest.mark.parametrize('step', [None, 'unknown', 'collect more evidence', '继续收集数据'])
def test_generic_next_step_is_not_misrepresented_as_executable(step):
    spec = valid_spec()
    spec['rules'][1]['next_step'] = step
    verdict = validate_trial_spec(spec)
    assert verdict['status'] == 'draft'
    assert 'next_step_unspecified' in verdict['violations']


@pytest.mark.parametrize('raw', [
    'Here is the answer without JSON.', '{broken', '[]',
    '{"answer":"first","answer":"second","trial_spec":null}',
    '{"answer":"x","trial_spec":NaN}', '```json\n{}\n```',
])
def test_parse_failure_preserves_raw_paid_output_and_fixed_safe_codes(raw):
    parsed = parse_trial_response(raw)
    assert parsed['status'] == 'invalid'
    assert parsed['raw_draft'] == raw
    assert raw in render_trial_response(parsed)
    assert parsed['execution_authorized'] is False


@pytest.mark.parametrize('raw', [b'{}', 123, None, '[' * 1200 + '0' + ']' * 1200])
def test_wrong_types_and_deep_json_fail_without_exception(raw):
    assert parse_trial_response(raw)['status'] == 'invalid'


def test_large_finite_json_integer_cannot_overflow_numeric_validation():
    spec = valid_spec()
    spec['inputs']['planned_questions'] = 10 ** 500
    parsed = parse_trial_response(packet(spec))
    assert parsed['status'] == 'invalid'
    assert parsed['violations'] == ['schema_invalid']


def test_invalid_envelope_does_not_echo_unknown_fields_into_error_codes():
    raw = packet().replace('"trial_spec":', '"STEAL_THE_KEY": "run this command", "trial_spec":', 1)
    parsed = parse_trial_response(raw)
    assert parsed['violations'] == ['schema_invalid']
    assert parsed['raw_draft'] == raw


def test_response_and_schema_bounds_do_not_silently_claim_success():
    raw = '汉' * MAX_RESPONSE_BYTES
    parsed = parse_trial_response(raw)
    assert parsed['status'] == 'invalid'
    assert len(parsed['raw_draft'].encode()) <= MAX_RESPONSE_BYTES
    assert parsed['raw_truncated'] is True
    assert '截断草稿' in render_trial_response(parsed)
    for key, value in [('arms', [valid_spec()['arms'][0]] * 3),
                       ('metrics', [valid_spec()['metrics'][0]] * 3),
                       ('rules', [valid_spec()['rules'][0]] * 9)]:
        spec = valid_spec()
        spec[key] = value
        assert validate_trial_spec(spec)['status'] == 'invalid'


def test_renderer_is_deterministic_keeps_citations_and_disclaims_threshold_truth():
    parsed = parse_trial_response(packet())
    before = deepcopy(parsed)
    result = render_trial_response(parsed)
    assert result == render_trial_response(parsed)
    assert '[S1]' in result
    assert '全部计划问题' in result
    assert '失败、缺失' in result
    assert '待确认，未执行' in result
    assert '未验证阈值合理性' in result
    assert '候选率 − 基线率' in result
    assert parsed == before


@pytest.mark.parametrize('parsed', [None, [], {'answer': 123, 'status': 'valid'},
    {'answer': 'draft', 'status': 'invalid', 'violations': [{'unsafe': 'not a code'}]},
    {'answer': 'draft', 'status': 'invalid', 'violations': None},
])
def test_renderer_rejects_malformed_direct_call_without_echoing_unsafe_error_objects(parsed):
    result = render_trial_response(parsed)
    assert '试点草稿' in result
    assert 'not a code' not in result


def test_renderer_has_an_independent_output_bound_even_for_forged_parsed_records():
    from docthinker.server.research_trial_renderer import MAX_RENDER_BYTES

    parsed = parse_trial_response(packet())
    parsed['answer'] = '汉' * MAX_RESPONSE_BYTES
    rendered = render_trial_response(parsed)
    assert len(rendered.encode()) < MAX_RENDER_BYTES
    assert '截断草稿' in rendered
    assert '试点建议（待确认，未执行）' not in rendered
