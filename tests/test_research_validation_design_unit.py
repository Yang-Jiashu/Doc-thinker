"""Pure relative-design compiler; no provider, files, execution or threshold."""
import copy
import hashlib
import socket

import pytest

from docthinker.server import research_validation_design as design


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('Pure design compiler must not use network or models')
    for owner, name in ((socket.socket, 'connect'), (socket.socket, 'connect_ex'),
                        (socket, 'create_connection'), (socket, 'getaddrinfo')):
        monkeypatch.setattr(owner, name, forbidden)


def source_ref(source, start=0, end=None):
    end = len(source['content']) if end is None else end
    return {'source_id': source['source_id'], 'label': source['label'], 'span': [start, end],
            'sha256': hashlib.sha256(source['content'][start:end].encode()).hexdigest()}


def fixture(*, paired=False, kind='count'):
    sources = [{'source_id': 'sample-1', 'label': 'S1', 'content': 'A fixed actual input sample.'},
               {'source_id': 'log-1', 'label': 'S2', 'content': 'Actual manually recorded task observations.', 'kind': 'observation_record'}]
    levels = ('baseline', 'high') if paired else design.LEVELS
    spec = {
        'design_type': 'paired' if paired else 'three_level',
        'target': {'id': 'error', 'description': 'Errors on the same fixed input'},
        'measurement': {'id': 'error-count', 'target_id': 'error', 'quantity': 'manual error count',
                        'kind': kind, 'unit': 'errors', 'method': 'Same rubric and observer for each level',
                        'observation_source': 'Manual task observation log, not a simulated model score',
                        'ordered_values': ['low', 'medium', 'high'] if kind == 'ordinal' else None},
        'factor': {'id': 'assistance', 'description': 'Amount of assistance'},
        'levels': [{'id': level, 'operation': 'Apply ' + level + ' assistance',
                    'settings': {'assistance': level, 'input_order': 'identical', 'rubric': 'unchanged'}} for level in levels],
        'fixed_conditions': {'input_order': 'identical', 'rubric': 'unchanged'},
        'samples': [{'id': 'case-1', 'task': 'Identify every stated boundary in this exact input using the fixed rubric.',
                     'source': source_ref(sources[0])}],
        'prediction': {'pairs': [{'from': left, 'to': right, 'direction': 'lower'} for left, right in zip(levels, levels[1:])],
                       'description': 'More assistance yields fewer errors'},
        'falsifier': 'Any complete adjacent pair has equal or more errors, under these fixed conditions',
        'missing_policy': 'retain_inconclusive',
        'activity_minutes': {'setup': 1, 'per_sample_level': 2, 'review': 1},
    }
    return spec, sources


def observed(spec, sources, values):
    return [{'sample_id': 'case-1', 'level': level['id'], 'target_id': 'error',
             'measurement_id': 'error-count', 'value': value, 'source': source_ref(sources[1])}
            for level, value in zip(spec['levels'], values)]


def compile(spec, sources, budget=10):
    return design.compile_design(spec, sources, trusted_budget_minutes=budget)


def evaluate(spec, sources, values):
    return design.evaluate_design(spec, observed(spec, sources, values), sources, trusted_budget_minutes=10)


def test_draft_compiles_complete_three_levels_and_relative_rules_not_fabricated_intervals():
    spec, sources = fixture()
    result = compile(spec, sources)
    assert result['status'] == 'review_required' and result['plan'] == spec
    assert result['comparisons'] == [{'from': 'baseline', 'to': 'intermediate', 'direction': 'lower'},
                                     {'from': 'intermediate', 'to': 'high', 'direction': 'lower'}]
    assert result['activity_minutes'] == 8
    assert not result['measurement_verified'] and not result['execution_authorized']
    assert not result['observations_present']
    assert 'threshold' not in str(result) and 'domain' not in str(result)


def test_original_counterexample_baseline_one_middle_three_cannot_support_fewer():
    spec, sources = fixture()
    result = evaluate(spec, sources, [1, 3, 0])
    assert result['verdict'] == 'refute'
    assert result['samples'][0]['comparisons'][0]['verdict'] == 'refute'


@pytest.mark.parametrize('values,verdict', [([3, 2, 1], 'support'), ([3, 3, 1], 'refute'),
                                         ([3, None, 1], 'inconclusive'), ([None, None, None], 'inconclusive')])
def test_relative_strict_order_has_complete_support_falsifier_and_missing_partition(values, verdict):
    spec, sources = fixture()
    result = evaluate(spec, sources, values)
    assert result['verdict'] == verdict
    assert len(result['samples']) == 1 and len(result['samples'][0]['levels']) == 3


def test_paired_design_and_fixed_sample_each_level_without_synthetic_samples():
    spec, sources = fixture(paired=True)
    assert compile(spec, sources)['comparisons'] == [{'from': 'baseline', 'to': 'high', 'direction': 'lower'}]
    assert evaluate(spec, sources, [2, 1])['verdict'] == 'support'
    spec['prediction']['pairs'][0]['direction'] = 'higher'
    assert evaluate(spec, sources, [2, 3])['verdict'] == 'support'


def test_missing_entire_observation_cell_and_extra_sample_cannot_be_silently_excluded():
    spec, sources = fixture()
    records = observed(spec, sources, [3, 2, 1])
    missing = design.evaluate_design(spec, records[:-1], sources, trusted_budget_minutes=10)
    assert missing['verdict'] == 'inconclusive' and missing['samples'][0]['levels']['high'] is None
    records[0]['sample_id'] = 'invented-case'
    invalid = design.evaluate_design(spec, records, sources, trusted_budget_minutes=10)
    assert invalid['violations'] == ['unpaired_sample_or_level']


def test_complete_good_sample_cannot_hide_incomplete_second_fixed_sample():
    spec, sources = fixture()
    sources.append({'source_id': 'sample-2', 'label': 'S3', 'content': 'A second actual fixed input.'})
    spec['samples'].append({'id': 'case-2', 'task': spec['samples'][0]['task'], 'source': source_ref(sources[2])})
    records = observed(spec, sources, [3, 2, 1])
    second = copy.deepcopy(records[:2])
    for row in second:
        row['sample_id'] = 'case-2'
    result = design.evaluate_design(spec, records + second, sources, trusted_budget_minutes=14)
    assert result['verdict'] == 'inconclusive'
    assert [row['verdict'] for row in result['samples']] == ['support', 'inconclusive']
    assert result['samples'][1]['levels']['high'] is None
    assert result['activity_minutes'] == 14


def test_intermediate_optimum_does_not_require_monotonic_effect_and_each_contrast_is_checked():
    spec, sources = fixture()
    spec['prediction']['pairs'] = [{'from': 'baseline', 'to': 'intermediate', 'direction': 'lower'},
                                  {'from': 'high', 'to': 'intermediate', 'direction': 'lower'}]
    assert evaluate(spec, sources, [3, 1, 2])['verdict'] == 'support'
    assert evaluate(spec, sources, [1, 3, 2])['verdict'] == 'refute'
    spec['prediction']['pairs'].append({'from': 'baseline', 'to': 'high', 'direction': 'higher'})
    assert compile(spec, sources)['status'] == 'review_required'
    spec['prediction']['pairs'][1] = {'from': 'intermediate', 'to': 'high', 'direction': 'lower'}
    spec['prediction']['pairs'][2]['direction'] = 'higher'
    assert compile(spec, sources)['violations'] == ['prediction_order_conflict']


@pytest.mark.parametrize('mutation,code', [
    (lambda s: s['levels'][1]['settings'].update(rubric='different'), 'confounded_factors'),
    (lambda s: s['levels'][1]['settings'].update(second_factor='changed'), 'confounded_factors'),
    (lambda s: s['levels'].pop(1), 'levels_incomplete'),
    (lambda s: s['measurement'].update(target_id='unrelated-target'), 'measurement_target_mismatch'),
    (lambda s: s.update(missing_policy='drop_missing'), 'missing_policy_invalid'),
    (lambda s: s['measurement'].update(ordered_values=[1, 2, 3]), 'invented_scale'),
    (lambda s: s['samples'].append(copy.deepcopy(s['samples'][0])), 'duplicate_sample'),
])
def test_mismatched_target_confounding_scale_and_cherry_picking_are_rejected(mutation, code):
    spec, sources = fixture()
    mutation(spec)
    assert compile(spec, sources)['violations'] == [code]


@pytest.mark.parametrize('mode,code', [('revoked', 'source_unavailable'), ('label', 'source_label_mismatch'),
                                     ('span', 'source_span_invalid'), ('hash', 'source_hash_mismatch'),
                                     ('missing', 'source_unavailable')])
def test_actual_current_sources_rechecked_for_withdrawal_coordinates_and_hash(mode, code):
    spec, sources = fixture()
    if mode == 'revoked':
        sources[0]['revoked'] = True
    elif mode == 'missing':
        sources.pop(0)
    elif mode == 'label':
        spec['samples'][0]['source']['label'] = 'S99'
    elif mode == 'span':
        spec['samples'][0]['source']['span'] = [0, 9999]
    else:
        spec['samples'][0]['source']['sha256'] = '0' * 64
    assert compile(spec, sources)['violations'] == [code]


def test_observation_source_rechecked_and_target_not_borrowed_from_a_proxy():
    spec, sources = fixture()
    records = observed(spec, sources, [3, 2, 1])
    records[0]['target_id'] = 'convenient-proxy'
    result = design.evaluate_design(spec, records, sources, trusted_budget_minutes=10)
    assert result['violations'] == ['measurement_target_mismatch']
    records[0]['target_id'] = 'error'
    sources[1]['revoked'] = True
    result = design.evaluate_design(spec, records, sources, trusted_budget_minutes=10)
    assert result['violations'] == ['source_unavailable']


def test_paper_claim_cannot_masquerade_as_an_actual_task_observation():
    spec, sources = fixture()
    records = observed(spec, sources, [3, 2, 1])
    records[0]['source'] = source_ref(sources[0])
    result = design.evaluate_design(spec, records, sources, trusted_budget_minutes=10)
    assert result['violations'] == ['observation_source_is_not_a_record']


def test_budget_uses_all_fixed_samples_and_all_levels_not_just_execution_or_one_arm():
    spec, sources = fixture()
    assert compile(spec, sources, budget=8)['status'] == 'review_required'
    assert compile(spec, sources, budget=7)['violations'] == ['activity_budget_exceeded']
    spec['activity_minutes']['setup'] = -1
    assert compile(spec, sources)['violations'] == ['activity_budget_invalid']


@pytest.mark.parametrize('value', [1.5, -1, True, float('nan'), float('inf')])
def test_count_observations_are_counts_not_invented_real_intervals(value):
    spec, sources = fixture()
    result = evaluate(spec, sources, [3, value, 1])
    assert result['verdict'] == 'inconclusive' and result['violations']


def test_qualitative_without_known_scale_stays_inconclusive_not_auto_numbered():
    spec, sources = fixture(kind='qualitative')
    result = evaluate(spec, sources, ['often', 'less often', 'rare'])
    assert result['verdict'] == 'inconclusive'
    assert [row['value'] for row in result['samples'][0]['levels'].values()] == ['often', 'less often', 'rare']


def test_known_declared_ordinal_order_compares_without_creating_numerical_thresholds():
    spec, sources = fixture(kind='ordinal')
    result = evaluate(spec, sources, ['high', 'medium', 'low'])
    assert result['verdict'] == 'support' and result['measurement_verified'] is False


def test_later_source_change_rejects_previously_compiled_design_and_no_inputs_are_mutated():
    spec, sources = fixture()
    before = copy.deepcopy((spec, sources))
    result = compile(spec, sources)
    assert (spec, sources) == before
    sources[0]['content'] = 'Changed sample at identical coordinates.'
    result = design.evaluate_design(result['plan'], [], sources, trusted_budget_minutes=10)
    assert result['verdict'] == 'inconclusive' and result['violations'] == ['source_hash_mismatch']


def test_bounded_plain_data_rejects_unbounded_and_custom_values_without_execution():
    spec, sources = fixture()
    spec['target']['description'] = 'x' * 32769
    assert compile(spec, sources)['violations'] == ['design_limit']
    spec, sources = fixture()
    spec['target']['description'] = object()
    assert compile(spec, sources)['violations'] == ['schema_invalid']


@pytest.mark.parametrize('value', [[], {}, ['lower'], {'kind': 'count'}])
@pytest.mark.parametrize('path,code', [
    (('design_type',), 'design_type_invalid'),
    (('measurement', 'kind'), 'measurement_kind_invalid'),
    (('prediction', 'pairs', 0, 'direction'), 'prediction_invalid'),
    (('prediction', 'pairs', 0, 'from'), 'prediction_invalid'),
    (('missing_policy',), 'missing_policy_invalid'),
])
def test_container_enum_values_fail_closed_without_typeerror(value, path, code):
    spec, sources = fixture()
    cursor = spec
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    result = compile(spec, sources)
    assert result['status'] == 'draft' and result['violations'] == [code]
    assert result['plan'] is None


@pytest.mark.parametrize('value', [[], {}, None, True, 1])
def test_bad_host_source_kind_and_ref_identity_are_safe(value):
    spec, sources = fixture()
    sources[0]['kind'] = value
    assert compile(spec, sources)['violations'] == ['sources_invalid']
    sources[0].pop('kind')
    spec['samples'][0]['source']['source_id'] = value
    assert compile(spec, sources)['violations'] == ['source_unavailable']


@pytest.mark.parametrize('value', [None, True, [], {}, float('nan'), float('inf'), 1 << 4096])
def test_invalid_operator_budget_is_safe_and_never_authorizes(value):
    spec, sources = fixture()
    result = compile(spec, sources, budget=value)
    assert result['status'] == 'draft' and result['plan'] is None
    assert not result['execution_authorized']


@pytest.mark.parametrize('content', ['x' * (design.MAX_SOURCE_BYTES + 1),
                                    '文' * (design.MAX_SOURCE_BYTES // 3 + 1), '\ud800'])
def test_host_source_utf8_capacity_and_invalid_unicode_fail_closed(content):
    spec, sources = fixture()
    sources[0]['content'] = content
    assert compile(spec, sources)['violations'] == ['sources_invalid']


def test_unicode_span_is_python_codepoints_not_utf8_bytes_and_changed_binding_rechecked():
    spec, sources = fixture()
    sources[0]['content'] = 'α甲🙂乙end'
    spec['samples'][0]['source'] = source_ref(sources[0], 1, 4)
    result = compile(spec, sources)
    assert result['status'] == 'review_required'
    assert sources[0]['content'][1:4] == '甲🙂乙'
    shifted = copy.deepcopy(result['plan'])
    shifted['samples'][0]['source']['span'] = [2, 5]
    assert compile(shifted, sources)['violations'] == ['source_hash_mismatch']
    sources[0]['content'] = 'α甲🙂丙end'
    stale = design.evaluate_design(result['plan'], [], sources, trusted_budget_minutes=10)
    assert stale['violations'] == ['source_hash_mismatch'] and stale['verdict'] == 'inconclusive'


def test_cyclic_and_excessively_nested_json_fail_closed_before_copy_or_execution():
    spec, sources = fixture()
    spec['target']['description'] = spec
    assert compile(spec, sources)['violations'] == ['design_limit']
    spec, sources = fixture()
    nested = []
    for _ in range(20):
        nested = [nested]
    spec['measurement']['kind'] = nested
    assert compile(spec, sources)['violations'] == ['design_limit']
