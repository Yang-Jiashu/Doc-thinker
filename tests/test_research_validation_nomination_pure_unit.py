"""Host skeleton arithmetic and relative counterexamples, with zero I/O."""
import json

import pytest

from docthinker.server import research_validation_nomination as nomination
from docthinker.server.research_validation_design import evaluate_design
from tests.test_research_validation_design_unit import source_ref


def fixture(count=1):
    sources = [{'id': 'source-' + str(i), 'label': 'S' + str(i),
                'content': 'Fixed source and task ' + str(i)} for i in range(1, count + 1)]
    data = {'evidence_analysis': 'Source-grounded proposed comparison [S1].',
        'baseline_operation': 'Read the fixed ungrouped instruction.',
        'candidate_operation': 'Read the fixed grouped instruction.',
        'measure_name': 'explicit omissions', 'measure_kind': 'count', 'measure_unit': 'omissions',
        'measurement_method': 'Count each explicitly stated unmet criterion once using the same input.',
        'observation_source': 'Future manual records, none collected.', 'direction': 'lower',
        'samples_json': json.dumps([{'label': row['label'], 'task': 'Apply the same fixed rubric to this input.'}
                                   for row in sources])}
    return data, sources


@pytest.mark.parametrize('count', [1, 2, 3])
def test_host_budget_pairing_and_decision_rule_are_complete_not_model_nominations(count):
    data, sources = fixture(count)
    result = nomination.parse(json.dumps(data), sources, trusted_budget_minutes=120, question='两天内完成设计')
    assert result['validation']['status'] == 'review_required'
    budget = result['allocation']
    assert budget['planned'] + budget['reserve'] == 120
    assert budget['first'] + budget['second'] == budget['planned'] <= 120
    assert result['two_days'] and not result['validation']['executed']
    plan = result['compiled']['plan']
    assert plan['factor']['id'] == 'protocol_bundle'
    assert plan['levels'][0]['settings']['rubric'] == plan['levels'][1]['settings']['rubric']
    assert len(plan['samples']) == count
    assert nomination.render(result).count('待观测') == count * 2


@pytest.mark.parametrize('key', ['rules', 'settings', 'activity_minutes', 'executed'])
def test_model_cannot_replace_host_owned_fields(key):
    data, sources = fixture()
    data[key] = 'forged'
    result = nomination.parse(json.dumps(data), sources, trusted_budget_minutes=120)
    assert result['validation']['status'] == 'draft' and not result['validation']['executed']


def test_original_one_to_three_counterexample_refutes_instead_of_passing_absolute_threshold():
    data, sources = fixture()
    result = nomination.parse(json.dumps(data), sources, trusted_budget_minutes=120)
    observation = {'source_id': 'log-1', 'label': 'S9', 'content': 'Manually recorded observations.',
                   'kind': 'observation_record'}
    current = [{'source_id': row['id'], 'label': row['label'], 'content': row['content']} for row in sources] + [observation]
    records = [{'sample_id': 'case-1', 'level': level, 'target_id': 'paired-bundle-observation',
                'measurement_id': 'paired-measure', 'value': value, 'source': source_ref(observation)}
               for level, value in [('baseline', 1), ('high', 3)]]
    compared = evaluate_design(result['compiled']['plan'], records, current, trusted_budget_minutes=120)
    assert compared['verdict'] == 'refute'
    assert not compared['semantic_verified'] and not compared['execution_authorized']
