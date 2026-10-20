"""Representations must not change the host comparison, input or budget."""
import json

import pytest

from docthinker.server import research_validation_nomination as base
from docthinker.server import research_validation_nomination_wire as wire
from tests.test_research_validation_nomination_pure_unit import fixture


@pytest.mark.parametrize('count', [1, 2, 3])
def test_wire_and_direct_nomination_produce_identical_plan_and_budget(count):
    direct, sources = fixture(count)
    data = dict(direct)
    analysis = data.pop('evidence_analysis')
    data['samples'] = json.loads(data.pop('samples_json'))
    encoded = json.dumps({'evidence_analysis': analysis, 'nominations_json': json.dumps(data)})
    left = base.parse(json.dumps(direct), sources, trusted_budget_minutes=120)
    right = wire.parse(encoded, sources, trusted_budget_minutes=120)
    assert right['validation']['status'] == 'review_required'
    assert right['compiled'] == left['compiled'] and right['allocation'] == left['allocation']
    assert right['validation']['version'] == 9
    assert not right['validation']['executed']


@pytest.mark.parametrize('bad', ['not json', '{}', '[]', '{"samples":1,"samples":2}', 'x' * 12001])
def test_bad_nomination_never_discards_valid_analysis_or_claims_valid_design(bad):
    _, sources = fixture()
    encoded = json.dumps({'evidence_analysis': 'A source-bound discussion [S1].', 'nominations_json': bad})
    parsed = wire.parse(encoded, sources, trusted_budget_minutes=120)
    assert 'source-bound discussion' in parsed['answer']
    assert parsed['validation']['status'] == 'draft'
    assert not parsed['validation']['executed'] and parsed['validation']['extra_model_calls'] == 0
