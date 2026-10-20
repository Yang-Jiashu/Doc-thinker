"""Portable scalar framing over the same paired compiler; no model calls."""
import json

import pytest

from docthinker.server import research_validation_delivery_v2 as wire
from tests.test_research_validation_design_unit import fixture


def packet():
    spec, sources = fixture(paired=True)
    spec['samples'][0]['source'] = 'S1'
    rows = [{'id': row['source_id'], 'label': row['label'], 'content': row['content']} for row in sources]
    return {'answer': 'A proposed source-grounded comparison [S1].',
            'validation_design': json.dumps(spec)}, rows


@pytest.mark.parametrize('fence', ['', 'json', None])
def test_scalar_json_and_one_complete_fence_preserve_exact_valid_plan(fence):
    data, sources = packet()
    raw = json.dumps(data)
    if fence is not None:
        raw = '```' + fence + '\n' + raw + '\n```'
    parsed = wire.parse(raw, sources, trusted_budget_minutes=120)
    assert parsed['validation']['status'] == 'review_required'
    assert parsed['validation']['version'] == 7
    assert parsed['validation']['response_fence_unwrapped'] is (fence is not None)
    assert not parsed['validation']['executed']
    assert wire.render(parsed).count('待观测') == 2


@pytest.mark.parametrize('bad', ['not json', '{}', '[]', 'NaN', '{"x":1,"x":2}', 'x' * 32769])
def test_invalid_inner_design_preserves_valid_prose_but_never_claims_plan_success(bad):
    data, sources = packet()
    data['validation_design'] = bad
    parsed = wire.parse(json.dumps(data), sources, trusted_budget_minutes=120)
    assert parsed['answer'] == data['answer']
    assert parsed['validation']['status'] == 'draft'
    assert parsed['validation']['answer_preserved_on_design_failure']
    assert '未通过' in wire.render(parsed)
    assert parsed['validation']['extra_model_calls'] == 0


def test_outer_extra_fields_do_not_hide_answer_or_grant_structural_success():
    data, sources = packet()
    data['executed'] = True
    parsed = wire.parse(json.dumps(data), sources, trusted_budget_minutes=120)
    assert parsed['answer'] == data['answer']
    assert parsed['validation']['status'] == 'draft'
    assert parsed['validation']['framing_failure'] == 'outer_schema'
    assert parsed['validation']['executed'] is False


def test_schema_is_closed_two_strings_and_complete_input_binding_is_not_optional():
    schema, contract = wire.schema(), wire.contract()
    assert set(schema['properties']) == set(schema['required']) == {'answer', 'validation_design'}
    assert all(value['type'] == 'string' for value in schema['properties'].values())
    assert schema['additionalProperties'] is False
    messages = [{'role': 'system', 'content': contract['policy']},
                {'role': 'user', 'content': contract['prompt']}]
    assert len(wire.input_sha256(messages)) == 64
    messages[1]['content'] = '{}'
    with pytest.raises(ValueError):
        wire.input_sha256(messages)
