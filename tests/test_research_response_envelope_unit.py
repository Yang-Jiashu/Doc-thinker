"""Strict shared response presentation, never inference/proof repair."""
import json

import pytest

from docthinker.server.research_response_envelope import decode_research_envelope


@pytest.mark.parametrize('fenced', [False, True])
def test_explicit_host_contract_can_bind_unique_metadata_only_tail_to_intact_prose(fenced):
    body = '🌿 An intact sourced draft [S1].\n\nNo action was taken.'
    metadata = {'task_assessment': {'requirements': []}}
    wire = json.dumps(metadata)
    raw = body + '\n\n' + ('```json\n' + wire + '\n```' if fenced else wire)
    assert decode_research_envelope(raw, allowed_metadata_fields=('task_assessment',)) == {
        'answer': body, **metadata}
    with pytest.raises(ValueError, match='Invalid research response envelope'):
        decode_research_envelope(raw)  # A chat example is not an implicit contract.


def test_combined_metadata_requires_every_field_to_have_an_actual_host_contract():
    metadata = {'task_assessment': {'requirements': []}, 'event_assessment': {'checks': []}}
    raw = 'The complete draft.\n' + json.dumps(metadata)
    with pytest.raises(ValueError):
        decode_research_envelope(raw, allowed_metadata_fields=('task_assessment',))
    assert decode_research_envelope(raw, allowed_metadata_fields=tuple(metadata)) == {
        'answer': 'The complete draft.', **metadata}


@pytest.mark.parametrize('raw', [
    '{"task_assessment":{}}',
    '```json\n{"task_assessment":{}}\n```',
    'Draft\n{"task_assessment":{},"tools":[]}',
    'Draft\n{"task_assessment":[]}',
    'Draft\n{"task_assessment":{},"task_assessment":{}}',
    'Draft\n{"task_assessment":{"value":NaN}}',
    'Draft\n{"task_assessment":{}}\nSuffix text.',
    'Draft\n{"task_assessment":{}}\n{"task_assessment":{}}',
    'Draft\n```json\n{"task_assessment":{}}\n```\n```json\n{"task_assessment":{}}\n```',
    'Draft\n```json\n{"task_assessment":{}}',
    'Draft\n{"nested":\n{"task_assessment":{}}\n}',
    'Draft {"task_assessment":{}}',
    'An example `{"task_assessment":{}}`.\n{"task_assessment":{}}',
])
def test_metadata_only_tail_never_repairs_ambiguity_or_malformed_protocol(raw):
    with pytest.raises(ValueError, match='Invalid research response envelope'):
        decode_research_envelope(raw, allowed_metadata_fields=('task_assessment',))


def test_existing_full_envelope_and_ordinary_prose_remain_compatible():
    value = {'answer': 'Original final body.', 'task_assessment': {'requirements': []}}
    assert decode_research_envelope(json.dumps(value)) == value
    assert decode_research_envelope('Ordinary prose and a nonprotocol {example}.') is None
    with pytest.raises(ValueError):
        decode_research_envelope('Ordinary prose', allowed_metadata_fields=('answer',))
