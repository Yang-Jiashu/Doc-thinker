"""Strict unique-root output decoding, never loose JSON example extraction."""
import json

import pytest

from docthinker.server.research_response_envelope import decode_research_envelope


def packet(**extra):
    return {'answer': 'The concise sourced answer [S1].', 'event_assessment': {'checks': []}, **extra}


@pytest.mark.parametrize('prefix,fence', [('', None), ('Normal prose first.\n\n', None),
    ('', 'json'), ('Normal prose first.\n\n', 'json'), ('Normal prose first.\n', ''),
    ('Normal prose first.\r\n\r\n', 'JSON')])
def test_unique_terminal_root_after_plain_prose_is_accepted(prefix, fence):
    value = packet(task_assessment={'requirements': []}, trial_spec={'version': 1})
    body = json.dumps(value, ensure_ascii=False, indent=2)
    text = prefix + (f'```{fence}\n{body}\n```' if fence is not None else body)
    assert decode_research_envelope(text) == value


@pytest.mark.parametrize('text', ['A normal source-grounded answer [S1].', 'Ordinary prose with {balanced braces}.',
    '{"ordinary_data":123}', '```python\nprint("hello")\n```', 'The word event_assessment is not a payload.'])
def test_nonprotocol_plain_content_has_no_envelope(text):
    assert decode_research_envelope(text) is None


@pytest.mark.parametrize('transform', [
    lambda body: body + '\nTrailing commentary.',
    lambda body: body + '\n' + body,
    lambda body: body + body,
    lambda body: 'An inline example: ' + body,
    lambda body: '{"wrapper":\n' + body + '\n}',
    lambda body: '{"wrapper":\n' + body,
    lambda body: '[\n' + body + '\n]',
    lambda body: '```python\n' + body + '\n```',
    lambda body: '```json\n' + body + '\n```\nTrailing text.',
    lambda body: '```json\n' + body,
    lambda body: 'Previous protocol: "event_assessment": {}\n' + body,
])
def test_ambiguous_nested_inline_or_nonterminal_metadata_is_never_extracted(transform):
    with pytest.raises(ValueError, match='^Invalid research response envelope$'):
        decode_research_envelope(transform(json.dumps(packet())))


@pytest.mark.parametrize('text', [
    '{"answer":"first","answer":"second","event_assessment":{}}',
    '{"answer":"ok","event_assessment":{"checks":[],"checks":[]}}',
    '{"answer":"ok","event_assessment":NaN}',
    '{"answer":"ok","event_assessment":Infinity}',
    '{"answer":"ok","tool_calls":[{"name":"send_email"}]}',
    '{"answer":"ok","event_assessment":{',
    '{"answer":[],"event_assessment":{}}',
    '{"answer":"","event_assessment":{}}',
    '{"answer":"\\ud800","event_assessment":{}}',
])
def test_bad_types_duplicate_nonfinite_and_unknown_keys_raise_fixed_error(text):
    with pytest.raises(ValueError, match='^Invalid research response envelope$'):
        decode_research_envelope('Normal prose.\n' + text)


def test_size_unicode_and_scan_bounds_are_fail_closed():
    with pytest.raises(ValueError, match='^Invalid research response envelope$'):
        decode_research_envelope(json.dumps(packet()), max_bytes=10)
    with pytest.raises(ValueError, match='^Invalid research response envelope$'):
        decode_research_envelope('\ud800')
    with pytest.raises(ValueError, match='^Invalid research response envelope$'):
        decode_research_envelope(('{}\n' * 65) + json.dumps(packet()))


def test_answer_field_text_and_joint_annotation_are_not_rewritten():
    value = packet(answer='Original exact answer witness text\nwith formatting [S1].',
        event_assessment={'checks': [{'id': 'L1', 'evidence': [{'label': 'S1', 'quote': 'Exact source\nquote'}]}]})
    assert decode_research_envelope('A prefix.\n' + json.dumps(value)) == value
