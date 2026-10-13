"""Definition handoff is bounded source geometry, not generated explanation."""
from copy import deepcopy
import hashlib
import json
from unittest.mock import patch

import pytest

from docthinker.server.research_answer_contract import (
    DATA_MARKER, SYSTEM_POLICY, answer_contract_receipt, source_quote_payload,
)
from docthinker.server.research_component_coverage import (
    MAX_WRITER_COMPONENT_BYTES, component_definition_locators,
)

QUESTION = 'Explain the architecture, component functions and conditions.'


def source(text, label='S1', start=31, **extra):
    return {'id': 'chunk-' + label, 'document_id': 'doc-' + label, 'label': label,
            'filename': label + '.txt', 'content': text, 'truncated': bool(start),
            'excerpt_start': start, 'excerpt_end': start + len(text), **extra}


def locators(packet):
    return [item for row in packet for item in row.get('component_definition_locators', [])]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def assert_geometry(packet):
    for row in packet:
        for item in row.get('component_definition_locators', []):
            assert item['source_label'] == row['label']
            assert item['offset_unit'] == 'characters_in_content'
            a, b = item['definition_start'], item['definition_end']
            c, d = item['name_start'], item['name_end']
            assert 0 <= a <= c < d <= b <= len(row['content'])
            assert row['content'][c:d] == item['name']
            # Relative offsets compose exactly with original chunk coordinates.
            assert row['excerpt_start'] + b <= row['excerpt_end']
            assert not {'content', 'quote', 'explanation', 'functionally_defined', 'verified'} & item.keys()


def test_english_handoff_locates_full_conditions_and_negation_without_rewriting():
    text = ('🌿 The amber queue stores immutable events only after consent. '
            'The cobalt cache does not return expired values. '
            'However, it is unavailable unless the integrity check succeeds.')
    rows = [source(text)]
    original = deepcopy(rows)
    packet = source_quote_payload(rows, question=QUESTION)
    by_name = {item['name']: item for item in locators(packet)}
    assert set(by_name) == {'amber queue', 'cobalt cache'}
    for name, expected in [('amber queue', 'only after consent'),
                           ('cobalt cache', 'unless the integrity check succeeds')]:
        item = by_name[name]
        definition = text[item['definition_start']:item['definition_end']]
        assert expected in definition
    cache = by_name['cobalt cache']
    assert 'does not return expired values' in text[cache['definition_start']:cache['definition_end']]
    assert_geometry(packet)
    assert rows == original and packet[0]['content'] == text


def test_chinese_handoff_preserves_unicode_and_negative_scope():
    text = '🌿 琥珀队列保存不可变事件，仅在授权后。钴蓝缓存不返回过期内容。但该缓存仅在校验成功时可用。'
    packet = source_quote_payload([source(text, start=117)], question='解释系统架构和组件分工。')
    by_name = {item['name']: item for item in locators(packet)}
    assert {'琥珀队列', '钴蓝缓存'} <= by_name.keys()
    marker = by_name['钴蓝缓存']
    quote = text[marker['definition_start']:marker['definition_end']]
    assert '不返回过期内容' in quote and '仅在校验成功时可用' in quote
    assert_geometry(packet)


@pytest.mark.parametrize('text', [
    'The amber queue and cobalt cache are mentioned.',
    'Memory stores records.',
    'The amber queue stores.',
    'The amber queue stores. However, the experimental conditions are unspecified.',
    'The amber queue stores records especially when',
    'The amber queue stores records especially when...',
    'Figure 3: The amber queue stores records.',
    'References\nThe amber queue stores records.',
    'To address this, the amber queue stores records.',
])
def test_names_incomplete_definitions_and_unsupported_context_do_not_create_locators(text):
    rows = [source(text)]
    packet = source_quote_payload(rows, question=QUESTION)
    assert locators(packet) == []
    assert packet[0]['content'] == text  # No source removal or repair.


@pytest.mark.parametrize('question', [
    '', 'Hello there.', 'Compare these.', 'What is the reported latency?',
    'Compare the accuracy scores only.', 'Compare results; do not explain architecture.',
    'Compare only exact-match accuracy numbers, not architecture.',
    'Compare accuracy, excluding the architecture.', '仅比较准确率，架构除外。',
    '只比较数值和准确率，不需要解释架构。',
])
def test_unknown_nonstructural_and_explicit_exclusion_do_not_extract_definition_data(question):
    rows = [source('The amber queue stores private records.')]
    with patch('docthinker.server.research_component_coverage.functional_definitions',
               side_effect=AssertionError('nonstructural task must not inspect definitions')):
        assert locators(source_quote_payload(rows, question=question)) == []


def test_explicit_disable_and_legacy_projection_leave_packet_unchanged():
    rows = [source('The amber queue stores records.')]
    expected = [{key: rows[0][key] for key in
                 ('label', 'filename', 'content', 'truncated', 'excerpt_start', 'excerpt_end')}]
    expected[0]['offset_unit'] = 'characters_in_source_chunk'
    assert source_quote_payload(rows) == expected
    with patch('docthinker.server.research_answer_contract.component_definition_locators',
               side_effect=AssertionError('disabled optional handoff must not run')):
        assert source_quote_payload(rows, question=QUESTION, include_component_locators=False) == expected


def test_truncated_but_complete_visible_quote_can_have_locator_partial_tail_cannot():
    text = 'The amber queue stores records only after consent.'
    full = source_quote_payload([source(text)], question=QUESTION)
    assert len(locators(full)) == 1 and full[0]['truncated'] is True
    partial = source_quote_payload([source(text[:-6])], question=QUESTION)
    assert locators(partial) == []


@pytest.mark.parametrize('extra', [
    {'excerpt_start': -1}, {'excerpt_start': True}, {'excerpt_end': 99999},
    {'label': 'S11'}, {'label': 'K1'},
])
def test_invalid_geometry_or_non_source_label_does_not_gain_a_locator(extra):
    row = source('The amber queue stores records.', **extra)
    assert component_definition_locators(QUESTION, [row]) == []


def test_packet_wide_bound_and_round_robin_do_not_equate_same_name_across_sources():
    text = ('The amber queue stores records. The cobalt cache holds values. '
            'The violet buffer returns batches. The silver index retrieves documents.')
    rows = [source(text, label=f'S{n}') for n in range(1, 11)]
    original = deepcopy(rows)
    plain = source_quote_payload(rows)
    packet = source_quote_payload(rows, question=QUESTION)
    items = locators(packet)
    assert len(items) == 4 and len({item['source_label'] for item in items}) == 4
    assert {item['name'] for item in items} == {'amber queue'}
    assert 0 < len(canonical(packet)) - len(canonical(plain)) <= MAX_WRITER_COMPONENT_BYTES
    assert rows == original and len(packet) == 10
    assert_geometry(packet)


def test_source_fit_variants_recompute_only_retained_locators_and_hash_actual_packet():
    rows = [source('The amber queue stores immutable events.', label='S1'),
            source('The cobalt cache does not return expired values.', label='S2')]
    digests = []
    for retained in (rows, rows[:1], []):
        packet = source_quote_payload(retained, question=QUESTION)
        messages = [{'role': 'system', 'content': SYSTEM_POLICY},
                    {'role': 'user', 'content': DATA_MARKER + json.dumps({'source_quotes': packet})}]
        receipt = answer_contract_receipt(messages, retained, original_source_count=2, question=QUESTION)
        assert receipt['evidence_packet_sha256'] == hashlib.sha256(canonical(packet)).hexdigest()
        assert receipt['component_locator_count'] == len(locators(packet)) == len(retained)
        assert receipt['semantic_support_verified'] is False and receipt['extra_provider_calls'] == 0
        assert {item['source_label'] for item in locators(packet)} <= {row['label'] for row in retained}
        digests.append(receipt['evidence_packet_sha256'])
    assert len(set(digests)) == 3


def test_shared_policy_handoff_is_bounded_and_does_not_claim_definitions_are_verified():
    assert len(SYSTEM_POLICY.encode()) <= 3800
    assert 'component_definition_locators 仅定位原文' in SYSTEM_POLICY
    assert '功能、分工及条件，勿只列名字，不视作语义验证' in SYSTEM_POLICY
