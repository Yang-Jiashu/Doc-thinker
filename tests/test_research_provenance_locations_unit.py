"""Review locations are literal display aids, not experimental entailment."""
import copy

import pytest

from docthinker.server.research_provenance import (
    MAX_REVIEW_QUOTE_CHARS, MAX_REVIEW_SCAN_CHARS, MAX_SOURCE_REFS,
    audit_answer, hydrate_audit,
)


def source(content, **fields):
    return {'id': 'source-one', 'label': 'S1', 'content': content, 'excerpt_start': 29, **fields}


def inspect(answer, content):
    row = source(content)
    audit = audit_answer(answer, [row])
    claim = audit['claims'][0]
    assert audit['entailment_verified'] is False
    assert claim['status'] == 'unverified'
    assert len(claim['source_refs']) <= MAX_SOURCE_REFS
    assert sum(len(ref['quote']) for ref in claim['source_refs']) <= MAX_REVIEW_QUOTE_CHARS
    for ref in claim['source_refs']:
        assert content[ref['start'] - row['excerpt_start']:ref['end'] - row['excerpt_start']] == ref['quote']
    return audit, claim, {item['code'] for item in claim['issues']}


def test_paraphrase_locates_tail_instead_of_unrelated_first_thousand_characters():
    prefix = 'Glaciers blanket mountains in winter. ' * 40
    target = 'After adaptation, the compact model consumed 83 GPU hours and reached 72 EM.'
    _, claim, _ = inspect('Adaptation required 83 GPU hours for 72 EM [S1].', prefix + target)
    assert any(target in ref['quote'] for ref in claim['source_refs'])
    assert all(ref['start'] >= 29 + len(prefix) for ref in claim['source_refs'])
    assert all('Glaciers' not in ref['quote'] for ref in claim['source_refs'])


def test_distinct_results_get_separate_exact_ranges_not_spliced_source_text():
    first = 'The validation score reached 72 EM.'
    gap = 'Rainfall was recorded throughout spring. ' * 6
    second = 'The adaptation consumed 83 GPU hours.'
    content = first + ' ' + gap + second
    _, claim, _ = inspect('Validation reached 72 EM with adaptation consuming 83 GPU hours [S1].', content)
    refs = claim['source_refs']
    assert len(refs) == 2
    assert refs[0]['end'] < refs[1]['start']
    assert first in refs[0]['quote'] and second in refs[1]['quote']
    assert all('Rainfall' not in ref['quote'] for ref in refs)


def test_shared_number_does_not_choose_wrong_experiment_over_distinct_lexical_anchors():
    first = 'The reactor experiment measured pressure at 41 units.'
    target = 'The vision experiment measured validation accuracy at 41 percent.'
    _, claim, _ = inspect('Vision validation accuracy was 41 percent [S1].', first + ' ' + target)
    assert len(claim['source_refs']) == 1
    assert target in claim['source_refs'][0]['quote']


def test_numbers_from_different_experiments_do_not_verify_a_combined_claim():
    content = 'Experiment amber used 31 samples. Experiment violet obtained 88 percent accuracy.'
    _, claim, _ = inspect('With 31 samples the experiment obtained 88 percent accuracy [S1].', content)
    assert len(claim['source_refs']) == 2
    assert claim['status'] == 'unverified'


@pytest.mark.parametrize('number', ['31', '62.7', '82.4%'])
def test_repeated_numeric_only_locations_are_disclosed_not_disambiguated(number):
    content = f'Experiment amber observed {number}. Experiment violet observed {number}.'
    _, claim, codes = inspect(f'观测值为{number} [S1]。', content)
    assert len(claim['source_refs']) == 2
    assert {'source_location_ambiguous', 'source_location_numeric_only'} <= codes


def test_only_three_repeated_locations_can_be_displayed_and_omissions_are_visible():
    content = ' '.join(f'Trial {name} recorded 31 units.' for name in ('amber', 'red', 'green', 'blue', 'violet'))
    _, claim, codes = inspect('数值为31 [S1]。', content)
    assert len(claim['source_refs']) == MAX_SOURCE_REFS
    assert {'source_location_ambiguous', 'source_location_budget'} <= codes


def test_disconnected_quotes_share_the_old_per_source_character_budget():
    content = ' '.join(f'Trial {name} used ' + 'intermediate calibration ' * 14 + '31 units.'
                       for name in ('amber', 'violet', 'green'))
    _, claim, codes = inspect('数值为31 [S1]。', content)
    assert len(claim['source_refs']) == 2
    assert sum(len(ref['quote']) for ref in claim['source_refs']) <= MAX_REVIEW_QUOTE_CHARS
    assert 'source_location_budget' in codes


def test_chinese_tail_keeps_qualifications_and_character_not_byte_offsets():
    prefix = '🐱春季进行了降雨观测。' * 110
    target = '实施增量微调后，验证准确率达到82.4%。但仅在固定测试集上成立。'
    _, claim, _ = inspect('增量微调带来的验证准确率为82.4% [S1]。', prefix + target)
    assert len(claim['source_refs']) == 1
    assert claim['source_refs'][0]['quote'] == target
    assert claim['source_refs'][0]['start'] == 29 + len(prefix)


def test_pdf_hyphenated_word_is_ranked_without_mutating_quote_or_offsets():
    target = 'The validation accu-\nracy improved substantially after adaptation.'
    _, claim, _ = inspect('Validation accuracy improved after adaptation [S1].', target)
    assert len(claim['source_refs']) == 1
    assert claim['source_refs'][0]['quote'] == target


def test_unmatched_claim_does_not_show_source_prefix_or_claim_source_was_deleted():
    _, claim, codes = inspect('Amortization necessitates contingency [S1].', 'Glaciers blanket mountains in winter.')
    assert claim['source_refs'] == []
    assert 'source_location_unresolved' in codes
    assert 'source_unavailable' not in codes


def test_a_single_oversized_relevant_unit_is_not_silently_truncated():
    content = 'The adaptation protocol ' + 'uses carefully controlled intermediate calibration ' * 30 + 'consumed 83 GPU hours.'
    _, claim, codes = inspect('Adaptation consumed 83 GPU hours [S1].', content)
    assert claim['source_refs'] == []
    assert {'source_location_unresolved', 'source_location_budget'} <= codes


def test_scan_cap_is_visible_and_cannot_create_a_partial_sentence_match():
    content = 'The adaptation consumed 83 GPU hours while ' + 'maintaining intermediate calibration ' * 500
    assert len(content) > MAX_REVIEW_SCAN_CHARS
    _, claim, codes = inspect('Adaptation consumed 83 GPU hours [S1].', content)
    assert claim['source_refs'] == []
    assert 'source_location_windowed' in codes


def test_three_cited_sources_keep_one_slot_each_and_existing_total_cap():
    rows = [source(f'Trial {name} consumed 83 GPU hours.', id=f'source-{index}', label=f'S{index}')
            for index, name in enumerate(('amber', 'violet', 'green'), 1)]
    audit = audit_answer('The trial consumed 83 GPU hours [S1] [S2] [S3].', rows)
    refs = audit['claims'][0]['source_refs']
    assert len(refs) == 3
    assert [ref['label'] for ref in refs] == ['S1', 'S2', 'S3']
    assert sum(len(ref['quote']) for ref in refs) <= 3 * MAX_REVIEW_QUOTE_CHARS


def test_missing_source_is_still_unavailable_not_just_unresolved():
    audit = audit_answer('The trial consumed 83 GPU hours [S9].', [source('The trial consumed 83 GPU hours.')])
    assert audit['claims'][0]['status'] == 'source_unavailable'


def test_hydration_preserves_separate_ranges_without_mutating_saved_audit():
    content = 'The model reached 72 EM. Rainfall was heavy. Adaptation consumed 83 GPU hours.'
    audit, claim, _ = inspect('Reaching 72 EM required adaptation using 83 GPU hours [S1].', content)
    saved = copy.deepcopy(audit)
    for item in saved['claims']:
        for ref in item['source_refs']:
            ref.pop('quote')  # Existing persistence stores pointers, not copies.
    before = copy.deepcopy(saved)
    hydrated = hydrate_audit(saved, [source(content)])
    assert saved == before
    assert hydrated['claims'][0]['source_refs'] == claim['source_refs']
    assert hydrated['claims'][0]['status'] == 'unverified'


def test_revocation_removes_every_disconnected_quote():
    content = 'The model reached 72 EM. Rainfall was heavy. Adaptation consumed 83 GPU hours.'
    audit, claim, _ = inspect('Reaching 72 EM required adaptation using 83 GPU hours [S1].', content)
    assert len(claim['source_refs']) == 2
    hydrated = hydrate_audit(audit, [])
    assert hydrated['claims'][0]['source_refs'] == []
    assert hydrated['claims'][0]['status'] == 'source_unavailable'
    assert '72 EM' not in str(hydrated['claims'][0]['source_refs'])


def test_same_length_changed_quote_is_rejected_when_rehydrating_live_audit():
    audit, _, _ = inspect('Adaptation took 83 GPU hours [S1].', 'Adaptation consumed 83 GPU hours.')
    result = hydrate_audit(audit, [source('Adaptation consumed 84 GPU hours.')])
    assert result['claims'][0]['source_refs'] == []
    assert result['claims'][0]['status'] == 'source_unavailable'


def test_ranking_and_budget_are_deterministic():
    content = 'Trial amber used 31 units. Trial violet used 31 units.'
    first, _, _ = inspect('数值为31 [S1]。', content)
    second, _, _ = inspect('数值为31 [S1]。', content)
    assert first == second


def test_one_literal_source_cannot_certify_a_second_paraphrased_source():
    rows = [source('The treatment improves survival.'),
            source('The treatment improves colour.', id='source-two', label='S2')]
    audit = audit_answer('The treatment improves survival.[S1][S2]', rows)
    claim = audit['claims'][0]
    assert claim['status'] == 'unverified'
    assert len(claim['source_refs']) == 2
    assert 'semantic_review' in {issue['code'] for issue in claim['issues']}
    hydrated = hydrate_audit(audit, rows)
    assert hydrated['claims'][0]['status'] == 'unverified'
    assert hydrated['claims'][0]['source_refs'] == claim['source_refs']
    # Old any-source exact badges are review failures, not source revocations.
    legacy = copy.deepcopy(audit)
    legacy['claims'][0].update(status='exact_quote', issues=[])
    restored = hydrate_audit(legacy, rows)['claims'][0]
    assert restored['status'] == 'unverified'
    assert len(restored['source_refs']) == 2
    assert 'source_unavailable' not in {issue['code'] for issue in restored['issues']}


@pytest.mark.parametrize('content,quote', [
    ('We do not conclude that\nthis method is effective.', 'this method is effective.'),
    ('We do not conclude that\nTHIS METHOD IS EFFECTIVE.', 'THIS METHOD IS EFFECTIVE.'),
    ("We don't conclude that\nthis method is effective.", 'this method is effective.'),
    ('If the control is fixed,\nthis method is effective.', 'this method is effective.'),
    ('This method is effective. However, only under controlled conditions.', 'This method is effective.'),
    ('If A holds, this method is effective only\nwhen B holds.', 'If A holds, this method is effective'),
])
def test_literal_substring_cannot_drop_negation_or_conditions_across_pdf_wraps(content, quote):
    audit = audit_answer(quote + '[S1]', [source(content)])
    claim = audit['claims'][0]
    assert claim['status'] == 'unverified'
    assert 'quote_context_review' in {issue['code'] for issue in claim['issues']}
    ref = claim['source_refs'][0]
    assert content[ref['start'] - 29:ref['end'] - 29] == ref['quote'] == quote
    assert hydrate_audit(audit, [source(content)])['claims'][0]['status'] == 'unverified'


def test_ordinary_pdf_wrap_is_not_a_revocation_or_false_context_warning():
    content = 'The study reports that\nthis method is effective.'
    audit = audit_answer('this method is effective.[S1]', [source(content)])
    assert audit['claims'][0]['status'] == 'exact_quote'
    assert hydrate_audit(audit, [source(content)])['claims'][0]['status'] == 'exact_quote'


def test_same_id_context_replacement_invalidates_fingerprinted_exact_quote():
    quote = 'the treatment improves survival.'
    old = source('Report: '.ljust(24) + quote)
    changed = source('It is not true that '.ljust(24) + quote)
    audit = audit_answer(quote + '[S1]', [old])
    assert audit['claims'][0]['status'] == 'exact_quote'
    # Exercise the real persistence shape: text removed, fingerprint retained.
    saved = copy.deepcopy(audit)
    saved['claims'][0]['source_refs'][0].pop('quote')
    hydrated = hydrate_audit(saved, [changed])['claims'][0]
    assert hydrated['status'] == 'source_unavailable'
    assert hydrated['source_refs'] == []
    assert hydrated['text'] == audit['claims'][0]['text']
    # A pre-fingerprint live record can retain its literal text for review,
    # but must not preserve an exact badge based on the previous context.
    legacy = copy.deepcopy(audit)
    legacy['claims'][0]['source_refs'][0].pop('source_window_sha256')
    restored = hydrate_audit(legacy, [changed])['claims'][0]
    assert restored['status'] == 'unverified'
    assert restored['source_refs'][0]['quote'] == quote
    assert 'quote_context_review' in {issue['code'] for issue in restored['issues']}


@pytest.mark.parametrize('replacement', ['Adaptation consumed 83 GPU hours.', 'Adaptation consumed 84 GPU hours.'])
def test_legacy_pointer_only_history_cannot_silently_repopulate_current_text(replacement):
    audit, _, _ = inspect('Adaptation took 83 GPU hours [S1].', 'Adaptation consumed 83 GPU hours.')
    saved = copy.deepcopy(audit)
    for ref in saved['claims'][0]['source_refs']:
        ref.pop('quote')
        ref.pop('source_window_sha256')
    before = copy.deepcopy(saved)
    claim = hydrate_audit(saved, [source(replacement)])['claims'][0]
    assert saved == before
    assert claim['status'] == 'unverified'
    assert claim['source_refs'] == []
    assert 'historical_source_unverified' in {issue['code'] for issue in claim['issues']}
    assert 'source_unavailable' not in {issue['code'] for issue in claim['issues']}


def test_fingerprinted_pointer_history_detects_same_length_quote_replacement():
    audit, _, _ = inspect('Adaptation took 83 GPU hours [S1].', 'Adaptation consumed 83 GPU hours.')
    audit['claims'][0]['source_refs'][0].pop('quote')
    claim = hydrate_audit(audit, [source('Adaptation consumed 84 GPU hours.')])['claims'][0]
    assert claim['status'] == 'source_unavailable'
    assert claim['source_refs'] == []


def test_exact_lookup_obeys_the_same_scan_cap_as_review_locations(monkeypatch):
    import docthinker.server.research_provenance as provenance
    original = provenance._locate_literal
    def bounded_lookup(text, content):
        assert len(content) <= MAX_REVIEW_SCAN_CHARS
        return original(text, content)
    monkeypatch.setattr(provenance, '_locate_literal', bounded_lookup)
    content = 'x' * (MAX_REVIEW_SCAN_CHARS + 500) + ' Tail match is present.'
    claim = audit_answer('Tail match is present.[S1]', [source(content)])['claims'][0]
    assert claim['status'] == 'unverified'
    assert claim['source_refs'] == []
    assert 'source_location_windowed' in {issue['code'] for issue in claim['issues']}


def test_hydration_cannot_exceed_per_source_quote_or_scan_budgets():
    content = 'x' * (MAX_REVIEW_SCAN_CHARS + 500)
    audit = audit_answer('A claim [S1]', [source(content)])
    claim = audit['claims'][0]
    claim['source_refs'] = [
        {'source_id': 'source-one', 'label': 'S1', 'start': 29 + start,
         'end': 29 + end, 'quote': content[start:end]}
        for start, end in [(0, 600), (600, 1200), (12001, 12401)]]
    restored = hydrate_audit(audit, [source(content)])['claims'][0]
    assert restored['status'] == 'unverified'
    assert sum(len(ref['quote']) for ref in restored['source_refs']) == 600
    assert {'source_location_budget', 'source_location_windowed'} <= {issue['code'] for issue in restored['issues']}
    claim['source_refs'] = [{'source_id': 'source-one', 'label': 'S1', 'start': 29,
                            'end': 5029, 'quote': content[:5000]}]
    restored = hydrate_audit(audit, [source(content)])['claims'][0]
    assert restored['source_refs'] == []
    assert restored['status'] == 'unverified'


def test_hydration_keeps_at_most_three_references_and_discloses_omission():
    row = source('Original statement.')
    audit = audit_answer('Original statement.[S1]', [row])
    claim = audit['claims'][0]
    claim['source_refs'] *= 4
    restored = hydrate_audit(audit, [row])['claims'][0]
    assert len(restored['source_refs']) == 3
    assert restored['status'] == 'unverified'
    assert 'citation_window' in {issue['code'] for issue in restored['issues']}


def test_long_whitespace_before_a_citation_remains_a_single_claim():
    from docthinker.server.research_provenance import _sentences
    text = 'Claim!' + ' ' * 100000 + '[S1]'
    assert list(_sentences(text)) == [text]
