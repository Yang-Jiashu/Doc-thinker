"""Literal provenance is testable; paraphrase entailment is not asserted."""
import pytest

from docthinker.server.research_provenance import audit_answer, hydrate_audit


def source(content, **kwargs):
    return {'id': 'chunk-one', 'label': 'S1', 'content': content, 'excerpt_start': 7, **kwargs}


@pytest.mark.parametrize('text', ['Renewable supply reduces imports.', '条件成立时才降低成本。', '🐱实验组测量值为 17%。'])
def test_exact_quote_preserves_unicode_offsets(text):
    audit = audit_answer(text + ' [S1]', [source(text)])
    claim = audit['claims'][0]
    assert claim['status'] == 'exact_quote'
    ref = claim['source_refs'][0]
    assert ref['start'] == 7 and ref['end'] == 7 + len(text)
    assert ref['quote'] == text
    assert audit['entailment_verified'] is False


@pytest.mark.parametrize('answer,expected', [
    ('Supply lowered imports [S1].', 'semantic_review'),
    ('Supply lowers imports by 70% [S1].', 'number_not_in_source'),
    ('Supply never lowers imports [S1].', 'polarity_review'),
])
def test_paraphrases_and_changes_cannot_claim_verification(answer, expected):
    audit = audit_answer(answer, [source('Supply lowers imports by 7%.')])
    assert audit['claims'][0]['status'] == 'unverified'
    assert expected in {issue['code'] for issue in audit['claims'][0]['issues']}


def test_omitting_conditions_requires_review():
    result = audit_answer('Treatment reduces risk [S1].', [source('Treatment may reduce risk only when adherence is high.')])
    assert 'condition_review' in {row['code'] for row in result['claims'][0]['issues']}
    assert not result['entailment_verified']


def test_missing_extra_source_cannot_receive_exact_badge():
    result = audit_answer('Supply reduces imports [S1] [S99].', [source('Supply reduces imports')])
    assert result['claims'][0]['status'] != 'exact_quote'


def test_revocation_removes_quotes_even_if_old_audit_contains_them():
    result = audit_answer('PRIVATE_SOURCE_LITERAL [S1]', [source('PRIVATE_SOURCE_LITERAL')])
    assert result['status'] == 'quoted_only'
    revoked = hydrate_audit(result, [])
    assert revoked['status'] == 'review_needed'
    assert revoked['claims'][0]['source_refs'] == []
    assert revoked['claims'][0]['status'] == 'source_unavailable'


def test_changed_excerpt_cannot_verify_unavailable_offsets():
    result = audit_answer('Measured result [S1]', [source('Measured result')])
    assert hydrate_audit(result, [source('Different', excerpt_start=100)])['claims'][0]['status'] == 'source_unavailable'


def test_audit_window_does_not_silently_claim_full_answer_coverage():
    result = audit_answer('\n'.join(f'Result number {number} [S1]' for number in range(20)), [source('Result number 0')])
    assert len(result['claims']) == 12
    assert result['summary']['omitted_statement_count'] == 8
    assert result['status'] == 'review_needed'
