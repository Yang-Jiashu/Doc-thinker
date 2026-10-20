"""One bounded syntax contract; citation recognition is not semantic support."""
from copy import deepcopy
from uuid import uuid4

import pytest

from docthinker.server.research_citation_syntax import (
    CITATION_GROUP_PATTERN, MAX_CITATION_SCAN_CHARS, MAX_CITATION_GROUPS,
    MAX_CITATION_LABELS, parse_citation_groups, extract_citation_labels, strip_citation_groups,
)
from docthinker.server.research_retrieval import citation_validation
from docthinker.server.research_knowledge import knowledge_receipt, select_task_knowledge
from docthinker.server.insight_runtime import InsightRuntime
from docthinker.server.research_provenance import audit_answer, hydrate_audit
from docthinker.server.research_store import ResearchStore, ResearchStoreError


def test_comma_groups_have_exact_unicode_offsets_and_first_seen_labels():
    answer = '🌿 原文 [s5, S8]，方法 [K1， s5, K2]。另见 [ S10 ]。'
    parsed = parse_citation_groups(answer)
    assert parsed['labels'] == ['S5', 'S8', 'K1', 'K2', 'S10']
    assert [answer[group['start']:group['end']] for group in parsed['groups']] == [
        '[s5, S8]', '[K1， s5, K2]', '[ S10 ]']
    assert parsed['groups'][0]['labels'] == ['S5', 'S8']
    assert parsed['truncated'] is False
    assert extract_citation_labels(answer, prefixes='S') == ['S5', 'S8', 'S10']
    assert extract_citation_labels(answer, prefixes='K') == ['K1', 'K2']
    assert all(CITATION_GROUP_PATTERN.fullmatch(answer[g['start']:g['end']]) for g in parsed['groups'])


@pytest.mark.parametrize('text', ['[S1-S3]', '[S1–S3]', '[S1,2]', '[S1; S2]', '[S1 / S2]',
                                   '[S1\nS2]', '[source 1]', '[S1, note]', '[SK1]'])
def test_ranges_and_ambiguous_syntax_are_not_expanded(text):
    assert parse_citation_groups(text)['labels'] == []
    assert strip_citation_groups(text) == text


def test_audit_view_preserves_unrequested_method_labels_and_original_answer():
    answer = 'Original fact [S1, K2, S8]. Procedure [K1].'
    assert strip_citation_groups(answer) == 'Original fact [K2]. Procedure [K1].'
    assert strip_citation_groups(answer, prefixes='SK') == 'Original fact . Procedure .'
    assert answer == 'Original fact [S1, K2, S8]. Procedure [K1].'
    assert strip_citation_groups('No tags. ' + '[S1] ' * 300).startswith('No tags. ')
    assert '[S1]' in strip_citation_groups('[S1] ' * 300)  # Limits never erase unscanned input.


def test_unknown_group_member_cannot_hide_behind_a_valid_single_citation():
    validation = citation_validation('Known [S1]. Other claim [S8, S999].', ['S1', 'S8'])
    assert validation['used_labels'] == ['S1', 'S8', 'S999']
    assert validation['unknown_labels'] == ['S999']
    assert validation['valid'] is False
    assert validation['warnings'][0]['code'] == 'unknown_citation_labels'


def test_only_comma_group_is_recognized_without_treating_k_as_document_evidence():
    result = citation_validation('Material [S5, S8].', ['S5', 'S8'])
    assert result['valid'] and not result['missing_citations']
    assert result['used_labels'] == ['S5', 'S8'] and result['entailment_verified'] is False
    method = citation_validation('Method [K1, K2].', ['S1'], knowledge_labels=['K1', 'K2'])
    assert method['missing_citations'] and not method['valid']
    mixed = citation_validation('Mixed [S1，K1, K999].', ['S1'], knowledge_labels=['K1'])
    assert mixed['unknown_labels'] == ['K999'] and not mixed['valid']


def test_invalid_canonical_numbers_remain_unknown_in_groups():
    result = citation_validation('[s1, S01, S0, k01, K0]', ['S1'], knowledge_labels=['K1'])
    assert result['used_labels'] == ['S1', 'S01', 'S0', 'K01', 'K0']
    assert result['unknown_labels'] == ['S01', 'S0', 'K01', 'K0']
    assert result['valid'] is False


@pytest.mark.parametrize('answer', [
    '[S1] ' * (MAX_CITATION_GROUPS + 1),
    '[S1] ' + 'x' * MAX_CITATION_SCAN_CHARS + ' [S999]',
    '[S1, S' + '9' * 5000 + ']',
    '[' + ', '.join('S' + str(n) for n in range(1, MAX_CITATION_LABELS + 2)) + ']',
])
def test_every_limit_is_visible_and_cannot_certify_an_unscanned_unknown(answer):
    parsed = parse_citation_groups(answer)
    assert parsed['truncated'] is True
    assert len(parsed['groups']) <= MAX_CITATION_GROUPS
    assert len(parsed['labels']) <= MAX_CITATION_LABELS
    assert all(len(label) <= 17 for label in parsed['labels'])
    validation = citation_validation(answer, ['S1'])
    assert validation['metadata_truncated'] and not validation['valid']


def test_knowledge_receipt_and_delivery_use_same_mixed_group_syntax():
    question = 'Compare these papers on evaluation conditions and conflicting results.'
    support, _ = select_task_knowledge(question)
    assert support and support['trace']['status'] == 'ready'
    known_methods = [row['label'] for row in support['sources']]
    answer = f'Fact [S1, S8]. Method [{known_methods[0]}, K999]. Unknown source [S1，S999].'
    receipt = knowledge_receipt(support, included=True, answer=answer, model_called=True)
    assert receipt['citation_validation']['unknown_labels'] == ['K999']
    assert receipt['citation_validation']['missing_citations'] is False
    sources = [{'id': f'chunk-{n}', 'label': f'S{n}', 'document_id': f'doc-{n}',
                'content': 'Actual original evidence with explicit conditions.'} for n in (1, 8)]
    snapshot = {'sources': sources, 'answer': answer}
    original = deepcopy(snapshot)
    delivery = InsightRuntime(question, support).on_stage('before_delivery', snapshot)
    check = delivery['actions'][0]['payload']
    assert check['unknown_source_labels'] == ['S999']
    assert check['unknown_method_labels'] == ['K999']
    assert check['missing_source_citations'] is False
    assert check['citation_parse_truncated'] is False
    assert check['entailment_verified'] is False and snapshot == original
    limited = InsightRuntime(question, support).on_stage('before_delivery', {
        **snapshot, 'answer': '[S1] ' * (MAX_CITATION_GROUPS + 1)})
    assert limited['actions'][0]['payload']['citation_parse_truncated'] is True


def test_knowledge_receipt_reports_truncated_syntax_without_claiming_support():
    support, _ = select_task_knowledge('Compare papers on evaluation conditions and conflicting results.')
    label = support['sources'][0]['label']
    receipt = knowledge_receipt(support, included=True,
                                answer=f'[{label}] ' * (MAX_CITATION_GROUPS + 1), model_called=True)
    assert receipt['citation_validation']['metadata_truncated'] is True
    assert receipt['citation_validation']['entailment_verified'] is False


def test_upstream_parse_limit_survives_persistence_and_restart(tmp_path):
    store = ResearchStore(tmp_path)
    answer_id = str(uuid4())
    validation = citation_validation('[K1] ' * (MAX_CITATION_GROUPS + 1), [], knowledge_labels=['K1'])
    assert validation['metadata_truncated'] and not validation['valid']
    assert validation['unknown_labels'] == [] and validation['used_labels'] == ['K1']
    store.record_answer('#00001', answer_id, [], validation, 'general')
    hydrated = ResearchStore(tmp_path).answer_metadata('#00001', [answer_id])[answer_id]['validation']
    assert hydrated['metadata_truncated'] is True and hydrated['valid'] is False
    assert hydrated['used_labels'] == ['K1'] and hydrated['unknown_labels'] == []
    assert hydrated['entailment_verified'] is False


@pytest.mark.parametrize('bad_flag', ['true', 1, None, {}, []])
def test_corrupt_upstream_parse_limit_flag_cannot_be_saved_as_checked(tmp_path, bad_flag):
    with pytest.raises(ResearchStoreError, match='flags'):
        ResearchStore(tmp_path).record_answer('#00001', str(uuid4()), [],
                                             {'valid': True, 'metadata_truncated': bad_flag}, 'general')


@pytest.mark.parametrize('suffix', [
    '[S1, K' + '9' * 100 + ']',
    '[S1, ' + ', '.join('K' + str(n) for n in range(1, MAX_CITATION_LABELS + 1)) + ']',
    '[S1] ' * (MAX_CITATION_GROUPS + 1),
])
def test_truncated_mixed_or_repeated_citations_never_gain_exact_badge(suffix):
    original = 'A literal recorded result'
    sources = [{'id': 'chunk-1', 'label': 'S1', 'content': original, 'excerpt_start': 9}]
    audit = audit_answer(original + ' ' + suffix, sources)
    assert audit['summary']['exact_quote_count'] == 0
    assert any(issue['code'] == 'citation_metadata_window'
               for claim in audit['claims'] for issue in claim['issues'])
    assert hydrate_audit(audit, sources)['summary']['exact_quote_count'] == 0


def test_grouped_exact_quotes_track_each_source_and_fail_closed_after_revocation():
    original = 'The recorded treatment did not improve the outcome.'
    sources = [{'id': f'chunk-{n}', 'label': f'S{n}', 'content': original, 'excerpt_start': n}
               for n in (5, 8)]
    answer = original + ' [S5, S8]'
    audit = audit_answer(answer, sources)
    claim = audit['claims'][0]
    assert claim['status'] == 'exact_quote'
    assert [ref['label'] for ref in claim['source_refs']] == ['S5', 'S8']
    assert all(ref['quote'] == original for ref in claim['source_refs'])
    revoked = hydrate_audit(audit, sources[:1])
    assert revoked['claims'][0]['status'] != 'exact_quote'
    assert revoked['summary']['exact_quote_count'] == 0
