"""Literal citation displacement signals, not model answer-quality scores."""
from docthinker.server.research_provenance import audit_answer, hydrate_audit, _literal_anchors, _review_anchors
from time import perf_counter


def source(label, text, start=0):
    return {'id': 'chunk-' + label, 'label': label, 'content': text,
            'excerpt_start': start, 'filename': label + '.txt'}


def codes(answer, rows):
    result = audit_answer(answer, rows)
    return result, {issue['code'] for claim in result['claims'] for issue in claim['issues']}


def test_displaced_architecture_markers_trigger_review_not_repair():
    rows = [source('S1', 'Both memory components are pretrained and knowledge access needs no additional training.'),
            source('S2', 'The dense vector index of Wikipedia is accessed by a neural retriever.', 41)]
    answer = 'The system uses a dense vector index of Wikipedia and a neural retriever [S1].'
    result, issues = codes(answer, rows)
    assert 'citation_anchor_drift' in issues
    assert result['entailment_verified'] is False
    assert result['claims'][0]['status'] == 'unverified'
    # A program must not silently assign the other label or claim it supports.
    assert all(ref['label'] == 'S1' for ref in result['claims'][0]['source_refs'])
    assert result['claims'][0]['text'].startswith('The system uses')


def test_joint_citations_compare_the_union_not_each_label_in_isolation():
    rows = [source('S1', 'Pretrained memory components expose knowledge access.'),
            source('S2', 'The dense vector index of Wikipedia uses a neural retriever.')]
    _, issues = codes('Memory components expose knowledge access with a dense vector index of Wikipedia [S1][S2].', rows)
    assert 'citation_anchor_drift' not in issues


def test_one_shared_name_or_two_words_cannot_trigger_displacement():
    rows = [source('S1', 'Performance depends on the benchmark and evaluation.'),
            source('S2', 'Wikipedia retrieval uses a dense vector index.')]
    for answer in ['Wikipedia improves performance [S1].',
                   'Wikipedia retrieval improves performance [S1].']:
        assert 'citation_anchor_drift' not in codes(answer, rows)[1]


def test_generic_research_discourse_does_not_trigger_a_warning_for_a_valid_comparison():
    rows = [source('S1', 'A scored 9 and B scored 3 on the same test.'),
            source('S2', 'The experimental results clearly demonstrate the advantages of a larger dataset.')]
    _, issues = codes('These experimental results clearly demonstrate that A scored higher than B [S1].', rows)
    assert 'citation_anchor_drift' not in issues


def test_exact_quote_retains_literal_status_and_offsets():
    quote = 'The dense vector index of Wikipedia uses a neural retriever.'
    rows = [source('S1', quote, 70), source('S2', quote)]
    result, issues = codes(quote + ' [S1]', rows)
    assert 'citation_anchor_drift' not in issues
    assert result['claims'][0]['status'] == 'exact_quote'
    assert result['claims'][0]['source_refs'][0]['start'] == 70


def test_pdf_wrap_normalization_does_not_modify_returned_quote():
    rows = [source('S1', 'Wikipedia provides a dense vector index and neural\nretriever.'),
            source('S2', 'Wikipedia provides a dense vector index and neural re-\ntriever.')]
    answer = 'Wikipedia dense vector retriever [S1].'
    assert 'citation_anchor_drift' not in codes(answer, rows)[1]
    assert 're-\ntriever' in rows[1]['content']


def test_only_one_pdf_soft_wrap_is_joined_and_coordinates_are_not_rewritten():
    for text in ['neural re-\ntriever', 'neural re- \r\n  triever']:
        assert 'retriever' in _literal_anchors(text)
        assert 'retriever' in _review_anchors(text)[0]
    # A blank paragraph cannot be normalized into a fabricated word.
    assert 'retriever' not in _literal_anchors('re-\n\ntriever')
    assert 'retriever' not in _review_anchors('re-\n\ntriever')[0]


def test_long_unterminated_pdf_wrap_has_no_quadratic_whitespace_backtracking():
    payload = 'a-' + '\n' * 11990 + '!'
    before = perf_counter()
    for _ in range(10):
        assert _literal_anchors(payload) == set()
        _review_anchors(payload)
    # Previously over 18s for these two bounded checks on the same machine;
    # use a generous wall limit, not a speed/latency product claim.
    assert perf_counter() - before < 3


def test_no_translation_stemming_or_uncited_claim_is_treated_as_proof():
    rows = [source('S1', 'A different design is evaluated.'),
            source('S2', 'The dense vector index of Wikipedia uses a neural retriever.')]
    for answer in ['使用稠密向量索引和神经检索器 [S1]。',
                   'The dense vector index of Wikipedia uses a neural retriever.']:
        result, issues = codes(answer, rows)
        assert 'citation_anchor_drift' not in issues
        assert result['entailment_verified'] is False


def test_unknown_labels_do_not_produce_a_replacement_citation_signal():
    rows = [source('S2', 'The dense vector index of Wikipedia uses a neural retriever.')]
    _, issues = codes('The dense vector index of Wikipedia uses a neural retriever [S99].', rows)
    assert 'source_unavailable' in issues
    assert 'citation_anchor_drift' not in issues


def test_diagnostic_is_bounded_to_ten_sources_and_claim_window():
    rows = [source('S' + str(i), 'Only ordinary material about evaluation.') for i in range(1, 11)]
    rows += [source('S11', 'Wikipedia dense vector neural retriever')]
    assert 'citation_anchor_drift' not in codes('Wikipedia dense vector neural retriever [S1].', rows)[1]
    _, issues = codes('x' * 810 + ' Wikipedia dense vector neural retriever [S1].', rows[:1] + rows[-1:])
    assert 'citation_anchor_drift' not in issues


def test_review_survives_hydration_without_importing_other_excerpt():
    rows = [source('S1', 'This study reports performance on a benchmark.'),
            source('S2', 'Wikipedia dense vector neural retriever')]
    result = audit_answer('Wikipedia dense vector neural retriever [S1].', rows)
    refreshed = hydrate_audit(result, rows[:1])
    assert refreshed['entailment_verified'] is False
    assert 'citation_anchor_drift' in {i['code'] for c in refreshed['claims'] for i in c['issues']}
    assert all(ref['label'] == 'S1' for c in refreshed['claims'] for ref in c['source_refs'])
