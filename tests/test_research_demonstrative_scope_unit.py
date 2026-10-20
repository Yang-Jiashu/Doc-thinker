"""Demonstrative source references bind literal adjacent context, not facts."""
import pytest

from docthinker.server.research_sentence_windows import evidence_units, opening_reference_unverified
from docthinker.server.research_evidence_selection import prepare_evidence_candidates
from docthinker.server.research_query_facets import build_query_facets
from docthinker.server.research_retrieval import refresh_query_coverage


@pytest.mark.parametrize('first,second', [
    ('We enrolled 31 independently sampled participants.', 'These participants completed the same questionnaire.'),
    ('The procedure produces two biodegradable materials.', 'Those materials remain stable only under dry conditions.'),
    ('The index stores balanced binary trees.', 'These trees are queried without changing their original nodes.'),
    ('The store keeps only authorized records.', 'Such records are never shared outside the workspace.'),
])
def test_unfamiliar_demonstrative_subjects_keep_the_adjacent_antecedent(first, second):
    text = first + ' ' + second
    spans = evidence_units(text)
    assert spans == [(0, len(text))]
    assert text[spans[0][0]:spans[0][1]] == text
    assert opening_reference_unverified(second)


@pytest.mark.parametrize('text', [
    'These are the results reported in the table.',
    'This is a direct definition of an archive.',
    'That was a planned intervention.',
    'Such as the preceding example.',
    'This paper describes a bounded archive.',
    'It is important to retain the original conditions.',
    'These materials Amber and Birch were tested separately.',
])
def test_copular_display_local_document_and_locally_named_objects_are_not_dangling_nominals(text):
    assert not opening_reference_unverified(text)


@pytest.mark.parametrize('boundary', ['\n3 Results\n', '\nFigure 4: Controlled evaluation.\n'])
def test_binding_does_not_cross_a_new_heading_or_caption(boundary):
    first = 'We enrolled 31 participants.'
    second = 'These participants completed the questionnaire.'
    text = first + boundary + second
    spans = evidence_units(text)
    containing = [text[a:b] for a, b in spans if second in text[a:b]]
    assert len(containing) == 1 and first not in containing[0]
    assert 'Figure 4:' not in containing[0] and '3 Results' not in containing[0]


def test_source_start_without_antecedent_is_preserved_and_warned_not_resolved():
    text = 'These participants completed the same questionnaire, but no control group was used.'
    source = {'id': 'one', 'document_id': 'one', 'filename': 'ArchiveStudy.pdf', 'content': text,
              'label': 'S1', 'ordinal': 0, 'excerpt_start': 0, 'excerpt_end': len(text)}
    plan = refresh_query_coverage({'gaps': [], 'query_terms': []}, [source])
    assert any(gap['code'] == 'source_antecedent_unverified' for gap in plan['gaps'])
    assert source['content'] == text


def test_bound_reference_larger_than_window_is_not_cut_into_a_fake_standalone_result():
    first = 'The preparation used ' + 'carefully documented materials ' * 55 + '.'
    second = ' These materials were evaluated only in dry conditions; no wet-condition result was measured.'
    text = '2 Experiments\n' + first + second
    row = {'id': 'one', 'document_id': 'one', 'filename': 'QuartzCache_2502.12345.pdf',
           'content': text, 'ordinal': 0, 'page': 1}
    question = '请说明 QuartzCache 的实验。'
    pool = prepare_evidence_candidates(question, [row], [], build_query_facets(question, [row]), {})
    assert not pool['candidates']
    assert any(text[a:b].strip() == (first + second).strip() for a, b in evidence_units(text))
