"""Excerpt scope safety is structural, not semantic verification; no model calls."""
from copy import deepcopy
import json

import pytest

from docthinker.server.research_prompt_budget import empty_assistance, fit_research_prompt
from docthinker.server.research_retrieval import _excerpt, build_retrieval_plan, refresh_query_coverage
from docthinker.server.research_sentence_windows import opening_reference_unverified
from docthinker.server.research_service import _supplement_insight_evidence


def source(identifier, text, ordinal=0):
    return {'id': identifier, 'document_id': 'study', 'filename': 'study.txt',
            'content': text, 'ordinal': ordinal, 'page': 4}


def test_one_quote_cannot_mix_previous_task_continuation_with_new_heading_even_below_budget():
    text = ('We report results on both test sets to permit comparison. '
            'The revised split performs better only under fixed controls.\n'
            'E Further Details on Saffron\n'
            'For Saffron classification, we regenerate each claim and predict its label. '
            'The label accuracy does not evaluate retrieval correctness.')
    row = source('boundary', text)
    original = deepcopy(row)
    for terms in ({'comparison', 'classification'}, {'comparison'}, {'saffron', 'label'}):
        chosen = _excerpt(row, terms, {}, 1000)
        assert chosen is not None
        assert not ('both test sets' in chosen['content'] and 'Saffron' in chosen['content'])
        assert text[chosen['excerpt_start']:chosen['excerpt_end']] == chosen['content']
        assert chosen['sentence_boundary_preserved'] is True
        assert chosen['truncated'] is True
    assert row == original


def test_result_heading_does_not_share_citation_with_previous_task_method():
    text = ('The Amber task reports classification accuracy without evidence supervision.\n'
            '4 Results\n4.1 Question Answering\n'
            'On all four question-answering datasets, we observe the best score only on the comparable split.')
    chosen = _excerpt(source('result', text), {'accuracy', 'score'}, {}, 800)
    assert not ('Amber' in chosen['content'] and 'all four' in chosen['content'])
    assert text[chosen['excerpt_start']:chosen['excerpt_end']] == chosen['content']


@pytest.mark.parametrize('text', [
    'We report results on both test sets to enable fair comparison to both approaches.',
    'This dataset contains only manually checked examples.',
    'It then loads the stored records.',
    'To overcome this, we retrieve the most frequent match.',
    '这些数据集只包含经过人工核验的样本。',
])
def test_cross_chunk_opening_reference_is_warning_not_sentence_boundary_failure(text):
    assert opening_reference_unverified(text)
    row = _excerpt(source('continuation', text), set(), {}, 1000)
    assert row['content'] == text and row['sentence_boundary_preserved'] is True
    plan = refresh_query_coverage({'query_terms': [], 'sources': [row], 'gaps': []}, [row])
    assert [gap['code'] for gap in plan['gaps']] == ['source_antecedent_unverified']


@pytest.mark.parametrize('text', [
    'It is important to preserve user consent before accessing records.',
    'This paper compares Birch and Elm under fixed conditions.',
    'We compare Birch and Elm. Both datasets contain 100 records.',
    'We report results on both datasets: Birch and Elm under fixed conditions.',
    'This dataset (Birch) contains 100 records.',
])
def test_locally_identified_objects_and_expletive_it_do_not_get_dangling_warning(text):
    assert not opening_reference_unverified(text)


def test_heading_detection_does_not_split_quantities_soft_wraps_or_figure_caption():
    for text in ('The experiment required\n12 hours of processing, but only under the tested conditions.',
                 'The system uses\nMemory Management\nfor its internal state only.',
                 'Figure 2. Processing starts after consent. The procedure does not remove source files.'):
        row = _excerpt(source('literal', text), set(), {}, 1000)
        assert row['content'] == text
        assert row['excerpt_start'] == 0 and row['excerpt_end'] == len(text)


def test_oversized_condition_is_not_trimmed_to_make_room_for_next_task():
    text = ('The Amber method succeeds. However, this holds only if ' + 'every permission is checked ' * 25
            + 'and no record is expired.\n2 Results\nThe Cobalt task reports no improvement.')
    chosen = _excerpt(source('long', text), {'amber', 'permission'}, {}, 120)
    assert chosen is None


def test_supplement_and_final_fit_share_scope_checks_without_new_calls_or_wider_budget():
    question = 'Compare cobalt retrieval conditions and classification results.'
    baseline = build_retrieval_plan(question, [source('base', 'Cobalt retrieval requires verified consent.')])
    detail = source('detail', 'We report results on both test sets under fixed conditions. '
                    'The test split was easier to answer.\nE Classification Details\n'
                    'For Amber classification, we regenerate the claim and predict its label only after validation.', 1)
    proposal = {'actions': [{'id': 'q1', 'kind': 'supplemental_query',
                            'payload': {'dimension_terms': ['conditions', 'classification', 'results']}}]}
    supplement = _supplement_insight_evidence(question, baseline['sources'], [detail], proposal)
    assert supplement['receipt']['model_calls'] == 0
    assert supplement['receipt']['retrieval_passes'] == 1
    assert len(supplement['sources']) <= 3
    added = supplement['sources'][1:]
    assert added
    assert all(not ('both test sets' in row['content'] and 'Amber' in row['content']) for row in added)
    assert sum(len(row['content'].encode()) for row in supplement['sources']) <= 12000
    # A continuation arriving after initial coverage must be checked by final
    # fitting; never inherit an earlier empty gap list as a safety certificate.
    continuation = _excerpt(source('late', 'We report results on both test sets under fixed conditions.', 2),
                            set(), {}, 800)
    continuation['label'] = 'S2'
    final_sources = [*baseline['sources'], continuation]
    plan = {**baseline, 'sources': final_sources, 'selected_chunks': final_sources, 'gaps': []}
    fitted = fit_research_prompt(question, plan, [], empty_assistance(final_sources), system_policy='')
    assert any(gap['code'] == 'source_antecedent_unverified' for gap in fitted['plan']['gaps'])
    payload, _ = json.JSONDecoder().raw_decode(fitted['prompt'].split('UNTRUSTED_RESEARCH_DATA_JSON:\n')[1])
    assert any(gap['code'] == 'source_antecedent_unverified' for gap in payload['retrieval_gaps'])
    assert fitted['budget']['input_bytes'] <= 24576
    refreshed = refresh_query_coverage(fitted['plan'], baseline['sources'])
    assert not any(gap['code'] == 'source_antecedent_unverified' for gap in refreshed['gaps'])
