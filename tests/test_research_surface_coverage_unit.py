"""Offline word-form diagnostics: no semantic assertions, ranking or API calls."""
from copy import deepcopy
import json

import pytest

from docthinker.server.research_prompt_budget import empty_assistance, fit_research_prompt
from docthinker.server.research_retrieval import (
    build_retrieval_plan, grounded_prompt, lexical_tokens, refresh_query_coverage,
)
from docthinker.server.research_surface_coverage import surface_term_coverage


def source(text, label='S1'):
    return {'id': label, 'document_id': 'd' + label, 'label': label,
            'filename': label + '.txt', 'content': text, 'ordinal': 0, 'page': 2,
            'excerpt_start': 0, 'excerpt_end': len(text), 'truncated': False,
            'original_content_length': len(text), 'offset_unit': 'characters_in_source_chunk'}


def terms_gap(value):
    return next((row for row in value['gaps'] if row['code'] == 'uncovered_query_terms'), None)


@pytest.mark.parametrize('query,text,matched', [
    ('assistant', 'personalized assistants', True),
    ('assistants', 'a personalized assistant', True),
    ('memory', 'memories', True),
    ('memories', 'memory', True),
    ('classes', 'class', True),
    ('status', 'statuses', True),
    ('day', 'days', False),  # Short potentially ambiguous words remain exact-only.
    ('new', 'news', False),
    ('axis', 'axes', False),  # No guess for irregular morphology.
    ('flow', 'flower', False),
    ('private', 'privately', False),
    ('retrieve', 'retrieval', False),  # Diagnostic != broader retrieval matcher.
    ('assistant', 'assistance', False),
    ('rag-1', 'rag-1s', False),
    ('助手', '助手们', True),  # Existing Han bigram token occurs exactly.
    ('助手', '助理', False),
    ('memory', '记忆', False),
])
def test_only_narrow_surface_forms_count_as_a_match(query, text, matched):
    result = surface_term_coverage(lexical_tokens(query), lexical_tokens(text))
    assert result['covered'] == ([query] if matched else [])
    assert result['uncovered'] == ([] if matched else [query])


def test_original_query_terms_remain_visible_without_stems_or_invented_terms():
    assert surface_term_coverage(['Assistant', 'privately', 'memories', 'Assistant'],
                                 ['assistants', 'memory', 'private']) == {
        'covered': ['Assistant', 'memories'], 'uncovered': ['privately']}


def test_chinese_bigrams_and_negative_prose_are_not_semantic_verification():
    question = '长期记忆助手'
    text = '长期记忆不适用于助手。'
    coverage = surface_term_coverage(lexical_tokens(question), lexical_tokens(text))
    assert '助手' in coverage['covered'] and '长期' in coverage['covered']
    # The cross-word bigram in the query is not guessed from nonadjacent words.
    assert '忆助' in coverage['uncovered']
    assert surface_term_coverage(['assistant'], lexical_tokens('No assistants were evaluated.')) == {
        'covered': ['assistant'], 'uncovered': []}


def test_initial_plan_plural_diagnostic_preserves_quotes_offsets_and_raw_terms():
    text = 'Personalized assistants use memory only in the tested setting.'
    plan = build_retrieval_plan('assistant memory private', [source(text)])
    assert plan['query_terms'] == ['assistant', 'memory', 'private']
    assert plan['covered_query_terms'] == ['assistant', 'memory']
    assert terms_gap(plan)['terms'] == ['private']
    assert '不表示概念缺失' in terms_gap(plan)['message']
    assert plan['sources'][0]['content'] == text
    assert plan['sources'][0]['excerpt_start'] == 0
    assert plan['sources'][0]['excerpt_end'] == len(text)
    assert plan['algorithm']['semantic_entailment_verified'] is False


def test_refresh_uses_only_retained_content_not_filenames_removed_context_or_old_gap():
    row = source('Personalized assistants use memory.')
    row['filename'] = 'private.txt'
    plan = {'sources': [row], 'query_terms': ['assistant', 'private'],
            'gaps': [{'code': 'uncovered_query_terms', 'terms': ['assistant', 'obsolete']},
                     {'code': 'source_boundary_unverified', 'message': 'keep this warning'}]}
    before = deepcopy(plan)
    refreshed = refresh_query_coverage(plan, [row])
    assert terms_gap(refreshed)['terms'] == ['private']
    assert refreshed['covered_query_terms'] == ['assistant']
    assert refreshed['gaps'][0] == before['gaps'][1]
    assert refresh_query_coverage(refreshed, [row]) == refreshed
    assert plan == before
    assert terms_gap(refresh_query_coverage(plan, []))['terms'] == ['assistant', 'private']


def test_bounded_overview_does_not_become_a_term_coverage_assertion():
    plan = {'query_terms': ['missing'], 'selection_strategy': 'document_overview',
            'gaps': [{'code': 'bounded_overview'}, {'code': 'uncovered_query_terms', 'terms': ['old']} ]}
    refreshed = refresh_query_coverage(plan, [source('An opening excerpt.')])
    assert refreshed['gaps'] == [{'code': 'bounded_overview'}]
    assert refreshed['covered_query_terms'] == []


def test_budget_drops_whole_supporting_excerpt_and_recomputes_final_missing_terms():
    first = source('Memory limits remain conditional. ' * 75)
    second = source('Personalized assistants require controlled evaluation. ' * 50, 'S2')
    plan = {'query_terms': ['assistant', 'memory'], 'sources': [first, second],
            'selected_chunks': [first, second], 'confirmed_memories': [], 'gaps': []}
    plan = refresh_query_coverage(plan, plan['sources'])
    one = refresh_query_coverage({**plan, 'selected_chunks': [first]}, [first])
    policy = 's' * (24576 - len(grounded_prompt('assistant memory', one).encode()) - 500)
    result = fit_research_prompt('assistant memory', plan, [], empty_assistance(plan['sources']), system_policy=policy)
    assert result['plan']['sources'] == [first]
    assert result['plan']['covered_query_terms'] == ['memory']
    assert terms_gap(result['plan'])['terms'] == ['assistant']
    assert not any('terms' in row for row in result['reasoning']['gaps'])
    payload, _ = json.JSONDecoder().raw_decode(result['prompt'].split('UNTRUSTED_RESEARCH_DATA_JSON:\n')[1])
    assert next(row for row in payload['retrieval_gaps']
                if row['code'] == 'uncovered_query_terms')['terms'] == ['assistant']
    assert payload['source_quotes'][0]['content'] == first['content']
    assert result['budget']['input_bytes'] <= 24576
    assert terms_gap(plan) is None  # Formatting never mutates upstream state.


def test_replaced_excerpt_with_same_id_cannot_reuse_old_coverage():
    row = source('Memory is bounded.')
    plan = {'query_terms': ['assistant', 'memory'], 'sources': [row], 'selected_chunks': [row],
            'confirmed_memories': [], 'gaps': [{'code': 'uncovered_query_terms', 'terms': ['assistant']}]}
    actual = source('Personalized assistants use memory.')
    support = empty_assistance([actual])
    support['reasoning']['gaps'] = deepcopy(plan['gaps'])
    result = fit_research_prompt('assistant memory', plan, [], support, system_policy='Use source excerpts.')
    assert result['plan']['sources'] == [actual]
    assert result['plan']['covered_query_terms'] == ['assistant', 'memory']
    assert terms_gap(result['plan']) is None
    assert result['reasoning']['gaps'] == []


def test_existing_reasoning_gap_is_refreshed_without_persisting_raw_query_terms():
    row = source('Personalized assistants use memory.')
    plan = {'query_terms': ['assistant', 'private'], 'sources': [row], 'selected_chunks': [row],
            'confirmed_memories': [], 'gaps': [{'code': 'uncovered_query_terms', 'terms': ['assistant']}]}
    support = empty_assistance([row])
    support['reasoning']['gaps'] = deepcopy(plan['gaps'])
    result = fit_research_prompt('assistant private', plan, [], support, system_policy='Use source excerpts.')
    assert terms_gap(result['plan'])['terms'] == ['private']
    assert result['reasoning']['gaps'] == [{
        'code': 'uncovered_query_terms', 'message': terms_gap(result['plan'])['message']}]
