"""Final input retention is literal/provenance checking, never semantic proof."""
from copy import deepcopy
import hashlib
import json

import pytest

from docthinker.server.research_need_ledger import (
    build_need_ledger, fallback_source_variants, final_need_ledger,
)
from docthinker.server.research_prompt_budget import fit_research_prompt, empty_assistance
from docthinker.server.research_retrieval import lexical_tokens, grounded_prompt


def source(identifier, text, label='S1', *, start=0):
    return {'id': identifier, 'document_id': 'doc-' + identifier, 'label': label,
        'filename': identifier + '.txt', 'content': text, 'ordinal': 0,
        'excerpt_start': start, 'excerpt_end': start + len(text),
        'truncated': start > 0, 'original_content_length': start + len(text)}


def plan(question, rows):
    return {'sources': rows, 'selected_chunks': rows, 'confirmed_memories': [],
        'gaps': [], 'query_terms': lexical_tokens(question), 'mode': 'evidence'}


def fit(question, value, policy='', **kwargs):
    return fit_research_prompt(question, value, [], empty_assistance(value['sources']),
        system_policy=policy, **kwargs)


@pytest.mark.parametrize('question,background,critical', [
    ('What is the approved deadline, exception and insurance?',
     'A deadline appears in the general agenda. ',
     'The approved deadline is 6 November. However, the exception needs client signoff; without signoff it is not approved.'),
    ('What is the current budget ceiling, exception and refund?',
     'A budget is useful in ordinary planning. ',
     'The current budget ceiling is 4500 yuan. However, the exception requires approval; it does not authorize any payment.'),
    ('When does output increase, when does it not, and what about humidity?',
     'Output is discussed in the introduction. ',
     'Output can increase only under cooling. It does not increase without cooling.'),
])
def test_tight_budget_keeps_nonfirst_whole_qualified_witness_and_discloses_missing_literal_need(question, background, critical):
    rows = [source('background', background + 'Ordinary background detail. ' * 55),
            source('critical', critical, 'S2', start=120)]
    value = plan(question, rows)
    original = deepcopy(value)
    # Manufacture a real 24,576-byte adapter budget allowing either whole
    # singleton, but not both. A feasible legacy prefix exists yet lacks detail.
    singleton_bytes = max(fit(question, plan(question, [row]))['budget']['input_bytes'] for row in rows)
    policy = 's' * (24576 - singleton_bytes - 24)
    legacy_prefix = fit(question, plan(question, rows[:1]), policy)
    assert legacy_prefix['plan']['sources'] == rows[:1]
    result = fit(question, value, policy)
    assert result['plan']['sources'] == rows[1:]
    assert value == original and result['budget']['input_bytes'] <= 24576
    assert critical in result['prompt']
    receipt = result['reasoning']['trace']['final_need_ledger']
    assert receipt['retained_need_count'] > 0 and receipt['unmatched_need_count'] > 0
    assert receipt['protected_unit_count'] > 0 and not receipt['semantic_verified']
    assert not receipt['whole_document_conditions_verified']
    assert any(row['code'] in {'uncovered_query_terms', 'final_need_witness_unmatched'} for row in result['plan']['gaps'])
    assert any(row['code'] == 'context_budget_omitted' for row in result['plan']['gaps'])
    bundle = next(row for row in result['need_ledger']['bundles'] if row['source_id'] == 'critical')
    assert bundle['retained_exactly'] and bundle['span'] == [120, 120 + len(critical)]
    assert bundle['sha256'] == hashlib.sha256(critical.encode()).hexdigest()


@pytest.mark.parametrize('mutation', ['content', 'span', 'source', 'document', 'label'])
def test_altered_geometry_never_keeps_old_literal_witness_credit(mutation):
    row = source('deadline', 'Deadline is Friday. However, only after approval.')
    ledger = build_need_ledger('deadline approval', [row])
    changed = deepcopy(row)
    if mutation == 'content':
        changed['content'] = changed['content'].replace('only ', 'also ')
    elif mutation == 'span':
        changed['excerpt_start'] += 1; changed['excerpt_end'] += 1
    elif mutation == 'source':
        changed['id'] = 'another-source'
    elif mutation == 'document':
        changed['document_id'] = 'another-document'
    else:
        changed['label'] = 'S2'
    packet = final_need_ledger(ledger, [changed])
    assert packet['receipt']['retained_need_count'] == 0
    assert packet['receipt']['omitted_need_count'] == 2
    assert packet['receipt']['invalid_retained_window_count'] == 1
    assert packet['receipt']['protected_unit_count'] == 0
    assert any(row['code'] == 'final_need_witness_omitted' for row in packet['gaps'])
    assert packet['ledger']['bundles'][0]['drop_reason'] in {'changed_or_partial_window', 'omitted_whole_window'}


def test_explicit_host_needs_are_not_reconstructed_from_source_content_or_raw_text_in_receipt():
    rows = [source('policy', 'Budget ceiling requires written approval. No automatic payment is authorized.')]
    required = [{'id': 'host-ceiling', 'text': 'budget ceiling'},
                {'id': 'host-approval', 'text': 'written approval'},
                {'id': 'host-unknown', 'text': 'refund amount'}]
    ledger = build_need_ledger('Plan the next action.', rows, required_needs=required)
    packet = final_need_ledger(ledger, rows)
    assert packet['receipt']['explicit_host_needs']
    assert packet['receipt']['need_count'] == 3 and packet['receipt']['retained_need_count'] == 2
    assert packet['receipt']['unmatched_need_count'] == 1
    encoded = json.dumps(packet, ensure_ascii=False)
    assert 'refund amount' not in encoded and rows[0]['content'] not in encoded


def test_greedy_ties_preserve_original_rank_and_no_quote_is_ever_cut():
    rows = [source('first', 'Energy imports decrease.'), source('second', 'Energy imports increase.', 'S2')]
    ledger = build_need_ledger('Energy imports', rows)
    variants = list(fallback_source_variants(ledger, rows))
    assert variants == [rows[:1], []]
    assert final_need_ledger(ledger, rows)['receipt']['retained_window_count'] == 2


def test_dropping_the_only_counterevidence_reports_omitted_witness_not_complete_coverage():
    positive = source('claim', 'Output increases under cooling.')
    negative = source('limit', 'Output does not increase without cooling.', 'S2')
    ledger = build_need_ledger('output not without cooling', [positive, negative])
    packet = final_need_ledger(ledger, [positive])
    assert packet['receipt']['omitted_need_count'] >= 1
    assert any(row['status'] == 'literal_witness_omitted' for row in packet['ledger']['needs'])
    assert not packet['receipt']['semantic_verified']


def test_nothing_fits_discloses_all_previously_witnessed_needs_as_omitted():
    rows = [source('approval', 'Approval is required. Do not execute without approval.')]
    ledger = build_need_ledger('approval', rows)
    packet = final_need_ledger(ledger, [])
    assert packet['receipt']['omitted_need_count'] == 1
    assert packet['receipt']['retained_need_count'] == 0
    assert packet['gaps'][0]['code'] == 'final_need_witness_omitted'
    assert packet['ledger']['bundles'][0]['drop_reason'] == 'omitted_whole_window'


def test_regular_plural_matches_reuse_existing_word_form_rule_not_semantic_aliases():
    rows = [source('policy', 'Budgets have ceilings and approvals.')]
    packet = final_need_ledger(build_need_ledger('budget ceiling approval salary', rows), rows)
    assert packet['receipt']['retained_need_count'] == 3
    assert packet['receipt']['unmatched_need_count'] == 1
    assert not packet['receipt']['semantic_verified']


def test_fitter_refreshes_old_need_warnings_against_current_complete_geometry():
    rows = [source('policy', 'Budget approval is needed.')]
    value = plan('budget approval', rows)
    value['gaps'] = [{'code': 'final_need_witness_omitted', 'count': 99, 'message': 'old packet'}]
    result = fit('budget approval', value)
    assert result['reasoning']['trace']['final_need_ledger']['omitted_need_count'] == 0
    assert not any(row['code'] == 'final_need_witness_omitted' for row in result['plan']['gaps'])


@pytest.mark.parametrize('offset_unit', ['characters_in_source_chunk', 'invalid-internal-coordinate'])
def test_new_literal_diagnostics_do_not_change_normal_wire_or_evict_quotes(offset_unit):
    row = source('policy', 'Budget approval is required. No automatic payment is authorized.')
    row['offset_unit'] = offset_unit
    question = 'Explain the budget and the insurance refund.'
    # Legacy/manual callers need not supply query_terms. New unmatched or
    # invalid-window bookkeeping must not inject an extra model warning.
    value = {'sources': [row], 'selected_chunks': [row], 'confirmed_memories': [],
             'gaps': [], 'mode': 'evidence'}
    expected = grounded_prompt(question, value)
    result = fit(question, value)
    assert result['prompt'] == expected
    assert result['messages'][1]['content'] == expected
    assert result['plan']['sources'] == [row]
    assert result['budget']['input_bytes'] == len(expected.encode()) + len('systemuser')
    assert result['reasoning']['trace']['final_need_ledger']['unmatched_need_count'] > 0
    assert any(gap['code'] == 'final_need_witness_unmatched' for gap in result['plan']['gaps'])
    if offset_unit != 'characters_in_source_chunk':
        assert result['need_ledger']['invalid_input_window_count'] == 1
    assert 'final_need_witness_unmatched' not in result['prompt']


@pytest.mark.parametrize('question', ['', 'x' * 9000, '\ud800', None])
def test_ledger_does_not_change_fitter_invalid_question_failure_boundary(question):
    with pytest.raises(ValueError, match='^Research prompt exceeds its budget$'):
        fit(question, plan('budget', [source('policy', 'Budget approval is required.')]))


@pytest.mark.parametrize('bad', [[{'id': 'a', 'text': ''}],
    [{'id': 'a', 'text': 'budget'}, {'id': 'a', 'text': 'deadline'}], [{'id': 'a', 'text': 'budget', 'trusted': True}]])
def test_malformed_host_contracts_are_rejected_not_silently_truncated(bad):
    with pytest.raises(ValueError):
        build_need_ledger('question', [], required_needs=bad)
