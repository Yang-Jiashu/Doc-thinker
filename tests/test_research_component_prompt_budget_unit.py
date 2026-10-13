"""Final-fit component attention aids cannot displace original evidence."""
from copy import deepcopy
import hashlib
import json

from docthinker.server.research_answer_contract import DATA_MARKER, source_quote_payload, SYSTEM_POLICY
from docthinker.server.research_prompt_budget import empty_assistance, fit_research_prompt
from docthinker.server.research_retrieval import build_retrieval_plan, grounded_prompt


QUESTION = 'Explain the architecture and component functions of the amber queue and cobalt cache.'


def definition_plan():
    return build_retrieval_plan(QUESTION, [
        {'id': 'first', 'document_id': 'first-doc', 'filename': 'Birch.txt',
         'content': 'The amber queue stores immutable events only after consent.'},
        {'id': 'second', 'document_id': 'second-doc', 'filename': 'Elm.txt',
         'content': 'The cobalt cache does not return expired entries.'},
    ])


def packet(result):
    return json.JSONDecoder().raw_decode(result['prompt'].split(DATA_MARKER, 1)[1])[0]


def evidence_hash(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True,
                                   separators=(',', ':')).encode()).hexdigest()


def test_fitted_locators_point_to_exact_final_evidence_and_receipt_same_projection():
    plan = definition_plan()
    original = deepcopy(plan)
    result = fit_research_prompt(QUESTION, plan, [], empty_assistance(plan['sources']),
                                 system_policy=SYSTEM_POLICY)
    quotes = packet(result)['source_quotes']
    hints = [(source, hint) for source in quotes for hint in source.get('component_definition_locators', [])]
    assert len(hints) == 2
    for source, hint in hints:
        assert hint['source_label'] == source['label']
        assert source['content'][hint['name_start']:hint['name_end']] == hint['name']
        assert 0 <= hint['definition_start'] <= hint['name_start'] < hint['name_end'] <= hint['definition_end'] <= len(source['content'])
    receipt = result['reasoning']['trace']['answer_contract']
    assert receipt['evidence_packet_sha256'] == evidence_hash(quotes)
    assert receipt['semantic_support_verified'] is False
    assert receipt['extra_provider_calls'] == 0
    assert result['plan']['sources'] == original['sources']
    assert plan == original
    assert result['budget']['input_bytes'] == sum(len((row['role'] + row['content']).encode()) for row in result['messages'])


def test_locators_leave_before_whole_evidence_when_input_budget_is_tight():
    plan = definition_plan()
    without = grounded_prompt(QUESTION, plan, include_component_locators=False)
    with_hints = grounded_prompt(QUESTION, plan)
    assert len(with_hints.encode()) > len(without.encode())
    policy = 's' * (24576 - len(without.encode()) - len('systemuser') - 2)
    result = fit_research_prompt(QUESTION, plan, [], empty_assistance(plan['sources']),
                                 system_policy=policy)
    quotes = packet(result)['source_quotes']
    assert result['plan']['sources'] == plan['sources']
    assert len(quotes) == 2
    assert all('component_definition_locators' not in source for source in quotes)
    assert result['budget']['input_bytes'] <= 24576
    assert result['reasoning']['trace']['answer_contract']['evidence_packet_sha256'] == evidence_hash(quotes)


def test_final_source_subset_has_no_locator_or_quote_for_omitted_source():
    plan = definition_plan()
    plain_one = grounded_prompt(QUESTION, {**plan, 'selected_chunks': plan['sources'][:1]},
                                include_component_locators=False)
    # Leave space for the omission receipt, but not the second complete quote.
    policy = 's' * (24576 - len(plain_one.encode()) - 190)
    result = fit_research_prompt(QUESTION, plan, [], empty_assistance(plan['sources']),
                                 system_policy=policy)
    quotes = packet(result)['source_quotes']
    assert len(quotes) == len(result['plan']['sources']) == 1
    kept = quotes[0]['label']
    assert all(hint['source_label'] == kept for hint in quotes[0].get('component_definition_locators', []))
    assert quotes == source_quote_payload(result['plan']['sources'], question=QUESTION,
        include_component_locators=bool(quotes[0].get('component_definition_locators')))
    assert any(row['code'] == 'context_budget_omitted' for row in result['plan']['gaps'])


def test_numeric_question_does_not_add_structural_obligations_or_locators():
    plan = definition_plan()
    question = 'Compare only exact-match accuracy numbers, not architecture.'
    result = fit_research_prompt(question, plan, [], empty_assistance(plan['sources']),
                                 system_policy=SYSTEM_POLICY)
    assert all('component_definition_locators' not in row for row in packet(result)['source_quotes'])
    assert result['plan']['sources'] == plan['sources']
