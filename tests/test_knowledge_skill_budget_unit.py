"""Knowledge is a bounded optional input, never a replacement for user facts."""
from copy import deepcopy
import json

from docthinker.server.knowledge_skills import retrieve_knowledge_skill
from docthinker.server.research_answer_contract import POLICY_VERSION, SYSTEM_POLICY, knowledge_skill_system_policy
from docthinker.server.research_prompt_budget import empty_assistance, fit_research_prompt
from docthinker.server.research_retrieval import build_retrieval_plan
from docthinker.server.research_knowledge import knowledge_receipt, unavailable_knowledge

QUESTION = '跨部门协作总是返工，帮我明确分工和沟通方式'


def setup_plan():
    return build_retrieval_plan(QUESTION, [
        {'id': 'user-fact', 'document_id': 'user-doc', 'filename': '工作记录.txt',
         'content': '跨部门协作返工：设计和销售未明确交付标准，沟通方式是群聊。'}])


def test_actual_method_input_is_separate_and_receipt_hashes_fitted_messages():
    plan = setup_plan()
    knowledge = retrieve_knowledge_skill('mao-workplace', QUESTION)
    original = deepcopy((plan, knowledge))
    fitted = fit_research_prompt(QUESTION, plan, [], empty_assistance(plan['sources']),
        system_policy=knowledge_skill_system_policy(True), knowledge_support=knowledge)
    baseline = fit_research_prompt(QUESTION, plan, [], empty_assistance(plan['sources']), system_policy=SYSTEM_POLICY)
    assert fitted['plan']['sources'] == baseline['plan']['sources'] == plan['sources']
    assert fitted['knowledge_skill_included']
    assert 0 < fitted['budget']['knowledge_skill_bytes'] <= 3500
    assert fitted['budget']['input_bytes'] <= 24576
    packet = json.loads(fitted['prompt'].split('KNOWLEDGE_SKILL_DATA_JSON:\n')[1])
    assert packet['kind'] == 'knowledge_skill_reference'
    assert packet['sources'][0]['label'] == 'K1'
    assert 'decision_rule' in packet['sources'][0]
    assert fitted['messages'][0]['content'] == knowledge_skill_system_policy(True)
    receipt = fitted['reasoning']['trace']['answer_contract']
    old_receipt = baseline['reasoning']['trace']['answer_contract']
    assert receipt['version'] == POLICY_VERSION + '+knowledge-skill-v1'
    assert receipt['evidence_packet_sha256'] == old_receipt['evidence_packet_sha256']
    assert receipt['messages_sha256'] != old_receipt['messages_sha256']
    assert (plan, knowledge) == original


def test_whole_method_block_and_policy_leave_before_source_or_history_budget():
    plan = setup_plan()
    assistance = empty_assistance(plan['sources'])
    history = [{'role': 'user', 'content': 'h' * 100}]
    first = fit_research_prompt(QUESTION, plan, history, assistance, system_policy=SYSTEM_POLICY)
    history[0]['content'] += 'h' * (24000 - first['budget']['input_bytes'])
    baseline = fit_research_prompt(QUESTION, plan, history, assistance, system_policy=SYSTEM_POLICY)
    assert baseline['budget']['input_bytes'] == 24000
    support = retrieve_knowledge_skill('mao-workplace', QUESTION)
    fitted = fit_research_prompt(QUESTION, plan, history, assistance,
        system_policy=knowledge_skill_system_policy(True), knowledge_support=support)
    assert fitted['knowledge_skill_included'] is False
    assert fitted['messages'] == baseline['messages']
    assert fitted['plan']['sources'] == baseline['plan']['sources']
    assert fitted['budget']['history_bytes'] == baseline['budget']['history_bytes']
    assert fitted['budget']['knowledge_skill_bytes'] == 0
    assert fitted['messages'][0]['content'] == SYSTEM_POLICY
    receipt = knowledge_receipt(support, included=False, answer='建议', model_called=True)
    assert receipt['status'] == 'budget_omitted' and receipt['sources'] == []


def test_literal_or_insufficient_evidence_answer_cannot_claim_method_usage():
    support = retrieve_knowledge_skill('mao-workplace', QUESTION)
    receipt = knowledge_receipt(support, included=True, answer='没有资料', model_called=False)
    assert receipt['status'] == 'not_applicable' and receipt['sources'] == []
    assert knowledge_receipt(None, included=False, answer='你好', model_called=True) is None


def test_receipt_does_not_claim_semantic_validation_and_revocation_removes_method_text():
    support = retrieve_knowledge_skill('mao-workplace', QUESTION)
    receipt = knowledge_receipt(support, included=True, answer='待验证建议 [k1]，未知 [K99]', model_called=True)
    assert receipt['citation_validation'] == {
        'unknown_labels': ['K99'], 'missing_citations': False, 'entailment_verified': False}
    assert receipt['authorship'] == 'project_original_method_design'
    assert receipt['validation_status'] == 'design_unvalidated'
    revoked = unavailable_knowledge(receipt)
    assert revoked['sources'] == [] and revoked['status'] == 'unavailable'
    assert receipt['sources']  # No mutation of the original versioned snapshot.


def test_generated_citation_noise_cannot_expand_saved_method_metadata_without_bound():
    support = retrieve_knowledge_skill('mao-workplace', QUESTION)
    answer = ' '.join(f'[K{n}]' for n in range(100, 1100))
    receipt = knowledge_receipt(support, included=True, answer=answer, model_called=True)
    assert len(receipt['citation_validation']['unknown_labels']) == 32
    assert receipt['citation_validation']['missing_citations'] is True
