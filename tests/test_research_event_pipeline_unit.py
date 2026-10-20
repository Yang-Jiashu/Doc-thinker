"""Real service/SQLite/prompt/lineage integration; model outputs are controlled.

These tests prove wiring and revocation, not model semantic accuracy.
"""
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from docthinker.server.personal_events import PersonalEvents
from docthinker.server.research_service import ResearchService, WITHDRAWN_ANALYSIS
from docthinker.server.research_store import ResearchNotFoundError
from docthinker.server.research_retrieval import build_retrieval_plan
from docthinker.server.research_prompt_budget import empty_assistance, fit_research_prompt
from docthinker.server.research_event_reasoning import EVENT_RESPONSE_POLICY
from docthinker.server.research_task_review import build_task_requirements

with patch('dotenv.load_dotenv', return_value=False):
    from docthinker.session_manager import SessionManager


def req(sid, question='What caused the Orion delivery delay?', **options):
    return SimpleNamespace(session_id=sid, question=question, use_memory=options.get('use_memory', False),
        use_conversation_context=options.get('history', False), use_llm_cache=False,
        use_self_evolution=False, knowledge_skill_mode='off', insight_mode='off',
        reasoning_mode=options.get('mode', 'auto'))


def packet(prompt, marker):
    return json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1].lstrip())[0]


@pytest.fixture
def setup(tmp_path):
    manager = SessionManager(str(tmp_path / '_system'), str(tmp_path), migrate_legacy=False)
    sid = manager.create_session('Orion') ['id']
    service = ResearchService(tmp_path, semantic_retriever=None, evidence_reranker=None, query_translator=None)
    return service, manager, sid


def material(service, sid, domain='delivery'):
    facts = [f'Orion {domain} scope expanded on day 10.',
             f'The Orion {domain} scope change added 30 hours of work on the critical path.',
             f'Orion {domain} finished three days late because the added work delayed the critical path. '
             'The team had no absences during this period.']
    doc = service.store.accept_document(sid, 'timeline.txt', '\n'.join(facts).encode())
    service.store.claim_document(sid, doc['id'])
    service.store.complete_document(sid, doc['id'], [{'text': text} for text in facts])
    chunks = service.all_chunks(sid)
    graph = PersonalEvents(service.store)
    rows = []
    for index, (fact, chunk) in enumerate(zip(facts, chunks)):
        rows.append(graph.record_event(sid, event_type='observation', action=fact,
            occurred_from=10 + index, occurred_until=10 + index,
            origins=[{'kind': 'source', 'source_id': chunk['id'], 'span': [0, len(fact)],
                      'sha256': hashlib.sha256(fact.encode()).hexdigest()}]))
    graph.relate(sid, rows[0]['id'], rows[1]['id'], 'reported_cause')
    graph.relate(sid, rows[1]['id'], rows[2]['id'], 'reported_cause')
    return doc, rows, chunks


class EnvelopeModel:
    def __init__(self, callback=None):
        self.prompts, self.packets, self.callback = [], [], callback

    async def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        context = packet(prompt, 'EVENT_REASONING_DATA_JSON:\n')
        self.packets.append(context)
        if self.callback:
            self.callback()
        return json.dumps({'answer': 'The records report a scope change and added critical-path work [S1]. '
                            'This is not proof of a unique cause; alternatives remain to be checked.',
                           'event_assessment': {'links': [], 'alternatives': [], 'missing_links': []}})


@pytest.mark.asyncio
@pytest.mark.parametrize('domain', ['delivery', 'family move', 'language study'])
async def test_events_reach_actual_single_call_and_gaps_are_not_completion(setup, domain):
    service, manager, sid = setup
    _, events, _ = material(service, sid, domain)
    model = EnvelopeModel()
    try:
        result = await service.answer(req(sid, f'Why was the Orion {domain} delayed?'), model, manager)
        assert len(model.prompts) == 1
        assert result['used_event_count'] >= 2
        assert result['used_event_relation_count'] >= 1
        assert result['event_reasoning']['unknown'] > 0
        assert not result['event_reasoning']['causality_verified']
        assert result['event_reasoning']['suggested_queries']
        assert result['reasoning']['trace']['answer_model_calls'] == 1
        assert result['reasoning']['trace']['context_budget']['input_bytes'] <= 24576
        assert {row['id'] for row in model.packets[0]['events']} <= {row['id'] for row in events}
        hydrated = service.hydrate_history(sid, manager.get_history(sid))[-1]
        assert hydrated['event_reasoning'] == result['event_reasoning']
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_event_revision_revokes_answer_history_and_sync_task_export(setup):
    service, manager, sid = setup
    doc, events, _ = material(service, sid)
    try:
        result = await service.answer(req(sid), EnvelopeModel(), manager)
        step = {'input_kind': 'documents', 'document_ids': [doc['id']],
                'input_snapshot': service.store.document_snapshots(sid, [doc['id']]),
                'context_snapshot': {}, 'answer_id': result['answer_id']}
        assert service.task_inputs_live(sid, [step])
        used = next(row for row in events if row['id'] in {
            item['id'] for item in service.events.retrieve_context(sid, req(sid).question)['events']})
        service.events.revise_event(sid, used['id'], expected_revision=1,
            event_type='observation', action='Corrected: Orion timing is not yet confirmed.', origins=used['origins'])
        assert not service.task_inputs_live(sid, [step])
        history = service.hydrate_history(sid, manager.get_history(sid))
        assert history[-1]['content'] == WITHDRAWN_ANALYSIS
        assert 'event_reasoning' not in history[-1]
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_event_change_during_generation_never_publishes_model_body(setup):
    service, manager, sid = setup
    _, events, _ = material(service, sid)
    def revise():
        for row in events:
            service.events.revise_event(sid, row['id'], expected_revision=1, event_type='observation',
                action='Corrected Orion event.', origins=row['origins'])
    try:
        model = EnvelopeModel(revise)
        result = await service.answer(req(sid), model, manager)
        assert len(model.prompts) == 1
        assert result['answer'] == WITHDRAWN_ANALYSIS
        assert 'event_reasoning' not in result
        assert result['citation_validation']['valid'] is False
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_scoped_task_cannot_borrow_other_document_events(setup):
    service, manager, sid = setup
    material(service, sid)
    doc = service.store.accept_document(sid, 'independent.txt', b'Orion schedule is unknown.')
    service.store.claim_document(sid, doc['id'])
    service.store.complete_document(sid, doc['id'], [{'text': 'Orion schedule is unknown.'}])
    prompts = []
    async def model(prompt, **kwargs):
        prompts.append(prompt)
        return 'The Orion schedule is unknown [S1].'
    try:
        await service.answer(req(sid), model, manager, document_ids=[doc['id']],
            project_instance_id=manager.get_session(sid)['instance_id'], project_context='Check only the selected schedule.')
        assert 'EVENT_REASONING_DATA_JSON' not in prompts[0]
        assert '30 hours' not in prompts[0]
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize('withdraw_during_generation', [False, True])
async def test_assessment_source_not_in_event_origins_still_revokes_receipt(setup, withdraw_during_generation):
    service, manager, sid = setup
    material(service, sid)
    text = 'Independent Orion workload records link 30 extra critical-path hours to the delivery delay.'
    extra = service.store.accept_document(sid, 'independent-records.txt', text.encode())
    service.store.claim_document(sid, extra['id'])
    service.store.complete_document(sid, extra['id'], [{'text': text}])
    calls = []
    async def model(prompt, **kwargs):
        calls.append(prompt)
        context = packet(prompt, 'EVENT_REASONING_DATA_JSON:\n')
        quotes = packet(prompt, 'UNTRUSTED_RESEARCH_DATA_JSON:\n')['source_quotes']
        source = next(row for row in quotes if row['filename'] == 'independent-records.txt')
        link = next(row for row in context['needs'] if row['kind'] == 'relation')
        if withdraw_during_generation:
            service.store.delete_document(sid, extra['id'])
        return json.dumps({'answer': f"The records report added critical-path work [{source['label']}].",
            'event_assessment': {'links': [{'obligation_id': link['id'], 'from_event': link['source_id'],
                'to_event': link['target_id'], 'verdict': 'support',
                'evidence': [{'label': source['label'], 'quote': source['content']}]}]}})
    try:
        result = await service.answer(req(sid), model, manager)
        if not withdraw_during_generation:
            assert result['event_reasoning']['supported'] >= 1
            service.store.delete_document(sid, extra['id'])
        else:
            assert result['answer'] == WITHDRAWN_ANALYSIS
            assert 'event_reasoning' not in result
            assert 'event_reasoning' not in result.get('reasoning', {}).get('trace', {})
        # The graph's own source revisions did not change. The independent
        # assessment witness is nonetheless a required source of THIS answer.
        assert service.event_dependencies.states(sid, [result['answer_id']])[result['answer_id']]
        hydrated = service.hydrate_history(sid, manager.get_history(sid))[-1]
        assert hydrated['content'] == WITHDRAWN_ANALYSIS
        assert 'event_reasoning' not in hydrated
        assert 'event_reasoning' not in hydrated.get('reasoning', {}).get('trace', {})
        assert len(calls) == 1
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_literal_and_operator_off_do_not_silently_add_calls_or_events(setup):
    service, manager, sid = setup
    material(service, sid)
    prompts = []
    async def model(prompt, **kwargs):
        prompts.append(prompt)
        return 'The report mentions added work [S1].'
    try:
        service.context_assistance_enabled = False
        result = await service.answer(req(sid), model, manager)
        assert len(prompts) == 1 and 'EVENT_REASONING_DATA_JSON' not in prompts[0]
        assert 'event_reasoning' not in result
        literal = await service.answer(req(sid, mode='evidence'), model, manager)
        assert len(prompts) == 1 and literal['answer_mode'] == 'evidence'
    finally:
        await service.stop()


def test_event_packet_mandatory_across_formatter_fallbacks():
    plan = build_retrieval_plan('Orion', [], [])
    fitted = fit_research_prompt('Orion', plan, [{'role': 'user', 'content': 'x' * 20000}],
        empty_assistance([]), system_policy='Answer faithfully.',
        event_context=json.dumps({'events': [{'id': 'event-placeholder', 'conditions': ['Do not drop me.']}]}),
        event_contract=EVENT_RESPONSE_POLICY)
    assert 'Do not drop me.' in fitted['messages'][1]['content']
    assert 'event_assessment' in fitted['messages'][0]['content']
    assert fitted['budget']['event_context_bytes'] > 0


@pytest.mark.asyncio
async def test_long_task_without_events_reviews_all_host_requirements(setup):
    service, manager, sid = setup
    doc = service.store.accept_document(sid, 'schedule.txt', b'Orion delivery date is unconfirmed.')
    service.store.claim_document(sid, doc['id'])
    service.store.complete_document(sid, doc['id'], [{'text': 'Orion delivery date is unconfirmed.'}])
    question = 'Draft an Orion delivery status brief and identify unknowns.'
    criteria = ['Separate recorded facts from proposed actions.', 'Obtain a confirmed delivery date.']
    requirements = build_task_requirements(question, criteria)
    calls = []
    async def model(prompt, **kwargs):
        calls.append(prompt)
        needs = packet(prompt, 'TASK_REQUIREMENTS_DATA_JSON:\n')
        assert [row['id'] for row in needs['requirements']] == ['Q1', 'C1', 'C2']
        assert 'EVENT_REASONING_DATA_JSON' not in prompt
        src = packet(prompt, 'UNTRUSTED_RESEARCH_DATA_JSON:\n')['source_quotes'][0]
        answer = f"Orion delivery date is unconfirmed [{src['label']}]. Proposed next step: ask the owner to confirm."
        return json.dumps({'answer': answer, 'task_assessment': {'requirements': [
            {'id': 'Q1', 'verdict': 'addressed', 'evidence': [{'label': src['label'], 'quote': src['content']}],
             'answer_witness': {'quote': answer}},
            {'id': 'C1', 'verdict': 'unknown', 'evidence': [],
             'missing_query': 'Check whether the owner approved the proposed communication.'}]}})
    try:
        result = await service.answer(req(sid, question), model, manager, document_ids=[doc['id']],
            project_instance_id=manager.get_session(sid)['instance_id'],
            project_context=json.dumps({'original_step_question': question, 'phase_acceptance_criteria': criteria}),
            task_requirements=requirements)
        assert len(calls) == 1 and 'event_reasoning' not in result
        assert result['task_review']['requirements_sha256'] == requirements['requirements_sha256']
        assert result['task_review']['addressed'] == 1
        assert result['task_review']['unknown'] == 2  # omitted C2 cannot vanish
        assert result['task_review']['goal_verified'] is False
        assert result['task_review']['suggested_queries']
        hydrated = service.hydrate_history(sid, manager.get_history(sid))[-1]
        assert hydrated['task_review'] == result['task_review']
        tampered = manager.get_history(sid)
        tampered[-1] = {**tampered[-1], 'content': 'A different or truncated draft.'}
        assert service.hydrate_history(sid, tampered)[-1]['content'] == WITHDRAWN_ANALYSIS
        service.store.delete_document(sid, doc['id'])
        withdrawn = service.hydrate_history(sid, manager.get_history(sid))[-1]
        assert withdrawn['content'] == WITHDRAWN_ANALYSIS and 'task_review' not in withdrawn
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_personal_draft_coverage_is_not_factual_support(setup):
    service, manager, sid = setup
    question = 'Draft a polite note asking to move the appointment; do not send it.'
    async def model(prompt, **kwargs):
        assert 'TASK_REQUIREMENTS_DATA_JSON' in prompt
        answer = 'Draft: Could we move the appointment to a mutually convenient time?'
        return json.dumps({'answer': answer, 'task_assessment': {'requirements': [
            {'id': 'Q1', 'verdict': 'addressed', 'evidence': [], 'answer_witness': {'quote': answer}}]}})
    try:
        result = await service.answer(req(sid, question), model, manager, document_ids=[], task_context_refs={},
            project_instance_id=manager.get_session(sid)['instance_id'], project_context='Draft only; no external action.',
            task_requirements=build_task_requirements(question, input_kind='personal'))
        review = result['task_review']
        assert review['addressed'] == 1 and review['supported'] == 0
        assert review['obligations'][0]['basis'] == 'answer_fulfilment_only'
        assert review['semantic_verified'] is False and review['goal_verified'] is False
        assert service.hydrate_history(sid, manager.get_history(sid))[-1]['task_review'] == review
    finally:
        await service.stop()
