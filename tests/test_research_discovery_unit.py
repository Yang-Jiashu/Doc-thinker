"""Real scoped store/retrieval flow, fake final writer; no quality claims."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from docthinker.server.research_discovery import build_discovery_context, fit_graph_receipt
from docthinker.server.research_evidence_index import hydrate_reasoning
from docthinker.server.research_extract import extract_chunks
from docthinker.server.research_prompt_budget import empty_assistance, fit_research_prompt
from docthinker.server.research_retrieval import build_retrieval_plan
from docthinker.server.research_service import ResearchService, WITHDRAWN_ANALYSIS, TRUSTED_RESEARCH_POLICY

with patch('dotenv.load_dotenv', return_value=False):
    from docthinker.session_manager import SessionManager


async def extractor(content, filename):
    return {'chunks': extract_chunks(content, '.txt')}


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHON_DOTENV_DISABLED', '1')
    manager = SessionManager(str(tmp_path / '_system'), str(tmp_path), migrate_legacy=False)
    sid = manager.create_session('Arbitrary own-material task')['id']
    service = ResearchService(tmp_path, extractor=extractor, semantic_retriever=None,
                              evidence_reranker=None, query_translator=None)
    return service, manager, sid


async def upload(project, text, name='source.txt'):
    service, _, sid = project
    doc = service.store.accept_document(sid, name, text.encode())
    assert await service.process(sid, doc['id'])
    return doc


async def ask(project, question, callback=None, **overrides):
    service, manager, sid = project
    request = SimpleNamespace(session_id=sid, question=question, reasoning_mode='auto',
                              use_memory=False, use_conversation_context=False)
    for name, value in overrides.items():
        setattr(request, name, value)
    calls = []
    async def writer(prompt, **kwargs):
        calls.append((prompt, kwargs))
        if callback:
            callback()
        return 'Synthetic integration answer; no scientific validity claimed. [S1]'
    result = await service.answer(request, writer, manager)
    return result, calls


@pytest.mark.asyncio
async def test_own_material_idea_uses_three_operators_and_only_one_writer(project):
    await upload(project, 'Redundancy reduces outage risk. Outage risk reduces availability.', 'systems.txt')
    await upload(project, 'Supplier diversity reduces concentration risk. Concentration risk reduces energy security.', 'energy.txt')
    result, calls = await ask(project, 'Based on redundancy and supplier diversity, propose testable research ideas.')
    receipt = result['reasoning']['trace']['hypothesis_support']
    assert len(calls) == 1
    assert receipt['status'] == 'ready'
    assert receipt['model_calls'] == 0
    assert set(receipt['algorithm_modes']) == {'analogy', 'deduction', 'induction'}
    assert 'model_background_unverified' in calls[0][0]
    assert '证伪' in calls[0][0]
    assert result['answer_mode'] == 'explore'
    assert len(result['sources']) <= 8
    assert result['reasoning']['trace']['context_budget']['input_bytes'] <= 24576
    assert project[0].store.list_memories(project[2]) == []
    history = project[0].hydrate_history(project[2], project[1].get_history(project[2]))
    assert history[-1]['reasoning']['trace']['hypothesis_support'] == receipt


@pytest.mark.asyncio
async def test_graph_rescues_bridge_not_matching_question_words(project):
    await upload(project, 'Orchard program improves root growth.', 'start.txt')
    bridge = await upload(project, 'Root growth increases water storage.', 'bridge.txt')
    await upload(project, 'Water storage improves stable harvest.', 'end.txt')
    question = 'Trace the multi-hop path between orchard program and stable harvest.'
    baseline = build_retrieval_plan(question, project[0].all_chunks(project[2]))
    assert bridge['id'] not in {row['document_id'] for row in baseline['sources']}
    result, calls = await ask(project, question)
    assert len(calls) == 1
    assert bridge['id'] in {row['document_id'] for row in result['sources']}
    receipt = result['reasoning']['trace']['graph_retrieval']
    assert receipt['status'] == 'expanded'
    assert receipt['model_calls'] == 0
    assert receipt['semantic_entailment_verified'] is False
    assert 'Root growth increases water storage.' in calls[0][0]


@pytest.mark.asyncio
@pytest.mark.parametrize('question', ['What does redundancy reduce?', '只摘录 redundancy 的原文，不要提出新假设。'])
async def test_ordinary_or_literal_request_does_not_schedule_new_work(project, monkeypatch, question):
    await upload(project, 'Redundancy reduces outage risk.')
    monkeypatch.setattr('docthinker.server.research_service.build_discovery_context',
                        lambda *args: pytest.fail('Discovery should not run'))
    result, calls = await ask(project, question)
    assert len(calls) == 1
    assert 'hypothesis_support' not in result['reasoning']['trace']


@pytest.mark.asyncio
async def test_ablation_disables_graph_and_hypothesis_work(project, monkeypatch):
    await upload(project, 'Redundancy reduces outage risk.')
    project[0].context_assistance_enabled = False
    monkeypatch.setattr('docthinker.server.research_service.build_discovery_context',
                        lambda *args: pytest.fail('Disabled means no work'))
    result, calls = await ask(project, 'Based on redundancy, propose new research ideas.')
    assert len(calls) == 1
    assert '后台假设辅助' not in calls[0][0]
    assert 'hypothesis_support' not in result['reasoning']['trace']


@pytest.mark.asyncio
async def test_failed_local_work_falls_back_without_retries(project, monkeypatch):
    await upload(project, 'Redundancy reduces outage risk.')
    def fail(*args):
        raise ValueError('private diagnostic must not be returned')
    monkeypatch.setattr('docthinker.server.research_service.build_discovery_context', fail)
    result, calls = await ask(project, 'Propose new research ideas about redundancy.')
    assert len(calls) == 1
    assert result['reasoning']['trace']['discovery_scheduler']['status'] == 'failed'
    assert 'private diagnostic' not in str(result)


@pytest.mark.asyncio
async def test_revocation_during_model_call_hides_hypothesis_and_does_not_rebill(project):
    doc = await upload(project, 'Redundancy reduces outage risk.')
    service, manager, sid = project
    result, calls = await ask(project, 'Propose testable ideas about redundancy.',
                              lambda: service.store.delete_document(sid, doc['id']))
    assert len(calls) == 1
    assert result['answer'] == WITHDRAWN_ANALYSIS
    assert result['reasoning']['trace']['hypothesis_support']['status'] == 'source_unavailable'
    assert service.hydrate_history(sid, manager.get_history(sid))[-1]['content'] == WITHDRAWN_ANALYSIS


@pytest.mark.asyncio
async def test_no_evidence_never_fakes_material_based_idea(project):
    result, calls = await ask(project, '根据上传的论文提出新的研究假设。')
    assert calls == []
    assert result['answer_mode'] == 'insufficient_evidence'


def test_budgeted_graph_receipt_only_describes_current_sources():
    trace = {'status': 'expanded', 'bridge_source_ids': ['bridge'],
             'kept_source_ids': ['start', 'bridge'], 'paths': [{'source_ids': ['start', 'bridge']}]}
    old = deepcopy(trace)
    receipt = fit_graph_receipt(trace, [{'id': 'start'}])
    assert receipt['status'] == 'unchanged'
    assert receipt['bridge_source_ids'] == receipt['paths'] == []
    assert trace == old


def test_graph_and_hypothesis_source_dependency_without_formal_proof_revokes():
    reasoning = {'mode': 'context_assist', 'status': 'hypothesis_support', 'claims': [], 'conclusions': [],
        'trace': {'graph_retrieval': {'status': 'expanded', 'kept_source_ids': ['gone'], 'paths': ['private material']},
                  'hypothesis_support': {'status': 'ready', 'source_ids': ['gone']}}}
    result = hydrate_reasoning(reasoning, [])
    assert result['status'] == 'source_unavailable'
    assert 'private material' not in str(result)
    assert reasoning['trace']['graph_retrieval']['paths'] == ['private material']


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['record_race', 'answer_store', 'proof_store'])
async def test_persistence_failure_cannot_resurface_analysis_after_restart(project, monkeypatch, failure):
    doc = await upload(project, 'Redundancy reduces outage risk.')
    service, manager, sid = project
    record = service.store.record_answer
    def unavailable(*args, **kwargs):
        if failure == 'record_race':
            service.store.delete_document(sid, doc['id'])
            return record(*args, **kwargs)
        raise OSError('synthetic metadata failure')
    monkeypatch.setattr(service.evidence if failure == 'proof_store' else service.store,
                        'record' if failure == 'proof_store' else 'record_answer', unavailable)
    result, calls = await ask(project, 'Propose testable research ideas about redundancy.')
    assert len(calls) == 1
    assert result['answer'] == WITHDRAWN_ANALYSIS
    assert result['sources'] == []
    assert result['research_provenance_required'] is True
    restored_manager = SessionManager(str(service.store.root / '_system'), str(service.store.root), migrate_legacy=False)
    restored = ResearchService(service.store.root, semantic_retriever=None)
    history = restored.hydrate_history(sid, restored_manager.get_history(sid))
    assert history[-1]['content'] == WITHDRAWN_ANALYSIS
    assert any(row['code'] == 'provenance_unavailable'
               for row in history[-1]['citation_validation']['warnings'])
    prompts = []
    async def writer(prompt, **kwargs):
        prompts.append(prompt)
        return 'Follow-up.'
    request = SimpleNamespace(session_id=sid, question='Explain the limitations.',
                              reasoning_mode='auto', use_memory=False, use_conversation_context=True)
    await restored.answer(request, writer, restored_manager)
    assert len(prompts) == 1
    assert 'Synthetic integration answer' not in prompts[0]


def test_graph_gaps_can_be_saved_without_free_text_persistence():
    from docthinker.server.research_store import ResearchStore, ANSWER_GAP_MESSAGES
    codes = [code for code in ANSWER_GAP_MESSAGES if code.startswith('graph_')]
    metadata = ResearchStore._answer_context_metadata({'retrieval_gaps': [{'code': code, 'message': 'untrusted text'} for code in codes]})
    assert len(metadata['retrieval_gaps']) == len(codes)
    assert 'untrusted text' not in str(metadata)


@pytest.mark.parametrize('changed', ['truncated', 'different_text', 'shifted'])
def test_fitted_graph_path_requires_original_window_not_just_source_id(changed):
    text = 'Root growth increases water storage.'
    source = {'id': 'bridge', 'content': text, 'excerpt_start': 20}
    path = {'source_ids': ['bridge'], 'source_windows': [{'source_id': 'bridge',
        'excerpt_start': 20, 'excerpt_end': 20 + len(text),
        'content_sha256': hashlib.sha256(text.encode()).hexdigest()}]}
    trace = {'paths': [path], 'bridge_source_ids': ['bridge'], 'status': 'expanded'}
    assert fit_graph_receipt(trace, [source])['paths'] == [path]
    if changed == 'truncated':
        source['content'] = text[:15]
    elif changed == 'different_text':
        source['content'] = text.replace('increases', 'decreases')
    else:
        source['excerpt_start'] = 21
    assert fit_graph_receipt(trace, [source])['paths'] == []


def test_discovery_uses_followup_search_context_but_does_not_force_new_ideas():
    seen = []
    class Graph:
        def expand(self, question, sources, chunks, **kwargs):
            seen.append(question)
            return {'sources': sources, 'trace': {}, 'gaps': []}
    question = '继续展开'
    previous = question + '\n前一问题（仅作检索指代）：Trace the multi-hop path between orchard and harvest.'
    plan = {'sources': [], 'gaps': []}
    result = build_discovery_context(question, plan, empty_assistance([]), [], Graph(), search_question=previous)
    assert seen == [previous]
    assert 'hypothesis_support' not in result['assistance']['reasoning']['trace']


@pytest.mark.asyncio
async def test_hypothesis_support_still_runs_after_faceted_reranking(project, monkeypatch):
    await upload(project, 'Redundancy reduces outage risk. Outage risk reduces availability.')
    service, _, _ = project
    service._semantic = object()
    service._evidence_reranker = object()
    async def semantic(*args, **kwargs):
        return {'scores': {}, 'spans': {}, 'trace': {'status': 'disabled'}}
    rerank_calls = []
    def rerank(question, sources, chunks, *args, **kwargs):
        rerank_calls.append(question)
        result = empty_assistance(sources)
        result['reasoning']['status'] = 'evidence_selected'
        return result
    monkeypatch.setattr(service, '_semantic_operation', semantic)
    monkeypatch.setattr('docthinker.server.research_faceted_assist.build_faceted_assistance', rerank)
    result, calls = await ask(project, 'Based on redundancy, propose testable research ideas.')
    assert len(rerank_calls) == len(calls) == 1
    assert result['reasoning']['trace']['hypothesis_support']['status'] == 'ready'
    assert 'model_background_unverified' in calls[0][0]


@pytest.mark.asyncio
async def test_mobile_journal_reply_preserves_provenance_marker_on_failed_record(project, monkeypatch):
    from docthinker.server.routers import mobile
    await upload(project, 'Redundancy reduces outage risk.')
    service, _, sid = project
    def unavailable(*args, **kwargs):
        raise OSError('synthetic failure')
    monkeypatch.setattr(service.store, 'record_answer', unavailable)
    result, calls = await ask(project, 'Propose testable ideas about redundancy.')
    stored = mobile._query_response(result)
    assert stored['research_provenance_required'] is True
    stored['answer'] = 'Old stored text must not resurface through polling.'
    stored['_research_session_id'] = sid
    monkeypatch.setattr(mobile.state, 'research_service', service)
    hydrated = await mobile._hydrate_research_reply(stored)
    assert hydrated['answer'] == WITHDRAWN_ANALYSIS
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_optional_hypothesis_block_is_shed_whole_before_original_evidence(project):
    from docthinker.server.research_graph_retrieval import BoundedEvidenceGraph
    await upload(project, 'Redundancy reduces outage risk. Outage risk reduces availability.')
    question = 'Propose testable research ideas about redundancy.'
    chunks = project[0].all_chunks(project[2])
    plan = build_retrieval_plan(question, chunks)
    composed = build_discovery_context(question, plan, empty_assistance(plan['sources']), chunks, BoundedEvidenceGraph())
    assert composed['assistance']['prompt']
    baseline = fit_research_prompt(question, composed['plan'], [], empty_assistance(composed['plan']['sources']),
                                   system_policy=TRUSTED_RESEARCH_POLICY)
    # Leave room for the complete sources, but not the optional support block.
    padding = 24576 - baseline['budget']['input_bytes'] - 100
    fitted = fit_research_prompt(question, composed['plan'], [], composed['assistance'],
                                 system_policy=TRUSTED_RESEARCH_POLICY + 'X' * padding)
    assert fitted['plan']['sources'] == composed['plan']['sources']
    assert fitted['reasoning']['trace']['assist_bytes'] == 0
    assert 'hypothesis_support' not in fitted['reasoning']['trace']
    assert '后台假设辅助' not in fitted['prompt']
