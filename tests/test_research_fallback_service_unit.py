"""Fallback metadata hygiene does not retry, expand, or resurrect late work."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from docthinker.server.research_prompt_budget import empty_assistance, fit_research_prompt
from docthinker.server.research_scheduler import ContextWorkResult
from docthinker.server.research_service import ResearchService

with patch('dotenv.load_dotenv', return_value=False):
    from docthinker.session_manager import SessionManager


REFERENCE = ('[1] Smith, A. Measurement method. Journal of Trials 12, 31–39 (2021).\n'
             '[2] Doe, B. Control procedure. arXiv preprint arXiv:2408.12345 (2024).\n19')
CONTENT = 'The experiment randomly assigned samples and reported a measured error of 0.3%.'
QUESTION = 'Explain the paper experiment and limitations.'


class Semantic:
    def semantic_scores(self, question, chunks, *, session_id):
        return {'scores': {row['id']: .9 for row in chunks}, 'spans': {},
                'trace': {'status': 'ready', 'provider_calls': 0}}


class Answer:
    def __init__(self):
        self.prompts = []

    async def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return 'The experiment randomly assigned samples. [S2]'


def setup(tmp_path, passages=(REFERENCE, CONTENT)):
    manager = SessionManager(str(tmp_path / '_system'), str(tmp_path), migrate_legacy=False)
    sid = manager.create_session('Research')['id']
    service = ResearchService(tmp_path, semantic_retriever=Semantic(), evidence_reranker=object())
    doc = service.store.accept_document(sid, 'study.txt', '\n'.join(passages).encode())
    service.store.claim_document(sid, doc['id'])
    service.store.complete_document(sid, doc['id'], [{'text': text, 'page': index + 1}
                                                  for index, text in enumerate(passages)])
    originals = service.all_chunks(sid)
    # Deterministic baseline isolates the service fallback from ranking quality.
    sources = [{**row, 'label': f'S{index + 1}', 'excerpt_start': 0,
                'excerpt_end': len(row['content']), 'truncated': False,
                'offset_unit': 'characters_in_source_chunk'} for index, row in enumerate(originals)]

    def plan(*args, **kwargs):
        values = deepcopy(sources)
        return {'sources': values, 'selected_chunks': values, 'confirmed_memories': [],
                'gaps': [{'code': 'uncovered_query_terms', 'terms': ['stale']},
                         {'code': 'documents_not_selected', 'count': 1},
                         {'code': 'excerpt_truncated', 'message': 'stale'}]}

    return service, manager, sid, sources, plan


def request(sid, question=QUESTION):
    return SimpleNamespace(session_id=sid, question=question, use_memory=False,
                           use_conversation_context=False, use_llm_cache=False, use_self_evolution=False)


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['timed_out', 'busy', 'failed', 'reranker_unavailable'])
async def test_fallback_pack_gaps_budget_and_assistance_are_consistent_without_extra_calls(tmp_path, status):
    service, manager, sid, sources, baseline = setup(tmp_path)
    model, captured = Answer(), []
    assistance = empty_assistance(deepcopy(sources), reason='reranker_unavailable')
    assistance['reasoning']['trace']['reranker'] = {'status': 'unavailable', 'provider_calls': 0}
    work = ContextWorkResult('completed', assistance) if status == 'reranker_unavailable' else ContextWorkResult(status)

    def inspect_fit(question, plan, history, supplied, **kwargs):
        captured.append((deepcopy(plan), deepcopy(supplied)))
        assert plan['sources'] == plan['selected_chunks'] == supplied['sources'] == [sources[1]]
        assert plan['gaps'] == supplied['reasoning']['gaps']
        return fit_research_prompt(question, plan, history, supplied, **kwargs)

    try:
        with patch('docthinker.server.research_service.build_retrieval_plan', side_effect=baseline), \
             patch.object(service._evidence_scheduler, 'run', AsyncMock(return_value=work)) as scheduled, \
             patch('docthinker.server.research_service.fit_research_prompt', side_effect=inspect_fit), \
             patch('docthinker.server.research_service.build_context_assist', side_effect=AssertionError('no legacy retry')):
            result = await service.answer(request(sid), model, manager)
        assert scheduled.await_count == 1 and len(captured) == 1
        assert len(model.prompts) == 1
        assert '[1] Smith' not in model.prompts[0] and CONTENT in model.prompts[0]
        assert result['sources'] == [sources[1]]
        trace = result['reasoning']['trace']
        assert trace['model_calls'] == 0 and trace['answer_model_calls'] == 1
        receipt = trace['fallback_metadata_filter']
        assert receipt['filtered_count'] == 1 and receipt['provider_calls'] == receipt['llm_tokens'] == 0
        assert trace['context_budget']['source_bytes'] == len(CONTENT.encode())
        assert trace['context_budget']['input_bytes'] <= trace['context_budget']['max_input_bytes']
        assert trace['context_budget']['retrieval_passes'] == 1
        codes = [gap['code'] for gap in result['retrieval_gaps']]
        assert 'fallback_metadata_filtered' in codes
        assert 'uncovered_query_terms' in codes and 'excerpt_truncated' not in codes
        assert 'documents_not_selected' in codes
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_all_metadata_fallback_uses_existing_insufficient_evidence_without_answer_call(tmp_path):
    service, manager, sid, _, baseline = setup(tmp_path, (REFERENCE,))
    model = Answer()
    try:
        with patch('docthinker.server.research_service.build_retrieval_plan', side_effect=baseline), \
             patch.object(service._evidence_scheduler, 'run', AsyncMock(return_value=ContextWorkResult('failed'))):
            result = await service.answer(request(sid), model, manager)
        assert result['sources'] == [] and result['answer_mode'] == 'insufficient_evidence'
        assert model.prompts == []
        assert result['reasoning']['trace']['answer_model_calls'] == 0
        assert result['reasoning']['trace']['context_budget']['source_bytes'] == 0
        assert 'insufficient_evidence' in [row['code'] for row in result['retrieval_gaps']]
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['disabled', 'successful', 'metadata_query'])
async def test_success_disabled_and_explicit_metadata_requests_do_not_lose_original_sources(tmp_path, mode):
    service, manager, sid, sources, baseline = setup(tmp_path)
    service.context_assistance_enabled = mode != 'disabled'
    assistance = empty_assistance(deepcopy(sources))
    assistance['reasoning']['status'] = 'evidence_selected'
    work = ContextWorkResult('timed_out') if mode == 'metadata_query' else ContextWorkResult('completed', assistance)
    try:
        with patch('docthinker.server.research_service.build_retrieval_plan', side_effect=baseline), \
             patch.object(service._evidence_scheduler, 'run', AsyncMock(return_value=work)):
            result = await service.answer(request(sid, 'List the paper references.' if mode == 'metadata_query' else QUESTION), Answer(), manager)
        assert result['sources'] == sources
        receipt = result['reasoning']['trace'].get('fallback_metadata_filter')
        if mode == 'metadata_query':
            assert receipt['filtered_count'] == 0 and receipt['status'] == 'metadata_query_retained'
        else:
            assert receipt is None
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_timeout_and_busy_filter_without_mutating_worker_input_or_using_late_result(tmp_path):
    service, manager, sid, sources, baseline = setup(tmp_path)
    service._evidence_scheduler.timeout_seconds = .01
    entered, release = threading.Event(), threading.Event()
    calls, snapshots = [], []

    def late(question, given, *args, **kwargs):
        calls.append(1)
        snapshots.append(given)
        entered.set()
        assert release.wait(3)
        result = empty_assistance(given)
        result['reasoning']['status'] = 'evidence_selected'
        return result

    try:
        with patch('docthinker.server.research_service.build_retrieval_plan', side_effect=baseline), \
             patch('docthinker.server.research_faceted_assist.build_faceted_assistance', side_effect=late):
            first_model, second_model = Answer(), Answer()
            first = await service.answer(request(sid), first_model, manager)
            assert entered.is_set() and service._evidence_scheduler.running
            assert first['reasoning']['trace']['scheduler_status'] == 'timed_out'
            timing = first['reasoning']['trace']['scheduler_timing']
            assert timing['worker_started'] and not timing['worker_finished']
            assert timing['worker_ms'] is None and timing['queue_ms'] >= 0
            second = await service.answer(request(sid), second_model, manager)
            assert second['reasoning']['trace']['scheduler_status'] == 'busy'
            assert not second['reasoning']['trace']['scheduler_timing']['work_submitted']
            assert first['sources'] == second['sources'] == [sources[1]]
            assert snapshots == [sources]  # Worker-owned original list was not edited.
            saved = deepcopy(first)
            release.set()
            async def idle():
                while service._evidence_scheduler.running:
                    await asyncio.sleep(.001)
            await asyncio.wait_for(idle(), 2)
            assert first == saved
            assert calls == [1] and len(first_model.prompts) == len(second_model.prompts) == 1
            assert first['reasoning']['trace']['fallback_metadata_filter']['provider_calls'] == 0
            history = service.hydrate_history(sid, manager.get_history(sid))
            assert all(len(item['sources']) == 1 for item in history if item.get('role') == 'assistant')
    finally:
        release.set()
        await service.stop()
