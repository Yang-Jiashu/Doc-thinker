"""Final excerpt diagnostics remain current even when assistance is a no-op."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from docthinker.server.research_prompt_budget import empty_assistance, fit_research_prompt
from docthinker.server.research_retrieval import build_retrieval_plan
from docthinker.server.research_scheduler import ContextWorkResult
from docthinker.server.research_service import ResearchService

with patch('dotenv.load_dotenv', return_value=False):
    from docthinker.session_manager import SessionManager


@pytest.mark.asyncio
@pytest.mark.parametrize('enabled', [False, True])
async def test_unchanged_source_packet_refreshes_stale_gap_without_new_retrieval(tmp_path, enabled):
    manager = SessionManager(str(tmp_path / '_system'), str(tmp_path), migrate_legacy=False)
    sid = manager.create_session('Study')['id']
    service = ResearchService(tmp_path, semantic_retriever=None, context_assistance_enabled=enabled)
    text = 'Personalized assistants use memory only in the tested setting.'
    document = service.store.accept_document(sid, 'study.txt', text.encode())
    service.store.claim_document(sid, document['id'])
    service.store.complete_document(sid, document['id'], [{'text': text, 'page': 2}])
    baseline = build_retrieval_plan('assistant memory private', service.all_chunks(sid))
    baseline['gaps'] = [{'code': 'uncovered_query_terms', 'terms': ['assistant', 'stale']},
                        {'code': 'source_boundary_unverified', 'message': 'Preserve caveat.'}]
    baseline['covered_query_terms'] = []
    original = deepcopy(baseline)
    assistance = empty_assistance(deepcopy(baseline['sources']))
    assistance['_lexical_coverage'] = {'sources': deepcopy(baseline['sources']),
        'gaps': [], 'trace': {'algorithm': 'offline_fixture', 'model_calls': 0}}
    prompts, checked = [], []

    async def answer(prompt, **kwargs):
        prompts.append(prompt)
        return 'Personalized assistants use memory only in the tested setting. [S1]'

    def inspect_fit(question, plan, *args, **kwargs):
        checked.append(deepcopy(plan))
        gap = next(row for row in plan['gaps'] if row['code'] == 'uncovered_query_terms')
        assert gap['terms'] == ['private']
        assert plan['covered_query_terms'] == ['assistant', 'memory']
        return fit_research_prompt(question, plan, *args, **kwargs)

    request = SimpleNamespace(session_id=sid, question='assistant memory private', use_memory=False,
                              use_conversation_context=False, use_llm_cache=False, use_self_evolution=False)
    try:
        with patch('docthinker.server.research_service.build_retrieval_plan', return_value=deepcopy(baseline)) as ranked, \
             patch.object(service._context_scheduler, 'run', AsyncMock(
                 return_value=ContextWorkResult('completed', assistance))) as scheduled, \
             patch('docthinker.server.research_service.fit_research_prompt', side_effect=inspect_fit):
            result = await service.answer(request, answer, manager)
        assert ranked.call_count == 1 and len(prompts) == len(checked) == 1
        assert scheduled.await_count == int(enabled)
        assert result['sources'] == baseline['sources']
        gap = next(row for row in result['retrieval_gaps'] if row['code'] == 'uncovered_query_terms')
        assert gap['terms'] == ['private'] and '不表示概念缺失' in gap['message']
        assert any(row['code'] == 'source_boundary_unverified' for row in result['retrieval_gaps'])
        payload, _ = json.JSONDecoder().raw_decode(prompts[0].split('UNTRUSTED_RESEARCH_DATA_JSON:\n')[1])
        assert next(row for row in payload['retrieval_gaps']
                    if row['code'] == 'uncovered_query_terms')['terms'] == ['private']
        assert payload['source_quotes'][0]['content'] == text
        assert baseline == original
        assert result['reasoning']['trace']['answer_model_calls'] == 1
    finally:
        await service.stop()
