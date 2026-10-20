"""Offline execution contracts, not measurements of generated answer quality."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from docthinker.harness_runtime import current_run, execute_run
from docthinker.server.insight_runtime import InsightRuntime
from docthinker.server.research_knowledge import select_task_knowledge
from docthinker.server.research_service import ResearchService, WITHDRAWN_ANALYSIS, _supplement_insight_evidence
from docthinker.server.research_prompt_budget import empty_assistance

with patch('dotenv.load_dotenv', return_value=False):
    from docthinker.session_manager import SessionManager

QUESTION = 'Compare these papers on cobalt retrieval conditions and conflicting results.'


def source(identifier, text, document='d1', label='S1'):
    return {'id': identifier, 'document_id': document, 'label': label, 'content': text,
            'filename': document + '.txt', 'ordinal': 0, 'truncated': False,
            'excerpt_start': 0, 'excerpt_end': len(text), 'original_content_length': len(text)}


def proposal():
    support, _ = select_task_knowledge(QUESTION)
    return InsightRuntime(QUESTION, support).on_stage('before_retrieval', {
        'allow_supplemental_retrieval': True})


def test_supplement_keeps_primary_and_rejects_unrelated_dimension_match():
    primary = source('base', 'Cobalt retrieval improves accuracy in this study.')
    detail = source('detail', 'Cobalt conditions require fixed controls and dataset sizes.')
    unrelated = source('unrelated', 'Cancer trials reported conflicting results with different controls and dataset sizes.', 'd2')
    remote_background = {**source('appendix', unrelated['content']), 'ordinal': 30}
    result = _supplement_insight_evidence(QUESTION, [primary], [primary, unrelated, remote_background, detail], proposal())
    assert result['sources'][0] == primary
    assert [row['id'] for row in result['sources']] == ['base', 'detail']
    assert result['sources'][1]['label'] == 'S2'
    assert result['receipt']['retrieval_passes'] == 1
    assert result['receipt']['model_calls'] == 0
    assert result['receipt']['candidate_count'] == 1


def test_supplement_budget_does_not_evict_or_truncate_originals():
    primary = source('base', 'x' * 12000)
    detail = source('detail', 'Cobalt conditions require fixed controls.')
    result = _supplement_insight_evidence(QUESTION, [primary], [detail], proposal())
    assert result['sources'] == [primary]
    assert result['receipt']['reason'] == 'evidence_budget'
    assert result['receipt']['retrieval_passes'] == 0


def test_supplement_only_adds_two_and_reports_scan_window():
    base = source('base', 'Cobalt retrieval improves accuracy.')
    candidates = [source(str(i), f'Cobalt dataset conditions {i} require fixed controls.') for i in range(300)]
    result = _supplement_insight_evidence(QUESTION, [base], candidates, proposal())
    assert len(result['sources']) <= 3
    assert result['receipt']['scan_limited']
    assert result['receipt']['scanned_chunks'] == 256
    assert result['receipt']['source_bytes'] <= 2400


def test_supplement_does_not_reuse_a_label_after_a_source_was_filtered():
    primary = source('base', 'Cobalt retrieval result.', label='S2')
    detail = source('detail', 'Cobalt conditions require fixed controls.')
    result = _supplement_insight_evidence(QUESTION, [primary], [detail], proposal())
    labels = [row['label'] for row in result['sources']]
    assert labels == ['S2', 'S3']


def test_supplement_caps_final_packet_at_ten_without_inventing_s11():
    from docthinker.server.research_service import _append_insight_sources

    primary = [source(f'base-{i}', f'Original excerpt {i}.', label=f'S{i}') for i in range(1, 10)]
    extras = [source(f'extra-{i}', f'Additional excerpt {i}.') for i in range(2)]
    result = _append_insight_sources(primary, extras)
    assert result[:9] == primary and len(result) == 10 and result[-1]['label'] == 'S10'
    # A filtered packet may have a hole but S10 already present. Fill only
    # unused valid space; never relabel original provenance.
    sparse = [{**row, 'label': f'S{i}'} for i, row in enumerate(primary, 2)]
    result = _append_insight_sources(sparse, extras)
    assert result[:9] == sparse and len(result) == 10 and result[-1]['label'] == 'S1'


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHON_DOTENV_DISABLED', '1')
    monkeypatch.setenv('DOCTHINKER_RESEARCH_SEMANTIC', 'off')
    manager = SessionManager(str(tmp_path / '_system'), str(tmp_path), migrate_legacy=False)
    sid = manager.create_session('Insight integration')['id']

    async def extract(content, filename):
        return {'chunks': [{'text': text} for text in content.decode().split('\n===chunk===\n')]}

    service = ResearchService(tmp_path, extractor=extract, semantic_retriever=None,
                              evidence_reranker=None, query_translator=None)
    return service, manager, sid


def request(sid, **overrides):
    return SimpleNamespace(**{'session_id': sid, 'question': QUESTION,
                              'use_memory': False, 'use_conversation_context': False,
                              'knowledge_skill_mode': 'auto', 'reasoning_mode': 'auto', **overrides})


async def upload(project):
    service, _, sid = project
    text = ('Cobalt retrieval improves accuracy in this study.\n===chunk===\n'
            'Cobalt conditions require fixed controls and dataset sizes.')
    doc = service.store.accept_document(sid, 'cobalt.txt', text.encode())
    assert await service.process(sid, doc['id'])
    return doc


@pytest.mark.asyncio
async def test_three_stages_reach_one_writer_and_persist_real_receipts(project, monkeypatch):
    service, manager, sid = project
    await upload(project)
    from docthinker.server import research_service as module
    original = module.build_retrieval_plan

    def leave_gap(question, chunks, *args, **kwargs):
        # Controlled retrieval omission; the host's supplemental pass must
        # really look up the missing chunk rather than forge an expanded trace.
        if 'query_terms_override' not in kwargs:
            chunks = chunks[:1]
        return original(question, chunks, *args, **kwargs)

    monkeypatch.setattr(module, 'build_retrieval_plan', leave_gap)
    monkeypatch.setattr(module, 'build_context_assist', lambda question, sources, *a, **kw: empty_assistance(sources))
    # Isolate the deliberately omitted primary pack from the separate literal
    # coverage lane; this test verifies the Insight pass itself retrieves it.
    monkeypatch.setattr('docthinker.server.research_lexical_coverage.select_lexical_coverage',
        lambda question, sources, chunks: {'sources': sources, 'gaps': [], 'trace': {'status': 'retained'}})
    seen = []

    async def writer(prompt, *, system_prompt):
        seen.append((prompt, system_prompt, current_run().run_id))
        return 'The result requires fixed controls, not a universal improvement. [S1] [S2] [K1]'

    result = await execute_run(lambda: service.answer(request(sid), writer, manager),
                               session_id=sid, run_id='one-shared-owner')
    assert len(seen) == 1
    assert seen[0][2] == 'one-shared-owner'
    assert 'fixed controls' in seen[0][0]
    receipt = result['reasoning']['trace']['insight_runtime']
    assert receipt['analysis_included']
    assert 'INSIGHT_CONTEXT_DATA_JSON' in seen[0][0]
    assert receipt['supplement']['retrieval_passes'] == 1
    assert receipt['supplement']['kept_source_ids']
    assert receipt['model_calls'] == 0
    assert receipt['semantic_entailment_verified'] is False
    assert receipt['delivery_actions'][0]['status'] == 'checked'
    assert receipt['delivery']['actions'][0]['payload']['unknown_source_labels'] == []
    assert len(manager.get_history(sid)) == 2
    saved = service.hydrate_history(sid, manager.get_history(sid))[-1]
    assert saved['reasoning']['trace']['insight_runtime'] == receipt
    harness = result['reasoning']['trace']['harness']
    assert harness['run_id'] == 'one-shared-owner'
    assert [event['stage'] for event in harness['events'] if event['type'] == 'stage.completed'] == [
        'retrieval', 'analysis', 'generation', 'delivery']
    budget = result['reasoning']['trace']['context_budget']
    assert budget['input_bytes'] <= 24576
    assert budget['knowledge_skill_bytes'] + budget['insight_bytes'] <= 3500


@pytest.mark.asyncio
async def test_noop_supplement_preserves_primary_budget_fallback(project, monkeypatch):
    service, manager, sid = project
    await upload(project)
    from docthinker.server import research_service as module
    retrieve, fit = module.build_retrieval_plan, module.fit_research_prompt
    chunks = service.all_chunks(sid)
    primary = retrieve(QUESTION, chunks[:1])
    assisted = retrieve(QUESTION, chunks[1:])['sources']
    assert primary['sources'][0]['id'] != assisted[0]['id']
    monkeypatch.setattr(module, 'build_retrieval_plan', lambda *a, **kw: primary.copy())
    monkeypatch.setattr(module, 'build_context_assist', lambda *a, **kw: empty_assistance(assisted))
    # This intentionally injects two different packs to test fallback identity,
    # independent of upstream evidence selection.
    monkeypatch.setattr('docthinker.server.research_lexical_coverage.select_lexical_coverage',
        lambda question, sources, chunks: {'sources': sources, 'gaps': [], 'trace': {'status': 'retained'}})
    inspected = []

    def capture(question, plan, history, assistance, **kwargs):
        assert plan['sources'] == primary['sources']
        assert assistance['sources'] == assisted
        inspected.append(True)
        return fit(question, plan, history, assistance, **kwargs)

    monkeypatch.setattr(module, 'fit_research_prompt', capture)

    async def writer(prompt, **options):
        return 'Cobalt condition [S1] [K1]'

    result = await service.answer(request(sid), writer, manager)
    assert inspected == [True]
    assert not result['reasoning']['trace']['insight_runtime']['supplement']['added_source_ids']


@pytest.mark.asyncio
@pytest.mark.parametrize('controls', [{'knowledge_skill_mode': 'off'}, {'reasoning_mode': 'evidence'}])
async def test_opt_out_and_literal_do_not_execute_insight_work(project, controls):
    service, manager, sid = project
    await upload(project)
    calls = []

    async def writer(prompt, **options):
        calls.append(prompt)
        return 'Cobalt result. [S1]'

    result = await service.answer(request(sid, **controls), writer, manager)
    trace = result['reasoning']['trace']['insight_runtime']
    assert trace['schema_versions'] == []
    assert trace['analysis_actions'] == []
    assert trace['supplement']['retrieval_passes'] == 0
    assert len(calls) == (0 if controls.get('reasoning_mode') == 'evidence' else 1)
    assert all('INSIGHT_CONTEXT_DATA_JSON' not in text for text in calls)


@pytest.mark.asyncio
async def test_insight_off_retains_static_methods_for_ab_comparison(project):
    service, manager, sid = project
    await upload(project)
    calls = []

    async def writer(prompt, **options):
        calls.append(prompt)
        return 'Cobalt result under stated conditions. [S1] [K1]'

    result = await service.answer(request(sid, insight_mode='off'), writer, manager)
    receipt = result['reasoning']['trace']['insight_runtime']
    assert receipt['enabled'] is False
    assert receipt['schema_versions'] == []
    assert receipt['supplement']['retrieval_passes'] == 0
    assert 'KNOWLEDGE_SKILL_DATA_JSON' in calls[0]
    assert 'INSIGHT_CONTEXT_DATA_JSON' not in calls[0]
    assert result['knowledge_skill']['status'] == 'ready'
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_source_revocation_during_generation_hides_insight_and_history(project):
    service, manager, sid = project
    doc = await upload(project)

    async def writer(prompt, **options):
        service.store.delete_document(sid, doc['id'])
        return 'PRIVATE DERIVED CONCLUSION [S1] [K1]'

    result = await service.answer(request(sid), writer, manager)
    assert result['answer'] == WITHDRAWN_ANALYSIS
    assert result['sources'] == []
    trace = result['reasoning']['trace']['insight_runtime']
    assert trace['status'] in {'inactive', 'source_unavailable'}
    assert 'PRIVATE' not in json.dumps(trace)
    history = service.hydrate_history(sid, manager.get_history(sid))
    assert history[-1]['content'] == WITHDRAWN_ANALYSIS


@pytest.mark.asyncio
async def test_eight_primary_plus_two_insight_sources_persist_and_s10_revocation_closes_history(project, monkeypatch):
    service, manager, sid = project
    for i in range(10):
        text = f'Cobalt retrieval conditions require fixed controls and dataset sizes for study {i}.'
        doc = service.store.accept_document(sid, f'cobalt-{i}.txt', text.encode())
        assert await service.process(sid, doc['id'])
    from docthinker.server import research_service as module
    original = module.build_retrieval_plan

    def leave_two_gaps(question, chunks, *args, **kwargs):
        if 'query_terms_override' not in kwargs:
            chunks = chunks[:8]
        return original(question, chunks, *args, **kwargs)

    monkeypatch.setattr(module, 'build_retrieval_plan', leave_two_gaps)
    monkeypatch.setattr(module, 'build_context_assist', lambda question, sources, *a, **kw: empty_assistance(sources))
    calls = []
    answer = 'The comparison needs the stated controls. ' + ' '.join(f'[S{i}]' for i in range(1, 11)) + ' [K1]'

    async def writer(prompt, **kwargs):
        calls.append(prompt)
        return answer

    try:
        result = await service.answer(request(sid), writer, manager)
        assert result['answer'] == answer and len(calls) == 1
        assert [row['label'] for row in result['sources']] == [f'S{i}' for i in range(1, 11)]
        supplement = result['reasoning']['trace']['insight_runtime']['supplement']
        assert len(supplement['kept_source_ids']) == 2 and supplement['retrieval_passes'] == 1
        assert sum(len(row['content'].encode()) for row in result['sources']) <= 12000
        assert not result['citation_validation']['unknown_labels']
        saved = service.hydrate_history(sid, manager.get_history(sid))[-1]
        assert saved['content'] == answer and len(saved['sources']) == 10
        tenth = result['sources'][-1]
        service.store.delete_document(sid, tenth['document_id'])
        revoked = service.hydrate_history(sid, manager.get_history(sid))[-1]
        assert revoked['content'] == WITHDRAWN_ANALYSIS
        assert len(revoked['sources']) == 9 and all(row['label'] != 'S10' for row in revoked['sources'])
        assert revoked['reasoning']['status'] == 'source_unavailable'
        assert revoked['reasoning']['trace']['insight_runtime']['status'] == 'source_unavailable'
        assert len(calls) == 1
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_cancellation_does_not_commit_or_retry_answer(project):
    service, manager, sid = project
    await upload(project)
    started = asyncio.Event()
    calls = []

    async def writer(prompt, **options):
        calls.append(1)
        started.set()
        await asyncio.Future()

    task = asyncio.create_task(service.answer(request(sid), writer, manager))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(calls) == 1
    assert manager.get_history(sid) == []
