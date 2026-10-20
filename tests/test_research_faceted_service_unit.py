"""Acceptance for evidence reselection through the real answer service.

Local scorer doubles verify routing/budgets/privacy, not learned model quality.
The separate reranker unit suite contains a real pinned-model offline smoke.
"""
from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from docthinker.server.research_service import ResearchService

with patch("dotenv.load_dotenv", return_value=False):
    from docthinker.session_manager import SessionManager


QUESTION = "论文通过什么实验设计判断治疗效果不是随机波动造成的？"
METHOD = ("In the fictional Orin experiment, participants were randomly assigned to treatment or placebo. "
          "Investigators preregistered the primary outcome and used confidence intervals to distinguish the treatment effect from chance.")
BACKGROUND = ("References and background: Clinical treatment studies appear in the annual overview of research. "
              "This bibliography lists prior work but does not describe the experimental assignment or statistical analysis.")


class Semantic:
    def __init__(self):
        self.calls = []

    def semantic_scores(self, question, chunks, *, session_id):
        self.calls.append((question, session_id, [row["content"] for row in chunks]))
        # Deliberately favor broad background; the relevant method remains a
        # candidate. This tests refinement, not a favorable first-stage double.
        return {"scores": {row["id"]: (0.83 if "randomly assigned" in row["content"] else 0.95) for row in chunks},
                "spans": {row["id"]: [0, len(row["content"])] for row in chunks},
                "trace": {"status": "ready", "provider_calls": 0}}

    def prepare_documents(self, chunks, *, session_id):
        return {"scores": {}, "spans": {}, "trace": {"status": "ready", "provider_calls": 0, "query_embedded": False}}


class Reranker:
    def __init__(self, *, failure=None, status="ready"):
        self.calls = []
        self.failure = failure
        self.status = status

    def rank(self, question, candidates, *, session_id):
        self.calls.append((question, session_id, [dict(row) for row in candidates]))
        if self.failure:
            raise RuntimeError(self.failure)
        return {"scores": ({row["id"]: (8.0 if "randomly assigned" in row["content"] else -8.0)
                            for row in candidates} if self.status == "ready" else {}),
                "trace": {"status": self.status, "provider_calls": 0, "score_kind": "raw_relevance_logit"}}


class Answer:
    def __init__(self):
        self.prompts = []

    async def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return "研究采用随机分组、安慰剂对照与预注册的结果指标，并用置信区间区分效果与随机波动。[S1]"


def workspace(root):
    manager = SessionManager(str(root / "_system"), str(root), migrate_legacy=False)
    return manager, manager.create_session("Research")['id']


def request(sid, question=QUESTION, *, history=False):
    return SimpleNamespace(session_id=sid, question=question, use_memory=False,
                           use_conversation_context=history, use_llm_cache=False, use_self_evolution=False)


def document(service, sid, passages=None):
    passages = passages or [BACKGROUND + f" Catalog entry {index}." for index in range(8)] + [METHOD]
    accepted = service.store.accept_document(sid, "orin-trial.txt", "\n".join(passages).encode())
    service.store.claim_document(sid, accepted['id'])
    service.store.complete_document(sid, accepted['id'], [{"text": text, "page": index + 1} for index, text in enumerate(passages)])
    return accepted


def identity(sources):
    return [(row['id'], row['content'], row['excerpt_start'], row['excerpt_end']) for row in sources]


async def worker_idle(scheduler):
    async def wait():
        while scheduler.running:
            await asyncio.sleep(0.001)
    await asyncio.wait_for(wait(), 2)


@pytest.mark.asyncio
async def test_faceted_service_recovers_method_and_rejects_background_with_one_answer_call(tmp_path):
    manager, sid = workspace(tmp_path)
    semantic, reranker, model = Semantic(), Reranker(), Answer()
    service = ResearchService(tmp_path, semantic_retriever=semantic, evidence_reranker=reranker)
    document(service, sid)
    result = await service.answer(request(sid), model, manager)
    assert len(model.prompts) == 1
    assert len(reranker.calls) == 1
    assert len(semantic.calls) >= 1
    assert any("randomly assigned" in row['content'] for row in result['sources'])
    assert not any("bibliography lists" in row['content'] for row in result['sources'])
    assert "randomly assigned" in model.prompts[0]
    assert "bibliography lists" not in model.prompts[0]
    assert result['reasoning']['mode'] == 'context_assist'
    assert not result['reasoning'].get('conclusions')
    assert result['reasoning']['trace']['model_calls'] == 0
    assert result['reasoning']['trace']['assist_bytes'] == 0
    assert result['reasoning']['trace']['retrieval_reselection'] is True
    assert result['reasoning']['trace']['answer_model_calls'] == 1
    originals = {row['id']: row['content'] for row in service.all_chunks(sid)}
    for source in result['sources']:
        assert source['content'] == originals[source['id']][source['excerpt_start']:source['excerpt_end']]
    assert len(result['sources']) <= 8
    assert sum(len(source['content'].encode()) for source in result['sources']) <= 12000
    await service.stop()


@pytest.mark.asyncio
async def test_pair_capable_reranker_receives_facet_queries_once_without_single_question_route(tmp_path):
    class PairReranker:
        def __init__(self):
            self.pairs = []
            self.single_calls = 0

        def rank(self, question, candidates, *, session_id):
            self.single_calls += 1
            raise AssertionError('A pair-capable scorer must use rank_pairs')

        def rank_pairs(self, candidates, *, session_id):
            self.pairs.append((session_id, [dict(row) for row in candidates]))
            return {'scores': {row['id']: (8.0 if 'randomly assigned' in row['content'] else -8.0)
                               for row in candidates},
                    'trace': {'status': 'ready', 'provider_calls': 0, 'score_kind': 'raw_relevance_logit'}}

    manager, sid = workspace(tmp_path)
    reranker, model = PairReranker(), Answer()
    service = ResearchService(tmp_path, semantic_retriever=Semantic(), evidence_reranker=reranker)
    document(service, sid)
    question = '论文的实验设计是什么？如何排除随机波动？'
    result = await service.answer(request(sid, question), model, manager)
    assert reranker.single_calls == 0
    assert len(reranker.pairs) == 1
    namespace, pairs = reranker.pairs[0]
    assert namespace.endswith('::' + sid) and str(tmp_path) not in namespace
    assert 1 <= len(pairs) <= 48
    assert len({row['id'] for row in pairs}) == len(pairs)
    for row in pairs:
        assert isinstance(row['query'], str) and row['query'].strip()
        assert isinstance(row['rerank_facet_id'], str) and row['rerank_facet_id']
        assert row['query'] in question
    assert len(model.prompts) == 1 and result['sources']
    originals = {row['id']: row['content'] for row in service.all_chunks(sid)}
    for source in result['sources']:
        assert source['content'] == originals[source['id']][source['excerpt_start']:source['excerpt_end']]
    assert result['reasoning']['trace']['answer_model_calls'] == 1
    assert result['reasoning']['trace']['reranker']['status'] == 'ready'
    await service.stop()


def test_literal_pair_assignment_preserves_conditions_and_does_not_expand_candidate_budget():
    from docthinker.server.research_faceted_assist import _rerank_pairs

    condition = 'If availability drops, why does treatment fail?'
    experiment = 'How did the study control random variation?'
    question = condition + ' ' + experiment
    facets = [
        {'id': 'original', 'kind': 'original', 'text': question, 'weight': .5},
        {'id': 'condition', 'kind': 'clause', 'text': condition, 'weight': 1.},
        {'id': 'experiment', 'kind': 'clause', 'text': experiment, 'weight': 1.},
        {'id': 'document', 'kind': 'document', 'text': 'untrusted-document-title', 'weight': 1.},
    ]
    candidates = [
        {'id': f'window-{index}', 'content': 'PRIVATE_SOURCE_ONLY_NOT_A_QUERY',
         'facet_scores': {'original': .6, 'condition': .8 if index % 2 == 0 else .2,
                          'experiment': .8 if index % 2 else .2, 'document': 1.}}
        for index in range(48)
    ]
    pairs = _rerank_pairs(question, candidates, facets)
    assert len(pairs) == len(candidates) == 48
    assert [row['id'] for row in pairs] == [row['id'] for row in candidates]
    assert {row['query'] for row in pairs} == {question}
    assert {row['rerank_facet_id'] for row in pairs} == {'original'}
    assert all(row['retrieval_support'] == .8 for row in pairs)
    assert all(row['query'] in question and row['query'] != row['content'] for row in pairs)
    assert all('query' not in row for row in candidates)  # No mutation of pool inputs.


def test_full_question_fallback_has_its_own_actual_query_group_and_credit():
    from docthinker.server.research_faceted_assist import _rerank_pairs

    question = 'How does Atlas operate, and what conditions restrict it?'
    facets = [
        {'id': 'original', 'kind': 'original', 'text': question, 'weight': .5},
        {'id': 'how', 'kind': 'clause', 'text': 'How does Atlas operate?', 'weight': 1.},
        {'id': 'when', 'kind': 'clause', 'text': 'What conditions restrict it?', 'weight': 1.},
    ]
    candidates = [{'id': 'literal', 'facet_scores': {'how': .8}},
                  {'id': 'fallback', 'facet_scores': {'original': .8}}]
    pairs = _rerank_pairs(question, candidates, facets)
    assert pairs[0]['query'] == pairs[1]['query'] == question
    assert pairs[0]['rerank_query_id'] == pairs[1]['rerank_query_id']
    assert pairs[0]['rerank_facet_id'] == 'original'
    assert pairs[1]['rerank_facet_id'] == 'original'
    assert len(pairs) == len(candidates)


def test_rank_only_scorer_keeps_one_actual_question_group_and_original_credit():
    from docthinker.server.research_faceted_assist import build_faceted_assistance
    from docthinker.server.research_query_facets import build_query_facets

    question = 'How were participants assigned? What did the experiment measure?'
    chunks = [{'id': 'method', 'document_id': 'study', 'filename': 'study.pdf',
               'ordinal': 0, 'content': METHOD}]
    original = next(facet['id'] for facet in build_query_facets(question, chunks)['facets']
                    if facet['kind'] == 'original')
    scorer = Reranker()
    result = build_faceted_assistance(question, [], chunks, Semantic(), scorer, session_id='synthetic::rank-only')
    assert len(scorer.calls) == 1
    actual_query, _, pairs = scorer.calls[0]
    assert actual_query == question
    assert {pair['query'] for pair in pairs} == {question}
    assert {pair['rerank_facet_id'] for pair in pairs} == {original}
    assert len({pair['rerank_query_id'] for pair in pairs}) == 1
    assert result['sources']
    assert result['trace']['selection']['rerank_query_group_count'] == 1


@pytest.mark.asyncio
async def test_disabled_assistance_performs_no_facets_or_reranking(tmp_path):
    manager, sid = workspace(tmp_path)
    semantic, reranker, model = Semantic(), Reranker(), Answer()
    service = ResearchService(tmp_path, semantic_retriever=semantic, evidence_reranker=reranker,
                              context_assistance_enabled=False)
    document(service, sid)
    result = await service.answer(request(sid), model, manager)
    assert len(semantic.calls) == 1
    assert reranker.calls == []
    assert len(model.prompts) == 1
    assert result['reasoning']['trace'].get('retrieval_reselection') is not True
    await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize('status,failure', [
    ('unavailable', None), ('busy', None),
    ('ready', '/private/customer/invoice api-secret private-source-phrase'),
])
async def test_reranker_failure_preserves_baseline_without_legacy_expansion_or_private_leak(tmp_path, status, failure):
    manager, sid = workspace(tmp_path)
    service = ResearchService(tmp_path, semantic_retriever=Semantic(), evidence_reranker=Reranker(status=status, failure=failure),
                              context_assistance_enabled=False)
    document(service, sid)
    baseline = await service.answer(request(sid), Answer(), manager)
    service.context_assistance_enabled = True
    model = Answer()
    with patch('docthinker.server.research_service.build_context_assist', side_effect=AssertionError('Do not fall through to legacy expansion')):
        result = await service.answer(request(sid), model, manager)
    assert identity(result['sources']) == identity(baseline['sources'])
    assert len(model.prompts) == 1
    if failure:
        assert failure not in json.dumps(result, ensure_ascii=False)
        assert failure not in model.prompts[0]
    await service.stop()


@pytest.mark.asyncio
async def test_completed_candidate_scan_filtering_everything_does_not_restore_baseline_noise(tmp_path):
    manager, sid = workspace(tmp_path)
    reranker = Reranker()
    service = ResearchService(tmp_path, semantic_retriever=Semantic(), evidence_reranker=reranker,
                              context_assistance_enabled=False)
    document(service, sid, [BACKGROUND])
    baseline = await service.answer(request(sid), Answer(), manager)
    assert baseline['sources']  # Establish that an accidental restoration is observable.
    service.context_assistance_enabled = True
    model = Answer()
    completed_scan = {'candidates': [], 'trace': {'live_chunks_scanned': 1,
                                                'bibliographic_units_filtered': 1,
                                                'corpus_scan_limited': False}}
    with patch('docthinker.server.research_faceted_assist.prepare_evidence_candidates', return_value=completed_scan):
        result = await service.answer(request(sid), model, manager)
    assert result['sources'] == []
    assert result['reasoning']['status'] == 'evidence_selected'
    assert result['answer_mode'] == 'insufficient_evidence'
    assert 'insufficient_evidence' in {gap['code'] for gap in result['retrieval_gaps']}
    assert result['reasoning']['trace']['retrieval_reselection'] is True
    assert reranker.calls == [] and model.prompts == []
    assert result['reasoning']['trace']['answer_model_calls'] == 0
    await service.stop()


@pytest.mark.asyncio
async def test_ready_reranking_selecting_no_usable_evidence_does_not_restore_old_sources(tmp_path):
    manager, sid = workspace(tmp_path)
    reranker = Reranker()
    service = ResearchService(tmp_path, semantic_retriever=Semantic(), evidence_reranker=reranker,
                              context_assistance_enabled=False)
    document(service, sid, [METHOD])
    baseline = await service.answer(request(sid), Answer(), manager)
    assert baseline['sources']
    service.context_assistance_enabled = True
    model = Answer()
    selection = {'sources': [], 'trace': {'selected_count': 0}, 'gaps': [
        {'code': 'insufficient_evidence', 'message': 'The completed selection found no usable evidence.'},
    ]}
    with patch('docthinker.server.research_faceted_assist.select_evidence', return_value=selection) as select:
        result = await service.answer(request(sid), model, manager)
    assert select.call_count == len(reranker.calls) == 1
    assert result['reasoning']['trace']['reranker']['status'] == 'ready'
    assert result['sources'] == []
    assert result['reasoning']['status'] == 'evidence_selected'
    assert result['answer_mode'] == 'insufficient_evidence'
    assert selection['gaps'][0] in result['retrieval_gaps']
    assert model.prompts == []
    restored = service.hydrate_history(sid, manager.get_history(sid))[-1]
    assert restored['sources'] == []
    assert 'insufficient_evidence' in {gap['code'] for gap in restored['retrieval_gaps']}
    await service.stop()


@pytest.mark.asyncio
async def test_refined_spans_persist_and_revoked_sources_cannot_reenter_history(tmp_path):
    manager, sid = workspace(tmp_path)
    semantic, reranker = Semantic(), Reranker()
    service = ResearchService(tmp_path, semantic_retriever=semantic, evidence_reranker=reranker)
    uploaded = document(service, sid)
    result = await service.answer(request(sid), Answer(), manager)
    history = service.hydrate_history(sid, manager.get_history(sid))
    persisted = next(row for row in history if row.get('id') == result['answer_id'])
    assert identity(persisted['sources']) == identity(result['sources'])
    service.store.delete_document(sid, uploaded['id'])
    history = service.hydrate_history(sid, manager.get_history(sid))
    revoked = next(row for row in history if row.get('id') == result['answer_id'])
    assert revoked['sources'] == []
    assert 'source_revoked' in {item['code'] for item in revoked['citation_validation']['warnings']}
    model = Answer()
    next_result = await service.answer(request(sid, history=True), model, manager)
    assert next_result['sources'] == []
    assert model.prompts == []
    await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize('gap_code', [
    'evidence_coverage_proxy', 'evidence_facets_uncovered', 'evidence_anchors_uncovered',
])
async def test_new_evidence_limitations_persist_without_erasing_completed_answer_sources(tmp_path, gap_code):
    from docthinker.server.research_evidence_selection import select_evidence

    manager, sid = workspace(tmp_path)
    service = ResearchService(tmp_path, semantic_retriever=Semantic(), evidence_reranker=Reranker())
    document(service, sid, [METHOD])
    model = Answer()

    def selected_with_gap(*args, **kwargs):
        selected = select_evidence(*args, **kwargs)
        selected['gaps'] = [{'code': gap_code, 'count': 1,
                             'message': 'PRIVATE_CALLER_MESSAGE_NOT_FOR_STORAGE'}]
        return selected

    with patch('docthinker.server.research_faceted_assist.select_evidence', side_effect=selected_with_gap):
        result = await service.answer(request(sid), model, manager)
    assert len(model.prompts) == 1
    assert result['sources']
    assert 'provenance_unavailable' not in {row['code'] for row in result['citation_validation']['warnings']}
    saved = next(row for row in service.hydrate_history(sid, manager.get_history(sid))
                 if row.get('id') == result['answer_id'])
    assert identity(saved['sources']) == identity(result['sources'])
    assert gap_code in {row['code'] for row in saved['retrieval_gaps']}
    assert 'PRIVATE_CALLER_MESSAGE_NOT_FOR_STORAGE' not in json.dumps(saved['retrieval_gaps'])
    await service.stop()


@pytest.mark.asyncio
async def test_same_display_session_across_tenants_never_shares_reranker_namespace_or_sources(tmp_path):
    semantic, reranker = Semantic(), Reranker()
    entries = []
    for tenant in ('tenant-one', 'tenant-two'):
        root = tmp_path / tenant
        manager, sid = workspace(root)
        service = ResearchService(root, semantic_retriever=semantic, evidence_reranker=reranker)
        document(service, sid, [METHOD + f" This passage belongs to {tenant}."])
        result = await service.answer(request(sid), Answer(), manager)
        entries.append((sid, reranker.calls[-1][1], result))
        own_chunks = {row['id']: row['content'] for row in service.all_chunks(sid)}
        assert result['sources']
        for row in result['sources']:
            assert row['id'] in own_chunks
            assert row['content'] == own_chunks[row['id']][row['excerpt_start']:row['excerpt_end']]
        other_tenant = 'tenant-two' if tenant == 'tenant-one' else 'tenant-one'
        assert all(other_tenant not in row['content'] for row in result['sources'])
        await service.stop()
    assert entries[0][0] == entries[1][0]
    assert entries[0][1] != entries[1][1]
    assert str(tmp_path) not in entries[0][1] and str(tmp_path) not in entries[1][1]


@pytest.mark.asyncio
async def test_short_followup_uses_only_enabled_previous_user_question_for_local_retrieval(tmp_path):
    manager, sid = workspace(tmp_path)
    semantic, reranker = Semantic(), Reranker()
    service = ResearchService(tmp_path, semantic_retriever=semantic, evidence_reranker=reranker)
    document(service, sid, [METHOD])
    manager.add_message(sid, 'user', QUESTION)
    manager.add_message(sid, 'assistant', 'INVENTED_ASSISTANT_QUERY_EXPANSION')
    result = await service.answer(request(sid, '再解释一下。', history=True), Answer(), manager)
    assert QUESTION in semantic.calls[0][0]
    assert QUESTION in reranker.calls[0][0]
    assert not any('INVENTED_ASSISTANT_QUERY_EXPANSION' in call[0] for call in semantic.calls + reranker.calls)
    assert result['reasoning']['trace']['retrieval_context'] == 'previous_user_turn'
    semantic.calls.clear()
    reranker.calls.clear()
    result = await service.answer(request(sid, '再解释一下。', history=False), Answer(), manager)
    assert semantic.calls[0][0] == '再解释一下。'
    assert all(QUESTION not in call[0] for call in semantic.calls + reranker.calls)
    assert result['reasoning']['trace']['retrieval_context'] == 'current_question'
    await service.stop()


@pytest.mark.asyncio
async def test_evidence_lane_timeout_keeps_single_actual_worker_and_stop_does_not_wait(tmp_path):
    manager, sid = workspace(tmp_path)
    service = ResearchService(tmp_path, semantic_retriever=Semantic(), evidence_reranker=Reranker())
    document(service, sid, [METHOD])
    assert service._evidence_scheduler.timeout_seconds == 8.0
    service._evidence_scheduler.timeout_seconds = 0.01
    started, release = threading.Event(), threading.Event()
    calls = []
    def blocked(*args, **kwargs):
        calls.append(1)
        started.set()
        assert release.wait(3)
        # The late result must never replace a response that already timed out.
        raise RuntimeError('private late worker failure')
    try:
        with patch('docthinker.server.research_faceted_assist.build_faceted_assistance', side_effect=blocked):
            first = await service.answer(request(sid), Answer(), manager)
            assert started.is_set() and service._evidence_scheduler.running
            assert first['sources']
            for _ in range(3):
                result = await service.answer(request(sid), Answer(), manager)
                assert identity(result['sources']) == identity(first['sources'])
            assert calls == [1]
            await asyncio.wait_for(service.stop(), 0.1)
            closed = await service._evidence_scheduler.run(lambda: 'never')
            assert closed.status == 'closed'
    finally:
        release.set()
        await worker_idle(service._evidence_scheduler)
        await service.stop()
