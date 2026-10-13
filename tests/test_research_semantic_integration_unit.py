"""Service acceptance for local semantic retrieval; no paid/model network calls.

The qualified cosine double checks plumbing, not embedding answer quality. Real
ONNX cross-language/no-network tests live in test_research_semantic_unit.py.
"""
from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from docthinker.server.research_semantic import LocalSemanticRetriever
from docthinker.server.research_service import ResearchService

with patch("dotenv.load_dotenv", return_value=False):
    from docthinker.session_manager import SessionManager


@pytest.mark.parametrize("mode,enabled", [(None, True), ("auto", True), ("local", True), ("off", False)])
def test_service_model_mode_preserves_production_backend_wiring(tmp_path, monkeypatch, mode, enabled):
    """Exercise default discovery explicitly without loading any model assets."""
    from docthinker.server import research_query_translation, research_reranker, research_service

    if mode is None:
        monkeypatch.delenv("DOCTHINKER_RESEARCH_SEMANTIC", raising=False)
    else:
        monkeypatch.setenv("DOCTHINKER_RESEARCH_SEMANTIC", mode)
    semantic = Mock(return_value=object())
    reranker = Mock(return_value=object())
    translator = Mock(return_value=object())
    monkeypatch.setattr(research_service, "LocalSemanticRetriever", semantic)
    monkeypatch.setattr(research_reranker, "LocalEvidenceReranker", reranker)
    monkeypatch.setattr(research_query_translation, "LocalQueryTranslator", translator)

    service = ResearchService(tmp_path)

    for factory, backend in ((semantic, service._semantic),
                             (reranker, service._evidence_reranker),
                             (translator, service._query_translator)):
        if enabled:
            factory.assert_called_once_with()
            assert backend is factory.return_value
        else:
            factory.assert_not_called()
            assert backend is None


PASSAGE = "Renewable electricity investment increases domestic energy supply and reduces dependence on imported fossil fuels."


class QualifiedSemantic:
    def __init__(self, *, stale=False, failure=None):
        self.calls = []
        self.saved_scores = {}
        self.stale = stale
        self.failure = failure

    def prepare_documents(self, chunks, *, session_id):
        self.calls.append(("prepare", session_id, [row["id"] for row in chunks]))
        if self.failure:
            raise RuntimeError(self.failure)
        return {"scores": {}, "trace": {"status": "ready", "query_embedded": False,
                                           "query_embedding_ms": 0.0, "provider_calls": 0}}

    def semantic_scores(self, question, chunks, *, session_id):
        self.calls.append(("query", session_id, [row["id"] for row in chunks]))
        if self.failure:
            raise RuntimeError(self.failure)
        scores = {row["id"]: 0.9 for row in chunks}
        self.saved_scores.update(scores)
        return {"scores": dict(self.saved_scores) if self.stale else scores,
                "trace": {"status": "ready", "provider_calls": 0, "llm_tokens": 0}}


class AnswerModel:
    def __init__(self):
        self.prompts = []

    async def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return "材料指出，可再生能源投资增加国内供给并降低进口依赖。[S1]"


class BlockingSemantic:
    """An actual synchronous worker that remains alive after caller timeout."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.calls = []
        self.active = 0
        self.max_active = 0

    def semantic_scores(self, question, chunks, *, session_id):
        with self.lock:
            self.calls.append(question)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            if not self.release.wait(3):
                raise RuntimeError("Synthetic worker was not released")
            return {"scores": {row["id"]: 0.9 for row in chunks},
                    "trace": {"status": "ready", "provider_calls": 0,
                              "question": question}}
        finally:
            with self.lock:
                self.active -= 1


async def semantic_worker_idle(service):
    async def wait_until_idle():
        while service._semantic_scheduler.running:
            await asyncio.sleep(0.001)
    await asyncio.wait_for(wait_until_idle(), 1.0)


async def semantic_worker_started(semantic):
    async def wait_until_started():
        while not semantic.started.is_set():
            await asyncio.sleep(0.001)
    await asyncio.wait_for(wait_until_started(), 1.0)


def request(sid, question):
    return SimpleNamespace(session_id=sid, question=question, use_memory=False,
                           use_conversation_context=False, use_llm_cache=False,
                           use_self_evolution=False)


def workspace(root):
    manager = SessionManager(str(root / "_system"), str(root), migrate_legacy=False)
    sid = manager.create_session("Research")["id"]
    return manager, sid


def ready_document(service, sid):
    document = service.store.accept_document(sid, "source.txt", PASSAGE.encode())
    service.store.claim_document(sid, document["id"])
    service.store.complete_document(sid, document["id"], [{"text": PASSAGE}])
    return document


@pytest.mark.asyncio
async def test_qualified_cross_language_retrieval_cites_actual_quote_with_one_answer_call(tmp_path):
    manager, sid = workspace(tmp_path)
    semantic = QualifiedSemantic()
    service = ResearchService(tmp_path, semantic_retriever=semantic, context_assistance_enabled=False)
    ready_document(service, sid)
    model = AnswerModel()
    result = await service.answer(request(sid, "论文说明可再生能源投资怎样降低进口依赖？"), model, manager)
    assert len(model.prompts) == 1
    assert len(result["sources"]) == 1
    source = result["sources"][0]
    assert source["content"] == PASSAGE[source["excerpt_start"]:source["excerpt_end"]]
    assert source["label"] == "S1"
    assert result["citation_validation"]["valid"] is True
    assert result["citation_validation"]["entailment_verified"] is False
    assert PASSAGE in model.prompts[0]
    assert result["reasoning"]["trace"]["semantic_retrieval"]["provider_calls"] == 0
    hydrated = service.hydrate_history(sid, manager.get_history(sid))
    assert hydrated[-1]["sources"][0]["content"] == PASSAGE


@pytest.mark.asyncio
async def test_semantic_location_reaches_retrieval_and_assistance_with_exact_offsets(tmp_path):
    manager, sid = workspace(tmp_path)
    content = ("A background study measures ocean salinity across several stations. " * 22) + PASSAGE

    class LocatedSemantic(QualifiedSemantic):
        def semantic_scores(self, question, chunks, *, session_id):
            result = super().semantic_scores(question, chunks, session_id=session_id)
            result["spans"] = {row["id"]: [row["content"].index(PASSAGE), len(row["content"])] for row in chunks}
            return result

    service = ResearchService(tmp_path, semantic_retriever=LocatedSemantic())
    document = service.store.accept_document(sid, "source.txt", content.encode())
    service.store.claim_document(sid, document["id"])
    service.store.complete_document(sid, document["id"], [{"text": content}])
    from docthinker.server.research_context_assist import build_context_assist
    with patch("docthinker.server.research_service.build_context_assist", wraps=build_context_assist) as assist:
        result = await service.answer(request(sid, "论文说明可再生能源投资怎样降低进口依赖？"), AnswerModel(), manager)
    assert assist.call_args.kwargs["semantic_spans"]
    assert len(result["sources"]) == 1
    source = result["sources"][0]
    assert PASSAGE in source["content"]
    assert source["excerpt_start"] > 0
    assert source["content"] == content[source["excerpt_start"]:source["excerpt_end"]]
    assert source["excerpt_end"] == len(content)
    assert result["reasoning"]["trace"]["semantic_retrieval"]["provider_calls"] == 0


@pytest.mark.asyncio
async def test_embedding_failure_falls_back_lexically_without_leaking_exception(tmp_path):
    manager, sid = workspace(tmp_path)
    private_error = "/private/customer/source secret-document-phrase bearer-secret"
    semantic = QualifiedSemantic(failure=private_error)
    service = ResearchService(tmp_path, semantic_retriever=semantic, context_assistance_enabled=False)
    ready_document(service, sid)
    model = AnswerModel()
    result = await service.answer(request(sid, "According to the paper, how does renewable electricity reduce fossil fuel import dependence?"), model, manager)
    assert len(model.prompts) == 1
    assert result["sources"]
    trace = result["reasoning"]["trace"]["semantic_retrieval"]
    assert trace["status"] == "inference_unavailable"
    assert "semantic_retrieval_unavailable" in {item["code"] for item in result["retrieval_gaps"]}
    assert private_error not in json.dumps(result, ensure_ascii=False)
    assert private_error not in model.prompts[0]
    restored = ResearchService(tmp_path, semantic_retriever=None).hydrate_history(sid, manager.get_history(sid))[-1]
    assert "semantic_retrieval_unavailable" in {item["code"] for item in restored["retrieval_gaps"]}
    assert private_error not in json.dumps(restored, ensure_ascii=False)


@pytest.mark.asyncio
async def test_same_display_session_in_distinct_storage_roots_has_distinct_cache_namespace(tmp_path):
    semantic = QualifiedSemantic()
    seen = []
    for directory in (tmp_path / "tenant-a", tmp_path / "tenant-b"):
        manager, sid = workspace(directory)
        service = ResearchService(directory, semantic_retriever=semantic, context_assistance_enabled=False)
        ready_document(service, sid)
        await service.prepare_retrieval(sid)
        seen.append((sid, semantic.calls[-1][1]))
    assert seen[0][0] == seen[1][0]
    assert seen[0][1] != seen[1][1]
    assert all(str(tmp_path) not in namespace for _, namespace in seen)


@pytest.mark.asyncio
async def test_withdrawn_source_cannot_return_through_stale_semantic_candidate_cache(tmp_path):
    manager, sid = workspace(tmp_path)
    semantic = QualifiedSemantic(stale=True)
    service = ResearchService(tmp_path, semantic_retriever=semantic, context_assistance_enabled=False)
    document = ready_document(service, sid)
    first = await service.answer(request(sid, "论文说明能源投资怎样降低进口依赖？"), AnswerModel(), manager)
    assert first["sources"]
    service.store.delete_document(sid, document["id"])
    second_model = AnswerModel()
    second = await service.answer(request(sid, "论文说明能源投资怎样降低进口依赖？"), second_model, manager)
    assert second["sources"] == []
    assert second["answer_mode"] == "insufficient_evidence"
    assert second_model.prompts == []
    assert semantic.saved_scores  # The deliberately stale candidate was offered.
    old = next(item for item in service.hydrate_history(sid, manager.get_history(sid)) if item.get("id") == first["answer_id"])
    assert old["sources"] == []
    assert "source_revoked" in {warning["code"] for warning in old["citation_validation"]["warnings"]}


@pytest.mark.asyncio
async def test_missing_local_model_fallback_gap_survives_restart(tmp_path):
    manager, sid = workspace(tmp_path)
    service = ResearchService(tmp_path, semantic_retriever=LocalSemanticRetriever(tmp_path / "not-installed"),
                              context_assistance_enabled=False)
    ready_document(service, sid)
    model = AnswerModel()
    result = await service.answer(request(sid, "According to the paper, explain renewable electricity investment and import dependence."), model, manager)
    assert len(model.prompts) == 1 and result["sources"]
    assert result["reasoning"]["trace"]["semantic_retrieval"]["status"] == "model_not_installed"
    restored = ResearchService(tmp_path, semantic_retriever=None).hydrate_history(sid, manager.get_history(sid))[-1]
    assert "semantic_retrieval_unavailable" in {item["code"] for item in restored["retrieval_gaps"]}
    assert "not-installed" not in json.dumps(restored, ensure_ascii=False)


@pytest.mark.asyncio
async def test_ingestion_and_explicit_prepare_never_embed_or_cache_a_query(tmp_path):
    manager, sid = workspace(tmp_path)
    semantic = QualifiedSemantic()

    async def extractor(content, filename):
        return {"chunks": [{"text": content.decode()}]}

    service = ResearchService(tmp_path, semantic_retriever=semantic, extractor=extractor,
                              context_assistance_enabled=False)
    document = service.store.accept_document(sid, "source.txt", PASSAGE.encode())
    assert await service.process(sid, document["id"])
    trace = await service.prepare_retrieval(sid)
    assert [operation for operation, _, _ in semantic.calls] == ["prepare", "prepare"]
    assert trace["query_embedded"] is False
    assert trace["query_embedding_ms"] == 0.0
    model = AnswerModel()
    await service.answer(request(sid, "论文说明能源投资怎样降低进口依赖？"), model, manager)
    assert [operation for operation, _, _ in semantic.calls] == ["prepare", "prepare", "query"]
    assert len(model.prompts) == 1


@pytest.mark.asyncio
async def test_service_semantic_deadline_retains_one_actual_worker_then_recovers(tmp_path):
    semantic = BlockingSemantic()
    service = ResearchService(tmp_path, semantic_retriever=semantic)
    service._semantic_scheduler.timeout_seconds = 0.01
    chunks = [{"id": "c1", "document_id": "d1", "content": PASSAGE}]
    try:
        first_request = asyncio.create_task(service._semantic_operation("#00001", chunks, "slow first query"))
        await semantic_worker_started(semantic)
        first = await first_request
        assert first["trace"]["status"] == "timed_out"
        assert first["scores"] == {}
        assert service._semantic_scheduler.running
        responses = await asyncio.gather(*[
            service._semantic_operation("#00001", chunks, f"must not run {index}")
            for index in range(20)
        ])
        assert all(result["trace"]["status"] == "busy" and result["scores"] == {}
                   for result in responses)
        assert semantic.calls == ["slow first query"]
        assert semantic.active == semantic.max_active == 1
        # Semantic congestion does not occupy the evidence-organization lane.
        local = await service._context_scheduler.run(lambda: "independent local work")
        assert local.status == "completed"
        assert local.value == "independent local work"
        semantic.release.set()
        await semantic_worker_idle(service)
        recovered = await service._semantic_operation("#00001", chunks, "fresh query")
        assert recovered["trace"]["status"] == "ready"
        assert recovered["trace"]["question"] == "fresh query"
        assert recovered["scores"] == {"c1": 0.9}
        assert semantic.calls == ["slow first query", "fresh query"]
        assert semantic.max_active == 1
    finally:
        semantic.release.set()
        await semantic_worker_idle(service)
        await service.stop()


@pytest.mark.asyncio
async def test_service_stop_does_not_wait_for_or_reschedule_timed_out_semantic_worker(tmp_path):
    semantic = BlockingSemantic()
    service = ResearchService(tmp_path, semantic_retriever=semantic)
    service._semantic_scheduler.timeout_seconds = 0.01
    chunks = [{"id": "c1", "document_id": "d1", "content": PASSAGE}]
    try:
        first_request = asyncio.create_task(service._semantic_operation("#00001", chunks, "blocked query"))
        await semantic_worker_started(semantic)
        response = await first_request
        assert response["trace"]["status"] == "timed_out"
        await asyncio.wait_for(service.stop(), 0.1)
        assert service._semantic_scheduler.running
        assert semantic.active == 1
        closed = await service._semantic_operation("#00001", chunks, "after stop")
        assert closed["trace"]["status"] == "closed"
        assert closed["scores"] == {}
        assert semantic.calls == ["blocked query"]
    finally:
        semantic.release.set()
        await semantic_worker_idle(service)
        await service.stop()
    assert semantic.active == 0
    assert semantic.max_active == 1
