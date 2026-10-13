"""Independent deferred lexical preparation checks; no provider/network calls.

Event gates prove ordering rather than asserting machine-dependent speedups.
All service data is temporary and every actual background thread is released.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import pytest

from docthinker.session_manager import SessionManager
from docthinker.server import research_corpus_index as corpus
from docthinker.server.research_extract import extract_chunks
from docthinker.server.research_service import ResearchService


def row(**extra):
    return {"id": "c1", "document_id": "d1", "filename": "synthetic.txt", "page": 1,
            "content": "The inventory archive preserves approved revisions.", **extra}


def request(sid, question="How does the inventory archive preserve revisions?"):
    return SimpleNamespace(session_id=sid, question=question, use_memory=False,
                           use_conversation_context=False, use_llm_cache=False,
                           use_self_evolution=False, knowledge_skill_mode="off", insight_mode="off")


class FakeModel:
    def __init__(self):
        self.calls = []

    async def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return "The synthetic inventory archive preserves approved revisions [S1]."


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    manager = SessionManager(str(tmp_path / "_system"), str(tmp_path), migrate_legacy=False)
    sid = manager.create_session("Synthetic warm-index audit")["id"]

    async def extract(content, filename):
        return {"chunks": extract_chunks(content, ".txt")}

    service = ResearchService(tmp_path, extractor=extract, context_assistance_enabled=False,
                              semantic_retriever=None, evidence_reranker=None, query_translator=None)
    return service, manager, sid


async def add_document(service, sid, *, filename="synthetic.txt", content=None):
    text = content or "The inventory archive preserves approved revisions only after validation."
    document = service.store.accept_document(sid, filename, text.encode())
    assert await service.process(sid, document["id"])
    return document


async def wait_event(event):
    async def waiting():
        while not event.is_set():
            await asyncio.sleep(.001)
    await asyncio.wait_for(waiting(), 2)


async def wait_warm(service):
    task = service._corpus_warm_task
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), 2)
    await asyncio.sleep(0)


def block_builder(monkeypatch):
    original = corpus._prepare
    entered, release = threading.Event(), threading.Event()
    calls = []

    def prepare(*args, **kwargs):
        calls.append(args[1])
        entered.set()
        assert release.wait(5), "synthetic builder gate must be released"
        return original(*args, **kwargs)

    monkeypatch.setattr(corpus, "_prepare", prepare)
    return entered, release, calls


def test_lookup_miss_does_not_build_or_cache_unknown_corpus(monkeypatch):
    cache = corpus.PreparedCorpusCache()
    monkeypatch.setattr(corpus, "_prepare", lambda *a, **k: pytest.fail("lookup must not prepare"))
    assert cache.lookup([row()], namespace="tenant::#00001") is None
    assert cache.lookup([], namespace="tenant::#00001") is None
    assert cache.info()["snapshots"] == cache.info()["feature_bytes"] == 0


def test_lookup_never_waits_behind_actual_cold_builder(monkeypatch):
    cache = corpus.PreparedCorpusCache()
    entered, release, _ = block_builder(monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        building = pool.submit(cache.get, [row()], namespace="tenant")
        try:
            assert entered.wait(2)
            checking = pool.submit(cache.lookup, [row()], namespace="tenant")
            # The producer gate remains closed: a blocking cache lock cannot pass.
            assert checking.result(timeout=.5) is None
            assert not building.done()
        finally:
            release.set()
        prepared = building.result(timeout=2)
    assert cache.lookup([row()], namespace="tenant") is prepared


def test_lookup_rechecks_live_membership_and_every_presented_locator():
    cache = corpus.PreparedCorpusCache()
    original = row()
    prepared = cache.get([original], namespace="tenant::#00001")
    assert cache.lookup([dict(original)], namespace="tenant::#00001") is prepared
    for current in ([], [row(content="An updated inventory archive removes revisions.")],
                    [row(filename="renamed.txt")], [row(page=4)], [row(is_generated=True)],
                    [row(document_id="different-document")], [row(ordinal=2)]):
        assert cache.lookup(current, namespace="tenant::#00001") is None
    assert cache.lookup([original], namespace="another-tenant::#00001") is None


def test_close_waits_for_builder_then_forbids_late_publication(monkeypatch):
    cache = corpus.PreparedCorpusCache()
    entered, release, _ = block_builder(monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        building = pool.submit(cache.get, [row()], namespace="tenant")
        try:
            assert entered.wait(2)
            closing = pool.submit(cache.close)
            assert not closing.done()
        finally:
            release.set()
        building.result(timeout=2)
        closing.result(timeout=2)
    assert cache.info()["snapshots"] == cache.info()["feature_bytes"] == 0
    with pytest.raises(ValueError, match="closed"):
        cache.get([row()], namespace="tenant")
    with pytest.raises(ValueError, match="closed"):
        cache.lookup([row()], namespace="tenant")
    assert cache.info()["snapshots"] == 0


@pytest.mark.asyncio
async def test_first_and_second_answers_do_not_wait_or_enqueue_while_warming(workspace, monkeypatch):
    service, manager, sid = workspace
    await add_document(service, sid)
    entered, release, builds = block_builder(monkeypatch)
    original_get = service._corpus_cache.get
    jobs = []

    def get(*args, **kwargs):
        jobs.append(kwargs["namespace"])
        return original_get(*args, **kwargs)

    monkeypatch.setattr(service._corpus_cache, "get", get)
    model = FakeModel()
    try:
        first = await asyncio.wait_for(service.answer(request(sid), model, manager), 1)
        await wait_event(entered)
        actual_task = service._corpus_warm_task
        assert actual_task is not None and not actual_task.done()
        # This delay models a slow build; foreground completion never waits for it.
        await asyncio.sleep(.06)
        second = await asyncio.wait_for(service.answer(request(sid), model, manager), 1)
        assert service._corpus_warm_task is actual_task and not actual_task.done()
        assert len(model.calls) == 2 and len(jobs) == len(builds) == 1
        assert first["sources"] == second["sources"]
        assert first["reasoning"]["trace"]["lexical_features"]["status"] == "cold_full_scan"
        assert second["reasoning"]["trace"]["lexical_features"]["status"] == "cold_full_scan"
        release.set()
        await wait_warm(service)
        third = await service.answer(request(sid), model, manager)
        assert third["reasoning"]["trace"]["lexical_features"]["status"] == "indexed"
        assert third["sources"] == first["sources"]
        assert len(model.calls) == 3 and len(jobs) == 1
    finally:
        release.set()
        await service.stop()


@pytest.mark.asyncio
async def test_scope_change_cannot_use_finished_old_warm_index(workspace):
    service, manager, sid = workspace
    first_doc = await add_document(service, sid)
    second_doc = await add_document(service, sid, filename="second.txt",
        content="The inventory archive validates revisions using signed version identifiers.")
    model = FakeModel()
    try:
        first = await service.answer(request(sid), model, manager, document_ids=[first_doc["id"]])
        await wait_warm(service)
        second = await service.answer(request(sid), model, manager, document_ids=[second_doc["id"]])
        assert first["sources"] and second["sources"]
        assert {source["document_id"] for source in second["sources"]} == {second_doc["id"]}
        assert second["reasoning"]["trace"]["lexical_features"]["status"] == "cold_full_scan"
        assert len(model.calls) == 2
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_source_withdrawal_cannot_be_resurrected_by_late_warmup(workspace, monkeypatch):
    service, manager, sid = workspace
    document = await add_document(service, sid)
    entered, release, _ = block_builder(monkeypatch)
    model = FakeModel()
    try:
        await service.answer(request(sid), model, manager)
        await wait_event(entered)
        service.store.delete_document(sid, document["id"])
        release.set()
        await wait_warm(service)
        current = service.all_chunks(sid)
        assert current == [] and service._corpus_cache.lookup(current, namespace=sid) is None
        unavailable = await service.answer(request(sid, "What does the uploaded paper say about inventory revisions?"), model, manager)
        assert unavailable["sources"] == []
        assert unavailable["answer_mode"] == "insufficient_evidence"
        assert len(model.calls) == 1
    finally:
        release.set()
        await service.stop()


@pytest.mark.asyncio
async def test_service_stop_waits_for_real_worker_without_blocking_event_loop(workspace, monkeypatch):
    service, _, sid = workspace
    entered, release, _ = block_builder(monkeypatch)
    assert service._warm_corpus(sid, [row()]) is True
    await wait_event(entered)
    stopping = asyncio.create_task(service.stop())
    try:
        await asyncio.sleep(.02)
        assert not stopping.done()
        # The event loop is alive while cache.close waits on a real worker lock.
        ticks = []
        for _ in range(3):
            await asyncio.sleep(.002)
            ticks.append(True)
        assert len(ticks) == 3 and not stopping.done()
        assert service._warm_corpus(sid, [row()]) is False
    finally:
        release.set()
        await asyncio.wait_for(stopping, 2)
    assert service._corpus_cache.info()["snapshots"] == 0
    assert service._corpus_warm_task is None or service._corpus_warm_task.done()
    assert service._warm_corpus(sid, [row()]) is False


@pytest.mark.asyncio
async def test_oversized_background_features_do_not_retry_or_reduce_answer_sources(workspace, monkeypatch):
    service, manager, sid = workspace
    await add_document(service, sid)
    service._corpus_cache = corpus.PreparedCorpusCache(max_bytes=1)
    original = corpus._prepare
    builds = []

    def prepare(*args, **kwargs):
        builds.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(corpus, "_prepare", prepare)
    model = FakeModel()
    try:
        first = await service.answer(request(sid), model, manager)
        task = service._corpus_warm_task
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        second = await service.answer(request(sid), model, manager)
        await asyncio.sleep(0)
        assert first["sources"] and first["sources"] == second["sources"]
        assert second["reasoning"]["trace"]["lexical_features"]["status"] == "full_scan_memory_budget"
        assert len(builds) == 1 and len(model.calls) == 2
        assert service._corpus_cache.info()["snapshots"] == 0
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_empty_corpus_does_not_prepare_or_spend_provider_call(workspace, monkeypatch):
    service, manager, sid = workspace
    model = FakeModel()
    monkeypatch.setattr(corpus, "_prepare", lambda *a, **k: pytest.fail("no live corpus to prepare"))
    try:
        answer = await service.answer(request(sid, "What does the uploaded paper say?"), model, manager)
        assert answer["answer_mode"] == "insufficient_evidence"
        assert answer["sources"] == [] and model.calls == []
        assert service._corpus_warm_task is None
    finally:
        await service.stop()
