"""One real preparation worker survives waiter cancellation, never an API call."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest

from docthinker.server.research_service import ResearchService


SOURCES = [{"id": "synthetic-c1", "document_id": "synthetic-d1", "content": "Archive evidence."}]


class BlockingPreparation:
    def __init__(self, *, fail=False):
        self.entered, self.release, self.exited = threading.Event(), threading.Event(), threading.Event()
        self.calls = []
        self.fail = fail
        self.active = 0
        self.maximum = 0
        self.publications = []

    def prepare_documents(self, chunks, *, session_id):
        self.calls.append(session_id)
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        self.entered.set()
        try:
            assert self.release.wait(5)
            if self.fail:
                raise RuntimeError("private-source-api-secret-must-not-be-logged")
            self.publications.append(session_id)
            return {"scores": {}, "spans": {}, "trace": {"status": "ready", "provider_calls": 0, "llm_tokens": 0}}
        finally:
            self.active -= 1
            self.exited.set()


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(.001)
    await asyncio.wait_for(wait(), 1)


def subject(tmp_path, model):
    return ResearchService(tmp_path, semantic_retriever=model, evidence_reranker=None,
                           query_translator=None, context_assistance_enabled=False)


@pytest.mark.asyncio
async def test_cancelled_prepare_waiter_retains_real_worker_and_20_calls_do_not_queue(tmp_path):
    model = BlockingPreparation()
    service = subject(tmp_path, model)
    request = asyncio.create_task(service._semantic_operation("project", SOURCES))
    try:
        await until(model.entered.is_set)
        actual_task = service._semantic_prepare_task
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert service._semantic_prepare_task is actual_task and not actual_task.done()
        responses = await asyncio.gather(*[service._semantic_operation(f"project-{i}", SOURCES) for i in range(20)])
        assert all(response["trace"]["status"] == "busy" for response in responses)
        assert len(model.calls) == model.active == model.maximum == 1
        assert model.publications == []
    finally:
        model.release.set()
        await until(lambda: service._semantic_prepare_task is None)
        await service.stop()
    assert model.exited.is_set() and len(model.publications) == 1


@pytest.mark.asyncio
async def test_stop_waits_for_real_prepare_without_blocking_event_loop_or_late_publish(tmp_path):
    model = BlockingPreparation()
    service = subject(tmp_path, model)
    request = asyncio.create_task(service._semantic_operation("project", SOURCES))
    stop = None
    try:
        await until(model.entered.is_set)
        stop = asyncio.create_task(service.stop())
        await until(lambda: service._closing)
        heartbeat = 0
        for _ in range(8):
            await asyncio.sleep(.001)
            heartbeat += 1
        assert heartbeat == 8 and not stop.done()
        assert (await service._semantic_operation("after-stop", SOURCES))["trace"]["status"] == "closed"
        assert len(model.calls) == 1 and model.publications == []
        model.release.set()
        await asyncio.wait_for(stop, 1)
        await request
        assert model.exited.is_set() and model.active == 0 and service._semantic_prepare_task is None
        frozen_publications = list(model.publications)
        await asyncio.sleep(.005)
        assert model.publications == frozen_publications
    finally:
        model.release.set()
        await asyncio.gather(request, *([stop] if stop is not None else []), return_exceptions=True)
        await service.stop()


@pytest.mark.asyncio
async def test_shutdown_before_preparation_does_not_read_sources_or_submit_model(tmp_path, monkeypatch):
    model = BlockingPreparation()
    service = subject(tmp_path, model)
    await service.stop()
    monkeypatch.setattr(service, "all_chunks", lambda sid: pytest.fail("closed prep must not read the corpus"))
    receipt = await service.prepare_retrieval("project")
    assert receipt["status"] == receipt["asset_preparation"]["status"] == "closed"
    assert receipt["provider_calls"] == 0 and not receipt["query_embedded"]
    assert model.calls == []


@pytest.mark.asyncio
async def test_failed_worker_after_cancel_is_observed_no_retry_or_sensitive_logs(tmp_path, caplog):
    model = BlockingPreparation(fail=True)
    service = subject(tmp_path, model)
    request = asyncio.create_task(service._semantic_operation("project", SOURCES))
    try:
        await until(model.entered.is_set)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        model.release.set()
        await until(lambda: service._semantic_prepare_task is None)
        await service.stop()
        assert len(model.calls) == 1 and model.publications == []
        assert "private-source" not in caplog.text
    finally:
        model.release.set()
        await service.stop()


@pytest.mark.asyncio
async def test_foreground_prepare_error_is_sanitized_and_slot_recovers(tmp_path):
    model = BlockingPreparation(fail=True)
    model.release.set()
    service = subject(tmp_path, model)
    first = await service._semantic_operation("project", SOURCES)
    assert first["trace"]["status"] == "inference_unavailable"
    assert "private-source" not in json.dumps(first)
    await until(lambda: service._semantic_prepare_task is None)
    model.fail = False
    second = await service._semantic_operation("other-project", SOURCES)
    assert second["trace"]["status"] == "ready" and len(model.calls) == 2
    assert model.maximum == 1
    await service.stop()


@pytest.mark.asyncio
async def test_task_queued_in_shared_pool_still_owns_one_preparation_slot(tmp_path):
    model = BlockingPreparation()
    model.release.set()
    service = subject(tmp_path, model)
    loop = asyncio.get_running_loop()
    # Keep the event loop's standard executor but temporarily admit one thread.
    # No production executor or additional lane is introduced by this fixture.
    original = loop._default_executor
    pool = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(pool)
    entered, release = threading.Event(), threading.Event()
    def blocker():
        entered.set()
        assert release.wait(5)
    active = loop.run_in_executor(None, blocker)
    request = None
    try:
        await until(entered.is_set)
        request = asyncio.create_task(service._semantic_operation("project", SOURCES))
        await until(lambda: service._semantic_prepare_task is not None)
        assert not model.entered.is_set()
        receipt = await service._semantic_operation("other-project", SOURCES)
        assert receipt["trace"]["status"] == "busy" and model.calls == []
        release.set()
        await active
        assert (await request)["trace"]["status"] == "ready"
        assert len(model.calls) == model.maximum == 1
        await service.stop()
    finally:
        release.set()
        await asyncio.gather(active, *([request] if request is not None else []), return_exceptions=True)
        loop._default_executor = original
        pool.shutdown(wait=True)


@pytest.mark.asyncio
async def test_cancelled_upload_does_not_change_ready_document_or_duplicate_warm(tmp_path):
    from docthinker.session_manager import SessionManager
    model = BlockingPreparation()
    async def extract(content, filename):
        return {"chunks": [{"text": content.decode()}]}
    service = ResearchService(tmp_path, semantic_retriever=model, extractor=extract,
                              evidence_reranker=None, query_translator=None,
                              context_assistance_enabled=False)
    manager = SessionManager(str(tmp_path / "_system"), str(tmp_path), migrate_legacy=False)
    sid = manager.create_session("Synthetic")['id']
    document = service.store.accept_document(sid, "synthetic.txt", b"Archive evidence.")
    job = asyncio.create_task(service.process(sid, document["id"]))
    try:
        await until(model.entered.is_set)
        assert service.store.get_document(sid, document["id"])["status"] == "ready"
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        assert not await service.process(sid, document["id"])
        assert len(model.calls) == 1 and service._semantic_prepare_task is not None
        assert service.store.get_document(sid, document["id"])["status"] == "ready"
    finally:
        model.release.set()
        await service.stop()
    assert model.active == 0 and len(model.publications) == 1
