"""Asset-only preparation: no queries, inference, new executors, or hidden cost."""
from __future__ import annotations

import asyncio
import json
import threading
from unittest.mock import Mock

import pytest

from docthinker.server import research_query_translation as qt
from docthinker.server import research_reranker as rr
from docthinker.server.research_scheduler import BackgroundContextScheduler
from docthinker.server.research_service import ResearchService
from docthinker.session_manager import SessionManager


@pytest.fixture(autouse=True)
def isolated_caches(monkeypatch):
    monkeypatch.setattr(rr, '_MODELS', rr.OrderedDict())
    monkeypatch.setattr(rr, '_SCORES', rr.OrderedDict())
    monkeypatch.setattr(qt, '_CACHE', qt.OrderedDict())
    monkeypatch.setattr(rr, '_INFERENCE_GATE', threading.BoundedSemaphore(1))
    monkeypatch.setattr(qt, '_INFERENCE_GATE', threading.BoundedSemaphore(1))


def fail_private():
    raise RuntimeError('private-api-key/path/question')


def test_reranker_prepare_only_validates_and_loads_assets_once(monkeypatch, tmp_path):
    model, tokenizer = Mock(), Mock()
    manifest = Mock(return_value=({}, 'asset-fingerprint'))
    loader = Mock(return_value=(model, tokenizer))
    monkeypatch.setattr(rr, '_model_manifest', manifest)
    monkeypatch.setattr(rr, '_load_model', loader)
    subject = rr.LocalEvidenceReranker(tmp_path)
    first = subject.prepare_assets()['trace']
    second = subject.prepare_assets()['trace']
    assert first['status'] == second['status'] == 'ready'
    assert first['model_reused'] is False and second['model_reused'] is True
    manifest.assert_called_once_with(tmp_path)
    loader.assert_called_once_with(tmp_path)
    assert not model.mock_calls and not tokenizer.mock_calls and not rr._SCORES
    assert first['asset_only'] and not first['query_inference']
    assert first['provider_calls'] == first['provider_tokens'] == 0
    assert first['total_ms'] >= first['initialization_ms'] >= 0


def test_translator_prepare_loads_runtime_once_without_encoding_or_query_cache(monkeypatch, tmp_path):
    subject = qt.LocalQueryTranslator(tmp_path)
    runtime = (Mock(), Mock(), Mock())
    loader = Mock(return_value=runtime)
    monkeypatch.setattr(subject, '_load_runtime', loader)
    first = subject.prepare_assets()['trace']
    second = subject.prepare_assets()['trace']
    loader.assert_called_once_with()
    assert subject._runtime is runtime and all(not item.mock_calls for item in runtime)
    assert not qt._CACHE and first['status'] == second['status'] == 'ready'
    assert first['model_reused'] is False and second['model_reused'] is True
    assert first['asset_only'] and not first['query_inference']
    assert first['provider_calls'] == first['llm_tokens'] == 0


@pytest.mark.parametrize('module,kind,loader', [
    (rr, rr.LocalEvidenceReranker, '_initialize'), (qt, qt.LocalQueryTranslator, '_load_runtime')])
def test_asset_busy_fast_skip_never_initializes(monkeypatch, module, kind, loader):
    subject = kind()
    operation = Mock(side_effect=AssertionError('must not initialize'))
    monkeypatch.setattr(subject, loader, operation)
    assert module._INFERENCE_GATE.acquire(blocking=False)
    try:
        assert subject.prepare_assets()['trace']['status'] == 'busy'
        operation.assert_not_called()
    finally:
        module._INFERENCE_GATE.release()


@pytest.mark.parametrize('module,kind,loader', [
    (rr, rr.LocalEvidenceReranker, '_initialize'), (qt, qt.LocalQueryTranslator, '_load_runtime')])
def test_asset_failure_is_sanitized_and_releases_gate(monkeypatch, module, kind, loader):
    subject = kind()
    monkeypatch.setattr(subject, loader, fail_private)
    receipt = subject.prepare_assets()
    assert receipt['trace']['status'] == 'unavailable'
    assert 'private' not in json.dumps(receipt)
    assert module._INFERENCE_GATE.acquire(blocking=False)
    module._INFERENCE_GATE.release()


@pytest.mark.parametrize('module,kind,loader', [
    (rr, rr.LocalEvidenceReranker, '_initialize'), (qt, qt.LocalQueryTranslator, '_load_runtime')])
def test_loading_asset_holds_actual_inference_gate(monkeypatch, module, kind, loader):
    entered, release = threading.Event(), threading.Event()
    subject = kind()

    def blocked():
        entered.set()
        assert release.wait(2)
        return (Mock(), Mock(), Mock())

    monkeypatch.setattr(subject, loader, blocked)
    worker = threading.Thread(target=subject.prepare_assets)
    worker.start()
    try:
        assert entered.wait(2)
        if module is rr:
            result = kind().rank('Question', [{'id': 'a', 'content': 'Evidence'}], session_id='project')
        else:
            result = kind().translate('测试问题', session_id='project')
        assert result['trace']['status'] == 'busy'
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()


class Semantic:
    def __init__(self):
        self.calls = []

    def prepare_documents(self, chunks, *, session_id):
        self.calls.append((chunks, session_id))
        return {'trace': {'status': 'ready', 'provider_calls': 0, 'llm_tokens': 0}}


class Assets:
    def __init__(self, callback=None):
        self.calls = 0
        self.callback = callback

    def prepare_assets(self):
        self.calls += 1
        if self.callback:
            return self.callback()
        return {'trace': {'status': 'ready', 'model_reused': False, 'initialization_ms': 1.0, 'total_ms': 1.0}}


def service(tmp_path, **kwargs):
    subject = ResearchService(tmp_path, semantic_retriever=Semantic(),
                              evidence_reranker=kwargs.pop('evidence_reranker', Assets()),
                              query_translator=kwargs.pop('query_translator', Assets()), **kwargs)
    return subject


@pytest.mark.asyncio
async def test_disabled_assistance_does_not_prepare_assets_but_vectors_still_prepare(tmp_path, monkeypatch):
    subject = service(tmp_path, context_assistance_enabled=False)
    monkeypatch.setattr(subject, 'all_chunks', lambda sid: [{'id': 'source'}])
    trace = await subject.prepare_retrieval('project')
    assert len(subject._semantic.calls) == 1
    assert subject._evidence_reranker.calls == subject._query_translator.calls == 0
    assert trace['status'] == 'ready' and trace['asset_preparation']['status'] == 'disabled'
    assert not trace['query_embedded'] and trace['preparation_wait_ms'] >= trace['vector_preparation_ms']
    assert subject._evidence_scheduler.timeout_seconds == 8.


@pytest.mark.asyncio
async def test_empty_corpus_does_not_load_assets(tmp_path):
    subject = service(tmp_path)
    trace = await subject.prepare_retrieval('#00001')
    assert trace['asset_preparation']['status'] == 'no_live_documents'
    assert subject._evidence_reranker.calls == subject._query_translator.calls == 0


@pytest.mark.asyncio
async def test_asset_stages_share_existing_worker_run_sequentially_and_report_cost(tmp_path):
    events = []
    subject = service(tmp_path)
    scheduler = subject._evidence_scheduler
    owner = threading.get_ident()

    def prepare(name):
        assert scheduler.running
        assert threading.get_ident() != owner
        events.append((name, threading.get_ident()))
        return {'trace': {'status': 'ready', 'initialization_ms': 1., 'total_ms': 1.,
                          'private': 'private-api-key/path/question'}}

    subject._evidence_reranker.callback = lambda: prepare('reranker')
    subject._query_translator.callback = lambda: prepare('translator')
    result = await subject.prepare_assistance_assets()
    assert [name for name, _ in events] == ['reranker', 'translator']
    assert len({thread for _, thread in events}) == 1
    assert result['status'] == 'completed' and result['wait_ms'] >= result['worker_ms'] >= 0
    assert not result['query_inference'] and not result['worker_running']
    assert result['provider_calls'] == 0 and 'private' not in json.dumps(result)
    assert subject._evidence_scheduler is scheduler
    assert result['scheduler_timing']['worker_finished']
    assert result['scheduler_timing']['worker_ms'] >= result['worker_ms']


@pytest.mark.asyncio
async def test_busy_model_does_not_queue_other_asset_stage(tmp_path):
    subject = service(tmp_path, evidence_reranker=Assets(lambda: {'trace': {'status': 'busy'}}))
    result = await subject.prepare_assistance_assets()
    assert result['components'] == {'reranker': {'status': 'busy'}}
    assert subject._query_translator.calls == 0


@pytest.mark.asyncio
async def test_timeout_keeps_actual_worker_slot_and_busy_reentry_skips(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def blocked():
        entered.set()
        assert release.wait(2)
        return {'trace': {'status': 'ready'}}

    subject = service(tmp_path, evidence_reranker=Assets(blocked))
    subject._evidence_scheduler = BackgroundContextScheduler(timeout_seconds=.02)
    try:
        result = await subject.prepare_assistance_assets()
        assert entered.is_set() and result['status'] == 'timed_out'
        assert result['worker_running'] and result['worker_ms'] is None
        second = await subject.prepare_assistance_assets()
        assert second['status'] == 'busy' and subject._evidence_reranker.calls == 1
        assert subject._query_translator.calls == 0
    finally:
        release.set()
        while subject._evidence_scheduler.running:
            await asyncio.sleep(.001)
    assert subject._query_translator.calls == 1  # same real worker, not a new job


@pytest.mark.asyncio
async def test_preparation_failure_never_downgrades_ready_document(tmp_path):
    manager = SessionManager(str(tmp_path / '_system'), str(tmp_path), migrate_legacy=False)
    sid = manager.create_session('Private research')['id']

    async def extract(content, filename):
        return {'chunks': [{'text': 'Source evidence.'}]}

    subject = service(tmp_path, extractor=extract, evidence_reranker=Assets(fail_private))
    document = subject.store.accept_document(sid, 'paper.txt', b'Source evidence.')
    assert await subject.process(sid, document['id'])
    assert subject.store.get_document(sid, document['id'])['status'] == 'ready'
    assert subject._evidence_reranker.calls == subject._query_translator.calls == 1
    receipt = await subject.prepare_assistance_assets()
    assert receipt['components']['reranker']['status'] == 'unavailable'
    assert 'private-api-key' not in json.dumps(receipt)
