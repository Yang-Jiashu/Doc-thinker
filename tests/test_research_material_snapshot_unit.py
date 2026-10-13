"""Live source reads and corpus fallback retain the original evidence envelope."""
import asyncio
from contextlib import contextmanager

import pytest

from docthinker.server.research_corpus_index import PreparedCorpusCache, CorpusIndexTooLarge
from docthinker.server.research_service import ResearchService
from tests.test_research_store_unit import ResearchStore, ready
from tests.test_research_service_unit import SessionManager, FakeModel, text_extractor, request


def pages(store, sid):
    rows = []
    for offset in range(0, 12000, 600):
        page = store.source_chunks(sid, limit=600, offset=offset)
        rows.extend(page)
        if len(page) < 600:
            break
    return rows


def test_single_snapshot_matches_paginated_shape_order_and_live_scope(tmp_path, monkeypatch):
    store = ResearchStore(tmp_path)
    docs = [ready(store, content=f'file {n}'.encode(), filename=f'{n}.txt', chunks=[
        {'text': f'Row {n}-{i}. Renewable supply reduces imports under conditions.', 'page': i + 1}
        for i in range(600)])[0] for n in range(3)]
    other, _ = ready(store, session='#00002', content=b'OTHER_USER_PRIVATE_SOURCE')
    queued = store.accept_document('#00001', 'queued.txt', b'Not parsed yet')
    expected = pages(store, '#00001')
    connections, selects = [], []
    original = store._read
    @contextmanager
    def observed():
        with original() as db:
            connections.append(1)
            if db:
                db.set_trace_callback(lambda sql: selects.append(sql) if sql.lstrip().upper().startswith('SELECT') else None)
            yield db
    monkeypatch.setattr(store, '_read', observed)
    assert store.all_source_chunks('#00001') == expected
    assert len(connections) == 1 and len(selects) == 1
    assert store.all_source_chunks('#00001', document_ids=[docs[1]['id']]) == [
        row for row in expected if row['document_id'] == docs[1]['id']]
    assert store.all_source_chunks('#00001', document_ids=[other['id']]) == []
    assert store.all_source_chunks('#00001', document_ids=[queued['id']]) == []
    store.delete_document('#00001', docs[1]['id'])
    assert store.all_source_chunks('#00001', document_ids=[docs[1]['id']]) == []


def test_empty_snapshot_read_never_creates_a_database(tmp_path):
    store = ResearchStore(tmp_path)
    assert store.all_source_chunks('#00001') == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('scope', [[], ['a', 'a'], [None], 'a', list(str(n) for n in range(21))])
def test_bad_scope_cannot_expand_into_a_whole_project(tmp_path, scope):
    store = ResearchStore(tmp_path)
    with pytest.raises(ValueError):
        store.all_source_chunks('#00001', document_ids=scope)


def test_real_service_memory_budget_fallback_keeps_full_evidence_and_model_count(tmp_path):
    async def scenario():
        manager = SessionManager(str(tmp_path / '_system'), str(tmp_path), migrate_legacy=False)
        sid = manager.create_session('Live source features')['id']
        service = ResearchService(tmp_path, extractor=text_extractor, context_assistance_enabled=False)
        model = FakeModel()
        try:
            doc = service.store.accept_document(sid, 'source.txt', b'Renewable supply reduces energy imports under conditions.')
            await service.process(sid, doc['id'])
            await asyncio.to_thread(service._corpus_cache.get, service.all_chunks(sid), namespace=sid)
            indexed = await service.answer(request(sid), model, manager)
            service._corpus_cache = PreparedCorpusCache(max_bytes=1)
            with pytest.raises(CorpusIndexTooLarge):
                service._corpus_cache.get(service.all_chunks(sid), namespace=sid)
            fallback = await service.answer(request(sid), model, manager)
            assert indexed['sources'] == fallback['sources']
            assert fallback['reasoning']['trace']['lexical_features']['status'] == 'full_scan_memory_budget'
            assert indexed['reasoning']['trace']['lexical_features']['status'] == 'indexed'
            assert len(model.prompts) == 2 and model.prompts[0] == model.prompts[1]
            await service.stop()
            assert service._corpus_cache.info()['snapshots'] == 0
        finally:
            await service.stop()
    asyncio.run(scenario())


def test_oversized_snapshot_hint_prevents_rebuilding_but_rechecks_live_content(monkeypatch):
    import docthinker.server.research_corpus_index as module
    cache, builds = PreparedCorpusCache(max_bytes=1), []
    original = module._prepare
    def observed(*args):
        builds.append(1)
        return original(*args)
    monkeypatch.setattr(module, '_prepare', observed)
    rows = [{'id': 'a', 'document_id': 'b', 'content': 'Renewable supply.'}]
    for _ in range(2):
        with pytest.raises(CorpusIndexTooLarge):
            cache.get(rows, namespace='#00001')
    assert len(builds) == 1
    rows[0]['content'] = 'Changed source.'
    with pytest.raises(CorpusIndexTooLarge):
        cache.get(rows, namespace='#00001')
    assert len(builds) == 2
    cache.clear()
    with pytest.raises(CorpusIndexTooLarge):
        cache.get(rows, namespace='#00001')
    assert len(builds) == 3
