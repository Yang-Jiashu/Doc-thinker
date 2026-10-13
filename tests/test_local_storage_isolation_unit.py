"""Exercise real file stores; identical logical namespaces must not share files' state."""

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from graphcore.coregraph import GraphCore
from graphcore.coregraph.kg.json_doc_status_impl import JsonDocStatusStorage
from graphcore.coregraph.kg.json_kv_impl import JsonKVStorage
from graphcore.coregraph.kg.nano_vector_db_impl import NanoVectorDBStorage
from graphcore.coregraph.kg.networkx_impl import NetworkXStorage
from graphcore.coregraph.kg.shared_storage import (
    finalize_share_data,
    initialize_share_data,
)
from graphcore.coregraph.utils import EmbeddingFunc, Tokenizer


@pytest.fixture(autouse=True)
def isolated_shared_storage():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


@pytest.mark.parametrize("storage_class", [JsonKVStorage, JsonDocStatusStorage])
@pytest.mark.parametrize("workspace", ["", "session_00001"])
async def test_dedup_and_persistence_are_isolated_by_directory(
    tmp_path, storage_class, workspace
):
    def create(directory):
        return storage_class(
            namespace="doc_status" if storage_class is JsonDocStatusStorage else "full_docs",
            workspace=workspace,
            global_config={"working_dir": str(tmp_path / directory)},
            embedding_func=None,
        )

    first, second = create("session2"), create("session3")
    await first.initialize()
    await second.initialize()
    doc_id = "identical-content-hash"
    await first.upsert({doc_id: {"content": "same document", "status": "processed"}})

    assert await first.filter_keys({doc_id}) == set()
    assert await second.filter_keys({doc_id}) == {doc_id}
    assert await second.get_by_ids([doc_id]) == [None]

    await second.upsert({doc_id: {"content": "same document", "status": "pending"}})
    await first.index_done_callback()
    await second.index_done_callback()
    assert json.loads(Path(first._file_name).read_text())[doc_id]["status"] == "processed"
    assert json.loads(Path(second._file_name).read_text())[doc_id]["status"] == "pending"

    # A fresh process must read each directory, including the old flat layout.
    finalize_share_data()
    initialize_share_data()
    reopened_first, reopened_second = create("session2"), create("session3")
    await reopened_first.initialize()
    await reopened_second.initialize()
    assert (await reopened_first.get_by_ids([doc_id]))[0]["status"] == "processed"
    assert (await reopened_second.get_by_ids([doc_id]))[0]["status"] == "pending"
    assert await reopened_second.filter_keys({doc_id}) == set()


async def test_same_physical_directory_still_shares_dedup_state(tmp_path):
    alias = tmp_path / "alias"
    directory = tmp_path / "original"
    directory.mkdir()
    alias.symlink_to(directory, target_is_directory=True)
    stores = [
        JsonKVStorage(
            namespace="full_docs", workspace="", embedding_func=None,
            global_config={"working_dir": str(path)},
        )
        for path in (directory, alias)
    ]
    for store in stores:
        await store.initialize()
    await stores[0].upsert({"doc": {"content": "same document"}})
    assert await stores[1].filter_keys({"doc"}) == set()


async def _embed(texts, **_kwargs):
    return np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (len(texts), 1))


@pytest.mark.parametrize(
    "storage_class",
    [NetworkXStorage, NanoVectorDBStorage, pytest.param("faiss", id="FaissVectorDBStorage")],
)
async def test_other_directory_commit_cannot_discard_unsaved_changes(tmp_path, storage_class):
    if storage_class == "faiss":
        pytest.importorskip("faiss")
        from graphcore.coregraph.kg.faiss_impl import FaissVectorDBStorage
        storage_class = FaissVectorDBStorage
    stores = [
        storage_class(
            namespace="entities", workspace="",
            embedding_func=EmbeddingFunc(embedding_dim=4, max_token_size=100, func=_embed),
            global_config={
                "working_dir": str(tmp_path / session), "embedding_batch_num": 4,
                "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
            },
        )
        for session in ("session2", "session3")
    ]
    for store in stores:
        await store.initialize()
    first, second = stores
    if storage_class is NetworkXStorage:
        await first.upsert_node("first", {"description": "first"})
        await second.upsert_node("second", {"description": "second"})
    else:
        await first.upsert({"first": {"content": "first"}})
        await second.upsert({"second": {"content": "second"}})
    # Committing first used to signal a reload of second from its still-empty file.
    await first.index_done_callback()
    if storage_class is NetworkXStorage:
        assert await second.has_node("second")
        assert not await second.has_node("first")
    else:
        assert await second.get_by_id("second") is not None
        assert await second.get_by_id("first") is None
    await second.index_done_callback()


class _CharacterTokenizer:
    def encode(self, text):
        return list(map(ord, text))

    def decode(self, tokens):
        return "".join(map(chr, tokens))


def _graphcore(directory, llm):
    return GraphCore(
        working_dir=str(directory), workspace="", llm_model_func=llm,
        embedding_func=EmbeddingFunc(embedding_dim=4, max_token_size=1000, func=_embed),
        tokenizer=Tokenizer("test-characters", _CharacterTokenizer()),
        entity_extract_max_gleaning=0,
    )


async def test_concurrent_same_document_builds_both_session_graphs_and_survives_restart(tmp_path):
    calls = 0
    both_started = asyncio.Event()

    async def llm(_prompt, **_kwargs):
        nonlocal calls
        calls += 1
        if calls >= 2:
            both_started.set()
        # Both independent pipelines must run; a shared busy flag strands session 3.
        await asyncio.wait_for(both_started.wait(), timeout=5)
        return (
            "entity<|#|>Solar<|#|>concept<|#|>Solar supplies domestic energy.\n"
            "entity<|#|>Energy<|#|>concept<|#|>Energy is supplied by Solar.\n"
            "relation<|#|>Solar<|#|>Energy<|#|>supply<|#|>Solar supplies Energy.\n"
            "<|COMPLETE|>"
        )

    stores = [_graphcore(tmp_path / session, llm) for session in ("session2", "session3")]
    for store in stores:
        await store.initialize_storages()
    try:
        await asyncio.wait_for(asyncio.gather(*(
            store.ainsert("Solar supplies Energy.") for store in stores
        )), timeout=15)
        for store in stores:
            assert await store.chunk_entity_relation_graph.has_node("Solar")
            assert await store.chunk_entity_relation_graph.has_edge("Solar", "Energy")
            assert (await store.doc_status.get_status_counts())["processed"] == 1
        assert calls == 2
        await stores[1].ainsert("Solar supplies Energy.")
        assert calls == 2
    finally:
        for store in stores:
            await store.finalize_storages()

    finalize_share_data()
    initialize_share_data()
    reopened = [_graphcore(tmp_path / session, llm) for session in ("session2", "session3")]
    try:
        for store in reopened:
            await store.initialize_storages()
            assert await store.chunk_entity_relation_graph.has_edge("Solar", "Energy")
            assert (await store.doc_status.get_status_counts())["processed"] == 1
            await store.ainsert("Solar supplies Energy.")
        assert calls == 2
    finally:
        for store in reopened:
            await store.finalize_storages()
