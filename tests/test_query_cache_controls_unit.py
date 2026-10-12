"""Regression coverage for request cache controls and retrieval policy propagation."""

import json
import logging
from dataclasses import fields, replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from docthinker.query import QueryMixin
from graphcore.coregraph import GraphCore, QueryParam, operate
from graphcore.coregraph.base import QueryContextResult, QueryResult


class Cache:
    workspace = "session_00001"

    def __init__(self):
        self.global_config = {"enable_llm_cache": True}
        self.entries = {}
        self.get_by_id = AsyncMock(side_effect=lambda key: self.entries.get(key))
        self.upsert = AsyncMock(side_effect=self.entries.update)
        self.index_done_callback = AsyncMock()


class CharacterTokenizer:
    def encode(self, text):
        return list(text)


@pytest.fixture
def query_setup(monkeypatch):
    model = AsyncMock(return_value="fresh answer")
    config = {
        "working_dir": "/test/session1",
        "tokenizer": CharacterTokenizer(),
        "llm_model_func": model,
        "llm_model_name": "answer-model",
        "addon_params": {"language": "English"},
    }
    context = QueryContextResult(context="source evidence", raw_data={"data": {}})
    monkeypatch.setattr(
        operate, "_build_query_context", AsyncMock(return_value=context)
    )
    monkeypatch.setattr(
        operate,
        "get_keywords_from_query",
        AsyncMock(return_value=(["topic"], ["entity"])),
    )
    chunk = {
        "content": "source evidence",
        "chunk_id": "chunk-1",
        "file_path": "doc.txt",
    }
    monkeypatch.setattr(operate, "_get_vector_context", AsyncMock(return_value=[chunk]))
    monkeypatch.setattr(
        operate, "process_chunks_unified", AsyncMock(return_value=[chunk])
    )
    return config, model, Cache()


async def run_query(mode, config, cache, param=None, system_prompt=None):
    param = param or QueryParam(mode=mode)
    if mode == "naive":
        return await operate.naive_query(
            "question", None, param, config, cache, system_prompt
        )
    return await operate.kg_query(
        "question", None, None, None, None, param, config, cache, system_prompt
    )


@pytest.mark.parametrize("mode", ["local", "naive"])
async def test_disabled_cache_never_reads_or_writes(query_setup, mode):
    config, model, cache = query_setup
    for _ in range(2):
        result = await run_query(
            mode, config, cache, QueryParam(mode=mode, use_llm_cache=False)
        )
        assert result.content == "fresh answer"
    assert model.await_count == 2
    cache.get_by_id.assert_not_awaited()
    cache.upsert.assert_not_awaited()


@pytest.mark.parametrize("mode", ["local", "naive"])
async def test_cached_answers_respect_history_prompt_and_policy(query_setup, mode):
    config, model, cache = query_setup
    base = QueryParam(mode=mode)
    await run_query(mode, config, cache, base)
    await run_query(mode, config, cache, base)
    assert model.await_count == 1
    for change in (
        {"conversation_history": [{"role": "user", "content": "different context"}]},
        {"user_prompt": "Only use original evidence"},
        {"include_discovered_edges": True},
        {"max_discovered_relations": 1},
        {"require_discovered_evidence": False},
        {"chunk_retrieval": "embedding"},
        {"graph_traversal_hops": 2},
        {"max_total_tokens": 10000},
    ):
        previous = model.await_count
        await run_query(mode, config, cache, replace(base, **change))
        assert model.await_count == previous + 1, change
    previous = model.await_count
    await run_query(mode, config, cache, base, "Custom instructions: {user_prompt}")
    assert model.await_count == previous + 1
    config["llm_model_name"] = "different-model"
    await run_query(mode, config, cache, base)
    assert model.await_count == previous + 2


async def test_keyword_cache_uses_prompt_language_and_can_be_absent(
    monkeypatch, query_setup
):
    config, model, cache = query_setup
    monkeypatch.setattr(operate, "_fast_extract_keywords", lambda text: None)
    model.return_value = (
        '{"high_level_keywords":["topic"],"low_level_keywords":["entity"]}'
    )
    param = QueryParam()
    await operate.extract_keywords_only("same question", param, config, cache)
    await operate.extract_keywords_only("same question", param, config, cache)
    assert model.await_count == 1
    config["addon_params"]["language"] = "Chinese"
    await operate.extract_keywords_only("same question", param, config, cache)
    assert model.await_count == 2
    # Cache storage is optional even when the request permits caching.
    await operate.extract_keywords_only("same question", param, config, None)
    assert model.await_count == 3
    cache.get_by_id.reset_mock()
    cache.upsert.reset_mock()
    await operate.extract_keywords_only(
        "same question", replace(param, use_llm_cache=False), config, cache
    )
    cache.get_by_id.assert_not_awaited()
    cache.upsert.assert_not_awaited()


@pytest.mark.parametrize("cache_enabled", [False, True])
async def test_data_query_keeps_all_retrieval_controls(monkeypatch, cache_enabled):
    import graphcore.coregraph.coregraph as core_module

    graph = object.__new__(GraphCore)
    for attribute in (
        "chunk_entity_relation_graph",
        "entities_vdb",
        "relationships_vdb",
        "text_chunks",
        "chunks_vdb",
        "llm_response_cache",
    ):
        setattr(graph, attribute, None)
    graph.llm_response_cache = Cache()
    monkeypatch.setattr(core_module, "asdict", lambda obj: {})
    query = AsyncMock(return_value=QueryResult(raw_data={"data": {}}))
    monkeypatch.setattr(core_module, "kg_query", query)
    param = QueryParam(
        stream=True,
        use_llm_cache=cache_enabled,
        include_discovered_edges=True,
        max_discovered_relations=1,
        max_relations=4,
        graph_traversal_hops=2,
        require_discovered_evidence=False,
        min_discovered_edge_confidence=0.93,
        entity_retrieval="bm25",
        relation_retrieval="embedding",
        chunk_retrieval="bm25",
    )
    await graph.aquery_data("question", param)
    copied = query.await_args.args[5]
    assert copied is not param
    assert (
        copied.only_need_context and not copied.only_need_prompt and not copied.stream
    )
    for item in fields(param):
        if item.name not in {"only_need_context", "only_need_prompt", "stream"}:
            assert getattr(copied, item.name) == getattr(param, item.name), item.name
    assert param.stream and not param.only_need_context
    if cache_enabled:
        graph.llm_response_cache.index_done_callback.assert_awaited_once()
    else:
        graph.llm_response_cache.index_done_callback.assert_not_awaited()


async def test_optional_cache_storage_does_not_break_query_cleanup():
    graph = object.__new__(GraphCore)
    graph.llm_response_cache = None
    await graph._query_done()


@pytest.mark.parametrize("mode", ["local", "naive"])
async def test_history_consumes_the_shared_context_budget(
    monkeypatch, query_setup, tmp_path, mode
):
    config, _, cache = query_setup
    monkeypatch.setenv("TOKEN_STATS_LOG", str(tmp_path / "stats.jsonl"))
    monkeypatch.setenv("CONTEXT_DUMP_DIR", str(tmp_path / "contexts"))
    process = operate.process_chunks_unified
    base = QueryParam(mode=mode, only_need_context=True, max_total_tokens=10000)

    async def build(param):
        if mode == "naive":
            await run_query(mode, config, cache, param)
        else:
            await operate._build_context_str([], [], [], "question", param, config)
        return process.await_args.kwargs["chunk_token_limit"]

    before = await build(base)
    history = [{"role": "user", "content": "history" * 10}]
    after = await build(replace(base, conversation_history=history))
    assert before - after == len("history" * 10) + len("user") + 8
    process.reset_mock()
    impossible = replace(base, max_total_tokens=1, conversation_history=history)
    if mode == "naive":
        result = await run_query(mode, config, cache, impossible)
        data = result.raw_data
    else:
        context, data = await operate._build_context_str(
            [], [], [], "question", impossible, config
        )
        assert context == ""
    assert data["metadata"]["failure_reason"] == "context_budget_exceeded"
    process.assert_not_awaited()


class MultimodalQuery(QueryMixin):
    def __init__(self, directory):
        self.working_dir = str(directory)
        self.graphcore = SimpleNamespace(
            workspace="session1", llm_response_cache=Cache()
        )
        self.logger = logging.getLogger(__name__)
        self._ensure_graphcore_initialized = AsyncMock()
        self._process_multimodal_query_content = AsyncMock(
            return_value="question + media description"
        )
        self.aquery = AsyncMock(
            side_effect=["old evidence answer", "new evidence answer"]
        )


async def test_multimodal_cache_keeps_retrieval_fresh(tmp_path):
    runner = MultimodalQuery(tmp_path)
    content = [{"type": "table", "table_data": "x,1"}]
    assert (
        await runner.aquery_with_multimodal("question", content)
        == "old evidence answer"
    )
    assert (
        await runner.aquery_with_multimodal("question", content)
        == "new evidence answer"
    )
    assert runner.aquery.await_count == 2
    # Expensive media analysis is reusable, current graph evidence is not skipped.
    runner._process_multimodal_query_content.assert_awaited_once()


async def test_multimodal_disabled_cache_performs_no_cache_io(tmp_path):
    runner = MultimodalQuery(tmp_path)
    cache = runner.graphcore.llm_response_cache
    await runner.aquery_with_multimodal(
        "question", [{"type": "table"}], use_llm_cache=False
    )
    cache.get_by_id.assert_not_awaited()
    cache.upsert.assert_not_awaited()
    cache.index_done_callback.assert_not_awaited()
    assert runner.aquery.await_args.kwargs["use_llm_cache"] is False


async def test_vlm_stops_when_text_context_budget_is_insufficient(tmp_path):
    runner = MultimodalQuery(tmp_path)
    failure = "上下文预算不足，无法完整保留证据。"
    runner.graphcore.aquery = AsyncMock(return_value=failure)
    runner.vision_model_func = AsyncMock()
    runner._activate_image_assets_for_query = AsyncMock()
    runner._call_vlm_with_multimodal_content = AsyncMock()
    assert await runner.aquery_vlm_enhanced("question", max_total_tokens=100) == failure
    runner._activate_image_assets_for_query.assert_not_awaited()
    runner._call_vlm_with_multimodal_content.assert_not_awaited()
    assert (
        runner.get_last_query_evidence()["failure_reason"] == "context_budget_exceeded"
    )


def test_multimodal_key_tracks_asset_identity_and_request_controls(tmp_path):
    runner = MultimodalQuery(tmp_path)
    first, second = tmp_path / "first" / "same.png", tmp_path / "second" / "same.png"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first image")
    second.write_bytes(b"different image")

    def key(path, **kwargs):
        return runner._generate_multimodal_cache_key(
            "q", [{"type": "image", "img_path": str(path)}], "local", **kwargs
        )

    original = key(first)
    assert original != key(second)
    assert original != key(first, user_prompt="faithful")
    assert original != key(
        first, conversation_history=[{"role": "user", "content": "old"}]
    )
    assert original != key(first, include_discovered_edges=True)
    first.write_bytes(b"replaced image")
    assert original != key(first)


def input_size(
    tokenizer, template, context, question, param, context_key="context_data"
):
    """Independent count of the actual answer prompt and its framing reserve."""
    prompt = template.format(
        **{context_key: context},
        user_prompt=f"\n\n{param.user_prompt}" if param.user_prompt else "n/a",
        response_type=param.response_type or "Multiple Paragraphs",
    )
    return (
        len(tokenizer.encode(prompt))
        + len(tokenizer.encode(question))
        + 200
        + sum(
            len(tokenizer.encode(item["content"]))
            + len(tokenizer.encode(item["role"]))
            + 8
            for item in param.conversation_history
        )
    )


@pytest.mark.parametrize("kind", ["entities", "relations", "mixed"])
async def test_real_kg_packing_respects_total_budget_and_preserves_whole_records(
    tmp_path, monkeypatch, kind
):
    monkeypatch.setenv("TOKEN_STATS_LOG", str(tmp_path / "stats.jsonl"))
    monkeypatch.setenv("CONTEXT_DUMP_DIR", str(tmp_path / "contexts"))
    tokenizer = CharacterTokenizer()
    template = "Answer {user_prompt}\n{context_data}"
    config = {"tokenizer": tokenizer, "system_prompt_template": template}
    entities = (
        [
            {"entity": f"Entity-{i}", "description": f"e{i}:" + "x" * 900}
            for i in range(4)
        ]
        if kind != "relations"
        else []
    )
    relations = (
        [
            {
                "entity1": f"From-{i}",
                "entity2": f"To-{i}",
                "description": f"r{i}:" + "y" * 900,
            }
            for i in range(4)
        ]
        if kind != "entities"
        else []
    )
    param = QueryParam(
        max_total_tokens=2048,
        enable_rerank=False,
        conversation_history=[{"role": "user", "content": "short history"}],
    )
    context, data = await operate._build_context_str(
        entities, relations, [], "q", param, config
    )
    assert data["status"] == "success"
    assert input_size(tokenizer, template, context, "q", param) <= 2048
    kept_entities = data["data"]["entities"]
    kept_relations = data["data"]["relationships"]
    assert 0 < len(kept_entities) + len(kept_relations) < len(entities) + len(relations)
    assert [item["description"] for item in kept_entities] == [
        item["description"] for item in entities[: len(kept_entities)]
    ]
    assert [item["description"] for item in kept_relations] == [
        item["description"] for item in relations[: len(kept_relations)]
    ]
    for item in kept_entities + kept_relations:
        assert json.dumps(item["description"]) in context
    for item in entities[len(kept_entities) :] + relations[len(kept_relations) :]:
        assert json.dumps(item["description"]) not in context


@pytest.mark.parametrize("mode", ["local", "naive"])
async def test_required_overhead_rejects_before_any_llm_call(query_setup, mode):
    config, model, cache = query_setup
    param = QueryParam(mode=mode, max_total_tokens=100, use_llm_cache=False)
    result = await run_query(mode, config, cache, param)
    assert result.raw_data["metadata"]["failure_reason"] == "context_budget_exceeded"
    assert "预算不足" in result.content
    model.assert_not_awaited()
    cache.get_by_id.assert_not_awaited()
    if mode == "local":
        operate.get_keywords_from_query.assert_not_awaited()


@pytest.mark.parametrize("mode", ["local", "naive"])
async def test_real_chunk_packing_counts_json_and_long_references(
    tmp_path, monkeypatch, mode
):
    tokenizer = CharacterTokenizer()
    config = {
        "tokenizer": tokenizer,
        "llm_model_func": AsyncMock(return_value="answer"),
    }
    chunks = [
        {
            "chunk_id": f"chunk-{i}",
            "content": f"Evidence {i}: " + "z" * 150,
            "file_path": f"document-{i}-" + "p" * 400 + ".txt",
        }
        for i in range(8)
    ]
    param = QueryParam(
        mode=mode,
        max_total_tokens=2048,
        chunk_top_k=8,
        enable_rerank=False,
        only_need_context=True,
    )
    if mode == "local":
        monkeypatch.setenv("TOKEN_STATS_LOG", str(tmp_path / "stats.jsonl"))
        monkeypatch.setenv("CONTEXT_DUMP_DIR", str(tmp_path / "contexts"))
        template = "{context_data}"
        context, data = await operate._build_context_str(
            [], [], chunks, "q", param, config, system_prompt=template
        )
    else:
        template = "{content_data}"
        monkeypatch.setattr(
            operate, "_get_vector_context", AsyncMock(return_value=chunks)
        )
        result = await operate.naive_query(
            "q", None, param, config, system_prompt=template
        )
        context, data = result.content, result.raw_data
    assert data["status"] == "success"
    assert (
        input_size(
            tokenizer,
            template,
            context,
            "q",
            param,
            "content_data" if mode == "naive" else "context_data",
        )
        <= 2048
    )
    kept = data["data"]["chunks"]
    assert 0 < len(kept) < len(chunks)
    assert [item["content"] for item in kept] == [
        item["content"] for item in chunks[: len(kept)]
    ]
    assert len(data["data"]["references"]) == len(kept)
    for item in kept:
        assert item["content"] in context and item["file_path"] in context
    config["llm_model_func"].assert_not_awaited()


async def test_custom_kg_prompt_budget_reaches_real_context_builder(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TOKEN_STATS_LOG", str(tmp_path / "stats.jsonl"))
    monkeypatch.setenv("CONTEXT_DUMP_DIR", str(tmp_path / "contexts"))
    tokenizer = CharacterTokenizer()
    model = AsyncMock(return_value="answer")
    config = {"tokenizer": tokenizer, "llm_model_func": model}
    entities = [{"entity": f"E{i}", "description": "evidence " * 30} for i in range(6)]
    monkeypatch.setattr(
        operate,
        "get_keywords_from_query",
        AsyncMock(return_value=(["topic"], ["entity"])),
    )
    monkeypatch.setattr(
        operate,
        "_perform_kg_search",
        AsyncMock(
            return_value={
                "final_entities": entities,
                "final_relations": [],
                "vector_chunks": [],
                "chunk_tracking": {},
                "query_embedding": None,
            }
        ),
    )
    monkeypatch.setattr(
        operate,
        "_apply_token_truncation",
        AsyncMock(
            return_value={
                "entities_context": entities,
                "relations_context": [],
                "filtered_entities": entities,
                "filtered_relations": [],
                "entity_id_to_original": {},
                "relation_id_to_original": {},
            }
        ),
    )
    monkeypatch.setattr(operate, "_merge_all_chunks", AsyncMock(return_value=[]))
    template = "Custom instruction " * 40 + "\n{context_data}\n{user_prompt}"
    param = QueryParam(
        mode="local", max_total_tokens=2048, use_llm_cache=False, enable_rerank=False
    )
    result = await operate.kg_query(
        "q",
        None,
        None,
        None,
        SimpleNamespace(global_config=config),
        param,
        config,
        system_prompt=template,
    )
    assert result.content == "answer"
    model.assert_awaited_once()
    actual_prompt = model.await_args.kwargs["system_prompt"]
    assert (
        len(tokenizer.encode(actual_prompt)) + len(tokenizer.encode("q")) + 200 <= 2048
    )
    assert 0 < len(result.raw_data["data"]["entities"]) < len(entities)
