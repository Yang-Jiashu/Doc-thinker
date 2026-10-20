"""Provider request compatibility; all clients are local fakes, never network calls."""

import asyncio
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from docthinker.auto_thinking.vlm_client import VLMClient
from docthinker.providers.clients import (
    CredentialBindingError,
    assert_credential_bindings,
    get_embed_client,
    get_vlm_client,
)
from docthinker.providers.request_compat import (
    chat_request_options,
    embedding_request_options,
)
from docthinker.utils import create_bltcy_rerank_func

server = importlib.import_module("docthinker.server.app")
utils = importlib.import_module("docthinker.utils")
clients = importlib.import_module("docthinker.providers.clients")


class FakeResponse:
    status = 200

    def __init__(self, data):
        self.data = data

    async def json(self):
        return self.data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False


class FakeSession:
    closed = False

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def post(self, endpoint, **kwargs):
        self.calls.append((endpoint, json.loads(kwargs["data"])))
        return FakeResponse(next(self.responses))


@pytest.mark.parametrize("base,endpoint,native", [
    ("https://api.cohere.com/v2", "https://api.cohere.com/v2/rerank", False),
    ("https://api.cohere.com/v2/rerank/", "https://api.cohere.com/v2/rerank", False),
    ("https://api.jina.ai/v1/rerank", "https://api.jina.ai/v1/rerank", False),
    ("https://dashscope.aliyuncs.com/compatible-mode/v1", "https://dashscope.aliyuncs.com/compatible-mode/v1/reranks", False),
    ("https://dashscope.aliyuncs.com/compatible-mode/v1/reranks", "https://dashscope.aliyuncs.com/compatible-mode/v1/reranks", False),
    ("https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank", "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank", True),
])
def test_rerank_preserves_endpoint_and_original_result_indices(monkeypatch, base, endpoint, native):
    results = [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]
    response = {"output": {"results": results}} if native else {"results": results}
    session = FakeSession([response])
    monkeypatch.setattr(utils.aiohttp, "ClientSession", lambda **_: session)
    monkeypatch.setattr(utils.aiohttp, "TCPConnector", lambda **_: None)
    rerank = create_bltcy_rerank_func(api_key="fake-key", base_url=base, model_name="fake-model")
    result = asyncio.run(rerank("query", ["doc0", "doc1", "doc2"], top_n=2))
    assert [item["index"] for item in result] == [2, 0]
    assert session.calls[0][0] == endpoint
    payload = session.calls[0][1]
    assert (payload["input"] if native else payload)["documents"] == ["doc0", "doc1", "doc2"]
    assert (payload["parameters"] if native else payload)["top_n"] == 2


def test_rerank_offsets_each_batch_without_fabricating_zero_scores(monkeypatch):
    session = FakeSession([
        {"results": [{"index": 24, "relevance_score": -0.2}]},
        {"results": [{"index": 0, "relevance_score": -0.1}]},
    ])
    monkeypatch.setattr(utils.aiohttp, "ClientSession", lambda **_: session)
    monkeypatch.setattr(utils.aiohttp, "TCPConnector", lambda **_: None)
    rerank = create_bltcy_rerank_func(api_key="fake", base_url="https://example.invalid/v1", model_name="model")
    result = asyncio.run(rerank("q", [str(i) for i in range(26)], top_n=1))
    assert result == [{"index": 25, "relevance_score": -0.1}]


@pytest.mark.parametrize("results", [
    [{"index": 8, "score": 0.9}],
    [{"index": 0, "score": 0.9}, {"index": 0, "score": 0.8}],
    [{"index": 0, "score": float("nan")}],
    [{"score": 0.9}],
    [{"index": 0, "score": 0.9}, {"score": 0.8}],
    [],
])
def test_rerank_invalid_or_ambiguous_mapping_fails_closed(monkeypatch, results):
    session = FakeSession([{"results": results}])
    monkeypatch.setattr(utils.aiohttp, "ClientSession", lambda **_: session)
    monkeypatch.setattr(utils.aiohttp, "TCPConnector", lambda **_: None)
    rerank = create_bltcy_rerank_func(api_key="fake", base_url="https://example.invalid/v1", model_name="model")
    with pytest.raises(ValueError):
        asyncio.run(rerank("q", ["a", "b"]))


@pytest.mark.parametrize("model", ["gpt-5.2", "openai/gpt-5-mini", "o1", "o3-mini", "o4-mini-2025-04-16"])
def test_reasoning_chat_parameters_do_not_send_temperature(model):
    options = chat_request_options(model, 800, 0.2)
    assert options == {"max_completion_tokens": 800}


@pytest.mark.parametrize("base,model,thinking", [
    ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-test", False),
    ("https://api.anthropic.com/v1", "claude-test", False),
    ("https://example.invalid/v1", "qwen-plus", False),
    ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus", True),
    ("https://test-workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", "qwen-plus", True),
    ("https://test-workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", "qwen3.8-max", True),
    ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3-235b-a22b-thinking-2507", False),
    ("https://dashscope.aliyuncs.com.example.invalid/v1", "qwen-plus", False),
])
def test_vlm_only_sends_explicitly_supported_thinking_extension(base, model, thinking):
    session = FakeSession([{"choices": [{"message": {"content": "ok"}}]}])
    client = VLMClient(api_key="fake", api_base=base, model=model)
    client._session = session
    assert asyncio.run(client.generate("hello", extra_body={"enable_thinking": False})) == "ok"
    assert ("enable_thinking" in session.calls[0][1]) is thinking


def test_vlm_reasoning_model_omits_sampling_and_legacy_token_parameter():
    session = FakeSession([{"choices": [{"message": {"content": "ok"}}]}])
    client = VLMClient(api_key="fake", api_base="https://api.openai.com/v1", model="gpt-5.2")
    client._session = session
    asyncio.run(client.generate("hello", max_tokens=800, temperature=0.2))
    payload = session.calls[0][1]
    assert payload["max_completion_tokens"] == 800
    assert "max_tokens" not in payload and "temperature" not in payload


@pytest.mark.parametrize("model,base,dimension,expected", [
    ("text-embedding-3-small", "https://api.openai.com/v1", 512, {"dimensions": 512}),
    ("text-embedding-3-large", "https://api.openai.com/v1", 3072, {"dimensions": 3072}),
    ("text-embedding-v3", "https://dashscope.aliyuncs.com/compatible-mode/v1", 768, {"dimensions": 768}),
    ("text-embedding-v4", "https://dashscope.aliyuncs.com/compatible-mode/v1", 2048, {"dimensions": 2048}),
    ("text-embedding-v4", "https://test-workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", 2048, {"dimensions": 2048}),
    ("BAAI/bge-m3", "https://example.invalid/v1", 1024, {}),
    ("text-embedding-3-small", "https://unknown.invalid/v1", 1536, {}),
])
def test_dimensions_only_sent_for_known_official_models(model, base, dimension, expected):
    assert embedding_request_options(base, model, dimension) == expected


@pytest.mark.parametrize("model,base,dimension", [
    ("text-embedding-3-small", "https://api.openai.com/v1", 2048),
    ("text-embedding-v3", "https://dashscope.aliyuncs.com/compatible-mode/v1", 1536),
    ("custom", "https://example.invalid/v1", 0),
])
def test_invalid_dimensions_fail_before_request(model, base, dimension):
    with pytest.raises(ValueError):
        embedding_request_options(base, model, dimension)


def make_embedding_settings(**overrides):
    values = {"embed_model": "custom", "embed_base_url": "https://example.invalid/v1", "embed_dim": 2}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_embedding_validates_dimension_and_does_not_hot_swap_model(monkeypatch):
    settings = make_embedding_settings()
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0, 2.0])])

    monkeypatch.setattr(server.state, "settings", settings)
    monkeypatch.setattr(server, "get_embed_client", lambda _: SimpleNamespace(embeddings=SimpleNamespace(create=create)))
    func = asyncio.run(server._get_embedding_func())
    settings.embed_model = "changed-model"
    result = asyncio.run(func(["text"]))
    assert result.shape == (1, 2)
    assert calls[0] == {"model": "custom", "input": ["text"]}
    assert asyncio.run(func([])).shape == (0, 2)


@pytest.mark.parametrize("items", [
    [SimpleNamespace(index=0, embedding=[1.0])],
    [SimpleNamespace(index=0, embedding=[1.0, float("inf")])],
    [],
    [SimpleNamespace(index=3, embedding=[1.0, 2.0])],
])
def test_embedding_rejects_mismatched_or_invalid_response(monkeypatch, items):
    monkeypatch.setattr(server.state, "settings", make_embedding_settings())
    create = AsyncMock(return_value=SimpleNamespace(data=items))
    monkeypatch.setattr(server, "get_embed_client", lambda _: SimpleNamespace(embeddings=SimpleNamespace(create=create)))
    func = asyncio.run(server._get_embedding_func())
    with pytest.raises(ValueError):
        asyncio.run(func(["text"]))
    assert create.await_count == 1


def test_dashscope_embeddings_batch_and_restore_index_order(monkeypatch):
    monkeypatch.setattr(server.state, "settings", make_embedding_settings(
        embed_model="text-embedding-v3", embed_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", embed_dim=64
    ))
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[
            SimpleNamespace(index=i, embedding=[float(text)] * 64)
            for i, text in reversed(list(enumerate(kwargs["input"])))
        ])

    monkeypatch.setattr(server, "get_embed_client", lambda _: SimpleNamespace(embeddings=SimpleNamespace(create=create)))
    func = asyncio.run(server._get_embedding_func())
    result = asyncio.run(func([str(i) for i in range(11)]))
    assert [len(call["input"]) for call in calls] == [10, 1]
    assert all(call["dimensions"] == 64 for call in calls)
    np.testing.assert_array_equal(result[:, 0], np.arange(11))


def test_auxiliary_factories_use_reasoning_compatible_parameters(monkeypatch):
    create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(server, "get_vlm_client", lambda _: client)
    monkeypatch.setattr(server.state, "settings", SimpleNamespace(keyword_llm_model="o3-mini", entity_extraction_llm_model="gpt-5.2"))
    keyword = asyncio.run(server._get_keyword_llm_func())
    entity = asyncio.run(server._get_entity_extraction_llm_func())
    assert asyncio.run(keyword("q")) == "ok"
    assert asyncio.run(entity("q")) == "ok"
    for call in create.call_args_list:
        assert "max_completion_tokens" in call.kwargs
        assert "max_tokens" not in call.kwargs


def test_dedicated_extraction_router_uses_sdk_client(monkeypatch):
    settings = SimpleNamespace(
        llm_api_key="fake", vlm_base_url="https://example.invalid/v1", vlm_model="vision",
        llm_model="chat", extraction_llm_model="extractor", llm_router_max_concurrency=1,
        rerank_api_key="", rerank_base_url="https://example.invalid/v1", rerank_model="",
        graphcore_llm_max_async=1, graphcore_embedding_max_async=1,
        graphcore_max_parallel_insert=1, graphcore_max_gleaning=0,
    )
    monkeypatch.setattr(server.state, "settings", settings)
    for name in ("_get_embedding_func", "_get_llm_model_func", "_get_keyword_llm_func", "_get_entity_extraction_llm_func"):
        monkeypatch.setattr(server, name, AsyncMock(return_value=object()))
    monkeypatch.setattr(server, "_create_rag_config", lambda: object())
    monkeypatch.setattr(server, "create_bltcy_rerank_func", lambda **_: None)
    create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(server, "get_vlm_client", lambda _: client)
    monkeypatch.setattr(server, "DocThinker", lambda **kwargs: kwargs)
    monkeypatch.setattr(server.state, "vision_vlm_client", None, raising=False)
    rag = asyncio.run(server._initialize_rag())
    extraction = rag["graphcore_kwargs"]["entity_extraction_llm_model_func"]
    assert asyncio.run(extraction("extract entities")) == "ok"
    assert create.call_args.kwargs["model"] == "extractor"


@pytest.mark.parametrize("group,factory", [("llm", get_vlm_client), ("embed", get_embed_client)])
def test_binding_error_blocks_client_factory_before_credentials_or_documents_leave(monkeypatch, group, factory):
    settings = SimpleNamespace(credential_binding_errors=[group])
    monkeypatch.setattr(clients, "_get_client", lambda *_: pytest.fail("Must not create or reuse a provider client"))
    with pytest.raises(CredentialBindingError, match=group) as error:
        factory(settings)
    assert "credential destination mismatch" in str(error.value)


def test_factory_binding_validation_only_blocks_its_own_service(monkeypatch):
    settings = SimpleNamespace(
        credential_binding_errors=["rerank"], llm_api_key="EMPTY",
        vlm_base_url="http://localhost:11434/v1", timeout_seconds=10,
    )
    sentinel = object()
    monkeypatch.setattr(clients, "_get_client", lambda *_: sentinel)
    assert get_vlm_client(settings) is sentinel


@pytest.mark.parametrize("errors", ["llm", None, ["unknown"], [42], {"llm": "private-value"}])
def test_malformed_binding_status_fails_without_echoing_untrusted_values(errors):
    with pytest.raises(CredentialBindingError, match="Invalid credential binding status") as error:
        assert_credential_bindings(SimpleNamespace(credential_binding_errors=errors))
    assert "private-value" not in str(error.value)


@pytest.mark.parametrize("group", ["llm", "embed", "rerank"])
def test_binding_error_disables_warmup_and_all_rag_provider_construction(monkeypatch, group):
    monkeypatch.setattr(server.state, "settings", SimpleNamespace(credential_binding_errors=[group]))
    monkeypatch.setattr(server, "get_vlm_client", lambda *_: pytest.fail("No warmup request allowed"))
    monkeypatch.setattr(server, "get_embed_client", lambda *_: pytest.fail("No embedding client allowed"))
    monkeypatch.setattr(server, "create_bltcy_rerank_func", lambda **_: pytest.fail("No rerank client allowed"))
    asyncio.run(server._warmup_llm_connection())
    assert asyncio.run(server._initialize_rag()) is None


def test_binding_error_keeps_settings_api_available_without_rag_subsystems(monkeypatch, tmp_path):
    from docthinker.providers.settings import AppSettings
    from docthinker.server.routers import settings as settings_router_module
    from docthinker.server.state import AppState

    settings = AppSettings(
        llm_api_key="EMPTY", vlm_base_url="https://example.invalid/v1", llm_model="model", vlm_model="vision",
        embed_api_key="EMPTY", embed_base_url="https://example.invalid/v1", embed_model="embed",
        rerank_api_key="EMPTY", rerank_base_url="https://example.invalid/v1", rerank_model="rerank",
        credential_binding_errors=["rerank"], workdir=str(tmp_path / "blocked-models"),
    )
    isolated = AppState()
    monkeypatch.setattr(server, "state", isolated)
    monkeypatch.setattr(settings_router_module, "state", isolated)
    monkeypatch.setattr(server, "load_settings", lambda: settings)
    monkeypatch.setattr(server, "APIConfig", lambda: object())
    monkeypatch.setattr(server, "SessionManager", lambda **_: object())
    monkeypatch.setattr(server, "get_vlm_client", lambda *_: pytest.fail("No provider client allowed"))
    monkeypatch.setattr(server, "get_embed_client", lambda *_: pytest.fail("No provider client allowed"))
    monkeypatch.setattr(server, "_cleanup_global_graphcore_artifacts", lambda *_: pytest.fail("Must not modify graph artifacts in recovery mode"))
    monkeypatch.setattr(server, "CognitiveProcessor", lambda **_: pytest.fail("Must not initialize a RAG subsystem"))
    monkeypatch.setattr(server, "IngestionService", lambda **_: pytest.fail("Must not initialize ingestion"))
    monkeypatch.setattr(server, "SessionLearningRunner", lambda *_: pytest.fail("Must not start background work"))

    async def exercise():
        async with server.lifespan(None):
            response = await settings_router_module.get_settings()
            assert response.llm_model == "model"
            assert any("rerank" in warning for warning in response.warnings)
            assert isolated.rag_instance is None
            assert isolated.ingestion_service is None
            assert isolated.memory_engine_factory is None
            assert isolated.post_upload_learning is None
            assert isolated.session_manager is not None

    asyncio.run(exercise())
