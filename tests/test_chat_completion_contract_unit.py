"""Real SDK, offline HTTP: incomplete provider output is never a saved answer."""
from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
from openai import AsyncOpenAI

from docthinker.server.research_service import ResearchService


ERROR = "Chat model did not return a complete text answer"
PRIVATE = "synthetic-private-provider-body"


def completion(content="Complete offline answer.", finish_reason="stop"):
    return {"id": "offline-contract", "model": "step-5-preview", "object": "chat.completion",
            "created": 0, "choices": [{"index": 0, "finish_reason": finish_reason,
                                       "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}


INVALID = [
    pytest.param(completion(PRIVATE, "length"), id="nonempty-length"),
    pytest.param(completion("", "length"), id="empty-length"),
    pytest.param(completion("", "stop"), id="empty-stop"),
    pytest.param(completion(" \n", "stop"), id="whitespace-stop"),
    pytest.param(completion(None, "stop"), id="null-content"),
    pytest.param(completion([PRIVATE], "stop"), id="nontext-content"),
    pytest.param(completion(PRIVATE, None), id="missing-finish"),
    pytest.param(completion(PRIVATE, "tool_calls"), id="unexpected-tool-call"),
    pytest.param({"id": PRIVATE}, id="missing-choices"),
    pytest.param({"choices": []}, id="empty-choices"),
    pytest.param({"choices": [{"finish_reason": "stop"}]}, id="missing-message"),
    pytest.param({"choices": [completion()["choices"][0]] * 2}, id="ambiguous-choices"),
]


@pytest.fixture
def adapter_factory(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("DOCTHINKER_RESEARCH_SEMANTIC", "off")
    with patch("dotenv.load_dotenv", return_value=False):
        server = importlib.import_module("docthinker.server.app")
    settings = SimpleNamespace(llm_model="step-5-preview", llm_models=["fallback-a", "fallback-b"],
                               llm_router_max_concurrency=1, llm_temperature=0.1)

    async def make(body, profile="research"):
        monkeypatch.setenv("DOCTHINKER_RESEARCH_ENABLED", "true" if profile == "research" else "false")
        monkeypatch.delenv("DOCTHINKER_INVITE_WORKER_ID", raising=False)
        if profile == "invite":
            monkeypatch.setenv("DOCTHINKER_INVITE_WORKER_ID", "0" * 32)
        monkeypatch.setattr(server, "state", SimpleNamespace(settings=settings, mobile_safe_mode=profile != "desktop"))
        calls = []

        def handle(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json=body)

        client = AsyncOpenAI(api_key="offline-fake-key", base_url="https://example.invalid/v1",
                             max_retries=0, http_client=httpx.AsyncClient(
                                 transport=httpx.MockTransport(handle), trust_env=False))
        factory = Mock(return_value=client)
        monkeypatch.setattr(server, "get_vlm_client", factory)
        answer = await server._get_llm_model_func()
        return answer, calls, client, factory

    return make


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["desktop", "research", "invite"])
@pytest.mark.parametrize("body", INVALID)
async def test_http_200_incomplete_or_malformed_answer_fails_closed(adapter_factory, body, profile):
    answer, calls, client, factory = await adapter_factory(body, profile)
    try:
        with pytest.raises(ValueError) as error:
            await answer("Offline question", system_prompt="Offline policy")
        assert str(error.value) == ERROR and PRIVATE not in str(error.value)
        assert len(calls) == 1  # No SDK retry or pool failover after an HTTP 200.
        if profile != "desktop":
            assert factory.call_args.kwargs == {"max_retries": 0}
            assert calls[0]["model"] == "step-5-preview"
            assert calls[0]["max_tokens"] == 2048
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["desktop", "research", "invite"])
async def test_http_200_nonempty_stop_is_returned_without_repair(adapter_factory, profile):
    body = completion()
    answer, calls, client, _ = await adapter_factory(body, profile)
    try:
        assert await answer("Offline question", system_prompt="Offline policy") == body["choices"][0]["message"]["content"]
        assert len(calls) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("body", INVALID)
async def test_research_never_persists_incomplete_sdk_answer(adapter_factory, tmp_path, monkeypatch, body):
    answer, calls, client, _ = await adapter_factory(body)
    service = ResearchService(tmp_path, context_assistance_enabled=False)
    manager = SimpleNamespace(add_message=Mock(), get_history=Mock(return_value=[]))
    store_answer, store_evidence = Mock(), Mock()
    monkeypatch.setattr(service.store, "record_answer", store_answer)
    monkeypatch.setattr(service.evidence, "record", store_evidence)
    request = SimpleNamespace(session_id="#00001", question="Hello", use_memory=False,
                              use_conversation_context=False, knowledge_skill_mode="off", insight_mode="off")
    try:
        with pytest.raises(ValueError, match=ERROR):
            await service.answer(request, answer, manager)
        assert len(calls) == 1
        manager.add_message.assert_not_called()
        store_answer.assert_not_called()
        store_evidence.assert_not_called()
    finally:
        await client.close()
