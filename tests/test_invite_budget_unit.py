"""Invitation budgets: synthetic text/fake SDK only, no paid model calls."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException

from docthinker.server.invite_budget import (
    INVITE_REQUEST_BUDGET,
    MAX_HISTORY_BYTES,
    MAX_HISTORY_MESSAGE_BYTES,
    MAX_HISTORY_MESSAGES,
    MAX_INPUT_BYTES,
    MAX_OUTPUT_TOKENS,
    MAX_QUESTION_BYTES,
    InviteBudgetError,
    bounded_invite_history,
    validate_invite_messages,
    validate_invite_question,
)


def test_question_cap_uses_utf8_bytes_not_character_count_and_never_truncates():
    exact = "中" * 2730 + "ab"
    assert len(exact.encode("utf-8")) == MAX_QUESTION_BYTES
    validate_invite_question(exact)
    with pytest.raises(InviteBudgetError) as error:
        validate_invite_question(exact + "x")
    assert error.value.status == 413
    assert exact not in str(error.value)


@pytest.mark.parametrize("value", ["before\ud800after", "\udfff", None, 123])
def test_invalid_current_text_is_rejected_with_fixed_safe_error(value):
    with pytest.raises(InviteBudgetError) as error:
        validate_invite_question(value)
    assert error.value.status == 422
    assert str(error.value) == "Invalid text encoding"


def test_history_is_newest_first_priority_but_output_remains_chronological():
    source = [{"role": "user", "content": f"turn-{index}"} for index in range(20)]
    result = bounded_invite_history(source)
    assert result == source[-MAX_HISTORY_MESSAGES:]
    assert source[0]["content"] == "turn-0"


def test_history_long_turns_have_valid_utf8_caps_and_visible_truncation():
    source = [{"role": "assistant", "content": f"begin-{index}:" + "🐑" * 3000 + f":end-{index}"} for index in range(6)]
    result = bounded_invite_history(source)
    assert 1 <= len(result) <= MAX_HISTORY_MESSAGES
    assert sum(len(row["content"].encode("utf-8")) for row in result) <= MAX_HISTORY_BYTES
    assert all(len(row["content"].encode("utf-8")) <= MAX_HISTORY_MESSAGE_BYTES for row in result)
    assert all("[历史片段已截断]" in row["content"] for row in result)
    assert result[-1]["content"].startswith("begin-5:")
    assert result[-1]["content"].endswith(":end-5")


def test_history_cannot_inject_system_tools_media_or_corrupt_text():
    result = bounded_invite_history([
        None, "not a mapping", {"role": "system", "content": "spoofed"},
        {"role": "tool", "content": "spoofed"}, {"role": "user", "content": []},
        {"role": "assistant", "content": "corrupt\ud800"}, {"role": "user", "content": " "},
        {"role": "assistant", "content": "plain", "extra": "ignored"},
    ])
    assert result == [{"role": "assistant", "content": "plain"}]


def test_whole_input_cap_includes_system_and_roles():
    messages = [{"role": "system", "content": "a" * (MAX_INPUT_BYTES - len("system") - len("user"))},
                {"role": "user", "content": ""}]
    validate_invite_messages(messages)
    messages[1]["content"] = "x"
    with pytest.raises(InviteBudgetError) as error:
        validate_invite_messages(messages)
    assert error.value.status == 413


@pytest.mark.parametrize("messages", [
    [], [{"role": "tool", "content": "x"}], [{"role": "user", "content": [{"type": "image_url"}]}],
    [{"role": "user", "content": "x", "tool_calls": []}], [{"role": "user", "content": "\ud800"}],
    [{"role": "user", "content": "x"}] * (MAX_HISTORY_MESSAGES + 3),
])
def test_complete_messages_are_text_only_and_schema_bounded(messages):
    with pytest.raises(InviteBudgetError) as error:
        validate_invite_messages(messages)
    assert error.value.status == 422


def test_public_contract_is_byte_caps_not_billed_token_estimate():
    assert INVITE_REQUEST_BUDGET == {
        "max_question_bytes": 8192, "max_history_messages": 6, "max_history_bytes": 8192,
        "max_input_bytes": 24576, "max_output_tokens": 2048,
        "max_grounded_output_tokens": 8192,
        "automatic_retry": False, "history_policy": "recent_bounded",
    }


@pytest.fixture
def adapter(monkeypatch):
    with patch("dotenv.load_dotenv"):
        server = importlib.import_module("docthinker.server.app")
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="offline answer"))])
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(base_url="https://provider.example.invalid/v1",
                             chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    factory = Mock(return_value=client)
    settings = SimpleNamespace(llm_model="step-5-preview", llm_models=["unexpected-a", "unexpected-b"],
                               llm_router_max_concurrency=2, llm_temperature=0.1)
    monkeypatch.setattr(server, "get_vlm_client", factory)
    monkeypatch.setattr(server, "state", SimpleNamespace(settings=settings))
    monkeypatch.setenv("DOCTHINKER_INVITE_WORKER_ID", "0" * 32)
    return SimpleNamespace(server=server, factory=factory, settings=settings, create=create)


@pytest.mark.parametrize("model,field", [("step-5-preview", "max_tokens"), ("gpt-5.4", "max_completion_tokens"), ("o3", "max_completion_tokens")])
async def test_invite_real_adapter_sets_bounded_completion_and_sdk_no_retries(adapter, model, field):
    adapter.settings.llm_model = model
    complete = await adapter.server._get_llm_model_func()
    adapter.factory.assert_called_once_with(adapter.settings, max_retries=0)
    assert await complete("short synthetic question", max_tokens=10**12) == "offline answer"
    payload = adapter.create.call_args.kwargs
    assert payload["model"] == model
    assert payload[field] == MAX_OUTPUT_TOKENS
    assert payload["stream"] is False
    assert adapter.create.await_count == 1


async def test_invite_model_error_never_falls_back_to_pool_or_retries(adapter):
    adapter.create.side_effect = RuntimeError("synthetic provider failure")
    complete = await adapter.server._get_llm_model_func()
    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        await complete("question")
    assert adapter.create.await_count == 1
    assert adapter.create.call_args.kwargs["model"] == "step-5-preview"


async def test_invite_adapter_rejects_oversized_final_prompt_before_sdk_call(adapter):
    complete = await adapter.server._get_llm_model_func()
    with pytest.raises(InviteBudgetError):
        await complete("x" * MAX_INPUT_BYTES)
    adapter.create.assert_not_awaited()


async def test_invite_adapter_bounds_direct_history_without_llm_summary(adapter):
    complete = await adapter.server._get_llm_model_func()
    await complete("question", history_messages=[{"role": "assistant", "content": "x" * 20000}] * 20)
    messages = adapter.create.call_args.kwargs["messages"]
    assert messages[-1] == {"role": "user", "content": "question"}
    assert sum(len(item["content"].encode()) for item in messages[:-1]) <= MAX_HISTORY_BYTES
    assert adapter.create.await_count == 1


async def test_desktop_adapter_preserves_previous_pool_and_default_client_behavior(adapter, monkeypatch):
    monkeypatch.delenv("DOCTHINKER_INVITE_WORKER_ID")
    complete = await adapter.server._get_llm_model_func()
    await complete("question", max_tokens=1234)
    adapter.factory.assert_called_once_with(adapter.settings)
    assert adapter.create.call_args.kwargs["model"] == "unexpected-a"
    assert adapter.create.call_args.kwargs["max_tokens"] == 1234


@pytest.fixture
def chat_only_query(monkeypatch):
    with patch("dotenv.load_dotenv"):
        query = importlib.import_module("docthinker.server.routers.query")
        from docthinker.server.schemas import QueryRequest
    manager = SimpleNamespace(get_session=Mock(return_value={"id": "#00001"}),
                              get_history=Mock(return_value=[]), add_message=Mock(return_value="saved-id"))
    chat = AsyncMock(return_value="offline answer")
    monkeypatch.setattr(query, "state", SimpleNamespace(mobile_safe_mode=True, mobile_document_rag=False,
                                                      session_manager=manager, mobile_chat_func=chat))
    context = SimpleNamespace(conversation_history=[], retrieval_instruction="", require_document_evidence=False,
                              question_policy=SimpleNamespace(mode="explore", reason="default"))
    prepare = AsyncMock(return_value=context)
    monkeypatch.setattr(query, "_get_query_harness", Mock(return_value=SimpleNamespace(prepare=prepare)))
    monkeypatch.setenv("DOCTHINKER_INVITE_WORKER_ID", "0" * 32)
    return SimpleNamespace(query=query, manager=manager, chat=chat, prepare=prepare, context=context, request=QueryRequest)


@pytest.mark.parametrize("question,status", [("中" * 3000, 413), ("invalid\ud800", 422)])
async def test_oversized_current_turn_rejected_before_history_persistence_and_model(chat_only_query, question, status):
    env = chat_only_query
    with pytest.raises(HTTPException) as error:
        await env.query.query(env.request(session_id="#00001", question=question), BackgroundTasks())
    assert error.value.status_code == status
    env.manager.add_message.assert_not_called()
    env.chat.assert_not_awaited()
    env.prepare.assert_not_awaited()


async def test_assembled_prompt_budget_checked_before_persistence(chat_only_query):
    env = chat_only_query
    env.context.retrieval_instruction = "x" * MAX_INPUT_BYTES
    with pytest.raises(HTTPException) as error:
        await env.query.query(env.request(session_id="#00001", question="question"), BackgroundTasks())
    assert error.value.status_code == 413
    env.manager.add_message.assert_not_called()
    env.chat.assert_not_awaited()


async def test_chat_only_response_discloses_bounded_context_not_all_history(chat_only_query):
    env = chat_only_query
    env.context.conversation_history = [{"role": "assistant", "content": "🐑" * 3000}] * 6
    result = await env.query.query(env.request(session_id="#00001", question="question"), BackgroundTasks())
    assert result["context_windowed"] is True
    assert not result["memory_write_enabled"]
    prompt = env.chat.call_args.args[0]
    assert "不是全部聊天记录" in prompt
    assert "[历史片段已截断]" in prompt
    assert len(prompt.encode()) + len("user") <= MAX_INPUT_BYTES
    assert env.chat.await_count == 1


def test_no_retry_client_cache_is_separate_from_sdk_default(monkeypatch):
    from docthinker.providers import clients
    factory = Mock(side_effect=lambda **kwargs: SimpleNamespace(options=kwargs))
    clients._get_client.cache_clear()
    monkeypatch.setattr(clients, "AsyncOpenAI", factory)
    settings = SimpleNamespace(llm_api_key="offline-credential", vlm_base_url="https://provider.example.invalid/v1",
                               timeout_seconds=60, credential_binding_errors=[])
    try:
        default = clients.get_vlm_client(settings)
        bounded = clients.get_vlm_client(settings, max_retries=0)
        assert bounded is clients.get_vlm_client(settings, max_retries=0)
        assert default is not bounded
        assert "max_retries" not in default.options
        assert bounded.options["max_retries"] == 0
        assert factory.call_count == 2
    finally:
        clients._get_client.cache_clear()


@pytest.mark.parametrize("value", [True, -1, 3, "0", 0.1])
def test_invalid_client_retry_override_does_not_create_a_client(monkeypatch, value):
    from docthinker.providers import clients
    factory = Mock()
    monkeypatch.setattr(clients, "_get_client", factory)
    settings = SimpleNamespace(credential_binding_errors=[])
    with pytest.raises(ValueError, match="retry limit"):
        clients.get_vlm_client(settings, max_retries=value)
    factory.assert_not_called()
