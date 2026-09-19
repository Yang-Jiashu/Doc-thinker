import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import BackgroundTasks

from docthinker.memory_core.core import RecallBundle
from docthinker.server.schemas import QueryRequest
from docthinker.ui.query_payload import build_query_payload

router = importlib.import_module("docthinker.server.routers.query")


@pytest.mark.parametrize("stream", [False, True])
async def test_actual_query_route_passes_prior_history_once_and_honors_budget(
    monkeypatch, stream
):
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    manager = SimpleNamespace(
        get_history=lambda _: list(history),
        add_message=lambda _, role, content: history.append(
            {"role": role, "content": content}
        ),
        get_files=lambda _: [],
    )
    llm = AsyncMock(side_effect=AssertionError("unexpected extra model call"))
    rag = SimpleNamespace(
        graphcore=SimpleNamespace(tokenizer=None),
        llm_model_func=llm,
        aquery=AsyncMock(return_value="answer"),
        aquery_stream=AsyncMock(return_value="answer"),
    )
    monkeypatch.setattr(router.state, "session_manager", manager)
    monkeypatch.setattr(router.state, "rag_instance", rag)
    monkeypatch.setattr(
        router, "_get_session_rag_or_raise", AsyncMock(return_value=rag)
    )
    request = QueryRequest(
        question="current",
        session_id="#00003",
        use_memory=False,
        use_llm_cache=False,
        top_k=4,
    )
    background = BackgroundTasks()
    if stream:
        response = await router.query_stream(request, background)
        async for _ in response.body_iterator:
            pass
        options = rag.aquery_stream.await_args.kwargs
    else:
        response = await router.query(request, background)
        options = rag.aquery.await_args.kwargs
        assert "context_budget" in response
    assert options["conversation_history"] == history[:2]
    assert options["top_k"] == 4
    assert options["use_llm_cache"] is False
    assert options["record_knowledge"] is False
    assert len(background.tasks) == 0
    llm.assert_not_awaited()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("evolution", [True, False])
async def test_missing_document_evidence_never_falls_back_to_unrestricted_llm(
    monkeypatch, stream, evolution
):
    llm = AsyncMock(side_effect=AssertionError("no evidence"))
    rag = SimpleNamespace(
        graphcore=SimpleNamespace(tokenizer=None),
        llm_model_func=llm,
        aquery=AsyncMock(return_value="[no-context]"),
        aquery_stream=AsyncMock(return_value="[no-context]"),
    )
    manager = SimpleNamespace(
        get_history=lambda _: [], add_message=lambda *args: None, get_files=lambda _: []
    )
    monkeypatch.setattr(router.state, "session_manager", manager)
    monkeypatch.setattr(router.state, "rag_instance", rag)
    monkeypatch.setattr(
        router, "_get_session_rag_or_raise", AsyncMock(return_value=rag)
    )
    request = QueryRequest(
        question="Who founded Acme?",
        session_id="#00003",
        use_memory=True,
        evolution_mode="faithful",
        use_self_evolution=evolution,
    )
    monkeypatch.setattr(
        router,
        "_get_agent_memory_core",
        lambda: SimpleNamespace(recall=AsyncMock(return_value=RecallBundle())),
    )
    background = BackgroundTasks()
    if stream:
        response = await router.query_stream(request, background)
        output = "".join([item async for item in response.body_iterator])
    else:
        output = (await router.query(request, background))["answer"]
    assert "原文证据" in output
    llm.assert_not_awaited()
    assert not background.tasks


@pytest.mark.parametrize("stream", [False, True])
async def test_budget_failure_is_not_retried_or_written_to_memory(monkeypatch, stream):
    message = "上下文预算不足，无法在限制内完整保留必要提示或证据。"
    llm = AsyncMock(side_effect=AssertionError("budget failure must not retry"))
    rag = SimpleNamespace(
        graphcore=SimpleNamespace(tokenizer=None),
        llm_model_func=llm,
        aquery=AsyncMock(return_value=message),
        aquery_stream=AsyncMock(return_value=message),
    )
    monkeypatch.setattr(router.state, "rag_instance", rag)
    monkeypatch.setattr(
        router.state,
        "session_manager",
        SimpleNamespace(
            get_history=lambda _: [],
            get_files=lambda _: [],
            add_message=lambda *args: None,
        ),
    )
    monkeypatch.setattr(
        router, "_get_session_rag_or_raise", AsyncMock(return_value=rag)
    )
    monkeypatch.setattr(
        router,
        "_get_agent_memory_core",
        lambda: SimpleNamespace(recall=AsyncMock(return_value=RecallBundle())),
    )
    request = QueryRequest(question="预算", session_id="#00003", use_memory=True)
    background = BackgroundTasks()
    if stream:
        response = await router.query_stream(request, background)
        output = "".join([item async for item in response.body_iterator])
        assert "context_budget_exceeded" in output
    else:
        result = await router.query(request, background)
        assert result["answer_mode"] == "context_budget_exceeded"
    assert not background.tasks
    llm.assert_not_awaited()


def test_ui_payload_preserves_explicit_controls_and_budgets():
    payload = build_query_payload(
        {
            "question": "q",
            "ui_mode": "deep",
            "use_memory": "false",
            "use_self_evolution": False,
            "include_discovered_edges": False,
            "max_total_tokens": 4096,
            "enable_path_completion": True,
            "adaptive_context": False,
        }
    )
    request = QueryRequest(**payload)
    assert request.mode == "mix"
    assert request.use_memory is False
    assert request.use_self_evolution is False
    assert request.include_discovered_edges is False
    assert request.max_total_tokens == 4096
    assert request.enable_path_completion is True
    assert request.adaptive_context is False


@pytest.mark.parametrize(
    "question",
    ["帮我总结刚才的对话", "我们聊过的预算是多少？", "Summarize our discussion"],
)
async def test_conversation_summary_can_use_history_without_document_retrieval(
    monkeypatch, question
):
    from docthinker.harness import QueryHarness

    harness = QueryHarness(
        memory_core_factory=lambda: None,
        history_loader=lambda _: [{"role": "user", "content": "预算为100元"}],
    )
    context = await harness.prepare(
        request=QueryRequest(question=question, use_memory=False)
    )
    llm = AsyncMock(return_value="预算为100元")
    monkeypatch.setattr(
        router.state, "rag_instance", SimpleNamespace(llm_model_func=llm)
    )
    assert (
        await router._conversation_fallback_answer(question, context) == "预算为100元"
    )
    assert "预算为100元" in llm.await_args.args[0]


async def test_sdk_legacy_write_can_be_disabled_without_forwarding_unknown_option():
    from docthinker.query import QueryMixin

    rag = SimpleNamespace(
        graphcore=object(),
        _execute_text_query=AsyncMock(return_value="answer"),
        add_knowledge_entry=Mock(),
        logger=Mock(),
    )
    answer = await QueryMixin.aquery(
        rag, "question", record_knowledge=False, vlm_enhanced=False
    )
    assert answer == "answer"
    rag.add_knowledge_entry.assert_not_called()
    assert "record_knowledge" not in rag._execute_text_query.await_args.kwargs
