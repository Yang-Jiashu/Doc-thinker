"""Real router paths share lifecycle, dispatch and completion ownership."""
import importlib
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import BackgroundTasks
from fastapi import HTTPException
import pytest

from docthinker.harness_runtime import current_run, execute_run
from docthinker.server.schemas import QueryRequest

query_api = importlib.import_module("docthinker.server.routers.query")


@pytest.mark.parametrize("provider_timeout", ["built_in", "legacy_asyncio"])
async def test_distinct_legacy_asyncio_and_builtin_timeouts_keep_unknown_outcome(monkeypatch, provider_timeout):
    """Exercise 3.10's distinct classes even when running on a 3.11 host."""
    class LegacyAsyncTimeout(Exception):
        pass

    monkeypatch.setattr(asyncio, "TimeoutError", LegacyAsyncTimeout)
    failure = (TimeoutError if provider_timeout == "built_in" else LegacyAsyncTimeout)("private timeout")
    seen = []

    async def operation():
        seen.append(current_run())
        raise failure

    with pytest.raises(type(failure)) as raised:
        await execute_run(operation, session_id="#00001")
    assert raised.value is failure and len(seen) == 1
    receipt = seen[0].snapshot()
    assert receipt["status"] == "unknown" and receipt["external_outcome_verified"] is False
    assert receipt["events"][-1]["payload"] == {"reason": "timeout", "external_outcome": "unknown"}
    assert not any(item["type"] == "run.completed" for item in receipt["events"])


def _stream_environment(monkeypatch, provider, write):
    rag = SimpleNamespace(graphcore=SimpleNamespace(tokenizer=None),
                          llm_model_func=AsyncMock(side_effect=AssertionError("extra model call")),
                          aquery_stream=AsyncMock(side_effect=provider))
    monkeypatch.setattr(query_api.state, "mobile_safe_mode", False)
    monkeypatch.setattr(query_api.state, "rag_instance", rag)
    monkeypatch.setattr(query_api.state, "session_manager", SimpleNamespace(
        get_history=lambda _: [], get_files=lambda _: [], add_message=write))
    monkeypatch.setattr(query_api, "_get_session_rag_or_raise", AsyncMock(return_value=rag))
    return rag


@pytest.mark.parametrize("failure_type", [TimeoutError, asyncio.TimeoutError, RuntimeError])
@pytest.mark.parametrize("failure_at", ["dispatch", "before_chunk", "after_chunk"])
async def test_stream_failure_never_commits_partial_answer_or_success(monkeypatch, failure_type, failure_at):
    seen, writes, output = [], [], []

    async def chunks():
        if failure_at == "after_chunk":
            yield "partial answer"
        raise failure_type("private provider detail")

    async def provider(**kwargs):
        seen.append(current_run())
        if failure_at == "dispatch":
            raise failure_type("private provider detail")
        return chunks()

    rag = _stream_environment(monkeypatch, provider,
                              lambda sid, role, content: writes.append((role, content)))
    background = BackgroundTasks()
    response = await query_api.query_stream(
        QueryRequest(question="q", session_id="#00001", use_memory=False), background)
    with pytest.raises(failure_type):
        async for item in response.body_iterator:
            output.append(item)
    assert writes == [("user", "q")]
    assert not background.tasks
    assert seen[0].status == ("unknown" if issubclass(failure_type, (TimeoutError, asyncio.TimeoutError)) else "failed")
    assert not any(event["type"] == "run.completed" for event in seen[0].snapshot()["events"])
    assert "private provider detail" not in "".join(output)
    assert ("partial answer" in "".join(output)) is (failure_at == "after_chunk")
    rag.aquery_stream.assert_awaited_once()
    rag.llm_model_func.assert_not_awaited()


async def test_stream_disconnect_closes_retained_provider_iterator(monkeypatch):
    seen, writes, retained = [], [], []
    closed = asyncio.Event()

    async def chunks():
        try:
            for number in range(100):
                yield f"piece-{number}"
        finally:
            closed.set()

    async def provider(**kwargs):
        seen.append(current_run())
        iterator = chunks()
        retained.append(iterator)
        return iterator

    _stream_environment(monkeypatch, provider, lambda sid, role, content: writes.append(role))
    response = await query_api.query_stream(
        QueryRequest(question="q", session_id="#00001", use_memory=False), BackgroundTasks())
    while "piece-0" not in await anext(response.body_iterator):
        pass
    await asyncio.wait_for(response.body_iterator.aclose(), 1)
    assert retained and closed.is_set()
    assert seen[0].status == "cancelled"
    assert writes == ["user"]


async def test_stream_deadline_covers_consumption_after_first_chunk(monkeypatch):
    seen, writes = [], []
    closed = asyncio.Event()

    async def chunks():
        try:
            yield "partial"
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def provider(**kwargs):
        seen.append(current_run())
        return chunks()

    rag = _stream_environment(monkeypatch, provider, lambda sid, role, content: writes.append(role))
    monkeypatch.setattr(query_api, "SESSION_QUERY_TIMEOUT_SECONDS", 0.03)
    response = await query_api.query_stream(
        QueryRequest(question="q", session_id="#00001", use_memory=False), BackgroundTasks())
    output = []
    with pytest.raises(asyncio.TimeoutError):
        async for item in response.body_iterator:
            output.append(item)
    assert "partial" in "".join(output)
    assert seen[0].status == "unknown"
    assert closed.is_set() and writes == ["user"]
    rag.aquery_stream.assert_awaited_once()
    rag.llm_model_func.assert_not_awaited()


async def test_stream_user_history_failure_does_not_break_generation_logger(monkeypatch):
    writes = []

    def write(sid, role, content):
        if role == "user":
            raise OSError("history unavailable")
        writes.append(content)

    async def provider(**kwargs):
        return "answer"

    _stream_environment(monkeypatch, provider, write)
    response = await query_api.query_stream(
        QueryRequest(question="q", session_id="#00001", use_memory=False), BackgroundTasks())
    output = [item async for item in response.body_iterator]
    assert "answer" in "".join(output)
    assert writes == ["answer"]


@pytest.mark.parametrize("failure", [TimeoutError(), asyncio.TimeoutError(), RuntimeError("provider unavailable")])
async def test_ambiguous_provider_failure_does_not_call_fallback(monkeypatch, failure):
    llm = AsyncMock(side_effect=AssertionError("must not retry paid generation"))
    rag = SimpleNamespace(graphcore=SimpleNamespace(tokenizer=None), llm_model_func=llm,
                          aquery=AsyncMock(side_effect=failure))
    writes = []
    monkeypatch.setattr(query_api.state, "mobile_safe_mode", False)
    monkeypatch.setattr(query_api.state, "rag_instance", rag)
    monkeypatch.setattr(query_api.state, "session_manager", SimpleNamespace(
        get_history=lambda _: [], get_files=lambda _: [],
        add_message=lambda sid, role, content: writes.append(role)))
    monkeypatch.setattr(query_api, "_get_session_rag_or_raise", AsyncMock(return_value=rag))
    with pytest.raises(HTTPException) as raised:
        await query_api.query(QueryRequest(question="please help", session_id="#00001", use_memory=False),
                              BackgroundTasks())
    assert raised.value.status_code == (504 if isinstance(failure, (TimeoutError, asyncio.TimeoutError)) else 500)
    rag.aquery.assert_awaited_once()
    llm.assert_not_awaited()
    assert writes == ["user"]


@pytest.mark.parametrize("stream", [False, True])
async def test_legacy_adapters_prepare_dispatch_and_commit_once(monkeypatch, stream):
    seen, writes = [], []

    async def provider(**kwargs):
        seen.append(current_run())
        assert kwargs["record_knowledge"] is False
        return "answer"

    rag = SimpleNamespace(graphcore=SimpleNamespace(tokenizer=None),
        llm_model_func=AsyncMock(side_effect=AssertionError("extra model call")),
        aquery=AsyncMock(side_effect=provider), aquery_stream=AsyncMock(side_effect=provider))
    manager = SimpleNamespace(get_history=lambda _: [], get_files=lambda _: [],
        add_message=lambda sid, role, content: writes.append((role, content)))
    monkeypatch.setattr(query_api.state, "mobile_safe_mode", False)
    monkeypatch.setattr(query_api.state, "rag_instance", rag)
    monkeypatch.setattr(query_api.state, "session_manager", manager)
    monkeypatch.setattr(query_api, "_get_session_rag_or_raise", AsyncMock(return_value=rag))
    request = QueryRequest(question="Analyze the project", session_id="#00001", use_memory=False)
    background = BackgroundTasks()
    if stream:
        response = await query_api.query_stream(request, background)
        assert not writes  # constructing a transport does not execute a turn
        output = [item async for item in response.body_iterator]
        assert "answer" in "".join(output)
    else:
        assert (await query_api.query(request, background))["answer"] == "answer"
    assert len(seen) == 1 and seen[0].status == "completed"
    assert writes == [("user", request.question), ("assistant", "answer")]
    assert not background.tasks
    stages = [event["stage"] for event in seen[0].snapshot()["events"]
              if event["type"] == "stage.started"]
    assert stages == ["prepare", "dispatch", "completion"]


async def test_research_sse_adapter_uses_same_research_operation_and_scope(monkeypatch):
    seen = []

    async def operation():
        seen.append(current_run())
        return {"answer": "verified answer", "sources": [], "research_core": True}

    async def answer(request, model, manager):
        return await execute_run(operation, session_id=request.session_id)

    service = SimpleNamespace(answer=AsyncMock(side_effect=answer))
    monkeypatch.setattr(query_api.state, "mobile_safe_mode", True)
    monkeypatch.setattr(query_api.state, "research_service", service)
    monkeypatch.setattr(query_api.state, "rag_instance", None)
    monkeypatch.setattr(query_api.state, "session_manager", SimpleNamespace(get_session=lambda _: {"id": "one"}))
    monkeypatch.setattr(query_api.state, "mobile_chat_func", None)
    response = await query_api.query_stream(QueryRequest(question="q", session_id="#00001"), BackgroundTasks())
    events = [json.loads(item.removeprefix("data: ")) async for item in response.body_iterator]
    assert events[0] == {"type": "chunk", "content": "verified answer"}
    assert events[1]["data"]["research_core"] is True
    service.answer.assert_awaited_once()
    assert len(seen) == 1 and seen[0].status == "completed"
    assert sum(row["type"] == "run.started" for row in seen[0].snapshot()["events"]) == 1


async def test_mobile_journal_request_id_becomes_shared_execution_id(tmp_path):
    from docthinker.server.request_journal import RequestJournal, request_fingerprint
    mobile = importlib.import_module("docthinker.server.routers.mobile")
    identifier = "79a2db6a-2710-47f1-a849-c70f4b5a880d"
    body = mobile.MobileQuestion(session_id="#00001", question="q", request_id=identifier)
    journal = RequestJournal(tmp_path.resolve(), boot_id="test-boot")
    assert journal.begin(body.session_id, identifier, request_fingerprint(body.model_dump()))
    seen = []

    async def answer(request, background):
        async def provider():
            seen.append(current_run())
            return {"answer": "answer", "sources": []}
        return await execute_run(provider, session_id=request.session_id)

    reply = await mobile._run_journal_query(answer, body, journal, ("unused", body.session_id, identifier))
    assert reply["request_status"] == "completed"
    assert len(seen) == 1 and seen[0].run_id == identifier
    assert seen[0].status == "completed"
    assert journal.read(body.session_id, identifier)["answer"] == "answer"


def test_mobile_insight_switch_is_independent_and_binds_request_identity(tmp_path):
    from docthinker.server.request_journal import RequestJournal, RequestConflictError, request_fingerprint
    mobile = importlib.import_module("docthinker.server.routers.mobile")
    identifier = "79a2db6a-2710-47f1-a849-c70f4b5a880d"
    original = {"session_id": "#00001", "question": "q"}
    default = mobile.MobileQuestion(**original, request_id=identifier)
    disabled = mobile.MobileQuestion(**original, request_id=identifier, insight_mode="off")
    assert disabled.knowledge_skill_mode == "auto"
    assert disabled.model_dump(exclude={"request_id"})["insight_mode"] == "off"
    assert request_fingerprint(original) == request_fingerprint(default.model_dump())
    assert request_fingerprint(default.model_dump()) != request_fingerprint(disabled.model_dump())
    journal = RequestJournal(tmp_path.resolve(), boot_id="test-boot")
    assert journal.begin(default.session_id, identifier, request_fingerprint(default.model_dump()))
    with pytest.raises(RequestConflictError):
        journal.begin(disabled.session_id, identifier, request_fingerprint(disabled.model_dump()))
