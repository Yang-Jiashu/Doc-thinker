"""Offline contracts for the dedicated private mobile entrypoint.

Production lifespan/model clients are never started. Handler delegation uses
fake stores/providers, and configuration imports cannot read a real .env file.
"""

import asyncio
import importlib
import sqlite3
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute

TOKEN = "test-only-mobile-access-token-not-a-real-secret"
PREFIX = "/api/v1/mobile"
ROUTES = {
    ("GET", "/capabilities"), ("GET", "/sessions"), ("POST", "/sessions"),
    ("GET", "/history"), ("GET", "/files"), ("GET", "/memory"), ("POST", "/query"),
    ("GET", "/feedback"), ("POST", "/feedback"),
    ("GET", "/query/status"),
    ("POST", "/upload"), ("GET", "/graph"), ("GET", "/chunks"),
    ("GET", "/voice/capabilities"), ("POST", "/voice/transcribe"),
    ("POST", "/voice/speak"),
    ("GET", "/research/capabilities"), ("GET", "/research/documents"),
    ("GET", "/research/knowledge-skills"),
    ("GET", "/research/document"), ("GET", "/research/source"),
    ("POST", "/research/upload"), ("POST", "/research/documents/delete"),
    ("POST", "/research/documents/retry"), ("GET", "/research/memory"),
    ("POST", "/research/memory"), ("POST", "/research/memory/delete"),
    ("GET", "/research/memory/history"), ("POST", "/research/memory/archive"),
    ("GET", "/research/export"),
    ("POST", "/research/entities"), ("GET", "/research/entities/resolve"),
    ("POST", "/research/events"), ("GET", "/research/events"),
    ("POST", "/research/events/revise"), ("POST", "/research/events/snapshots"),
    ("GET", "/research/event"), ("GET", "/research/event/history"),
    ("POST", "/research/event-relations"), ("GET", "/research/event-relations"),
    ("GET", "/workbench/brief"), ("POST", "/workbench/brief"),
    ("GET", "/workbench/artifacts"), ("POST", "/workbench/artifacts"),
    ("GET", "/workbench/artifact"),
    ("POST", "/workbench/artifacts/delete"),
    ("GET", "/workbench/tasks"), ("POST", "/workbench/tasks"),
    ("GET", "/workbench/task"), ("POST", "/workbench/tasks/update"),
    ("POST", "/workbench/tasks/run"), ("POST", "/workbench/tasks/control"),
    ("GET", "/workbench/task/artifact"),
    ("POST", "/workbench/tasks/replan/preview"), ("POST", "/workbench/tasks/replan"),
    ("POST", "/workbench/tasks/evidence/preview"),
    ("POST", "/workbench/long-projects"), ("GET", "/workbench/long-projects"),
    ("GET", "/workbench/long-project"), ("POST", "/workbench/long-project/phases"),
    ("GET", "/workbench/long-project/phases"), ("POST", "/workbench/long-project/phases/review"),
    ("POST", "/workbench/long-project/review"),
}


@asynccontextmanager
async def _offline_lifespan(_app):
    yield


@pytest.fixture
def mobile(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCTHINKER_MOBILE_ENABLED", "true")
    monkeypatch.setenv("DOCTHINKER_MOBILE_ACCESS_TOKEN", TOKEN)
    monkeypatch.setenv("DOCTHINKER_MOBILE_CORS_ORIGINS", "")
    monkeypatch.setenv("DOCTHINKER_VOICE_ENABLED", "false")
    with patch("dotenv.load_dotenv"):
        entry = importlib.import_module("docthinker.server.mobile_app")
        api = importlib.import_module("docthinker.server.routers.mobile")
        voice = importlib.import_module("docthinker.server.routers.voice")
        guard = importlib.import_module("docthinker.server.mobile_access")
        app = entry.create_mobile_app(lifespan=_offline_lifespan)
    manager = SimpleNamespace(
        base_storage_path=tmp_path.resolve(),
        get_session=Mock(side_effect=lambda sid: {"id": sid} if sid == "#00001" else None),
        list_sessions=Mock(return_value=[{"id": "#00001", "title": "测试会话", "path": "/private/internal"}]),
        get_history=Mock(return_value=[{"role": "user", "content": "只读测试消息"}]),
        get_files=Mock(return_value=[{
            "id": "file-1", "filename": "example.txt", "file_size": 12,
            "status": "processing", "file_path": "/private/example.txt",
        }]),
    )
    monkeypatch.setattr(api.state, "session_manager", manager)
    monkeypatch.setattr(voice, "capabilities", lambda: {"enabled": False, "asr": {"available": False}, "tts": {"available": False}})
    api._active_queries.clear()
    api._starting_query_keys.clear()
    yield SimpleNamespace(app=app, api=api, voice=voice, entry=entry, guard=guard, manager=manager)
    api._active_queries.clear()
    api._starting_query_keys.clear()


def _client(app, *, token=TOKEN):
    headers = {"Authorization": "Bearer " + token} if token is not None else {}
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://mobile.test", headers=headers,
    )


def test_mobile_entrypoint_mounts_only_allowlisted_routes_and_shared_dependency(mobile):
    registered = {
        (method.upper(), path.removeprefix(PREFIX))
        for path, methods in mobile.app.openapi()["paths"].items()
        for method in methods
    }
    # Newer FastAPI keeps lazy included-router wrappers; older versions flatten
    # them into APIRoutes. Both must exclude arbitrary mounts/admin routers.
    for route in mobile.app.routes:
        if isinstance(route, APIRoute):
            assert route.path.startswith(PREFIX + "/")
            assert any(item.call is mobile.guard.require_mobile_access for item in route.dependant.dependencies), route.path
        else:
            assert getattr(route, "original_router", None) in (mobile.api.router, mobile.voice.router)
    def check_protected(router):
        for route in router.routes:
            if isinstance(route, APIRoute):
                assert any(item.call is mobile.guard.require_mobile_access for item in route.dependant.dependencies), route.path
            else:
                assert getattr(route, "original_router", None) in (
                    mobile.api.workbench_router, mobile.api.project_tasks_router,
                    mobile.api.personal_events_router, mobile.api.long_projects_router)
                check_protected(route.original_router)
    for router in (mobile.api.router, mobile.voice.router):
        check_protected(router)
    assert registered == ROUTES


@pytest.mark.parametrize("path", [
    "/docs", "/redoc", "/openapi.json", "/api/v1/settings", "/api/v1/sessions",
    "/api/v1/query", "/api/v1/ingest", "/api/v1/knowledge-graph/data",
    "/api/v1/mobile/settings", "/api/v1/mobile/admin",
])
async def test_legacy_and_administration_endpoints_are_not_exposed(mobile, path):
    async with _client(mobile.app) as client:
        assert (await client.get(path)).status_code == 404
        assert (await client.post(path, json={})).status_code == 404


@pytest.mark.parametrize("method,path", sorted(ROUTES))
async def test_every_mobile_route_requires_token_before_handler(mobile, method, path):
    options = {"params": {"session_id": "#00001", "entity_id": "能源"}}
    if path in {"/upload", "/voice/transcribe"}:
        options = {"data": {"session_id": "#00001"} if path == "/upload" else {}, "files": {"file": ("test.txt", b"test", "text/plain")}}
    elif method == "POST":
        options = {"json": {"title": "测试", "session_id": "#00001", "question": "测试", "text": "测试"}}
    async with _client(mobile.app, token=None) as client:
        response = await client.request(method, PREFIX + path, **options)
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"
    assert TOKEN not in response.text
    mobile.manager.get_session.assert_not_called()
    mobile.manager.list_sessions.assert_not_called()


@pytest.mark.parametrize("enabled,token,status", [("false", TOKEN, 404), ("", TOKEN, 404), ("true", "short", 503), ("true", "", 503)])
async def test_mobile_is_opt_in_and_rejects_missing_server_configuration(mobile, monkeypatch, enabled, token, status):
    monkeypatch.setenv("DOCTHINKER_MOBILE_ENABLED", enabled)
    monkeypatch.setenv("DOCTHINKER_MOBILE_ACCESS_TOKEN", token)
    async with _client(mobile.app) as client:
        result = await client.get(PREFIX + "/capabilities")
    assert result.status_code == status


@pytest.mark.parametrize("authorization", ["Basic " + TOKEN, "Bearer wrong", "Bearer ", "Bearer " + TOKEN + "extra"])
async def test_incorrect_tokens_are_rejected_without_echoing_them(mobile, authorization):
    async with _client(mobile.app, token=None) as client:
        response = await client.get(PREFIX + "/capabilities", headers={"Authorization": authorization})
    assert response.status_code == 401
    assert TOKEN not in response.text
    assert authorization not in response.text


async def test_unauthenticated_multipart_is_rejected_before_body_parsing(mobile):
    async with _client(mobile.app, token=None) as client:
        for path in ("/upload", "/voice/transcribe"):
            response = await client.post(PREFIX + path, content=b"not multipart", headers={"Content-Type": "multipart/form-data"})
            assert response.status_code == 401, (path, response.text)


async def test_unauthorized_and_oversized_uploads_do_not_consume_request_stream(mobile):
    consumed = []

    async def body():
        consumed.append(True)
        yield b"not a real user file"

    async with _client(mobile.app, token=None) as client:
        response = await client.post(PREFIX + "/upload", content=body(), headers={"Content-Type": "multipart/form-data"})
        assert response.status_code == 401
    assert not consumed
    async with _client(mobile.app) as client:
        response = await client.post(PREFIX + "/upload", content=body(), headers={
            "Content-Type": "multipart/form-data", "Content-Length": str(21 * 1024 * 1024),
        })
        assert response.status_code == 413
    assert not consumed


async def test_private_capability_and_read_dtos_do_not_claim_customer_auth(mobile):
    async with _client(mobile.app) as client:
        capability = (await client.get(PREFIX + "/capabilities")).json()
        assert capability["customer_auth"] is False
        assert capability["audience"] == "private-development"
        assert capability["realtime_voice"] is False
        sessions = await client.get(PREFIX + "/sessions")
        assert sessions.json() == {"sessions": [{"id": "#00001", "title": "测试会话"}]}
        files = await client.get(PREFIX + "/files", params={"session_id": "#00001"})
        assert files.json()["files"][0]["status"] == "processing"
        assert "file_path" not in files.json()["files"][0]
        history = await client.get(PREFIX + "/history", params={"session_id": "#00001"})
        assert history.json()["history"][0]["content"] == "只读测试消息"


@pytest.mark.parametrize("sid,status", [("#00002", 404), ("../#00001", 400), ("#00001/anything", 400), ("", 400)])
async def test_invalid_or_missing_sessions_never_fall_back(mobile, sid, status):
    async with _client(mobile.app) as client:
        for path in ("/history", "/files", "/graph", "/chunks", "/memory", "/feedback"):
            response = await client.get(PREFIX + path, params={"session_id": sid, "entity_id": "能源"})
            assert response.status_code == status
    mobile.manager.get_history.assert_not_called()
    mobile.manager.get_files.assert_not_called()


@pytest.mark.parametrize("path,payload", [
    ("/sessions", {"title": ""}), ("/sessions", {"title": "a" * 81}),
    ("/query", {"session_id": "#00001", "question": ""}),
    ("/query", {"session_id": "#00001", "question": "q" * 8001}),
])
async def test_mobile_input_lengths_are_bounded(mobile, path, payload):
    async with _client(mobile.app) as client:
        response = await client.post(PREFIX + path, json=payload)
    assert response.status_code == 422


async def test_mobile_query_forwards_explicit_controls_without_hidden_retries(mobile, monkeypatch):
    with patch("dotenv.load_dotenv"):
        underlying = importlib.import_module("docthinker.server.routers.query")
    ask = AsyncMock(return_value={"answer": "mock only", "sources": [], "internal_debug": "/private/secret"})
    monkeypatch.setattr(underlying, "query", ask)
    async with _client(mobile.app) as client:
        response = await client.post(PREFIX + "/query", json={
            "session_id": "#00001", "question": "只运行mock",
            "use_memory": False, "use_conversation_context": False,
            "use_llm_cache": False, "use_self_evolution": False,
        })
    assert response.status_code == 200
    assert response.json() == {"answer": "mock only", "sources": []}
    ask.assert_awaited_once()
    request = ask.await_args.args[0]
    assert request.session_id == "#00001"
    assert not any((request.use_memory, request.use_conversation_context, request.use_llm_cache, request.use_self_evolution))
    assert not mobile.api._active_queries


FEEDBACK_ANSWER_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
FEEDBACK_USER_ID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


async def test_feedback_reads_do_not_create_storage_and_auth_rejects_before_read(mobile):
    async with _client(mobile.app) as client:
        response = await client.get(PREFIX + "/feedback", params={"session_id": "#00001"})
    assert response.json() == {"items": []}
    assert not (mobile.manager.base_storage_path / "answer_feedback.db").exists()


async def test_feedback_storage_outage_does_not_hide_chat_history_or_leak_path(mobile):
    sensitive = mobile.manager.base_storage_path / "synthetic-private-feedback.txt"
    sensitive.write_text("synthetic private data")
    (mobile.manager.base_storage_path / "answer_feedback.db").symlink_to(sensitive)
    async with _client(mobile.app) as client:
        history = await client.get(PREFIX + "/history", params={"session_id": "#00001"})
        feedback = await client.get(PREFIX + "/feedback", params={"session_id": "#00001"})
    assert history.status_code == 200
    assert history.json()["history"] == mobile.manager.get_history.return_value
    assert feedback.status_code == 503
    assert str(sensitive) not in feedback.text and "synthetic private data" not in feedback.text
    assert sensitive.read_text() == "synthetic private data"


async def test_feedback_resolves_real_assistant_turn_and_durably_updates_without_model(mobile, monkeypatch):
    mobile.manager.get_history.return_value = [
        {"id": FEEDBACK_USER_ID, "role": "user", "content": "synthetic question"},
        {"id": FEEDBACK_ANSWER_ID, "role": "assistant", "content": "synthetic answer"},
    ]
    payload = {"session_id": "#00001", "answer_id": FEEDBACK_ANSWER_ID, "rating": "helpful", "note": "明确人工反馈"}
    with patch("docthinker.server.routers.query.query", side_effect=AssertionError("Feedback must not ask a model")):
        async with _client(mobile.app) as client:
            first = await client.post(PREFIX + "/feedback", json=payload)
            assert first.status_code == 200
            assert first.json()["feedback"]["note"] == payload["note"]
            second = await client.post(PREFIX + "/feedback", json={**payload, "rating": "inaccurate", "note": "修正了评价"})
            assert second.status_code == 200
            loaded = await client.get(PREFIX + "/feedback", params={"session_id": "#00001"})
            history = await client.get(PREFIX + "/history", params={"session_id": "#00001"})
    assert loaded.json()["items"] == [second.json()["feedback"]]
    assert history.json()["history"][1]["feedback"] == second.json()["feedback"]
    assert "feedback" not in history.json()["history"][0]
    assert mobile.manager.get_history.call_count == 3


@pytest.mark.parametrize("answer_id", [FEEDBACK_USER_ID, "cccccccc-cccc-4ccc-8ccc-cccccccccccc"])
async def test_feedback_rejects_user_or_foreign_turn_before_any_storage_write(mobile, answer_id):
    mobile.manager.get_history.return_value = [{"id": FEEDBACK_USER_ID, "role": "user", "content": "synthetic"}]
    async with _client(mobile.app) as client:
        response = await client.post(PREFIX + "/feedback", json={
            "session_id": "#00001", "answer_id": answer_id, "rating": "helpful",
        })
    assert response.status_code == 404
    assert not (mobile.manager.base_storage_path / "answer_feedback.db").exists()


@pytest.mark.parametrize("updates", [
    {"session_id": "#00002"}, {"session_id": "../#00001"},
    {"answer_id": "0"}, {"answer_id": "../anything"}, {"rating": "train"},
    {"note": "a" * 501}, {"note": None}, {"note": 123},
    {"user_id": "someone-else"}, {"answer": "forged"},
])
async def test_feedback_identity_and_content_contract_is_bounded(mobile, updates):
    payload = {"session_id": "#00001", "answer_id": FEEDBACK_ANSWER_ID, "rating": "helpful", **updates}
    async with _client(mobile.app) as client:
        response = await client.post(PREFIX + "/feedback", json=payload)
    assert response.status_code in {400, 404, 422}
    assert not (mobile.manager.base_storage_path / "answer_feedback.db").exists()


async def test_mobile_query_returns_real_persisted_answer_reference(mobile, monkeypatch):
    with patch("dotenv.load_dotenv"):
        underlying = importlib.import_module("docthinker.server.routers.query")
    ask = AsyncMock(return_value={"answer": "synthetic", "answer_id": FEEDBACK_ANSWER_ID, "sources": []})
    monkeypatch.setattr(underlying, "query", ask)
    async with _client(mobile.app) as client:
        response = await client.post(PREFIX + "/query", json={"session_id": "#00001", "question": "synthetic"})
    assert response.json() == {"answer": "synthetic", "answer_id": FEEDBACK_ANSWER_ID, "sources": []}


@pytest.mark.parametrize("answer_id", [None, "not-persisted", 123])
async def test_mobile_query_does_not_offer_feedback_for_missing_or_invalid_reference(mobile, monkeypatch, answer_id):
    with patch("dotenv.load_dotenv"):
        underlying = importlib.import_module("docthinker.server.routers.query")
    ask = AsyncMock(return_value={"answer": "synthetic", "answer_id": answer_id, "sources": []})
    monkeypatch.setattr(underlying, "query", ask)
    async with _client(mobile.app) as client:
        response = await client.post(PREFIX + "/query", json={"session_id": "#00001", "question": "synthetic"})
    assert response.json() == {"answer": "synthetic", "sources": []}


async def test_session_creation_strips_legacy_internal_paths(mobile, monkeypatch):
    with patch("dotenv.load_dotenv"):
        underlying = importlib.import_module("docthinker.server.routers.sessions")
    create = AsyncMock(return_value={"status": "success", "session": {
        "id": "#00002", "title": "新对话", "path": "/private/internal",
        "talk_dir": "/private/talk", "knowledge_dir": "/private/knowledge",
    }})
    monkeypatch.setattr(underlying, "create_session", create)
    async with _client(mobile.app) as client:
        response = await client.post(PREFIX + "/sessions", json={"title": "新对话"})
    assert response.status_code == 200
    assert response.json() == {"session": {"id": "#00002", "title": "新对话"}}
    create.assert_awaited_once()


@pytest.mark.parametrize("status", [404, 500, None])
async def test_delegated_errors_never_echo_internal_provider_or_path_details(mobile, monkeypatch, status):
    with patch("dotenv.load_dotenv"):
        underlying = importlib.import_module("docthinker.server.routers.graph")
    sensitive = "PRIVATE_TEST_PROVIDER_KEY /private/configuration/path"
    error = HTTPException(status, sensitive) if status else RuntimeError(sensitive)
    get_graph = AsyncMock(side_effect=error)
    monkeypatch.setattr(underlying, "get_graph_data", get_graph)
    async with _client(mobile.app) as client:
        response = await client.get(PREFIX + "/graph", params={"session_id": "#00001"})
    assert response.status_code == (status or 500)
    assert "PRIVATE_TEST_PROVIDER_KEY" not in response.text
    assert "/private/" not in response.text
    assert response.headers["cache-control"] == "no-store"


async def test_same_session_query_singleflight_releases_after_completion(mobile, monkeypatch):
    with patch("dotenv.load_dotenv"):
        underlying = importlib.import_module("docthinker.server.routers.query")
    entered, release = asyncio.Event(), asyncio.Event()

    async def ask(*_args):
        entered.set()
        await release.wait()
        return {"answer": "mock only", "sources": []}

    monkeypatch.setattr(underlying, "query", ask)
    async with _client(mobile.app) as client:
        first = asyncio.create_task(client.post(PREFIX + "/query", json={"session_id": "#00001", "question": "first"}))
        await asyncio.wait_for(entered.wait(), 2)
        try:
            second = await client.post(PREFIX + "/query", json={"session_id": "#00001", "question": "second"})
            assert second.status_code == 429
        finally:
            release.set()
            response = await first
        assert response.status_code == 200
        assert not mobile.api._active_queries


REQUEST_ID = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"


async def test_query_status_is_authenticated_read_only_without_model_or_database(mobile, monkeypatch):
    with patch("dotenv.load_dotenv"):
        underlying = importlib.import_module("docthinker.server.routers.query")
    ask = AsyncMock(side_effect=AssertionError("Status must not ask a model"))
    monkeypatch.setattr(underlying, "query", ask)
    async with _client(mobile.app) as browser:
        response = await browser.get(PREFIX + "/query/status", params={"session_id": "#00001", "request_id": REQUEST_ID})
    assert response.json() == {"request_id": REQUEST_ID, "request_status": "not_found"}
    assert not (mobile.manager.base_storage_path / "request_journal.db").exists()
    ask.assert_not_called()


async def test_idempotent_query_replays_exact_completion_and_survives_restart(mobile, monkeypatch):
    with patch("dotenv.load_dotenv"):
        underlying = importlib.import_module("docthinker.server.routers.query")
    ask = AsyncMock(return_value={"answer": "synthetic once", "sources": [], "answer_id": FEEDBACK_ANSWER_ID, "context_windowed": True})
    monkeypatch.setattr(underlying, "query", ask)
    payload = {"session_id": "#00001", "question": "synthetic private question", "request_id": REQUEST_ID}
    async with _client(mobile.app) as browser:
        first = await browser.post(PREFIX + "/query", json=payload)
        assert first.status_code == 200
        monkeypatch.setattr(mobile.api, "_query_boot_id", "restarted-test-process")
        replay = await browser.post(PREFIX + "/query", json={**payload, "use_memory": True})
        status = await browser.get(PREFIX + "/query/status", params={"session_id": "#00001", "request_id": REQUEST_ID})
    assert first.json() == replay.json() == status.json() == {
        "answer": "synthetic once", "sources": [], "answer_id": FEEDBACK_ANSWER_ID,
        "context_windowed": True, "request_id": REQUEST_ID, "request_status": "completed",
    }
    ask.assert_awaited_once()
    assert not hasattr(ask.await_args.args[0], "request_id")
    assert "synthetic private question".encode() not in (mobile.manager.base_storage_path / "request_journal.db").read_bytes()


async def test_changed_payload_id_is_rejected_without_a_second_call(mobile, monkeypatch):
    with patch("dotenv.load_dotenv"):
        underlying = importlib.import_module("docthinker.server.routers.query")
    ask = AsyncMock(return_value={"answer": "original", "sources": []})
    monkeypatch.setattr(underlying, "query", ask)
    payload = {"session_id": "#00001", "question": "synthetic", "request_id": REQUEST_ID}
    async with _client(mobile.app) as browser:
        assert (await browser.post(PREFIX + "/query", json=payload)).status_code == 200
        for changed in ({"question": "different"}, {"use_self_evolution": True}):
            conflict = await browser.post(PREFIX + "/query", json={**payload, **changed})
            assert conflict.status_code == 409
    ask.assert_awaited_once()


async def test_timeout_and_duplicate_inflight_return_running_without_resending(mobile, monkeypatch):
    with patch("dotenv.load_dotenv"):
        underlying = importlib.import_module("docthinker.server.routers.query")
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def ask(*args):
        calls.append(args)
        entered.set()
        await release.wait()
        return {"answer": "completed later", "sources": []}
    monkeypatch.setattr(underlying, "query", ask)
    monkeypatch.setattr(mobile.api, "QUERY_WAIT_SECONDS", 0.01)
    payload = {"session_id": "#00001", "question": "synthetic", "request_id": REQUEST_ID}
    async with _client(mobile.app) as browser:
        first = await browser.post(PREFIX + "/query", json=payload)
        assert first.status_code == 202 and first.json()["request_status"] == "running"
        task = next(iter(mobile.api._query_tasks.values()))
        try:
            replay = await browser.post(PREFIX + "/query", json=payload)
            assert replay.status_code == 202 and replay.json()["request_status"] == "running"
            history = await browser.get(PREFIX + "/history", params={"session_id": "#00001"})
            assert history.json()["query_requests"] == [{"request_id": REQUEST_ID, "request_status": "running"}]
            status = await browser.get(PREFIX + "/query/status", params={"session_id": "#00001", "request_id": REQUEST_ID})
            assert status.json()["request_status"] == "running"
            assert len(calls) == 1
            assert "#00001" in mobile.api._active_queries
        finally:
            release.set()
            await asyncio.wait_for(task, 2)
        completed = await browser.get(PREFIX + "/query/status", params={"session_id": "#00001", "request_id": REQUEST_ID})
        assert completed.json()["request_status"] == "completed"
        assert completed.json()["answer"] == "completed later"
        history = await browser.get(PREFIX + "/history", params={"session_id": "#00001"})
        assert "query_requests" not in history.json()
    assert not mobile.api._active_queries and not mobile.api._query_tasks


async def test_disconnected_http_request_does_not_cancel_the_one_model_task(mobile, monkeypatch):
    with patch("dotenv.load_dotenv"):
        underlying = importlib.import_module("docthinker.server.routers.query")
    entered, release = asyncio.Event(), asyncio.Event()
    ask_calls = []
    async def ask(*args):
        ask_calls.append(args)
        entered.set()
        await release.wait()
        return {"answer": "not canceled", "sources": []}
    monkeypatch.setattr(underlying, "query", ask)
    payload = {"session_id": "#00001", "question": "synthetic", "request_id": REQUEST_ID}
    async with _client(mobile.app) as browser:
        connection = asyncio.create_task(browser.post(PREFIX + "/query", json=payload))
        await asyncio.wait_for(entered.wait(), 2)
        model_task = next(iter(mobile.api._query_tasks.values()))
        try:
            connection.cancel()
            with pytest.raises(asyncio.CancelledError):
                await connection
            assert not model_task.done()
            assert "#00001" in mobile.api._active_queries
        finally:
            release.set()
            await asyncio.wait_for(model_task, 2)
        replay = await browser.post(PREFIX + "/query", json=payload)
        assert replay.json()["answer"] == "not canceled"
    assert len(ask_calls) == 1


async def test_interrupted_old_process_request_is_unknown_never_retried(mobile, monkeypatch):
    from docthinker.server.request_journal import RequestJournal, request_fingerprint
    with patch("dotenv.load_dotenv"):
        underlying = importlib.import_module("docthinker.server.routers.query")
    ask = AsyncMock(side_effect=AssertionError("Unknown requests must not be retried"))
    monkeypatch.setattr(underlying, "query", ask)
    payload = {"session_id": "#00001", "question": "synthetic", "request_id": REQUEST_ID}
    old = RequestJournal(mobile.manager.base_storage_path, boot_id="lost-process")
    old.begin("#00001", REQUEST_ID, request_fingerprint(payload))
    async with _client(mobile.app) as browser:
        replay = await browser.post(PREFIX + "/query", json=payload)
        assert replay.status_code == 202 and replay.json()["request_status"] == "unknown"
        status = await browser.get(PREFIX + "/query/status", params={"session_id": "#00001", "request_id": REQUEST_ID})
        assert status.json()["request_status"] == "unknown"
        history = await browser.get(PREFIX + "/history", params={"session_id": "#00001"})
        assert history.json()["query_requests"] == [{"request_id": REQUEST_ID, "request_status": "unknown"}]
    ask.assert_not_called()


async def test_provider_failure_is_unknown_and_never_echoes_or_retries_error(mobile, monkeypatch):
    with patch("dotenv.load_dotenv"):
        underlying = importlib.import_module("docthinker.server.routers.query")
    ask = AsyncMock(side_effect=RuntimeError("private-synthetic-key /internal/path"))
    monkeypatch.setattr(underlying, "query", ask)
    payload = {"session_id": "#00001", "question": "synthetic", "request_id": REQUEST_ID}
    async with _client(mobile.app) as browser:
        first = await browser.post(PREFIX + "/query", json=payload)
        second = await browser.post(PREFIX + "/query", json=payload)
    assert first.status_code == second.status_code == 202
    assert first.json() == second.json() == {"request_id": REQUEST_ID, "request_status": "unknown"}
    assert "private-synthetic-key" not in first.text
    ask.assert_awaited_once()


@pytest.mark.parametrize("code", ["model_response_truncated", "model_response_empty", "model_response_invalid"])
async def test_known_model_response_failure_survives_real_query_router_and_replays_without_cost(mobile, monkeypatch, code):
    from docthinker.server.model_response_error import ModelResponseError, model_response_failure
    with patch("dotenv.load_dotenv"):
        underlying = importlib.import_module("docthinker.server.routers.query")
    answer = AsyncMock(side_effect=ModelResponseError(code))
    monkeypatch.setattr(underlying.state, "mobile_safe_mode", True)
    monkeypatch.setattr(underlying.state, "research_service", SimpleNamespace(answer=answer))
    monkeypatch.setattr(underlying.state, "mobile_chat_func", None)
    payload = {"session_id": "#00001", "question": "private question", "request_id": REQUEST_ID}
    expected = {"request_id": REQUEST_ID, "request_status": "failed", "request_error": model_response_failure(code)}
    async with _client(mobile.app) as browser:
        first = await browser.post(PREFIX + "/query", json=payload)
        monkeypatch.setattr(mobile.api, "_query_boot_id", "restarted-test-process")
        replay = await browser.post(PREFIX + "/query", json=payload)
        status = await browser.get(PREFIX + "/query/status", params={"session_id": "#00001", "request_id": REQUEST_ID})
        conflict = await browser.post(PREFIX + "/query", json={**payload, "question": "different"})
        history = await browser.get(PREFIX + "/history", params={"session_id": "#00001"})
    assert first.status_code == replay.status_code == status.status_code == 200
    assert first.json() == replay.json() == status.json() == expected
    assert conflict.status_code == 409
    assert "query_requests" not in history.json()
    assert "answer" not in first.json()
    answer.assert_awaited_once()


async def test_known_failure_without_legacy_request_id_is_not_reported_as_unknown(mobile, monkeypatch):
    from docthinker.server.model_response_error import ModelResponseError
    with patch("dotenv.load_dotenv"):
        underlying = importlib.import_module("docthinker.server.routers.query")
    ask = AsyncMock(side_effect=ModelResponseError("model_response_empty"))
    monkeypatch.setattr(underlying, "query", ask)
    async with _client(mobile.app) as browser:
        result = await browser.post(PREFIX + "/query", json={"session_id": "#00001", "question": "hello"})
    assert result.status_code == 200 and result.json()["request_status"] == "failed"
    assert "request_id" not in result.json() and "answer" not in result.json()
    ask.assert_awaited_once()


async def test_invitation_worker_allows_one_live_model_across_two_sessions_but_status_works(mobile, monkeypatch):
    with patch("dotenv.load_dotenv"):
        underlying = importlib.import_module("docthinker.server.routers.query")
    monkeypatch.setenv("DOCTHINKER_INVITE_WORKER_ID", "offline-worker")
    mobile.manager.get_session.side_effect = lambda sid: {"id": sid} if sid in {"#00001", "#00002"} else None
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def ask(*args):
        calls.append(args)
        entered.set()
        await release.wait()
        return {"answer": "synthetic", "sources": []}
    monkeypatch.setattr(underlying, "query", ask)
    payload = {"session_id": "#00001", "question": "synthetic", "request_id": REQUEST_ID}
    async with _client(mobile.app) as browser:
        first = asyncio.create_task(browser.post(PREFIX + "/query", json=payload))
        await asyncio.wait_for(entered.wait(), 2)
        try:
            second = await browser.post(PREFIX + "/query", json={**payload, "session_id": "#00002", "request_id": FEEDBACK_ANSWER_ID})
            legacy = await browser.post(PREFIX + "/query", json={"session_id": "#00002", "question": "synthetic"})
            assert second.status_code == legacy.status_code == 429
            replay = await browser.post(PREFIX + "/query", json=payload)
            assert replay.status_code == 202 and replay.json()["request_status"] == "running"
            status = await browser.get(PREFIX + "/query/status", params={"session_id": "#00001", "request_id": REQUEST_ID})
            assert status.json()["request_status"] == "running"
            assert len(calls) == 1
        finally:
            release.set()
            assert (await first).status_code == 200
        assert not mobile.api._active_queries


async def test_execution_deadline_is_unknown_and_legacy_shape_is_unchanged(mobile, monkeypatch):
    with patch("dotenv.load_dotenv"):
        underlying = importlib.import_module("docthinker.server.routers.query")
    ask = AsyncMock(side_effect=lambda *_: asyncio.sleep(10))
    async def blocking(*_):
        await asyncio.Event().wait()
    monkeypatch.setattr(underlying, "query", blocking)
    monkeypatch.setattr(mobile.api, "QUERY_EXECUTION_SECONDS", 0.01)
    async with _client(mobile.app) as browser:
        body = {"session_id": "#00001", "question": "synthetic", "request_id": REQUEST_ID}
        response = await browser.post(PREFIX + "/query", json=body)
        assert response.status_code == 202 and response.json()["request_status"] == "unknown"
        body.pop("request_id")
        old = await browser.post(PREFIX + "/query", json=body)
        assert old.status_code == 504 and "request_status" not in old.json()
    assert not mobile.api._active_queries


@pytest.mark.parametrize("request_id", ["not-a-uuid", "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA", "aaaaaaaa-aaaa-3aaa-8aaa-aaaaaaaaaaaa"])
async def test_query_request_id_validation_happens_before_model(mobile, monkeypatch, request_id):
    with patch("dotenv.load_dotenv"):
        underlying = importlib.import_module("docthinker.server.routers.query")
    ask = AsyncMock()
    monkeypatch.setattr(underlying, "query", ask)
    async with _client(mobile.app) as browser:
        response = await browser.post(PREFIX + "/query", json={"session_id": "#00001", "question": "synthetic", "request_id": request_id})
        status = await browser.get(PREFIX + "/query/status", params={"session_id": "#00001", "request_id": request_id})
    assert response.status_code == status.status_code == 422
    ask.assert_not_called()


async def test_mobile_graph_and_chunks_delegate_only_the_requested_bounded_scope(mobile, monkeypatch):
    with patch("dotenv.load_dotenv"):
        underlying = importlib.import_module("docthinker.server.routers.graph")
    graph = AsyncMock(return_value={
        "nodes": [{"id": "能源", "label": "能源", "file_path": "/private/document.pdf"}],
        "edges": [], "metadata": {"graph_file": "/private/graph.graphml"},
    })
    chunks = AsyncMock(return_value={"chunks": [{"chunk_id": "chunk-1", "content": "mock evidence", "file_path": "/private/document.pdf"}]})
    monkeypatch.setattr(underlying, "get_graph_data", graph)
    monkeypatch.setattr(underlying, "get_entity_chunks", chunks)
    async with _client(mobile.app) as client:
        result = await client.get(PREFIX + "/graph", params={"session_id": "#00001", "focus_node": "能源 #?&", "scope": "full"})
        assert result.status_code == 200
        assert "/private/" not in result.text
        result = await client.get(PREFIX + "/chunks", params={"session_id": "#00001", "entity_id": "能源 #?&", "max_chunks": 0})
        assert result.status_code == 200
        assert result.json() == {"chunks": [{"id": "chunk-1", "content": "mock evidence"}]}
    graph.assert_awaited_once_with(session_id="#00001", scope="summary", focus_node="能源 #?&")
    chunks.assert_awaited_once_with(session_id="#00001", entity_id="能源 #?&", max_chunks=3)


async def test_mobile_cors_requires_explicit_origin(mobile, monkeypatch):
    monkeypatch.setenv("DOCTHINKER_MOBILE_CORS_ORIGINS", "*")
    with pytest.raises(ValueError, match="explicit"):
        mobile.entry.create_mobile_app(lifespan=_offline_lifespan)
    monkeypatch.setenv("DOCTHINKER_MOBILE_CORS_ORIGINS", "http://127.0.0.1:5174")
    app = mobile.entry.create_mobile_app(lifespan=_offline_lifespan)
    async with _client(app, token=None) as client:
        headers = {"Origin": "http://127.0.0.1:5174", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type"}
        allowed = await client.options(PREFIX + "/query", headers=headers)
        assert allowed.status_code == 200
        assert allowed.headers["access-control-allow-origin"] == "http://127.0.0.1:5174"
        headers["Origin"] = "https://untrusted.invalid"
        rejected = await client.options(PREFIX + "/query", headers=headers)
        assert rejected.status_code == 400
        assert "access-control-allow-origin" not in rejected.headers


@pytest.fixture
def memory_db(mobile, monkeypatch, tmp_path):
    path = tmp_path / "offline-memory.db"
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE long_horizon_memories (
            id TEXT PRIMARY KEY, session_id TEXT, scope_key TEXT, scope TEXT,
            status TEXT, kind TEXT, summary TEXT, question TEXT,
            created_at REAL, last_seen_at REAL, confidence REAL,
            use_count INTEGER, audit_json TEXT
        )""")

    def insert(id="memory-1", sid="#00001", scope="session", status="active", **values):
        data = {
            "id": id, "session_id": sid, "scope_key": f"{scope}:{sid}" if scope == "session" else f"{scope}:default",
            "scope": scope, "status": status, "kind": "insight", "summary": "助手整理的离线测试摘要",
            "question": "用户的离线测试提问", "created_at": 1700000000, "last_seen_at": 1700000000,
            "confidence": 0.97, "use_count": 7, "audit_json": '{"file_path":"/private/secret"}',
            **values,
        }
        with sqlite3.connect(path) as connection:
            connection.execute(
                f"INSERT INTO long_horizon_memories ({','.join(data)}) VALUES ({','.join('?' for _ in data)})",
                list(data.values()),
            )
    monkeypatch.setattr(mobile.api, "_memory_database_path", lambda: path)
    return SimpleNamespace(path=path, insert=insert)


async def test_memory_reads_only_exact_session_active_scope_with_safe_dto(mobile, memory_db):
    memory_db.insert()
    memory_db.insert("other-session", sid="#00002")
    memory_db.insert("shared-project", scope="project")
    memory_db.insert("shared-user", scope="user")
    memory_db.insert("deleted", status="deleted")
    memory_db.insert("candidate", status="candidate")
    memory_db.insert("mismatched-owner", sid="#00002", scope_key="session:#00001")
    memory_db.insert("mismatched-key", scope_key="session:#00002")
    before = memory_db.path.read_bytes()
    async with _client(mobile.app) as client:
        response = await client.get(PREFIX + "/memory", params={"session_id": "#00001", "scope": "user", "status": "deleted"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["scope"] == "session"
    assert payload["has_more"] is False and payload["next_offset"] is None
    assert [item["id"] for item in payload["items"]] == ["memory-1"]
    item = payload["items"][0]
    assert item["source_type"] == "assistant_summary" and item["question_source"] == "user_message"
    assert set(item) == {"id", "kind", "summary", "question", "created_at", "source_type", "question_source", "truncated_fields"}
    assert all(field not in response.text for field in ("confidence", "audit", "file_path", "use_count", "/private/"))
    assert memory_db.path.read_bytes() == before


async def test_memory_paginates_without_loading_shared_records_and_marks_truncation(mobile, memory_db):
    for i in range(23):
        memory_db.insert(f"m-{i:02d}", summary="摘" * 1600, question="问" * 900, kind="k" * 60, last_seen_at=1700000100 - i)
    async with _client(mobile.app) as client:
        first = (await client.get(PREFIX + "/memory", params={"session_id": "#00001"})).json()
        second = (await client.get(PREFIX + "/memory", params={"session_id": "#00001", "offset": 20})).json()
    assert len(first["items"]) == 20 and first["has_more"] is True and first["next_offset"] == 20
    assert len(second["items"]) == 3 and second["has_more"] is False
    assert not ({item["id"] for item in first["items"]} & {item["id"] for item in second["items"]})
    assert len(first["items"][0]["summary"]) == 1500
    assert len(first["items"][0]["question"]) == 800
    assert len(first["items"][0]["kind"]) == 48
    assert first["items"][0]["truncated_fields"] == ["kind", "summary", "question"]
    assert "total" not in first


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 21}, {"offset": -1}, {"offset": 1001}, {"limit": "not-a-number"}])
async def test_memory_rejects_unbounded_pages_before_reading(mobile, monkeypatch, params):
    read = Mock(side_effect=AssertionError("must not read"))
    monkeypatch.setattr(mobile.api, "_read_session_memories", read)
    async with _client(mobile.app) as client:
        response = await client.get(PREFIX + "/memory", params={"session_id": "#00001", **params})
    assert response.status_code == 422
    read.assert_not_called()


async def test_memory_missing_database_stays_missing_and_errors_are_sanitized(mobile, monkeypatch, tmp_path):
    missing = tmp_path / "not-created" / "memory.db"
    monkeypatch.setattr(mobile.api, "_memory_database_path", lambda: missing)
    async with _client(mobile.app) as client:
        response = await client.get(PREFIX + "/memory", params={"session_id": "#00001"})
        assert response.status_code == 200 and response.json()["items"] == []
        assert not missing.parent.exists()
        monkeypatch.setattr(mobile.api, "_read_session_memories", Mock(side_effect=RuntimeError("/private/database secret-provider-error")))
        error = await client.get(PREFIX + "/memory", params={"session_id": "#00001"})
    assert error.status_code == 503
    assert "private" not in error.text and "secret" not in error.text


async def test_memory_connection_is_read_only_and_does_not_mutate_usage(mobile, memory_db, monkeypatch):
    memory_db.insert()
    connect = sqlite3.connect
    statements = []
    def readonly_connection(database, **kwargs):
        assert database.endswith("?mode=ro") and kwargs["uri"] is True
        connection = connect(database, **kwargs)
        connection.set_trace_callback(statements.append)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("UPDATE long_horizon_memories SET use_count = 999")
        connection.rollback()
        statements.clear()
        return connection
    monkeypatch.setattr(mobile.api.sqlite3, "connect", readonly_connection)
    async with _client(mobile.app) as client:
        result = await client.get(PREFIX + "/memory", params={"session_id": "#00001"})
    assert result.status_code == 200
    assert statements and all(statement.lstrip().upper().startswith("SELECT") for statement in statements)
    with connect(memory_db.path) as connection:
        assert connection.execute("SELECT use_count FROM long_horizon_memories").fetchone()[0] == 7


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
async def test_memory_endpoint_has_no_mutation_actions(mobile, method):
    async with _client(mobile.app) as client:
        response = await client.request(method, PREFIX + "/memory", params={"session_id": "#00001"}, json={"summary": "do not save"})
    assert response.status_code == 405


@pytest.mark.parametrize("original", ["../report.pdf", "folder/report.pdf", "folder\\report.pdf", "report\n.pdf", "report.txt", "文" * 100 + ".pdf"])
async def test_original_upload_name_cannot_change_type_or_escape_directory(mobile, original):
    async with _client(mobile.app) as client:
        response = await client.post(PREFIX + "/upload", data={"session_id": "#00001", "original_filename": original}, files={"file": ("temporary.pdf", b"offline mock", "application/pdf")})
    assert response.status_code == 422


async def test_original_upload_name_preserves_same_type_without_bypassing_unknown_type(mobile, monkeypatch):
    with patch("dotenv.load_dotenv"):
        underlying = importlib.import_module("docthinker.server.routers.ingest")
    names = []
    async def ingest(**kwargs):
        names.append(kwargs["files"][0].filename)
        return {"status": "processing"}
    monkeypatch.setattr(underlying, "ingest_files", ingest)
    async with _client(mobile.app) as client:
        response = await client.post(PREFIX + "/upload", data={"session_id": "#00001", "original_filename": "能源报告.PDF"}, files={"file": ("temporary.pdf", b"offline mock", "application/pdf")})
        rejected = await client.post(PREFIX + "/upload", data={"session_id": "#00001", "original_filename": "能源报告.pdf"}, files={"file": ("temporary", b"offline mock", "application/pdf")})
    assert response.status_code == 200 and response.json()["status"] == "processing"
    assert names == ["能源报告.PDF"]
    assert rejected.status_code == 415
