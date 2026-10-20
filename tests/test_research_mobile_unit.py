"""Private research/mobile and invitation contracts, with no real providers.

The real temporary research store and service are exercised through the mobile
HTTP routes. App startup is replaced with an offline lifespan; the session manager
and answer model are synthetic. Gateways use an in-process mock HTTP transport.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
import uuid

import httpx
import pytest

# This existing fixture binds startup settings/models to temporary paths and
# synthetic adapters; production lifespan can therefore be tested without I/O
# to any provider, embedding service or desktop data.
from tests.test_mobile_safe_startup_unit import safe_mobile

TOKEN = "offline-research-mobile-token-never-a-real-credential"
PREFIX = "/api/v1/mobile"
ORIGIN = "http://127.0.0.1:8120"
PASSWORD = "offline-test-password-123"
RESEARCH_ROUTES = [
    ("GET", "capabilities"), ("GET", "documents"), ("GET", "document"),
    ("GET", "source"), ("POST", "upload"), ("POST", "documents/delete"), ("POST", "documents/retry"),
    ("GET", "memory"), ("POST", "memory"), ("POST", "memory/delete"),
    ("GET", "export"),
]
SID, OTHER_SID = "#00001", "#00002"
QUESTION = "How does renewable supply reduce energy imports?"
SOURCE_TEXT = "Renewable supply reduces energy imports. PRIVATE_SOURCE_QUOTE_TOKEN_ONLY."


class OfflineManager:
    """Synthetic project/history IDs with a real owner-only journal directory."""
    def __init__(self, root):
        self.base_storage_path = root / "_system"
        self.base_storage_path.mkdir(mode=0o700)
        self.histories = {SID: [], OTHER_SID: []}
        self.get_session = Mock(side_effect=lambda sid: {"id": sid, "title": sid} if sid in self.histories else None)

    def list_sessions(self):
        return [{"id": sid, "title": sid} for sid in self.histories]

    def get_history(self, sid):
        return deepcopy(self.histories[sid])

    def get_files(self, sid):
        return []

    def add_message(self, sid, role, content, *, research_provenance_required=False, research_memory_required=False):
        identifier = str(uuid.uuid4())
        self.histories[sid].append({"id": identifier, "role": role, "content": content,
            **({'research_memory_required': True} if role == 'assistant' and research_memory_required else {}),
            **({'research_core': True, 'research_provenance_required': True}
               if role == 'assistant' and research_provenance_required else {})})
        return identifier


@asynccontextmanager
async def offline_lifespan(_app):
    yield


@pytest.fixture
def mobile(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("DOCTHINKER_MOBILE_ENABLED", "true")
    monkeypatch.setenv("DOCTHINKER_MOBILE_ACCESS_TOKEN", TOKEN)
    monkeypatch.setenv("DOCTHINKER_MOBILE_CORS_ORIGINS", "")
    monkeypatch.delenv("DOCTHINKER_INVITE_WORKER_ID", raising=False)
    monkeypatch.delenv("DOCTHINKER_RESEARCH_ENABLED", raising=False)
    with patch("dotenv.load_dotenv", return_value=False):
        entry = importlib.import_module("docthinker.server.mobile_app")
        api = importlib.import_module("docthinker.server.routers.mobile")
        query = importlib.import_module("docthinker.server.routers.query")
        from docthinker.server.research_extract import extract_chunks
        from docthinker.server.research_service import ResearchService
        from docthinker.server.state import AppState
    root = (tmp_path / "worker").resolve()
    root.mkdir(mode=0o700)
    manager = OfflineManager(root)

    async def text_extractor(content, filename):
        return {"chunks": extract_chunks(content, Path(filename).suffix.lower())}

    service = ResearchService(root, extractor=text_extractor)
    model = AsyncMock(return_value="Renewable supply reduced imports in this source [S1].")
    state = AppState(session_manager=manager, mobile_safe_mode=True,
                     mobile_document_rag=False, mobile_chat_func=model, research_service=service)
    for module in (api, query):
        monkeypatch.setattr(module, "state", state)
    api._active_queries.clear()
    api._starting_query_keys.clear()
    app = entry.create_mobile_app(lifespan=offline_lifespan)
    yield SimpleNamespace(app=app, api=api, query=query, entry=entry, state=state,
                          manager=manager, root=root, service=service, model=model)
    assert not service._runner  # Tests never start production/background workers.
    api._active_queries.clear()
    api._starting_query_keys.clear()


def mobile_client(mobile, *, token=TOKEN):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mobile.app, raise_app_exceptions=False),
        base_url="http://mobile.test",
        headers={"Authorization": "Bearer " + token} if token is not None else {},
    )


async def upload_ready(mobile, browser, *, sid=SID, text=SOURCE_TEXT, filename="energy-paper.txt"):
    response = await browser.post(PREFIX + "/research/upload",
                                  data={"session_id": sid}, files={"file": (filename, text.encode(), "text/plain")})
    assert response.status_code == 200, response.text
    result = response.json()
    await mobile.service.process(sid, result["document"]["id"])
    return result["document"]["id"]


async def test_research_capabilities_are_read_only_and_disabled_without_optin(mobile, monkeypatch):
    monkeypatch.setattr(mobile.state, "research_service", None)
    async with mobile.app.router.lifespan_context(mobile.app):
        async with mobile_client(mobile) as browser:
            response = await browser.get(PREFIX + "/research/capabilities")
            assert response.status_code == 200
            caps = response.json()
            from docthinker.server.research_source_contract import MAX_ANSWER_SOURCES
            assert caps["max_source_chunks"] == MAX_ANSWER_SOURCES == 10
            assert caps["enabled"] is False and caps["memory_write_enabled"] is False
            assert caps["automatic_learning"] is False and caps["ocr"] is False
            assert caps["autonomous_tasks"] is False and caps["semantic_entailment_verified"] is False
            assert (await browser.get(PREFIX + "/research/documents", params={"session_id": SID})).status_code == 404
    assert not mobile.service.store.directory.exists()
    mobile.model.assert_not_awaited()


async def test_actual_safe_lifespan_does_not_initialize_research_without_optin(safe_mobile, monkeypatch):
    monkeypatch.setenv("DOCTHINKER_RESEARCH_ENABLED", "false")
    async with safe_mobile.entry._safe_mobile_lifespan(None):
        assert safe_mobile.state.research_service is None
        assert (await safe_mobile.mobile.research_capabilities())["enabled"] is False
        assert not (safe_mobile.root / "_system" / "research").exists()
    safe_mobile.chat.assert_not_awaited()
    safe_mobile.warmup.assert_not_awaited()
    safe_mobile.initialize.assert_not_awaited()


async def test_actual_safe_lifespan_starts_opted_in_offline_research_without_model_probe(safe_mobile, monkeypatch):
    from docthinker.server.research_service import ResearchService
    monkeypatch.setenv("DOCTHINKER_RESEARCH_ENABLED", "true")
    async with safe_mobile.entry._safe_mobile_lifespan(None):
        service = safe_mobile.state.research_service
        assert isinstance(service, ResearchService) and service._runner is not None
        assert service.context_assistance_enabled is False
        runner = service._runner
        capability = await safe_mobile.mobile.research_capabilities()
        assert capability["enabled"] is True and capability["memory_write_enabled"] is True
        assert capability["automatic_learning"] is False
        assert not service.store.path.exists()  # A readiness read creates no DB.
    assert runner.done() and safe_mobile.state.research_service is None
    safe_mobile.chat.assert_not_awaited()
    safe_mobile.warmup.assert_not_awaited()
    safe_mobile.initialize.assert_not_awaited()


@pytest.mark.parametrize("value,enabled", [("true", True), ("1", True), ("false", False), ("0", False),
                                          (" TRUE ", True), (" False ", False)])
async def test_startup_research_assistance_requires_explicit_operator_flag(safe_mobile, monkeypatch, value, enabled):
    monkeypatch.setenv("DOCTHINKER_RESEARCH_ENABLED", "true")
    monkeypatch.setenv("DOCTHINKER_RESEARCH_CONTEXT_ASSIST", value)
    async with safe_mobile.entry._safe_mobile_lifespan(None):
        assert safe_mobile.state.research_service.context_assistance_enabled is enabled
    safe_mobile.chat.assert_not_awaited()
    safe_mobile.warmup.assert_not_awaited()


@pytest.mark.parametrize("value", ["", "off", "on", "yes", "2", "private-secret-invalid-flag"])
async def test_bad_assistance_flag_is_rejected_before_settings_stores_or_models(safe_mobile, monkeypatch, value):
    monkeypatch.setenv("DOCTHINKER_RESEARCH_ENABLED", "true")
    monkeypatch.setenv("DOCTHINKER_RESEARCH_CONTEXT_ASSIST", value)
    with pytest.raises(ValueError, match="DOCTHINKER_RESEARCH_CONTEXT_ASSIST") as error:
        async with safe_mobile.entry._safe_mobile_lifespan(None):
            pytest.fail("Invalid experiment opt-in accepted")
    assert "private-secret" not in str(error.value)
    safe_mobile.server.load_settings.assert_not_called()
    safe_mobile.server._get_llm_model_func.assert_not_awaited()
    safe_mobile.chat.assert_not_awaited()
    assert not safe_mobile.root.exists()


@pytest.mark.parametrize("method,route", RESEARCH_ROUTES)
async def test_research_routes_authenticate_before_store_or_body(mobile, method, route):
    consumed = []

    async def body():
        consumed.append(True)
        yield b"not valid multipart or JSON"

    async with mobile_client(mobile, token=None) as browser:
        options = {"params": {"session_id": SID}}
        if method == "POST":
            options = {"content": body(), "headers": {"Content-Type": "multipart/form-data"}}
        response = await browser.request(method, PREFIX + "/research/" + route, **options)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert TOKEN not in response.text and not consumed
    assert not mobile.service.store.directory.exists()
    mobile.manager.get_session.assert_not_called()
    mobile.model.assert_not_awaited()


async def test_authenticated_upload_is_offline_then_explicitly_processed_with_page_truth(mobile):
    async with mobile_client(mobile) as browser:
        caps = (await browser.get(PREFIX + "/research/capabilities")).json()
        assert caps["enabled"] and caps["supported_upload_types"] == ["pdf", "txt", "md"]
        uploaded = await browser.post(PREFIX + "/research/upload", data={"session_id": SID},
                                      files={"file": ("energy.txt", SOURCE_TEXT.encode(), "text/plain")})
        assert uploaded.status_code == 200 and uploaded.headers["cache-control"] == "no-store"
        document = uploaded.json()["document"]
        assert document["status"] == "queued" and document["chunk_count"] == 0
        assert not any(field in document for field in ("file_path", "content_hash", "upstream_token"))
        mobile.model.assert_not_awaited()
        await mobile.service.process(SID, document["id"])
        fetched = await browser.get(PREFIX + "/research/document", params={"session_id": SID, "document_id": document["id"]})
        assert fetched.json()["document"]["status"] == "ready"
        quote = fetched.json()["document"]["chunks"][0]
        assert quote["content"] == SOURCE_TEXT and quote["page"] is None
        assert quote["document_id"] == document["id"]
        assert fetched.json()["document"]["source_ids"] == [quote["id"]]
    mobile.model.assert_not_awaited()


async def test_same_document_is_deduped_only_inside_its_project(mobile):
    async with mobile_client(mobile) as browser:
        first = await upload_ready(mobile, browser)
        duplicate = await browser.post(PREFIX + "/research/upload", data={"session_id": SID},
                                       files={"file": ("again.txt", SOURCE_TEXT.encode(), "text/plain")})
        second = await upload_ready(mobile, browser, sid=OTHER_SID)
        assert duplicate.json()["deduplicated"] is True
        assert duplicate.json()["document"]["id"] == first
        assert second != first
        documents = []
        for sid in (SID, OTHER_SID):
            result = await browser.get(PREFIX + "/research/documents", params={"session_id": sid})
            documents.append(result.json()["documents"])
        assert len(documents[0]) == len(documents[1]) == 1
        assert documents[0][0]["id"] != documents[1][0]["id"]
    mobile.model.assert_not_awaited()


async def test_pdf_multipart_is_parsed_locally_and_keeps_actual_page_number(mobile):
    from tests.test_research_service_unit import pdf_with_text
    pytest.importorskip("pypdf")
    async with mobile_client(mobile) as browser:
        accepted = await browser.post(PREFIX + "/research/upload", data={"session_id": SID},
                                      files={"file": ("energy.pdf", pdf_with_text(), "application/pdf")})
        assert accepted.status_code == 200
        document_id = accepted.json()["document"]["id"]
        await mobile.service.process(SID, document_id)
        document = (await browser.get(PREFIX + "/research/document", params={"session_id": SID, "document_id": document_id})).json()["document"]
        assert document["status"] == "ready"
        assert document["chunks"][0]["page"] == 1
        assert "Solar investment increases renewable supply" in document["chunks"][0]["content"]
    mobile.model.assert_not_awaited()


async def test_document_source_and_memory_refs_do_not_cross_projects(mobile):
    async with mobile_client(mobile) as browser:
        document_id = await upload_ready(mobile, browser)
        source = mobile.service.store.source_chunks(SID)[0]
        assert (await browser.get(PREFIX + "/research/document", params={"session_id": OTHER_SID, "document_id": document_id})).status_code == 404
        assert (await browser.get(PREFIX + "/research/source", params={"session_id": OTHER_SID, "chunk_id": source["id"]})).status_code == 404
        bad_memory = await browser.post(PREFIX + "/research/memory", json={
            "session_id": OTHER_SID, "kind": "finding", "text": "Not my project source.", "source_ids": [source["id"]],
        })
        assert bad_memory.status_code == 422
        assert (await browser.get(PREFIX + "/research/memory", params={"session_id": OTHER_SID})).json()["items"] == []
        for action in ("delete", "retry"):
            denied = await browser.post(PREFIX + "/research/documents/" + action, json={
                "session_id": OTHER_SID, "document_id": document_id,
            })
            assert denied.status_code == 404
        assert mobile.service.store.get_document(SID, document_id)["status"] == "ready"
        assert mobile.service.store.source_chunks(SID, [source["id"]])[0]["content"] == source["content"]
        owned = (await browser.post(PREFIX + "/research/memory", json={
            "session_id": SID, "kind": "finding", "text": "A source in this project only.", "source_ids": [source["id"]],
        })).json()["memory"]
        denied = await browser.post(PREFIX + "/research/memory/delete", json={
            "session_id": OTHER_SID, "memory_id": owned["id"], "expected_revision": owned["revision"],
        })
        assert denied.status_code == 404
        assert mobile.service.store.get_memory(SID, owned["id"])["status"] == "active"
    mobile.model.assert_not_awaited()


@pytest.mark.parametrize("override", [
    {"user_id": "other-user"}, {"confirmed_by_user": False}, {"kind": "implicit_preference"},
    {"expected_revision": "1"}, {"text": "x" * 2001}, {"source_ids": [str(uuid.uuid4())] * 21},
])
async def test_memory_request_is_explicit_strict_and_bounded(mobile, override):
    payload = {"session_id": SID, "kind": "preference", "text": "Answer concisely."} | override
    async with mobile_client(mobile) as browser:
        response = await browser.post(PREFIX + "/research/memory", json=payload)
    assert response.status_code == 422
    assert mobile.service.store.list_memories(SID) == []
    mobile.model.assert_not_awaited()


async def test_memory_confirm_correct_revision_and_delete_without_model_calls(mobile):
    async with mobile_client(mobile) as browser:
        payload = {"session_id": SID, "kind": "preference", "text": "Prefer concise Chinese."}
        saved = await browser.post(PREFIX + "/research/memory", json=payload)
        assert saved.status_code == 200
        memory = saved.json()["memory"]
        assert memory["confirmed_by_user"] is True and memory["revision"] == 1 and memory["status"] == "active"
        assert (await browser.post(PREFIX + "/research/memory", json={**payload, "memory_id": memory["id"]})).status_code == 422
        correction = {**payload, "memory_id": memory["id"], "expected_revision": 1, "text": "Prefer concise English."}
        changed = await browser.post(PREFIX + "/research/memory", json=correction)
        assert changed.status_code == 200 and changed.json()["memory"]["revision"] == 2
        assert (await browser.post(PREFIX + "/research/memory", json=correction)).status_code == 409
        assert (await browser.post(PREFIX + "/research/memory", json={**correction, "session_id": OTHER_SID, "expected_revision": 2})).status_code == 404
        removed = await browser.post(PREFIX + "/research/memory/delete", json={
            "session_id": SID, "memory_id": memory["id"], "expected_revision": 2,
        })
        assert removed.status_code == 200
        assert (await browser.get(PREFIX + "/research/memory", params={"session_id": SID})).json()["items"] == []
        assert mobile.service.store.active_memory_context(SID) == []
        exported = (await browser.get(PREFIX + "/research/export", params={"session_id": SID})).json()["markdown"]
        assert "Prefer concise Chinese." not in exported and "Prefer concise English." not in exported
    mobile.model.assert_not_awaited()


async def test_document_delete_invalidates_confirmed_finding_and_hides_source(mobile):
    async with mobile_client(mobile) as browser:
        document_id = await upload_ready(mobile, browser)
        source_id = mobile.service.store.source_chunks(SID)[0]["id"]
        saved = await browser.post(PREFIX + "/research/memory", json={
            "session_id": SID, "kind": "finding", "text": "Supply reduces imports in this source.", "source_ids": [source_id],
        })
        removed = await browser.post(PREFIX + "/research/documents/delete", json={"session_id": SID, "document_id": document_id})
        assert removed.json() == {"removed": True, "invalidated_memory_count": 1}
        assert (await browser.get(PREFIX + "/research/source", params={"session_id": SID, "chunk_id": source_id})).status_code == 404
        visible = (await browser.get(PREFIX + "/research/memory", params={"session_id": SID})).json()["items"]
        assert visible[0]["id"] == saved.json()["memory"]["id"] and visible[0]["status"] == "invalidated"
        assert mobile.service.store.active_memory_context(SID) == []
        exported = (await browser.get(PREFIX + "/research/export", params={"session_id": SID})).json()["markdown"]
        assert "Supply reduces imports in this source." not in exported
    mobile.model.assert_not_awaited()


async def test_tracked_query_replay_and_status_never_resurrect_deleted_quotes(mobile):
    request_id = str(uuid.uuid4())
    payload = {"session_id": SID, "question": QUESTION, "request_id": request_id,
               "use_memory": False, "use_conversation_context": False,
               "use_llm_cache": False, "use_self_evolution": False}
    async with mobile_client(mobile) as browser:
        document_id = await upload_ready(mobile, browser)
        response = await browser.post(PREFIX + "/query", json=payload)
        assert response.status_code == 200, response.text
        completed = response.json()
        assert completed["request_status"] == "completed" and completed["research_core"] is True
        assert completed["sources"][0]["content"] == SOURCE_TEXT
        assert completed["citation_validation"]["entailment_verified"] is False
        assert completed["citation_validation"]["scope"] == "citation_syntax_only"
        assert "PRIVATE_SOURCE_QUOTE_TOKEN_ONLY" not in mobile.api._query_journal().path.read_bytes().decode("latin-1")
        assert (await browser.post(PREFIX + "/query", json=payload)).json()["answer_id"] == completed["answer_id"]
        await browser.post(PREFIX + "/research/documents/delete", json={"session_id": SID, "document_id": document_id})
        status = await browser.get(PREFIX + "/query/status", params={"session_id": SID, "request_id": request_id})
        replay = await browser.post(PREFIX + "/query", json=payload)
        history = await browser.get(PREFIX + "/history", params={"session_id": SID})
        restored = history.json()["history"][-1]
        for result in (status.json(), replay.json(), restored):
            assert result["sources"] == [] and result["unavailable_source_count"] == 1
            assert result["citation_validation"]["valid"] is False
            assert "PRIVATE_SOURCE_QUOTE_TOKEN_ONLY" not in json.dumps(result)
        other = await browser.get(PREFIX + "/query/status", params={"session_id": OTHER_SID, "request_id": request_id})
        assert other.json()["request_status"] == "not_found"
    assert mobile.model.await_count == 1
    assert len(mobile.manager.get_history(SID)) == 2


async def test_model_failure_receipt_is_unknown_and_never_implicitly_retried(mobile):
    mobile.model.side_effect = RuntimeError("fake-private-provider-key /private/provider-path")
    payload = {"session_id": SID, "question": QUESTION, "request_id": str(uuid.uuid4())}
    async with mobile_client(mobile) as browser:
        await upload_ready(mobile, browser)
        first = await browser.post(PREFIX + "/query", json=payload)
        second = await browser.post(PREFIX + "/query", json=payload)
    assert first.status_code == second.status_code == 202
    assert first.json()["request_status"] == second.json()["request_status"] == "unknown"
    assert "fake-private-provider-key" not in first.text and "/private/provider-path" not in first.text
    assert mobile.model.await_count == 1


async def test_no_matching_original_evidence_does_not_spend_a_model_call(mobile):
    async with mobile_client(mobile) as browser:
        await upload_ready(mobile, browser)
        await browser.post(PREFIX + "/research/memory", json={"session_id": SID, "kind": "preference", "text": "Be concise."})
        answer = await browser.post(PREFIX + "/query", json={"session_id": SID, "question": "论文对 protein folding 有什么证据？", "use_memory": True})
        assert answer.status_code == 200 and answer.json()["answer_mode"] == "insufficient_evidence"
        assert answer.json()["sources"] == []
    mobile.model.assert_not_awaited()


@pytest.fixture
async def gateway_factory(monkeypatch, tmp_path):
    from docthinker.server.invite_app import create_invite_app
    from docthinker.server.invite_store import InviteStore
    store = InviteStore.initialize((tmp_path / "registry").resolve())
    auth = {}
    for name, port, uploads in (("alice", 8101, True), ("bobby", 8102, False)):
        invite = store.add_user(name, port, allow_uploads=uploads)
        auth[name] = store.register(invite["invite_code"], name, PASSWORD)
    calls, apps = [], []

    async def upstream(request):
        calls.append(request)
        if request.url.path.endswith("/research/capabilities"):
            return httpx.Response(200, json={"enabled": True, "memory_write_enabled": True, "automatic_learning": False})
        return httpx.Response(200, json={"ok": True, "worker_port": request.url.port})

    def create(enabled):
        monkeypatch.setenv("DOCTHINKER_RESEARCH_ENABLED", "true" if enabled else "false")
        monkeypatch.setenv("DOCTHINKER_INVITE_UPLOADS_ENABLED", "false")
        app = create_invite_app(store=store, origin=ORIGIN, transport=httpx.MockTransport(upstream))
        apps.append(app)
        return app

    yield SimpleNamespace(create=create, store=store, auth=auth, calls=calls)
    for app in apps:
        await app.state.invite_upstream_client.aclose()
        await app.state.invite_status_upstream_client.aclose()


def gateway_client(app, identity):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                             base_url=ORIGIN,
                             headers={"Origin": ORIGIN, "Authorization": "Bearer " + identity["access_token"]})


async def test_disabled_gateway_research_routes_never_reach_worker_or_break_me(gateway_factory):
    fixture = gateway_factory
    app = fixture.create(False)
    async with gateway_client(app, fixture.auth["alice"]) as browser:
        me = await browser.get("/api/v1/invite/me")
        assert me.status_code == 200
        for method, route in RESEARCH_ROUTES:
            if route == "capabilities":
                continue
            kwargs = {"params": {"session_id": SID}} if method == "GET" else {"json": {"session_id": SID}}
            response = await browser.request(method, PREFIX + "/research/" + route, **kwargs)
            assert response.status_code == 403, (route, response.text)
    assert not fixture.calls


async def test_gateway_capabilities_mask_disabled_profile_and_user_upload_permissions(gateway_factory):
    fixture = gateway_factory
    for enabled in (False, True):
        app = fixture.create(enabled)
        for name, uploads in (("alice", True), ("bobby", False)):
            async with gateway_client(app, fixture.auth[name]) as browser:
                response = await browser.get(PREFIX + "/research/capabilities")
                assert response.status_code == 200
                caps = response.json()
                assert caps["enabled"] is enabled
                assert caps["uploads_allowed"] is (enabled and uploads)
                assert caps["memory_write_enabled"] is enabled


async def test_gateway_document_reparse_requires_upload_permission_before_forwarding(gateway_factory):
    fixture = gateway_factory
    app = fixture.create(True)
    async with gateway_client(app, fixture.auth["bobby"]) as browser:
        response = await browser.post(PREFIX + "/research/documents/retry", json={
            "session_id": SID, "document_id": str(uuid.uuid4()),
        })
    assert response.status_code == 403
    assert fixture.calls == []
    with fixture.store._connect() as db:
        assert db.execute("SELECT count(*) FROM usage").fetchone()[0] == 0


async def test_gateway_document_reparse_shares_upload_count_without_recharging_stored_bytes(gateway_factory):
    from docthinker.server.invite_store import UPLOAD_DAILY_BYTES, UPLOAD_DAILY_LIMIT

    fixture = gateway_factory
    app = fixture.create(True)
    user_id = fixture.auth["alice"]["user"]["id"]
    # Already-ingested bytes do not get counted a second time when the user
    # explicitly reparses. Only the remaining extraction-count slot is spent.
    for index in range(UPLOAD_DAILY_LIMIT - 1):
        fixture.store.reserve_action(user_id, "upload", body_bytes=UPLOAD_DAILY_BYTES if index == 0 else 0)
    body = {"session_id": SID, "document_id": str(uuid.uuid4())}
    async with gateway_client(app, fixture.auth["alice"]) as browser:
        accepted = await browser.post(PREFIX + "/research/documents/retry", json=body)
        denied = await browser.post(PREFIX + "/research/documents/retry", json=body)
    assert accepted.status_code == 200 and denied.status_code == 429
    assert len(fixture.calls) == 1 and fixture.calls[0].url.port == 8101
    with fixture.store._connect() as db:
        row = db.execute("SELECT kind,count,bytes FROM usage WHERE user_id=?", (user_id,)).fetchone()
        assert tuple(row) == ("upload", UPLOAD_DAILY_LIMIT, UPLOAD_DAILY_BYTES)


async def test_enabled_gateway_scopes_worker_and_replaces_auth_without_model_quota(gateway_factory):
    fixture = gateway_factory
    app = fixture.create(True)
    for name, port in (("alice", 8101), ("bobby", 8102)):
        async with gateway_client(app, fixture.auth[name]) as browser:
            result = await browser.get(PREFIX + "/research/memory", params={"session_id": SID},
                                       headers={"Cookie": "private=cookie", "X-Forwarded-Host": "attacker.invalid"})
            assert result.json()["worker_port"] == port
        forwarded = fixture.calls[-1]
        worker = next(row for row in fixture.store.list_worker_bindings() if row["username"] == name)
        assert forwarded.headers["authorization"] == "Bearer " + worker["upstream_token"]
        assert fixture.auth[name]["access_token"] not in str(forwarded.headers)
        assert "cookie" not in forwarded.headers and "x-forwarded-host" not in forwarded.headers
    with fixture.store._connect() as db:
        assert db.execute("SELECT count(*) FROM usage").fetchone()[0] == 0


async def test_research_upload_has_explicit_permission_and_a_separate_daily_quota(gateway_factory):
    fixture = gateway_factory
    app = fixture.create(True)
    alice = fixture.auth["alice"]
    user_id = alice["user"]["id"]
    for _ in range(50):
        fixture.store.reserve_action(user_id, "query")
    async with gateway_client(app, fixture.auth["bobby"]) as browser:
        denied = await browser.post(PREFIX + "/research/upload", data={"session_id": SID},
                                    files={"file": ("paper.txt", b"offline source", "text/plain")})
        assert denied.status_code == 403
    assert not fixture.calls
    async with gateway_client(app, alice) as browser:
        for _ in range(5):
            uploaded = await browser.post(PREFIX + "/research/upload", data={"session_id": SID},
                                          files={"file": ("paper.txt", b"offline source", "text/plain")})
            assert uploaded.status_code == 200
        blocked = await browser.post(PREFIX + "/research/upload", data={"session_id": SID},
                                     files={"file": ("paper.txt", b"offline source", "text/plain")})
        assert blocked.status_code == 429
    assert len(fixture.calls) == 5 and all(call.url.port == 8101 for call in fixture.calls)
    with fixture.store._connect() as db:
        counts = {row["kind"]: row["count"] for row in db.execute("SELECT kind,count FROM usage WHERE user_id=?", (user_id,))}
        assert counts == {"query": 50, "upload": 5}


async def test_research_gateway_json_cannot_override_identity_or_enable_implicit_learning(gateway_factory):
    fixture = gateway_factory
    app = fixture.create(True)
    async with gateway_client(app, fixture.auth["alice"]) as browser:
        for extra in ({"user_id": "bobby"}, {"upstream_url": "http://127.0.0.1:8102"}, {"confirmed_by_user": False}):
            response = await browser.post(PREFIX + "/research/memory", json={
                "session_id": SID, "kind": "preference", "text": "Be concise.", **extra,
            })
            assert response.status_code == 422
    assert not fixture.calls
