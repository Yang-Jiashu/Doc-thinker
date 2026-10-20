"""Offline HTTP contracts for task-routed method knowledge, not quality evals.

The imported fixtures use temporary real stores, a synthetic writer and mock
upstream transport. No provider, user's project data or network is required.
"""

from __future__ import annotations

import hashlib
import json
from unittest.mock import AsyncMock
import uuid

import httpx
import pytest

from tests.test_research_mobile_unit import (
    ORIGIN, PREFIX, SID, gateway_client, gateway_factory, mobile, mobile_client,
)


WORK_QUESTION = "跨部门协作卡住，项目延期，如何安排负责人、优先级并做进度汇报？"


def _no_optional_models(mobile):
    mobile.service._semantic = None
    mobile.service._evidence_reranker = None
    mobile.service._query_translator = None
    mobile.service.context_assistance_enabled = False


async def test_knowledge_catalog_is_authenticated_read_only_metadata(mobile, monkeypatch):
    from docthinker.server import knowledge_skills

    def forbidden_pack_read(_skill_id):
        pytest.fail("Catalog metadata must not load knowledge resources")

    monkeypatch.setattr(knowledge_skills, "_read_pack", forbidden_pack_read)
    async with mobile_client(mobile) as browser:
        caps = await browser.get(PREFIX + "/research/capabilities")
        response = await browser.get(PREFIX + "/research/knowledge-skills")
    assert caps.status_code == response.status_code == 200
    assert caps.json()["knowledge_skills"] is True
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {"items": knowledge_skills.knowledge_skill_catalog()}
    assert response.json()["items"][0]["version"] == "2"
    assert "entries" not in response.json()["items"][0]
    assert not mobile.service.store.directory.exists()
    mobile.manager.get_session.assert_not_called()
    mobile.model.assert_not_awaited()


async def test_knowledge_catalog_without_auth_does_not_touch_store(mobile):
    async with mobile_client(mobile, token=None) as browser:
        response = await browser.get(PREFIX + "/research/knowledge-skills")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert not mobile.service.store.directory.exists()
    mobile.manager.get_session.assert_not_called()
    mobile.model.assert_not_awaited()


async def test_knowledge_catalog_follows_research_feature_gate(mobile, monkeypatch):
    monkeypatch.setattr(mobile.state, "research_service", None)
    async with mobile_client(mobile) as browser:
        caps = await browser.get(PREFIX + "/research/capabilities")
        catalog = await browser.get(PREFIX + "/research/knowledge-skills")
    assert caps.status_code == 200
    assert caps.json()["enabled"] is False
    assert caps.json()["knowledge_skills"] is False
    assert catalog.status_code == 404
    assert not mobile.service.store.directory.exists()
    mobile.model.assert_not_awaited()


@pytest.mark.parametrize("mode", [None, "auto", "off"])
@pytest.mark.parametrize("tracked", [False, True])
async def test_mobile_mode_is_forwarded_to_query_request(mobile, monkeypatch, mode, tracked):
    from docthinker.server.schemas import QueryRequest

    ask = AsyncMock(return_value={"answer": "offline answer", "sources": []})
    monkeypatch.setattr(mobile.query, "query", ask)
    payload = {"session_id": SID, "question": WORK_QUESTION}
    if mode is not None:
        payload["knowledge_skill_mode"] = mode
    if tracked:
        payload["request_id"] = str(uuid.uuid4())
    async with mobile_client(mobile) as browser:
        response = await browser.post(PREFIX + "/query", json=payload)
    assert response.status_code == 200, response.text
    ask.assert_awaited_once()
    request = ask.await_args.args[0]
    assert isinstance(request, QueryRequest)
    assert request.knowledge_skill_mode == (mode or "auto")
    assert not hasattr(request, "request_id")
    mobile.model.assert_not_awaited()


async def test_automatic_cards_survive_status_replay_history_without_becoming_s_evidence(mobile):
    _no_optional_models(mobile)
    mobile.model.return_value = "建议先明确负责人、交付标准和期限；这是方法建议，不是公司事实判断。[K1]"
    request_id = str(uuid.uuid4())
    payload = {"session_id": SID, "question": WORK_QUESTION, "request_id": request_id,
               "use_memory": False, "use_conversation_context": False,
               "use_llm_cache": False, "use_self_evolution": False}
    async with mobile_client(mobile) as browser:
        response = await browser.post(PREFIX + "/query", json=payload)
        assert response.status_code == 200, response.text
        first = response.json()
        assert first["request_status"] == "completed"
        receipt = first["knowledge_skill"]
        assert receipt["status"] == "ready" and receipt["version"] == "2"
        assert receipt["authorship"] == "project_original_method_design"
        assert receipt["validation_status"] == "design_unvalidated"
        assert 1 <= len(receipt["sources"]) <= 3
        assert [s["label"] for s in receipt["sources"]] == [
            f"K{index + 1}" for index in range(len(receipt["sources"]))]
        assert all(s["knowledge_scope"] == "method_reference_not_user_evidence"
                   for s in receipt["sources"])
        assert first["sources"] == []
        assert receipt["citation_validation"]["unknown_labels"] == []
        replay = await browser.post(PREFIX + "/query", json={**payload, "knowledge_skill_mode": "auto"})
        status = await browser.get(PREFIX + "/query/status", params={"session_id": SID, "request_id": request_id})
        history = await browser.get(PREFIX + "/history", params={"session_id": SID})
        for item in (replay.json(), status.json(), history.json()["history"][-1]):
            assert item["knowledge_skill"] == receipt
            assert item["sources"] == []
    assert mobile.model.await_count == 1
    assert "project_original_method_design" in mobile.model.await_args.args[0]
    assert "decision_rule" in mobile.model.await_args.args[0]
    assert mobile.service.store.active_memory_context(SID) == []


async def test_off_omits_cards_and_skill_prompt_without_disabling_normal_chat(mobile):
    _no_optional_models(mobile)
    mobile.model.return_value = "请先说明交付目标与当前阻塞。"
    async with mobile_client(mobile) as browser:
        response = await browser.post(PREFIX + "/query", json={
            "session_id": SID, "question": WORK_QUESTION, "knowledge_skill_mode": "off",
            "use_memory": False, "use_conversation_context": False,
        })
        assert response.status_code == 200, response.text
        assert "knowledge_skill" not in response.json()
        history = await browser.get(PREFIX + "/history", params={"session_id": SID})
        assert "knowledge_skill" not in history.json()["history"][-1]
    mobile.model.assert_awaited_once()
    assert "knowledge_skill_reference" not in mobile.model.await_args.args[0]
    assert "decision_rule" not in mobile.model.await_args.args[0]


def test_default_auto_retains_legacy_fingerprint_but_off_changes_identity():
    from docthinker.server.request_journal import CONTROL_DEFAULTS, request_fingerprint

    original = {"session_id": SID, "question": WORK_QUESTION}
    old_raw = json.dumps({**original, **CONTROL_DEFAULTS}, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    legacy = hashlib.sha256(old_raw.encode("utf-8")).hexdigest()
    assert request_fingerprint(original) == legacy
    assert request_fingerprint({**original, "knowledge_skill_mode": "auto"}) == legacy
    assert request_fingerprint({**original, "knowledge_skill_mode": "off"}) != legacy


async def test_reusing_completed_request_with_changed_skill_mode_is_conflict(mobile):
    _no_optional_models(mobile)
    mobile.model.return_value = "先核实交付依赖，再安排分工。[K1]"
    payload = {"session_id": SID, "question": WORK_QUESTION, "request_id": str(uuid.uuid4()),
               "use_memory": False, "use_conversation_context": False}
    async with mobile_client(mobile) as browser:
        first = await browser.post(PREFIX + "/query", json=payload)
        assert first.status_code == 200, first.text
        changed = await browser.post(PREFIX + "/query", json={**payload, "knowledge_skill_mode": "off"})
    assert changed.status_code == 409
    assert mobile.model.await_count == 1
    assert len(mobile.manager.get_history(SID)) == 2


@pytest.mark.parametrize("extra", [
    {"knowledge_skill_mode": "on"}, {"knowledge_skill_mode": True},
    {"knowledge_skill_mode": None}, {"knowledge_skill_mode": []},
    {"knowledge_skill_id": "mao-workplace"},
])
async def test_mobile_rejects_invalid_mode_and_manual_skill_id_before_model(mobile, extra):
    async with mobile_client(mobile) as browser:
        response = await browser.post(PREFIX + "/query", json={
            "session_id": SID, "question": WORK_QUESTION, **extra,
        })
    assert response.status_code == 422, response.text
    mobile.model.assert_not_awaited()
    assert not mobile.service.store.directory.exists()


async def test_invite_catalog_requires_auth_feature_flag_and_uses_authenticated_worker(gateway_factory):
    fixture = gateway_factory
    enabled = fixture.create(True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=enabled),
                                 base_url=ORIGIN, headers={"Origin": ORIGIN}) as browser:
        denied = await browser.get(PREFIX + "/research/knowledge-skills")
    assert denied.status_code == 401 and not fixture.calls
    disabled = fixture.create(False)
    async with gateway_client(disabled, fixture.auth["alice"]) as browser:
        denied = await browser.get(PREFIX + "/research/knowledge-skills")
        caps = await browser.get(PREFIX + "/research/capabilities")
    assert denied.status_code == 403
    assert caps.json()["knowledge_skills"] is False
    fixture.calls.clear()
    async with gateway_client(enabled, fixture.auth["alice"]) as browser:
        response = await browser.get(PREFIX + "/research/knowledge-skills")
    assert response.status_code == 200
    assert len(fixture.calls) == 1 and fixture.calls[0].url.port == 8101
    assert fixture.auth["alice"]["access_token"] not in str(fixture.calls[0].headers)
    with fixture.store._connect() as db:
        assert db.execute("SELECT count(*) FROM usage").fetchone()[0] == 0


@pytest.mark.parametrize("mode", ["auto", "off"])
async def test_invite_forwards_valid_skill_mode(gateway_factory, mode):
    fixture = gateway_factory
    app = fixture.create(True)
    payload = {"session_id": SID, "question": WORK_QUESTION, "knowledge_skill_mode": mode,
               "request_id": str(uuid.uuid4())}
    async with gateway_client(app, fixture.auth["alice"]) as browser:
        response = await browser.post(PREFIX + "/query", json=payload)
    assert response.status_code == 200, response.text
    assert len(fixture.calls) == 1
    assert json.loads(fixture.calls[0].content) == payload


@pytest.mark.parametrize("extra,tracked", [
    ({"knowledge_skill_mode": "on"}, True),
    ({"knowledge_skill_mode": False}, True),
    ({"knowledge_skill_mode": None}, True),
    ({"knowledge_skill_id": "mao-workplace"}, True),
    ({"knowledge_skill_id": "mao-workplace"}, False),
])
async def test_invite_rejects_invalid_skill_fields_before_upstream(gateway_factory, extra, tracked):
    fixture = gateway_factory
    app = fixture.create(True)
    payload = {"session_id": SID, "question": WORK_QUESTION, **extra}
    if tracked:
        payload["request_id"] = str(uuid.uuid4())
    async with gateway_client(app, fixture.auth["alice"]) as browser:
        response = await browser.post(PREFIX + "/query", json=payload)
    assert response.status_code == 422, response.text
    assert not fixture.calls
    with fixture.store._connect() as db:
        assert db.execute("SELECT count(*) FROM usage").fetchone()[0] == 0
