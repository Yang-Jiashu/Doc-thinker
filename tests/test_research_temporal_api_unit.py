"""HTTP acceptance of temporal memory and bounded reasoning; synthetic data only.

The mobile app uses the real store/service, with a synthetic model and offline
lifespan. Gateway tests check its actual authorization/forwarding boundary using
an in-process transport; they do not claim deployed network acceptance.
"""
from __future__ import annotations

import asyncio
from copy import copy
import json
import uuid

import httpx
import pytest

from tests.test_research_mobile_unit import (
    mobile, mobile_client, upload_ready, gateway_factory, gateway_client,
    PREFIX, SID, OTHER_SID, OfflineManager, ORIGIN, PASSWORD, TOKEN,
)


async def test_temporal_create_correct_archive_history_and_restore_over_http(mobile):
    now = [100.0]
    mobile.service.store.clock = lambda: now[0]
    async with mobile_client(mobile) as client:
        payload = {"session_id": SID, "kind": "decision", "text": "Use wind generation", "valid_from": 10}
        response = await client.post(PREFIX + "/research/memory", json=payload)
        assert response.status_code == 200, response.text
        original = response.json()["memory"]
        assert original["recorded_at"] == 100 and original["valid_from"] == 10
        now[0] = 200
        correction = {**payload, "memory_id": original["id"], "expected_revision": 1,
                      "text": "Use solar generation", "valid_from": 50}
        response = await client.post(PREFIX + "/research/memory", json=correction)
        assert response.status_code == 200 and response.json()["memory"]["revision"] == 2
        for at, expected in ((49, "Use wind generation"), (50, "Use solar generation")):
            page = (await client.get(PREFIX + "/research/memory", params={"session_id": SID, "as_of": at})).json()
            assert page["items"][0]["text"] == expected
        conflict = await client.post(PREFIX + "/research/memory", json=correction)
        assert conflict.status_code == 409
        foreign = await client.get(PREFIX + "/research/memory/history", params={"session_id": OTHER_SID, "memory_id": original["id"]})
        assert foreign.status_code == 404
        history = (await client.get(PREFIX + "/research/memory/history", params={"session_id": SID, "memory_id": original["id"], "limit": 1})).json()
        assert history["total"] == 2 and history["items"][0]["revision"] == 2 and history["next_cursor"]
        previous = (await client.get(PREFIX + "/research/memory/history", params={"session_id": SID, "memory_id": original["id"], "cursor": history["next_cursor"]})).json()
        assert previous["items"][0]["valid_until"] == 50 and previous["items"][0]["text"] == "Use wind generation"
        archive = {"session_id": SID, "memory_id": original["id"], "expected_revision": 2, "archived": True}
        archived = await client.post(PREFIX + "/research/memory/archive", json=archive)
        assert archived.status_code == 200 and archived.json()["memory"]["archived"]
        assert (await client.get(PREFIX + "/research/memory", params={"session_id": SID})).json()["items"] == []
        visible = (await client.get(PREFIX + "/research/memory", params={"session_id": SID, "include_archived": "true"})).json()["items"]
        assert visible[0]["text"] == "Use solar generation"
        assert (await client.post(PREFIX + "/research/memory/archive", json=archive)).status_code == 409
        restored = await client.post(PREFIX + "/research/memory/archive", json={**archive, "expected_revision": 3, "archived": False})
        assert restored.status_code == 200 and restored.json()["memory"]["revision"] == 4
    mobile.model.assert_not_awaited()


async def test_memory_as_of_reaches_answer_context_and_receipt_binds_time(mobile):
    now = [10]
    mobile.service.store.clock = lambda: now[0]
    memory = mobile.service.store.create_memory(SID, "decision", "We chose wind generation.")
    now[0] = 20
    mobile.service.store.update_memory(SID, memory["id"], expected_revision=1, text="We chose solar generation.")
    request_id = str(uuid.uuid4())
    payload = {"session_id": SID, "question": "What did we decide?", "request_id": request_id,
               "memory_as_of": 15, "reasoning_mode": "auto", "use_memory": True,
               "use_conversation_context": False, "use_llm_cache": False, "use_self_evolution": False}
    async with mobile_client(mobile) as client:
        response = await client.post(PREFIX + "/query", json=payload)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["memory_as_of"] == 15 and result["used_memory_count"] == 1
        prompt = mobile.model.await_args.args[0]
        assert "We chose wind generation." in prompt and "We chose solar generation." not in prompt
        assert (await client.post(PREFIX + "/query", json={**payload, "memory_as_of": 25})).status_code == 409
        assert (await client.post(PREFIX + "/query", json={**payload, "reasoning_mode": "deduction"})).status_code == 409
        replay = (await client.post(PREFIX + "/query", json=payload)).json()
        assert replay["answer_id"] == result["answer_id"] and replay["memory_as_of"] == 15
        history = (await client.get(PREFIX + "/history", params={"session_id": SID})).json()["history"]
        assert history[-1]["memory_as_of"] == 15
    assert mobile.model.await_count == 1


async def test_paginated_search_keeps_old_matches_without_cross_project_cursor_reuse(mobile):
    old = mobile.service.store.create_memory(SID, "finding", "Old nebula research archive")
    for i in range(107):
        mobile.service.store.create_memory(SID, "finding", "ordinary later note " + str(i))
    mobile.service.store.create_memory(OTHER_SID, "finding", "nebula in another project")
    async with mobile_client(mobile) as client:
        search = await client.get(PREFIX + "/research/memory", params={"session_id": SID, "query": "nebula", "limit": 20})
        assert search.status_code == 200 and search.json()["items"][0]["id"] == old["id"]
        first = (await client.get(PREFIX + "/research/memory", params={"session_id": SID, "limit": 20})).json()
        assert first["total"] == 108 and len(first["items"]) == 20 and first["next_cursor"]
        cursor = first["next_cursor"]
        second = (await client.get(PREFIX + "/research/memory", params={"session_id": SID, "cursor": cursor, "limit": 20})).json()
        assert not {row["id"] for row in first["items"]}.intersection(row["id"] for row in second["items"])
        assert (await client.get(PREFIX + "/research/memory", params={"session_id": OTHER_SID, "cursor": cursor})).status_code == 422
    mobile.model.assert_not_awaited()


async def test_omitted_source_ids_preserve_evidence_on_correction_and_revoke_historical_text(mobile):
    async with mobile_client(mobile) as client:
        doc_id = await upload_ready(mobile, client)
        source_id = mobile.service.store.source_chunks(SID)[0]["id"]
        payload = {"session_id": SID, "kind": "finding", "text": "Evidence-derived saved finding", "source_ids": [source_id], "valid_from": 10}
        saved = (await client.post(PREFIX + "/research/memory", json=payload)).json()["memory"]
        corrected = await client.post(PREFIX + "/research/memory", json={
            "session_id": SID, "memory_id": saved["id"], "expected_revision": 1,
            "kind": "finding", "text": "Corrected evidence-derived finding", "valid_from": 20,
        })
        assert corrected.status_code == 200 and corrected.json()["memory"]["source_ids"] == [source_id]
        await client.post(PREFIX + "/research/documents/delete", json={"session_id": SID, "document_id": doc_id})
        current = (await client.get(PREFIX + "/research/memory", params={"session_id": SID})).json()["items"]
        assert current[0]["status"] == "invalidated" and current[0]["text"] == ""
        historical = (await client.get(PREFIX + "/research/memory", params={"session_id": SID, "as_of": 15})).json()["items"]
        assert historical == []
        history = (await client.get(PREFIX + "/research/memory/history", params={"session_id": SID, "memory_id": saved["id"]})).json()["items"]
        assert all(item["text"] == "" for item in history)
    mobile.model.assert_not_awaited()


async def test_archived_memory_remains_in_explicit_project_export(mobile):
    memory = mobile.service.store.create_memory(SID, "decision", "Archived project milestone sentinel")
    mobile.service.store.archive_memory(SID, memory["id"], expected_revision=1)
    async with mobile_client(mobile) as client:
        result = await client.get(PREFIX + "/research/export", params={"session_id": SID})
        assert result.status_code == 200 and "Archived project milestone sentinel" in result.json()["markdown"]
    mobile.model.assert_not_awaited()


async def test_literal_mode_binds_real_quotes_without_model_generation(mobile):
    async with mobile_client(mobile) as client:
        await upload_ready(mobile, client)
        response = await client.post(PREFIX + "/query", json={
            "session_id": SID, "question": "renewable supply energy imports", "reasoning_mode": "evidence",
            "request_id": str(uuid.uuid4()), "use_memory": False,
        })
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["sources"] and result["claim_audit"]["summary"]["exact_quote_count"] >= 1
        assert result["claim_audit"]["entailment_verified"] is False
        assert result["reasoning_mode"] == "evidence"
    mobile.model.assert_not_awaited()


async def test_literal_source_revocation_removes_excerpt_from_history_and_same_id_reply(mobile):
    payload = {"session_id": SID, "question": "PRIVATE_LITERAL_SENTINEL", "reasoning_mode": "evidence",
               "request_id": str(uuid.uuid4()), "use_memory": False}
    async with mobile_client(mobile) as client:
        doc_id = await upload_ready(mobile, client, text="PRIVATE_LITERAL_SENTINEL original observation.")
        result = (await client.post(PREFIX + "/query", json=payload)).json()
        assert "PRIVATE_LITERAL_SENTINEL" in result['answer']
        await client.post(PREFIX + "/research/documents/delete", json={"session_id": SID, "document_id": doc_id})
        for route in ('/query', '/query/status'):
            response = (await client.post(PREFIX + route, json=payload) if route == '/query' else
                        await client.get(PREFIX + route, params={"session_id": SID, "request_id": payload['request_id']}))
            replay = response.json()
            assert response.status_code == 200 and replay['answer_id'] == result['answer_id']
            assert "PRIVATE_LITERAL_SENTINEL" not in replay['answer'] and not replay['sources']
            assert all(not claim['source_refs'] for claim in replay['claim_audit']['claims'])
        messages = mobile.service.hydrate_history(SID, mobile.state.session_manager.get_history(SID))
        stored = next(row for row in messages if row['id'] == result['answer_id'])
        assert "PRIVATE_LITERAL_SENTINEL" not in stored['content']
    mobile.model.assert_not_awaited()


async def test_normal_answer_uses_background_premises_and_revokes_replay_without_second_model(mobile):
    text = "Observation: grid ready. If grid ready then supply grows. If supply grows and imports decrease then security improves. Observation: imports decrease."
    request_id = str(uuid.uuid4())
    payload = {"session_id": SID, "question": "derive grid ready supply grows security improves",
               "reasoning_mode": "deduction", "request_id": request_id, "use_memory": False}
    async with mobile_client(mobile) as client:
        doc_id = await upload_ready(mobile, client, text=text)
        response = await client.post(PREFIX + "/query", json=payload)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["answer_mode"] == "evidence" and result["reasoning"]["status"] == "derived"
        assert result["reasoning"]["mode"] == "context_assist"
        assert result["answer"] == mobile.model.return_value
        assert 'supporting_material' in mobile.model.await_args.args[0]
        conclusions = result["reasoning"]["conclusions"]
        security = next(row for row in conclusions if row["subject"] == "security improves")
        assert security["depth"] == 2 and security["proof"] and security["source_ids"]
        assert security["epistemic_status"] == "conditional_on_source_premises"
        assert result["reasoning"]["trace"]["model_calls"] == 0
        assert mobile.service.store.list_memories(SID) == []
        await client.post(PREFIX + "/research/documents/delete", json={"session_id": SID, "document_id": doc_id})
        replay = (await client.post(PREFIX + "/query", json=payload)).json()
        assert replay["answer_id"] == result["answer_id"] and replay["sources"] == []
        assert replay["reasoning"]["status"] == "source_unavailable"
        assert replay["reasoning"]["conclusions"] == [] and replay["reasoning"]["claims"] == []
    mobile.model.assert_awaited_once()


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "253402300800", "true"])
async def test_invalid_temporal_query_is_rejected_without_model(mobile, value):
    async with mobile_client(mobile) as client:
        response = await client.get(PREFIX + "/research/memory", params={"session_id": SID, "as_of": value})
        assert response.status_code == 422
    mobile.model.assert_not_awaited()


@pytest.mark.parametrize("fragment", ['true', '"100"', '-1', 'NaN', 'Infinity', '253402300800'])
async def test_invalid_temporal_json_fails_with_validation_response(mobile, fragment):
    raw = '{"session_id":"#00001","kind":"decision","text":"bad time","valid_from":' + fragment + '}'
    async with mobile_client(mobile) as client:
        response = await client.post(PREFIX + "/research/memory", content=raw, headers={"Content-Type": "application/json"})
        assert response.status_code == 422, response.text
    assert mobile.service.store.list_memories(SID) == []
    mobile.model.assert_not_awaited()


@pytest.mark.parametrize("method,route", [("GET", "memory/history"), ("POST", "memory/archive")])
async def test_new_memory_routes_authenticate_before_body_or_storage(mobile, method, route):
    consumed = []
    async def body():
        consumed.append(True)
        yield b"not json"
    async with mobile_client(mobile, token=None) as client:
        response = await client.request(method, PREFIX + "/research/" + route, content=body())
    assert response.status_code == 401 and not consumed
    assert not mobile.service.store.directory.exists()


async def test_gateway_forwards_temporal_fields_to_the_bound_worker_only(gateway_factory):
    fixture = gateway_factory
    app = fixture.create(True)
    mid = str(uuid.uuid4())
    async with gateway_client(app, fixture.auth["alice"]) as client:
        history = await client.get(PREFIX + "/research/memory/history", params={"session_id": SID, "memory_id": mid, "limit": 10})
        assert history.status_code == 200 and history.json()["worker_port"] == 8101
        archived = await client.post(PREFIX + "/research/memory/archive", json={"session_id": SID, "memory_id": mid, "expected_revision": 1, "archived": True})
        assert archived.status_code == 200
        saved = await client.post(PREFIX + "/research/memory", json={"session_id": SID, "kind": "decision", "text": "Temporal note", "valid_from": 100, "valid_until": 200})
        assert saved.status_code == 200
        page = await client.get(PREFIX + "/research/memory", params={"session_id": SID, "as_of": 150, "query": "Temporal", "include_archived": "true", "limit": 20})
        assert page.status_code == 200
    assert len(fixture.calls) == 4 and all(call.url.port == 8101 for call in fixture.calls)
    assert json.loads(fixture.calls[1].content)["archived"] is True
    assert json.loads(fixture.calls[2].content)["valid_from"] == 100
    assert fixture.calls[3].url.params["as_of"] == "150"
    with fixture.store._connect() as db:
        assert db.execute("SELECT count(*) FROM usage WHERE kind='query'").fetchone()[0] == 0


async def test_gateway_rejects_new_route_identity_overrides_and_disabled_profile(gateway_factory):
    fixture = gateway_factory
    mid = str(uuid.uuid4())
    enabled = fixture.create(True)
    async with gateway_client(enabled, fixture.auth["alice"]) as client:
        denied = await client.post(PREFIX + "/research/memory/archive", json={"session_id": SID, "memory_id": mid, "expected_revision": 1, "archived": True, "user_id": "bobby"})
        assert denied.status_code == 422
        denied = await client.get(PREFIX + "/research/memory/history", params={"session_id": SID, "memory_id": mid, "user_id": "bobby"})
        assert denied.status_code == 400
    disabled = fixture.create(False)
    async with gateway_client(disabled, fixture.auth["alice"]) as client:
        assert (await client.get(PREFIX + "/research/memory/history", params={"session_id": SID, "memory_id": mid})).status_code == 403
        assert (await client.post(PREFIX + "/research/memory/archive", json={"session_id": SID, "memory_id": mid, "expected_revision": 1})).status_code == 403
    assert not fixture.calls


async def test_gateway_and_mobile_memory_chain_keep_two_worker_stores_separate(mobile, monkeypatch, tmp_path):
    """Run actual gateway and HTTP endpoints; model and network stay offline.

    A serialized in-process transport selects the same independent state that
    separate production worker processes own. It checks the bound worker token
    before adapting it to the fixture app's synthetic authentication token.
    """
    from docthinker.server.invite_app import create_invite_app
    from docthinker.server.invite_store import InviteStore
    from docthinker.server.research_service import ResearchService

    second_root = (tmp_path / "second-worker").resolve()
    second_root.mkdir(mode=0o700)
    other_state = copy(mobile.state)
    other_state.session_manager = OfflineManager(second_root)
    other_state.research_service = ResearchService(second_root, extractor=mobile.service.extractor)
    registry = InviteStore.initialize((tmp_path / "gateway-registry").resolve())
    auth = {}
    for name, port in (("alice", 8101), ("bobby", 8102)):
        invite = registry.add_user(name, port)
        auth[name] = registry.register(invite["invite_code"], name, PASSWORD)
    bindings = {int(row["upstream_url"].rsplit(":", 1)[1]): row for row in registry.list_worker_bindings()}
    states = {8101: mobile.state, 8102: other_state}
    lock = asyncio.Lock()

    async def upstream(request):
        assert request.headers["authorization"] == "Bearer " + bindings[request.url.port]["upstream_token"]
        headers = {key: value for key, value in request.headers.items() if key not in {"authorization", "host"}}
        headers["authorization"] = "Bearer " + TOKEN
        async with lock:
            with monkeypatch.context() as isolated:
                isolated.setattr(mobile.api, "state", states[request.url.port])
                isolated.setattr(mobile.query, "state", states[request.url.port])
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mobile.app, raise_app_exceptions=False), base_url="http://worker.test") as worker:
                    response = await worker.request(request.method, request.url.raw_path.decode(), headers=headers, content=request.content)
                    return httpx.Response(response.status_code, headers=response.headers, content=response.content)

    monkeypatch.setenv("DOCTHINKER_RESEARCH_ENABLED", "true")
    monkeypatch.setenv("DOCTHINKER_INVITE_UPLOADS_ENABLED", "false")
    app = create_invite_app(store=registry, origin=ORIGIN, transport=httpx.MockTransport(upstream))
    try:
        async with gateway_client(app, auth["alice"]) as client:
            saved = await client.post(PREFIX + "/research/memory", json={"session_id": SID, "kind": "decision", "text": "Alice private temporal note", "valid_from": 100})
            assert saved.status_code == 200, saved.text
            mid = saved.json()["memory"]["id"]
        async with gateway_client(app, auth["bobby"]) as client:
            assert (await client.get(PREFIX + "/research/memory", params={"session_id": SID, "as_of": 150})).json()["items"] == []
            assert (await client.get(PREFIX + "/research/memory/history", params={"session_id": SID, "memory_id": mid})).status_code == 404
            assert (await client.post(PREFIX + "/research/memory/archive", json={"session_id": SID, "memory_id": mid, "expected_revision": 1})).status_code == 404
        async with gateway_client(app, auth["alice"]) as client:
            found = (await client.get(PREFIX + "/research/memory", params={"session_id": SID, "as_of": 150})).json()["items"]
            assert found[0]["id"] == mid and found[0]["text"] == "Alice private temporal note"
    finally:
        await app.state.invite_upstream_client.aclose()
        await app.state.invite_status_upstream_client.aclose()
    assert other_state.research_service.store.page_memories(SID)["items"] == []
    assert b"Alice private temporal note" not in other_state.research_service.store.path.read_bytes()
    mobile.model.assert_not_awaited()
