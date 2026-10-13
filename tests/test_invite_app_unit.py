import asyncio
import json
import sqlite3

import httpx
import pytest

from docthinker.server.invite_app import create_invite_app, validated_origin
from docthinker.server.invite_store import InviteError, InviteStore

ORIGIN = "http://127.0.0.1:8120"
PASSWORD = "offline-test-password-123"


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    monkeypatch.delenv("DOCTHINKER_INVITE_UPLOADS_ENABLED", raising=False)
    store = InviteStore.initialize((tmp_path / "registry").resolve())
    invites, auth = {}, {}
    for name, port in [("alice", 8101), ("bobby", 8102)]:
        invites[name] = store.add_user(name, port)
        auth[name] = store.register(invites[name]["invite_code"], name, PASSWORD)
    calls = []
    async def upstream(request):
        calls.append(request)
        return httpx.Response(200, json={"worker_port": request.url.port, "data": "synthetic-only"})
    app = create_invite_app(store=store, origin=ORIGIN, transport=httpx.MockTransport(upstream))
    return app, store, auth, calls


def client(app, *, token=None, origin=ORIGIN):
    headers = {"Origin": origin} if origin is not None else {}
    if token:
        headers["Authorization"] = "Bearer " + token
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=ORIGIN, headers=headers)


async def test_public_status_and_protected_routes_are_no_store(gateway):
    app, _, _, calls = gateway
    async with client(app, origin=None) as browser:
        status = await browser.get("/api/v1/invite/status")
        assert status.json() == {"service": "docthinker-invite", "registration": "invitation-only"}
        assert status.headers["cache-control"] == "no-store"
        for path in ("/api/v1/invite/me", "/api/v1/mobile/sessions"):
            denied = await browser.get(path)
            assert denied.status_code == 401 and denied.headers["cache-control"] == "no-store"
    assert not calls


async def test_registration_login_me_and_logout(gateway):
    app, store, _, _ = gateway
    invitation = store.add_user("carol", 8103)
    async with client(app) as browser:
        register = await browser.post("/api/v1/invite/register", json={**{k: invitation[k] for k in ("invite_code", "username")}, "password": PASSWORD})
        assert register.status_code == 200
        assert set(register.json()) == {"access_token", "expires_in", "user"}
        login = await browser.post("/api/v1/invite/login", json={"username": "carol", "password": PASSWORD})
        assert login.status_code == 200
        token = login.json()["access_token"]
        headers = {"Authorization": "Bearer " + token}
        me = await browser.get("/api/v1/invite/me", headers=headers)
        assert me.json() == {"user": login.json()["user"]}
        assert (await browser.post("/api/v1/invite/logout", headers=headers)).json() == {"ok": True}
        assert (await browser.get("/api/v1/invite/me", headers=headers)).status_code == 401


async def test_same_session_identifier_routes_to_different_workers_and_replaces_authorization(gateway):
    app, store, auth, calls = gateway
    for name, port in [("alice", 8101), ("bobby", 8102)]:
        async with client(app, token=auth[name]["access_token"]) as browser:
            response = await browser.get("/api/v1/mobile/history", params={"session_id": "#00001"}, headers={"Cookie": "never-forward=secret", "X-Forwarded-Host": "attacker.invalid"})
        assert response.json()["worker_port"] == port
        worker = next(row for row in store.list_worker_bindings() if row["username"] == name)
        request = calls[-1]
        assert request.headers["authorization"] == "Bearer " + worker["upstream_token"]
        assert auth[name]["access_token"] not in str(request.headers)
        assert "cookie" not in request.headers and "x-forwarded-host" not in request.headers
        assert request.url.params["session_id"] == "#00001"


@pytest.mark.parametrize("path", ["/api/v1/settings", "/api/v1/mobile/settings", "/api/v1/mobile/admin", "/docs", "/openapi.json", "/api/v1/mobile/query/stream", "/api/v1/invite/reissue-invite", "/api/v1/mobile/reissue-invite"])
async def test_admin_and_unknown_paths_never_reach_workers(gateway, path):
    app, _, auth, calls = gateway
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        assert (await browser.get(path)).status_code == 404
    assert not calls


@pytest.mark.parametrize("params", [{"access_token": "must-not-authenticate"}, {"user_id": "bobby"}, {"backend": "http://attacker.invalid"}, {"session_id": "#00001", "upstream_url": "http://127.0.0.1:8102"}])
async def test_query_identity_or_backend_overrides_are_rejected(gateway, params):
    app, _, auth, calls = gateway
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        assert (await browser.get("/api/v1/mobile/history", params=params)).status_code == 400
    assert not calls


async def test_json_identity_override_is_rejected_before_upstream(gateway):
    app, _, auth, calls = gateway
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        result = await browser.post("/api/v1/mobile/query", json={"session_id": "#00001", "question": "synthetic", "user_id": "bobby"})
    assert result.status_code == 422 and not calls


async def test_feedback_uses_authenticated_worker_without_consuming_query_credit(gateway):
    app, store, auth, calls = gateway
    # Exhaust AI queries; explicit human feedback still works independently.
    for _ in range(50):
        store.reserve_action(auth["alice"]["user"]["id"], "query")
    payload = {"session_id": "#00001", "answer_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "rating": "helpful", "note": "synthetic only"}
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        assert (await browser.post("/api/v1/mobile/feedback", json=payload)).status_code == 200
        assert (await browser.get("/api/v1/mobile/feedback", params={"session_id": "#00001"})).status_code == 200
    assert len(calls) == 2
    assert all(request.url.port == 8101 for request in calls)
    assert json.loads(calls[0].content) == payload
    assert calls[1].url.params["session_id"] == "#00001"
    with store._connect() as db:
        usages = {row["kind"]: row["count"] for row in db.execute("SELECT kind,count FROM usage WHERE user_id=?", (auth["alice"]["user"]["id"],))}
    assert usages == {"query": 50, "feedback": 1}


async def test_workbench_brief_writes_use_separate_budget_and_worker_identity(gateway):
    app, store, auth, calls = gateway
    uid = auth["alice"]["user"]["id"]
    for _ in range(50):
        store.reserve_action(uid, "query")
    payload = {"session_id": "#00001", "expected_revision": 0, "goal": "write", "output_format": "report", "constraints": "cite"}
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        assert (await browser.post("/api/v1/mobile/workbench/brief", json=payload)).status_code == 200
        assert (await browser.get("/api/v1/mobile/workbench/brief", params={"session_id": "#00001"})).status_code == 200
        invalid = await browser.post("/api/v1/mobile/workbench/brief", json={**payload, "user_id": "bobby"})
        assert invalid.status_code == 422
    assert len(calls) == 2 and all(request.url.port == 8101 for request in calls)
    assert json.loads(calls[0].content) == payload
    with store._connect() as db:
        usage = {row["kind"]: row["count"] for row in db.execute("SELECT kind,count FROM usage WHERE user_id=?", (uid,))}
    assert usage == {"query": 50, "workbench_write": 1}


async def test_workbench_artifact_reads_and_save_require_research_but_cleanup_does_not(gateway, monkeypatch):
    app, store, auth, calls = gateway
    payload = {"session_id": "#00001", "answer_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "title": "result"}
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        for path in ("artifacts", "artifact"):
            response = await browser.get("/api/v1/mobile/workbench/" + path, params={"session_id": "#00001"})
            assert response.status_code == 403
        assert (await browser.post("/api/v1/mobile/workbench/artifacts", json=payload)).status_code == 403
        assert (await browser.post("/api/v1/mobile/workbench/artifacts/delete", json={k: payload[k] for k in ("session_id", "answer_id")})).status_code == 200
    assert len(calls) == 1
    monkeypatch.setenv("DOCTHINKER_RESEARCH_ENABLED", "true")
    async def upstream(request):
        calls.append(request)
        return httpx.Response(200, json={"synthetic": True})
    enabled = create_invite_app(store=store, origin=ORIGIN, transport=httpx.MockTransport(upstream))
    async with client(enabled, token=auth["alice"]["access_token"]) as browser:
        assert (await browser.get("/api/v1/mobile/workbench/artifact", params={"session_id": "#00001", "answer_id": payload["answer_id"]})).status_code == 200
        assert (await browser.post("/api/v1/mobile/workbench/artifacts", json=payload)).status_code == 200
        assert (await browser.post("/api/v1/mobile/workbench/artifacts", json={**payload, "sources": []})).status_code == 422
    assert len(calls) == 3


async def test_workbench_reads_keep_gateway_single_active_boundary(gateway):
    _, store, auth, calls = gateway
    entered, release = asyncio.Event(), asyncio.Event()
    async def upstream(request):
        calls.append(request)
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"brief": {}})
    app = create_invite_app(store=store, origin=ORIGIN, transport=httpx.MockTransport(upstream))
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        first = asyncio.create_task(browser.get("/api/v1/mobile/workbench/brief", params={"session_id": "#00001"}))
        await entered.wait()
        try:
            second = await browser.get("/api/v1/mobile/workbench/brief", params={"session_id": "#00001"})
            assert second.status_code == 429
        finally:
            release.set()
            assert (await first).status_code == 200
    assert len(calls) == 1


async def test_workbench_capability_cannot_claim_artifacts_when_gateway_research_disabled(gateway):
    _, store, auth, _ = gateway
    async def upstream(_request):
        return httpx.Response(200, json={"workbench": {"brief": True, "artifacts": True, "max_artifacts_per_session": 100}})
    app = create_invite_app(store=store, origin=ORIGIN, transport=httpx.MockTransport(upstream))
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        result = await browser.get("/api/v1/mobile/capabilities")
        assert result.json()["workbench"]["brief"] is True
        assert result.json()["workbench"]["artifacts"] is False


async def test_query_idempotent_reservation_survives_quota_exhaustion_and_conflicts(gateway):
    app, store, auth, calls = gateway
    identity = auth["alice"]
    uid = identity["user"]["id"]
    payload = {"session_id": "#00001", "question": "synthetic", "request_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"}
    async with client(app, token=identity["access_token"]) as browser:
        assert (await browser.post("/api/v1/mobile/query", json=payload)).status_code == 200
        for _ in range(49):
            store.reserve_action(uid, "query")
        # Quota reservation remains once; the actual worker journal decides
        # whether this is completion replay/running/unknown, never this ledger.
        assert (await browser.post("/api/v1/mobile/query", json={**payload, "use_memory": True})).status_code == 200
        conflict = await browser.post("/api/v1/mobile/query", json={**payload, "question": "changed"})
        assert conflict.status_code == 409
        status = await browser.get("/api/v1/mobile/query/status", params={"session_id": "#00001", "request_id": payload["request_id"]})
        assert status.status_code == 200
    assert len(calls) == 3  # Two same-ID HTTP forwards and one read; not model calls.
    with store._connect() as db:
        assert db.execute("SELECT count FROM usage WHERE user_id=? AND kind='query'", (uid,)).fetchone()[0] == 50
        assert db.execute("SELECT count(*) FROM query_reservations WHERE user_id=?", (uid,)).fetchone()[0] == 1


async def test_insight_ab_switch_reaches_worker_and_cannot_replay_other_mode(gateway):
    app, store, auth, calls = gateway
    payload = {"session_id": "#00001", "question": "synthetic", "insight_mode": "off",
               "knowledge_skill_mode": "auto", "request_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"}
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        assert (await browser.post("/api/v1/mobile/query", json=payload)).status_code == 200
        conflict = await browser.post("/api/v1/mobile/query", json={**payload, "insight_mode": "auto"})
        assert conflict.status_code == 409
    assert len(calls) == 1
    assert json.loads(calls[0].content) == payload
    with store._connect() as db:
        assert db.execute("SELECT SUM(count) FROM usage WHERE kind='query'").fetchone()[0] == 1


async def test_status_and_same_id_reconnect_are_not_blocked_by_running_user(gateway):
    app, store, auth, calls = gateway
    entered, release = asyncio.Event(), asyncio.Event()
    first = [True]
    rid = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    async def upstream(request):
        calls.append(request)
        if request.method == "POST" and first[0]:
            first[0] = False
            entered.set()
            await release.wait()
            return httpx.Response(200, json={"answer": "synthetic", "sources": [], "request_id": rid, "request_status": "completed"})
        return httpx.Response(200 if request.method == "GET" else 202, json={"request_id": rid, "request_status": "running"})
    await app.state.invite_upstream_client.aclose()
    app = create_invite_app(store=store, origin=ORIGIN, transport=httpx.MockTransport(upstream))
    payload = {"session_id": "#00001", "question": "synthetic", "request_id": rid}
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        connection = asyncio.create_task(browser.post("/api/v1/mobile/query", json=payload))
        await asyncio.wait_for(entered.wait(), 2)
        try:
            status = await browser.get("/api/v1/mobile/query/status", params={"session_id": "#00001", "request_id": rid})
            replay = await browser.post("/api/v1/mobile/query", json=payload)
            assert status.status_code == 200 and status.json()["request_status"] == "running"
            assert replay.status_code == 202 and replay.json()["request_status"] == "running"
        finally:
            release.set()
            assert (await connection).status_code == 200
    with store._connect() as db:
        assert db.execute("SELECT SUM(count) FROM usage WHERE kind='query'").fetchone()[0] == 1


@pytest.mark.parametrize("overrides", [
    {"request_id": "bad-id"}, {"use_memory": "false"}, {"question": "q" * 8193}, {"question": "\ud800"},
    {"insight_mode": "invalid"},
])
async def test_invalid_reconnect_request_does_not_charge_or_reach_workers(gateway, overrides):
    app, store, auth, calls = gateway
    payload = {"session_id": "#00001", "question": "synthetic", "request_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"}
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        response = await browser.post("/api/v1/mobile/query", content=json.dumps({**payload, **overrides}), headers={"Content-Type": "application/json"})
    assert response.status_code in {413, 422}
    assert not calls
    with store._connect() as db:
        assert db.execute("SELECT count(*) FROM usage").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM query_reservations").fetchone()[0] == 0


async def test_unknown_worker_status_and_conflict_messages_never_expose_provider_body(gateway):
    app, store, auth, calls = gateway
    await app.state.invite_upstream_client.aclose()
    async def upstream(request):
        return httpx.Response(409, json={"detail": "synthetic-private-provider-key /private/path"})
    app = create_invite_app(store=store, origin=ORIGIN, transport=httpx.MockTransport(upstream))
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        response = await browser.post("/api/v1/mobile/query", json={"session_id": "#00001", "question": "synthetic", "request_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"})
    assert response.status_code == 409
    assert "synthetic-private-provider-key" not in response.text and "/private/path" not in response.text


async def test_four_waiting_tracked_posts_cannot_starve_the_status_pool(gateway):
    app, store, auth, calls = gateway
    for name, port in [("carol", 8103), ("david", 8104)]:
        invitation = store.add_user(name, port)
        auth[name] = store.register(invitation["invite_code"], name, PASSWORD)
    await app.state.invite_upstream_client.aclose()
    await app.state.invite_status_upstream_client.aclose()
    entered, release = asyncio.Event(), asyncio.Event()
    post_count = [0]
    async def upstream(request):
        if request.method == "POST":
            post_count[0] += 1
            if post_count[0] == 4:
                entered.set()
            await release.wait()
            return httpx.Response(202, json={"request_id": json.loads(request.content)["request_id"], "request_status": "running"})
        return httpx.Response(200, json={"request_id": request.url.params["request_id"], "request_status": "running"})
    app = create_invite_app(store=store, origin=ORIGIN, transport=httpx.MockTransport(upstream))
    assert app.state.invite_status_upstream_client is not app.state.invite_upstream_client
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        requests = [asyncio.create_task(browser.post("/api/v1/mobile/query", json={
            "session_id": "#00001", "question": "synthetic", "request_id": f"{index:08x}-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        }, headers={"Authorization": "Bearer " + auth[name]["access_token"]}))
                    for index, name in enumerate(("alice", "bobby", "carol", "david"))]
        await asyncio.wait_for(entered.wait(), 2)
        try:
            status = await asyncio.wait_for(browser.get("/api/v1/mobile/query/status", params={
                "session_id": "#00001", "request_id": "00000000-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            }), 1)
            assert status.status_code == 200 and status.json()["request_status"] == "running"
            assert all(not task.done() for task in requests)
        finally:
            release.set()
            responses = await asyncio.gather(*requests)
            assert all(response.status_code == 202 for response in responses)


async def test_feedback_quota_is_separate_and_persisted(gateway):
    app, store, auth, calls = gateway
    for _ in range(200):
        store.reserve_action(auth["alice"]["user"]["id"], "feedback")
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        response = await browser.post("/api/v1/mobile/feedback", json={"session_id": "#00001", "answer_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "rating": "helpful"})
    assert response.status_code == 429 and not calls
    store.reserve_action(auth["alice"]["user"]["id"], "query")


@pytest.mark.parametrize("payload", [
    {"session_id": "#00001", "answer_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "rating": "helpful", "user_id": "bobby"},
    {"session_id": "#00001", "answer_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "rating": "helpful", "backend": "http://attacker.invalid"},
])
async def test_feedback_identity_overrides_never_reach_worker(gateway, payload):
    app, _, auth, calls = gateway
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        assert (await browser.post("/api/v1/mobile/feedback", json=payload)).status_code == 422
    assert not calls


async def test_feedback_request_is_small_and_same_origin_authenticated(gateway):
    app, _, auth, calls = gateway
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        assert (await browser.post("/api/v1/mobile/feedback", content=b"a" * 4097, headers={"Content-Type": "application/json"})).status_code == 413
    async with client(app, origin=None, token=auth["alice"]["access_token"]) as browser:
        assert (await browser.post("/api/v1/mobile/feedback", json={})).status_code == 403
    async with client(app) as browser:
        assert (await browser.get("/api/v1/mobile/feedback", params={"session_id": "#00001"})).status_code == 401
    assert not calls


async def test_unauthenticated_post_never_reads_stream(gateway):
    app, _, _, calls = gateway
    consumed = []
    async def payload():
        consumed.append(True)
        raise AssertionError("Unauthenticated request body was read")
        yield b"unreachable"
    async with client(app) as browser:
        response = await browser.post("/api/v1/mobile/query", content=payload(), headers={"Content-Type": "application/json"})
    assert response.status_code == 401 and not consumed and not calls


@pytest.mark.parametrize("headers", [{"Origin": "https://other.invalid"}, {"Origin": "null"}, {"Host": "localhost:8120"}, {"Host": "127.0.0.1:8121"}, {"Sec-Fetch-Site": "cross-site"}])
async def test_origin_host_and_port_are_exact_even_with_forwarded_headers(gateway, headers):
    app, _, auth, calls = gateway
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        result = await browser.get("/api/v1/mobile/sessions", headers={**headers, "X-Forwarded-Host": "127.0.0.1:8120", "X-Forwarded-For": "127.0.0.1"})
    assert result.status_code in {400, 403} and not calls
    assert "access-control-allow-origin" not in result.headers


async def test_write_requests_require_origin_and_auth_bodies_are_bounded(gateway):
    app, _, _, _ = gateway
    async with client(app, origin=None) as browser:
        result = await browser.post("/api/v1/invite/login", json={"username": "alice", "password": PASSWORD})
        assert result.status_code == 403
    async with client(app) as browser:
        large = await browser.post("/api/v1/invite/login", content=b"a" * 8193, headers={"Content-Type": "application/json"})
        assert large.status_code == 413


async def test_business_body_cap_upload_disabled_and_no_implicit_retries(gateway):
    app, _, auth, calls = gateway
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        huge = await browser.post("/api/v1/mobile/query", content=b"a" * 65537, headers={"Content-Type": "application/json"})
        assert huge.status_code == 413
        upload = await browser.post("/api/v1/mobile/upload", data={"session_id": "#00001"}, files={"file": ("synthetic.txt", b"test")})
        assert upload.status_code == 403
    assert not calls


async def test_daily_query_quota_applies_even_if_frontend_sends_extra_calls(gateway):
    app, store, auth, calls = gateway
    for _ in range(50):
        store.reserve_action(auth["alice"]["user"]["id"], "query")
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        result = await browser.post("/api/v1/mobile/query", json={"session_id": "#00001", "question": "synthetic"})
    assert result.status_code == 429 and not calls


async def test_concurrent_same_user_request_rejected_and_other_user_stays_independent(gateway):
    _, store, auth, _ = gateway
    entered, release = asyncio.Event(), asyncio.Event()
    async def blocked(request):
        if request.url.port == 8101:
            entered.set()
            await release.wait()
        return httpx.Response(200, json={"worker": request.url.port})
    app = create_invite_app(store=store, origin=ORIGIN, transport=httpx.MockTransport(blocked))
    async with client(app, token=auth["alice"]["access_token"]) as alice, client(app, token=auth["bobby"]["access_token"]) as bob:
        first = asyncio.create_task(alice.get("/api/v1/mobile/sessions"))
        await asyncio.wait_for(entered.wait(), 1)
        assert (await alice.get("/api/v1/mobile/sessions")).status_code == 429
        assert (await bob.get("/api/v1/mobile/sessions")).status_code == 200
        release.set()
        assert (await first).status_code == 200
        assert (await alice.get("/api/v1/mobile/sessions")).status_code == 200


async def test_global_four_request_limit_and_release_after_failure(gateway):
    _, store, auth, _ = gateway
    for index in range(3):
        username = f"extra{index}"
        invitation = store.add_user(username, 8103 + index)
        auth[username] = store.register(invitation["invite_code"], username, PASSWORD)
    started, release = asyncio.Queue(), asyncio.Event()
    async def blocked(request):
        await started.put(request.url.port)
        await release.wait()
        raise httpx.ConnectError("synthetic failure, never echo me")
    app = create_invite_app(store=store, origin=ORIGIN, transport=httpx.MockTransport(blocked))
    browsers = [client(app, token=row["access_token"]) for row in auth.values()]
    try:
        tasks = [asyncio.create_task(browser.get("/api/v1/mobile/sessions")) for browser in browsers[:4]]
        for _ in range(4):
            await asyncio.wait_for(started.get(), 1)
        assert (await browsers[4].get("/api/v1/mobile/sessions")).status_code == 429
        release.set()
        assert all(result.status_code == 502 for result in await asyncio.gather(*tasks))
        later = await browsers[4].get("/api/v1/mobile/sessions")
        assert later.status_code == 502 and "echo" not in later.text
    finally:
        release.set()
        for browser in browsers:
            await browser.aclose()


async def test_total_upstream_timeout_releases_the_slot_without_retry(gateway, monkeypatch):
    from docthinker.server import invite_app
    _, store, auth, _ = gateway
    calls = []
    async def stalled(request):
        calls.append(request)
        await asyncio.Event().wait()
    monkeypatch.setattr(invite_app, "UPSTREAM_TIMEOUT_SECONDS", 0.01)
    app = create_invite_app(store=store, origin=ORIGIN, transport=httpx.MockTransport(stalled))
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        assert (await browser.get("/api/v1/mobile/sessions")).status_code == 504
        assert len(calls) == 1
        assert (await browser.get("/api/v1/mobile/sessions")).status_code == 504
        assert len(calls) == 2


@pytest.mark.parametrize("user_allowed,worker_ready,expected", [(False, True, 403), (True, False, 503), (True, True, 200)])
async def test_upload_requires_owner_global_user_and_worker_opt_in(gateway, monkeypatch, user_allowed, worker_ready, expected):
    _, store, auth, _ = gateway
    monkeypatch.setenv("DOCTHINKER_INVITE_UPLOADS_ENABLED", "true")
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE users SET allow_uploads=? WHERE username='alice'", (int(user_allowed),))
    calls = []
    async def worker(request):
        calls.append(request)
        return httpx.Response(200, json={"uploads": worker_ready, "document_rag": worker_ready} if request.url.path.endswith("capabilities") else {"status": "processing"})
    app = create_invite_app(store=store, origin=ORIGIN, transport=httpx.MockTransport(worker))
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        result = await browser.post("/api/v1/mobile/upload", data={"session_id": "#00001"}, files={"file": ("synthetic.txt", b"mock")})
    assert result.status_code == expected
    assert sum(request.url.path.endswith("upload") for request in calls) == (1 if expected == 200 else 0)


async def test_oversized_upstream_response_is_not_returned(gateway, monkeypatch):
    from docthinker.server import invite_app
    _, store, auth, _ = gateway
    monkeypatch.setattr(invite_app, "RESPONSE_LIMIT", 40)
    async def worker(_request):
        return httpx.Response(200, json={"too_large": "a" * 41})
    app = create_invite_app(store=store, origin=ORIGIN, transport=httpx.MockTransport(worker))
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        result = await browser.get("/api/v1/mobile/sessions")
    assert result.status_code == 502 and "too_large" not in result.text


@pytest.mark.parametrize("upstream_status", [302, 307, 500, 401])
async def test_upstream_redirects_and_provider_errors_are_not_followed_or_echoed(gateway, upstream_status):
    _, store, auth, _ = gateway
    calls = []
    async def bad(request):
        calls.append(request)
        return httpx.Response(upstream_status, text="secret provider token /private/data", headers={"Location": "https://attacker.invalid", "Set-Cookie": "upstream-secret=true"})
    app = create_invite_app(store=store, origin=ORIGIN, transport=httpx.MockTransport(bad))
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        result = await browser.get("/api/v1/mobile/sessions")
    assert result.status_code == 502 and len(calls) == 1
    assert "secret" not in result.text and "private" not in result.text
    assert "location" not in result.headers and "set-cookie" not in result.headers


async def test_corrupt_nonloopback_binding_fails_closed(gateway):
    app, store, auth, calls = gateway
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE users SET upstream_url='http://localhost:8101' WHERE username='alice'")
    async with client(app, token=auth["alice"]["access_token"]) as browser:
        result = await browser.get("/api/v1/mobile/sessions")
    assert result.status_code == 503 and not calls


async def test_login_rate_limit_uses_socket_ip_not_forwarded_header(gateway):
    app, _, _, _ = gateway
    async with client(app) as browser:
        for i in range(20):
            result = await browser.post("/api/v1/invite/login", json={}, headers={"X-Forwarded-For": f"198.51.100.{i}"})
            assert result.status_code == 422
        result = await browser.post("/api/v1/invite/login", json={}, headers={"X-Forwarded-For": "192.0.2.200"})
    assert result.status_code == 429


@pytest.mark.parametrize("origin", ["", "http://example.com", "https://user:secret@example.com", "https://example.com/path", "https://example.com?token=bad", "https://example.com/#x", "http://127.0.0.1:0", "https://example.com/"])
def test_origin_configuration_fails_closed(origin):
    with pytest.raises(InviteError):
        validated_origin(origin)
