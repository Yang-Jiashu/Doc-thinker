"""Explicit local integration check with two real workers and a fake model.

No repository .env is loaded, no real credentials are used, and no public port
is opened. Run manually, not in the default offline unit suite.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import tempfile
import threading
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request

from docthinker.server.invite_app import create_invite_app
from docthinker.server.invite_store import InviteStore
from docthinker.server.invite_workers import WorkerSupervisor, SHARING_ACK_ENV, SHARING_ACK_VALUE


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def fake_model():
    requests = []
    app = FastAPI()
    @app.post("/v1/chat/completions")
    async def complete(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        requests.append(messages)
        return {"id": "offline-check", "object": "chat.completion", "created": 1,
                "model": "offline-test", "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "离线联调模拟回复，不是真实 AI。"},
                "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                            access_log=False, log_level="critical", proxy_headers=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("Local fake provider did not start")
    try:
        yield port, requests
    finally:
        server.should_exit = True
        thread.join(timeout=10)


async def wait_workers(store):
    async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
        deadline = time.monotonic() + 55
        pending = store.list_worker_bindings()
        while pending and time.monotonic() < deadline:
            remaining = []
            for worker in pending:
                try:
                    result = await client.get(worker["upstream_url"] + "/api/v1/mobile/capabilities",
                        headers={"Authorization": "Bearer " + worker["upstream_token"]})
                    if result.status_code == 200:
                        assert result.json()["chat_ready"]
                        assert not result.json()["uploads"]
                        continue
                except httpx.TransportError:
                    pass
                remaining.append(worker)
            pending = remaining
            if pending:
                await asyncio.sleep(0.15)
        assert not pending, "Isolated workers did not become ready"


async def exercise(store, alice, bob, requests):
    origin = "http://127.0.0.1:8088"
    app = create_invite_app(store=store, origin=origin)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=origin,
                                     headers={"Origin": origin}) as client:
            identities = []
            for invitation in (alice, bob):
                result = await client.post("/api/v1/invite/register", json={
                    "username": invitation["username"], "invite_code": invitation["invite_code"],
                    "password": "offline-integration-password",
                })
                assert result.status_code == 200, result.status_code
                identities.append({"Authorization": "Bearer " + result.json()["access_token"]})
            a, b = identities
            sid = None
            for identity, title in ((a, "Alice only"), (b, "Bob only")):
                result = await client.post("/api/v1/mobile/sessions", headers=identity, json={"title": title})
                assert result.status_code == 200, result.status_code
                current = result.json()["session"]["id"]
                sid = current if sid is None else sid
                assert current == sid, "Independent workers should both start with their own first session"
            # Startup, registration and workspace creation must not use a model.
            assert not requests
            a_private = "alice-private-test-phrase"
            result = await client.post("/api/v1/mobile/query", headers=a, json={
                "session_id": sid, "question": a_private, "use_memory": False,
                "use_conversation_context": True, "use_llm_cache": False, "use_self_evolution": False,
            })
            assert result.status_code == 200, result.status_code
            assert "离线联调" in result.json()["answer"]
            result = await client.get("/api/v1/mobile/history", headers=b, params={"session_id": sid})
            assert result.status_code == 200 and result.json()["history"] == []
            result = await client.get("/api/v1/mobile/history", headers=a, params={"session_id": sid})
            assert a_private in result.text
            result = await client.post("/api/v1/mobile/query", headers=b, json={
                "session_id": sid, "question": "bob-private-test-phrase", "use_memory": False,
                "use_conversation_context": True, "use_llm_cache": False, "use_self_evolution": False,
            })
            assert result.status_code == 200
            assert len(requests) == 2
            assert a_private not in str(requests[-1])
            assert (await client.get("/api/v1/settings", headers=a)).status_code == 404
            await client.post("/api/v1/invite/logout", headers=a)
            assert (await client.get("/api/v1/mobile/sessions", headers=a)).status_code == 401
            store.revoke_user("bob")
            assert (await client.get("/api/v1/mobile/sessions", headers=b)).status_code == 401
    print("PASS: real isolated workers, separate same-number sessions/history/model context, logout/revocation; fake local model only.")


def main():
    # Resolve /var and /tmp aliases before passing the strict private-store check.
    with tempfile.TemporaryDirectory(prefix="docthinker-invite-check-") as temporary, fake_model() as (port, requests):
        root = Path(temporary).resolve() / "registry"
        store = InviteStore.initialize(root)
        used = {port}
        def worker_port():
            candidate = free_port()
            while candidate in used:
                candidate = free_port()
            used.add(candidate)
            return candidate
        alice = store.add_user("alice", worker_port())
        bob = store.add_user("bob", worker_port())
        # Permission is for our own loopback fake provider, not the user's Step Plan.
        environment = {SHARING_ACK_ENV: SHARING_ACK_VALUE,
                       "LLM_BINDING_API_KEY": "offline-not-a-real-provider-key",
                       "LLM_VLM_HOST": f"http://127.0.0.1:{port}/v1", "LLM_MODEL": "offline-test",
                       "PATH": os.defpath, "LANG": "en_US.UTF-8"}
        supervisor = WorkerSupervisor(store, environment)
        try:
            supervisor.start()
            asyncio.run(wait_workers(store))
            assert not requests
            asyncio.run(exercise(store, alice, bob, requests))
        finally:
            supervisor.close()


if __name__ == "__main__":
    main()
