"""Explicit loopback-only check of the packaged invitation runtime.

Runs the real supervisor, two isolated workers and built H5 gateway as separate
processes. Uses only a temporary private registry, a fake model and a fake key
file; never loads repository dotenv or inherited provider credentials. This is
manual integration coverage, not part of the normal offline unit-test suite.

Run with the deployment environment's Python from the repository root:
    python -m tests.manual_invite_runtime_check
"""
from __future__ import annotations

from contextlib import contextmanager
from html.parser import HTMLParser
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request

from docthinker.server.invite_store import InviteStore
from docthinker.server.invite_workers import SHARING_ACK_ENV, SHARING_ACK_VALUE


ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "local-integration-only-not-a-real-provider-key"
PASSWORD = "local-integration-test-password"


def free_port(used):
    while True:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        if port not in used and port >= 1024:
            used.add(port)
            return port


def port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


@contextmanager
def fake_model(used):
    calls = []
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def complete(request: Request):
        assert request.headers.get("authorization") == "Bearer " + FAKE_KEY
        body = await request.json()
        assert body.get("model") == "offline-runtime-check"
        calls.append(body.get("messages", []))
        return {
            "id": "offline-runtime-check", "object": "chat.completion", "created": 1,
            "model": "offline-runtime-check", "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "离线运行包联调回复，不是真实 AI。"},
                "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    port = free_port(used)
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, access_log=False,
        log_level="critical", proxy_headers=False,
    ))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "Loopback fake provider failed to start"
    try:
        yield port, calls
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        assert not thread.is_alive(), "Fake provider did not stop"


class AssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        for name in ("src", "href"):
            value = attributes.get(name, "")
            if value.startswith(("/assets/", "./static/brand/", "./manifest.", "./pwa-register.")):
                self.paths.append(value.removeprefix("./"))


def wait_ready(process, origin, store):
    with httpx.Client(base_url=origin, timeout=1, trust_env=False,
                      follow_redirects=False) as client:
        deadline = time.monotonic() + 110
        while time.monotonic() < deadline:
            assert process.poll() is None, "Packaged runtime exited during startup"
            try:
                response = client.get("/index.html")
                if response.status_code == 200:
                    ready = True
                    for binding in store.list_worker_bindings():
                        capabilities = client.get(
                            binding["upstream_url"] + "/api/v1/mobile/capabilities",
                            headers={"Authorization": "Bearer " + binding["upstream_token"]},
                        )
                        ready = ready and capabilities.status_code == 200
                        if capabilities.status_code == 200:
                            assert capabilities.json()["chat_ready"] is True
                            assert capabilities.json()["uploads"] is False
                    if ready:
                        return
            except httpx.TransportError:
                pass
            time.sleep(0.15)
    raise AssertionError("Packaged runtime did not become ready in time")


def exercise(origin, invitations, calls):
    with httpx.Client(base_url=origin, headers={"Origin": origin}, timeout=30,
                      trust_env=False, follow_redirects=False) as client:
        redirect = client.get("/")
        assert redirect.status_code == 302
        assert redirect.headers["location"] == "/index.html#/pages/invite/index"
        shell = client.get("/index.html")
        assert shell.status_code == 200 and shell.headers["content-type"].startswith("text/html")
        assert "<div id=\"app\"></div>" in shell.text
        assert shell.headers["cache-control"] == "no-store"
        assert shell.headers["x-frame-options"] == "DENY"
        assert "connect-src 'self'" in shell.headers["content-security-policy"]
        parser = AssetParser()
        parser.feed(shell.text)
        assert any(path.startswith("/assets/") and path.endswith(".js") for path in parser.paths)
        for path in set(parser.paths + ["/service-worker.js"]):
            asset = client.get("/" + path.lstrip("/"))
            assert asset.status_code == 200 and asset.content, "Built H5 asset unavailable"

        assert client.get("/api/v1/mobile/sessions").status_code == 401
        identities = []
        for invitation in invitations:
            registered = client.post("/api/v1/invite/register", json={
                "username": invitation["username"], "invite_code": invitation["invite_code"],
                "password": PASSWORD,
            })
            assert registered.status_code == 200, "Invitation registration failed"
            logged_in = client.post("/api/v1/invite/login", json={
                "username": invitation["username"], "password": PASSWORD,
            })
            assert logged_in.status_code == 200, "Invitation login failed"
            identities.append({"Authorization": "Bearer " + logged_in.json()["access_token"]})

        sessions = []
        for identity, title in zip(identities, ("Alice private workspace", "Bob private workspace")):
            response = client.post("/api/v1/mobile/sessions", headers=identity, json={"title": title})
            assert response.status_code == 200, "Isolated workspace creation failed"
            sessions.append(response.json()["session"]["id"])
            capabilities = client.get("/api/v1/mobile/capabilities", headers=identity)
            assert capabilities.status_code == 200
            assert capabilities.json()["chat_ready"] is True
            assert capabilities.json()["uploads"] is False
        assert sessions[0] == sessions[1], "Fresh isolated workers should both have their own first session"
        assert not calls, "Startup, assets, authentication and workspace creation must not call a model"
        alice, bob = identities
        session_id = sessions[0]
        questions = ("alice-only-private-runtime-phrase", "bob-only-private-runtime-phrase")
        request_id = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
        answer_ids = []
        for index, identity in enumerate(identities):
            payload = {
                "session_id": session_id, "question": questions[index], "use_memory": False,
                "use_conversation_context": True, "use_llm_cache": False,
                "use_self_evolution": False, "request_id": request_id,
            }
            answered = client.post("/api/v1/mobile/query", headers=identity, json=payload)
            assert answered.status_code == 200, "Fake-provider chat through packaged runtime failed"
            assert "离线运行包联调" in answered.json()["answer"]
            assert answered.json()["request_status"] == "completed"
            replay = client.post("/api/v1/mobile/query", headers=identity, json=payload)
            assert replay.status_code == 200 and replay.json() == answered.json()
            status = client.get("/api/v1/mobile/query/status", headers=identity,
                                params={"session_id": session_id, "request_id": request_id})
            assert status.status_code == 200 and status.json() == answered.json()
            answer_ids.append(answered.json()["answer_id"])
            if index == 0:
                other = client.get("/api/v1/mobile/history", headers=bob, params={"session_id": session_id})
                assert other.status_code == 200 and other.json()["history"] == []
                foreign = client.get("/api/v1/mobile/query/status", headers=bob,
                                     params={"session_id": session_id, "request_id": request_id})
                assert foreign.json() == {"request_id": request_id, "request_status": "not_found"}
        assert len(calls) == 2, "Each user's query should call the local fake provider exactly once"
        assert questions[0] in str(calls[0]) and questions[1] not in str(calls[0])
        assert questions[1] in str(calls[1]) and questions[0] not in str(calls[1])
        for index, identity in enumerate(identities):
            history = client.get("/api/v1/mobile/history", headers=identity, params={"session_id": session_id})
            assert history.status_code == 200 and questions[index] in history.text
            assert questions[1 - index] not in history.text

        assert answer_ids[0] != answer_ids[1], "Persisted answer IDs must not collide between users"
        feedback = {"session_id": session_id, "answer_id": answer_ids[0], "rating": "inaccurate", "note": "alice-only-feedback"}
        recorded = client.post("/api/v1/mobile/feedback", headers=alice, json=feedback)
        assert recorded.status_code == 200, "Explicit feedback failed through the real gateway/worker"
        foreign = client.post("/api/v1/mobile/feedback", headers=bob, json=feedback)
        assert foreign.status_code == 404, "A user could rate another user's persisted answer"
        bob_feedback = client.get("/api/v1/mobile/feedback", headers=bob, params={"session_id": session_id})
        assert bob_feedback.json() == {"items": []}, "Feedback crossed isolated user directories"
        restored = client.get("/api/v1/mobile/history", headers=alice, params={"session_id": session_id})
        assert restored.json()["history"][-1]["id"] == answer_ids[0]
        assert restored.json()["history"][-1]["feedback"] == recorded.json()["feedback"]
        assert "alice-only-feedback" not in client.get(
            "/api/v1/mobile/history", headers=bob, params={"session_id": session_id},
        ).text
        assert len(calls) == 2, "Explicit feedback must not call the model or add learning"

        for path in ("/api/v1/settings", "/api/v1/knowledge-graph/data", "/docs", "/openapi.json",
                     "/.env", "/pyproject.toml", "/assets/../../.env"):
            assert client.get(path, headers=alice).status_code == 404, "Non-allowlisted route was exposed"
        assert client.get("/index.html", headers={"Host": "unexpected.invalid"}).status_code == 400
        assert client.get("/index.html", headers={"Origin": "https://unexpected.invalid"}).status_code == 403


def main():
    build_dir = (ROOT / "clients/mobile/dist/build/h5").resolve()
    assert build_dir.is_dir(), "Build the mobile H5 shell before running this manual check"
    with tempfile.TemporaryDirectory(prefix="docthinker-runtime-check-") as temporary:
        temporary = Path(temporary).resolve()
        home, tmp = temporary / "home", temporary / "tmp"
        home.mkdir(mode=0o700)
        tmp.mkdir(mode=0o700)
        store = InviteStore.initialize(temporary / "registry")
        used = set()
        invitations = (store.add_user("alice", free_port(used)), store.add_user("bob", free_port(used)))
        gateway_port = free_port(used)
        owned_ports = [gateway_port, *(int(item["upstream_url"].rsplit(":", 1)[1]) for item in store.list_worker_bindings())]
        origin = f"http://127.0.0.1:{gateway_port}"
        # Fixture writes are confined to the temporary directory. Never copy any
        # real environment credential or dotenv content into this deployment.
        secret = temporary / "fake-model-key"
        descriptor = os.open(secret, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(FAKE_KEY + "\n")
        with fake_model(used) as (provider_port, calls):
            environment = {
                "PATH": os.defpath, "LANG": "en_US.UTF-8", "HOME": str(home), "TMPDIR": str(tmp),
                "PYTHONPATH": str(ROOT), "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
                "PYTHON_DOTENV_DISABLED": "1", SHARING_ACK_ENV: SHARING_ACK_VALUE,
                "DOCTHINKER_INVITE_DATA_DIR": str(store.root), "DOCTHINKER_INVITE_ORIGIN": origin,
                "DOCTHINKER_INVITE_BIND_HOST": "127.0.0.1", "DOCTHINKER_INVITE_PORT": str(gateway_port),
                "DOCTHINKER_INVITE_WEB_DIR": str(build_dir), "LLM_BINDING_API_KEY_FILE": str(secret),
                "LLM_VLM_HOST": f"http://127.0.0.1:{provider_port}/v1", "LLM_MODEL": "offline-runtime-check",
            }
            command = [sys.executable, "-m", "docthinker.server.invite_runtime"]
            checked = subprocess.run(command + ["--check"], env=environment, cwd=temporary,
                                     capture_output=True, timeout=20)
            assert checked.returncode == 0, "Packaged runtime configuration check failed"
            assert not calls and all(not port_open(port) for port in owned_ports)
            process = subprocess.Popen(command, env=environment, cwd=temporary,
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, start_new_session=True)
            try:
                wait_ready(process, origin, store)
                exercise(origin, invitations, calls)
                health_environment = {key: value for key, value in environment.items()
                                      if key not in {SHARING_ACK_ENV, "LLM_BINDING_API_KEY_FILE", "LLM_VLM_HOST", "LLM_MODEL"}}
                # Health probes must not even resolve a provider secret, so a
                # deliberately nonexistent secret path cannot break readiness.
                health_environment["LLM_BINDING_API_KEY_FILE"] = str(temporary / "does-not-exist")
                before = len(calls)
                healthy = subprocess.run(command + ["--healthcheck"], env=health_environment,
                                         cwd=temporary, capture_output=True, timeout=12)
                assert healthy.returncode == 0, "Credential-free packaged healthcheck failed"
                assert len(calls) == before == 2, "Healthcheck unexpectedly called a provider"
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                        raise AssertionError("Runtime failed to exit after SIGTERM") from None
                deadline = time.monotonic() + 5
                while any(port_open(port) for port in owned_ports) and time.monotonic() < deadline:
                    time.sleep(0.1)
                assert all(not port_open(port) for port in owned_ports), "SIGTERM left an owned gateway or worker listener running"
            assert process.returncode == 0, "SIGTERM shutdown was not clean"
    print("PASS: packaged runtime, two isolated workers, real HTTP login/chat/history/answer feedback, idempotent answer/status replay, built H5 assets, "
          "blocked administration routes, credential-free healthcheck and SIGTERM cleanup; loopback fake model only.")


if __name__ == "__main__":
    main()
