"""Real loopback memory/graph acceptance, with zero permitted model requests.

Build clients/mobile H5, then run python -m tests.manual_memory_graph_runtime_check.
--serve retains a disposable two-user preview after all assertions pass. Only
synthetic project notes, accounts and credentials enter this private fixture.
This checks persistence/provenance and isolation, not AI or causal reliability.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
from urllib.parse import urljoin, urlsplit
import signal
import subprocess
import sys
import tempfile
import threading
import time

# Set before importing application/test helpers; never load a user's dotenv.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from docthinker.server.invite_store import InviteStore
from docthinker.server.invite_workers import SHARING_ACK_ENV, SHARING_ACK_VALUE
from tests.manual_invite_runtime_check import FAKE_KEY, ROOT, free_port, port_open
from tests.manual_research_runtime_check import PASSWORD, PREFIX, client, login, ready_document, running, upload

BASE = PREFIX + "/research"
PROJECT = "周五客户演示 · 合成项目"


class _ShellIcons(HTMLParser):
    """Read the build's actual icon declarations instead of old brand paths."""
    def __init__(self):
        super().__init__()
        self.manifests, self.icons = [], []

    def handle_starttag(self, tag, attrs):
        if tag != "link":
            return
        values = dict(attrs)
        relations = values.get("rel", "").split()
        if "manifest" in relations and values.get("href"):
            self.manifests.append(values["href"])
        if any(value in relations for value in ("icon", "apple-touch-icon")) and values.get("href"):
            self.icons.append(values["href"])


def check_shell_assets(browser, origin):
    """Actual public HTTP image delivery, with no SPA/source-file fallback."""
    shell = browser.get("/index.html")
    assert shell.status_code == 200 and shell.headers.get("cache-control") == "no-store"
    assert "connect-src 'self'" in shell.headers.get("content-security-policy", "")
    declared = _ShellIcons()
    declared.feed(shell.text)
    assert len(declared.manifests) == 1 and declared.icons, "Current H5 icon declarations are missing"

    def local_path(reference):
        resolved = urlsplit(urljoin(origin + "/index.html", reference))
        expected = urlsplit(origin)
        assert (resolved.scheme, resolved.netloc) == (expected.scheme, expected.netloc), "Icon must stay loopback same-origin"
        assert not resolved.query and not resolved.fragment, "Build icon must be a stable local asset path"
        return resolved.path

    manifest = browser.get(local_path(declared.manifests[0]))
    assert manifest.status_code == 200 and manifest.headers.get("cache-control") == "no-store"
    icons = manifest.json().get("icons")
    assert type(icons) is list and icons and all(type(item) is dict and type(item.get("src")) is str for item in icons)
    paths = {local_path(reference) for reference in declared.icons}
    paths.update(local_path(item["src"]) for item in icons)
    # The header mark shares the current icon directory, not a frozen old path.
    logo_path = str(Path(local_path(icons[0]["src"])).parent / "logo-64.png")
    paths.add(logo_path)
    for path in sorted(paths):
        response = browser.get(path)
        assert response.status_code == 200, "Current H5 image route unavailable: " + path
        assert response.content and response.headers.get("content-type", "").startswith("image/")
        assert response.headers.get("cache-control") == "no-store"
        assert response.headers.get("x-content-type-options") == "nosniff"
        if path.lower().endswith(".png"):
            assert response.content.startswith(b"\x89PNG\r\n\x1a\n"), "Image route returned non-PNG data"
    forbidden = {"/.env", "/README.md", "/package.json", "/assets/private.js.map",
                 str(Path(logo_path).parent / "private.txt")}
    if logo_path.startswith("/static/"):
        # An image outside the explicit static namespace must still be refused.
        forbidden.add("/" + logo_path.removeprefix("/static/"))
    for path in sorted(forbidden):
        denied = browser.get(path)
        assert denied.status_code == 404 and "<div id=\"app\"" not in denied.text, "Non-asset route leaked or became SPA HTML"


@contextmanager
def provider_tripwire(used):
    """Count every attempted request, not only successfully parsed completions."""
    calls = []
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def reject_model_request(request: Request, _next):
        calls.append((request.method, request.url.path))
        return JSONResponse({"error": "This synthetic observation check forbids model requests"}, status_code=503)

    port = free_port(used)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
        access_log=False, log_level="critical", proxy_headers=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(.05)
    assert server.started, "Loopback provider guard did not start"
    try:
        yield port, calls
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        assert not thread.is_alive(), "Loopback provider guard did not stop"
        assert not calls, "Observation unexpectedly attempted a model request"


def request(browser, identity, method, path, **kwargs):
    response = browser.request(method, PREFIX + path, headers=identity, **kwargs)
    assert response.status_code == 200, f"Synthetic observation route failed: {method} {path} ({response.status_code})"
    assert response.headers.get("cache-control") == "no-store", "Private response must not be cached"
    return response.json()


def new_session(browser, identity, title):
    return request(browser, identity, "POST", "/sessions", json={"title": title})["session"]["id"]


def observation(browser, identity, sid, **params):
    result = request(browser, identity, "GET", "/research/observability", params={"session_id": sid, **params})
    assert result["schema_version"] == 1 and result["session_id"] == sid
    assert result["read_only"] is True and result["profile"] == "bounded_research"
    assert result["graph"]["causality_verified"] is False
    assert len(result["graph"]["edges"]) <= 400
    return result


def seed_project(browser, identity, sid, owner):
    marker = owner + "独立合成资料"
    text = (f"{marker}。这是虚构项目记录，不是真实客户资料。\n"
            "周五客户演示先完成登录与资料核对，再检查讲解路线。\n"
            "上午负责人确认采用轻量方案，下午安排演示彩排。\n"
            "彩排任务依赖方案确认；这是用户记录的计划依赖，不是因果证明。\n"
            "当前记录没有证明需求变化导致延期，也不代表系统自动推断出这些关系。")
    document = upload(browser, identity, sid, owner + "-项目笔记-合成.txt", text.encode())["document"]
    ready = ready_document(browser, identity, sid, document["id"])
    source = ready["chunks"][0]
    source_id, source_text = source["id"], source["text"]
    memory = request(browser, identity, "POST", "/research/memory", json={"session_id": sid,
        "kind": "decision", "text": marker + "：本轮演示采用轻量方案，先核对原文再讲解。",
        "source_ids": [source_id]})["memory"]
    request(browser, identity, "POST", "/research/memory", json={"session_id": sid,
        "kind": "preference", "text": "我希望先看一句话结论，再看三条依据。"})
    source_origin = {"kind": "source", "source_id": source_id, "span": [0, len(source_text)],
                     "sha256": hashlib.sha256(source_text.encode()).hexdigest()}
    memory_origin = {"kind": "memory", "memory_id": memory["id"], "revision": memory["revision"]}
    now = time.time() - 3600
    event_a = request(browser, identity, "POST", "/research/events", json={"session_id": sid,
        "confirmed": True, "event_type": "decision", "action": marker + "：确认轻量演示方案",
        "occurred_from": now, "occurred_until": now + 60,
        "conditions": ["仅为用户显式记录的合成项目决定"], "origins": [source_origin]})["event"]
    event_b = request(browser, identity, "POST", "/research/events", json={"session_id": sid,
        "confirmed": True, "event_type": "task", "action": marker + "：安排演示彩排",
        "occurred_from": now + 120, "occurred_until": now + 180,
        "conditions": ["确认方案后安排彩排；不授权自动执行"], "origins": [memory_origin]})["event"]
    for left, right, relation in ((event_a, event_b, "before"), (event_b, event_a, "depends_on")):
        request(browser, identity, "POST", "/research/event-relations", json={"session_id": sid,
            "confirmed": True, "source_id": left["id"], "target_id": right["id"], "relation": relation,
            "conditions": ["用户显式记录；有出处不代表因果已被证明"]})
    return {"sid": sid, "marker": marker, "document_id": document["id"], "source_id": source_id,
            "source_text": source_text, "memory": memory, "event_a": event_a, "event_b": event_b}


def assert_seed(browser, identity, fixture):
    result = observation(browser, identity, fixture["sid"])
    assert result["counts"]["documents"]["total"] == 1
    assert result["counts"]["chunks"]["available"] >= 1
    assert result["counts"]["memories"]["total"] == 2
    assert result["counts"]["events"]["total"] == 2 and result["counts"]["relations"]["total"] == 2
    assert {node["type"] for node in result["graph"]["nodes"]} == {"document", "source", "memory", "event"}
    assert {edge["type"] for edge in result["graph"]["edges"]} == {
        "contains_source", "memory_source", "event_origin", "event_relation"}
    assert {edge["relation"] for edge in result["graph"]["edges"] if edge["type"] == "event_relation"} == {"before", "depends_on"}
    assert all(edge["causality_verified"] is False and edge["is_discovered"] is False for edge in result["graph"]["edges"])
    assert not result["graph"]["window_limited"]
    assert fixture["source_text"] not in json.dumps(result, ensure_ascii=False), "Projection must not export source bodies"
    for node in result["graph"]["nodes"]:
        for identifier in node["source_ids"]:
            source = request(browser, identity, "GET", "/research/source",
                             params={"session_id": fixture["sid"], "chunk_id": identifier})["chunk"]
            assert source["id"] == identifier and fixture["marker"] in source["content"]
    narrow = observation(browser, identity, fixture["sid"], limit=1)
    assert narrow["counts"] == result["counts"] and narrow["window"]["limited"]["memories"] is True
    assert narrow["graph"]["window_limited"] is True
    return result


def check_revision_and_withdrawal(browser, identity, fixture):
    sid, memory = fixture["sid"], fixture["memory"]
    changed = request(browser, identity, "POST", "/research/memory", json={"session_id": sid,
        "memory_id": memory["id"], "expected_revision": memory["revision"],
        "text": "修订后的合成决定：缩短讲解内容，但不自动执行任务。", "kind": "decision"})["memory"]
    assert changed["revision"] == 2
    result = observation(browser, identity, sid)
    events = {row["id"]: row for row in result["events"]}
    assert events[fixture["event_a"]["id"]]["status"] == "active"
    assert events[fixture["event_b"]["id"]]["redacted"] is True
    assert all(row["redacted"] is True for row in result["relations"])
    assert not any(edge["type"] == "event_relation" for edge in result["graph"]["edges"])
    assert result["counts"]["memories"]["versions"] == 3
    request(browser, identity, "POST", "/research/documents/delete", json={"session_id": sid,
        "document_id": fixture["document_id"]})
    result = observation(browser, identity, sid)
    assert fixture["marker"] not in json.dumps(result, ensure_ascii=False)
    assert result["counts"]["documents"]["by_status"]["deleted"] == 1
    assert result["counts"]["memories"]["by_status"]["invalidated"] == 1
    assert result["sources"] == [] and result["graph"]["edges"] == []
    assert all(row["redacted"] is True for row in result["events"] + result["relations"])
    withdrawn = browser.get(BASE + "/source", headers=identity,
                            params={"session_id": sid, "chunk_id": fixture["source_id"]})
    assert withdrawn.status_code == 404 and fixture["marker"] not in withdrawn.text


def exercise(origin, invitations, store, calls):
    with client(origin) as browser:
        check_shell_assets(browser, origin)
        assert browser.get(PREFIX + "/research/observability", params={"session_id": "#00001"}).status_code == 401
        identities = {}
        for invitation in invitations:
            registered = browser.post("/api/v1/invite/register", json={"username": invitation["username"],
                "invite_code": invitation["invite_code"], "password": PASSWORD})
            assert registered.status_code == 200, "Synthetic account registration failed"
            identities[invitation["username"]] = login(browser, invitation["username"])

        # Destructive checks have their own session. Preserved demos are created
        # afterwards so the browser initially selects a healthy recent project.
        checked_sid = new_session(browser, identities["preview"], "撤回与修订检查 · 合成资料")
        foreign_checked_sid = new_session(browser, identities["bobby"], "隔离边界检查 · 空会话")
        assert checked_sid == foreign_checked_sid, "Same-ID worker-isolation fixture was not established"
        checked = seed_project(browser, identities["preview"], checked_sid, "撤回检查")
        assert_seed(browser, identities["preview"], checked)
        empty = observation(browser, identities["bobby"], foreign_checked_sid)
        assert empty["counts"]["memories"]["total"] == empty["counts"]["events"]["total"] == 0
        assert empty["graph"]["nodes"] == []
        check_revision_and_withdrawal(browser, identities["preview"], checked)

        demos, snapshots = {}, {}
        for owner in ("preview", "bobby"):
            sid = new_session(browser, identities[owner], PROJECT)
            demos[owner] = seed_project(browser, identities[owner], sid, owner)
            snapshots[owner] = assert_seed(browser, identities[owner], demos[owner])
        assert demos["preview"]["sid"] == demos["bobby"]["sid"]
        for owner, other in (("preview", "bobby"), ("bobby", "preview")):
            sid = demos[owner]["sid"]
            result = observation(browser, identities[owner], sid)
            assert demos[other]["marker"] not in json.dumps(result, ensure_ascii=False)
            denied = browser.get(BASE + "/source", headers=identities[owner],
                params={"session_id": sid, "chunk_id": demos[other]["source_id"]})
            assert denied.status_code == 404, "Source crossed an authenticated worker boundary"
            assert result["counts"] == snapshots[owner]["counts"]
            assert browser.get(PREFIX + "/memory/dashboard", headers=identities[owner],
                               params={"session_id": sid}).status_code == 404
        assert not calls, "Read/edit/observe unexpectedly called the model"
        with store._connect() as db:
            assert db.execute("SELECT count(*) FROM usage WHERE kind='query'").fetchone()[0] == 0


def serve(origin, calls):
    stop = threading.Event()
    previous = {number: signal.signal(number, lambda *_: stop.set()) for number in (signal.SIGINT, signal.SIGTERM)}
    print(json.dumps({"preview_origin": origin, "accounts": [
        {"username": owner, "password": PASSWORD} for owner in ("preview", "bobby")]}), flush=True)
    try:
        while not stop.wait(1):
            assert not calls, "Preview unexpectedly attempted a model request"
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="Keep the synthetic loopback preview alive after acceptance")
    args = parser.parse_args(argv)
    build_dir = (ROOT / "clients/mobile/dist/build/h5").resolve()
    assert build_dir.is_dir(), "Build current mobile H5 before running this check"
    with tempfile.TemporaryDirectory(prefix="docthinker-memory-graph-check-") as directory:
        temporary = Path(directory).resolve()
        temporary.chmod(0o700)
        home, tmp = temporary / "home", temporary / "tmp"
        home.mkdir(mode=0o700)
        tmp.mkdir(mode=0o700)
        store, used = InviteStore.initialize(temporary / "registry"), set()
        invitations = [store.add_user(owner, free_port(used), allow_uploads=True) for owner in ("preview", "bobby")]
        gateway_port = free_port(used)
        owned_ports = [gateway_port, *(int(row["upstream_url"].rsplit(":", 1)[1]) for row in store.list_worker_bindings())]
        origin = f"http://127.0.0.1:{gateway_port}"
        secret = temporary / "synthetic-model-key"
        descriptor = os.open(secret, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(FAKE_KEY + "\n")
        with provider_tripwire(used) as (provider_port, calls):
            environment = {"PATH": os.defpath, "LANG": "en_US.UTF-8", "HOME": str(home), "TMPDIR": str(tmp),
                "PYTHONPATH": str(ROOT), "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHON_DOTENV_DISABLED": "1",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                SHARING_ACK_ENV: SHARING_ACK_VALUE, "DOCTHINKER_INVITE_DATA_DIR": str(store.root),
                "DOCTHINKER_INVITE_ORIGIN": origin, "DOCTHINKER_INVITE_BIND_HOST": "127.0.0.1",
                "DOCTHINKER_INVITE_PORT": str(gateway_port), "DOCTHINKER_INVITE_WEB_DIR": str(build_dir),
                "DOCTHINKER_RESEARCH_ENABLED": "true", "DOCTHINKER_RESEARCH_CONTEXT_ASSIST": "false",
                "DOCTHINKER_RESEARCH_SEMANTIC": "off", "DOCTHINKER_RESEARCH_MEMORY_SEMANTIC": "off",
                "LLM_BINDING_API_KEY_FILE": str(secret), "LLM_VLM_HOST": f"http://127.0.0.1:{provider_port}/v1",
                "LLM_MODEL": "offline-runtime-check"}
            checked = subprocess.run([sys.executable, "-m", "docthinker.server.invite_runtime", "--check"],
                env=environment, cwd=temporary, capture_output=True, timeout=20)
            assert checked.returncode == 0, "Synthetic runtime configuration check failed"
            assert not calls and not any(port_open(port) for port in owned_ports)
            with running(environment, temporary, origin, store, owned_ports):
                exercise(origin, invitations, store, calls)
                if args.serve:
                    serve(origin, calls)
                else:
                    print("PASS: real gateway + two workers; project TXT, confirmed memory, explicit event relations, "
                          "source links, same-session tenant isolation, revisions and withdrawal; "
                          "zero provider requests, zero vendor calls. Not AI quality or causal-proof acceptance.", flush=True)
        assert not any(port_open(port) for port in used), "Synthetic loopback listeners leaked"


if __name__ == "__main__":
    main()
