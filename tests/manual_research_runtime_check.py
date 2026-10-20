"""Loopback acceptance: real invitation workers, offline research, fake answers.

Run ``python -m tests.manual_research_runtime_check`` after building mobile H5.
``--serve`` keeps the already-exercised disposable preview alive for browser QA.
All documents, accounts and model credentials are synthetic. This does not test
real-model quality, Internet hosting, customer capacity or vendor permissions.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import httpx
import uvicorn
from fastapi import FastAPI, Request

from docthinker.server.invite_store import InviteStore
from docthinker.server.invite_workers import SHARING_ACK_ENV, SHARING_ACK_VALUE
from tests.manual_invite_runtime_check import FAKE_KEY, ROOT, free_port, port_open
from tests.test_research_service_unit import pdf_with_text

PASSWORD = "OfflinePreviewOnly2026"
PREFIX = "/api/v1/mobile"
TEXT = ("Renewable supply reduces energy imports in the synthetic study. "
        "PRIVATE_RESEARCH_FIXTURE_QUOTE. Climate policy increases renewable investment.")
PREFERENCE_FIRST = "Prefer concise Chinese explanations."
PREFERENCE_UPDATED = "Prefer concise English explanations."
FINDING = "Renewable supply and energy imports are connected in this quoted example."
QUESTION = "How does renewable supply reduce energy imports?"


def model_prompt_data(messages):
    """Read synthetic test inputs as data; this is not a semantic evaluator."""
    prompt = next(row["content"] for row in reversed(messages) if row["role"] == "user")
    decoder = json.JSONDecoder()
    marker = "UNTRUSTED_RESEARCH_DATA_JSON:\n"
    data, _ = decoder.raw_decode(prompt.split(marker, 1)[1])
    supporting = {"supporting_material": [], "limits": []}
    helper = prompt.find('{"supporting_material":')
    if helper >= 0:
        supporting, _ = decoder.raw_decode(prompt[helper:])
    return data, supporting


def fake_grounded_reply(messages):
    data, supporting = model_prompt_data(messages)
    paragraphs = ["这是离线流程测试回答，不是真实模型效果。"]
    quotes = data.get("source_quotes", [])
    if quotes:
        # Quote only a bounded complete synthetic sentence. This fixture must
        # not manufacture a source, or mistake helper material for source text.
        source = quotes[0]
        literal = source["content"].split(". ", 1)[0].rstrip(".") + "."
        paragraphs.append(f"资料摘录：{literal} [{source['label']}]")
    else:
        paragraphs.append("这次没有匹配原文，仅作普通交流与背景梳理。")
    for item in supporting["supporting_material"]:
        labels = " ".join(f"[{label}]" for label in item["sources"])
        paragraphs.append(f"写作参考（不是原文直接结论）：{item['line']} {labels}")
        for hypothesis in item.get("hypotheses", []):
            conditions = "、".join(hypothesis.get("conditions", [])) or "尚需检查"
            paragraphs.append(f"待验证思路：{hypothesis['line']}；适用条件：{conditions}；目标场景尚未验证。")
        if "counterexample_groups" in item:
            paragraphs.append(f"本次样本有 {item['support_groups']} 组支持、{item['counterexample_groups']} 组反例，不代表普遍规律。")
    paragraphs.extend("材料限制：" + hint for hint in supporting["limits"])
    paragraphs.append("缺失的因果环节仍需原文，不会自动补齐。")
    return "\n\n".join(paragraphs)


def expected_output_cap(messages):
    """Assert the real production profile, not the pre-review 8K fixture cap.

    Only a separately rendered, validated host requirement packet activates
    the reviewed profile. A marker inside a quoted user/source JSON string
    cannot change this offline fake-provider contract.
    """
    data, _ = model_prompt_data(messages)
    prompt = next(row['content'] for row in reversed(messages) if row['role'] == 'user')
    marker = '\nTASK_REQUIREMENTS_DATA_JSON:\n'
    if marker in '\n' + prompt:
        packet, _ = json.JSONDecoder().raw_decode(('\n' + prompt).split(marker, 1)[1])
        # The rendered contract is a compact pointer catalog, not the full
        # private host packet validated before production rendering.
        rows = packet.get('requirements') if type(packet) is dict else None
        if (type(packet) is not dict or set(packet) != {'version', 'requirements_sha256', 'requirements'}
                or type(packet['version']) is not int or packet['version'] not in {1, 2}
                or type(packet['requirements_sha256']) is not str
                or not re.fullmatch(r'[0-9a-f]{64}', packet['requirements_sha256'])
                or type(rows) is not list or not 1 <= len(rows) <= 26):
            raise ValueError('Invalid rendered host requirements')
        seen = set()
        for row in rows:
            if type(row) is not dict or set(row) != {'id', 'kind', 'slot'}:
                raise ValueError('Invalid rendered requirement identity')
            key = row['id']
            if type(key) is not str or key in seen or not re.fullmatch(r'Q1|G1|[PC](?:[1-9]|1[0-2])', key):
                raise ValueError('Invalid rendered requirement identity')
            seen.add(key)
            kind, slot = (('task', 'original_step_question') if key == 'Q1' else
                ('project_goal', 'project_goal') if key == 'G1' else
                ('project_criterion', f'project_acceptance_criteria[{int(key[1:])-1}]') if key.startswith('P') else
                ('criterion', f'phase_acceptance_criteria[{int(key[1:])-1}]'))
            if row['kind'] != kind or row['slot'] != slot:
                raise ValueError('Invalid rendered requirement slot')
        if 'Q1' not in seen:
            raise ValueError('Missing original task requirement')
        return 16384
    return 8192 if data.get('source_quotes') else 2048


@contextmanager
def fake_research_model(used):
    calls = []
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def complete(request: Request):
        assert request.headers.get("authorization") == "Bearer " + FAKE_KEY
        body = await request.json()
        assert body.get("model") == "offline-runtime-check"
        messages = body.get("messages", [])
        data, _ = model_prompt_data(messages)
        actual_cap = body.get("max_tokens", body.get("max_completion_tokens"))
        expected_cap = expected_output_cap(messages)
        if actual_cap != expected_cap:
            # Numeric test-contract diagnostics only; never print prompts,
            # user data, source bodies, tokens or credentials.
            print(json.dumps({'fake_provider_expected_cap': expected_cap,
                              'fake_provider_actual_cap': actual_cap}), flush=True)
        assert actual_cap == expected_cap
        assert sum(len((row["role"] + row["content"]).encode()) for row in messages) <= 24576
        calls.append(messages)
        return {"id": "offline-research-check", "object": "chat.completion", "created": 1,
                "model": "offline-runtime-check", "choices": [{"index": 0, "message": {
                    "role": "assistant", "content": fake_grounded_reply(messages)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    port = free_port(used)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                            access_log=False, log_level="critical", proxy_headers=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "Loopback fake provider did not start"
    try:
        yield port, calls
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        assert not thread.is_alive(), "Loopback fake provider failed to stop"


def wait_research_ready(process, origin, store):
    with httpx.Client(base_url=origin, timeout=1, trust_env=False) as browser:
        deadline = time.monotonic() + 110
        while time.monotonic() < deadline:
            assert process.poll() is None, "Research runtime exited during startup"
            try:
                if browser.get("/index.html").status_code == 200:
                    ready = True
                    for binding in store.list_worker_bindings():
                        response = browser.get(binding["upstream_url"] + PREFIX + "/capabilities",
                            headers={"Authorization": "Bearer " + binding["upstream_token"]})
                        ready = ready and response.status_code == 200
                        if response.status_code == 200:
                            caps = response.json()
                            assert caps["chat_ready"] is True and caps["research_core"] is True
                            assert caps["uploads"] is True and caps["automatic_learning"] is False
                    if ready:
                        return
            except httpx.TransportError:
                pass
            time.sleep(0.15)
    raise AssertionError("Research runtime did not become ready")


@contextmanager
def running(environment, temporary, origin, store, owned_ports):
    command = [sys.executable, "-m", "docthinker.server.invite_runtime"]
    process = subprocess.Popen(command, env=environment, cwd=temporary, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        wait_research_ready(process, origin, store)
        yield process
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                raise AssertionError("Research runtime did not terminate cleanly") from None
        deadline = time.monotonic() + 5
        while any(port_open(port) for port in owned_ports) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert all(not port_open(port) for port in owned_ports), "Owned research listeners remain running"
        assert process.returncode == 0, "Research runtime shutdown was not clean"


def client(origin):
    return httpx.Client(base_url=origin, headers={"Origin": origin}, timeout=30,
                        trust_env=False, follow_redirects=False)


def login(browser, username):
    response = browser.post("/api/v1/invite/login", json={"username": username, "password": PASSWORD})
    assert response.status_code == 200, "Synthetic account login failed"
    return {"Authorization": "Bearer " + response.json()["access_token"]}


def upload(browser, identity, sid, filename, content):
    response = browser.post(PREFIX + "/research/upload", headers=identity,
                            data={"session_id": sid}, files={"file": (filename, content, "application/octet-stream")})
    assert response.status_code == 200, f"Offline research upload failed ({response.status_code}): " + response.text[:600]
    return response.json()


def ready_document(browser, identity, sid, document_id):
    deadline = time.monotonic() + 35
    while time.monotonic() < deadline:
        response = browser.get(PREFIX + "/research/document", headers=identity,
                               params={"session_id": sid, "document_id": document_id})
        assert response.status_code == 200, "Research document disappeared"
        document = response.json()["document"]
        assert document["status"] != "failed", "Offline document extraction failed"
        if document["status"] == "ready":
            assert document["chunks"], "Ready document has no literal sources"
            return document
        time.sleep(0.1)
    raise AssertionError("Offline document extraction did not complete")


def completed_query(browser, identity, payload):
    response = browser.post(PREFIX + "/query", headers=identity, json=payload)
    assert response.status_code in {200, 202}, "Research question was rejected"
    deadline = time.monotonic() + 30
    while response.json().get("request_status") == "running" and time.monotonic() < deadline:
        time.sleep(0.1)
        response = browser.get(PREFIX + "/query/status", headers=identity,
                                params={"session_id": payload["session_id"], "request_id": payload["request_id"]})
    result = response.json()
    assert result.get("request_status") == "completed", "Research request did not complete"
    assert result.get("research_core") is True
    assert result["citation_validation"]["scope"] == "citation_syntax_only"
    assert result["citation_validation"]["entailment_verified"] is False
    return result


@contextmanager
def held_parser_slot(path):
    """Hold the actual deployment inode to test both worker subprocess gates."""
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def workbench_before_restart(browser, alice, bob, sid, other_sid, answer, calls):
    """Real gateway/worker persistence, with no extra synthetic model calls."""
    before = len(calls)
    for identity, project in ((alice, sid), (bob, sid), (alice, other_sid)):
        response = browser.get(PREFIX + "/workbench/brief", headers=identity, params={"session_id": project})
        assert response.status_code == 200, "Workbench brief route unavailable"
        assert response.json()["brief"]["revision"] == 0
    payload = {"session_id": sid, "expected_revision": 0,
               "goal": "Compare evidence for this offline project.", "output_format": "A reviewable note",
               "constraints": "Do not fill missing links with invented facts."}
    response = browser.post(PREFIX + "/workbench/brief", headers=alice, json=payload)
    assert response.status_code == 200 and response.json()["brief"]["revision"] == 1
    brief = response.json()["brief"]
    # A stale client must not silently overwrite the project goal.
    assert browser.post(PREFIX + "/workbench/brief", headers=alice, json={**payload, "goal": "Stale update"}).status_code == 409
    for identity, project in ((bob, sid), (alice, other_sid)):
        assert browser.get(PREFIX + "/workbench/brief", headers=identity,
                           params={"session_id": project}).json()["brief"]["revision"] == 0
    bookmark = {"session_id": sid, "answer_id": answer["answer_id"], "title": "Offline reviewed output"}
    for identity, project in ((bob, sid), (alice, other_sid)):
        assert browser.post(PREFIX + "/workbench/artifacts", headers=identity,
                            json={**bookmark, "session_id": project}).status_code == 404
    saved = browser.post(PREFIX + "/workbench/artifacts", headers=alice, json=bookmark)
    assert saved.status_code == 200
    artifact = saved.json()["artifact"]
    assert artifact["available"] is True and artifact["answer"]["sources"] == answer["sources"]
    assert browser.post(PREFIX + "/workbench/artifacts", headers=alice, json=bookmark).json()["artifact"] == artifact
    assert browser.get(PREFIX + "/workbench/artifacts", headers=bob, params={"session_id": sid}).json()["items"] == []
    assert len(calls) == before, "Workbench metadata or provenance reads invoked a model"
    return {"brief": brief, "artifact": artifact}


def phase_one(origin, invitations, calls, parser_lock):
    with client(origin) as browser:
        assert browser.get(PREFIX + "/research/documents", params={"session_id": "#00001"}).status_code == 401
        identities, sessions = [], []
        for invitation in invitations:
            registered = browser.post("/api/v1/invite/register", json={"username": invitation["username"],
                "invite_code": invitation["invite_code"], "password": PASSWORD})
            assert registered.status_code == 200, "Synthetic invitation registration failed"
            identity = login(browser, invitation["username"])
            identities.append(identity)
            created = browser.post(PREFIX + "/sessions", headers=identity, json={"title": "离线研究流程验证"})
            assert created.status_code == 200
            sessions.append(created.json()["session"]["id"])
            cap = browser.get(PREFIX + "/research/capabilities", headers=identity).json()
            assert cap["enabled"] is True and cap["uploads_allowed"] is True
        assert sessions == ["#00001", "#00001"], "User worker directories are not isolated"
        alice, bob = identities
        sid = sessions[0]
        with held_parser_slot(parser_lock):
            alice_text = upload(browser, alice, sid, "synthetic-energy.txt", TEXT.encode())
            duplicate = upload(browser, alice, sid, "duplicate-energy.txt", TEXT.encode())
            bob_text = upload(browser, bob, sid, "synthetic-energy.txt", TEXT.encode())
            time.sleep(0.25)
            for identity, document in ((alice, alice_text), (bob, bob_text)):
                queued = browser.get(PREFIX + "/research/document", headers=identity,
                    params={"session_id": sid, "document_id": document["document"]["id"]}).json()["document"]
                assert queued["status"] in {"queued", "processing"} and queued["chunk_count"] == 0
                assert not queued["chunks"], "A worker bypassed the shared parser slot"
            assert not calls, "Waiting for an offline parser billed a model"
        assert duplicate["deduplicated"] is True
        assert duplicate["document"]["id"] == alice_text["document"]["id"]
        assert alice_text["document"]["id"] != bob_text["document"]["id"]
        another = browser.post(PREFIX + "/sessions", headers=alice, json={"title": "第二个隔离项目"}).json()["session"]["id"]
        assert another == "#00002"
        second_text = upload(browser, alice, another, "same-energy.txt", TEXT.encode())
        assert second_text["deduplicated"] is False
        pdf = upload(browser, alice, sid, "synthetic-paper.pdf", pdf_with_text())
        alice_doc = ready_document(browser, alice, sid, alice_text["document"]["id"])
        bob_doc = ready_document(browser, bob, sid, bob_text["document"]["id"])
        other_doc = ready_document(browser, alice, another, second_text["document"]["id"])
        pdf_doc = ready_document(browser, alice, sid, pdf["document"]["id"])
        assert pdf_doc["chunks"][0]["page"] == 1
        assert alice_doc["chunks"][0]["content"] == bob_doc["chunks"][0]["content"] == other_doc["chunks"][0]["content"] == TEXT
        assert not calls, "Startup/auth/upload/extraction unexpectedly called a model"
        alice_source = alice_doc["chunks"][0]["id"]
        assert browser.get(PREFIX + "/research/source", headers=bob, params={"session_id": sid, "chunk_id": alice_source}).status_code == 404
        preference = browser.post(PREFIX + "/research/memory", headers=alice, json={
            "session_id": sid, "kind": "preference", "text": PREFERENCE_FIRST}).json()["memory"]
        finding = browser.post(PREFIX + "/research/memory", headers=alice, json={
            "session_id": sid, "kind": "finding", "text": FINDING, "source_ids": [alice_source]}).json()["memory"]
        assert not browser.get(PREFIX + "/research/memory", headers=bob, params={"session_id": sid}).json()["items"]
        rid = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
        results, payloads = [], []
        for index, identity in enumerate(identities):
            payload = {"session_id": sid, "question": QUESTION + (" alice-only" if index == 0 else " bob-only"),
                       "request_id": rid, "use_memory": True, "use_conversation_context": False,
                       "use_llm_cache": False, "use_self_evolution": False}
            answer = completed_query(browser, identity, payload)
            assert answer["sources"] and answer["citation_validation"]["valid"] is True
            assert answer["retrieval_gaps"] and answer["used_memory_count"] == (2 if index == 0 else 0)
            replay = completed_query(browser, identity, payload)
            assert replay["answer_id"] == answer["answer_id"] and replay["sources"] == answer["sources"]
            assert replay["used_memory_count"] == answer["used_memory_count"]
            results.append(answer)
            payloads.append(payload)
        assert len(calls) == 2, "Replayed research query invoked another provider call"
        assert PREFERENCE_FIRST in str(calls[0]) and PREFERENCE_FIRST not in str(calls[1])
        assert "bob-only" not in str(calls[0]) and "alice-only" not in str(calls[1])
        workbench = workbench_before_restart(browser, alice, bob, sid, another, results[0], calls)
        corrected = browser.post(PREFIX + "/research/memory", headers=alice, json={
            "session_id": sid, "kind": "preference", "text": PREFERENCE_UPDATED,
            "memory_id": preference["id"], "expected_revision": 1})
        assert corrected.status_code == 200 and corrected.json()["memory"]["revision"] == 2
        assert len(calls) == 2, "Confirmed memory mutation invoked a model"
        return {"sid": sid, "second_sid": another, "payloads": payloads, "results": results,
                "alice_document": alice_doc, "bob_document": bob_doc, "other_document": other_doc,
                "pdf_document": pdf_doc, "preference": corrected.json()["memory"], "finding": finding,
                "workbench": workbench}


def assert_restarted_answer(before, replay, history, *, memory_revoked):
    """A restart replays request identity, never restores revoked premises."""
    assert replay['answer_id'] == before['answer_id'], 'Restart changed an accepted request identity'
    assert len(history) == 2 and history[-1]['id'] == before['answer_id']
    current = history[-1]
    assert current['sources'] == replay['sources']
    assert current['content'] == replay['answer']
    if memory_revoked:
        assert replay['sources'] == [] and current['sources'] == []
        assert replay['used_memory_count'] == current['used_memory_count'] == 0
        for answer in (replay, current):
            assert answer['citation_validation']['valid'] is False
            assert 'memory_context_revoked' in {row['code'] for row in answer['citation_validation']['warnings']}
            assert 'PRIVATE_RESEARCH_FIXTURE_QUOTE' not in json.dumps(answer)
            assert before['answer'] not in json.dumps(answer, ensure_ascii=False), 'A withdrawn answer body leaked through replay/history'
        assert replay['answer'] != before['answer'], 'Corrected memory restored an old generated answer'
    else:
        assert replay['sources'] == before['sources']
        assert replay['answer'] == before['answer']
        assert replay['used_memory_count'] == current['used_memory_count'] == before['used_memory_count']
        assert replay['citation_validation'] == before['citation_validation']
    # Selection metadata describes the historical request, not permission to
    # reuse its old revisions. Current usage is independently zero on revoke.
    assert replay['memory_selection'] == current['memory_selection'] == before['memory_selection']
    assert [gap['code'] for gap in before['retrieval_gaps']] == [gap['code'] for gap in current['retrieval_gaps']]
    assert all(set(gap) <= {'code', 'count', 'message'} for gap in current['retrieval_gaps'])


def phase_two(origin, saved, calls):
    with client(origin) as browser:
        alice, bob = login(browser, "preview"), login(browser, "bobby")
        sid = saved["sid"]
        for index, (identity, payload, before) in enumerate(zip((alice, bob), saved["payloads"], saved["results"])):
            replay = completed_query(browser, identity, payload)
            history = browser.get(PREFIX + "/history", headers=identity, params={"session_id": sid}).json()["history"]
            assert_restarted_answer(before, replay, history, memory_revoked=index == 0)
        assert len(calls) == 2, "Worker restart replay unexpectedly invoked a provider"
        assert browser.get(PREFIX + "/workbench/brief", headers=alice,
                           params={"session_id": sid}).json()["brief"] == saved["workbench"]["brief"]
        catalog = browser.get(PREFIX + "/workbench/artifacts", headers=alice,
                              params={"session_id": sid}).json()["items"]
        assert len(catalog) == 1 and "answer" not in catalog[0], "Listing should not hydrate every saved answer"
        artifact = browser.get(PREFIX + "/workbench/artifact", headers=alice,
                               params={"session_id": sid, "answer_id": saved["results"][0]["answer_id"]}).json()["artifact"]
        assert artifact['available'] is False and artifact['answer'] is None
        assert artifact['unavailable_reason'] == 'memory_context_revoked'
        assert 'PRIVATE_RESEARCH_FIXTURE_QUOTE' not in json.dumps(artifact)
        for key, value in saved['workbench']['artifact'].items():
            if key not in {'available', 'answer'}:
                assert artifact[key] == value, 'Restart lost the saved bookmark identity'
        memories = browser.get(PREFIX + "/research/memory", headers=alice, params={"session_id": sid}).json()["items"]
        assert next(row for row in memories if row["id"] == saved["preference"]["id"])["text"] == PREFERENCE_UPDATED
        assert len(browser.get(PREFIX + "/sessions", headers=alice).json()["sessions"]) == 2
        removed = browser.post(PREFIX + "/research/documents/delete", headers=alice,
            json={"session_id": sid, "document_id": saved["alice_document"]["id"]})
        assert removed.status_code == 200 and removed.json()["invalidated_memory_count"] == 1
        revoked = completed_query(browser, alice, saved["payloads"][0])
        assert revoked["unavailable_source_count"] == 1 and revoked["citation_validation"]["valid"] is False
        assert "PRIVATE_RESEARCH_FIXTURE_QUOTE" not in json.dumps(revoked)
        history = browser.get(PREFIX + "/history", headers=alice, params={"session_id": sid}).json()["history"]
        assert "PRIVATE_RESEARCH_FIXTURE_QUOTE" not in json.dumps(history)
        artifact = browser.get(PREFIX + "/workbench/artifact", headers=alice,
                               params={"session_id": sid, "answer_id": saved["results"][0]["answer_id"]}).json()["artifact"]
        assert artifact["available"] is False and artifact["answer"] is None
        assert "PRIVATE_RESEARCH_FIXTURE_QUOTE" not in json.dumps(artifact)
        assert browser.get(PREFIX + "/workbench/artifacts", headers=bob, params={"session_id": sid}).json()["items"] == []
        for _ in range(2):
            removed_bookmark = browser.post(PREFIX + "/workbench/artifacts/delete", headers=alice,
                json={"session_id": sid, "answer_id": saved["results"][0]["answer_id"]})
            assert removed_bookmark.status_code == 200 and removed_bookmark.json()["removed"] is True
        assert browser.get(PREFIX + "/workbench/artifacts", headers=alice, params={"session_id": sid}).json()["items"] == []
        for identity, project, document in ((bob, sid, saved["bob_document"]),
                                            (alice, saved["second_sid"], saved["other_document"])):
            source = browser.get(PREFIX + "/research/source", headers=identity,
                params={"session_id": project, "chunk_id": document["chunks"][0]["id"]})
            assert source.status_code == 200 and source.json()["chunk"]["content"] == TEXT
        memories = browser.get(PREFIX + "/research/memory", headers=alice, params={"session_id": sid}).json()["items"]
        placeholder = next(row for row in memories if row["id"] == saved["finding"]["id"])
        assert placeholder["status"] == "invalidated" and placeholder["redacted"] is True and not placeholder["text"]
        revoked_memory = browser.get(PREFIX + "/research/memory/history", headers=alice,
            params={"session_id": sid, "memory_id": saved["finding"]["id"]}).json()
        assert revoked_memory["items"] and all(row.get("redacted") is True and not row["text"]
                                               for row in revoked_memory["items"])
        exported = browser.get(PREFIX + "/research/export", headers=alice, params={"session_id": sid}).json()["markdown"]
        assert PREFERENCE_UPDATED in exported and FINDING not in exported
        assert len(calls) == 2, "Source revocation/export unexpectedly called a model"
        fresh = {"session_id": sid, "question": "How does solar investment increase renewable supply?",
                 "request_id": str(uuid.uuid4()), "use_memory": True, "use_conversation_context": False,
                 "use_llm_cache": False, "use_self_evolution": False}
        answered = completed_query(browser, alice, fresh)
        assert answered["sources"][0]["page"] == 1
        assert len(calls) == 3 and PREFERENCE_UPDATED in str(calls[-1])
        assert PREFERENCE_FIRST not in str(calls[-1]) and FINDING not in str(calls[-1])
        assert "PRIVATE_RESEARCH_FIXTURE_QUOTE" not in str(calls[-1])


def create_project(browser, identity, title):
    response = browser.post(PREFIX + "/sessions", headers=identity, json={"title": title})
    assert response.status_code == 200, "Synthetic reasoning project creation failed"
    return response.json()["session"]["id"]


def chat_payload(sid, question, **options):
    return {"session_id": sid, "question": question,
            "request_id": str(uuid.uuid4()), "use_memory": False,
            "use_conversation_context": False, "use_llm_cache": False,
            "use_self_evolution": False, **options}


def literal_reasoning(result, documents):
    """Audit against uploaded bytes, not only response labels or HTTP success."""
    originals = {chunk["id"]: chunk["content"] for document in documents for chunk in document["chunks"]}
    refs = {row["id"]: row for row in result["sources"]}
    assert result["reasoning"]["mode"] == "context_assist", "Backend analysis became an answer mode"
    assert result["reasoning"]["trace"]["model_calls"] == 0
    assert result["claim_audit"]["entailment_verified"] is False
    for ref in refs.values():
        assert ref["id"] in originals, "Reasoning crossed source ownership"
        assert originals[ref["id"]][ref["excerpt_start"]:ref["excerpt_end"]] == ref["content"]
    for claim in result["reasoning"].get("claims", []):
        source = claim["source_ids"][0]
        assert source in refs
        assert originals[source][claim["excerpt_start"]:claim["excerpt_end"]] == claim["quote"]
    for conclusion in result["reasoning"]["conclusions"]:
        assert set(conclusion["source_ids"]) <= refs.keys(), "A proof lacks a live premise source"


def phase_reasoning(origin, calls):
    """Natural writing calls a model once; optional evidence operators never do."""
    before = len(calls)
    with client(origin) as browser:
        alice, bob = login(browser, "preview"), login(browser, "bobby")
        deduction_sid = create_project(browser, alice, "写作参考 · 前提与完整链路")
        deduction_documents = []
        for filename, text in (("observation.txt", "Observation: grid ready."),
                               ("rules.txt", "If grid ready then supply grows. If supply grows then security improves.")):
            uploaded = upload(browser, alice, deduction_sid, filename, text.encode())
            deduction_documents.append(ready_document(browser, alice, deduction_sid, uploaded["document"]["id"]))
        deduction_request = chat_payload(deduction_sid, "请参考资料写一段分析，解释 security improves 为什么成立。")
        deduction = completed_query(browser, alice, deduction_request)
        literal_reasoning(deduction, deduction_documents)
        assert deduction["reasoning"]["trace"].get("skipped_reason") != "operator_disabled", (
            "This acceptance requires DOCTHINKER_RESEARCH_CONTEXT_ASSIST=true in the worker environment")
        assert any(row["kind"] == "deduced" and row["depth"] == 2 and "security improves" in row["text"]
                   for row in deduction["reasoning"]["conclusions"]), "Complete two-step deduction was not recovered"
        assert {row["document_id"] for row in deduction["sources"]} == {row["id"] for row in deduction_documents}
        assert len(calls) == before + 1 and "写作参考" in deduction["answer"]
        _, support = model_prompt_data(calls[-1])
        assert any(row["type"] == "deduced" and "security improves" in row["line"] and row["steps"]
                   for row in support["supporting_material"])

        analogy_sid = create_project(browser, alice, "参考案例 · 云服务与能源供应")
        analogy_documents = []
        for filename, text in (("cloud.txt", "Redundancy reduces outage risk. Outage risk reduces availability. Redundancy increases availability under low load."),
                               ("energy.txt", "Supplier diversity reduces concentration risk. Concentration risk reduces energy security.")):
            uploaded = upload(browser, alice, analogy_sid, filename, text.encode())
            analogy_documents.append(ready_document(browser, alice, analogy_sid, uploaded["document"]["id"]))
        analogy_request = chat_payload(analogy_sid, "请借鉴 redundancy outage risk availability 的案例，帮我写一段讨论。")
        analogy = completed_query(browser, alice, analogy_request)
        literal_reasoning(analogy, analogy_documents)
        assert any(row["kind"] == "analogy_hypothesis" and row["mappings"] and row["differences"]
                   and row["candidate_inferences"] for row in analogy["reasoning"]["conclusions"]), "Structural analogy lost its available transfer hypothesis"
        assert all(candidate["epistemic_status"] == "unverified_transfer_hypothesis"
                   for row in analogy["reasoning"]["conclusions"] for candidate in row["candidate_inferences"])
        assert "supplier diversity" in analogy["answer"] and "low load" in analogy["answer"]
        assert len(calls) == before + 2
        _, support = model_prompt_data(calls[-1])
        assert any(row["type"] == "analogy_hypothesis" and row["mappings"] and row["hypotheses"]
                   for row in support["supporting_material"])

        induction_sid = create_project(browser, bob, "文献梳理 · 支持案例与反例")
        induction_documents = []
        for region, claim in (("north", "Solar reduces imports."), ("south", "Solar reduces imports."),
                              ("winter", "Solar does not reduce imports.")):
            uploaded = upload(browser, bob, induction_sid, region + ".txt", ("Case: " + region + "\n" + claim).encode())
            induction_documents.append(ready_document(browser, bob, induction_sid, uploaded["document"]["id"]))
        induction_request = chat_payload(induction_sid, "请综合资料写一段 solar imports 关系的综述，保留反例。")
        induction = completed_query(browser, bob, induction_request)
        literal_reasoning(induction, induction_documents)
        assert any(row["kind"] == "inductive_candidate" and row["support_count"] == 2 and row["counterexample_count"] == 1
                   for row in induction["reasoning"]["conclusions"]), "Counterexample disappeared from induction"
        assert len(calls) == before + 3
        _, support = model_prompt_data(calls[-1])
        assert any(row["support_groups"] == 2 and row["counterexample_groups"] == 1
                   for row in support["supporting_material"] if row["type"] == "inductive_candidate")

        missing_sid = create_project(browser, bob, "论证检查 · 缺失前提时停下")
        missing_upload = upload(browser, bob, missing_sid, "missing-condition.txt",
            b"Observation: grid ready. If grid ready and imports decrease then security improves.")
        ready_document(browser, bob, missing_sid, missing_upload["document"]["id"])
        missing = completed_query(browser, bob, chat_payload(missing_sid, "请分析 security improves 的论证，说明缺少什么前提。"))
        assert missing["reasoning"]["conclusions"] == []
        assert "missing_premises" in {row["code"] for row in missing["reasoning"]["gaps"]}
        assert len(calls) == before + 4
        _, support = model_prompt_data(calls[-1])
        assert not support["supporting_material"] and any("缺少原文前提" in row for row in support["limits"])

        # A literal extract is also zero-model, and quotation checking must bind
        # to original bytes rather than merely accepting an [S1] label.
        extract_request = chat_payload(deduction_sid, "grid ready", reasoning_mode="evidence")
        extract = completed_query(browser, alice, extract_request)
        assert extract["sources"] and extract["claim_audit"]["entailment_verified"] is False
        assert any(row["status"] == "exact_quote" for row in extract["claim_audit"]["claims"])
        assert len(calls) == before + 4, "Legacy literal extract unexpectedly requested a model"

        source = deduction_documents[0]["chunks"][0]["id"]
        first = browser.post(PREFIX + "/research/memory", headers=alice, json={"session_id": deduction_sid,
            "kind": "decision", "text": "Portfolio old decision: study grid readiness first.", "valid_from": 1600000000,
            "source_ids": [source]})
        assert first.status_code == 200
        memory = first.json()["memory"]
        second = browser.post(PREFIX + "/research/memory", headers=alice, json={"session_id": deduction_sid,
            "kind": "decision", "text": "Portfolio new decision: check supply mechanisms first.", "valid_from": 1700000000,
            "source_ids": [source], "memory_id": memory["id"], "expected_revision": 1})
        assert second.status_code == 200 and second.json()["memory"]["revision"] == 2
        for timestamp, expected in ((1650000000, "Portfolio old decision"), (1750000000, "Portfolio new decision")):
            page = browser.get(PREFIX + "/research/memory", headers=alice,
                params={"session_id": deduction_sid, "query": "Portfolio", "as_of": timestamp, "limit": 20}).json()
            assert len(page["items"]) == 1 and expected in page["items"][0]["text"]
        temporal_request = chat_payload(deduction_sid, "请参考 Portfolio 决定，写一段 grid ready 和 supply 的分析。",
                                             use_memory=True, memory_as_of=1650000000)
        temporal_answer = completed_query(browser, alice, temporal_request)
        assert temporal_answer["memory_as_of"] == 1650000000 and temporal_answer["used_memory_count"] == 1
        literal_reasoning(temporal_answer, deduction_documents)
        assert all("Portfolio" not in row["quote"] for row in temporal_answer["reasoning"].get("claims", [])), "A confirmed memory became a logical axiom"
        assert len(calls) == before + 5
        assert "Portfolio old decision" in str(calls[-1]) and "Portfolio new decision" not in str(calls[-1])
        temporal_context_request = chat_payload(deduction_sid, "What did we decide about Portfolio?",
                                                     use_memory=True, memory_as_of=1650000000)
        temporal_context = completed_query(browser, alice, temporal_context_request)
        assert temporal_context["memory_as_of"] == 1650000000 and temporal_context["used_memory_count"] == 1
        assert len(calls) == before + 6
        assert "Portfolio old decision" in str(calls[-1]) and "Portfolio new decision" not in str(calls[-1])
        before = len(calls)
        versions = browser.get(PREFIX + "/research/memory/history", headers=alice,
            params={"session_id": deduction_sid, "memory_id": memory["id"]}).json()
        assert [row["revision"] for row in versions["items"]] == [2, 1]
        archived = browser.post(PREFIX + "/research/memory/archive", headers=alice,
            json={"session_id": deduction_sid, "memory_id": memory["id"], "expected_revision": 2, "archived": True})
        assert archived.status_code == 200 and archived.json()["memory"]["archived"] is True
        hidden = browser.get(PREFIX + "/research/memory", headers=alice, params={"session_id": deduction_sid}).json()
        assert hidden["items"] == []
        visible = browser.get(PREFIX + "/research/memory", headers=alice,
            params={"session_id": deduction_sid, "include_archived": "true"}).json()
        assert visible["items"][0]["id"] == memory["id"] and visible["items"][0]["archived"] is True
        restored = browser.post(PREFIX + "/research/memory/archive", headers=alice,
            json={"session_id": deduction_sid, "memory_id": memory["id"], "expected_revision": 3, "archived": False})
        assert restored.status_code == 200 and restored.json()["memory"]["revision"] == 4
        for index in range(22):
            response = browser.post(PREFIX + "/research/memory", headers=alice,
                json={"session_id": deduction_sid, "kind": "decision", "text": f"Offline pagination record {index:02d}."})
            assert response.status_code == 200
        first_page = browser.get(PREFIX + "/research/memory", headers=alice,
            params={"session_id": deduction_sid, "limit": 20}).json()
        assert len(first_page["items"]) == 20 and first_page["total"] == 23 and first_page["next_cursor"]
        second_page = browser.get(PREFIX + "/research/memory", headers=alice,
            params={"session_id": deduction_sid, "limit": 20, "cursor": first_page["next_cursor"]}).json()
        assert len(second_page["items"]) == 3 and not second_page["next_cursor"]
        assert not ({row["id"] for row in first_page["items"]} & {row["id"] for row in second_page["items"]})
        assert browser.get(PREFIX + "/research/memory/history", headers=bob,
            params={"session_id": "#00001", "memory_id": memory["id"]}).status_code == 404
        assert len(calls) == before, "Offline reasoning, search or memory timeline called a provider"
        return {"deduction": ("preview", deduction_request, deduction, deduction_documents),
                "analogy": ("preview", analogy_request, analogy, analogy_documents),
                "induction": ("bobby", induction_request, induction, induction_documents),
                "temporal": ("preview", temporal_request, temporal_answer, deduction_documents),
                "temporal_context": ("preview", temporal_context_request, temporal_context, deduction_documents),
                "literal": (extract_request, extract),
                "memory_id": memory["id"]}


def phase_reasoning_restart(origin, saved, calls):
    before = len(calls)
    with client(origin) as browser:
        identities = {name: login(browser, name) for name in ("preview", "bobby")}
        for name in ("deduction", "analogy", "induction", "temporal", "temporal_context"):
            username, payload, original, documents = saved[name]
            replay = completed_query(browser, identities[username], payload)
            assert replay["answer_id"] == original["answer_id"]
            assert replay["reasoning"]["conclusions"] == original["reasoning"]["conclusions"]
            assert replay["memory_as_of"] == original["memory_as_of"]
            literal_reasoning(replay, documents)
            history = browser.get(PREFIX + "/history", headers=identities[username],
                params={"session_id": payload["session_id"]}).json()["history"]
            persisted = next(row for row in history if row.get("id") == original["answer_id"])
            assert persisted["reasoning"]["conclusions"] == original["reasoning"]["conclusions"]
            assert persisted["claim_audit"] == replay["claim_audit"]
        username, payload, original, documents = saved["deduction"]
        identity = identities[username]
        deleted = browser.post(PREFIX + "/research/documents/delete", headers=identity,
            json={"session_id": payload["session_id"], "document_id": documents[0]["id"]})
        assert deleted.status_code == 200
        revoked = completed_query(browser, identity, payload)
        assert revoked["reasoning"]["conclusions"] == []
        assert revoked["reasoning"]["status"] == "source_unavailable"
        revoked_sources = {row["id"] for row in documents[0]["chunks"]}
        assert all(not (set(row["source_ids"]) & revoked_sources) for row in revoked["reasoning"]["claims"])
        assert all(ref["source_id"] not in revoked_sources for row in revoked["claim_audit"]["claims"] for ref in row["source_refs"])
        assert "Observation: grid ready." not in json.dumps(revoked)
        assert "source_revoked" in {row["code"] for row in revoked["citation_validation"]["warnings"]}
        literal_request, original_literal = saved["literal"]
        literal_replay = completed_query(browser, identity, literal_request)
        assert literal_replay["answer_id"] == original_literal["answer_id"]
        assert "Observation: grid ready." not in literal_replay["answer"]
        assert "If grid ready then supply grows." in literal_replay["answer"]
        answers = browser.get(PREFIX + "/history", headers=identity,
            params={"session_id": payload["session_id"]}).json()["history"]
        literal_history = next(row for row in answers if row.get("id") == original_literal["answer_id"])
        assert literal_history["content"] == literal_replay["answer"]
        assert "Observation: grid ready." not in json.dumps(literal_history)
        history = browser.get(PREFIX + "/research/memory/history", headers=identity,
            params={"session_id": payload["session_id"], "memory_id": saved["memory_id"]}).json()
        assert history["items"] and all(row["redacted"] is True and row["text"] == "" for row in history["items"])
        assert len(calls) == before, "Restart, replay or source revocation invoked a provider"


def next_synthetic_upload_day(store, temporary):
    """Separate acceptance fixtures without changing the five-upload policy.

    The old upload/extraction scenario and new reasoning scenario each fit the
    real daily allowance. Only this disposable registry's existing upload rows
    are dated yesterday; model-call, request-journal and other quotas remain
    untouched. This is test setup, never a production quota control.
    """
    assert temporary.name.startswith("docthinker-research-check-") and store.root == temporary / "registry"
    with store._connect() as db:
        day = int(store.clock() // 86400)
        db.execute("UPDATE usage SET day=? WHERE day=? AND kind='upload'", (day - 1, day))


def serve_until_stopped(origin):
    stop = threading.Event()
    previous = {number: signal.signal(number, lambda *_: stop.set()) for number in (signal.SIGINT, signal.SIGTERM)}
    print(json.dumps({"preview_origin": origin, "username": "preview", "password": PASSWORD,
                      "mode": "loopback_fake_provider_only"}), flush=True)
    try:
        while not stop.wait(1):
            pass
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="Keep disposable loopback preview alive after acceptance")
    parser.add_argument("--workbench-only", action="store_true",
                        help="Test workbench persistence/isolation/revocation without the optional reasoning scenarios")
    args = parser.parse_args(argv)
    build_dir = (ROOT / "clients/mobile/dist/build/h5").resolve()
    assert build_dir.is_dir(), "Build mobile H5 before running acceptance"
    with tempfile.TemporaryDirectory(prefix="docthinker-research-check-") as directory:
        temporary = Path(directory).resolve()
        home, tmp = temporary / "home", temporary / "tmp"
        home.mkdir(mode=0o700)
        tmp.mkdir(mode=0o700)
        store = InviteStore.initialize(temporary / "registry")
        used = set()
        invitations = (store.add_user("preview", free_port(used), allow_uploads=True),
                       store.add_user("bobby", free_port(used), allow_uploads=True))
        gateway_port = free_port(used)
        owned_ports = [gateway_port, *(int(row["upstream_url"].rsplit(":", 1)[1]) for row in store.list_worker_bindings())]
        origin = f"http://127.0.0.1:{gateway_port}"
        secret = temporary / "fake-model-key"
        descriptor = os.open(secret, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(FAKE_KEY + "\n")
        with fake_research_model(used) as (provider_port, calls):
            environment = {"PATH": os.defpath, "LANG": "en_US.UTF-8", "HOME": str(home), "TMPDIR": str(tmp),
                "PYTHONPATH": str(ROOT), "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1", "PYTHON_DOTENV_DISABLED": "1",
                SHARING_ACK_ENV: SHARING_ACK_VALUE, "DOCTHINKER_INVITE_DATA_DIR": str(store.root),
                "DOCTHINKER_INVITE_ORIGIN": origin, "DOCTHINKER_INVITE_BIND_HOST": "127.0.0.1",
                "DOCTHINKER_INVITE_PORT": str(gateway_port), "DOCTHINKER_INVITE_WEB_DIR": str(build_dir),
                "DOCTHINKER_RESEARCH_ENABLED": "true", "LLM_BINDING_API_KEY_FILE": str(secret),
                # Exercise the optional operators explicitly; the product's
                # default remains off. Keep this grammar/transport acceptance
                # independent of optional local model assets or downloads.
                "DOCTHINKER_RESEARCH_CONTEXT_ASSIST": "true",
                "DOCTHINKER_RESEARCH_SEMANTIC": "off",
                "LLM_VLM_HOST": f"http://127.0.0.1:{provider_port}/v1", "LLM_MODEL": "offline-runtime-check"}
            command = [sys.executable, "-m", "docthinker.server.invite_runtime"]
            checked = subprocess.run(command + ["--check"], env=environment, cwd=temporary,
                                     capture_output=True, timeout=20)
            assert checked.returncode == 0, ("Research runtime configuration check failed: "
                + (checked.stdout + checked.stderr).decode("utf-8", errors="replace")[-4000:].replace(FAKE_KEY, "[synthetic-key]"))
            assert not calls and not any(port_open(port) for port in owned_ports)
            with running(environment, temporary, origin, store, owned_ports):
                saved = phase_one(origin, invitations, calls, store.root / "workers" / "research-parser.lock")
            with running(environment, temporary, origin, store, owned_ports):
                phase_two(origin, saved, calls)
                if not args.workbench_only:
                    next_synthetic_upload_day(store, temporary)
                    reasoning_saved = phase_reasoning(origin, calls)
            with running(environment, temporary, origin, store, owned_ports):
                if not args.workbench_only:
                    phase_reasoning_restart(origin, reasoning_saved, calls)
                health_environment = {key: value for key, value in environment.items()
                                      if key not in {SHARING_ACK_ENV, "LLM_BINDING_API_KEY_FILE", "LLM_VLM_HOST", "LLM_MODEL"}}
                health_environment["LLM_BINDING_API_KEY_FILE"] = str(temporary / "nonexistent-key")
                healthy = subprocess.run(command + ["--healthcheck"], env=health_environment,
                                         cwd=temporary, capture_output=True, timeout=12)
                assert healthy.returncode == 0, "Credential-free research healthcheck failed"
                expected_calls = 3 if args.workbench_only else 9
                assert len(calls) == expected_calls, "A replay, status read or healthcheck made an extra model call"
                if args.workbench_only:
                    print("PASS: real two-worker gateway; per-user and per-session briefs; stale revision rejection; "
                          "answer-reference bookmarks; restart identity persistence with corrected-memory withdrawal; live source-revocation suppression; "
                          "idempotent unbookmark; no model calls for workbench actions; three fake answer requests only. "
                          "Not a real-model quality or Internet deployment test.", flush=True)
                else:
                    print("PASS: real two-worker research gateway; shared single parser slot; offline TXT/PDF extraction; scoped dedup; grounded citations; "
                          "confirmed memory correction; indexed temporal paging/archive/version history; natural writing with complete premise chains and explicit missing-premise limits; "
                          "background analogy with conditions and induction with counterexample; zero-model legacy literal evidence; restart replay and "
                          "source-revocation suppression; historical memory prompt isolation; nine fake model requests total, one per normal answer.", flush=True)
                if args.serve:
                    serve_until_stopped(origin)


if __name__ == "__main__":
    main()
