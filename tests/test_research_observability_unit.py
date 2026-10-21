"""Authenticated real SQLite projections; no providers, growth or migrations."""
import hashlib
import json
import sqlite3
import socket

import pytest

from tests.test_research_mobile_unit import (
    mobile, mobile_client, upload_ready, gateway_factory, gateway_client, PREFIX, SID, OTHER_SID,
)
from docthinker.server.personal_events import PersonalEvents
from docthinker.server.research_observability import project_observability
from docthinker.server.research_store import ResearchStore

PATH = PREFIX + "/research/observability"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Observation must not call a network/provider")
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


async def observe(browser, sid=SID, **params):
    response = await browser.get(PATH, params={"session_id": sid, **params})
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    return response.json()


async def test_empty_read_is_real_zero_without_creating_schema_or_calling_models(mobile):
    async with mobile_client(mobile) as browser:
        result = await observe(browser)
    assert result["profile"] == "bounded_research" and result["session_id"] == SID
    assert result["read_only"] is True and result["sampled_at"] > 0
    assert result["counts"]["documents"]["total"] == 0
    assert result["counts"]["memories"]["versions"] == 0
    assert result["graph"]["nodes"] == result["graph"]["edges"] == []
    assert result["graph"]["totals"] == {"nodes": None, "edges": None}
    assert not mobile.service.store.path.exists()
    mobile.model.assert_not_awaited()


async def test_optin_auth_session_and_page_validation_precede_reads(mobile, monkeypatch):
    async with mobile_client(mobile, token=None) as browser:
        assert (await browser.get(PATH, params={"session_id": "../private", "limit": "bad"})).status_code == 401
    async with mobile_client(mobile) as browser:
        for limit in ("0", "101", "1.2", "bad"):
            assert (await browser.get(PATH, params={"session_id": SID, "limit": limit})).status_code == 422
        for sid, status in (("../private", 400), ("#99999", 404)):
            assert (await browser.get(PATH, params={"session_id": sid})).status_code == status
        monkeypatch.setattr(mobile.state, "research_service", None)
        assert (await browser.get(PATH, params={"session_id": SID})).status_code == 404
    assert not mobile.service.store.path.exists()


async def test_real_counts_provenance_and_session_isolation_without_loading_source_text(mobile):
    store = mobile.service.store
    async with mobile_client(mobile) as browser:
        doc = await upload_ready(mobile, browser, text="Original evidence body only visible through the source endpoint.")
        await upload_ready(mobile, browser, sid=OTHER_SID, text="FOREIGN_PROJECT_PRIVATE", filename="foreign.txt")
        source = store.source_chunks(SID)[0]
        memory = store.create_memory(SID, "finding", "User-confirmed finding", source_chunk_ids=[source["id"]])
        store.create_memory(OTHER_SID, "preference", "FOREIGN_MEMORY_PRIVATE")
        events = PersonalEvents(store)
        origin = {"kind": "memory", "memory_id": memory["id"], "revision": 1}
        a = events.record_event(SID, event_type="decision", action="First explicit event", origins=[origin])
        b = events.record_event(SID, event_type="change", action="Second explicit event", origins=[origin])
        relation = events.relate(SID, a["id"], b["id"], "reported_cause", conditions=["Only a user assertion"])
        before = store.path.read_bytes()
        result = await observe(browser)
        assert store.path.read_bytes() == before, "Read mutated SQLite or recall usage"
        other = await observe(browser, OTHER_SID)
    assert result["counts"]["documents"]["total"] == 1
    assert result["counts"]["chunks"] == {"total": 1, "available": 1}
    assert result["counts"]["memories"]["versions"] == 1
    assert result["counts"]["memories"]["source_links"] == 1
    assert result["counts"]["events"]["total"] == 2
    assert result["counts"]["relations"]["total"] == 1
    assert result["memories"][0]["source_count"] == 1
    assert "FOREIGN" not in str(result) and "First explicit event" not in str(other)
    assert "Original evidence body" not in str(result), "Projection must not include source bodies"
    graph = result["graph"]
    assert {item["type"] for item in graph["nodes"]} == {"document", "source", "memory", "event"}
    assert {item["type"] for item in graph["edges"]} == {"contains_source", "memory_source", "event_origin", "event_relation"}
    edge = next(item for item in graph["edges"] if item["id"] == relation["id"])
    assert edge["relation"] == "reported_cause" and edge["causality_verified"] is False
    assert not graph["window_limited"] and graph["causality_verified"] is False
    assert all(item["is_discovered"] is False for item in graph["edges"])
    mobile.model.assert_not_awaited()


async def test_source_withdrawal_redacts_memory_events_relations_and_document_labels(mobile):
    secret = "SECRET-WITHDRAWN-CONTENT"
    async with mobile_client(mobile) as browser:
        doc = await upload_ready(mobile, browser, filename=secret + ".txt", text="Source supporting explicit records.")
        source = mobile.service.store.source_chunks(SID)[0]
        memory = mobile.service.store.create_memory(SID, "finding", secret, source_chunk_ids=[source["id"]])
        events = PersonalEvents(mobile.service.store)
        origin = {"kind": "source", "source_id": source["id"], "span": [0, len(source["text"])],
                  "sha256": hashlib.sha256(source["text"].encode()).hexdigest()}
        a = events.record_event(SID, event_type="decision", action=secret, origins=[origin], conditions=[secret])
        b = events.record_event(SID, event_type="change", action=secret, origins=[origin])
        events.relate(SID, a["id"], b["id"], "hypothesis", conditions=[secret])
        mobile.service.store.delete_document(SID, doc)
        response = await browser.get(PATH, params={"session_id": SID})
    assert response.status_code == 200 and secret not in response.text
    result = response.json()
    assert result["counts"]["documents"]["by_status"]["deleted"] == 1
    assert result["counts"]["memories"]["by_status"]["invalidated"] == 1
    assert result["memories"][0]["redacted"] and result["memories"][0]["source_ids"] == []
    assert all(item["redacted"] for item in result["events"] + result["relations"])
    assert result["sources"] == [] and result["graph"]["edges"] == []
    mobile.model.assert_not_awaited()


async def test_stale_event_endpoints_and_memory_revisions_are_not_live_graph_edges(mobile):
    store, events = mobile.service.store, PersonalEvents(mobile.service.store)
    memory = store.create_memory(SID, "decision", "Original plan")
    origin = {"kind": "memory", "memory_id": memory["id"], "revision": 1}
    a = events.record_event(SID, event_type="decision", action="OLD_PRIVATE_ACTION", origins=[origin])
    b = events.record_event(SID, event_type="decision", action="OTHER_OLD_PRIVATE_ACTION", origins=[origin])
    events.relate(SID, a["id"], b["id"], "depends_on")
    events.revise_event(SID, a["id"], expected_revision=1, event_type="decision", action="Revised event", origins=[origin])
    async with mobile_client(mobile) as browser:
        revised = await observe(browser)
        assert revised["relations"][0]["status"] == "stale_endpoint"
        assert not any(edge["type"] == "event_relation" for edge in revised["graph"]["edges"])
        store.update_memory(SID, memory["id"], expected_revision=1, text="New plan")
        result = await observe(browser)
    assert all(item["redacted"] for item in result["events"])
    assert "PRIVATE_ACTION" not in str(result)
    assert result["graph"]["edges"] == []


async def test_window_totals_and_version_counts_are_not_confused(mobile):
    store = mobile.service.store
    memories = [store.create_memory(SID, "preference", f"Preference {index}") for index in range(4)]
    store.update_memory(SID, memories[0]["id"], expected_revision=1, text="Revised preference")
    store.archive_memory(SID, memories[1]["id"], expected_revision=1, archived=True)
    async with mobile_client(mobile) as browser:
        result = await observe(browser, limit=1)
    assert result["counts"]["memories"]["total"] == 4
    assert result["counts"]["memories"]["versions"] == 6
    assert result["counts"]["memories"]["by_status"]["archived"] == 1
    assert len(result["memories"]) == result["window"]["returned"]["memories"] == 1
    assert result["window"]["limited"]["memories"] and result["graph"]["window_limited"]
    assert result["window"]["counts_scope"] == "stored_session_rows"


def test_same_session_identifier_in_two_real_worker_stores_never_crosses(tmp_path):
    a, b = tmp_path / "alice", tmp_path / "bobby"
    a.mkdir(mode=0o700)
    b.mkdir(mode=0o700)
    alice, bobby = ResearchStore(a), ResearchStore(b)
    alice.create_memory(SID, "decision", "ALICE_PRIVATE_ONLY")
    bobby.create_memory(SID, "decision", "BOBBY_PRIVATE_ONLY")
    left, right = project_observability(alice, SID), project_observability(bobby, SID)
    assert left["session_id"] == right["session_id"] == SID
    assert "ALICE_PRIVATE_ONLY" in str(left) and "BOBBY_PRIVATE_ONLY" not in str(left)
    assert "BOBBY_PRIVATE_ONLY" in str(right) and "ALICE_PRIVATE_ONLY" not in str(right)


def test_projection_edge_cap_is_explicit_and_sources_stay_available_on_demand(tmp_path):
    store = ResearchStore(tmp_path)
    doc = store.accept_document(SID, "sources.txt", b"Five independent stored source windows")
    store.claim_document(SID, doc["id"])
    store.complete_document(SID, doc["id"], [{"text": f"Original source window {index}"} for index in range(5)])
    sources = [row["id"] for row in store.source_chunks(SID)]
    for index in range(85):
        store.create_memory(SID, "finding", f"Explicit confirmed record {index}", source_chunk_ids=sources)
    result = project_observability(store, SID, limit=100)
    assert len(result["graph"]["edges"]) == 400
    assert result["graph"]["window_limited"] is True
    assert len(result["memories"]) == 85 and result["counts"]["memories"]["total"] == 85
    assert all(set(node["source_ids"]).issubset(set(sources)) for node in result["graph"]["nodes"])
    assert all("Original source window" not in str(node) for node in result["graph"]["nodes"])


def test_restart_document_status_is_current_but_raw_counts_are_explicit(tmp_path):
    store = ResearchStore(tmp_path, boot_id="first")
    document = store.accept_document(SID, "queued.txt", b"Pending local extraction")
    store.claim_document(SID, document["id"])
    before = store.path.read_bytes()
    result = project_observability(ResearchStore(tmp_path, boot_id="second"), SID)
    assert result["documents"][0]["status"] == "queued"
    assert result["documents"][0]["stored_status"] == "processing"
    assert result["counts"]["documents"]["by_status"]["processing"] == 1
    assert store.path.read_bytes() == before


def test_large_event_windows_reduce_whole_records_to_an_explicit_byte_budget(tmp_path):
    store = ResearchStore(tmp_path)
    memory = store.create_memory(SID, "decision", "A user-confirmed anchor")
    origins = [{"kind": "memory", "memory_id": memory["id"], "revision": 1}]
    conditions = [chr(65 + index) + ":" + "条" * 254 for index in range(16)]
    events = PersonalEvents(store)
    for index in range(75):
        events.record_event(SID, event_type="observation", action="据" * 1000,
                            origins=origins, conditions=conditions)
    result = project_observability(store, SID, limit=100)
    assert result["counts"]["events"]["total"] == 75
    assert 0 < len(result["events"]) < 75
    assert result["window"]["byte_limited"] is True
    assert result["window"]["limited"]["events"] and result["graph"]["window_limited"]
    assert len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode()) <= 1024 * 1024
    assert all(item["action"] == "据" * 1000 and item["conditions"] == conditions for item in result["events"])


async def test_dependency_inspection_outage_is_503_not_zero_or_revocation(mobile):
    store = mobile.service.store
    store.create_memory(SID, "decision", "PRIVATE_OLD_NOTE")
    with sqlite3.connect(store.path) as db:
        db.execute("DROP TABLE research_memory_sources")
    async with mobile_client(mobile) as browser:
        response = await browser.get(PATH, params={"session_id": SID})
    assert response.status_code == 503
    assert "PRIVATE_OLD_NOTE" not in response.text and str(store.path) not in response.text


async def test_bounded_memory_inspection_unknown_is_not_redacted_as_withdrawn(mobile, monkeypatch):
    store = mobile.service.store
    store.create_memory(SID, "decision", "PRIVATE_NOTE")
    def unavailable(db, row, *, at=None, seen=None):
        seen["inspection_unavailable"] = True
        return False
    monkeypatch.setattr(store, "_memory_sources_live", unavailable)
    async with mobile_client(mobile) as browser:
        response = await browser.get(PATH, params={"session_id": SID})
    assert response.status_code == 503 and "PRIVATE_NOTE" not in response.text


async def test_bounded_event_inspection_unknown_is_not_reported_as_source_revocation(mobile, monkeypatch):
    import docthinker.server.personal_events as module
    store = mobile.service.store
    memory = store.create_memory(SID, "decision", "Current plan")
    PersonalEvents(store).record_event(SID, event_type="decision", action="PRIVATE_EVENT",
        origins=[{"kind": "memory", "memory_id": memory["id"], "revision": 1}])
    def unavailable(db, sid, bindings, *, now, seen=None):
        seen["inspection_unavailable"] = True
        return False
    monkeypatch.setattr(module, "_bindings_live", unavailable)
    async with mobile_client(mobile) as browser:
        response = await browser.get(PATH, params={"session_id": SID})
    assert response.status_code == 503 and "PRIVATE_EVENT" not in response.text


async def test_legacy_summary_metadata_and_expanded_flags_are_truthful(mobile, monkeypatch):
    from unittest.mock import AsyncMock
    import docthinker.server.routers.graph as graph_module
    monkeypatch.setattr(graph_module, "get_graph_data", AsyncMock(return_value={
        "nodes": [{"id": "original", "is_expanded": False}, {"id": "extra", "is_expanded": "1"},
                  {"id": "false-string", "is_expanded": "false"}],
        "edges": [{"source": "original", "target": "extra", "is_expanded": True}],
        "metadata": {"total_nodes": 250, "total_edges": 400, "nodes_returned": 3, "edges_returned": 1,
                     "truncated": True, "graph_file": "/private/owner/graph.graphml"}}))
    async with mobile_client(mobile) as browser:
        response = await browser.get(PREFIX + "/graph", params={"session_id": SID})
    assert response.status_code == 200 and "/private/owner" not in response.text
    result = response.json()
    assert [node["is_discovered"] for node in result["nodes"]] == [False, True, False]
    assert result["edges"][0]["is_discovered"] is True
    assert result["metadata"]["scope"] == "summary" and result["metadata"]["window_limited"] is True
    assert result["metadata"]["total_nodes"] == 250
    assert result["metadata"]["total_edges_basis"] == "raw_graph_records"


async def test_gateway_optin_identity_and_unknown_legacy_dashboard_stay_separate(gateway_factory):
    fixture = gateway_factory
    disabled = fixture.create(False)
    async with gateway_client(disabled, fixture.auth["alice"]) as browser:
        assert (await browser.get(PATH, params={"session_id": SID})).status_code == 403
    assert fixture.calls == []
    enabled = fixture.create(True)
    for name, port in (("alice", 8101), ("bobby", 8102)):
        async with gateway_client(enabled, fixture.auth[name]) as browser:
            response = await browser.get(PATH, params={"session_id": SID, "limit": 2})
            assert response.status_code == 200 and response.json()["worker_port"] == port
            assert (await browser.get(PREFIX + "/memory/dashboard", params={"session_id": SID})).status_code == 404
            assert (await browser.get(PATH, params={"session_id": SID, "user_id": "other"})).status_code == 400
    assert len(fixture.calls) == 2
    with fixture.store._connect() as db:
        assert db.execute("SELECT count(*) FROM usage").fetchone()[0] == 0


async def test_legacy_adapter_counts_only_exact_active_session_and_unknowns_are_null(mobile, monkeypatch, tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE long_horizon_memories(id TEXT,session_id TEXT,scope TEXT,scope_key TEXT,status TEXT,"
                   "kind TEXT,summary TEXT,created_at REAL,last_seen_at REAL)")
        for identifier, sid, scope, scope_key in (("own", SID, "session", "session:" + SID),
                ("other", OTHER_SID, "session", "session:" + OTHER_SID),
                ("shared", SID, "user", "user:owner"), ("mismatch", SID, "session", "session:" + OTHER_SID)):
            db.execute("INSERT INTO long_horizon_memories VALUES(?,?,?,?,?,?,?,?,?)",
                (identifier, sid, scope, scope_key, "active", "summary", identifier + "_PRIVATE", 100, 100))
    monkeypatch.setattr(mobile.api, "_memory_database_path", lambda: path)
    before = path.read_bytes()
    async with mobile_client(mobile) as browser:
        response = await browser.get(PREFIX + "/memory/dashboard", params={"session_id": SID})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["profile"] == "legacy_session_memory" and result["counts"]["memories"]["total"] == 1
        assert result["counts"]["documents"]["total"] is None
        assert result["counts"]["memories"]["versions"] is None
        assert result["availability"]["confirmed_memories"] is False
        assert [item["id"] for item in result["memories"]] == ["own"]
        assert "other_PRIVATE" not in response.text and "shared_PRIVATE" not in response.text
        monkeypatch.setenv("DOCTHINKER_INVITE_WORKER_ID", "isolated-test-worker")
        assert (await browser.get(PREFIX + "/memory/dashboard", params={"session_id": SID})).status_code == 404
    assert path.read_bytes() == before
    mobile.model.assert_not_awaited()
