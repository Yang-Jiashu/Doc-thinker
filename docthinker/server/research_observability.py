"""Bounded SELECT-only session projection, not GraphCore or an inference engine.

Stored-row totals are distinct from live validity checks on the returned window.
No model, embedding, migration, usage update, or document-body scan belongs here.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import sqlite3
import time

from .personal_events import PersonalEvents
from .research_store import ResearchStoreError, _session

MEMORY_STATES = ("active", "archived", "invalidated", "deleted")
DOCUMENT_STATES = ("queued", "processing", "ready", "failed", "deleted")
MAX_GRAPH_EDGES = 400
MAX_NODE_SOURCES = 20
MAX_RESPONSE_BYTES = 1024 * 1024


class ObservabilityUnavailable(RuntimeError):
    """Inspection did not finish; this must not be reported as zero or revoked."""


def envelope(session_id, limit, *, profile, sampled_at):
    return {"schema_version": 1, "profile": profile, "session_id": session_id,
            "read_only": True, "sampled_at": sampled_at,
            "availability": {key: True for key in
                ("documents", "confirmed_memories", "events", "event_relations", "graph")},
            "counts": {"documents": {"total": 0, "by_status": dict.fromkeys(DOCUMENT_STATES, 0)},
                       "chunks": {"total": 0, "available": 0},
                       "memories": {"total": 0, "by_status": dict.fromkeys(MEMORY_STATES, 0),
                                    "versions": 0, "source_links": 0},
                       "events": {"total": 0}, "relations": {"total": 0}},
            "window": {"limit": limit, "counts_scope": "stored_session_rows",
                       "validity_scope": "returned_window", "returned": {},
                       "limited": dict.fromkeys(("documents", "memories", "events", "relations", "sources"), False),
                       "states": {}, "max_response_bytes": MAX_RESPONSE_BYTES, "byte_limited": False},
            "documents": [], "memories": [], "events": [], "relations": [], "sources": [],
            "graph": {"kind": "provenance_projection", "nodes": [], "edges": [],
                      "window_limited": False, "returned": {"nodes": 0, "edges": 0},
                      "totals": {"nodes": None, "edges": None}, "causality_verified": False}}


def _count(db, table, sid):
    # Table names are private constants, never caller input.
    return db.execute("SELECT count(*) FROM " + table + " WHERE session_id=?", (sid,)).fetchone()[0]


def _states(db, table, column, sid, allowed):
    result = dict.fromkeys(allowed, 0)
    for row in db.execute("SELECT " + column + ",count(*) FROM " + table +
                          " WHERE session_id=? GROUP BY " + column, (sid,)):
        if row[0] not in result:
            raise ObservabilityUnavailable("Unknown stored state")
        result[row[0]] = row[1]
    return result


def _memory_view(store, db, row, at, seen):
    value = store._memory(row)
    if row["status"] not in {"deleted", "invalidated"}:
        live = store._memory_sources_live(db, row, at=at, seen=seen)
        if seen.get("inspection_unavailable"):
            raise ObservabilityUnavailable("Memory dependencies could not be inspected")
        if not live:
            value.update(status="invalidated", text="", source_chunk_ids=[], source_answer_id=None, redacted=True)
    return value


def _origins_live(owner, db, sid, payload, at, seen):
    try:
        owner._origins(db, sid, payload["origins"], maximum=32, at=at, seen=seen)
        live = True
    except ResearchStoreError:
        live = False
    if seen.get("inspection_unavailable"):
        raise ObservabilityUnavailable("Event dependencies could not be inspected")
    return live


def _event_view(owner, db, row, at, seen):
    payload = json.loads(row["payload"])
    result = {"id": row["event_id"], "session_id": row["session_id"], "revision": row["revision"],
        "recorded_at": row["recorded_at"], "semantic_verified": False, "causality_verified": False,
        "execution_authorized": False}
    if not _origins_live(owner, db, row["session_id"], payload, at, seen):
        return {**result, "status": "source_unavailable", "redacted": True}
    return {**result, **payload, "status": "active", "redacted": False}


def _relation_view(owner, db, row, at, seen):
    result = {"id": row["relation_id"], "session_id": row["session_id"], "recorded_at": row["recorded_at"],
        "semantic_verified": False, "causality_verified": False, "execution_authorized": False}
    left = owner._required_event(db, row["session_id"], row["source_id"])
    right = owner._required_event(db, row["session_id"], row["target_id"])
    if left["revision"] != row["source_revision"] or right["revision"] != row["target_revision"]:
        return {**result, "status": "stale_endpoint", "redacted": True}
    payload = json.loads(row["payload"])
    if (not _origins_live(owner, db, row["session_id"], payload, at, seen)
            or _event_view(owner, db, left, at, seen)["status"] != "active"
            or _event_view(owner, db, right, at, seen)["status"] != "active"):
        return {**result, "status": "source_unavailable", "redacted": True}
    return {**result, **payload, "source_id": row["source_id"], "source_revision": row["source_revision"],
        "target_id": row["target_id"], "target_revision": row["target_revision"],
        "relation": row["relation"], "status": "active", "redacted": False}


def _node(kind, identifier, label, *, description="", status="active", redacted=False,
          source_ids=(), revision=None):
    sources = list(dict.fromkeys(source_ids))
    value = {"id": kind + ":" + identifier, "record_id": identifier, "type": kind,
             "label": label[:160], "description": description[:500], "status": status,
             "redacted": redacted, "source_ids": sources[:MAX_NODE_SOURCES],
             "label_truncated": len(label) > 160, "description_truncated": len(description) > 500,
             "sources_limited": len(sources) > MAX_NODE_SOURCES}
    if revision is not None:
        value["revision"] = revision
    return value


def _finish(result):
    for key in ("documents", "memories", "events", "relations", "sources"):
        result["window"]["returned"][key] = len(result[key])
    graph = result["graph"]
    graph["returned"] = {"nodes": len(graph["nodes"]), "edges": len(graph["edges"])}
    graph["window_limited"] = graph["window_limited"] or any(result["window"]["limited"].values())
    return result


def _bounded_result(result):
    """Keep complete records, shrinking a large display window rather than text.

    At most seven halvings for the 100-row cap; no re-query, re-ranking or model.
    Stored totals do not change, and omitted rows are explicitly marked limited.
    """
    for _ in range(8):
        _finish(result)
        if len(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()) <= MAX_RESPONSE_BYTES:
            return result
        result["window"]["byte_limited"] = True
        reduced = False
        for key in ("documents", "memories", "events", "relations", "sources"):
            if len(result[key]) > 1:
                result[key] = result[key][:(len(result[key]) + 1) // 2]
                result["window"]["limited"][key] = True
                reduced = True
        if not reduced:
            break
        for key in ("memories", "events", "relations"):
            result["window"]["states"][key] = dict(Counter(item["status"] for item in result[key]))
        result["graph"]["nodes"], result["graph"]["edges"] = [], []
        _graph(result)
    raise ObservabilityUnavailable("Observation exceeds response budget")


def project_observability(store, session_id, *, limit=20):
    """Read one owner/project snapshot; full totals, bounded records and graph.

    Content is revalidated using the same memory/event readers as their public
    APIs. Only source metadata is read here; opening a passage uses /source.
    Counts describe stored records, not an unbounded recursive semantic audit.
    """
    _session(session_id)
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ResearchStoreError("Invalid observation window")
    sampled_at = store._timestamp(store.clock())
    result = envelope(session_id, limit, profile="bounded_research", sampled_at=sampled_at)
    deadline, ticks, interrupted = time.monotonic() + 2.0, 0, False

    def budget():
        nonlocal ticks, interrupted
        ticks += 1
        interrupted = ticks > 4000 or time.monotonic() > deadline
        return int(interrupted)

    try:
        with store._read(timeout=1) as db:
            if db is None:
                return _finish(result)
            db.set_progress_handler(budget, 1000)
            # A read transaction keeps counts, validity and nodes on one snapshot.
            db.execute("BEGIN")
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            result["counts"]["documents"]["by_status"] = _states(
                db, "research_documents", "state", session_id, DOCUMENT_STATES)
            result["counts"]["documents"]["total"] = sum(result["counts"]["documents"]["by_status"].values())
            result["counts"]["memories"]["by_status"] = _states(
                db, "research_memories", "status", session_id, MEMORY_STATES)
            result["counts"]["memories"]["total"] = sum(result["counts"]["memories"]["by_status"].values())
            result["counts"]["memories"]["versions"] = _count(db, "research_memory_versions", session_id)
            result["counts"]["memories"]["source_links"] = (
                _count(db, "research_memory_sources", session_id) if "research_memory_sources" in tables else None)
            result["counts"]["chunks"]["total"] = _count(db, "research_chunks", session_id)
            result["counts"]["chunks"]["available"] = db.execute(
                "SELECT count(*) FROM research_chunks c JOIN research_documents d "
                "ON d.document_id=c.document_id AND d.session_id=c.session_id "
                "WHERE c.session_id=? AND d.state='ready'", (session_id,)).fetchone()[0]

            for row in db.execute("SELECT document_id,filename,state,boot_id,chunk_count,byte_size "
                    "FROM research_documents WHERE session_id=? ORDER BY updated_at DESC,document_id DESC LIMIT ?",
                    (session_id, limit)).fetchall():
                deleted = row["state"] == "deleted"
                status = "queued" if row["state"] == "processing" and row["boot_id"] != store.boot_id else row["state"]
                result["documents"].append({"id": row["document_id"], "filename": "" if deleted else row["filename"],
                    "status": status, "stored_status": row["state"], "chunk_count": 0 if deleted else row["chunk_count"],
                    "size_bytes": 0 if deleted else row["byte_size"], "redacted": deleted})
            memory_rows = db.execute("SELECT * FROM research_memories WHERE session_id=? "
                "ORDER BY updated_at DESC,memory_id DESC LIMIT ?", (session_id, limit)).fetchall()
            requested_sources = []
            seen = {"active": set(), "memo": {}}
            for row in memory_rows:
                memory = _memory_view(store, db, row, sampled_at, seen)
                sources = memory["source_chunk_ids"]
                result["memories"].append({key: memory[key] for key in
                    ("id", "kind", "text", "status", "revision", "archived", "valid_from", "valid_until", "recorded_at")}
                    | {"redacted": memory.get("redacted", False), "source_ids": sources,
                       "source_count": len(sources), "confirmed_by_user": True,
                       "applicable_now": memory["status"] == "active" and memory["valid_from"] <= sampled_at
                           and (memory["valid_until"] is None or sampled_at < memory["valid_until"])})
                requested_sources.extend(sources)

            owner = PersonalEvents(store)
            if "personal_events" in tables:
                result["counts"]["events"]["total"] = _count(db, "personal_events", session_id)
                result["events"] = [_event_view(owner, db, row, sampled_at, seen) for row in db.execute(
                    "SELECT * FROM personal_events WHERE session_id=? ORDER BY recorded_at DESC,event_id DESC LIMIT ?",
                    (session_id, limit)).fetchall()]
                result["counts"]["relations"]["total"] = _count(db, "personal_event_relations", session_id)
                result["relations"] = [_relation_view(owner, db, row, sampled_at, seen) for row in db.execute(
                    "SELECT * FROM personal_event_relations WHERE session_id=? ORDER BY recorded_at DESC,relation_id DESC LIMIT ?",
                    (session_id, limit)).fetchall()]
            # Resolve at most 100 x 16 explicit origins, reusing repeated IDs.
            memory_source_cache = {}
            for event in result["events"]:
                if event.get("redacted"):
                    continue
                event_sources = []
                for origin in event.get("origins", []):
                    if origin["kind"] == "source":
                        event_sources.append(origin["source_id"])
                    else:
                        # Current checked memory origins may lie outside the UI window.
                        key = (origin["memory_id"], origin["revision"])
                        if key not in memory_source_cache:
                            row = db.execute("SELECT source_chunk_ids FROM research_memories "
                                "WHERE session_id=? AND memory_id=? AND revision=? AND status='active'",
                                (session_id, *key)).fetchone()
                            memory_source_cache[key] = json.loads(row[0]) if row is not None else []
                        event_sources.extend(memory_source_cache[key])
                event_sources = list(dict.fromkeys(event_sources))
                event["source_ids"] = event_sources[:MAX_NODE_SOURCES]
                event["sources_limited"] = len(event_sources) > MAX_NODE_SOURCES
                requested_sources.extend(event_sources)
            for relation in result["relations"]:
                if not relation.get("redacted"):
                    requested_sources.extend(origin["source_id"] for origin in relation.get("origins", [])
                                             if origin["kind"] == "source")
            requested_sources = list(dict.fromkeys(requested_sources))
            source_ids = requested_sources[:limit]
            columns = ("SELECT c.chunk_id,c.document_id,c.ordinal,c.page,length(c.text) AS text_chars,d.filename "
                "FROM research_chunks c JOIN research_documents d ON d.document_id=c.document_id "
                "AND d.session_id=c.session_id WHERE c.session_id=? AND d.state='ready' ")
            source_rows = []
            if source_ids:
                source_rows = db.execute(columns + "AND c.chunk_id IN (" + ",".join("?" for _ in source_ids) +
                    ") ORDER BY c.document_id,c.ordinal,c.chunk_id", [session_id, *source_ids]).fetchall()
            if not source_ids:
                source_rows = db.execute(columns + "ORDER BY c.document_id,c.ordinal,c.chunk_id LIMIT ?",
                    (session_id, limit)).fetchall()
            result["sources"] = [{"id": row["chunk_id"], "document_id": row["document_id"],
                "filename": row["filename"], "ordinal": row["ordinal"], "page": row["page"],
                "text_chars": row["text_chars"]} for row in source_rows]
            if interrupted:
                raise ObservabilityUnavailable("Inspection budget exhausted")
            result["window"]["limited"] = {key: result["counts"][key]["total"] > len(result[key])
                for key in ("documents", "memories", "events", "relations")}
            result["window"]["limited"]["sources"] = (
                result["counts"]["chunks"]["available"] > len(source_rows) or len(requested_sources) > limit)
            for key in ("memories", "events", "relations"):
                result["window"]["states"][key] = dict(Counter(item["status"] for item in result[key]))
            _graph(result)
            return _bounded_result(result)
    except (sqlite3.Error, ValueError, TypeError, KeyError, OverflowError) as error:
        # Missing/corrupt dependency tables or a bounded-inspection failure are
        # unavailable, never an empty graph or a proven source withdrawal.
        raise ObservabilityUnavailable("Session observation unavailable") from error


def _graph(result):
    graph, nodes = result["graph"], {}
    source_set = {source["id"] for source in result["sources"]}
    for document in result["documents"]:
        sources = [source["id"] for source in result["sources"] if source["document_id"] == document["id"]]
        node = _node("document", document["id"], document["filename"] or "已移除资料",
            status=document["status"], redacted=document["redacted"], source_ids=sources)
        nodes[node["id"]] = node
    for source in result["sources"]:
        node = _node("source", source["id"], source["filename"],
            description="原文片段 · 按需读取" + (" · 第 " + str(source["page"]) + " 页" if source["page"] else ""),
            source_ids=[source["id"]])
        nodes[node["id"]] = node
    for memory in result["memories"]:
        node = _node("memory", memory["id"], memory["text"] or "内容已失效",
            description=memory["text"], status=memory["status"], redacted=memory["redacted"],
            source_ids=[value for value in memory["source_ids"] if value in source_set], revision=memory["revision"])
        node["applicable_now"] = memory["applicable_now"]
        nodes[node["id"]] = node
    for event in result["events"]:
        sources = list(event.get("source_ids", []))
        for origin in event.get("origins", []):
            if origin["kind"] == "memory":
                memory = nodes.get("memory:" + origin["memory_id"])
                if memory:
                    sources.extend(memory["source_ids"])
        node = _node("event", event["id"], event.get("action", "事件依据不可用"),
            description=event.get("action", ""), status=event["status"], redacted=event["redacted"],
            source_ids=[value for value in sources if value in source_set], revision=event["revision"])
        nodes[node["id"]] = node

    def edge(source, target, kind, *, identifier=None, relation="", description="", origins=(), conditions=()):
        if source not in nodes or target not in nodes:
            graph["window_limited"] = True
            return
        if len(graph["edges"]) >= MAX_GRAPH_EDGES:
            graph["window_limited"] = True
            return
        identifier = identifier or hashlib.sha256((kind + source + target).encode()).hexdigest()
        source_ids = list(dict.fromkeys(nodes[source]["source_ids"] + nodes[target]["source_ids"] +
            [origin["source_id"] for origin in origins if origin["kind"] == "source" and origin["source_id"] in source_set]))
        graph["edges"].append({"id": identifier, "source": source, "target": target, "type": kind,
            "relation": relation or kind, "description": description[:500], "is_discovered": False,
            "source_ids": source_ids[:MAX_NODE_SOURCES], "sources_limited": len(source_ids) > MAX_NODE_SOURCES,
            "origins": list(origins), "conditions": list(conditions), "causality_verified": False,
            "source_scope": "endpoint_and_explicit_origins_not_entailment"})

    for source in result["sources"]:
        edge("document:" + source["document_id"], "source:" + source["id"], "contains_source", description="资料包含此片段")
    for memory in result["memories"]:
        for source in memory["source_ids"]:
            edge("source:" + source, "memory:" + memory["id"], "memory_source", description="用户确认记忆所关联的原文")
    for event in result["events"]:
        for origin in event.get("origins", []):
            kind, identifier = ("source", origin["source_id"]) if origin["kind"] == "source" else ("memory", origin["memory_id"])
            edge(kind + ":" + identifier, "event:" + event["id"], "event_origin", description="事件的显式来源锚点",
                 origins=[origin], conditions=event.get("conditions", []))
    for relation in result["relations"]:
        if not relation["redacted"]:
            edge("event:" + relation["source_id"], "event:" + relation["target_id"], "event_relation",
                identifier=relation["id"], relation=relation["relation"], description="用户显式关联 · 非因果证明",
                origins=relation.get("origins", []), conditions=relation.get("conditions", []))
    graph["nodes"] = list(nodes.values())
