#!/usr/bin/env python3
"""Reproducible synthetic session/history benchmark, never model latency.

Builds a private TemporaryDirectory with real SQLite and invented messages,
document/graph entries, relations and legacy-compatible talk.json. Reports all
five timing samples and complete output equality without printing any content.
No real database, credentials, provider, network or persisted benchmark output
is accessed. OS page-cache state is not controlled; this is not cold-disk I/O.
Full mirror rewriting is intentionally retained and reported, not hidden.

Run from the full development checkout:
  PYTHON_DOTENV_DISABLED=1 python scripts/benchmark_session_reads.py
Use --messages/--other to change fixture size; repetitions stay fixed at five.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import sys
import tempfile
import time


os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["DOCTHINKER_RESEARCH_SEMANTIC"] = "off"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Optional dependency notices belong to stderr; stdout remains one JSON object.
with redirect_stdout(sys.stderr):
    from docthinker.session_manager import SessionManager


REPEATS = 5
MAX_FIXTURE_TEXT_BYTES = 12_000_000


@contextmanager
def _offline():
    """Fail this local benchmark if ordinary Python networking is attempted."""
    originals = (socket.socket.connect, socket.socket.connect_ex,
                 socket.create_connection, socket.getaddrinfo)
    counts = {"network_attempts": 0}

    def denied(*args, **kwargs):
        counts["network_attempts"] += 1
        raise AssertionError("Synthetic session benchmark must not access the network")

    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.create_connection = denied
    socket.getaddrinfo = denied
    try:
        yield counts
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo = originals


def _summary(samples):
    ordered = sorted(samples)
    return {"samples_ms": [round(value, 3) for value in samples],
            "median_ms": round(statistics.median(samples), 3),
            "p95_ms": round(ordered[max(0, math.ceil(.95 * len(ordered)) - 1)], 3),
            "p95_method": "nearest_rank_observed_max_with_five_samples_not_tail_guarantee",
            "sample_count": len(samples)}


def _old_history(manager, session_id):
    """Frozen pre-round-2 projection; current unchanged SQLite converters.

    A differential performance reference, not an independent semantic oracle.
    """
    history = []
    for entry in manager.kb_storage.query_entries("session_" + manager._canonical_session_id(session_id), ""):
        if entry.entry_type in ("question", "answer"):
            history.append({
                "id": entry.id, "role": entry.metadata.get("role", "user"), "content": entry.content,
                "created_at": entry.created_at.isoformat() if entry.created_at else "",
                "timestamp": entry.created_at.timestamp() if entry.created_at else 0.0,
                **({"research_core": True, "research_provenance_required": True}
                   if entry.metadata.get("research_provenance_required") is True else {}),
            })
    history.sort(key=lambda row: row["created_at"])
    return history


def _old_session(manager, session_id):
    canonical = manager._canonical_session_id(session_id)
    kb = manager.kb_storage.get_knowledge_base("session_" + canonical)
    if not kb and canonical != str(session_id).strip():
        kb = manager.kb_storage.get_knowledge_base("session_" + str(session_id).strip())
    if not kb:
        return None
    sid = str((kb.metadata or {}).get("session_id", canonical))
    merged = {**manager._build_default_paths(sid), **(kb.metadata or {})}
    manager._ensure_session_dirs(merged)
    return {
        "id": sid, "instance_id": "kb-" + str(kb.storage_id),
        "title": merged.get("title", kb.name), "created_at": kb.created_at.isoformat(),
        **{key: merged.get(key) for key in (
            "path", "data_dir", "content_dir", "talk_dir", "code_dir", "knowledge_dir", "talk_file")},
        "metadata": merged,
    }


def _seed(manager, sid, *, messages, other, long_messages=False):
    stamp = datetime(2025, 7, 2, 12, 0, tzinfo=timezone(timedelta(hours=8)))
    rows = []
    text_bytes = 0
    conn = manager.kb_storage._get_connection()
    try:
        kb_id = conn.execute("SELECT id FROM knowledge_bases WHERE name = ?", ("session_" + sid,)).fetchone()[0]
        for index in range(messages + other):
            if index < messages:
                kind = "question" if index % 2 == 0 else "answer"
                metadata = {"role": "user" if index % 2 == 0 else "assistant",
                            "extra_nested": {"list": [index, False, None]},
                            "research_provenance_required": True if index % 5 == 0 else "true"}
                created_at = stamp + timedelta(seconds=index % 7)
                content = ("Synthetic memory content " + "X" * 4090 if long_messages
                           else "Message 演绎\\\"\n" + str(index))
            else:
                kind = ("document", "entity", "relationship", "generic")[index % 4]
                metadata = {"filename": f"unused-{index}.pdf", "graph": {"weights": [0.1, 0.9]}}
                created_at = stamp
                content = "UNUSED_PRIVATE_SYNTHETIC_BODY_" * 200
            text_bytes += len(content.encode("utf-8"))
            if text_bytes > MAX_FIXTURE_TEXT_BYTES:
                raise ValueError("Synthetic fixture exceeds its 12,000,000-text-byte bound")
            rows.append((f"entry-{index}", kb_id, content, kind, json.dumps(metadata),
                         created_at.isoformat(), created_at.isoformat()))
        conn.executemany("INSERT INTO knowledge_entries (id,kb_id,content,type,metadata,created_at,updated_at) "
                         "VALUES (?,?,?,?,?,?,?)", rows)
        conn.executemany("INSERT INTO knowledge_relations (source_id,target_id,relation_type) VALUES (?,?,?)",
                         [(row[0], rows[(index + 1) % len(rows)][0], "related-to")
                          for index, row in enumerate(rows)])
        conn.commit()
    finally:
        conn.close()
    return text_bytes


def _new_manager(root):
    manager = SessionManager(str(root / "_system"), str(root / "data"), migrate_legacy=False)
    return manager, manager.create_session("Synthetic performance fixture")["id"]


def _query_counts(manager, function):
    statements = []
    original = manager.kb_storage._get_connection

    def connect():
        conn = original()
        conn.set_trace_callback(statements.append)
        return conn

    manager.kb_storage._get_connection = connect
    try:
        function()
    finally:
        manager.kb_storage._get_connection = original
    selects = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
    return {"selects": len(selects),
            "relation_selects": sum("knowledge_relations" in sql.lower() for sql in selects)}


def benchmark_session_reads(*, messages=1000, other=1000):
    with tempfile.TemporaryDirectory(prefix="docthinker-session-reads-synthetic-") as directory:
        manager, sid = _new_manager(Path(directory))
        text_bytes = _seed(manager, sid, messages=messages, other=other)
        expected_session, expected_history = _old_session(manager, sid), _old_history(manager, sid)
        functions = {
            "baseline_session_metadata": (lambda: _old_session(manager, sid), expected_session),
            "current_session_metadata": (lambda: manager.get_session(sid), expected_session),
            "baseline_history": (lambda: _old_history(manager, sid), expected_history),
            "current_history": (lambda: manager.get_history(sid), expected_history),
        }
        counts = {key: _query_counts(manager, function) for key, (function, _) in functions.items()}
        samples = {key: [] for key in functions}
        # Alternate old/new order between samples to reduce simple order bias.
        for repeat in range(REPEATS):
            order = list(functions) if repeat % 2 == 0 else list(reversed(functions))
            for key in order:
                function, expected = functions[key]
                started = time.perf_counter()
                actual = function()
                elapsed = (time.perf_counter() - started) * 1000
                if actual != expected:
                    raise AssertionError("Complete session/history output changed")
                samples[key].append(elapsed)
        return {
            "fixture": {"messages": messages, "unused_document_graph_entries": other,
                        "relations": messages + other, "text_bytes": text_bytes},
            "sql": counts, "timings": {key: _summary(values) for key, values in samples.items()},
            "full_output_equal": True, "query_cache": False, "history_truncated": False,
            "scope": "current_metadata_and_complete_history_not_model_or_network_latency",
        }


def _mirror_fixture(root, *, messages, other):
    manager, sid = _new_manager(root)
    _seed(manager, sid, messages=messages, other=other, long_messages=True)
    talk = Path(manager.get_session(sid)["talk_file"])
    payload = json.loads(talk.read_text(encoding="utf-8"))
    payload["customer_custom"] = {"preserved": ["synthetic", False, None]}
    payload["messages"] = [{"id": row["id"], "role": row["role"], "content": row["content"],
                            "timestamp": row["created_at"]} for row in manager.get_history(sid)]
    talk.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manager, sid, talk, payload


def benchmark_mirror_append(*, messages=200, other=1000):
    """Isolated fresh fixtures per arm/sample; actual complete add_message().

    Database setup, JSON reset and equality checks are outside timed blocks.
    Generated UUIDs and initial wall-clock creation stamps are normalized only
    for old/new equality; their geometry/role/content/metadata are not removed.
    """
    text = "New synthetic short message"
    stamp = datetime(2026, 1, 1)
    samples = {"baseline_metadata_append": [], "current_metadata_append": []}
    bytes_written = {key: [] for key in samples}
    fixture_bytes = None
    with tempfile.TemporaryDirectory(prefix="docthinker-session-mirror-synthetic-") as directory:
        for repeat in range(REPEATS):
            normalized = []
            order = list(samples) if repeat % 2 == 0 else list(reversed(samples))
            for key in order:
                root = Path(directory) / f"{repeat}-{key}"
                manager, sid, talk, payload = _mirror_fixture(root, messages=messages, other=other)
                fixture_bytes = talk.stat().st_size
                if key.startswith("baseline"):
                    manager.get_session = lambda value: _old_session(manager, value)
                started = time.perf_counter()
                mid = manager.add_message(sid, "user", text, stamp)
                elapsed = (time.perf_counter() - started) * 1000
                after = json.loads(talk.read_text(encoding="utf-8"))
                expected = {**payload, "messages": [*payload["messages"], {
                    "id": mid, "role": "user", "content": text, "timestamp": stamp.isoformat()}],
                    "updated_at": stamp.isoformat()}
                if after != expected:
                    raise AssertionError("Complete JSON mirror output changed")
                if manager.get_history(sid) != _old_history(manager, sid):
                    raise AssertionError("Complete committed history output changed")
                # Preserve every field, normalizing only nondeterministic IDs and
                # automatically generated session creation stamps across arms.
                comparable = {**after, "created_at": "generated-session-stamp",
                              "messages": [*after["messages"][:-1],
                                           {**after["messages"][-1], "id": "generated-message-id"}]}
                normalized.append(comparable)
                samples[key].append(elapsed)
                bytes_written[key].append(talk.stat().st_size)
            if normalized[0] != normalized[1]:
                raise AssertionError("Old/new full JSON mirror differs")
        return {
            "fixture_messages": messages, "unused_document_graph_entries": other,
            "fixture_mirror_bytes": fixture_bytes, "append_content_bytes": len(text.encode("utf-8")),
            "timings": {key: {**_summary(values), "mirror_bytes_each_append": bytes_written[key]}
                        for key, values in samples.items()},
            "full_output_equal": True, "comparison_normalizes_only_generated_id_and_creation_stamp": True,
            "full_json_mirror_still_rewritten": True, "fsync_or_sqlite_pragma_changed": False,
            "scope": "complete_existing_sqlite_transaction_and_realtime_json_mirror_not_model_latency",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", type=int, default=1000)
    parser.add_argument("--other", type=int, default=1000)
    parser.add_argument("--mirror-messages", type=int, default=200)
    args = parser.parse_args()
    if not 1 <= args.messages <= 2000 or not 0 <= args.other <= 2000 or not 1 <= args.mirror_messages <= 2000:
        parser.error("message counts must be 1..2000 and other entries 0..2000")
    fingerprints = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (ROOT / "docthinker/session_manager.py", ROOT / "docthinker/knowledge_base_storage.py")}
    with _offline() as offline:
        result = {"kind": "synthetic_session_read_and_mirror_benchmark", "version": 1,
                  "python": platform.python_version(), "platform": platform.platform(), "samples": REPEATS,
                  "source_fingerprints_before": fingerprints,
                  "reads": benchmark_session_reads(messages=args.messages, other=args.other),
                  "mirror": benchmark_mirror_append(messages=args.mirror_messages, other=args.other),
                  "vendor_calls": 0, "network_attempts": offline["network_attempts"],
                  "private_database_or_configuration_read": False,
                  "temporary_fixture_cleanup": "TemporaryDirectory_on_success_or_exception",
                  "os_page_cache_controlled": False}
    after = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in fingerprints}
    if after != fingerprints:
        raise AssertionError("Production sources changed during benchmark; discard results")
    result["source_fingerprints_after"] = after
    result["production_sources_unchanged_during_measurement"] = True
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
