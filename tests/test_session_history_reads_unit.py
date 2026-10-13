"""Lossless live-history/metadata reads against private temporary SQLite data.

The reference path deliberately keeps the previous fully hydrated storage
calls. No model, real user database, persisted credential or network is used.
"""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from docthinker.knowledge_base import KnowledgeEntry
from docthinker.session_manager import SessionManager


@pytest.fixture
def manager(tmp_path):
    return SessionManager(str(tmp_path / "_system"), str(tmp_path / "data"), migrate_legacy=False)


def reference_history(manager, session_id):
    name = "session_" + manager._canonical_session_id(session_id)
    history = []
    for entry in manager.kb_storage.query_entries(name, ""):
        if entry.entry_type not in ("question", "answer"):
            continue
        history.append({
            "id": entry.id,
            "role": entry.metadata.get("role", "user"),
            "content": entry.content,
            "created_at": entry.created_at.isoformat() if entry.created_at else "",
            "timestamp": entry.created_at.timestamp() if entry.created_at else 0.0,
            **({"research_core": True, "research_provenance_required": True}
               if entry.metadata.get("research_provenance_required") is True else {}),
        })
    history.sort(key=lambda row: row["created_at"])
    return history


def reference_session(manager, session_id):
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


def seed_mixed(manager, sid, *, messages=64, other=128):
    """Include equal timestamps, large unused document/graph bodies and relations."""
    stamp = datetime(2025, 7, 2, 12, 0, tzinfo=timezone(timedelta(hours=8)))
    conn = manager.kb_storage._get_connection()
    try:
        kb_id = conn.execute("SELECT id FROM knowledge_bases WHERE name = ?", ("session_" + sid,)).fetchone()[0]
        rows = []
        for index in range(messages + other):
            if index < messages:
                kind = "question" if index % 2 == 0 else "answer"
                metadata = {"role": "user" if index % 2 == 0 else "assistant",
                            "extra_nested": {"list": [index, False, None]},
                            "research_provenance_required": True if index % 5 == 0 else "true"}
                # Deliberately unsorted timestamps, with ties that preserve insertion order.
                created_at = stamp + timedelta(seconds=index % 7)
                content = "Message 演绎\\\"\n" + str(index)
            else:
                kind = ("document", "entity", "relationship", "generic")[index % 4]
                metadata = {"filename": f"unused-{index}.pdf", "graph": {"weights": [0.1, 0.9]}}
                created_at = stamp
                content = "UNUSED_PRIVATE_SYNTHETIC_BODY_" * 200
            rows.append((f"entry-{sid}-{index}", kb_id, content, kind, json.dumps(metadata),
                         created_at.isoformat(), created_at.isoformat()))
        conn.executemany(
            "INSERT INTO knowledge_entries (id,kb_id,content,type,metadata,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?)", rows)
        conn.executemany("INSERT INTO knowledge_relations (source_id,target_id,relation_type) VALUES (?,?,?)",
                         [(row[0], rows[(index + 1) % len(rows)][0], "related-to")
                          for index, row in enumerate(rows)])
        conn.commit()
    finally:
        conn.close()


def trace_reads(manager, monkeypatch):
    statements = []
    original = manager.kb_storage._get_connection

    def connect():
        conn = original()
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(manager.kb_storage, "_get_connection", connect)
    return statements


@pytest.mark.parametrize("alias", ["#00001", "#1", "1", "  #1  "])
def test_live_history_matches_original_all_fields_and_stable_ties(manager, alias):
    sid = manager.create_session("私密合成工作区")["id"]
    seed_mixed(manager, sid)
    before = reference_history(manager, sid)
    after = manager.get_history(alias)
    assert after == before
    assert json.dumps(after, ensure_ascii=False) == json.dumps(before, ensure_ascii=False)
    assert len(after) == 64 and all("UNUSED_PRIVATE_SYNTHETIC_BODY" not in row["content"] for row in after)


def test_history_one_select_and_zero_unrelated_relation_materialization(manager, monkeypatch):
    sid = manager.create_session()["id"]
    seed_mixed(manager, sid, messages=256, other=512)
    expected = reference_history(manager, sid)
    statements = trace_reads(manager, monkeypatch)
    assert manager.get_history(sid) == expected
    selects = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
    assert len(selects) == 1
    assert not any("knowledge_relations" in sql.lower() for sql in selects)
    assert "type IN ('question', 'answer')" in selects[0]
    assert "ORDER BY rowid" in selects[0]


@pytest.mark.parametrize("alias", ["#00001", "#1", "1"])
def test_metadata_matches_original_without_any_entry_reads(manager, monkeypatch, alias):
    sid = manager.create_session("标题")["id"]
    seed_mixed(manager, sid)
    expected = reference_session(manager, alias)
    statements = trace_reads(manager, monkeypatch)
    assert manager.get_session(alias) == expected
    selects = [sql for sql in statements if sql.lstrip().upper().startswith("SELECT")]
    assert len(selects) == 1
    assert all("knowledge_entries" not in sql.lower() and "knowledge_relations" not in sql.lower() for sql in selects)


def test_missing_and_legacy_raw_session_name_still_follow_original_lookup(manager):
    assert manager.get_history("#99999") == []
    assert manager.get_session("#99999") is None
    manager.kb_storage.create_knowledge_base("session_1", "session", {"title": "legacy", "session_id": "1"})
    assert manager.get_session("1") == reference_session(manager, "1")
    assert manager.get_history("1") == reference_history(manager, "1") == []


def test_history_keeps_legacy_missing_role_and_exact_boolean_provenance(manager):
    sid = manager.create_session()["id"]
    for index, metadata in enumerate(({}, {"role": "system"}, {"research_provenance_required": 1},
                                      {"research_provenance_required": "true"},
                                      {"research_provenance_required": True})):
        manager.kb_storage.add_entry("session_" + sid, KnowledgeEntry(
            id=f"legacy-{index}", content=f"row-{index}", entry_type="answer", metadata=metadata))
    rows = manager.get_history(sid)
    assert rows == reference_history(manager, sid)
    assert rows[0]["role"] == "user"
    assert rows[1]["role"] == "system"
    assert not any("research_core" in row for row in rows[:-1])
    assert rows[-1]["research_provenance_required"] is True


def test_live_commits_rollback_removal_rename_and_restart_are_not_cached(manager):
    sid = manager.create_session("before")["id"]
    other = SessionManager(str(manager.base_storage_path), str(manager.data_root), migrate_legacy=False)
    stamp = datetime(2025, 1, 1, 1, 1, 1)
    mid = other.add_message(sid, "assistant", "original", stamp, research_provenance_required=True)
    assert manager.get_history(sid) == reference_history(other, sid)
    assert manager.update_session(sid, "after")
    assert other.get_session(sid)["title"] == "after"
    conn = other.kb_storage._get_connection()
    try:
        conn.execute("UPDATE knowledge_entries SET content = ? WHERE id = ?", ("uncommitted", mid))
        assert manager.get_history(sid)[0]["content"] == "original"
        conn.rollback()
        conn.execute("UPDATE knowledge_entries SET content = ? WHERE id = ?", ("committed", mid))
        conn.commit()
    finally:
        conn.close()
    assert manager.get_history(sid)[0]["content"] == "committed"
    assert other.kb_storage.delete_entry("session_" + sid, mid)
    assert manager.get_history(sid) == []
    restarted = SessionManager(str(manager.base_storage_path), str(manager.data_root), migrate_legacy=False)
    assert restarted.get_history(sid) == [] and restarted.get_session(sid)["title"] == "after"


def test_other_process_write_is_visible_without_cache_and_keeps_full_json(manager):
    sid = manager.create_session()["id"]
    code = (
        "from docthinker.session_manager import SessionManager; "
        "import sys; m=SessionManager(sys.argv[1],sys.argv[2],migrate_legacy=False); "
        "m.add_message(sys.argv[3],'user','cross-process synthetic write')"
    )
    env = {**os.environ, "PYTHON_DOTENV_DISABLED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run([sys.executable, "-c", code, str(manager.base_storage_path),
                                str(manager.data_root), sid], env=env, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    assert manager.get_history(sid) == reference_history(manager, sid)
    payload = json.loads(Path(manager.get_session(sid)["talk_file"]).read_text(encoding="utf-8"))
    assert payload["messages"][0]["content"] == "cross-process synthetic write"


def test_add_message_preserves_live_complete_json_unknown_fields_and_message_order(manager):
    sid = manager.create_session("mirror title")["id"]
    talk = Path(manager.get_session(sid)["talk_file"])
    payload = json.loads(talk.read_text(encoding="utf-8"))
    payload["customer_custom"] = {"list": ["私密合成", 0, None]}
    payload["messages"] = [{"id": "legacy", "role": "user", "content": "old", "timestamp": "legacy-date",
                            "extra": {"preserved": True}}]
    talk.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    stamp = datetime(2025, 5, 4, 3, 2, 1, 123456, tzinfo=timezone.utc)
    new_id = manager.add_message(sid, "assistant", "new 演绎\n\\\"", stamp,
                                 research_provenance_required=True)
    expected = dict(payload)
    expected["messages"] = [*payload["messages"], {"id": new_id, "role": "assistant",
                            "content": "new 演绎\n\\\"", "timestamp": stamp.isoformat()}]
    expected["updated_at"] = stamp.isoformat()
    assert json.loads(talk.read_text(encoding="utf-8")) == expected
    assert talk.read_text(encoding="utf-8") == json.dumps(expected, ensure_ascii=False, indent=2)
    assert manager.get_history(sid)[0]["research_provenance_required"] is True


def test_mirror_write_failure_still_preserves_original_canonical_sqlite_semantics(manager, monkeypatch, caplog):
    sid = manager.create_session()["id"]
    talk = Path(manager.get_session(sid)["talk_file"])
    original_text = talk.read_text(encoding="utf-8")
    write = Path.write_text

    def fail_mirror(path, *args, **kwargs):
        if path == talk:
            raise OSError("synthetic mirror failure")
        return write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_mirror)
    mid = manager.add_message(sid, "user", "durably stored by existing SQLite transaction")
    assert manager.get_history(sid)[0]["id"] == mid
    assert talk.read_text(encoding="utf-8") == original_text
    assert "Failed to append talk log" in caplog.text


def test_canonical_write_failure_does_not_append_mirror(manager, monkeypatch):
    sid = manager.create_session()["id"]
    talk = Path(manager.get_session(sid)["talk_file"])
    original = talk.read_text(encoding="utf-8")

    def fail(*args, **kwargs):
        raise OSError("synthetic canonical write failure")

    monkeypatch.setattr(manager.kb_storage, "add_entry", fail)
    with pytest.raises(OSError, match="canonical write"):
        manager.add_message(sid, "user", "not committed")
    assert manager.get_history(sid) == []
    assert talk.read_text(encoding="utf-8") == original


def test_deleted_then_reused_number_never_returns_old_history_or_instance(manager):
    sid = manager.create_session()["id"]
    old_instance = manager.get_session(sid)["instance_id"]
    manager.add_message(sid, "user", "OLD_SYNTHETIC_PRIVATE_MESSAGE")
    assert manager.delete_session(sid)
    assert manager.get_session(sid) is None and manager.get_history(sid) == []
    assert not (manager.data_root / sid).exists()
    replacement = manager.create_session("replacement")
    assert replacement["id"] == sid
    assert manager.get_session(sid)["instance_id"] != old_instance
    assert manager.get_history(sid) == []
    assert json.loads(Path(replacement["talk_file"]).read_text(encoding="utf-8"))["messages"] == []


def test_live_read_errors_are_not_hidden_or_replaced_by_stale_history(manager, monkeypatch):
    sid = manager.create_session()["id"]
    manager.add_message(sid, "user", "visible")
    assert manager.get_history(sid)

    def fail():
        raise OSError("synthetic live read failure")

    monkeypatch.setattr(manager.kb_storage, "_get_connection", fail)
    with pytest.raises(OSError, match="live read"):
        manager.get_history(sid)
    with pytest.raises(OSError, match="live read"):
        manager.get_session(sid)


def test_malformed_message_metadata_keeps_failure_visible(manager):
    sid = manager.create_session()["id"]
    mid = manager.add_message(sid, "user", "message")
    conn = manager.kb_storage._get_connection()
    try:
        conn.execute("UPDATE knowledge_entries SET metadata = ? WHERE id = ?", ("{bad", mid))
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(json.JSONDecodeError):
        manager.get_history(sid)
