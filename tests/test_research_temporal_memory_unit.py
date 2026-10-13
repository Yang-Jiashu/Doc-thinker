"""Temporal, privacy and indexed lifetime recall contracts, without a model."""
from contextlib import contextmanager
import sqlite3
import time
import uuid

import pytest

from docthinker.server.research_memory import select_confirmed_memories
from docthinker.server.research_store import (
    ResearchStore, ResearchStoreError, ResearchRevisionConflictError,
)


def test_event_time_corrections_keep_recording_time_and_half_open_history(tmp_path):
    now = [100.0]
    store = ResearchStore(tmp_path, clock=lambda: now[0])
    original = store.create_memory("#00001", "decision", "Use wind generation", valid_from=10)
    now[0] = 200
    corrected = store.update_memory("#00001", original["id"], expected_revision=1,
                                    text="Use solar generation", valid_from=50)
    assert corrected["recorded_at"] == 200 and corrected["valid_from"] == 50
    before = store.search_memories("#00001", "generation", as_of=49)
    after = store.search_memories("#00001", "generation", as_of=50)
    assert [row["text"] for row in before] == ["Use wind generation"]
    assert [row["text"] for row in after] == ["Use solar generation"]
    assert before[0]["valid_until"] == 50 and before[0]["recorded_until"] == 200
    assert select_confirmed_memories("generation", before)["memories"][0]["revision"] == 1
    assert store.search_memories("#00001", "generation", as_of=9) == []
    with pytest.raises(ResearchRevisionConflictError):
        store.update_memory("#00001", original["id"], expected_revision=2,
                            text="Cannot overwrite earlier timeline", valid_from=49)
    restarted = ResearchStore(tmp_path, clock=lambda: 201)
    assert restarted.search_memories("#00001", "wind", as_of=20)[0]["revision"] == 1


def test_timed_expiry_future_notes_and_archive_are_not_forgetting(tmp_path):
    now = [100]
    store = ResearchStore(tmp_path, clock=lambda: now[0])
    expiring = store.create_memory("#00001", "finding", "hydrogen trial", valid_from=20, valid_until=101)
    future = store.create_memory("#00001", "decision", "hydrogen expansion", valid_from=110)
    archived = store.archive_memory("#00001", expiring["id"], expected_revision=1)
    assert archived["status"] == "archived" and archived["revision"] == 2
    assert store.page_memories("#00001")["total"] == 0
    assert store.page_memories("#00001", include_archived=True)["total"] == 1
    recalled = store.search_memories("#00001", "hydrogen")
    assert recalled[0]["archived"] and recalled[0]["id"] == expiring["id"]
    assert select_confirmed_memories("hydrogen", recalled)["memories"][0]["id"] == expiring["id"]
    now[0] = 105
    assert store.search_memories("#00001", "hydrogen") == []
    assert store.search_memories("#00001", "hydrogen", as_of=110)[0]["id"] == future["id"]
    restored = store.archive_memory("#00001", expiring["id"], expected_revision=2, archived=False)
    assert restored["status"] == "active" and restored["revision"] == 3
    assert store.search_memories("#00001", "hydrogen", as_of=30)[0]["text"] == "hydrogen trial"


def test_expired_correction_cannot_revive_older_archive_snapshot(tmp_path):
    now = [10]
    store = ResearchStore(tmp_path, clock=lambda: now[0])
    note = store.create_memory("#00001", "finding", "hydrogen initial conclusion")
    now[0] = 20
    store.archive_memory("#00001", note["id"], expected_revision=1)
    now[0] = 30
    store.archive_memory("#00001", note["id"], expected_revision=2, archived=False)
    now[0] = 40
    store.update_memory("#00001", note["id"], expected_revision=3,
                        text="hydrogen temporary correction", valid_until=50)
    assert store.search_memories("#00001", "hydrogen", as_of=35)[0]["text"] == "hydrogen initial conclusion"
    assert store.search_memories("#00001", "hydrogen", as_of=45)[0]["text"] == "hydrogen temporary correction"
    assert store.search_memories("#00001", "hydrogen", as_of=50) == []
    assert store.page_memories("#00001", as_of=50, include_archived=True)["total"] == 0


def test_deleting_document_revokes_old_version_even_after_independent_correction(tmp_path):
    now = [10]
    store = ResearchStore(tmp_path, clock=lambda: now[0])
    doc = store.accept_document("#00001", "paper.txt", b"external observations")
    store.claim_document("#00001", doc["id"])
    store.complete_document("#00001", doc["id"], [{"text": "external observations"}])
    chunk = store.source_chunks("#00001")[0]
    memory = store.create_memory("#00001", "finding", "external observations remembered", source_chunk_ids=[chunk["id"]])
    now[0] = 20
    store.update_memory("#00001", memory["id"], expected_revision=1,
                        text="independent later observations", source_chunk_ids=[])
    now[0] = 30
    store.delete_document("#00001", doc["id"])
    assert store.search_memories("#00001", "external", as_of=15) == []
    assert store.search_memories("#00001", "independent")[0]["revision"] == 2
    history = store.memory_history("#00001", memory["id"])
    assert history["items"][1]["text"] == "" and history["items"][1]["redacted"]
    assert history["items"][0]["text"] == "independent later observations"
    # Earlier databases did not mark all historical dependencies invalidated.
    # The live-source guard must still redact those inherited snapshots.
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE research_memory_versions SET status='superseded' WHERE memory_id=? AND revision=1", (memory["id"],))
    assert store.memory_history("#00001", memory["id"])["items"][1]["redacted"]
    assert store.memory_versions("#00001", memory["id"])[1]["text"] == ""


def test_delete_scrubs_search_index_and_historical_recall(tmp_path):
    now = [10]
    store = ResearchStore(tmp_path, clock=lambda: now[0])
    note = store.create_memory("#00001", "preference", "erasurewordone")
    now[0] = 20
    store.update_memory("#00001", note["id"], expected_revision=1, text="erasurewordtwo")
    store.archive_memory("#00001", note["id"], expected_revision=2)
    store.delete_memory("#00001", note["id"], expected_revision=3)
    assert store.search_memories("#00001", "erasurewordone", as_of=15) == []
    assert store.search_memories("#00001", "erasurewordtwo", as_of=20) == []
    assert all(row["text"] == "" for row in store.memory_history("#00001", note["id"])["items"])
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT count(*) FROM research_memory_search WHERE research_memory_search MATCH 'erasurewordone OR erasurewordtwo'").fetchone()[0] == 0


def test_current_browser_keeps_redacted_revocation_but_recall_and_history_do_not(tmp_path):
    store = ResearchStore(tmp_path, clock=lambda: 100)
    doc = store.accept_document("#00001", "source.txt", b"Source evidence")
    store.claim_document("#00001", doc["id"])
    store.complete_document("#00001", doc["id"], [{"text": "Source evidence"}])
    chunk = store.source_chunks("#00001")[0]
    memory = store.create_memory("#00001", "finding", "Revocation sentinel", source_chunk_ids=[chunk["id"]])
    store.delete_document("#00001", doc["id"])
    page = store.page_memories("#00001")
    assert page["total"] == 1 and page["items"][0]["id"] == memory["id"]
    assert page["items"][0]["status"] == "invalidated" and page["items"][0]["redacted"]
    assert page["items"][0]["text"] == "" and page["items"][0]["source_chunk_ids"] == []
    assert store.page_memories("#00001", as_of=100)["items"] == []
    assert store.page_memories("#00001", query="sentinel")["items"] == []
    assert store.search_memories("#00001", "sentinel") == []
    assert store.active_memory_context("#00001") == []
    assert store.page_memories("#00002")["items"] == []


def test_keyset_pages_are_disjoint_query_and_session_bound(tmp_path):
    store = ResearchStore(tmp_path, clock=lambda: 100)
    ids = {store.create_memory("#00001", "finding", "能源政策资料 " + str(i))["id"] for i in range(47)}
    store.create_memory("#00002", "finding", "能源政策资料 unrelated project")
    page = store.page_memories("#00001", query="能源政策", limit=20)
    collected = [row["id"] for row in page["items"]]
    assert page["total"] == 47 and page["next_cursor"]
    with pytest.raises(ResearchStoreError):
        store.page_memories("#00002", query="能源政策", cursor=page["next_cursor"])
    with pytest.raises(ResearchStoreError):
        store.page_memories("#00001", query="changed query", cursor=page["next_cursor"])
    while page["next_cursor"]:
        page = store.page_memories("#00001", query="能源政策", cursor=page["next_cursor"], limit=20)
        collected.extend(row["id"] for row in page["items"])
    assert len(collected) == len(set(collected)) == 47 and set(collected) == ids
    assert store.page_memories("#00001", query="\" OR * NOT malicious") ["items"] == []


def test_revision_history_pages_do_not_repeat_and_reject_wrong_identity(tmp_path):
    now = [10]
    store = ResearchStore(tmp_path, clock=lambda: now[0])
    memory = store.create_memory("#00001", "decision", "initial")
    for i in range(1, 7):
        now[0] += 1
        store.update_memory("#00001", memory["id"], expected_revision=i, text="revision " + str(i + 1))
    first = store.memory_history("#00001", memory["id"], limit=3)
    second = store.memory_history("#00001", memory["id"], limit=3, cursor=first["next_cursor"])
    assert first["total"] == 7 and [r["revision"] for r in first["items"] + second["items"]] == [7, 6, 5, 4, 3, 2]
    with pytest.raises(ResearchStoreError):
        store.memory_history("#00002", memory["id"], cursor=first["next_cursor"])


@pytest.mark.parametrize("invalid", [-1, True, float("nan"), float("inf"), "2026-01-01", 253402300800])
def test_invalid_event_times_rejected_without_memory(tmp_path, invalid):
    store = ResearchStore(tmp_path)
    with pytest.raises(ResearchStoreError):
        store.create_memory("#00001", "finding", "a note", valid_from=invalid)
    with pytest.raises(ResearchStoreError):
        store.search_memories("#00001", "a note", as_of=invalid)
    assert store.list_memories("#00001") == []


def test_more_than_ten_thousand_notes_keep_indexed_old_recall_bounded(tmp_path, monkeypatch):
    store = ResearchStore(tmp_path, clock=lambda: 100_000)
    old_id = str(uuid.uuid4())
    # Batch-seed through the same snapshot/index writer. Per-request validation
    # is covered separately; this fixture measures the retained lifetime size.
    with store._write() as db:
        for i in range(10_050):
            mid = old_id if i == 0 else str(uuid.uuid4())
            text = "zebrasolar 历史光伏方案" if i == 0 else "unrelated ordinary note " + str(i)
            db.execute("INSERT INTO research_memories(memory_id,session_id,kind,text,source_chunk_ids,revision,status,created_at,updated_at,valid_from) VALUES(?,?,'finding',?,'[]',1,'active',?,?,?)",
                       (mid, "#00001", text, i, i, i))
            store._snapshot(db, store._required_memory(db, "#00001", mid))
    # SQL instruction counting is repeatable across fast/slow CI hosts, unlike
    # tight wall-clock assertions. No corpus-sized Python list is materialized.
    work = [0]
    original_read = store._read
    @contextmanager
    def metered_read():
        with original_read() as db:
            if db is not None:
                db.set_progress_handler(lambda: work.__setitem__(0, work[0] + 100) or 0, 100)
            yield db
    monkeypatch.setattr(store, "_read", metered_read)
    started = time.monotonic()
    matches = store.search_memories("#00001", "zebrasolar 历史光伏", limit=20)
    elapsed = time.monotonic() - started
    assert len(matches) == 1 and matches[0]["id"] == old_id
    assert work[0] < 20_000, f"Unexpected corpus scan: {work[0]} SQL instructions, {elapsed:.3f}s"
    assert store.search_memories("#00002", "zebrasolar 历史光伏") == []
    assert store.page_memories("#00001", limit=20)["total"] == 10_050
    # Public writes also remain possible after the old 100/500/5,000 limits.
    assert store.create_memory("#00001", "decision", "new lifetime note")["revision"] == 1


def test_legacy_memory_rows_migrate_once_and_remain_temporally_searchable(tmp_path):
    now = [10]
    store = ResearchStore(tmp_path, clock=lambda: now[0])
    memory = store.create_memory("#00001", "finding", "legacy renewable conclusion")
    now[0] = 20
    store.update_memory("#00001", memory["id"], expected_revision=1, text="corrected renewable conclusion")
    with sqlite3.connect(store.path) as db:
        db.executescript("""
            DROP TABLE research_memory_search;
            DROP TABLE research_memory_sources;
            DROP INDEX research_memory_versions_time;
            DROP INDEX research_memory_versions_identity_time;
            DROP INDEX research_memory_versions_kind_time;
            ALTER TABLE research_memories DROP COLUMN valid_from;
            ALTER TABLE research_memories DROP COLUMN valid_until;
            ALTER TABLE research_memories DROP COLUMN recorded_until;
            ALTER TABLE research_memory_versions DROP COLUMN valid_from;
            ALTER TABLE research_memory_versions DROP COLUMN valid_until;
            ALTER TABLE research_memory_versions DROP COLUMN recorded_until;
        """)
    restarted = ResearchStore(tmp_path, clock=lambda: 30)
    assert restarted.search_memories("#00001", "renewable", as_of=15)[0]["text"] == "legacy renewable conclusion"
    assert restarted.search_memories("#00001", "renewable")[0]["text"] == "corrected renewable conclusion"
    before = store.path.read_bytes()
    restarted.search_memories("#00001", "renewable", as_of=15)
    assert before == store.path.read_bytes()
