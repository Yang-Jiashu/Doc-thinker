"""Offline durable request contracts; no model clients or real credentials."""
import concurrent.futures
import hashlib
import json
import os
import sqlite3

import pytest

from docthinker.server import request_journal as module
from docthinker.server.request_journal import RequestConflictError, RequestJournal, request_fingerprint

SID = "#00001"
RID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
RID2 = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
PAYLOAD = {"session_id": SID, "question": "synthetic private question"}


@pytest.fixture
def journal(tmp_path):
    return RequestJournal(tmp_path.resolve(), boot_id="first-process", clock=lambda: 1700000000)


def test_status_is_read_only_and_does_not_initialize_storage(journal):
    assert journal.read(SID, RID) == {"request_id": RID, "request_status": "not_found"}
    assert not journal.path.exists()


def test_canonical_fingerprint_binds_semantics_without_id():
    digest = request_fingerprint(PAYLOAD)
    assert digest == request_fingerprint({**PAYLOAD, **module.CONTROL_DEFAULTS, "request_id": RID2})
    assert digest != request_fingerprint({**PAYLOAD, "question": "different"})
    for key in module.CONTROL_DEFAULTS:
        assert digest != request_fingerprint({**PAYLOAD, key: not module.CONTROL_DEFAULTS[key]})


@pytest.mark.parametrize("data", [
    {"session_id": "#00002/path", "question": "q"},
    {"session_id": SID, "question": "  "},
    {"session_id": SID, "question": "q", "use_memory": "false"},
    {"session_id": SID, "question": 12},
])
def test_invalid_semantics_never_get_a_fingerprint(data):
    with pytest.raises(ValueError):
        request_fingerprint(data)


def test_begin_is_atomic_and_never_copies_question_or_credentials(journal):
    digest = request_fingerprint(PAYLOAD)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(lambda _: journal.begin(SID, RID, digest), range(8))) == 1
    assert journal.path.stat().st_mode & 0o777 == 0o600
    assert PAYLOAD["question"].encode() not in journal.path.read_bytes()
    assert journal.read(SID, RID, active=True)["request_status"] == "running"
    assert journal.read(SID, RID)["request_status"] == "unknown"


def test_restart_preserves_unknown_and_never_reexecutes(journal):
    digest = request_fingerprint(PAYLOAD)
    assert journal.begin(SID, RID, digest)
    restarted = RequestJournal(journal.directory, boot_id="second-process", clock=journal.clock)
    assert restarted.read(SID, RID, active=True)["request_status"] == "unknown"
    assert not restarted.begin(SID, RID, digest)
    assert not restarted.complete(SID, RID, {"answer": "must not adopt old running execution", "sources": []})


def test_completed_answer_replays_after_restart(journal):
    digest = request_fingerprint(PAYLOAD)
    assert journal.begin(SID, RID, digest)
    result = {"answer": "synthetic result", "sources": [], "answer_id": RID2, "context_windowed": True}
    assert journal.complete(SID, RID, result)
    restarted = RequestJournal(journal.directory, boot_id="second-process", clock=journal.clock)
    assert restarted.read(SID, RID) == {**result, "request_id": RID, "request_status": "completed"}
    assert not restarted.begin(SID, RID, digest)


@pytest.mark.parametrize("code", ["model_response_truncated", "model_response_empty", "model_response_invalid"])
def test_known_response_failure_is_terminal_bounded_and_survives_restart_and_expiry(journal, code):
    from docthinker.server.model_response_error import model_response_failure
    digest = request_fingerprint(PAYLOAD)
    assert journal.begin(SID, RID, digest)
    assert journal.fail(SID, RID, code)
    expected = {"request_id": RID, "request_status": "failed", "request_error": model_response_failure(code)}
    assert journal.read(SID, RID, active=True) == expected
    assert journal.recent_unresolved(SID, active_ids={RID}) == []
    restarted = RequestJournal(journal.directory, boot_id="next", clock=lambda: journal.clock() + 365 * 86400)
    assert restarted.read(SID, RID) == expected
    assert not restarted.begin(SID, RID, digest)
    assert not journal.complete(SID, RID, {"answer": "must not override terminal failure"})
    assert not journal.fail(SID, RID, "model_response_empty")
    journal.mark_unknown(SID, RID)
    assert journal.read(SID, RID) == expected
    with pytest.raises(RequestConflictError):
        restarted.begin(SID, RID, request_fingerprint({**PAYLOAD, "question": "changed"}))
    assert PAYLOAD["question"].encode() not in journal.path.read_bytes()


def test_failure_code_cannot_overwrite_completed_or_other_boot_and_cannot_store_diagnostics(journal):
    digest = request_fingerprint(PAYLOAD)
    journal.begin(SID, RID, digest)
    other = RequestJournal(journal.directory, boot_id="other", clock=journal.clock)
    assert not other.fail(SID, RID, "model_response_empty")
    with pytest.raises(ValueError, match="Invalid model response failure code"):
        journal.fail(SID, RID, "private-provider-body")
    assert journal.complete(SID, RID, {"answer": "original"})
    assert not journal.fail(SID, RID, "model_response_truncated")
    assert journal.read(SID, RID)["answer"] == "original"
    assert b"private-provider-body" not in journal.path.read_bytes()


def test_known_host_predispatch_rejection_is_distinct_from_model_response_and_not_replayed(journal):
    digest = request_fingerprint(PAYLOAD)
    assert journal.begin(SID, RID, digest)
    assert journal.fail(SID, RID, 'inputs_unavailable')
    result = journal.read(SID, RID)
    assert result['request_status'] == 'failed'
    assert result['request_error']['model_dispatched'] is False
    assert result['request_error']['automatic_retry'] is False
    assert journal.recent_unresolved(SID) == []
    restarted = RequestJournal(journal.directory, boot_id='next', clock=journal.clock)
    assert restarted.read(SID, RID) == result
    assert not restarted.begin(SID, RID, digest)
    assert not journal.complete(SID, RID, {'answer': 'must not replace a terminal host rejection'})
    from docthinker.server.model_response_error import model_response_failure
    with pytest.raises(ValueError):
        model_response_failure('inputs_unavailable')  # Not a fake provider response.


def test_old_journal_schema_read_is_read_only_then_upgrades_without_reusing_id(journal):
    digest = request_fingerprint(PAYLOAD)
    with sqlite3.connect(journal.path) as db:
        db.execute("CREATE TABLE query_requests (session_id TEXT,request_id TEXT,fingerprint TEXT,boot_id TEXT,"
                   "state TEXT CHECK(state IN ('running','completed','unknown')),result TEXT,result_bytes INTEGER,"
                   "created_at REAL,updated_at REAL,PRIMARY KEY(session_id,request_id))")
        db.execute("INSERT INTO query_requests VALUES(?,?,?,?,?,?,?,?,?)",
                   (SID, RID, digest, journal.boot_id, "running", None, 0, journal.clock(), journal.clock()))
    journal.path.chmod(0o600)
    assert journal.read(SID, RID)["request_status"] == "unknown"
    with sqlite3.connect(journal.path) as db:
        assert "failure_code" not in {row[1] for row in db.execute("PRAGMA table_info(query_requests)")}
    assert journal.fail(SID, RID, "model_response_empty")
    assert not journal.begin(SID, RID, digest)
    assert journal.read(SID, RID)["request_status"] == "failed"


def test_known_failures_do_not_hide_older_live_recovery_references(journal):
    digest = request_fingerprint(PAYLOAD)
    journal.begin(SID, RID, digest)
    for index in range(7):
        identifier = f"{index:08x}-ffff-4fff-8fff-ffffffffffff"
        journal.begin(SID, identifier, digest)
        journal.fail(SID, identifier, "model_response_empty")
    assert journal.recent_unresolved(SID, active_ids={RID}) == [{"request_id": RID, "request_status": "running"}]


def test_id_conflict_does_not_replace_result(journal):
    journal.begin(SID, RID, request_fingerprint(PAYLOAD))
    journal.complete(SID, RID, {"answer": "original", "sources": []})
    different = request_fingerprint({**PAYLOAD, "question": "new content"})
    with pytest.raises(RequestConflictError):
        journal.begin(SID, RID, different)
    with pytest.raises(RequestConflictError):
        journal.read(SID, RID, fingerprint=different)
    assert journal.read(SID, RID)["answer"] == "original"


def test_worker_directory_and_sessions_do_not_share_ids(journal, tmp_path):
    journal.begin(SID, RID, request_fingerprint(PAYLOAD))
    other_dir = tmp_path / "other-worker"
    other_dir.mkdir()
    other = RequestJournal(other_dir.resolve(), boot_id="other")
    assert other.read(SID, RID)["request_status"] == "not_found"
    assert journal.read("#00002", RID)["request_status"] == "not_found"


def test_result_expiration_preserves_non_retryable_tombstone(journal):
    digest = request_fingerprint(PAYLOAD)
    journal.begin(SID, RID, digest)
    journal.complete(SID, RID, {"answer": "old result", "sources": []})
    journal.clock = lambda: 1700000000 + module.RESULT_RETENTION_SECONDS + 1
    assert journal.read(SID, RID)["request_status"] == "unknown"
    assert not journal.begin(SID, RID, digest)
    journal.begin(SID, RID2, digest)
    journal.complete(SID, RID2, {"answer": "new result", "sources": []})
    with sqlite3.connect(journal.path) as db:
        assert db.execute("SELECT state,result FROM query_requests WHERE request_id=?", (RID,)).fetchone() == ("unknown", None)


def test_cached_bytes_do_not_drop_id_and_legacy_row_cap_no_longer_stops_lifetime(journal, monkeypatch):
    monkeypatch.setattr(module, "MAX_CACHED_BYTES", 100)
    monkeypatch.setattr(module, "MAX_JOURNAL_RECORDS", 2)
    digest = request_fingerprint(PAYLOAD)
    journal.begin(SID, RID, digest)
    journal.complete(SID, RID, {"answer": "x" * 50, "sources": []})
    journal.begin(SID, RID2, digest)
    journal.complete(SID, RID2, {"answer": "y" * 50, "sources": []})
    assert journal.read(SID, RID)["request_status"] == "unknown"
    assert journal.read(SID, RID2)["request_status"] == "completed"
    assert not journal.begin(SID, RID, digest)
    assert journal.begin("#00002", RID, digest)


def test_storage_quota_rejects_new_requests_but_keeps_idempotency_and_allows_expansion(journal):
    digest = request_fingerprint(PAYLOAD)
    assert journal.begin(SID, RID, digest)
    journal.max_storage_bytes = journal.path.stat().st_size
    assert not journal.begin(SID, RID, digest)
    with pytest.raises(module.RequestJournalLimitError):
        journal.begin(SID, RID2, digest)
    assert journal.read(SID, RID2)["request_status"] == "not_found"
    journal.max_storage_bytes *= 4
    assert journal.begin(SID, RID2, digest)
    restarted = RequestJournal(journal.directory, boot_id="restarted", clock=journal.clock)
    assert not restarted.begin(SID, RID, digest)


def test_completion_at_storage_pressure_keeps_unknown_tombstone(journal):
    digest = request_fingerprint(PAYLOAD)
    journal.begin(SID, RID, digest)
    journal.max_storage_bytes = journal.path.stat().st_size
    assert not journal.complete(SID, RID, {"answer": "can't cache this at quota", "sources": []})
    assert journal.read(SID, RID)["request_status"] == "unknown"
    journal.max_storage_bytes = module.DEFAULT_JOURNAL_BYTES
    assert not journal.begin(SID, RID, digest)


@pytest.mark.parametrize("quota", [0, -1, 65535, True, "100000", 1 << 51])
def test_invalid_journal_quotas_rejected_without_files(tmp_path, quota):
    with pytest.raises(ValueError):
        RequestJournal(tmp_path.resolve(), boot_id="test", max_storage_bytes=quota)
    assert not list(tmp_path.iterdir())


def test_operator_can_expand_storage_using_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCTHINKER_REQUEST_JOURNAL_BYTES", "134217728")
    assert RequestJournal(tmp_path.resolve(), boot_id="test").max_storage_bytes == 134217728
    monkeypatch.setenv("DOCTHINKER_REQUEST_JOURNAL_BYTES", "not a quota")
    with pytest.raises(ValueError):
        RequestJournal(tmp_path.resolve(), boot_id="test")


def test_fingerprints_preserve_legacy_defaults_and_bind_reasoning_and_event_time(journal):
    legacy = hashlib.sha256(json.dumps({**PAYLOAD, **module.CONTROL_DEFAULTS}, ensure_ascii=False,
                                      sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert request_fingerprint(PAYLOAD) == legacy
    assert request_fingerprint({**PAYLOAD, "reasoning_mode": "auto", "memory_as_of": None}) == legacy
    digests = {legacy}
    for mode in ("evidence", "analogy", "deduction", "induction"):
        digests.add(request_fingerprint({**PAYLOAD, "reasoning_mode": mode}))
    assert len(digests) == 5
    assert request_fingerprint({**PAYLOAD, "memory_as_of": 100}) == request_fingerprint({**PAYLOAD, "memory_as_of": 100.0})
    assert request_fingerprint({**PAYLOAD, "memory_as_of": -0.0}) == request_fingerprint({**PAYLOAD, "memory_as_of": 0})
    journal.begin(SID, RID, legacy)
    for data in ({"reasoning_mode": "analogy"}, {"memory_as_of": 100}, {"memory_as_of": 101}):
        changed = request_fingerprint({**PAYLOAD, **data})
        with pytest.raises(RequestConflictError):
            journal.begin(SID, RID, changed)
        with pytest.raises(RequestConflictError):
            journal.read(SID, RID, fingerprint=changed)


@pytest.mark.parametrize("extra", [
    {"reasoning_mode": []}, {"reasoning_mode": None}, {"reasoning_mode": "causal"},
    {"memory_as_of": True}, {"memory_as_of": -1}, {"memory_as_of": float("nan")},
    {"memory_as_of": float("inf")}, {"memory_as_of": "100"}, {"memory_as_of": 253402300800},
    {"memory_as_of": 10 ** 1000}, {"session_id": "#００００１"},
])
def test_invalid_reasoning_semantics_are_rejected(extra):
    with pytest.raises(ValueError):
        request_fingerprint({**PAYLOAD, **extra})


def test_more_than_ten_thousand_receipts_keep_ids_and_constant_work_completion(journal, monkeypatch):
    digest = request_fingerprint(PAYLOAD)
    journal.begin(SID, RID, digest)
    with sqlite3.connect(journal.path) as db:
        db.executemany("INSERT INTO query_requests (session_id,request_id,fingerprint,boot_id,state,result,result_bytes,created_at,updated_at) VALUES(?,?,?,'old','unknown',NULL,0,?,?)",
                       [(SID, f"{i:08x}-aaaa-4aaa-8aaa-aaaaaaaaaaaa", digest, 1, 1) for i in range(10_010)])
    assert journal.begin(SID, RID2, digest)
    work = [0]
    original = sqlite3.connect
    def metered_connect(*args, **kwargs):
        db = original(*args, **kwargs)
        db.set_progress_handler(lambda: work.__setitem__(0, work[0] + 100) or 0, 100)
        return db
    monkeypatch.setattr(module.sqlite3, "connect", metered_connect)
    assert journal.complete(SID, RID2, {"answer": "bounded completion", "sources": []})
    assert work[0] < 10_000, f"Completion unexpectedly scanned the lifetime journal: {work[0]} instructions"
    assert not journal.begin(SID, "00000000-aaaa-4aaa-8aaa-aaaaaaaaaaaa", digest)


def test_legacy_journal_usage_migrates_once_and_remains_exact(journal, monkeypatch):
    digest = request_fingerprint(PAYLOAD)
    journal.begin(SID, RID, digest)
    first_result = {"answer": "first retained reply", "sources": []}
    journal.complete(SID, RID, first_result)
    with sqlite3.connect(journal.path) as db:
        db.executescript("""
            DROP TRIGGER query_requests_usage_insert;
            DROP TRIGGER query_requests_usage_update;
            DROP TRIGGER query_requests_usage_delete;
            DROP TABLE query_request_usage;
            DROP INDEX query_requests_completed_time;
            DROP INDEX query_requests_unresolved_time;
        """)
    before = journal.path.read_bytes()
    assert journal.read(SID, RID)["answer"] == first_result["answer"]
    assert before == journal.path.read_bytes()
    journal.begin(SID, RID2, digest)
    journal.complete(SID, RID2, {"answer": "second retained reply", "sources": []})
    with sqlite3.connect(journal.path) as db:
        assert db.execute("SELECT result_bytes FROM query_request_usage").fetchone()[0] == db.execute("SELECT SUM(result_bytes) FROM query_requests").fetchone()[0]
    monkeypatch.setattr(module, "MAX_CACHED_BYTES", 70)
    third = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
    journal.begin(SID, third, digest)
    journal.complete(SID, third, {"answer": "new result", "sources": []})
    with sqlite3.connect(journal.path) as db:
        assert db.execute("SELECT result_bytes FROM query_request_usage").fetchone()[0] == db.execute("SELECT SUM(result_bytes) FROM query_requests").fetchone()[0]
        assert db.execute("SELECT count(*) FROM query_requests").fetchone()[0] == 3


def test_oversized_result_is_unknown_not_auto_retriable(journal, monkeypatch):
    monkeypatch.setattr(module, "MAX_RESULT_BYTES", 40)
    digest = request_fingerprint(PAYLOAD)
    journal.begin(SID, RID, digest)
    assert not journal.complete(SID, RID, {"answer": "x" * 100, "sources": []})
    assert journal.read(SID, RID)["request_status"] == "unknown"
    assert not journal.begin(SID, RID, digest)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "public"])
def test_unsafe_database_files_fail_closed(journal, tmp_path, kind):
    other = tmp_path / "do-not-read"
    other.write_text("synthetic private file")
    if kind == "symlink":
        journal.path.symlink_to(other)
    elif kind == "hardlink":
        os.link(other, journal.path)
    else:
        journal.path.write_text("synthetic private file")
        journal.path.chmod(0o644)
    with pytest.raises(ValueError):
        journal.read(SID, RID)
    with pytest.raises(ValueError):
        journal.begin(SID, RID, request_fingerprint(PAYLOAD))


def test_explicit_unknown_is_never_promoted_by_late_completion(journal):
    journal.begin(SID, RID, request_fingerprint(PAYLOAD))
    journal.mark_unknown(SID, RID)
    assert not journal.complete(SID, RID, {"answer": "late", "sources": []})
    assert journal.read(SID, RID)["request_status"] == "unknown"


def test_recent_unresolved_is_bounded_current_session_and_read_only(journal):
    assert journal.recent_unresolved(SID) == []
    assert not journal.path.exists()
    digest = request_fingerprint(PAYLOAD)
    for index in range(8):
        rid = f"{index:08x}-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        journal.begin(SID, rid, digest)
    journal.begin("#00002", RID, digest)
    unresolved = journal.recent_unresolved(SID, active_ids={"00000007-aaaa-4aaa-8aaa-aaaaaaaaaaaa"})
    assert len(unresolved) == 5
    assert unresolved[0] == {"request_id": "00000007-aaaa-4aaa-8aaa-aaaaaaaaaaaa", "request_status": "running"}
    assert all(item["request_id"] != RID for item in unresolved)
    journal.clock = lambda: 1700000000 + module.RECOVERY_WINDOW_SECONDS + 1
    assert journal.recent_unresolved(SID) == []
    assert not journal.begin(SID, "00000007-aaaa-4aaa-8aaa-aaaaaaaaaaaa", digest)
