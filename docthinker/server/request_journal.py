"""Small per-worker journal for one-shot chat execution, not an agent job queue.

An accepted ID is never silently reused. A process loss or ambiguous provider
failure is reported as unknown, never retried. No credentials or question text
are stored: the request is bound by a digest, and only bounded completed answers
are retained for reconnects. Status reads never create storage or call a model.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import time

from .model_response_error import model_response_failure


def _request_failure(code):
    # Keep host pre-dispatch rejections separate from provider-response errors:
    # no model result arrived and there is no uncertain attempt to reconcile.
    # Only the adapter that observes the actual model boundary uses this code.
    if code == 'inputs_unavailable':
        return {'code': code, 'message': '任务输入或请求预算未通过检查，未调用回答模型；请检查后重新授权。',
                'automatic_retry': False, 'model_dispatched': False}
    return model_response_failure(code)

REQUEST_ID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
MAX_JOURNAL_RECORDS = 10000
DEFAULT_JOURNAL_BYTES = 256 * 1024 * 1024
MAX_RESULT_BYTES = 1024 * 1024
MAX_CACHED_BYTES = 64 * 1024 * 1024
RESULT_RETENTION_SECONDS = 7 * 86400
RECOVERY_WINDOW_SECONDS = 86400
CONTROL_DEFAULTS = {
    "use_memory": True, "use_conversation_context": True,
    "use_llm_cache": True, "use_self_evolution": False,
}


class RequestConflictError(ValueError):
    pass


class RequestJournalLimitError(ValueError):
    pass


def request_fingerprint(data: dict) -> str:
    """Match omitted and explicit default controls; reject untrusted semantics."""
    session_id, question = data.get("session_id"), data.get("question")
    if (not isinstance(session_id, str) or not re.fullmatch(r"#[0-9]{5}", session_id)
            or not isinstance(question, str) or not question.strip() or len(question) > 8000):
        raise ValueError("Invalid request semantics")
    values = {"session_id": session_id, "question": question}
    for key, default in CONTROL_DEFAULTS.items():
        value = data.get(key, default)
        if type(value) is not bool:
            raise ValueError("Request controls must be boolean")
        values[key] = value
    mode, as_of = data.get("reasoning_mode", "auto"), data.get("memory_as_of")
    if not isinstance(mode, str) or mode not in {"auto", "evidence", "analogy", "deduction", "induction"}:
        raise ValueError("Invalid reasoning mode")
    if as_of is not None and (type(as_of) not in (int, float) or not 0 <= as_of <= 253402300799 or not math.isfinite(as_of)):
        raise ValueError("Invalid memory time")
    # Preserve fingerprints of pre-temporal default requests across upgrades.
    if mode != "auto":
        values["reasoning_mode"] = mode
    if as_of is not None:
        values["memory_as_of"] = float(as_of) if as_of else 0.0
    skill_mode = data.get('knowledge_skill_mode', 'auto')
    if type(skill_mode) is not str or skill_mode not in {'auto', 'off'}:
        raise ValueError('Invalid knowledge skill mode')
    # Default/absent fields keep pre-skill fingerprints stable. A changed
    # assistance setting is a different request, never a paid silent replay.
    if skill_mode != 'auto':
        values['knowledge_skill_mode'] = skill_mode
    insight_mode = data.get('insight_mode', 'auto')
    if type(insight_mode) is not str or insight_mode not in {'auto', 'off'}:
        raise ValueError('Invalid insight mode')
    # A/B assistance controls belong to request identity. Preserve historical
    # default fingerprints, but never replay auto output for an off request.
    if insight_mode != 'auto':
        values['insight_mode'] = insight_mode
    raw = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_request_reference(session_id: str, request_id: str) -> None:
    if (not isinstance(session_id, str) or not re.fullmatch(r"#[0-9]{5}", session_id)
            or not isinstance(request_id, str) or not re.fullmatch(REQUEST_ID_PATTERN, request_id)):
        raise ValueError("Invalid request reference")


class RequestJournal:
    def __init__(self, directory: Path, *, boot_id: str, clock=time.time,
                 max_storage_bytes: int | None = None):
        self.directory = Path(directory)
        if (not self.directory.is_absolute() or self.directory.is_symlink()
                or self.directory.resolve(strict=True) != self.directory
                or not self.directory.is_dir()):
            raise ValueError("Request journal requires an existing worker directory")
        self.path = self.directory / "request_journal.db"
        self.boot_id, self.clock = boot_id, clock
        if max_storage_bytes is None:
            configured = os.environ.get("DOCTHINKER_REQUEST_JOURNAL_BYTES", str(DEFAULT_JOURNAL_BYTES))
            if not re.fullmatch(r"[0-9]{1,16}", configured):
                raise ValueError("Invalid request journal storage quota")
            max_storage_bytes = int(configured)
        if type(max_storage_bytes) is not int or not 65_536 <= max_storage_bytes <= 1 << 50:
            raise ValueError("Invalid request journal storage quota")
        self.max_storage_bytes = max_storage_bytes

    def _check_file(self) -> bool:
        try:
            info = self.path.lstat()
        except FileNotFoundError:
            return False
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != os.getuid() or info.st_mode & 0o077):
            raise ValueError("Request journal must be an owner-only regular file")
        return True

    def _ensure_file(self) -> None:
        if not self._check_file():
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(descriptor)
            self._check_file()

    @staticmethod
    def _schema(db) -> None:
        db.execute(
            "CREATE TABLE IF NOT EXISTS query_requests ("
            "session_id TEXT NOT NULL, request_id TEXT NOT NULL, fingerprint TEXT NOT NULL, "
            "boot_id TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN ('running','completed','unknown')), "
            "result TEXT, result_bytes INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, "
            "updated_at REAL NOT NULL, PRIMARY KEY(session_id,request_id))"
        )
        # Preserve the old CHECK/table and accepted IDs during upgrades. A
        # bounded outcome code refines the old unknown tombstone into a known
        # terminal failure; old workers still safely refuse to execute it.
        if "failure_code" not in {row[1] for row in db.execute("PRAGMA table_info(query_requests)")}:
            db.execute("ALTER TABLE query_requests ADD COLUMN failure_code TEXT")
        db.execute("CREATE INDEX IF NOT EXISTS query_requests_completed_time ON query_requests(updated_at,request_id) WHERE state='completed'")
        db.execute("CREATE INDEX IF NOT EXISTS query_requests_unresolved_time ON query_requests(session_id,updated_at DESC,request_id DESC) WHERE state IN ('running','unknown')")
        db.execute("CREATE INDEX IF NOT EXISTS query_requests_live_unresolved_time ON query_requests(session_id,updated_at DESC,request_id DESC) WHERE state IN ('running','unknown') AND failure_code IS NULL")
        db.execute("CREATE TABLE IF NOT EXISTS query_request_usage (singleton INTEGER PRIMARY KEY CHECK(singleton=1), result_bytes INTEGER NOT NULL)")
        if not db.execute("SELECT 1 FROM query_request_usage WHERE singleton=1").fetchone():
            # One migration scan for an existing journal, not one scan per
            # answer. Triggers maintain the aggregate inside the same commit.
            db.execute("INSERT INTO query_request_usage SELECT 1,COALESCE(SUM(result_bytes),0) FROM query_requests")
        db.execute("CREATE TRIGGER IF NOT EXISTS query_requests_usage_insert AFTER INSERT ON query_requests BEGIN "
                   "UPDATE query_request_usage SET result_bytes=result_bytes+NEW.result_bytes WHERE singleton=1; END")
        db.execute("CREATE TRIGGER IF NOT EXISTS query_requests_usage_update AFTER UPDATE OF result_bytes ON query_requests BEGIN "
                   "UPDATE query_request_usage SET result_bytes=result_bytes+NEW.result_bytes-OLD.result_bytes WHERE singleton=1; END")
        db.execute("CREATE TRIGGER IF NOT EXISTS query_requests_usage_delete AFTER DELETE ON query_requests BEGIN "
                   "UPDATE query_request_usage SET result_bytes=result_bytes-OLD.result_bytes WHERE singleton=1; END")

    @staticmethod
    def _used_bytes(db) -> int:
        return (db.execute("PRAGMA page_count").fetchone()[0] - db.execute("PRAGMA freelist_count").fetchone()[0]) * db.execute("PRAGMA page_size").fetchone()[0]

    @staticmethod
    def _evict_oldest_result(db) -> bool:
        row = db.execute("SELECT session_id,request_id FROM query_requests WHERE state='completed' "
                         "ORDER BY updated_at,request_id LIMIT 1").fetchone()
        if row is None:
            return False
        db.execute("UPDATE query_requests SET state='unknown',result=NULL,result_bytes=0 WHERE session_id=? AND request_id=?", row)
        return True

    def _make_space(self, db, *, reserve: int = 16_384) -> bool:
        # Reclaim only replay blobs. Accepted IDs/fingerprints survive expiry,
        # quota pressure, process restart and later storage expansion.
        while self._used_bytes(db) + reserve > self.max_storage_bytes:
            if not self._evict_oldest_result(db):
                return False
        return True

    def _public(self, row, *, active: bool) -> dict:
        request_id = row["request_id"]
        if "failure_code" in row.keys() and row["failure_code"] is not None:
            return {"request_id": request_id, "request_status": "failed",
                    "request_error": _request_failure(row["failure_code"])}
        if row["state"] == "completed" and row["result"] is not None:
            if row["updated_at"] > self.clock() - RESULT_RETENTION_SECONDS:
                result = json.loads(row["result"])
                return {**result, "request_id": request_id, "request_status": "completed"}
        running = row["state"] == "running" and row["boot_id"] == self.boot_id and active
        return {"request_id": request_id, "request_status": "running" if running else "unknown"}

    def read(self, session_id: str, request_id: str, *, active: bool = False, fingerprint: str | None = None) -> dict:
        validate_request_reference(session_id, request_id)
        if not self._check_file():
            return {"request_id": request_id, "request_status": "not_found"}
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
            db.row_factory = sqlite3.Row
            # An interrupted initial schema transaction is still no accepted ID.
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='query_requests'").fetchone():
                return {"request_id": request_id, "request_status": "not_found"}
            row = db.execute("SELECT * FROM query_requests WHERE session_id=? AND request_id=?", (session_id, request_id)).fetchone()
            if row and fingerprint is not None and row["fingerprint"] != fingerprint:
                raise RequestConflictError("Request ID belongs to different content")
            return self._public(row, active=active) if row else {"request_id": request_id, "request_status": "not_found"}

    def recent_unresolved(self, session_id: str, *, active_ids: set[str] | None = None) -> list[dict]:
        """At most five recent references for refresh recovery; SELECT-only.

        It is not a resumable task list. Completed answers stay in ordinary
        history; old/evicted receipts cannot permanently block a new session.
        """
        if not isinstance(session_id, str) or not re.fullmatch(r"#[0-9]{5}", session_id):
            raise ValueError("Invalid session reference")
        if not self._check_file():
            return []
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
            db.row_factory = sqlite3.Row
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='query_requests'").fetchone():
                return []
            has_failure = "failure_code" in {row[1] for row in db.execute("PRAGMA table_info(query_requests)")}
            rows = db.execute("SELECT * FROM query_requests WHERE session_id=? AND state IN ('running','unknown') "
                              + ("AND failure_code IS NULL " if has_failure else "") +
                              "AND updated_at>? ORDER BY updated_at DESC,request_id DESC LIMIT 5",
                              (session_id, self.clock() - RECOVERY_WINDOW_SECONDS)).fetchall()
            return [{"request_id": row["request_id"], "request_status": self._public(row, active=row["request_id"] in (active_ids or set()))["request_status"]}
                    for row in rows]

    def begin(self, session_id: str, request_id: str, fingerprint: str) -> bool:
        validate_request_reference(session_id, request_id)
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError("Invalid request fingerprint")
        self._ensure_file()
        with closing(sqlite3.connect(self.path, timeout=2)) as db:
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("BEGIN IMMEDIATE")
            self._schema(db)
            row = db.execute("SELECT fingerprint FROM query_requests WHERE session_id=? AND request_id=?", (session_id, request_id)).fetchone()
            if row:
                if row[0] != fingerprint:
                    raise RequestConflictError("Request ID belongs to different content")
                return False
            if not self._make_space(db):
                raise RequestJournalLimitError("Request journal storage quota reached; expand storage")
            now = self.clock()
            db.execute("INSERT INTO query_requests (session_id,request_id,fingerprint,boot_id,state,result,result_bytes,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                       (session_id, request_id, fingerprint, self.boot_id, "running", None, 0, now, now))
            if self._used_bytes(db) > self.max_storage_bytes:
                raise RequestJournalLimitError("Request journal storage quota reached; expand storage")
            db.commit()
            return True

    def complete(self, session_id: str, request_id: str, result: dict) -> bool:
        validate_request_reference(session_id, request_id)
        if not isinstance(result, dict):
            raise ValueError("Invalid request result")
        raw = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        size = len(raw.encode("utf-8"))
        if size > min(MAX_RESULT_BYTES, MAX_CACHED_BYTES):
            self.mark_unknown(session_id, request_id)
            return False
        self._check_file()
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=2)) as db:
            db.execute("BEGIN IMMEDIATE")
            self._schema(db)
            row = db.execute("SELECT state,boot_id FROM query_requests WHERE session_id=? AND request_id=?", (session_id, request_id)).fetchone()
            if row is None or row != ("running", self.boot_id):
                return False
            # Evict result blobs only, never the accepted ID/fingerprint tombstone.
            db.execute("UPDATE query_requests SET state='unknown',result=NULL,result_bytes=0 "
                       "WHERE state='completed' AND updated_at<=?", (self.clock() - RESULT_RETENTION_SECONDS,))
            while db.execute("SELECT result_bytes FROM query_request_usage WHERE singleton=1").fetchone()[0] + size > MAX_CACHED_BYTES:
                if not self._evict_oldest_result(db):
                    break
            if not self._make_space(db, reserve=size + 16_384):
                db.execute("UPDATE query_requests SET state='unknown',updated_at=? WHERE session_id=? AND request_id=?", (self.clock(), session_id, request_id))
                db.commit()
                return False
            db.execute("SAVEPOINT completed_result")
            changed = db.execute("UPDATE query_requests SET state='completed',result=?,result_bytes=?,updated_at=? "
                                 "WHERE session_id=? AND request_id=? AND state='running' AND boot_id=?",
                                 (raw, size, self.clock(), session_id, request_id, self.boot_id)).rowcount
            if self._used_bytes(db) > self.max_storage_bytes:
                db.execute("ROLLBACK TO completed_result")
                db.execute("UPDATE query_requests SET state='unknown',updated_at=? WHERE session_id=? AND request_id=?", (self.clock(), session_id, request_id))
                changed = 0
            db.execute("RELEASE completed_result")
            db.commit()
            return bool(changed)

    def mark_unknown(self, session_id: str, request_id: str) -> None:
        if not self._check_file():
            return
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=2)) as db:
            db.execute("UPDATE query_requests SET state='unknown',updated_at=? "
                       "WHERE session_id=? AND request_id=? AND state='running' AND boot_id=?",
                       (self.clock(), session_id, request_id, self.boot_id))
            db.commit()

    def fail(self, session_id: str, request_id: str, code: str) -> bool:
        """Keep a terminal safe code, never raw response text or retry authority."""
        validate_request_reference(session_id, request_id)
        _request_failure(code)
        if not self._check_file():
            return False
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=2)) as db:
            db.execute("BEGIN IMMEDIATE")
            self._schema(db)
            changed = db.execute("UPDATE query_requests SET state='unknown',failure_code=?,result=NULL,result_bytes=0,updated_at=? "
                                 "WHERE session_id=? AND request_id=? AND state='running' AND boot_id=?",
                                 (code, self.clock(), session_id, request_id, self.boot_id)).rowcount
            db.commit()
            return bool(changed)
