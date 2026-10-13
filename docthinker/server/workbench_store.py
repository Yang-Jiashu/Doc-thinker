"""Bounded user-authored briefs and answer bookmarks, not an agent queue.

Answers, sources, provider credentials and hidden prompts are never copied.
The router must resolve every answer against this worker's current session
history and rehydrate its provenance before returning it.
"""
from __future__ import annotations

from contextlib import closing
import os
from pathlib import Path
import re
import sqlite3
import stat
import time

MAX_BRIEFS = 100
MAX_SESSION_ARTIFACTS = 100
MAX_WORKER_ARTIFACTS = 1000
BRIEF_LIMITS = {"goal": 1200, "output_format": 400, "constraints": 1200}
ANSWER_ID = r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"


class WorkbenchConflictError(ValueError):
    pass


class WorkbenchLimitError(ValueError):
    pass


def _session(value):
    if not isinstance(value, str) or not re.fullmatch(r"#[0-9]{5}", value):
        raise ValueError("Invalid session reference")


def _answer(value):
    if not isinstance(value, str) or not re.fullmatch(ANSWER_ID, value):
        raise ValueError("Invalid answer reference")


def _text(value, limit, *, required=False):
    if (not isinstance(value, str) or len(value) > limit or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", value)
            or (required and not value.strip())):
        raise ValueError("Invalid workbench text")
    value.encode("utf-8")


class WorkbenchStore:
    def __init__(self, directory: Path, *, clock=time.time, session_instance_id=None):
        self.directory = Path(directory)
        if (not self.directory.is_absolute() or self.directory.is_symlink()
                or self.directory.resolve(strict=True) != self.directory
                or not self.directory.is_dir() or self.directory.stat().st_uid != os.getuid()):
            raise ValueError("Workbench requires an existing owned worker directory")
        self.path, self.clock = self.directory / "workbench.db", clock
        if session_instance_id is not None and (not isinstance(session_instance_id, str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_instance_id)):
            raise ValueError("Invalid stable session identity")
        self.session_instance_id = session_instance_id

    def _key(self, session_id):
        _session(session_id)
        # Human-readable session numbers are reused after deletion. Bind new
        # route writes to the immutable knowledge-base instance as well.
        return session_id + ":" + self.session_instance_id if self.session_instance_id else session_id

    def _check_file(self):
        try:
            info = self.path.lstat()
        except FileNotFoundError:
            return False
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != os.getuid() or info.st_mode & 0o077):
            raise ValueError("Workbench requires owner-only regular storage")
        return True

    def _ensure_file(self):
        if not self._check_file():
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(fd)
            self._check_file()

    @staticmethod
    def _schema(db):
        db.execute("CREATE TABLE IF NOT EXISTS workbench_briefs ("
                   "session_id TEXT PRIMARY KEY, goal TEXT NOT NULL, output_format TEXT NOT NULL, "
                   "constraints_text TEXT NOT NULL, revision INTEGER NOT NULL, updated_at REAL NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS workbench_artifacts ("
                   "session_id TEXT NOT NULL, answer_id TEXT NOT NULL, title TEXT NOT NULL, "
                   "created_at REAL NOT NULL, updated_at REAL NOT NULL, PRIMARY KEY(session_id,answer_id))")

    @staticmethod
    def _has_table(db, table):
        return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()

    @staticmethod
    def _brief(row, session_id):
        if row is None:
            return {"session_id": session_id, **dict.fromkeys(BRIEF_LIMITS, ""), "revision": 0, "updated_at": None}
        return {"session_id": session_id, "goal": row["goal"], "output_format": row["output_format"],
                "constraints": row["constraints_text"], "revision": row["revision"], "updated_at": row["updated_at"]}

    def get_brief(self, session_id):
        key = self._key(session_id)
        if not self._check_file():
            return self._brief(None, session_id)
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
            db.row_factory = sqlite3.Row
            if not self._has_table(db, "workbench_briefs"):
                return self._brief(None, session_id)
            row = db.execute("SELECT * FROM workbench_briefs WHERE session_id=?", (key,)).fetchone()
            return self._brief(row, session_id)

    def save_brief(self, session_id, *, expected_revision, goal, output_format, constraints):
        key = self._key(session_id)
        if type(expected_revision) is not int or not 0 <= expected_revision < 2_147_483_647:
            raise ValueError("Invalid brief revision")
        for field_name, value in (("goal", goal), ("output_format", output_format), ("constraints", constraints)):
            _text(value, BRIEF_LIMITS[field_name])
        self._ensure_file()
        with closing(sqlite3.connect(self.path, timeout=2)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("BEGIN IMMEDIATE")
            self._schema(db)
            row = db.execute("SELECT * FROM workbench_briefs WHERE session_id=?", (key,)).fetchone()
            revision = row["revision"] if row else 0
            if revision != expected_revision:
                raise WorkbenchConflictError("Brief has changed; reload it before saving")
            if row is None and db.execute("SELECT count(*) FROM workbench_briefs").fetchone()[0] >= MAX_BRIEFS:
                raise WorkbenchLimitError("Workbench brief capacity reached")
            now = self.clock()
            db.execute("INSERT INTO workbench_briefs VALUES (?,?,?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET "
                       "goal=excluded.goal,output_format=excluded.output_format,constraints_text=excluded.constraints_text,"
                       "revision=excluded.revision,updated_at=excluded.updated_at",
                       (key, goal, output_format, constraints, revision + 1, now))
            db.commit()
            return {"session_id": session_id, "goal": goal, "output_format": output_format,
                    "constraints": constraints, "revision": revision + 1, "updated_at": now}

    def list_artifacts(self, session_id):
        key = self._key(session_id)
        if not self._check_file():
            return []
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
            db.row_factory = sqlite3.Row
            if not self._has_table(db, "workbench_artifacts"):
                return []
            return [dict(row) for row in db.execute(
                "SELECT answer_id,title,created_at,updated_at FROM workbench_artifacts WHERE session_id=? "
                "ORDER BY updated_at DESC,answer_id LIMIT ?", (key, MAX_SESSION_ARTIFACTS)).fetchall()]

    def save_artifact(self, session_id, answer_id, title):
        key = self._key(session_id)
        _answer(answer_id)
        _text(title, 120, required=True)
        title = title.strip()
        self._ensure_file()
        with closing(sqlite3.connect(self.path, timeout=2)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("BEGIN IMMEDIATE")
            self._schema(db)
            previous = db.execute("SELECT answer_id,title,created_at,updated_at FROM workbench_artifacts "
                                  "WHERE session_id=? AND answer_id=?", (key, answer_id)).fetchone()
            if previous and previous["title"] == title:
                return dict(previous)
            if previous is None:
                count = db.execute("SELECT count(*) FROM workbench_artifacts").fetchone()[0]
                session_count = db.execute("SELECT count(*) FROM workbench_artifacts WHERE session_id=?", (key,)).fetchone()[0]
                if count >= MAX_WORKER_ARTIFACTS or session_count >= MAX_SESSION_ARTIFACTS:
                    raise WorkbenchLimitError("Workbench artifact capacity reached")
            now = self.clock()
            created = previous["created_at"] if previous else now
            db.execute("INSERT INTO workbench_artifacts VALUES (?,?,?,?,?) ON CONFLICT(session_id,answer_id) "
                       "DO UPDATE SET title=excluded.title,updated_at=excluded.updated_at",
                       (key, answer_id, title, created, now))
            db.commit()
            return {"answer_id": answer_id, "title": title, "created_at": created, "updated_at": now}

    def delete_artifact(self, session_id, answer_id):
        key = self._key(session_id)
        _answer(answer_id)
        if not self._check_file():
            return
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=2)) as db:
            if self._has_table(db, "workbench_artifacts"):
                db.execute("DELETE FROM workbench_artifacts WHERE session_id=? AND answer_id=?", (key, answer_id))
                db.commit()

    def delete_session_metadata(self, session_id):
        """Remove every generation before the parent permits number reuse.

        Failure propagates so session deletion cannot silently leave metadata.
        Generation-scoped reads additionally close a concurrent late-writer race.
        """
        _session(session_id)
        if not self._check_file():
            return
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=2)) as db:
            db.execute("BEGIN IMMEDIATE")
            for table in ("workbench_briefs", "workbench_artifacts"):
                if self._has_table(db, table):
                    db.execute("DELETE FROM " + table + " WHERE session_id=? OR session_id LIKE ?",
                               (session_id, session_id + ":%"))
            db.commit()
