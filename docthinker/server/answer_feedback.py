"""Bounded human feedback, separate from memory, training and model clients.

The caller must resolve an actual assistant turn in the worker's own session
before writing. Only the turn reference and the user's explicit feedback are
stored; answers, questions and credentials are never copied into this store.
"""
from __future__ import annotations

from contextlib import closing
import os
from pathlib import Path
import sqlite3
import stat
import time

MAX_SESSION_FEEDBACK = 500
MAX_WORKER_FEEDBACK = 2000


class FeedbackLimitError(ValueError):
    pass


class AnswerFeedbackStore:
    def __init__(self, directory: Path, *, clock=time.time):
        self.directory = Path(directory)
        self.clock = clock
        # Never infer or fall back to the desktop/user's default storage.
        if (not self.directory.is_absolute() or self.directory.is_symlink()
                or self.directory.resolve(strict=True) != self.directory
                or not self.directory.is_dir()):
            raise ValueError("Feedback requires an existing absolute worker directory")
        self.path = self.directory / "answer_feedback.db"

    def _check_file(self) -> bool:
        try:
            info = self.path.lstat()
        except FileNotFoundError:
            return False
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != os.getuid() or info.st_mode & 0o077):
            raise ValueError("Feedback storage must be an owner-only regular file")
        return True

    def _ensure_file(self) -> None:
        if not self._check_file():
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(fd)
            self._check_file()

    def list_for_session(self, session_id: str) -> list[dict]:
        # Reads do not initialize a database, schema, provider or memory.
        if not self._check_file():
            return []
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT answer_id,rating,note,updated_at FROM answer_feedback "
                "WHERE session_id=? ORDER BY updated_at DESC,answer_id ASC LIMIT ?",
                (session_id, MAX_SESSION_FEEDBACK),
            ).fetchall()
            return [dict(row) for row in rows]

    def put(self, session_id: str, answer_id: str, rating: str, note: str) -> dict:
        self._ensure_file()
        with closing(sqlite3.connect(self.path, timeout=2)) as db:
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS answer_feedback ("
                "session_id TEXT NOT NULL, answer_id TEXT NOT NULL, "
                "rating TEXT NOT NULL CHECK(rating IN ('helpful','inaccurate')), "
                "note TEXT NOT NULL CHECK(length(note)<=500), "
                "updated_at REAL NOT NULL, PRIMARY KEY(session_id,answer_id))"
            )
            previous = db.execute(
                "SELECT 1 FROM answer_feedback WHERE session_id=? AND answer_id=?",
                (session_id, answer_id),
            ).fetchone()
            if not previous:
                total = db.execute("SELECT count(*) FROM answer_feedback").fetchone()[0]
                session_total = db.execute(
                    "SELECT count(*) FROM answer_feedback WHERE session_id=?", (session_id,),
                ).fetchone()[0]
                if total >= MAX_WORKER_FEEDBACK or session_total >= MAX_SESSION_FEEDBACK:
                    raise FeedbackLimitError("Feedback storage limit reached")
            updated_at = self.clock()
            db.execute(
                "INSERT INTO answer_feedback VALUES (?,?,?,?,?) "
                "ON CONFLICT(session_id,answer_id) DO UPDATE SET "
                "rating=excluded.rating,note=excluded.note,updated_at=excluded.updated_at",
                (session_id, answer_id, rating, note, updated_at),
            )
            db.commit()
        return {"answer_id": answer_id, "rating": rating, "note": note, "updated_at": updated_at}
