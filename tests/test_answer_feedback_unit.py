"""Offline durability, bounds and isolation for explicit human answer feedback."""
import os
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from docthinker.server.answer_feedback import AnswerFeedbackStore, FeedbackLimitError

ANSWER = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


def test_read_without_feedback_creates_nothing(tmp_path):
    store = AnswerFeedbackStore(tmp_path.resolve())
    assert store.list_for_session("#00001") == []
    assert not list(tmp_path.iterdir())


def test_explicit_feedback_is_durable_upsert_and_does_not_copy_answers(tmp_path):
    store = AnswerFeedbackStore(tmp_path.resolve(), clock=lambda: 123.5)
    first = store.put("#00001", ANSWER, "helpful", "我的明确评价")
    assert first == {"answer_id": ANSWER, "rating": "helpful", "note": "我的明确评价", "updated_at": 123.5}
    second = store.put("#00001", ANSWER, "inaccurate", "我修改了评价")
    assert AnswerFeedbackStore(tmp_path.resolve()).list_for_session("#00001") == [second]
    assert store.path.stat().st_mode & 0o777 == 0o600
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT count(*) FROM answer_feedback").fetchone()[0] == 1
        columns = [row[1] for row in db.execute("PRAGMA table_info(answer_feedback)")]
    assert set(columns) == {"session_id", "answer_id", "rating", "note", "updated_at"}


def test_identical_session_and_answer_are_separate_between_worker_directories(tmp_path):
    alice = tmp_path / "alice"
    bobby = tmp_path / "bobby"
    alice.mkdir()
    bobby.mkdir()
    a, b = AnswerFeedbackStore(alice.resolve()), AnswerFeedbackStore(bobby.resolve())
    a.put("#00001", ANSWER, "helpful", "Alice only")
    b.put("#00001", ANSWER, "inaccurate", "Bobby only")
    assert a.list_for_session("#00001")[0]["note"] == "Alice only"
    assert b.list_for_session("#00001")[0]["note"] == "Bobby only"
    assert a.list_for_session("#00002") == []


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "public", "directory"])
def test_untrusted_feedback_files_are_rejected_without_writes(tmp_path, kind):
    target = tmp_path / "target"
    target.write_text("synthetic unrelated data")
    target.chmod(0o600)
    db = tmp_path / "answer_feedback.db"
    if kind == "symlink":
        db.symlink_to(target)
    elif kind == "hardlink":
        os.link(target, db)
    elif kind == "public":
        db.write_text("synthetic")
        db.chmod(0o644)
    else:
        db.mkdir()
    store = AnswerFeedbackStore(tmp_path.resolve())
    with pytest.raises(ValueError):
        store.put("#00001", ANSWER, "helpful", "")
    with pytest.raises(ValueError):
        store.list_for_session("#00001")
    assert target.read_text() == "synthetic unrelated data"


def test_store_never_falls_back_or_resolves_symlink_directory(tmp_path):
    with pytest.raises(ValueError):
        AnswerFeedbackStore(Path("relative"))
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        AnswerFeedbackStore(link)


def test_session_bound_rejects_new_rows_but_allows_existing_update(tmp_path, monkeypatch):
    monkeypatch.setattr("docthinker.server.answer_feedback.MAX_SESSION_FEEDBACK", 2)
    store = AnswerFeedbackStore(tmp_path.resolve())
    store.put("#00001", "a", "helpful", "")
    store.put("#00001", "b", "helpful", "")
    with pytest.raises(FeedbackLimitError):
        store.put("#00001", "c", "helpful", "")
    store.put("#00001", "a", "inaccurate", "updated at capacity")
    assert len(store.list_for_session("#00001")) == 2
    assert any(row["note"] == "updated at capacity" for row in store.list_for_session("#00001"))


def test_worker_bound_is_atomic_under_concurrent_writes(tmp_path, monkeypatch):
    monkeypatch.setattr("docthinker.server.answer_feedback.MAX_WORKER_FEEDBACK", 2)
    store = AnswerFeedbackStore(tmp_path.resolve())
    def attempt(index):
        try:
            store.put(f"#{index:05}", str(index), "helpful", "")
            return True
        except FeedbackLimitError:
            return False
    with ThreadPoolExecutor(max_workers=5) as pool:
        assert sum(pool.map(attempt, range(1, 6))) == 2
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT count(*) FROM answer_feedback").fetchone()[0] == 2


@pytest.mark.parametrize("rating,note", [("train", ""), ("helpful", "a" * 501)])
def test_database_bounds_are_defense_in_depth(tmp_path, rating, note):
    with pytest.raises(sqlite3.IntegrityError):
        AnswerFeedbackStore(tmp_path.resolve()).put("#00001", ANSWER, rating, note)
