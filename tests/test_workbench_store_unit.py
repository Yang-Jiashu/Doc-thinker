"""Local user-authored workbench storage; no model, source or execution data."""
from concurrent.futures import ThreadPoolExecutor
import os
import sqlite3
import uuid

import pytest

from docthinker.server import workbench_store as module
from docthinker.server.workbench_store import WorkbenchStore, WorkbenchConflictError, WorkbenchLimitError

SID, OTHER = "#00001", "#00002"
ANSWER = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
BRIEF = {"goal": "比较证据", "output_format": "结论与来源", "constraints": "不补造事实"}


@pytest.fixture
def store(tmp_path):
    return WorkbenchStore(tmp_path.resolve(), clock=lambda: 1234.0)


def test_empty_reads_and_delete_do_not_create_storage(store):
    assert store.get_brief(SID) == {"session_id": SID, "goal": "", "output_format": "", "constraints": "", "revision": 0, "updated_at": None}
    assert store.list_artifacts(SID) == []
    store.delete_artifact(SID, ANSWER)
    assert not store.path.exists()


def test_brief_cas_and_explicit_clear_survive_restart(store):
    saved = store.save_brief(SID, expected_revision=0, **BRIEF)
    assert saved == {"session_id": SID, **BRIEF, "revision": 1, "updated_at": 1234.0}
    assert WorkbenchStore(store.directory).get_brief(SID) == saved
    before = store.path.read_bytes()
    with pytest.raises(WorkbenchConflictError):
        store.save_brief(SID, expected_revision=0, **{**BRIEF, "goal": "stale overwrite"})
    assert store.path.read_bytes() == before
    assert store.get_brief(OTHER)["revision"] == 0
    cleared = store.save_brief(SID, expected_revision=1, goal="", output_format="", constraints="")
    assert cleared["revision"] == 2 and cleared["goal"] == ""


def test_brief_concurrent_cas_has_one_winner(store):
    def save(value):
        try:
            return store.save_brief(SID, expected_revision=0, **{**BRIEF, "goal": value})["goal"]
        except WorkbenchConflictError:
            return "conflict"
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(save, ["a", "b"]))
    assert outcomes.count("conflict") == 1
    assert store.get_brief(SID)["revision"] == 1


def test_artifact_reference_only_idempotent_and_session_scoped(store):
    first = store.save_artifact(SID, ANSWER, "  保留此结论  ")
    before = store.path.read_bytes()
    assert store.save_artifact(SID, ANSWER, "保留此结论") == first
    assert store.path.read_bytes() == before
    assert store.list_artifacts(OTHER) == []
    assert WorkbenchStore(store.directory).list_artifacts(SID) == [first]
    store.save_artifact(OTHER, ANSWER, "另一个项目")
    store.delete_artifact(SID, ANSWER)
    store.delete_artifact(SID, ANSWER)
    assert store.list_artifacts(SID) == []
    assert store.list_artifacts(OTHER)[0]["title"] == "另一个项目"
    with sqlite3.connect(store.path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(workbench_artifacts)")}
    assert columns == {"session_id", "answer_id", "title", "created_at", "updated_at"}
    assert store.path.stat().st_mode & 0o777 == 0o600


def test_worker_roots_with_identical_ids_are_isolated(store, tmp_path):
    other_root = tmp_path / "second-worker"
    other_root.mkdir(mode=0o700)
    other_store = WorkbenchStore(other_root.resolve())
    store.save_artifact(SID, ANSWER, "owner A")
    store.save_brief(SID, expected_revision=0, **BRIEF)
    assert other_store.list_artifacts(SID) == []
    assert other_store.get_brief(SID)["revision"] == 0


def test_artifact_limits_allow_updates_and_delete(store, monkeypatch):
    monkeypatch.setattr(module, "MAX_SESSION_ARTIFACTS", 1)
    monkeypatch.setattr(module, "MAX_WORKER_ARTIFACTS", 2)
    store.save_artifact(SID, ANSWER, "first")
    with pytest.raises(WorkbenchLimitError):
        store.save_artifact(SID, str(uuid.uuid4()), "second")
    store.save_artifact(SID, ANSWER, "renamed")
    store.save_artifact(OTHER, ANSWER, "second session")
    with pytest.raises(WorkbenchLimitError):
        store.save_artifact("#00003", ANSWER, "third session")
    store.delete_artifact(SID, ANSWER)
    store.save_artifact("#00003", ANSWER, "room after deletion")


def test_brief_limit_allows_updates(store, monkeypatch):
    monkeypatch.setattr(module, "MAX_BRIEFS", 1)
    store.save_brief(SID, expected_revision=0, **BRIEF)
    with pytest.raises(WorkbenchLimitError):
        store.save_brief(OTHER, expected_revision=0, **BRIEF)
    assert store.save_brief(SID, expected_revision=1, **BRIEF)["revision"] == 2


def test_stable_generation_hides_old_and_late_writer_rows(store):
    old = WorkbenchStore(store.directory, session_instance_id="old-instance")
    new = WorkbenchStore(store.directory, session_instance_id="new-instance")
    old.save_brief(SID, expected_revision=0, **BRIEF)
    old.save_artifact(SID, ANSWER, "old sensitive title")
    store.delete_session_metadata(SID)
    # A writer that passed session validation before deletion may finish late.
    old.save_brief(SID, expected_revision=0, **BRIEF)
    old.save_artifact(SID, ANSWER, "late old sensitive title")
    assert new.get_brief(SID)["revision"] == 0
    assert new.list_artifacts(SID) == []


def test_real_session_delete_recreate_cleans_all_workbench_generations(tmp_path):
    from docthinker.session_manager import SessionManager
    manager = SessionManager(base_storage_path=tmp_path / "_system", data_root_path=tmp_path / "data", migrate_legacy=False)
    sid = manager.create_session("old project")["id"]
    first = manager.get_session(sid)["instance_id"]
    store = WorkbenchStore(manager.base_storage_path, session_instance_id=first)
    store.save_brief(sid, expected_revision=0, **BRIEF)
    store.save_artifact(sid, ANSWER, "sensitive title")
    assert manager.delete_session(sid)
    recreated = manager.create_session("new project")["id"]
    assert recreated == sid
    second = manager.get_session(sid)["instance_id"]
    assert first != second
    current = WorkbenchStore(manager.base_storage_path, session_instance_id=second)
    assert current.get_brief(sid)["goal"] == ""
    assert current.list_artifacts(sid) == []
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT count(*) FROM workbench_briefs").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM workbench_artifacts").fetchone()[0] == 0


def test_workbench_cleanup_failure_blocks_real_session_deletion(tmp_path, monkeypatch):
    from docthinker.session_manager import SessionManager
    manager = SessionManager(base_storage_path=tmp_path / "_system", data_root_path=tmp_path / "data", migrate_legacy=False)
    sid = manager.create_session("keep on failed cleanup")["id"]
    def fail(*_args):
        raise ValueError("synthetic storage failure")
    monkeypatch.setattr(WorkbenchStore, "delete_session_metadata", fail)
    with pytest.raises(ValueError):
        manager.delete_session(sid)
    assert manager.get_session(sid) is not None


@pytest.mark.parametrize("session", ["#1", "../x", "#００００１", "#00001/", None])
def test_invalid_session_never_creates_data(store, session):
    with pytest.raises(ValueError):
        store.get_brief(session)
    with pytest.raises(ValueError):
        store.save_artifact(session, ANSWER, "invalid")
    assert not store.path.exists()


@pytest.mark.parametrize("changes", [{"goal": "x" * 1201}, {"output_format": "x" * 401},
                                    {"constraints": "x" * 1201}, {"goal": "\ud800"}, {"goal": "\x00"},
                                    {"expected_revision": True}, {"expected_revision": -1}])
def test_invalid_brief_never_creates_data(store, changes):
    with pytest.raises(ValueError):
        store.save_brief(SID, **{**BRIEF, "expected_revision": 0, **changes})
    assert not store.path.exists()


@pytest.mark.parametrize("title", ["", "  ", "x" * 121, "\x00", "\ud800", None])
def test_invalid_artifact_never_creates_data(store, title):
    with pytest.raises(ValueError):
        store.save_artifact(SID, ANSWER, title)
    assert not store.path.exists()


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "public"])
def test_storage_link_and_permission_attacks_fail_closed(store, tmp_path, kind):
    target = tmp_path / "target"
    target.touch(mode=0o600)
    if kind == "symlink":
        store.path.symlink_to(target)
    elif kind == "hardlink":
        os.link(target, store.path)
    else:
        store.path.touch(mode=0o644)
        store.path.chmod(0o644)
    with pytest.raises(ValueError):
        store.get_brief(SID)
    with pytest.raises(ValueError):
        store.save_artifact(SID, ANSWER, "unsafe")
    assert target.read_bytes() == b""
