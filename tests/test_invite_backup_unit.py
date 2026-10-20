"""Cold backup tests use generated data only; no deployment keys or Docker."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import stat

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("invite_backup_tests", ROOT / "deploy/invite/backup.py")
backup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backup)


@pytest.fixture
def data(tmp_path):
    source = tmp_path.resolve() / "source"
    source.mkdir(mode=0o700)
    with sqlite3.connect(source / "invite.sqlite3") as connection:
        connection.execute("CREATE TABLE samples (value TEXT)")
        connection.execute("INSERT INTO samples VALUES ('synthetic')")
    (source / "documents").mkdir(mode=0o700)
    (source / "empty").mkdir(mode=0o700)
    (source / "documents" / "paper.txt").write_text("synthetic paper", encoding="utf-8")
    return source


def test_backup_verify_restore_private_and_complete(data, tmp_path):
    destination, restored = tmp_path / "backup", tmp_path / "restored"
    report = backup.create_backup(data, destination)
    assert report["ok"] and not report["encrypted"] and not report["docker_restore_verified"]
    assert report["files"] == 3
    assert backup.verify_backup(destination)["ok"]
    assert backup.restore_backup(destination, restored)["ok"]
    assert (restored / "documents/paper.txt").read_text() == "synthetic paper"
    assert (restored / "empty").is_dir()
    with sqlite3.connect(restored / "invite.sqlite3") as connection:
        assert connection.execute("SELECT * FROM samples").fetchall() == [("synthetic",)]
    for root in (destination, restored):
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        for path in root.rglob("*"):
            assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)


def test_never_overwrites_existing_directory(data, tmp_path):
    destination = tmp_path / "backup"
    backup.create_backup(data, destination)
    with pytest.raises(backup.BackupError, match="already exists"):
        backup.create_backup(data, destination)
    with pytest.raises(backup.BackupError, match="already exists"):
        backup.restore_backup(destination, data)
    assert (data / "documents/paper.txt").read_text() == "synthetic paper"


@pytest.mark.parametrize("change", ["tamper", "extra_file", "extra_directory", "missing_file", "missing_directory", "extra_root"])
def test_corruption_or_unlisted_data_blocks_restore(data, tmp_path, change):
    destination, restored = tmp_path / "backup", tmp_path / "restored"
    backup.create_backup(data, destination)
    paper = destination / "data/documents/paper.txt"
    if change == "tamper":
        paper.write_text("not original")
    elif change == "extra_file":
        (destination / "data/unknown").touch()
    elif change == "extra_directory":
        (destination / "data/unlisted").mkdir()
    elif change == "missing_file":
        paper.unlink()
    elif change == "missing_directory":
        (destination / "data/empty").rmdir()
    else:
        (destination / "extra").touch()
    with pytest.raises(backup.BackupError):
        backup.restore_backup(destination, restored)
    assert not restored.exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../escape", "a//b", "a\\b"])
def test_manifest_path_traversal_rejected(data, tmp_path, name):
    destination = tmp_path / "backup"
    backup.create_backup(data, destination)
    manifest_path = destination / backup.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][name] = {"size": 0, "sha256": "0" * 64}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(backup.BackupError, match="paths"):
        backup.restore_backup(destination, tmp_path / "restored")


@pytest.mark.parametrize("kind", ["file_symlink", "directory_symlink", "hardlink"])
def test_source_links_rejected(data, tmp_path, kind):
    if kind == "file_symlink":
        (data / "link").symlink_to(data / "invite.sqlite3")
    elif kind == "directory_symlink":
        (data / "link").symlink_to(data / "documents", target_is_directory=True)
    else:
        os.link(data / "invite.sqlite3", data / "hardlink")
    with pytest.raises(backup.BackupError):
        backup.create_backup(data, tmp_path / "backup")
    assert not (tmp_path / "backup").exists()


def test_running_supervisor_blocks_backup(data, tmp_path):
    descriptor = os.open(data / "workers.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(backup.BackupError, match="Workers are running"):
            backup.create_backup(data, tmp_path / "backup")
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("suffix", ["-wal", "-journal"])
def test_uncheckpointed_sqlite_blocks_cold_backup(data, tmp_path, suffix):
    (data / ("invite.sqlite3" + suffix)).write_bytes(b"pending transaction")
    with pytest.raises(backup.BackupError, match="Uncheckpointed"):
        backup.create_backup(data, tmp_path / "backup")


@pytest.mark.parametrize("body", [b"", b"not a SQLite database"])
def test_invalid_database_does_not_produce_successful_backup(data, tmp_path, body):
    (data / "invite.sqlite3").write_bytes(body)
    with pytest.raises(backup.BackupError, match="SQLite integrity"):
        backup.create_backup(data, tmp_path / "backup")
    assert not (tmp_path / "backup").exists()


def test_modification_during_snapshot_rejected_and_cleaned(data, tmp_path, monkeypatch):
    original = backup._transfer
    def changing_transfer(source, identity, destination=None):
        result = original(source, identity, destination)
        if source.name == "paper.txt":
            source.write_text("changed while backing up")
        return result
    monkeypatch.setattr(backup, "_transfer", changing_transfer)
    with pytest.raises(backup.BackupError, match="tree changed"):
        backup.create_backup(data, tmp_path / "backup")
    assert not (tmp_path / "backup").exists()


def test_private_output_parent_required(data, tmp_path):
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    with pytest.raises(backup.BackupError, match="owner-only"):
        backup.create_backup(data, public / "backup")


def test_limits_prevent_partial_backup(data, tmp_path, monkeypatch):
    monkeypatch.setattr(backup, "MAX_FILES", 1)
    with pytest.raises(backup.BackupError, match="limit"):
        backup.create_backup(data, tmp_path / "backup")
    assert not (tmp_path / "backup").exists()


def test_offline_self_test_is_truthful():
    report = backup.self_test()
    assert report["ok"] and report["synthetic_data_only"]
    assert not report["deployment_data_read"] and not report["docker_restore_verified"]
    assert len(report["checks"]) == 5
