"""Synthetic owner-only lifecycle checks; never touches real users or providers."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import zipfile

import pytest

from docthinker.server import invite_account_lifecycle as lifecycle
from docthinker.server.invite_store import InviteError, InviteStore


def private_file(path, text):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Synthetic fixture only: parent hierarchy must match private worker layout.
    path.write_text(text)
    path.chmod(0o600)


@pytest.fixture
def account(tmp_path):
    old = os.umask(0o077)
    try:
        store = InviteStore.initialize((tmp_path / "registry").resolve())
        invitation = store.add_user("alice", 8101)
        auth = store.register(invitation["invite_code"], "alice", "synthetic-password-123")
        binding = store.list_worker_bindings()[0]
        worker = store.root / "workers" / binding["id"]
        private_file(worker / "data" / "_system" / "history.txt", "User conversation and preference")
        private_file(worker / "_system" / "research" / "files" / "paper.txt", "Original complete paper text")
        private_file(worker / "runtime" / "private.env", "worker-runtime-secret")
        private_file(worker / "home" / "cache.txt", "private-home-cache")
        private_file(worker / "tmp" / "temp.txt", "temporary-parser-data")
        private_file(worker / "cache" / "cached.txt", "model-cache-data")
        store.add_user("bravo", 8102)
        other = next(row for row in store.list_worker_bindings() if row["username"] == "bravo")
        private_file(store.root / "workers" / other["id"] / "data" / "other.txt", "must-not-export-or-delete")
        output = tmp_path / "exports"
        output.mkdir(mode=0o700)
        store.revoke_user("alice")
        yield store, binding, worker, output, invitation, auth, other
    finally:
        os.umask(old)


def test_export_is_complete_private_business_archive_without_registry_credentials(account):
    store, binding, worker, output, invitation, auth, other = account
    target = output / "alice.zip"
    report = lifecycle.export_account(store, "alice", "alice", target)
    assert report["status"] == "exported" and report["files"] == 2
    assert target.stat().st_mode & 0o777 == 0o600
    with zipfile.ZipFile(target) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        metadata = json.loads(archive.read("account.json"))
        assert set(metadata) == {"id", "username", "status", "created_at"}
        assert metadata["id"] == binding["id"]
        for name, item in manifest["files"].items():
            content = archive.read("files/" + name)
            assert content == (worker / name).read_bytes()
            assert item == {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    raw = target.read_bytes()
    for secret in (binding["upstream_token"], invitation["invite_code"], auth["access_token"],
                   "synthetic-password-123", "worker-runtime-secret", "private-home-cache", other["id"]):
        assert secret.encode() not in raw
    assert worker.exists()


def test_cleanup_preserves_tombstone_and_other_account_and_is_not_secure_erase(account):
    store, binding, worker, output, invitation, auth, other = account
    store.reserve_action(other["id"], "query") if other["status"] == "active" else None
    report = lifecycle.purge_account(store, "alice", "alice")
    assert report["status"] == "retained_data_removed" and not report["secure_erase"]
    assert "backups" in report["not_removed"] and "username" in report["retained_tombstone"]
    assert not worker.exists()
    assert (store.root / "workers" / other["id"] / "data" / "other.txt").read_text() == "must-not-export-or-delete"
    assert (store.root / "account-locks" / (binding["id"] + ".lock")).exists()
    with store._connect() as db:
        row = dict(db.execute("SELECT * FROM users WHERE id=?", (binding["id"],)).fetchone())
        assert row["status"] == "revoked" and row["username"] == "alice" and row["upstream_url"] == binding["upstream_url"]
        assert row["password_hash"] is None and row["upstream_token"] == "" and row["created_at"] == 0
        for table in ("invitations", "auth_sessions", "usage", "query_reservations"):
            assert db.execute(f"SELECT count(*) FROM {table} WHERE user_id=?", (binding["id"],)).fetchone()[0] == 0
    with pytest.raises(InviteError, match="already complete"):
        lifecycle.purge_account(store, "alice", "alice")
    with pytest.raises(InviteError, match="reserved"):
        store.add_user("new-user", 8101)
    with pytest.raises(InviteError, match="reserved"):
        store.add_user("alice", 8103)
    assert [row["username"] for row in store.list_worker_bindings()] == ["bravo"]
    with pytest.raises(InviteError):
        store.authenticate(auth["access_token"])


@pytest.mark.parametrize("operation", ["export", "purge"])
@pytest.mark.parametrize("case", ["active", "pending", "missing", "legacy", "confirmation"])
def test_unsafe_account_states_or_wrong_confirmation_cannot_touch_files(account, operation, case):
    store, binding, worker, output, *_ = account
    username, confirmation = "alice", "alice"
    with store._connect() as db:
        if case in {"active", "pending"}:
            db.execute("UPDATE users SET status=? WHERE id=?", (case, binding["id"]))
        elif case == "legacy":
            db.execute("DELETE FROM account_lifecycle WHERE user_id=?", (binding["id"],))
        elif case == "missing":
            username = confirmation = "missing"
        else:
            confirmation = "bravo"
    with pytest.raises(InviteError):
        if operation == "export":
            lifecycle.export_account(store, username, confirmation, output / "archive.zip")
        else:
            lifecycle.purge_account(store, username, confirmation)
    assert (worker / "data" / "_system" / "history.txt").exists()
    assert not (output / "archive.zip").exists()


@pytest.mark.parametrize("operation", ["export", "purge"])
@pytest.mark.parametrize("kind", ["worker_lock", "runtime_lock", "symlink", "hardlink", "special", "unsafe_directory"])
def test_active_locks_or_unsafe_files_fail_closed(account, operation, kind):
    store, binding, worker, output, *_ = account
    from contextlib import nullcontext
    context = nullcontext()
    if kind == "worker_lock":
        context = lifecycle.worker_lifetime_lock(store.root, binding["id"])
    elif kind == "runtime_lock":
        from docthinker.server.invite_workers import supervisor_lock
        context = supervisor_lock(store.root)
    elif kind == "symlink":
        (worker / "data" / "link").symlink_to(output)
    elif kind == "hardlink":
        os.link(worker / "data" / "_system" / "history.txt", worker / "data" / "hardlink")
    elif kind == "special":
        os.mkfifo(worker / "data" / "fifo", 0o600)
    else:
        (worker / "data").chmod(0o755)
    with context, pytest.raises((InviteError, OSError)):
        if operation == "export":
            lifecycle.export_account(store, "alice", "alice", output / "archive.zip")
        else:
            lifecycle.purge_account(store, "alice", "alice")
    assert (worker / "data" / "_system" / "history.txt").exists()
    assert not (output / "archive.zip").exists()


def test_failed_partial_cleanup_can_resume_but_cannot_export_or_start_worker(account, monkeypatch):
    store, binding, worker, output, *_ = account
    original = lifecycle._remove_tree
    def fail_after_one(fd, entries, prefix=""):
        descriptor = os.open(worker / "data" / "_system", os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.unlink("history.txt", dir_fd=descriptor)
        finally:
            os.close(descriptor)
        raise OSError("private filesystem detail must not escape")
    monkeypatch.setattr(lifecycle, "_remove_tree", fail_after_one)
    with pytest.raises(InviteError, match="incomplete") as error:
        lifecycle.purge_account(store, "alice", "alice")
    assert "private filesystem" not in str(error.value)
    with store._connect() as db:
        assert db.execute("SELECT state FROM account_lifecycle WHERE user_id=?", (binding["id"],)).fetchone()[0] == "purging"
    with pytest.raises(InviteError, match="complete account export"):
        lifecycle.export_account(store, "alice", "alice", output / "archive.zip")
    with pytest.raises(InviteError, match="no longer authorized"):
        store.authorize_worker(binding["id"], 8101, binding["upstream_token"])
    monkeypatch.setattr(lifecycle, "_remove_tree", original)
    assert lifecycle.purge_account(store, "alice", "alice")["status"] == "retained_data_removed"


def test_mutating_file_export_is_removed_not_claimed_complete(account, monkeypatch):
    store, binding, worker, output, *_ = account
    original = lifecycle._read_copy
    def changed(parent, name, expected, target):
        result = original(parent, name, expected, target)
        private_file(worker / "data" / "new.txt", "concurrent mutation")
        return result
    monkeypatch.setattr(lifecycle, "_read_copy", changed)
    with pytest.raises(InviteError, match="changed"):
        lifecycle.export_account(store, "alice", "alice", output / "archive.zip")
    assert not (output / "archive.zip").exists()


def test_final_database_failure_leaves_recoverable_purging_not_fake_success(account):
    store, binding, worker, output, *_ = account
    with store._connect() as db:
        db.execute("CREATE TRIGGER fail_finish BEFORE UPDATE OF state ON account_lifecycle "
                   "WHEN NEW.state='purged' BEGIN SELECT RAISE(ABORT, 'private-db-detail'); END")
    with pytest.raises(InviteError, match="incomplete") as error:
        lifecycle.purge_account(store, "alice", "alice")
    assert "private-db-detail" not in str(error.value)
    assert not worker.exists()
    with store._connect() as db:
        assert db.execute("SELECT state FROM account_lifecycle WHERE user_id=?", (binding["id"],)).fetchone()[0] == "purging"
        db.execute("DROP TRIGGER fail_finish")
    assert lifecycle.purge_account(store, "alice", "alice")["status"] == "retained_data_removed"


def test_initial_database_failure_never_starts_filesystem_cleanup(account):
    store, binding, worker, output, *_ = account
    with store._connect() as db:
        db.execute("CREATE TRIGGER fail_begin BEFORE UPDATE OF state ON account_lifecycle "
                   "WHEN NEW.state='purging' BEGIN SELECT RAISE(ABORT, 'blocked'); END")
    with pytest.raises(sqlite3.IntegrityError):
        lifecycle.purge_account(store, "alice", "alice")
    assert (worker / "data" / "_system" / "history.txt").exists()
    with store._connect() as db:
        assert db.execute("SELECT state FROM account_lifecycle WHERE user_id=?", (binding["id"],)).fetchone()[0] == "retained"


@pytest.mark.parametrize("case", ["existing", "symlink", "inside_registry", "public_parent", "unknown_durable_root"])
def test_export_destinations_and_layout_cannot_overwrite_or_silently_omit(account, case):
    store, binding, worker, output, *_ = account
    target = output / "archive.zip"
    if case == "existing":
        private_file(target, "must-not-overwrite")
    elif case == "symlink":
        target.symlink_to(worker / "data" / "_system" / "history.txt")
    elif case == "inside_registry":
        target = store.root / "archive.zip"
    elif case == "public_parent":
        output.chmod(0o755)
    else:
        private_file(worker / "unknown-business-record", "must-not-silently-omit")
    with pytest.raises((InviteError, OSError)):
        lifecycle.export_account(store, "alice", "alice", target)
    assert (worker / "data" / "_system" / "history.txt").exists()
    if case == "existing":
        assert target.read_text() == "must-not-overwrite"
    elif case != "symlink":
        assert not target.exists()


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory_symlink"])
def test_lock_paths_cannot_be_links(account, kind):
    store, binding, worker, output, *_ = account
    locks = store.root / "account-locks"
    if kind == "directory_symlink":
        locks.symlink_to(output)
    else:
        locks.mkdir(mode=0o700)
        private_file(output / "lock-target", "")
        if kind == "symlink":
            (locks / (binding["id"] + ".lock")).symlink_to(output / "lock-target")
        else:
            os.link(output / "lock-target", locks / (binding["id"] + ".lock"))
    with pytest.raises((InviteError, OSError)):
        lifecycle.purge_account(store, "alice", "alice")
    assert worker.exists()


def test_live_child_lock_blocks_cleanup_even_without_supervisor_lock(account):
    store, binding, worker, output, *_ = account
    source = "from docthinker.server.invite_account_lifecycle import worker_lifetime_lock; import sys\nwith worker_lifetime_lock(sys.argv[1],sys.argv[2]):\n print('locked',flush=True)\n sys.stdin.read()\n"
    process = subprocess.Popen([sys.executable, "-c", source, str(store.root), binding["id"]],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(InviteError, match="still running"):
            lifecycle.purge_account(store, "alice", "alice")
        assert worker.exists()
    finally:
        process.communicate(timeout=5)
        if process.poll() is None:
            process.kill()
    assert lifecycle.purge_account(store, "alice", "alice")["status"] == "retained_data_removed"


def test_two_administrators_cannot_cleanup_same_account_concurrently(account, monkeypatch):
    store, binding, worker, output, *_ = account
    original = lifecycle._remove_tree
    entered, release = threading.Event(), threading.Event()
    def pause(fd, entries, prefix=""):
        if not prefix:
            entered.set()
            assert release.wait(5)
        return original(fd, entries, prefix)
    monkeypatch.setattr(lifecycle, "_remove_tree", pause)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lifecycle.purge_account, store, "alice", "alice")
        assert entered.wait(5)
        try:
            with pytest.raises(InviteError, match="still running"):
                lifecycle.purge_account(store, "alice", "alice")
            with pytest.raises(InviteError, match="still running"):
                with lifecycle.worker_lifetime_lock(store.root, binding["id"]):
                    pytest.fail("worker could start during deletion")
        finally:
            release.set()
        assert future.result()["status"] == "retained_data_removed"


def test_admin_cli_uses_explicit_confirmation_and_never_emits_content_or_credentials(account, monkeypatch, capsys):
    from docthinker.server.invite_admin import main
    store, binding, worker, output, invitation, auth, other = account
    monkeypatch.setenv("DOCTHINKER_INVITE_DATA_DIR", str(store.root))
    assert main(["export-account", "--username", "alice", "--confirm-username", "alice", "--output", str(output / "archive.zip")]) == 0
    assert main(["purge-account", "--username", "alice", "--confirm-username", "wrong"]) == 1
    assert worker.exists()
    assert main(["purge-account", "--username", "alice", "--confirm-username", "alice"]) == 0
    captured = capsys.readouterr()
    for value in (binding["upstream_token"], invitation["invite_code"], auth["access_token"], other["id"], "User conversation"):
        assert value not in captured.out + captured.err


def _worker_environment(store, binding, worker, monkeypatch):
    from docthinker.server import invite_workers as workers
    environment = {workers.SHARING_ACK_ENV: workers.SHARING_ACK_VALUE,
                   "LLM_BINDING_API_KEY": "synthetic-model-key-not-used",
                   "LLM_VLM_HOST": "https://provider.example.invalid/v1", "LLM_MODEL": "synthetic-model"}
    identity = workers.WorkerBinding(binding["id"], 8101, binding["upstream_token"])
    monkeypatch.setattr(os, "environ", workers.child_environment(store.root, identity, environment))
    monkeypatch.chdir(worker / "runtime")
    return workers


def test_real_worker_entry_rechecks_revoked_binding_before_app_or_model_start(account, monkeypatch):
    store, binding, worker, *_ = account
    workers = _worker_environment(store, binding, worker, monkeypatch)
    monkeypatch.setattr(workers, "_serve_worker", lambda _port: pytest.fail("Revoked worker must never start the app/model"))
    with pytest.raises(workers.WorkerConfigurationError, match="no longer authorized"):
        workers.worker_main()


def test_real_worker_entry_holds_account_lock_through_serve_and_shutdown(account, monkeypatch):
    store, binding, worker, *_ = account
    workers = _worker_environment(store, binding, worker, monkeypatch)
    with store._connect() as db:
        db.execute("UPDATE users SET status='active' WHERE id=?", (binding["id"],))
    seen = []
    def serve(port):
        seen.append(port)
        store.revoke_user("alice")
        with pytest.raises(InviteError, match="still running"):
            lifecycle.purge_account(store, "alice", "alice")
        assert worker.exists()
    monkeypatch.setattr(workers, "_serve_worker", serve)
    workers.worker_main()
    assert seen == [8101]
    monkeypatch.chdir(store.root)
    assert lifecycle.purge_account(store, "alice", "alice")["status"] == "retained_data_removed"


def test_directory_swap_during_cleanup_never_removes_other_account(account, monkeypatch):
    store, binding, worker, output, _, _, other = account
    other_worker = store.root / "workers" / other["id"]
    original = lifecycle._inventory
    changed = []
    def inventory(fd, prefix="", **kwargs):
        result = original(fd, prefix, **kwargs)
        if not prefix and not changed:
            changed.append(True)
            worker.rename(store.root / "workers" / "held-original")
            other_worker.rename(worker)
        return result
    monkeypatch.setattr(lifecycle, "_inventory", inventory)
    with pytest.raises(InviteError, match="incomplete"):
        lifecycle.purge_account(store, "alice", "alice")
    assert (worker / "data" / "other.txt").read_text() == "must-not-export-or-delete"


def test_missing_never_started_worker_can_export_metadata_then_cleanup(account):
    store, binding, worker, output, _, _, other = account
    store.add_user("never-started", 8103)
    store.revoke_user("never-started")
    archive = output / "never-started.zip"
    assert lifecycle.export_account(store, "never-started", "never-started", archive)["files"] == 0
    assert lifecycle.purge_account(store, "never-started", "never-started")["status"] == "retained_data_removed"


def test_empty_business_directories_are_retained_in_manifest_and_archive(account):
    store, _, worker, output, *_ = account
    directories = ("data/empty-project/empty-results", "_system/research/files/empty-import")
    for name in directories:
        (worker / name).mkdir(parents=True, mode=0o700)
    target = output / "empty-dirs.zip"
    lifecycle.export_account(store, "alice", "alice", target)
    with zipfile.ZipFile(target) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for name in directories:
            assert name in manifest["directories"]
            info = archive.getinfo("files/" + name + "/")
            assert info.is_dir() and info.file_size == 0
            assert (info.external_attr >> 16) & 0o777 == 0o700
        assert set(manifest["directories"]) == {
            item.filename.removeprefix("files/").removesuffix("/")
            for item in archive.infolist() if item.is_dir()
        }


def test_export_fstat_failure_closes_descriptor_without_unverified_cleanup_or_overwrite(account, monkeypatch):
    store, _, worker, output, *_ = account
    target = output / "failed-fstat.zip"
    opened, closed, unlinked = [], [], []
    original_open, original_fstat = os.open, os.fstat
    original_close, original_unlink = os.close, os.unlink
    def open_file(path, flags, *args, **kwargs):
        fd = original_open(path, flags, *args, **kwargs)
        if path == target.name and flags & os.O_CREAT:
            opened.append(fd)
        return fd
    def fail_stat(fd):
        if opened and fd == opened[-1]:
            raise OSError("synthetic fstat failure")
        return original_fstat(fd)
    def close_file(fd):
        closed.append(fd)
        return original_close(fd)
    def unlink_file(name, *args, **kwargs):
        unlinked.append(name)
        return original_unlink(name, *args, **kwargs)
    monkeypatch.setattr(os, "open", open_file)
    monkeypatch.setattr(os, "fstat", fail_stat)
    monkeypatch.setattr(os, "close", close_file)
    monkeypatch.setattr(os, "unlink", unlink_file)
    with pytest.raises(OSError, match="synthetic fstat failure"):
        lifecycle.export_account(store, "alice", "alice", target)
    assert len(opened) == 1 and closed.count(opened[0]) == 1
    assert target.name not in unlinked
    # Without a confirmed inode we leave the private empty artifact for owner
    # inspection, rather than risking deletion of a replaced path.
    assert target.exists() and target.stat().st_size == 0
    assert target.stat().st_mode & 0o777 == 0o600
    monkeypatch.setattr(os, "fstat", original_fstat)
    with pytest.raises(FileExistsError):
        lifecycle.export_account(store, "alice", "alice", target)
    assert len(opened) == 1 and target.stat().st_size == 0
    assert (worker / "data" / "_system" / "history.txt").exists()
