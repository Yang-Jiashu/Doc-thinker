"""Offline owner-only account export and retained-data cleanup.

No public endpoints, provider calls or process killing. Lifecycle-protocol-1
accounts only: old identities cannot be safely upgraded by inventing a lock.
Cleanup is logical deletion, not physical secure erase or backup deletion.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import zipfile

from .invite_store import InviteError, normalize_username

MAX_FILES = 100_000
MAX_BYTES = 2 * 1024 ** 3
_ID = re.compile(r"[0-9a-f]{32}\Z")
_RETAINED = ["account_id", "username", "reserved_worker_port", "revoked_status", "lifecycle_state"]


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _exists(parent, name):
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False


def _private(info, *, directory=False):
    if (not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
            or info.st_uid != os.getuid() or info.st_mode & 0o077
            or (not directory and info.st_nlink != 1)):
        raise InviteError("Account storage must be private, owned, and contain no links or special files")


@contextmanager
def _directory(path):
    path = Path(path)
    if not path.is_absolute() or path != path.resolve():
        raise InviteError("Account paths must be canonical absolute paths without symlinks")
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        _private(info, directory=True)
        yield fd
        if _identity(os.stat(path, follow_symlinks=False))[:2] != _identity(info)[:2]:
            raise InviteError("Account directory changed; operation is not confirmed complete")
    finally:
        os.close(fd)


@contextmanager
def _child_directory(parent, name, *, create=False):
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent)
        except FileExistsError:
            pass
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    try:
        before = os.fstat(fd)
        _private(before, directory=True)
        yield fd
        after = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if _identity(before)[:2] != _identity(after)[:2]:
            raise InviteError("Account directory changed; operation is not confirmed complete")
    finally:
        os.close(fd)


@contextmanager
def _lock(parent, name):
    fd = os.open(name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=parent)
    try:
        info = os.fstat(fd)
        _private(info)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise InviteError("Account or runtime is still running; stop it before this operation", 409) from None
        if _identity(os.stat(name, dir_fd=parent, follow_symlinks=False))[:2] != _identity(info)[:2]:
            raise InviteError("Account lock changed; operation refused")
        yield
        if _identity(os.stat(name, dir_fd=parent, follow_symlinks=False))[:2] != _identity(info)[:2]:
            raise InviteError("Account lock changed; operation is not confirmed complete")
    finally:
        os.close(fd)


@contextmanager
def worker_lifetime_lock(root, user_id):
    """Held by the child, so it survives loss of its supervisor process."""
    if not isinstance(user_id, str) or not _ID.fullmatch(user_id):
        raise InviteError("Invalid account identity")
    with _directory(root) as root_fd:
        with _child_directory(root_fd, "account-locks", create=True) as locks:
            with _lock(locks, user_id + ".lock"):
                yield


def _account(db, username):
    row = db.execute("SELECT u.id,u.username,u.status,u.created_at,l.protocol,l.state "
                     "FROM users u LEFT JOIN account_lifecycle l ON l.user_id=u.id WHERE u.username=?",
                     (username,)).fetchone()
    if (row is None or row["status"] != "revoked" or row["protocol"] != 1
            or not _ID.fullmatch(row["id"])):
        raise InviteError("Account unavailable: revoked lifecycle-protocol-1 account required", 409)
    if row["state"] == "purged":
        raise InviteError("Account retained-data cleanup is already complete; tombstone remains", 409)
    return dict(row)


@contextmanager
def _maintenance(store, username, confirmation):
    username = normalize_username(username)
    if confirmation != username:
        raise InviteError("Confirmation must exactly match the normalized username")
    # Reject old protocol before even inventing an account lock file.
    with store._connect() as db:
        account = _account(db, username)
    with _directory(store.root) as root_fd, _lock(root_fd, "workers.lock"):
        with worker_lifetime_lock(store.root, account["id"]):
            with store._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                current = _account(db, username)
                if current["id"] != account["id"]:
                    raise InviteError("Account identity changed; operation refused")
            yield current, root_fd


def _inventory(fd, prefix="", *, entries=None, total=None):
    entries = {} if entries is None else entries
    total = [0] if total is None else total
    for name in sorted(os.listdir(fd)):
        if (name in {".", ".."} or any(ord(char) < 32 for char in name)
                or "\\" in name or ":" in name or len(name) > 255):
            raise InviteError("Account filename cannot be represented safely in an export")
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        path = f"{prefix}/{name}" if prefix else name
        is_dir = stat.S_ISDIR(info.st_mode)
        _private(info, directory=is_dir)
        if len(entries) >= MAX_FILES:
            raise InviteError("Account file count exceeds the bounded maintenance limit")
        entries[path] = (is_dir, _identity(info))
        if is_dir:
            with _child_directory(fd, name) as child:
                _inventory(child, path, entries=entries, total=total)
        else:
            total[0] += info.st_size
            if total[0] > MAX_BYTES:
                raise InviteError("Account bytes exceed the bounded maintenance limit")
    return entries


def _read_copy(parent, name, expected, target):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    try:
        _private(os.fstat(fd))
        if _identity(os.fstat(fd)) != expected:
            raise InviteError("Account file changed during export")
        digest, count = hashlib.sha256(), 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, expected[2] - count + 1))
            if not chunk:
                break
            count += len(chunk)
            if count > expected[2]:
                raise InviteError("Account file changed during export")
            target.write(chunk)
            digest.update(chunk)
        if count != expected[2] or _identity(os.fstat(fd)) != expected:
            raise InviteError("Account file changed during export")
        return {"bytes": count, "sha256": digest.hexdigest()}
    finally:
        os.close(fd)


def _copy_tree(fd, archive, entries, prefix="", *, hashes=None):
    hashes = {} if hashes is None else hashes
    for name in sorted(os.listdir(fd)):
        path = f"{prefix}/{name}" if prefix else name
        if path not in entries:
            raise InviteError("Account files changed during export")
        is_dir, expected = entries[path]
        if is_dir:
            with _child_directory(fd, name) as child:
                if _identity(os.fstat(child)) != expected:
                    raise InviteError("Account directory changed during export")
                _copy_tree(child, archive, entries, path, hashes=hashes)
        else:
            with archive.open("files/" + path, "w", force_zip64=True) as target:
                hashes[path] = _read_copy(fd, name, expected, target)
    return hashes


def _business_entries(all_entries):
    # Never copy registry or worker environment/home/runtime/cache/temp files.
    # Unknown durable roots fail closed rather than silently claiming completeness.
    for path in all_entries:
        parts = path.split("/")
        if parts[0] not in {"data", "_system", "runtime", "home", "tmp", "cache"}:
            raise InviteError("Unrecognized account storage layout; export completeness is unverified")
        if parts[0] == "_system" and len(parts) > 1 and parts[1] != "research":
            raise InviteError("Unrecognized account storage layout; export completeness is unverified")
    return {path: entry for path, entry in all_entries.items()
            if path == "data" or path.startswith("data/") or path == "_system"
            or path.startswith("_system/research")}


def export_account(store, username, confirmation, output):
    """New private ZIP containing retained business files, never registry secrets."""
    output = Path(output)
    repository = Path(__file__).resolve().parents[2]
    if (not output.is_absolute() or output != output.resolve() or output == store.root
            or output.is_relative_to(store.root) or store.root.is_relative_to(output)
            or output == repository or output.is_relative_to(repository)):
        raise InviteError("Export requires a new private path outside the registry and repository")
    fd, created = None, None
    with _maintenance(store, username, confirmation) as (account, root_fd):
        if account["state"] != "retained":
            raise InviteError("Cleanup has started; a complete account export is no longer available")
        with _directory(output.parent) as out_parent:
            try:
                fd = os.open(output.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=out_parent)
                created = _identity(os.fstat(fd))[:2]
                with os.fdopen(fd, "wb", closefd=False) as target, zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
                    hashes = {}
                    directories = []
                    if _exists(root_fd, "workers"):
                        with _child_directory(root_fd, "workers") as workers:
                            if _exists(workers, account["id"]):
                                with _child_directory(workers, account["id"]) as worker:
                                    before = _inventory(worker)
                                    entries = _business_entries(before)
                                    directories = [path for path, (is_dir, _) in entries.items() if is_dir]
                                    for path in directories:
                                        item = zipfile.ZipInfo("files/" + path + "/")
                                        item.external_attr = (stat.S_IFDIR | 0o700) << 16
                                        archive.writestr(item, b"")
                                    for name in ("data", "_system"):
                                        if name in entries:
                                            with _child_directory(worker, name) as child:
                                                _copy_tree(child, archive, entries, name, hashes=hashes)
                                    if _inventory(worker) != before:
                                        raise InviteError("Account changed during export; no export was completed")
                    metadata = {key: account[key] for key in ("id", "username", "status", "created_at")}
                    archive.writestr("account.json", json.dumps(metadata, sort_keys=True))
                    archive.writestr("manifest.json", json.dumps({"format": "docthinker-account-export-v1", "files": hashes, "directories": directories,
                        "scope": "all retained business data and research files, including original uploads",
                        "excluded": ["registry credentials", "provider credentials", "worker runtime/home/cache/tmp", "backups", "external provider records"],
                        "warning": "User-supplied secrets inside documents or chats remain part of private user content."}, sort_keys=True))
                os.fsync(fd)
            except BaseException:
                if fd is not None and created is not None:
                    try:
                        if _identity(os.stat(output.name, dir_fd=out_parent, follow_symlinks=False))[:2] == created:
                            os.unlink(output.name, dir_fd=out_parent)
                    except FileNotFoundError:
                        pass
                raise
            finally:
                if fd is not None:
                    os.close(fd)
    return {"status": "exported", "username": account["username"], "files": len(hashes), "format": "private_zip", "contains_user_content": True}


def _remove_tree(fd, entries, prefix=""):
    for name in sorted(os.listdir(fd)):
        path = f"{prefix}/{name}" if prefix else name
        current = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if path not in entries or _identity(current) != entries[path][1]:
            raise InviteError("Account changed during cleanup; cleanup is incomplete")
        if entries[path][0]:
            with _child_directory(fd, name) as child:
                if _identity(os.fstat(child)) != entries[path][1]:
                    raise InviteError("Account directory changed during cleanup; cleanup is incomplete")
                _remove_tree(child, entries, path)
            os.rmdir(name, dir_fd=fd)
        else:
            os.unlink(name, dir_fd=fd)


def purge_account(store, username, confirmation):
    """Remove retained worker content; retain a non-reusable identity tombstone."""
    with _maintenance(store, username, confirmation) as (account, root_fd):
        # Commit the recovery marker before any irreversible filesystem operation.
        with store._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _account(db, account["username"])
            db.execute("UPDATE account_lifecycle SET state='purging' WHERE user_id=?", (account["id"],))
            db.execute("UPDATE users SET password_hash=NULL,upstream_token='',allow_uploads=0,created_at=0 WHERE id=?", (account["id"],))
            for table in ("invitations", "auth_sessions", "usage", "query_reservations"):
                db.execute(f"DELETE FROM {table} WHERE user_id=?", (account["id"],))
        try:
            if _exists(root_fd, "workers"):
                with _child_directory(root_fd, "workers") as workers:
                    if _exists(workers, account["id"]):
                        with _child_directory(workers, account["id"]) as worker:
                            entries = _inventory(worker)
                            _remove_tree(worker, entries)
                        os.rmdir(account["id"], dir_fd=workers)
                        os.fsync(workers)
            with store._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                _account(db, account["username"])
                db.execute("UPDATE account_lifecycle SET state='purged' WHERE user_id=?", (account["id"],))
        except Exception:
            raise InviteError("Account cleanup is incomplete; access remains revoked. Resolve local storage errors and repeat the same confirmed cleanup") from None
    return {"status": "retained_data_removed", "username": account["username"], "retained_tombstone": _RETAINED,
            "not_removed": ["backups", "external provider records", "physical disk remnants"], "secure_erase": False}
