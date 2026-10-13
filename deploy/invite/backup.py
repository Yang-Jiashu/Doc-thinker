"""Bounded, offline directory backups. No credentials, Docker, or network APIs.

Call only after stopping the gateway, workers, and administrative writes. The
worker lock is an additional guard, not a general-purpose live database lock.
Backups contain private user data; mode 0700/0600 is not encryption.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import stat
import tempfile
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "BACKUP-MANIFEST.json"
MAX_FILES = 100_000
MAX_BYTES = 50 * 1024**3
MAX_MANIFEST_BYTES = 16 * 1024**2
BLOCK_SIZE = 1024 * 1024


class BackupError(ValueError):
    """Only static diagnostics; filenames and document contents stay private."""


def _path(value: Path) -> Path:
    path = Path(value).expanduser()
    if (not path.is_absolute() or path != path.resolve()
            or path == ROOT or path.is_relative_to(ROOT) or ROOT.is_relative_to(path)):
        raise BackupError("Use a canonical absolute path outside the source checkout, without links.")
    return path


def _private_directory(path: Path) -> None:
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise BackupError("Data, backup, and destination parent directories must be owner-only (0700).")


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _regular(info):
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
        raise BackupError("Only owned regular files without hard links can be backed up or restored.")


def _inventory(root: Path):
    _private_directory(root)
    files, directories, total = {}, [], 0
    def unreadable(_error):
        raise BackupError("A data directory cannot be read; incomplete backups are not permitted.")
    for directory, children, names in os.walk(root, followlinks=False, onerror=unreadable):
        current = Path(directory)
        for name in children:
            path = current / name
            info = path.lstat()
            if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_mode & 0o022 or path.resolve() != path):
                raise BackupError("Linked, unowned, or writable-by-others directories are not supported.")
            directories.append(path.relative_to(root).as_posix())
        for name in names:
            path = current / name
            info = path.lstat()
            _regular(info)
            if path.resolve() != path:
                raise BackupError("Symbolic links are not supported.")
            if name.endswith(("-wal", "-journal")) and info.st_size:
                raise BackupError("Uncheckpointed SQLite files found; stop all writers and checkpoint before cold backup.")
            files[path.relative_to(root).as_posix()] = _identity(info)
            total += info.st_size
            if len(files) + len(directories) > MAX_FILES or total > MAX_BYTES:
                raise BackupError("Backup exceeds the bounded file-count or size limit.")
        if len(files) + len(directories) > MAX_FILES:
            raise BackupError("Backup exceeds the bounded file-count or size limit.")
    return files, sorted(directories)


@contextmanager
def _stopped(root: Path):
    descriptor = os.open(root / "workers.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        _regular(info)
        if info.st_mode & 0o077:
            raise BackupError("Worker lock permissions are unsafe.")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BackupError("Workers are running; stop the deployment before backup.") from None
        yield
    finally:
        os.close(descriptor)


def _transfer(source: Path, identity, destination: Path | None = None):
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    output = None
    try:
        info = os.fstat(descriptor)
        _regular(info)
        if _identity(info) != identity:
            raise BackupError("Files changed during the operation; no successful backup or restore was produced.")
        if destination is not None:
            output = os.fdopen(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb")
        digest, size = hashlib.sha256(), 0
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while block := stream.read(BLOCK_SIZE):
                size += len(block)
                if size > identity[2]:
                    raise BackupError("Files changed during the operation; retry only after stopping all writers.")
                digest.update(block)
                if output:
                    output.write(block)
        if _identity(os.fstat(descriptor)) != identity or size != identity[2]:
            raise BackupError("Files changed during the operation; retry only after stopping all writers.")
        if output:
            output.flush()
            os.fsync(output.fileno())
        return {"size": size, "sha256": digest.hexdigest()}
    finally:
        if output:
            output.close()
        os.close(descriptor)


@contextmanager
def _new_directory(destination: Path):
    _private_directory(destination.parent)
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError:
        raise BackupError("Destination already exists; existing data will never be overwritten.") from None
    identity = destination.stat()
    try:
        yield
    except BaseException:
        # Remove only the directory created by this operation, never another inode.
        current = destination.lstat()
        if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
            shutil.rmtree(destination)
        raise


def _sqlite_checks(root: Path, files) -> None:
    if "invite.sqlite3" not in files:
        raise BackupError("This is not an invitation data directory (invite.sqlite3 is required).")
    for name in files:
        if not name.endswith((".sqlite3", ".db")):
            continue
        try:
            with (root / name).open("rb") as handle:
                if handle.read(16) != b"SQLite format 3\x00":
                    raise BackupError("SQLite integrity verification failed.")
            with sqlite3.connect(f"{(root / name).as_uri()}?mode=ro&immutable=1", uri=True) as connection:
                if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                    raise BackupError("SQLite integrity verification failed.")
        except sqlite3.Error:
            raise BackupError("SQLite integrity verification failed.") from None


def _report(manifest, operation):
    return {"operation": operation, "ok": True, "format": 1,
            "files": len(manifest["files"]), "bytes": sum(item["size"] for item in manifest["files"].values()),
            "encrypted": False, "docker_restore_verified": False}


def create_backup(data_dir: Path, output: Path):
    source, destination = _path(data_dir), _path(output)
    _private_directory(source)
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise BackupError("Backup destination must be separate from its data directory.")
    with _stopped(source):
        files, directories = _inventory(source)
        with _new_directory(destination):
            data = destination / "data"
            data.mkdir(mode=0o700)
            for name in sorted(directories, key=lambda value: (value.count("/"), value)):
                (data / name).mkdir(mode=0o700)
            manifest = {"format": 1, "directories": directories, "files": {
                name: _transfer(source / name, identity, data / name) for name, identity in sorted(files.items())}}
            if _inventory(source) != (files, directories):
                raise BackupError("The data tree changed during backup; stop all writers before retrying.")
            _sqlite_checks(data, files)
            body = json.dumps(manifest, sort_keys=True).encode("utf-8")
            if len(body) > MAX_MANIFEST_BYTES:
                raise BackupError("Backup manifest exceeds its size limit.")
            with os.fdopen(os.open(destination / MANIFEST, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
    return _report(manifest, "backup")


def _load_backup(backup_dir: Path):
    backup = _path(backup_dir)
    _private_directory(backup)
    if {path.name for path in backup.iterdir()} != {"data", MANIFEST}:
        raise BackupError("Unexpected or missing entries in the backup directory.")
    descriptor = os.open(backup / MANIFEST, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        _regular(info)
        if info.st_mode & 0o077 or info.st_size > MAX_MANIFEST_BYTES:
            raise BackupError("Backup manifest size or permissions are unsafe.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            manifest = json.loads(handle.read(MAX_MANIFEST_BYTES + 1))
    except (UnicodeError, json.JSONDecodeError):
        raise BackupError("Invalid backup manifest.") from None
    finally:
        os.close(descriptor)
    if (not isinstance(manifest, dict) or set(manifest) != {"format", "directories", "files"}
            or type(manifest["format"]) is not int or manifest["format"] != 1
            or not isinstance(manifest["files"], dict) or not isinstance(manifest["directories"], list)):
        raise BackupError("Invalid backup manifest schema.")
    paths = [*manifest["files"], *manifest["directories"]]
    if len(paths) > MAX_FILES or any(not isinstance(name, str) or not name or "\\" in name
            or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            or str(PurePosixPath(name)) != name for name in paths):
        raise BackupError("Invalid paths in backup manifest.")
    for record in manifest["files"].values():
        if (not isinstance(record, dict) or set(record) != {"size", "sha256"}
                or type(record["size"]) is not int or not 0 <= record["size"] <= MAX_BYTES
                or not isinstance(record["sha256"], str) or len(record["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in record["sha256"])):
            raise BackupError("Invalid file record in backup manifest.")
    files, directories = _inventory(backup / "data")
    if set(files) != set(manifest["files"]) or directories != manifest["directories"]:
        raise BackupError("Backup inventory does not match its manifest.")
    return backup, manifest, files, directories


def verify_backup(backup_dir: Path):
    backup, manifest, files, _ = _load_backup(backup_dir)
    for name, identity in files.items():
        if _transfer(backup / "data" / name, identity) != manifest["files"][name]:
            raise BackupError("Backup checksum mismatch; restore is blocked.")
    _sqlite_checks(backup / "data", files)
    return _report(manifest, "verify")


def restore_backup(backup_dir: Path, target_dir: Path):
    backup, manifest, files, directories = _load_backup(backup_dir)
    destination = _path(target_dir)
    if destination.is_relative_to(backup) or backup.is_relative_to(destination):
        raise BackupError("Restore destination must be separate from the backup directory.")
    with _new_directory(destination):
        for name in sorted(directories, key=lambda value: (value.count("/"), value)):
            (destination / name).mkdir(mode=0o700)
        for name, identity in files.items():
            if _transfer(backup / "data" / name, identity, destination / name) != manifest["files"][name]:
                raise BackupError("Backup checksum mismatch; partial restore was removed.")
        _sqlite_checks(destination, files)
    return _report(manifest, "restore")


def self_test():
    """Only generated temporary data; no deployment configuration is loaded."""
    with tempfile.TemporaryDirectory(prefix="docthinker-cold-backup-") as temporary:
        root = Path(temporary).resolve()
        source, backup, restored = root / "source", root / "backup", root / "restored"
        source.mkdir(mode=0o700)
        with sqlite3.connect(source / "invite.sqlite3") as connection:
            connection.execute("CREATE TABLE test_records (value TEXT)")
            connection.execute("INSERT INTO test_records VALUES ('synthetic-record')")
        (source / "documents").mkdir(mode=0o700)
        (source / "empty-memory").mkdir(mode=0o700)
        (source / "documents" / "sample.txt").write_bytes(b"synthetic source evidence\n")
        create_backup(source, backup)
        verify_backup(backup)
        restore_backup(backup, restored)
        with sqlite3.connect(restored / "invite.sqlite3") as connection:
            if connection.execute("SELECT value FROM test_records").fetchall() != [("synthetic-record",)]:
                raise BackupError("Self-test database restoration failed.")
        if ((restored / "documents" / "sample.txt").read_bytes() != b"synthetic source evidence\n"
                or not (restored / "empty-memory").is_dir()):
            raise BackupError("Self-test document restoration failed.")
        rejected = 0
        try:
            restore_backup(backup, restored)
        except BackupError:
            rejected += 1
        (backup / "data" / "documents" / "sample.txt").write_bytes(b"tampered")
        try:
            restore_backup(backup, root / "corrupt-restore")
        except BackupError:
            rejected += 1
        if rejected != 2 or (root / "corrupt-restore").exists():
            raise BackupError("Self-test rejection or cleanup failed.")
    return {"operation": "backup-self-test", "ok": True, "synthetic_data_only": True,
            "checks": ["sqlite_restore", "document_restore", "empty_directory_restore", "overwrite_rejected", "corruption_rejected"],
            "docker_restore_verified": False, "deployment_data_read": False}
