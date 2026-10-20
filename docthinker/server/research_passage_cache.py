"""Private exact-batch derived-vector cache; never evidence or authority.

The encoder owns normalization and batching. This leaf only transports its
float32 bits for an identical scoped request. No text, query, pickle or model
response is stored. Any unsafe/missing/corrupt entry falls back to that encoder.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
import struct
import time
import uuid

VERSION = 'private-exact-passage-batch-v1'
MAX_ENTRIES = 128
MAX_BYTES = 64 * 1024 * 1024
MAX_MATRIX_BYTES = 8192 * 384 * 4
MAX_IDLE_SECONDS = 7 * 24 * 60 * 60
_HEX = re.compile(r'[0-9a-f]{64}\Z')
_MAGIC = b'DTPV1\x00'
_NOFOLLOW = getattr(os, 'O_NOFOLLOW', 0)
_DIRECTORY = getattr(os, 'O_DIRECTORY', 0)


class Unavailable(ValueError):
    pass


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def _private(info, mode, *, directory=False):
    kind = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if (not kind or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != mode
            or not directory and info.st_nlink != 1):
        raise Unavailable('private_cache_object_required')


def _mkdir(path):
    # Never resolve an attacker-supplied symlink into an apparently safe path.
    path = Path(path).absolute()
    for parent in (*reversed(path.parents), path):
        try:
            info = parent.lstat()
        except FileNotFoundError:
            try:
                parent.mkdir(mode=0o700)
            except FileExistsError:
                pass
            info = parent.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise Unavailable('cache_directory_symlink_or_type')
    _private(path.stat(), 0o700, directory=True)
    return path


def _open_private(directory, name, flags):
    # Non-regular poisoned entries (especially FIFOs) must not block on open.
    fd = os.open(name, flags | _NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=directory)
    try:
        _private(os.fstat(fd), 0o600)
        return fd
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def _locked(root, wait_seconds):
    directory = os.open(root, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
    lock = None
    try:
        _private(os.fstat(directory), 0o700, directory=True)
        deadline = time.monotonic() + wait_seconds
        # macOS can return ENOENT for simultaneous O_NOFOLLOW|O_CREAT opens
        # before the winning creator's directory entry becomes observable.
        while lock is None:
            try:
                lock = _open_private(directory, '.lock', os.O_RDWR | os.O_CREAT)
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    raise Unavailable('cache_lock_unavailable')
                time.sleep(min(0.02, max(0, deadline - time.monotonic())))
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise Unavailable('cache_lock_busy')
                time.sleep(min(0.02, max(0, deadline - time.monotonic())))
        yield directory
    finally:
        if lock is not None:
            os.close(lock)  # Releases the OS lock, including on cancellation.
        os.close(directory)


def _read_secret(directory):
    fd = _open_private(directory, '.auth', os.O_RDONLY)
    try:
        value = os.read(fd, 33)
    finally:
        os.close(fd)
    if len(value) != 32:
        raise Unavailable('cache_auth_unavailable')
    return value


def _secret(directory):
    try:
        fd = _open_private(directory, '.auth', os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        return _read_secret(directory)
    value = os.urandom(32)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(directory)
    except BaseException:
        # An interrupted empty key is not trusted or silently replaced.
        raise
    return value


def _read_existing(root, scope, token, rows, dimension):
    """Atomic committed entries need no writer lock, including during a cold job.

    This path creates nothing and never reads source/permission state. Writers
    replace complete authenticated files, so an opened old inode is also exact.
    """
    directory = os.open(root, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
    try:
        _private(os.fstat(directory), 0o700, directory=True)
        key = _read_secret(directory)
        scoped = os.open(scope, os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=directory)
        try:
            _private(os.fstat(scoped), 0o700, directory=True)
            return _load(scoped, token + '.pv', key, token, rows, dimension)
        finally:
            os.close(scoped)
    finally:
        os.close(directory)


def _load(directory, name, key, expected, rows, dimension):
    import numpy as np

    fd = _open_private(directory, name, os.O_RDONLY)
    try:
        info = os.fstat(fd)
        length = info.st_size
        if time.time() - info.st_mtime > MAX_IDLE_SECONDS:
            raise Unavailable('cache_entry_expired')
        if not 40 <= length <= MAX_MATRIX_BYTES + 4096:
            raise Unavailable('cache_entry_size')
        with os.fdopen(fd, 'rb') as stream:
            fd = None
            raw = stream.read(MAX_MATRIX_BYTES + 4097)
    finally:
        if fd is not None:
            os.close(fd)
    if not raw.startswith(_MAGIC) or len(raw) < len(_MAGIC) + 4 + 32:
        raise Unavailable('cache_entry_format')
    header_size = struct.unpack('>I', raw[len(_MAGIC):len(_MAGIC) + 4])[0]
    if not 1 <= header_size <= 2048:
        raise Unavailable('cache_header_size')
    payload_start = len(_MAGIC) + 4 + header_size
    if not hmac.compare_digest(hmac.digest(key, raw[:-32], 'sha256'), raw[-32:]):
        raise Unavailable('cache_authentication')
    header = json.loads(raw[len(_MAGIC) + 4:payload_start])
    payload = raw[payload_start:-32]
    required = {'v': VERSION, 'identity': expected, 'shape': [rows, dimension],
                'dtype': '<f4', 'normalized': True,
                'payload_sha256': hashlib.sha256(payload).hexdigest()}
    if _json(header) != _json(required) or len(payload) != rows * dimension * 4:
        raise Unavailable('cache_identity_or_shape')
    matrix = np.frombuffer(payload, dtype='<f4').reshape(rows, dimension).copy()
    if not np.isfinite(matrix).all() or (np.linalg.norm(matrix, axis=1) <= 0).any():
        raise Unavailable('cache_invalid_vector')
    os.utime(name, None, dir_fd=directory, follow_symlinks=False)
    return matrix


def _store(directory, name, key, identity, matrix, rows, dimension):
    import numpy as np

    if (not isinstance(matrix, np.ndarray) or matrix.dtype != np.dtype('float32')
            or matrix.shape != (rows, dimension) or not np.isfinite(matrix).all()
            or (np.linalg.norm(matrix, axis=1) <= 0).any()):
        raise Unavailable('encoder_matrix_not_cacheable')
    payload = matrix.astype('<f4', copy=False).tobytes(order='C')
    if len(payload) > MAX_MATRIX_BYTES:
        raise Unavailable('cache_matrix_limit')
    header = _json({'v': VERSION, 'identity': identity, 'shape': [rows, dimension],
                    'dtype': '<f4', 'normalized': True,
                    'payload_sha256': hashlib.sha256(payload).hexdigest()})
    raw = _MAGIC + struct.pack('>I', len(header)) + header + payload
    raw += hmac.digest(key, raw, 'sha256')
    temporary = '.' + uuid.uuid4().hex + '.tmp'
    fd = _open_private(directory, temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)
    finally:
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass


def _trim(root, max_entries, max_bytes):
    # One workspace lock owns maintenance; scope directories never share keys.
    entries, total, scanned = [], 0, 0
    for scope in root.iterdir():
        if not _HEX.fullmatch(scope.name):
            continue
        _private(scope.lstat(), 0o700, directory=True)
        for entry in scope.iterdir():
            scanned += 1
            if scanned > 4096:
                raise Unavailable('cache_maintenance_bound')
            info = entry.lstat()
            _private(info, 0o600)
            if entry.name.startswith('.') and entry.name.endswith('.tmp'):
                entry.unlink()  # Abandoned atomic writes, never source files.
            elif entry.suffix == '.pv' and _HEX.fullmatch(entry.stem):
                if time.time() - info.st_mtime > MAX_IDLE_SECONDS:
                    entry.unlink()
                    continue
                entries.append((info.st_mtime_ns, str(entry), info.st_size))
                total += info.st_size
    count = len(entries)
    for _, name, size in sorted(entries):
        if count <= max_entries and total <= max_bytes:
            break
        Path(name).unlink()
        count -= 1
        total -= size
    for scope in root.iterdir():
        if _HEX.fullmatch(scope.name):
            try:
                scope.rmdir()
            except OSError as error:
                if error.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                    raise


class PrivatePassageCache:
    def __init__(self, root, *, max_entries=MAX_ENTRIES, max_bytes=MAX_BYTES, wait_seconds=30.0):
        if (type(max_entries) is not int or not 1 <= max_entries <= MAX_ENTRIES
                or type(max_bytes) is not int or not 1024 <= max_bytes <= MAX_BYTES
                or type(wait_seconds) not in (int, float) or not 0 <= wait_seconds <= 30):
            raise ValueError('Invalid bounded passage cache')
        self.root = root
        self.max_entries, self.max_bytes, self.wait_seconds = max_entries, max_bytes, wait_seconds

    def matrix(self, namespace, identity, *, rows, dimension, encode):
        """Exact once encoder fallback; encoder failures are never retried here."""
        result = None
        invoked = False
        try:
            if (type(namespace) is not str or not namespace or len(namespace) > 1024
                    or type(rows) is not int or not 1 <= rows <= 8192 or dimension != 384):
                raise Unavailable('cache_request_bound')
            root = _mkdir(self.root)
            scope = hashlib.sha256(namespace.encode()).hexdigest()
            token = hashlib.sha256(_json({'v': VERSION, 'root': str(root),
                'scope': scope, 'identity': identity})).hexdigest()
            try:
                # A different session's twenty-second encoder cannot block
                # already committed warm vectors. Maintenance runs on writes.
                return _read_existing(root, scope, token, rows, dimension), 'hit'
            except (OSError, ValueError, KeyError, TypeError, UnicodeError):
                pass
            with _locked(root, self.wait_seconds) as directory:
                key = _secret(directory)
                scoped = _mkdir(root / scope)
                scope_fd = os.open(scoped, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
                try:
                    name = token + '.pv'
                    try:
                        result = _load(scope_fd, name, key, token, rows, dimension)
                    except (OSError, ValueError, KeyError, TypeError, UnicodeError):
                        pass  # No cached bytes can enter an answer after failure.
                    if result is not None:
                        _trim(root, self.max_entries, self.max_bytes)
                        return result, 'hit'
                    invoked = True
                    result = encode()
                    try:
                        _store(scope_fd, name, key, token, result, rows, dimension)
                    except (OSError, ValueError, TypeError):
                        return result, 'write_unavailable'
                finally:
                    os.close(scope_fd)
                _trim(root, self.max_entries, self.max_bytes)
                return result, 'stored'
        except (OSError, ValueError, KeyError, TypeError, UnicodeError):
            # In particular do not call encode twice if maintenance failed.
            if invoked and result is None:
                raise
            if result is not None and not invoked:
                return result, 'hit_maintenance_unavailable'
            return (encode() if result is None else result), 'unavailable'
