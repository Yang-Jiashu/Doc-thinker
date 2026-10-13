"""Small, local invitation registry; never imports model or desktop state.

Upstream credentials deliberately live only in this owner-only SQLite file.
Passwords, invitation codes and browser bearer tokens are never stored verbatim.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import stat
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

SESSION_SECONDS = 7 * 24 * 3600
MAX_USERS = 5
QUERY_DAILY_LIMIT = 50
GLOBAL_QUERY_DAILY_LIMIT = 250
DEFAULT_REGISTRY_BYTES = 64 * 1024 * 1024
UPLOAD_DAILY_LIMIT = 5
UPLOAD_DAILY_BYTES = 50 * 1024 * 1024
SESSION_CREATION_LIMIT = 100


class InviteError(Exception):
    """Only fixed public messages belong here, never SQL/credential details."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def registry_storage_bytes(value: str) -> int:
    """Validate the owner-only admission setting without echoing its input."""
    if (not isinstance(value, str) or not re.fullmatch(r"[0-9]{1,16}", value)
            or not 65_536 <= int(value) <= 1 << 50):
        raise InviteError("Invalid invitation registry storage quota")
    return int(value)


def normalize_username(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{3,32}", value):
        raise InviteError("Username must contain 3–32 ASCII letters, digits, underscores or hyphens", 422)
    return value.lower()


def validate_password(value: str) -> None:
    if not isinstance(value, str) or not 12 <= len(value) <= 128 or len(value.encode("utf-8")) > 512 or not value.strip():
        raise InviteError("Password must contain 12–128 characters", 422)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _derive(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=32768, r=8, p=1, dklen=32, maxmem=64 * 1024 * 1024)


def _password_hash(password: str) -> str:
    salt = secrets.token_bytes(16)
    return "scrypt$32768$8$1$" + salt.hex() + "$" + _derive(password, salt).hex()


def _verify_password(password: str, encoded: str | None) -> bool:
    # Missing users still pay the same KDF cost, reducing account enumeration.
    salt, expected, valid = b"\x00" * 16, b"\x00" * 32, False
    if encoded:
        try:
            algorithm, n, r, p, salt_hex, hash_hex = encoded.split("$")
            if (algorithm, n, r, p) == ("scrypt", "32768", "8", "1"):
                salt, expected = bytes.fromhex(salt_hex), bytes.fromhex(hash_hex)
                valid = len(salt) == 16 and len(expected) == 32
        except (ValueError, TypeError):
            pass
    return hmac.compare_digest(_derive(password, salt), expected) and valid


def _private_path(path: Path, *, directory: bool) -> None:
    info = path.lstat()
    wanted = stat.S_ISDIR if directory else stat.S_ISREG
    if (not wanted(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != (0o700 if directory else 0o600)
            or (not directory and info.st_nlink != 1)):
        raise InviteError("Invitation storage must be owner-only (directory 0700, file 0600)")


def data_directory(value: str | Path) -> Path:
    given = Path(value).expanduser()
    if not given.is_absolute():
        raise InviteError("Invitation data directory must be explicitly absolute")
    resolved = given.resolve()
    repository = Path(__file__).resolve().parents[2]
    if resolved == repository or resolved.is_relative_to(repository) or repository.is_relative_to(resolved):
        raise InviteError("Invitation data directory must be outside the repository")
    if given != resolved:
        raise InviteError("Invitation data directory must not contain symlinks or relative components")
    return resolved


class InviteStore:
    def __init__(self, root: str | Path, *, clock: Callable[[], float] = time.time,
                 max_storage_bytes: int | None = None):
        if max_storage_bytes is None:
            max_storage_bytes = registry_storage_bytes(os.environ.get(
                "DOCTHINKER_INVITE_REGISTRY_BYTES", str(DEFAULT_REGISTRY_BYTES)))
        if type(max_storage_bytes) is not int or not 65_536 <= max_storage_bytes <= 1 << 50:
            raise InviteError("Invalid invitation registry storage quota")
        self.max_storage_bytes = max_storage_bytes
        self.root = data_directory(root)
        self.path = self.root / "invite.sqlite3"
        self.clock = clock
        self._validate_files()
        with self._connect() as db:
            if db.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise InviteError("Invitation store is not initialized")
            expected = {"users", "invitations", "auth_sessions", "usage"}
            actual = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not expected.issubset(actual):
                raise InviteError("Invitation store is not initialized")
            # Additive startup migration for already-initialized local stores.
            # Gateway creation resolves this store before accepting HTTP; old
            # credentials/bindings/usage are retained and user_version stays 1.
            db.execute("CREATE TABLE IF NOT EXISTS query_reservations ("
                       "user_id TEXT NOT NULL REFERENCES users(id), session_id TEXT NOT NULL, "
                       "request_id TEXT NOT NULL, fingerprint TEXT NOT NULL, created_at REAL NOT NULL, "
                       "PRIMARY KEY(user_id,session_id,request_id))")
            # Existing identities deliberately do not gain a lifecycle protocol
            # marker: an old, orphaned worker may not hold the new lifetime lock.
            db.execute("CREATE TABLE IF NOT EXISTS account_lifecycle ("
                       "user_id TEXT PRIMARY KEY REFERENCES users(id), "
                       "protocol INTEGER NOT NULL, state TEXT NOT NULL "
                       "CHECK(state IN ('retained','purging','purged')))")

    @classmethod
    def from_environment(cls, *, require_initialized: bool = True) -> "InviteStore":
        # Initialization is always an explicit CLI action, never an HTTP side effect.
        value = os.getenv("DOCTHINKER_INVITE_DATA_DIR", "").strip()
        if not value or not require_initialized:
            raise InviteError("Set DOCTHINKER_INVITE_DATA_DIR and initialize the store explicitly")
        return cls(value)

    @classmethod
    def initialize(cls, root: str | Path) -> "InviteStore":
        # Reject bad configuration before creating an otherwise unusable store.
        max_storage_bytes = registry_storage_bytes(os.environ.get(
            "DOCTHINKER_INVITE_REGISTRY_BYTES", str(DEFAULT_REGISTRY_BYTES)))
        directory = data_directory(root)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        _private_path(directory, directory=True)
        path = directory / "invite.sqlite3"
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        except FileExistsError:
            raise InviteError("Invitation store already exists; it was not changed") from None
        with sqlite3.connect(path) as db:
            db.executescript("""
                PRAGMA user_version = 1;
                CREATE TABLE users (
                    id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK(status IN ('pending','active','revoked')),
                    upstream_url TEXT NOT NULL UNIQUE, upstream_token TEXT NOT NULL,
                    password_hash TEXT, allow_uploads INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL
                );
                CREATE TABLE invitations (
                    code_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                    expires_at REAL NOT NULL, used_at REAL, revoked INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE auth_sessions (
                    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                    expires_at REAL NOT NULL, revoked INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX auth_sessions_user ON auth_sessions(user_id);
                CREATE TABLE usage (
                    user_id TEXT NOT NULL REFERENCES users(id), day INTEGER NOT NULL,
                    kind TEXT NOT NULL, count INTEGER NOT NULL, bytes INTEGER NOT NULL,
                    PRIMARY KEY(user_id, day, kind)
                );
            """)
        return cls(directory, max_storage_bytes=max_storage_bytes)

    def _validate_files(self) -> None:
        try:
            _private_path(self.root, directory=True)
            _private_path(self.path, directory=False)
        except FileNotFoundError:
            raise InviteError("Invitation store is not initialized") from None

    @contextmanager
    def _connect(self):
        self._validate_files()
        db = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def add_user(self, username: str, upstream_port: int, *, upstream_token: str | None = None,
                 expires_hours: int = 48, allow_uploads: bool = False) -> dict:
        username = normalize_username(username)
        if type(upstream_port) is not int or not 1024 <= upstream_port <= 65535:
            raise InviteError("Upstream port must be an explicit integer between 1024 and 65535")
        if type(expires_hours) is not int or not 1 <= expires_hours <= 168:
            raise InviteError("Invitation lifetime must be 1–168 hours")
        token = upstream_token if upstream_token is not None else secrets.token_urlsafe(48)
        if not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", token):
            raise InviteError("Upstream token must be 32–256 URL-safe random characters")
        user_id, invitation = secrets.token_hex(16), secrets.token_urlsafe(32)
        now = self.clock()
        expires = now + expires_hours * 3600
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT count(*) FROM users WHERE status != 'revoked'").fetchone()[0] >= MAX_USERS:
                raise InviteError("This private gateway supports at most five non-revoked users")
            if db.execute("SELECT 1 FROM users WHERE upstream_token=?", (token,)).fetchone():
                raise InviteError("Each worker requires a distinct internal token")
            try:
                db.execute("INSERT INTO users VALUES (?,?,?,?,?,?,?,?)", (
                    user_id, username, "pending", f"http://127.0.0.1:{upstream_port}", token,
                    None, int(allow_uploads), now,
                ))
                db.execute("INSERT INTO invitations(code_hash,user_id,expires_at) VALUES (?,?,?)", (_digest(invitation), user_id, expires))
                db.execute("INSERT INTO account_lifecycle VALUES (?,1,'retained')", (user_id,))
            except sqlite3.IntegrityError:
                raise InviteError("Username or upstream port is already reserved") from None
        return {"username": username, "invite_code": invitation, "expires_at": expires}

    def _new_session(self, db, user: sqlite3.Row) -> dict:
        token = secrets.token_urlsafe(32)
        db.execute("INSERT INTO auth_sessions(token_hash,user_id,expires_at) VALUES (?,?,?)", (_digest(token), user["id"], self.clock() + SESSION_SECONDS))
        return {"access_token": token, "expires_in": SESSION_SECONDS, "user": {"id": user["id"], "username": user["username"]}}

    def reissue_invitation(self, username: str, *, expires_hours: int = 48) -> dict:
        """Replace one pending user's code without recreating its identity.

        Owner-only CLI operation. Every successful invocation invalidates all
        previous codes for this pending user, including a previous reissue.
        Registration and reissue serialize on the same SQLite write lock: an
        already registered/revoked user can never be reset by this operation.
        """
        username = normalize_username(username)
        if type(expires_hours) is not int or not 1 <= expires_hours <= 168:
            raise InviteError("Invitation lifetime must be 1–168 hours")
        invitation = secrets.token_urlsafe(32)
        digest = _digest(invitation)
        expires = self.clock() + expires_hours * 3600
        try:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                user = db.execute("SELECT id FROM users WHERE username=? AND status='pending'", (username,)).fetchone()
                if user is None:
                    # Same fixed error for missing, active and revoked users.
                    raise InviteError("Pending invitation unavailable; no invitation was changed", 404)
                if db.execute("SELECT 1 FROM invitations WHERE code_hash=?", (digest,)).fetchone():
                    raise InviteError("Could not reissue invitation; no invitation was changed")
                # Pending codes have no historical login identity to retain.
                # Keep one row per pending user rather than accumulating codes.
                db.execute("DELETE FROM invitations WHERE user_id=?", (user["id"],))
                db.execute("INSERT INTO invitations(code_hash,user_id,expires_at) VALUES (?,?,?)", (digest, user["id"], expires))
        except sqlite3.IntegrityError:
            # The context rolls back deletion too; never expose raw SQL errors.
            raise InviteError("Could not reissue invitation; no invitation was changed") from None
        return {"username": username, "invite_code": invitation, "expires_at": expires}

    def register(self, invite_code: str, username: str, password: str) -> dict:
        username = normalize_username(username)
        validate_password(password)
        if not isinstance(invite_code, str) or not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", invite_code):
            raise InviteError("Invitation or credentials are invalid", 401)
        encoded = _password_hash(password)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            user = db.execute("""SELECT u.* FROM invitations i JOIN users u ON u.id=i.user_id
                WHERE i.code_hash=? AND i.used_at IS NULL AND i.revoked=0
                  AND i.expires_at>? AND u.status='pending' AND u.username=?""",
                (_digest(invite_code), self.clock(), username)).fetchone()
            if user is None:
                raise InviteError("Invitation or credentials are invalid", 401)
            db.execute("UPDATE users SET password_hash=?,status='active' WHERE id=?", (encoded, user["id"]))
            db.execute("UPDATE invitations SET used_at=? WHERE code_hash=?", (self.clock(), _digest(invite_code)))
            return self._new_session(db, user)

    def login(self, username: str, password: str) -> dict:
        username = normalize_username(username)
        validate_password(password)
        with self._connect() as db:
            user = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        valid = _verify_password(password, user["password_hash"] if user else None)
        if not valid or user is None or user["status"] != "active":
            raise InviteError("Username or password is invalid", 401)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute("SELECT * FROM users WHERE id=? AND status='active'", (user["id"],)).fetchone()
            if current is None:
                raise InviteError("Username or password is invalid", 401)
            # Bound retained sessions; a new login replaces the oldest if needed.
            db.execute("DELETE FROM auth_sessions WHERE expires_at<=? OR revoked=1", (self.clock(),))
            old = db.execute("SELECT token_hash FROM auth_sessions WHERE user_id=? ORDER BY expires_at DESC LIMIT -1 OFFSET 9", (user["id"],)).fetchall()
            db.executemany("DELETE FROM auth_sessions WHERE token_hash=?", [(row[0],) for row in old])
            return self._new_session(db, current)

    def authenticate(self, token: str) -> dict:
        if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", token):
            raise InviteError("Authentication required", 401)
        with self._connect() as db:
            user = db.execute("""SELECT u.* FROM auth_sessions s JOIN users u ON u.id=s.user_id
                WHERE s.token_hash=? AND s.expires_at>? AND s.revoked=0 AND u.status='active'""",
                (_digest(token), self.clock())).fetchone()
        if user is None:
            raise InviteError("Authentication required", 401)
        return dict(user)

    def logout(self, token: str) -> None:
        self.authenticate(token)
        with self._connect() as db:
            db.execute("UPDATE auth_sessions SET revoked=1 WHERE token_hash=?", (_digest(token),))

    def revoke_user(self, username: str) -> bool:
        username = normalize_username(username)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            user = db.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
            if user is None:
                return False
            db.execute("UPDATE users SET status='revoked' WHERE id=?", (user["id"],))
            db.execute("UPDATE invitations SET revoked=1 WHERE user_id=?", (user["id"],))
            db.execute("UPDATE auth_sessions SET revoked=1 WHERE user_id=?", (user["id"],))
            return True

    def list_worker_bindings(self) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT id,username,status,upstream_url,upstream_token,allow_uploads FROM users WHERE status IN ('pending','active') ORDER BY created_at,id").fetchall()
        return [{**dict(row), "allow_uploads": bool(row["allow_uploads"])} for row in rows]

    worker_records = list_worker_bindings

    def authorize_worker(self, user_id: str, port: int, token: str) -> None:
        """Recheck a captured spawn binding while the worker lifetime lock is held."""
        with self._connect() as db:
            row = db.execute("SELECT upstream_url,upstream_token,status FROM users WHERE id=?", (user_id,)).fetchone()
            lifecycle = db.execute("SELECT state FROM account_lifecycle WHERE user_id=?", (user_id,)).fetchone()
        if (row is None or row["status"] not in {"pending", "active"}
                or row["upstream_url"] != f"http://127.0.0.1:{port}"
                or not hmac.compare_digest(row["upstream_token"], token)
                or (lifecycle is not None and lifecycle["state"] != "retained")):
            raise InviteError("Worker identity is no longer authorized")

    def reserve_action(self, user_id: str, kind: str, *, body_bytes: int = 0) -> None:
        limits = {"query": QUERY_DAILY_LIMIT, "upload": UPLOAD_DAILY_LIMIT, "sessions": SESSION_CREATION_LIMIT,
                  "voice": QUERY_DAILY_LIMIT, "feedback": 200, "research_write": 200, "workbench_write": 200}
        if kind not in limits or type(body_bytes) is not int or body_bytes < 0:
            raise InviteError("Invalid quota request")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._reserve_action(db, user_id, kind, body_bytes, limits[kind])

    def _reserve_action(self, db, user_id: str, kind: str, body_bytes: int, limit: int, *, units: int = 1) -> None:
        day = -1 if kind == "sessions" else int(self.clock() // 86400)
        if not db.execute("SELECT 1 FROM users WHERE id=? AND status='active'", (user_id,)).fetchone():
            raise InviteError("Authentication required", 401)
        previous = db.execute("SELECT count,bytes FROM usage WHERE user_id=? AND day=? AND kind=?", (user_id, day, kind)).fetchone()
        count, size = (previous[0], previous[1]) if previous else (0, 0)
        if kind == "query":
            global_count = db.execute("SELECT COALESCE(SUM(count),0) FROM usage WHERE day=? AND kind='query'", (day,)).fetchone()[0]
            if global_count + units > GLOBAL_QUERY_DAILY_LIMIT:
                raise InviteError("Service daily query quota reached; contact the instance owner", 429)
        if count + units > limit or (kind == "upload" and size + body_bytes > UPLOAD_DAILY_BYTES):
            raise InviteError("Usage quota reached; contact the instance owner", 429)
        db.execute("INSERT INTO usage VALUES (?,?,?,?,?) ON CONFLICT(user_id,day,kind) DO UPDATE SET count=excluded.count,bytes=excluded.bytes", (user_id, day, kind, count + units, size + body_bytes))

    def reserve_query(self, user_id: str, session_id: str, request_id: str, fingerprint: str, *, body_bytes: int = 0,
                      units: int = 1) -> bool:
        """Bind an authenticated request and charge daily quota exactly once.

        This ledger is not proof of provider execution. The isolated worker's
        journal decides running/completed/unknown. No automatic retry/refund.
        Accepted IDs are permanent; only new requests require storage admission.
        """
        from .request_journal import validate_request_reference
        try:
            validate_request_reference(session_id, request_id)
        except ValueError:
            raise InviteError("Invalid request reference", 422) from None
        if (not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint)
                or type(body_bytes) is not int or body_bytes < 0 or type(units) is not int or not 1 <= units <= 6):
            raise InviteError("Invalid query reservation", 422)
        if units != 1:
            # Default chat tombstones keep their historical identity. A bounded
            # multi-step approval reserves all possible calls in one atomic
            # transaction; changed budgets cannot reuse an accepted ID.
            fingerprint = hashlib.sha256((fingerprint + ':units:' + str(units)).encode()).hexdigest()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM users WHERE id=? AND status='active'", (user_id,)).fetchone():
                raise InviteError("Authentication required", 401)
            previous = db.execute("SELECT fingerprint FROM query_reservations WHERE user_id=? AND session_id=? AND request_id=?",
                                  (user_id, session_id, request_id)).fetchone()
            if previous:
                if previous[0] != fingerprint:
                    raise InviteError("Request ID belongs to different content", 409)
                return False
            if self._used_bytes(db) > self.max_storage_bytes:
                raise InviteError("Request registry storage quota reached; contact the instance owner to expand storage", 429)
            self._reserve_action(db, user_id, "query", body_bytes, QUERY_DAILY_LIMIT, units=units)
            db.execute("INSERT INTO query_reservations VALUES (?,?,?,?,?)", (user_id, session_id, request_id, fingerprint, self.clock()))
            # Measure actual allocated live pages (including indexes and usage),
            # not an estimated per-row count. Rollback also undoes quota charges.
            # Existing free pages can be reused; no ID is evicted to make space.
            if self._used_bytes(db) > self.max_storage_bytes:
                raise InviteError("Request registry storage quota reached; contact the instance owner to expand storage", 429)
            return True

    @staticmethod
    def _used_bytes(db) -> int:
        return (db.execute("PRAGMA page_count").fetchone()[0]
                - db.execute("PRAGMA freelist_count").fetchone()[0]) * db.execute("PRAGMA page_size").fetchone()[0]
