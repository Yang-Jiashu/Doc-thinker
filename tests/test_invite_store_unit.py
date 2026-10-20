import concurrent.futures
import os
import sqlite3
from pathlib import Path

import pytest

from docthinker.server.invite_store import InviteError, InviteStore, SESSION_SECONDS

PASSWORD = "offline-test-password-123"


@pytest.fixture
def registry(tmp_path):
    store = InviteStore.initialize((tmp_path / "private-registry").resolve())
    now = [1700000000.0]
    store.clock = lambda: now[0]
    return store, now


def enroll(store, name="alice", port=8101):
    invite = store.add_user(name, port)
    return invite, store.register(invite["invite_code"], name, PASSWORD)


def test_registry_stores_only_hashes_of_browser_credentials_and_owner_only_files(registry):
    store, _ = registry
    invite, session = enroll(store)
    assert store.root.stat().st_mode & 0o777 == 0o700
    assert store.path.stat().st_mode & 0o777 == 0o600
    raw = store.path.read_bytes()
    assert invite["invite_code"].encode() not in raw
    assert session["access_token"].encode() not in raw
    assert PASSWORD.encode() not in raw
    with sqlite3.connect(store.path) as db:
        password_hash = db.execute("SELECT password_hash FROM users").fetchone()[0]
    assert password_hash.startswith("scrypt$32768$8$1$")
    assert set(session) == {"access_token", "expires_in", "user"}
    assert set(session["user"]) == {"id", "username"}
    assert session["expires_in"] == SESSION_SECONDS


def test_invitation_is_bound_to_exact_reserved_username_and_single_use(registry):
    store, _ = registry
    invitation = store.add_user("Alice", 8101)
    with pytest.raises(InviteError, match="invalid"):
        store.register(invitation["invite_code"], "bob", PASSWORD)
    registered = store.register(invitation["invite_code"], "ALICE", PASSWORD)
    assert registered["user"]["username"] == "alice"
    with pytest.raises(InviteError, match="invalid"):
        store.register(invitation["invite_code"], "alice", PASSWORD)


def test_expired_invite_and_expired_session_are_rejected(registry):
    store, now = registry
    invite = store.add_user("alice", 8101, expires_hours=1)
    now[0] += 3600
    with pytest.raises(InviteError):
        store.register(invite["invite_code"], "alice", PASSWORD)
    _, auth = enroll(store, "bob", 8102)
    now[0] += SESSION_SECONDS
    with pytest.raises(InviteError) as error:
        store.authenticate(auth["access_token"])
    assert error.value.status == 401


def test_wrong_login_missing_user_and_revoked_user_have_same_error(registry):
    store, _ = registry
    _, auth = enroll(store)
    failures = []
    for name, password in [("alice", "wrong-password-123"), ("missing", PASSWORD)]:
        with pytest.raises(InviteError) as error:
            store.login(name, password)
        failures.append((error.value.status, str(error.value)))
    store.revoke_user("alice")
    with pytest.raises(InviteError) as error:
        store.login("alice", PASSWORD)
    assert failures[0] == failures[1] == (error.value.status, str(error.value))
    with pytest.raises(InviteError):
        store.authenticate(auth["access_token"])
    assert store.list_worker_bindings() == []


def test_logout_revokes_only_that_browser_session_and_revoke_user_revokes_all(registry):
    store, _ = registry
    _, auth = enroll(store)
    other = store.login("alice", PASSWORD)
    store.logout(auth["access_token"])
    with pytest.raises(InviteError):
        store.authenticate(auth["access_token"])
    assert store.authenticate(other["access_token"])["username"] == "alice"
    assert store.revoke_user("alice")
    with pytest.raises(InviteError):
        store.authenticate(other["access_token"])


def test_atomic_registration_allows_only_one_winner(registry):
    store, _ = registry
    invitation = store.add_user("alice", 8101)
    def attempt(_):
        try:
            store.register(invitation["invite_code"], "alice", PASSWORD)
            return True
        except InviteError:
            return False
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(attempt, range(2))) == [False, True]


def test_expired_pending_invite_can_be_reissued_without_replacing_worker_or_other_users(registry):
    store, now = registry
    invitations = [store.add_user("tester" + str(index), 8101 + index, allow_uploads=index == 0) for index in range(5)]
    before = store.list_worker_bindings()
    first = next(user for user in before if user["username"] == "tester0")
    marker = store.root / "workers" / first["id"] / "retained.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("synthetic retained project")
    now[0] += 48 * 3600
    replacement = store.reissue_invitation("Tester0", expires_hours=12)
    assert replacement["expires_at"] == now[0] + 12 * 3600
    assert replacement["invite_code"] != invitations[0]["invite_code"]
    assert store.list_worker_bindings() == before
    assert marker.read_text() == "synthetic retained project"
    with pytest.raises(InviteError):
        store.register(invitations[0]["invite_code"], "tester0", PASSWORD)
    auth = store.register(replacement["invite_code"], "tester0", PASSWORD)
    assert auth["user"]["id"] == first["id"]
    assert store.authenticate(auth["access_token"])["allow_uploads"] == 1
    with store._connect() as db:
        assert db.execute("SELECT count(*) FROM users").fetchone()[0] == 5
        assert db.execute("SELECT count(*) FROM invitations").fetchone()[0] == 5


def test_each_successful_reissue_replaces_previous_code_and_has_exact_expiry(registry):
    store, now = registry
    original = store.add_user("alice", 8101)
    first = store.reissue_invitation("alice")
    second = store.reissue_invitation("alice", expires_hours=1)
    for expired in (original, first):
        with pytest.raises(InviteError):
            store.register(expired["invite_code"], "alice", PASSWORD)
    with store._connect() as db:
        assert db.execute("SELECT count(*) FROM invitations").fetchone()[0] == 1
    now[0] += 3600
    with pytest.raises(InviteError):
        store.register(second["invite_code"], "alice", PASSWORD)
    newest = store.reissue_invitation("alice")
    assert store.register(newest["invite_code"], "alice", PASSWORD)["user"]["username"] == "alice"


def test_reissue_cannot_reset_active_revoked_or_missing_users_and_errors_match(registry):
    store, _ = registry
    _, active = enroll(store, "alice", 8101)
    revoked = store.add_user("bobby", 8102)
    store.revoke_user("bobby")
    errors = []
    for username in ("alice", "bobby", "missing"):
        with pytest.raises(InviteError) as error:
            store.reissue_invitation(username)
        errors.append((error.value.status, str(error.value)))
    assert errors[0] == errors[1] == errors[2]
    assert active["access_token"] not in repr(errors) and revoked["invite_code"] not in repr(errors)
    assert store.authenticate(active["access_token"])["username"] == "alice"
    with pytest.raises(InviteError):
        store.register(revoked["invite_code"], "bobby", PASSWORD)


@pytest.mark.parametrize("lifetime", [0, 169, True, None, "12", 1.5])
def test_invalid_reissue_expiry_cannot_invalidate_previous_code(registry, lifetime):
    store, _ = registry
    original = store.add_user("alice", 8101)
    with pytest.raises(InviteError):
        store.reissue_invitation("alice", expires_hours=lifetime)
    assert store.register(original["invite_code"], "alice", PASSWORD)["user"]["username"] == "alice"


def test_reissue_failure_rolls_back_old_code_removal_and_hides_sql_error(registry):
    store, _ = registry
    original = store.add_user("alice", 8101)
    with store._connect() as db:
        db.execute("CREATE TRIGGER fail_reissue BEFORE INSERT ON invitations BEGIN SELECT RAISE(ABORT,'PRIVATE_SQL_SENTINEL'); END")
    with pytest.raises(InviteError) as error:
        store.reissue_invitation("alice")
    assert "PRIVATE_SQL_SENTINEL" not in str(error.value)
    assert store.register(original["invite_code"], "alice", PASSWORD)["user"]["username"] == "alice"


def test_random_code_collision_cannot_revalidate_old_code_or_touch_another_user(registry, monkeypatch):
    from docthinker.server import invite_store
    store, _ = registry
    first = store.add_user("alice", 8101)
    other = store.add_user("bobby", 8102)
    for collision in (first, other):
        monkeypatch.setattr(invite_store.secrets, "token_urlsafe", lambda _size, value=collision["invite_code"]: value)
        with pytest.raises(InviteError):
            store.reissue_invitation("alice")
    monkeypatch.undo()
    assert store.register(first["invite_code"], "alice", PASSWORD)["user"]["username"] == "alice"
    assert store.register(other["invite_code"], "bobby", PASSWORD)["user"]["username"] == "bobby"


def test_reissue_winning_a_race_rejects_registration_using_old_code(registry, monkeypatch):
    import threading
    from docthinker.server import invite_store
    store, _ = registry
    original = store.add_user("alice", 8101)
    hashed, release = threading.Event(), threading.Event()
    original_hash = invite_store._password_hash
    def delayed_hash(password):
        result = original_hash(password)
        hashed.set()
        assert release.wait(5)
        return result
    monkeypatch.setattr(invite_store, "_password_hash", delayed_hash)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(store.register, original["invite_code"], "alice", PASSWORD)
        try:
            assert hashed.wait(5)
            replacement = store.reissue_invitation("alice")
        finally:
            release.set()
        with pytest.raises(InviteError):
            pending.result(timeout=5)
    assert store.register(replacement["invite_code"], "alice", PASSWORD)["user"]["username"] == "alice"


def test_concurrent_reissues_leave_one_usable_code_and_one_original_identity(registry):
    import threading
    store, _ = registry
    store.add_user("alice", 8101)
    before = store.list_worker_bindings()
    barrier = threading.Barrier(2)
    def rotate(_):
        barrier.wait(timeout=5)
        return store.reissue_invitation("alice")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        replacements = list(pool.map(rotate, range(2)))
    assert store.list_worker_bindings() == before
    with store._connect() as db:
        assert db.execute("SELECT count(*) FROM invitations").fetchone()[0] == 1
    winners = []
    for replacement in replacements:
        try:
            winners.append(store.register(replacement["invite_code"], "alice", PASSWORD))
        except InviteError:
            pass
    assert len(winners) == 1 and winners[0]["user"]["id"] == before[0]["id"]


def test_registration_winning_a_race_cannot_be_reset_by_waiting_reissue(registry, monkeypatch):
    import threading
    store, _ = registry
    original = store.add_user("alice", 8101)
    registered, release, rotating = threading.Event(), threading.Event(), threading.Event()
    new_session = store._new_session
    def held_registration(db, user):
        result = new_session(db, user)
        registered.set()
        assert release.wait(5)
        return result
    def rotate():
        rotating.set()
        return store.reissue_invitation("alice")
    monkeypatch.setattr(store, "_new_session", held_registration)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        registration = pool.submit(store.register, original["invite_code"], "alice", PASSWORD)
        try:
            assert registered.wait(5)
            replacement = pool.submit(rotate)
            assert rotating.wait(5)
        finally:
            release.set()
        auth = registration.result(timeout=5)
        with pytest.raises(InviteError, match="Pending invitation unavailable"):
            replacement.result(timeout=5)
    assert store.authenticate(auth["access_token"])["username"] == "alice"


def test_quota_is_atomic_persistent_and_uses_utc_day(registry):
    store, now = registry
    _, auth = enroll(store)
    uid = auth["user"]["id"]
    def attempt(_):
        try:
            store.reserve_action(uid, "query")
            return True
        except InviteError as error:
            assert error.status == 429
            return False
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(attempt, range(56))) == 50
    reopened = InviteStore(store.root, clock=store.clock)
    with pytest.raises(InviteError):
        reopened.reserve_action(uid, "query")
    now[0] += 86400
    reopened.reserve_action(uid, "query")
    for _ in range(100):
        reopened.reserve_action(uid, "sessions")
    now[0] += 86400
    with pytest.raises(InviteError):
        reopened.reserve_action(uid, "sessions")


def test_upload_count_and_bytes_are_bounded(registry):
    store, _ = registry
    _, auth = enroll(store)
    uid = auth["user"]["id"]
    store.reserve_action(uid, "upload", body_bytes=50 * 1024 * 1024)
    with pytest.raises(InviteError) as error:
        store.reserve_action(uid, "upload", body_bytes=1)
    assert error.value.status == 429


@pytest.mark.parametrize("name", ["aa", "with spaces", " ../alice", "中文用户名", "a" * 33])
def test_username_validation(registry, name):
    with pytest.raises(InviteError):
        registry[0].add_user(name, 8101)


@pytest.mark.parametrize("password", ["short", "a" * 129, " " * 12])
def test_password_validation(registry, password):
    store, _ = registry
    invite = store.add_user("alice", 8101)
    with pytest.raises(InviteError):
        store.register(invite["invite_code"], "alice", password)


def test_port_is_unique_and_fixed_loopback_only_and_user_count_is_bounded(registry):
    store, _ = registry
    store.add_user("alice", 8101)
    with pytest.raises(InviteError):
        store.add_user("alice", 8102)
    with pytest.raises(InviteError):
        store.add_user("bob", 8101)
    for i in range(4):
        store.add_user(f"person{i}", 8102 + i)
    with pytest.raises(InviteError):
        store.add_user("sixth", 8106)
    records = store.worker_records()
    assert len(records) == 5 and all(r["upstream_url"].startswith("http://127.0.0.1:") for r in records)
    assert all(not r["allow_uploads"] for r in records)
    assert all(type(r["allow_uploads"]) is bool for r in records)
    assert len({r["upstream_token"] for r in records}) == 5


def test_internal_worker_tokens_cannot_be_reused_across_users(registry):
    store, _ = registry
    token = "synthetic-only-distinct-worker-token-12345"
    store.add_user("alice", 8101, upstream_token=token)
    with pytest.raises(InviteError, match="distinct"):
        store.add_user("bobby", 8102, upstream_token=token)


def test_registry_refuses_missing_insecure_symlink_or_repository_storage(tmp_path, monkeypatch):
    with pytest.raises(InviteError):
        InviteStore((tmp_path / "missing").resolve())
    with pytest.raises(InviteError):
        InviteStore.initialize(Path(__file__).resolve().parents[1] / "test-do-not-create")
    insecure = (tmp_path / "insecure").resolve()
    insecure.mkdir(mode=0o755)
    with pytest.raises(InviteError):
        InviteStore.initialize(insecure)
    target = (tmp_path / "secure").resolve()
    store = InviteStore.initialize(target)
    store.path.chmod(0o644)
    with pytest.raises(InviteError):
        store.list_worker_bindings()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(InviteError):
        InviteStore(linked)
    monkeypatch.delenv("DOCTHINKER_INVITE_DATA_DIR", raising=False)
    with pytest.raises(InviteError):
        InviteStore.from_environment()


def test_cli_prints_only_invitation_not_internal_token_or_password(tmp_path, monkeypatch, capsys):
    from docthinker.server.invite_admin import main
    import json
    directory = (tmp_path / "cli-private").resolve()
    monkeypatch.setenv("DOCTHINKER_INVITE_DATA_DIR", str(directory))
    assert main(["init"]) == 0
    capsys.readouterr()
    assert main(["add-user", "--username", "alice", "--upstream-port", "8101"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert set(payload) == {"username", "invite_code", "expires_at"}
    store = InviteStore(directory)
    assert store.list_worker_bindings()[0]["upstream_token"] not in output
    assert main(["reissue-invite", "--username", "alice", "--expires-hours", "12"]) == 0
    reissued_output = capsys.readouterr().out
    replacement = json.loads(reissued_output)
    assert set(replacement) == {"username", "invite_code", "expires_at"}
    assert payload["invite_code"] not in reissued_output
    assert store.list_worker_bindings()[0]["upstream_token"] not in reissued_output
    with pytest.raises(InviteError):
        store.register(payload["invite_code"], "alice", PASSWORD)
    auth = store.register(replacement["invite_code"], "alice", PASSWORD)
    assert main(["reissue-invite", "--username", "alice"]) == 1
    failed = capsys.readouterr()
    assert failed.out == "" and "Pending invitation unavailable" in failed.err
    assert all(secret not in failed.err for secret in (payload["invite_code"], replacement["invite_code"], auth["access_token"], PASSWORD))
    assert main(["revoke-user", "--username", "alice"]) == 0
    assert store.list_worker_bindings() == []


def test_request_reservation_charges_once_across_threads_restart_and_utc_days(registry):
    from docthinker.server.request_journal import request_fingerprint
    store, now = registry
    _, auth = enroll(store)
    uid = auth["user"]["id"]
    rid = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    fingerprint = request_fingerprint({"session_id": "#00001", "question": "synthetic"})
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(lambda _: store.reserve_query(uid, "#00001", rid, fingerprint), range(8))) == 1
    reopened = InviteStore(store.root, clock=store.clock)
    assert not reopened.reserve_query(uid, "#00001", rid, fingerprint)
    now[0] += 86400
    assert not reopened.reserve_query(uid, "#00001", rid, fingerprint)
    with reopened._connect() as db:
        assert db.execute("SELECT SUM(count) FROM usage WHERE kind='query'").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM query_reservations").fetchone()[0] == 1


def test_reservation_conflict_is_409_without_mutating_quota(registry):
    from docthinker.server.request_journal import request_fingerprint
    store, _ = registry
    _, auth = enroll(store)
    uid = auth["user"]["id"]
    rid = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    first = request_fingerprint({"session_id": "#00001", "question": "first"})
    store.reserve_query(uid, "#00001", rid, first)
    with pytest.raises(InviteError) as error:
        store.reserve_query(uid, "#00001", rid, request_fingerprint({"session_id": "#00001", "question": "changed"}))
    assert error.value.status == 409
    with store._connect() as db:
        assert db.execute("SELECT SUM(count) FROM usage WHERE kind='query'").fetchone()[0] == 1


def test_same_reference_is_independent_for_users_and_sessions(registry):
    from docthinker.server.request_journal import request_fingerprint
    store, _ = registry
    _, alice = enroll(store)
    _, bob = enroll(store, "bobby", 8102)
    rid = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    for uid, sid in [(alice["user"]["id"], "#00001"), (alice["user"]["id"], "#00002"), (bob["user"]["id"], "#00001")]:
        assert store.reserve_query(uid, sid, rid, request_fingerprint({"session_id": sid, "question": "synthetic"}))
    with store._connect() as db:
        assert db.execute("SELECT SUM(count) FROM usage WHERE kind='query'").fetchone()[0] == 3


def test_global_daily_quota_is_atomic_counts_revoked_users_and_resets(registry, monkeypatch):
    from docthinker.server import invite_store as module
    from docthinker.server.request_journal import request_fingerprint
    monkeypatch.setattr(module, "GLOBAL_QUERY_DAILY_LIMIT", 5)
    store, now = registry
    _, alice = enroll(store)
    _, bob = enroll(store, "bobby", 8102)
    alice_id, bob_id = alice["user"]["id"], bob["user"]["id"]
    for _ in range(3):
        store.reserve_action(alice_id, "query")
    store.revoke_user("alice")
    def reserve(index):
        rid = f"{index:08x}-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        try:
            return store.reserve_query(bob_id, "#00001", rid, request_fingerprint({"session_id": "#00001", "question": "synthetic"}))
        except InviteError as error:
            assert error.status == 429
            return False
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(reserve, range(8))) == 2
    with pytest.raises(InviteError) as error:
        store.reserve_action(bob_id, "query")
    assert error.value.status == 429
    with store._connect() as db:
        assert db.execute("SELECT SUM(count) FROM usage WHERE kind='query'").fetchone()[0] == 5
    now[0] += 86400
    store.reserve_action(bob_id, "query")


def test_additive_reservation_startup_migration_preserves_existing_auth(registry):
    store, _ = registry
    _, auth = enroll(store)
    with store._connect() as db:
        db.execute("DROP TABLE query_reservations")
    reopened = InviteStore(store.root, clock=store.clock)
    assert reopened.authenticate(auth["access_token"])["id"] == auth["user"]["id"]
    with reopened._connect() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM query_reservations").fetchone()[0] == 0


def reservation_id(index):
    return f"{index:08x}-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


def test_old_ten_thousand_reservations_do_not_disable_a_long_lived_account(registry):
    store, _ = registry
    _, auth = enroll(store)
    uid, fingerprint = auth["user"]["id"], "a" * 64
    with store._connect() as db:
        db.executemany("INSERT INTO query_reservations VALUES (?,?,?,?,?)", (
            (uid, "#00001", reservation_id(index), fingerprint, store.clock() - 86400)
            for index in range(10_000)))
    reopened = InviteStore(store.root, clock=store.clock)
    assert reopened.reserve_query(uid, "#00001", reservation_id(10_000), fingerprint)
    assert not reopened.reserve_query(uid, "#00001", reservation_id(0), fingerprint)
    # An operator may lower the budget below an existing registry's live size.
    # Authentication/conflict/replay still work, but new work cannot be charged.
    lowered = InviteStore(store.root, clock=store.clock, max_storage_bytes=65_536)
    assert lowered.authenticate(auth["access_token"])["id"] == uid
    assert not lowered.reserve_query(uid, "#00001", reservation_id(0), fingerprint)
    with pytest.raises(InviteError, match="storage quota"):
        lowered.reserve_query(uid, "#00001", reservation_id(10_001), fingerprint)
    with reopened._connect() as db:
        assert db.execute("SELECT count(*) FROM query_reservations").fetchone()[0] == 10_001
        assert db.execute("SELECT SUM(count) FROM usage WHERE kind='query'").fetchone()[0] == 1


def test_registry_budget_rejects_new_requests_atomically_but_retains_replays_and_allows_expansion(registry):
    store, now = registry
    _, auth = enroll(store)
    uid, fingerprint = auth["user"]["id"], "a" * 64
    bounded = InviteStore(store.root, clock=store.clock, max_storage_bytes=65_536)
    accepted = 0
    for index in range(2000):
        if index % 20 == 0:
            now[0] += 86400
        try:
            bounded.reserve_query(uid, "#00001", reservation_id(index), fingerprint, body_bytes=7)
        except InviteError as error:
            assert error.status == 429 and "storage quota" in str(error)
            break
        accepted += 1
    else:
        pytest.fail("Physical database growth was not bounded")
    assert accepted > 0
    with bounded._connect() as db:
        assert bounded._used_bytes(db) <= bounded.max_storage_bytes
        assert db.execute("SELECT count(*) FROM query_reservations").fetchone()[0] == accepted
        assert db.execute("SELECT SUM(count),SUM(bytes) FROM usage WHERE kind='query'").fetchone()[:] == (accepted, accepted * 7)
    # Pressure, restart and expansion never cause old requests to be charged again.
    reopened = InviteStore(store.root, clock=store.clock, max_storage_bytes=65_536)
    assert not reopened.reserve_query(uid, "#00001", reservation_id(0), fingerprint)
    with pytest.raises(InviteError) as conflict:
        reopened.reserve_query(uid, "#00001", reservation_id(0), "b" * 64)
    assert conflict.value.status == 409
    expanded = InviteStore(store.root, clock=store.clock, max_storage_bytes=131_072)
    assert expanded.reserve_query(uid, "#00001", reservation_id(accepted), fingerprint)
    assert not expanded.reserve_query(uid, "#00001", reservation_id(0), fingerprint)


def test_concurrent_storage_admission_cannot_overfill_or_overcharge(registry):
    store, now = registry
    _, auth = enroll(store)
    uid, fingerprint = auth["user"]["id"], "a" * 64
    # Imported history puts admission near a small budget without consuming today.
    with store._connect() as db:
        db.executemany("INSERT INTO query_reservations VALUES (?,?,?,?,?)", (
            (uid, "#00001", reservation_id(index), fingerprint, now[0] - 86400)
            for index in range(200)))
        current = store._used_bytes(db)
        page = db.execute("PRAGMA page_size").fetchone()[0]
    bounded = InviteStore(store.root, clock=store.clock, max_storage_bytes=max(65_536, current + page))
    def attempt(index):
        try:
            return bounded.reserve_query(uid, "#00001", reservation_id(index + 200), fingerprint)
        except InviteError as error:
            assert error.status == 429 and "storage quota" in str(error)
            return False
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        accepted = sum(pool.map(attempt, range(40)))
    assert 0 < accepted < 40  # Actually exercise both admission and page pressure.
    with bounded._connect() as db:
        assert bounded._used_bytes(db) <= bounded.max_storage_bytes
        assert db.execute("SELECT count(*) FROM query_reservations").fetchone()[0] == 200 + accepted
        assert db.execute("SELECT COALESCE(SUM(count),0) FROM usage WHERE kind='query'").fetchone()[0] == accepted


def test_storage_budget_reuses_free_pages_instead_of_using_file_length(registry):
    store, _ = registry
    _, auth = enroll(store)
    with store._connect() as db:
        db.execute("CREATE TABLE temporary_padding (value BLOB)")
        db.execute("INSERT INTO temporary_padding VALUES (zeroblob(200000))")
        db.execute("DROP TABLE temporary_padding")
    assert store.path.stat().st_size > 65_536
    bounded = InviteStore(store.root, clock=store.clock, max_storage_bytes=65_536)
    assert bounded.reserve_query(auth["user"]["id"], "#00001", reservation_id(0), "a" * 64)


def test_exact_budget_is_admitted_but_post_insert_overflow_rolls_back_quota(registry, monkeypatch):
    store, _ = registry
    _, auth = enroll(store)
    uid = auth["user"]["id"]
    # Page allocation can stay unchanged for a row, or jump by multiple pages
    # when a tree splits. Exercise both boundaries without SQLite-version shape.
    monkeypatch.setattr(store, "_used_bytes", lambda db: store.max_storage_bytes)
    assert store.reserve_query(uid, "#00001", reservation_id(0), "a" * 64, body_bytes=7)
    measurements = iter([store.max_storage_bytes, store.max_storage_bytes + 4096])
    monkeypatch.setattr(store, "_used_bytes", lambda db: next(measurements))
    with pytest.raises(InviteError, match="storage quota"):
        store.reserve_query(uid, "#00001", reservation_id(1), "a" * 64, body_bytes=7)
    with store._connect() as db:
        assert db.execute("SELECT count(*) FROM query_reservations").fetchone()[0] == 1
        assert db.execute("SELECT SUM(count),SUM(bytes) FROM usage WHERE kind='query'").fetchone()[:] == (1, 7)


@pytest.mark.parametrize("invalid", [True, False, "65536", 65535, (1 << 50) + 1, 65536.0, -1])
def test_invalid_explicit_registry_budget_is_rejected(registry, invalid):
    with pytest.raises(InviteError, match="storage quota"):
        InviteStore(registry[0].root, max_storage_bytes=invalid)


@pytest.mark.parametrize("invalid", ["", "0", "65535", "1e9", "100000\n", " 65536", str((1 << 50) + 1), "private-not-a-number"])
def test_invalid_registry_budget_fails_before_creating_files_and_does_not_echo(tmp_path, monkeypatch, invalid):
    monkeypatch.setenv("DOCTHINKER_INVITE_REGISTRY_BYTES", invalid)
    root = (tmp_path / "not-created").resolve()
    with pytest.raises(InviteError, match="storage quota") as error:
        InviteStore.initialize(root)
    assert not root.exists()
    assert "private-not-a-number" not in str(error.value)


def test_registry_budget_default_environment_and_explicit_override(registry, monkeypatch):
    from docthinker.server.invite_store import DEFAULT_REGISTRY_BYTES
    store, _ = registry
    monkeypatch.delenv("DOCTHINKER_INVITE_REGISTRY_BYTES", raising=False)
    assert InviteStore(store.root).max_storage_bytes == DEFAULT_REGISTRY_BYTES
    monkeypatch.setenv("DOCTHINKER_INVITE_REGISTRY_BYTES", "131072")
    assert InviteStore(store.root).max_storage_bytes == 131_072
    assert InviteStore(store.root, max_storage_bytes=65_536).max_storage_bytes == 65_536
