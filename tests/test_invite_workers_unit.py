"""Offline worker-supervision tests: fake subprocesses, no model/server startup."""

import builtins
import os
import signal
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from docthinker.server import invite_workers as workers


def record(index=1, **updates):
    return {"id": f"{index:032x}", "username": f"user{index}", "status": "active",
            "upstream_url": f"http://127.0.0.1:{18000 + index}",
            "upstream_token": f"test_internal_{index}_" + "x" * 48,
            "allow_uploads": False, **updates}


def model_environment(**updates):
    return {workers.SHARING_ACK_ENV: workers.SHARING_ACK_VALUE,
            "LLM_BINDING_API_KEY": "offline-only-model-credential",
            "LLM_VLM_HOST": "https://provider.example.invalid/v1",
            "LLM_MODEL": "offline-only-model", **updates}


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    rows = [record(1), record(2)]
    return SimpleNamespace(root=root, rows=rows, list_worker_bindings=lambda: list(rows))


def fake_runner(store, monkeypatch, environment=None):
    monkeypatch.setattr(workers, "ensure_ports_available", Mock())
    started, stopped = [], []

    def popen(*args, **kwargs):
        process = SimpleNamespace(poll=Mock(return_value=None), pid=100 + len(started))
        started.append((process, args, kwargs))
        return process

    runner = workers.WorkerSupervisor(store, model_environment() if environment is None else environment,
                                      popen=popen, stopper=stopped.append)
    return runner, started, stopped


def test_missing_ack_cannot_read_store_import_runtime_or_start_child(store, monkeypatch):
    reader = Mock(side_effect=AssertionError("Do not read store before permission acknowledgement"))
    store.list_worker_bindings = reader
    runner, started, _ = fake_runner(store, monkeypatch, {})
    with pytest.raises(workers.WorkerConfigurationError, match="startup is blocked"):
        runner.start()
    reader.assert_not_called()
    assert not started and not (store.root / "workers").exists()
    monkeypatch.delenv(workers.SHARING_ACK_ENV, raising=False)
    importer = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name not in {"dotenv", "uvicorn", "mobile_app", "invite_store"}
        return importer(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(workers.WorkerConfigurationError):
        workers.worker_main()
    assert workers.main(["--check"]) == 2


@pytest.mark.parametrize("updates", [
    {"LLM_BINDING_API_KEY": ""}, {"LLM_MODEL": ""}, {"LLM_VLM_HOST": ""},
    {"LLM_BINDING_API_KEY": "EMPTY"}, {"LLM_BINDING_API_KEY": "secret\r\nInjected: header"},
    {"LLM_MODEL": "first,second"}, {"LLM_MODEL": "model with spaces"},
    {"LLM_VLM_HOST": "http://remote.example.invalid/v1"},
    {"LLM_VLM_HOST": "https://user:secret@provider.example.invalid/v1"},
    {"LLM_VLM_HOST": "https://provider.example.invalid/v1?secret=value"},
    {"EMBEDDING_BINDING_API_KEY": "another-secret"}, {"RERANK_API_KEY": "another-secret"},
    {"VOICE_ASR_KEY": "another-secret"}, {"DOCTHINKER_VOICE_ENABLED": "yes"},
    {"DOCTHINKER_INVITE_UPLOADS_ENABLED": "true"},
])
def test_configuration_fails_closed_and_errors_do_not_echo_provider_values(updates):
    with pytest.raises(workers.WorkerConfigurationError) as result:
        workers.provider_environment(model_environment(**updates))
    assert "another-secret" not in str(result.value)
    assert "provider.example" not in str(result.value)
    assert "Injected" not in str(result.value)


@pytest.mark.parametrize("records", [
    [], [record(i) for i in range(1, 7)],
    [record(), record(2, id=record()["id"])],
    [record(), record(2, upstream_url=record()["upstream_url"])],
    [record(), record(2, upstream_token=record()["upstream_token"])],
    [record(id="../../desktop")], [record(status="revoked")],
    [record(upstream_url="http://localhost:18001")],
    [record(upstream_url="http://0.0.0.0:18001")],
    [record(upstream_url="http://127.0.0.1:18001/path")],
    [record(upstream_url="http://127.0.0.1:80")],
    [record(upstream_token="too-short")], [record(allow_uploads="false")],
])
def test_worker_mapping_is_fixed_bounded_unique_and_loopback_only(records):
    with pytest.raises(workers.WorkerConfigurationError):
        workers.parse_bindings(records)


def test_check_makes_no_directories_imports_clients_or_launches_processes(store, monkeypatch):
    runner, started, stopped = fake_runner(store, monkeypatch)
    bindings = runner.check()
    assert len(bindings) == 2 and not started and not stopped
    assert not (store.root / "workers").exists()
    assert bindings[0].token not in repr(bindings)


def test_workers_get_independent_private_paths_tokens_and_allowlisted_environments(store, monkeypatch):
    environment = model_environment(
        OPENAI_API_KEY="never-inherit", HTTP_PROXY="http://unexpected.invalid",
        RAG_WORKDIR="/desktop/global", LONG_HORIZON_DB_PATH="/desktop/shared.db",
        DATABASE_URL="postgres://secret", PYTHONPATH="/untrusted", HOME="/real-user-home",
        DOCTHINKER_MOBILE_ACCESS_TOKEN="do-not-reuse", UNRELATED_SECRET="private",
    )
    runner, started, stopped = fake_runner(store, monkeypatch, environment)
    runner.start()
    assert len(started) == 2
    for (process, args, options), row in zip(started, store.rows):
        env = options["env"]
        root = store.root / "workers" / row["id"]
        assert options["cwd"] == root / "runtime"
        assert options["start_new_session"] is True
        assert options["stdout"] == options["stderr"] == subprocess.DEVNULL
        assert row["upstream_token"] not in str(args)
        assert env["DOCTHINKER_MOBILE_ACCESS_TOKEN"] == row["upstream_token"]
        assert env["DOCTHINKER_MOBILE_DATA_DIR"] == str(root)
        assert env["PYTHON_DOTENV_DISABLED"] == "1"
        assert env["EMBEDDING_BINDING_API_KEY"] == "EMPTY"
        assert env["DOCTHINKER_VOICE_ENABLED"] == "false"
        assert env["DOCTHINKER_RESEARCH_CONTEXT_ASSIST"] == "false"
        assert env["LLM_MODELS"] == env["LLM_MODEL"]
        for key in ("OPENAI_API_KEY", "HTTP_PROXY", "DATABASE_URL", "LONG_HORIZON_DB_PATH", "RAG_WORKDIR", "UNRELATED_SECRET"):
            assert key not in env
        for path in (root, root / "runtime", root / "home", root / "tmp", root / "cache", root / "data"):
            assert path.stat().st_mode & 0o777 == 0o700
    assert started[0][2]["env"]["HOME"] != started[1][2]["env"]["HOME"]
    runner.close()
    assert stopped == [item[0] for item in started]


def test_invitation_model_pool_cannot_override_explicit_single_model():
    environment = workers.provider_environment(model_environment(LLM_MODELS="expensive-one,expensive-two"))
    assert environment["LLM_MODEL"] == "offline-only-model"
    assert environment["LLM_MODELS"] == "offline-only-model"


@pytest.mark.parametrize("value,expected", [("true", "true"), ("1", "true"), ("false", "false"), ("0", "false"),
                                           (" TRUE ", "true"), (" False ", "false")])
def test_research_assistance_optin_is_validated_and_forwarded_to_each_worker(store, monkeypatch, value, expected):
    environment = model_environment(DOCTHINKER_RESEARCH_ENABLED="true", DOCTHINKER_RESEARCH_CONTEXT_ASSIST=value)
    runner, started, _ = fake_runner(store, monkeypatch, environment)
    runner.start()
    assert len(started) == 2
    assert all(options["env"]["DOCTHINKER_RESEARCH_CONTEXT_ASSIST"] == expected for _, _, options in started)
    runner.close()


@pytest.mark.parametrize("value,expected", [("auto", "auto"), ("off", "off"), ("local", "local"),
                                           (" OFF ", "off")])
def test_research_semantic_mode_reaches_every_isolated_worker(store, monkeypatch, value, expected):
    environment = model_environment(DOCTHINKER_RESEARCH_ENABLED="true", DOCTHINKER_RESEARCH_SEMANTIC=value)
    runner, started, _ = fake_runner(store, monkeypatch, environment)
    runner.start()
    assert len(started) == 2
    assert all(options["env"]["DOCTHINKER_RESEARCH_SEMANTIC"] == expected for _, _, options in started)
    runner.close()


@pytest.mark.parametrize("value", [None, False, "", "disabled", "https://private-secret.invalid"])
def test_invalid_semantic_mode_fails_without_echoing_operator_input(value):
    with pytest.raises(workers.WorkerConfigurationError, match="DOCTHINKER_RESEARCH_SEMANTIC") as error:
        workers.provider_environment(model_environment(DOCTHINKER_RESEARCH_SEMANTIC=value))
    assert "private-secret" not in str(error.value)


@pytest.mark.parametrize("value", [None, True, "", "off", "yes", "2", "private-secret-invalid-flag"])
def test_invalid_worker_assistance_flag_is_rejected_without_echoing_values(value):
    with pytest.raises(workers.WorkerConfigurationError, match="DOCTHINKER_RESEARCH_CONTEXT_ASSIST") as error:
        workers.provider_environment(model_environment(DOCTHINKER_RESEARCH_CONTEXT_ASSIST=value))
    assert "private-secret" not in str(error.value)


def test_trusted_storage_quotas_are_validated_and_forwarded_to_independent_workers(store, monkeypatch):
    environment = model_environment(DOCTHINKER_RESEARCH_STORAGE_BYTES="1073741824",
                                    DOCTHINKER_REQUEST_JOURNAL_BYTES="536870912")
    runner, started, _ = fake_runner(store, monkeypatch, environment)
    runner.start()
    for _, _, options in started:
        assert options["env"]["DOCTHINKER_RESEARCH_STORAGE_BYTES"] == "1073741824"
        assert options["env"]["DOCTHINKER_REQUEST_JOURNAL_BYTES"] == "536870912"
    runner.close()
    default = workers.provider_environment(model_environment())
    assert "DOCTHINKER_RESEARCH_STORAGE_BYTES" not in default
    assert "DOCTHINKER_REQUEST_JOURNAL_BYTES" not in default


@pytest.mark.parametrize("key", ["DOCTHINKER_RESEARCH_STORAGE_BYTES", "DOCTHINKER_REQUEST_JOURNAL_BYTES"])
@pytest.mark.parametrize("invalid", ["", "0", "65535", "-1", "1e9", "１00000", "100000\n", str((1 << 50) + 1), "private-secret-not-a-number"])
def test_invalid_worker_storage_quota_rejected_without_echo(key, invalid):
    with pytest.raises(workers.WorkerConfigurationError) as result:
        workers.provider_environment(model_environment(**{key: invalid}))
    assert "private-secret" not in str(result.value)


def test_chat_only_profile_rejects_uploads_and_reused_provider_key(store, monkeypatch):
    store.rows[0]["allow_uploads"] = True
    runner, started, _ = fake_runner(store, monkeypatch)
    with pytest.raises(workers.WorkerConfigurationError, match="chat-only"):
        runner.start()
    assert not started
    store.rows[0]["allow_uploads"] = False
    environment = model_environment(LLM_BINDING_API_KEY=store.rows[0]["upstream_token"])
    with pytest.raises(workers.WorkerConfigurationError, match="must not reuse"):
        workers.child_environment(store.root, workers.parse_bindings(store.rows)[0], environment)


def test_symlink_shared_directory_and_dotenv_are_rejected_before_start(store, monkeypatch, tmp_path):
    binding = workers.parse_bindings(store.rows)[0]
    (store.root / "workers").symlink_to(tmp_path, target_is_directory=True)
    runner, started, _ = fake_runner(store, monkeypatch)
    with pytest.raises(workers.WorkerConfigurationError, match="not symlinks"):
        runner.check()
    (store.root / "workers").unlink()
    path = workers.prepare_directories(store.root, binding)
    (path / "runtime" / ".env").write_text("LLM_BINDING_API_KEY=do-not-load", encoding="utf-8")
    with pytest.raises(workers.WorkerConfigurationError, match="dotenv"):
        runner.check()
    assert not started


def test_failed_partial_start_terminates_previously_created_children(store, monkeypatch):
    runner, started, stopped = fake_runner(store, monkeypatch)
    original = runner._popen

    def fail_second(*args, **kwargs):
        if started:
            raise RuntimeError("fake spawn failure")
        return original(*args, **kwargs)

    runner._popen = fail_second
    with pytest.raises(RuntimeError, match="fake spawn"):
        runner.start()
    assert stopped == [started[0][0]] and not runner.processes


def test_revocations_stop_existing_workers_but_new_users_do_not_start_workers(store, monkeypatch):
    runner, started, stopped = fake_runner(store, monkeypatch)
    runner.start()
    store.rows.pop(0)
    store.rows.append(record(3))
    runner.reconcile()
    assert stopped == [started[0][0]] and len(started) == 2
    assert set(runner.processes) == {record(2)["id"]}
    store.rows.clear()
    runner.reconcile()
    assert len(stopped) == 2 and not runner.processes


@pytest.mark.parametrize("failure", ["binding", "process"])
def test_running_worker_failure_requires_manual_restart_and_closes_all(store, monkeypatch, failure):
    runner, started, stopped = fake_runner(store, monkeypatch)
    event = Mock()

    def change(_seconds):
        if failure == "binding":
            store.rows[0]["upstream_token"] = "changed-" + "x" * 48
        else:
            started[0][0].poll.return_value = 1
        return False

    event.wait.side_effect = change
    with pytest.raises(workers.WorkerConfigurationError):
        runner.run(event)
    assert len(started) == len(stopped) == 2


def test_shutdown_attempts_every_owned_child_even_if_one_stop_fails(store, monkeypatch):
    runner, started, _ = fake_runner(store, monkeypatch)
    runner.start()
    stopped = []

    def stop(process):
        stopped.append(process)
        if process is started[0][0]:
            raise OSError("fake termination failure")

    runner._stopper = stop
    with pytest.raises(workers.WorkerConfigurationError, match="every owned child"):
        runner.close()
    assert len(stopped) == 2


def test_shutdown_signal_and_kill_target_only_the_created_process_group(monkeypatch):
    terminate = Mock()
    monkeypatch.setattr(os, "killpg", terminate)
    process = SimpleNamespace(pid=12345, wait=Mock(side_effect=[subprocess.TimeoutExpired("fake", 5), 0]))
    workers.stop_child(process)
    assert terminate.call_args_list == [((12345, signal.SIGTERM),), ((12345, signal.SIGKILL),)]
    assert process.wait.call_count == 2


def test_second_supervisor_for_same_store_cannot_acquire_lock(store):
    with workers.supervisor_lock(store.root), pytest.raises(workers.WorkerConfigurationError, match="already running"), workers.supervisor_lock(store.root):
        pytest.fail("Duplicate supervisor")


def test_sqlite_binding_contract_and_cli_check_redact_all_credentials(tmp_path, monkeypatch, capsys):
    from docthinker.server.invite_store import InviteStore

    store = InviteStore.initialize((tmp_path / "actual-store").resolve())
    invitation = store.add_user("alice", 18001)
    bindings = workers.parse_bindings(store.list_worker_bindings())
    assert len(bindings) == 1 and bindings[0].allow_uploads is False
    monkeypatch.setattr(os, "environ", model_environment(DOCTHINKER_INVITE_DATA_DIR=str(store.root)))
    assert workers.main(["--check"]) == 0
    output = capsys.readouterr()
    assert "chat/history only" in output.out
    for secret in (bindings[0].token, invitation["invite_code"], "offline-only-model-credential"):
        assert secret not in output.out + output.err
    assert not (store.root / "workers").exists()


def test_child_wrapper_blocks_all_dotenv_before_import_and_is_loopback_only(tmp_path, monkeypatch):
    from docthinker.server.invite_store import InviteStore

    store = InviteStore.initialize((tmp_path / "worker-registry").resolve())
    store.add_user("worker-test", 18001)
    binding = workers.parse_bindings(store.list_worker_bindings())[0]
    directory = workers.prepare_directories(store.root, binding)
    env = workers.child_environment(store.root, binding, model_environment())
    env["DATABASE_URL"] = "must-not-inherit"
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.chdir(directory / "runtime")
    umask = Mock()
    monkeypatch.setattr(os, "umask", umask)
    import dotenv
    import dotenv.main

    original_load, original_main_load = dotenv.load_dotenv, dotenv.main.load_dotenv
    importer = builtins.__import__
    application = object()
    run = Mock()

    def guarded(name, *args, **kwargs):
        if name == "mobile_app":
            assert dotenv.load_dotenv("never-open.env") is False
            assert dotenv.main.load_dotenv("never-open.env") is False
            assert "DATABASE_URL" not in os.environ
            return SimpleNamespace(create_mobile_app=lambda: application)
        if name == "uvicorn":
            return SimpleNamespace(run=run)
        return importer(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    try:
        workers.worker_main()
    finally:
        dotenv.load_dotenv = original_load
        dotenv.main.load_dotenv = original_main_load
    umask.assert_called_once_with(0o077)
    run.assert_called_once_with(application, host="127.0.0.1", port=binding.port,
                                access_log=False, log_level="warning", proxy_headers=False)
