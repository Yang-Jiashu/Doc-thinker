"""Offline deployment-supervision tests; no real listener or model is started."""

import builtins
import os
import signal
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from docthinker.server import invite_runtime as runtime
from docthinker.server import invite_workers as workers
from docthinker.server.invite_store import InviteError


def model_environment(**updates):
    return {
        workers.SHARING_ACK_ENV: workers.SHARING_ACK_VALUE,
        "LLM_BINDING_API_KEY": "offline-provider-credential",
        "LLM_VLM_HOST": "https://provider.example.invalid/v1",
        "LLM_MODEL": "offline-model",
        **updates,
    }


@pytest.fixture
def web_dir(tmp_path):
    directory = tmp_path / "built-web"
    directory.mkdir()
    for name in ("index.html", "manifest.webmanifest", "service-worker.js"):
        (directory / name).write_text("offline build fixture", encoding="utf-8")
    return directory


@pytest.fixture
def configuration(web_dir):
    return runtime.RuntimeConfiguration.from_environment({
        "DOCTHINKER_INVITE_ORIGIN": "https://invite.example.invalid",
        "DOCTHINKER_INVITE_WEB_DIR": str(web_dir),
    })


@pytest.fixture
def store(tmp_path):
    directory = tmp_path / "private-store"
    directory.mkdir(mode=0o700)
    return SimpleNamespace(root=directory)


def fake_runtime(store, configuration, monkeypatch, **environment):
    monkeypatch.setattr(runtime, "ensure_gateway_port_available", Mock())
    binding = SimpleNamespace(user_id="a" * 32, port=18001)
    fake_workers = SimpleNamespace(
        check=Mock(return_value=(binding,)), processes={}, bindings={binding.user_id: binding},
        reconcile=Mock(),
    )
    events = []

    def start():
        events.append("workers-start")
        fake_workers.processes[binding.user_id] = SimpleNamespace(poll=Mock(return_value=None))

    def close():
        events.append("workers-stop")
        fake_workers.processes.clear()

    fake_workers.start = Mock(side_effect=start)
    fake_workers.close = Mock(side_effect=close)
    process = SimpleNamespace(pid=12345, poll=Mock(return_value=None))

    def popen(*args, **kwargs):
        events.append("gateway-start")
        return process

    popen_mock = Mock(side_effect=popen)
    stopper = Mock(side_effect=lambda _process: events.append("gateway-stop"))
    result = runtime.InviteRuntime(
        store, configuration, model_environment(**environment), workers=fake_workers,
        popen=popen_mock, stopper=stopper,
    )
    return result, fake_workers, process, popen_mock, stopper, events


@pytest.mark.parametrize("updates", [
    {"DOCTHINKER_INVITE_BIND_HOST": "localhost"},
    {"DOCTHINKER_INVITE_BIND_HOST": "192.168.1.1"},
    {"DOCTHINKER_INVITE_PORT": "80"},
    {"DOCTHINKER_INVITE_PORT": "65536"},
    {"DOCTHINKER_INVITE_PORT": "8080\n"},
    {"DOCTHINKER_INVITE_PORT": "٨٠٨٠"},
    {"DOCTHINKER_INVITE_ORIGIN": "http://public.example.invalid"},
    {"DOCTHINKER_INVITE_ORIGIN": "https://invite.example.invalid/path"},
    {"DOCTHINKER_INVITE_WEB_DIR": ""},
    {"DOCTHINKER_INVITE_WEB_DIR": "."},
])
def test_explicit_origin_binding_and_web_build_are_required(web_dir, updates):
    environment = {"DOCTHINKER_INVITE_ORIGIN": "https://invite.example.invalid",
                   "DOCTHINKER_INVITE_WEB_DIR": str(web_dir), **updates}
    with pytest.raises((ValueError, InviteError)):
        runtime.RuntimeConfiguration.from_environment(environment)


def test_container_bind_is_opt_in_but_workers_stay_loopback(web_dir):
    configuration = runtime.RuntimeConfiguration.from_environment({
        "DOCTHINKER_INVITE_ORIGIN": "https://invite.example.invalid",
        "DOCTHINKER_INVITE_WEB_DIR": str(web_dir),
        "DOCTHINKER_INVITE_BIND_HOST": "0.0.0.0",
        "DOCTHINKER_INVITE_PORT": "8088",
    })
    assert configuration.host == "0.0.0.0" and configuration.port == 8088
    with pytest.raises(workers.WorkerConfigurationError):
        workers.parse_bindings([{"id": "a" * 32, "status": "active", "upstream_token": "a" * 48,
                                 "upstream_url": "http://0.0.0.0:18001"}])


def test_missing_ack_precedes_store_or_runtime_imports(monkeypatch, capsys):
    monkeypatch.setattr(os, "environ", {})
    importer = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name not in {"invite_store", "invite_app", "invite_site", "uvicorn", "dotenv", "mobile_app"}
        return importer(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert runtime.main(["--check"]) == 2
    assert "startup is blocked" in capsys.readouterr().err


def test_docker_secret_key_is_loaded_explicitly_and_not_forwarded_as_a_path(tmp_path):
    secret = (tmp_path / "llm_api_key").resolve()
    secret.write_text("offline-file-credential\n", encoding="utf-8")
    secret.chmod(0o444)  # Docker bind-mounted secrets need not be owner-mode 0600.
    environment = model_environment(LLM_BINDING_API_KEY_FILE=str(secret))
    del environment["LLM_BINDING_API_KEY"]
    resolved = runtime.explicit_runtime_environment(environment)
    assert resolved["LLM_BINDING_API_KEY"] == "offline-file-credential"
    assert "LLM_BINDING_API_KEY_FILE" not in resolved
    assert "LLM_BINDING_API_KEY" not in environment


@pytest.mark.parametrize("kind", ["relative", "symlink", "directory", "too-large", "empty", "invalid-key", "conflict"])
def test_secret_files_fail_closed_without_echoing_path_or_contents(tmp_path, kind):
    secret = (tmp_path / "private-secret").resolve()
    secret.write_text("never-print-this-secret", encoding="utf-8")
    environment = model_environment(LLM_BINDING_API_KEY_FILE=str(secret))
    if kind != "conflict":
        del environment["LLM_BINDING_API_KEY"]
    if kind == "relative":
        environment["LLM_BINDING_API_KEY_FILE"] = "private-secret"
    elif kind == "symlink":
        link = tmp_path / "secret-link"
        link.symlink_to(secret)
        environment["LLM_BINDING_API_KEY_FILE"] = str(link)
    elif kind == "directory":
        environment["LLM_BINDING_API_KEY_FILE"] = str(tmp_path)
    elif kind == "too-large":
        secret.write_text("x" * 4097, encoding="utf-8")
    elif kind == "empty":
        secret.write_text(" \n", encoding="utf-8")
    elif kind == "invalid-key":
        secret.write_text("secret\ninjected-header", encoding="utf-8")
    with pytest.raises(workers.WorkerConfigurationError) as error:
        runtime.explicit_runtime_environment(environment)
    assert "never-print-this-secret" not in str(error.value)
    assert str(tmp_path) not in str(error.value)


def test_check_is_read_only_and_gateway_env_never_inherits_model_credentials(store, configuration, monkeypatch):
    runner, fake_workers, _, popen, stopper, _ = fake_runtime(store, configuration, monkeypatch)
    assert len(runner.check()) == 1
    assert not (store.root / "gateway").exists()
    fake_workers.start.assert_not_called()
    popen.assert_not_called()
    stopper.assert_not_called()
    environment = runtime.gateway_environment(store.root, configuration, model_environment(
        OPENAI_API_KEY="do-not-pass", HTTP_PROXY="http://untrusted.invalid", PYTHONPATH="/untrusted",
        DATABASE_URL="postgres://secret", HOME="/owner/home", DOCTHINKER_INVITE_UPLOADS_ENABLED="true",
    ))
    for key in ("LLM_BINDING_API_KEY", "LLM_VLM_HOST", "LLM_MODEL", "OPENAI_API_KEY", "HTTP_PROXY", "DATABASE_URL", workers.SHARING_ACK_ENV):
        assert key not in environment
    assert environment["PYTHON_DOTENV_DISABLED"] == "1"
    assert environment["DOCTHINKER_INVITE_UPLOADS_ENABLED"] == "false"
    assert environment["HOME"] == str(store.root / "gateway" / "home")


def test_registry_budget_survives_gateway_environment_cleaning(store, configuration):
    default = runtime.gateway_environment(store.root, configuration, {})
    assert default["DOCTHINKER_INVITE_REGISTRY_BYTES"] == "67108864"
    expanded = runtime.gateway_environment(store.root, configuration, {
        "DOCTHINKER_INVITE_REGISTRY_BYTES": "134217728", "UNRELATED_VALUE": "private-not-inherited"})
    assert expanded["DOCTHINKER_INVITE_REGISTRY_BYTES"] == "134217728"
    assert "UNRELATED_VALUE" not in expanded


@pytest.mark.parametrize("value", ["", "65535", "1e9", "100000\n", str((1 << 50) + 1), 65536, None])
def test_bad_registry_budget_cannot_reach_gateway_child(store, configuration, value):
    with pytest.raises(InviteError, match="storage quota"):
        runtime.gateway_environment(store.root, configuration, {"DOCTHINKER_INVITE_REGISTRY_BYTES": value})


def test_gateway_and_worker_ports_cannot_overlap(store, configuration, monkeypatch):
    runner, fake_workers, _, popen, _, _ = fake_runtime(store, configuration, monkeypatch)
    fake_workers.check.return_value[0].port = configuration.port
    with pytest.raises(workers.WorkerConfigurationError, match="ports must be distinct"):
        runner.start()
    fake_workers.start.assert_not_called()
    popen.assert_not_called()


def test_gateway_paths_reject_symlinks_and_dotenv_without_starting(store, configuration, monkeypatch, tmp_path):
    runner, fake_workers, _, popen, _, _ = fake_runtime(store, configuration, monkeypatch)
    (store.root / "gateway").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(workers.WorkerConfigurationError, match="not symlinks"):
        runner.check()
    (store.root / "gateway").unlink()
    directory = runtime.prepare_gateway_directory(store.root)
    (directory / ".env").write_text("LLM_BINDING_API_KEY=never-load", encoding="utf-8")
    with pytest.raises(workers.WorkerConfigurationError, match="dotenv"):
        runner.start()
    fake_workers.start.assert_not_called()
    popen.assert_not_called()


def test_start_owns_only_sanitized_children_and_stops_gateway_first(store, configuration, monkeypatch):
    runner, fake_workers, process, popen, stopper, events = fake_runtime(store, configuration, monkeypatch,
                                                                      OTHER_SECRET="not-for-gateway")
    runner.start()
    options = popen.call_args.kwargs
    assert options["cwd"] == store.root / "gateway" / "runtime"
    assert options["start_new_session"] is True
    assert options["stdout"] == options["stderr"] == subprocess.DEVNULL
    assert "OTHER_SECRET" not in options["env"]
    assert "offline-provider-credential" not in repr(popen.call_args)
    assert popen.call_args.args[0][-1] == "--gateway"
    for child in (store.root / "gateway").iterdir():
        assert child.stat().st_mode & 0o777 == 0o700
    with pytest.raises(workers.WorkerConfigurationError, match="already started"):
        runner.start()
    runner.close()
    stopper.assert_called_once_with(process)
    assert events == ["workers-start", "gateway-start", "gateway-stop", "workers-stop"]
    assert runner.gateway is None and not fake_workers.processes


def test_failed_gateway_spawn_closes_already_started_workers(store, configuration, monkeypatch):
    runner, fake_workers, _, popen, stopper, _ = fake_runtime(store, configuration, monkeypatch)
    popen.side_effect = RuntimeError("secret failure")
    with pytest.raises(RuntimeError):
        runner.start()
    fake_workers.close.assert_called_once()
    stopper.assert_not_called()
    assert not fake_workers.processes


@pytest.mark.parametrize("failed_child", ["gateway", "worker"])
def test_child_failure_fails_service_and_cleans_up_every_child(store, configuration, monkeypatch, failed_child):
    runner, fake_workers, process, popen, stopper, events = fake_runtime(store, configuration, monkeypatch)
    monkeypatch.setattr(runner, "wait_ready", Mock(return_value=True))
    event = Mock()

    def fail_child(_seconds):
        if failed_child == "gateway":
            process.poll.return_value = 1
        else:
            fake_workers.reconcile.side_effect = workers.WorkerConfigurationError("A worker exited")
        return False

    event.wait.side_effect = fail_child
    with pytest.raises(workers.WorkerConfigurationError):
        runner.run(event)
    assert events[-2:] == ["gateway-stop", "workers-stop"]
    assert popen.call_count == 1 and stopper.call_count == 1


def test_revoking_all_users_stops_gateway_without_adding_users(store, configuration, monkeypatch):
    runner, fake_workers, _, popen, stopper, events = fake_runtime(store, configuration, monkeypatch)
    monkeypatch.setattr(runner, "wait_ready", Mock(return_value=True))
    event = Mock()
    event.wait.return_value = False
    fake_workers.reconcile.side_effect = fake_workers.processes.clear
    runner.run(event)
    assert events[-2:] == ["gateway-stop", "workers-stop"]
    assert popen.call_count == stopper.call_count == 1


def test_shutdown_attempts_worker_cleanup_after_gateway_stop_failure(store, configuration, monkeypatch):
    runner, fake_workers, _, _, stopper, _ = fake_runtime(store, configuration, monkeypatch)
    runner.start()
    stopper.side_effect = RuntimeError("secret stop failure")
    with pytest.raises(workers.WorkerConfigurationError, match="every owned invitation process"):
        runner.close()
    fake_workers.close.assert_called_once()
    assert not fake_workers.processes


def test_readiness_is_only_loopback_tcp_and_honors_stop_and_timeout(store, configuration, monkeypatch):
    runner, _, _, _, _, _ = fake_runtime(store, configuration, monkeypatch)
    runner.start()
    probe = Mock(return_value=True)
    monkeypatch.setattr(runtime, "port_ready", probe)
    assert runner.wait_ready(threading.Event()) is True
    assert {call.args[0] for call in probe.call_args_list} == {configuration.port, 18001}
    stopped = threading.Event()
    stopped.set()
    assert runner.wait_ready(stopped) is False
    probe.return_value = False
    monkeypatch.setattr(runtime.time, "monotonic", Mock(side_effect=[0, 121]))
    with pytest.raises(workers.WorkerConfigurationError, match="startup timed out"):
        runner.wait_ready(threading.Event())
    runner.close()


def test_cli_check_and_errors_never_echo_secrets_or_create_accounts(tmp_path, web_dir, monkeypatch, capsys):
    from docthinker.server.invite_store import InviteStore

    store = InviteStore.initialize((tmp_path / "actual-store").resolve())
    invitation = store.add_user("alice", 18001)
    environment = model_environment(DOCTHINKER_INVITE_DATA_DIR=str(store.root),
                                    DOCTHINKER_INVITE_WEB_DIR=str(web_dir),
                                    DOCTHINKER_INVITE_ORIGIN="https://invite.example.invalid")
    monkeypatch.setattr(os, "environ", environment)
    assert runtime.main(["--check"]) == 0
    output = capsys.readouterr()
    assert "No servers or provider requests started" in output.out
    assert not (store.root / "gateway").exists() and not (store.root / "workers").exists()
    assert len(store.list_worker_bindings()) == 1
    for secret in (invitation["invite_code"], "offline-provider-credential", store.list_worker_bindings()[0]["upstream_token"]):
        assert secret not in output.out + output.err
    monkeypatch.setattr(runtime.RuntimeConfiguration, "from_environment", Mock(side_effect=RuntimeError("secret-runtime-error")))
    assert runtime.main(["--check"]) == 1
    assert "secret-runtime-error" not in capsys.readouterr().err


def test_main_shutdown_signal_is_forwarded_and_handlers_restored(tmp_path, web_dir, monkeypatch):
    from docthinker.server.invite_store import InviteStore

    store = InviteStore.initialize((tmp_path / "actual-store").resolve())
    store.add_user("alice", 18001)
    monkeypatch.setattr(os, "environ", model_environment(
        DOCTHINKER_INVITE_DATA_DIR=str(store.root), DOCTHINKER_INVITE_WEB_DIR=str(web_dir),
        DOCTHINKER_INVITE_ORIGIN="https://invite.example.invalid",
    ))
    handlers, restored = {}, []

    def install(sig, handler):
        if callable(handler):
            handlers[sig] = handler
            return f"previous-{sig}"
        restored.append((sig, handler))

    def run(_self, stop_event):
        assert not stop_event.is_set()
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        assert stop_event.is_set()

    monkeypatch.setattr(signal, "signal", install)
    monkeypatch.setattr(runtime.InviteRuntime, "run", run)
    assert runtime.main([]) == 0
    assert restored == [(signal.SIGINT, f"previous-{signal.SIGINT}"),
                        (signal.SIGTERM, f"previous-{signal.SIGTERM}")]


def test_gateway_child_requires_supervision_without_loading_core(monkeypatch):
    monkeypatch.delenv("DOCTHINKER_INVITE_GATEWAY_CHILD", raising=False)
    importer = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name not in {"uvicorn", "dotenv", "mobile_app", "docthinker.core", "app"}
        return importer(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(workers.WorkerConfigurationError, match="supervised startup"):
        runtime.gateway_main()


def test_gateway_child_clears_legacy_secrets_and_runs_one_gateway_only(tmp_path, web_dir, monkeypatch):
    import uvicorn

    from docthinker.server import invite_site
    from docthinker.server.invite_store import InviteStore

    store = InviteStore.initialize((tmp_path / "actual-store").resolve())
    directory = runtime.prepare_gateway_directory(store.root)
    monkeypatch.chdir(directory)
    monkeypatch.setattr(os, "environ", {
        "DOCTHINKER_INVITE_GATEWAY_CHILD": "1", "DOCTHINKER_INVITE_DATA_DIR": str(store.root),
        "DOCTHINKER_INVITE_WEB_DIR": str(web_dir), "DOCTHINKER_INVITE_ORIGIN": "https://invite.example.invalid",
        "DOCTHINKER_INVITE_BIND_HOST": "0.0.0.0", "LLM_BINDING_API_KEY": "must-not-reach-gateway",
        "OPENAI_API_KEY": "legacy-never-inherit", "DOCTHINKER_INVITE_UPLOADS_ENABLED": "true",
    })
    fake_app = object()

    def create_site():
        assert "LLM_BINDING_API_KEY" not in os.environ and "OPENAI_API_KEY" not in os.environ
        assert os.environ["DOCTHINKER_INVITE_UPLOADS_ENABLED"] == "false"
        assert os.environ["PYTHON_DOTENV_DISABLED"] == "1"
        return fake_app

    serve = Mock()
    monkeypatch.setattr(invite_site, "create_invite_site", create_site)
    monkeypatch.setattr(uvicorn, "run", serve)
    monkeypatch.setattr(os, "umask", Mock())
    importer = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name not in {"dotenv", "mobile_app", "docthinker.core", "app"}
        return importer(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    runtime.gateway_main()
    serve.assert_called_once_with(fake_app, host="0.0.0.0", port=8080, workers=1,
                                  access_log=False, log_level="warning", proxy_headers=False)


def test_stop_during_startup_closes_both_gateway_and_workers(store, configuration, monkeypatch):
    runner, _, _, _, _, events = fake_runtime(store, configuration, monkeypatch)
    monkeypatch.setattr(runner, "wait_ready", Mock(return_value=False))
    runner.run(threading.Event())
    assert events == ["workers-start", "gateway-start", "gateway-stop", "workers-stop"]


def health_fixture(tmp_path, web_dir, count=2):
    from docthinker.server.invite_store import InviteStore

    store = InviteStore.initialize((tmp_path / "health-store").resolve())
    for index in range(count):
        store.add_user(f"user{index}", 18001 + index)
    environment = {
        "DOCTHINKER_INVITE_DATA_DIR": str(store.root), "DOCTHINKER_INVITE_WEB_DIR": str(web_dir),
        "DOCTHINKER_INVITE_ORIGIN": "https://invite.example.invalid",
        "LLM_BINDING_API_KEY_FILE": "/never-read-this-secret-file",
    }
    return store, environment


def health_transport(handler):
    import httpx

    def streamed(request):
        response = handler(request)
        return httpx.Response(response.status_code, headers=response.headers,
                              stream=httpx.ByteStream(response.content))

    return httpx.MockTransport(streamed)


def test_healthcheck_uses_only_loopback_gets_without_permission_or_secret_reads(tmp_path, web_dir, monkeypatch):
    import httpx

    store, environment = health_fixture(tmp_path, web_dir, count=5)
    records = store.list_worker_bindings()
    requests = []
    monkeypatch.setattr(runtime, "explicit_runtime_environment", Mock(side_effect=AssertionError("Do not read provider secret")))

    def respond(request):
        requests.append(request)
        assert request.method == "GET" and request.url.host == "127.0.0.1"
        if request.url.path == "/index.html":
            assert request.headers["host"] == "invite.example.invalid"
            assert "authorization" not in request.headers
            return httpx.Response(200, text="<html>built invite shell</html>", headers={"Content-Type": "text/html"})
        assert request.url.path == "/api/v1/mobile/capabilities"
        row = next(record for record in records if record["upstream_url"] == f"http://127.0.0.1:{request.url.port}")
        assert request.headers["authorization"] == "Bearer " + row["upstream_token"]
        return httpx.Response(200, json={"chat_ready": True, "uploads": False})

    assert runtime.healthcheck(environment, transport=health_transport(respond)) is True
    assert len(requests) == 6


@pytest.mark.parametrize("case", ["gateway-error", "gateway-redirect", "gateway-empty", "gateway-json", "worker-error", "not-ready", "uploads", "missing-uploads", "not-object", "invalid-json", "oversized"])
def test_healthcheck_fails_closed_for_gateway_and_worker_failures(tmp_path, web_dir, case):
    import httpx

    _, environment = health_fixture(tmp_path, web_dir, count=1)

    def respond(request):
        if request.url.path == "/index.html":
            if case == "gateway-error":
                return httpx.Response(503, text="private-error")
            if case == "gateway-redirect":
                return httpx.Response(302, headers={"location": "https://provider.example.invalid"})
            if case == "gateway-json":
                return httpx.Response(200, json={"not": "the shell"})
            return httpx.Response(200, text="" if case == "gateway-empty" else "<html>shell</html>",
                                  headers={"Content-Type": "text/html"})
        if case == "worker-error":
            return httpx.Response(503, text="private-worker-error")
        if case == "invalid-json":
            return httpx.Response(200, text="not json")
        if case == "oversized":
            return httpx.Response(200, content=b"x" * (runtime.HEALTHCHECK_BODY_LIMIT + 1))
        body = {"chat_ready": case != "not-ready", "uploads": case == "uploads"}
        if case == "missing-uploads":
            del body["uploads"]
        return httpx.Response(200, json=[] if case == "not-object" else body)

    try:
        healthy = runtime.healthcheck(environment, transport=health_transport(respond))
    except (ValueError, httpx.HTTPError):
        healthy = False
    assert healthy is False


def test_healthcheck_enforces_timeout_and_absolute_deadline(tmp_path, web_dir, monkeypatch):
    import httpx

    _, environment = health_fixture(tmp_path, web_dir, count=1)
    request_count = []

    def timeout(request):
        request_count.append(request)
        assert request.extensions["timeout"] == {"connect": 1, "read": 1, "write": 1, "pool": 1}
        raise httpx.ReadTimeout("private-timeout-detail")

    with pytest.raises(httpx.ReadTimeout):
        runtime.healthcheck(environment, transport=health_transport(timeout))
    assert len(request_count) == 1
    monkeypatch.setattr(runtime.time, "monotonic", Mock(side_effect=[0, 9]))
    with pytest.raises(ValueError):
        runtime.healthcheck(environment, transport=health_transport(timeout))
    assert len(request_count) == 1


def test_healthcheck_cli_diagnostics_never_echo_internal_error_or_secret(monkeypatch, capsys):
    monkeypatch.setattr(os, "environ", {})
    probe = Mock(return_value=True)
    monkeypatch.setattr(runtime, "healthcheck", probe)
    assert runtime.main(["--healthcheck"]) == 0
    assert capsys.readouterr().out == "Invitation runtime is healthy.\n"
    probe.side_effect = RuntimeError("private-path-or-token-do-not-print")
    assert runtime.main(["--healthcheck"]) == 1
    output = capsys.readouterr()
    assert output.err == "Invitation runtime is unavailable.\n"
    assert "private-path-or-token" not in output.out + output.err


@pytest.mark.parametrize("other_mode", ["--gateway", "--check"])
def test_healthcheck_cli_modes_are_mutually_exclusive(other_mode, capsys):
    with pytest.raises(SystemExit) as error:
        runtime.main(["--healthcheck", other_mode])
    assert error.value.code == 2
    assert "not allowed" in capsys.readouterr().err
