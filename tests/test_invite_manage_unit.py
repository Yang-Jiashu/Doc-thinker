"""Offline deployment-helper contracts. All Docker calls and prompts are mocked."""

import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("invite_manage_helper", ROOT / "deploy/invite/manage.py")
manage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = manage
SPEC.loader.exec_module(manage)

FAKE_KEY = "synthetic-test-key-NEVER-A-REAL-CREDENTIAL"
DEFAULT_SETTINGS = {
    "origin": "http://127.0.0.1:8088",
    "api_base": "https://api.example.test/v1",
    "model": "test-preview",
    "sharing_authorized": True,
}


@pytest.mark.parametrize("command", ["export-account", "purge-account"])
def test_account_maintenance_is_explicit_local_cli_without_secrets_docker_or_config(monkeypatch, command):
    calls = []
    def forbidden(*_args, **_kwargs):
        pytest.fail("Cold account maintenance must not load configuration, Docker or model credentials")
    monkeypatch.setattr(manage, "load_configuration", forbidden)
    monkeypatch.setattr(manage, "run_compose", forbidden)
    monkeypatch.setattr(manage, "default_config_dir", forbidden)
    monkeypatch.setenv("LLM_BINDING_API_KEY", FAKE_KEY)
    monkeypatch.setenv("DOCTHINKER_MOBILE_ACCESS_TOKEN", "browser-must-not-leak")
    monkeypatch.setattr(manage.subprocess, "run", lambda *a, **kw: (calls.append((a, kw)) or SimpleNamespace(returncode=0)))
    arguments = [command, "--data-dir", "/synthetic/private", "--username", "Alice", "--confirm-username", "alice"]
    if command == "export-account":
        arguments.extend(["--output", "/synthetic/output/alice.zip"])
    assert manage.main(arguments) == 0
    argv, options = calls[0][0][0], calls[0][1]
    assert argv[:4] == [sys.executable, "-m", "docthinker.server.invite_admin", command]
    assert options["shell"] is False and options["check"] is False
    assert options["env"]["DOCTHINKER_INVITE_DATA_DIR"] == "/synthetic/private"
    assert options["env"]["PYTHON_DOTENV_DISABLED"] == "1"
    assert FAKE_KEY not in str(calls) and "browser-must-not-leak" not in str(calls)


def test_account_maintenance_confirmation_mismatch_never_launches_process(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Confirmation mismatch must be rejected before a child starts")
    monkeypatch.setattr(manage.subprocess, "run", forbidden)
    assert manage.main(["purge-account", "--data-dir", "/synthetic/private", "--username", "alice", "--confirm-username", "bravo"]) == 1


def test_backup_self_test_never_loads_real_configuration_or_docker(monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Offline backup self-test must not load deployment credentials or Docker")
    monkeypatch.setattr(manage, "load_configuration", forbidden)
    monkeypatch.setattr(manage, "run_compose", forbidden)
    monkeypatch.setattr(manage, "default_config_dir", forbidden)
    assert manage.main(["backup-self-test"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and report["synthetic_data_only"]
    assert not report["deployment_data_read"] and not report["docker_restore_verified"]


@pytest.fixture
def private_dir(tmp_path):
    path = tmp_path.resolve() / "private-config"
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def write_file(path, content, *, mode=0o600):
    path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    path.chmod(mode)


def configure(path, **settings):
    values = {**DEFAULT_SETTINGS, **settings}
    write_file(path / manage.SETTINGS_FILE, json.dumps(values))
    write_file(path / manage.KEY_FILE, FAKE_KEY)
    return manage.load_configuration(path)


def prompts(monkeypatch, *, permission=manage.SHARING_ACK, key=FAKE_KEY):
    responses = iter(["", DEFAULT_SETTINGS["api_base"], DEFAULT_SETTINGS["model"], permission])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(responses))
    monkeypatch.setattr(manage.getpass, "getpass", lambda _prompt: key)


@pytest.fixture
def docker_mock(monkeypatch):
    calls = []
    monkeypatch.setattr(manage.shutil, "which", lambda *_args, **_kwargs: "/test/docker")

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(manage.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("command", ["check", "up"])
def test_research_profile_is_explicit_per_invocation(private_dir, docker_mock, command):
    configuration = configure(private_dir)
    manage.run_compose(command, SimpleNamespace(research=True), configuration, private_dir)
    assert all(kwargs["env"]["DOCTHINKER_DEPLOY_RESEARCH_ENABLED"] == "true" for _, kwargs in docker_mock)
    docker_mock.clear()
    manage.run_compose(command, SimpleNamespace(), configuration, private_dir)
    assert all(kwargs["env"]["DOCTHINKER_DEPLOY_RESEARCH_ENABLED"] == "false" for _, kwargs in docker_mock)


def test_research_upload_authorization_is_per_new_account(private_dir, docker_mock):
    configuration = configure(private_dir)
    manage.run_compose("add-user", SimpleNamespace(username="researcher", port=8102, research_uploads=True), configuration, private_dir)
    assert docker_mock[-1][0][-1] == "--allow-uploads"
    assert "--allow-uploads" not in docker_mock[-1][1]["env"].values()


def test_setup_keeps_key_outside_checkout_and_json(private_dir, monkeypatch, capsys):
    prompts(monkeypatch)
    manage.setup(private_dir)
    assert stat.S_IMODE(private_dir.stat().st_mode) == 0o700
    for name in (manage.SETTINGS_FILE, manage.KEY_FILE):
        assert stat.S_IMODE((private_dir / name).stat().st_mode) == 0o600
    assert json.loads((private_dir / manage.SETTINGS_FILE).read_text()) == DEFAULT_SETTINGS
    assert (private_dir / manage.KEY_FILE).read_text() == FAKE_KEY
    assert FAKE_KEY not in (private_dir / manage.SETTINGS_FILE).read_text()
    assert FAKE_KEY not in capsys.readouterr().out
    assert not (private_dir / manage.CONTAINER_KEY_FILE).exists()


def test_setup_without_share_permission_can_save_but_not_start(private_dir, monkeypatch, capsys):
    prompts(monkeypatch, permission="")
    manage.setup(private_dir)
    assert not manage.load_configuration(private_dir).sharing_authorized
    assert "check 和 up" in capsys.readouterr().out


@pytest.mark.parametrize("name", [manage.SETTINGS_FILE, manage.KEY_FILE, manage.CONTAINER_KEY_FILE])
def test_setup_does_not_prompt_or_overwrite_existing_files(private_dir, monkeypatch, name):
    target = private_dir / name
    write_file(target, "existing private value")
    monkeypatch.setattr("builtins.input", lambda _prompt: pytest.fail("must not prompt"))
    with pytest.raises(manage.ConfigurationError, match="不会覆盖"):
        manage.setup(private_dir)
    assert target.read_text() == "existing private value"


def test_setup_does_not_follow_existing_secret_symlink(private_dir, tmp_path, monkeypatch):
    external = tmp_path / "do-not-touch"
    external.write_text("untouched")
    (private_dir / manage.KEY_FILE).symlink_to(external)
    monkeypatch.setattr("builtins.input", lambda _prompt: pytest.fail("must not prompt"))
    with pytest.raises(manage.ConfigurationError, match="不会覆盖"):
        manage.setup(private_dir)
    assert external.read_text() == "untouched"


def test_setup_invalid_ack_does_not_leave_key(private_dir, monkeypatch):
    prompts(monkeypatch, permission="yes")
    with pytest.raises(manage.ConfigurationError, match="未保存"):
        manage.setup(private_dir)
    assert list(private_dir.iterdir()) == []


def test_setup_rolls_back_only_its_new_key_on_settings_write_failure(private_dir, monkeypatch):
    prompts(monkeypatch)
    original = manage._write_new

    def fail_settings(fd, name, body, **kwargs):
        if name == manage.SETTINGS_FILE:
            raise OSError("synthetic failure")
        return original(fd, name, body, **kwargs)

    monkeypatch.setattr(manage, "_write_new", fail_settings)
    with pytest.raises(OSError):
        manage.setup(private_dir)
    assert list(private_dir.iterdir()) == []


def test_write_new_is_exclusive_and_cleans_up_partial_file(private_dir, monkeypatch):
    with manage._directory_fd(private_dir) as fd:
        manage._write_new(fd, "existing", b"unchanged")
        with pytest.raises(FileExistsError):
            manage._write_new(fd, "existing", b"replacement")
        monkeypatch.setattr(manage.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError("disk failed")))
        with pytest.raises(OSError):
            manage._write_new(fd, "partial", b"partial body")
    assert (private_dir / "existing").read_bytes() == b"unchanged"
    assert not (private_dir / "partial").exists()


def test_cleanup_does_not_unlink_file_replaced_by_someone_else(private_dir):
    original = private_dir / "original"
    replacement = private_dir / "replacement"
    write_file(original, "original")
    write_file(replacement, "replacement")
    identity = (original.stat().st_dev, original.stat().st_ino)
    original.unlink()
    replacement.rename(original)
    with manage._directory_fd(private_dir) as fd:
        manage._remove_owned_new(fd, "original", identity)
    assert original.read_text() == "replacement"


@pytest.mark.parametrize("location", ["relative", "inside", "repo", "parent"])
def test_config_directory_must_be_outside_checkout(tmp_path, monkeypatch, location):
    repository = tmp_path.resolve() / "repo"
    repository.mkdir()
    monkeypatch.setattr(manage, "REPOSITORY", repository)
    candidates = {"relative": Path("relative"), "inside": repository / "private",
                  "repo": repository, "parent": repository.parent}
    with pytest.raises(manage.ConfigurationError):
        manage.config_directory(candidates[location], create=True)
    assert not (repository / "private").exists()


def test_config_directory_rejects_symlink_and_symlink_ancestor(private_dir, tmp_path):
    link = tmp_path / "linked"
    link.symlink_to(private_dir, target_is_directory=True)
    for path in (link, link / "child"):
        with pytest.raises(manage.ConfigurationError):
            manage.config_directory(path, create=True)
    assert not (private_dir / "child").exists()


@pytest.mark.parametrize("mode", [0o755, 0o777, 0o750, 0o600])
def test_config_directory_rejects_unsafe_permissions(private_dir, mode):
    private_dir.chmod(mode)
    with pytest.raises(manage.ConfigurationError):
        manage.config_directory(private_dir)


def test_config_directory_rejects_wrong_owner(private_dir, monkeypatch):
    monkeypatch.setattr(manage.os, "getuid", lambda: private_dir.stat().st_uid + 1)
    with pytest.raises(manage.ConfigurationError):
        manage.config_directory(private_dir)


def test_missing_config_has_actionable_static_error(tmp_path):
    with pytest.raises(manage.ConfigurationError, match="先运行 setup"):
        manage.config_directory(tmp_path.resolve() / "missing")


@pytest.mark.parametrize("command, suffix", [("status", "ps"), ("down", "down")])
@pytest.mark.parametrize("condition", ["missing", "unsafe", "symlink", "invalid_json", "missing_key"])
def test_status_and_shutdown_do_not_depend_on_damaged_provider_configuration(tmp_path, monkeypatch, docker_mock, command, suffix, condition):
    path = tmp_path.resolve() / "configuration"
    if condition != "missing":
        path.mkdir(mode=0o700)
    if condition == "unsafe":
        path.chmod(0o777)
    elif condition == "symlink":
        target = path
        path = tmp_path.resolve() / "config-link"
        path.symlink_to(target, target_is_directory=True)
    elif condition == "invalid_json":
        write_file(path / manage.SETTINGS_FILE, "not valid JSON")
    elif condition == "missing_key":
        write_file(path / manage.SETTINGS_FILE, json.dumps(DEFAULT_SETTINGS))
    def no_credentials(*_args, **_kwargs):
        pytest.fail("Stopping/inspecting cannot depend on configuration, model credentials, or key copies")
    for name in ("load_configuration", "prepare_container_key", "config_directory", "default_config_dir", "compose_environment"):
        monkeypatch.setattr(manage, name, no_credentials)
    assert manage.main(["--config-dir", str(path), command]) == 0
    assert len(docker_mock) == 1
    argv, options = docker_mock[0]
    assert argv == ["/test/docker", "compose", "-p", manage.PROJECT, "-f", str(manage.COMPOSE_FILE), "--env-file", os.devnull, suffix]
    assert options["env"]["DOCTHINKER_DEPLOY_KEY_FILE"] == os.devnull
    assert "--volumes" not in argv and "-v" not in argv
    assert not (path / manage.CONTAINER_KEY_FILE).exists()


@pytest.mark.parametrize("command", ["status", "down"])
def test_control_path_ignores_model_secrets_and_invalid_startup_options(monkeypatch, docker_mock, command):
    for key, value in {"LLM_BINDING_API_KEY": FAKE_KEY, "OPENAI_API_KEY": FAKE_KEY,
                       "DOCTHINKER_DEPLOY_KEY_FILE": "/do-not-read/key", "DOCTHINKER_DEPLOY_MODEL": "bad model",
                       "DOCTHINKER_RESEARCH_SEMANTIC": "bad-mode", "DOCTHINKER_RESEARCH_STORAGE_BYTES": "bad-quota",
                       "COMPOSE_FILE": "/other-project.yaml", "COMPOSE_PROJECT_NAME": "other-project",
                       "COMPOSE_PROFILES": "public"}.items():
        monkeypatch.setenv(key, value)
    assert manage.main([command]) == 0
    argv, options = docker_mock[0]
    assert FAKE_KEY not in repr(options) and "other-project" not in repr((argv, options))
    assert set(options["env"]) <= manage._DOCKER_ENV | {"COMPOSE_DISABLE_ENV_FILE", "DOCTHINKER_DEPLOY_KEY_FILE"}
    assert options["env"]["COMPOSE_DISABLE_ENV_FILE"] == "true"


@pytest.mark.parametrize("command", ["up", "build", "init", "check", "add-user", "revoke-user", "reissue-invite", "down --volumes", "ps"])
def test_nonstarting_control_path_cannot_start_mutate_accounts_or_delete_volumes(monkeypatch, command):
    monkeypatch.setattr(manage.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must not execute Docker"))
    with pytest.raises(manage.ConfigurationError, match="只允许"):
        manage.run_control_compose(command)


def test_control_failures_are_returned_without_retry(monkeypatch, docker_mock, capsys):
    calls = []
    def failed(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(manage.subprocess, "run", failed)
    assert manage.main(["status"]) == 7 and len(calls) == 1
    assert "没有启动服务或删除数据卷" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["status", "down"])
def test_control_missing_docker_does_not_require_configuration(monkeypatch, capsys, command):
    monkeypatch.setattr(manage.shutil, "which", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(manage, "default_config_dir", lambda: pytest.fail("must not inspect credentials"))
    monkeypatch.setattr(manage.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must not execute"))
    assert manage.main([command]) == 1
    assert "未找到 Docker" in capsys.readouterr().err


def test_control_placeholder_covers_only_required_compose_interpolation():
    import re

    required = re.findall(r"\$\{([^}:]+):\?", manage.COMPOSE_FILE.read_text())
    assert required == ["DOCTHINKER_DEPLOY_KEY_FILE"]


def test_preflight_reports_missing_conditions_without_reading_key_or_calling_docker(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(manage, "load_configuration", lambda *_args: pytest.fail("must not read the key"))
    monkeypatch.setattr(manage, "run_compose", lambda *_args: pytest.fail("must not call Docker"))
    result = manage.main(["--config-dir", str(tmp_path / "missing"), "preflight", "--web-dir", str(tmp_path / "unbuilt")])
    assert result == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "blocked" and report["secret_contents_read"] is False


@pytest.mark.parametrize("value", ["", "ftp://host.test/v1", "http://host.test/v1",
    "https://user:password@host.test/v1", "https://host.test/v1?api_key=private",
    "https://host.test/v1#private", "https://host.test:99999/v1", "https://host.test/\npath",
    "https://host.test/\x7fpath", "https://host.test/{key}", "https://host.test/\\path", None])
def test_api_url_rejects_insecure_or_credential_bearing_input(value):
    with pytest.raises(manage.ConfigurationError):
        manage._url(value)


@pytest.mark.parametrize("value", ["http://127.0.0.1:9000/v1", "http://localhost:9000/v1",
    "https://api.example.test/step_plan/v1"])
def test_api_url_accepts_https_and_localhost(value):
    assert manage._url(value) == value


@pytest.mark.parametrize("value", ["https://host.test/", "https://host.test/path", "https://HOST.test",
    "https://host.test?x=1"])
def test_origin_is_canonical_without_path(value):
    with pytest.raises(manage.ConfigurationError):
        manage._url(value, origin=True)


@pytest.mark.parametrize("key", ["", "EMPTY", " key", "key ", "key\nvalue", "key\x7fvalue", "x" * 4097, "\ud800"])
def test_key_validation_is_static_and_does_not_echo_input(key):
    with pytest.raises(manage.ConfigurationError) as caught:
        manage._key_bytes(key)
    assert str(caught.value) and (not key or str(caught.value) != key)
    assert FAKE_KEY not in str(caught.value)


@pytest.mark.parametrize("model", ["", " model", "model ", "model\nkey", "model\x7fkey", None, "x" * 201,
                                  "model with spaces", "first,second", "-leading", "模型", "model;command"])
def test_model_validation_rejects_bad_identifiers(model):
    with pytest.raises(manage.ConfigurationError):
        manage._model(model)


@pytest.mark.parametrize("model", ["step-5-preview", "org/model:preview-v2.1", "provider_model", "x" * 200])
def test_model_validation_accepts_single_worker_compatible_identifier(model):
    assert manage._model(model) == model


def test_load_configuration_has_exact_non_secret_fields(private_dir):
    configuration = configure(private_dir)
    assert configuration == manage.Configuration(**DEFAULT_SETTINGS)
    assert FAKE_KEY not in repr(configuration)


@pytest.mark.parametrize("change", [{"key": FAKE_KEY}, {"sharing_authorized": 1}, {"sharing_authorized": "true"}])
def test_configuration_rejects_extra_secret_fields_and_non_boolean_ack(private_dir, change):
    configure(private_dir)
    write_file(private_dir / manage.SETTINGS_FILE, json.dumps({**DEFAULT_SETTINGS, **change}))
    with pytest.raises(manage.ConfigurationError, match="字段"):
        manage.load_configuration(private_dir)


@pytest.mark.parametrize("body", [b"", b"{broken " + FAKE_KEY.encode(), b"\xff", b"[]", b"{}"])
def test_load_rejects_invalid_settings_without_echoing_content(private_dir, body):
    configure(private_dir)
    write_file(private_dir / manage.SETTINGS_FILE, body)
    with pytest.raises(manage.ConfigurationError) as caught:
        manage.load_configuration(private_dir)
    assert FAKE_KEY not in str(caught.value)


@pytest.mark.parametrize("name", [manage.SETTINGS_FILE, manage.KEY_FILE])
@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory", "unsafe-mode", "empty", "oversized"])
def test_load_rejects_unsafe_file_types_permissions_and_size(private_dir, tmp_path, name, kind):
    configure(private_dir)
    file = private_dir / name
    if kind == "symlink":
        target = tmp_path / "external"
        file.rename(target)
        file.symlink_to(target)
    elif kind == "hardlink":
        os.link(file, tmp_path / "linked-copy")
    elif kind == "directory":
        file.unlink()
        file.mkdir(mode=0o600)
    elif kind == "unsafe-mode":
        file.chmod(0o644)
    elif kind == "empty":
        write_file(file, b"")
    elif kind == "oversized":
        write_file(file, b"x" * 17000)
    with pytest.raises((manage.ConfigurationError, OSError)):
        manage.load_configuration(private_dir)


def test_container_key_is_readonly_copy_beneath_private_directory(private_dir):
    configure(private_dir)
    key_file = manage.prepare_container_key(private_dir)
    assert key_file == private_dir / manage.CONTAINER_KEY_FILE
    assert key_file.read_bytes() == (private_dir / manage.KEY_FILE).read_bytes()
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o444
    assert stat.S_IMODE(private_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((private_dir / manage.KEY_FILE).stat().st_mode) == 0o600
    original = key_file.stat().st_ino
    assert manage.prepare_container_key(private_dir).stat().st_ino == original


@pytest.mark.parametrize("kind", ["different", "unsafe-mode", "symlink", "hardlink"])
def test_container_key_never_overwrites_unsafe_or_different_existing_copy(private_dir, tmp_path, kind):
    configure(private_dir)
    key_file = private_dir / manage.CONTAINER_KEY_FILE
    write_file(key_file, FAKE_KEY if kind != "different" else "old key", mode=0o444)
    if kind == "unsafe-mode":
        key_file.chmod(0o600)
    elif kind == "symlink":
        key_file.unlink()
        external = tmp_path / "external-key"
        write_file(external, "do not touch", mode=0o444)
        key_file.symlink_to(external)
    elif kind == "hardlink":
        os.link(key_file, tmp_path / "key-link")
    before = key_file.read_bytes()
    with pytest.raises((manage.ConfigurationError, OSError)):
        manage.prepare_container_key(private_dir)
    assert key_file.read_bytes() == before


def test_compose_environment_does_not_inherit_any_model_keys_or_compose_flags(private_dir):
    configuration = configure(private_dir)
    environment = manage.compose_environment(configuration, private_dir / manage.CONTAINER_KEY_FILE, {
        "PATH": "/usr/bin", "HOME": "/example/home", "DOCKER_HOST": "unix:///docker.sock",
        "LLM_BINDING_API_KEY": FAKE_KEY, "OPENAI_API_KEY": FAKE_KEY, "STEPFUN_API_KEY": FAKE_KEY,
        "COMPOSE_FILE": "/untrusted/compose.yaml", "COMPOSE_PROFILES": "https",
        "DOCTHINKER_DEPLOY_KEY_FILE": "/untrusted/key", "PYTHONPATH": "/untrusted/modules",
    })
    assert environment["PATH"] == "/usr/bin"
    assert environment["HOME"] == "/example/home"
    assert environment["COMPOSE_DISABLE_ENV_FILE"] == "true"
    assert environment["DOCTHINKER_DEPLOY_KEY_FILE"] == str(private_dir / manage.CONTAINER_KEY_FILE)
    assert FAKE_KEY not in repr(environment)
    for name in ("LLM_BINDING_API_KEY", "OPENAI_API_KEY", "STEPFUN_API_KEY", "COMPOSE_FILE", "COMPOSE_PROFILES", "PYTHONPATH"):
        assert name not in environment


def test_storage_quota_environment_flows_through_compose_without_other_parent_values(private_dir):
    configuration = manage.Configuration("http://127.0.0.1:8088", "https://provider.example.invalid/v1", "offline-model", True)
    result = manage.compose_environment(configuration, private_dir / manage.CONTAINER_KEY_FILE, {
        "DOCTHINKER_RESEARCH_STORAGE_BYTES": "1073741824",
        "DOCTHINKER_REQUEST_JOURNAL_BYTES": "536870912", "ARBITRARY_PRIVATE_VALUE": "not inherited",
        "DOCTHINKER_INVITE_REGISTRY_BYTES": "134217728",
    })
    assert result["DOCTHINKER_RESEARCH_STORAGE_BYTES"] == "1073741824"
    assert result["DOCTHINKER_REQUEST_JOURNAL_BYTES"] == "536870912"
    assert result["DOCTHINKER_INVITE_REGISTRY_BYTES"] == "134217728"
    assert "ARBITRARY_PRIVATE_VALUE" not in result
    compose = manage.COMPOSE_FILE.read_text()
    assert '${DOCTHINKER_RESEARCH_STORAGE_BYTES:-536870912}' in compose
    assert '${DOCTHINKER_REQUEST_JOURNAL_BYTES:-268435456}' in compose
    assert '${DOCTHINKER_INVITE_REGISTRY_BYTES:-67108864}' in compose


@pytest.mark.parametrize("value, expected", [(None, "false"), ("false", "false"), ("0", "false"), ("1", "true"), (" TRUE ", "true")])
def test_experimental_assistance_is_explicitly_opt_in_through_compose(private_dir, value, expected):
    configuration = configure(private_dir)
    environment = {} if value is None else {"DOCTHINKER_RESEARCH_CONTEXT_ASSIST": value}
    result = manage.compose_environment(configuration, private_dir / manage.CONTAINER_KEY_FILE, environment)
    assert result["DOCTHINKER_RESEARCH_CONTEXT_ASSIST"] == expected
    assert '${DOCTHINKER_RESEARCH_CONTEXT_ASSIST:-false}' in manage.COMPOSE_FILE.read_text()


@pytest.mark.parametrize("value", ["", "yes", "invalid-private-value", 1, None])
def test_invalid_experimental_assistance_flag_is_rejected_without_echo(private_dir, value):
    configuration = configure(private_dir)
    with pytest.raises(manage.ConfigurationError) as error:
        manage.compose_environment(configuration, private_dir / manage.CONTAINER_KEY_FILE,
                                   {"DOCTHINKER_RESEARCH_CONTEXT_ASSIST": value})
    assert "invalid-private-value" not in str(error.value)


@pytest.mark.parametrize("value, expected", [(None, "auto"), ("auto", "auto"), ("off", "off"),
                                            ("local", "local"), (" OFF ", "off")])
def test_semantic_mode_passes_through_compose_with_explicit_default(private_dir, value, expected):
    configuration = configure(private_dir)
    environment = {} if value is None else {"DOCTHINKER_RESEARCH_SEMANTIC": value}
    result = manage.compose_environment(configuration, private_dir / manage.CONTAINER_KEY_FILE, environment)
    assert result["DOCTHINKER_RESEARCH_SEMANTIC"] == expected
    assert '${DOCTHINKER_RESEARCH_SEMANTIC:-auto}' in manage.COMPOSE_FILE.read_text()


@pytest.mark.parametrize("value", ["", "false", "invalid-private-value", 1, None])
def test_invalid_semantic_mode_is_rejected_without_echo(private_dir, value):
    configuration = configure(private_dir)
    with pytest.raises(manage.ConfigurationError) as error:
        manage.compose_environment(configuration, private_dir / manage.CONTAINER_KEY_FILE,
                                   {"DOCTHINKER_RESEARCH_SEMANTIC": value})
    assert "invalid-private-value" not in str(error.value)


@pytest.mark.parametrize("key", ["DOCTHINKER_RESEARCH_STORAGE_BYTES", "DOCTHINKER_REQUEST_JOURNAL_BYTES", "DOCTHINKER_INVITE_REGISTRY_BYTES"])
@pytest.mark.parametrize("invalid", ["", "0", "65535", "1e9", "100000\n", str((1 << 50) + 1), "private-not-a-number"])
def test_invalid_storage_quota_fails_before_compose_and_does_not_echo(private_dir, key, invalid):
    configuration = manage.Configuration("http://127.0.0.1:8088", "https://provider.example.invalid/v1", "offline-model", True)
    with pytest.raises(manage.ConfigurationError) as result:
        manage.compose_environment(configuration, private_dir / manage.CONTAINER_KEY_FILE, {key: invalid})
    assert "private-not-a-number" not in str(result.value)


@pytest.mark.parametrize("command, suffix", [
    ("build", ["build", "gateway"]),
    ("init", ["run", "--rm", "--no-deps", "gateway", "python", "-m", "docthinker.server.invite_admin", "init"]),
    ("add-user", ["run", "--rm", "--no-deps", "gateway", "python", "-m", "docthinker.server.invite_admin", "add-user", "--username", "tester_1", "--upstream-port", "8102"]),
    ("revoke-user", ["run", "--rm", "--no-deps", "gateway", "python", "-m", "docthinker.server.invite_admin", "revoke-user", "--username", "tester_1"]),
    ("reissue-invite", ["run", "--rm", "--no-deps", "gateway", "python", "-m", "docthinker.server.invite_admin", "reissue-invite", "--username", "tester_1", "--expires-hours", "48"]),
    ("check", ["run", "--rm", "--no-deps", "gateway", "python", "-m", "docthinker.server.invite_runtime", "--check"]),
    ("down", ["down"]),
    ("status", ["ps"]),
])
def test_commands_use_only_fixed_local_compose_file(private_dir, docker_mock, command, suffix):
    configuration = configure(private_dir)
    assert manage.run_compose(command, SimpleNamespace(username="tester_1", port=8102, expires_hours=48), configuration, private_dir) == 0
    assert len(docker_mock) == 1
    argv, kwargs = docker_mock[0]
    assert argv == ["/test/docker", "compose", "-p", manage.PROJECT, "-f", str(manage.COMPOSE_FILE), "--env-file", "/dev/null", *suffix]
    assert kwargs["cwd"] == manage.REPOSITORY
    assert kwargs["shell"] is False and kwargs["check"] is False
    assert FAKE_KEY not in repr(argv) and FAKE_KEY not in repr(kwargs)
    assert "https" not in argv
    if command == "down":
        assert "--volumes" not in argv and "-v" not in argv


def test_up_checks_configuration_before_starting(private_dir, docker_mock):
    configuration = configure(private_dir)
    assert manage.run_compose("up", SimpleNamespace(), configuration, private_dir) == 0
    assert len(docker_mock) == 2
    assert docker_mock[0][0][-2:] == ["docthinker.server.invite_runtime", "--check"]
    assert docker_mock[1][0][-6:] == ["up", "-d", "--wait", "--wait-timeout", "150", "gateway"]


def test_up_does_not_start_after_preflight_failure(private_dir, monkeypatch, docker_mock, capsys):
    configuration = configure(private_dir)

    def fail(argv, **kwargs):
        docker_mock.append((argv, kwargs))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(manage.subprocess, "run", fail)
    assert manage.run_compose("up", SimpleNamespace(), configuration, private_dir) == 7
    assert len(docker_mock) == 1 and "up" not in docker_mock[0][0]
    assert "没有启动服务" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["up", "check"])
def test_permission_false_blocks_start_and_check_before_any_docker_or_key_copy(private_dir, monkeypatch, command):
    configuration = configure(private_dir, sharing_authorized=False)
    monkeypatch.setattr(manage.shutil, "which", lambda *_args, **_kwargs: pytest.fail("must not look up Docker"))
    monkeypatch.setattr(manage.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must not execute Docker"))
    with pytest.raises(manage.ConfigurationError, match="共享许可"):
        manage.run_compose(command, SimpleNamespace(), configuration, private_dir)
    assert not (private_dir / manage.CONTAINER_KEY_FILE).exists()


def test_missing_docker_reports_actionable_static_error_without_copying_secret(private_dir, monkeypatch, capsys):
    configure(private_dir)
    monkeypatch.setattr(manage.shutil, "which", lambda *_args, **_kwargs: None)
    assert manage.main(["--config-dir", str(private_dir), "build"]) == 1
    error = capsys.readouterr().err
    assert "未找到 Docker" in error and "Docker Desktop" in error
    assert FAKE_KEY not in error
    assert not (private_dir / manage.CONTAINER_KEY_FILE).exists()


@pytest.mark.parametrize("code, expected", [(3, 3), (-9, 1), (None, 1)])
def test_docker_failure_has_nonzero_exit_and_static_message(private_dir, monkeypatch, docker_mock, capsys, code, expected):
    configuration = configure(private_dir)
    monkeypatch.setattr(manage.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=code))
    assert manage.run_compose("status", SimpleNamespace(), configuration, private_dir) == expected
    assert "不会自动重试" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["build", "init", "check", "up", "down", "status"])
def test_main_dispatches_supported_commands_without_real_docker(private_dir, docker_mock, command):
    configure(private_dir)
    assert manage.main(["--config-dir", str(private_dir), command]) == 0
    assert len(docker_mock) == (2 if command == "up" else 1)


@pytest.mark.parametrize("command", ["add-user", "revoke-user", "reissue-invite"])
def test_main_validates_and_dispatches_user_management(private_dir, docker_mock, command):
    configure(private_dir)
    arguments = ["--config-dir", str(private_dir), command, "--username", "tester_1"]
    if command == "add-user":
        arguments.extend(["--port", "8103"])
    assert manage.main(arguments) == 0
    assert "tester_1" in docker_mock[0][0]


def test_reissue_uses_validated_admin_path_and_explicit_expiry(private_dir, docker_mock):
    configure(private_dir)
    assert manage.main(["--config-dir", str(private_dir), "reissue-invite", "--username", "tester_1", "--expires-hours", "12"]) == 0
    assert len(docker_mock) == 1
    argv, options = docker_mock[0]
    assert argv[-5:] == ["reissue-invite", "--username", "tester_1", "--expires-hours", "12"]
    assert "--no-deps" in argv and "up" not in argv and "--volumes" not in argv
    assert options["env"]["DOCTHINKER_DEPLOY_KEY_FILE"] == str(private_dir / manage.CONTAINER_KEY_FILE)
    assert FAKE_KEY not in repr((argv, options))


@pytest.mark.parametrize("lifetime", ["0", "169", "1.5", "not-a-number"])
def test_reissue_rejects_invalid_expiry_before_configuration_or_docker(monkeypatch, lifetime):
    monkeypatch.setattr(manage, "default_config_dir", lambda: pytest.fail("must not access credentials"))
    monkeypatch.setattr(manage, "run_compose", lambda *_args: pytest.fail("must not run Docker"))
    with pytest.raises(SystemExit) as error:
        manage.main(["reissue-invite", "--username", "tester_1", "--expires-hours", lifetime])
    assert error.value.code == 2


def test_reissue_cannot_use_stop_only_path_when_configuration_is_missing(tmp_path, docker_mock, capsys):
    assert manage.main(["--config-dir", str(tmp_path.resolve() / "missing"), "reissue-invite", "--username", "tester_1"]) == 1
    assert not docker_mock
    assert "先运行 setup" in capsys.readouterr().err


@pytest.mark.parametrize("username", ["ab", "../tester", "test;echo", " space ", "x" * 33])
def test_main_rejects_bad_username_before_docker(private_dir, monkeypatch, capsys, username):
    monkeypatch.setattr(manage.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must not run"))
    assert manage.main(["--config-dir", str(private_dir), "revoke-user", "--username", username]) == 1
    assert username not in capsys.readouterr().err


def test_cli_does_not_accept_arbitrary_public_compose_or_ports(private_dir):
    for arguments in (["up", "--public"], ["up", "--compose-file", "other.yaml"],
                      ["add-user", "--username", "tester", "--port", "80"]):
        with pytest.raises(SystemExit) as caught:
            manage.main(["--config-dir", str(private_dir), *arguments])
        assert caught.value.code == 2


def test_main_does_not_echo_raw_filesystem_or_subprocess_exception(private_dir, monkeypatch, docker_mock, capsys):
    configure(private_dir)
    monkeypatch.setattr(manage.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError(FAKE_KEY)))
    assert manage.main(["--config-dir", str(private_dir), "status"]) == 1
    assert FAKE_KEY not in capsys.readouterr().err


def test_main_cancellation_has_static_message_and_no_key(private_dir, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert manage.main(["--config-dir", str(private_dir), "setup"]) == 1
    assert "操作已取消" in capsys.readouterr().err
    assert list(private_dir.iterdir()) == []


def test_helper_is_stdlib_and_has_no_network_or_dotenv_imports():
    import ast

    source = (ROOT / "deploy/invite/manage.py").read_text()
    tree = ast.parse(source)
    imports = {node.names[0].name.split(".", 1)[0] if isinstance(node, ast.Import) else (node.module or "").split(".", 1)[0]
               for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))}
    assert imports <= sys.stdlib_module_names
    assert not imports & {"socket", "urllib.request", "http", "requests", "dotenv"}
    assert "shell=True" not in source
