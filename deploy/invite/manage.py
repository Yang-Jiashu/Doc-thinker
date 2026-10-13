#!/usr/bin/env python3
"""Owner-only deployment helper; never reads dotenv or shell model credentials.

Settings and the original model key live outside the checkout (0700 directory,
0600 files). Docker Compose receives only a key-file path. A separate 0444 key
copy inside that same owner-only directory permits the fixed non-root container
UID to read its single-file, read-only secret mount; other host users cannot
traverse the parent directory. Existing copies are verified, never overwritten.
No command enables a public listener or selects the optional HTTPS overlay.
"""

from __future__ import annotations

import argparse
import getpass
import hmac
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

REPOSITORY = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPOSITORY / "deploy" / "invite" / "compose.yaml"
PROJECT = "docthinker-invite"
SHARING_ACK = "I_HAVE_PERMISSION_TO_SHARE"
SETTINGS_FILE = "settings.json"
KEY_FILE = "llm-api-key"
CONTAINER_KEY_FILE = "container-llm-key"
DEFAULT_ORIGIN = "http://127.0.0.1:8088"
_USERNAME = re.compile(r"[A-Za-z0-9_-]{3,32}\Z")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,199}\Z")
_HOST = re.compile(r"[A-Za-z0-9.-]+\Z")
_SETTINGS_FIELDS = {"origin", "api_base", "model", "sharing_authorized"}
_DOCKER_ENV = {
    "PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TERM",
    "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH", "XDG_CONFIG_HOME",
}


class ConfigurationError(ValueError):
    """Messages are static and safe to display, never raw filesystem failures."""


@dataclass(frozen=True)
class Configuration:
    origin: str
    api_base: str
    model: str
    sharing_authorized: bool


def default_config_dir() -> Path:
    return Path.home() / ".local" / "share" / "docthinker-deployment"


def _url(value, *, origin=False) -> str:
    if not isinstance(value, str) or not value or len(value) > 2048:
        raise ConfigurationError("服务地址不符合要求。")
    try:
        parsed = urlsplit(value)
        port = parsed.port
        if (any(ord(c) <= 32 or ord(c) == 127 or c in "\\{}" for c in value)
                or parsed.scheme not in {"http", "https"}
                or not parsed.hostname or not _HOST.fullmatch(parsed.hostname)
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment
                or (port is not None and not 1 <= port <= 65535)
                or (parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1"})
                or (origin and parsed.path)):
            raise ValueError
        if origin and value != f"{parsed.scheme}://{parsed.netloc.lower()}":
            raise ValueError
    except ValueError:
        raise ConfigurationError("服务地址须为 HTTPS 或本机 HTTP，不能含账号、密钥、查询参数；网站地址不能含路径。") from None
    return value


def _model(value) -> str:
    # Match the invitation worker's actual model contract at setup time,
    # before accepting a key or leaving a configuration that cannot start.
    if not isinstance(value, str) or not _MODEL.fullmatch(value):
        raise ConfigurationError("请填写有效的模型标识，不要填写密钥。")
    return value


def _key_bytes(value: str) -> bytes:
    if (not isinstance(value, str) or not value or value == "EMPTY" or value != value.strip()
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ConfigurationError("模型密钥不能为空或包含控制字符、首尾空格。")
    try:
        body = value.encode("utf-8")
    except UnicodeError:
        raise ConfigurationError("模型密钥编码无效。") from None
    if len(body) > 4096:
        raise ConfigurationError("模型密钥过长。")
    return body


def config_directory(value, *, create=False) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or path != path.resolve():
        raise ConfigurationError("配置目录须为不含符号链接的绝对路径。")
    if path == REPOSITORY or path.is_relative_to(REPOSITORY) or REPOSITORY.is_relative_to(path):
        raise ConfigurationError("配置与密钥目录必须位于项目目录之外，不能是项目的父目录。")
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        info = path.lstat()
    except OSError:
        raise ConfigurationError("配置尚未就绪，请先运行 setup。") from None
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise ConfigurationError("配置目录必须由当前用户拥有，权限为 0700，且不能是符号链接。")
    return path


@contextmanager
def _directory_fd(path: Path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise ConfigurationError("配置目录必须由当前用户拥有且权限为 0700。")
        yield descriptor
    finally:
        os.close(descriptor)


def _read_file(directory_fd: int, name: str, *, mode=0o600, limit=16384) -> bytes:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != mode or info.st_nlink != 1):
            raise ConfigurationError("配置或密钥文件的所有者、权限或类型不安全。")
        if not 0 < info.st_size <= limit:
            raise ConfigurationError("配置或密钥文件大小不符合要求。")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            body = handle.read(limit + 1)
        if not 0 < len(body) <= limit:
            raise ConfigurationError("配置或密钥文件大小不符合要求。")
        return body
    finally:
        os.close(descriptor)


def _write_new(directory_fd: int, name: str, body: bytes, *, mode=0o600):
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         mode, dir_fd=directory_fd)
    info = os.fstat(descriptor)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(body)
            handle.flush()
            os.fsync(descriptor)
        return info.st_dev, info.st_ino
    except BaseException:
        _remove_owned_new(directory_fd, name, (info.st_dev, info.st_ino))
        raise
    finally:
        os.close(descriptor)


def _remove_owned_new(directory_fd, name, identity):
    try:
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (info.st_dev, info.st_ino) == identity:
            os.unlink(name, dir_fd=directory_fd)
    except FileNotFoundError:
        pass


def setup(path: Path) -> None:
    directory = config_directory(path, create=True)
    with _directory_fd(directory) as descriptor:
        for name in (SETTINGS_FILE, KEY_FILE, CONTAINER_KEY_FILE):
            try:
                os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise ConfigurationError("配置或密钥已存在；不会覆盖。请使用新的私有配置目录。")
        origin = _url(input(f"网站地址 [{DEFAULT_ORIGIN}]：").strip() or DEFAULT_ORIGIN, origin=True)
        api_base = _url(input("模型 API Base URL（例如 https://api.example.com/v1）：").strip())
        model = _model(input("模型标识：").strip())
        key = _key_bytes(getpass.getpass("模型 API Key（输入不显示）："))
        permission = input(
            "仅当你已确认模型服务允许多人共享使用，输入 I_HAVE_PERMISSION_TO_SHARE；否则直接回车："
        ).strip()
        if permission not in {"", SHARING_ACK}:
            raise ConfigurationError("共享许可确认词不匹配；未保存配置。")
        settings = {"origin": origin, "api_base": api_base, "model": model,
                    "sharing_authorized": permission == SHARING_ACK}
        identity = _write_new(descriptor, KEY_FILE, key)
        try:
            _write_new(descriptor, SETTINGS_FILE,
                       (json.dumps(settings, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        except BaseException:
            _remove_owned_new(descriptor, KEY_FILE, identity)
            raise
    print("配置已保存到项目之外的私有目录；模型密钥未写入 JSON、网址或命令参数。")
    if not settings["sharing_authorized"]:
        print("共享许可尚未确认，check 和 up 将保持阻止。")


def load_configuration(path: Path) -> Configuration:
    directory = config_directory(path)
    with _directory_fd(directory) as descriptor:
        try:
            value = json.loads(_read_file(descriptor, SETTINGS_FILE).decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            raise ConfigurationError("设置文件格式无效，请检查私有配置。") from None
        if not isinstance(value, dict) or set(value) != _SETTINGS_FIELDS or type(value.get("sharing_authorized")) is not bool:
            raise ConfigurationError("设置文件字段不符合要求；密钥不能写入设置 JSON。")
        configuration = Configuration(_url(value["origin"], origin=True), _url(value["api_base"]),
                                      _model(value["model"]), value["sharing_authorized"])
        # Validate the source key without ever echoing it or placing it in env.
        try:
            _key_bytes(_read_file(descriptor, KEY_FILE, limit=4096).decode("utf-8"))
        except UnicodeError:
            raise ConfigurationError("模型密钥编码无效。") from None
        return configuration


def prepare_container_key(path: Path) -> Path:
    directory = config_directory(path)
    with _directory_fd(directory) as descriptor:
        source = _read_file(descriptor, KEY_FILE, limit=4096)
        try:
            _key_bytes(source.decode("utf-8"))
        except UnicodeError:
            raise ConfigurationError("模型密钥编码无效。") from None
        try:
            existing = _read_file(descriptor, CONTAINER_KEY_FILE, mode=0o444, limit=4096)
        except FileNotFoundError:
            try:
                _write_new(descriptor, CONTAINER_KEY_FILE, source, mode=0o444)
            except FileExistsError:
                pass  # Concurrent creation is accepted only after the same validation below.
            existing = _read_file(descriptor, CONTAINER_KEY_FILE, mode=0o444, limit=4096)
        if not hmac.compare_digest(source, existing):
            raise ConfigurationError("容器密钥副本与原密钥不一致；不会覆盖，请由运营者检查。")
    return directory / CONTAINER_KEY_FILE


def compose_environment(configuration: Configuration, key_file: Path, environment=None) -> dict[str, str]:
    inherited = os.environ if environment is None else environment
    result = {name: value for name, value in inherited.items() if name in _DOCKER_ENV}
    assist = inherited.get("DOCTHINKER_RESEARCH_CONTEXT_ASSIST", "false")
    if not isinstance(assist, str) or assist.strip().lower() not in {"true", "false", "1", "0"}:
        raise ConfigurationError("实验性证据整理开关须为 true、false、1 或 0。")
    result["DOCTHINKER_RESEARCH_CONTEXT_ASSIST"] = "true" if assist.strip().lower() in {"true", "1"} else "false"
    semantic_mode = inherited.get("DOCTHINKER_RESEARCH_SEMANTIC", "auto")
    if not isinstance(semantic_mode, str) or semantic_mode.strip().lower() not in {"auto", "off", "local"}:
        raise ConfigurationError("研究语义检索模式须为 auto、off 或 local。")
    result["DOCTHINKER_RESEARCH_SEMANTIC"] = semantic_mode.strip().lower()
    for key in ("DOCTHINKER_RESEARCH_STORAGE_BYTES", "DOCTHINKER_REQUEST_JOURNAL_BYTES", "DOCTHINKER_INVITE_REGISTRY_BYTES"):
        if key in inherited:
            value = inherited[key]
            if (not isinstance(value, str) or not re.fullmatch(r"[0-9]{1,16}", value)
                    or not 65_536 <= int(value) <= 1 << 50):
                raise ConfigurationError("存储配额须为 65536 至 1125899906842624 之间的整数字节数。")
            result[key] = str(int(value))
    result.update({
        "COMPOSE_DISABLE_ENV_FILE": "true",
        "DOCTHINKER_DEPLOY_ORIGIN": configuration.origin,
        "DOCTHINKER_DEPLOY_API_BASE": configuration.api_base,
        "DOCTHINKER_DEPLOY_MODEL": configuration.model,
        "DOCTHINKER_DEPLOY_SHARING_ACK": SHARING_ACK if configuration.sharing_authorized else "",
        "DOCTHINKER_DEPLOY_KEY_FILE": str(key_file),
    })
    return result


def _compose_arguments(command: str, args) -> list[str]:
    run = ["run", "--rm", "--no-deps", "gateway", "python", "-m"]
    if command == "build":
        return ["build", "gateway"]
    if command == "init":
        return [*run, "docthinker.server.invite_admin", "init"]
    if command == "add-user":
        return [*run, "docthinker.server.invite_admin", "add-user", "--username", args.username,
                "--upstream-port", str(args.port), *(["--allow-uploads"] if getattr(args, "research_uploads", False) else [])]
    if command == "revoke-user":
        return [*run, "docthinker.server.invite_admin", "revoke-user", "--username", args.username]
    if command == "reissue-invite":
        return [*run, "docthinker.server.invite_admin", "reissue-invite", "--username", args.username,
                "--expires-hours", str(args.expires_hours)]
    if command == "check":
        return [*run, "docthinker.server.invite_runtime", "--check"]
    if command == "up":
        return ["up", "-d", "--wait", "--wait-timeout", "150", "gateway"]
    if command == "down":
        return ["down"]  # Never --volumes/-v: accounts and data must survive shutdown.
    if command == "status":
        return ["ps"]
    raise ConfigurationError("不支持的部署命令。")


def run_control_compose(command: str) -> int:
    """Inspect/stop the fixed project even if credentials are lost or invalid.

    Compose must parse its required secret-file interpolation for ps/down,
    although neither operation mounts secrets or creates a container. The
    placeholder is only available on this two-command allowlist; starting,
    building, account writes, and volume deletion cannot use this path.
    """
    if command not in {"down", "status"}:
        raise ConfigurationError("此控制路径只允许查看状态或停止现有服务。")
    environment = {key: value for key, value in os.environ.items() if key in _DOCKER_ENV}
    environment.update({"COMPOSE_DISABLE_ENV_FILE": "true", "DOCTHINKER_DEPLOY_KEY_FILE": os.devnull})
    docker = shutil.which("docker", path=environment.get("PATH", os.defpath))
    if not docker:
        raise ConfigurationError("未找到 Docker。请安装并启动 Docker Desktop 或 Docker Engine（含 Compose），再重试。")
    result = subprocess.run(
        [docker, "compose", "-p", PROJECT, "-f", str(COMPOSE_FILE), "--env-file", os.devnull,
         *_compose_arguments(command, None)],
        cwd=REPOSITORY, env=environment, check=False, shell=False,
    )
    code = result.returncode if isinstance(result.returncode, int) and result.returncode >= 0 else 1
    if code:
        print("Docker Compose 控制命令未完成；请检查 Docker 状态，没有启动服务或删除数据卷，不会自动重试。", file=sys.stderr)
    return code


def run_compose(command: str, args, configuration: Configuration, path: Path) -> int:
    if command in {"down", "status"}:
        return run_control_compose(command)
    if command in {"up", "check"} and not configuration.sharing_authorized:
        raise ConfigurationError("尚未明确确认模型服务的多人共享许可；不会启动服务。")
    environment = compose_environment(configuration, path / CONTAINER_KEY_FILE)
    environment["DOCTHINKER_DEPLOY_RESEARCH_ENABLED"] = "true" if getattr(args, "research", False) else "false"
    docker = shutil.which("docker", path=environment.get("PATH", os.defpath))
    if not docker:
        raise ConfigurationError("未找到 Docker。请安装并启动 Docker Desktop 或 Docker Engine（含 Compose），再重试。")
    environment["DOCTHINKER_DEPLOY_KEY_FILE"] = str(prepare_container_key(path))
    prefix = [docker, "compose", "-p", PROJECT, "-f", str(COMPOSE_FILE), "--env-file", "/dev/null"]

    def execute(name):
        result = subprocess.run([*prefix, *_compose_arguments(name, args)], cwd=REPOSITORY,
                                env=environment, check=False, shell=False)
        return result.returncode if isinstance(result.returncode, int) and result.returncode >= 0 else 1

    if command == "up":
        result = execute("check")
        if result:
            print("启动前检查未通过；没有启动服务。", file=sys.stderr)
            return result
    result = execute(command)
    if result:
        print("Docker Compose 未完成操作；请检查 Docker 状态和上方诊断，不会自动重试。", file=sys.stderr)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="封装部署与受邀用户管理；默认仅本机开放，不读取 .env。")
    parser.add_argument("--config-dir", type=Path, help="项目之外的绝对私有目录，默认 ~/.local/share/docthinker-deployment")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("setup", "build", "init", "check", "up", "down", "status"):
        command_parser = commands.add_parser(command)
        if command in {"check", "up"}:
            command_parser.add_argument("--research", action="store_true", help="显式启用有界离线论文解析、原文检索与确认记忆，不启用旧版 RAG 或语音")
    preflight = commands.add_parser("preflight", help="只读上线门禁报告；不读取密钥内容、不调用 Docker 或模型、不开放端口")
    preflight.add_argument("--web-dir", type=Path, default=REPOSITORY / "clients/mobile/dist/build/h5")
    backup = commands.add_parser("backup", help="离线私有目录冷备；须先停服务/管理写入，不备份配置密钥，不调用 Docker")
    backup.add_argument("--data-dir", type=Path, required=True)
    backup.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("backup-verify", help="核验离线备份清单、SHA256 与 SQLite 完整性")
    verify.add_argument("--backup-dir", type=Path, required=True)
    restore = commands.add_parser("restore", help="将离线备份恢复到新的私有目录，绝不覆盖现有数据")
    restore.add_argument("--backup-dir", type=Path, required=True)
    restore.add_argument("--target-dir", type=Path, required=True)
    commands.add_parser("backup-self-test", help="只用临时模拟数据验证备份/恢复/篡改拒绝；不读真实配置或数据")
    for command in ("export-account", "purge-account"):
        account = commands.add_parser(command, help="仅离线本地账号维护；新协议、撤销账号与停机锁必需，不调用 Docker/模型")
        account.add_argument("--data-dir", type=Path, required=True)
        account.add_argument("--username", required=True)
        account.add_argument("--confirm-username", required=True)
        if command == "export-account":
            account.add_argument("--output", type=Path, required=True)
    add = commands.add_parser("add-user", help="添加一名受邀用户；邀请码只在管理员输出中显示，请私下转交")
    add.add_argument("--username", required=True)
    add.add_argument("--port", type=int, choices=range(8101, 8106), required=True)
    add.add_argument("--research-uploads", action="store_true", help="授权此用户上传研究资料；启动还须使用 up --research")
    revoke = commands.add_parser("revoke-user")
    revoke.add_argument("--username", required=True)
    reissue = commands.add_parser("reissue-invite", help="仅为未注册账号重发邀请码；旧码立即失效，不改身份/权限/数据")
    reissue.add_argument("--username", required=True)
    reissue.add_argument("--expires-hours", type=int, choices=range(1, 169), default=48)
    args = parser.parse_args(argv)
    try:
        if args.command in {"down", "status"}:
            return run_control_compose(args.command)
        if args.command in {"export-account", "purge-account"}:
            # Explicit owner-side directory workflow, like cold backup. Never
            # infer a Docker volume host path or mount credentials for this.
            if not _USERNAME.fullmatch(args.username) or args.confirm_username != args.username.lower():
                raise ConfigurationError("用户名不符合要求，或确认名称与小写用户名不完全一致。")
            arguments = [sys.executable, "-m", "docthinker.server.invite_admin", args.command,
                         "--username", args.username, "--confirm-username", args.confirm_username]
            if args.command == "export-account":
                arguments.extend(["--output", str(args.output)])
            environment = {key: value for key, value in os.environ.items()
                           if key in {"PATH", "LANG", "LC_ALL", "LC_CTYPE"}}
            environment.update({"PYTHONPATH": str(REPOSITORY), "PYTHONNOUSERSITE": "1",
                                "PYTHONDONTWRITEBYTECODE": "1", "PYTHON_DOTENV_DISABLED": "1",
                                "DOCTHINKER_INVITE_DATA_DIR": str(args.data_dir)})
            result = subprocess.run(arguments, cwd=REPOSITORY, env=environment, check=False, shell=False)
            return result.returncode if isinstance(result.returncode, int) and result.returncode >= 0 else 1
        if args.command in {"backup", "backup-verify", "restore", "backup-self-test"}:
            spec = importlib.util.spec_from_file_location("invite_backup_helper", Path(__file__).with_name("backup.py"))
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            try:
                if args.command == "backup":
                    report = helper.create_backup(args.data_dir, args.output)
                elif args.command == "backup-verify":
                    report = helper.verify_backup(args.backup_dir)
                elif args.command == "restore":
                    report = helper.restore_backup(args.backup_dir, args.target_dir)
                else:
                    report = helper.self_test()
            except helper.BackupError as error:
                raise ConfigurationError(str(error)) from None
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0
        path = args.config_dir if args.config_dir is not None else default_config_dir()
        if args.command == "preflight":
            spec = importlib.util.spec_from_file_location("invite_preflight_helper", Path(__file__).with_name("preflight.py"))
            helper = importlib.util.module_from_spec(spec)
            helper.manage = sys.modules[__name__]
            spec.loader.exec_module(helper)
            return helper.main(["--config-dir", str(path), "--web-dir", str(args.web_dir)])
        if args.command in {"add-user", "revoke-user", "reissue-invite"} and not _USERNAME.fullmatch(args.username):
            raise ConfigurationError("用户名须为 3–32 位字母、数字、下划线或短横线。")
        if args.command == "setup":
            setup(path)
            return 0
        path = config_directory(path)
        configuration = load_configuration(path)
        return run_compose(args.command, args, configuration, path)
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
    except (OSError, ValueError, TypeError):
        print("部署配置或命令未完成。请检查私有目录、文件权限及 Docker；不会显示密钥或覆盖已有文件。", file=sys.stderr)
    except (KeyboardInterrupt, EOFError):
        print("操作已取消，未启动服务。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
