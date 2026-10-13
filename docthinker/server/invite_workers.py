"""Bounded, local-only workers for the fixed invitation gateway user list.

This isolates Python global state and storage, not hostile OS accounts. The
operator acknowledgement is not evidence of a provider's sharing permission.
No dotenv files are read, provider requests made, or accounts created here.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

MAX_WORKERS = 5
WORKER_PROFILE = "chat/history only; uploads/RAG/voice disabled"
SHARING_ACK_ENV = "DOCTHINKER_INVITE_SHARED_MODEL_ACK"
SHARING_ACK_VALUE = "I_HAVE_PERMISSION_TO_SHARE"
_ROOT = Path(__file__).resolve().parents[2]
_USER_ID = re.compile(r"[0-9a-f]{32}\Z")
_TOKEN = re.compile(r"[A-Za-z0-9_-]{32,256}\Z")
_MODEL_OPTIONS = (
    "LLM_MODELS", "LLM_TEMPERATURE", "KEYWORD_LLM_MODEL", "ENTITY_EXTRACTION_LLM_MODEL",
    "EXTRACTION_LLM_MODEL", "VLM_MODEL", "DOCTHINKER_LLM_CREDENTIAL_BINDING",
)


class WorkerConfigurationError(ValueError):
    """Only static, non-secret diagnostics may be passed to this exception."""


def worker_profile(environment: Mapping[str, str]) -> str:
    return ("bounded offline research + confirmed project memory; legacy RAG/voice disabled"
            if environment.get("DOCTHINKER_RESEARCH_ENABLED", "").lower() in {"true", "1"} else WORKER_PROFILE)


@dataclass(frozen=True)
class WorkerBinding:
    user_id: str
    port: int
    token: str = field(repr=False)
    allow_uploads: bool = False


def require_sharing_ack(environment: Mapping[str, str]) -> None:
    if environment.get(SHARING_ACK_ENV) != SHARING_ACK_VALUE:
        raise WorkerConfigurationError(
            "Shared-model startup is blocked: verify provider permission, then explicitly set "
            + SHARING_ACK_ENV + "=" + SHARING_ACK_VALUE
        )


def _api_base(value: str) -> None:
    try:
        parsed = urlsplit(value)
        if (not parsed.hostname or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or any(ord(c) <= 32 or c in "\\{}" for c in value)
                or parsed.scheme not in {"https", "http"}
                or (parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"})):
            raise ValueError
        _ = parsed.port
    except ValueError:
        raise WorkerConfigurationError("Provider endpoints require HTTPS or loopback HTTP without URL credentials") from None


def provider_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Explicit model destinations only; never inherit OpenAI/default key fallbacks."""
    require_sharing_ack(environment)
    context_assist = environment.get("DOCTHINKER_RESEARCH_CONTEXT_ASSIST", "false")
    if not isinstance(context_assist, str) or context_assist.strip().lower() not in {"true", "false", "1", "0"}:
        raise WorkerConfigurationError("DOCTHINKER_RESEARCH_CONTEXT_ASSIST must be true, false, 1 or 0")
    context_assist = context_assist.strip().lower() in {"true", "1"}
    semantic_mode = environment.get("DOCTHINKER_RESEARCH_SEMANTIC", "auto")
    if not isinstance(semantic_mode, str) or semantic_mode.strip().lower() not in {"auto", "off", "local"}:
        raise WorkerConfigurationError("DOCTHINKER_RESEARCH_SEMANTIC must be auto, off or local")
    semantic_mode = semantic_mode.strip().lower()
    if (any(environment.get(key, "").strip() not in {"", "EMPTY"} for key in (
        "EMBEDDING_BINDING_API_KEY", "RERANK_API_KEY", "VOICE_ASR_KEY", "VOICE_TTS_KEY",
    )) or any(environment.get(key, "").strip().lower() not in {"", "false", "0", "no", "off"} for key in (
        "DOCTHINKER_VOICE_ENABLED", "DOCTHINKER_INVITE_UPLOADS_ENABLED",
    ))):
        raise WorkerConfigurationError("This launcher supports chat/history only; remove embedding, rerank, upload and voice configuration")
    required = ("LLM_BINDING_API_KEY", "LLM_VLM_HOST", "LLM_MODEL")
    if any(not environment.get(key, "").strip() for key in required):
        raise WorkerConfigurationError("Explicit LLM_BINDING_API_KEY, LLM_VLM_HOST and LLM_MODEL are required")
    if (environment["LLM_BINDING_API_KEY"] == "EMPTY"
            or len(environment["LLM_BINDING_API_KEY"]) > 4096
            or any(ord(c) < 32 for c in environment["LLM_BINDING_API_KEY"])
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,199}", environment["LLM_MODEL"])):
        raise WorkerConfigurationError("Invalid explicit model credential or model identifier")
    _api_base(environment["LLM_VLM_HOST"])
    result = {key: environment[key] for key in (*required, *_MODEL_OPTIONS) if key in environment}
    # Only trusted operator settings may expand persistence. Client payloads
    # cannot set these values, and no arbitrary parent environment is inherited.
    for key in ("DOCTHINKER_RESEARCH_STORAGE_BYTES", "DOCTHINKER_REQUEST_JOURNAL_BYTES"):
        if key in environment:
            value = environment[key]
            if (not isinstance(value, str) or not re.fullmatch(r"[0-9]{1,16}", value)
                    or not 65_536 <= int(value) <= 1 << 50):
                raise WorkerConfigurationError("Invalid worker storage quota")
            result[key] = str(int(value))
    # Keep the explicit model authoritative. Invitation chat deliberately does
    # not rotate through LLM_MODELS or retry on a different billed destination.
    result["LLM_MODELS"] = environment["LLM_MODEL"]
    # Auxiliary calls follow this explicit provider, never legacy qwen defaults.
    for key in ("KEYWORD_LLM_MODEL", "ENTITY_EXTRACTION_LLM_MODEL", "VLM_MODEL"):
        result.setdefault(key, "")
    result.update({
        SHARING_ACK_ENV: SHARING_ACK_VALUE, "EMBEDDING_BINDING_API_KEY": "EMPTY",
        "RERANK_API_KEY": "", "DOCTHINKER_VOICE_ENABLED": "false",
        "LLM_ROUTER_MAX_CONCURRENCY": "2", "GRAPHCORE_LLM_MAX_ASYNC": "2",
        "GRAPHCORE_EMBEDDING_MAX_ASYNC": "2", "GRAPHCORE_MAX_PARALLEL_INSERT": "1",
        "TIMEOUT": "120",
        "DOCTHINKER_RESEARCH_ENABLED": "true" if environment.get("DOCTHINKER_RESEARCH_ENABLED", "").lower() in {"true", "1"} else "false",
        "DOCTHINKER_RESEARCH_CONTEXT_ASSIST": "true" if context_assist else "false",
        "DOCTHINKER_RESEARCH_SEMANTIC": semantic_mode,
    })
    return result


def parse_bindings(records) -> tuple[WorkerBinding, ...]:
    if not isinstance(records, (list, tuple)) or not 0 < len(records) <= MAX_WORKERS:
        raise WorkerConfigurationError("Configure between one and five fixed pending/active users")
    bindings, ids, ports, tokens = [], set(), set(), set()
    for record in records:
        try:
            user_id, token = record["id"], record["upstream_token"]
            parsed = urlsplit(record["upstream_url"])
            port = parsed.port
            valid = (
                isinstance(user_id, str) and _USER_ID.fullmatch(user_id)
                and isinstance(token, str) and _TOKEN.fullmatch(token)
                and record["status"] in {"pending", "active"}
                and parsed.scheme == "http" and parsed.hostname == "127.0.0.1"
                and parsed.username is None and parsed.password is None
                and not parsed.path and not parsed.query and not parsed.fragment
                and isinstance(port, int) and 1024 <= port <= 65535
                and record["upstream_url"] == f"http://127.0.0.1:{port}"
                and type(record.get("allow_uploads", False)) in {int, bool}
                and record.get("allow_uploads", False) in {0, 1}
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            valid = False
        if not valid:
            raise WorkerConfigurationError("Invalid fixed worker binding")
        if user_id in ids or port in ports or token in tokens:
            raise WorkerConfigurationError("Worker identities, ports and internal tokens must be distinct")
        ids.add(user_id)
        ports.add(port)
        tokens.add(token)
        bindings.append(WorkerBinding(user_id, port, token, bool(record.get("allow_uploads", False))))
    return tuple(bindings)


def _private_directory(path: Path, *, create=False) -> None:
    if create:
        path.mkdir(mode=0o700, exist_ok=True)
    try:
        info = path.lstat()
    except OSError:
        raise WorkerConfigurationError("A required private worker directory is unavailable") from None
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise WorkerConfigurationError("Worker directories must be owned by this user, private and not symlinks")


def worker_directory(root: Path, binding: WorkerBinding) -> Path:
    if not root.is_absolute():
        raise WorkerConfigurationError("Invite storage requires an absolute private directory")
    _private_directory(root)
    return root / "workers" / binding.user_id


def child_environment(root: Path, binding: WorkerBinding, environment: Mapping[str, str]) -> dict[str, str]:
    result = provider_environment(environment)
    if binding.token == result["LLM_BINDING_API_KEY"]:
        raise WorkerConfigurationError("Internal worker access tokens must not reuse provider credentials")
    directory = worker_directory(root, binding)
    # Do not pass arbitrary database/cache paths, proxies, credentials or dotenv configuration.
    result.update({key: environment[key] for key in ("PATH", "LANG", "LC_ALL") if key in environment})
    result.update({
        "PYTHONPATH": str(_ROOT), "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "PYTHON_DOTENV_DISABLED": "1", "HOME": str(directory / "home"),
        "TMPDIR": str(directory / "tmp"), "XDG_CACHE_HOME": str(directory / "cache"),
        "DOCTHINKER_MOBILE_ENABLED": "true", "DOCTHINKER_MOBILE_ACCESS_TOKEN": binding.token,
        "DOCTHINKER_MOBILE_DATA_DIR": str(directory), "DOCTHINKER_MOBILE_CORS_ORIGINS": "",
        "DOCTHINKER_INVITE_WORKER_ID": binding.user_id, "DOCTHINKER_INVITE_WORKER_PORT": str(binding.port),
        # All workers use one deployment-generated inode, never a supplied path.
        "DOCTHINKER_RESEARCH_PARSER_LOCK_PATH": str(root / "workers" / "research-parser.lock"),
    })
    # Uploads/parsing and voice remain disabled in this initial worker launcher.
    # Gateway permission alone cannot enable an unreviewed parser/provider setup.
    if binding.allow_uploads and result["DOCTHINKER_RESEARCH_ENABLED"] != "true":
        raise WorkerConfigurationError("This worker launcher is chat-only; uploads must remain disabled")
    return result


def prepare_directories(root: Path, binding: WorkerBinding) -> Path:
    directory = worker_directory(root, binding)
    for path in (root / "workers", directory, *(directory / name for name in ("runtime", "home", "tmp", "cache", "data"))):
        _private_directory(path, create=True)
    dotenv_path = directory / "runtime" / ".env"
    if dotenv_path.exists() or dotenv_path.is_symlink():
        raise WorkerConfigurationError("Worker runtime directory must not contain a dotenv file")
    return directory


@contextmanager
def supervisor_lock(root: Path):
    _private_directory(root)
    descriptor = os.open(root / "workers.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
            raise WorkerConfigurationError("Worker lock file must be private and owned by this user")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise WorkerConfigurationError("An invitation worker supervisor is already running") from None
        yield
    finally:
        os.close(descriptor)


def ensure_ports_available(bindings: tuple[WorkerBinding, ...]) -> None:
    # The child bind remains authoritative; a race fails startup rather than selecting a new port.
    sockets = []
    try:
        for binding in bindings:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(listener)
            listener.bind(("127.0.0.1", binding.port))
    except OSError:
        raise WorkerConfigurationError("A configured worker port is unavailable; no automatic reassignment") from None
    finally:
        for listener in sockets:
            listener.close()


def stop_child(process) -> None:
    """Terminate only a child process group that this supervisor created."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


class WorkerSupervisor:
    """One startup snapshot; revocations stop workers, additions require restart."""

    def __init__(self, store, environment=None, *, popen=None, stopper=None):
        self.store = store
        self.environment = dict(os.environ if environment is None else environment)
        self._popen = popen or subprocess.Popen
        self._stopper = stopper or stop_child
        self.processes = {}
        self.bindings = {}

    def check(self) -> tuple[WorkerBinding, ...]:
        require_sharing_ack(self.environment)  # Before store or any runtime imports.
        bindings = parse_bindings(self.store.list_worker_bindings())
        for binding in bindings:
            child_environment(Path(self.store.root), binding, self.environment)
            directory = worker_directory(Path(self.store.root), binding)
            for path in (directory.parent, directory, *(directory / name for name in ("runtime", "home", "tmp", "cache", "data"))):
                if path.exists() or path.is_symlink():
                    _private_directory(path)
            if (directory / "runtime" / ".env").exists() or (directory / "runtime" / ".env").is_symlink():
                raise WorkerConfigurationError("Worker runtime directory must not contain a dotenv file")
        return bindings

    def start(self) -> None:
        bindings = self.check()
        if self.processes:
            raise WorkerConfigurationError("Workers are already started")
        ensure_ports_available(bindings)
        try:
            for binding in bindings:
                directory = prepare_directories(Path(self.store.root), binding)
                process = self._popen(
                    [sys.executable, "-m", "docthinker.server.invite_workers", "--worker"],
                    cwd=directory / "runtime",
                    env=child_environment(Path(self.store.root), binding, self.environment),
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                self.processes[binding.user_id] = process
                self.bindings[binding.user_id] = binding
        except BaseException:
            self.close()
            raise

    def reconcile(self) -> None:
        records = self.store.list_worker_bindings()
        active = {item.user_id: item for item in parse_bindings(records)} if records else {}
        for user_id, process in list(self.processes.items()):
            if user_id not in active:
                self._stopper(process)
                del self.processes[user_id]
                continue
            if active[user_id] != self.bindings[user_id]:
                raise WorkerConfigurationError("A running user binding changed; restart the supervisor")
            if process.poll() is not None:
                raise WorkerConfigurationError("A worker exited; inspect configuration before manually restarting")

    def close(self) -> None:
        processes, self.processes = self.processes, {}
        failed = False
        for process in processes.values():
            try:
                self._stopper(process)
            except Exception:  # noqa: BLE001 - still attempt cleanup of every other owned child
                failed = True
        if failed:
            raise WorkerConfigurationError("Could not stop every owned child process; inspect local process status")

    def run(self, stop_event=None) -> None:
        stop_event = stop_event or threading.Event()
        with supervisor_lock(Path(self.store.root)):
            try:
                self.start()
                print(f"Started {len(self.processes)} local invitation workers. Profile: {worker_profile(self.environment)}.", flush=True)
                while self.processes and not stop_event.wait(1):
                    self.reconcile()
            finally:
                self.close()


def worker_main() -> None:
    # Both entrypoints check independently. No create_mobile_app/core import occurs before this.
    require_sharing_ack(os.environ)
    provider_environment(os.environ)
    user_id = os.getenv("DOCTHINKER_INVITE_WORKER_ID", "")
    try:
        port = int(os.environ["DOCTHINKER_INVITE_WORKER_PORT"])
        directory = Path(os.environ["DOCTHINKER_MOBILE_DATA_DIR"])
    except (KeyError, ValueError):
        raise WorkerConfigurationError("Missing isolated worker startup configuration") from None
    if (not _USER_ID.fullmatch(user_id) or directory.name != user_id or directory.parent.name != "workers" or not directory.is_absolute()
            or not 1024 <= port <= 65535 or Path.cwd() != directory / "runtime"
            or not _TOKEN.fullmatch(os.getenv("DOCTHINKER_MOBILE_ACCESS_TOKEN", ""))
            or os.getenv("DOCTHINKER_MOBILE_ENABLED") != "true"):
        raise WorkerConfigurationError("Invalid isolated worker startup configuration")
    _private_directory(directory)
    _private_directory(Path.cwd())
    clean_environment = child_environment(directory.parent.parent, WorkerBinding(
        user_id, port, os.environ["DOCTHINKER_MOBILE_ACCESS_TOKEN"],
    ), os.environ)
    os.environ.clear()
    os.environ.update(clean_environment)
    os.umask(0o077)
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    from .invite_account_lifecycle import worker_lifetime_lock
    from .invite_store import InviteError, InviteStore

    try:
        # The child owns this lock for its whole lifetime, including startup and
        # shutdown. A supervisor crash therefore cannot make a live worker look
        # stopped to account maintenance. Recheck a possibly stale spawn binding.
        with worker_lifetime_lock(directory.parent.parent, user_id):
            InviteStore(directory.parent.parent).authorize_worker(
                user_id, port, os.environ["DOCTHINKER_MOBILE_ACCESS_TOKEN"])
            _serve_worker(port)
    except InviteError as error:
        raise WorkerConfigurationError(str(error)) from None


def _serve_worker(port: int) -> None:
    # python-dotenv >=1.0 is supported by this repository; older versions do not
    # implement PYTHON_DOTENV_DISABLED. Block module-relative .env loads as well.
    import dotenv
    import dotenv.main

    def no_dotenv(*_args, **_kwargs):
        return False

    dotenv.load_dotenv = no_dotenv
    dotenv.main.load_dotenv = no_dotenv
    import uvicorn

    from .mobile_app import create_mobile_app

    uvicorn.run(create_mobile_app(), host="127.0.0.1", port=port,
                access_log=False, log_level="warning", proxy_headers=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run at most five isolated invitation workers; never reads .env")
    parser.add_argument("--check", action="store_true", help="Validate explicit configuration without starting workers or model clients")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        require_sharing_ack(os.environ)
        if args.worker:
            if args.check:
                raise WorkerConfigurationError("Worker and check modes cannot be combined")
            worker_main()
            return 0
        from .invite_store import InviteStore

        store = InviteStore.from_environment(require_initialized=True)
        supervisor = WorkerSupervisor(store)
        if args.check:
            bindings = supervisor.check()
            print(f"Validated {len(bindings)} fixed worker bindings. Profile: {worker_profile(os.environ)}. No workers or provider requests started.")
            return 0
        stop_event = threading.Event()
        previous = {sig: signal.signal(sig, lambda *_: stop_event.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            supervisor.run(stop_event)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        return 0
    except WorkerConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001 - CLI boundary must not expose provider/store exception details
        # Store/provider errors can contain secrets or absolute storage paths.
        print("Worker startup failed; check local configuration. No automatic restart.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
