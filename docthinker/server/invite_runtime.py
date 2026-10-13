"""Single-process supervisor for the deployable, invitation-only H5 service.

The public-facing shell and gateway share one origin. Model workers remain on
loopback, with independent storage and credentials. Accounts and the explicit
model-sharing acknowledgement must already exist; this entrypoint never reads
dotenv files, provisions users, opens tunnels or restarts failed processes.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .invite_workers import (
    WORKER_PROFILE,
    WorkerConfigurationError,
    WorkerSupervisor,
    _private_directory,
    parse_bindings,
    require_sharing_ack,
    stop_child,
    supervisor_lock,
)

_ROOT = Path(__file__).resolve().parents[2]
STARTUP_TIMEOUT_SECONDS = 120
HEALTHCHECK_DEADLINE_SECONDS = 8
HEALTHCHECK_BODY_LIMIT = 64 * 1024


def explicit_runtime_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Resolve only an explicitly selected Docker secret, never a dotenv file."""
    require_sharing_ack(environment)
    result = dict(environment)
    if "LLM_BINDING_API_KEY_FILE" not in result:
        return result
    if "LLM_BINDING_API_KEY" in result:
        raise WorkerConfigurationError("Configure only one explicit model key source: environment or secret file")
    value = result.pop("LLM_BINDING_API_KEY_FILE")
    descriptor = None
    try:
        path = Path(value)
        if not path.is_absolute() or path != path.resolve(strict=True) or path.is_symlink():
            raise ValueError
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 4096:
            raise ValueError
        content = os.read(descriptor, 4097)
        key = content.decode("utf-8").strip()
        if len(content) > 4096 or not key or any(ord(character) < 32 for character in key):
            raise ValueError
    except (OSError, ValueError, TypeError):
        raise WorkerConfigurationError("Explicit model secret file must be an absolute real regular file containing at most 4096 bytes") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    result["LLM_BINDING_API_KEY"] = key
    return result


@dataclass(frozen=True)
class RuntimeConfiguration:
    host: str
    port: int
    origin: str
    web_dir: Path

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> "RuntimeConfiguration":
        from .invite_app import validated_origin
        from .invite_site import InviteShell

        host = environment.get("DOCTHINKER_INVITE_BIND_HOST", "127.0.0.1")
        if host not in {"127.0.0.1", "0.0.0.0"}:
            raise WorkerConfigurationError("Gateway bind host must be 127.0.0.1 or 0.0.0.0")
        value = environment.get("DOCTHINKER_INVITE_PORT", "8080")
        if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
            raise WorkerConfigurationError("Gateway port must be an explicit integer between 1024 and 65535")
        port = int(value)
        if not 1024 <= port <= 65535:
            raise WorkerConfigurationError("Gateway port must be an explicit integer between 1024 and 65535")
        origin, _ = validated_origin(environment.get("DOCTHINKER_INVITE_ORIGIN", ""))
        directory = environment.get("DOCTHINKER_INVITE_WEB_DIR", "").strip()
        if not directory:
            raise WorkerConfigurationError("Set an explicit built DOCTHINKER_INVITE_WEB_DIR")
        shell = InviteShell(Path(directory))
        return cls(host, port, origin, Path(shell.directory))


def gateway_environment(root: Path, configuration: RuntimeConfiguration,
                        environment: Mapping[str, str]) -> dict[str, str]:
    """The gateway never receives provider credentials or arbitrary legacy paths."""
    from .invite_store import DEFAULT_REGISTRY_BYTES, registry_storage_bytes
    registry_bytes = registry_storage_bytes(environment.get(
        "DOCTHINKER_INVITE_REGISTRY_BYTES", str(DEFAULT_REGISTRY_BYTES)))
    result = {key: environment[key] for key in ("PATH", "LANG", "LC_ALL") if key in environment}
    directory = root / "gateway"
    result.update({
        "PYTHONPATH": str(_ROOT), "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
        "PYTHON_DOTENV_DISABLED": "1", "HOME": str(directory / "home"),
        "TMPDIR": str(directory / "tmp"), "XDG_CACHE_HOME": str(directory / "cache"),
        "DOCTHINKER_INVITE_GATEWAY_CHILD": "1",
        "DOCTHINKER_INVITE_DATA_DIR": str(root),
        "DOCTHINKER_INVITE_REGISTRY_BYTES": str(registry_bytes),
        "DOCTHINKER_INVITE_WEB_DIR": str(configuration.web_dir),
        "DOCTHINKER_INVITE_ORIGIN": configuration.origin,
        "DOCTHINKER_INVITE_BIND_HOST": configuration.host,
        "DOCTHINKER_INVITE_PORT": str(configuration.port),
        "DOCTHINKER_INVITE_UPLOADS_ENABLED": "false",
        "DOCTHINKER_RESEARCH_ENABLED": "true" if environment.get("DOCTHINKER_RESEARCH_ENABLED", "").lower() in {"true", "1"} else "false",
    })
    return result


def prepare_gateway_directory(root: Path) -> Path:
    _private_directory(root)
    directory = root / "gateway"
    for path in (directory, *(directory / name for name in ("runtime", "home", "tmp", "cache"))):
        _private_directory(path, create=True)
    runtime = directory / "runtime"
    if (runtime / ".env").exists() or (runtime / ".env").is_symlink():
        raise WorkerConfigurationError("Gateway runtime directory must not contain a dotenv file")
    return runtime


def ensure_gateway_port_available(configuration: RuntimeConfiguration) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind((configuration.host, configuration.port))
    except OSError:
        raise WorkerConfigurationError("The configured gateway port is unavailable") from None


def port_ready(port: int) -> bool:
    """Only probe the already-owned local listener; no HTTP/provider request."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


class InviteRuntime:
    """Keep a single gateway and its fixed user workers in one failure domain."""

    def __init__(self, store, configuration: RuntimeConfiguration, environment=None,
                 *, workers=None, popen=None, stopper=None):
        self.store = store
        self.configuration = configuration
        self.environment = explicit_runtime_environment(os.environ if environment is None else environment)
        self.workers = workers or WorkerSupervisor(store, self.environment)
        self._popen = popen or subprocess.Popen
        self._stopper = stopper or stop_child
        self.gateway = None

    def check(self):
        require_sharing_ack(self.environment)
        bindings = self.workers.check()
        if any(binding.port == self.configuration.port for binding in bindings):
            raise WorkerConfigurationError("Gateway and worker ports must be distinct")
        directory = Path(self.store.root) / "gateway"
        for path in (directory, *(directory / name for name in ("runtime", "home", "tmp", "cache"))):
            if path.exists() or path.is_symlink():
                _private_directory(path)
        if (directory / "runtime" / ".env").exists() or (directory / "runtime" / ".env").is_symlink():
            raise WorkerConfigurationError("Gateway runtime directory must not contain a dotenv file")
        return bindings

    def start(self) -> None:
        if self.gateway is not None or self.workers.processes:
            raise WorkerConfigurationError("The invitation runtime is already started")
        self.check()
        ensure_gateway_port_available(self.configuration)
        runtime_directory = prepare_gateway_directory(Path(self.store.root))
        try:
            self.workers.start()
            self.gateway = self._popen(
                [sys.executable, "-m", "docthinker.server.invite_runtime", "--gateway"],
                cwd=runtime_directory,
                env=gateway_environment(Path(self.store.root), self.configuration, self.environment),
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except BaseException:
            self.close()
            raise

    def reconcile(self) -> None:
        if self.gateway is None or self.gateway.poll() is not None:
            raise WorkerConfigurationError("The invitation gateway exited; inspect configuration before restarting")
        self.workers.reconcile()

    def wait_ready(self, stop_event) -> bool:
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while not stop_event.is_set():
            self.reconcile()
            if not self.workers.processes:
                return False  # All users were revoked during startup.
            ports = [self.configuration.port, *(self.workers.bindings[key].port for key in self.workers.processes)]
            if all(port_ready(port) for port in ports):
                return True
            if time.monotonic() >= deadline:
                raise WorkerConfigurationError("Invitation runtime startup timed out; no automatic restart")
            stop_event.wait(0.25)
        return False

    def close(self) -> None:
        # Stop accepting new requests before stopping the workers they target.
        gateway, self.gateway = self.gateway, None
        failed = False
        if gateway is not None:
            try:
                self._stopper(gateway)
            except Exception:
                failed = True
        try:
            self.workers.close()
        except Exception:
            failed = True
        if failed:
            raise WorkerConfigurationError("Could not stop every owned invitation process; inspect local process status")

    def run(self, stop_event=None) -> None:
        stop_event = stop_event or threading.Event()
        with supervisor_lock(Path(self.store.root)):
            try:
                self.start()
                if not self.wait_ready(stop_event):
                    return
                print(f"Invitation runtime ready with {len(self.workers.processes)} isolated users. Profile: {WORKER_PROFILE}.", flush=True)
                while not stop_event.wait(1):
                    self.reconcile()
                    if not self.workers.processes:
                        break
            finally:
                self.close()


def gateway_main() -> None:
    """Internal gateway child; no core/model imports and no worker credentials."""
    from .invite_store import InviteStore

    if os.getenv("DOCTHINKER_INVITE_GATEWAY_CHILD") != "1":
        raise WorkerConfigurationError("Gateway child requires supervised startup configuration")
    store = InviteStore.from_environment(require_initialized=True)
    configuration = RuntimeConfiguration.from_environment(os.environ)
    runtime = Path(store.root) / "gateway" / "runtime"
    _private_directory(runtime)
    if Path.cwd() != runtime or (runtime / ".env").exists() or (runtime / ".env").is_symlink():
        raise WorkerConfigurationError("Gateway child requires its private dotenv-free runtime directory")
    environment = gateway_environment(Path(store.root), configuration, os.environ)
    os.environ.clear()
    os.environ.update(environment)
    os.umask(0o077)
    import uvicorn

    from .invite_site import create_invite_site

    uvicorn.run(create_invite_site(), host=configuration.host, port=configuration.port,
                workers=1, access_log=False, log_level="warning", proxy_headers=False)


def healthcheck(environment=None, *, transport=None) -> bool:
    """Read-only local liveness checks; never resolve the model secret or call it."""
    import httpx

    from .invite_app import validated_origin
    from .invite_store import InviteStore

    environment = os.environ if environment is None else environment
    configuration = RuntimeConfiguration.from_environment(environment)
    store = InviteStore(environment.get("DOCTHINKER_INVITE_DATA_DIR", ""))
    bindings = parse_bindings(store.list_worker_bindings())
    _, host = validated_origin(configuration.origin)
    deadline = time.monotonic() + HEALTHCHECK_DEADLINE_SECONDS

    def get(client, url, headers):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError
        request_deadline = min(deadline, time.monotonic() + 1)
        # Reserve time for connect/pool/write/read instead of letting each
        # phase consume the entire remaining overall deadline.
        with client.stream("GET", url, headers=headers, timeout=min(1, remaining / 4)) as response:
            if response.status_code != 200 or time.monotonic() >= request_deadline:
                raise ValueError
            chunks, size = [], 0
            for chunk in response.iter_raw():
                size += len(chunk)
                if size > HEALTHCHECK_BODY_LIMIT or time.monotonic() >= request_deadline:
                    raise ValueError
                chunks.append(chunk)
            return b"".join(chunks), response.headers.get("content-type", "")

    # At most six loopback requests, each with one-second phase timeouts and
    # explicit body/deadline bounds. No proxy, redirect, retry or business route.
    with httpx.Client(transport=transport, timeout=1, trust_env=False, follow_redirects=False) as client:
        body, content_type = get(client, f"http://127.0.0.1:{configuration.port}/index.html", {"Host": host})
        if not body or not content_type.startswith("text/html"):
            return False
        for binding in bindings:
            body, _ = get(client, f"http://127.0.0.1:{binding.port}/api/v1/mobile/capabilities",
                          {"Authorization": "Bearer " + binding.token})
            capabilities = json.loads(body)
            research = environment.get("DOCTHINKER_RESEARCH_ENABLED", "").lower() in {"true", "1"}
            if (not isinstance(capabilities, dict) or capabilities.get("chat_ready") is not True
                    or capabilities.get("uploads") is not research
                    or (research and capabilities.get("research_core") is not True)):
                return False
    return time.monotonic() < deadline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the invitation-only H5 gateway and isolated workers; never reads .env")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="Validate explicit deployment configuration without starting servers or models")
    modes.add_argument("--healthcheck", action="store_true", help="Check local gateway and worker readiness without reading the provider secret")
    modes.add_argument("--gateway", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.healthcheck:
        try:
            if healthcheck():
                print("Invitation runtime is healthy.")
                return 0
        except Exception:
            pass
        print("Invitation runtime is unavailable.", file=sys.stderr)
        return 1
    try:
        if args.gateway:
            gateway_main()
            return 0
        require_sharing_ack(os.environ)  # Before store access or gateway configuration imports.
        from .invite_store import InviteStore

        store = InviteStore.from_environment(require_initialized=True)
        configuration = RuntimeConfiguration.from_environment(os.environ)
        runtime = InviteRuntime(store, configuration)
        if args.check:
            bindings = runtime.check()
            print(f"Validated invitation runtime for {len(bindings)} fixed users. Profile: {WORKER_PROFILE}. No servers or provider requests started.")
            return 0
        stop_event = threading.Event()
        previous = {sig: signal.signal(sig, lambda *_: stop_event.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            runtime.run(stop_event)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        return 0
    except WorkerConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2
    except Exception:
        print("Invitation runtime failed; check explicit deployment configuration. No credential details or automatic restart.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
