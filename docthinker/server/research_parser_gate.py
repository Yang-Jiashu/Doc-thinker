"""One offline parser slot per invitation deployment, with cancellable waits.

Worker launchers generate the lock path from the owned deployment layout. No HTTP
field supplies it. A crash releases the process-held flock; the file is an opaque
owner-only coordination inode, not a task queue or storage for document content.
Direct private/mobile development keeps its existing independent parser behavior.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import errno
import os
from pathlib import Path
import re
import stat
import weakref

LOCK_ENV = "DOCTHINKER_RESEARCH_PARSER_LOCK_PATH"
LOCK_FILENAME = "research-parser.lock"
PARSER_WAIT_SECONDS = 120.0
PARSER_POLL_SECONDS = 0.05
MAX_LOCAL_PARSER_WAITERS = 4

# A single service normally submits one document at a time. Keep accidental
# parallel submissions bounded too, without making one flock contender per
# queued document. asyncio.Lock grants waiting tasks their local FIFO turn.
# The OS lock below still owns the cross-process exclusion; local FIFO is not a
# promise of strict FIFO ordering among different worker processes.
_LOCAL_GATES = weakref.WeakKeyDictionary()


class _LocalGate:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.users = 0


@asynccontextmanager
async def _local_turn(key: str, deadline: float):
    loop = asyncio.get_running_loop()
    gates = _LOCAL_GATES.setdefault(loop, {})
    gate = gates.setdefault(key, _LocalGate())
    if gate.users >= MAX_LOCAL_PARSER_WAITERS + 1:
        raise ParserSlotUnavailable("Offline parser coordination is unavailable")
    gate.users += 1
    acquired = False
    try:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise ParserSlotUnavailable("Offline parser coordination is unavailable")
        try:
            await asyncio.wait_for(gate.lock.acquire(), remaining)
            acquired = True
        except asyncio.TimeoutError:
            raise ParserSlotUnavailable("Offline parser coordination is unavailable") from None
        yield
    finally:
        if acquired:
            gate.lock.release()
        gate.users -= 1
        if not gate.users and gates.get(key) is gate:
            gates.pop(key, None)
        if not gates:
            _LOCAL_GATES.pop(loop, None)


class ParserSlotUnavailable(ValueError):
    pass


def _private_directory(path: Path) -> None:
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or info.st_mode & 0o077 or path.resolve(strict=True) != path):
        raise ParserSlotUnavailable("Offline parser coordination is unavailable")


def _lock_descriptor(environment) -> int:
    descriptor = None
    try:
        identifier = environment.get("DOCTHINKER_INVITE_WORKER_ID", "")
        directory = Path(environment.get("DOCTHINKER_MOBILE_DATA_DIR", ""))
        if (not re.fullmatch(r"[0-9a-f]{32}", identifier) or not directory.is_absolute()
                or directory.name != identifier or directory.parent.name != "workers"):
            raise ValueError
        expected = directory.parent / LOCK_FILENAME
        if environment.get(LOCK_ENV) != str(expected):
            raise ValueError
        for parent in (directory, directory.parent, directory.parent.parent):
            _private_directory(parent)
        descriptor = os.open(expected, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        info = os.fstat(descriptor)
        visible = expected.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != 0
                or info.st_uid != os.getuid() or info.st_mode & 0o077
                or (info.st_dev, info.st_ino) != (visible.st_dev, visible.st_ino)):
            raise ValueError
        return descriptor
    except (OSError, ValueError, TypeError):
        if descriptor is not None:
            os.close(descriptor)
        raise ParserSlotUnavailable("Offline parser coordination is unavailable") from None


@asynccontextmanager
async def offline_parser_slot(environment=None):
    """Hold one deployment-wide slot; finite waits never block the event loop.

    The 120-second budget includes the local queue and cross-process wait, not
    extraction itself. Extraction already has its separate child-process limit.
    There are at most four waiting local submissions plus one flock contender.
    A timeout starts no parser/model and is not automatically resubmitted.
    """
    environment = os.environ if environment is None else environment
    if not environment.get("DOCTHINKER_INVITE_WORKER_ID", "").strip():
        yield
        return
    try:
        import fcntl
    except ImportError:
        raise ParserSlotUnavailable("Offline parser coordination is unavailable") from None
    loop = asyncio.get_running_loop()
    deadline = loop.time() + PARSER_WAIT_SECONDS
    # Validate the deployment-owned path before allowing it to select a local
    # scheduling key. Close it while queued so waiters do not retain file FDs.
    descriptor = _lock_descriptor(environment)
    os.close(descriptor)
    async with _local_turn(environment[LOCK_ENV], deadline):
        descriptor = _lock_descriptor(environment)
        acquired = False
        try:
            while not acquired:
                if loop.time() >= deadline:
                    raise ParserSlotUnavailable("Offline parser coordination is unavailable")
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except OSError as error:
                    if error.errno not in {errno.EAGAIN, errno.EWOULDBLOCK}:
                        raise ParserSlotUnavailable("Offline parser coordination is unavailable") from None
                    # Waiting starts no child and reads no provider credentials.
                    await asyncio.sleep(min(PARSER_POLL_SECONDS, max(0, deadline - loop.time())))
            yield
        finally:
            try:
                if acquired:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
