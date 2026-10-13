"""Shared parser containment: private temporary locks, no model credentials."""
from __future__ import annotations

import asyncio
import os
import stat
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from docthinker.server.research_parser_gate import (
    LOCK_ENV, LOCK_FILENAME, ParserSlotUnavailable, offline_parser_slot,
)
from docthinker.server import research_service
from docthinker.server import research_parser_gate as gate_module

fcntl = pytest.importorskip("fcntl")


@pytest.fixture
def deployment(tmp_path):
    root = (tmp_path / "registry").resolve()
    root.mkdir(mode=0o700)
    workers = root / "workers"
    workers.mkdir(mode=0o700)
    environments = []
    for identifier in ("a" * 32, "b" * 32):
        directory = workers / identifier
        directory.mkdir(mode=0o700)
        environments.append({"DOCTHINKER_INVITE_WORKER_ID": identifier,
                             "DOCTHINKER_MOBILE_DATA_DIR": str(directory),
                             LOCK_ENV: str(workers / LOCK_FILENAME)})
    return SimpleNamespace(root=root, workers=workers, environments=environments,
                           lock=workers / LOCK_FILENAME)


async def test_different_workers_share_exactly_one_cancellable_parser_slot(deployment):
    first, second = deployment.environments
    entered = asyncio.Event()

    async def other_worker():
        async with offline_parser_slot(second):
            entered.set()

    async with offline_parser_slot(first):
        task = asyncio.create_task(other_worker())
        await asyncio.sleep(0.12)
        assert not entered.is_set()
        assert stat.S_IMODE(deployment.lock.stat().st_mode) == 0o600
        assert deployment.lock.read_bytes() == b""
    await asyncio.wait_for(task, 1)
    assert entered.is_set()


async def test_cancelled_waiter_does_not_unlock_another_worker_or_leave_a_slot(deployment):
    entered = []

    async def waiter():
        async with offline_parser_slot(deployment.environments[1]):
            entered.append(True)

    async with offline_parser_slot(deployment.environments[0]):
        waiting = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert not entered
        descriptor = os.open(deployment.lock, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)
    await asyncio.wait_for(waiter(), 1)
    assert entered == [True]


async def test_local_wait_is_finite_and_never_enters_a_busy_parser(deployment, monkeypatch):
    monkeypatch.setattr(gate_module, "PARSER_WAIT_SECONDS", 0.04)
    async with offline_parser_slot(deployment.environments[0]):
        with pytest.raises(ParserSlotUnavailable) as error:
            async with offline_parser_slot(deployment.environments[1]):
                pytest.fail("Timed-out waiter entered another worker's slot")
        assert str(error.value) == "Offline parser coordination is unavailable"
    assert not gate_module._LOCAL_GATES.get(asyncio.get_running_loop())
    async with offline_parser_slot(deployment.environments[1]):
        pass


async def test_os_lock_wait_has_the_same_finite_deadline(deployment, monkeypatch):
    monkeypatch.setattr(gate_module, "PARSER_WAIT_SECONDS", 0.04)
    descriptor = gate_module._lock_descriptor(deployment.environments[0])
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(ParserSlotUnavailable):
            async with offline_parser_slot(deployment.environments[1]):
                pytest.fail("Acquired an externally held parser lock")
        with pytest.raises(BlockingIOError):
            second = os.open(deployment.lock, os.O_RDWR)
            try:
                fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(second)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert not gate_module._LOCAL_GATES.get(asyncio.get_running_loop())
    async with offline_parser_slot(deployment.environments[1]):
        pass


async def test_local_queue_is_fifo_bounded_and_cancelled_entry_does_not_block_it(deployment):
    order = []
    async def enter(index):
        async with offline_parser_slot(deployment.environments[1]):
            order.append(index)
            await asyncio.sleep(0)
    async with offline_parser_slot(deployment.environments[0]):
        tasks = []
        for index in range(gate_module.MAX_LOCAL_PARSER_WAITERS):
            tasks.append(asyncio.create_task(enter(index)))
            await asyncio.sleep(0)
        with pytest.raises(ParserSlotUnavailable):
            await enter(99)
        assert not order
        tasks[1].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[1]
        tasks.append(asyncio.create_task(enter(4)))
        await asyncio.sleep(0)
    await asyncio.wait_for(asyncio.gather(*(task for task in tasks if not task.cancelled())), 1)
    assert order == [0, 2, 3, 4]
    assert not gate_module._LOCAL_GATES.get(asyncio.get_running_loop())


async def test_os_process_cannot_acquire_slot_held_by_worker(deployment):
    code = (
        "import fcntl,os,sys; f=os.open(sys.argv[1],os.O_RDWR); "
        "\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); print('ACQUIRED')"
        "\nexcept BlockingIOError: print('BLOCKED')"
        "\nfinally: os.close(f)"
    )
    async with offline_parser_slot(deployment.environments[0]):
        child = await asyncio.create_subprocess_exec(sys.executable, "-c", code, str(deployment.lock),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                    env={"PATH": os.defpath})
        stdout, _ = await asyncio.wait_for(child.communicate(), 5)
        assert child.returncode == 0 and stdout.strip() == b"BLOCKED"


async def test_direct_private_mode_ignores_shared_lock_environment(deployment, monkeypatch):
    opened = Mock(side_effect=AssertionError("Private mode must not touch deployment lock"))
    monkeypatch.setattr(os, "open", opened)
    async with offline_parser_slot({LOCK_ENV: "/unsafe/supplied/path"}):
        pass
    opened.assert_not_called()
    assert not deployment.lock.exists()


@pytest.mark.parametrize("override", [
    {LOCK_ENV: "/tmp/not-the-owned-lock"}, {LOCK_ENV: "relative.lock"}, {LOCK_ENV: ""},
    {"DOCTHINKER_INVITE_WORKER_ID": "../other"}, {"DOCTHINKER_MOBILE_DATA_DIR": "relative"},
])
async def test_worker_rejects_missing_or_redirected_paths_without_echoing_them(deployment, override):
    environment = deployment.environments[0] | override
    with pytest.raises(ParserSlotUnavailable) as error:
        async with offline_parser_slot(environment):
            pytest.fail("Unsafe parser lock accepted")
    assert str(error.value) == "Offline parser coordination is unavailable"
    assert not deployment.lock.exists()


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "public", "content", "directory"])
async def test_lock_rejects_unsafe_inode_without_modifying_target(deployment, kind):
    target = deployment.root / "sensitive-placeholder"
    target.write_bytes(b"untouched")
    target.chmod(0o600)
    if kind == "symlink":
        deployment.lock.symlink_to(target)
    elif kind == "hardlink":
        os.link(target, deployment.lock)
    elif kind == "directory":
        deployment.lock.mkdir(mode=0o700)
    else:
        deployment.lock.touch(mode=0o600)
        if kind == "public":
            deployment.lock.chmod(0o644)
        else:
            deployment.lock.write_bytes(b"not a coordination inode")
    with pytest.raises(ParserSlotUnavailable):
        async with offline_parser_slot(deployment.environments[0]):
            pytest.fail("Unsafe inode accepted")
    assert target.read_bytes() == b"untouched"


async def test_shared_parent_must_be_private_and_canonical(deployment):
    deployment.workers.chmod(0o755)
    with pytest.raises(ParserSlotUnavailable):
        async with offline_parser_slot(deployment.environments[0]):
            pytest.fail("Public lock directory accepted")
    assert not deployment.lock.exists()


async def test_extract_offline_cannot_spawn_before_shared_slot_is_available(deployment, monkeypatch):
    for key, value in deployment.environments[0].items():
        monkeypatch.setenv(key, value)
    called = AsyncMock(return_value={"chunks": [{"content": "synthetic source"}]})
    monkeypatch.setattr(research_service, "_extract_offline_unlocked", called)
    async with offline_parser_slot(deployment.environments[1]):
        extraction = asyncio.create_task(research_service.extract_offline(b"synthetic", "paper.txt"))
        await asyncio.sleep(0.10)
        called.assert_not_awaited()
        extraction.cancel()
        with pytest.raises(asyncio.CancelledError):
            await extraction
    assert (await research_service.extract_offline(b"synthetic", "paper.txt"))["chunks"]
    called.assert_awaited_once()


async def test_extract_failure_releases_slot_and_sanitizes_bad_coordination(deployment, monkeypatch):
    for key, value in deployment.environments[0].items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv(LOCK_ENV, "/private/never-return-this-path")
    spawn = AsyncMock(side_effect=AssertionError("Invalid lock must not spawn"))
    monkeypatch.setattr(research_service, "_extract_offline_unlocked", spawn)
    assert await research_service.extract_offline(b"synthetic", "paper.txt") == {"error_code": "parser_capacity_unavailable"}
    spawn.assert_not_awaited()


async def test_shutdown_during_spawn_reaps_child_before_releasing_global_slot(deployment, monkeypatch):
    for key, value in deployment.environments[0].items():
        monkeypatch.setenv(key, value)
    spawning, finish_spawn = asyncio.Event(), asyncio.Event()
    child = SimpleNamespace(returncode=None, kill=Mock(), wait=AsyncMock())

    def kill():
        child.returncode = -9

    child.kill.side_effect = kill

    async def spawn(*args, **kwargs):
        spawning.set()
        await finish_spawn.wait()
        return child

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    extraction = asyncio.create_task(research_service.extract_offline(b"synthetic", "paper.txt"))
    await asyncio.wait_for(spawning.wait(), 1)
    extraction.cancel()
    entered = asyncio.Event()

    async def next_worker():
        async with offline_parser_slot(deployment.environments[1]):
            assert child.kill.called and child.wait.await_count == 1
            entered.set()

    waiting = asyncio.create_task(next_worker())
    await asyncio.sleep(0.1)
    assert not entered.is_set()
    finish_spawn.set()
    with pytest.raises(asyncio.CancelledError):
        await extraction
    await asyncio.wait_for(waiting, 1)
    assert entered.is_set()


def test_launcher_overwrites_arbitrary_parser_path_with_shared_deployment_inode(tmp_path):
    from docthinker.server.invite_workers import (
        SHARING_ACK_ENV, SHARING_ACK_VALUE, WorkerBinding, child_environment,
    )
    root = tmp_path.resolve()
    root.chmod(0o700)
    environment = {SHARING_ACK_ENV: SHARING_ACK_VALUE, "LLM_BINDING_API_KEY": "synthetic-model-key",
                   "LLM_VLM_HOST": "http://127.0.0.1:12345/v1", "LLM_MODEL": "synthetic-model",
                   "DOCTHINKER_RESEARCH_ENABLED": "true", LOCK_ENV: "/do/not/use-this.lock"}
    paths = []
    for index in ("a", "b"):
        binding = WorkerBinding(index * 32, 8101 if index == "a" else 8102, "synthetic-worker-token-" + index * 32, True)
        paths.append(child_environment(root, binding, environment)[LOCK_ENV])
    assert paths == [str(root / "workers" / LOCK_FILENAME)] * 2
