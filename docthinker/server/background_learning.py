"""Process-local, event-loop-owned scheduling for post-upload learning.

This is not a durable queue or a lock against document ingestion. Each session
has one worker, and uploads arriving during a run request at most one rerun.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

_log = logging.getLogger("docthinker.background_learning")


@dataclass
class _SessionWork:
    task: asyncio.Task | None = None
    pending: bool = True
    running: bool = False
    cancelling: bool = False


class SessionLearningRunner:
    """Coalesce session jobs without blocking uploads or spawning duplicate work.

    Call all methods from the same asyncio event loop. Queued requests for a
    session collapse into its next run; requests during a run set one pending
    rerun. Different sessions share a bounded semaphore. Cancellation drops
    that session's pending work; shutdown rejects all subsequent submissions.
    """

    def __init__(
        self,
        run_session: Callable[[str], Awaitable[None]],
        *,
        max_concurrent_sessions: int = 2,
    ) -> None:
        if type(max_concurrent_sessions) is not int or max_concurrent_sessions < 1:
            raise ValueError("max_concurrent_sessions must be a positive integer")
        self._run_session = run_session
        self._semaphore = asyncio.Semaphore(max_concurrent_sessions)
        self._sessions: dict[str, _SessionWork] = {}
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def snapshot(self) -> dict[str, dict[str, bool]]:
        """Small read-only scheduling status, with no document or prompt data."""
        return {
            sid: {"running": work.running, "pending": work.pending}
            for sid, work in self._sessions.items()
        }

    def submit(self, session_id: str) -> bool:
        """Schedule/coalesce a request synchronously; False means it was rejected."""
        if self._closed or not session_id:
            return False
        work = self._sessions.get(session_id)
        if work is not None and work.task is not None and not work.task.done():
            if work.cancelling:
                return False
            work.pending = True
            return True

        work = _SessionWork()
        task = asyncio.create_task(
            self._run(session_id, work), name=f"post-upload-learning:{session_id}"
        )
        work.task = task
        self._sessions[session_id] = work
        task.add_done_callback(lambda done: self._finished(session_id, work, done))
        return True

    async def _run(self, session_id: str, work: _SessionWork) -> None:
        while work.pending and not self._closed and not work.cancelling:
            async with self._semaphore:
                if self._closed or work.cancelling:
                    return
                # Requests coalesced while waiting already belong to this run.
                work.pending = False
                work.running = True
                try:
                    await self._run_session(session_id)
                except Exception:  # noqa: BLE001 - isolate jobs; cancellation still propagates
                    _log.exception(
                        "Post-upload learning failed for session %s", session_id
                    )
                finally:
                    work.running = False
            # Release the shared slot before a rerun so other sessions get a turn.

    def _finished(
        self, session_id: str, work: _SessionWork, task: asyncio.Task
    ) -> None:
        if self._sessions.get(session_id) is work:
            self._sessions.pop(session_id, None)
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                _log.error(
                    "Post-upload learning task crashed for session %s",
                    session_id,
                    exc_info=(type(error), error, error.__traceback__),
                )

    async def wait_idle(self) -> None:
        """Wait for current work and coalesced reruns (not used on upload paths)."""
        while self._sessions:
            tasks = [
                work.task for work in self._sessions.values() if work.task is not None
            ]
            await asyncio.gather(
                *(asyncio.shield(task) for task in tasks), return_exceptions=True
            )

    async def cancel(self, session_id: str) -> bool:
        """Cancel one session and its pending rerun without affecting other sessions."""
        work = self._sessions.get(session_id)
        if work is None or work.task is None:
            return False
        work.cancelling = True
        work.pending = False
        work.task.cancel()
        await asyncio.gather(work.task, return_exceptions=True)
        return True

    async def shutdown(self) -> None:
        """Cancel and observe every task before the host closes shared resources."""
        self._closed = True
        tasks = []
        for work in self._sessions.values():
            work.cancelling = True
            work.pending = False
            if work.task is not None:
                work.task.cancel()
                tasks.append(work.task)
        await asyncio.gather(*tasks, return_exceptions=True)
