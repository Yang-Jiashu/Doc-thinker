"""
Unified LLM call tracing middleware.

Every LLM invocation goes through LLMTrace, which records the full
prompt, response, timing, and metadata into a per-session JSONL file
at ``data/_traces/{session_id}_traces.jsonl``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_log = logging.getLogger("docthinker.llm_trace")


class LLMTrace:
    """Wrap an LLM call with automatic tracing."""

    __slots__ = ("stage", "sub_stage", "session_id", "trace_dir")

    def __init__(
        self,
        *,
        stage: str,
        sub_stage: str = "",
        session_id: str = "",
        trace_dir: Optional[Path] = None,
    ) -> None:
        self.stage = stage
        self.sub_stage = sub_stage
        self.session_id = session_id
        self.trace_dir = trace_dir or Path("data/_traces")

    async def call(
        self,
        llm_func: Callable[..., Any],
        prompt: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Invoke *llm_func* with *prompt*, record the trace, and return the response."""
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        call_id = hashlib.md5(
            f"{time.time():.6f}{prompt[:80]}".encode()
        ).hexdigest()[:12]
        t0 = time.perf_counter()

        _log.info(
            "[%s/%s] call_id=%s | prompt_chars=%d",
            self.stage,
            self.sub_stage,
            call_id,
            len(prompt),
        )

        status = "ok"
        response = ""
        try:
            if asyncio.iscoroutinefunction(llm_func):
                response = await llm_func(prompt)
            else:
                response = llm_func(prompt)
            response = str(response or "")
        except Exception as exc:
            status = f"error: {exc}"
            raise
        finally:
            elapsed = time.perf_counter() - t0
            record: Dict[str, Any] = {
                "call_id": call_id,
                "stage": self.stage,
                "sub_stage": self.sub_stage,
                "session_id": self.session_id,
                "timestamp": time.time(),
                "elapsed_s": round(elapsed, 3),
                "status": status,
                "prompt_chars": len(prompt),
                "response_chars": len(response),
                "prompt_preview": prompt[:500],
                "response_preview": response[:500],
                "prompt_full": prompt,
                "response_full": response,
                "metadata": metadata or {},
            }
            trace_file = self.trace_dir / f"{self.session_id or 'global'}_traces.jsonl"
            try:
                with open(trace_file, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as io_err:
                _log.warning("trace write failed: %s", io_err)

            _log.info(
                "[%s/%s] call_id=%s | %.2fs | status=%s | resp_chars=%d",
                self.stage,
                self.sub_stage,
                call_id,
                elapsed,
                status,
                len(response),
            )


def list_traces(
    session_id: str,
    *,
    stage: Optional[str] = None,
    limit: int = 50,
    trace_dir: Optional[Path] = None,
) -> list[Dict[str, Any]]:
    """Return recent trace records for a session (preview only, no full prompt/response)."""
    base = trace_dir or Path("data/_traces")
    trace_file = base / f"{session_id}_traces.jsonl"
    if not trace_file.exists():
        return []
    traces: list[Dict[str, Any]] = []
    for line in trace_file.read_text(encoding="utf-8").strip().split("\n"):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if stage and rec.get("stage") != stage:
            continue
        traces.append(
            {
                "call_id": rec.get("call_id"),
                "stage": rec.get("stage"),
                "sub_stage": rec.get("sub_stage"),
                "elapsed_s": rec.get("elapsed_s"),
                "status": rec.get("status"),
                "prompt_chars": rec.get("prompt_chars"),
                "response_chars": rec.get("response_chars"),
                "prompt_preview": rec.get("prompt_preview"),
                "response_preview": rec.get("response_preview"),
                "timestamp": rec.get("timestamp"),
            }
        )
    traces.sort(key=lambda x: -(x.get("timestamp") or 0))
    return traces[:limit]


def get_trace_detail(
    call_id: str,
    session_id: str,
    *,
    trace_dir: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Return the full trace record (including complete prompt/response) for *call_id*."""
    base = trace_dir or Path("data/_traces")
    trace_file = base / f"{session_id}_traces.jsonl"
    if not trace_file.exists():
        return None
    for line in trace_file.read_text(encoding="utf-8").strip().split("\n"):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("call_id") == call_id:
            return rec
    return None
