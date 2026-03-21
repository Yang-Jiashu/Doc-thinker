"""Warm Layer: Core Memory — LLM-compressed MEMORY.md per session."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

from .models import LLMFunc, TurnRecord
from .prompts import CORE_MEMORY_INIT_PROMPT, CORE_MEMORY_UPDATE_PROMPT

_log = logging.getLogger("claw.core_memory")

_EMPTY_MEMORY = "# Core Memory\n\n*No memory recorded yet.*\n"


class CoreMemory:
    """Manages the per-session MEMORY.md file.

    The file is updated by calling an LLM to compress recent conversation
    turns into structured key facts.  It is always injected into the
    query context so the model has persistent session awareness.
    """

    def __init__(
        self,
        talk_dir: str,
        llm_func: LLMFunc,
        max_bytes: int = 10240,
        update_interval: int = 5,
    ):
        self._talk_dir = Path(talk_dir)
        self._llm_func = llm_func
        self._max_bytes = max_bytes
        self._update_interval = update_interval
        self._turns_since_update = 0

    @property
    def memory_file(self) -> Path:
        return self._talk_dir / "MEMORY.md"

    def read(self) -> str:
        """Return current MEMORY.md content, or empty placeholder."""
        if not self.memory_file.is_file():
            return ""
        try:
            return self.memory_file.read_text(encoding="utf-8")
        except Exception:
            return ""

    def exists(self) -> bool:
        return self.memory_file.is_file()

    def size_bytes(self) -> int:
        if not self.memory_file.is_file():
            return 0
        return self.memory_file.stat().st_size

    def record_turn(self) -> None:
        """Increment the turn counter (called after each Q&A)."""
        self._turns_since_update += 1

    def should_update(self) -> bool:
        """Whether enough turns have accumulated to trigger an LLM update."""
        return self._turns_since_update >= self._update_interval

    async def update(self, recent_turns: List[TurnRecord], force: bool = False) -> str:
        """Compress recent turns into MEMORY.md via LLM.

        Returns the new MEMORY.md content.
        """
        if not force and not self.should_update():
            return self.read()

        if not recent_turns:
            return self.read()

        turns_text = self._format_turns(recent_turns)
        current = self.read()

        try:
            if current and current.strip() != _EMPTY_MEMORY.strip():
                prompt = CORE_MEMORY_UPDATE_PROMPT.format(
                    current_memory_md=current,
                    start_turn=1,
                    end_turn=len(recent_turns),
                    recent_turns=turns_text,
                )
            else:
                prompt = CORE_MEMORY_INIT_PROMPT.format(
                    recent_turns=turns_text,
                )

            new_content = await self._llm_func(prompt)

            if new_content and len(new_content.strip()) > 10:
                content = new_content.strip()
                if len(content.encode("utf-8")) > self._max_bytes:
                    content = content[: self._max_bytes]
                    _log.warning("MEMORY.md truncated to %d bytes", self._max_bytes)

                self._write(content)
                self._turns_since_update = 0
                _log.info(
                    "MEMORY.md updated (%d bytes) for %s",
                    len(content.encode("utf-8")),
                    self._talk_dir.name,
                )
                return content
            else:
                _log.warning("LLM returned empty/short content for MEMORY.md update")
                return self.read()
        except Exception as exc:
            _log.error("Core memory update failed: %s", exc)
            return self.read()

    def _write(self, content: str) -> None:
        self._talk_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file.write_text(content, encoding="utf-8")

    @staticmethod
    def _format_turns(turns: List[TurnRecord]) -> str:
        lines = []
        for i, t in enumerate(turns, 1):
            label = "用户" if t.role == "user" else "助手"
            lines.append(f"第{i}轮 [{label}]: {t.content}")
        return "\n\n".join(lines)
