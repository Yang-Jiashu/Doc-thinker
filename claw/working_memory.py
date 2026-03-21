"""Hot Layer: Working Memory — recent N turns from talk.json, always injected."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .models import TurnRecord

_log = logging.getLogger("claw.working_memory")


class WorkingMemory:
    """Reads recent conversation turns from a session's talk.json."""

    def __init__(self, talk_dir: str, max_turns: int = 6):
        self._talk_dir = Path(talk_dir)
        self._max_turns = max_turns

    @property
    def talk_file(self) -> Path:
        return self._talk_dir / "talk.json"

    def get_recent_turns(self, n: Optional[int] = None) -> List[TurnRecord]:
        """Return the most recent *n* turns (default: configured max)."""
        limit = n or self._max_turns
        messages = self._load_messages()
        recent = messages[-limit * 2:] if len(messages) > limit * 2 else messages
        return [TurnRecord.from_dict(m) for m in recent]

    def get_all_turns(self) -> List[TurnRecord]:
        """Return every message in the session."""
        return [TurnRecord.from_dict(m) for m in self._load_messages()]

    def get_turn_count(self) -> int:
        return len(self._load_messages())

    def format_for_context(self, turns: Optional[List[TurnRecord]] = None) -> str:
        """Format recent turns into a context string for the LLM."""
        if turns is None:
            turns = self.get_recent_turns()
        if not turns:
            return ""
        lines = []
        for t in turns:
            label = "用户" if t.role == "user" else "助手"
            lines.append(f"[{label}] {t.content}")
        return "\n".join(lines)

    def _load_messages(self) -> List[Dict]:
        if not self.talk_file.is_file():
            return []
        try:
            data = json.loads(self.talk_file.read_text(encoding="utf-8"))
            return data.get("messages", [])
        except Exception as exc:
            _log.warning("Failed to read talk.json: %s", exc)
            return []
