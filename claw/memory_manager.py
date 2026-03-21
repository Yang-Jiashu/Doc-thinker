"""Orchestrator: ties the three memory layers together."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .core_memory import CoreMemory
from .models import EmbeddingFunc, LLMFunc, MemoryConfig, TurnRecord
from .semantic_archive import SemanticArchive
from .working_memory import WorkingMemory

_log = logging.getLogger("claw.manager")


class ClawMemoryManager:
    """Per-session tiered memory manager.

    Layers:
      1. Hot  – WorkingMemory   (recent N turns, always injected)
      2. Warm – CoreMemory      (LLM-compressed MEMORY.md, always injected)
      3. Cold – SemanticArchive (older turns embedded, top-k retrieval)
    """

    def __init__(
        self,
        talk_dir: str,
        llm_func: LLMFunc,
        embedding_func: Optional[EmbeddingFunc] = None,
        config: Optional[MemoryConfig] = None,
    ):
        cfg = config or MemoryConfig()
        self.talk_dir = talk_dir
        self.config = cfg

        self.working = WorkingMemory(talk_dir, max_turns=cfg.working_memory_turns)
        self.core = CoreMemory(
            talk_dir,
            llm_func=llm_func,
            max_bytes=cfg.core_memory_max_bytes,
            update_interval=cfg.core_memory_update_interval,
        )
        self.archive = SemanticArchive(
            talk_dir,
            embedding_func=embedding_func,
            chunk_size=cfg.archive_chunk_size,
            top_k=cfg.archive_top_k,
            min_score=cfg.archive_min_score,
        )

    async def build_memory_context(
        self, query: str, *, enable_archive: bool = True
    ) -> str:
        """Assemble a memory context string from all three layers.

        Returns a block of text ready to be prepended to the LLM prompt.
        """
        sections: List[str] = []

        core_md = self.core.read()
        if not core_md and not self.core.exists():
            all_turns = self.working.get_all_turns()
            if len(all_turns) >= 4:
                try:
                    core_md = await self.core.update(all_turns, force=True)
                    _log.info("Auto-created MEMORY.md on first query (%d turns)", len(all_turns))
                except Exception as exc:
                    _log.warning("Auto-create MEMORY.md failed: %s", exc)

        if core_md and core_md.strip():
            sections.append(f"## 核心记忆\n{core_md}")

        recent_turns = self.working.get_recent_turns()
        if recent_turns:
            formatted = self.working.format_for_context(recent_turns)
            sections.append(f"## 近期对话\n{formatted}")

        if enable_archive:
            archive_hits = await self.archive.search(query)
            if archive_hits:
                lines = []
                for hit in archive_hits:
                    lines.append(f"- [{hit.score:.2f}] {hit.text[:300]}")
                sections.append("## 历史记忆检索\n" + "\n".join(lines))

        if not sections:
            return ""

        return "---\n" + "\n\n".join(sections) + "\n---"

    async def post_query_update(
        self,
        question: str,
        answer: str,
        session_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Called after each Q&A turn to update all memory layers."""
        ts = timestamp or time.time()

        self.core.record_turn()

        all_turns = self.working.get_all_turns()

        if self.core.should_update() and len(all_turns) >= 2:
            try:
                await self.core.update(all_turns, force=True)
            except Exception as exc:
                _log.warning("Core memory update failed: %s", exc)

        wm_count = self.config.working_memory_turns * 2
        if len(all_turns) > wm_count:
            older = all_turns[:-wm_count]
            try:
                await self.archive.archive_turns(older)
            except Exception as exc:
                _log.warning("Archive update failed: %s", exc)

    def get_stats(self) -> Dict[str, Any]:
        """Return stats for the /memory/stats API."""
        return {
            "turn_count": self.working.get_turn_count(),
            "core_memory_exists": self.core.exists(),
            "core_memory_bytes": self.core.size_bytes(),
            "archive_chunks": self.archive.chunk_count,
            "working_memory_window": self.config.working_memory_turns,
            "core_update_interval": self.config.core_memory_update_interval,
        }

    def save(self) -> None:
        """Persist archive to disk (core memory is auto-persisted on update)."""
        self.archive.save()
