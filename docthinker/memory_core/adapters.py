"""Adapters from existing DocThinker components to memory backend protocols."""

from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from .protocols import ExpandedKnowledgeBackend

_log = logging.getLogger("docthinker.memory_core")


def serialize_episode_match(raw: Any) -> Optional[Dict[str, Any]]:
    """Normalize Neuro Memory's tuple output into a stable API shape."""

    try:
        episode, score, reason = raw
    except Exception:
        return None
    summary = str(getattr(episode, "summary", "") or "").strip()
    if not summary:
        return None
    return {
        "episode_id": str(getattr(episode, "episode_id", "") or ""),
        "summary": summary,
        "score": round(float(score or 0.0), 4),
        "reason": str(reason or ""),
        "source_type": str(getattr(episode, "source_type", "") or ""),
        "concepts": list(getattr(episode, "concepts", []) or [])[:8],
        "entity_ids": list(getattr(episode, "entity_ids", []) or [])[:8],
    }


class ClawConversationBackend:
    """Conversation-memory adapter for existing Claw managers."""

    def __init__(self, get_claw_manager: Callable[[Optional[str]], Optional[Any]]) -> None:
        self.get_claw_manager = get_claw_manager

    async def build_context(self, session_id: Optional[str], query: str) -> str:
        manager = self.get_claw_manager(session_id)
        if not manager:
            return ""
        return str(await manager.build_memory_context(query, enable_archive=True) or "")

    async def consolidate(
        self,
        session_id: Optional[str],
        question: str,
        answer: str,
    ) -> bool:
        manager = self.get_claw_manager(session_id)
        if not manager:
            return False
        await manager.post_query_update(question, answer, session_id, time.time())
        return True


class NeuroEpisodicBackend:
    """Episodic-memory adapter for existing Neuro Memory engines."""

    def __init__(
        self,
        get_memory_engine: Callable[[Optional[str]], Optional[Any]],
    ) -> None:
        self.get_memory_engine = get_memory_engine

    async def retrieve(
        self,
        session_id: Optional[str],
        query: str,
        *,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        engine = self.get_memory_engine(session_id)
        if not engine:
            return []
        raw_matches = await engine.retrieve_analogies(
            query,
            top_k=max(1, int(top_k)),
            then_spread=True,
            spread_top_k=3,
        )
        matches: List[Dict[str, Any]] = []
        for raw in raw_matches:
            match = serialize_episode_match(raw)
            if match:
                matches.append(match)
        return matches

    async def write(
        self,
        session_id: Optional[str],
        question: str,
        answer: str,
        *,
        concepts: Sequence[str],
        timestamp: float,
    ) -> Optional[str]:
        engine = self.get_memory_engine(session_id)
        if not engine:
            return None
        episode = await engine.add_observation(
            summary=f"User asked: {question}\nAssistant answered: {answer[:500]}",
            key_points=[question, answer[:300]],
            concepts=list(concepts),
            entity_ids=list(concepts),
            source_type="chat",
            session_id=session_id,
            timestamp=timestamp,
        )
        if episode is None:
            return None
        return str(getattr(episode, "episode_id", "") or "")


class ExpandedNodeBackend:
    """Expanded-KG adapter for existing ExpandedNodeManager instances."""

    def __init__(
        self,
        get_expanded_node_manager: Callable[[Optional[str]], Optional[Any]],
    ) -> None:
        self.get_expanded_node_manager = get_expanded_node_manager

    def _manager(self, session_id: Optional[str]) -> Optional[Any]:
        return self.get_expanded_node_manager(session_id)

    def match(
        self,
        session_id: Optional[str],
        query: str,
        *,
        top_k: int,
        min_score: float,
    ) -> List[Dict[str, Any]]:
        manager = self._manager(session_id)
        if not manager:
            return []
        matches = manager.match_nodes(
            query=query,
            top_k=max(1, int(top_k)),
            min_score=max(0.0, float(min_score)),
        )
        if matches:
            manager.mark_hits([str(match.get("entity") or "") for match in matches])
        return list(matches or [])

    def build_instruction(
        self,
        session_id: Optional[str],
        matches: Sequence[Dict[str, Any]],
        *,
        limit: int,
    ) -> str:
        manager = self._manager(session_id)
        if not manager:
            return ""
        return str(manager.build_forced_instruction(matches, limit=max(1, int(limit))) or "")

    def record_usage(
        self,
        session_id: Optional[str],
        answer: str,
        matches: Sequence[Dict[str, Any]],
        *,
        attached_entities: Sequence[str],
    ) -> List[str]:
        manager = self._manager(session_id)
        if not manager:
            return []
        manager.record_response_usage(
            answer=answer,
            matches=matches,
            attached_entities=list(attached_entities),
        )
        # Older managers may still return a frequency-based promotion list.
        # Recording usage is allowed; this adapter supplies no source evidence
        # and must not report such a list as successfully promoted facts.
        return []

    def get_record(self, session_id: Optional[str], name: str) -> Optional[Dict[str, Any]]:
        manager = self._manager(session_id)
        if not manager:
            return None
        record = manager.get(name)
        return record if isinstance(record, dict) else None


class GraphCorePromotionBackend:
    """Fail-closed compatibility adapter for legacy frequency-based promotion.

    Names, answer associations, and usage scores do not provide independent
    document evidence. This interface therefore cannot authorize source-graph
    writes. Verified ECLRR relations use their separate gate/writeback path.
    """

    def __init__(
        self,
        get_session_rag: Callable[[Optional[str]], Awaitable[Any]],
    ) -> None:
        self.get_session_rag = get_session_rag

    async def promote(
        self,
        session_id: Optional[str],
        promoted_names: Sequence[str],
        *,
        answer_entities: Sequence[str],
        expanded_backend: ExpandedKnowledgeBackend,
    ) -> List[str]:
        if not session_id or not promoted_names:
            return []

        _log.warning(
            "Skipped frequency-based promotion of %d expanded candidates: "
            "independently verified source evidence is required",
            len(promoted_names),
        )
        return []


class ChatTurnIngestBackend:
    """Adapter for the optional chat-turn-to-KG ingestion hook."""

    def __init__(
        self,
        ingest_chat_turn: Callable[[str, str, Optional[str], Optional[float]], Awaitable[None]],
        enabled: Callable[[], bool],
    ) -> None:
        self.ingest_chat_turn = ingest_chat_turn
        self.enabled = enabled

    async def ingest(
        self,
        session_id: Optional[str],
        question: str,
        answer: str,
        *,
        timestamp: float,
    ) -> bool:
        if not self.enabled():
            return False
        await self.ingest_chat_turn(question, answer, session_id, timestamp)
        return True
