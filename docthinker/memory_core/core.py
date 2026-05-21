"""Unified facade for agentic memory behavior.

The first version intentionally wraps existing systems instead of replacing
them:
- Claw provides conversational working/core/archive memory.
- Neuro memory provides episodic analogies and activation traces.
- Expanded KG nodes provide evolving semantic hypotheses.
- GraphCore receives promoted expansion nodes after repeated use.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from .adapters import (
    ChatTurnIngestBackend,
    ClawConversationBackend,
    ExpandedNodeBackend,
    GraphCorePromotionBackend,
    NeuroEpisodicBackend,
)
from .protocols import AgentMemoryBackends, MemoryPolicy

_log = logging.getLogger("docthinker.memory_core")


def _extract_entities_from_text(text: str, max_entities: int = 12) -> List[str]:
    try:
        from docthinker.kg_expansion import extract_entities_from_text

        return extract_entities_from_text(text, max_entities=max_entities)
    except Exception:
        source = str(text or "")
        entities: List[str] = []
        seen: set[str] = set()
        for pattern in (r"[\u4e00-\u9fff]{2,10}", r"\b[A-Za-z][A-Za-z0-9_\-]{2,32}\b"):
            for match in re.finditer(pattern, source):
                name = match.group(0).strip()
                if name and name not in seen:
                    seen.add(name)
                    entities.append(name)
                    if len(entities) >= max_entities:
                        return entities
        return entities


@dataclass
class MemoryTrace:
    """Small, serializable trace of memory work done for one turn."""

    memory_mode: str = "session"
    memory_hits: int = 0
    episodic_hits: int = 0
    expanded_hits: int = 0
    memory_context_injected: bool = False
    retrieval_instruction_applied: bool = False
    events: List[Dict[str, Any]] = field(default_factory=list)

    def format_for_response(self, mode: str = "") -> str:
        lines: List[str] = [f"memory_mode: {self.memory_mode}"]
        if self.retrieval_instruction_applied:
            lines.append("retrieval_instruction: provided")
        lines.append(f"memory_hits: {self.memory_hits}")
        lines.append(f"episodic_hits: {self.episodic_hits}")
        lines.append(f"expanded_hits: {self.expanded_hits}")
        if mode:
            lines.append(f"mode: {mode}")
        if self.memory_context_injected:
            lines.append("memory_context: injected")
        return "\n".join(lines)

    def to_schema(self, *, consolidation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        conversation_hits = sum(
            1 for event in self.events if event.get("type") == "memory_context"
        )
        return {
            "memory_mode": self.memory_mode,
            "recall": {
                "memory_sources": self.memory_hits,
                "conversation_hits": conversation_hits,
                "episodic_hits": self.episodic_hits,
                "expanded_hits": self.expanded_hits,
                "context_injected": self.memory_context_injected,
                "retrieval_instruction_applied": self.retrieval_instruction_applied,
            },
            "events": list(self.events),
            "consolidation": dict(consolidation or {}),
        }


@dataclass
class RecallBundle:
    """Memory context prepared before answer generation."""

    retrieval_instruction: str = ""
    memory_summaries: List[Dict[str, Any]] = field(default_factory=list)
    episodic_matches: List[Dict[str, Any]] = field(default_factory=list)
    expanded_matches: List[Dict[str, Any]] = field(default_factory=list)
    trace: MemoryTrace = field(default_factory=MemoryTrace)


class AgentMemoryCore:
    """Facade that presents memory as a coherent agent-facing subsystem."""

    def __init__(
        self,
        *,
        backends: Optional[AgentMemoryBackends] = None,
        policy: Optional[MemoryPolicy] = None,
        get_claw_manager: Optional[Callable[[Optional[str]], Optional[Any]]] = None,
        get_expanded_node_manager: Optional[Callable[[Optional[str]], Optional[Any]]] = None,
        get_session_rag: Optional[Callable[[Optional[str]], Awaitable[Any]]] = None,
        get_memory_engine: Optional[Callable[[Optional[str]], Optional[Any]]] = None,
        ingest_chat_turn: Optional[
            Callable[[str, str, Optional[str], Optional[float]], Awaitable[None]]
        ] = None,
        chat_turn_ingest_enabled: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.policy = policy or MemoryPolicy()
        self.backends = backends or self._build_legacy_backends(
            get_claw_manager=get_claw_manager,
            get_expanded_node_manager=get_expanded_node_manager,
            get_session_rag=get_session_rag,
            get_memory_engine=get_memory_engine,
            ingest_chat_turn=ingest_chat_turn,
            chat_turn_ingest_enabled=chat_turn_ingest_enabled,
        )

    @staticmethod
    def _build_legacy_backends(
        *,
        get_claw_manager: Optional[Callable[[Optional[str]], Optional[Any]]],
        get_expanded_node_manager: Optional[Callable[[Optional[str]], Optional[Any]]],
        get_session_rag: Optional[Callable[[Optional[str]], Awaitable[Any]]],
        get_memory_engine: Optional[Callable[[Optional[str]], Optional[Any]]],
        ingest_chat_turn: Optional[
            Callable[[str, str, Optional[str], Optional[float]], Awaitable[None]]
        ],
        chat_turn_ingest_enabled: Optional[Callable[[], bool]],
    ) -> AgentMemoryBackends:
        expanded = (
            ExpandedNodeBackend(get_expanded_node_manager)
            if get_expanded_node_manager
            else None
        )
        return AgentMemoryBackends(
            conversation=(
                ClawConversationBackend(get_claw_manager)
                if get_claw_manager
                else None
            ),
            episodic=(
                NeuroEpisodicBackend(get_memory_engine)
                if get_memory_engine
                else None
            ),
            expanded=expanded,
            graph=(
                GraphCorePromotionBackend(get_session_rag)
                if get_session_rag
                else None
            ),
            chat_turn=(
                ChatTurnIngestBackend(
                    ingest_chat_turn,
                    chat_turn_ingest_enabled or (lambda: False),
                )
                if ingest_chat_turn
                else None
            ),
        )

    @staticmethod
    def _merge_instructions(*instructions: Optional[str]) -> str:
        parts: List[str] = []
        for instruction in instructions:
            text = str(instruction or "").strip()
            if text:
                parts.append(text)
        return "\n\n".join(parts).strip()

    @staticmethod
    def _format_episodic_instruction(matches: Sequence[Dict[str, Any]]) -> str:
        if not matches:
            return ""
        lines = [
            "## 情节记忆与类比参考",
            "以下是 agent 过往经历中与当前问题相似的片段。请将其作为类比线索，而不是直接事实来源。",
            "",
        ]
        for item in matches[:5]:
            summary = str(item.get("summary") or "").strip()
            reason = str(item.get("reason") or "").strip()
            score = float(item.get("score") or 0.0)
            lines.append(f"- [{score:.2f}] {summary[:240]}")
            if reason:
                lines.append(f"  关联原因: {reason[:160]}")
        return "\n".join(lines)

    async def recall(
        self,
        *,
        session_id: Optional[str],
        query: str,
        base_instruction: str = "",
        mode: str = "",
        enable_thinking: bool = False,
        enable_expanded_matching: bool = True,
        expanded_top_k: Optional[int] = None,
        expanded_min_score: Optional[float] = None,
        skip_memory: bool = False,
    ) -> RecallBundle:
        """Prepare memory context and KG hypothesis matches for one query."""

        trace = MemoryTrace()
        merged_instruction = str(base_instruction or "").strip()
        memory_summaries: List[Dict[str, Any]] = []
        episodic_matches: List[Dict[str, Any]] = []
        expanded_matches: List[Dict[str, Any]] = []

        if skip_memory:
            trace.retrieval_instruction_applied = bool(merged_instruction)
            return RecallBundle(
                retrieval_instruction=merged_instruction,
                memory_summaries=memory_summaries,
                episodic_matches=episodic_matches,
                expanded_matches=expanded_matches,
                trace=trace,
            )

        if enable_thinking:
            if self.backends.conversation and self.policy.layer_enabled("conversation"):
                try:
                    claw_context = await self.backends.conversation.build_context(session_id, query)
                    if claw_context:
                        merged_instruction = self._merge_instructions(
                            merged_instruction,
                            claw_context,
                        )
                        memory_summaries = [{"source": "claw", "injected": True}]
                        trace.memory_context_injected = True
                        trace.events.append({
                            "type": "memory_context",
                            "source": "claw",
                            "chars": len(claw_context),
                        })
                except Exception as exc:
                    _log.warning("[recall] claw context failed: %s", exc)

            if self.backends.episodic and self.policy.layer_enabled("episodic"):
                try:
                    episodic_matches = await self.backends.episodic.retrieve(
                        session_id,
                        query,
                        top_k=self.policy.episodic_top_k,
                    )
                    if episodic_matches:
                        episodic_instruction = self._format_episodic_instruction(
                            episodic_matches,
                        )
                        merged_instruction = self._merge_instructions(
                            merged_instruction,
                            episodic_instruction,
                        )
                        memory_summaries.append({
                            "source": "neuro_memory",
                            "kind": "episodic_analogy",
                            "count": len(episodic_matches),
                        })
                        trace.memory_context_injected = True
                        trace.events.append({
                            "type": "episodic_recall",
                            "source": "neuro_memory",
                            "count": len(episodic_matches),
                        })
                except Exception as exc:
                    _log.warning("[recall] episodic recall failed: %s", exc)

        if enable_expanded_matching and self.backends.expanded and self.policy.layer_enabled("expanded"):
            try:
                effective_expanded_top_k = (
                    expanded_top_k
                    if expanded_top_k is not None
                    else self.policy.expanded_top_k
                )
                effective_expanded_min_score = (
                    expanded_min_score
                    if expanded_min_score is not None
                    else self.policy.expanded_min_score
                )
                expanded_matches = self.backends.expanded.match(
                    session_id,
                    query,
                    top_k=effective_expanded_top_k,
                    min_score=effective_expanded_min_score,
                )
                if expanded_matches:
                    expanded_instruction = self.backends.expanded.build_instruction(
                        session_id,
                        expanded_matches,
                        limit=min(
                            self.policy.expanded_instruction_limit,
                            max(1, int(effective_expanded_top_k)),
                        ),
                    )
                    merged_instruction = self._merge_instructions(
                        merged_instruction,
                        expanded_instruction,
                    )
                    trace.events.append({
                        "type": "expanded_match",
                        "count": len(expanded_matches),
                    })
            except Exception as exc:
                _log.warning("[recall] expanded matching failed: %s", exc)

        trace.memory_hits = len(memory_summaries)
        trace.episodic_hits = len(episodic_matches)
        trace.expanded_hits = len(expanded_matches)
        trace.retrieval_instruction_applied = bool(merged_instruction)
        return RecallBundle(
            retrieval_instruction=merged_instruction,
            memory_summaries=memory_summaries,
            episodic_matches=episodic_matches,
            expanded_matches=expanded_matches,
            trace=trace,
        )

    async def after_response(
        self,
        *,
        session_id: Optional[str],
        question: str,
        answer: str,
        matched_expanded: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Consolidate memories after a response has been produced."""

        if not session_id or not answer:
            return {"updated": False}

        t0 = time.time()
        result: Dict[str, Any] = {
            "updated": True,
            "claw_updated": False,
            "episode_added": False,
            "chat_turn_ingested": False,
            "expanded_promoted": [],
        }

        if self.backends.conversation and self.policy.layer_enabled("conversation"):
            try:
                result["claw_updated"] = await self.backends.conversation.consolidate(
                    session_id,
                    question,
                    answer,
                )
            except Exception as exc:
                _log.warning("[after_response] claw update failed: %s", exc)

        concepts = _extract_entities_from_text(
            f"{question}\n{answer}",
            max_entities=self.policy.answer_entity_limit,
        )

        if self.backends.episodic and self.policy.layer_enabled("episodic"):
            try:
                episode_id = await self.backends.episodic.write(
                    session_id,
                    question,
                    answer,
                    concepts=concepts,
                    timestamp=time.time(),
                )
                if episode_id is not None:
                    result["episode_added"] = True
                    result["episode_id"] = episode_id
            except Exception as exc:
                _log.warning("[after_response] episode add failed: %s", exc)

        if self.backends.chat_turn and self.policy.layer_enabled("chat_turn"):
            try:
                result["chat_turn_ingested"] = await self.backends.chat_turn.ingest(
                    session_id,
                    question,
                    answer,
                    timestamp=time.time(),
                )
            except Exception as exc:
                _log.warning("[after_response] chat turn ingest failed: %s", exc)

        if matched_expanded and self.backends.expanded and self.policy.layer_enabled("expanded"):
            try:
                promoted_names = self.backends.expanded.record_usage(
                    session_id,
                    answer,
                    matched_expanded,
                    attached_entities=concepts,
                )
                if promoted_names and self.backends.graph and self.policy.layer_enabled("graph"):
                    promoted_names = await self.backends.graph.promote(
                        session_id,
                        promoted_names,
                        answer_entities=concepts,
                        expanded_backend=self.backends.expanded,
                    )
                result["expanded_promoted"] = promoted_names
            except Exception as exc:
                _log.warning("[after_response] expanded promotion failed: %s", exc)

        result["elapsed_seconds"] = round(time.time() - t0, 3)
        result["memory_trace"] = {
            "memory_mode": "session",
            "consolidation": {
                "conversation_updated": bool(result["claw_updated"]),
                "episode_added": bool(result["episode_added"]),
                "chat_turn_ingested": bool(result["chat_turn_ingested"]),
                "expanded_promoted": list(result["expanded_promoted"]),
                "elapsed_seconds": result["elapsed_seconds"],
                "enabled_layers": sorted(self.policy.enabled_layer_set()),
            },
        }
        return result
