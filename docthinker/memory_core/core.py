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
        get_claw_manager: Callable[[Optional[str]], Optional[Any]],
        get_expanded_node_manager: Callable[[Optional[str]], Optional[Any]],
        get_session_rag: Callable[[Optional[str]], Awaitable[Any]],
        get_memory_engine: Optional[Callable[[Optional[str]], Optional[Any]]] = None,
        ingest_chat_turn: Optional[
            Callable[[str, str, Optional[str], Optional[float]], Awaitable[None]]
        ] = None,
        chat_turn_ingest_enabled: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.get_claw_manager = get_claw_manager
        self.get_memory_engine = get_memory_engine or (lambda _sid: None)
        self.get_expanded_node_manager = get_expanded_node_manager
        self.get_session_rag = get_session_rag
        self.ingest_chat_turn = ingest_chat_turn
        self.chat_turn_ingest_enabled = chat_turn_ingest_enabled or (lambda: False)

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

    @staticmethod
    def _serialize_episode_match(raw: Any) -> Optional[Dict[str, Any]]:
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

    async def recall(
        self,
        *,
        session_id: Optional[str],
        query: str,
        base_instruction: str = "",
        mode: str = "",
        enable_thinking: bool = False,
        enable_expanded_matching: bool = True,
        expanded_top_k: int = 2,
        expanded_min_score: float = 0.2,
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
            claw_mgr = self.get_claw_manager(session_id)
            if claw_mgr:
                try:
                    claw_context = await claw_mgr.build_memory_context(
                        query,
                        enable_archive=True,
                    )
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

            memory_engine = self.get_memory_engine(session_id)
            if memory_engine:
                try:
                    raw_matches = await memory_engine.retrieve_analogies(
                        query,
                        top_k=5,
                        then_spread=True,
                        spread_top_k=3,
                    )
                    for raw in raw_matches:
                        match = self._serialize_episode_match(raw)
                        if match:
                            episodic_matches.append(match)
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

        if enable_expanded_matching:
            expanded_manager = self.get_expanded_node_manager(session_id)
            if expanded_manager:
                expanded_matches = expanded_manager.match_nodes(
                    query=query,
                    top_k=max(1, int(expanded_top_k)),
                    min_score=max(0.0, float(expanded_min_score)),
                )
                if expanded_matches:
                    expanded_manager.mark_hits([
                        str(match.get("entity") or "")
                        for match in expanded_matches
                    ])
                    expanded_instruction = expanded_manager.build_forced_instruction(
                        expanded_matches,
                        limit=min(2, max(1, int(expanded_top_k))),
                    )
                    merged_instruction = self._merge_instructions(
                        merged_instruction,
                        expanded_instruction,
                    )
                    trace.events.append({
                        "type": "expanded_match",
                        "count": len(expanded_matches),
                    })

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

        claw_mgr = self.get_claw_manager(session_id)
        if claw_mgr:
            try:
                await claw_mgr.post_query_update(question, answer, session_id, time.time())
                result["claw_updated"] = True
            except Exception as exc:
                _log.warning("[after_response] claw update failed: %s", exc)

        memory_engine = self.get_memory_engine(session_id)
        if memory_engine:
            try:
                concepts = _extract_entities_from_text(
                    f"{question}\n{answer}",
                    max_entities=12,
                )
                episode = await memory_engine.add_observation(
                    summary=f"User asked: {question}\nAssistant answered: {answer[:500]}",
                    key_points=[question, answer[:300]],
                    concepts=concepts,
                    entity_ids=concepts,
                    source_type="chat",
                    session_id=session_id,
                    timestamp=time.time(),
                )
                if episode is not None:
                    result["episode_added"] = True
                    result["episode_id"] = getattr(episode, "episode_id", "")
            except Exception as exc:
                _log.warning("[after_response] episode add failed: %s", exc)

        if self.ingest_chat_turn and self.chat_turn_ingest_enabled():
            try:
                await self.ingest_chat_turn(question, answer, session_id, time.time())
                result["chat_turn_ingested"] = True
            except Exception as exc:
                _log.warning("[after_response] chat turn ingest failed: %s", exc)

        if matched_expanded:
            try:
                promoted = await self._promote_expanded_nodes(
                    session_id=session_id,
                    answer=answer,
                    matched_nodes=matched_expanded,
                )
                result["expanded_promoted"] = promoted
            except Exception as exc:
                _log.warning("[after_response] expanded promotion failed: %s", exc)

        result["elapsed_seconds"] = round(time.time() - t0, 3)
        return result

    async def _promote_expanded_nodes(
        self,
        *,
        session_id: str,
        answer: str,
        matched_nodes: Sequence[Dict[str, Any]],
    ) -> List[str]:
        manager = self.get_expanded_node_manager(session_id)
        if manager is None:
            return []

        entities = _extract_entities_from_text(answer, max_entities=12)
        usage = manager.record_response_usage(
            answer=answer,
            matches=matched_nodes,
            attached_entities=entities,
        )
        promoted_names = [
            str(name).strip()
            for name in (usage.get("promoted") or [])
            if str(name).strip()
        ]
        if not promoted_names:
            return []

        session_rag = await self.get_session_rag(session_id)
        graphcore = getattr(session_rag, "graphcore", None)
        if graphcore is None:
            return []
        graph = graphcore.chunk_entity_relation_graph
        changed = False

        for name in promoted_names:
            record = manager.get(name)
            if not record:
                continue
            roots = [
                str(root).strip()
                for root in (record.get("root_ids") or [])
                if str(root).strip()
            ]

            await graph.upsert_node(
                name,
                {
                    "entity_id": name,
                    "entity_type": "concept",
                    "description": record.get("reason") or record.get("description") or name,
                    "source_id": "promoted_expansion",
                    "is_expanded": "0",
                },
            )
            changed = True

            for ent in entities[:8]:
                if not ent or ent == name:
                    continue
                await graph.upsert_node(
                    ent,
                    {
                        "entity_id": ent,
                        "entity_type": "concept",
                        "description": f"Extracted from answer for expansion node {name}",
                        "source_id": "answer_entity",
                    },
                )
                await graph.upsert_edge(
                    name,
                    ent,
                    {
                        "keywords": "co_mentioned",
                        "description": f"Assistant answer associated {name} with {ent}",
                        "source_id": "answer_entity",
                    },
                )
                changed = True

            for root in roots[:6]:
                if not root or root == name:
                    continue
                await graph.upsert_edge(
                    name,
                    root,
                    {
                        "keywords": "expanded_from_root",
                        "description": f"Promoted expansion node linked to root node {root}",
                        "source_id": "llm_expansion",
                    },
                )
                changed = True

        if changed and hasattr(graph, "index_done_callback"):
            try:
                await graph.index_done_callback(force_save=True)
            except TypeError:
                await graph.index_done_callback()

        return promoted_names
