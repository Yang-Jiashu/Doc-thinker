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
import threading
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
    long_horizon_hits: int = 0
    memory_context_injected: bool = False
    retrieval_instruction_applied: bool = False
    recall_plan: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def format_for_response(self, mode: str = "") -> str:
        lines: List[str] = [f"memory_mode: {self.memory_mode}"]
        if self.retrieval_instruction_applied:
            lines.append("retrieval_instruction: provided")
        lines.append(f"memory_hits: {self.memory_hits}")
        lines.append(f"episodic_hits: {self.episodic_hits}")
        lines.append(f"expanded_hits: {self.expanded_hits}")
        lines.append(f"long_horizon_hits: {self.long_horizon_hits}")
        if self.recall_plan:
            lines.append(f"recall_plan: {self.recall_plan.get('question_type', 'general')}")
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
                "long_horizon_hits": self.long_horizon_hits,
                "context_injected": self.memory_context_injected,
                "retrieval_instruction_applied": self.retrieval_instruction_applied,
                "plan": dict(self.recall_plan or {}),
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
    long_horizon_matches: List[Dict[str, Any]] = field(default_factory=list)
    trace: MemoryTrace = field(default_factory=MemoryTrace)


class InMemoryLongHorizonBackend:
    """Process-local long-horizon memory backend.

    It keeps the memory-core contract useful out of the box while remaining
    easy to replace with SQLite, vector storage, or a graph-backed backend.
    """

    def __init__(self) -> None:
        self._items: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _tokens(text: str) -> set[str]:
        source = str(text or "").lower()
        tokens: set[str] = set()
        for match in re.finditer(r"[\u4e00-\u9fff]{2,8}|[a-z][a-z0-9_\-]{2,32}", source):
            tokens.add(match.group(0))
        return tokens

    @staticmethod
    def _scope_key(session_id: Optional[str], scope: str) -> str:
        sid = str(session_id or "global").strip() or "global"
        normalized = str(scope or "session").strip() or "session"
        if normalized == "session":
            return f"session:{sid}"
        return f"{normalized}:default"

    @staticmethod
    def _clean_summary(answer: str) -> str:
        text = re.sub(r"\s+", " ", str(answer or "")).strip()
        if not text:
            return ""
        chunks = re.split(r"(?<=[。！？.!?])\s+", text)
        summary = chunks[0] if chunks else text
        return summary[:260]

    @staticmethod
    def _classify_query(query: str) -> str:
        q = str(query or "").lower()
        if any(k in q for k in ("时间线", "timeline", "history", "演化", "long horizon")):
            return "temporal"
        if any(k in q for k in ("冲突", "矛盾", "conflict", "inconsistent")):
            return "conflict"
        if any(k in q for k in ("比较", "compare", "对比", "difference", "区别")):
            return "comparison"
        if any(k in q for k in ("计划", "roadmap", "next", "下一步", "怎么做", "怎么办")):
            return "planning"
        if any(k in q for k in ("为什么", "why", "推理", "reason", "cause")):
            return "causal_reasoning"
        if any(k in q for k in ("总结", "summary", "归纳", "synthesize")):
            return "synthesis"
        return "general"

    @staticmethod
    def _classify_insight(question: str, answer: str) -> str:
        q = str(question or "").lower()
        a = str(answer or "").lower()
        if any(k in q for k in ("喜欢", "prefer", "不要", "别", "风格")):
            return "preference"
        if any(k in q for k in ("修复", "实现", "改", "优化", "todo", "next", "roadmap")):
            return "task_memory"
        if any(k in a for k in ("because", "therefore", "因为", "所以", "推理")):
            return "reasoning_trace"
        return "insight"

    def build_recall_plan(
        self,
        query: str,
        *,
        mode: str = "",
        enable_thinking: bool = False,
    ) -> Dict[str, Any]:
        question_type = self._classify_query(query)
        layers = ["long_horizon", "expanded"]
        if enable_thinking:
            layers.insert(0, "conversation")
            layers.insert(1, "episodic")
        if question_type in {"temporal", "planning", "conflict", "synthesis"}:
            reason = "query benefits from cross-turn state and consolidated insights"
        elif question_type == "comparison":
            reason = "query benefits from prior contrasts and graph hypotheses"
        else:
            reason = "default recall plan"
        return {
            "question_type": question_type,
            "mode": mode or "default",
            "layers": layers,
            "reason": reason,
        }

    def retrieve(
        self,
        session_id: Optional[str],
        query: str,
        *,
        scopes: Sequence[str],
        top_k: int,
        min_confidence: float,
    ) -> List[Dict[str, Any]]:
        query_tokens = self._tokens(query)
        scored: List[tuple[float, Dict[str, Any]]] = []
        with self._lock:
            for scope in scopes:
                for item in self._items.get(self._scope_key(session_id, scope), []):
                    confidence = float(item.get("confidence") or 0.0)
                    if confidence < min_confidence:
                        continue
                    item_tokens = self._tokens(
                        " ".join([
                            str(item.get("summary") or ""),
                            str(item.get("question") or ""),
                            " ".join(str(c) for c in item.get("concepts") or []),
                        ])
                    )
                    overlap = len(query_tokens & item_tokens)
                    if query_tokens and overlap == 0:
                        continue
                    score = confidence + (overlap * 0.12) + min(0.2, float(item.get("use_count") or 0) * 0.03)
                    payload = dict(item)
                    payload["score"] = round(score, 3)
                    scored.append((score, payload))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [payload for _, payload in scored[: max(0, int(top_k))]]

    def build_instruction(
        self,
        matches: Sequence[Dict[str, Any]],
        *,
        limit: int,
    ) -> str:
        if not matches:
            return ""
        lines = [
            "## 长期记忆与跨回合推理",
            "以下是系统从过往交互中巩固出的长期线索。请优先把它们当作用户偏好、项目状态和推理连续性的约束。",
            "",
        ]
        for item in matches[: max(1, int(limit))]:
            summary = str(item.get("summary") or "").strip()
            kind = str(item.get("kind") or "insight")
            confidence = float(item.get("confidence") or 0.0)
            scope = str(item.get("scope") or "session")
            lines.append(f"- [{scope}/{kind}/{confidence:.2f}] {summary[:240]}")
        return "\n".join(lines)

    def consolidate(
        self,
        session_id: Optional[str],
        question: str,
        answer: str,
        *,
        concepts: Sequence[str],
        scope: str,
        timestamp: float,
        matched_expanded: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        summary = self._clean_summary(answer)
        if not summary or len(summary) < 24:
            return None
        negative = ("抱歉", "不知道", "没有检索到", "无法回答", "i don't know", "sorry")
        if any(marker in summary.lower() for marker in negative):
            return None

        scope_name = str(scope or "session")
        key = self._scope_key(session_id, scope_name)
        concepts_list = [str(c) for c in concepts[:12] if str(c).strip()]
        expanded_entities = [
            str(item.get("entity") or item.get("name") or "")
            for item in (matched_expanded or [])
            if item.get("entity") or item.get("name")
        ]
        new_tokens = self._tokens(summary)
        with self._lock:
            bucket = self._items.setdefault(key, [])
            for item in bucket:
                existing_tokens = self._tokens(str(item.get("summary") or ""))
                if new_tokens and len(new_tokens & existing_tokens) >= max(2, min(5, len(new_tokens) // 2)):
                    item["summary"] = summary
                    item["question"] = str(question or "")[:260]
                    item["concepts"] = sorted(set((item.get("concepts") or []) + concepts_list))[:16]
                    item["expanded_entities"] = sorted(set((item.get("expanded_entities") or []) + expanded_entities))[:12]
                    item["last_seen_at"] = timestamp
                    item["use_count"] = int(item.get("use_count") or 0) + 1
                    item["confidence"] = min(0.95, float(item.get("confidence") or 0.55) + 0.05)
                    return dict(item)

            item = {
                "id": f"lh-{len(bucket) + 1}",
                "scope": scope_name,
                "kind": self._classify_insight(question, answer),
                "question": str(question or "")[:260],
                "summary": summary,
                "concepts": concepts_list,
                "expanded_entities": expanded_entities[:12],
                "confidence": 0.62 if expanded_entities else 0.55,
                "created_at": timestamp,
                "last_seen_at": timestamp,
                "use_count": 1,
            }
            bucket.append(item)
            if len(bucket) > 200:
                del bucket[: len(bucket) - 200]
            return dict(item)

    def stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if session_id:
                keys = [self._scope_key(session_id, "session"), self._scope_key(session_id, "project"), self._scope_key(session_id, "user")]
                items = [dict(item) for key in keys for item in self._items.get(key, [])]
            else:
                items = [dict(item) for bucket in self._items.values() for item in bucket]
            by_kind: Dict[str, int] = {}
            for item in items:
                kind = str(item.get("kind") or "insight")
                by_kind[kind] = by_kind.get(kind, 0) + 1
            recent = sorted(items, key=lambda x: float(x.get("last_seen_at") or 0), reverse=True)[:12]
        return {
            "enabled": True,
            "system": "long_horizon_memory",
            "session_id": session_id,
            "count": len(items),
            "by_kind": by_kind,
            "recent": recent,
        }


_DEFAULT_LONG_HORIZON_BACKEND = InMemoryLongHorizonBackend()


def get_default_long_horizon_backend() -> InMemoryLongHorizonBackend:
    return _DEFAULT_LONG_HORIZON_BACKEND


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
            long_horizon=get_default_long_horizon_backend(),
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
        long_horizon_matches: List[Dict[str, Any]] = []

        if skip_memory:
            trace.retrieval_instruction_applied = bool(merged_instruction)
            return RecallBundle(
                retrieval_instruction=merged_instruction,
                memory_summaries=memory_summaries,
                episodic_matches=episodic_matches,
                expanded_matches=expanded_matches,
                long_horizon_matches=long_horizon_matches,
                trace=trace,
            )

        if self.backends.long_horizon and self.policy.layer_enabled("long_horizon"):
            try:
                plan = self.backends.long_horizon.build_recall_plan(
                    query,
                    mode=mode,
                    enable_thinking=enable_thinking,
                )
                trace.recall_plan = dict(plan or {})
                trace.events.append({
                    "type": "recall_plan",
                    "question_type": trace.recall_plan.get("question_type", "general"),
                    "layers": trace.recall_plan.get("layers", []),
                })
                long_horizon_matches = self.backends.long_horizon.retrieve(
                    session_id,
                    query,
                    scopes=self.policy.long_horizon_scopes,
                    top_k=self.policy.long_horizon_top_k,
                    min_confidence=self.policy.long_horizon_min_confidence,
                )
                if long_horizon_matches:
                    long_horizon_instruction = self.backends.long_horizon.build_instruction(
                        long_horizon_matches,
                        limit=self.policy.long_horizon_instruction_limit,
                    )
                    merged_instruction = self._merge_instructions(
                        merged_instruction,
                        long_horizon_instruction,
                    )
                    memory_summaries.append({
                        "source": "long_horizon",
                        "kind": "consolidated_insight",
                        "count": len(long_horizon_matches),
                    })
                    trace.memory_context_injected = True
                    trace.events.append({
                        "type": "long_horizon_recall",
                        "count": len(long_horizon_matches),
                    })
            except Exception as exc:
                _log.warning("[recall] long-horizon recall failed: %s", exc)

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
        trace.long_horizon_hits = len(long_horizon_matches)
        trace.retrieval_instruction_applied = bool(merged_instruction)
        return RecallBundle(
            retrieval_instruction=merged_instruction,
            memory_summaries=memory_summaries,
            episodic_matches=episodic_matches,
            expanded_matches=expanded_matches,
            long_horizon_matches=long_horizon_matches,
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
            "long_horizon_insight_added": False,
            "long_horizon_insight": None,
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

        if self.backends.long_horizon and self.policy.layer_enabled("long_horizon"):
            try:
                insight = self.backends.long_horizon.consolidate(
                    session_id,
                    question,
                    answer,
                    concepts=concepts,
                    scope=self.policy.long_horizon_write_scope,
                    timestamp=time.time(),
                    matched_expanded=matched_expanded,
                )
                if insight:
                    result["long_horizon_insight_added"] = True
                    result["long_horizon_insight"] = insight
            except Exception as exc:
                _log.warning("[after_response] long-horizon consolidation failed: %s", exc)

        result["elapsed_seconds"] = round(time.time() - t0, 3)
        result["memory_trace"] = {
            "memory_mode": "session",
            "consolidation": {
                "conversation_updated": bool(result["claw_updated"]),
                "episode_added": bool(result["episode_added"]),
                "chat_turn_ingested": bool(result["chat_turn_ingested"]),
                "expanded_promoted": list(result["expanded_promoted"]),
                "long_horizon_insight_added": bool(result["long_horizon_insight_added"]),
                "long_horizon_insight": result["long_horizon_insight"],
                "elapsed_seconds": result["elapsed_seconds"],
                "enabled_layers": sorted(self.policy.enabled_layer_set()),
            },
        }
        return result
