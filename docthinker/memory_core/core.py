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
    memory_reasoning: Dict[str, Any] = field(default_factory=dict)
    write_policy: Dict[str, Any] = field(default_factory=dict)
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
        if self.memory_reasoning:
            lines.append(f"memory_reasoning: {self.memory_reasoning.get('mode', 'derived')}")
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
                "memory_reasoning": dict(self.memory_reasoning or {}),
            },
            "events": list(self.events),
            "write_policy": dict(self.write_policy or {}),
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
    memory_reasoning: Dict[str, Any] = field(default_factory=dict)
    trace: MemoryTrace = field(default_factory=MemoryTrace)


class InMemoryLongHorizonBackend:
    """Process-local long-horizon memory backend.

    It keeps the memory-core contract useful out of the box while remaining
    easy to replace with SQLite, vector storage, or a graph-backed backend.
    """

    def __init__(self) -> None:
        self._items: Dict[str, List[Dict[str, Any]]] = {}
        self._last_decision: Dict[str, Any] = {"action": "init"}
        self._lock = threading.RLock()

    @staticmethod
    def _tokens(text: str) -> set[str]:
        source = str(text or "").lower()
        tokens: set[str] = set()
        for match in re.finditer(r"[\u4e00-\u9fff]{2,8}|[a-z][a-z0-9_\-]{2,32}", source):
            tokens.add(match.group(0))
        for match in re.finditer(r"[\u4e00-\u9fff]{2,}", source):
            chunk = match.group(0)
            for idx in range(max(0, len(chunk) - 1)):
                tokens.add(chunk[idx:idx + 2])
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
        text = f"{q}\n{a}"
        if any(k in text for k in ("用户说", "user prefers", "prefer", "喜欢", "不要", "别", "风格")):
            return "user_preference"
        if any(k in text for k in ("反馈", "feedback", "规则", "rule", "准则")):
            return "feedback_rule"
        if any(k in text for k in ("reference", "引用", "资料", "论文", "文档来源")):
            return "reference_pointer"
        if any(k in q for k in ("喜欢", "prefer", "不要", "别", "风格")):
            return "user_preference"
        if any(k in q for k in ("修复", "实现", "改", "优化", "todo", "next", "roadmap")):
            return "project_state"
        if any(k in a for k in ("because", "therefore", "因为", "所以", "推理")):
            return "reasoning_trace"
        return "insight"

    @staticmethod
    def _contains_secret(text: str) -> bool:
        source = str(text or "")
        secret_patterns = (
            r"sk-[A-Za-z0-9_\-]{20,}",
            r"gh[pousr]_[A-Za-z0-9_]{20,}",
            r"AKIA[0-9A-Z]{16}",
            r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[^\s]{8,}",
        )
        return any(re.search(pattern, source) for pattern in secret_patterns)

    @staticmethod
    def _explicit_memory_request(question: str) -> bool:
        q = str(question or "").lower()
        return any(k in q for k in ("remember", "记住", "作为记忆", "保存到记忆", "长期记忆"))

    def _write_decision(
        self,
        question: str,
        answer: str,
        concepts: Sequence[str],
        *,
        summary: str,
    ) -> Dict[str, Any]:
        text = f"{question}\n{answer}"
        if self._contains_secret(text):
            return {
                "action": "skip",
                "reason": "secret_guard",
                "detail": "possible API key, token, password, or cloud credential",
            }
        if not summary or len(summary) < 24:
            return {"action": "skip", "reason": "too_short"}
        negative = ("抱歉", "不知道", "没有检索到", "无法回答", "i don't know", "sorry")
        if any(marker in summary.lower() for marker in negative):
            return {"action": "skip", "reason": "low_information_answer"}

        explicit = self._explicit_memory_request(question)
        ephemeral_markers = (
            "临时", "temporary", "just this once", "这次", "本次",
            "debug log", "stack trace", "报错日志", "日志",
            "git diff", "git status", "commit hash", "文件路径",
            "line ", "第几行", "/tmp/", "node_modules",
        )
        durable_markers = (
            "长期", "架构", "原则", "偏好", "preference", "rule", "规则",
            "roadmap", "设计", "可控", "memory", "记忆", "reasoning", "推理",
        )
        lowered = text.lower()
        if not explicit and any(marker in lowered for marker in ephemeral_markers):
            return {"action": "skip", "reason": "ephemeral_or_verification_needed"}
        if explicit or any(marker in lowered for marker in durable_markers) or concepts:
            return {"action": "store", "reason": "durable_signal"}
        return {"action": "skip", "reason": "no_durable_signal"}

    def last_write_decision(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._last_decision)

    @staticmethod
    def _candidate_text(item: Dict[str, Any]) -> str:
        parts: List[str] = [
            str(item.get("id") or ""),
            str(item.get("summary") or ""),
            str(item.get("question") or ""),
            str(item.get("kind") or ""),
            str(item.get("scope") or ""),
            " ".join(str(c) for c in item.get("concepts") or []),
            " ".join(str(c) for c in item.get("expanded_entities") or []),
        ]
        return " ".join(part for part in parts if part)

    @staticmethod
    def _infer_edit_action(instruction: str) -> str:
        text = str(instruction or "").lower()
        if any(k in text for k in ("删除", "忘记", "移除", "delete", "remove", "forget")):
            return "delete"
        return "update"

    @staticmethod
    def _extract_rewrite_summary(instruction: str) -> str:
        text = re.sub(r"\s+", " ", str(instruction or "")).strip()
        patterns = (
            r"(?:改成|改为|更新为|设为|写成|总结为|rewrite to|change to|update to)[:：]?\s*(.+)$",
            r"(?:新的内容|new summary)[:：]\s*(.+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip(" 。.!?;；")
        return ""

    def _score_edit_candidate(self, instruction: str, item: Dict[str, Any]) -> Dict[str, Any]:
        query_tokens = self._tokens(instruction)
        item_tokens = self._tokens(self._candidate_text(item))
        overlap = query_tokens & item_tokens
        exact_boost = 0.0
        lowered = str(instruction or "").lower()
        for value in (
            item.get("id"),
            item.get("summary"),
            item.get("question"),
            *(item.get("concepts") or []),
            *(item.get("expanded_entities") or []),
        ):
            value_text = str(value or "").lower().strip()
            if value_text and value_text in lowered:
                exact_boost += 0.25
        confidence = float(item.get("confidence") or 0.0)
        token_score = (len(overlap) / max(1, len(query_tokens))) if query_tokens else 0.0
        score = min(1.0, token_score * 0.7 + min(0.2, confidence * 0.2) + exact_boost)
        return {
            "score": round(score, 3),
            "matched_terms": sorted(overlap)[:12],
        }

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

    def reason_over_memory(
        self,
        query: str,
        matches: Sequence[Dict[str, Any]],
        *,
        plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Derive a compact inference layer from recalled memory."""
        question_type = str((plan or {}).get("question_type") or self._classify_query(query))
        if not matches:
            return {
                "mode": "memory_reasoning",
                "question_type": question_type,
                "conclusions": [],
                "constraints": [],
                "open_questions": ["no_long_horizon_match"],
            }
        preferences: List[str] = []
        tasks: List[str] = []
        reasoning: List[str] = []
        for item in matches:
            summary = str(item.get("summary") or "").strip()
            kind = str(item.get("kind") or "insight")
            if not summary:
                continue
            if kind in {"preference", "user_preference", "feedback_rule"}:
                preferences.append(summary)
            elif kind in {"task_memory", "project_state"}:
                tasks.append(summary)
            else:
                reasoning.append(summary)
        conclusions: List[str] = []
        if preferences:
            conclusions.append("preserve_user_preferences")
        if tasks:
            conclusions.append("continue_project_state")
        if reasoning:
            conclusions.append("reuse_prior_reasoning")
        return {
            "mode": "memory_reasoning",
            "question_type": question_type,
            "conclusions": conclusions or ["use_recalled_context"],
            "constraints": (preferences + tasks + reasoning)[:5],
            "open_questions": [],
        }

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
        decision = self._write_decision(question, answer, concepts, summary=summary)
        with self._lock:
            self._last_decision = {
                **decision,
                "scope": str(scope or "session"),
                "question": str(question or "")[:120],
            }
        if decision.get("action") != "store":
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
                    item["audit"] = {"last_decision": decision.get("reason"), "source": "after_response"}
                    self._last_decision = {
                        **decision,
                        "action": "update",
                        "memory_id": item.get("id"),
                        "kind": item.get("kind"),
                        "scope": item.get("scope"),
                    }
                    return dict(item)

            item = {
                "id": f"lh-{scope_name}-{int(timestamp * 1000)}-{len(bucket) + 1}",
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
                "audit": {
                    "last_decision": decision.get("reason"),
                    "source": "after_response",
                    "schema": "docthinker.long_horizon.v1",
                },
            }
            bucket.append(item)
            if len(bucket) > 200:
                del bucket[: len(bucket) - 200]
            self._last_decision = {
                **decision,
                "action": "store",
                "memory_id": item.get("id"),
                "kind": item.get("kind"),
                "scope": item.get("scope"),
            }
            return dict(item)

    def list_insights(
        self,
        session_id: Optional[str] = None,
        *,
        scope: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            if scope:
                keys = [self._scope_key(session_id, scope)]
            elif session_id:
                keys = [
                    self._scope_key(session_id, "session"),
                    self._scope_key(session_id, "project"),
                    self._scope_key(session_id, "user"),
                ]
            else:
                keys = list(self._items.keys())
            items = [dict(item) for key in keys for item in self._items.get(key, [])]
        items.sort(key=lambda item: float(item.get("last_seen_at") or 0), reverse=True)
        return items[: max(0, int(limit))]

    def delete_insight(self, memory_id: str, session_id: Optional[str] = None) -> bool:
        target = str(memory_id or "").strip()
        if not target:
            return False
        with self._lock:
            keys = (
                [
                    self._scope_key(session_id, "session"),
                    self._scope_key(session_id, "project"),
                    self._scope_key(session_id, "user"),
                ]
                if session_id
                else list(self._items.keys())
            )
            for key in keys:
                bucket = self._items.get(key, [])
                kept = [item for item in bucket if str(item.get("id")) != target]
                if len(kept) != len(bucket):
                    self._items[key] = kept
                    self._last_decision = {
                        "action": "delete",
                        "reason": "user_controlled_memory_management",
                        "memory_id": target,
                    }
                    return True
        return False

    def update_insight(
        self,
        memory_id: str,
        patch: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        target = str(memory_id or "").strip()
        if not target:
            return None
        allowed = {"summary", "question", "kind", "concepts", "expanded_entities", "confidence"}
        clean_patch = {key: value for key, value in (patch or {}).items() if key in allowed}
        if not clean_patch:
            return None

        with self._lock:
            keys = (
                [
                    self._scope_key(session_id, "session"),
                    self._scope_key(session_id, "project"),
                    self._scope_key(session_id, "user"),
                ]
                if session_id
                else list(self._items.keys())
            )
            for key in keys:
                for item in self._items.get(key, []):
                    if str(item.get("id")) != target:
                        continue
                    if "summary" in clean_patch:
                        item["summary"] = self._clean_summary(str(clean_patch["summary"])) or str(item.get("summary") or "")
                    if "question" in clean_patch:
                        item["question"] = str(clean_patch["question"] or "")[:260]
                    if "kind" in clean_patch:
                        item["kind"] = str(clean_patch["kind"] or "insight")[:64]
                    if "concepts" in clean_patch and isinstance(clean_patch["concepts"], (list, tuple)):
                        item["concepts"] = [str(c) for c in clean_patch["concepts"][:16] if str(c).strip()]
                    if "expanded_entities" in clean_patch and isinstance(clean_patch["expanded_entities"], (list, tuple)):
                        item["expanded_entities"] = [str(c) for c in clean_patch["expanded_entities"][:12] if str(c).strip()]
                    if "confidence" in clean_patch:
                        try:
                            item["confidence"] = max(0.05, min(0.99, float(clean_patch["confidence"])))
                        except (TypeError, ValueError):
                            pass
                    item["last_seen_at"] = time.time()
                    item["use_count"] = int(item.get("use_count") or 0) + 1
                    item["audit"] = {
                        **(item.get("audit") or {}),
                        "last_decision": "natural_language_edit",
                        "source": "memory_editor",
                    }
                    self._last_decision = {
                        "action": "update",
                        "reason": "natural_language_memory_edit",
                        "memory_id": target,
                        "fields": sorted(clean_patch),
                    }
                    return dict(item)
        return None

    def plan_edit(
        self,
        session_id: Optional[str],
        instruction: str,
        *,
        scope: Optional[str] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        action = self._infer_edit_action(instruction)
        rewrite_summary = self._extract_rewrite_summary(instruction)
        candidates: List[Dict[str, Any]] = []
        for item in self.list_insights(session_id=session_id, scope=scope, limit=500):
            match = self._score_edit_candidate(instruction, item)
            if match["score"] <= 0:
                continue
            candidate = dict(item)
            candidate["match"] = match
            candidate["proposed_action"] = action
            candidate["suggested_patch"] = (
                {"summary": rewrite_summary}
                if action == "update" and rewrite_summary
                else {}
            )
            candidates.append(candidate)
        candidates.sort(key=lambda item: item["match"]["score"], reverse=True)
        return {
            "instruction": instruction,
            "session_id": session_id,
            "scope": scope,
            "action": action,
            "strategy": "token_embedding_fallback",
            "candidates": candidates[: max(0, int(limit))],
            "suggested_patch": {"summary": rewrite_summary} if rewrite_summary else {},
        }

    def export_markdown(self, session_id: Optional[str] = None) -> str:
        items = self.list_insights(session_id=session_id, limit=500)
        lines = [
            "# DocThinker MEMORY.md",
            "",
            "This is an auditable index of DocThinker long-horizon memory. It is generated from agentic memory records, not used as the only source of truth.",
            "",
            "## What Not To Save",
            "",
            "- Secrets, API keys, passwords, or credentials.",
            "- One-off debug traces, transient file paths, git history, and already documented code facts.",
            "- User content explicitly marked as not memory.",
            "",
        ]
        if not items:
            lines.extend(["## Memories", "", "_No long-horizon memories yet._", ""])
            return "\n".join(lines)
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in items:
            grouped.setdefault(str(item.get("kind") or "insight"), []).append(item)
        for kind in sorted(grouped):
            lines.extend([f"## {kind}", ""])
            for item in grouped[kind]:
                scope = str(item.get("scope") or "session")
                confidence = float(item.get("confidence") or 0.0)
                summary = str(item.get("summary") or "").strip()
                concepts = ", ".join(str(c) for c in (item.get("concepts") or [])[:5])
                suffix = f" Concepts: {concepts}." if concepts else ""
                lines.append(f"- `{item.get('id')}` [{scope}, confidence {confidence:.2f}] {summary}{suffix}")
            lines.append("")
        return "\n".join(lines)

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
            "last_write_decision": self.last_write_decision(),
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

    @staticmethod
    def _format_memory_reasoning_instruction(memory_reasoning: Dict[str, Any]) -> str:
        constraints = [
            str(item).strip()
            for item in (memory_reasoning.get("constraints") or [])
            if str(item).strip()
        ]
        conclusions = [
            str(item).strip()
            for item in (memory_reasoning.get("conclusions") or [])
            if str(item).strip()
        ]
        if not constraints and not conclusions:
            return ""
        lines = [
            "## Memory-side reasoning",
            "The memory layer has already reasoned over recalled insights. Treat these as continuity constraints, not as standalone facts.",
            "",
        ]
        for item in conclusions[:4]:
            lines.append(f"- conclusion: {item}")
        for item in constraints[:5]:
            lines.append(f"- constraint: {item[:220]}")
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
        memory_reasoning: Dict[str, Any] = {}

        if skip_memory:
            trace.retrieval_instruction_applied = bool(merged_instruction)
            return RecallBundle(
                retrieval_instruction=merged_instruction,
                memory_summaries=memory_summaries,
                episodic_matches=episodic_matches,
                expanded_matches=expanded_matches,
                long_horizon_matches=long_horizon_matches,
                memory_reasoning=memory_reasoning,
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
                    memory_reasoning = self.backends.long_horizon.reason_over_memory(
                        query,
                        long_horizon_matches,
                        plan=trace.recall_plan,
                    )
                    reasoning_instruction = self._format_memory_reasoning_instruction(
                        memory_reasoning,
                    )
                    merged_instruction = self._merge_instructions(
                        merged_instruction,
                        long_horizon_instruction,
                        reasoning_instruction,
                    )
                    memory_summaries.append({
                        "source": "long_horizon",
                        "kind": "consolidated_insight",
                        "count": len(long_horizon_matches),
                        "reasoned": bool(memory_reasoning.get("conclusions")),
                    })
                    trace.memory_context_injected = True
                    trace.memory_reasoning = dict(memory_reasoning or {})
                    trace.events.append({
                        "type": "long_horizon_recall",
                        "count": len(long_horizon_matches),
                    })
                    trace.events.append({
                        "type": "memory_reasoning",
                        "question_type": memory_reasoning.get("question_type"),
                        "conclusions": memory_reasoning.get("conclusions", []),
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
            memory_reasoning=memory_reasoning,
            trace=trace,
        )

    async def after_response(
        self,
        *,
        session_id: Optional[str],
        question: str,
        answer: str,
        matched_expanded: Optional[Sequence[Dict[str, Any]]] = None,
        remember: bool = True,
        excluded_layers: Optional[Sequence[str]] = None,
        write_scope: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Consolidate memories after a response has been produced."""

        if not session_id or not answer:
            return {"updated": False}

        t0 = time.time()
        excluded = set(str(layer) for layer in (excluded_layers or ()))
        excluded.update(str(layer) for layer in self.policy.write_excluded_layers)
        writes_allowed = bool(remember and self.policy.allow_memory_writes)
        result: Dict[str, Any] = {
            "updated": writes_allowed,
            "memory_write_skipped": not writes_allowed,
            "excluded_layers": sorted(excluded),
            "claw_updated": False,
            "episode_added": False,
            "chat_turn_ingested": False,
            "expanded_promoted": [],
            "long_horizon_insight_added": False,
            "long_horizon_insight": None,
            "long_horizon_write_decision": {},
        }

        if not writes_allowed:
            result["elapsed_seconds"] = round(time.time() - t0, 3)
            result["memory_trace"] = {
                "memory_mode": "session",
                "write_policy": {
                    "remember": bool(remember),
                    "writes_allowed": False,
                    "excluded_layers": sorted(excluded),
                },
                "consolidation": {
                    "skipped": True,
                    "reason": "memory_writes_disabled",
                    "enabled_layers": sorted(self.policy.enabled_layer_set()),
                },
            }
            return result

        if (
            self.backends.conversation
            and self.policy.layer_enabled("conversation")
            and "conversation" not in excluded
        ):
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

        if (
            self.backends.episodic
            and self.policy.layer_enabled("episodic")
            and "episodic" not in excluded
        ):
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

        if (
            self.backends.chat_turn
            and self.policy.layer_enabled("chat_turn")
            and "chat_turn" not in excluded
        ):
            try:
                result["chat_turn_ingested"] = await self.backends.chat_turn.ingest(
                    session_id,
                    question,
                    answer,
                    timestamp=time.time(),
                )
            except Exception as exc:
                _log.warning("[after_response] chat turn ingest failed: %s", exc)

        if (
            matched_expanded
            and self.backends.expanded
            and self.policy.layer_enabled("expanded")
            and "expanded" not in excluded
        ):
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

        if (
            self.backends.long_horizon
            and self.policy.layer_enabled("long_horizon")
            and "long_horizon" not in excluded
        ):
            try:
                insight = self.backends.long_horizon.consolidate(
                    session_id,
                    question,
                    answer,
                    concepts=concepts,
                    scope=write_scope or self.policy.long_horizon_write_scope,
                    timestamp=time.time(),
                    matched_expanded=matched_expanded,
                )
                if insight:
                    result["long_horizon_insight_added"] = True
                    result["long_horizon_insight"] = insight
                if hasattr(self.backends.long_horizon, "last_write_decision"):
                    result["long_horizon_write_decision"] = self.backends.long_horizon.last_write_decision()
            except Exception as exc:
                _log.warning("[after_response] long-horizon consolidation failed: %s", exc)

        result["elapsed_seconds"] = round(time.time() - t0, 3)
        result["memory_trace"] = {
            "memory_mode": "session",
            "write_policy": {
                "remember": bool(remember),
                "writes_allowed": True,
                "excluded_layers": sorted(excluded),
                "write_scope": write_scope or self.policy.long_horizon_write_scope,
            },
            "consolidation": {
                "conversation_updated": bool(result["claw_updated"]),
                "episode_added": bool(result["episode_added"]),
                "chat_turn_ingested": bool(result["chat_turn_ingested"]),
                "expanded_promoted": list(result["expanded_promoted"]),
                "long_horizon_insight_added": bool(result["long_horizon_insight_added"]),
                "long_horizon_insight": result["long_horizon_insight"],
                "long_horizon_write_decision": result["long_horizon_write_decision"],
                "elapsed_seconds": result["elapsed_seconds"],
                "enabled_layers": sorted(self.policy.enabled_layer_set()),
            },
        }
        return result
