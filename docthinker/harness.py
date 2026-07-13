"""Thin query harness for request-scoped policy and execution controls.

The harness deliberately does not own retrieval or generation.  It gives the
existing components one place to decide which optional capabilities are
allowed for a run, keeping those decisions out of HTTP routers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from docthinker.memory_core.core import MemoryTrace


@dataclass(frozen=True)
class QueryControls:
    use_memory: bool = True
    use_conversation_context: bool = True
    use_llm_cache: bool = True
    use_self_evolution: bool = True
    remember_turn: bool = True

    @classmethod
    def from_request(cls, request: Any) -> "QueryControls":
        return cls(
            use_memory=bool(getattr(request, "use_memory", True)),
            use_conversation_context=bool(
                getattr(request, "use_conversation_context", True)
            ),
            use_llm_cache=bool(getattr(request, "use_llm_cache", True)),
            use_self_evolution=bool(
                getattr(request, "use_self_evolution", True)
            ),
            remember_turn=bool(getattr(request, "remember_turn", True)),
        )

    def to_schema(self) -> Dict[str, bool]:
        return {
            "use_memory": self.use_memory,
            "use_conversation_context": self.use_conversation_context,
            "use_llm_cache": self.use_llm_cache,
            "use_self_evolution": self.use_self_evolution,
            "remember_turn": self.remember_turn,
        }


@dataclass
class QueryRunContext:
    controls: QueryControls
    retrieval_instruction: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    memory_summaries: List[Dict[str, Any]] = field(default_factory=list)
    episodic_matches: List[Dict[str, Any]] = field(default_factory=list)
    expanded_matches: List[Dict[str, Any]] = field(default_factory=list)
    long_horizon_matches: List[Dict[str, Any]] = field(default_factory=list)
    cognition_matches: List[Dict[str, Any]] = field(default_factory=list)
    memory_reasoning: Dict[str, Any] = field(default_factory=dict)
    trace: MemoryTrace = field(default_factory=MemoryTrace)

    def graph_query_options(self, request: Any) -> Dict[str, Any]:
        """Translate one run into the options understood by DocThinker."""
        use_evolution = self.controls.use_self_evolution
        return {
            "enable_rerank": request.enable_rerank,
            "top_k": request.top_k,
            "chunk_top_k": request.chunk_top_k,
            "max_relation_tokens": request.max_relation_tokens,
            "max_total_tokens": request.max_total_tokens,
            "include_discovered_edges": bool(
                use_evolution and request.include_discovered_edges
            ),
            "max_relations": request.max_relations,
            "max_discovered_relations": request.max_discovered_relations,
            "min_discovered_edge_confidence": request.min_discovered_edge_confidence,
            "require_discovered_evidence": request.require_discovered_evidence,
            "enable_image_asset_activation": request.enable_image_asset_activation,
            "image_activation_threshold": request.image_activation_threshold,
            "image_activation_top_k": request.image_activation_top_k,
            "user_prompt": self.retrieval_instruction or None,
            "conversation_history": self.conversation_history,
            "use_llm_cache": self.controls.use_llm_cache,
        }


class QueryHarness:
    """Prepare optional context for a single query run."""

    def __init__(
        self,
        *,
        memory_core_factory: Callable[[], Any],
        history_loader: Callable[[Optional[str]], List[Dict[str, str]]],
    ) -> None:
        self._memory_core_factory = memory_core_factory
        self._history_loader = history_loader

    async def prepare(
        self,
        *,
        request: Any,
        skip_memory: bool = False,
    ) -> QueryRunContext:
        controls = QueryControls.from_request(request)
        context = QueryRunContext(
            controls=controls,
            retrieval_instruction=str(request.retrieval_instruction or "").strip(),
        )

        if controls.use_conversation_context:
            context.conversation_history = self._history_loader(request.session_id)

        if not controls.use_memory or skip_memory:
            context.trace.memory_mode = "off"
            context.trace.retrieval_instruction_applied = bool(
                context.retrieval_instruction
            )
            context.trace.events.append({
                "type": "memory_skipped",
                "reason": "request_disabled" if not controls.use_memory else "identity_query",
            })
            return context

        recall = await self._memory_core_factory().recall(
            session_id=request.session_id,
            query=request.question,
            base_instruction=context.retrieval_instruction,
            mode=request.mode,
            enable_thinking=request.enable_thinking,
            enable_expanded_matching=bool(
                controls.use_self_evolution and request.enable_expanded_matching
            ),
            expanded_top_k=request.expanded_top_k,
            expanded_min_score=request.expanded_min_score,
            skip_memory=False,
        )
        context.retrieval_instruction = recall.retrieval_instruction
        context.memory_summaries = recall.memory_summaries
        context.episodic_matches = recall.episodic_matches
        context.expanded_matches = recall.expanded_matches
        context.long_horizon_matches = recall.long_horizon_matches
        context.cognition_matches = recall.cognition_matches
        context.memory_reasoning = recall.memory_reasoning
        context.trace = recall.trace
        return context

    @staticmethod
    def should_enrich(context: QueryRunContext) -> bool:
        return bool(context.controls.use_memory and context.controls.remember_turn)
