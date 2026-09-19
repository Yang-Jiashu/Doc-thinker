"""Thin query harness for request-scoped policy and execution controls.

The harness deliberately does not own retrieval or generation.  It gives the
existing components one place to decide which optional capabilities are
allowed for a run, keeping those decisions out of HTTP routers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Callable, Dict, List, Optional

from docthinker.memory_core.core import MemoryTrace
from docthinker.reasoning_policy import (
    QuestionPolicy,
    classify_question,
    complete_path_from_evidence,
    evidence_constrained_paths,
    exploratory_pagerank,
    format_exploration_instruction,
    format_path_instruction,
)


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
    question_policy: QuestionPolicy = field(
        default_factory=lambda: QuestionPolicy("faithful", 1.0, "default")
    )
    graph_reasoning: Dict[str, Any] = field(default_factory=dict)
    retrieval_instruction: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    memory_summaries: List[Dict[str, Any]] = field(default_factory=list)
    episodic_matches: List[Dict[str, Any]] = field(default_factory=list)
    expanded_matches: List[Dict[str, Any]] = field(default_factory=list)
    long_horizon_matches: List[Dict[str, Any]] = field(default_factory=list)
    memory_reasoning: Dict[str, Any] = field(default_factory=dict)
    trace: MemoryTrace = field(default_factory=MemoryTrace)

    def graph_query_options(self, request: Any) -> Dict[str, Any]:
        """Translate one run into the options understood by DocThinker."""
        # Strict path mode injects only the selected path. It must not reopen
        # the broad discovered-edge pool in GraphCore retrieval.
        use_discovered_pool = self.question_policy.mode == "explore"
        return {
            "enable_rerank": request.enable_rerank,
            "top_k": request.top_k,
            "chunk_top_k": request.chunk_top_k,
            "max_relation_tokens": request.max_relation_tokens,
            "max_total_tokens": request.max_total_tokens,
            "include_discovered_edges": bool(
                self.controls.use_self_evolution
                and use_discovered_pool
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
            question_policy=classify_question(
                request.question,
                requested_mode=getattr(request, "evolution_mode", "auto"),
                self_evolution_enabled=controls.use_self_evolution,
            ),
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
        context.memory_reasoning = recall.memory_reasoning
        context.trace = recall.trace
        return context

    async def enrich_graph_reasoning(
        self,
        *,
        context: QueryRunContext,
        graphcore: Any,
        question: str,
        min_candidate_confidence: float = 0.80,
        llm_func: Any = None,
    ) -> None:
        """Attach path or exploration guidance selected by the question policy."""
        mode = context.question_policy.mode
        context.graph_reasoning = {
            "policy": context.question_policy.to_schema(),
            "applied": False,
        }
        graph = getattr(graphcore, "chunk_entity_relation_graph", None)
        if mode == "faithful" or graph is None:
            return

        nodes = graph.get_all_nodes()
        edges = graph.get_all_edges()
        if inspect.isawaitable(nodes):
            nodes = await nodes
        if inspect.isawaitable(edges):
            edges = await edges
        node_list = list(nodes or [])
        edge_list = list(edges or [])

        instruction = ""
        if mode == "path":
            paths, diagnostic = evidence_constrained_paths(
                question,
                node_list,
                edge_list,
                allow_candidates=context.controls.use_self_evolution,
                min_candidate_confidence=min_candidate_confidence,
            )
            if not paths and len(diagnostic.get("anchors") or []) >= 2:
                chunks = await self._load_gap_evidence_chunks(
                    graphcore,
                    node_list,
                    diagnostic,
                )
                anchors = diagnostic["anchors"]
                completed, completion_diagnostic = await complete_path_from_evidence(
                    question=question,
                    start=anchors[0],
                    goal=anchors[-1],
                    chunks=chunks,
                    llm_func=llm_func,
                    min_confidence=min_candidate_confidence,
                )
                diagnostic["completion"] = completion_diagnostic
                if completed is not None:
                    paths = [completed]
                    diagnostic["reason"] = "query_local_path_completion"
            instruction = format_path_instruction(paths, diagnostic)
            context.graph_reasoning.update({
                "paths": [path.to_schema() for path in paths],
                "diagnostic": diagnostic,
            })
        elif mode == "explore":
            exploration = exploratory_pagerank(
                question,
                node_list,
                edge_list,
                min_candidate_confidence=min_candidate_confidence,
            )
            instruction = format_exploration_instruction(exploration)
            context.graph_reasoning["exploration"] = exploration

        if instruction:
            context.retrieval_instruction = self._merge_instruction(
                context.retrieval_instruction,
                instruction,
            )
            context.graph_reasoning["applied"] = True

    @staticmethod
    async def _load_gap_evidence_chunks(
        graphcore: Any,
        nodes: List[Dict[str, Any]],
        diagnostic: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        storage = getattr(graphcore, "text_chunks", None)
        if storage is None or not hasattr(storage, "get_by_ids"):
            return []
        relevant = set(diagnostic.get("anchors") or [])
        relevant.update(diagnostic.get("reached_frontier") or [])
        chunk_ids: List[str] = []
        for node in nodes:
            name = str(node.get("id") or node.get("entity_id") or "").strip()
            if name not in relevant:
                continue
            raw = str(node.get("source_id") or "")
            for chunk_id in raw.split("<SEP>"):
                chunk_id = chunk_id.strip()
                if chunk_id and chunk_id not in chunk_ids:
                    chunk_ids.append(chunk_id)
        chunk_ids = chunk_ids[:12]
        if not chunk_ids:
            return []
        values = storage.get_by_ids(chunk_ids)
        if inspect.isawaitable(values):
            values = await values
        output: List[Dict[str, Any]] = []
        for chunk_id, value in zip(chunk_ids, list(values or [])):
            if not isinstance(value, dict):
                continue
            item = dict(value)
            item.setdefault("id", chunk_id)
            output.append(item)
        return output

    @staticmethod
    def _merge_instruction(*parts: str) -> str:
        clean = [str(part or "").strip() for part in parts if str(part or "").strip()]
        return "\n\n".join(clean)

    @staticmethod
    def should_enrich(context: QueryRunContext) -> bool:
        return bool(context.controls.use_memory and context.controls.remember_turn)
