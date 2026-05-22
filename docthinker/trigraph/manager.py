"""Coordinator for causal DAG, activation flow, and self-evolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional

from docthinker.causal import CausalDAG, CausalExtractor
from docthinker.flow import KnowledgeFlowDynamics
from docthinker.seal import SEALEvolution

from .interaction import CrossGraphInteraction


class TriGraphManager:
    def __init__(
        self,
        *,
        knowledge_dir: Path | str,
        llm_func: Callable[..., Awaitable[str]],
    ) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.llm_func = llm_func
        self.causal_dag = CausalDAG(self.knowledge_dir / "causal_dag.json")
        self.flow_dynamics = KnowledgeFlowDynamics(self.knowledge_dir / "flow.json")
        self.cross_graph = CrossGraphInteraction(
            causal_dag=self.causal_dag,
            flow_dynamics=self.flow_dynamics,
        )
        self.seal_evolution = SEALEvolution(
            llm_func=llm_func,
            causal_dag=self.causal_dag,
        )

    async def build_causal_from_text(
        self,
        text: str,
        *,
        source_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        extractor = CausalExtractor(llm_func=self.llm_func, dag=self.causal_dag, min_chars=20)
        result = await extractor.extract_from_text(text, source_id=source_id)
        self.causal_dag.save()
        return result

    async def deep_mode_query_context(
        self,
        *,
        question: str,
        episodic_concepts: Iterable[str] = (),
    ) -> Dict[str, Any]:
        concepts = [str(item) for item in episodic_concepts if str(item).strip()]
        causal_context = self.cross_graph.build_causal_context(concepts, max_chains=5, max_depth=3)
        activated = self.flow_dynamics.propagate_causal(self.causal_dag._forward, concepts)
        return {
            "question": question,
            "causal_context": causal_context,
            "flow_activated": activated,
            "top_activated": self.flow_dynamics.get_top_activated(5),
        }

    async def post_query_evolution(
        self,
        *,
        question: str,
        answer: str,
        semantic_entities: Iterable[str] = (),
    ) -> Dict[str, Any]:
        result = await self.seal_evolution.run(
            question=question,
            answer=answer,
            semantic_entities=semantic_entities,
        )
        self.save_all()
        return result

    def save_all(self) -> None:
        self.causal_dag.save()
        self.flow_dynamics.save()
