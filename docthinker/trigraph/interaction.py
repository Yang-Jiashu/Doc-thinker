"""Interactions between semantic, causal, and activation-flow memory."""

from __future__ import annotations

from typing import Dict, Iterable, List

from docthinker.causal import CausalDAG
from docthinker.flow import KnowledgeFlowDynamics


class CrossGraphInteraction:
    def __init__(self, *, causal_dag: CausalDAG, flow_dynamics: KnowledgeFlowDynamics):
        self.causal_dag = causal_dag
        self.flow_dynamics = flow_dynamics

    def build_causal_context(
        self,
        entities: Iterable[str],
        *,
        max_chains: int = 5,
        max_depth: int = 3,
    ) -> str:
        entity_list = [str(entity) for entity in entities if str(entity).strip()]
        self.flow_dynamics.record_batch_access(entity_list)
        chains = self.causal_dag.get_causal_chains(
            entity_list,
            max_depth=max_depth,
            max_chains=max_chains,
        )
        if not chains:
            return "因果推理链\n- " + "、".join(entity_list)
        lines = ["因果推理链"]
        for chain in chains[:max_chains]:
            lines.append("- " + " → ".join(chain))
        return "\n".join(lines)

    def verify_semantic_with_causal(self, entities: Iterable[str]) -> Dict[str, object]:
        with_backing: List[str] = []
        without_backing: List[str] = []
        for entity in entities:
            key = str(entity)
            has_backing = (
                key in self.causal_dag.nodes
                or bool(self.causal_dag.get_ancestors(key))
                or bool(self.causal_dag.get_descendants(key))
            )
            if has_backing:
                with_backing.append(key)
            else:
                without_backing.append(key)
        total = len(with_backing) + len(without_backing)
        return {
            "with_causal_backing": with_backing,
            "without_causal_backing": without_backing,
            "coverage_ratio": len(with_backing) / total if total else 0.0,
        }
