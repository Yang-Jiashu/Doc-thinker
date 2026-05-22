"""Minimal SEAL-style loop for validating causal memory hypotheses."""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable, Dict, Iterable, List

from docthinker.causal import CausalDAG


class SEALEvolution:
    def __init__(self, *, llm_func: Callable[..., Awaitable[str]], causal_dag: CausalDAG):
        self.llm_func = llm_func
        self.causal_dag = causal_dag

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        source = str(text or "").strip()
        try:
            return json.loads(source)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", source, re.DOTALL)
            return json.loads(match.group(0)) if match else {}

    async def run(
        self,
        *,
        question: str,
        answer: str,
        semantic_entities: Iterable[str] = (),
    ) -> Dict[str, Any]:
        assessment_raw = await self.llm_func(
            "Assess memory quality for this answer.\n"
            f"Question: {question}\nAnswer: {answer}"
        )
        assessment = self._parse_json(assessment_raw)

        hypothesis_raw = await self.llm_func(
            "Augment memory with hypotheses as JSON key hypotheses.\n"
            f"Entities: {list(semantic_entities)}\nQuestion: {question}\nAnswer: {answer}"
        )
        hypotheses = self._parse_json(hypothesis_raw).get("hypotheses") or []

        validate_raw = await self.llm_func(
            "Validate causal hypotheses as JSON key validated.\n"
            f"Hypotheses: {json.dumps(hypotheses, ensure_ascii=False)}"
        )
        validated = self._parse_json(validate_raw).get("validated") or []

        inserted = 0
        for item in validated:
            edge = self.causal_dag.add_edge(
                item.get("cause") or "",
                item.get("effect") or "",
                mechanism=item.get("mechanism") or "validated_hypothesis",
                strength=item.get("validity_score") or item.get("confidence") or 0.7,
                evidence=item.get("reason") or "",
                source_id="seal",
            )
            if edge is not None:
                inserted += 1
        return {
            "assessment": assessment,
            "hypotheses": hypotheses,
            "validated": validated,
            "inserted": inserted,
        }
