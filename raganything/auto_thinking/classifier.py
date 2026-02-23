"""Query complexity classifier for auto-thinking routing."""
#复杂度判断，与建图方式有关。
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Optional

from .prompts import COMPLEXITY_PROMPT
from .vlm_client import VLMClient


@dataclass
class ComplexityVote:
    complexity: str
    confidence: float
    features: Dict[str, float]
    use_hyper: bool


class ComplexityClassifier:
    """Combine heuristics with a lightweight LLM judgement for routing."""

    def __init__(
        self,
        vlm_client: VLMClient,
        *,
        low_length_threshold: int = 6,
        high_length_threshold: int = 15,
    ) -> None:
        self.vlm_client = vlm_client
        self.low_length_threshold = low_length_threshold
        self.high_length_threshold = high_length_threshold

    async def assess(self, query: str) -> ComplexityVote:
        heuristics = self._compute_heuristics(query)

        if heuristics["token_estimate"] <= self.low_length_threshold and not heuristics["has_multi_intent"]:
            return ComplexityVote("low", 0.85, heuristics, False)

        llm_vote = await self._ask_model(query)
        complexity = llm_vote.get("complexity", "medium")
        confidence = float(llm_vote.get("confidence", 0.5))
        reasons = llm_vote.get("reasons", [])

        features = {**heuristics, "confidence": confidence}
        if isinstance(reasons, list):
            features.update({f"reason_{idx}": reason for idx, reason in enumerate(reasons)})

        use_hyper = complexity == "high" or (
            complexity == "medium"
            and confidence < 0.6
            and heuristics["token_estimate"] >= self.high_length_threshold
        )

        return ComplexityVote(complexity, confidence, features, use_hyper)

    async def _ask_model(self, query: str) -> Dict[str, Optional[str]]:
        response = await self.vlm_client.generate(
            COMPLEXITY_PROMPT.format(query=query),
            images=None,
            max_tokens=300,
        )
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON substring
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        return {"complexity": "medium", "confidence": 0.5, "reasons": ["LLM parse failure"]}

    def _compute_heuristics(self, query: str) -> Dict[str, float]:
        words = query.strip().split()
        token_estimate = len(words)
        heuristics = {
            "token_estimate": token_estimate,
            "has_multi_intent": 1.0 if re.search(r"\b(and|或者|同时|分别|compare|versus)\b", query, re.IGNORECASE) else 0.0,
            "has_reasoning_keyword": 1.0
            if re.search(r"(why|cause|impact|trend|plan|workflow|流程|策略|比较)", query, re.IGNORECASE)
            else 0.0,
        }
        return heuristics
