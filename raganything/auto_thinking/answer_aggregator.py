"""Result aggregation utilities for auto-thinking responses."""
#当同时得到raganything和hypergraphrag的答案时，按置信度选择最终输出。
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RetrievalAnswer:
    content: str
    source: str
    meta: Dict[str, float]


class AnswerAggregator:
    """Select the best answer among multiple backends."""

    def select(self, rag_answer: Optional[RetrievalAnswer], hyper_answer: Optional[RetrievalAnswer]) -> RetrievalAnswer:
        if rag_answer and not hyper_answer:
            return rag_answer
        if hyper_answer and not rag_answer:
            return hyper_answer
        if not rag_answer and not hyper_answer:
            raise ValueError("No answers available to select from.")

        assert rag_answer and hyper_answer
        rag_score = rag_answer.meta.get("confidence", 0.5)
        hyper_score = hyper_answer.meta.get("confidence", 0.6)

        return hyper_answer if hyper_score >= rag_score else rag_answer
