"""Auto-thinking orchestration utilities for DocThinker."""

from __future__ import annotations


def __getattr__(name: str):
    if name in {"ComplexityClassifier", "ComplexityVote"}:
        from .classifier import ComplexityClassifier, ComplexityVote

        return ComplexityClassifier if name == "ComplexityClassifier" else ComplexityVote
    if name in {"QuestionDecomposer", "QuestionPlan", "SubQuestion", "SubQuestionAnswer"}:
        from .decomposer import QuestionDecomposer, QuestionPlan, SubQuestion, SubQuestionAnswer

        return {
            "QuestionDecomposer": QuestionDecomposer,
            "QuestionPlan": QuestionPlan,
            "SubQuestion": SubQuestion,
            "SubQuestionAnswer": SubQuestionAnswer,
        }[name]
    if name == "HybridRAGOrchestrator":
        from .orchestrator import HybridRAGOrchestrator

        return HybridRAGOrchestrator
    if name == "VLMClient":
        from .vlm_client import VLMClient

        return VLMClient
    raise AttributeError(name)

__all__ = [
    "HybridRAGOrchestrator",
    "ComplexityClassifier",
    "ComplexityVote",
    "VLMClient",
    "QuestionDecomposer",
    "QuestionPlan",
    "SubQuestion",
    "SubQuestionAnswer",
]
