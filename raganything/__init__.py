from .raganything import RAGAnything as RAGAnything
from .config import RAGAnythingConfig as RAGAnythingConfig
from .auto_thinking import (
    HybridRAGOrchestrator,
    ComplexityClassifier,
    ComplexityVote,
    VLMClient,
)

__version__ = "1.2.8"
__author__ = "Zirui Guo"
__url__ = "https://github.com/HKUDS/RAG-Anything"

__all__ = [
    "RAGAnything",
    "RAGAnythingConfig",
    "HybridRAGOrchestrator",
    "ComplexityClassifier",
    "ComplexityVote",
    "VLMClient",
]
