from .core import DocThinker as DocThinker
from .config import DocThinkerConfig as DocThinkerConfig
from .auto_thinking import (
    HybridRAGOrchestrator,
    ComplexityClassifier,
    ComplexityVote,
    VLMClient,
)

__version__ = "1.2.8"
__author__ = "Zirui Guo"
__url__ = "https://github.com/Yang-Jiashu/doc-thinker"

__all__ = [
    "DocThinker",
    "DocThinkerConfig",
    "HybridRAGOrchestrator",
    "ComplexityClassifier",
    "ComplexityVote",
    "VLMClient",
]
