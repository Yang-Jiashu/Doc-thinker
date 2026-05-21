"""Public package exports for DocThinker.

Imports are resolved lazily so lightweight subpackages such as
``docthinker.memory_core`` do not pull in server/runtime dependencies.
"""

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
    "AgentMemoryCore",
    "AgentMemoryBackends",
    "RecallBundle",
    "MemoryTrace",
]


def __getattr__(name):
    if name == "DocThinker":
        from .core import DocThinker

        return DocThinker
    if name == "DocThinkerConfig":
        from .config import DocThinkerConfig

        return DocThinkerConfig
    if name in {
        "HybridRAGOrchestrator",
        "ComplexityClassifier",
        "ComplexityVote",
        "VLMClient",
    }:
        from . import auto_thinking

        return getattr(auto_thinking, name)
    if name in {"AgentMemoryCore", "AgentMemoryBackends", "RecallBundle", "MemoryTrace"}:
        from . import memory_core

        return getattr(memory_core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
