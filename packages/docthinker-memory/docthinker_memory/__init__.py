"""Standalone import path for the DocThinker agentic memory facade.

This package is intentionally thin: it gives plugin authors and agent
frameworks a dependency target that is smaller and clearer than the full
DocThinker application surface.
"""

from docthinker.memory_core import (
    AgentMemoryBackends,
    AgentMemoryCore,
    ChatTurnBackend,
    ConversationMemoryBackend,
    EpisodicMemoryBackend,
    ExpandedKnowledgeBackend,
    GraphPromotionBackend,
    MemoryPolicy,
    MemoryTrace,
    RecallBundle,
)

__all__ = [
    "AgentMemoryBackends",
    "AgentMemoryCore",
    "ChatTurnBackend",
    "ConversationMemoryBackend",
    "EpisodicMemoryBackend",
    "ExpandedKnowledgeBackend",
    "GraphPromotionBackend",
    "MemoryPolicy",
    "MemoryTrace",
    "RecallBundle",
]
