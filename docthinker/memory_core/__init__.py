"""Agentic memory core facade.

This package is the stable entrypoint agents should use for memory work.
It wraps the current Claw, neuro/episode, and KG expansion pieces without
forcing a large rewrite of the existing RAG stack.
"""

from .core import AgentMemoryCore, RecallBundle, MemoryTrace
from .protocols import (
    AgentMemoryBackends,
    ChatTurnBackend,
    ConversationMemoryBackend,
    EpisodicMemoryBackend,
    ExpandedKnowledgeBackend,
    GraphPromotionBackend,
)

__all__ = [
    "AgentMemoryCore",
    "RecallBundle",
    "MemoryTrace",
    "AgentMemoryBackends",
    "ChatTurnBackend",
    "ConversationMemoryBackend",
    "EpisodicMemoryBackend",
    "ExpandedKnowledgeBackend",
    "GraphPromotionBackend",
]
