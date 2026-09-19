"""Agentic memory core facade.

This package is the stable entrypoint agents should use for memory work.
It wraps the current Claw, neuro/episode, and KG expansion pieces without
forcing a large rewrite of the existing RAG stack.
"""

from .core import (
    AgentMemoryCore,
    InMemoryLongHorizonBackend,
    MemoryTrace,
    RecallBundle,
    get_default_long_horizon_backend,
)
from .protocols import (
    AgentMemoryBackends,
    ChatTurnBackend,
    ConversationMemoryBackend,
    EpisodicMemoryBackend,
    ExpandedKnowledgeBackend,
    GraphPromotionBackend,
    LongHorizonMemoryBackend,
    VersionedLongHorizonMemoryBackend,
    MemoryPolicy,
)
from .sqlite_backend import SQLiteLongHorizonBackend

__all__ = [
    "AgentMemoryCore",
    "InMemoryLongHorizonBackend",
    "SQLiteLongHorizonBackend",
    "RecallBundle",
    "MemoryTrace",
    "get_default_long_horizon_backend",
    "AgentMemoryBackends",
    "ChatTurnBackend",
    "ConversationMemoryBackend",
    "EpisodicMemoryBackend",
    "ExpandedKnowledgeBackend",
    "GraphPromotionBackend",
    "LongHorizonMemoryBackend",
    "VersionedLongHorizonMemoryBackend",
    "MemoryPolicy",
]
