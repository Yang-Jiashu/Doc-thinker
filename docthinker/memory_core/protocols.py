"""Backend contracts for the DocThinker agentic memory layer.

These protocols are intentionally small. They define what a memory backend
must do without forcing downstream projects to depend on DocThinker's server,
GraphCore, Claw, or Neuro Memory implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence


class ConversationMemoryBackend(Protocol):
    """Short/long conversation memory used before and after generation."""

    async def build_context(self, session_id: Optional[str], query: str) -> str:
        """Return memory context to inject into the next retrieval/generation step."""

    async def consolidate(
        self,
        session_id: Optional[str],
        question: str,
        answer: str,
    ) -> bool:
        """Persist a completed turn. Return True when an update happened."""


class EpisodicMemoryBackend(Protocol):
    """Experience memory used for analogy recall and turn-level writes."""

    async def retrieve(
        self,
        session_id: Optional[str],
        query: str,
        *,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Return serialized episodic matches sorted by relevance."""

    async def write(
        self,
        session_id: Optional[str],
        question: str,
        answer: str,
        *,
        concepts: Sequence[str],
        timestamp: float,
    ) -> Optional[str]:
        """Write one completed turn as an episode and return its id if available."""


class ExpandedKnowledgeBackend(Protocol):
    """Hypothesis layer for self-evolving KG nodes."""

    def match(
        self,
        session_id: Optional[str],
        query: str,
        *,
        top_k: int,
        min_score: float,
    ) -> List[Dict[str, Any]]:
        """Return expanded-node matches for this query."""

    def build_instruction(
        self,
        session_id: Optional[str],
        matches: Sequence[Dict[str, Any]],
        *,
        limit: int,
    ) -> str:
        """Return an instruction that nudges retrieval toward matched hypotheses."""

    def record_usage(
        self,
        session_id: Optional[str],
        answer: str,
        matches: Sequence[Dict[str, Any]],
        *,
        attached_entities: Sequence[str],
    ) -> List[str]:
        """Record answer usage and return promoted node names."""

    def get_record(self, session_id: Optional[str], name: str) -> Optional[Dict[str, Any]]:
        """Return metadata for an expanded node."""


class GraphPromotionBackend(Protocol):
    """Writes promoted memory hypotheses into an authoritative graph."""

    async def promote(
        self,
        session_id: Optional[str],
        promoted_names: Sequence[str],
        *,
        answer_entities: Sequence[str],
        expanded_backend: ExpandedKnowledgeBackend,
    ) -> List[str]:
        """Persist promoted node names and return the names that were written."""


class ChatTurnBackend(Protocol):
    """Optional bridge that feeds completed chat turns back into a KG pipeline."""

    async def ingest(
        self,
        session_id: Optional[str],
        question: str,
        answer: str,
        *,
        timestamp: float,
    ) -> bool:
        """Return True when the turn was ingested."""


@dataclass
class AgentMemoryBackends:
    """Pluggable backend bundle used by :class:`AgentMemoryCore`."""

    conversation: Optional[ConversationMemoryBackend] = None
    episodic: Optional[EpisodicMemoryBackend] = None
    expanded: Optional[ExpandedKnowledgeBackend] = None
    graph: Optional[GraphPromotionBackend] = None
    chat_turn: Optional[ChatTurnBackend] = None


@dataclass(frozen=True)
class MemoryPolicy:
    """Tunable policy for recall breadth and consolidation behavior.

    A policy lets host applications choose which memory layers are active and
    how much context each layer may contribute without changing backend code.
    """

    episodic_top_k: int = 5
    expanded_top_k: int = 2
    expanded_min_score: float = 0.2
    expanded_instruction_limit: int = 2
    answer_entity_limit: int = 12
    enabled_layers: Sequence[str] = field(
        default_factory=lambda: (
            "conversation",
            "episodic",
            "expanded",
            "graph",
            "chat_turn",
        )
    )

    def layer_enabled(self, layer: str) -> bool:
        return layer in self.enabled_layer_set()

    def enabled_layer_set(self) -> set[str]:
        return {str(item) for item in self.enabled_layers}
