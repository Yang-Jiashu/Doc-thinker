"""Data models for the Claw tiered memory system."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional


@dataclass
class TurnRecord:
    """A single Q&A turn in conversation history."""

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    turn_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "turn_id": self.turn_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TurnRecord":
        return cls(
            role=d.get("role", "user"),
            content=d.get("content", ""),
            timestamp=d.get("timestamp", 0.0),
            turn_id=d.get("turn_id") or d.get("id"),
        )


@dataclass
class ArchiveChunk:
    """An embedded chunk stored in the semantic archive."""

    chunk_id: str
    text: str
    embedding: Optional[List[float]] = None
    timestamp: float = 0.0
    turn_ids: List[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "timestamp": self.timestamp,
            "turn_ids": self.turn_ids,
        }
        if self.embedding is not None:
            d["embedding"] = self.embedding
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArchiveChunk":
        return cls(
            chunk_id=d.get("chunk_id", ""),
            text=d.get("text", ""),
            embedding=d.get("embedding"),
            timestamp=d.get("timestamp", 0.0),
            turn_ids=d.get("turn_ids", []),
        )


LLMFunc = Callable[..., Coroutine[Any, Any, str]]
EmbeddingFunc = Callable[[List[str]], Coroutine[Any, Any, Any]]


@dataclass
class MemoryConfig:
    """Configuration for the Claw memory system."""

    working_memory_turns: int = 6
    core_memory_update_interval: int = 5
    core_memory_max_bytes: int = 10240
    archive_chunk_size: int = 600
    archive_top_k: int = 5
    archive_min_score: float = 0.35
