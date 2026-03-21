#多模态结构化的缓存
"""Shared schema utilities for HyperGraphRAG and orchestrator pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class StructuredChunk:
    """Structured chunk representing text plus associated multimodal metadata."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def ensure_text(self) -> "StructuredChunk":
        """Return self ensuring text is a string."""
        if self.text is None:
            return StructuredChunk(text="", metadata=dict(self.metadata))
        return self

    @property
    def type(self) -> str:
        return self.metadata.get("type", "text")

