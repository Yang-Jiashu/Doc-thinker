"""Claw: Tiered episodic memory system for DocThinker.

Three-layer architecture:
  - Hot Layer (Working Memory): recent N turns, always injected
  - Warm Layer (Core Memory): LLM-compressed MEMORY.md per session
  - Cold Layer (Semantic Archive): embedded chunks of older turns, vector-retrieved
"""

from .models import TurnRecord, MemoryConfig
from .memory_manager import ClawMemoryManager

__all__ = ["TurnRecord", "MemoryConfig", "ClawMemoryManager"]
