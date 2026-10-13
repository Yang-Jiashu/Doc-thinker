"""Deterministic query budgets; no model calls are needed to choose a budget."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextBudget:
    history_tokens: int = 1200
    instruction_tokens: int = 2000
    tokenizer: Any = None

    def count(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        # A deliberately conservative fallback, not a provider usage estimate.
        return len(text.encode("utf-8"))

    def clip(self, text: str, limit: int) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.decode(self.tokenizer.encode(text)[: max(0, limit)])
        return text.encode("utf-8")[: max(0, limit)].decode("utf-8", errors="ignore")

    def history(self, messages: list[dict]) -> list[dict]:
        """Keep recent messages within budget, without mutating stored history."""
        kept = []
        remaining = self.history_tokens
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") not in {
                "user",
                "assistant",
            }:
                continue
            content = str(message.get("content") or "").strip()
            if not content or remaining <= 8:
                continue
            content = self.clip(content, remaining - 8)
            kept.append({"role": message["role"], "content": content})
            remaining -= self.count(content) + 8
        return list(reversed(kept))


def retrieval_limits(request: Any, mode: str) -> dict[str, int]:
    names = (
        "top_k",
        "chunk_top_k",
        "max_relation_tokens",
        "max_total_tokens",
        "max_relations",
        "max_discovered_relations",
    )
    limits = {name: int(getattr(request, name)) for name in names}
    if not getattr(request, "adaptive_context", True):
        return limits
    # Initial engineering caps, not quality-optimal parameters. Request values
    # are upper bounds and are never increased by a profile.
    profiles = {
        "faithful": (12, 8, 2400, 12000, 16, 0),
        "path": (20, 12, 4000, 18000, 24, 4),
        "explore": (16, 10, 3200, 16000, 24, 6),
    }
    return {name: min(limits[name], cap) for name, cap in zip(names, profiles[mode])}
