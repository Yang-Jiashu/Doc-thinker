"""Custom backend example for docthinker-memory."""

from __future__ import annotations

import asyncio
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_PACKAGE_ROOT), str(_ROOT)]

from docthinker_memory import AgentMemoryBackends, AgentMemoryCore, MemoryPolicy


class InMemoryConversationBackend:
    def __init__(self) -> None:
        self.turns: Dict[str, List[str]] = {}

    async def build_context(self, session_id: Optional[str], query: str) -> str:
        history = self.turns.get(session_id or "default", [])[-3:]
        if not history:
            return ""
        return "## Conversation memory\n" + "\n".join(f"- {item}" for item in history)

    async def consolidate(self, session_id: Optional[str], question: str, answer: str) -> bool:
        key = session_id or "default"
        self.turns.setdefault(key, []).append(f"Q: {question}\nA: {answer[:160]}")
        return True


class InMemoryEpisodicBackend:
    def __init__(self) -> None:
        self.episodes: List[Dict[str, Any]] = []

    async def retrieve(self, session_id: Optional[str], query: str, *, top_k: int) -> List[Dict[str, Any]]:
        scored = []
        query_terms = set(query.lower().split())
        for item in self.episodes:
            terms = set(str(item["summary"]).lower().split())
            score = len(query_terms & terms) / max(1, len(query_terms))
            if score > 0:
                scored.append({**item, "score": score, "reason": "overlapping query terms"})
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]

    async def write(
        self,
        session_id: Optional[str],
        question: str,
        answer: str,
        *,
        concepts: Sequence[str],
        timestamp: float,
    ) -> Optional[str]:
        episode_id = f"episode-{len(self.episodes) + 1}"
        self.episodes.append({
            "episode_id": episode_id,
            "summary": f"{question} -> {answer[:160]}",
            "concepts": list(concepts),
            "timestamp": timestamp,
        })
        return episode_id


async def main() -> None:
    conversation = InMemoryConversationBackend()
    episodic = InMemoryEpisodicBackend()
    memory = AgentMemoryCore(
        backends=AgentMemoryBackends(
            conversation=conversation,
            episodic=episodic,
        ),
        policy=MemoryPolicy(
            episodic_top_k=2,
            enabled_layers=("conversation", "episodic"),
        ),
    )

    await memory.after_response(
        session_id="demo",
        question="How should an agent remember goals?",
        answer="Keep a compact working memory and promote durable facts into long-term memory.",
    )

    bundle = await memory.recall(
        session_id="demo",
        query="How should this agent remember goals?",
        enable_thinking=True,
    )
    print(bundle.retrieval_instruction)
    print(bundle.trace.to_schema())


if __name__ == "__main__":
    asyncio.run(main())
