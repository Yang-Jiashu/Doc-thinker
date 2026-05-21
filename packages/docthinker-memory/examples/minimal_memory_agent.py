"""Minimal agent loop using docthinker-memory."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_PACKAGE_ROOT), str(_ROOT)]

from custom_backend import InMemoryConversationBackend, InMemoryEpisodicBackend
from docthinker_memory import AgentMemoryBackends, AgentMemoryCore


async def answer_with_memory(memory: AgentMemoryCore, question: str) -> str:
    bundle = await memory.recall(
        session_id="agent-demo",
        query=question,
        base_instruction="Answer briefly and use memory only when it is relevant.",
        enable_thinking=True,
    )
    answer = f"I considered {bundle.trace.memory_hits} memory source(s): {question}"
    await memory.after_response(
        session_id="agent-demo",
        question=question,
        answer=answer,
    )
    return answer


async def main() -> None:
    memory = AgentMemoryCore(
        backends=AgentMemoryBackends(
            conversation=InMemoryConversationBackend(),
            episodic=InMemoryEpisodicBackend(),
        )
    )
    print(await answer_with_memory(memory, "Remember that I am building an agent memory framework."))
    print(await answer_with_memory(memory, "What framework am I building?"))


if __name__ == "__main__":
    asyncio.run(main())
