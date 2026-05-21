# docthinker-memory

`docthinker-memory` is the lightweight agentic memory facade from DocThinker.
It gives external agents a stable interface for recall and consolidation
without coupling them to DocThinker's server runtime.

The package currently re-exports the memory core shipped in `docthinker`:

- `AgentMemoryCore`
- `AgentMemoryBackends`
- `MemoryPolicy`
- backend protocols for conversation, episodic, expanded KG, graph promotion,
  and chat-turn ingestion
- `RecallBundle` and `MemoryTrace`

## Install

Once the package is published:

```bash
pip install docthinker-memory
```

For local development from this repository today:

```bash
pip install -e .
pip install -e packages/docthinker-memory
```

## Minimal Usage

```python
from docthinker_memory import AgentMemoryCore, MemoryPolicy

memory = AgentMemoryCore(
    policy=MemoryPolicy(
        episodic_top_k=3,
        expanded_top_k=2,
        enabled_layers=("conversation", "episodic", "expanded"),
    )
)

bundle = await memory.recall(
    session_id="demo",
    query="What should this agent remember?",
    enable_thinking=True,
)
```

Implement the backend protocols when you want to plug in your own vector store,
database, graph store, or long-term memory service.
