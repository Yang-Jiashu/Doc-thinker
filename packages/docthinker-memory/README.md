# docthinker-memory

`docthinker-memory` is the lightweight agentic memory facade from DocThinker.
It gives external agents a stable interface for recall and consolidation
without coupling them to DocThinker's server runtime.
It is backend-agnostic: Claw/OpenClaw-style memory is only one adapter, not a
required dependency.

The package currently re-exports the memory core shipped in `docthinker`:

- `AgentMemoryCore`
- `AgentMemoryBackends`
- `MemoryPolicy`
- backend protocols for conversation, episodic, long-horizon insight memory,
  expanded KG, graph promotion, and chat-turn ingestion
- `InMemoryLongHorizonBackend` as a default process-local implementation
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
        long_horizon_top_k=3,
        enabled_layers=("conversation", "episodic", "expanded", "long_horizon"),
    )
)

bundle = await memory.recall(
    session_id="demo",
    query="What should this agent remember?",
    enable_thinking=True,
)
```

Implement the backend protocols when you want to plug in your own vector store,
database, graph store, or long-term memory service. Use `remember_turn=False`,
`memory_excluded_layers`, or `MemoryPolicy.allow_memory_writes` when a turn
should be answered but not remembered. Long-horizon backends can also expose
list/delete/export controls so host agents can keep memory auditable instead of
only appending hidden state.
