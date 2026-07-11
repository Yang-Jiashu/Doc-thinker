"""Compatibility entry point for ECLRR-v4 path relation discovery.

The former root-degree DFS implementation lived in this module. It has been
replaced by the evidence-carrying ECLRR-v4 pipeline; no candidate state is
written by this compatibility function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .eclrr_v4.models import ECLRRConfig, GateResult
from .eclrr_v4.service import run_eclrr_v4

GRAPH_FIELD_SEP = "<SEP>"


@dataclass(frozen=True)
class PathDiscoveryConfig(ECLRRConfig):
    """Backward-compatible name for the ECLRR-v4 configuration."""


PathDiscoveredEdge = GateResult


async def discover_path_edges(
    *,
    graph: Any,
    text_chunks: Any,
    llm_func: Callable,
    judge_func: Callable | None = None,
    config: PathDiscoveryConfig | None = None,
) -> list[GateResult]:
    """Run discovery without writing graph or vector storage."""
    result = await run_eclrr_v4(
        graph=graph,
        text_chunks=text_chunks,
        generator_func=llm_func,
        judge_func=judge_func or llm_func,
        config=config or PathDiscoveryConfig(),
        write_graph=False,
    )
    return [item for item in result.gate_results if item.action in {"create", "refine"}]


__all__ = [
    "GRAPH_FIELD_SEP",
    "PathDiscoveryConfig",
    "PathDiscoveredEdge",
    "discover_path_edges",
]
