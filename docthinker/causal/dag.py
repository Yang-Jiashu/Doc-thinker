"""Persistent causal DAG with cycle-safe edge writes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


@dataclass
class CausalNode:
    id: str
    node_type: str = "event"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalEdge:
    source: str
    target: str
    mechanism: str = "causes"
    strength: float = 1.0
    evidence: str = ""
    source_id: str = ""


class CausalDAG:
    """Small DAG used by the tri-graph memory layer."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: Dict[str, CausalEdge] = {}
        self._forward: Dict[str, List[str]] = {}
        self._backward: Dict[str, List[str]] = {}
        self._load()

    @staticmethod
    def _edge_id(source: str, target: str) -> str:
        return f"{source}->{target}"

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.nodes = {
            key: CausalNode(**value)
            for key, value in (data.get("nodes") or {}).items()
        }
        self.edges = {
            key: CausalEdge(**value)
            for key, value in (data.get("edges") or {}).items()
        }
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        self._forward = {node_id: [] for node_id in self.nodes}
        self._backward = {node_id: [] for node_id in self.nodes}
        for edge in self.edges.values():
            self._forward.setdefault(edge.source, []).append(edge.target)
            self._backward.setdefault(edge.target, []).append(edge.source)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": {key: asdict(value) for key, value in self.nodes.items()},
            "edges": {key: asdict(value) for key, value in self.edges.items()},
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_node(self, node_id: str, node_type: str = "event", **metadata: Any) -> CausalNode:
        key = str(node_id).strip()
        if not key:
            raise ValueError("node_id is required")
        node = self.nodes.get(key)
        if node is None:
            node = CausalNode(id=key, node_type=node_type, metadata=dict(metadata))
            self.nodes[key] = node
            self._forward.setdefault(key, [])
            self._backward.setdefault(key, [])
        elif metadata:
            node.metadata.update(metadata)
        return node

    def _has_path(self, source: str, target: str) -> bool:
        seen: Set[str] = set()
        stack = [source]
        while stack:
            current = stack.pop()
            if current == target:
                return True
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self._forward.get(current, []))
        return False

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        mechanism: str = "causes",
        strength: float = 1.0,
        evidence: str = "",
        source_id: str = "",
    ) -> Optional[CausalEdge]:
        src = str(source).strip()
        tgt = str(target).strip()
        if not src or not tgt or src == tgt:
            return None
        self.add_node(src)
        self.add_node(tgt)
        if self._has_path(tgt, src):
            return None
        edge_id = self._edge_id(src, tgt)
        edge = CausalEdge(
            source=src,
            target=tgt,
            mechanism=str(mechanism or "causes"),
            strength=max(0.0, min(1.0, float(strength or 0.0))),
            evidence=str(evidence or ""),
            source_id=str(source_id or ""),
        )
        self.edges[edge_id] = edge
        self._rebuild_indexes()
        return edge

    def get_ancestors(self, node_id: str) -> Set[str]:
        return self._walk(node_id, self._backward)

    def get_descendants(self, node_id: str) -> Set[str]:
        return self._walk(node_id, self._forward)

    @staticmethod
    def _walk(start: str, graph: Dict[str, List[str]]) -> Set[str]:
        seen: Set[str] = set()
        stack = list(graph.get(start, []))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(graph.get(current, []))
        return seen

    def get_causal_chains(
        self,
        starts: Iterable[str],
        *,
        max_depth: int = 4,
        max_chains: int = 20,
    ) -> List[List[str]]:
        chains: List[List[str]] = []

        def dfs(path: List[str]) -> None:
            if len(chains) >= max_chains:
                return
            current = path[-1]
            next_nodes = self._forward.get(current, [])
            if not next_nodes or len(path) > max_depth:
                if len(path) > 1:
                    chains.append(list(path))
                return
            for nxt in next_nodes:
                if nxt not in path:
                    dfs(path + [nxt])

        for start in starts:
            if start in self.nodes:
                dfs([start])
        return chains

    def topological_sort(self) -> List[str]:
        indegree = {node_id: len(self._backward.get(node_id, [])) for node_id in self.nodes}
        queue = [node_id for node_id, degree in indegree.items() if degree == 0]
        order: List[str] = []
        while queue:
            current = queue.pop(0)
            order.append(current)
            for nxt in self._forward.get(current, []):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        return order

    def stats(self) -> Dict[str, int]:
        return {"nodes": len(self.nodes), "edges": len(self.edges)}

    def to_graph_data(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {
                    "id": node.id,
                    "label": node.id,
                    "type": node.node_type,
                    **node.metadata,
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "id": key,
                    "source": edge.source,
                    "target": edge.target,
                    "label": edge.mechanism,
                    "description": edge.evidence,
                    "weight": edge.strength,
                    "source_id": edge.source_id,
                }
                for key, edge in self.edges.items()
            ],
            "metadata": {"source": "causal_dag", **self.stats()},
        }
