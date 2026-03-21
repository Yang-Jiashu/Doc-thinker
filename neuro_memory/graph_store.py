# neuro_memory/graph_store.py
"""记忆联想图存储：节点与边的增删查，供扩散激活与巩固使用。"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    EdgeType,
    MemoryEdge,
    get_decay_for_edge_type,
    DEFAULT_MAX_EDGE_WEIGHT,
    DEFAULT_ACTIVATION_WEIGHT_DELTA,
)

# 秒/天
SECONDS_PER_DAY = 86400.0


class MemoryGraphStore:
    """内存版记忆图：节点为 episode_id / entity_id / chunk_id，边带类型与权重。"""

    def __init__(self):
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._out_edges: Dict[str, List[MemoryEdge]] = defaultdict(list)
        self._edge_index: Dict[str, MemoryEdge] = {}  # edge_key -> edge

    def add_node(self, node_id: str, node_type: str = "episode", data: Optional[Dict[str, Any]] = None) -> None:
        if node_id not in self._nodes:
            self._nodes[node_id] = {"id": node_id, "type": node_type, "created_at": time.time()}
        if data:
            self._nodes[node_id].update(data)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self._nodes.get(node_id)

    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType, weight: float = 1.0,
                 metadata: Optional[Dict[str, Any]] = None,
                 last_activated_at: Optional[float] = None) -> MemoryEdge:
        self.add_node(source_id)
        self.add_node(target_id)
        key = f"{source_id}\t{edge_type.value}\t{target_id}"
        if key in self._edge_index:
            e = self._edge_index[key]
            e.weight = min(DEFAULT_MAX_EDGE_WEIGHT, e.weight + weight * 0.3)  # 重复添加略加强
            if metadata:
                e.metadata.update(metadata)
            if last_activated_at is not None:
                e.last_activated_at = last_activated_at
            return e
        edge = MemoryEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata or {},
            last_activated_at=last_activated_at,
        )
        self._out_edges[source_id].append(edge)
        self._edge_index[key] = edge
        return edge

    def record_edge_activation(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        *,
        weight_delta: float = DEFAULT_ACTIVATION_WEIGHT_DELTA,
        max_weight: float = DEFAULT_MAX_EDGE_WEIGHT,
    ) -> bool:
        """记录边被激活：更新 last_activated_at，并按增量加强权重（不超过 max_weight）。"""
        key = f"{source_id}\t{edge_type.value}\t{target_id}"
        if key not in self._edge_index:
            return False
        e = self._edge_index[key]
        e.last_activated_at = time.time()
        if e.weight < max_weight:
            e.weight = min(max_weight, e.weight + weight_delta)
        return True

    def decay_edges(self, decay_factor: float = 0.9, max_age_days: float = 30.0) -> int:
        """对超过 max_age_days 未激活的边做 weight *= decay_factor。返回被衰减的边数。"""
        cutoff = time.time() - max_age_days * SECONDS_PER_DAY
        count = 0
        for e in self._edge_index.values():
            last = e.last_activated_at or e.created_at
            if last < cutoff:
                e.weight *= decay_factor
                count += 1
        return count

    def prune_edges(self, min_weight: float = 0.05) -> int:
        """删除 weight < min_weight 的边。返回被删除的边数。"""
        to_remove = [(k, self._edge_index[k]) for k, e in list(self._edge_index.items()) if e.weight < min_weight]
        for k, e in to_remove:
            self._edge_index.pop(k, None)
            out_list = self._out_edges.get(e.source_id, [])
            self._out_edges[e.source_id] = [ex for ex in out_list if ex.edge_key() != k]
        return len(to_remove)

    def get_out_edges(self, node_id: str) -> List[MemoryEdge]:
        return list(self._out_edges.get(node_id, []))

    def get_neighbors_with_edges(self, node_id: str) -> List[Tuple[str, MemoryEdge]]:
        return [(e.target_id, e) for e in self.get_out_edges(node_id)]

    def get_all_edges(self) -> List[MemoryEdge]:
        return list(self._edge_index.values())

    def get_all_nodes(self) -> List[Tuple[str, Dict[str, Any]]]:
        """返回 (node_id, node_data) 列表，便于可视化与导出。"""
        return list(self._nodes.items())

    def get_nodes_by_type(self, node_type: str) -> List[str]:
        return [nid for nid, n in self._nodes.items() if n.get("type") == node_type]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": dict(self._nodes),
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "edge_type": e.edge_type.value,
                    "weight": e.weight,
                    "created_at": e.created_at,
                    "last_activated_at": e.last_activated_at,
                    "metadata": e.metadata,
                }
                for e in self._edge_index.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryGraphStore":
        g = cls()
        g._nodes = data.get("nodes", {})
        for e in data.get("edges", []):
            edge = MemoryEdge(
                source_id=e["source_id"],
                target_id=e["target_id"],
                edge_type=EdgeType(e["edge_type"]),
                weight=e.get("weight", 1.0),
                created_at=e.get("created_at", time.time()),
                last_activated_at=e.get("last_activated_at"),
                metadata=e.get("metadata", {}),
            )
            key = edge.edge_key()
            g._edge_index[key] = edge
            g._out_edges[edge.source_id].append(edge)
        return g
