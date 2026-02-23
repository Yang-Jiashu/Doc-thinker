# neuro_memory/graph_store.py
"""记忆联想图存储：节点与边的增删查，供扩散激活与巩固使用。"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .models import EdgeType, MemoryEdge, get_decay_for_edge_type


class MemoryGraphStore:
    """内存版记忆图：节点为 episode_id / entity_id / chunk_id，边带类型与权重。"""

    def __init__(self):
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._out_edges: Dict[str, List[MemoryEdge]] = defaultdict(list)
        self._edge_index: Dict[str, MemoryEdge] = {}  # edge_key -> edge
    
    @property
    def nodes(self) -> Dict[str, Dict[str, Any]]:
        """获取所有节点"""
        return self._nodes
    
    @property
    def edges(self) -> List[MemoryEdge]:
        """获取所有边"""
        return list(self._edge_index.values())

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
                 metadata: Optional[Dict[str, Any]] = None) -> MemoryEdge:
        self.add_node(source_id)
        self.add_node(target_id)
        key = f"{source_id}\t{edge_type.value}\t{target_id}"
        if key in self._edge_index:
            e = self._edge_index[key]
            e.weight = min(1.0, e.weight + weight * 0.3)  # 重复添加略加强
            if metadata:
                e.metadata.update(metadata)
            return e
        edge = MemoryEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata or {},
        )
        self._out_edges[source_id].append(edge)
        self._edge_index[key] = edge
        return edge

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
                metadata=e.get("metadata", {}),
            )
            key = edge.edge_key()
            g._edge_index[key] = edge
            g._out_edges[edge.source_id].append(edge)
        return g
