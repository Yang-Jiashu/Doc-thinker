# neuro_memory/spreading_activation.py
"""扩散激活算法：模拟人脑联想时激活沿图传播、多路径叠加、按边类型衰减。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .models import EdgeType, get_decay_for_edge_type
from .graph_store import MemoryGraphStore


def spreading_activation(
    graph: MemoryGraphStore,
    seed_ids: List[str],
    *,
    max_hops: int = 3,
    initial_activation: float = 1.0,
    decay_per_hop: Optional[Dict[EdgeType, float]] = None,
    query_similarity_fn: Optional[Callable[[str], float]] = None,
    min_activation: float = 1e-4,
    record_activation: bool = True,
) -> List[Tuple[str, float]]:
    """
    从种子节点出发做扩散激活，返回 (node_id, activation_score) 列表，按分数降序。

    - seed_ids: 种子节点 ID 列表（如当前 query 命中的 entity/episode/chunk）。
    - max_hops: 最大传播跳数。
    - initial_activation: 种子节点初始激活值（会均分到多个种子）。
    - decay_per_hop: 每种边类型每跳的衰减系数，默认用 models.EDGE_TYPE_DECAY。
    - query_similarity_fn: 若提供，对每个 node_id 返回与 query 的相似度 [0,1]，用于对传出边加权。
    - min_activation: 低于此值的激活不再传播。
    """
    decay = decay_per_hop or {}
    activation: Dict[str, float] = defaultdict(float)
    per_seed = initial_activation / max(1, len(seed_ids))
    for nid in seed_ids:
        if graph.has_node(nid):
            activation[nid] += per_seed

    for hop in range(max_hops):
        next_activation: Dict[str, float] = defaultdict(float)
        for node_id, score in list(activation.items()):
            if score < min_activation:
                continue
            for target_id, edge in graph.get_neighbors_with_edges(node_id):
                if record_activation:
                    graph.record_edge_activation(node_id, target_id, edge.edge_type)
                decay_factor = decay.get(edge.edge_type) or get_decay_for_edge_type(edge.edge_type)
                transfer = score * edge.weight * (decay_factor ** (hop + 1))
                if query_similarity_fn:
                    try:
                        qs = query_similarity_fn(target_id)
                        transfer *= (0.5 + 0.5 * qs)
                    except Exception:
                        pass
                if transfer >= min_activation:
                    next_activation[target_id] += transfer
        for nid, delta in next_activation.items():
            activation[nid] += delta

    out: List[Tuple[str, float]] = [(nid, s) for nid, s in activation.items() if s >= min_activation]
    out.sort(key=lambda x: -x[1])
    return out


def top_k_activated(
    graph: MemoryGraphStore,
    seed_ids: List[str],
    k: int = 20,
    *,
    max_hops: int = 3,
    exclude_seeds: bool = True,
    query_similarity_fn: Optional[Callable[[str], float]] = None,
) -> List[Tuple[str, float]]:
    """扩散激活后取 top-k；可选排除种子本身。"""
    full = spreading_activation(
        graph, seed_ids,
        max_hops=max_hops,
        query_similarity_fn=query_similarity_fn,
    )
    seen: Set[str] = set(seed_ids)
    result: List[Tuple[str, float]] = []
    for nid, score in full:
        if exclude_seeds and nid in seen:
            continue
        result.append((nid, score))
        if len(result) >= k:
            break
    return result


def spreading_activation_traced(
    graph: MemoryGraphStore,
    seed_ids: List[str],
    *,
    max_hops: int = 3,
    initial_activation: float = 1.0,
    decay_per_hop: Optional[Dict[EdgeType, float]] = None,
    min_activation: float = 1e-4,
    record_activation: bool = True,
) -> List[Tuple[str, float, List[str], List[Tuple[str, EdgeType, float]]]]:
    """
    带路径追踪的扩散激活。
    返回 (node_id, activation_score, path_nodes, path_edges) 列表，按分数降序。
    path_nodes: [seed_id, hop1_node, ..., target_node]
    path_edges: [(source_id, edge_type, edge_weight), ...]
    """
    decay = decay_per_hop or {}
    activation: Dict[str, float] = defaultdict(float)
    best_path: Dict[str, Tuple[List[str], List[Tuple[str, EdgeType, float]], float]] = {}

    per_seed = initial_activation / max(1, len(seed_ids))
    for nid in seed_ids:
        if graph.has_node(nid):
            activation[nid] += per_seed
            best_path[nid] = ([nid], [], per_seed)

    for hop in range(max_hops):
        next_activation: Dict[str, float] = defaultdict(float)
        for node_id, score in list(activation.items()):
            if score < min_activation:
                continue
            for target_id, edge in graph.get_neighbors_with_edges(node_id):
                if record_activation:
                    graph.record_edge_activation(node_id, target_id, edge.edge_type)
                decay_factor = decay.get(edge.edge_type) or get_decay_for_edge_type(edge.edge_type)
                transfer = score * edge.weight * (decay_factor ** (hop + 1))
                if transfer >= min_activation:
                    next_activation[target_id] += transfer
                    cur_path, cur_edges, _ = best_path.get(node_id, ([node_id], [], 0.0))
                    new_path = cur_path + [target_id]
                    new_edges = cur_edges + [(node_id, edge.edge_type, edge.weight)]
                    prev = best_path.get(target_id)
                    if prev is None or transfer > prev[2]:
                        best_path[target_id] = (new_path, new_edges, transfer)
        for nid, delta in next_activation.items():
            activation[nid] += delta

    out: List[Tuple[str, float, List[str], List[Tuple[str, EdgeType, float]]]] = []
    for nid, s in activation.items():
        if s >= min_activation:
            path_info = best_path.get(nid, ([nid], [], 0.0))
            out.append((nid, s, path_info[0], path_info[1]))
    out.sort(key=lambda x: -x[1])
    return out
