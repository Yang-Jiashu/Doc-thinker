# neuro_memory/analogical_retrieval.py
"""类比检索：内容相似 + 结构相似 + 显著性，支持区分信息。"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import Episode


def score_episode(
    episode: Episode,
    content_sim: float,
    structure_sim: float = 0.0,
    *,
    alpha: float = 0.6,
    beta: float = 0.25,
    gamma: float = 0.15,
    recency_weight: float = 0.5,
) -> float:
    """
    综合得分 = α*content_sim + β*structure_sim + γ*salience。
    salience 用 retrieval_count 与 last_retrieved_at 的简单归一化。
    """
    import time
    salience = 0.0
    if episode.retrieval_count > 0 or episode.last_retrieved_at:
        salience = min(1.0, 0.3 + 0.4 * min(1.0, episode.retrieval_count / 10))
        if episode.last_retrieved_at:
            age = time.time() - episode.last_retrieved_at
            # 24 小时内检索过则加分
            salience += recency_weight * max(0, 1.0 - age / (24 * 3600))
        salience = min(1.0, salience)
    return alpha * content_sim + beta * structure_sim + gamma * salience


async def retrieve_analogies(
    query_text: str,
    query_structure_description: str,
    episodes_by_id: Dict[str, Episode],
    *,
    content_embed_fn: Optional[Callable[[str], Any]] = None,
    content_search_fn: Optional[Callable[[List[float], int], List[Tuple[str, float]]]] = None,
    structure_sim_fn: Optional[Callable[[str, str], Any]] = None,
    top_k: int = 10,
    alpha: float = 0.6,
    beta: float = 0.25,
    gamma: float = 0.15,
) -> List[Tuple[Episode, float, Optional[str]]]:
    """
    类比检索：先按内容向量找候选，再算结构相似与显著性，综合排序。
    返回 [(episode, score, differentiation_hint), ...]，differentiation_hint 在 top2 很相似时由调用方用 LLM 生成。
    """
    import asyncio

    if not episodes_by_id:
        return []

    # 若无向量检索，则退化为按 episode 列表顺序或随机（仅用于测试）
    candidates: List[Tuple[str, float]] = []
    if content_embed_fn and content_search_fn:
        try:
            q_emb = content_embed_fn(query_text)
            if asyncio.iscoroutine(q_emb):
                q_emb = await q_emb
            if q_emb and isinstance(q_emb, list) and isinstance(q_emb[0], list):
                q_emb = q_emb[0]
            candidates = content_search_fn(q_emb, top_k * 3)
        except Exception:
            candidates = [(eid, 0.5) for eid in list(episodes_by_id.keys())[: top_k * 3]]
    else:
        candidates = [(eid, 0.5) for eid in list(episodes_by_id.keys())[: top_k * 3]]

    scored: List[Tuple[Episode, float]] = []
    for eid, content_sim in candidates:
        ep = episodes_by_id.get(eid)
        if not ep:
            continue
        structure_sim = 0.0
        if structure_sim_fn and query_structure_description and (ep.structure_description or ep.relation_triples or ep.entity_ids):
            try:
                from .consolidation import build_structure_description
                struct_b = ep.structure_description or build_structure_description(ep)
                out = structure_sim_fn(query_structure_description, struct_b)
                structure_sim = float(await out if asyncio.iscoroutine(out) else out)
            except Exception:
                pass
        s = score_episode(ep, content_sim, structure_sim, alpha=alpha, beta=beta, gamma=gamma)
        scored.append((ep, s))

    scored.sort(key=lambda x: -x[1])
    result: List[Tuple[Episode, float, Optional[str]]] = [(ep, s, None) for ep, s in scored[:top_k]]
    return result


def structure_description_from_triples(
    entity_ids: List[str],
    relation_triples: List[Tuple[str, str, str]],
    max_rels: int = 15,
) -> str:
    """从实体与关系三元组生成简短结构描述，供 embedding 或相似度用。"""
    from collections import Counter
    parts = []
    if entity_ids:
        parts.append(f"Entities({len(entity_ids)}): {', '.join(entity_ids[:10])}")
    if relation_triples:
        rels = [r for _, r, _ in relation_triples[:max_rels]]
        cnt = Counter(rels)
        parts.append("Relations: " + ", ".join(f"{r}({c})" for r, c in cnt.most_common(10)))
    return " | ".join(parts) if parts else ""
