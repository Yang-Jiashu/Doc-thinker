# neuro_memory/consolidation.py
"""记忆巩固：重放、跨事件关系推断、权重更新、主题聚类。"""

from __future__ import annotations

import asyncio
import random
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .models import EdgeType, Episode
from .graph_store import MemoryGraphStore


async def infer_cross_episode_relations(
    episode_a: Episode,
    episode_b: Episode,
    llm_func: Callable[[str], Any],
) -> List[Dict[str, Any]]:
    """
    用 LLM 推断两个 episode 之间的类比/主题关系。
    返回 [{"relation": "analogous_to"|"same_theme", "reason": str, "mapping": [...]}, ...]
    """
    prompt = f"""你是一个记忆与类比分析助手。请比较两段经历是否属于同一主题或可类比，并简要说明理由。

经历A摘要: {episode_a.summary[:800]}
经历A关键概念: {", ".join(episode_a.concepts[:15])}

经历B摘要: {episode_b.summary[:800]}
经历B关键概念: {", ".join(episode_b.concepts[:15])}

请用简短中文回答，格式如下（若无关则 relation 填 none）：
- relation: analogous_to 或 same_theme 或 none
- reason: 一句话理由
"""
    try:
        if asyncio.iscoroutinefunction(llm_func):
            resp = await llm_func(prompt)
        else:
            resp = llm_func(prompt)
        if not resp or not isinstance(resp, str):
            return []
        text = resp.strip().lower()
        result = []
        if "analogous" in text or "类比" in resp:
            result.append({"relation": "analogous_to", "reason": resp[:200]})
        if "same_theme" in text or "同一主题" in resp or "同一类" in resp:
            result.append({"relation": "same_theme", "reason": resp[:200]})
        return result
    except Exception:
        return []


def build_structure_description(episode: Episode) -> str:
    """从 episode 的实体与关系生成结构描述，用于结构相似度。"""
    parts = []
    if episode.entity_ids:
        parts.append(f"Entities({len(episode.entity_ids)}): {', '.join(episode.entity_ids[:10])}")
    if episode.relation_triples:
        rels = [r for _, r, _ in episode.relation_triples[:15]]
        cnt = Counter(rels)
        parts.append("Relations: " + ", ".join(f"{r}({c})" for r, c in cnt.most_common(10)))
    return " | ".join(parts) if parts else episode.summary[:200]


async def consolidate(
    graph: MemoryGraphStore,
    episodes: Dict[str, Episode],
    *,
    recent_n: int = 50,
    high_salience_n: int = 20,
    content_sim_threshold: float = 0.5,
    structure_sim_threshold: float = 0.3,
    llm_func: Optional[Callable[..., Any]] = None,
    content_sim_fn: Optional[Callable[..., Union[float, Any]]] = None,
    structure_sim_fn: Optional[Callable[..., Union[float, Any]]] = None,
) -> Dict[str, Any]:
    """
    巩固流程：采样 → 配对 → 跨事件推断 → 写回图。
    不要求必传 llm_func；无 LLM 时只做基于相似度的边添加。
    content_sim_fn / structure_sim_fn 可为 async，本函数会 await。
    """

    async def _sim(a: str, b: str, fn: Optional[Callable[..., Any]]) -> float:
        if not fn:
            return 0.0
        try:
            out = fn(a, b)
            return float(await out if asyncio.iscoroutine(out) else out)
        except Exception:
            return 0.0

    episode_ids = list(episodes.keys())
    if not episode_ids:
        return {"edges_added": 0, "pairs_processed": 0}

    # 按时间与 retrieval_count 排序，取近期 + 高显著性
    def salience_key(eid: str) -> Tuple[float, int]:
        ep = episodes.get(eid)
        if not ep:
            return 0.0, 0
        rec = ep.last_retrieved_at or ep.timestamp
        return rec, ep.retrieval_count

    sorted_ids = sorted(episode_ids, key=salience_key, reverse=True)
    recent = sorted_ids[:recent_n]
    high_salience = sorted(episode_ids, key=lambda eid: (episodes.get(eid) or Episode(episode_id="")).retrieval_count, reverse=True)[:high_salience_n]
    pool_ids = list(dict.fromkeys(recent + high_salience))
    random.shuffle(pool_ids)

    edges_added = 0
    pairs_processed = 0

    for i in range(len(pool_ids)):
        for j in range(i + 1, min(i + 5, len(pool_ids))):
            eid_a, eid_b = pool_ids[i], pool_ids[j]
            ep_a, ep_b = episodes.get(eid_a), episodes.get(eid_b)
            if not ep_a or not ep_b:
                continue

            content_sim = await _sim(ep_a.content_for_embedding(), ep_b.content_for_embedding(), content_sim_fn)
            structure_sim = await _sim(
                ep_a.structure_description or build_structure_description(ep_a),
                ep_b.structure_description or build_structure_description(ep_b),
                structure_sim_fn,
            )

            if content_sim < content_sim_threshold and structure_sim < structure_sim_threshold:
                continue

            pairs_processed += 1

            if llm_func and content_sim >= 0.4:
                try:
                    relations = await infer_cross_episode_relations(ep_a, ep_b, llm_func)
                except Exception:
                    relations = []
                for rel in relations:
                    rtype = rel.get("relation")
                    if rtype == "analogous_to":
                        graph.add_edge(eid_a, eid_b, EdgeType.ANALOGOUS_TO, weight=0.7, metadata={"reason": rel.get("reason", "")})
                        graph.add_edge(eid_b, eid_a, EdgeType.ANALOGOUS_TO, weight=0.7, metadata={"reason": rel.get("reason", "")})
                        edges_added += 2
                    elif rtype == "same_theme":
                        graph.add_edge(eid_a, eid_b, EdgeType.SAME_THEME, weight=0.6)
                        graph.add_edge(eid_b, eid_a, EdgeType.SAME_THEME, weight=0.6)
                        edges_added += 2
            else:
                if content_sim >= content_sim_threshold:
                    graph.add_edge(eid_a, eid_b, EdgeType.EPISODE_SIMILARITY, weight=content_sim)
                    graph.add_edge(eid_b, eid_a, EdgeType.EPISODE_SIMILARITY, weight=content_sim)
                    edges_added += 2

    return {"edges_added": edges_added, "pairs_processed": pairs_processed}
