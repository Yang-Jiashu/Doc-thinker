"""
Density clustering for KG nodes + LLM-powered cluster summarization.

After PDF/TXT ingestion completes, run ``build_cluster_summaries`` on the
node embedding matrix.  Dense groups (≥ ``MIN_CLUSTER_NODES`` nodes) get
an LLM-generated thematic summary stored to disk.  These summaries are
later consumed by the expansion pipeline (Prompt A).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

_log = logging.getLogger("docthinker.kg_expansion.clustering")

MIN_CLUSTER_NODES = 4

CLUSTER_SUMMARY_PROMPT = """你是一名知识分析专家。

## 任务
以下是通过密度聚类算法自动发现的一组语义紧密的实体（{node_count} 个）：

{entities_with_descriptions}

请用 2-4 句话总结这组实体共同代表的主题、核心概念和它们之间的关系模式。

## 输出格式
直接输出摘要文本，不要 JSON 包装，不要标题。"""


@dataclass
class ClusterSummary:
    cluster_id: int
    node_ids: List[str] = field(default_factory=list)
    node_descriptions: List[str] = field(default_factory=list)
    summary: str = ""


def _try_import_hdbscan():
    try:
        import hdbscan  # type: ignore
        return hdbscan
    except ImportError:
        return None


def _try_import_sklearn_dbscan():
    try:
        from sklearn.cluster import DBSCAN  # type: ignore
        return DBSCAN
    except ImportError:
        return None


def cluster_nodes(
    node_ids: List[str],
    embeddings: np.ndarray,
    *,
    min_cluster_size: int = MIN_CLUSTER_NODES,
    eps: float = 0.35,
) -> List[List[int]]:
    """Return groups of indices that form dense clusters.

    Tries HDBSCAN first, falls back to sklearn DBSCAN, then to a
    simple greedy cosine-threshold approach.
    """
    n = embeddings.shape[0]
    if n < min_cluster_size:
        return []

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    normed = embeddings / norms

    labels: Optional[np.ndarray] = None

    hdbscan_mod = _try_import_hdbscan()
    if hdbscan_mod is not None:
        clusterer = hdbscan_mod.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=2,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(normed)
        _log.info("[cluster] HDBSCAN labels: %d unique", len(set(labels) - {-1}))

    if labels is None:
        DBSCAN = _try_import_sklearn_dbscan()
        if DBSCAN is not None:
            clusterer = DBSCAN(eps=eps, min_samples=max(2, min_cluster_size - 1), metric="cosine")
            labels = clusterer.fit_predict(normed)
            _log.info("[cluster] DBSCAN labels: %d unique", len(set(labels) - {-1}))

    if labels is None:
        _log.warning("[cluster] neither HDBSCAN nor sklearn available; using greedy cosine clustering")
        sim = normed @ normed.T
        assigned = np.full(n, -1, dtype=int)
        cid = 0
        for i in range(n):
            if assigned[i] != -1:
                continue
            members = [i]
            for j in range(i + 1, n):
                if assigned[j] == -1 and sim[i, j] >= (1.0 - eps):
                    members.append(j)
            if len(members) >= min_cluster_size:
                for m in members:
                    assigned[m] = cid
                cid += 1
        labels = assigned

    groups: Dict[int, List[int]] = {}
    for idx, lab in enumerate(labels):
        if lab < 0:
            continue
        groups.setdefault(int(lab), []).append(idx)

    return [idxs for idxs in groups.values() if len(idxs) >= min_cluster_size]


async def build_cluster_summaries(
    nodes_data: List[Dict[str, Any]],
    embeddings: np.ndarray,
    llm_func: Callable[..., Any],
    *,
    min_cluster_size: int = MIN_CLUSTER_NODES,
    session_id: str = "",
) -> List[ClusterSummary]:
    """Cluster nodes by embedding density and generate LLM summaries for each cluster."""
    node_ids = [n.get("id") or n.get("entity_id") or "" for n in nodes_data]
    if len(node_ids) != embeddings.shape[0]:
        _log.warning("[cluster] node_ids length (%d) != embeddings rows (%d)", len(node_ids), embeddings.shape[0])
        return []

    clusters = cluster_nodes(node_ids, embeddings, min_cluster_size=min_cluster_size)
    _log.info("[cluster] found %d dense clusters (≥%d nodes) out of %d nodes",
              len(clusters), min_cluster_size, len(node_ids))
    if not clusters:
        return []

    from ..llm_trace import LLMTrace

    async def _summarize(cid: int, indices: List[int]) -> ClusterSummary:
        ids = [node_ids[i] for i in indices]
        descs = []
        for i in indices:
            name = node_ids[i]
            desc = str(nodes_data[i].get("description") or "")
            etype = str(nodes_data[i].get("entity_type") or "")
            line = f"- {name}"
            if etype:
                line += f" [{etype}]"
            if desc:
                line += f": {desc[:120]}"
            descs.append(line)
        entities_text = "\n".join(descs)

        prompt = CLUSTER_SUMMARY_PROMPT.format(
            node_count=len(ids),
            entities_with_descriptions=entities_text,
        )
        tracer = LLMTrace(stage="clustering", sub_stage="cluster_summary", session_id=session_id)
        resp = await tracer.call(llm_func, prompt, metadata={"cluster_id": cid, "node_count": len(ids)})
        return ClusterSummary(
            cluster_id=cid,
            node_ids=ids,
            node_descriptions=[str(nodes_data[i].get("description") or "") for i in indices],
            summary=str(resp).strip(),
        )

    summaries = await asyncio.gather(
        *[_summarize(cid, idxs) for cid, idxs in enumerate(clusters)],
        return_exceptions=True,
    )
    results: List[ClusterSummary] = []
    for s in summaries:
        if isinstance(s, ClusterSummary):
            results.append(s)
        else:
            _log.warning("[cluster] summarization failed for one cluster: %s", s)
    _log.info("[cluster] generated %d cluster summaries", len(results))
    return results


def save_cluster_summaries(summaries: List[ClusterSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(s) for s in summaries]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    _log.info("[cluster] saved %d summaries to %s", len(data), path)


def load_cluster_summaries(path: Path) -> List[ClusterSummary]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [
            ClusterSummary(
                cluster_id=item.get("cluster_id", i),
                node_ids=item.get("node_ids", []),
                node_descriptions=item.get("node_descriptions", []),
                summary=item.get("summary", ""),
            )
            for i, item in enumerate(raw)
        ]
    except Exception as exc:
        _log.warning("[cluster] failed to load %s: %s", path, exc)
        return []
