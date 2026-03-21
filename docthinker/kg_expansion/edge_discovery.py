"""Background edge discovery: find latent relationships between existing KG entities.

After initial entity extraction, many potential edges are missed because each
chunk is processed independently.  This module takes the full entity set,
groups them into overlapping windows, and asks the LLM to identify plausible
relationships.  Discovered edges are persisted with ``is_discovered=1`` so the
frontend can render them distinctly (e.g. red).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

_log = logging.getLogger("docthinker.edge_discovery")

EDGE_DISCOVERY_PROMPT = """你是知识图谱关系发现专家。

下面列出了从文档中提取的实体（节点），每个实体包含名称、类型和描述。
请仔细分析这些实体之间是否存在 **尚未显式建立的潜在关系**。

关注以下类型的隐含关系：
1. **层级关系**：A 是 B 的子类/上位概念/组成部分
2. **因果关系**：A 导致/影响/促进 B
3. **对比关系**：A 与 B 形成对比/替代/竞争
4. **时序关系**：A 发生在 B 之前/之后/同时
5. **应用关系**：A 被用于/应用于 B
6. **协作关系**：A 与 B 共同参与/协作完成某事

## 实体列表

{entities_block}

## 要求

- 只输出**确信度较高**的关系，不要猜测
- 每条关系必须包含 source、target、keywords（关系类型简述）、description（具体描述）
- source 和 target 必须是上面列出的实体名称（精确匹配）
- 不要重复已经存在的关系

## 已有关系（避免重复）

{existing_edges_block}

## 输出格式

严格输出 JSON 数组，每个元素格式：
```json
{{"source": "实体A", "target": "实体B", "keywords": "关系类型", "description": "具体描述"}}
```

如果没有发现有价值的潜在关系，输出空数组 `[]`。
"""


@dataclass
class DiscoveredEdge:
    source: str
    target: str
    keywords: str
    description: str


async def discover_edges(
    nodes: List[Dict[str, Any]],
    existing_edges: List[Dict[str, Any]],
    llm_func: Callable,
    *,
    window_size: int = 30,
    overlap: int = 10,
    max_parallel: int = 3,
) -> List[DiscoveredEdge]:
    """Scan entity windows and ask LLM to find latent relationships.

    Splits entities into overlapping windows so the LLM sees enough context
    to spot cross-entity connections without exceeding token limits.
    """
    if len(nodes) < 3:
        return []

    existing_pairs = set()
    for e in existing_edges:
        s, t = str(e.get("source", "")), str(e.get("target", ""))
        existing_pairs.add((s, t))
        existing_pairs.add((t, s))

    windows = _build_windows(nodes, window_size, overlap)
    _log.info("[edge_discovery] %d entities -> %d windows (size=%d, overlap=%d)",
              len(nodes), len(windows), window_size, overlap)

    sem = asyncio.Semaphore(max_parallel)
    all_discovered: List[DiscoveredEdge] = []
    t0 = time.time()

    async def _process_window(win_nodes: List[Dict[str, Any]], idx: int):
        async with sem:
            return await _discover_in_window(
                win_nodes, existing_edges, existing_pairs, llm_func, idx,
            )

    tasks = [_process_window(w, i) for i, w in enumerate(windows)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    seen = set()
    for r in results:
        if isinstance(r, Exception):
            _log.warning("[edge_discovery] window failed: %s", r)
            continue
        for edge in r:
            key = (edge.source, edge.target)
            rev = (edge.target, edge.source)
            if key in seen or rev in seen or key in existing_pairs:
                continue
            seen.add(key)
            all_discovered.append(edge)

    elapsed = time.time() - t0
    _log.info("[edge_discovery] completed in %.1fs — discovered %d new edges",
              elapsed, len(all_discovered))
    return all_discovered


def _build_windows(
    nodes: List[Dict[str, Any]], window_size: int, overlap: int,
) -> List[List[Dict[str, Any]]]:
    if len(nodes) <= window_size:
        return [nodes]
    step = max(1, window_size - overlap)
    windows = []
    for start in range(0, len(nodes), step):
        chunk = nodes[start : start + window_size]
        if len(chunk) >= 3:
            windows.append(chunk)
    return windows


def _format_entities_block(nodes: List[Dict[str, Any]]) -> str:
    lines = []
    for n in nodes:
        name = n.get("id") or n.get("entity_id") or ""
        etype = n.get("entity_type", "unknown")
        desc = (n.get("description") or "")[:200]
        lines.append(f"- **{name}** (type: {etype}): {desc}")
    return "\n".join(lines)


def _format_existing_edges(edges: List[Dict[str, Any]], node_names: set) -> str:
    relevant = []
    for e in edges:
        s, t = str(e.get("source", "")), str(e.get("target", ""))
        if s in node_names or t in node_names:
            kw = e.get("keywords", "related")
            relevant.append(f"- {s} -> {t}: {kw}")
    if not relevant:
        return "(无)"
    return "\n".join(relevant[:50])


async def _discover_in_window(
    win_nodes: List[Dict[str, Any]],
    all_edges: List[Dict[str, Any]],
    existing_pairs: set,
    llm_func: Callable,
    window_idx: int,
) -> List[DiscoveredEdge]:
    node_names = {str(n.get("id") or n.get("entity_id") or "") for n in win_nodes}
    entities_block = _format_entities_block(win_nodes)
    existing_block = _format_existing_edges(all_edges, node_names)

    prompt = EDGE_DISCOVERY_PROMPT.format(
        entities_block=entities_block,
        existing_edges_block=existing_block,
    )

    try:
        raw = await llm_func(prompt)
    except Exception as exc:
        _log.warning("[edge_discovery] LLM call failed for window %d: %s", window_idx, exc)
        return []

    return _parse_edges(raw, node_names, existing_pairs)


def _parse_edges(
    raw: str, valid_names: set, existing_pairs: set,
) -> List[DiscoveredEdge]:
    text = raw.strip()
    # Extract JSON array from response
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        items = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []

    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        src = str(item.get("source", "")).strip()
        tgt = str(item.get("target", "")).strip()
        kw = str(item.get("keywords", "related")).strip()
        desc = str(item.get("description", "")).strip()
        if not src or not tgt or src == tgt:
            continue
        if src not in valid_names or tgt not in valid_names:
            continue
        if (src, tgt) in existing_pairs or (tgt, src) in existing_pairs:
            continue
        results.append(DiscoveredEdge(source=src, target=tgt, keywords=kw, description=desc))
    return results
