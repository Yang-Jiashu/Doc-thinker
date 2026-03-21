"""
KG Expansion via LLM: two-part expansion with rich entities and edges.

Part A – Cluster-based expansion: for each density-cluster summary,
         generate related entities with descriptions and edges.
Part B – Top-node expansion: for the top-N highest-degree nodes,
         generate entities across 6 cognitive dimensions.

All LLM calls run in parallel via asyncio.gather.
Every call is traced through LLMTrace for full observability.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

_log = logging.getLogger("docthinker.kg_expansion.expander")

EXPANSION_SOURCE_ID = "llm_expansion"
EXPANDED_NODE_FLAG = "1"

# ---------------------------------------------------------------------------
# Prompt A – Cluster-based expansion
# ---------------------------------------------------------------------------
CLUSTER_EXPAND_PROMPT = """你是一名知识图谱构建专家。

## 背景
以下是从文档中提取的一组语义紧密的实体簇（通过向量密度聚类自动发现）：

### 簇摘要
{cluster_summary}

### 簇内实体
{cluster_entities}

### 当前图谱中已有的所有实体（禁止重复）
{all_existing_entities}

## 任务
基于此簇的主题，推理出 **尚未存在于图谱中** 的相关知识。你需要：
1. 思考这个簇代表的核心主题是什么
2. 从以下角度联想新知识：
   - 该主题的前提条件或理论基础
   - 该主题的实际应用或案例
   - 该主题的相关对比概念
   - 该主题的演进历史或最新发展
3. 为每个新实体提供**具体的、有事实依据的描述**（不少于 20 字）
4. 为每个新实体指明它与簇内**哪个已有实体**存在关系，并说明关系类型

## 输出格式
严格输出 JSON 数组，不要其他文字：
[
  {{
    "entity": "新实体名称",
    "entity_type": "concept|person|technology|method|organization|event|other",
    "description": "该实体的具体描述，包含关键事实，不少于20字",
    "edges": [
      {{
        "target": "簇内已有实体名",
        "relation": "关系类型关键词",
        "description": "为什么存在这个关系的简要说明"
      }}
    ]
  }}
]

## 质量要求
- 数量：不少于 {min_count} 个
- 描述必须具体：❌ "一种算法" → ✅ "一种基于密度的聚类算法，由Ester等人1996年提出，核心思想是..."
- 每个实体至少有 1 条边连接到簇内已有实体
- 不得与已有实体重复或高度相似
"""

# ---------------------------------------------------------------------------
# Prompt B – Top-node expansion (6 cognitive dimensions)
# ---------------------------------------------------------------------------
TOPNODE_EXPAND_PROMPT = """你是一名知识图谱构建专家，擅长知识推理和关联发现。

## 背景
以下是知识图谱中连接最紧密的核心实体（按连接度排序的前 {top_n} 个）：

{top_nodes_with_descriptions}

### 当前图谱中已有的所有实体（禁止重复）
{all_existing_entities}

## 任务
围绕这些核心实体，从 **6 个认知维度** 推理新知识：

| 维度 | 说明 | 示例 |
|------|------|------|
| 层级关系 | 上位概念、下位概念、所属领域 | "机器学习" → 上位: "人工智能"; 下位: "强化学习" |
| 因果关联 | 前置条件、后续影响、因果链 | "过拟合" → 因: "训练数据不足"; 果: "泛化能力下降" |
| 类比迁移 | 其他领域的相似概念 | "注意力机制" ↔ "人类选择性注意" |
| 对立互补 | 相对立或互补的概念 | "监督学习" ↔ "无监督学习" |
| 时间演进 | 历史渊源、版本迭代、未来趋势 | "BERT" → "GPT" → "LLaMA" |
| 应用实践 | 具体场景、工具、案例 | "Transformer" → 应用: "机器翻译、文本摘要" |

## 输出格式
严格输出 JSON 数组：
[
  {{
    "entity": "新实体名称",
    "entity_type": "concept|person|technology|method|organization|event|other",
    "description": "具体描述（含关键事实，不少于20字）",
    "dimension": "层级关系|因果关联|类比迁移|对立互补|时间演进|应用实践",
    "edges": [
      {{
        "target": "已有核心实体名",
        "relation": "关系关键词",
        "description": "关系说明"
      }}
    ]
  }}
]

## 质量要求
- 数量：不少于 {min_count} 个
- 每个维度至少覆盖 2 个新实体
- 描述必须具体且有事实依据
- 每个实体至少 1 条边
- 禁止生成已有实体的变体（如 "XX的应用"、"XX方法"）
"""

# ---------------------------------------------------------------------------
# Prompt C – Self-validation
# ---------------------------------------------------------------------------
VALIDATION_PROMPT = """你是一名知识图谱质量审核员。

## 任务
以下是 LLM 自动扩展生成的候选实体列表。请逐一审核：

{candidate_entities_json}

## 审核标准
对每个候选实体，检查：
1. **事实性**：描述是否包含可验证的事实？（0-1分）
2. **非冗余性**：是否与以下已有实体语义重复？（0-1分）
   已有实体：{existing_entities}
3. **边有效性**：声称的关系是否合理？目标实体是否存在？（0-1分）
4. **具体性**：描述是否足够具体，还是空泛的废话？（0-1分）

## 输出格式
[
  {{
    "entity": "候选实体名",
    "factuality": 0.8,
    "non_redundancy": 0.9,
    "edge_validity": 1.0,
    "specificity": 0.7,
    "overall_score": 0.85,
    "verdict": "accept|revise|reject",
    "revision_note": "如果 verdict=revise，说明需要修改什么"
  }}
]

只输出 JSON，不要其他文字。"""


# ── Helpers ─────────────────────────────────────────────────────────

def _parse_rich_entities(text: str) -> List[Dict[str, Any]]:
    """Parse LLM output into a list of entity dicts with edges."""
    text = text.strip()
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
        if not isinstance(data, list):
            return []
        results: List[Dict[str, Any]] = []
        for x in data:
            if not isinstance(x, dict):
                continue
            name = str(x.get("entity") or "").strip()
            if not name:
                continue
            edges_raw = x.get("edges") or []
            edges: List[Dict[str, str]] = []
            for e in (edges_raw if isinstance(edges_raw, list) else []):
                if isinstance(e, dict) and e.get("target"):
                    edges.append({
                        "target": str(e["target"]).strip(),
                        "relation": str(e.get("relation") or "related_to").strip(),
                        "description": str(e.get("description") or "").strip(),
                    })
            results.append({
                "entity": name,
                "entity_type": str(x.get("entity_type") or "concept").strip(),
                "description": str(x.get("description") or "").strip(),
                "dimension": str(x.get("dimension") or "").strip(),
                "edges": edges,
            })
        return results
    except json.JSONDecodeError:
        _log.warning("[expander] JSON parse failed, response length=%d", len(text))
        return []


def _parse_validation(text: str) -> List[Dict[str, Any]]:
    """Parse self-validation response."""
    text = text.strip()
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _string_dedup(entities: List[Dict[str, Any]], existing: Set[str]) -> List[Dict[str, Any]]:
    """Filter entities that duplicate existing names."""
    existing_lower = {e.lower().strip() for e in existing}
    result: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for item in entities:
        name = (item.get("entity") or "").strip()
        if not name:
            continue
        low = name.lower()
        if low in existing_lower or low in seen:
            continue
        if any(ex in name for ex in existing if len(ex) > 2):
            continue
        seen.add(low)
        result.append(item)
    return result


def _pick_top_nodes(
    nodes_data: List[Dict[str, Any]],
    edges_data: List[Dict[str, Any]],
    top_n: int = 50,
) -> List[Dict[str, Any]]:
    """Select the top-N nodes by degree (non-expanded only)."""
    degree: Dict[str, int] = {}
    for e in edges_data:
        src = str(e.get("source") or "").strip()
        tgt = str(e.get("target") or "").strip()
        if src:
            degree[src] = degree.get(src, 0) + 1
        if tgt:
            degree[tgt] = degree.get(tgt, 0) + 1

    scored: List[Tuple[int, Dict[str, Any]]] = []
    for n in nodes_data:
        nid = str(n.get("id") or n.get("entity_id") or "").strip()
        if not nid:
            continue
        is_exp = str(n.get("source_id") or "") == EXPANSION_SOURCE_ID or str(n.get("is_expanded") or "") == "1"
        if is_exp:
            continue
        scored.append((degree.get(nid, 0), n))
    scored.sort(key=lambda x: -x[0])
    return [n for _, n in scored[:top_n]]


# ── Main Expander ───────────────────────────────────────────────────

class KGExpander:
    """Two-part KG expansion with tracing and self-validation."""

    def __init__(
        self,
        *,
        llm_func: Callable[..., Any],
        embedding_func: Optional[Callable[..., Any]] = None,
        min_per_cluster: int = 8,
        min_per_topnode: int = 15,
        semantic_dedup_threshold: float = 0.92,
        enable_validation: bool = True,
        validation_min_score: float = 0.6,
        session_id: str = "",
    ):
        self.llm_func = llm_func
        self.embedding_func = embedding_func
        self.min_per_cluster = min_per_cluster
        self.min_per_topnode = min_per_topnode
        self.semantic_dedup_threshold = semantic_dedup_threshold
        self.enable_validation = enable_validation
        self.validation_min_score = validation_min_score
        self.session_id = session_id

    async def _call_llm(self, prompt: str, *, sub_stage: str = "", metadata: Optional[Dict] = None) -> str:
        from ..llm_trace import LLMTrace
        tracer = LLMTrace(stage="expansion", sub_stage=sub_stage, session_id=self.session_id)
        return await tracer.call(self.llm_func, prompt, metadata=metadata)

    async def _embed(self, texts: List[str]) -> List[List[float]]:
        if not self.embedding_func or not texts:
            return []
        try:
            fn = self.embedding_func
            out = await fn(texts) if asyncio.iscoroutinefunction(fn) else fn(texts)
            if out is None:
                return []
            if hasattr(out, "tolist"):
                out = out.tolist()
            out = list(out) if out else []
            if out and not isinstance(out[0], (list, tuple)):
                return [out]
            return out
        except Exception:
            return []

    # ── Part A: Cluster-based expansion ──

    async def _expand_from_cluster(
        self,
        cluster_summary: Any,
        all_existing: List[str],
    ) -> List[Dict[str, Any]]:
        cid = getattr(cluster_summary, "cluster_id", 0)
        summary_text = getattr(cluster_summary, "summary", "")
        node_ids = getattr(cluster_summary, "node_ids", [])
        node_descs = getattr(cluster_summary, "node_descriptions", [])

        entities_lines: List[str] = []
        for name, desc in zip(node_ids, node_descs):
            line = f"- {name}"
            if desc:
                line += f": {desc[:150]}"
            entities_lines.append(line)

        prompt = CLUSTER_EXPAND_PROMPT.format(
            cluster_summary=summary_text,
            cluster_entities="\n".join(entities_lines),
            all_existing_entities="、".join(all_existing[:120]),
            min_count=self.min_per_cluster,
        )
        resp = await self._call_llm(
            prompt,
            sub_stage="cluster_expand",
            metadata={"cluster_id": cid, "node_count": len(node_ids)},
        )
        return _parse_rich_entities(resp)

    # ── Part B: Top-node expansion ──

    async def _expand_from_top_nodes(
        self,
        top_nodes: List[Dict[str, Any]],
        all_existing: List[str],
    ) -> List[Dict[str, Any]]:
        lines: List[str] = []
        for n in top_nodes:
            nid = str(n.get("id") or n.get("entity_id") or "")
            desc = str(n.get("description") or "")
            etype = str(n.get("entity_type") or "")
            line = f"- {nid}"
            if etype:
                line += f" [{etype}]"
            if desc:
                line += f": {desc[:150]}"
            lines.append(line)

        prompt = TOPNODE_EXPAND_PROMPT.format(
            top_n=len(top_nodes),
            top_nodes_with_descriptions="\n".join(lines),
            all_existing_entities="、".join(all_existing[:120]),
            min_count=self.min_per_topnode,
        )
        resp = await self._call_llm(
            prompt,
            sub_stage="topnode_expand",
            metadata={"top_n": len(top_nodes)},
        )
        return _parse_rich_entities(resp)

    # ── Self-validation ──

    async def _validate(
        self,
        candidates: List[Dict[str, Any]],
        existing_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Run self-validation and filter low-quality candidates."""
        if not candidates or not self.enable_validation:
            return candidates

        batch_size = 25
        accepted: List[Dict[str, Any]] = []
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start: start + batch_size]
            prompt = VALIDATION_PROMPT.format(
                candidate_entities_json=json.dumps(batch, ensure_ascii=False, indent=2),
                existing_entities="、".join(existing_names[:100]),
            )
            try:
                resp = await self._call_llm(
                    prompt,
                    sub_stage="validation",
                    metadata={"batch_start": start, "batch_size": len(batch)},
                )
                verdicts = _parse_validation(resp)
                verdict_map = {str(v.get("entity") or "").strip().lower(): v for v in verdicts}
                for c in batch:
                    cname = (c.get("entity") or "").strip().lower()
                    v = verdict_map.get(cname)
                    if v is None:
                        accepted.append(c)
                        continue
                    score = float(v.get("overall_score") or 1.0)
                    verdict = str(v.get("verdict") or "accept").lower()
                    if verdict == "reject" or score < self.validation_min_score:
                        _log.info("[validate] rejected: %s (score=%.2f)", c.get("entity"), score)
                        continue
                    c["validation_score"] = score
                    accepted.append(c)
            except Exception as exc:
                _log.warning("[validate] validation call failed, accepting batch: %s", exc)
                accepted.extend(batch)

        _log.info("[validate] %d/%d candidates passed validation", len(accepted), len(candidates))
        return accepted

    # ── Semantic dedup ──

    async def _semantic_dedup(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.embedding_func or not candidates or self.semantic_dedup_threshold >= 1.0:
            return candidates
        names = [c.get("entity", "") for c in candidates]
        embs = await self._embed(names)
        if len(embs) != len(names):
            return candidates
        import numpy as np
        emb_matrix = np.array(embs)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        normed = emb_matrix / norms

        keep: List[int] = []
        for i in range(len(normed)):
            skip = False
            for j in keep:
                sim = float(np.dot(normed[i], normed[j]))
                if sim >= self.semantic_dedup_threshold:
                    skip = True
                    break
            if not skip:
                keep.append(i)
        _log.info("[dedup] semantic dedup: %d → %d", len(candidates), len(keep))
        return [candidates[i] for i in keep]

    # ── Main entry point ──

    async def expand(
        self,
        nodes_data: List[Dict[str, Any]],
        edges_data: List[Dict[str, Any]],
        *,
        cluster_summaries: Optional[List[Any]] = None,
        apply_to_graph: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute two-part expansion pipeline.

        Returns dict with ``added_nodes``, ``added_edges``, ``cluster_count``,
        ``top_node_count``, ``validated``, ``raw_count``.
        """
        existing_ids = {
            str(n.get("id") or n.get("entity_id") or "").strip()
            for n in nodes_data if n.get("id") or n.get("entity_id")
        }
        existing_names = sorted(existing_ids)

        # ── Part A: cluster-based (parallel across clusters) ──
        cluster_tasks = []
        if cluster_summaries:
            for cs in cluster_summaries:
                cluster_tasks.append(self._expand_from_cluster(cs, existing_names))

        # ── Part B: top-node (single call) ──
        top_nodes = _pick_top_nodes(nodes_data, edges_data, top_n=50)

        # Run A + B in parallel
        all_results: List[Any] = []
        tasks = list(cluster_tasks)
        if top_nodes:
            tasks.append(self._expand_from_top_nodes(top_nodes, existing_names))

        if tasks:
            all_results = await asyncio.gather(*tasks, return_exceptions=True)

        all_candidates: List[Dict[str, Any]] = []
        for r in all_results:
            if isinstance(r, list):
                all_candidates.extend(r)
            elif isinstance(r, Exception):
                _log.warning("[expand] one expansion task failed: %s", r)

        raw_count = len(all_candidates)
        _log.info("[expand] raw candidates: %d (cluster_tasks=%d, top_nodes=%d)",
                  raw_count, len(cluster_tasks), len(top_nodes))

        # ── Dedup ──
        deduped = _string_dedup(all_candidates, existing_ids)
        deduped = await self._semantic_dedup(deduped)

        # ── Validate ──
        validated = await self._validate(deduped, existing_names)

        # ── Apply to graph ──
        added_nodes: List[Dict[str, Any]] = []
        added_edges: List[Dict[str, Any]] = []

        if apply_to_graph and validated:
            G = apply_to_graph
            for c in validated:
                entity_name = (c.get("entity") or "").strip()
                if not entity_name:
                    continue
                node_data = {
                    "entity_id": entity_name,
                    "entity_type": c.get("entity_type", "concept"),
                    "description": c.get("description", ""),
                    "source_id": EXPANSION_SOURCE_ID,
                    "is_expanded": EXPANDED_NODE_FLAG,
                }
                try:
                    exists = False
                    if hasattr(G, "has_node"):
                        fn = getattr(G, "has_node")
                        exists = await fn(entity_name) if asyncio.iscoroutinefunction(fn) else fn(entity_name)
                    if exists:
                        continue
                    await G.upsert_node(entity_name, node_data)
                    added_nodes.append(c)

                    for edge in c.get("edges", []):
                        target = str(edge.get("target") or "").strip()
                        if not target:
                            continue
                        has_target = False
                        if hasattr(G, "has_node"):
                            fn = getattr(G, "has_node")
                            has_target = await fn(target) if asyncio.iscoroutinefunction(fn) else fn(target)
                        if has_target:
                            edge_props = {
                                "keywords": edge.get("relation", "related_to"),
                                "description": edge.get("description", ""),
                                "source_id": EXPANSION_SOURCE_ID,
                                "weight": 0.6,
                            }
                            await G.upsert_edge(entity_name, target, edge_props)
                            added_edges.append({
                                "source": entity_name,
                                "target": target,
                                "relation": edge.get("relation", "related_to"),
                                "description": edge.get("description", ""),
                            })
                        else:
                            _log.debug("[expand] edge target '%s' not in graph, skipped", target)
                except Exception as exc:
                    _log.warning("[expand] failed to upsert node '%s': %s", entity_name, exc)
                    continue

            if added_nodes and hasattr(G, "index_done_callback"):
                try:
                    cb = G.index_done_callback
                    try:
                        await cb(force_save=True)
                    except TypeError:
                        await cb()
                except Exception:
                    pass

        return {
            "added_nodes": added_nodes,
            "added_edges": added_edges,
            "suggested": validated if not apply_to_graph else [],
            "raw_count": raw_count,
            "deduped_count": len(deduped),
            "validated_count": len(validated),
            "cluster_count": len(cluster_tasks),
            "top_node_count": len(top_nodes),
        }
