# docthinker/kg_expansion/expander.py
"""
知识图谱 LLM 扩展：分层送入 + 多角度 prompting

- 送社区摘要而非原始图谱 → 控制 token
- 明确列出已有实体 → 避免重复
- 多角度分批请求 → 数量多且多样
- embedding 语义去重 → 质量保障
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

EXPANSION_SOURCE_ID = "llm_expansion"
# 扩展节点在存储中的显式标记：is_expanded=1 表示 LLM 扩展节点，0 或缺失表示原始节点
EXPANDED_NODE_FLAG = "1"

# 多角度 prompt 配置
EXPANSION_ANGLES: List[Tuple[str, str]] = [
    ("上位概念", "这些实体的父类、所属领域是什么？"),
    ("下位概念", "这些实体可以细分出哪些子概念？"),
    ("因果关联", "这些实体的前置条件和后续影响是什么？"),
    ("跨领域类比", "在其他领域有没有类似的概念？"),
    ("对立面", "与这些实体相对立或互补的概念是什么？"),
    ("应用场景", "这些实体在实际中会涉及哪些具体场景？"),
    ("时间维度", "这些实体的历史演变和未来趋势涉及什么？"),
]

PROMPT_TEMPLATE = """你是一个知识图谱扩展专家。

## 已有知识图谱摘要
{community_summaries}

## 已有实体（不要重复这些）
{existing_entities}

## 你的任务
从【{angle_name}】的角度（{angle_hint}），联想出与上述知识图谱相关但尚未包含的知识实体。

要求：
1. 数量不少于{min_count}个
2. 不得与已有实体重复或高度相似
3. 每个实体给出一句话说明它与图谱的潜在关联
4. 严格输出 JSON 数组，不要其他文字：
[
  {{"entity": "实体名", "reason": "潜在关联说明"}},
  ...
]
"""


def _build_community_summary(nodes_data: List[Dict], edges_data: List[Dict]) -> str:
    """从节点和边构建社区摘要（小图时整图当作一个社区）。"""
    entity_names = [n.get("id", "") for n in nodes_data if n.get("id")]
    entity_types = {}
    for n in nodes_data:
        eid = n.get("id")
        if eid:
            entity_types[eid] = n.get("entity_type", "unknown")

    if not entity_names:
        return "该知识图谱为空。"

    edges_text = []
    for e in edges_data[:50]:
        u = e.get("source", "")
        v = e.get("target", "")
        kw = e.get("keywords", "related")
        if u and v:
            edges_text.append(f"{u} -{kw}- {v}")

    summary = f"该知识图谱包含 {len(entity_names)} 个实体：{', '.join(entity_names[:30])}"
    if len(entity_names) > 30:
        summary += " 等"
    summary += "。"
    if edges_text:
        summary += f" 主要关系包括：{'；'.join(edges_text[:15])}"
        if len(edges_text) > 15:
            summary += " 等"
        summary += "。"
    return summary


def _parse_llm_json(text: str) -> List[Dict[str, str]]:
    """从 LLM 输出中解析 JSON 数组。"""
    text = text.strip()
    # 尝试提取 [...] 部分
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [
                {"entity": str(x.get("entity", "")).strip(), "reason": str(x.get("reason", "")).strip()}
                for x in data
                if isinstance(x, dict) and x.get("entity")
            ]
    except json.JSONDecodeError:
        pass
    return []


def _string_dedup(entities: List[Dict], existing: Set[str]) -> List[Dict]:
    """字面去重：过滤与已有实体相同或高度相似的。"""
    existing_lower = {e.lower().strip() for e in existing}
    result = []
    for item in entities:
        name = (item.get("entity") or "").strip()
        if not name:
            continue
        if name.lower() in existing_lower:
            continue
        # 包含关系也跳过（如 "X的父类" 与 X 重复）
        if any(ex in name for ex in existing):
            continue
        result.append(item)
        existing_lower.add(name.lower())
    return result


class KGExpander:
    """知识图谱 LLM 扩展器：多角度联想生成新节点。"""

    def __init__(
        self,
        *,
        llm_func: Callable[..., Any],
        embedding_func: Optional[Callable[..., Any]] = None,
        angles: Optional[List[Tuple[str, str]]] = None,
        min_per_angle: int = 20,
        semantic_dedup_threshold: float = 0.92,
    ):
        self.llm_func = llm_func
        self.embedding_func = embedding_func
        self.angles = angles or EXPANSION_ANGLES
        self.min_per_angle = min_per_angle
        self.semantic_dedup_threshold = semantic_dedup_threshold

    async def _call_llm(self, prompt: str) -> str:
        if asyncio.iscoroutinefunction(self.llm_func):
            return await self.llm_func(prompt)
        return self.llm_func(prompt)

    async def _embed(self, texts: List[str]) -> List[List[float]]:
        if not self.embedding_func or not texts:
            return []
        try:
            fn = self.embedding_func
            if asyncio.iscoroutinefunction(fn):
                out = await fn(texts)
            else:
                out = fn(texts)
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

    def _suggest_expansion(
        self,
        community_summaries: str,
        existing_entities: List[str],
        angle_name: str,
        angle_hint: str,
    ) -> str:
        entities_str = "、".join(existing_entities[:80])
        if len(existing_entities) > 80:
            entities_str += " 等"
        return PROMPT_TEMPLATE.format(
            community_summaries=community_summaries,
            existing_entities=entities_str,
            angle_name=angle_name,
            angle_hint=angle_hint,
            min_count=self.min_per_angle,
        )

    async def expand(
        self,
        nodes_data: List[Dict],
        edges_data: List[Dict],
        *,
        angle_indices: Optional[List[int]] = None,
        apply_to_graph: Optional[Any] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        执行知识图谱扩展。

        - nodes_data, edges_data: 当前图谱的节点和边（来自 get_all_nodes/get_all_edges）
        - angle_indices: 使用哪些角度 [0..6]，默认全部
        - apply_to_graph: 若提供（chunk_entity_relation_graph），将新节点写入图谱
        - session_id: 可选，用于 session 级图谱

        Returns:
            {
                "added": [{"entity": str, "reason": str, "angle": str}],
                "suggested": [...],  # 若未 apply 则全部在 suggested
                "community_summary": str,
                "angles_used": [...],
            }
        """
        existing_ids = {n.get("id", "") for n in nodes_data if n.get("id")}
        existing_names = list(existing_ids)

        community_summary = _build_community_summary(nodes_data, edges_data)

        indices = angle_indices if angle_indices is not None else list(range(len(self.angles)))
        all_suggestions: List[Dict[str, Any]] = []

        for i in indices:
            if i >= len(self.angles):
                continue
            angle_name, angle_hint = self.angles[i]
            prompt = self._suggest_expansion(
                community_summary, existing_names, angle_name, angle_hint
            )
            try:
                resp = await self._call_llm(prompt)
                parsed = _parse_llm_json(resp)
                for p in parsed:
                    p["angle"] = angle_name
                all_suggestions.extend(parsed)
            except Exception:
                continue

        # 字面去重
        deduped = _string_dedup(all_suggestions, existing_ids)
        seen_names: Set[str] = set()
        unique_suggestions: List[Dict] = []
        for s in deduped:
            name = (s.get("entity") or "").strip()
            if name and name not in seen_names:
                seen_names.add(name)
                unique_suggestions.append(s)

        # 可选：语义去重
        if self.embedding_func and unique_suggestions and self.semantic_dedup_threshold < 1.0:
            try:
                names = [s["entity"] for s in unique_suggestions]
                embs = await self._embed(names)
                if len(embs) == len(names):
                    to_keep = []
                    for j, emb in enumerate(embs):
                        skip = False
                        for k, e2 in enumerate(embs[:j]):
                            if k >= len(to_keep):
                                break
                            # 简单余弦相似度
                            dot = sum(a * b for a, b in zip(emb, e2))
                            na = (sum(a * a for a in emb) ** 0.5) or 1e-8
                            nb = (sum(b * b for b in e2) ** 0.5) or 1e-8
                            sim = dot / (na * nb)
                            if sim >= self.semantic_dedup_threshold:
                                skip = True
                                break
                        if not skip:
                            to_keep.append(unique_suggestions[j])
                    unique_suggestions = to_keep
            except Exception:
                pass

        added: List[Dict] = []
        if apply_to_graph and unique_suggestions:
            for s in unique_suggestions:
                entity_name = (s.get("entity") or "").strip()
                if not entity_name:
                    continue
                reason = s.get("reason", "")
                node_data = {
                    "entity_id": entity_name,
                    "entity_type": "concept",
                    "description": reason or entity_name,
                    "source_id": EXPANSION_SOURCE_ID,
                    "is_expanded": EXPANDED_NODE_FLAG,
                }
                try:
                    exists = False
                    if hasattr(apply_to_graph, "has_node") and asyncio.iscoroutinefunction(
                        getattr(apply_to_graph, "has_node")
                    ):
                        exists = await apply_to_graph.has_node(entity_name)
                    if not exists:
                        await apply_to_graph.upsert_node(entity_name, node_data)
                        added.append({**s, "entity": entity_name})
                    else:
                        cur = None
                        if hasattr(apply_to_graph, "get_node"):
                            get_fn = getattr(apply_to_graph, "get_node")
                            cur = await get_fn(entity_name) if asyncio.iscoroutinefunction(get_fn) else get_fn(entity_name)
                        if cur and str(cur.get("source_id") or "").strip() == EXPANSION_SOURCE_ID:
                            if cur.get("is_expanded") not in (1, "1"):
                                await apply_to_graph.upsert_node(entity_name, node_data)
                                added.append({**s, "entity": entity_name})
                except Exception:
                    continue
            if added and hasattr(apply_to_graph, "index_done_callback"):
                try:
                    cb = apply_to_graph.index_done_callback
                    try:
                        await cb(force_save=True)
                    except TypeError:
                        await cb()
                except Exception:
                    pass

        added_ids = {a["entity"] for a in added}
        suggested = [s for s in unique_suggestions if (s.get("entity") or "").strip() not in added_ids]

        return {
            "added": added,
            "suggested": unique_suggestions if not apply_to_graph else suggested,
            "community_summary": community_summary,
            "angles_used": [self.angles[i][0] for i in indices],
        }
