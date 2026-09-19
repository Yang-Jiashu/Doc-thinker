"""Question-aware graph reasoning policies.

This module keeps the source graph authoritative and treats discovered edges
as penalised candidates.  Multi-hop questions use evidence-constrained path
search; exploratory questions use personalised PageRank plus MMR diversity.
"""

from __future__ import annotations

import math
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from docthinker.retrieval_policy import (
    relation_confidence,
    relation_has_evidence,
    truthy_metadata,
)


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+")

_FAITHFUL_TERMS = (
    "原文", "文档中", "资料中", "明确提到", "具体数值", "多少", "哪一家",
    "叫什么", "是否提到", "according to", "exact", "which supplier",
)
_PATH_TERMS = (
    "为什么", "如何导致", "原因链", "完整原因", "因果链", "传导路径",
    "连锁影响", "怎么影响", "如何影响", "追溯路径", "why", "how does",
    "causal", "chain", "lead to",
)
_EXPLORE_TERMS = (
    "还可能", "可能有哪些", "还有哪些", "潜在影响", "提出方案", "改进建议",
    "有哪些思路", "发散", "探索", "brainstorm", "what else", "possible",
    "potential", "ideas",
)


def _tokens(text: str) -> set[str]:
    output: set[str] = set()
    for raw in _TOKEN_RE.findall(str(text or "").lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", raw) and len(raw) > 1:
            output.update(raw[index : index + 2] for index in range(len(raw) - 1))
        else:
            output.add(raw)
    return output


def _similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True)
class QuestionPolicy:
    mode: str
    confidence: float
    reason: str

    def to_schema(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }


def classify_question(
    question: str,
    *,
    requested_mode: str = "auto",
    self_evolution_enabled: bool = True,
) -> QuestionPolicy:
    """Route a question conservatively to faithful, path, or explore mode."""
    requested = str(requested_mode or "auto").strip().lower()
    if not self_evolution_enabled or requested == "off":
        return QuestionPolicy("faithful", 1.0, "self_evolution_disabled")
    if requested in {"faithful", "path", "explore"}:
        return QuestionPolicy(requested, 1.0, "explicit_override")

    text = str(question or "").strip().lower()
    scores = {
        "faithful": sum(1 for term in _FAITHFUL_TERMS if term in text),
        "path": sum(1 for term in _PATH_TERMS if term in text),
        "explore": sum(1 for term in _EXPLORE_TERMS if term in text),
    }
    best_score = max(scores.values(), default=0)
    if best_score <= 0:
        return QuestionPolicy("faithful", 0.55, "uncertain_defaults_to_faithful")

    # Exploration wins mixed prompts such as "还可能有哪些影响", while exact
    # document questions remain conservative even if they contain "为什么".
    if scores["explore"] == best_score:
        mode = "explore"
    elif scores["faithful"] >= scores["path"]:
        mode = "faithful"
    else:
        mode = "path"
    total = sum(scores.values())
    confidence = min(0.98, 0.65 + 0.1 * best_score + 0.05 * (best_score / max(1, total)))
    return QuestionPolicy(mode, confidence, f"keyword_route:{scores}")


def _node_name(node: Dict[str, Any]) -> str:
    return str(node.get("id") or node.get("entity_id") or node.get("name") or "").strip()


def _edge_endpoints(edge: Dict[str, Any]) -> Tuple[str, str]:
    source = str(edge.get("source") or edge.get("src_id") or "").strip()
    target = str(edge.get("target") or edge.get("tgt_id") or edge.get("target_id") or "").strip()
    return source, target


def _edge_text(edge: Dict[str, Any]) -> str:
    source, target = _edge_endpoints(edge)
    return " ".join(
        str(value or "")
        for value in (
            source,
            edge.get("keywords"),
            edge.get("description"),
            edge.get("relation"),
            target,
        )
    )


def _candidate_edge(edge: Dict[str, Any]) -> bool:
    return truthy_metadata(edge.get("is_discovered"))


def _eligible_edge(
    edge: Dict[str, Any],
    *,
    allow_candidates: bool,
    min_candidate_confidence: float,
) -> bool:
    source, target = _edge_endpoints(edge)
    if not source or not target or source == target:
        return False
    if not _candidate_edge(edge):
        return True
    return bool(
        allow_candidates
        and truthy_metadata(edge.get("query_eligible"))
        and relation_confidence(edge) >= min_candidate_confidence
        and relation_has_evidence(edge)
    )


def _extract_anchors(
    question: str,
    nodes: Sequence[Dict[str, Any]],
    *,
    limit: int = 5,
    add_fallback_to_exact: bool = False,
) -> List[str]:
    text = str(question or "").lower()
    exact: List[Tuple[int, str]] = []
    fallback: List[Tuple[float, str]] = []
    for node in nodes:
        name = _node_name(node)
        if not name:
            continue
        position = text.find(name.lower())
        if position >= 0:
            exact.append((position, name))
            continue
        description = str(node.get("description") or "")
        score = max(_similarity(question, name), _similarity(question, description))
        if score > 0:
            fallback.append((score, name))
    exact.sort(key=lambda item: item[0])
    fallback.sort(key=lambda item: (-item[0], item[1]))
    names = [name for _, name in exact]
    if names and not add_fallback_to_exact:
        return names[:limit]
    names.extend(name for _, name in fallback if name not in names)
    return names[: max(2, limit)]


def _edge_quality(edge: Dict[str, Any], question: str) -> float:
    candidate = _candidate_edge(edge)
    confidence = relation_confidence(edge)
    if confidence <= 0:
        confidence = 0.82 if candidate else 0.95
    grounded = 1.0 if (not candidate or relation_has_evidence(edge)) else 0.0
    relevance = _similarity(question, _edge_text(edge))
    provenance = 0.65 if candidate else 1.0
    return 0.35 * relevance + 0.30 * grounded + 0.20 * confidence + 0.15 * provenance


def _path_score(path_edges: Sequence[Dict[str, Any]], question: str) -> float:
    if not path_edges:
        return 0.0
    qualities = [_edge_quality(edge, question) for edge in path_edges]
    candidate_ratio = sum(_candidate_edge(edge) for edge in path_edges) / len(path_edges)
    source_ids = {
        str(edge.get("source_id") or "").strip()
        for edge in path_edges
        if str(edge.get("source_id") or "").strip()
    }
    source_diversity = min(1.0, len(source_ids) / max(1, len(path_edges)))
    length_penalty = max(0, len(path_edges) - 4) * 0.025
    score = (
        0.70 * (sum(qualities) / len(qualities))
        + 0.20
        + 0.10 * source_diversity
        - 0.18 * candidate_ratio
        - length_penalty
    )
    return max(0.0, min(1.0, score))


@dataclass
class GraphPath:
    nodes: List[str]
    edges: List[Dict[str, Any]]
    score: float

    def to_schema(self) -> Dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "score": round(self.score, 4),
            "candidate_edges": sum(_candidate_edge(edge) for edge in self.edges),
            "hops": [
                {
                    "source": _edge_endpoints(edge)[0],
                    "target": _edge_endpoints(edge)[1],
                    "relation": str(edge.get("keywords") or edge.get("description") or "related"),
                    "candidate": _candidate_edge(edge),
                    "confidence": round(relation_confidence(edge), 4),
                    "source_id": str(edge.get("source_id") or ""),
                }
                for edge in self.edges
            ],
        }


def _extract_json_object(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


async def complete_path_from_evidence(
    *,
    question: str,
    start: str,
    goal: str,
    chunks: Sequence[Dict[str, Any]],
    llm_func: Any,
    max_hops: int = 8,
    min_confidence: float = 0.80,
) -> Tuple[GraphPath | None, Dict[str, Any]]:
    """Ask the LLM to bridge one known gap, then verify every quote locally.

    The generated path is query-local and is never persisted here.  A hop is
    accepted only when its quote is an exact substring of the declared chunk.
    """
    evidence: Dict[str, str] = {}
    blocks: List[str] = []
    for index, chunk in enumerate(chunks[:12]):
        chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or f"chunk-{index}")
        content = str(chunk.get("content") or chunk.get("text") or "").strip()
        if not content:
            continue
        evidence[chunk_id] = " ".join(content.split())
        blocks.append(f"[{chunk_id}]\n{content[:1200]}")
    if not evidence or not callable(llm_func):
        return None, {"reason": "no_bridge_evidence"}

    prompt = f"""你是证据约束的知识图谱路径修复器。
问题：{question}
起点：{start}
终点：{goal}

请只使用下面的原文片段，尝试构造从起点到终点的连续有向路径。
每一跳必须提供 chunk_id 和逐字 evidence_quote；不允许使用外部常识。
如果证据不足，请输出 {{"hops": []}}。
最多 {max_hops} 跳。严格输出 JSON：
{{"hops":[{{"source":"...","target":"...","relation":"...","confidence":0.9,"chunk_id":"...","evidence_quote":"原文逐字片段"}}]}}

原文片段：
{chr(10).join(blocks)}
"""
    try:
        raw = await llm_func(prompt)
    except Exception as exc:
        return None, {"reason": "llm_bridge_failed", "error": str(exc)}
    hops = _extract_json_object(raw).get("hops") or []
    if not isinstance(hops, list) or not 1 <= len(hops) <= max_hops:
        return None, {"reason": "invalid_hop_count"}

    path_nodes = [start]
    path_edges: List[Dict[str, Any]] = []
    current = start
    for hop in hops:
        if not isinstance(hop, dict):
            return None, {"reason": "invalid_hop_shape"}
        source = str(hop.get("source") or "").strip()
        target = str(hop.get("target") or "").strip()
        chunk_id = str(hop.get("chunk_id") or "").strip()
        quote = " ".join(str(hop.get("evidence_quote") or "").split())
        try:
            confidence = float(hop.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if source != current or not target or target in path_nodes:
            return None, {"reason": "discontinuous_or_cyclic_path"}
        if confidence < min_confidence:
            return None, {"reason": "low_confidence_hop"}
        if (
            len(quote) < 4
            or source not in quote
            or target not in quote
            or chunk_id not in evidence
            or quote not in evidence[chunk_id]
        ):
            return None, {"reason": "ungrounded_hop", "chunk_id": chunk_id}
        edge = {
            "source": source,
            "target": target,
            "keywords": str(hop.get("relation") or "related"),
            "description": str(hop.get("relation") or ""),
            "confidence": confidence,
            "is_discovered": "1",
            "query_eligible": "1",
            "evidence": json.dumps(
                [{"chunk_id": chunk_id, "quote": quote}], ensure_ascii=False
            ),
            "evidence_chunk_ids": json.dumps([chunk_id], ensure_ascii=False),
            "source_id": chunk_id,
            "provenance": "query_local_path_completion",
        }
        path_edges.append(edge)
        path_nodes.append(target)
        current = target
    if current != goal:
        return None, {"reason": "goal_not_reached"}
    average_confidence = sum(
        relation_confidence(edge) for edge in path_edges
    ) / len(path_edges)
    completion_score = (
        0.55 * average_confidence
        + 0.30  # every hop has an exact local quote
        + 0.15  # start and goal are explicitly constrained
        - max(0, len(path_edges) - 4) * 0.025
    )
    path = GraphPath(
        path_nodes,
        path_edges,
        max(0.0, min(1.0, completion_score)),
    )
    if path.score < 0.70:
        return None, {"reason": "path_score_below_threshold", "score": path.score}
    return path, {"reason": "query_local_bridge", "hops": len(path_edges)}


def evidence_constrained_paths(
    question: str,
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
    *,
    allow_candidates: bool = True,
    min_candidate_confidence: float = 0.80,
    max_hops: int = 8,
    beam_width: int = 20,
    top_k: int = 3,
    max_candidate_edges: int = 2,
) -> Tuple[List[GraphPath], Dict[str, Any]]:
    """Find high-quality directed paths between question-aligned anchors."""
    anchors = _extract_anchors(
        question,
        nodes,
        limit=2,
        add_fallback_to_exact=True,
    )
    if len(anchors) < 2:
        return [], {"anchors": anchors, "reason": "insufficient_anchors"}

    eligible = [
        edge for edge in edges
        if _eligible_edge(
            edge,
            allow_candidates=allow_candidates,
            min_candidate_confidence=min_candidate_confidence,
        )
    ]
    adjacency: Dict[str, List[Dict[str, Any]]] = {}
    reverse_adjacency: Dict[str, List[Dict[str, Any]]] = {}
    for edge in eligible:
        source, target = _edge_endpoints(edge)
        adjacency.setdefault(source, []).append(edge)
        reverse_adjacency.setdefault(target, []).append(edge)
    for values in list(adjacency.values()) + list(reverse_adjacency.values()):
        values.sort(key=lambda edge: _edge_quality(edge, question), reverse=True)

    exact_positions = {
        anchor: str(question).lower().find(anchor.lower()) for anchor in anchors
    }
    ordered = sorted(
        anchors,
        key=lambda name: (
            exact_positions[name] < 0,
            exact_positions[name] if exact_positions[name] >= 0 else anchors.index(name),
        ),
    )
    endpoint_pairs: List[Tuple[str, str]] = []
    if len(ordered) >= 2:
        endpoint_pairs.append((ordered[0], ordered[-1]))
    endpoint_pairs.extend(
        (left, right)
        for left in anchors[:3]
        for right in anchors[:3]
        if left != right and (left, right) not in endpoint_pairs
    )

    found: List[GraphPath] = []
    reached: set[str] = set()

    def expand_side(
        seed: str,
        side_adjacency: Dict[str, List[Dict[str, Any]]],
        *,
        reverse: bool,
    ) -> Dict[str, List[Tuple[List[str], List[Dict[str, Any]], float]]]:
        records: Dict[str, List[Tuple[List[str], List[Dict[str, Any]], float]]] = {
            seed: [([seed], [], 0.0)]
        }
        frontier = [([seed], [], 0.0)]
        side_depth = max(1, math.ceil(max_hops / 2))
        for _ in range(side_depth):
            expanded: List[Tuple[List[str], List[Dict[str, Any]], float]] = []
            for path_nodes, path_edges, _ in frontier:
                current = path_nodes[-1]
                reached.add(current)
                for edge in side_adjacency.get(current, []):
                    source, target = _edge_endpoints(edge)
                    next_node = source if reverse else target
                    if next_node in path_nodes:
                        continue
                    next_edges = path_edges + [edge]
                    if sum(_candidate_edge(item) for item in next_edges) > max_candidate_edges:
                        continue
                    next_nodes = path_nodes + [next_node]
                    score = _path_score(next_edges, question)
                    expanded.append((next_nodes, next_edges, score))
                    bucket = records.setdefault(next_node, [])
                    bucket.append((next_nodes, next_edges, score))
                    bucket.sort(key=lambda item: item[2], reverse=True)
                    del bucket[2:]
            expanded.sort(key=lambda item: item[2], reverse=True)
            frontier = expanded[:beam_width]
            if not frontier:
                break
        return records

    for start, goal in endpoint_pairs:
        forward = expand_side(start, adjacency, reverse=False)
        backward = expand_side(goal, reverse_adjacency, reverse=True)
        for meeting in set(forward) & set(backward):
            for forward_path in forward[meeting]:
                for backward_path in backward[meeting]:
                    forward_nodes, forward_edges, _ = forward_path
                    backward_nodes, backward_edges, _ = backward_path
                    combined_nodes = forward_nodes + list(reversed(backward_nodes[:-1]))
                    combined_edges = forward_edges + list(reversed(backward_edges))
                    if not combined_edges or len(combined_edges) > max_hops:
                        continue
                    if len(set(combined_nodes)) != len(combined_nodes):
                        continue
                    if combined_nodes[0] != start or combined_nodes[-1] != goal:
                        continue
                    candidate_count = sum(_candidate_edge(edge) for edge in combined_edges)
                    candidate_ratio = candidate_count / len(combined_edges)
                    score = _path_score(combined_edges, question)
                    if (
                        candidate_count <= max_candidate_edges
                        and candidate_ratio <= 0.34
                        and score >= 0.70
                    ):
                        found.append(GraphPath(combined_nodes, combined_edges, score))

    found.sort(key=lambda path: path.score, reverse=True)
    selected: List[GraphPath] = []
    for candidate in found:
        edge_set = {
            _edge_endpoints(edge) for edge in candidate.edges
        }
        too_similar = False
        for existing in selected:
            other = {_edge_endpoints(edge) for edge in existing.edges}
            union = edge_set | other
            overlap = len(edge_set & other) / len(union) if union else 1.0
            if overlap >= 0.75:
                too_similar = True
                break
        if not too_similar:
            selected.append(candidate)
        if len(selected) >= top_k:
            break

    diagnostic = {
        "anchors": anchors,
        "eligible_edges": len(eligible),
        "reached_frontier": sorted(reached)[:20],
        "reason": "ok" if selected else "no_continuous_evidence_path",
    }
    return selected, diagnostic


def exploratory_pagerank(
    question: str,
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
    *,
    min_candidate_confidence: float = 0.80,
    top_k: int = 8,
    damping: float = 0.85,
    iterations: int = 30,
    mmr_lambda: float = 0.70,
) -> Dict[str, Any]:
    """Expand around query anchors with PPR, then diversify results with MMR."""
    names = [_node_name(node) for node in nodes if _node_name(node)]
    if not names:
        return {"anchors": [], "items": [], "reason": "empty_graph"}
    anchors = _extract_anchors(question, nodes, limit=4)
    if not anchors:
        return {"anchors": [], "items": [], "reason": "insufficient_anchors"}

    eligible = [
        edge for edge in edges
        if _eligible_edge(
            edge,
            allow_candidates=True,
            min_candidate_confidence=min_candidate_confidence,
        )
    ]
    adjacency: Dict[str, List[Tuple[str, float, Dict[str, Any]]]] = {}
    for edge in eligible:
        source, target = _edge_endpoints(edge)
        weight = max(0.05, _edge_quality(edge, question))
        adjacency.setdefault(source, []).append((target, weight, edge))
        adjacency.setdefault(target, []).append((source, weight, edge))

    scores = {name: 0.0 for name in names}
    restart = 1.0 / len(anchors)
    for anchor in anchors:
        scores[anchor] = scores.get(anchor, 0.0) + restart
    for _ in range(max(1, iterations)):
        updated = {name: 0.0 for name in scores}
        for anchor in anchors:
            updated[anchor] = updated.get(anchor, 0.0) + (1.0 - damping) * restart
        for source, value in scores.items():
            neighbours = adjacency.get(source, [])
            if not neighbours:
                continue
            total_weight = sum(weight for _, weight, _ in neighbours)
            for target, weight, _ in neighbours:
                updated[target] = updated.get(target, 0.0) + damping * value * weight / total_weight
        scores = updated

    node_map = {_node_name(node): node for node in nodes if _node_name(node)}
    ranked: List[Tuple[float, str]] = []
    for name, page_rank in scores.items():
        if name in anchors or page_rank <= 0:
            continue
        description = str((node_map.get(name) or {}).get("description") or "")
        relevance = max(_similarity(question, name), _similarity(question, description))
        ranked.append((0.7 * page_rank + 0.3 * relevance, name))
    ranked.sort(reverse=True)

    selected: List[str] = []
    while ranked and len(selected) < top_k:
        best_name = ""
        best_score = -math.inf
        for base_score, name in ranked:
            duplicate = max((_similarity(name, chosen) for chosen in selected), default=0.0)
            mmr = mmr_lambda * base_score - (1.0 - mmr_lambda) * duplicate
            if mmr > best_score:
                best_name, best_score = name, mmr
        selected.append(best_name)
        ranked = [(score, name) for score, name in ranked if name != best_name]

    items: List[Dict[str, Any]] = []
    for name in selected:
        supporting = []
        for neighbour, _, edge in adjacency.get(name, []):
            if neighbour in anchors or neighbour in selected:
                supporting.append(edge)
        candidate = any(_candidate_edge(edge) for edge in supporting)
        items.append({
            "entity": name,
            "description": str((node_map.get(name) or {}).get("description") or ""),
            "score": round(scores.get(name, 0.0), 6),
            "candidate": candidate,
            "relations": [
                {
                    "source": _edge_endpoints(edge)[0],
                    "target": _edge_endpoints(edge)[1],
                    "relation": str(edge.get("keywords") or edge.get("description") or "related"),
                    "candidate": _candidate_edge(edge),
                }
                for edge in supporting[:3]
            ],
        })
    return {"anchors": anchors, "items": items, "reason": "ok"}


def format_path_instruction(paths: Iterable[GraphPath], diagnostic: Dict[str, Any]) -> str:
    selected = list(paths)
    if not selected:
        anchors = "、".join(diagnostic.get("anchors") or [])
        return (
            "[严格路径检查]\n"
            f"未找到连接问题锚点（{anchors or '不足'}）的连续证据路径。"
            "不要用零散候选关系拼凑完整因果链；请明确说明证据缺口。"
        )
    lines = [
        "[经过证据约束和连续性检查的图谱路径]",
        "优先沿以下路径回答。标记为候选的边只能作为推测，并须与原文事实分开。",
    ]
    for index, path in enumerate(selected, 1):
        lines.append(f"路径{index}（score={path.score:.3f}）：" + " → ".join(path.nodes))
        for edge in path.edges:
            source, target = _edge_endpoints(edge)
            kind = "候选" if _candidate_edge(edge) else "原文"
            relation = str(edge.get("keywords") or edge.get("description") or "related")
            lines.append(f"- [{kind}] {source} --{relation}--> {target}")
    return "\n".join(lines)


def format_exploration_instruction(result: Dict[str, Any]) -> str:
    items = result.get("items") or []
    if not items:
        return ""
    lines = [
        "[开放探索候选]",
        "下面内容经过图扩散和多样性去重。回答时必须分成“原文支持”和“探索性推测”两部分。",
    ]
    for item in items:
        kind = "候选推测" if item.get("candidate") else "原文关联"
        description = str(item.get("description") or "")[:180]
        lines.append(f"- [{kind}] {item.get('entity')}: {description}")
    return "\n".join(lines)
