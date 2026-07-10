"""Path-grounded edge discovery for self-evolving knowledge graphs.

This module replaces the older entity-window edge discovery.  It asks the LLM
to infer new edges only from a bounded graph path bundle plus the original text
chunks that support the nodes/edges in those paths.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

GRAPH_FIELD_SEP = "<SEP>"

_log = logging.getLogger("docthinker.path_edge_discovery")


PATH_EDGE_DISCOVERY_PROMPT = """你是一个知识图谱自进化关系推理器。

下面给你的内容由两部分构成：

1. 源知识图谱子图：包含节点、已有边、边的描述、边的来源 chunk。
2. 原文证据：包含上述节点和边回查到的原始 chunk 文本。

你的任务是：基于已有子图链条和原文证据，判断是否可以写回新的直接关系边。

严格要求：
- 只能基于给定子图和原文 chunk 证据推导关系，不要使用外部常识。
- 不能因为两个节点共同出现就连边。
- 只能输出 source/target 都出现在子图中的关系。
- 新关系必须由至少两条已有边组成的路径支持。
- evidence_chain 必须逐条说明使用了哪条已有边、哪个 chunk_id、哪段逐字 quote。
- quote 必须原样出现在对应 chunk 文本中。
- confidence 必须 >= 0.80；不确定时输出空数组。
- 不要重复已有关系。

## 源知识图谱子图 JSON

{subgraph_json}

## 原文证据 JSON

{chunks_json}

## 输出格式

严格输出 JSON 数组，每个元素格式如下：
```json
[
  {{
    "source": "实体A",
    "target": "实体C",
    "relation": "grandparent_of",
    "keywords": "亲属推理",
    "description": "实体A是实体B的父亲，实体B是实体C的父亲，因此实体A是实体C的祖父。",
    "inference_type": "path_composition",
    "path_used": ["实体A", "实体B", "实体C"],
    "evidence_chain": [
      {{"edge": ["实体A", "实体B"], "chunk_id": "chunk-1", "quote": "实体A是实体B的父亲"}},
      {{"edge": ["实体B", "实体C"], "chunk_id": "chunk-2", "quote": "实体B是实体C的父亲"}}
    ],
    "evidence_chunk_ids": ["chunk-1", "chunk-2"],
    "confidence": 0.91
  }}
]
```

如果没有发现可由证据链支持的新关系，输出空数组 `[]`。
"""


PATH_EDGE_DISCOVERY_PROMPT = """你是知识图谱自进化关系推理器。

下面给你的内容由两部分构成：
1. 源知识图谱子图 JSON：包含节点、已有边、路径、节点/边描述、source_ids。
2. 原文 chunk 证据 JSON：包含这些节点和边回查到的原始 chunk 文本。

你的任务：沿着子图里的 paths，检查路径中的非相邻节点是否能形成新的直接关系边。
例如 A-B-C 可以检查 A-C；A-B-C-D 可以检查 A-C、B-D、A-D。

必须遵守：
- 只能基于给定子图和原文 chunk 证据推理，不要使用外部常识。
- 新边必须由至少两条已有边组成的 path 支持，不能只因为两个节点共现就连边。
- source 和 target 必须精确使用子图 nodes 里的 id。
- 不要输出已经存在于子图 edges 的直接边。
- evidence_chain 必须逐条说明使用了哪条已有边、哪个 chunk_id、哪段逐字 quote。
- quote 必须原样出现在对应 chunk 文本中。
- confidence 必须 >= 0.80；低于 0.80 的候选不要输出。
- 如果存在可由证据链支持的候选，请优先输出 1-5 条最强的新边；只有完全没有证据链时才输出 []。

## 源知识图谱子图 JSON

{subgraph_json}

## 原文 chunk 证据 JSON

{chunks_json}

## 输出格式

严格只输出 JSON 数组，不要 Markdown，不要解释。每个元素格式如下：
[
  {{
    "source": "实体A",
    "target": "实体C",
    "relation": "inferred_relation_type",
    "keywords": "关系类型简述",
    "description": "根据 A-B-C 路径和原文证据，说明为什么 A 与 C 可以形成直接关系。",
    "inference_type": "path_composition",
    "path_used": ["实体A", "实体B", "实体C"],
    "evidence_chain": [
      {{"edge": ["实体A", "实体B"], "chunk_id": "chunk-1", "quote": "来自 chunk-1 的逐字片段"}},
      {{"edge": ["实体B", "实体C"], "chunk_id": "chunk-2", "quote": "来自 chunk-2 的逐字片段"}}
    ],
    "evidence_chunk_ids": ["chunk-1", "chunk-2"],
    "confidence": 0.91
  }}
]

如果完全没有可由证据链支持的新边，输出 []。
"""


PATH_EDGE_DISCOVERY_PROMPT = """You are a knowledge-graph self-evolution edge discovery engine.

Input has two parts:
1. A source knowledge-graph subgraph JSON with nodes, existing edges, paths,
   descriptions, and source_ids.
2. Original chunk evidence JSON for the nodes and edges in that subgraph.
   `content` contains focused verbatim windows from the original chunk, not a
   prefix-only truncation. `full_content_chars` shows the original chunk length.

Task:
Follow the `paths` in the subgraph and check whether non-adjacent nodes can form
a new direct edge. For example, in A-B-C check A-C; in A-B-C-D check A-C, B-D,
and A-D.

Rules:
- Use only the provided subgraph and chunk evidence. Do not use outside knowledge.
- A new edge must be supported by a path with at least two existing edges.
- Do not create an edge only because two nodes co-occur.
- `source` and `target` must exactly match node ids from the subgraph.
- Do not output a direct edge already present in `edges`.
- `evidence_chain` must include evidence for every adjacent edge in `path_used`.
- Every `quote` must be a short verbatim substring from the referenced chunk.
- Prefer copying `quote` from `quote_candidates` when present.
- Do not quote node/edge descriptions. Do not summarize quotes. Do not use "...".
- Keep each quote under 80 characters when possible.
- Keep `description` under 120 Chinese characters.
- `confidence` must be >= 0.80.
- Return at most 2 strongest candidates. If there is only one strong candidate,
  return one object. If there is no evidence-supported candidate, return [].
- Output complete valid JSON only. Close every object and the array.

## Source Subgraph JSON

{subgraph_json}

## Original Chunk Evidence JSON

{chunks_json}

## Output JSON schema

[
  {{
    "source": "EntityA",
    "target": "EntityC",
    "relation": "inferred_relation_type",
    "keywords": "short relation type",
    "description": "Concise explanation grounded in the path and chunks.",
    "inference_type": "path_composition",
    "path_used": ["EntityA", "EntityB", "EntityC"],
    "evidence_chain": [
      {{"edge": ["EntityA", "EntityB"], "chunk_id": "chunk-1", "quote": "verbatim quote"}},
      {{"edge": ["EntityB", "EntityC"], "chunk_id": "chunk-2", "quote": "verbatim quote"}}
    ],
    "evidence_chunk_ids": ["chunk-1", "chunk-2"],
    "confidence": 0.91
  }}
]
"""


@dataclass
class PathDiscoveryConfig:
    max_depth: int = 10
    max_branch: int = 10
    max_roots: int = 6
    max_paths: int = 24
    max_chunks: int = 80
    min_confidence: float = 0.80
    max_prompt_chars: int = 120000
    max_chunk_chars: int = 1200
    evidence_window_chars: int = 320
    max_output_tokens: int = 8192
    max_parallel: int = 1
    artifact_dir: Optional[str] = None


@dataclass
class PathDiscoveredEdge:
    source: str
    target: str
    keywords: str
    description: str
    relation: str
    inference_type: str
    path_used: List[str]
    evidence_chain: List[Dict[str, Any]]
    evidence_chunk_ids: List[str]
    confidence: float


def _node_name(node: Dict[str, Any]) -> str:
    return str(
        node.get("id")
        or node.get("entity_id")
        or node.get("entity_name")
        or node.get("label")
        or ""
    ).strip()


def _edge_endpoints(edge: Dict[str, Any]) -> Tuple[str, str]:
    if "src_tgt" in edge and isinstance(edge.get("src_tgt"), (list, tuple)):
        pair = edge["src_tgt"]
        if len(pair) >= 2:
            return str(pair[0]).strip(), str(pair[1]).strip()
    return (
        str(edge.get("src_id") or edge.get("source") or "").strip(),
        str(edge.get("tgt_id") or edge.get("target") or "").strip(),
    )


def _split_source_ids(raw: Any) -> List[str]:
    ids: List[str] = []
    for item in str(raw or "").split(GRAPH_FIELD_SEP):
        item = item.strip()
        if item and item not in ids:
            ids.append(item)
    return ids


def _edge_key(source: str, target: str) -> Tuple[str, str]:
    return tuple(sorted((str(source), str(target))))


def _compact_node(node: Dict[str, Any]) -> Dict[str, Any]:
    name = _node_name(node)
    return {
        "id": name,
        "entity_type": node.get("entity_type", "unknown"),
        "description": str(node.get("description") or "")[:400],
        "source_ids": _split_source_ids(node.get("source_id")),
    }


def _compact_edge(edge: Dict[str, Any]) -> Dict[str, Any]:
    src, tgt = _edge_endpoints(edge)
    return {
        "source": src,
        "target": tgt,
        "keywords": edge.get("keywords", edge.get("relation", "related")),
        "description": str(edge.get("description") or "")[:500],
        "confidence": edge.get("confidence", edge.get("weight", "")),
        "source_ids": _split_source_ids(edge.get("source_id")),
        "provenance": edge.get("provenance", "source_document"),
    }


def _build_graph_indexes(
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Any]], Dict[str, List[str]]]:
    nodes_by_name = {
        name: node for node in nodes if (name := _node_name(node))
    }
    edges_by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    adjacency: Dict[str, List[str]] = {name: [] for name in nodes_by_name}

    for edge in edges:
        src, tgt = _edge_endpoints(edge)
        if not src or not tgt or src == tgt:
            continue
        if src not in nodes_by_name or tgt not in nodes_by_name:
            continue
        key = _edge_key(src, tgt)
        edges_by_pair.setdefault(key, edge)
        adjacency.setdefault(src, []).append(tgt)
        adjacency.setdefault(tgt, []).append(src)

    for name, neighbours in adjacency.items():
        neighbours.sort(key=lambda n: (len(adjacency.get(n, [])), n), reverse=True)

    return nodes_by_name, edges_by_pair, adjacency


def select_root_node(
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
) -> str:
    nodes_by_name, _, adjacency = _build_graph_indexes(nodes, edges)
    if not nodes_by_name:
        return ""
    return max(nodes_by_name, key=lambda name: (len(adjacency.get(name, [])), name))


def select_root_nodes(
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
    *,
    limit: int = 6,
) -> List[str]:
    nodes_by_name, _, adjacency = _build_graph_indexes(nodes, edges)
    ranked = sorted(
        nodes_by_name,
        key=lambda name: (len(adjacency.get(name, [])), name),
        reverse=True,
    )
    return ranked[: max(1, int(limit))]


def build_bounded_paths(
    root: str,
    nodes: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, Any]],
    config: Optional[PathDiscoveryConfig] = None,
) -> List[List[str]]:
    cfg = config or PathDiscoveryConfig()
    nodes_by_name, _, adjacency = _build_graph_indexes(nodes, edges)
    if root not in nodes_by_name:
        return []

    max_depth = max(2, int(cfg.max_depth))
    max_branch = max(1, int(cfg.max_branch))
    max_paths = max(1, int(cfg.max_paths))
    paths: List[List[str]] = []

    def dfs(path: List[str]) -> None:
        if len(paths) >= max_paths:
            return
        if len(path) >= 3:
            paths.append(path[:])
        if len(path) >= max_depth + 1:
            return
        current = path[-1]
        for neighbour in adjacency.get(current, [])[:max_branch]:
            if neighbour in path:
                continue
            dfs([*path, neighbour])
            if len(paths) >= max_paths:
                return

    dfs([root])
    return paths[:max_paths]


def _path_edges(path: Sequence[str], edges_by_pair: Dict[Tuple[str, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for src, tgt in zip(path, path[1:]):
        edge = edges_by_pair.get(_edge_key(src, tgt))
        if edge:
            out.append(edge)
    return out


def build_path_subgraph(
    paths: Sequence[Sequence[str]],
    nodes_by_name: Dict[str, Dict[str, Any]],
    edges_by_pair: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    node_names: List[str] = []
    edge_pairs: List[Tuple[str, str]] = []
    path_payloads: List[Dict[str, Any]] = []

    for idx, path in enumerate(paths, start=1):
        clean_path = [name for name in path if name in nodes_by_name]
        if len(clean_path) < 3:
            continue
        for name in clean_path:
            if name not in node_names:
                node_names.append(name)
        path_edge_payloads = []
        for src, tgt in zip(clean_path, clean_path[1:]):
            key = _edge_key(src, tgt)
            if key not in edges_by_pair:
                continue
            if key not in edge_pairs:
                edge_pairs.append(key)
            path_edge_payloads.append(_compact_edge(edges_by_pair[key]))
        if path_edge_payloads:
            path_payloads.append(
                {
                    "path_id": f"path_{idx:03d}",
                    "nodes": clean_path,
                    "edges": path_edge_payloads,
                }
            )

    return {
        "nodes": [_compact_node(nodes_by_name[name]) for name in node_names],
        "edges": [_compact_edge(edges_by_pair[pair]) for pair in edge_pairs],
        "paths": path_payloads,
    }


def collect_path_chunk_ids(
    paths: Sequence[Sequence[str]],
    nodes_by_name: Dict[str, Dict[str, Any]],
    edges_by_pair: Dict[Tuple[str, str], Dict[str, Any]],
    *,
    max_chunks: int,
) -> List[str]:
    chunk_ids: List[str] = []

    def add_many(ids: Iterable[str]) -> None:
        for chunk_id in ids:
            if chunk_id and chunk_id not in chunk_ids:
                chunk_ids.append(chunk_id)

    for path in paths:
        for name in path:
            add_many(_split_source_ids((nodes_by_name.get(name) or {}).get("source_id")))
        for edge in _path_edges(path, edges_by_pair):
            add_many(_split_source_ids(edge.get("source_id")))
        if len(chunk_ids) >= max_chunks:
            break
    return chunk_ids[:max_chunks]


def _collect_focus_terms(paths: Sequence[Sequence[str]]) -> List[str]:
    terms: List[str] = []
    for path in paths:
        for name in path:
            clean = str(name or "").strip()
            if clean and clean not in terms:
                terms.append(clean)
    terms.sort(key=len, reverse=True)
    return terms


def _merge_ranges(ranges: List[Tuple[int, int]], *, max_gap: int = 80) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    ordered = sorted((max(0, start), max(0, end)) for start, end in ranges if end > start)
    merged: List[Tuple[int, int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1] + max_gap:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _extract_focus_excerpt(
    content: str,
    focus_terms: Sequence[str],
    *,
    max_chars: int,
    window_chars: int,
) -> Tuple[str, List[Dict[str, Any]]]:
    text = str(content or "")
    if len(text) <= max_chars:
        return text, [{"start": 0, "end": len(text), "matched_terms": []}]

    ranges: List[Tuple[int, int]] = []
    term_hits: Dict[Tuple[int, int], List[str]] = {}
    half_window = max(80, int(window_chars) // 2)
    for term in focus_terms:
        if not term:
            continue
        start = 0
        while True:
            idx = text.find(term, start)
            if idx < 0:
                break
            left = max(0, idx - half_window)
            right = min(len(text), idx + len(term) + half_window)
            pair = (left, right)
            ranges.append(pair)
            term_hits.setdefault(pair, []).append(term)
            start = idx + max(1, len(term))

    if not ranges:
        return text[:max_chars], [{"start": 0, "end": min(len(text), max_chars), "matched_terms": []}]

    merged = _merge_ranges(ranges)
    scored: List[Tuple[int, int, int, List[str]]] = []
    for start, end in merged:
        matched = [term for term in focus_terms if term and term in text[start:end]]
        score = len(matched) * 1000 - (end - start)
        scored.append((score, start, end, matched))
    scored.sort(key=lambda item: item[0], reverse=True)

    selected: List[Tuple[int, int, List[str]]] = []
    used = 0
    separator_cost = len("\n...\n")
    for _, start, end, matched in scored:
        piece_len = end - start
        extra = piece_len + (separator_cost if selected else 0)
        if selected and used + extra > max_chars:
            continue
        if not selected and piece_len > max_chars:
            end = min(len(text), start + max_chars)
            piece_len = end - start
            extra = piece_len
        if used + extra <= max_chars:
            selected.append((start, end, matched))
            used += extra
        if used >= max_chars:
            break

    selected.sort(key=lambda item: item[0])
    excerpts = [
        {"start": start, "end": end, "matched_terms": matched}
        for start, end, matched in selected
    ]
    excerpt_text = "\n...\n".join(text[start:end] for start, end, _ in selected)
    return excerpt_text, excerpts


def _quote_candidates_from_excerpt(
    excerpt: str,
    focus_terms: Sequence[str],
    *,
    limit: int = 8,
    max_chars: int = 90,
) -> List[str]:
    text = str(excerpt or "")
    if not text:
        return []

    separators = "。！？；\n"
    pieces: List[str] = []
    start = 0
    for idx, ch in enumerate(text):
        if ch in separators:
            piece = text[start : idx + 1].strip()
            if piece:
                pieces.append(piece)
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)

    scored: List[Tuple[int, int, str]] = []
    for piece in pieces:
        clean = " ".join(piece.split())
        if not clean or clean == "...":
            continue
        matches = sum(1 for term in focus_terms if term and term in clean)
        if matches <= 0:
            continue
        if len(clean) > max_chars:
            best = ""
            for term in focus_terms:
                idx = clean.find(term)
                if idx >= 0:
                    left = max(0, idx - max_chars // 2)
                    right = min(len(clean), left + max_chars)
                    best = clean[left:right]
                    break
            clean = best or clean[:max_chars]
        scored.append((matches, -len(clean), clean))

    scored.sort(reverse=True)
    out: List[str] = []
    for _, _, candidate in scored:
        if candidate and candidate not in out:
            out.append(candidate)
        if len(out) >= limit:
            break
    return out


async def load_chunk_evidence(
    text_chunks: Any,
    chunk_ids: Sequence[str],
    *,
    max_chunk_chars: int = 1200,
    focus_terms: Optional[Sequence[str]] = None,
    evidence_window_chars: int = 320,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    if not text_chunks or not chunk_ids:
        return [], {}
    chunk_data_list = await text_chunks.get_by_ids(list(chunk_ids))
    chunks: List[Dict[str, str]] = []
    full_by_id: Dict[str, str] = {}
    for chunk_id, data in zip(chunk_ids, chunk_data_list):
        if not isinstance(data, dict):
            continue
        content = str(data.get("content") or "")
        if not content:
            continue
        full_by_id[chunk_id] = content
        excerpt, excerpts = _extract_focus_excerpt(
            content,
            list(focus_terms or []),
            max_chars=max(200, int(max_chunk_chars)),
            window_chars=max(120, int(evidence_window_chars)),
        )
        chunks.append(
            {
                "chunk_id": chunk_id,
                "file_path": str(data.get("file_path") or "unknown_source"),
                "content": excerpt,
                "content_strategy": "focus_term_windows",
                "full_content_chars": str(len(content)),
                "excerpts": excerpts,
                "quote_candidates": _quote_candidates_from_excerpt(
                    excerpt,
                    list(focus_terms or []),
                ),
            }
        )
    return chunks, full_by_id


def _safe_artifact_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value or "root"))
    return cleaned[:80] or "root"


def _write_artifacts(
    artifact_dir: Optional[str],
    *,
    root: str,
    subgraph: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    prompt: str,
    raw: str,
    accepted: List["PathDiscoveredEdge"],
    rejected: Dict[str, int],
) -> None:
    if not artifact_dir:
        return
    try:
        run_dir = Path(artifact_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base = run_dir / f"{stamp}_{_safe_artifact_name(root)}"
        base.mkdir(parents=True, exist_ok=True)
        (base / "subgraph.json").write_text(
            json.dumps(subgraph, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (base / "chunks.json").write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (base / "prompt.txt").write_text(prompt, encoding="utf-8")
        (base / "raw_response.txt").write_text(str(raw or ""), encoding="utf-8")
        (base / "result.json").write_text(
            json.dumps(
                {
                    "root": root,
                    "accepted": [edge.__dict__ for edge in accepted],
                    "rejected": rejected,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        _log.warning("[path_edge_discovery] failed to write artifacts: %s", exc)


def _fit_prompt_sections(
    subgraph: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    config: PathDiscoveryConfig,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Keep prompt structure intact while fitting large evidence bundles."""
    chunk_limit = max(300, int(config.max_chunk_chars))
    working_chunks = [dict(item) for item in chunks]
    while True:
        for item in working_chunks:
            item["content"] = str(item.get("content") or "")[:chunk_limit]
        subgraph_json = json.dumps(subgraph, ensure_ascii=False, indent=2)
        chunks_json = json.dumps(working_chunks, ensure_ascii=False, indent=2)
        prompt = PATH_EDGE_DISCOVERY_PROMPT.format(
            subgraph_json=subgraph_json,
            chunks_json=chunks_json,
        )
        if len(prompt) <= int(config.max_prompt_chars) or chunk_limit <= 300:
            return subgraph_json, chunks_json, working_chunks
        chunk_limit = max(300, chunk_limit // 2)


def _strip_json_fence(raw: str) -> str:
    text = str(raw or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _escape_control_chars_in_strings(text: str) -> str:
    out: List[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            out.append(ch)
            in_string = not in_string
            continue
        if in_string and ch == "\n":
            out.append("\\n")
            continue
        if in_string and ch == "\r":
            out.append("\\r")
            continue
        if in_string and ch == "\t":
            out.append("\\t")
            continue
        if in_string and ord(ch) < 32:
            out.append("\\u%04x" % ord(ch))
            continue
        out.append(ch)
    return "".join(out)


def _loads_jsonish(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_escape_control_chars_in_strings(text))


def _extract_complete_objects_from_array(text: str) -> List[Any]:
    start = text.find("[")
    if start == -1:
        return []

    objects: List[Any] = []
    square_depth = 0
    curly_depth = 0
    object_start: Optional[int] = None
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == "[":
            square_depth += 1
            continue
        if ch == "]":
            square_depth = max(0, square_depth - 1)
            continue
        if ch == "{":
            if square_depth == 1 and curly_depth == 0:
                object_start = idx
            if object_start is not None:
                curly_depth += 1
            continue
        if ch == "}" and object_start is not None:
            curly_depth -= 1
            if curly_depth == 0:
                candidate = text[object_start : idx + 1]
                try:
                    parsed = _loads_jsonish(candidate)
                except json.JSONDecodeError:
                    _log.warning(
                        "[path_edge_discovery] failed to parse complete JSON object near %r",
                        candidate[:200],
                    )
                else:
                    objects.append(parsed)
                object_start = None

    return objects


def _extract_json_array(raw: str) -> List[Any]:
    text = _strip_json_fence(raw)
    if not text:
        return []

    try:
        parsed = _loads_jsonish(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = _loads_jsonish(text[start : end + 1])
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    objects = _extract_complete_objects_from_array(text)
    if objects:
        _log.warning(
            "[path_edge_discovery] recovered %d complete JSON object(s) from an incomplete array",
            len(objects),
        )
        return objects

    if "{" in text:
        _log.warning(
            "[path_edge_discovery] failed to parse LLM JSON array; raw prefix=%r",
            text[:500],
        )
    return []


def _normalise_evidence_chain(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        edge = item.get("edge")
        if not isinstance(edge, list) or len(edge) < 2:
            continue
        chunk_id = str(item.get("chunk_id") or "").strip()
        quote = " ".join(str(item.get("quote") or "").split())
        if not chunk_id or not quote:
            continue
        out.append(
            {
                "edge": [str(edge[0]).strip(), str(edge[1]).strip()],
                "chunk_id": chunk_id,
                "quote": quote,
            }
        )
    return out


def validate_path_edge(
    item: Dict[str, Any],
    *,
    valid_names: set[str],
    existing_pairs: set[Tuple[str, str]],
    edges_by_pair: Dict[Tuple[str, str], Dict[str, Any]],
    chunk_text_by_id: Dict[str, str],
    min_confidence: float = 0.80,
) -> Tuple[Optional[PathDiscoveredEdge], str]:
    src = str(item.get("source") or "").strip()
    tgt = str(item.get("target") or "").strip()
    if not src or not tgt or src == tgt:
        return None, "invalid_endpoints"
    if src not in valid_names or tgt not in valid_names:
        return None, "unknown_endpoint"
    if _edge_key(src, tgt) in existing_pairs:
        return None, "edge_exists"

    try:
        confidence = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < float(min_confidence):
        return None, "low_confidence"

    path_used = [str(name).strip() for name in (item.get("path_used") or []) if str(name).strip()]
    if len(path_used) < 3:
        return None, "short_path"
    if path_used[0] != src or path_used[-1] != tgt:
        # Allow reverse path, but keep the model honest about the path endpoints.
        if not (path_used[0] == tgt and path_used[-1] == src):
            return None, "path_endpoint_mismatch"
    for name in path_used:
        if name not in valid_names:
            return None, "path_unknown_node"
    for left, right in zip(path_used, path_used[1:]):
        if _edge_key(left, right) not in edges_by_pair:
            return None, "path_not_connected"

    evidence_chain = _normalise_evidence_chain(item.get("evidence_chain"))
    if len(evidence_chain) < 2:
        return None, "insufficient_evidence_chain"

    evidence_chunk_ids: List[str] = []
    covered_path_pairs: set[Tuple[str, str]] = set()
    for ev in evidence_chain:
        ev_src, ev_tgt = ev["edge"][0], ev["edge"][1]
        if _edge_key(ev_src, ev_tgt) not in edges_by_pair:
            return None, "evidence_edge_not_in_graph"
        covered_path_pairs.add(_edge_key(ev_src, ev_tgt))
        chunk_id = ev["chunk_id"]
        quote = ev["quote"]
        chunk_text = " ".join(str(chunk_text_by_id.get(chunk_id) or "").split())
        if not chunk_text or quote not in chunk_text:
            return None, "quote_not_grounded"
        named_endpoints = [name for name in (ev_src, ev_tgt) if len(str(name)) >= 2]
        if named_endpoints and not any(name in quote for name in named_endpoints):
            return None, "quote_not_about_evidence_edge"
        if chunk_id not in evidence_chunk_ids:
            evidence_chunk_ids.append(chunk_id)

    required_path_pairs = {_edge_key(left, right) for left, right in zip(path_used, path_used[1:])}
    if not required_path_pairs.issubset(covered_path_pairs):
        return None, "missing_path_evidence"

    provided_chunk_ids = [
        str(chunk_id).strip()
        for chunk_id in (item.get("evidence_chunk_ids") or [])
        if str(chunk_id).strip()
    ]
    if provided_chunk_ids:
        for chunk_id in provided_chunk_ids:
            if chunk_id not in evidence_chunk_ids and chunk_id in chunk_text_by_id:
                evidence_chunk_ids.append(chunk_id)

    relation = str(item.get("relation") or item.get("keywords") or "related").strip()
    keywords = str(item.get("keywords") or relation or "related").strip()
    description = str(item.get("description") or "").strip()
    if not description:
        return None, "missing_description"

    edge = PathDiscoveredEdge(
        source=src,
        target=tgt,
        keywords=keywords,
        description=description,
        relation=relation,
        inference_type=str(item.get("inference_type") or "path_composition").strip(),
        path_used=path_used,
        evidence_chain=evidence_chain,
        evidence_chunk_ids=evidence_chunk_ids,
        confidence=min(1.0, confidence),
    )
    return edge, "accepted"


def parse_path_discovered_edges(
    raw: str,
    *,
    valid_names: set[str],
    existing_pairs: set[Tuple[str, str]],
    edges_by_pair: Dict[Tuple[str, str], Dict[str, Any]],
    chunk_text_by_id: Dict[str, str],
    min_confidence: float = 0.80,
) -> Tuple[List[PathDiscoveredEdge], Dict[str, int]]:
    accepted: List[PathDiscoveredEdge] = []
    rejected: Dict[str, int] = {}
    seen: set[Tuple[str, str]] = set()

    items = _extract_json_array(raw)
    if not items and "{" in str(raw or ""):
        rejected["json_parse_failed"] = 1

    for item in items:
        if not isinstance(item, dict):
            rejected["invalid_item"] = rejected.get("invalid_item", 0) + 1
            continue
        edge, reason = validate_path_edge(
            item,
            valid_names=valid_names,
            existing_pairs=existing_pairs | seen,
            edges_by_pair=edges_by_pair,
            chunk_text_by_id=chunk_text_by_id,
            min_confidence=min_confidence,
        )
        if edge is None:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        seen.add(_edge_key(edge.source, edge.target))
        accepted.append(edge)

    return accepted, rejected


async def discover_path_edges(
    *,
    graph: Any,
    text_chunks: Any,
    llm_func: Callable,
    config: Optional[PathDiscoveryConfig] = None,
) -> List[PathDiscoveredEdge]:
    cfg = config or PathDiscoveryConfig()
    nodes = await graph.get_all_nodes()
    edges = await graph.get_all_edges()
    if len(nodes) < 3 or len(edges) < 2:
        return []

    nodes_by_name, edges_by_pair, _ = _build_graph_indexes(nodes, edges)
    roots = select_root_nodes(nodes, edges, limit=cfg.max_roots)
    if not roots:
        _log.info("[path_edge_discovery] no root nodes")
        return []

    existing_pairs = set(edges_by_pair)
    all_discovered: List[PathDiscoveredEdge] = []
    aggregate_rejected: Dict[str, int] = {}

    for root in roots:
        paths = build_bounded_paths(root, nodes, edges, cfg)
        if not paths:
            _log.info("[path_edge_discovery] no paths from root=%s", root)
            continue

        subgraph = build_path_subgraph(paths, nodes_by_name, edges_by_pair)
        chunk_ids = collect_path_chunk_ids(
            paths,
            nodes_by_name,
            edges_by_pair,
            max_chunks=max(1, int(cfg.max_chunks)),
        )
        chunks, chunk_text_by_id = await load_chunk_evidence(
            text_chunks,
            chunk_ids,
            max_chunk_chars=max(200, int(cfg.max_chunk_chars)),
            focus_terms=_collect_focus_terms(paths),
            evidence_window_chars=max(120, int(cfg.evidence_window_chars)),
        )
        if not chunks:
            _log.info("[path_edge_discovery] no chunk evidence for root=%s", root)
            continue

        subgraph_json, chunks_json, fitted_chunks = _fit_prompt_sections(subgraph, chunks, cfg)
        prompt = PATH_EDGE_DISCOVERY_PROMPT.format(
            subgraph_json=subgraph_json,
            chunks_json=chunks_json,
        )

        t0 = time.time()
        raw = await llm_func(prompt, max_tokens=max(2048, int(cfg.max_output_tokens)))
        discovered, rejected = parse_path_discovered_edges(
            raw,
            valid_names=set(nodes_by_name),
            existing_pairs=existing_pairs | {
                _edge_key(edge.source, edge.target) for edge in all_discovered
            },
            edges_by_pair=edges_by_pair,
            chunk_text_by_id=chunk_text_by_id,
            min_confidence=cfg.min_confidence,
        )
        for reason, count in rejected.items():
            aggregate_rejected[reason] = aggregate_rejected.get(reason, 0) + count

        _write_artifacts(
            cfg.artifact_dir,
            root=root,
            subgraph=subgraph,
            chunks=fitted_chunks,
            prompt=prompt,
            raw=raw,
            accepted=discovered,
            rejected=rejected,
        )

        all_discovered.extend(discovered)
        _log.info(
            "[path_edge_discovery] root=%s paths=%d chunks=%d prompt_chars=%d raw_chars=%d accepted=%d rejected=%s in %.1fs",
            root,
            len(paths),
            len(fitted_chunks),
            len(prompt),
            len(str(raw or "")),
            len(discovered),
            rejected,
            time.time() - t0,
        )

    _log.info(
        "[path_edge_discovery] roots=%d accepted_total=%d rejected_total=%s",
        len(roots),
        len(all_discovered),
        aggregate_rejected,
    )
    return all_discovered


__all__ = [
    "GRAPH_FIELD_SEP",
    "PATH_EDGE_DISCOVERY_PROMPT",
    "PathDiscoveryConfig",
    "PathDiscoveredEdge",
    "select_root_node",
    "select_root_nodes",
    "build_bounded_paths",
    "build_path_subgraph",
    "collect_path_chunk_ids",
    "load_chunk_evidence",
    "validate_path_edge",
    "parse_path_discovered_edges",
    "discover_path_edges",
]
