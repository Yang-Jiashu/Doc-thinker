"""Bounded literal-entity graph expansion, not graph-generated evidence.

The sparse graph is built from the existing conservative source-claim parser.
Exact entity endpoints join source statements; a shared relation word never
does. Selected source statements form starting components. At most two new
edges may bridge those components to another question anchor. For explicit
idea requests a bounded related-source route is also allowed. Edges retain
polarity and conditions and are association/search hints, never implications.

No provider, database, generated memory, or global tenant cache is used here.
The caller supplies the current authorized project view on every call. The
single immutable cache contains source pointers, not quotations; additions are
rehydrated from that current view. Expansion is append-only: a full evidence
budget is disclosed, not made to fit by dropping a literal answer or premise.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from types import MappingProxyType

from .research_reasoning import extract_document_claims
from .research_sentence_windows import evidence_units
from .research_hypotheses import detect_hypothesis_intent

MAX_INPUT_CHUNKS = 8192
MAX_GRAPH_CHUNKS = 2048
MAX_GRAPH_CHARS = 2_000_000
MAX_GRAPH_EDGES = 6144
MAX_DEGREE = 24
MAX_START_NODES = 16
MAX_PATH_STATES = 1024
MAX_PATHS = 32
MAX_ADDED_SOURCES = 3
MAX_SOURCE_CHARS = 1000
MAX_QUERY_CHARS = 8192

_LITERAL_ONLY = re.compile(
    r"逐字|翻译|润色|校对|只(?:要|需)?(?:引用|摘录|原文)|不要(?:分析|推理|发挥)|"
    r"\b(?:verbatim|translate|translation|proofread|rewrite)\b|"
    r"\b(?:quote|extract)\s+only\b|\b(?:no|without)\s+(?:analysis|inference)\b", re.I)
_NO_GRAPH = re.compile(
    r"(?:不要|不用|无需|不做|不需要|别|请勿|禁止).{0,5}(?:多跳|扩展|补边|关联|推理|假设|类比|构思|idea)|"
    r"\b(?:do not|don['’]t|no|without|avoid|skip)\s+(?:\w+\s+){0,3}"
    r"(?:graph|multi.hop|expansion|hypotheses|hypothesis|ideas|reasoning)\b", re.I)
_MULTIHOP = re.compile(
    r"多跳|因果链|推理链|证据链|机制链|完整.{0,5}(?:链路|链条)|"
    r"\bmulti[- ]?hop\b|\b(?:causal|reasoning|evidence|mechanism) (?:chain|path)\b|"
    r"\bconnect(?:ing)? (?:the )?dots\b", re.I)
_GENERATED = frozenset({"llm", "self_study", "self_evolution", "llm_expansion", "hypothesis", "inferred", "memory"})
_GENERIC_ENTITIES = frozenset({"it", "this", "that", "they", "model", "models", "method", "methods", "result", "results", "system", "systems", "data", "research", "study", "studies", "thing", "模型", "方法", "结果", "系统", "数据", "研究", "效果", "性能"})
_ADJACENT_QUALIFICATION = re.compile(
    r"\b(?:no|not|never|cannot|without|only|except|unless|when|under|if|conditions?|"
    r"assumptions?|limited|limitations?|restricted|restricts?)\b|\b\w+n['’]t\b|"
    r"不能|并非|未能|没有|无证据|尚未|不支持|仅|只有|条件|前提|如果|除非|限于|限制", re.I)
# Match the existing ASCII-literal re.I behavior without compiling a pattern
# for every entity on every query. Python's four extra ASCII-casefold matches
# are retained; full Unicode casefold (e.g. ß -> ss) would broaden them unsafely.
_ASCII_CASE_TRANSLATION = str.maketrans({
    **{chr(value): chr(value + 32) for value in range(ord("A"), ord("Z") + 1)},
    "İ": "i", "ı": "i", "ſ": "s", "K": "k",
})
_STRUCTURED_LINE = re.compile(r"(?m)^\s*(?:case|案例|情境|role|角色|类型)\s*[:：]", re.I)


def _text(value):
    if not isinstance(value, str):
        return ""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return ""
    return value


def _limit(value, default, maximum):
    return min(maximum, max(0, value)) if type(value) is int else default


def graph_retrieval_intent(question: str) -> bool:
    """Ordinary chat, literal-only and explicitly declined expansion stay off."""
    text = _text(question)[:MAX_QUERY_CHARS]
    return bool(text and not (_LITERAL_ONLY.search(text) or _NO_GRAPH.search(text))
                and (detect_hypothesis_intent(text) or _MULTIHOP.search(text)))


def _gap(code):
    return {"code": code, "message": {
        "graph_association_not_proof": "图中连接只帮助寻找相关原文，不证明因果、蕴含或新假设成立。",
        "graph_window_limited": "部分资料超过本次图索引窗口；没有找到路径不代表不存在。",
        "graph_hub_omitted": "高连接度通用节点已停止扩展，避免无关背景膨胀。",
        "graph_search_limited": "图搜索达到有界窗口，本次未穷尽全部关系。",
        "graph_no_source_seed": "没有可核对的原文起点，未任意扩展其他材料。",
        "graph_no_connected_path": "本次可解析的有界图中没有找到所需连接；不能据此补全推理链。",
        "graph_capacity_reached": "关联原文无法完整放入当前证据预算，已保留原有证据，未截断或挤掉它们。",
        "graph_source_unavailable": "部分原文已变更、撤回或不属于当前资料窗口，未使用旧图指针。",
        "graph_counterevidence_in_path": "连接中包含否定陈述，应核对反例，不能把连接当成正向因果链。",
        "graph_path_conditions_differ": "连接原文的条件未完全对齐，应分别核对，不能默认跨条件迁移成立。",
        "graph_parser_is_partial": "图索引仅覆盖保守解析出的明确关系，未解析段落仍由正常检索处理。",
        "graph_excerpt_windowed": "关联依据只载入覆盖路径陈述及相邻显式限定的连续原文窗口，窗口之外仍可能有其他条件。",
        "graph_path_window_incomplete": "部分连接所需陈述不在已保留的原文窗口中，未把未展示的陈述当成证据。",
    }[code]}


@dataclass(frozen=True)
class _Edge:
    source_id: str
    document_id: str
    left: tuple[str, str]
    right: tuple[str, str]
    relation: str
    polarity: int
    conditions: tuple[str, ...]
    kind: str
    excerpt_start: int
    excerpt_end: int


@dataclass(frozen=True)
class _Graph:
    key: str
    edges: tuple[_Edge, ...]
    adjacency: Mapping
    source_edges: Mapping
    hubs: frozenset
    limited: bool
    indexed_chunks: int


def _live_chunks(chunks):
    """Fail closed for ambiguous IDs and generated/revoked source records."""
    rows, conflicts = {}, set()
    for raw in chunks:
        if not isinstance(raw, Mapping):
            continue
        identifier, document = _text(raw.get("id")), _text(raw.get("document_id"))
        content = _text(raw.get("content"))
        if not identifier or not document or len(identifier) > 256 or len(document) > 256 or not content.strip():
            continue
        if (raw.get("is_generated") in (True, "true", "1") or raw.get("is_discovered") in (True, "true", "1")
                or _text(raw.get("provenance")) in _GENERATED
                or _text(raw.get("state")) in {"deleted", "failed", "invalidated"}
                or raw.get("truncated") is True or raw.get("excerpt_start", 0)):
            continue
        if identifier in rows and rows[identifier] != raw:
            conflicts.add(identifier)
        else:
            rows[identifier] = raw
    return {identifier: row for identifier, row in rows.items() if identifier not in conflicts}


def _canonical_window(live):
    documents = defaultdict(list)
    for row in live.values():
        documents[row["document_id"]].append(row)
    selected, remaining = [], MAX_GRAPH_CHARS
    limited = False
    for document in sorted(documents):
        rows = documents[document]
        size = sum(len(row["content"]) for row in rows)
        # Omit an entire document rather than invent a complete graph from a
        # clipped case. Lexical retrieval remains available for omitted text.
        if (len(rows) > 600 or any(len(row["content"]) > 20_000 for row in rows)
                or len(selected) + len(rows) > MAX_GRAPH_CHUNKS or size > remaining):
            limited = True
            continue
        rows.sort(key=lambda row: (row.get("ordinal", 0) if type(row.get("ordinal", 0)) is int else 0, row["id"]))
        selected.extend(rows)
        remaining -= size
    digest = hashlib.sha256()
    # Membership is authoritative even when a source was outside the window.
    digest.update(json.dumps(sorted(live), ensure_ascii=False).encode())
    for row in selected:
        digest.update(json.dumps([row["id"], row["document_id"], row.get("ordinal"), row["content"]],
                                 ensure_ascii=False, separators=(",", ":")).encode())
    return selected, digest.hexdigest(), limited


def _entity_allowed(value):
    return bool(value and value not in _GENERIC_ENTITIES and len(value) <= 160)


def _parser_rows(rows):
    """Normalize PDF soft wraps without changing a single source offset.

    Only interior CR/LF characters of complete evidence units become spaces.
    Unit boundaries/heading separators and explicit Case/Role label blocks
    remain untouched. No word, punctuation, negation, or condition is removed;
    actual quotations always come from the canonical, unnormalized chunk.
    """
    prepared = []
    for row in rows:
        text = row["content"]
        if "\n" not in text and "\r" not in text:
            prepared.append(row)
            continue
        characters = list(text)
        for left, right in evidence_units(text):
            unit = text[left:right]
            if _STRUCTURED_LINE.search(unit):
                continue
            start = left + len(unit) - len(unit.lstrip())
            end = right - (len(unit) - len(unit.rstrip()))
            for index in range(start, end):
                if characters[index] in "\r\n":
                    characters[index] = " "
        prepared.append({**row, "content": "".join(characters)})
    return prepared


def _build(rows, key, limited):
    documents = defaultdict(list)
    for row in rows:
        documents[row["document_id"]].append(row)
    edges, seen = [], set()
    for document in sorted(documents):
        parsed = extract_document_claims(_parser_rows(documents[document]), max_claims=1536)
        limited |= parsed["limited"]
        pending = []
        for claim in parsed["claims"]:
            atoms = [claim] if claim["kind"] == "observation" else [*claim["premises"], claim["consequence"]]
            for atom in atoms:
                if not (_entity_allowed(atom["subject"]) and _entity_allowed(atom["object"])):
                    continue
                edge = _Edge(claim["source_ids"][0], document,
                             (claim["scope"], atom["subject"]), (claim["scope"], atom["object"]),
                             atom["relation"], atom["polarity"], tuple(atom["conditions"]), claim["kind"],
                             claim["excerpt_start"], claim["excerpt_end"])
                if edge.left != edge.right and edge not in seen:
                    pending.append(edge)
                    seen.add(edge)
        if len(edges) + len(pending) > MAX_GRAPH_EDGES:
            limited = True
            continue
        edges.extend(pending)
    edges.sort(key=lambda edge: (edge.document_id, edge.source_id, edge.excerpt_start,
                                 edge.left, edge.right, edge.relation, edge.polarity))
    adjacency, source_edges = defaultdict(list), defaultdict(list)
    for index, edge in enumerate(edges):
        adjacency[edge.left].append(index)
        adjacency[edge.right].append(index)
        source_edges[edge.source_id].append(index)
    hubs = frozenset(node for node, indexes in adjacency.items() if len(indexes) > MAX_DEGREE)
    return _Graph(key, tuple(edges),
                  MappingProxyType({node: tuple(indexes) if node not in hubs else () for node, indexes in adjacency.items()}),
                  MappingProxyType({source: tuple(indexes) for source, indexes in source_edges.items()}),
                  hubs, limited, len(rows))


def _question_mentions(question, entity, folded=None, unicode_folded=None):
    if entity.isascii():
        # A/B/C in experimental questions are identifiers, not English articles.
        if len(entity) == 1 and entity.isalpha():
            value, needle = question, entity.upper()
        else:
            value = folded if folded is not None else question.translate(_ASCII_CASE_TRANSLATION)
            needle = entity.lower()
        offset = value.find(needle)
        while offset >= 0:
            end = offset + len(needle)
            if ((not offset or not (value[offset - 1].isalnum() or value[offset - 1] == "_"))
                    and (end == len(value) or not (value[end].isalnum() or value[end] == "_"))):
                return True
            offset = value.find(needle, offset + 1)
        return False
    return entity in (unicode_folded if unicode_folded is not None else question.casefold())


def _seed_span(source, canonical):
    start = source.get("excerpt_start", 0)
    content = _text(source.get("content"))
    end = source.get("excerpt_end", start + len(content)) if type(start) is int else -1
    if (type(start) is not int or type(end) is not int or start < 0 or end <= start
            or end - start != len(content) or end > len(canonical["content"])
            or source.get("document_id") != canonical["document_id"]
            or canonical["content"][start:end] != content):
        return None
    return start, end


def _candidate_window(canonical, required_edges):
    """Hydrate a single untouched source span containing every route claim.

    ``evidence_units`` binds visible qualifications and PDF sentence structure.
    Immediately adjacent explicit negative/conditional units are conservatively
    retained as well, including e.g. ``No trials ...`` which need not start with
    a discourse connector. If this union exceeds the window, omit it whole.
    This is a local lexical boundary check, not semantic completeness proof.
    """
    content = canonical["content"]
    if (not required_edges or any(edge.source_id != canonical["id"]
            or edge.document_id != canonical["document_id"]
            or not 0 <= edge.excerpt_start < edge.excerpt_end <= len(content)
            for edge in required_edges)):
        return None
    start, end = 0, len(content)
    if len(content) > MAX_SOURCE_CHARS:
        claim_start = min(edge.excerpt_start for edge in required_edges)
        claim_end = max(edge.excerpt_end for edge in required_edges)
        units = evidence_units(content)
        overlapping = [index for index, (left, right) in enumerate(units)
                       if left < claim_end and claim_start < right]
        if not overlapping:
            return None
        first, last = overlapping[0], overlapping[-1]
        # The evidence units themselves already include their bound preceding
        # assumptions/following caveats. Add adjacent negation/condition units
        # to avoid removing e.g. a standalone null-result sentence.
        while first and _ADJACENT_QUALIFICATION.search(content[slice(*units[first - 1])]):
            first -= 1
        while last + 1 < len(units) and _ADJACENT_QUALIFICATION.search(content[slice(*units[last + 1])]):
            last += 1
        start, end = units[first][0], units[last][1]
        if end - start > MAX_SOURCE_CHARS:
            return None
    if any(not start <= edge.excerpt_start < edge.excerpt_end <= end for edge in required_edges):
        return None
    return {"content": content[start:end], "excerpt_start": start, "excerpt_end": end,
            "original_content_length": len(content), "truncated": bool(start or end < len(content)),
            "offset_unit": "characters_in_source_chunk",
            "evidence_scope": "excerpt_only" if start or end < len(content) else "source_chunk",
            "sentence_boundary_preserved": True}


def _seed_components(graph, seeds):
    adjacency = defaultdict(set)
    support = defaultdict(set)
    covered = set()
    for row in seeds:
        start, end = row.get("excerpt_start", 0), row.get("excerpt_start", 0) + len(row["content"])
        for index in graph.source_edges.get(row["id"], ()):
            edge = graph.edges[index]
            if start <= edge.excerpt_start < edge.excerpt_end <= end:
                adjacency[edge.left].add(edge.right)
                adjacency[edge.right].add(edge.left)
                support[edge.left].add(row["id"])
                support[edge.right].add(row["id"])
                covered.add(index)
    components, visited = [], set()
    for node in sorted(adjacency):
        if node in visited:
            continue
        nodes, stack = set(), [node]
        while stack:
            current = stack.pop()
            if current in nodes:
                continue
            nodes.add(current)
            stack.extend(adjacency[current] - nodes)
        visited.update(nodes)
        components.append((nodes, set().union(*(support[node] for node in nodes))))
    return components, covered


def _paths(graph, question, seeds, idea):
    folded, unicode_folded = question.translate(_ASCII_CASE_TRANSLATION), question.casefold()
    anchors = {node for node in graph.adjacency if _question_mentions(question, node[1], folded, unicode_folded)}
    components, covered = _seed_components(graph, seeds)
    labels = {node: {node} for node in anchors}
    supports, roots = {}, []
    for nodes, sources in components:
        matched = nodes & anchors
        if matched or idea:
            for node in nodes:
                labels[node] = matched or {node}
                supports[node] = sources
                roots.append(node)
    roots.sort(key=lambda node: (min(next(i for i, row in enumerate(seeds) if row["id"] == sid)
                                   for sid in supports[node]), len(graph.adjacency[node]), node))
    limited = len(roots) > MAX_START_NODES
    found, states, hub_seen = [], 0, False
    selected_ids = {row["id"] for row in seeds}
    for root in roots[:MAX_START_NODES]:
        queue = [(root, (root,), ())]
        while queue:
            current, nodes, route = queue.pop(0)
            hub_seen |= current in graph.hubs
            for index in graph.adjacency.get(current, ()):
                states += 1
                if states > MAX_PATH_STATES:
                    return found, True, hub_seen, len(anchors), states - 1
                if index in covered or index in route:
                    continue
                edge = graph.edges[index]
                # A claim elsewhere in an already selected excerpt's chunk is
                # not visible evidence. Append-only expansion cannot pretend
                # that this hidden statement was supplied to the writer.
                if edge.source_id in selected_ids:
                    continue
                other = edge.right if edge.left == current else edge.left
                if other in nodes or other in graph.hubs:
                    hub_seen |= other in graph.hubs
                    continue
                route_next, nodes_next = (*route, index), (*nodes, other)
                new_ids = {graph.edges[item].source_id for item in route_next} - selected_ids
                target_labels = labels.get(other, set()) - labels[root]
                if new_ids and (target_labels or idea):
                    source_ids = set().union(supports[root], supports.get(other, set()),
                                             (graph.edges[item].source_id for item in route_next))
                    # Idea discovery still requires a real connected statement
                    # extending a selected source, not a relation-only match.
                    found.append({"nodes": nodes_next, "edges": route_next,
                                  "source_ids": source_ids, "new_ids": new_ids,
                                  "anchor_gain": len(target_labels), "root": root})
                    if len(found) >= MAX_PATHS:
                        return found, True, hub_seen, len(anchors), states
                if len(route_next) < 2:
                    queue.append((other, nodes_next, route_next))
    found.sort(key=lambda path: (-path["anchor_gain"], len(path["new_ids"]), len(path["edges"]),
                                  tuple(sorted(path["new_ids"])), path["nodes"]))
    return found, limited, hub_seen, len(anchors), states


class BoundedEvidenceGraph:
    """One-project-view-at-a-time cache; safe to share if calls pass live scope.

    Concurrent calls keep their immutable snapshot locally; replacing this
    cache cannot change an in-flight graph. No process-global or cross-user
    source cache is created. ``prepare`` is optional; ``expand`` lazily prepares.
    """
    def __init__(self):
        self._graph = None

    def _prepare(self, live):
        rows, key, limited = _canonical_window(live)
        graph = self._graph
        hit = graph is not None and graph.key == key
        if not hit:
            graph = _build(rows, key, limited)
            self._graph = graph
        return graph, hit

    def prepare(self, chunks: Sequence[Mapping]) -> dict:
        if len(chunks) > MAX_INPUT_CHUNKS:
            self._graph = None
            return {"indexed_chunks": 0, "edge_count": 0, "limited": True, "cache_hit": False}
        graph, hit = self._prepare(_live_chunks(chunks))
        return {"indexed_chunks": graph.indexed_chunks, "edge_count": len(graph.edges),
                "limited": graph.limited, "cache_hit": hit}

    def expand(self, question: str, seed_sources: Sequence[Mapping], chunks: Sequence[Mapping], *,
               max_sources: int = 8, max_source_bytes: int = 12_000,
               protected_source_ids: Sequence[str] = ()) -> dict:
        """Append complete real source windows; no extra model pass."""
        seeds = [dict(row) for row in seed_sources]
        trace = {"algorithm": "bounded_evidence_graph_v1", "status": "unchanged", "max_hops": 2,
                 "model_calls": 0, "semantic_entailment_verified": False,
                 "association_not_proof": True, "kept_source_ids": [row["id"] for row in seeds],
                 "bridge_source_ids": [], "paths": [], "cache_hit": False,
                 "protected_source_ids": sorted(set(protected_source_ids)), "search_states": 0}
        gaps = []

        def finish():
            trace["kept_source_ids"] = [row["id"] for row in seeds]
            trace["source_bytes"] = sum(len(row["content"].encode("utf-8")) for row in seeds)
            return {"sources": seeds, "trace": trace, "gaps": [_gap(code) for code in dict.fromkeys(gaps)]}

        if not graph_retrieval_intent(question):
            trace["reason"] = "not_requested"
            return finish()
        if len(chunks) > MAX_INPUT_CHUNKS:
            gaps.append("graph_window_limited")
            return finish()
        live = _live_chunks(chunks)
        valid = [row for row in seeds if row.get("id") in live and _seed_span(row, live[row["id"]])]
        if len(valid) != len(seeds):
            gaps.append("graph_source_unavailable")
            seeds = valid
        if not seeds:
            gaps.append("graph_no_source_seed")
            return finish()
        graph, hit = self._prepare(live)
        trace.update(cache_hit=hit, indexed_chunks=graph.indexed_chunks, edge_count=len(graph.edges))
        gaps += ["graph_association_not_proof", "graph_parser_is_partial"]
        if graph.limited:
            gaps.append("graph_window_limited")
        paths, limited, hubs, anchors, states = _paths(graph, _text(question)[:MAX_QUERY_CHARS], seeds, detect_hypothesis_intent(question))
        trace.update(query_anchor_count=anchors, search_states=states)
        if limited:
            gaps.append("graph_search_limited")
        if hubs:
            gaps.append("graph_hub_omitted")
        if not paths:
            gaps.append("graph_no_connected_path")
            return finish()
        cap, byte_cap = _limit(max_sources, 8, 8), _limit(max_source_bytes, 12_000, 12_000)
        selected = {row["id"] for row in seeds}
        used = sum(len(row["content"].encode("utf-8")) for row in seeds)
        for path in paths:
            missing = sorted(path["new_ids"] - selected)
            if not missing:
                continue
            edge_rows = [graph.edges[index] for index in path["edges"]]
            selected_rows = {row["id"]: row for row in seeds}
            # Earlier candidate paths may already have appended a window from
            # this source. Reusing its ID cannot expose a different hidden
            # claim farther away in the canonical chunk.
            if any(edge.source_id in selected_rows and not (
                    selected_rows[edge.source_id].get("excerpt_start", 0) <= edge.excerpt_start
                    < edge.excerpt_end <= selected_rows[edge.source_id].get("excerpt_start", 0)
                    + len(selected_rows[edge.source_id]["content"])) for edge in edge_rows):
                gaps.append("graph_path_window_incomplete")
                continue
            windows = {identifier: _candidate_window(live[identifier],
                       [edge for edge in edge_rows if edge.source_id == identifier]) for identifier in missing}
            if any(window is None for window in windows.values()):
                gaps.append("graph_capacity_reached")
                continue
            additions = [{**live[identifier], **windows[identifier]} for identifier in missing]
            size = sum(len(row["content"].encode("utf-8")) for row in additions)
            if (len(seeds) + len(additions) > cap or used + size > byte_cap
                    or len(trace["bridge_source_ids"]) + len(additions) > MAX_ADDED_SOURCES):
                gaps.append("graph_capacity_reached")
                continue
            for row in additions:
                content = row["content"]
                item = {key: row[key] for key in ("id", "document_id", "filename", "ordinal", "page") if key in row}
                used_labels = {item.get("label") for item in seeds}
                label_number = len(seeds) + 1
                while f"S{label_number}" in used_labels:
                    label_number += 1
                item.update(windows[row["id"]], filename=_text(row.get("filename"))[:120] or "未命名资料",
                            ordinal=row.get("ordinal", 0), label=f"S{label_number}")
                seeds.append(item)
                trace["bridge_source_ids"].append(row["id"])
                selected.add(row["id"])
                if item["truncated"]:
                    gaps.append("graph_excerpt_windowed")
            used += size
            if any(edge.polarity < 0 for edge in edge_rows):
                gaps.append("graph_counterevidence_in_path")
            conditions = {edge.conditions for edge in edge_rows}
            for identifier in path["source_ids"] & (selected - path["new_ids"]):
                conditions.update(graph.edges[index].conditions for index in graph.source_edges.get(identifier, ()))
            if len(conditions) > 1:
                gaps.append("graph_path_conditions_differ")
            trace["paths"].append({"source_ids": sorted(path["source_ids"]),
                                   "bridge_source_ids": missing, "expansion_hops": len(path["edges"]),
                                   "entities": [node[1] for node in path["nodes"]],
                                   "edges": [{"source_id": edge.source_id, "relation": edge.relation,
                                              "polarity": edge.polarity, "conditional": bool(edge.conditions),
                                              "excerpt_start": edge.excerpt_start, "excerpt_end": edge.excerpt_end,
                                              "kind": edge.kind} for edge in edge_rows],
                                   "source_windows": [{"source_id": row["id"],
                                                       "excerpt_start": row.get("excerpt_start", 0),
                                                       "excerpt_end": row.get("excerpt_start", 0) + len(row["content"]),
                                                       "content_sha256": hashlib.sha256(row["content"].encode()).hexdigest()}
                                                      for row in seeds if row["id"] in path["source_ids"]],
                                   "epistemic_status": "related_sources_not_entailment"})
        if trace["bridge_source_ids"]:
            trace["status"] = "expanded"
        return finish()
