"""Build a deterministic, evidence-aware view of the source graph."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .models import FactEdge, PathStep

GRAPH_FIELD_SEP = "<SEP>"

_FUZZY_TERMS = (
    "未明说",
    "未明确",
    "关系不明",
    "隐秘关系",
    "某种联系",
    "物品关联",
    "暗示",
    "无法公开命名",
    "纸背关系",
    "可能关联",
    "unspecified",
    "unclear relationship",
    "implicit relation",
    "some connection",
)
_GENERIC_RELATIONS = {
    "related",
    "associated",
    "relation",
    "联系",
    "相关",
    "有关",
    "某种联系",
    "关系不明",
}
_PROMOTED_PROVENANCE = {"eclrr_v4"}
_LEGACY_PROVENANCE = {
    "path_edge_discovery",
    "self_study",
    "llm_expansion",
    "legacy_synthetic",
}


def node_name(node: dict[str, Any]) -> str:
    return str(
        node.get("id")
        or node.get("entity_id")
        or node.get("entity_name")
        or node.get("label")
        or ""
    ).strip()


def edge_endpoints(edge: dict[str, Any]) -> tuple[str, str]:
    pair = edge.get("src_tgt")
    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
        return str(pair[0]).strip(), str(pair[1]).strip()
    return (
        str(edge.get("src_id") or edge.get("source") or "").strip(),
        str(edge.get("tgt_id") or edge.get("target") or "").strip(),
    )


def split_source_ids(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = str(value or "").split(GRAPH_FIELD_SEP)
    return tuple(
        dict.fromkeys(str(item).strip() for item in values if str(item).strip())
    )


def _truthy(value: Any) -> bool:
    if value is True or value == 1:
        return True
    return isinstance(value, str) and value.strip().lower() in {"1", "true", "yes"}


def classify_edge(edge: dict[str, Any]) -> str:
    provenance = str(edge.get("provenance") or "").strip().lower()
    review_status = str(edge.get("review_status") or "").strip().lower()
    if review_status == "promoted" and provenance in _PROMOTED_PROVENANCE:
        return "promoted"
    if review_status in {"candidate", "pending", "proposed"}:
        return "legacy"
    if _truthy(edge.get("is_discovered")) or provenance in _LEGACY_PROVENANCE:
        return "legacy"
    text = " ".join(
        str(edge.get(key) or "") for key in ("relation", "keywords", "description")
    ).casefold()
    if any(term.casefold() in text for term in _FUZZY_TERMS):
        return "fuzzy"
    return "fact"


def relation_name(edge: dict[str, Any]) -> str:
    return str(edge.get("relation") or edge.get("keywords") or "related").strip()


def relation_family(edge: dict[str, Any]) -> str:
    value = str(edge.get("relation_family") or "").strip().lower()
    if value:
        return value
    relation = relation_name(edge).casefold()
    if any(
        term in relation
        for term in ("parent", "child", "sibling", "kin", "亲", "父", "母")
    ):
        return "kinship"
    if any(term in relation for term in ("cause", "lead", "导致", "造成")):
        return "causation"
    if any(term in relation for term in ("influence", "影响")):
        return "influence"
    if any(term in relation for term in ("conflict", "oppose", "冲突", "对立")):
        return "opposition"
    if any(term in relation for term in ("love", "romantic", "情感", "伴侣")):
        return "romantic"
    return "other_specific"


def _stored_promoted_relations(edge: dict[str, Any]) -> list[dict[str, Any]]:
    raw = edge.get("eclrr_relations")
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, dict)]
    if not raw:
        return []
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return (
        [dict(item) for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, list)
        else []
    )


def _specificity(edge: dict[str, Any]) -> float:
    relation = relation_name(edge).strip().casefold()
    description = str(edge.get("description") or "").strip()
    generic = relation in _GENERIC_RELATIONS or any(
        term.casefold() in relation for term in _FUZZY_TERMS
    )
    value = 0.25 if generic else 0.75
    if len(description) >= 24:
        value += 0.15
    if edge.get("relation"):
        value += 0.10
    return min(1.0, value)


def _edge_id(source: str, target: str, edge: dict[str, Any]) -> str:
    identity = "|".join(
        (
            source,
            relation_family(edge),
            relation_name(edge),
            str(edge.get("direction") or "undirected"),
            target,
            GRAPH_FIELD_SEP.join(split_source_ids(edge.get("source_id"))),
        )
    )
    return "fact-" + hashlib.md5(identity.encode("utf-8")).hexdigest()


@dataclass
class FactGraphView:
    nodes: dict[str, dict[str, Any]]
    fact_edges: dict[str, FactEdge]
    fuzzy_by_pair: dict[tuple[str, str], list[dict[str, Any]]]
    exact_relations: set[str]
    adjacency: dict[str, list[PathStep]]

    @classmethod
    def build(
        cls,
        nodes: Iterable[dict[str, Any]],
        edges: Iterable[dict[str, Any]],
    ) -> "FactGraphView":
        nodes_by_name = {
            name: dict(node) for node in nodes if (name := node_name(node))
        }
        fact_edges: dict[str, FactEdge] = {}
        fuzzy_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
        exact_relations: set[str] = set()
        adjacency: dict[str, list[PathStep]] = {name: [] for name in nodes_by_name}

        ordered_edges = sorted(
            (dict(edge) for edge in edges),
            key=lambda edge: (
                *edge_endpoints(edge),
                relation_name(edge),
                json.dumps(edge, sort_keys=True, default=str),
            ),
        )
        for raw in ordered_edges:
            source, target = edge_endpoints(raw)
            if not source or not target or source == target:
                continue
            if source not in nodes_by_name or target not in nodes_by_name:
                continue
            edge_class = classify_edge(raw)
            for promoted in _stored_promoted_relations(raw):
                key = promoted.get("canonical_key") or canonical_relation_key(
                    str(promoted.get("source") or source),
                    str(promoted.get("relation_family") or "other_specific"),
                    str(promoted.get("relation") or ""),
                    str(promoted.get("direction") or "undirected"),
                    str(promoted.get("target") or target),
                )
                exact_relations.add(str(key))
            if edge_class == "promoted":
                exact_relations.add(
                    str(
                        raw.get("canonical_key")
                        or canonical_relation_key(
                            source,
                            relation_family(raw),
                            relation_name(raw),
                            str(raw.get("direction") or "undirected"),
                            target,
                        )
                    )
                )
            pair = tuple(sorted((source, target)))
            if edge_class == "fuzzy":
                fuzzy_by_pair.setdefault(pair, []).append(raw)
                continue
            if edge_class != "fact":
                continue
            edge_id = _edge_id(source, target, raw)
            fact = FactEdge(
                edge_id=edge_id,
                source=source,
                target=target,
                relation=relation_name(raw),
                relation_family=relation_family(raw),
                description=str(raw.get("description") or ""),
                source_ids=split_source_ids(raw.get("source_id")),
                edge_class="fact",
                declared_direction=str(raw.get("direction") or "undirected")
                .strip()
                .lower(),
                specificity=_specificity(raw),
                grounding=1.0 if split_source_ids(raw.get("source_id")) else 0.0,
                raw=raw,
            )
            fact_edges[edge_id] = fact
            exact_relations.add(
                canonical_relation_key(
                    source,
                    fact.relation_family,
                    fact.relation,
                    fact.declared_direction,
                    target,
                )
            )
            adjacency[source].append(_step(fact, source, target))
            adjacency[target].append(_step(fact, target, source))

        for name, steps in adjacency.items():
            steps.sort(
                key=lambda step: (
                    -fact_edges[step.edge_id].grounding,
                    -fact_edges[step.edge_id].specificity,
                    len(adjacency.get(step.traversal_target, [])),
                    step.traversal_target,
                    step.edge_id,
                )
            )
        return cls(nodes_by_name, fact_edges, fuzzy_by_pair, exact_relations, adjacency)

    def degree(self, node: str) -> int:
        return len(self.adjacency.get(node, ()))

    def node_type(self, node: str) -> str:
        return (
            str(self.nodes.get(node, {}).get("entity_type") or "unknown")
            .strip()
            .lower()
        )

    def fuzzy_edge(self, source: str, target: str) -> dict[str, Any] | None:
        values = self.fuzzy_by_pair.get(tuple(sorted((source, target)))) or []
        return dict(values[0]) if values else None


def _step(edge: FactEdge, traversal_source: str, traversal_target: str) -> PathStep:
    declared = edge.declared_direction
    if declared in {"forward", "source_to_target", "outbound", "directed", "->"}:
        direction = "forward" if traversal_source == edge.source else "inverse"
    elif declared in {"reverse", "target_to_source", "inbound", "<-"}:
        direction = "inverse" if traversal_source == edge.source else "forward"
    else:
        direction = "undirected"
    return PathStep(
        edge_id=edge.edge_id,
        source=edge.source,
        target=edge.target,
        traversal_source=traversal_source,
        traversal_target=traversal_target,
        traversal_direction=direction,
        relation=edge.relation,
        relation_family=edge.relation_family,
        source_ids=edge.source_ids,
    )


def canonical_relation_key(
    source: str,
    family: str,
    relation: str,
    direction: str,
    target: str,
) -> str:
    source = str(source).strip()
    target = str(target).strip()
    direction = str(direction or "undirected").strip().lower()
    family = re.sub(r"\s+", "_", str(family).strip().lower())
    relation = re.sub(r"\s+", "_", str(relation).strip().lower())
    if direction == "target_to_source":
        source, target = target, source
        direction = "source_to_target"
    if direction in {"undirected", "bidirectional", "both"} and source > target:
        source, target = target, source
    return "|".join((source, family, relation, direction, target))
