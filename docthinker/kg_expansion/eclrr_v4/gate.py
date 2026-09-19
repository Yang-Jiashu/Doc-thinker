"""Deterministic post-Judge gate for ECLRR-v4."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from typing import Any

from .graph_view import FactGraphView, canonical_relation_key, split_source_ids
from .clue_semantics import PERSON_ONLY_RELATIONS, clue_relation_ontology
from .models import (
    ECLRRConfig,
    EvidencePackage,
    GateResult,
    JudgeDecision,
    Proposal,
    path_to_dict,
)
from .prompts import RELATION_FAMILIES

_GENERIC_RELATIONS = {
    "related",
    "associated",
    "connected",
    "relation",
    "联系",
    "相关",
    "有关",
    "某种联系",
    "evidence_linkage",
    "shared_object_connection",
    "route_alignment",
    "reserved_seat_ritual",
    "temporal_alignment",
}
_DIRECTIONS = {"source_to_target", "target_to_source", "undirected"}
_SURFACE_RELATION_TERMS = (
    "artifact",
    "ritual",
    "trace",
    "linkage",
    "alignment",
    "reserved_seat",
    "symbolic",
    "archival",
)
def _node_type(node: dict[str, Any]) -> str:
    return str(node.get("entity_type") or node.get("type") or "").strip().casefold()


def _is_person(node: dict[str, Any]) -> bool:
    value = _node_type(node)
    return value in {"person", "human", "character", "\u4eba\u7269", "\u4eba"}


async def _chunk_map(text_chunks: Any, chunk_ids: list[str]) -> dict[str, str]:
    records = await text_chunks.get_by_ids(chunk_ids)
    result: dict[str, str] = {}
    for index, record in enumerate(records or []):
        if not isinstance(record, dict):
            continue
        chunk_id = str(record.get("chunk_id") or record.get("id") or "").strip()
        if not chunk_id and index < len(chunk_ids):
            chunk_id = chunk_ids[index]
        if chunk_id and isinstance(record.get("content"), str):
            result[chunk_id] = record["content"]
    return result


def _node_aliases(name: str, node: dict[str, Any]) -> tuple[str, ...]:
    aliases: list[str] = [name]
    raw = node.get("aliases") or node.get("alias") or []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            raw = parsed if isinstance(parsed, list) else re.split(r"[,，;|]", raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            raw = re.split(r"[,，;|]", raw)
    if isinstance(raw, (list, tuple, set)):
        aliases.extend(str(value).strip() for value in raw if str(value).strip())
    return tuple(dict.fromkeys(aliases))


def _noop(review_id: str, source: str, target: str, reason: str) -> GateResult:
    return GateResult(
        action="no-op",
        reason=reason,
        review_id=review_id,
        source=source,
        target=target,
    )


async def deterministic_gate(
    package: EvidencePackage,
    proposal: Proposal,
    decision: JudgeDecision,
    view: FactGraphView,
    text_chunks: Any,
    config: ECLRRConfig,
) -> GateResult:
    item = package.review_item
    if proposal.review_id != item.review_id or decision.review_id != item.review_id:
        return _noop(item.review_id, item.source, item.target, "review_id_mismatch")
    if proposal.source != item.source or proposal.target != item.target:
        return _noop(item.review_id, item.source, item.target, "endpoint_mismatch")
    if proposal.source not in view.nodes or proposal.target not in view.nodes:
        return _noop(item.review_id, item.source, item.target, "unknown_endpoint")
    if decision.decision == "reject":
        return _noop(item.review_id, item.source, item.target, "judge_reject")
    if not (
        decision.total >= 8
        and decision.evidence_coverage == 4
        and decision.semantic_composability >= 2
        and decision.relation_direction >= 1
    ):
        return _noop(item.review_id, item.source, item.target, "judge_threshold_failed")

    relation = (
        decision.revised_relation
        if decision.decision == "revise"
        else proposal.relation
    )
    family = (
        decision.revised_relation_family
        if decision.decision == "revise"
        else proposal.relation_family
    )
    relation = str(relation or "").strip()
    family = str(family or "").strip().lower()
    if (
        not relation
        or relation.casefold() in _GENERIC_RELATIONS
        or any(term in relation.casefold() for term in _SURFACE_RELATION_TERMS)
    ):
        return _noop(item.review_id, item.source, item.target, "generic_relation")
    if family not in RELATION_FAMILIES:
        return _noop(
            item.review_id, item.source, item.target, "relation_family_outside_ontology"
        )
    if proposal.direction not in _DIRECTIONS:
        return _noop(item.review_id, item.source, item.target, "invalid_direction")
    expected_relations = clue_relation_ontology(
        [evidence.quote for evidence in package.direct_evidence]
    )
    if expected_relations is not None and relation not in expected_relations:
        return _noop(
            item.review_id, item.source, item.target, "relation_conflicts_with_direct_clue"
        )
    if relation in PERSON_ONLY_RELATIONS and not (
        _is_person(view.nodes[proposal.source])
        and _is_person(view.nodes[proposal.target])
    ):
        return _noop(
            item.review_id, item.source, item.target, "relation_endpoint_type_mismatch"
        )

    path = item.primary_path
    if not config.min_hops <= path.hops <= config.max_hops:
        return _noop(item.review_id, item.source, item.target, "path_hops_out_of_range")
    if path.nodes[0] != item.source or path.nodes[-1] != item.target:
        return _noop(item.review_id, item.source, item.target, "path_endpoint_mismatch")
    if len(set(path.nodes)) != len(path.nodes):
        return _noop(item.review_id, item.source, item.target, "path_cycle")
    for hop_index, step in enumerate(path.steps):
        edge = view.fact_edges.get(step.edge_id)
        if edge is None or edge.edge_class != "fact":
            return _noop(item.review_id, item.source, item.target, "non_fact_path_edge")
        if step.traversal_source not in {
            edge.source,
            edge.target,
        } or step.traversal_target not in {edge.source, edge.target}:
            return _noop(
                item.review_id, item.source, item.target, "path_not_continuous"
            )
        if (
            step.traversal_source != path.nodes[hop_index]
            or step.traversal_target != path.nodes[hop_index + 1]
        ):
            return _noop(
                item.review_id, item.source, item.target, "path_not_continuous"
            )

    primary_by_hop = {
        evidence.hop_index: evidence for evidence in package.primary_evidence
    }
    if set(primary_by_hop) != set(range(path.hops)):
        return _noop(
            item.review_id, item.source, item.target, "missing_primary_evidence"
        )
    if item.fuzzy_edge and not package.direct_evidence:
        return _noop(
            item.review_id, item.source, item.target, "missing_direct_endpoint_evidence"
        )
    required_refs = {
        evidence.evidence_id
        for evidence in (*package.primary_evidence, *package.direct_evidence)
    }
    if not required_refs.issubset(set(proposal.evidence_refs)):
        return _noop(
            item.review_id, item.source, item.target, "generator_missing_evidence_ref"
        )
    if not required_refs.issubset(set(decision.verified_evidence_refs)):
        return _noop(
            item.review_id, item.source, item.target, "judge_missing_evidence_ref"
        )

    chunk_ids = list(dict.fromkeys(
        evidence.chunk_id
        for evidence in (*package.primary_evidence, *package.direct_evidence)
    ))
    chunks = await _chunk_map(text_chunks, chunk_ids)
    evidence_chain: list[dict[str, Any]] = []
    for hop_index, step in enumerate(path.steps):
        evidence = primary_by_hop[hop_index]
        if evidence.edge_id != step.edge_id or evidence.chunk_id not in step.source_ids:
            return _noop(
                item.review_id, item.source, item.target, "evidence_not_owned_by_edge"
            )
        text = chunks.get(evidence.chunk_id)
        if text is None:
            return _noop(item.review_id, item.source, item.target, "missing_chunk")
        if not (0 <= evidence.start < evidence.end <= len(text)):
            return _noop(
                item.review_id, item.source, item.target, "invalid_quote_offsets"
            )
        if text[evidence.start : evidence.end] != evidence.quote:
            return _noop(
                item.review_id, item.source, item.target, "quote_not_exact_substring"
            )
        aliases = (
            *_node_aliases(step.source, view.nodes[step.source]),
            *_node_aliases(step.target, view.nodes[step.target]),
        )
        if not any(alias.casefold() in evidence.quote.casefold() for alias in aliases):
            return _noop(
                item.review_id, item.source, item.target, "quote_not_about_hop"
            )
        evidence_chain.append({"evidence_kind": "path", **asdict(evidence)})

    if item.fuzzy_edge:
        fuzzy_source_ids = set(split_source_ids(item.fuzzy_edge.get("source_id")))
        endpoint_aliases = (
            _node_aliases(item.source, view.nodes[item.source]),
            _node_aliases(item.target, view.nodes[item.target]),
        )
        for evidence in package.direct_evidence:
            text = chunks.get(evidence.chunk_id)
            if text is None or evidence.chunk_id not in fuzzy_source_ids:
                return _noop(
                    item.review_id,
                    item.source,
                    item.target,
                    "direct_evidence_not_owned_by_fuzzy_edge",
                )
            if not (0 <= evidence.start < evidence.end <= len(text)):
                return _noop(
                    item.review_id, item.source, item.target, "invalid_direct_quote_offsets"
                )
            if text[evidence.start : evidence.end] != evidence.quote:
                return _noop(
                    item.review_id, item.source, item.target, "direct_quote_not_exact_substring"
                )
            if not all(
                any(alias.casefold() in evidence.quote.casefold() for alias in aliases)
                for aliases in endpoint_aliases
            ):
                return _noop(
                    item.review_id, item.source, item.target, "direct_quote_missing_endpoint"
                )
            evidence_chain.insert(
                0, {"evidence_kind": "direct_endpoint", **asdict(evidence)}
            )

    canonical_key = canonical_relation_key(
        proposal.source,
        family,
        relation,
        proposal.direction,
        proposal.target,
    )
    if canonical_key in view.exact_relations:
        return _noop(
            item.review_id, item.source, item.target, "equivalent_relation_exists"
        )
    relation_id = "rel-eclrr-" + hashlib.md5(canonical_key.encode("utf-8")).hexdigest()
    action = "refine" if item.fuzzy_edge else "create"
    chain = " → ".join(path.nodes)
    description = (
        f"{chain} 的逐步关系由 chunk 证据完整支持；结合端点直接暗示，"
        f"推断 {proposal.source} → {proposal.target} 为 {relation}。"
        "该结论属于证据支持的隐藏关系，并非原文明示。"
    )
    return GateResult(
        action=action,
        reason="gate_passed",
        review_id=item.review_id,
        source=proposal.source,
        target=proposal.target,
        relation=relation,
        relation_family=family,
        direction=proposal.direction,
        description=description,
        canonical_key=canonical_key,
        relation_id=relation_id,
        evidence_chain=evidence_chain,
        evidence_chunk_ids=chunk_ids,
        path_used=list(path.nodes),
        supporting_paths=[path_to_dict(path)],
        proposal=asdict(proposal),
        judge_decision=asdict(decision),
    )
