"""Evidence and budget policy for inferred knowledge-graph relations."""

from __future__ import annotations

import json
import logging
from typing import Any

_log = logging.getLogger("docthinker.retrieval_policy")


def truthy_metadata(value: Any) -> bool:
    if value is True or value == 1:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def relation_has_evidence(relation: dict[str, Any]) -> bool:
    chain = _json_value(relation.get("evidence_chain"))
    chunk_ids = _json_value(relation.get("evidence_chunk_ids"))
    if not isinstance(chain, list) or not chain:
        return False
    if not isinstance(chunk_ids, list) or not chunk_ids:
        return False
    source_ids = {
        item.strip()
        for item in str(relation.get("source_id") or "").split("<SEP>")
        if item.strip()
    }
    if not set(map(str, chunk_ids)).issubset(source_ids):
        return False
    for evidence in chain:
        if not isinstance(evidence, dict):
            return False
        if not all(
            key in evidence for key in ("chunk_id", "quote", "start", "end", "edge_id")
        ):
            return False
        if not isinstance(evidence.get("start"), int) or not isinstance(
            evidence.get("end"), int
        ):
            return False
        if evidence["start"] < 0 or evidence["end"] <= evidence["start"]:
            return False
    return True


def relation_confidence(relation: dict[str, Any]) -> float:
    """Legacy helper retained for original-edge ranking compatibility."""
    for key in ("confidence", "weight"):
        try:
            return max(0.0, min(1.0, float(relation.get(key, 0.0))))
        except (TypeError, ValueError):
            continue
    return 0.0


def _judge_scores(relation: dict[str, Any]) -> dict[str, int]:
    value = _json_value(relation.get("judge_scores"))
    if not isinstance(value, dict):
        decision = _json_value(relation.get("judge_decision"))
        value = decision if isinstance(decision, dict) else {}
    scores: dict[str, int] = {}
    for key in (
        "evidence_coverage",
        "semantic_composability",
        "relation_direction",
        "uncertainty_calibration",
        "total",
    ):
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool):
            scores[key] = item
    return scores


def is_promoted_relation(relation: dict[str, Any]) -> bool:
    if str(relation.get("review_status") or "").strip().lower() != "promoted":
        return False
    if str(relation.get("provenance") or "").strip().lower() != "eclrr_v4":
        return False
    if str(relation.get("algorithm_version") or "").strip().lower() != "eclrr_v4":
        return False
    if not truthy_metadata(relation.get("query_eligible")):
        return False
    scores = _judge_scores(relation)
    return (
        scores.get("total", -1) >= 8
        and scores.get("evidence_coverage") == 4
        and scores.get("semantic_composability", -1) >= 2
        and scores.get("relation_direction", -1) >= 1
        and relation_has_evidence(relation)
    )


def _retrieval_relevance(relation: dict[str, Any]) -> tuple[int, float]:
    try:
        distance = float(relation.get("retrieval_distance"))
    except (TypeError, ValueError):
        return 0, 0.0
    return 1, max(0.0, min(1.0, 1.0 - distance))


def _expand_promoted_records(relation: dict[str, Any]) -> list[dict[str, Any]]:
    expanded = [relation]
    records = _json_value(relation.get("eclrr_relations"))
    if not isinstance(records, list):
        return expanded
    source = relation.get("src_id") or relation.get("source")
    target = relation.get("tgt_id") or relation.get("target")
    for record in records:
        if not isinstance(record, dict):
            continue
        item = dict(record)
        item["src_id"] = item.get("src_id") or item.get("source") or source
        item["tgt_id"] = item.get("tgt_id") or item.get("target") or target
        if item.get("canonical_key") == relation.get("canonical_key"):
            continue
        expanded.append(item)
    return expanded


def select_relations_for_query(
    relations: list[dict[str, Any]], query_param: Any
) -> list[dict[str, Any]]:
    """Allow source facts plus explicitly enabled, fully promoted ECLRR-v4 edges."""
    originals: list[dict[str, Any]] = []
    promoted: list[dict[str, Any]] = []
    include_discovered = bool(getattr(query_param, "include_discovered_edges", False))

    for physical_relation in relations:
        for relation in _expand_promoted_records(physical_relation):
            review_status = str(relation.get("review_status") or "").strip().lower()
            provenance = str(relation.get("provenance") or "").strip().lower()
            discovered = (
                truthy_metadata(relation.get("is_discovered"))
                or review_status in {"candidate", "pending", "proposed", "promoted"}
                or provenance
                in {
                    "eclrr_v4",
                    "path_edge_discovery",
                    "self_study",
                    "legacy_synthetic",
                    "llm_expansion",
                }
            )
            if not discovered:
                originals.append(relation)
                continue
            if include_discovered and is_promoted_relation(relation):
                promoted.append(relation)

    promoted_by_id: dict[str, dict[str, Any]] = {}
    for relation in promoted:
        key = str(
            relation.get("relation_id")
            or relation.get("canonical_key")
            or f"{relation.get('src_id')}|{relation.get('relation')}|{relation.get('tgt_id')}"
        )
        current = promoted_by_id.get(key)
        if current is None or _retrieval_relevance(relation) > _retrieval_relevance(
            current
        ):
            promoted_by_id[key] = relation
    promoted = list(promoted_by_id.values())
    promoted.sort(
        key=lambda item: (
            *_retrieval_relevance(item),
            _judge_scores(item).get("total", 0),
            str(item.get("relation_id") or item.get("canonical_key") or ""),
        ),
        reverse=True,
    )
    max_relations = max(1, int(getattr(query_param, "max_relations", 32)))
    max_promoted = max(0, int(getattr(query_param, "max_discovered_relations", 8)))
    selected_promoted = promoted[: min(max_promoted, max_relations)]
    original_budget = max_relations - len(selected_promoted)
    selected = originals[:original_budget] + selected_promoted
    _log.info(
        "Relation evidence gate: %d original + %d ECLRR-v4 promoted retained from %d",
        min(len(originals), original_budget),
        len(selected_promoted),
        len(relations),
    )
    return selected


__all__ = [
    "is_promoted_relation",
    "relation_confidence",
    "relation_has_evidence",
    "select_relations_for_query",
    "truthy_metadata",
]
