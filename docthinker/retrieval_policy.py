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


def relation_has_evidence(relation: dict[str, Any]) -> bool:
    for key in ("evidence", "evidence_chain", "evidence_chunk_ids"):
        value = relation.get(key)
        if isinstance(value, (list, tuple, dict)) and bool(value):
            return True
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value.strip())
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = None
            if isinstance(parsed, (list, tuple, dict)) and bool(parsed):
                return True
    return False


def relation_confidence(relation: dict[str, Any]) -> float:
    for key in ("confidence", "weight"):
        try:
            return max(0.0, min(1.0, float(relation.get(key, 0.0))))
        except (TypeError, ValueError):
            continue
    return 0.0


def _retrieval_relevance(relation: dict[str, Any]) -> tuple[int, float]:
    try:
        distance = float(relation.get("retrieval_distance"))
    except (TypeError, ValueError):
        return 0, 0.0
    return 1, max(0.0, min(1.0, 1.0 - distance))


def select_relations_for_query(
    relations: list[dict[str, Any]], query_param: Any
) -> list[dict[str, Any]]:
    """Apply provenance, confidence, evidence, and count budgets to KG relations."""
    originals: list[dict[str, Any]] = []
    discovered: list[dict[str, Any]] = []
    include_discovered = bool(
        getattr(query_param, "include_discovered_edges", False)
    )
    min_confidence = float(
        getattr(query_param, "min_discovered_edge_confidence", 0.80)
    )
    require_evidence = bool(
        getattr(query_param, "require_discovered_evidence", True)
    )

    for relation in relations:
        if not truthy_metadata(relation.get("is_discovered")):
            originals.append(relation)
            continue
        if not include_discovered:
            continue
        if not truthy_metadata(relation.get("query_eligible")):
            continue
        if relation_confidence(relation) < min_confidence:
            continue
        if require_evidence and not relation_has_evidence(relation):
            continue
        discovered.append(relation)

    discovered.sort(
        key=lambda item: (
            *_retrieval_relevance(item),
            relation_confidence(item),
            float(item.get("rank", 0) or 0),
        ),
        reverse=True,
    )
    max_relations = max(1, int(getattr(query_param, "max_relations", 32)))
    max_discovered = max(
        0, int(getattr(query_param, "max_discovered_relations", 8))
    )
    selected_discovered = discovered[: min(max_discovered, max_relations)]
    original_budget = max_relations - len(selected_discovered)
    selected = originals[:original_budget] + selected_discovered
    _log.info(
        "Relation evidence gate: %d original + %d discovered retained from %d",
        min(len(originals), original_budget),
        len(selected_discovered),
        len(relations),
    )
    return selected
