"""Compensating atomic writeback for promoted ECLRR-v4 relations."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections import defaultdict
from copy import deepcopy
from typing import Any

from .models import GateResult

GRAPH_FIELD_SEP = "<SEP>"
_LOCKS: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def _json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def _formal_record(result: GateResult) -> dict[str, Any]:
    decision = result.judge_decision
    scores = {
        key: decision.get(key)
        for key in (
            "evidence_coverage",
            "semantic_composability",
            "relation_direction",
            "uncertainty_calibration",
            "total",
        )
    }
    return {
        "relation_id": result.relation_id,
        "canonical_key": result.canonical_key,
        "source": result.source,
        "target": result.target,
        "relation": result.relation,
        "relation_family": result.relation_family,
        "direction": result.direction,
        "description": result.description,
        "path_used": result.path_used,
        "supporting_paths": result.supporting_paths,
        "evidence_chain": result.evidence_chain,
        "evidence_chunk_ids": result.evidence_chunk_ids,
        "source_id": GRAPH_FIELD_SEP.join(result.evidence_chunk_ids),
        "generator_output": result.proposal,
        "judge_decision": result.judge_decision,
        "judge_scores": scores,
        "decision_score": scores.get("total"),
        "algorithm_version": "eclrr_v4",
        "review_status": "promoted",
        "query_eligible": "1",
        "is_discovered": "1",
        "provenance": "eclrr_v4",
    }


def _parse_relation_list(edge: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not edge:
        return []
    raw = edge.get("eclrr_relations")
    if not raw:
        return []
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, dict)]
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return (
        [dict(item) for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, list)
        else []
    )


def _graph_edge_payload(
    result: GateResult, old_edge: dict[str, Any] | None
) -> dict[str, Any]:
    formal = _formal_record(result)
    relations = _parse_relation_list(old_edge)
    relations = [
        item for item in relations if item.get("canonical_key") != result.canonical_key
    ]
    relations.append(formal)
    relations.sort(key=lambda item: str(item.get("canonical_key") or ""))

    if result.action == "refine" or old_edge is None:
        payload = {
            "keywords": result.relation,
            "description": result.description,
            "source_id": formal["source_id"],
            "relation": result.relation,
            "relation_family": result.relation_family,
            "direction": result.direction,
            "path_used": _json(result.path_used),
            "supporting_paths": _json(result.supporting_paths),
            "evidence_chain": _json(result.evidence_chain),
            "evidence_chunk_ids": _json(result.evidence_chunk_ids),
            "generator_output": _json(result.proposal),
            "judge_decision": _json(result.judge_decision),
            "judge_scores": _json(formal["judge_scores"]),
            "decision_score": str(formal["decision_score"]),
            "algorithm_version": "eclrr_v4",
            "review_status": "promoted",
            "query_eligible": "1",
            "is_discovered": "1",
            "provenance": "eclrr_v4",
            "relation_id": result.relation_id,
            "canonical_key": result.canonical_key,
            "weight": "1.0",
        }
    else:
        payload = dict(old_edge)
        payload["has_promoted_relations"] = "1"
    payload["eclrr_relations"] = _json(relations)
    return {key: value for key, value in payload.items() if value is not None}


def _vdb_payload(result: GateResult) -> dict[str, Any]:
    formal = _formal_record(result)
    evidence_text = " | ".join(
        f"{item.get('source')}->{item.get('target')}"
        f"[{item.get('chunk_id')}]:{item.get('quote')}"
        for item in result.evidence_chain
    )
    content = (
        f"{result.source}\n{result.relation_family}:{result.relation}\n"
        f"direction:{result.direction}\n{result.target}\n{result.description}\n"
        f"evidence:{evidence_text}\nchunks:{','.join(result.evidence_chunk_ids)}"
    )
    return {
        "src_id": result.source,
        "tgt_id": result.target,
        "source_id": formal["source_id"],
        "content": content,
        "file_path": "",
        "relation_id": result.relation_id,
        "canonical_key": result.canonical_key,
        "relation": result.relation,
        "relation_family": result.relation_family,
        "direction": result.direction,
        "description": result.description,
        "path_used": _json(result.path_used),
        "supporting_paths": _json(result.supporting_paths),
        "evidence_chain": _json(result.evidence_chain),
        "evidence_chunk_ids": _json(result.evidence_chunk_ids),
        "generator_output": _json(result.proposal),
        "judge_decision": _json(result.judge_decision),
        "judge_scores": _json(formal["judge_scores"]),
        "decision_score": str(formal["decision_score"]),
        "algorithm_version": "eclrr_v4",
        "review_status": "promoted",
        "query_eligible": "1",
        "is_discovered": "1",
        "provenance": "eclrr_v4",
    }


async def _commit(storage: Any, *, force_graph: bool = False) -> None:
    if storage is None:
        return
    callback = storage.index_done_callback
    parameters = inspect.signature(callback).parameters
    if force_graph and "force_save" in parameters:
        result = await storage.index_done_callback(force_save=True)
    else:
        result = await storage.index_done_callback()
    if result is False:
        raise RuntimeError("storage_commit_returned_false")


async def commit_promotion(
    result: GateResult,
    graph: Any,
    relationships_vdb: Any,
) -> None:
    if result.action not in {"create", "refine"}:
        return
    lock_name = (
        str(getattr(graph, "workspace", ""))
        + ":"
        + str(getattr(graph, "namespace", "graph"))
    )
    async with _LOCKS[lock_name]:
        old_edge = await graph.get_edge(result.source, result.target)
        old_edge = deepcopy(dict(old_edge)) if old_edge is not None else None
        old_vector = None
        if relationships_vdb is not None and hasattr(relationships_vdb, "get_by_id"):
            old_vector = await relationships_vdb.get_by_id(result.relation_id)
        graph_written = False
        vector_written = False
        try:
            await graph.upsert_edge(
                result.source,
                result.target,
                _graph_edge_payload(result, old_edge),
            )
            graph_written = True
            await _commit(graph, force_graph=True)
            if relationships_vdb is not None:
                await relationships_vdb.upsert(
                    {result.relation_id: _vdb_payload(result)}
                )
                vector_written = True
                await _commit(relationships_vdb)
        except Exception:
            if vector_written and relationships_vdb is not None:
                if old_vector and old_vector.get("content"):
                    restored = {
                        key: value
                        for key, value in old_vector.items()
                        if not key.startswith("__") and key not in {"id", "created_at"}
                    }
                    await relationships_vdb.upsert({result.relation_id: restored})
                else:
                    await relationships_vdb.delete([result.relation_id])
                await _commit(relationships_vdb)
            if graph_written:
                if old_edge is None:
                    await graph.remove_edges([(result.source, result.target)])
                else:
                    await graph.upsert_edge(result.source, result.target, old_edge)
                await _commit(graph, force_graph=True)
            raise


async def commit_promotions(
    results: list[GateResult],
    graph: Any,
    relationships_vdb: Any,
) -> tuple[list[GateResult], dict[str, str]]:
    committed: list[GateResult] = []
    failed: dict[str, str] = {}
    for result in results:
        if result.action not in {"create", "refine"}:
            continue
        try:
            await commit_promotion(result, graph, relationships_vdb)
            committed.append(result)
        except Exception as exc:
            failed[result.review_id] = f"writeback_failure:{type(exc).__name__}:{exc}"
    return committed, failed
