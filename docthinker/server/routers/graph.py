import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import FileResponse, Response

from ..schemas import EntityRelationshipRequest, RelationshipRequest
from ..state import state
from ..memory import get_session_memory_engine
from docthinker.kg_expansion import ExpandedNodeManager
from docthinker.image_assets import is_image_node, resolve_graph_node_color
from docthinker.memory_core import get_default_long_horizon_backend

import logging

_log = logging.getLogger("docthinker.graph")


router = APIRouter()

_ECLRR_RUN_TASKS: Dict[str, asyncio.Task] = {}
_ECLRR_RUN_STATUS: Dict[str, Dict[str, Any]] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_memory_engine_or_raise(session_id: Optional[str]):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    engine = get_session_memory_engine(session_id)
    if engine is None:
        raise HTTPException(status_code=501, detail="Memory engine not initialized")
    return engine


async def _get_session_rag_or_raise(session_id: Optional[str]):
    if not state.rag_instance or not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    try:
        config = state.rag_instance.config
        graphcore_kwargs = getattr(state.rag_instance, "graphcore_kwargs", {})
        session_rag = state.session_manager.get_session_rag(
            session_id, config, graphcore_kwargs
        )
        session_rag.llm_model_func = state.rag_instance.llm_model_func
        session_rag.embedding_func = state.rag_instance.embedding_func
        init_result = await session_rag._ensure_graphcore_initialized()
        if session_rag.graphcore is None:
            detail = (
                init_result.get("error")
                if isinstance(init_result, dict)
                else "unknown initialization failure"
            )
            raise RuntimeError(f"GraphCore initialization failed: {detail}")
        return session_rag
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {e}")


def _get_expanded_node_manager_or_raise(session_id: Optional[str]) -> ExpandedNodeManager:
    if not state.session_manager:
        raise HTTPException(status_code=500, detail="Session manager not initialized")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    session = state.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    metadata = session.get("metadata") or {}
    knowledge_dir = metadata.get("knowledge_dir") or session.get("knowledge_dir")
    if not knowledge_dir:
        raise HTTPException(status_code=500, detail="knowledge_dir not found in session metadata")

    if not hasattr(state, "expanded_node_managers") or state.expanded_node_managers is None:
        state.expanded_node_managers = {}
    if not hasattr(state, "expanded_node_lock") or state.expanded_node_lock is None:
        from threading import RLock

        state.expanded_node_lock = RLock()

    storage_path = Path(str(knowledge_dir)) / "expanded_nodes.json"
    lock = state.expanded_node_lock
    with lock:
        mgr = state.expanded_node_managers.get(session_id)
        if mgr is None:
            mgr = ExpandedNodeManager(storage_path=storage_path)
            state.expanded_node_managers[session_id] = mgr
        return mgr


def _pick_root_entity_ids(
    nodes_data: List[Dict[str, Any]],
    edges_data: List[Dict[str, Any]],
    *,
    limit: int = 6,
) -> List[str]:
    degree: Dict[str, int] = {}
    for edge in edges_data:
        src = str(edge.get("source") or "").strip()
        tgt = str(edge.get("target") or "").strip()
        if src:
            degree[src] = degree.get(src, 0) + 1
        if tgt:
            degree[tgt] = degree.get(tgt, 0) + 1

    candidates = []
    for n in nodes_data:
        entity = str(n.get("id") or n.get("entity_id") or "").strip()
        if not entity:
            continue
        is_expanded = str(n.get("source_id") or "").strip() == "llm_expansion" or str(
            n.get("is_expanded") or ""
        ).strip() in {"1", "true", "True"}
        if is_expanded:
            continue
        candidates.append((degree.get(entity, 0), entity))

    candidates.sort(key=lambda x: (-x[0], x[1]))
    return [entity for _, entity in candidates[: max(1, int(limit))]]


def _is_expanded_node(node_data: Dict[str, Any]) -> bool:
    ie = node_data.get("is_expanded")
    if ie is not None and ie != "":
        if ie == 1 or ie == "1" or str(ie).strip() == "1":
            return True
    return str(node_data.get("source_id") or "").strip() == "llm_expansion"


@router.post("/config")
async def update_config(payload: Dict[str, Any] = Body(...)):
    """Update system configuration"""
    if not state.rag_instance:
        raise HTTPException(status_code=500, detail="Service not initialized")
    
    config_type = payload.get("type")
    config_data = payload.get("data", {})
    
    try:
        if config_type == "kg":
            # Update Knowledge Graph configuration
            if "kg-storage" in config_data:
                state.rag_instance.config.knowledge_graph_storage_type = config_data["kg-storage"]
            
            if "kg-path" in config_data:
                state.rag_instance.config.knowledge_graph_path = config_data["kg-path"]
            
            if "entity-threshold" in config_data:
                state.rag_instance.config.entity_disambiguation_threshold = float(config_data["entity-threshold"])
            
            if "rel-threshold" in config_data:
                state.rag_instance.config.relationship_validation_threshold = float(config_data["rel-threshold"])
                
            if "enable-auto-validation" in config_data:
                state.rag_instance.config.enable_auto_validation = config_data["enable-auto-validation"] == "on"
            
            # New dual mode parameters
            if "graph-construction-mode" in config_data:
                state.rag_instance.config.graph_construction_mode = config_data["graph-construction-mode"]
                # Also update orchestrator if it exists
                if hasattr(state, "orchestrator") and state.orchestrator:
                    if hasattr(state.orchestrator, "hyper_system") and state.orchestrator.hyper_system:
                        state.orchestrator.hyper_system.graph_construction_mode = config_data["graph-construction-mode"]
            
            if "spacy-model" in config_data:
                state.rag_instance.config.spacy_model = config_data["spacy-model"]
                # Also update orchestrator if it exists
                if hasattr(state, "orchestrator") and state.orchestrator:
                    if hasattr(state.orchestrator, "hyper_system") and state.orchestrator.hyper_system:
                        state.orchestrator.hyper_system.spacy_model = config_data["spacy-model"]

            return {"success": True, "message": "Knowledge graph configuration updated"}
            
        elif config_type == "ui":
            # UI config might not be directly updateable in backend RAG instance
            # but we could store it if needed
            return {"success": True, "message": "UI configuration received (not all fields are persistent)"}
            
        elif config_type == "api":
            # API config might require restart to take effect
            return {"success": True, "message": "API configuration received (restart may be required)"}
            
        else:
            return {"success": False, "message": f"Unknown configuration type: {config_type}"}
            
    except Exception as e:
        return {"success": False, "message": f"Error updating configuration: {str(e)}"}


@router.get("/knowledge-graph/stats-all")
async def get_all_graph_stats():
    """Return node/edge counts for all session graphs."""
    if not state.rag_instance or not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")
    result: Dict[str, Any] = {"sessions": {}}
    for s in state.session_manager.list_sessions():
        sid = s.get("id")
        if not sid:
            continue
        try:
            config = state.rag_instance.config
            graphcore_kwargs = getattr(state.rag_instance, "graphcore_kwargs", {})
            session_rag = state.session_manager.get_session_rag(sid, config, graphcore_kwargs)
            session_rag.llm_model_func = state.rag_instance.llm_model_func
            session_rag.embedding_func = state.rag_instance.embedding_func
            await session_rag._ensure_graphcore_initialized()
            SG = session_rag.graphcore.chunk_entity_relation_graph
            snd = await SG.get_all_nodes()
            sed = await SG.get_all_edges()
            result["sessions"][sid] = {
                "nodes": len(snd),
                "edges": len(sed),
                "title": s.get("title", "unknown"),
            }
            if hasattr(SG, "_graphml_xml_file"):
                result["sessions"][sid]["path"] = str(getattr(SG, "_graphml_xml_file", ""))
        except Exception as e:
            result["sessions"][sid] = {"error": str(e), "title": s.get("title", "unknown")}
    return result


def _select_graph_nodes(
    nodes_data: List[Dict[str, Any]],
    edge_degree: Dict[str, int],
    *,
    scope: str = "summary",
    max_nodes: int = 200,
) -> List[Dict[str, Any]]:
    """Select graph nodes without changing the classic graph's default cap."""
    normalized_scope = str(scope or "summary").strip().lower()
    if normalized_scope == "full":
        return list(nodes_data)

    expanded_nodes = [n for n in nodes_data if _is_expanded_node(n)]
    other_nodes = sorted(
        [n for n in nodes_data if not _is_expanded_node(n)],
        key=lambda n: edge_degree.get(
            str(n.get("id") or n.get("entity_id") or ""), 0
        ),
        reverse=True,
    )
    budget = max(0, max_nodes - len(expanded_nodes))
    return expanded_nodes + other_nodes[:budget]


@router.get("/knowledge-graph/data")
async def get_graph_data(
    session_id: Optional[str] = None,
    scope: str = "summary",
):
    if not state.rag_instance or not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")

    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    try:
        config = state.rag_instance.config
        graphcore_kwargs = getattr(state.rag_instance, "graphcore_kwargs", {})
        session_rag = state.session_manager.get_session_rag(session_id, config, graphcore_kwargs)
        session_rag.llm_model_func = state.rag_instance.llm_model_func
        session_rag.embedding_func = state.rag_instance.embedding_func
        await session_rag._ensure_graphcore_initialized()
        target_rag = session_rag
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session graph not found: {e}")

    try:
        if not target_rag.graphcore:
            await target_rag._ensure_graphcore_initialized()
        G = target_rag.graphcore.chunk_entity_relation_graph
        nodes_data = await G.get_all_nodes()
        edges_data = await G.get_all_edges()

        nodes = []
        edges = []
        # Sort non-expanded nodes by degree (most connected first) so the
        # 200-node cap keeps the most important entities visible.
        edge_degree: Dict[str, int] = {}
        for e in edges_data:
            for k in ("source", "target"):
                nid = e.get(k, "")
                edge_degree[nid] = edge_degree.get(nid, 0) + 1

        nodes_to_use = _select_graph_nodes(
            nodes_data,
            edge_degree,
            scope=scope,
        )
        for node_info in nodes_to_use:
            node_id = node_info.get("id") or node_info.get("entity_id") or ""
            if not node_id:
                continue
            is_expanded = _is_expanded_node(node_info)
            nodes.append(
                {
                    "id": node_id,
                    "label": node_id,
                    "type": node_info.get("entity_type", "unknown"),
                    "description": node_info.get("description", ""),
                    "source_id": node_info.get("source_id", ""),
                    "file_path": node_info.get("file_path", ""),
                    "size": 20,
                    "color": resolve_graph_node_color(
                        node_info, is_expanded=is_expanded
                    ),
                    "is_expanded": is_expanded,
                    "is_image_node": is_image_node(node_info),
                    "degree": edge_degree.get(node_id, 0),
                }
            )

        node_ids = set(n["id"] for n in nodes)
        for edge_index, edge_info in enumerate(edges_data):
            for edge in _expand_graph_edge_records(edge_info, edge_index):
                if edge["source"] in node_ids and edge["target"] in node_ids:
                    edges.append(edge)

        expanded_in_response = sum(1 for x in nodes if x.get("is_expanded"))
        image_nodes_in_response = sum(1 for x in nodes if x.get("is_image_node"))
        discovered_edges_count = sum(1 for x in edges if x.get("is_discovered"))
        promoted_edges_count = sum(1 for x in edges if x.get("is_promoted"))
        meta = {
            "total_nodes": len(nodes_data),
            "total_edges": len(edges_data),
            "discovered_edges": discovered_edges_count,
            "promoted_edges": promoted_edges_count,
            "session_id": session_id,
            "nodes_returned": len(nodes),
            "edges_returned": len(edges),
            "truncated": len(nodes) < len(nodes_data),
            "expanded_in_response": expanded_in_response,
            "image_nodes_in_response": image_nodes_in_response,
        }
        if hasattr(G, "_graphml_xml_file"):
            meta["graph_file"] = str(getattr(G, "_graphml_xml_file", ""))
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": meta,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract graph data: {str(e)}")


def _split_chunk_source_ids(raw: Any) -> List[str]:
    seen = set()
    chunk_ids: List[str] = []
    for item in str(raw or "").split("<SEP>"):
        chunk_id = item.strip()
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            chunk_ids.append(chunk_id)
    return chunk_ids


def _limit_chunk_source_ids(source_ids: List[str], max_chunks: int) -> List[str]:
    """Use zero for all chunks while preserving the existing positive cap."""
    requested_max = int(max_chunks)
    if requested_max <= 0:
        return list(source_ids)
    return list(source_ids[: min(requested_max, 80)])


def _json_list(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [dict(item) for item in parsed if isinstance(item, dict)]


def _is_eclrr_v4_promoted(edge: Dict[str, Any]) -> bool:
    return (
        str(edge.get("review_status") or "").strip().lower() == "promoted"
        and str(edge.get("provenance") or "").strip().lower() == "eclrr_v4"
        and str(edge.get("algorithm_version") or "").strip().lower() == "eclrr_v4"
    )


def _graph_edge_response(
    edge: Dict[str, Any],
    source: str,
    target: str,
    *,
    fallback_id: str,
) -> Dict[str, Any]:
    promoted = _is_eclrr_v4_promoted(edge)
    discovered = str(edge.get("is_discovered", "0")) in {"1", "true", "True"}
    relation_id = str(
        edge.get("relation_id") or edge.get("canonical_key") or ""
    ).strip()
    return {
        "id": relation_id or fallback_id,
        "source": str(edge.get("source") or edge.get("src_id") or source),
        "target": str(edge.get("target") or edge.get("tgt_id") or target),
        "label": edge.get("relation") or edge.get("keywords") or edge.get("label") or "related",
        "relation": edge.get("relation") or edge.get("keywords") or edge.get("label") or "related",
        "relation_family": edge.get("relation_family", ""),
        "direction": edge.get("direction", ""),
        "description": edge.get("description", ""),
        "weight": edge.get("weight", 1.0),
        "source_id": edge.get("source_id", ""),
        "is_discovered": discovered,
        "is_promoted": promoted,
        "edge_kind": "eclrr_v4" if promoted else "original",
        "color": "#ef4444" if discovered else "#95a5a6",
        "width": 2 if discovered else 1,
        "review_status": edge.get("review_status", ""),
        "query_eligible": edge.get("query_eligible", ""),
        "provenance": edge.get("provenance", ""),
        "algorithm_version": edge.get("algorithm_version", ""),
        "relation_id": relation_id,
        "canonical_key": edge.get("canonical_key", ""),
        "path_used": edge.get("path_used", ""),
        "supporting_paths": edge.get("supporting_paths", ""),
        "evidence_chain": edge.get("evidence_chain", ""),
        "evidence_chunk_ids": edge.get("evidence_chunk_ids", ""),
        "judge_scores": edge.get("judge_scores", ""),
        "decision_score": edge.get("decision_score", ""),
    }


def _expand_graph_edge_records(edge: Dict[str, Any], index: int) -> List[Dict[str, Any]]:
    """Expose one physical fact edge plus any independently selectable ECLRR relations."""
    source = str(edge.get("source") or edge.get("src_id") or "").strip()
    target = str(edge.get("target") or edge.get("tgt_id") or "").strip()
    if not source or not target:
        return []

    physical = _graph_edge_response(
        edge,
        source,
        target,
        fallback_id=f"edge-{index}-{source}-{target}",
    )
    records = [physical]
    physical_identity = str(edge.get("relation_id") or edge.get("canonical_key") or "")
    for nested_index, relation in enumerate(_json_list(edge.get("eclrr_relations"))):
        if not _is_eclrr_v4_promoted(relation):
            continue
        nested_identity = str(relation.get("relation_id") or relation.get("canonical_key") or "")
        if nested_identity and nested_identity == physical_identity:
            continue
        records.append(
            _graph_edge_response(
                relation,
                source,
                target,
                fallback_id=f"edge-{index}-eclrr-{nested_index}",
            )
        )
    return records


async def _load_source_chunks(
    graphcore: Any,
    source_id: Any,
    max_chunks: int,
) -> tuple[List[str], List[Dict[str, Any]]]:
    source_ids = _limit_chunk_source_ids(_split_chunk_source_ids(source_id), max_chunks)
    if not source_ids or not getattr(graphcore, "text_chunks", None):
        return source_ids, []

    chunk_data_list = await graphcore.text_chunks.get_by_ids(source_ids)
    chunks: List[Dict[str, Any]] = []
    for chunk_id, data in zip(source_ids, chunk_data_list):
        if not data:
            chunks.append({"chunk_id": chunk_id, "content": "", "missing": True})
            continue
        chunks.append(
            {
                "chunk_id": chunk_id,
                "content": str(data.get("content") or data.get("text") or ""),
                "file_path": data.get("file_path") or data.get("full_doc_id") or "",
                "tokens": data.get("tokens"),
                "chunk_order_index": data.get("chunk_order_index"),
            }
        )
    return source_ids, chunks


@router.get("/knowledge-graph/entity-chunks")
async def get_entity_chunks(
    session_id: Optional[str] = None,
    entity_id: Optional[str] = None,
    max_chunks: int = 20,
):
    """Return source chunk ids and chunk text for one graph entity."""
    session_rag = await _get_session_rag_or_raise(session_id)
    if not entity_id:
        raise HTTPException(status_code=400, detail="entity_id is required")

    try:
        if not session_rag.graphcore:
            await session_rag._ensure_graphcore_initialized()
        graphcore = session_rag.graphcore
        graph = graphcore.chunk_entity_relation_graph
        node_info = await graph.get_node(entity_id)
        if not node_info:
            return {"entity_id": entity_id, "source_ids": [], "chunks": []}

        source_ids, chunks = await _load_source_chunks(
            graphcore,
            node_info.get("source_id", ""),
            max_chunks,
        )
        return {"entity_id": entity_id, "source_ids": source_ids, "chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load entity chunks: {str(e)}")


@router.get("/knowledge-graph/edge-chunks")
async def get_edge_chunks(
    session_id: Optional[str] = None,
    source_id: Optional[str] = None,
    edge_id: Optional[str] = None,
    max_chunks: int = 20,
):
    """Return the exact source chunks carried by one displayed graph relation."""
    session_rag = await _get_session_rag_or_raise(session_id)
    if source_id is None:
        raise HTTPException(status_code=400, detail="source_id is required")
    try:
        if not session_rag.graphcore:
            await session_rag._ensure_graphcore_initialized()
        source_ids, chunks = await _load_source_chunks(
            session_rag.graphcore,
            source_id,
            max_chunks,
        )
        return {"edge_id": edge_id or "", "source_ids": source_ids, "chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load edge chunks: {str(e)}")


async def _run_eclrr_for_session(session_id: str, max_new_edges: int) -> None:
    status = _ECLRR_RUN_STATUS[session_id]
    try:
        session_rag = await _get_session_rag_or_raise(session_id)
        graphcore = session_rag.graphcore
        if graphcore is None:
            await session_rag._ensure_graphcore_initialized()
            graphcore = session_rag.graphcore
        if graphcore is None:
            raise RuntimeError("GraphCore initialization failed")
        graph = graphcore.chunk_entity_relation_graph
        nodes = await graph.get_all_nodes()
        if len(nodes) < 4:
            raise RuntimeError("At least four graph nodes are required")
        llm_fn = getattr(session_rag, "llm_model_func", None)
        if not llm_fn:
            raise RuntimeError("LLM function is not configured")

        session = state.session_manager.get_session(session_id) if state.session_manager else None
        metadata = (session or {}).get("metadata") or {}
        knowledge_dir = metadata.get("knowledge_dir") or (session or {}).get("knowledge_dir")
        from docthinker.kg_expansion.eclrr_v4 import ECLRRConfig, run_eclrr_v4

        result = await run_eclrr_v4(
            graph=graph,
            text_chunks=graphcore.text_chunks,
            generator_func=llm_fn,
            judge_func=llm_fn,
            relationships_vdb=graphcore.relationships_vdb,
            config=ECLRRConfig(
                max_review_items=min(128, max(60, max_new_edges * 4)),
                max_promotions=max_new_edges,
                artifact_dir=(
                    str(Path(knowledge_dir) / "eclrr_v4_runs")
                    if knowledge_dir
                    else None
                ),
            ),
        )
        status.update(
            {
                "status": "completed",
                "finished_at": _utc_now(),
                "reviewed": len(result.review_items),
                "proposed": len(result.proposals),
                "accepted": len(result.committed),
                "rejected": result.metrics.get("no_op", 0),
                "create": result.metrics.get("create", 0),
                "refine": result.metrics.get("refine", 0),
                "write_failures": len(result.metrics.get("write_failures", {})),
                "relations": [
                    {
                        "id": item.relation_id,
                        "source": item.source,
                        "target": item.target,
                        "relation": item.relation,
                        "action": item.action,
                    }
                    for item in result.committed
                ],
            }
        )
    except Exception as exc:
        _log.exception("ECLRR-v4 manual run failed for session %s", session_id)
        status.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        _ECLRR_RUN_TASKS.pop(session_id, None)


@router.post("/knowledge-graph/eclrr-v4/run")
async def start_eclrr_v4(payload: Dict[str, Any] = Body(default={})):
    session_id = str(payload.get("session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    try:
        requested_limit = int(payload.get("max_new_edges", 40))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="max_new_edges must be an integer")
    max_new_edges = min(40, max(1, requested_limit))

    running = _ECLRR_RUN_TASKS.get(session_id)
    if running and not running.done():
        return _ECLRR_RUN_STATUS[session_id]

    await _get_session_rag_or_raise(session_id)
    status = {
        "session_id": session_id,
        "status": "running",
        "max_new_edges": max_new_edges,
        "started_at": _utc_now(),
        "reviewed": 0,
        "proposed": 0,
        "accepted": 0,
        "rejected": 0,
    }
    _ECLRR_RUN_STATUS[session_id] = status
    _ECLRR_RUN_TASKS[session_id] = asyncio.create_task(
        _run_eclrr_for_session(session_id, max_new_edges)
    )
    return status


@router.get("/knowledge-graph/eclrr-v4/status")
async def get_eclrr_v4_status(session_id: Optional[str] = None):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    return _ECLRR_RUN_STATUS.get(
        session_id,
        {
            "session_id": session_id,
            "status": "idle",
            "max_new_edges": 40,
            "reviewed": 0,
            "proposed": 0,
            "accepted": 0,
            "rejected": 0,
        },
    )


@router.get("/knowledge-graph/image/{session_id}/{filename}")
async def serve_image_node(session_id: str, filename: str):
    """Serve an image file from a session's multimodal/images directory."""
    if not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")

    session = state.session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    knowledge_dir = session.get("knowledge_dir") or ""
    if not knowledge_dir:
        raise HTTPException(status_code=404, detail="Knowledge dir not found")

    safe_name = Path(filename).name
    img_path = Path(knowledge_dir) / "multimodal" / "images" / safe_name

    if not img_path.is_file():
        raise HTTPException(status_code=404, detail=f"Image file not found: {safe_name}")

    _MIME = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
    }
    mime = _MIME.get(img_path.suffix.lower(), "image/png")
    return FileResponse(str(img_path), media_type=mime)


@router.post("/knowledge-graph/expand")
async def expand_knowledge_graph(payload: Dict[str, Any] = Body(default={})):
    """Expand a session KG via two-part pipeline: cluster-based + top-node expansion.

    Loads pre-computed cluster summaries (from ingest), runs parallel LLM
    expansion with self-validation, writes nodes + edges to graph + VDB.
    """
    session_id = payload.get("session_id")
    apply = payload.get("apply", True)
    root_entity_ids = payload.get("root_entity_ids")
    if not state.rag_instance or not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    try:
        config = state.rag_instance.config
        graphcore_kwargs = getattr(state.rag_instance, "graphcore_kwargs", {})
        session_rag = state.session_manager.get_session_rag(
            session_id, config, graphcore_kwargs
        )
        session_rag.llm_model_func = state.rag_instance.llm_model_func
        session_rag.embedding_func = state.rag_instance.embedding_func
        await session_rag._ensure_graphcore_initialized()
        target_rag = session_rag
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {e}")
    try:
        if not target_rag.graphcore:
            await target_rag._ensure_graphcore_initialized()
        G = target_rag.graphcore.chunk_entity_relation_graph
        nodes_data = await G.get_all_nodes()
        edges_data = await G.get_all_edges()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get graph: {e}")

    llm_fn = getattr(target_rag, "llm_model_func", None)
    if not llm_fn:
        raise HTTPException(status_code=500, detail="LLM not available")
    embed_fn = getattr(target_rag, "embedding_func", None)
    if embed_fn and hasattr(embed_fn, "func"):
        embed_fn = embed_fn.func

    try:
        from docthinker.kg_expansion import KGExpander, load_cluster_summaries
        from docthinker.hypergraph.utils import compute_mdhash_id

        # Load cluster summaries generated during ingestion
        cluster_summaries = load_cluster_summaries(
            Path(target_rag.graphcore.working_dir) / "cluster_summaries.json"
        )
        _log.info("[expand] loaded %d cluster summaries for session %s",
                  len(cluster_summaries), session_id)

        expander = KGExpander(
            llm_func=llm_fn,
            embedding_func=embed_fn,
            min_per_cluster=8,
            min_per_topnode=15,
            semantic_dedup_threshold=0.92,
            enable_validation=True,
            validation_min_score=0.6,
            session_id=session_id,
        )
        result = await expander.expand(
            nodes_data,
            edges_data,
            cluster_summaries=cluster_summaries if cluster_summaries else None,
            apply_to_graph=G if apply else None,
        )

        # ── Upsert expanded nodes/edges into VDB for retrieval ───────
        added = result.get("added_nodes") or []
        added_edges = result.get("added_edges") or []
        gc = target_rag.graphcore
        if apply and gc and (added or added_edges):
            entity_vdb_data = {}
            for ent in added:
                name = str(ent.get("entity") or "").strip()
                desc = str(ent.get("description") or name)
                etype = str(ent.get("entity_type") or "concept")
                entity_vdb_data[compute_mdhash_id(name, prefix="ent-")] = {
                    "content": f"{name}\n{desc}",
                    "entity_name": name,
                    "source_id": "llm_expansion",
                    "description": desc,
                    "entity_type": etype,
                    "file_path": "llm_expansion",
                }
            if entity_vdb_data:
                try:
                    await gc.entities_vdb.upsert(entity_vdb_data)
                    _log.info("[expand] upserted %d expanded entities into VDB", len(entity_vdb_data))
                except Exception as vdb_exc:
                    _log.warning("[expand] entities_vdb upsert failed: %s", vdb_exc)

            rel_vdb_data = {}
            for edge in added_edges:
                src = str(edge.get("source") or "").strip()
                tgt = str(edge.get("target") or "").strip()
                rel = str(edge.get("relation") or "related")
                edesc = str(edge.get("description") or "")
                if src and tgt:
                    rel_vdb_data[compute_mdhash_id(src + tgt, prefix="rel-")] = {
                        "src_id": src,
                        "tgt_id": tgt,
                        "source_id": "llm_expansion",
                        "content": f"{rel}\t{src}\n{tgt}\n{edesc}",
                        "keywords": rel,
                        "description": edesc,
                    }
            if rel_vdb_data:
                try:
                    await gc.relationships_vdb.upsert(rel_vdb_data)
                    _log.info("[expand] upserted %d expanded relations into VDB", len(rel_vdb_data))
                except Exception as vdb_exc:
                    _log.warning("[expand] relationships_vdb upsert failed: %s", vdb_exc)

        # ── Update lifecycle manager ──
        manager = _get_expanded_node_manager_or_raise(session_id)
        if isinstance(root_entity_ids, list) and root_entity_ids:
            root_ids = [str(x).strip() for x in root_entity_ids if str(x).strip()]
        else:
            root_ids = _pick_root_entity_ids(nodes_data, edges_data)

        lifecycle = manager.upsert_candidates(
            added,
            default_root_ids=root_ids,
            source="llm_expansion",
        )

        return {
            "success": True,
            "added_nodes": len(added),
            "added_edges": len(added_edges),
            "raw_count": result.get("raw_count", 0),
            "validated_count": result.get("validated_count", 0),
            "cluster_count": result.get("cluster_count", 0),
            "top_node_count": result.get("top_node_count", 0),
            "root_entity_ids": root_ids,
            "lifecycle": lifecycle,
        }
    except Exception as e:
        err = str(e)
        _log.error("[expand] expansion failed: %s", err, exc_info=True)
        raise HTTPException(status_code=500, detail=err or "Expansion failed")


@router.get("/knowledge-graph/debug-expanded")
async def debug_expanded_nodes(session_id: Optional[str] = None):
    """Return diagnostics for expanded nodes in a session graph."""
    if not state.rag_instance or not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    try:
        config = state.rag_instance.config
        graphcore_kwargs = getattr(state.rag_instance, "graphcore_kwargs", {})
        session_rag = state.session_manager.get_session_rag(
            session_id, config, graphcore_kwargs
        )
        session_rag.llm_model_func = state.rag_instance.llm_model_func
        session_rag.embedding_func = state.rag_instance.embedding_func
        await session_rag._ensure_graphcore_initialized()
        target_rag = session_rag
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {e}")

    try:
        if not target_rag.graphcore:
            await target_rag._ensure_graphcore_initialized()
        G = target_rag.graphcore.chunk_entity_relation_graph
        nodes_data = await G.get_all_nodes()

        expanded = [
            {"id": n.get("id") or n.get("entity_id"), "is_expanded": n.get("is_expanded")}
            for n in nodes_data
            if _is_expanded_node(n)
        ]
        total = len(nodes_data)
        storage_info = {}
        if hasattr(G, "_graphml_xml_file"):
            storage_info["graph_file"] = getattr(G, "_graphml_xml_file", "N/A")

        return {
            "expanded_count": len(expanded),
            "total_nodes": total,
            "expanded_sample": expanded[:20],
            "storage_info": storage_info,
            "session_id": session_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-graph/stats")
async def get_knowledge_graph_stats(session_id: Optional[str] = None):
    target_rag = await _get_session_rag_or_raise(session_id)
    try:
        G = target_rag.graphcore.chunk_entity_relation_graph
        nodes_data = await G.get_all_nodes()
        edges_data = await G.get_all_edges()
        entity_types = sorted(
            {
                str(n.get("entity_type") or "unknown")
                for n in nodes_data
            }
        )
        relationship_types = sorted(
            {
                str(e.get("keywords") or e.get("description") or "related")
                for e in edges_data
            }
        )
        return {
            "session_id": session_id,
            "total_entities": len(nodes_data),
            "total_relationships": len(edges_data),
            "entity_types": entity_types,
            "relationship_types": relationship_types,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-graph/expanded-nodes")
async def list_expanded_nodes(
    session_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 200,
):
    manager = _get_expanded_node_manager_or_raise(session_id)
    nodes = manager.list_nodes(status=status, limit=limit)
    return {
        "session_id": session_id,
        "status": status,
        "count": len(nodes),
        "nodes": nodes,
    }


@router.post("/knowledge-graph/expanded-nodes/match")
async def match_expanded_nodes(payload: Dict[str, Any] = Body(default={})):
    session_id = payload.get("session_id")
    query = str(payload.get("query") or "").strip()
    top_k = int(payload.get("top_k") or 2)
    memory_terms = payload.get("memory_terms") or []
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    manager = _get_expanded_node_manager_or_raise(session_id)
    matches = manager.match_nodes(
        query=query,
        top_k=max(1, top_k),
        memory_terms=memory_terms if isinstance(memory_terms, list) else [],
    )
    if matches:
        manager.mark_hits([m.get("entity", "") for m in matches])
    instruction = manager.build_forced_instruction(matches, limit=min(2, max(1, top_k)))
    return {
        "session_id": session_id,
        "query": query,
        "count": len(matches),
        "matches": matches,
        "instruction": instruction,
    }


@router.post("/knowledge-graph/entity")
async def add_entity(request: EntityRelationshipRequest):
    target_rag = await _get_session_rag_or_raise(request.session_id)
    try:
        G = target_rag.graphcore.chunk_entity_relation_graph
        props = dict(request.properties or {})
        props.setdefault("entity_type", request.entity_type)
        props.setdefault("source_id", request.document_id)
        await G.upsert_node(request.entity_name, props)
        await G.index_done_callback()
        return {
            "status": "success",
            "entity": {
                "id": request.entity_name,
                "name": request.entity_name,
                "type": props.get("entity_type", request.entity_type),
                "properties": props,
                "document_ids": [request.document_id] if request.document_id else [],
                "session_id": request.session_id,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge-graph/relationship")
async def add_relationship(request: RelationshipRequest):
    target_rag = await _get_session_rag_or_raise(request.session_id)
    try:
        G = target_rag.graphcore.chunk_entity_relation_graph
        if not await G.has_node(request.source_entity) or not await G.has_node(request.target_entity):
            raise HTTPException(status_code=404, detail="Source or target entity not found")
        props = dict(request.properties or {})
        props.setdefault("keywords", request.relationship_type)
        props.setdefault("description", request.relationship_type)
        props.setdefault("source_id", request.document_id)
        await G.upsert_edge(request.source_entity, request.target_entity, props)
        await G.index_done_callback()
        return {
            "status": "success",
            "relationship": {
                "id": f"{request.source_entity}-{request.target_entity}",
                "source_id": request.source_entity,
                "target_id": request.target_entity,
                "type": request.relationship_type,
                "properties": props,
                "document_ids": [request.document_id] if request.document_id else [],
                "session_id": request.session_id,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/knowledge-graph/entity/{entity_name}")
async def update_entity(entity_name: str, properties: Dict[str, Any], session_id: Optional[str] = None):
    target_rag = await _get_session_rag_or_raise(session_id)
    try:
        G = target_rag.graphcore.chunk_entity_relation_graph
        if await G.has_node(entity_name):
            await G.upsert_node(entity_name, properties)
            await G.index_done_callback()
            return {"status": "success", "message": f"Entity {entity_name} updated", "session_id": session_id}
        raise HTTPException(status_code=404, detail="Entity not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/knowledge-graph/relationship")
async def delete_relationship(source: str, target: str, session_id: Optional[str] = None):
    target_rag = await _get_session_rag_or_raise(session_id)
    try:
        G = target_rag.graphcore.chunk_entity_relation_graph
        if await G.has_edge(source, target):
            await G.remove_edges([(source, target)])
            await G.index_done_callback()
            return {"status": "success", "message": f"Relationship {source}->{target} deleted", "session_id": session_id}
        raise HTTPException(status_code=404, detail="Relationship not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/stats")
async def memory_stats(session_id: Optional[str] = None):
    """Memory system status — returns claw tiered memory stats."""
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    from ..memory import get_session_claw_manager
    claw_mgr = get_session_claw_manager(session_id)
    if claw_mgr:
        return {
            "enabled": True,
            "system": "claw",
            "session_id": session_id,
            **claw_mgr.get_stats(),
        }

    try:
        engine = _get_memory_engine_or_raise(session_id)
        episodes = engine.episode_store.all_episodes()
        edges = engine.graph.get_all_edges()
        return {
            "enabled": True,
            "system": "neuro_memory",
            "session_id": session_id,
            "episodes": len(episodes),
            "edges": len(edges),
        }
    except Exception as e:
        return {"enabled": False, "session_id": session_id, "error": str(e)}


@router.get("/memory/dashboard")
async def memory_dashboard(session_id: Optional[str] = None):
    """Aggregated dashboard state for KG + memory visualization."""
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    kg_stats: Dict[str, Any]
    try:
        kg_stats = await get_knowledge_graph_stats(session_id=session_id)
    except Exception as exc:
        kg_stats = {"error": str(exc), "total_entities": 0, "total_relationships": 0}

    expanded_payload: Dict[str, Any]
    try:
        manager = _get_expanded_node_manager_or_raise(session_id)
        expanded_nodes = manager.list_nodes(limit=100)
        lifecycle = {"candidate": 0, "active": 0, "promoted": 0, "deprecated": 0}
        for item in expanded_nodes:
            status = str(item.get("status") or "candidate")
            lifecycle[status] = lifecycle.get(status, 0) + 1
        expanded_payload = {
            "count": len(expanded_nodes),
            "lifecycle": lifecycle,
            "nodes": expanded_nodes,
        }
    except Exception as exc:
        expanded_payload = {"count": 0, "lifecycle": {}, "nodes": [], "error": str(exc)}

    memory_payload = await memory_stats(session_id=session_id)
    long_horizon_payload = get_default_long_horizon_backend().stats(session_id)
    return {
        "session_id": session_id,
        "kg": kg_stats,
        "expanded": expanded_payload,
        "memory": memory_payload,
        "long_horizon": long_horizon_payload,
    }


@router.get("/memory/long-horizon")
async def list_long_horizon_memory(
    session_id: Optional[str] = None,
    scope: Optional[str] = None,
    limit: int = 50,
):
    """List editable long-horizon memories for audit and management."""
    backend = get_default_long_horizon_backend()
    return {
        "session_id": session_id,
        "scope": scope,
        "items": backend.list_insights(session_id=session_id, scope=scope, limit=limit),
        "last_write_decision": backend.last_write_decision(),
    }


@router.delete("/memory/long-horizon/{memory_id}")
async def delete_long_horizon_memory(memory_id: str, session_id: Optional[str] = None):
    """Delete one long-horizon memory record."""
    deleted = get_default_long_horizon_backend().delete_insight(
        memory_id,
        session_id=session_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Memory not found: {memory_id}")
    return {"deleted": True, "memory_id": memory_id}


@router.patch("/memory/long-horizon/{memory_id}")
async def update_long_horizon_memory(
    memory_id: str,
    payload: Dict[str, Any] = Body(default={}),
    session_id: Optional[str] = None,
):
    """Update one long-horizon memory record after user confirmation."""
    updated = get_default_long_horizon_backend().update_insight(
        memory_id,
        payload,
        session_id=session_id,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Memory not found or patch empty: {memory_id}")
    return {"updated": True, "memory_id": memory_id, "item": updated}


@router.post("/memory/long-horizon/edit-plan")
async def plan_long_horizon_memory_edit(payload: Dict[str, Any] = Body(default={})):
    """Map a natural-language memory edit command to editable candidates."""
    instruction = str(payload.get("instruction") or "").strip()
    if not instruction:
        raise HTTPException(status_code=400, detail="instruction is required")
    session_id = payload.get("session_id")
    scope = payload.get("scope")
    limit = int(payload.get("limit") or 5)
    return get_default_long_horizon_backend().plan_edit(
        session_id=session_id,
        instruction=instruction,
        scope=scope,
        limit=limit,
    )


@router.get("/memory/long-horizon/export")
async def export_long_horizon_memory(session_id: Optional[str] = None):
    """Export long-horizon memory as a MEMORY.md-style index."""
    markdown = get_default_long_horizon_backend().export_markdown(session_id=session_id)
    return Response(
        content=markdown,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": "inline; filename=MEMORY.md"},
    )

# ── LLM Trace observability endpoints ──────────────────────────────

@router.get("/traces")
async def get_traces(
    session_id: Optional[str] = None,
    stage: Optional[str] = None,
    limit: int = 50,
):
    """Return recent LLM call traces for a session (preview mode)."""
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    from docthinker.llm_trace import list_traces
    traces = list_traces(session_id, stage=stage, limit=limit)
    return {"session_id": session_id, "count": len(traces), "traces": traces}


@router.get("/traces/{call_id}")
async def get_trace_detail_endpoint(
    call_id: str,
    session_id: Optional[str] = None,
):
    """Return full trace record including complete prompt and response."""
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    from docthinker.llm_trace import get_trace_detail
    record = get_trace_detail(call_id, session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Trace {call_id} not found")
    return record
