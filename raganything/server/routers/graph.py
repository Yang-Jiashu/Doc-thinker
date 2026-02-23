from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Body

from ..schemas import EntityRelationshipRequest, RelationshipRequest
from ..state import state


router = APIRouter()


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


@router.get("/knowledge-graph/data")
async def get_graph_data(session_id: Optional[str] = None):
    if not state.rag_instance or not state.session_manager:
        raise HTTPException(status_code=500, detail="Service not initialized")

    target_rag = state.rag_instance
    if session_id:
        try:
            config = state.rag_instance.config
            session_rag = state.session_manager.get_session_rag(session_id, config)
            session_rag.llm_model_func = state.rag_instance.llm_model_func
            session_rag.embedding_func = state.rag_instance.embedding_func
            await session_rag._ensure_lightrag_initialized()
            target_rag = session_rag
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Session graph not found: {e}")

    try:
        if not target_rag.lightrag:
            await target_rag._ensure_lightrag_initialized()
        G = target_rag.lightrag.chunk_entity_relation_graph
        nodes_data = await G.get_all_nodes()
        edges_data = await G.get_all_edges()

        nodes = []
        edges = []
        max_nodes = 200

        for i, node_info in enumerate(nodes_data):
            if i >= max_nodes:
                break
            node_id = node_info["id"]
            nodes.append(
                {
                    "id": node_id,
                    "label": node_id,
                    "type": node_info.get("entity_type", "unknown"),
                    "size": 20,
                    "color": "#3498db",
                }
            )

        node_ids = set(n["id"] for n in nodes)
        for edge_info in edges_data:
            u = edge_info["source"]
            v = edge_info["target"]
            if u in node_ids and v in node_ids:
                edges.append(
                    {
                        "id": f"{u}-{v}",
                        "source": u,
                        "target": v,
                        "label": edge_info.get("keywords", "related"),
                        "color": "#95a5a6",
                        "width": 1,
                    }
                )

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_nodes": len(nodes_data),
                "total_edges": len(edges_data),
                "session_id": session_id,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract graph data: {str(e)}")


@router.get("/knowledge-graph/stats")
async def get_knowledge_graph_stats():
    if not state.rag_instance:
        raise HTTPException(status_code=500, detail="Service not initialized")
    try:
        kg = state.rag_instance.knowledge_graph
        return {
            "total_entities": len(kg.entities),
            "total_relationships": len(kg.relationships),
            "entity_types": list(set(entity.type for entity in kg.entities.values())),
            "relationship_types": list(set(rel.type for rel in kg.relationships.values())),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge-graph/entity")
async def add_entity(request: EntityRelationshipRequest):
    if not state.rag_instance:
        raise HTTPException(status_code=500, detail="Service not initialized")
    try:
        entity = state.rag_instance.knowledge_graph.add_entity(
            name=request.entity_name,
            type=request.entity_type,
            properties=request.properties,
            document_id=request.document_id,
        )
        state.rag_instance.knowledge_graph.save(str(state.rag_instance.graph_path))
        return {
            "status": "success",
            "entity": {
                "id": entity.id,
                "name": entity.name,
                "type": entity.type,
                "properties": entity.properties,
                "document_ids": list(entity.document_ids),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge-graph/relationship")
async def add_relationship(request: RelationshipRequest):
    if not state.rag_instance:
        raise HTTPException(status_code=500, detail="Service not initialized")
    try:
        source_entity = state.rag_instance.knowledge_graph.get_entity_by_name(request.source_entity)
        target_entity = state.rag_instance.knowledge_graph.get_entity_by_name(request.target_entity)
        if not source_entity or not target_entity:
            raise HTTPException(status_code=404, detail="Source or target entity not found")

        relationship = state.rag_instance.knowledge_graph.add_relationship(
            source_id=source_entity.id,
            target_id=target_entity.id,
            type=request.relationship_type,
            properties=request.properties,
            document_id=request.document_id,
        )
        state.rag_instance.knowledge_graph.save(str(state.rag_instance.graph_path))
        return {
            "status": "success",
            "relationship": {
                "id": relationship.id,
                "source_id": relationship.source_id,
                "target_id": relationship.target_id,
                "type": relationship.type,
                "properties": relationship.properties,
                "document_ids": list(relationship.document_ids),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/knowledge-graph/entity/{entity_name}")
async def update_entity(entity_name: str, properties: Dict[str, Any]):
    if not state.rag_instance:
        raise HTTPException(status_code=500, detail="Service not initialized")
    try:
        if not state.rag_instance.lightrag:
            await state.rag_instance._ensure_lightrag_initialized()
        G = state.rag_instance.lightrag.chunk_entity_relation_graph
        if await G.has_node(entity_name):
            await G.upsert_node(entity_name, properties)
            await G.index_done_callback()
            return {"status": "success", "message": f"Entity {entity_name} updated"}
        raise HTTPException(status_code=404, detail="Entity not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/knowledge-graph/relationship")
async def delete_relationship(source: str, target: str):
    if not state.rag_instance:
        raise HTTPException(status_code=500, detail="Service not initialized")
    try:
        if not state.rag_instance.lightrag:
            await state.rag_instance._ensure_lightrag_initialized()
        G = state.rag_instance.lightrag.chunk_entity_relation_graph
        if await G.has_edge(source, target):
            await G.remove_edges([(source, target)])
            await G.index_done_callback()
            return {"status": "success", "message": f"Relationship {source}->{target} deleted"}
        raise HTTPException(status_code=404, detail="Relationship not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/stats")
async def memory_stats():
    """类人脑记忆引擎状态：episode 数量、图边数等。"""
    if not getattr(state, "memory_engine", None) or state.memory_engine is None:
        return {"enabled": False, "episodes": 0, "edges": 0}
    try:
        episodes = state.memory_engine.episode_store.all_episodes()
        edges = state.memory_engine.graph.get_all_edges()
        return {
            "enabled": True,
            "episodes": len(episodes),
            "edges": len(edges),
        }
    except Exception as e:
        return {"enabled": True, "error": str(e)}


@router.get("/memory/graph-data")
async def memory_graph_data():
    """记忆联想图可视化数据：节点（episode/entity/chunk）+ 边（联想类型），与 /knowledge-graph/data 同结构便于前端复用。"""
    if not getattr(state, "memory_engine", None) or state.memory_engine is None:
        return {"nodes": [], "edges": [], "metadata": {"source": "memory", "enabled": False}}
    try:
        graph = state.memory_engine.graph
        episodes = state.memory_engine.episode_store.all_episodes()
        nodes = []
        edges = []
        # 颜色：episode=蓝，entity=绿，chunk=橙
        type_color = {"episode": "#3498db", "entity": "#2ecc71", "chunk": "#e67e22"}
        type_label = {"episode": "经历", "entity": "实体", "chunk": "片段"}
        for nid, nd in graph.get_all_nodes():
            ntype = nd.get("type", "episode")
            label = nid
            if ntype == "episode" and nid in episodes:
                summary = (episodes[nid].summary or nid)[:30]
                label = summary + "…" if len(episodes[nid].summary or "") > 30 else (episodes[nid].summary or nid)
            nodes.append({
                "id": nid,
                "label": label,
                "type": type_label.get(ntype, ntype),
                "size": 20,
                "color": type_color.get(ntype, "#95a5a6"),
            })
        for e in graph.get_all_edges():
            edges.append({
                "id": f"{e.source_id}-{e.edge_type.value}-{e.target_id}",
                "source": e.source_id,
                "target": e.target_id,
                "label": e.edge_type.value,
                "type": e.edge_type.value,
                "color": "#9b59b6",
                "width": max(1, int(e.weight * 3)),
            })
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "source": "memory",
                "enabled": True,
                "total_nodes": len(nodes),
                "total_edges": len(edges),
            },
        }
    except Exception as e:
        return {"nodes": [], "edges": [], "metadata": {"source": "memory", "enabled": True, "error": str(e)}}


@router.post("/memory/consolidate")
async def memory_consolidate(recent_n: int = 50, run_llm: bool = True):
    """触发一次记忆巩固（重放、跨事件联想、权重更新）。"""
    if not getattr(state, "memory_engine", None) or state.memory_engine is None:
        raise HTTPException(status_code=501, detail="Memory engine not initialized")
    try:
        result = await state.memory_engine.consolidate(
            recent_n=recent_n,
            run_llm=run_llm,
        )
        state.memory_engine.save()
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

