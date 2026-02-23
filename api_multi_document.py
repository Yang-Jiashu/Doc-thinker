"""
API service for multi-document enhanced RAG system

This FastAPI service demonstrates the new multi-document query enhancement feature
that leverages the knowledge graph to find related documents and provide more
comprehensive answers.
"""

import os
import asyncio
import logging
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import shutil
import tempfile

from raganything import RAGAnything, RAGAnythingConfig
from raganything.session_manager import SessionManager
from raganything.cognitive import CognitiveProcessor
from raganything.providers import load_settings, get_embed_client, get_vlm_client
from raganything.services import IngestionService
from lightrag.utils import EmbeddingFunc
import numpy as np

settings = load_settings()

WORKDIR = settings.workdir

logging.basicConfig(level=os.getenv("LOG_LEVEL") or "INFO")
logger = logging.getLogger("rag_api")

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Document Enhanced RAG API",
    description="API service for multi-document enhanced RAG system using knowledge graph",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Session Manager
session_manager = SessionManager(base_storage_path=WORKDIR)

# Define request/response models
class CreateSessionRequest(BaseModel):
    title: Optional[str] = None

class QueryRequest(BaseModel):
    question: str
    mode: str = "hybrid"
    enable_rerank: bool = True
    session_id: Optional[str] = None  # Add session_id
    memory_mode: str = "session"      # session, global, or hybrid
    retrieval_instruction: Optional[str] = None # User instruction for retrieval/merge

class MultiDocumentQueryRequest(BaseModel):
    question: str
    mode: str = "hybrid"
    enable_rerank: bool = True
    session_id: Optional[str] = None

class EntityRelationshipRequest(BaseModel):
    entity_name: str
    entity_type: str
    document_id: str
    properties: Optional[Dict[str, Any]] = None

class RelationshipRequest(BaseModel):
    source_entity: str
    target_entity: str
    relationship_type: str
    document_id: str
    properties: Optional[Dict[str, Any]] = None

class IngestRequest(BaseModel):
    content: str
    source_type: str = "text" # text, url, json_blob, etc.
    session_id: Optional[str] = None

# Global RAG instance
rag_instance: Optional[RAGAnything] = None
# Global Cognitive Processor
cognitive_processor: Optional[CognitiveProcessor] = None
ingestion_service: Optional[IngestionService] = None


async def ingest_chat_turn(user_text: str, assistant_text: str, session_id: Optional[str] = None):
    """Ingest chat turn into Global and Session graphs"""
    text_to_ingest = f"User Question: {user_text}\nAssistant Answer: {assistant_text}"
    if cognitive_processor:
        try:
            insight = await cognitive_processor.process(text_to_ingest, source_type="chat")
            link_names = ", ".join([l.name for l in insight.potential_links[:10]])
            text_to_ingest += (
                f"\n\n[Cognitive Analysis]:\n"
                f"Summary: {insight.summary}\n"
                f"Concepts: {', '.join(insight.concepts)}\n"
                f"Potential Links: {link_names}\n"
                f"Reasoning: {insight.reasoning}"
            )
        except Exception as e:
            logger.exception("Cognitive processing failed for chat turn: %s", e)

    logger.info("Background task: ingesting chat turn")
    try:
        if ingestion_service:
            await ingestion_service.ingest_text(text_to_ingest, session_id=session_id)
        else:
            raise RuntimeError("Ingestion service not initialized")
    except Exception as e:
        logger.exception("Error ingesting chat turn: %s", e)



def create_rag_config() -> RAGAnythingConfig:
    return RAGAnythingConfig(
        working_dir=WORKDIR,
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

async def get_embedding_func():
    embed_client = get_embed_client(settings)
    
    async def embedding_func_impl(texts: List[str]) -> Any:
        resp = await embed_client.embeddings.create(
            model=settings.embed_model,
            input=texts,
        )
        if not hasattr(resp, "data"):
            raise RuntimeError(f"Unexpected embedding response: {resp}")
        vectors: List[List[float]] = []
        for item in resp.data:
            emb = getattr(item, "embedding", None)
            if isinstance(emb, list):
                vectors.append(emb)
        return np.array(vectors, dtype=np.float32)

    return EmbeddingFunc(
        embedding_dim=settings.embed_dim,
        max_token_size=8192,
        func=embedding_func_impl,
    )

async def get_llm_model_func():
    vlm_client = get_vlm_client(settings)

    async def chat_complete(prompt: str, system_prompt: str | None = None, **_: Any) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = await vlm_client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            max_tokens=2048,
            stream=False,
        )
        if not hasattr(resp, "choices") or not resp.choices:
            return str(resp)
        return resp.choices[0].message.content
    
    return chat_complete

async def initialize_rag() -> RAGAnything:
    """Initialize RAG instance with configuration"""
    embedding_func = await get_embedding_func()
    chat_complete = await get_llm_model_func()
    config = create_rag_config()

    return RAGAnything(
        config=config,
        llm_model_func=chat_complete,
        embedding_func=embedding_func,
    )


@app.on_event("startup")
async def startup_event():
    """Initialize RAG instance on startup"""
    global rag_instance, cognitive_processor, ingestion_service
    rag_instance = await initialize_rag()

    cognitive_processor = CognitiveProcessor(
        llm_func=rag_instance.llm_model_func,
        embedding_func=rag_instance.embedding_func,
        knowledge_graph=rag_instance.knowledge_graph,
    )

    ingestion_service = IngestionService(
        rag_global=rag_instance,
        session_manager=session_manager,
        create_rag_config=create_rag_config,
        get_llm_model_func=get_llm_model_func,
        get_embedding_func=get_embedding_func,
    )
    
    # Pre-initialize LightRAG to ensure it's ready for queries even without new ingestion
    try:
        if hasattr(rag_instance, '_ensure_lightrag_initialized'):
             logger.info("Pre-initializing LightRAG instance...")
             await rag_instance._ensure_lightrag_initialized()
    except Exception as e:
        logger.exception("Failed to pre-initialize LightRAG: %s", e)
        
    logger.info("RAG instance initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Finalize RAG storages on shutdown"""
    global rag_instance
    if rag_instance:
        await rag_instance.finalize_storages()
        print("RAG storages finalized successfully")


@app.post("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Multi-Document Enhanced RAG API",
        "version": "1.0.0"
    }


# ====== Session Management Endpoints ======

@app.post("/sessions")
async def create_session(request: CreateSessionRequest):
    """Create a new session"""
    try:
        session = session_manager.create_session(title=request.title)
        return {"status": "success", "session": session}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def list_sessions():
    """List all sessions"""
    try:
        sessions = session_manager.list_sessions()
        return {"status": "success", "sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success", "session": session}

@app.put("/sessions/{session_id}")
async def update_session(session_id: str, request: CreateSessionRequest):
    """Update session details (e.g. title)"""
    if session_manager.update_session(session_id, request.title):
        return {"status": "success", "message": "Session updated"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_manager.delete_session(session_id):
        return {"status": "success", "message": "Session deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get session chat history"""
    try:
        history = session_manager.get_history(session_id)
        return {"status": "success", "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/files")
async def get_session_files(session_id: str):
    """Get files uploaded in a session"""
    try:
        files = session_manager.get_files(session_id)
        return {"status": "success", "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/stream")
async def ingest_stream(request: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest arbitrary information stream (text, url, json)"""
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG instance not initialized")

    async def _process_and_ingest(content: str, source_type: str, session_id: Optional[str]):
        # 1. Cognitive Processing
        processed_text = content
        if cognitive_processor:
            try:
                insight = await cognitive_processor.process(content, source_type=source_type)
                link_names = ", ".join([l.name for l in insight.potential_links[:10]])
                processed_text = f"Source: {source_type}\nContent:\n{content}\n\n[Cognitive Analysis]:\nSummary: {insight.summary}\nReasoning: {insight.reasoning}\nConcepts: {', '.join(insight.concepts)}"
                processed_text += f"\nPotential Links: {link_names}"
                logger.info("Stream processed: %s", insight.summary)
            except Exception as e:
                logger.exception("Stream processing failed: %s", e)
        
        try:
            if ingestion_service:
                await ingestion_service.ingest_text(processed_text, session_id=session_id)
            else:
                raise RuntimeError("Ingestion service not initialized")
        except Exception as e:
            logger.exception("Stream ingestion failed: %s", e)

    background_tasks.add_task(
        _process_and_ingest,
        request.content,
        request.source_type,
        request.session_id
    )
    
    return {"status": "processing", "message": "Stream accepted for cognitive processing"}


@app.post("/ingest")
async def ingest_files(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = None
):
    """Ingest files into RAG system (Global and optionally Session)"""
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG instance not initialized")

    uploaded_files: List[str] = []
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in files:
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                await file.close()
                uploaded_files.append(file_path)

                if session_id:
                    try:
                        session_manager.add_document_record(
                            session_id,
                            file.filename,
                            file_path=file_path,
                            file_size=os.path.getsize(file_path),
                            file_ext=os.path.splitext(file.filename)[1].lower(),
                        )
                    except Exception as e:
                        print(f"Warning: Failed to record document in session history: {e}")

            print(f"Starting ingestion of {len(uploaded_files)} files from {temp_dir}")

            if ingestion_service:
                await ingestion_service.ingest_folder(temp_dir, session_id=session_id)
            else:
                raise RuntimeError("Ingestion service not initialized")

        return {
            "status": "success",
            "message": f"Successfully ingested {len(uploaded_files)} files",
            "files": [os.path.basename(f) for f in uploaded_files],
            "session_id": session_id,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge-graph/data")
async def get_graph_data(session_id: Optional[str] = None):
    """Get real graph data for visualization"""
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG instance not initialized")
    
    target_rag = rag_instance
    
    if session_id:
        try:
            config = create_rag_config()
            session_rag = session_manager.get_session_rag(session_id, config)
            # Initialize minimal dependencies to load graph
            session_rag.llm_model_func = await get_llm_model_func()
            session_rag.embedding_func = await get_embedding_func()
            await session_rag._ensure_lightrag_initialized()
            target_rag = session_rag
        except Exception as e:
            print(f"Error loading session RAG for visualization: {e}")
            # Fallback to global or return empty if strictly session requested?
            # Let's return error to indicate session graph issue
            raise HTTPException(status_code=404, detail=f"Session graph not found: {e}")

    try:
        # Extract data from NetworkX graph
        # Fix: RAGAnything uses .lightrag, not .rag
        if not target_rag.lightrag:
            # Try to initialize if missing (should be handled by _ensure_lightrag_initialized)
             await target_rag._ensure_lightrag_initialized()
             
        G = target_rag.lightrag.chunk_entity_relation_graph
        
        # Get internal NetworkX graph for iteration (accessing protected member for efficiency)
        # Or use public API:
        # nodes_data = await G.get_all_nodes()
        # edges_data = await G.get_all_edges()
        
        # Using public API is safer
        nodes_data = await G.get_all_nodes()
        edges_data = await G.get_all_edges()
        
        nodes = []
        edges = []
        
        # Limit nodes for visualization performance
        max_nodes = 200
        
        # Get nodes (entities)
        for i, node_info in enumerate(nodes_data):
            if i >= max_nodes:
                break
            
            node_id = node_info["id"]
            # Determine color/size based on degree or type if available
            # We don't have degree readily available in node_info unless we calculate it
            # But let's just use default size for now
            size = 20
            
            nodes.append({
                "id": node_id,
                "label": node_id,  # Entity name is the node ID in LightRAG
                "type": node_info.get("entity_type", "unknown"), # field name might be entity_type
                "size": size,
                "color": "#3498db" # Default blue
            })
            
        # Get edges (relations)
        node_ids = set(n["id"] for n in nodes)
        
        for edge_info in edges_data:
            u = edge_info["source"]
            v = edge_info["target"]
            
            if u in node_ids and v in node_ids:
                edges.append({
                    "id": f"{u}-{v}",
                    "source": u,
                    "target": v,
                    "label": edge_info.get("keywords", "related"), # Use relation keywords or type
                    "color": "#95a5a6",
                    "width": 1
                })
                
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_nodes": len(nodes_data),
                "total_edges": len(edges_data),
                "session_id": session_id
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to extract graph data: {str(e)}")


@app.post("/query")
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """RAG query endpoint with session support"""
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG instance not initialized")
    
    # Save user question to history if session_id provided
    if request.session_id:
        try:
            session_manager.add_message(request.session_id, "user", request.question)
        except Exception as e:
            print(f"Warning: Failed to save user message: {e}")
    
    answer = ""
    try:
        # Determine query strategy
        
        if request.session_id and request.memory_mode == "hybrid":
            # Hybrid Mode: Query both Session and Global, then merge
            print(f"Executing Hybrid Query for session {request.session_id}")
            
            # 1. Query Session RAG
            session_result = ""
            try:
                config = create_rag_config()
                session_rag = session_manager.get_session_rag(request.session_id, config)
                session_rag.llm_model_func = await get_llm_model_func()
                session_rag.embedding_func = await get_embedding_func()
                await session_rag._ensure_lightrag_initialized()
                
                session_result = await session_rag.aquery(
                    query=request.question,
                    mode=request.mode,
                    enable_rerank=request.enable_rerank
                )
            except Exception as e:
                print(f"Session query failed: {e}")
                session_result = "(No session context available)"
                
            # 2. Query Global RAG
            global_result = await rag_instance.aquery(
                query=request.question,
                mode=request.mode,
                enable_rerank=request.enable_rerank
            )
            
            # 3. Merge Results using LLM
            # Simple merge prompt
            instruction_text = ""
            if request.retrieval_instruction:
                instruction_text = f"\nUSER INSTRUCTION FOR MERGING: {request.retrieval_instruction}\n"

            merge_prompt = f"""
            You are an intelligent assistant with access to two knowledge sources:
            
            [Session Context (Current Conversation Files)]:
            {session_result}
            
            [Global Knowledge Base (Historical/All Files)]:
            {global_result}
            
            Please synthesize a comprehensive answer to the user's question: "{request.question}"
            
            {instruction_text}
            
            Default Policy: Prioritize the Session Context for specific details about current files, but use Global Knowledge to provide broader context or connections if relevant.
            If the User Instruction conflicts with the Default Policy, follow the User Instruction.
            """
            
            llm_func = await get_llm_model_func()
            answer = await llm_func(merge_prompt)
            
        elif request.session_id and request.memory_mode == "session":
            # Session-only mode
            try:
                config = create_rag_config()
                session_rag = session_manager.get_session_rag(request.session_id, config)
                session_rag.llm_model_func = await get_llm_model_func()
                session_rag.embedding_func = await get_embedding_func()
                await session_rag._ensure_lightrag_initialized()
                target_rag = session_rag
            except Exception as e:
                print(f"Error loading session RAG: {e}, falling back to global")
                target_rag = rag_instance
                
            answer = await target_rag.aquery(
                query=request.question,
                mode=request.mode,
                enable_rerank=request.enable_rerank
            )
            
        else:
            # Global/Default mode
            answer = await rag_instance.aquery(
                query=request.question,
                mode=request.mode,
                enable_rerank=request.enable_rerank
            )
        
        # Save answer to history
        if request.session_id:
            try:
                session_manager.add_message(request.session_id, "assistant", answer)
            except Exception as e:
                print(f"Warning: Failed to save assistant message: {e}")
        
        # Trigger background ingestion of this interaction
        background_tasks.add_task(
            ingest_chat_turn, 
            request.question, 
            answer, 
            request.session_id
        )
                
        return {
            "answer": answer,
            "query": request.question,
            "mode": request.mode,
            "session_id": request.session_id,
            "memory_mode": request.memory_mode
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/multi-document")
async def query_multi_document(request: MultiDocumentQueryRequest):
    """Multi-document enhanced query endpoint"""
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG instance not initialized")
    
    try:
        result = await rag_instance.aquery_multi_document_enhanced(
            query=request.question,
            mode=request.mode,
            enable_rerank=request.enable_rerank
        )
        return {
            "answer": result["answer"],
            "query": request.question,
            "mode": request.mode,
            "related_documents": result["related_documents"],
            "extracted_entities": result["extracted_entities"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge-graph/stats")
async def get_knowledge_graph_stats():
    """Get knowledge graph statistics"""
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG instance not initialized")
    
    try:
        kg = rag_instance.knowledge_graph
        return {
            "total_entities": len(kg.entities),
            "total_relationships": len(kg.relationships),
            "entity_types": list(set(entity.type for entity in kg.entities.values())),
            "relationship_types": list(set(rel.type for rel in kg.relationships.values()))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/knowledge-graph/entity")
async def add_entity(request: EntityRelationshipRequest):
    """Add entity to knowledge graph"""
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG instance not initialized")
    
    try:
        entity = rag_instance.knowledge_graph.add_entity(
            name=request.entity_name,
            type=request.entity_type,
            properties=request.properties,
            document_id=request.document_id
        )
        
        # Save knowledge graph
        rag_instance.knowledge_graph.save(str(rag_instance.graph_path))
        
        return {
            "status": "success",
            "entity": {
                "id": entity.id,
                "name": entity.name,
                "type": entity.type,
                "properties": entity.properties,
                "document_ids": list(entity.document_ids)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/knowledge-graph/relationship")
async def add_relationship(request: RelationshipRequest):
    """Add relationship to knowledge graph"""
    if not rag_instance:
        raise HTTPException(status_code=500, detail="RAG instance not initialized")
    
    try:
        # Find source and target entities
        source_entity = rag_instance.knowledge_graph.get_entity_by_name(request.source_entity)
        target_entity = rag_instance.knowledge_graph.get_entity_by_name(request.target_entity)
        
        if not source_entity or not target_entity:
            raise HTTPException(status_code=404, detail="Source or target entity not found")
        
        relationship = rag_instance.knowledge_graph.add_relationship(
            source_id=source_entity.id,
            target_id=target_entity.id,
            type=request.relationship_type,
            properties=request.properties,
            document_id=request.document_id
        )
        
        # Save knowledge graph
        rag_instance.knowledge_graph.save(str(rag_instance.graph_path))
        
        return {
            "status": "success",
            "relationship": {
                "id": relationship.id,
                "source_id": relationship.source_id,
                "target_id": relationship.target_id,
                "type": relationship.type,
                "properties": relationship.properties,
                "document_ids": list(relationship.document_ids)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ====== Graph Editing Endpoints ======

@app.put("/knowledge-graph/entity/{entity_name}")
async def update_entity(entity_name: str, properties: Dict[str, Any]):
    """Update entity properties (Persistent)"""
    if not rag_instance:
         raise HTTPException(status_code=500, detail="RAG instance not initialized")
    
    try:
        if not rag_instance.lightrag:
            await rag_instance._ensure_lightrag_initialized()
            
        G = rag_instance.lightrag.chunk_entity_relation_graph
        
        # Check if node exists
        if await G.has_node(entity_name):
            # Update node
            await G.upsert_node(entity_name, properties)
            # Persist changes
            await G.index_done_callback()
            
            return {"status": "success", "message": f"Entity {entity_name} updated"}
        else:
            raise HTTPException(status_code=404, detail="Entity not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/knowledge-graph/relationship")
async def delete_relationship(source: str, target: str):
    """Delete relationship (Persistent)"""
    if not rag_instance:
         raise HTTPException(status_code=500, detail="RAG instance not initialized")
    
    try:
        if not rag_instance.lightrag:
            await rag_instance._ensure_lightrag_initialized()
            
        G = rag_instance.lightrag.chunk_entity_relation_graph
        
        if await G.has_edge(source, target):
            await G.remove_edges([(source, target)])
            # Persist changes
            await G.index_done_callback()
            
            return {"status": "success", "message": f"Relationship {source}->{target} deleted"}
        else:
            raise HTTPException(status_code=404, detail="Relationship not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from raganything.server.app import app as app

if __name__ == "__main__":
    import uvicorn
    import time
    print("正在启动 RAG API 服务，延时 5 秒...", flush=True)
    time.sleep(5)
    uvicorn.run(
        "api_multi_document:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
