from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None


class QueryRequest(BaseModel):
    question: str
    mode: str = "hybrid"
    enable_rerank: bool = True
    top_k: int = Field(default=20, ge=1, le=100)
    chunk_top_k: int = Field(default=12, ge=1, le=50)
    max_relation_tokens: int = Field(default=5000, ge=256, le=30000)
    max_total_tokens: int = Field(default=24000, ge=2048, le=120000)
    include_discovered_edges: bool = False
    max_relations: int = Field(default=32, ge=1, le=200)
    max_discovered_relations: int = Field(default=8, ge=0, le=50)
    min_discovered_edge_confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    require_discovered_evidence: bool = True
    session_id: Optional[str] = None
    memory_mode: str = "session"
    retrieval_instruction: Optional[str] = None
    enable_thinking: bool = False
    enable_expanded_matching: bool = True
    expanded_top_k: int = 2
    expanded_min_score: float = 0.2
    remember_turn: bool = True
    memory_excluded_layers: List[str] = Field(default_factory=list)
    memory_write_scope: Optional[str] = None
    enable_image_asset_activation: bool = True
    image_activation_threshold: float = 0.62
    image_activation_top_k: int = 3


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
    session_id: Optional[str] = None


class RelationshipRequest(BaseModel):
    source_entity: str
    target_entity: str
    relationship_type: str
    document_id: str
    properties: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class IngestRequest(BaseModel):
    content: str
    source_type: str = "text"
    session_id: Optional[str] = None


class SignalIngestRequest(BaseModel):
    payload: Any
    modality: Optional[str] = None
    source_type: str = "signal"
    source_uri: Optional[str] = None
    timestamp: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
