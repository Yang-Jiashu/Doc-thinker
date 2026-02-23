from typing import Dict, List, Any, Optional
from pydantic import BaseModel


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None


class QueryRequest(BaseModel):
    question: str
    mode: str = "hybrid"
    enable_rerank: bool = True
    session_id: Optional[str] = None
    memory_mode: str = "session"
    retrieval_instruction: Optional[str] = None
    enable_thinking: bool = False


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

