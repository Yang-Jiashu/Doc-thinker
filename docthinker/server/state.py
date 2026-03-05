from dataclasses import dataclass, field
from typing import Optional, Any, Set

from docthinker.session_manager import SessionManager
from docthinker.cognitive import CognitiveProcessor
from docthinker.services import IngestionService
from docthinker.providers import AppSettings


@dataclass
class AppState:
    settings: Optional[AppSettings] = None
    api_config: Optional[Any] = None
    session_manager: Optional[SessionManager] = None

    rag_instance: Optional[Any] = None
    cognitive_processor: Optional[CognitiveProcessor] = None
    ingestion_service: Optional[IngestionService] = None
    orchestrator: Optional[Any] = None
    memory_engine: Optional[Any] = None  # neuro_memory.MemoryEngine，类人脑联想与记忆重连
    kg_entity_ids: Set[str] = field(default_factory=set)  # 主 KG 实体 ID 集合，供记忆桥接边判断


state = AppState()
