from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Awaitable, Any, List


@dataclass(frozen=True)
class IngestionService:
    rag_global: Any
    session_manager: Any
    create_rag_config: Callable[[], Any]
    get_llm_model_func: Callable[[], Awaitable[Any]]
    get_embedding_func: Callable[[], Awaitable[Any]]

    async def _ensure_ready(self, rag: Any) -> None:
        if not getattr(rag, "lightrag", None):
            await rag._ensure_lightrag_initialized()

    async def _insert_text(self, rag: Any, text: str) -> None:
        await self._ensure_ready(rag)
        lightrag = rag.lightrag
        if hasattr(lightrag, "ainsert"):
            await lightrag.ainsert(text)
        else:
            await lightrag.insert(text)

    async def _get_session_rag(self, session_id: str) -> Any:
        config = self.create_rag_config()
        lightrag_kwargs = self.rag_global.lightrag_kwargs
        session_rag = self.session_manager.get_session_rag(session_id, config, lightrag_kwargs)
        session_rag.llm_model_func = await self.get_llm_model_func()
        session_rag.embedding_func = await self.get_embedding_func()
        await session_rag._ensure_lightrag_initialized()
        return session_rag

    async def ingest_text(self, text: str, session_id: Optional[str] = None) -> None:
        await self._insert_text(self.rag_global, text)
        if session_id:
            session_rag = await self._get_session_rag(session_id)
            await self._insert_text(session_rag, text)

    async def ingest_folder(self, folder_path: str, session_id: Optional[str] = None) -> None:
        await self.rag_global.process_folder_complete(folder_path)
        if session_id:
            session_rag = await self._get_session_rag(session_id)
            await session_rag.process_folder_complete(folder_path)

    async def ingest_files(self, file_paths: List[str], session_id: Optional[str] = None) -> None:
        for file_path in file_paths:
            await self.rag_global.process_document_complete(file_path)
        
        if session_id:
            session_rag = await self._get_session_rag(session_id)
            for file_path in file_paths:
                await session_rag.process_document_complete(file_path)

