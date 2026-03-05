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
        if not getattr(rag, "graphcore", None):
            await rag._ensure_graphcore_initialized()

    async def _insert_text(self, rag: Any, text: str) -> None:
        await self._ensure_ready(rag)
        graphcore = rag.graphcore
        if hasattr(graphcore, "ainsert"):
            await graphcore.ainsert(text)
        else:
            await graphcore.insert(text)

    async def _get_session_rag(self, session_id: str) -> Any:
        config = self.create_rag_config()
        graphcore_kwargs = getattr(self.rag_global, "graphcore_kwargs", {})
        try:
            session_rag = self.session_manager.get_session_rag(
                session_id, config, graphcore_kwargs
            )
        except TypeError:
            session_rag = self.session_manager.get_session_rag(session_id, config)
        session_rag.llm_model_func = await self.get_llm_model_func()
        session_rag.embedding_func = await self.get_embedding_func()
        await session_rag._ensure_graphcore_initialized()
        return session_rag

    async def ingest_text(self, text: str, session_id: Optional[str] = None) -> None:
        """所有 ingest 均写入全局图谱，不考虑 session。"""
        await self._insert_text(self.rag_global, text)

    async def ingest_folder(self, folder_path: str, session_id: Optional[str] = None) -> None:
        """所有 ingest 均写入全局图谱，不考虑 session。"""
        await self.rag_global.process_folder_complete(folder_path)

    async def ingest_files(self, file_paths: List[str], session_id: Optional[str] = None) -> None:
        """所有 ingest 均写入全局图谱，不考虑 session。"""
        for file_path in file_paths:
            await self.rag_global.process_document_complete(file_path)
