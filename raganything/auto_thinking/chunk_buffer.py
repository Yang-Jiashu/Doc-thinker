"""In-memory buffer for synchronising parsed chunks with HyperGraphRAG."""
##暂存raganything流程解析出的结构化文本，如果选择hypergraphrag流程，则直接同步。
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..hypergraph.schemas import StructuredChunk


@dataclass
class ChunkRecord:
    doc_id: str
    content: StructuredChunk
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class DocumentBuffer:
    doc_id: str
    chunks: List[ChunkRecord] = field(default_factory=list)
    synced: bool = False
    file_path: Optional[str] = None


class ChunkBuffer:
    """Simple dictionary-backed buffer for chunk synchronisation."""

    def __init__(self) -> None:
        self._docs: Dict[str, DocumentBuffer] = {}

    def append(self, doc_id: str, chunk: ChunkRecord) -> None:
        doc = self._docs.setdefault(doc_id, DocumentBuffer(doc_id=doc_id))
        doc.chunks.append(chunk)

    def append_structured(
        self, doc_id: str, *, chunk: Union[ChunkRecord, StructuredChunk, str]
    ) -> None:
        """Append chunk allowing raw StructuredChunk or str for compatibility."""
        if isinstance(chunk, ChunkRecord):
            record = chunk
        elif isinstance(chunk, StructuredChunk):
            record = ChunkRecord(doc_id=doc_id, content=chunk, metadata={})
        elif isinstance(chunk, str):
            record = ChunkRecord(
                doc_id=doc_id,
                content=StructuredChunk(text=chunk, metadata={"type": "text"}),
                metadata={},
            )
        else:
            raise TypeError(f"Unsupported chunk type: {type(chunk)!r}")
        self.append(doc_id, record)

    def mark_synced(self, doc_id: str) -> None:
        if doc_id in self._docs:
            self._docs[doc_id].synced = True

    def set_file_path(self, doc_id: str, file_path: str) -> None:
        doc = self._docs.setdefault(doc_id, DocumentBuffer(doc_id=doc_id))
        doc.file_path = file_path

    def get_pending(self) -> List[DocumentBuffer]:
        return [doc for doc in self._docs.values() if not doc.synced and doc.chunks]

    def get(self, doc_id: str) -> Optional[DocumentBuffer]:
        return self._docs.get(doc_id)
