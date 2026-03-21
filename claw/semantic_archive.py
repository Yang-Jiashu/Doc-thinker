"""Cold Layer: Semantic Archive — chunk, embed, and vector-search older turns."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .models import ArchiveChunk, EmbeddingFunc, TurnRecord

_log = logging.getLogger("claw.semantic_archive")


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def _make_chunk_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


class SemanticArchive:
    """Embeds older conversation turns and retrieves relevant chunks via cosine similarity."""

    def __init__(
        self,
        talk_dir: str,
        embedding_func: Optional[EmbeddingFunc] = None,
        chunk_size: int = 600,
        top_k: int = 5,
        min_score: float = 0.35,
    ):
        self._talk_dir = Path(talk_dir)
        self._embedding_func = embedding_func
        self._chunk_size = chunk_size
        self._top_k = top_k
        self._min_score = min_score
        self._chunks: List[ArchiveChunk] = []
        self._loaded = False

    @property
    def archive_file(self) -> Path:
        return self._talk_dir / "archive_vectors.json"

    @property
    def chunk_count(self) -> int:
        self._ensure_loaded()
        return len(self._chunks)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.archive_file.is_file():
            return
        try:
            data = json.loads(self.archive_file.read_text(encoding="utf-8"))
            self._chunks = [ArchiveChunk.from_dict(c) for c in data.get("chunks", [])]
        except Exception as exc:
            _log.warning("Failed to load archive: %s", exc)

    def save(self) -> None:
        self._talk_dir.mkdir(parents=True, exist_ok=True)
        data = {"chunks": [c.to_dict() for c in self._chunks]}
        self.archive_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    async def archive_turns(self, turns: List[TurnRecord]) -> int:
        """Chunk and embed a batch of turns. Returns the number of new chunks added."""
        if not turns or self._embedding_func is None:
            return 0

        self._ensure_loaded()
        existing_ids = {c.chunk_id for c in self._chunks}

        new_chunks_text: List[str] = []
        new_chunks_meta: List[Dict[str, Any]] = []

        text_block = self._turns_to_text(turns)
        for segment in self._split_text(text_block, self._chunk_size):
            cid = _make_chunk_id(segment)
            if cid in existing_ids:
                continue
            new_chunks_text.append(segment)
            new_chunks_meta.append({
                "chunk_id": cid,
                "timestamp": turns[-1].timestamp if turns else 0.0,
                "turn_ids": [t.turn_id or "" for t in turns if t.turn_id],
            })

        if not new_chunks_text:
            return 0

        try:
            embeddings = await self._embedding_func(new_chunks_text)
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()

            for i, text in enumerate(new_chunks_text):
                emb = embeddings[i] if i < len(embeddings) else None
                chunk = ArchiveChunk(
                    chunk_id=new_chunks_meta[i]["chunk_id"],
                    text=text,
                    embedding=emb,
                    timestamp=new_chunks_meta[i]["timestamp"],
                    turn_ids=new_chunks_meta[i]["turn_ids"],
                )
                self._chunks.append(chunk)

            self.save()
            _log.info("Archived %d new chunks (total: %d)", len(new_chunks_text), len(self._chunks))
            return len(new_chunks_text)
        except Exception as exc:
            _log.error("Failed to embed/archive turns: %s", exc)
            return 0

    async def search(self, query: str, top_k: Optional[int] = None) -> List[ArchiveChunk]:
        """Return the most relevant archived chunks for a query."""
        if self._embedding_func is None:
            return []

        self._ensure_loaded()
        if not self._chunks:
            return []

        k = top_k or self._top_k
        try:
            query_emb_raw = await self._embedding_func([query])
            if hasattr(query_emb_raw, "tolist"):
                query_emb_raw = query_emb_raw.tolist()
            query_emb = query_emb_raw[0] if query_emb_raw else None
        except Exception as exc:
            _log.warning("Failed to embed query for archive search: %s", exc)
            return []

        if query_emb is None:
            return []

        scored: List[ArchiveChunk] = []
        for chunk in self._chunks:
            if chunk.embedding is None:
                continue
            sim = _cosine_similarity(query_emb, chunk.embedding)
            if sim >= self._min_score:
                chunk.score = sim
                scored.append(chunk)

        scored.sort(key=lambda c: -c.score)
        return scored[:k]

    @staticmethod
    def _turns_to_text(turns: List[TurnRecord]) -> str:
        lines = []
        for t in turns:
            label = "Q" if t.role == "user" else "A"
            lines.append(f"{label}: {t.content}")
        return "\n".join(lines)

    @staticmethod
    def _split_text(text: str, max_chars: int) -> List[str]:
        if len(text) <= max_chars:
            return [text] if text.strip() else []
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end
        return chunks
