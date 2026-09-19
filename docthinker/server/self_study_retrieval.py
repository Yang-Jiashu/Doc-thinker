"""Bounded lexical retrieval from a self-study graph snapshot, without model calls."""

from __future__ import annotations

import asyncio
import heapq
import json
import re
from collections.abc import Sequence
from typing import Any

from graphcore.coregraph.constants import GRAPH_FIELD_SEP


def _tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    for word in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.add(word)
        tokens.update(word[index : index + 2] for index in range(len(word) - 1))
    return tokens


def _source_only(record: dict[str, Any]) -> bool:
    for key in ("is_discovered", "is_expanded"):
        if str(record.get(key, "")).lower() in {"1", "true", "yes"}:
            return False
    if str(record.get("provenance", "")).lower() in {"self_study", "seal", "eclrr_v4"}:
        return False
    return str(record.get("source_id", "")) not in {
        "promoted_expansion",
        "answer_entity",
        "llm_expansion",
        "self_study",
        "seal",
    }


class SelfStudySnapshotRetriever:
    """Rank snapshot records, then fetch only their referenced source chunks.

    No GraphCore query pipeline, keyword extraction, embedding, or reranker is
    invoked. This trades semantic recall for predictable cost in audit-only
    background learning. It does not change the main user-facing retriever.
    """

    def __init__(self, nodes: Sequence[dict], edges: Sequence[dict], text_chunks: Any):
        self._text_chunks = text_chunks
        self._nodes = self._index(
            nodes, ("id", "entity_id", "entity_type", "description", "source_id")
        )
        self._edges = self._index(
            edges, ("source", "target", "keywords", "description", "source_id")
        )

    @staticmethod
    def _index(
        records: Sequence[dict], fields: Sequence[str]
    ) -> list[tuple[dict, set[str]]]:
        indexed = []
        for record in records:
            if not _source_only(record):
                continue
            compact = {
                key: str(record[key]) for key in fields if record.get(key) is not None
            }
            compact["evidence_scope"] = "graph_summary_not_source_quote"
            # Preserve the complete description/relationship or skip the record;
            # never remove a qualifying clause or truncate a source identifier.
            if len(json.dumps(compact, ensure_ascii=False)) > 2000:
                continue
            terms = _tokens(
                " ".join(
                    value
                    for key, value in compact.items()
                    if key not in {"source_id", "evidence_scope"}
                )
            )
            indexed.append((compact, terms))
        return indexed

    @staticmethod
    def _rank(indexed: list[tuple[dict, set[str]]], terms: set[str]) -> list[dict]:
        scored = (
            (len(terms & words) / max(1, len(words)), -position, record)
            for position, (record, words) in enumerate(indexed)
            if terms & words
        )
        selected = []
        remaining = 6000
        for _, _, record in heapq.nlargest(12, scored, key=lambda row: row[:2]):
            size = len(json.dumps(record, ensure_ascii=False))
            if size <= remaining:
                selected.append(dict(record))
                remaining -= size
        return selected

    async def query(self, question: str) -> dict[str, list[dict]]:
        terms = _tokens(str(question)[:2000])
        # Keep large snapshot scans off the request event loop.
        entities = await asyncio.to_thread(self._rank, self._nodes, terms)
        relations = await asyncio.to_thread(self._rank, self._edges, terms)
        chunk_ids = list(
            dict.fromkeys(
                chunk_id.strip()
                for row in [*entities, *relations]
                for chunk_id in str(row.get("source_id", "")).split(GRAPH_FIELD_SEP)
                if chunk_id.strip()
            )
        )[:5]
        chunks = []
        if chunk_ids and self._text_chunks is not None:
            records = await self._text_chunks.get_by_ids(chunk_ids)
            for chunk_id, row in zip(chunk_ids, records):
                if isinstance(row, dict) and row.get("content"):
                    content = str(row["content"])
                    end = min(len(content), 2000)
                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "source_id": chunk_id,
                            "content": content[:end],
                            "file_path": str(row.get("file_path", "")),
                            "truncated": end < len(content),
                            "excerpt_start": 0,
                            "excerpt_end": end,
                            "offset_unit": "characters_in_source_chunk",
                            "original_content_length": len(content),
                            "evidence_scope": "excerpt_only"
                            if end < len(content)
                            else "source_chunk",
                        }
                    )
        return {"entities": entities, "relations": relations, "chunks": chunks}
