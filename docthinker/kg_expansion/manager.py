"""
Session-scoped lifecycle manager for LLM-expanded nodes.

Tracks candidate → active → promoted → deprecated lifecycle.
Supports token-overlap matching (fast) and embedding-based matching
(accurate) for query-time expanded-node retrieval.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip()).lower()


def _tokenize(text: str) -> List[str]:
    lowered = str(text or "").lower()
    return [t for t in re.split(r"[^a-z0-9\u4e00-\u9fff]+", lowered) if t]


def extract_entities_from_text(text: str, max_entities: int = 12) -> List[str]:
    """Lightweight regex entity extraction for promotion edge building."""
    source = str(text or "")
    if not source:
        return []
    entities: List[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"[\u4e00-\u9fff]{2,10}", source):
        name = m.group(0).strip()
        if name and name not in seen:
            seen.add(name)
            entities.append(name)
            if len(entities) >= max_entities:
                return entities
    for m in re.finditer(r"\b[A-Za-z][A-Za-z0-9_\-]{2,32}\b", source):
        name = m.group(0).strip()
        if len(name) < 3:
            continue
        if name.lower() in {"the", "and", "for", "with", "from", "this", "that"}:
            continue
        if name not in seen:
            seen.add(name)
            entities.append(name)
            if len(entities) >= max_entities:
                return entities
    return entities


class ExpandedNodeManager:
    """Session-scoped lifecycle manager for expanded knowledge nodes."""

    def __init__(
        self,
        storage_path: Path,
        *,
        promote_score_threshold: float = 1.2,
        promote_use_threshold: int = 2,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.promote_score_threshold = float(promote_score_threshold)
        self.promote_use_threshold = int(promote_use_threshold)
        self._records: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
        self._lock = RLock()

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            if self.storage_path.exists():
                try:
                    payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
                    items = payload.get("nodes", []) if isinstance(payload, dict) else (payload if isinstance(payload, list) else [])
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        entity = str(item.get("entity") or "").strip()
                        if not entity:
                            continue
                        key = _normalize_name(entity)
                        self._records[key] = self._normalize_record(item)
                except Exception:
                    self._records = {}
            self._loaded = True

    def _normalize_record(self, item: Dict[str, Any]) -> Dict[str, Any]:
        now = _utc_now_iso()
        entity = str(item.get("entity") or "").strip()
        roots = item.get("root_ids") or []
        entities = item.get("attached_entities") or []
        edges_raw = item.get("edges") or []
        return {
            "entity": entity,
            "entity_type": str(item.get("entity_type") or "concept"),
            "description": str(item.get("description") or ""),
            "status": str(item.get("status") or "candidate"),
            "reason": str(item.get("reason") or ""),
            "angle": str(item.get("angle") or ""),
            "dimension": str(item.get("dimension") or ""),
            "source": str(item.get("source") or "llm_expansion"),
            "root_ids": [str(x).strip() for x in roots if str(x).strip()],
            "edges": edges_raw if isinstance(edges_raw, list) else [],
            "hit_count": int(item.get("hit_count") or 0),
            "use_count": int(item.get("use_count") or 0),
            "promotion_score": float(item.get("promotion_score") or 0.0),
            "validation_score": float(item.get("validation_score") or 0.0),
            "last_hit_at": item.get("last_hit_at"),
            "last_used_at": item.get("last_used_at"),
            "created_at": item.get("created_at") or now,
            "updated_at": item.get("updated_at") or now,
            "attached_entities": [str(x).strip() for x in entities if str(x).strip()],
        }

    def _persist(self) -> None:
        with self._lock:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": _utc_now_iso(),
                "nodes": sorted(self._records.values(), key=lambda x: x.get("entity", "")),
            }
            self.storage_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def upsert_candidates(
        self,
        candidates: Sequence[Dict[str, Any]],
        *,
        default_root_ids: Optional[Iterable[str]] = None,
        source: str = "llm_expansion",
    ) -> Dict[str, int]:
        self._ensure_loaded()
        added = 0
        updated = 0
        roots = [str(x).strip() for x in (default_root_ids or []) if str(x).strip()]
        now = _utc_now_iso()

        with self._lock:
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                entity = str(item.get("entity") or "").strip()
                if not entity:
                    continue
                key = _normalize_name(entity)
                existing = self._records.get(key)
                if existing is None:
                    self._records[key] = self._normalize_record({
                        "entity": entity,
                        "entity_type": item.get("entity_type", "concept"),
                        "description": item.get("description", ""),
                        "status": "candidate",
                        "reason": item.get("reason", ""),
                        "angle": item.get("angle", ""),
                        "dimension": item.get("dimension", ""),
                        "source": source,
                        "root_ids": list(roots),
                        "edges": item.get("edges", []),
                        "validation_score": float(item.get("validation_score") or 0.0),
                        "created_at": now,
                        "updated_at": now,
                    })
                    added += 1
                else:
                    merged_roots = list(dict.fromkeys([*(existing.get("root_ids") or []), *roots]))
                    existing["reason"] = str(item.get("reason") or existing.get("reason") or "")
                    existing["angle"] = str(item.get("angle") or existing.get("angle") or "")
                    existing["root_ids"] = merged_roots
                    existing["updated_at"] = now
                    if item.get("description") and not existing.get("description"):
                        existing["description"] = str(item["description"])
                    if item.get("entity_type") and existing.get("entity_type") == "concept":
                        existing["entity_type"] = str(item["entity_type"])
                    if item.get("edges") and not existing.get("edges"):
                        existing["edges"] = item["edges"]
                    updated += 1
            self._persist()

        return {"added": added, "updated": updated}

    def list_nodes(
        self, *, status: Optional[str] = None, limit: int = 200,
    ) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        with self._lock:
            items = list(self._records.values())
        if status:
            items = [x for x in items if str(x.get("status") or "") == status]
        items.sort(key=lambda x: (
            -float(x.get("promotion_score") or 0.0),
            -int(x.get("use_count") or 0),
            x.get("entity") or "",
        ))
        return items[:max(1, int(limit))]

    def get(self, entity: str) -> Optional[Dict[str, Any]]:
        self._ensure_loaded()
        key = _normalize_name(entity)
        with self._lock:
            item = self._records.get(key)
            return dict(item) if item else None

    # ── Token-based matching (fast fallback) ──

    def match_nodes(
        self,
        query: str,
        *,
        top_k: int = 3,
        memory_terms: Optional[Sequence[str]] = None,
        min_score: float = 0.2,
    ) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        q = str(query or "").strip()
        if not q:
            return []

        q_tokens = set(_tokenize(q))
        mem_tokens = set(_tokenize(" ".join(memory_terms or [])))
        matches: List[Dict[str, Any]] = []

        with self._lock:
            items = list(self._records.values())

        for item in items:
            if str(item.get("status") or "") == "deprecated":
                continue

            entity = str(item.get("entity") or "")
            description = str(item.get("description") or "")
            reason = str(item.get("reason") or "")
            corpus = f"{entity} {description} {reason}".strip()
            corpus_lower = corpus.lower()
            score = 0.0

            if entity and entity.lower() in q.lower():
                score += 0.6
            if corpus and q.lower() in corpus_lower:
                score += 0.2

            c_tokens = set(_tokenize(corpus))
            if q_tokens and c_tokens:
                inter = len(q_tokens & c_tokens)
                union = len(q_tokens | c_tokens) or 1
                score += 0.5 * (inter / union)

            if mem_tokens and c_tokens:
                inter_mem = len(mem_tokens & c_tokens)
                if inter_mem:
                    score += min(0.2, 0.05 * inter_mem)

            if score >= min_score:
                enriched = dict(item)
                enriched["score"] = round(float(score), 4)
                matches.append(enriched)

        matches.sort(key=lambda x: (
            -float(x.get("score") or 0.0),
            -float(x.get("promotion_score") or 0.0),
            x.get("entity") or "",
        ))
        return matches[:max(1, int(top_k))]

    # ── Embedding-based matching (accurate) ──

    async def match_nodes_embedding(
        self,
        query: str,
        *,
        top_k: int = 3,
        embedding_func: Optional[Callable[..., Any]] = None,
        min_score: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """Match expanded nodes using embedding cosine similarity."""
        if not embedding_func:
            return self.match_nodes(query, top_k=top_k)

        self._ensure_loaded()
        with self._lock:
            items = [dict(v) for v in self._records.values() if v.get("status") != "deprecated"]

        if not items:
            return []

        import asyncio
        import numpy as np

        corpus_texts = [
            f"{it['entity']} {it.get('description', '')} {it.get('reason', '')}".strip()
            for it in items
        ]
        all_texts = [query] + corpus_texts

        if asyncio.iscoroutinefunction(embedding_func):
            embs = await embedding_func(all_texts)
        else:
            embs = embedding_func(all_texts)

        if embs is None or len(embs) < 2:
            return self.match_nodes(query, top_k=top_k)

        if hasattr(embs, "tolist"):
            embs = embs.tolist()
        embs = list(embs)

        q_emb = np.array(embs[0])
        c_embs = np.array(embs[1:])
        q_norm = np.linalg.norm(q_emb) or 1e-8
        c_norms = np.linalg.norm(c_embs, axis=1)
        c_norms = np.where(c_norms == 0, 1e-8, c_norms)
        sims = c_embs @ q_emb / (c_norms * q_norm)

        matches: List[Dict[str, Any]] = []
        for idx, sim in enumerate(sims):
            if sim >= min_score and idx < len(items):
                items[idx]["score"] = round(float(sim), 4)
                matches.append(items[idx])

        matches.sort(key=lambda x: -float(x.get("score") or 0.0))
        return matches[:max(1, int(top_k))]

    def mark_hits(self, entities: Sequence[str]) -> None:
        self._ensure_loaded()
        now = _utc_now_iso()
        with self._lock:
            for entity in entities:
                key = _normalize_name(entity)
                item = self._records.get(key)
                if not item:
                    continue
                item["hit_count"] = int(item.get("hit_count") or 0) + 1
                item["last_hit_at"] = now
                item["updated_at"] = now
            self._persist()

    def record_response_usage(
        self,
        *,
        answer: str,
        matches: Sequence[Dict[str, Any]],
        attached_entities: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        self._ensure_loaded()
        answer_text = str(answer or "").lower()
        now = _utc_now_iso()
        used: List[str] = []
        promoted: List[str] = []
        attached = [str(x).strip() for x in (attached_entities or []) if str(x).strip()]

        with self._lock:
            for matched in matches:
                entity = str(matched.get("entity") or "").strip()
                if not entity:
                    continue
                key = _normalize_name(entity)
                item = self._records.get(key)
                if not item:
                    continue
                if entity.lower() in answer_text:
                    used.append(entity)
                    item["use_count"] = int(item.get("use_count") or 0) + 1
                    item["last_used_at"] = now
                    item["promotion_score"] = float(item.get("promotion_score") or 0.0) + max(
                        0.4, float(matched.get("score") or 0.4)
                    )
                    item["status"] = self._next_status(item)
                    if item["status"] == "promoted":
                        promoted.append(entity)
                    for ent in attached:
                        if ent not in item["attached_entities"]:
                            item["attached_entities"].append(ent)
                else:
                    item["promotion_score"] = max(
                        0.0, float(item.get("promotion_score") or 0.0) - 0.05
                    )
                item["updated_at"] = now
            self._persist()

        return {"used": used, "promoted": promoted}

    def _next_status(self, item: Dict[str, Any]) -> str:
        score = float(item.get("promotion_score") or 0.0)
        uses = int(item.get("use_count") or 0)
        current = str(item.get("status") or "candidate")

        if current == "promoted":
            return "promoted"
        if uses >= self.promote_use_threshold and score >= self.promote_score_threshold:
            return "promoted"
        if score >= 0.6:
            return "active"
        return "candidate"

    # ── Forced instruction for query-time injection ──

    def build_forced_instruction(
        self,
        matches: Sequence[Dict[str, Any]],
        *,
        limit: int = 3,
    ) -> str:
        selected = list(matches)[:max(1, int(limit))]
        if not selected:
            return ""

        lines: List[str] = [
            "## 扩展知识参考",
            "系统通过知识图谱自我进化生成了以下扩展知识节点，它们与当前问题高度相关。",
            "请在回答中优先核对这些知识，如果它们与你的分析一致，请自然地融入回答中。",
            "",
        ]
        for item in selected:
            entity = str(item.get("entity") or "").strip()
            desc = str(item.get("description") or "").strip()
            etype = str(item.get("entity_type") or "").strip()
            score = item.get("score", 0)
            edges = item.get("edges") or []

            lines.append(f"### 节点: {entity}")
            if etype:
                lines.append(f"- 类型: {etype}")
            if desc:
                lines.append(f"- 描述: {desc[:300]}")
            if edges:
                edge_strs = [f"{e.get('target', '?')} ({e.get('relation', '?')})" for e in edges[:3]]
                lines.append(f"- 关联实体: {', '.join(edge_strs)}")
            lines.append(f"- 匹配置信度: {score:.2f}")
            lines.append("")

        lines.append("注意：")
        lines.append("1. 只采纳与问题确实相关的节点")
        lines.append("2. 如果某个节点的信息与文档事实矛盾，请忽略并在回答末尾注明")
        lines.append("3. 被采纳的节点将获得更高权重，未被采纳的将逐渐衰减")
        return "\n".join(lines)
