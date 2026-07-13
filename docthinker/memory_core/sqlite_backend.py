"""SQLite-backed long-horizon memory with revisions and durable edges."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .core import InMemoryLongHorizonBackend


class SQLiteLongHorizonBackend(InMemoryLongHorizonBackend):
    """Persistent implementation of the long-horizon memory contract.

    SQLite is the source of truth. The inherited class only supplies the
    deterministic classification, tokenization, edit-planning, instruction,
    and memory-reasoning helpers.
    """

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        # Keep the parent lock/decision attributes for helper compatibility,
        # but do not use its process-local ``_items`` store.
        super().__init__()
        self.db_path = Path(db_path or self.default_path()).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_lock = threading.RLock()
        self._initialize_schema()

    @staticmethod
    def default_path() -> Path:
        explicit = str(os.getenv("LONG_HORIZON_DB_PATH", "")).strip()
        if explicit:
            return Path(explicit)
        workdir = str(os.getenv("RAG_WORKDIR", "./rag_storage_api")).strip()
        return Path(workdir or "./rag_storage_api") / "long_horizon_memory.db"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _initialize_schema(self) -> None:
        with self._db_lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS long_horizon_memories (
                    id TEXT PRIMARY KEY,
                    scope_key TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    question TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    concepts_json TEXT NOT NULL DEFAULT '[]',
                    expanded_entities_json TEXT NOT NULL DEFAULT '[]',
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    use_count INTEGER NOT NULL DEFAULT 1,
                    audit_json TEXT NOT NULL DEFAULT '{}',
                    version INTEGER NOT NULL DEFAULT 1
                );

                CREATE INDEX IF NOT EXISTS idx_lh_scope_status
                    ON long_horizon_memories(scope_key, status, last_seen_at DESC);
                CREATE INDEX IF NOT EXISTS idx_lh_session
                    ON long_horizon_memories(session_id, last_seen_at DESC);

                CREATE TABLE IF NOT EXISTS long_horizon_revisions (
                    revision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    operator TEXT NOT NULL DEFAULT 'system',
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_lh_revision_memory
                    ON long_horizon_revisions(memory_id, revision_id DESC);

                CREATE TABLE IF NOT EXISTS long_horizon_edges (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 0.5,
                    evidence_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (source_id, target_id, relation_type)
                );

                CREATE INDEX IF NOT EXISTS idx_lh_edge_source
                    ON long_horizon_edges(source_id, status, weight DESC);
                CREATE INDEX IF NOT EXISTS idx_lh_edge_target
                    ON long_horizon_edges(target_id, status, weight DESC);

                CREATE TABLE IF NOT EXISTS cognitions (
                    id TEXT PRIMARY KEY,
                    scope_key TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    cognition_type TEXT NOT NULL,
                    statement TEXT NOT NULL,
                    conditions_json TEXT NOT NULL DEFAULT '[]',
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    use_count INTEGER NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 1,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_cognition_scope_status
                    ON cognitions(scope_key, status, updated_at DESC);

                CREATE TABLE IF NOT EXISTS cognition_evidence (
                    cognition_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    stance TEXT NOT NULL DEFAULT 'supports',
                    weight REAL NOT NULL DEFAULT 1.0,
                    relation_type TEXT NOT NULL DEFAULT 'derived_from',
                    created_at REAL NOT NULL,
                    PRIMARY KEY (cognition_id, memory_id, stance),
                    FOREIGN KEY (cognition_id) REFERENCES cognitions(id),
                    FOREIGN KEY (memory_id) REFERENCES long_horizon_memories(id)
                );

                CREATE INDEX IF NOT EXISTS idx_cognition_evidence_memory
                    ON cognition_evidence(memory_id, cognition_id);

                CREATE TABLE IF NOT EXISTS cognition_revisions (
                    revision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cognition_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    operator TEXT NOT NULL DEFAULT 'system',
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_cognition_revision
                    ON cognition_revisions(cognition_id, revision_id DESC);

                CREATE TABLE IF NOT EXISTS long_horizon_metadata (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )

    @staticmethod
    def _json_load(value: Any, fallback: Any) -> Any:
        try:
            return json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return fallback

    @staticmethod
    def _json_dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    @classmethod
    def _row_to_item(cls, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "scope": row["scope"],
            "kind": row["kind"],
            "question": row["question"],
            "summary": row["summary"],
            "concepts": cls._json_load(row["concepts_json"], []),
            "expanded_entities": cls._json_load(row["expanded_entities_json"], []),
            "confidence": float(row["confidence"]),
            "status": row["status"],
            "created_at": float(row["created_at"]),
            "last_seen_at": float(row["last_seen_at"]),
            "use_count": int(row["use_count"]),
            "audit": cls._json_load(row["audit_json"], {}),
            "version": int(row["version"]),
        }

    @classmethod
    def _row_to_cognition(
        cls,
        row: sqlite3.Row,
        evidence: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        evidence_items = [dict(item) for item in (evidence or [])]
        return {
            "id": row["id"],
            "layer": "cognition",
            "scope": row["scope"],
            "cognition_type": row["cognition_type"],
            "statement": row["statement"],
            "conditions": cls._json_load(row["conditions_json"], []),
            "confidence": float(row["confidence"]),
            "status": row["status"],
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
            "use_count": int(row["use_count"]),
            "version": int(row["version"]),
            "metadata": cls._json_load(row["metadata_json"], {}),
            "evidence": evidence_items,
            "evidence_memory_ids": [
                str(item.get("memory_id")) for item in evidence_items
                if item.get("memory_id")
            ],
        }

    @staticmethod
    def _normalized_session_id(session_id: Optional[str], scope: str) -> str:
        if str(scope or "session") != "session":
            return "global"
        return str(session_id or "global").strip() or "global"

    def _scope_keys(
        self,
        session_id: Optional[str],
        scope: Optional[str] = None,
    ) -> List[str]:
        if scope:
            return [self._scope_key(session_id, scope)]
        if session_id:
            return [
                self._scope_key(session_id, "session"),
                self._scope_key(session_id, "project"),
                self._scope_key(session_id, "user"),
            ]
        return []

    def _set_last_decision(
        self,
        decision: Dict[str, Any],
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        payload = dict(decision or {})
        self._last_decision = payload
        owns_connection = conn is None
        active = conn or self._connect()
        try:
            active.execute(
                """
                INSERT INTO long_horizon_metadata(key, value_json, updated_at)
                VALUES ('last_write_decision', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (self._json_dump(payload), time.time()),
            )
            if owns_connection:
                active.commit()
        finally:
            if owns_connection:
                active.close()

    def last_write_decision(self) -> Dict[str, Any]:
        with self._db_lock, self._connect() as conn:
            row = conn.execute(
                "SELECT value_json FROM long_horizon_metadata WHERE key = ?",
                ("last_write_decision",),
            ).fetchone()
        if not row:
            return {"action": "init", "storage": "sqlite"}
        return dict(self._json_load(row["value_json"], {"action": "init"}))

    def _record_revision(
        self,
        conn: sqlite3.Connection,
        item: Dict[str, Any],
        *,
        action: str,
        reason: str,
        operator: str,
        created_at: Optional[float] = None,
    ) -> int:
        cursor = conn.execute(
            """
            INSERT INTO long_horizon_revisions(
                memory_id, version, action, snapshot_json, reason, operator, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(item["id"]),
                int(item.get("version") or 1),
                str(action),
                self._json_dump(item),
                str(reason or ""),
                str(operator or "system"),
                float(created_at if created_at is not None else time.time()),
            ),
        )
        return int(cursor.lastrowid)

    def _write_item(self, conn: sqlite3.Connection, item: Dict[str, Any]) -> None:
        scope = str(item.get("scope") or "session")
        session_id = str(item.pop("_session_id", "global"))
        conn.execute(
            """
            INSERT INTO long_horizon_memories(
                id, scope_key, session_id, scope, kind, question, summary,
                concepts_json, expanded_entities_json, confidence, status,
                created_at, last_seen_at, use_count, audit_json, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                scope_key = excluded.scope_key,
                session_id = excluded.session_id,
                scope = excluded.scope,
                kind = excluded.kind,
                question = excluded.question,
                summary = excluded.summary,
                concepts_json = excluded.concepts_json,
                expanded_entities_json = excluded.expanded_entities_json,
                confidence = excluded.confidence,
                status = excluded.status,
                created_at = excluded.created_at,
                last_seen_at = excluded.last_seen_at,
                use_count = excluded.use_count,
                audit_json = excluded.audit_json,
                version = excluded.version
            """,
            (
                str(item["id"]),
                self._scope_key(session_id, scope),
                session_id,
                scope,
                str(item.get("kind") or "insight"),
                str(item.get("question") or "")[:260],
                str(item.get("summary") or "")[:260],
                self._json_dump(list(item.get("concepts") or [])[:16]),
                self._json_dump(list(item.get("expanded_entities") or [])[:12]),
                float(item.get("confidence") or 0.55),
                str(item.get("status") or "active"),
                float(item.get("created_at") or time.time()),
                float(item.get("last_seen_at") or time.time()),
                int(item.get("use_count") or 1),
                self._json_dump(dict(item.get("audit") or {})),
                int(item.get("version") or 1),
            ),
        )

    @staticmethod
    def _relation_recall_factor(relation_type: str) -> float:
        return {
            "supports": 0.95,
            "causes": 0.92,
            "derived_from": 0.9,
            "analogous_to": 0.86,
            "similar_to": 0.78,
            "related_to": 0.7,
            "contradicts": 0.62,
        }.get(str(relation_type or "related_to"), 0.68)

    def _auto_link_memory(
        self,
        conn: sqlite3.Connection,
        item: Dict[str, Any],
        *,
        scope_key: str,
    ) -> int:
        """Create conservative similarity edges without rewriting either memory."""
        source_tokens = self._tokens(self._candidate_text(item))
        source_concepts = {str(value).strip().lower() for value in item.get("concepts") or [] if str(value).strip()}
        if not source_tokens and not source_concepts:
            return 0
        rows = conn.execute(
            """
            SELECT * FROM long_horizon_memories
            WHERE scope_key = ? AND status = 'active' AND id != ?
            ORDER BY last_seen_at DESC LIMIT 200
            """,
            (scope_key, str(item["id"])),
        ).fetchall()
        linked = 0
        now = time.time()
        for row in rows:
            other = self._row_to_item(row)
            other_tokens = self._tokens(self._candidate_text(other))
            other_concepts = {str(value).strip().lower() for value in other.get("concepts") or [] if str(value).strip()}
            token_union = source_tokens | other_tokens
            token_score = len(source_tokens & other_tokens) / max(1, len(token_union))
            concept_union = source_concepts | other_concepts
            concept_score = len(source_concepts & other_concepts) / max(1, len(concept_union))
            weight = min(0.95, token_score * 0.7 + concept_score * 0.3)
            if weight < 0.18:
                continue
            source_id, target_id = sorted((str(item["id"]), str(other["id"])))
            relation = "analogous_to" if str(item.get("kind")) == str(other.get("kind")) and weight >= 0.34 else "similar_to"
            conn.execute(
                """
                INSERT INTO long_horizon_edges(
                    source_id, target_id, relation_type, weight,
                    evidence_json, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
                ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
                    weight = excluded.weight,
                    evidence_json = excluded.evidence_json,
                    status = 'active',
                    updated_at = excluded.updated_at
                """,
                (
                    source_id,
                    target_id,
                    relation,
                    round(weight, 4),
                    self._json_dump({
                        "source": "automatic_memory_linker",
                        "token_similarity": round(token_score, 4),
                        "concept_similarity": round(concept_score, 4),
                    }),
                    now,
                    now,
                ),
            )
            linked += 1
        return linked

    def retrieve(
        self,
        session_id: Optional[str],
        query: str,
        *,
        scopes: Sequence[str],
        top_k: int,
        min_confidence: float,
    ) -> List[Dict[str, Any]]:
        query_tokens = self._tokens(query)
        scope_keys = [self._scope_key(session_id, scope) for scope in scopes]
        if not scope_keys:
            return []
        placeholders = ",".join("?" for _ in scope_keys)
        with self._db_lock, self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM long_horizon_memories
                WHERE status = 'active'
                  AND scope_key IN ({placeholders})
                  AND confidence >= ?
                """,
                (*scope_keys, float(min_confidence)),
            ).fetchall()

            allowed_ids = {str(row["id"]) for row in rows}
            edge_rows = []
            if allowed_ids:
                edge_rows = conn.execute(
                    """
                    SELECT * FROM long_horizon_edges
                    WHERE status = 'active'
                    ORDER BY weight DESC, updated_at DESC
                    """
                ).fetchall()

        row_items = {str(row["id"]): self._row_to_item(row) for row in rows}
        scored: List[tuple[float, Dict[str, Any]]] = []
        for row in rows:
            item = row_items[str(row["id"])]
            item_tokens = self._tokens(
                " ".join(
                    [
                        str(item.get("summary") or ""),
                        str(item.get("question") or ""),
                        " ".join(str(c) for c in item.get("concepts") or []),
                    ]
                )
            )
            overlap = len(query_tokens & item_tokens)
            if query_tokens and overlap == 0:
                continue
            score = float(item["confidence"]) + overlap * 0.12 + min(
                0.2, float(item.get("use_count") or 0) * 0.03
            )
            item["score"] = round(score, 3)
            item["recall_origin"] = "direct"
            item["graph_path"] = []
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)

        limit = max(0, int(top_k))
        if limit == 0 or not scored:
            return []
        # Keep the highest lexical match as the graph seed. Other direct
        # matches can still be returned later, but a connected neighbour is
        # explicitly marked as graph-derived so the relation path is auditable.
        seed_count = 1
        seeds = scored[:seed_count]
        adjacency: Dict[str, List[Dict[str, Any]]] = {}
        for edge in edge_rows:
            source = str(edge["source_id"])
            target = str(edge["target_id"])
            if source not in allowed_ids or target not in allowed_ids:
                continue
            payload = {
                "source_id": source,
                "target_id": target,
                "relation_type": edge["relation_type"],
                "weight": float(edge["weight"]),
            }
            adjacency.setdefault(source, []).append({**payload, "neighbor_id": target})
            adjacency.setdefault(target, []).append({**payload, "neighbor_id": source})

        graph_best: Dict[str, tuple[float, Dict[str, Any]]] = {}
        direct_ids = {str(item[1]["id"]) for item in seeds}
        for seed_score, seed in seeds:
            seed_id = str(seed["id"])
            queue: List[tuple[str, int, float, List[Dict[str, Any]]]] = [
                (seed_id, 0, float(seed_score), [])
            ]
            visited = {seed_id}
            while queue:
                current_id, depth, path_score, path = queue.pop(0)
                if depth >= 2:
                    continue
                for edge in adjacency.get(current_id, []):
                    neighbor_id = str(edge["neighbor_id"])
                    if neighbor_id in visited:
                        continue
                    visited.add(neighbor_id)
                    factor = self._relation_recall_factor(str(edge["relation_type"]))
                    next_score = path_score * float(edge["weight"]) * factor * 0.82
                    next_path = path + [{
                        "source_id": current_id,
                        "target_id": neighbor_id,
                        "relation_type": edge["relation_type"],
                        "weight": round(float(edge["weight"]), 4),
                    }]
                    neighbor = row_items.get(neighbor_id)
                    if neighbor is None:
                        continue
                    graph_score = next_score + float(neighbor.get("confidence") or 0.0) * 0.15
                    candidate = dict(neighbor)
                    candidate.update({
                        "score": round(graph_score, 3),
                        "recall_origin": "graph",
                        "seed_memory_id": seed_id,
                        "graph_depth": depth + 1,
                        "graph_path": next_path,
                    })
                    previous = graph_best.get(neighbor_id)
                    if neighbor_id not in direct_ids and (previous is None or graph_score > previous[0]):
                        graph_best[neighbor_id] = (graph_score, candidate)
                    queue.append((neighbor_id, depth + 1, next_score, next_path))

        graph_ranked = sorted(graph_best.values(), key=lambda pair: pair[0], reverse=True)
        graph_slots = min(len(graph_ranked), max(0, limit // 2))
        # Add the lexical seed first, then reserve explicit graph slots. This
        # prevents the same neighbour from being labelled as a direct match
        # merely because it also shared one weak token with the query.
        results = [item for _, item in seeds]
        existing_ids = {str(item["id"]) for item in results}
        graph_added = 0
        for _, item in graph_ranked:
            if len(results) >= limit or graph_added >= graph_slots:
                break
            if str(item["id"]) not in existing_ids:
                results.append(item)
                existing_ids.add(str(item["id"]))
                graph_added += 1
        if len(results) < limit:
            for _, item in scored[seed_count:]:
                if len(results) >= limit:
                    break
                if str(item["id"]) not in existing_ids:
                    results.append(item)
                    existing_ids.add(str(item["id"]))
        return results

    def consolidate(
        self,
        session_id: Optional[str],
        question: str,
        answer: str,
        *,
        concepts: Sequence[str],
        scope: str,
        timestamp: float,
        matched_expanded: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        summary = self._clean_summary(answer)
        decision = self._write_decision(question, answer, concepts, summary=summary)
        scope_name = str(scope or "session")
        base_decision = {
            **decision,
            "scope": scope_name,
            "question": str(question or "")[:120],
            "storage": "sqlite",
        }
        if decision.get("action") != "store":
            with self._db_lock, self._connect() as conn:
                self._set_last_decision(base_decision, conn)
            return None

        session_key = self._normalized_session_id(session_id, scope_name)
        scope_key = self._scope_key(session_id, scope_name)
        concepts_list = [str(c) for c in concepts[:12] if str(c).strip()]
        expanded_entities = [
            str(item.get("entity") or item.get("name") or "")
            for item in (matched_expanded or [])
            if item.get("entity") or item.get("name")
        ]
        normalized_summary = re.sub(r"\s+", " ", summary).strip().lower()

        with self._db_lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM long_horizon_memories
                WHERE scope_key = ? AND status = 'active'
                ORDER BY last_seen_at DESC
                """,
                (scope_key,),
            ).fetchall()
            for row in rows:
                item = self._row_to_item(row)
                existing_summary = re.sub(
                    r"\s+", " ", str(item.get("summary") or "")
                ).strip().lower()
                # An identical observation reinforces the same memory. A
                # merely similar observation remains a separate source node;
                # _auto_link_memory connects it after insertion instead of
                # overwriting what previously happened.
                if not normalized_summary or normalized_summary != existing_summary:
                    continue
                item.update(
                    {
                        "concepts": sorted(set(item["concepts"] + concepts_list))[:16],
                        "expanded_entities": sorted(
                            set(item["expanded_entities"] + expanded_entities)
                        )[:12],
                        "last_seen_at": float(timestamp),
                        "use_count": int(item.get("use_count") or 0) + 1,
                        "confidence": min(0.95, float(item["confidence"]) + 0.05),
                        "audit": {
                            "last_decision": decision.get("reason"),
                            "source": "duplicate_reinforcement",
                            "schema": "docthinker.long_horizon.v2",
                        },
                        "version": int(item.get("version") or 1) + 1,
                        "status": "active",
                        "_session_id": row["session_id"],
                    }
                )
                self._write_item(conn, item)
                linked_edges = self._auto_link_memory(conn, item, scope_key=scope_key)
                item.pop("_session_id", None)
                item["graph_edges_created"] = linked_edges
                self._record_revision(
                    conn,
                    item,
                    action="reinforce",
                    reason="identical_memory_observed_again",
                    operator="after_response",
                    created_at=timestamp,
                )
                final_decision = {
                    **base_decision,
                    "action": "reinforce",
                    "memory_id": item["id"],
                    "kind": item["kind"],
                }
                self._set_last_decision(final_decision, conn)
                return dict(item)

            item = {
                "id": f"lh-{scope_name}-{int(timestamp * 1000)}-{uuid.uuid4().hex[:8]}",
                "scope": scope_name,
                "kind": self._classify_insight(question, answer),
                "question": str(question or "")[:260],
                "summary": summary,
                "concepts": concepts_list,
                "expanded_entities": expanded_entities[:12],
                "confidence": 0.62 if expanded_entities else 0.55,
                "status": "active",
                "created_at": float(timestamp),
                "last_seen_at": float(timestamp),
                "use_count": 1,
                "version": 1,
                "audit": {
                    "last_decision": decision.get("reason"),
                    "source": "after_response",
                    "schema": "docthinker.long_horizon.v2",
                },
                "_session_id": session_key,
            }
            self._write_item(conn, item)
            linked_edges = self._auto_link_memory(conn, item, scope_key=scope_key)
            item.pop("_session_id", None)
            item["graph_edges_created"] = linked_edges
            self._record_revision(
                conn,
                item,
                action="create",
                reason=str(decision.get("reason") or "durable_signal"),
                operator="after_response",
                created_at=timestamp,
            )
            final_decision = {
                **base_decision,
                "action": "store",
                "memory_id": item["id"],
                "kind": item["kind"],
            }
            self._set_last_decision(final_decision, conn)
            return dict(item)

    def list_insights(
        self,
        session_id: Optional[str] = None,
        *,
        scope: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        scope_keys = self._scope_keys(session_id, scope)
        where = ["status = 'active'"]
        params: List[Any] = []
        if scope_keys:
            where.append(f"scope_key IN ({','.join('?' for _ in scope_keys)})")
            params.extend(scope_keys)
        elif scope:
            return []
        params.append(max(0, int(limit)))
        with self._db_lock, self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM long_horizon_memories
                WHERE {' AND '.join(where)}
                ORDER BY last_seen_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def _find_row(
        self,
        conn: sqlite3.Connection,
        memory_id: str,
        session_id: Optional[str],
    ) -> Optional[sqlite3.Row]:
        params: List[Any] = [str(memory_id)]
        where = ["id = ?"]
        scope_keys = self._scope_keys(session_id)
        if scope_keys:
            where.append(f"scope_key IN ({','.join('?' for _ in scope_keys)})")
            params.extend(scope_keys)
        return conn.execute(
            f"SELECT * FROM long_horizon_memories WHERE {' AND '.join(where)}",
            params,
        ).fetchone()

    def _mark_cognitions_for_review(
        self,
        conn: sqlite3.Connection,
        memory_id: str,
    ) -> int:
        rows = conn.execute(
            """
            SELECT c.* FROM cognitions c
            JOIN cognition_evidence ce ON ce.cognition_id = c.id
            WHERE ce.memory_id = ? AND c.status = 'active'
            """,
            (str(memory_id),),
        ).fetchall()
        changed = 0
        for row in rows:
            cognition_id = str(row["id"])
            metadata = self._json_load(row["metadata_json"], {})
            metadata.update({
                "needs_review_reason": "source_memory_changed",
                "changed_memory_id": str(memory_id),
            })
            conn.execute(
                """
                UPDATE cognitions SET status = 'needs_review', updated_at = ?,
                    version = version + 1, metadata_json = ? WHERE id = ?
                """,
                (time.time(), self._json_dump(metadata), cognition_id),
            )
            updated_row = conn.execute(
                "SELECT * FROM cognitions WHERE id = ?",
                (cognition_id,),
            ).fetchone()
            updated = self._row_to_cognition(
                updated_row,
                self._cognition_evidence(conn, cognition_id),
            )
            self._record_cognition_revision(
                conn,
                updated,
                action="invalidate",
                reason="source_memory_changed",
                operator="memory_editor",
            )
            changed += 1
        return changed

    def delete_insight(self, memory_id: str, session_id: Optional[str] = None) -> bool:
        target = str(memory_id or "").strip()
        if not target:
            return False
        with self._db_lock, self._connect() as conn:
            row = self._find_row(conn, target, session_id)
            if not row or row["status"] == "deleted":
                return False
            item = self._row_to_item(row)
            item.update(
                {
                    "status": "deleted",
                    "last_seen_at": time.time(),
                    "version": int(item["version"]) + 1,
                    "audit": {
                        **dict(item.get("audit") or {}),
                        "last_decision": "user_controlled_delete",
                        "source": "memory_editor",
                    },
                    "_session_id": row["session_id"],
                }
            )
            self._write_item(conn, item)
            item.pop("_session_id", None)
            conn.execute(
                "UPDATE long_horizon_edges SET status = 'inactive', updated_at = ? WHERE source_id = ? OR target_id = ?",
                (time.time(), target, target),
            )
            self._mark_cognitions_for_review(conn, target)
            self._record_revision(
                conn,
                item,
                action="delete",
                reason="user_controlled_memory_management",
                operator="memory_editor",
            )
            self._set_last_decision(
                {
                    "action": "delete",
                    "reason": "user_controlled_memory_management",
                    "memory_id": target,
                    "storage": "sqlite",
                },
                conn,
            )
            return True

    def update_insight(
        self,
        memory_id: str,
        patch: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        target = str(memory_id or "").strip()
        allowed = {"summary", "question", "kind", "concepts", "expanded_entities", "confidence"}
        clean_patch = {key: value for key, value in (patch or {}).items() if key in allowed}
        if not target or not clean_patch:
            return None
        with self._db_lock, self._connect() as conn:
            row = self._find_row(conn, target, session_id)
            if not row or row["status"] != "active":
                return None
            item = self._row_to_item(row)
            if "summary" in clean_patch:
                item["summary"] = self._clean_summary(str(clean_patch["summary"])) or item["summary"]
            if "question" in clean_patch:
                item["question"] = str(clean_patch["question"] or "")[:260]
            if "kind" in clean_patch:
                item["kind"] = str(clean_patch["kind"] or "insight")[:64]
            if isinstance(clean_patch.get("concepts"), (list, tuple)):
                item["concepts"] = [str(c) for c in clean_patch["concepts"][:16] if str(c).strip()]
            if isinstance(clean_patch.get("expanded_entities"), (list, tuple)):
                item["expanded_entities"] = [
                    str(c) for c in clean_patch["expanded_entities"][:12] if str(c).strip()
                ]
            if "confidence" in clean_patch:
                try:
                    item["confidence"] = max(0.05, min(0.99, float(clean_patch["confidence"])))
                except (TypeError, ValueError):
                    pass
            item.update(
                {
                    "last_seen_at": time.time(),
                    "use_count": int(item.get("use_count") or 0) + 1,
                    "version": int(item["version"]) + 1,
                    "audit": {
                        **dict(item.get("audit") or {}),
                        "last_decision": "natural_language_edit",
                        "source": "memory_editor",
                    },
                    "_session_id": row["session_id"],
                }
            )
            self._write_item(conn, item)
            linked_edges = self._auto_link_memory(conn, item, scope_key=row["scope_key"])
            item.pop("_session_id", None)
            item["graph_edges_created"] = linked_edges
            item["cognitions_marked_for_review"] = self._mark_cognitions_for_review(conn, target)
            self._record_revision(
                conn,
                item,
                action="update",
                reason="natural_language_memory_edit",
                operator="memory_editor",
            )
            self._set_last_decision(
                {
                    "action": "update",
                    "reason": "natural_language_memory_edit",
                    "memory_id": target,
                    "fields": sorted(clean_patch),
                    "storage": "sqlite",
                },
                conn,
            )
            return dict(item)

    def list_revisions(
        self,
        memory_id: str,
        session_id: Optional[str] = None,
        *,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        target = str(memory_id or "").strip()
        if not target:
            return []
        with self._db_lock, self._connect() as conn:
            if not self._find_row(conn, target, session_id):
                return []
            rows = conn.execute(
                """
                SELECT revision_id, memory_id, version, action, snapshot_json,
                       reason, operator, created_at
                FROM long_horizon_revisions
                WHERE memory_id = ?
                ORDER BY revision_id DESC
                LIMIT ?
                """,
                (target, max(0, int(limit))),
            ).fetchall()
        return [
            {
                "revision_id": int(row["revision_id"]),
                "memory_id": row["memory_id"],
                "version": int(row["version"]),
                "action": row["action"],
                "snapshot": self._json_load(row["snapshot_json"], {}),
                "reason": row["reason"],
                "operator": row["operator"],
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    def restore_revision(
        self,
        memory_id: str,
        revision_id: int,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        target = str(memory_id or "").strip()
        if not target:
            return None
        with self._db_lock, self._connect() as conn:
            current_row = self._find_row(conn, target, session_id)
            if not current_row:
                return None
            revision = conn.execute(
                """
                SELECT snapshot_json FROM long_horizon_revisions
                WHERE revision_id = ? AND memory_id = ?
                """,
                (int(revision_id), target),
            ).fetchone()
            if not revision:
                return None
            restored = dict(self._json_load(revision["snapshot_json"], {}))
            if not restored:
                return None
            restored.update(
                {
                    "id": target,
                    "status": "active",
                    "last_seen_at": time.time(),
                    "version": int(current_row["version"]) + 1,
                    "audit": {
                        **dict(restored.get("audit") or {}),
                        "last_decision": "restore_revision",
                        "source": "memory_editor",
                        "restored_from_revision": int(revision_id),
                    },
                    "_session_id": current_row["session_id"],
                }
            )
            self._write_item(conn, restored)
            linked_edges = self._auto_link_memory(conn, restored, scope_key=current_row["scope_key"])
            restored.pop("_session_id", None)
            restored["graph_edges_created"] = linked_edges
            self._record_revision(
                conn,
                restored,
                action="restore",
                reason=f"restore_revision:{int(revision_id)}",
                operator="memory_editor",
            )
            self._set_last_decision(
                {
                    "action": "restore",
                    "reason": "user_controlled_revision_restore",
                    "memory_id": target,
                    "revision_id": int(revision_id),
                    "storage": "sqlite",
                },
                conn,
            )
            return dict(restored)

    def upsert_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        *,
        weight: float = 0.5,
        evidence: Optional[Dict[str, Any]] = None,
        status: str = "active",
    ) -> Dict[str, Any]:
        source = str(source_id or "").strip()
        target = str(target_id or "").strip()
        relation = str(relation_type or "related_to").strip() or "related_to"
        if not source or not target or source == target:
            raise ValueError("source_id and target_id must be different, non-empty memory ids")
        now = time.time()
        normalized_weight = max(0.0, min(1.0, float(weight)))
        with self._db_lock, self._connect() as conn:
            known = conn.execute(
                "SELECT COUNT(*) AS count FROM long_horizon_memories WHERE id IN (?, ?)",
                (source, target),
            ).fetchone()
            if int(known["count"]) != 2:
                raise ValueError("both edge endpoints must exist")
            conn.execute(
                """
                INSERT INTO long_horizon_edges(
                    source_id, target_id, relation_type, weight,
                    evidence_json, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
                    weight = excluded.weight,
                    evidence_json = excluded.evidence_json,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    source,
                    target,
                    relation,
                    normalized_weight,
                    self._json_dump(dict(evidence or {})),
                    str(status or "active"),
                    now,
                    now,
                ),
            )
        return {
            "source_id": source,
            "target_id": target,
            "relation_type": relation,
            "weight": normalized_weight,
            "evidence": dict(evidence or {}),
            "status": str(status or "active"),
            "updated_at": now,
        }

    def list_edges(
        self,
        *,
        memory_id: Optional[str] = None,
        session_id: Optional[str] = None,
        status: str = "active",
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        where = ["e.status = ?"]
        params: List[Any] = [str(status or "active")]
        if memory_id:
            where.append("(e.source_id = ? OR e.target_id = ?)")
            params.extend([str(memory_id), str(memory_id)])
        scope_keys = self._scope_keys(session_id)
        if scope_keys:
            placeholders = ",".join("?" for _ in scope_keys)
            where.append(f"(s.scope_key IN ({placeholders}) OR t.scope_key IN ({placeholders}))")
            params.extend(scope_keys)
            params.extend(scope_keys)
        params.append(max(0, int(limit)))
        with self._db_lock, self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT e.* FROM long_horizon_edges e
                JOIN long_horizon_memories s ON s.id = e.source_id
                JOIN long_horizon_memories t ON t.id = e.target_id
                WHERE {' AND '.join(where)}
                ORDER BY e.weight DESC, e.updated_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [
            {
                "source_id": row["source_id"],
                "target_id": row["target_id"],
                "relation_type": row["relation_type"],
                "weight": float(row["weight"]),
                "evidence": self._json_load(row["evidence_json"], {}),
                "status": row["status"],
                "created_at": float(row["created_at"]),
                "updated_at": float(row["updated_at"]),
            }
            for row in rows
        ]

    def _cognition_evidence(
        self,
        conn: sqlite3.Connection,
        cognition_id: str,
    ) -> List[Dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT ce.memory_id, ce.stance, ce.weight, ce.relation_type,
                   ce.created_at, m.summary, m.kind, m.status AS memory_status
            FROM cognition_evidence ce
            JOIN long_horizon_memories m ON m.id = ce.memory_id
            WHERE ce.cognition_id = ?
            ORDER BY ce.weight DESC, ce.created_at ASC
            """,
            (str(cognition_id),),
        ).fetchall()
        return [
            {
                "memory_id": row["memory_id"],
                "stance": row["stance"],
                "weight": float(row["weight"]),
                "relation_type": row["relation_type"],
                "created_at": float(row["created_at"]),
                "summary": row["summary"],
                "kind": row["kind"],
                "memory_status": row["memory_status"],
            }
            for row in rows
        ]

    def _record_cognition_revision(
        self,
        conn: sqlite3.Connection,
        item: Dict[str, Any],
        *,
        action: str,
        reason: str,
        operator: str,
    ) -> int:
        cursor = conn.execute(
            """
            INSERT INTO cognition_revisions(
                cognition_id, version, action, snapshot_json,
                reason, operator, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(item["id"]),
                int(item.get("version") or 1),
                str(action),
                self._json_dump(item),
                str(reason or ""),
                str(operator or "system"),
                time.time(),
            ),
        )
        return int(cursor.lastrowid)

    def create_cognition(
        self,
        session_id: Optional[str],
        statement: str,
        *,
        evidence_memory_ids: Sequence[str],
        cognition_type: str = "induction",
        scope: str = "session",
        conditions: Sequence[str] = (),
        confidence: float = 0.55,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        clean_statement = re.sub(r"\s+", " ", str(statement or "")).strip()
        evidence_ids = list(dict.fromkeys(
            str(value).strip() for value in evidence_memory_ids if str(value).strip()
        ))
        if not clean_statement:
            raise ValueError("statement is required")
        if not evidence_ids:
            raise ValueError("at least one evidence memory is required")
        scope_name = str(scope or "session")
        scope_key = self._scope_key(session_id, scope_name)
        session_key = self._normalized_session_id(session_id, scope_name)
        placeholders = ",".join("?" for _ in evidence_ids)
        now = time.time()
        cognition_id = f"cg-{cognition_type}-{int(now * 1000)}-{uuid.uuid4().hex[:8]}"
        with self._db_lock, self._connect() as conn:
            evidence_rows = conn.execute(
                f"""
                SELECT * FROM long_horizon_memories
                WHERE id IN ({placeholders}) AND status = 'active' AND scope_key = ?
                """,
                (*evidence_ids, scope_key),
            ).fetchall()
            known_ids = {str(row["id"]) for row in evidence_rows}
            missing = [value for value in evidence_ids if value not in known_ids]
            if missing:
                raise ValueError(f"unknown or out-of-scope evidence memories: {', '.join(missing)}")
            conn.execute(
                """
                INSERT INTO cognitions(
                    id, scope_key, session_id, scope, cognition_type, statement,
                    conditions_json, confidence, status, created_at, updated_at,
                    use_count, version, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, 0, 1, ?)
                """,
                (
                    cognition_id,
                    scope_key,
                    session_key,
                    scope_name,
                    str(cognition_type or "induction")[:48],
                    clean_statement[:1200],
                    self._json_dump([str(value) for value in conditions if str(value).strip()][:16]),
                    max(0.05, min(0.99, float(confidence))),
                    now,
                    now,
                    self._json_dump({
                        "source": "cognition_engine",
                        "schema": "docthinker.cognition.v1",
                        **dict(metadata or {}),
                    }),
                ),
            )
            for memory_id in evidence_ids:
                conn.execute(
                    """
                    INSERT INTO cognition_evidence(
                        cognition_id, memory_id, stance, weight,
                        relation_type, created_at
                    ) VALUES (?, ?, 'supports', 1.0, 'derived_from', ?)
                    """,
                    (cognition_id, memory_id, now),
                )
            row = conn.execute("SELECT * FROM cognitions WHERE id = ?", (cognition_id,)).fetchone()
            item = self._row_to_cognition(row, self._cognition_evidence(conn, cognition_id))
            self._record_cognition_revision(
                conn,
                item,
                action="create",
                reason="derived_from_memory_evidence",
                operator="cognition_engine",
            )
        return item

    def list_cognitions(
        self,
        session_id: Optional[str] = None,
        *,
        scope: Optional[str] = None,
        status: str = "active",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        where = ["status = ?"]
        params: List[Any] = [str(status or "active")]
        scope_keys = self._scope_keys(session_id, scope)
        if scope_keys:
            where.append(f"scope_key IN ({','.join('?' for _ in scope_keys)})")
            params.extend(scope_keys)
        elif scope:
            return []
        params.append(max(0, int(limit)))
        with self._db_lock, self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM cognitions
                WHERE {' AND '.join(where)}
                ORDER BY updated_at DESC LIMIT ?
                """,
                params,
            ).fetchall()
            return [
                self._row_to_cognition(row, self._cognition_evidence(conn, str(row["id"])))
                for row in rows
            ]

    def retrieve_cognitions(
        self,
        session_id: Optional[str],
        query: str,
        *,
        scopes: Sequence[str],
        top_k: int,
        min_confidence: float,
        evidence_memory_ids: Sequence[str] = (),
    ) -> List[Dict[str, Any]]:
        query_tokens = self._tokens(query)
        evidence_seed_ids = {str(value) for value in evidence_memory_ids if str(value)}
        scope_keys = [self._scope_key(session_id, scope) for scope in scopes]
        if not scope_keys:
            return []
        placeholders = ",".join("?" for _ in scope_keys)
        with self._db_lock, self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM cognitions
                WHERE status = 'active'
                  AND scope_key IN ({placeholders})
                  AND confidence >= ?
                """,
                (*scope_keys, float(min_confidence)),
            ).fetchall()
            scored: List[tuple[float, Dict[str, Any]]] = []
            for row in rows:
                evidence = self._cognition_evidence(conn, str(row["id"]))
                item = self._row_to_cognition(row, evidence)
                cognition_tokens = self._tokens(
                    " ".join([
                        str(item.get("statement") or ""),
                        " ".join(str(value) for value in item.get("conditions") or []),
                    ])
                )
                overlap = len(query_tokens & cognition_tokens)
                evidence_overlap = len(evidence_seed_ids & set(item["evidence_memory_ids"]))
                if query_tokens and overlap == 0 and evidence_overlap == 0:
                    continue
                score = (
                    float(item["confidence"])
                    + overlap * 0.14
                    + min(0.35, evidence_overlap * 0.18)
                    + min(0.12, int(item.get("use_count") or 0) * 0.02)
                )
                item["score"] = round(score, 3)
                item["recall_origin"] = "evidence_graph" if evidence_overlap else "direct"
                item["matched_evidence_memory_ids"] = sorted(
                    evidence_seed_ids & set(item["evidence_memory_ids"])
                )
                scored.append((score, item))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            selected = [item for _, item in scored[: max(0, int(top_k))]]
            if selected:
                selected_ids = [str(item["id"]) for item in selected]
                conn.execute(
                    f"""
                    UPDATE cognitions SET use_count = use_count + 1, updated_at = updated_at
                    WHERE id IN ({','.join('?' for _ in selected_ids)})
                    """,
                    selected_ids,
                )
        return selected

    def build_cognition_instruction(
        self,
        matches: Sequence[Dict[str, Any]],
        *,
        limit: int,
    ) -> str:
        if not matches:
            return ""
        lines = [
            "## Derived cognition (separate from source memory)",
            "These are revisable conclusions derived from preserved memories. Apply only when their conditions fit, and keep their evidence trace visible.",
            "",
        ]
        for item in matches[: max(1, int(limit))]:
            conditions = "; ".join(str(value) for value in item.get("conditions") or [])
            evidence_ids = ", ".join(str(value) for value in item.get("evidence_memory_ids") or [])
            lines.append(
                f"- [{item.get('cognition_type', 'induction')}/{float(item.get('confidence') or 0):.2f}] "
                f"{str(item.get('statement') or '')[:500]}"
            )
            if conditions:
                lines.append(f"  conditions: {conditions[:300]}")
            if evidence_ids:
                lines.append(f"  evidence memories: {evidence_ids[:500]}")
        return "\n".join(lines)

    def update_cognition(
        self,
        cognition_id: str,
        patch: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        target = str(cognition_id or "").strip()
        allowed = {"statement", "conditions", "confidence", "status", "cognition_type", "metadata"}
        clean = {key: value for key, value in dict(patch or {}).items() if key in allowed}
        if not target or not clean:
            return None
        with self._db_lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM cognitions WHERE id = ?", (target,)).fetchone()
            if not row:
                return None
            item = self._row_to_cognition(row, self._cognition_evidence(conn, target))
            statement = re.sub(r"\s+", " ", str(clean.get("statement", item["statement"]))).strip()
            conditions = clean.get("conditions", item["conditions"])
            if not isinstance(conditions, (list, tuple)):
                conditions = item["conditions"]
            try:
                confidence = max(0.05, min(0.99, float(clean.get("confidence", item["confidence"]))))
            except (TypeError, ValueError):
                confidence = float(item["confidence"])
            metadata = dict(item.get("metadata") or {})
            if isinstance(clean.get("metadata"), dict):
                metadata.update(clean["metadata"])
            version = int(item["version"]) + 1
            conn.execute(
                """
                UPDATE cognitions SET cognition_type = ?, statement = ?,
                    conditions_json = ?, confidence = ?, status = ?, updated_at = ?,
                    version = ?, metadata_json = ? WHERE id = ?
                """,
                (
                    str(clean.get("cognition_type", item["cognition_type"]))[:48],
                    statement[:1200],
                    self._json_dump([str(value) for value in conditions if str(value).strip()][:16]),
                    confidence,
                    str(clean.get("status", item["status"]))[:32],
                    time.time(),
                    version,
                    self._json_dump(metadata),
                    target,
                ),
            )
            updated_row = conn.execute("SELECT * FROM cognitions WHERE id = ?", (target,)).fetchone()
            updated = self._row_to_cognition(updated_row, self._cognition_evidence(conn, target))
            self._record_cognition_revision(
                conn,
                updated,
                action="update",
                reason="user_controlled_cognition_edit",
                operator="cognition_editor",
            )
        return updated

    def list_cognition_revisions(
        self,
        cognition_id: str,
        *,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        with self._db_lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM cognition_revisions WHERE cognition_id = ?
                ORDER BY revision_id DESC LIMIT ?
                """,
                (str(cognition_id), max(0, int(limit))),
            ).fetchall()
        return [
            {
                "revision_id": int(row["revision_id"]),
                "cognition_id": row["cognition_id"],
                "version": int(row["version"]),
                "action": row["action"],
                "snapshot": self._json_load(row["snapshot_json"], {}),
                "reason": row["reason"],
                "operator": row["operator"],
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    def stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        items = self.list_insights(session_id=session_id, limit=100000)
        by_kind: Dict[str, int] = {}
        for item in items:
            kind = str(item.get("kind") or "insight")
            by_kind[kind] = by_kind.get(kind, 0) + 1
        with self._db_lock, self._connect() as conn:
            revision_count = int(
                conn.execute("SELECT COUNT(*) AS count FROM long_horizon_revisions").fetchone()["count"]
            )
            edge_count = int(
                conn.execute(
                    "SELECT COUNT(*) AS count FROM long_horizon_edges WHERE status = 'active'"
                ).fetchone()["count"]
            )
            cognition_count = int(
                conn.execute(
                    "SELECT COUNT(*) AS count FROM cognitions WHERE status = 'active'"
                ).fetchone()["count"]
            )
        return {
            "enabled": True,
            "system": "long_horizon_memory",
            "storage": "sqlite",
            "persistent": True,
            "database_path": str(self.db_path),
            "session_id": session_id,
            "count": len(items),
            "by_kind": by_kind,
            "recent": items[:12],
            "revision_count": revision_count,
            "edge_count": edge_count,
            "cognition_count": cognition_count,
            "last_write_decision": self.last_write_decision(),
        }
