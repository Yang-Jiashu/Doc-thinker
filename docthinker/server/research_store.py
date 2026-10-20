"""Bounded, private research documents and explicitly confirmed project memory.

This is deterministic storage, not an agent, extractor or model client. Documents
are deduplicated *within* a project, never across projects. Extraction jobs can be
reclaimed after a worker restart because extraction has no external side effects.
Memory is never learned implicitly: every mutation must be initiated by a user.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import time
import uuid

from .research_memory import classify_memory_query
from .research_retrieval import lexical_tokens
from .research_source_contract import MAX_ANSWER_SOURCES, SOURCE_LABEL_PATTERN

MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
MAX_DOCUMENT_TEXT_CHARS = 300_000
MAX_DOCUMENT_CHUNKS = 600
MAX_CHUNK_CHARS = 8_000
MAX_DOCUMENTS_PER_SESSION = 20
MAX_DOCUMENTS_TOTAL = 100
MAX_DOCUMENT_RECORDS = 1_000
MAX_MEMORY_TEXT_CHARS = 2_000
MAX_MEMORY_SOURCES = 20
MAX_MEMORIES_PER_SESSION = 100
MAX_MEMORIES_TOTAL = 500
MAX_MEMORY_VERSIONS = 5_000
# Memory counts above remain legacy pagination constants rather than lifetime
# capacities; memory growth is governed by storage bytes. Document admission
# still enforces the 20/session and 100/worker limits, and source snapshots stay
# within the 20-document view. Do not imply an unbounded paper catalog here.
DEFAULT_STORAGE_BYTES = 512 * 1024 * 1024
MAX_MEMORY_PAGE = 100
MAX_MEMORY_OFFSET = 2_147_483_647
MAX_MEMORY_SEARCH_TERMS = 64
_UNSET = object()
MAX_MEMORY_CONTEXT_CHARS = 6_000
MAX_ANSWER_RECORDS = 2_000
MAX_ANSWER_LOOKUP = 200
ANSWER_READ_BATCH = 384  # Including scope binds, each query stays below 400 variables.
MAX_ANSWER_CONTEXT_BYTES = 2_048
ANSWER_GAP_MESSAGES = {
    "final_need_witness_omitted": "部分已找到的需求见证未进入最终请求，不能视为已完成覆盖。",
    "final_need_witness_unmatched": "部分需求未在当前原文窗口中找到字面见证，不能据此猜补，也不代表全文没有证据。",
    "literal_coverage_scan_limited": "本地原文筛选达到扫描上限，未覆盖全部资料；没有找到不代表不存在。",
    "semantic_retrieval_unavailable": "本地语义检索暂不可用，已降级为词面检索；跨语言和同义表达可能遗漏。",
    "semantic_retrieval_windowed": "本地语义检索达到扫描窗口上限，不代表已完整覆盖问题或全部资料。",
    "insufficient_evidence": "未找到匹配的原文依据；请补充资料或换用原文术语。",
    "bounded_overview": "仅抽取各资料的开头片段，不代表全文总结或完整比较。",
    "uncovered_query_terms": "部分问题用词没有字面匹配，可能涉及跨语言或同义表达，不等于概念缺失；仍须核对原文，不能猜测补齐因果链。",
    "evidence_coverage_proxy": "检索覆盖是相关性代理，不代表关键事实完整覆盖或引用已充分支持结论。",
    "fallback_metadata_filtered": "辅助筛选不可用；仅从原有来源移除明确的纯元信息，未验证其余证据的相关性或完整性。",
    "evidence_facets_uncovered": "部分子问题缺少足够相关的原文，不应将缺失证据补成结论。",
    "evidence_anchors_uncovered": "部分显式技术要点缺少原文细节，不能用只列出名称的概述代替。",
    "documents_not_selected": "部分资料未进入本次有限上下文；未引用不等于资料不支持或反对。",
    "excerpt_truncated": "部分依据仅为连续节选，限定条件可能在节选之外。",
    "source_boundary_unverified": "部分来源末尾没有完整句界，可能跨页或截断；未提供的续文和条件仍需核对。",
    "source_antecedent_unverified": "部分节选开头回指缺少先行对象；不能从下一实验或模型常识补造其对象和条件。",
    "complete_statement_exceeds_budget": "部分匹配原句及相邻限定条件无法完整放入预算，已跳过；不能把缺失部分当成已核实。",
    "duplicate_chunks_removed": "重复原文已合并，不能当作多份独立证据。",
    "confirmed_memory_budget_omitted": "部分有效确认记忆因上下文预算未载入本次回答；原记忆未被截断或删除。",
    "context_budget_omitted": "为遵守输入预算，部分历史或确认记忆未加入本次回答。",
    "graph_association_not_proof": "图路径仅用于查找关联原文，不能据此断言完整因果链已成立。",
    "graph_window_limited": "图索引达到窗口上限，本次没有遍历全部资料。",
    "graph_hub_omitted": "为避免无关内容扩散，跳过了连接过多的节点。",
    "graph_search_limited": "图搜索达到步数预算，不能将未找到当作不存在。",
    "graph_no_source_seed": "没有可用于图扩展的有效原文起点。",
    "graph_no_connected_path": "本次有限图搜索未找到可追加的关联原文。",
    "graph_capacity_reached": "关联原文无法完整放入本次来源预算，已省略，未挤掉原有证据。",
    "graph_source_unavailable": "部分图关联来源已不可用，本次未使用。",
    "graph_counterevidence_in_path": "关联路径含否定信息，不能把路径方向当作支持结论。",
    "graph_path_conditions_differ": "关联原文的成立条件不同，需要逐项核对，不能直接串成因果链。",
    "graph_parser_is_partial": "本地图仅识别部分明确关系表达，不能覆盖论文中的全部语义。",
    "graph_excerpt_windowed": "关联依据仅包含完整陈述及相邻显式限定的连续节选，窗口之外仍可能有其他条件。",
    "graph_path_window_incomplete": "连接所需的部分陈述不在已保留的原文窗口中，未把未展示的陈述当作证据。",
}
ANSWER_MEMORY_STRATEGIES = {
    "disabled", "lexical_bm25_with_preferences", "recent_project_context", "explicit_preferences", "explicit_task_memory",
}
ANSWER_MEMORY_STRATEGIES |= {'indexed_temporal_' + value for value in ANSWER_MEMORY_STRATEGIES if value != 'disabled'}
UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
SESSION_PATTERN = r"#[0-9]{5}"
DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md"}
MEMORY_KINDS = {"preference", "decision", "finding"}


class ResearchStoreError(ValueError):
    """A safe, caller-correctable storage error without private path details."""


class ResearchLimitError(ResearchStoreError):
    pass


class ResearchNotFoundError(ResearchStoreError):
    pass


class ResearchRevisionConflictError(ResearchStoreError):
    pass


def _session(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(SESSION_PATTERN, value):
        raise ResearchStoreError("Invalid project reference")
    return value


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(UUID_PATTERN, value):
        raise ResearchStoreError("Invalid record reference")
    return value


def _integer(value, minimum: int, maximum: int, message: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ResearchStoreError(message)
    return value


def _text(value: str, maximum: int, message: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ResearchStoreError(message)
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ResearchStoreError(message) from error
    if "\x00" in value:
        raise ResearchStoreError(message)
    return value


class ResearchStore:
    def __init__(self, root: Path, *, clock=time.time, boot_id: str | None = None,
                 max_storage_bytes: int = DEFAULT_STORAGE_BYTES):
        self.root = Path(root)
        if (not self.root.is_absolute() or self.root.is_symlink()
                or not self.root.is_dir() or self.root.resolve(strict=True) != self.root
                or self.root.stat().st_uid != os.getuid()):
            raise ResearchStoreError("An existing owned worker directory is required")
        self.directory = self.root / "_system" / "research"
        self.files_directory = self.directory / "files"
        self.path = self.directory / "research.db"
        self.clock = clock
        self.boot_id = boot_id or str(uuid.uuid4())
        self.max_storage_bytes = _integer(max_storage_bytes, 65_536, 1 << 50, "Invalid research storage quota")

    @staticmethod
    def _check_directory(path: Path, *, private: bool) -> bool:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return False
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or path.resolve(strict=True) != path or (private and info.st_mode & 0o077)
                or (not private and info.st_mode & 0o022)):
            raise ResearchStoreError("Research storage must use owned private directories")
        return True

    def _check_layout(self) -> bool:
        # Inspect every parent without following a request-controlled path.
        if not self._check_directory(self.root, private=False):
            raise ResearchStoreError("Worker directory is unavailable")
        system = self.root / "_system"
        if not self._check_directory(system, private=False):
            return False
        if not self._check_directory(self.directory, private=True):
            return False
        if not self._check_directory(self.files_directory, private=True):
            return False
        return True

    @staticmethod
    def _check_regular(path: Path) -> bool:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return False
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != os.getuid() or info.st_mode & 0o077):
            raise ResearchStoreError("Research storage must use owner-only regular files")
        return True

    def _exists(self) -> bool:
        return self._check_layout() and self._check_regular(self.path)

    def _ensure(self) -> None:
        for path, private in ((self.root / "_system", False), (self.directory, True),
                              (self.files_directory, True)):
            if not self._check_directory(path, private=private):
                try:
                    path.mkdir(mode=0o700)
                except FileExistsError:
                    pass
                self._check_directory(path, private=private)
        if not self._check_regular(self.path):
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(fd)
            self._check_regular(self.path)

    @staticmethod
    def _schema(db) -> None:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS research_documents (
                document_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                filename TEXT NOT NULL, extension TEXT NOT NULL, content_hash TEXT NOT NULL,
                byte_size INTEGER NOT NULL, state TEXT NOT NULL, boot_id TEXT,
                attempt INTEGER NOT NULL DEFAULT 0, chunk_count INTEGER NOT NULL DEFAULT 0,
                text_chars INTEGER NOT NULL DEFAULT 0, error_code TEXT,
                created_at REAL NOT NULL, updated_at REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS research_documents_project
                ON research_documents(session_id,state,created_at);
            CREATE TABLE IF NOT EXISTS research_chunks (
                chunk_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                document_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
                text TEXT NOT NULL, page INTEGER,
                UNIQUE(document_id,ordinal));
            CREATE INDEX IF NOT EXISTS research_chunks_project
                ON research_chunks(session_id,document_id,ordinal);
            CREATE TABLE IF NOT EXISTS research_memories (
                memory_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                kind TEXT NOT NULL, text TEXT NOT NULL, source_chunk_ids TEXT NOT NULL,
                source_answer_id TEXT, revision INTEGER NOT NULL, status TEXT NOT NULL,
                created_at REAL NOT NULL, updated_at REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS research_memories_project
                ON research_memories(session_id,status,updated_at);
            CREATE TABLE IF NOT EXISTS research_memory_versions (
                memory_id TEXT NOT NULL, session_id TEXT NOT NULL, revision INTEGER NOT NULL,
                kind TEXT NOT NULL, text TEXT NOT NULL, source_chunk_ids TEXT NOT NULL,
                source_answer_id TEXT, status TEXT NOT NULL, created_at REAL NOT NULL,
                PRIMARY KEY(memory_id,revision));
            CREATE TABLE IF NOT EXISTS research_answers (
                answer_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                source_refs TEXT NOT NULL, validation TEXT NOT NULL,
                answer_mode TEXT NOT NULL, created_at REAL NOT NULL,
                context_metadata_json TEXT NOT NULL DEFAULT '{}');
            CREATE INDEX IF NOT EXISTS research_answers_project
                ON research_answers(session_id,created_at);
        """)
        # Additive migration only on writes. SELECT-only status/history calls
        # remain compatible with the original six-column table without mutation.
        if "context_metadata_json" not in {row[1] for row in db.execute("PRAGMA table_info(research_answers)")}:
            try:
                db.execute("ALTER TABLE research_answers ADD COLUMN context_metadata_json TEXT NOT NULL DEFAULT '{}'")
            except sqlite3.OperationalError:
                # Concurrent first writers may both observe the old schema.
                if "context_metadata_json" not in {row[1] for row in db.execute("PRAGMA table_info(research_answers)")}:
                    raise
        ResearchStore._memory_schema(db)

    @staticmethod
    def _memory_schema(db) -> None:
        """One-time additive migration, then incremental indexes on each write.

        The FTS row contains tokenized text rather than another authoritative
        note. Session is an indexed FTS term, so other projects never become
        candidate rows. Historical revisions share the same index.
        """
        db.execute("BEGIN IMMEDIATE")
        needs_index = not db.execute("SELECT 1 FROM sqlite_master WHERE name='research_memory_search'").fetchone()
        for table in ("research_memories", "research_memory_versions"):
            columns = {row[1] for row in db.execute("PRAGMA table_info(" + table + ")")}
            for name, declaration in (("valid_from", "REAL"), ("valid_until", "REAL"),
                                      ("recorded_until", "REAL")):
                if name not in columns:
                    db.execute("ALTER TABLE " + table + " ADD COLUMN " + name + " " + declaration)
                    if name == "valid_from":
                        timestamp = "updated_at" if table == "research_memories" else "created_at"
                        db.execute("UPDATE " + table + " SET valid_from=" + timestamp)
        statements = """
            CREATE INDEX IF NOT EXISTS research_memory_versions_time
                ON research_memory_versions(session_id,valid_from,memory_id,revision);
            CREATE INDEX IF NOT EXISTS research_memory_versions_identity_time
                ON research_memory_versions(memory_id,revision,valid_from,valid_until);
            CREATE INDEX IF NOT EXISTS research_memory_versions_kind_time
                ON research_memory_versions(session_id,kind,valid_from DESC,memory_id DESC,revision DESC);
            CREATE INDEX IF NOT EXISTS research_memories_kind
                ON research_memories(session_id,kind,status,updated_at DESC,memory_id DESC);
            CREATE VIRTUAL TABLE IF NOT EXISTS research_memory_search USING fts5(
                memory_id UNINDEXED, revision UNINDEXED, scope, tokens,
                tokenize='unicode61 remove_diacritics 0');
            CREATE TABLE IF NOT EXISTS research_memory_sources (
                memory_id TEXT NOT NULL, revision INTEGER NOT NULL,
                session_id TEXT NOT NULL, chunk_id TEXT NOT NULL,
                PRIMARY KEY(memory_id,revision,chunk_id));
            CREATE INDEX IF NOT EXISTS research_memory_sources_chunk
                ON research_memory_sources(session_id,chunk_id,memory_id,revision);
        """
        for statement in statements.split(";"):
            if statement.strip():
                db.execute(statement)
        if needs_index:
            # Legacy revisions used recording time as their only timestamp.
            # Recover half-open validity intervals once; do not scan on queries.
            db.execute("UPDATE research_memory_versions AS old SET valid_until=(SELECT min(new.created_at) "
                       "FROM research_memory_versions AS new WHERE new.memory_id=old.memory_id AND new.revision>old.revision),"
                       "recorded_until=(SELECT min(new.created_at) FROM research_memory_versions AS new "
                       "WHERE new.memory_id=old.memory_id AND new.revision>old.revision)")
            for row in db.execute("SELECT * FROM research_memory_versions WHERE status!='deleted'"):
                ResearchStore._index_version(db, row)
        db.commit()

    @staticmethod
    def _index_version(db, row) -> None:
        rowid = db.execute("SELECT rowid FROM research_memory_versions WHERE memory_id=? AND revision=?",
                           (row["memory_id"], row["revision"])).fetchone()[0]
        db.execute("DELETE FROM research_memory_search WHERE rowid=?", (rowid,))
        db.execute("DELETE FROM research_memory_sources WHERE memory_id=? AND revision=?",
                   (row["memory_id"], row["revision"]))
        if row["status"] == "deleted":
            return
        db.execute("INSERT INTO research_memory_search(rowid,memory_id,revision,scope,tokens) VALUES(?,?,?,?,?)",
                   (rowid, row["memory_id"], row["revision"], "p" + row["session_id"][1:],
                    " ".join(lexical_tokens(row["text"]))))
        db.executemany("INSERT INTO research_memory_sources VALUES(?,?,?,?)",
                       [(row["memory_id"], row["revision"], row["session_id"], chunk_id)
                        for chunk_id in json.loads(row["source_chunk_ids"])])

    def _ensure_memory_schema(self) -> None:
        with self._read() as db:
            needs_migration = db is not None and not db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='research_memory_search'").fetchone()
        if needs_migration:
            with self._write():
                pass

    def _storage_capacity(self, db=None) -> dict:
        if db is None:
            used = self.path.stat().st_size if self._exists() else 0
        else:
            used = (db.execute("PRAGMA page_count").fetchone()[0]
                    - db.execute("PRAGMA freelist_count").fetchone()[0]) * db.execute("PRAGMA page_size").fetchone()[0]
        return {"available": used + 16_384 < self.max_storage_bytes,
                "remaining": max(0, self.max_storage_bytes - used), "limit": self.max_storage_bytes,
                "unit": "bytes", "used": used}

    def _require_storage(self, db) -> None:
        if not self._storage_capacity(db)["available"]:
            raise ResearchLimitError("Research storage quota reached; expand storage or remove data")

    @contextmanager
    def _write(self):
        self._ensure()
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=5)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=DELETE")
            # Logical deletion also clears SQLite freed payload bytes. This is
            # not an erasure guarantee for filesystems, journals or backups.
            db.execute("PRAGMA secure_delete=ON")
            self._schema(db)
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    @contextmanager
    def _read(self, *, timeout=5):
        if not self._exists():
            yield None
            return
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=timeout)) as db:
            db.row_factory = sqlite3.Row
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='research_documents'").fetchone():
                yield None
            else:
                yield db

    def _document(self, row, *, deduplicated: bool = False) -> dict:
        state = row["state"]
        if state == "processing" and row["boot_id"] != self.boot_id:
            state = "queued"
        return {"id": row["document_id"], "document_id": row["document_id"],
                "session_id": row["session_id"], "filename": row["filename"],
                "status": state, "state": state, "byte_size": row["byte_size"],
                "chunk_count": row["chunk_count"], "text_chars": row["text_chars"],
                "attempt": row["attempt"], "error_code": row["error_code"],
                "created_at": row["created_at"], "updated_at": row["updated_at"],
                "deduplicated": deduplicated}

    @staticmethod
    def _required_document(db, session_id: str, document_id: str):
        row = db.execute("SELECT * FROM research_documents WHERE session_id=? AND document_id=? AND state!='deleted'",
                         (session_id, document_id)).fetchone() if db is not None else None
        if row is None:
            raise ResearchNotFoundError("Document not found in this project")
        return row

    def accept_document(self, session_id: str, filename: str, content: bytes) -> dict:
        _session(session_id)
        _text(filename, 120, "Invalid document filename")
        if (filename != filename.strip() or filename in {".", ".."}
                or "/" in filename or "\\" in filename
                or any(ord(character) < 32 or ord(character) == 127 for character in filename)):
            raise ResearchStoreError("Filename is a display name, not a path")
        extension = Path(filename).suffix.lower()
        if extension not in DOCUMENT_EXTENSIONS:
            raise ResearchStoreError("Only PDF, TXT and MD documents are supported")
        if not isinstance(content, bytes) or not content or len(content) > MAX_DOCUMENT_BYTES:
            raise ResearchLimitError("Document must contain 1 to 10 MiB of data")
        if extension == ".pdf":
            if not content.startswith(b"%PDF-"):
                raise ResearchStoreError("Invalid PDF signature")
        else:
            try:
                decoded = content.decode("utf-8-sig", errors="strict")
            except UnicodeDecodeError as error:
                raise ResearchStoreError("Text documents must be UTF-8") from error
            _text(decoded, MAX_DOCUMENT_TEXT_CHARS, "Text document is empty or exceeds its extraction limit")
        content_hash = hashlib.sha256(content).hexdigest()
        created_path = None
        try:
            with self._write() as db:
                existing = db.execute("SELECT * FROM research_documents WHERE session_id=? AND content_hash=? AND state!='deleted' ORDER BY created_at LIMIT 1",
                                      (session_id, content_hash)).fetchone()
                if existing:
                    return self._document(existing, deduplicated=True)
                if db.execute("SELECT count(*) FROM research_documents WHERE session_id=? AND state!='deleted'", (session_id,)).fetchone()[0] >= MAX_DOCUMENTS_PER_SESSION:
                    raise ResearchLimitError("Project document limit reached")
                if db.execute("SELECT count(*) FROM research_documents WHERE state!='deleted'").fetchone()[0] >= MAX_DOCUMENTS_TOTAL:
                    raise ResearchLimitError("Worker document limit reached")
                if db.execute("SELECT count(*) FROM research_documents").fetchone()[0] >= MAX_DOCUMENT_RECORDS:
                    raise ResearchLimitError("Document metadata capacity reached")
                document_id, now = str(uuid.uuid4()), self.clock()
                created_path = self.files_directory / (document_id + extension)
                fd = os.open(created_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                db.execute("INSERT INTO research_documents(document_id,session_id,filename,extension,content_hash,byte_size,state,created_at,updated_at) VALUES(?,?,?,?,?,?,'queued',?,?)",
                           (document_id, session_id, filename, extension, content_hash, len(content), now, now))
                return self._document(db.execute("SELECT * FROM research_documents WHERE document_id=?", (document_id,)).fetchone())
        except BaseException:
            if created_path is not None and created_path.exists() and not created_path.is_symlink():
                created_path.unlink()
            raise

    def list_documents(self, session_id: str) -> list[dict]:
        _session(session_id)
        with self._read() as db:
            if db is None:
                return []
            rows = db.execute("SELECT * FROM research_documents WHERE session_id=? AND state!='deleted' ORDER BY created_at,document_id LIMIT ?",
                              (session_id, MAX_DOCUMENTS_PER_SESSION)).fetchall()
            return [self._document(row) for row in rows]

    def document_snapshots(self, session_id: str, document_ids=None) -> dict:
        """Content/state identities for durable tasks, no source text or paths.

        A task's declared corpus is an input dependency, not only the excerpts
        ultimately selected. New/removed/parsed documents therefore invalidate
        project-wide work; explicit scopes protect unrelated completed work.
        """
        _session(session_id)
        if document_ids is not None:
            if not isinstance(document_ids, (list, tuple)) or not 1 <= len(document_ids) <= MAX_DOCUMENTS_PER_SESSION:
                raise ResearchStoreError("Invalid document scope")
            for identifier in document_ids:
                _identifier(identifier)
            if len(set(document_ids)) != len(document_ids):
                raise ResearchStoreError("Invalid document scope")
        with self._read() as db:
            return self._document_snapshots_in(db, session_id, document_ids)

    @staticmethod
    def _document_snapshots_in(db, session_id, document_ids=None):
        """Connection-local variant for atomic multi-input publication checks."""
        if db is None:
            return {}
        rows = db.execute("SELECT document_id,content_hash,state,chunk_count,text_chars,updated_at "
                          "FROM research_documents WHERE session_id=? AND state!='deleted' "
                          "ORDER BY document_id LIMIT ?", (session_id, MAX_DOCUMENTS_PER_SESSION)).fetchall()
        allowed = set(document_ids) if document_ids is not None else None
        return {row['document_id']: hashlib.sha256(json.dumps(
            [row['content_hash'], row['state'], row['chunk_count'], row['text_chars'], row['updated_at']],
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            for row in rows if allowed is None or row['document_id'] in allowed}

    def get_document(self, session_id: str, document_id: str) -> dict:
        _session(session_id); _identifier(document_id)
        with self._read() as db:
            return self._document(self._required_document(db, session_id, document_id))

    def pending_documents(self, limit: int = 20) -> list[dict]:
        _integer(limit, 1, MAX_DOCUMENTS_TOTAL, "Invalid job limit")
        with self._read() as db:
            if db is None:
                return []
            rows = db.execute("SELECT * FROM research_documents WHERE state='queued' OR (state='processing' AND boot_id!=?) ORDER BY created_at,document_id LIMIT ?",
                              (self.boot_id, limit)).fetchall()
            return [self._document(row) for row in rows]

    def claim_document(self, session_id: str, document_id: str) -> dict | None:
        _session(session_id); _identifier(document_id)
        with self._write() as db:
            row = self._required_document(db, session_id, document_id)
            if row["state"] not in {"queued", "processing"} or (row["state"] == "processing" and row["boot_id"] == self.boot_id):
                return None
            db.execute("UPDATE research_documents SET state='processing',boot_id=?,attempt=attempt+1,error_code=NULL,updated_at=? WHERE document_id=?",
                       (self.boot_id, self.clock(), document_id))
            return self._document(db.execute("SELECT * FROM research_documents WHERE document_id=?", (document_id,)).fetchone())

    def retry_document(self, session_id: str, document_id: str) -> dict:
        """Explicit retry of a failed local extraction, never a model retry."""
        _session(session_id); _identifier(document_id)
        with self._write() as db:
            row = self._required_document(db, session_id, document_id)
            if row["state"] != "failed":
                raise ResearchStoreError("Only failed document extraction can be retried")
            db.execute("UPDATE research_documents SET state='queued',boot_id=NULL,error_code=NULL,updated_at=? WHERE document_id=?",
                       (self.clock(), document_id))
            return self._document(db.execute("SELECT * FROM research_documents WHERE document_id=?", (document_id,)).fetchone())

    def read_document_bytes(self, session_id: str, document_id: str) -> bytes:
        _session(session_id); _identifier(document_id)
        with self._read() as db:
            row = self._required_document(db, session_id, document_id)
            path = self.files_directory / (document_id + row["extension"])
            if not self._check_regular(path):
                raise ResearchStoreError("Document source is unavailable")
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as handle:
                info = os.fstat(handle.fileno())
                if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                        or info.st_uid != os.getuid() or info.st_mode & 0o077
                        or info.st_size != row["byte_size"]):
                    raise ResearchStoreError("Document source is unavailable")
                content = handle.read(MAX_DOCUMENT_BYTES + 1)
            if len(content) != row["byte_size"] or hashlib.sha256(content).hexdigest() != row["content_hash"]:
                raise ResearchStoreError("Document source failed integrity validation")
            return content

    def complete_document(self, session_id: str, document_id: str, chunks: list[dict]) -> dict:
        _session(session_id); _identifier(document_id)
        if not isinstance(chunks, list) or not 1 <= len(chunks) <= MAX_DOCUMENT_CHUNKS:
            raise ResearchLimitError("Extraction must produce 1 to 600 chunks")
        normalized, total = [], 0
        for ordinal, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ResearchStoreError("Invalid extraction chunk")
            text = _text(chunk.get("text", chunk.get("content")), MAX_CHUNK_CHARS, "Invalid extraction chunk text")
            total += len(text)
            if total > MAX_DOCUMENT_TEXT_CHARS:
                raise ResearchLimitError("Extracted document text exceeds its limit")
            page = chunk.get("page")
            if page is not None:
                _integer(page, 1, 100_000, "Invalid source page")
            normalized.append((str(uuid.uuid4()), session_id, document_id, ordinal, text, page))
        with self._write() as db:
            row = self._required_document(db, session_id, document_id)
            if row["state"] != "processing" or row["boot_id"] != self.boot_id:
                raise ResearchRevisionConflictError("Document extraction claim is no longer current")
            db.execute("DELETE FROM research_chunks WHERE document_id=?", (document_id,))
            db.executemany("INSERT INTO research_chunks VALUES(?,?,?,?,?,?)", normalized)
            db.execute("UPDATE research_documents SET state='ready',chunk_count=?,text_chars=?,error_code=NULL,updated_at=? WHERE document_id=?",
                       (len(normalized), total, self.clock(), document_id))
            return self._document(db.execute("SELECT * FROM research_documents WHERE document_id=?", (document_id,)).fetchone())

    def fail_document(self, session_id: str, document_id: str, error_code: str = "extraction_failed") -> dict:
        _session(session_id); _identifier(document_id)
        if not isinstance(error_code, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", error_code):
            raise ResearchStoreError("Invalid extraction error code")
        with self._write() as db:
            row = self._required_document(db, session_id, document_id)
            if row["state"] != "processing" or row["boot_id"] != self.boot_id:
                raise ResearchRevisionConflictError("Document extraction claim is no longer current")
            db.execute("UPDATE research_documents SET state='failed',error_code=?,updated_at=? WHERE document_id=?",
                       (error_code, self.clock(), document_id))
            return self._document(db.execute("SELECT * FROM research_documents WHERE document_id=?", (document_id,)).fetchone())

    def source_chunks(self, session_id: str, chunk_ids: list[str] | None = None, *,
                      document_id: str | None = None, limit: int = 600, offset: int = 0) -> list[dict]:
        _session(session_id)
        _integer(limit, 1, MAX_DOCUMENT_CHUNKS, "Invalid chunk page size")
        _integer(offset, 0, MAX_DOCUMENTS_PER_SESSION * MAX_DOCUMENT_CHUNKS, "Invalid chunk offset")
        if document_id is not None:
            _identifier(document_id)
        if chunk_ids is not None:
            if not isinstance(chunk_ids, list) or len(chunk_ids) > MAX_DOCUMENT_CHUNKS:
                raise ResearchLimitError("Too many chunk references")
            for chunk_id in chunk_ids:
                _identifier(chunk_id)
            if not chunk_ids:
                return []
        with self._read() as db:
            if db is None:
                return []
            sql = ("SELECT c.*,d.filename,d.content_hash FROM research_chunks c JOIN research_documents d "
                   "ON d.document_id=c.document_id WHERE c.session_id=? AND d.session_id=? AND d.state='ready'")
            parameters = [session_id, session_id]
            if document_id is not None:
                sql += " AND c.document_id=?"
                parameters.append(document_id)
            if chunk_ids is not None:
                sql += " AND c.chunk_id IN (" + ",".join("?" for _ in chunk_ids) + ")"
                parameters.extend(chunk_ids)
            sql += " ORDER BY d.created_at,d.document_id,c.ordinal LIMIT ? OFFSET ?"
            parameters.extend([limit, offset])
            rows = db.execute(sql, parameters).fetchall()
            return [{"id": row["chunk_id"], "chunk_id": row["chunk_id"], "document_id": row["document_id"],
                     "filename": row["filename"], "ordinal": row["ordinal"], "page": row["page"], "content_hash": row["content_hash"],
                     "text": row["text"], "content": row["text"]} for row in rows]

    def all_source_chunks(self, session_id: str, *, document_ids=None) -> list[dict]:
        """One live SELECT snapshot, not 20 independently sorted OFFSET pages.

        The old 12,000-chunk envelope/order is unchanged. Explicit scope is
        applied before materializing text; no cache grants access to documents.
        """
        _session(session_id)
        if document_ids is not None:
            if not isinstance(document_ids, (list, tuple)) or not 1 <= len(document_ids) <= MAX_DOCUMENTS_PER_SESSION:
                raise ResearchLimitError('Invalid source document scope')
            for identifier in document_ids:
                _identifier(identifier)
            if len(set(document_ids)) != len(document_ids):
                raise ResearchStoreError('Duplicate source document scope')
        with self._read() as db:
            if db is None:
                return []
            sql = ('SELECT c.*,d.filename,d.content_hash FROM research_chunks c '
                   'JOIN research_documents d ON d.document_id=c.document_id '
                   "WHERE c.session_id=? AND d.session_id=? AND d.state='ready'")
            parameters = [session_id, session_id]
            if document_ids is not None:
                sql += ' AND c.document_id IN (' + ','.join('?' for _ in document_ids) + ')'
                parameters.extend(document_ids)
            sql += ' ORDER BY d.created_at,d.document_id,c.ordinal LIMIT ?'
            parameters.append(MAX_DOCUMENTS_PER_SESSION * MAX_DOCUMENT_CHUNKS)
            rows = db.execute(sql, parameters).fetchall()
            return [{"id": row["chunk_id"], "chunk_id": row["chunk_id"], "document_id": row["document_id"],
                     "filename": row["filename"], "ordinal": row["ordinal"], "page": row["page"], "content_hash": row["content_hash"],
                     "text": row["text"], "content": row["text"]} for row in rows]

    @staticmethod
    def _memory_answer_sources(db, session_id: str, answer_id: str) -> list[str]:
        """Resolve provenance only inside this private project's answer ledger."""
        _identifier(answer_id)
        answer = db.execute("SELECT source_refs FROM research_answers WHERE session_id=? AND answer_id=?",
                            (session_id, answer_id)).fetchone()
        if answer is None:
            raise ResearchStoreError("Memory source answer must belong to this project")
        try:
            refs = json.loads(answer["source_refs"])
            if (not isinstance(refs, list) or len(refs) > MAX_ANSWER_SOURCES
                    or not all(isinstance(ref, dict) and "id" in ref for ref in refs)):
                raise ValueError
            return list(dict.fromkeys(_identifier(ref["id"]) for ref in refs))
        except (ValueError, TypeError, KeyError):
            raise ResearchStoreError("Memory source answer references are unavailable") from None

    def _memory_payload(self, db, session_id: str, kind: str, text: str,
                        source_chunk_ids: list[str] | None, source_answer_id: str | None):
        if kind not in MEMORY_KINDS:
            raise ResearchStoreError("Invalid confirmed memory kind")
        _text(text, MAX_MEMORY_TEXT_CHARS, "Memory must contain 1 to 2000 characters")
        source_chunk_ids = [] if source_chunk_ids is None else source_chunk_ids
        if not isinstance(source_chunk_ids, list) or len(source_chunk_ids) > MAX_MEMORY_SOURCES:
            raise ResearchLimitError("Too many memory source references")
        for chunk_id in source_chunk_ids:
            _identifier(chunk_id)
        if source_answer_id is not None:
            # Remembering an answer must retain its document dependencies.
            # Otherwise omitting source_chunk_ids bypasses source withdrawal.
            source_chunk_ids = source_chunk_ids + self._memory_answer_sources(db, session_id, source_answer_id)
            from .research_memory_dependencies import source_answer_memory_live
            if not source_answer_memory_live(db, session_id, source_answer_id, at=self.clock()):
                raise ResearchStoreError("Memory source answer has unavailable memory dependencies")
        source_chunk_ids = list(dict.fromkeys(source_chunk_ids))
        if len(source_chunk_ids) > MAX_MEMORY_SOURCES:
            raise ResearchLimitError("Too many memory source references")
        if source_chunk_ids:
            placeholders = ",".join("?" for _ in source_chunk_ids)
            found = db.execute("SELECT c.chunk_id FROM research_chunks c JOIN research_documents d ON d.document_id=c.document_id "
                               "WHERE c.session_id=? AND d.session_id=? AND d.state='ready' AND c.chunk_id IN (" + placeholders + ")",
                               [session_id, session_id, *source_chunk_ids]).fetchall()
            if len(found) != len(source_chunk_ids):
                raise ResearchStoreError("Memory sources must belong to ready documents in this project")
        return json.dumps(source_chunk_ids, separators=(",", ":"))

    @staticmethod
    def _memory(row) -> dict:
        keys = row.keys()
        result = {"id": row["memory_id"], "memory_id": row["memory_id"], "session_id": row["session_id"],
                "kind": row["kind"], "text": row["text"], "source_chunk_ids": json.loads(row["source_chunk_ids"]),
                "source_answer_id": row["source_answer_id"], "revision": row["revision"],
                "status": row["status"], "confirmed_by_user": True,
                "created_at": row["created_at"], "updated_at": row["updated_at"],
                "recorded_at": row["updated_at"],
                "recorded_until": row["recorded_until"] if "recorded_until" in keys else None,
                "valid_from": row["valid_from"] if "valid_from" in keys else row["updated_at"],
                "valid_until": row["valid_until"] if "valid_until" in keys else None,
                "archived": row["status"] == "archived"}
        if row["status"] in {"invalidated", "deleted"}:
            result.update(text="", source_chunk_ids=[], source_answer_id=None, redacted=True)
        return result

    @staticmethod
    def _required_memory(db, session_id: str, memory_id: str):
        row = db.execute("SELECT * FROM research_memories WHERE session_id=? AND memory_id=?",
                         (session_id, memory_id)).fetchone() if db is not None else None
        if row is None:
            raise ResearchNotFoundError("Memory not found in this project")
        return row

    @staticmethod
    def _snapshot(db, row) -> None:
        db.execute("INSERT INTO research_memory_versions(memory_id,session_id,revision,kind,text,source_chunk_ids,source_answer_id,status,created_at,valid_from,valid_until,recorded_until) "
                   "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                   (row["memory_id"], row["session_id"], row["revision"], row["kind"], row["text"],
                    row["source_chunk_ids"], row["source_answer_id"], row["status"], row["updated_at"],
                    row["valid_from"], row["valid_until"], row["recorded_until"]))
        ResearchStore._index_version(db, row)

    @staticmethod
    def _timestamp(value, message="Invalid memory time") -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 253_402_300_799 or not math.isfinite(value):
            raise ResearchStoreError(message)
        return float(value)

    def _validity(self, now, valid_from, valid_until) -> tuple[float, float | None]:
        start = self._timestamp(now if valid_from is None else valid_from)
        end = None if valid_until is None else self._timestamp(valid_until)
        if end is not None and end <= start:
            raise ResearchStoreError("Memory validity must end after it starts")
        return start, end

    def create_memory(self, session_id: str, kind: str, text: str, *, source_chunk_ids: list[str] | None = None,
                      source_answer_id: str | None = None, valid_from: float | None = None,
                      valid_until: float | None = None) -> dict:
        _session(session_id)
        with self._write() as db:
            sources = self._memory_payload(db, session_id, kind, text, source_chunk_ids, source_answer_id)
            self._require_storage(db)
            memory_id, now = str(uuid.uuid4()), self.clock()
            start, end = self._validity(now, valid_from, valid_until)
            db.execute("INSERT INTO research_memories(memory_id,session_id,kind,text,source_chunk_ids,source_answer_id,revision,status,created_at,updated_at,valid_from,valid_until) "
                       "VALUES(?,?,?,?,?,?,1,'active',?,?,?,?)",
                       (memory_id, session_id, kind, text, sources, source_answer_id, now, now, start, end))
            row = self._required_memory(db, session_id, memory_id)
            self._snapshot(db, row)
            self._require_storage(db)
            return self._memory(row)

    def get_memory(self, session_id: str, memory_id: str) -> dict:
        _session(session_id); _identifier(memory_id)
        with self._read() as db:
            return self._read_memory(db, self._required_memory(db, session_id, memory_id), at=self.clock())

    def update_memory(self, session_id: str, memory_id: str, *, expected_revision: int, text: str,
                      kind: str | None = None, source_chunk_ids: list[str] | None = None,
                      source_answer_id=_UNSET, valid_from: float | None = None,
                      valid_until=_UNSET) -> dict:
        _session(session_id); _identifier(memory_id)
        _integer(expected_revision, 1, MAX_MEMORY_OFFSET, "Invalid memory revision")
        with self._write() as db:
            row = self._required_memory(db, session_id, memory_id)
            if row["revision"] != expected_revision or row["status"] not in {"active", "archived"}:
                raise ResearchRevisionConflictError("Memory changed; refresh before correcting it")
            # Omission keeps provenance; explicit None removes it when the
            # user confirms an independent correction with its own sources.
            answer_id = row["source_answer_id"] if source_answer_id is _UNSET else source_answer_id
            sources = self._memory_payload(db, session_id, row["kind"] if kind is None else kind, text,
                                           json.loads(row["source_chunk_ids"]) if source_chunk_ids is None else source_chunk_ids,
                                           answer_id)
            self._require_storage(db)
            now = self.clock()
            start, end = self._validity(now, valid_from, row["valid_until"] if valid_until is _UNSET else valid_until)
            if start < row["valid_from"]:
                raise ResearchRevisionConflictError("A correction cannot precede the version it corrects")
            # Event time may be earlier than recording time. Closing the old
            # half-open interval preserves what applied before the correction.
            previous_end = min(start, row["valid_until"]) if row["valid_until"] is not None else start
            db.execute("UPDATE research_memory_versions SET status='superseded',valid_until=?,recorded_until=? WHERE memory_id=? AND revision=?",
                       (previous_end, now, memory_id, expected_revision))
            db.execute("UPDATE research_memories SET kind=?,text=?,source_chunk_ids=?,source_answer_id=?,revision=revision+1,updated_at=?,valid_from=?,valid_until=? WHERE memory_id=?",
                       (row["kind"] if kind is None else kind, text, sources,
                        answer_id, now, start, end, memory_id))
            updated = self._required_memory(db, session_id, memory_id)
            self._snapshot(db, updated)
            self._require_storage(db)
            return self._memory(updated)

    def archive_memory(self, session_id: str, memory_id: str, *, expected_revision: int, archived: bool = True) -> dict:
        """Hide from the active browser, retaining indexed long-term recall."""
        _session(session_id); _identifier(memory_id)
        _integer(expected_revision, 1, MAX_MEMORY_OFFSET, "Invalid memory revision")
        if type(archived) is not bool:
            raise ResearchStoreError("Invalid memory archive flag")
        with self._write() as db:
            row = self._required_memory(db, session_id, memory_id)
            if row["revision"] != expected_revision or row["status"] not in {"active", "archived"}:
                raise ResearchRevisionConflictError("Memory changed; refresh before archiving it")
            if not self._memory_sources_live(db, row, at=self.clock()):
                raise ResearchRevisionConflictError("Memory source is unavailable; cannot change archive visibility")
            status = "archived" if archived else "active"
            if row["status"] == status:
                return self._memory(row)
            # Archiving changes visibility rather than factual validity. Keep
            # the version's event interval and record the administration time.
            self._require_storage(db)
            now = self.clock()
            db.execute("UPDATE research_memory_versions SET status='superseded',recorded_until=? WHERE memory_id=? AND revision=?",
                       (now, memory_id, expected_revision))
            db.execute("UPDATE research_memories SET status=?,revision=revision+1,updated_at=? WHERE memory_id=?",
                       (status, now, memory_id))
            updated = self._required_memory(db, session_id, memory_id)
            self._snapshot(db, updated)
            self._require_storage(db)
            return self._memory(updated)

    def delete_memory(self, session_id: str, memory_id: str, *, expected_revision: int) -> dict:
        _session(session_id); _identifier(memory_id)
        _integer(expected_revision, 1, MAX_MEMORY_OFFSET, "Invalid memory revision")
        with self._write() as db:
            row = self._required_memory(db, session_id, memory_id)
            if row["revision"] != expected_revision:
                raise ResearchRevisionConflictError("Memory changed; refresh before deleting it")
            if row["status"] == "deleted":
                return self._memory(row)
            db.execute("DELETE FROM research_memory_search WHERE rowid IN (SELECT rowid FROM research_memory_versions WHERE memory_id=?)", (memory_id,))
            db.execute("DELETE FROM research_memory_sources WHERE memory_id=?", (memory_id,))
            # Deletion must still work at capacity: old snapshots are scrubbed,
            # and the latest snapshot becomes a revision tombstone in-place.
            db.execute("UPDATE research_memory_versions SET text='',source_chunk_ids='[]',source_answer_id=NULL,status='deleted' WHERE memory_id=?",
                       (memory_id,))
            db.execute("UPDATE research_memories SET text='',source_chunk_ids='[]',source_answer_id=NULL,status='deleted',revision=revision+1,updated_at=? WHERE memory_id=?",
                       (self.clock(), memory_id))
            updated = self._required_memory(db, session_id, memory_id)
            db.execute("UPDATE research_memory_versions SET revision=?,created_at=? WHERE memory_id=? AND revision=?",
                       (updated["revision"], updated["updated_at"], memory_id, expected_revision))
            return self._memory(updated)

    def list_memories(self, session_id: str, *, offset: int = 0, limit: int = 20,
                      include_deleted: bool = False) -> list[dict]:
        _session(session_id)
        _integer(offset, 0, MAX_MEMORY_OFFSET, "Invalid memory offset")
        _integer(limit, 1, 20, "Invalid memory page size")
        if type(include_deleted) is not bool:
            raise ResearchStoreError("Invalid memory status filter")
        with self._read() as db:
            if db is None:
                return []
            sql = "SELECT * FROM research_memories WHERE session_id=?"
            if not include_deleted:
                sql += " AND status='active'"
            sql += " ORDER BY updated_at DESC,memory_id DESC LIMIT ? OFFSET ?"
            at = self.clock()
            return [self._read_memory(db, row, at=at) for row in db.execute(sql, (session_id, limit, offset)).fetchall()]

    def list_visible_memories(self, session_id: str, *, limit: int = 100) -> list[dict]:
        """Bounded review surface: active and revoked notes, never deleted text."""
        _session(session_id)
        _integer(limit, 1, 100, "Invalid visible memory limit")
        with self._read() as db:
            if db is None:
                return []
            rows = db.execute("SELECT * FROM research_memories WHERE session_id=? AND status IN ('active','invalidated') ORDER BY updated_at DESC,memory_id DESC LIMIT ?",
                              (session_id, limit)).fetchall()
            at = self.clock()
            return [self._read_memory(db, row, at=at) for row in rows]

    @staticmethod
    def _memory_predicate(*, include_archived: bool = True) -> str:
        statuses = "('active','archived')" if include_archived else "('active')"
        return ("m.status IN " + statuses + " AND v.status NOT IN ('deleted','invalidated') "
                "AND v.valid_from<=:at AND (v.valid_until IS NULL OR :at<v.valid_until) "
                "AND NOT EXISTS (SELECT 1 FROM research_memory_versions newer "
                "WHERE newer.memory_id=v.memory_id AND newer.revision>v.revision "
                "AND newer.status NOT IN ('deleted','invalidated') AND newer.valid_from<=:at) "
                "AND NOT EXISTS (SELECT 1 FROM research_memory_sources s "
                "LEFT JOIN research_chunks c ON c.chunk_id=s.chunk_id AND c.session_id=s.session_id "
                "LEFT JOIN research_documents d ON d.document_id=c.document_id AND d.state='ready' "
                "WHERE s.memory_id=v.memory_id AND s.revision=v.revision AND d.document_id IS NULL)")

    @staticmethod
    def _search_expression(session_id: str, question: str) -> str | None:
        terms = list(dict.fromkeys(lexical_tokens(question)))[:MAX_MEMORY_SEARCH_TERMS]
        if not terms:
            return None
        # Only tokenizer output reaches FTS grammar, and every term is quoted.
        return 'scope : "p' + session_id[1:] + '" AND tokens : (' + " OR ".join(
            '"' + term.replace('"', '""') + '"' for term in terms) + ')'

    @staticmethod
    def _selected_memory(row, *, historical: bool) -> dict:
        result = ResearchStore._memory(row)
        result["version_status"] = result["status"]
        result["status"] = row["current_status"]
        result["archived"] = row["current_status"] == "archived"
        result["historical"] = historical
        if row["current_status"] == "invalidated":
            result.update(text="", source_chunk_ids=[], source_answer_id=None, redacted=True)
        return result

    @staticmethod
    def _memory_sources_live(db, row, *, at=None, seen=None) -> bool:
        """Current dependencies for a current or historical memory row.

        The direct answer check also protects older rows written before answer
        dependencies were included in the reverse index. This is provenance
        availability only, not factual or semantic verification of a note.
        """
        if row["source_answer_id"] is not None:
            try:
                answer_sources = ResearchStore._memory_answer_sources(db, row["session_id"], row["source_answer_id"])
            except ResearchStoreError:
                return False
            from .research_memory_dependencies import source_answer_memory_live
            if not source_answer_memory_live(db, row["session_id"], row["source_answer_id"], at=at, seen=seen):
                return False
            if answer_sources:
                placeholders = ",".join("?" for _ in answer_sources)
                live = db.execute("SELECT count(*) FROM research_chunks c JOIN research_documents d "
                                  "ON d.document_id=c.document_id WHERE c.session_id=? AND d.session_id=? "
                                  "AND d.state='ready' AND c.chunk_id IN (" + placeholders + ")",
                                  [row["session_id"], row["session_id"], *answer_sources]).fetchone()[0]
                if live != len(answer_sources):
                    return False
        return not db.execute("SELECT 1 FROM research_memory_sources s "
                              "LEFT JOIN research_chunks c ON c.chunk_id=s.chunk_id AND c.session_id=s.session_id "
                              "LEFT JOIN research_documents d ON d.document_id=c.document_id AND d.state='ready' "
                              "WHERE s.memory_id=? AND s.revision=? AND d.document_id IS NULL LIMIT 1",
                              (row["memory_id"], row["revision"])).fetchone()

    @staticmethod
    def _version_sources_available(db, row) -> bool:
        return ResearchStore._memory_sources_live(db, row)

    @staticmethod
    def _read_memory(db, row, *, at=None) -> dict:
        result = ResearchStore._memory(row)
        if row["status"] not in {"deleted", "invalidated"} and not ResearchStore._memory_sources_live(db, row, at=at):
            # Older installations allowed answer-only provenance without a
            # reverse dependency row. Do not re-expose it after withdrawal.
            result.update(status="invalidated", text="", source_chunk_ids=[], source_answer_id=None, redacted=True)
        return result

    def search_memories(self, session_id: str, question: str, *, limit: int = 100,
                        as_of: float | None = None) -> list[dict]:
        """Indexed recall across the complete retained lifetime, bounded output.

        Archived notes remain candidates. Relevance never upgrades a user note
        into source evidence. Event-time reads use the latest correction known
        now for that interval, not a simulation of what the system knew then.
        """
        _session(session_id)
        _text(question, 8_192, "Invalid memory search query")
        if len(question.encode("utf-8")) > 8_192:
            raise ResearchLimitError("Memory query exceeds its byte budget")
        _integer(limit, 1, MAX_MEMORY_PAGE, "Invalid memory search limit")
        at = self._timestamp(self.clock() if as_of is None else as_of)
        self._ensure_memory_schema()
        with self._read() as db:
            if db is None:
                return []
            expression = self._search_expression(session_id, question)
            parameters = {"session": session_id, "at": at, "limit": limit, "query": expression}
            predicate = "v.session_id=:session AND " + self._memory_predicate()
            base = ("SELECT v.*,v.created_at AS updated_at,m.status AS current_status "
                    "FROM research_memory_versions v JOIN research_memories m ON m.memory_id=v.memory_id ")
            rows = []
            if expression:
                # CROSS JOIN deliberately fixes FTS as the outer candidate
                # iterator. Otherwise SQLite may choose the temporal index and
                # execute an FTS lookup once per lifetime revision.
                search_base = ("SELECT v.*,v.created_at AS updated_at,m.status AS current_status "
                               "FROM research_memory_search CROSS JOIN research_memory_versions v "
                               "ON v.rowid=research_memory_search.rowid "
                               "CROSS JOIN research_memories m ON m.memory_id=v.memory_id ")
                rows = db.execute(search_base +
                                  "WHERE research_memory_search MATCH :query AND " + predicate +
                                  " ORDER BY bm25(research_memory_search),v.created_at DESC,v.memory_id LIMIT :limit",
                                  parameters).fetchall()
            strategy = classify_memory_query(question)
            fallback_limit = limit if strategy in {"explicit_preferences", "recent_project_context"} else min(2, limit)
            parameters["fallback_limit"] = fallback_limit
            fallback_filter = " AND v.kind='preference'" if strategy != "recent_project_context" else ""
            fallback = db.execute(base + "WHERE " + predicate + fallback_filter +
                                  " ORDER BY v.valid_from DESC,v.memory_id DESC,v.revision DESC LIMIT :fallback_limit", parameters).fetchall()
            # Reserve at most two style preferences for ordinary questions.
            # Explicit preference/progress queries may browse recent memories.
            if strategy == "explicit_preferences":
                ordered = fallback + rows
            elif strategy == "recent_project_context":
                ordered = rows + fallback
            else:
                fallback_ids = {row["memory_id"] for row in fallback}
                relevant = [row for row in rows if row["memory_id"] not in fallback_ids]
                reserve = min(len(fallback), max(0, limit - (1 if relevant else 0)))
                ordered = relevant[:limit - reserve] + fallback[:reserve]
            seen, result = set(), []
            for row in ordered:
                if row["memory_id"] in seen or not self._memory_sources_live(db, row, at=parameters['at']):
                    continue
                seen.add(row["memory_id"])
                result.append(self._selected_memory(row, historical=as_of is not None))
                if len(result) == limit:
                    break
            return result

    def page_memories(self, session_id: str, *, query: str = "", cursor: str | None = None,
                      limit: int = 20, as_of: float | None = None, include_archived: bool = False) -> dict:
        """Keyset paging; only the requested page is materialized in Python."""
        _session(session_id)
        _integer(limit, 1, MAX_MEMORY_PAGE, "Invalid memory page size")
        if not isinstance(query, str) or len(query) > 8_192 or "\x00" in query:
            raise ResearchStoreError("Invalid memory search query")
        try:
            if len(query.encode("utf-8")) > 8_192:
                raise ResearchLimitError("Memory query exceeds its byte budget")
        except UnicodeEncodeError as error:
            raise ResearchStoreError("Invalid memory search query") from error
        if type(include_archived) is not bool:
            raise ResearchStoreError("Invalid memory archive filter")
        at = self._timestamp(self.clock() if as_of is None else as_of)
        fingerprint = hashlib.sha256(json.dumps([session_id, query, as_of, include_archived], ensure_ascii=False).encode()).hexdigest()[:24]
        after = None
        if cursor is not None:
            try:
                if not isinstance(cursor, str) or len(cursor) > 512:
                    raise ValueError
                decoded = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
                if not isinstance(decoded, list) or len(decoded) != 5 or decoded[0] != fingerprint:
                    raise ValueError
                at = self._timestamp(decoded[1])
                after = (self._timestamp(decoded[2]), _identifier(decoded[3]),
                         _integer(decoded[4], 1, MAX_MEMORY_OFFSET, "Invalid page cursor"))
            except (ValueError, TypeError, UnicodeError) as error:
                raise ResearchStoreError("Invalid memory page cursor") from error
        self._ensure_memory_schema()
        with self._read() as db:
            if db is None:
                return {"items": [], "next_cursor": None, "total": 0}
            expression = self._search_expression(session_id, query) if query.strip() else None
            if query.strip() and expression is None:
                return {"items": [], "next_cursor": None, "total": 0}
            parameters = {"session": session_id, "at": at, "limit": limit + 1, "query": expression}
            joins = " FROM research_memory_versions v JOIN research_memories m ON m.memory_id=v.memory_id "
            visibility = self._memory_predicate(include_archived=include_archived)
            if as_of is None and not query.strip():
                # Keep an explicit revoked placeholder on the current review
                # surface. It is never eligible for search, historical recall
                # or the answer context, and does not expose revoked text.
                visibility = "(" + visibility + " OR (m.status='invalidated' AND v.revision=m.revision))"
            predicate = "v.session_id=:session AND " + visibility
            if expression:
                joins = (" FROM research_memory_search CROSS JOIN research_memory_versions v "
                         "ON v.rowid=research_memory_search.rowid "
                         "CROSS JOIN research_memories m ON m.memory_id=v.memory_id ")
                predicate += " AND research_memory_search MATCH :query"
            total = db.execute("SELECT count(*)" + joins + "WHERE " + predicate, parameters).fetchone()[0]
            if after is not None:
                predicate += " AND (v.valid_from,v.memory_id,v.revision)<(:after_time,:after_id,:after_revision)"
                parameters.update(after_time=after[0], after_id=after[1], after_revision=after[2])
            rows = db.execute("SELECT v.*,v.created_at AS updated_at,m.status AS current_status" + joins +
                              "WHERE " + predicate + " ORDER BY v.valid_from DESC,v.memory_id DESC,v.revision DESC LIMIT :limit", parameters).fetchall()
            next_cursor = None
            if len(rows) > limit:
                last = rows[limit - 1]
                next_cursor = base64.urlsafe_b64encode(json.dumps([fingerprint, at, last["valid_from"], last["memory_id"], last["revision"]], separators=(",", ":")).encode()).decode()
            items = []
            for row in rows[:limit]:
                item = self._selected_memory(row, historical=as_of is not None)
                if not self._memory_sources_live(db, row, at=parameters['at']):
                    item.update(status="invalidated", text="", source_chunk_ids=[], source_answer_id=None, redacted=True)
                items.append(item)
            return {"items": items,
                    "next_cursor": next_cursor, "total": total}

    def memory_history(self, session_id: str, memory_id: str, *, limit: int = 50,
                       cursor: str | None = None) -> dict:
        _session(session_id); _identifier(memory_id)
        _integer(limit, 1, MAX_MEMORY_PAGE, "Invalid memory history page size")
        after = MAX_MEMORY_OFFSET
        if cursor is not None:
            try:
                if not isinstance(cursor, str) or len(cursor) > 200:
                    raise ValueError
                decoded = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
                if not isinstance(decoded, list) or len(decoded) != 3 or decoded[:2] != [session_id, memory_id]:
                    raise ValueError
                after = _integer(decoded[2], 1, MAX_MEMORY_OFFSET, "Invalid history cursor")
            except (ValueError, TypeError, UnicodeError) as error:
                raise ResearchStoreError("Invalid memory history cursor") from error
        self._ensure_memory_schema()
        with self._read() as db:
            parent = self._required_memory(db, session_id, memory_id)
            total = db.execute("SELECT count(*) FROM research_memory_versions WHERE memory_id=?", (memory_id,)).fetchone()[0]
            rows = db.execute("SELECT *,created_at AS updated_at FROM research_memory_versions "
                              "WHERE memory_id=? AND revision<? ORDER BY revision DESC LIMIT ?", (memory_id, after, limit + 1)).fetchall()
            items = []
            for row in rows[:limit]:
                item = self._memory(row)
                if (parent["status"] == "deleted" or row["status"] in {"deleted", "invalidated"}
                        or not self._memory_sources_live(db, row, at=self.clock())):
                    item.update(text="", source_chunk_ids=[], source_answer_id=None, redacted=True)
                items.append(item)
            next_cursor = base64.urlsafe_b64encode(json.dumps([session_id, memory_id, rows[limit - 1]["revision"]]).encode()).decode() if len(rows) > limit else None
            return {"items": items, "next_cursor": next_cursor, "total": total}

    def memory_versions(self, session_id: str, memory_id: str, *, offset: int = 0, limit: int = 20) -> list[dict]:
        _session(session_id); _identifier(memory_id)
        _integer(offset, 0, MAX_MEMORY_OFFSET, "Invalid memory revision offset")
        _integer(limit, 1, 20, "Invalid memory revision page size")
        self._ensure_memory_schema()
        with self._read() as db:
            parent = self._required_memory(db, session_id, memory_id)
            rows = db.execute("SELECT *,created_at AS updated_at FROM research_memory_versions WHERE session_id=? AND memory_id=? ORDER BY revision DESC LIMIT ? OFFSET ?",
                              (session_id, memory_id, limit, offset)).fetchall()
            result = []
            for row in rows:
                item = self._memory(row)
                if (parent["status"] == "deleted" or row["status"] in {"deleted", "invalidated"}
                        or not self._memory_sources_live(db, row, at=self.clock())):
                    item.update(text="", source_chunk_ids=[], source_answer_id=None, redacted=True)
                result.append(item)
            return result

    def active_memory_context(self, session_id: str, *, max_chars: int = MAX_MEMORY_CONTEXT_CHARS,
                              max_items: int = 20) -> list[dict]:
        _integer(max_chars, 1, MAX_MEMORY_CONTEXT_CHARS, "Invalid memory context budget")
        _integer(max_items, 1, 20, "Invalid memory context item limit")
        memories = self.page_memories(session_id, limit=max_items)["items"]
        selected, used = [], 0
        for memory in memories:
            if memory["status"] != "active":
                continue
            # Only complete explicit notes fit; do not silently truncate or alter
            # a user's preference/decision/finding to manufacture a new meaning.
            if used + len(memory["text"]) > max_chars:
                continue
            selected.append({key: value for key, value in memory.items() if key not in {"version_status", "historical"}})
            used += len(memory["text"])
        return selected

    def export_memories(self, session_id: str, *, cursor: str | None = None, limit: int = 100) -> dict:
        page = self.page_memories(session_id, cursor=cursor, limit=limit, include_archived=True)
        return {"session_id": session_id, "schema_version": 2, "memories": page["items"],
                "next_cursor": page["next_cursor"], "total": page["total"]}

    @staticmethod
    def _answer_validation(validation: dict) -> dict:
        """Keep bounded syntax-check metadata, never a semantic correctness claim."""
        if not isinstance(validation, dict):
            raise ResearchStoreError("Invalid citation validation metadata")

        clipped_labels = False

        def labels(value):
            nonlocal clipped_labels
            if not isinstance(value, list):
                raise ResearchStoreError("Invalid citation validation labels")
            bounded = []
            for label in value[:32]:
                # Fabricated S/K syntax such as [S0]/[K0001] remains a warning,
                # not turn a paid answer into a metadata persistence failure.
                # This metadata is not a source reference: record_answer below
                # still accepts only canonical S labels for uploaded chunks.
                if not isinstance(label, str) or not re.fullmatch(r"[SK][0-9]+", label):
                    raise ResearchStoreError("Invalid citation validation label")
                if len(label) > 17:
                    clipped_labels = True
                    label = label[:17] + "…"
                if label not in bounded:
                    bounded.append(label)
            return bounded

        used = labels(validation.get("used_labels", []))
        unknown = labels(validation.get("unknown_labels", []))
        missing = validation.get("missing_citations", False)
        valid = validation.get("valid", False)
        upstream_truncated = validation.get("metadata_truncated", False)
        if type(missing) is not bool or type(valid) is not bool or type(upstream_truncated) is not bool:
            raise ResearchStoreError("Invalid citation validation flags")
        raw_warnings = validation.get("warnings", [])
        if not isinstance(raw_warnings, list):
            raise ResearchStoreError("Invalid citation warnings")
        messages = {
            "unknown_citation_labels": "回答包含未提供的引用标签，不能将其作为资料依据。",
            "missing_citations": "回答未引用本次资料；资料性结论尚未核对。",
            "empty_answer": "未收到可用回答。",
            "source_revoked": "部分原文依据已被删除，无法继续核对。",
            "provenance_unavailable": "依据记录不可用，不能继续核对本次引用。",
            "memory_context_revoked": "相关记忆已改变或无法核验，旧回答已收起。",
        }
        warnings = []
        for warning in raw_warnings[:8]:
            if not isinstance(warning, dict) or warning.get("code") not in messages:
                raise ResearchStoreError("Invalid citation warning code")
            normalized = {"code": warning["code"], "message": messages[warning["code"]]}
            if "labels" in warning:
                normalized["labels"] = labels(warning["labels"])
            warnings.append(normalized)
        # Model/user supplied entailment fields are deliberately not preserved.
        truncated = upstream_truncated or clipped_labels or len(validation.get("used_labels", [])) > 32 or len(validation.get("unknown_labels", [])) > 32 or len(raw_warnings) > 8
        return {"valid": valid and not missing and not unknown and not truncated,
                "used_labels": used, "unknown_labels": unknown,
                "missing_citations": missing, "warnings": warnings,
                "scope": "citation_syntax_only", "entailment_verified": False,
                "metadata_truncated": truncated}

    def answer_capacity(self, session_id: str) -> dict:
        """SELECT-only provisioned storage check, without a lifetime answer cap."""
        _session(session_id)
        with self._read() as db:
            return self._storage_capacity(db)

    @staticmethod
    def _answer_context_metadata(value: dict | None) -> dict:
        """Persist bounded limitation codes/counts, never query or memory text."""
        if value is None:
            return {}
        if not isinstance(value, dict) or set(value) - {"retrieval_gaps", "used_memory_count", "memory_selection", "memory_dependency_required", "answer_dependency_required"}:
            raise ResearchStoreError("Invalid answer context metadata")
        normalized = {}
        if 'answer_dependency_required' in value:
            if type(value['answer_dependency_required']) is not bool:
                raise ResearchStoreError('Invalid analysis dependency flag')
            normalized['answer_dependency_required'] = value['answer_dependency_required']
        if "memory_dependency_required" in value:
            if type(value['memory_dependency_required']) is not bool:
                raise ResearchStoreError('Invalid memory dependency flag')
            normalized['memory_dependency_required'] = value['memory_dependency_required']
        if "retrieval_gaps" in value:
            gaps = value["retrieval_gaps"]
            if not isinstance(gaps, list) or len(gaps) > len(ANSWER_GAP_MESSAGES):
                raise ResearchLimitError("Too many retrieval limitation codes")
            saved, seen = [], set()
            for gap in gaps:
                if (not isinstance(gap, dict) or not isinstance(gap.get("code"), str) or gap["code"] not in ANSWER_GAP_MESSAGES
                        or set(gap) - {"code", "count", "message", "terms"}):
                    raise ResearchStoreError("Invalid retrieval limitation code")
                if gap["code"] in seen:
                    raise ResearchStoreError("Duplicate retrieval limitation code")
                seen.add(gap["code"])
                item = {"code": gap["code"]}
                if "count" in gap:
                    item["count"] = _integer(gap["count"], 0, MAX_DOCUMENTS_PER_SESSION * MAX_DOCUMENT_CHUNKS,
                                             "Invalid retrieval limitation count")
                # Caller messages and query terms are deliberately discarded.
                saved.append(item)
            normalized["retrieval_gaps"] = saved
        if "used_memory_count" in value:
            normalized["used_memory_count"] = _integer(value["used_memory_count"], 0, 8, "Invalid used memory count")
        if "memory_selection" in value:
            selection = value["memory_selection"]
            if (not isinstance(selection, dict) or set(selection) != {"strategy", "omitted_count", "selected_count"}
                    or not isinstance(selection["strategy"], str) or selection["strategy"] not in ANSWER_MEMORY_STRATEGIES):
                raise ResearchStoreError("Invalid memory selection metadata")
            normalized["memory_selection"] = {
                "strategy": selection["strategy"],
                "omitted_count": _integer(selection["omitted_count"], 0, MAX_MEMORIES_PER_SESSION, "Invalid omitted memory count"),
                "selected_count": _integer(selection["selected_count"], 0, 8, "Invalid selected memory count"),
            }
        return normalized

    @staticmethod
    def _public_answer_context(value: dict) -> dict:
        result = dict(value)
        if "retrieval_gaps" in result:
            result["retrieval_gaps"] = [{**gap, "message": ANSWER_GAP_MESSAGES[gap["code"]]} for gap in result["retrieval_gaps"]]
        return result

    def record_answer(self, session_id: str, answer_id: str, source_refs: list[dict],
                      validation: dict, answer_mode: str, *, context_metadata: dict | None = None) -> dict:
        """Persist citation references only, not another answer or source snapshot."""
        _session(session_id); _identifier(answer_id)
        if answer_mode not in {"evidence", "explore", "general", "insufficient_evidence", "memory", "analogy", "deduction", "induction"}:
            raise ResearchStoreError("Invalid research answer mode")
        if not isinstance(source_refs, list) or len(source_refs) > MAX_ANSWER_SOURCES:
            raise ResearchLimitError("Too many answer source references")
        normalized = []
        ids, labels_seen = set(), set()
        for source in source_refs:
            if not isinstance(source, dict):
                raise ResearchStoreError("Invalid answer source reference")
            chunk_id, label = _identifier(source.get("id")), source.get("label")
            if not isinstance(label, str) or not SOURCE_LABEL_PATTERN.fullmatch(label) or chunk_id in ids or label in labels_seen:
                raise ResearchStoreError("Invalid or duplicate answer source label")
            reference = {"id": chunk_id, "label": label}
            if "excerpt_start" in source or "excerpt_end" in source:
                start = _integer(source.get("excerpt_start"), 0, MAX_CHUNK_CHARS - 1, "Invalid source excerpt start")
                end = _integer(source.get("excerpt_end"), 1, MAX_CHUNK_CHARS, "Invalid source excerpt end")
                truncated = source.get("truncated")
                if end <= start or type(truncated) is not bool:
                    raise ResearchStoreError("Invalid source excerpt bounds")
                reference.update({"excerpt_start": start, "excerpt_end": end, "truncated": truncated})
            normalized.append(reference)
            ids.add(chunk_id); labels_seen.add(label)
        refs_raw = json.dumps(normalized, separators=(",", ":"))
        validation_raw = json.dumps(self._answer_validation(validation), ensure_ascii=False, separators=(",", ":"))
        context_raw = json.dumps(self._answer_context_metadata(context_metadata), separators=(",", ":"), sort_keys=True)
        if len(context_raw.encode("utf-8")) > MAX_ANSWER_CONTEXT_BYTES:
            raise ResearchLimitError("Answer context metadata exceeds its budget")
        with self._write() as db:
            existing = db.execute("SELECT * FROM research_answers WHERE answer_id=?", (answer_id,)).fetchone()
            if existing:
                if (existing["session_id"] != session_id or existing["source_refs"] != refs_raw
                        or existing["validation"] != validation_raw or existing["answer_mode"] != answer_mode
                        or existing["context_metadata_json"] != context_raw):
                    raise ResearchRevisionConflictError("Answer reference belongs to different content")
                return {"answer_id": answer_id, "recorded": True}
            self._require_storage(db)
            if ids:
                found = db.execute("SELECT c.chunk_id,length(c.text) AS text_length FROM research_chunks c JOIN research_documents d ON d.document_id=c.document_id "
                                   "WHERE c.session_id=? AND d.session_id=? AND d.state='ready' AND c.chunk_id IN (" + ",".join("?" for _ in ids) + ")",
                                   [session_id, session_id, *ids]).fetchall()
                if len(found) != len(ids):
                    raise ResearchStoreError("Answer sources must belong to this project")
                lengths = {row["chunk_id"]: row["text_length"] for row in found}
                for reference in normalized:
                    if "excerpt_end" in reference:
                        full_length = lengths[reference["id"]]
                        if (reference["excerpt_end"] > full_length
                                or reference["truncated"] != (reference["excerpt_start"] > 0 or reference["excerpt_end"] < full_length)):
                            raise ResearchStoreError("Source excerpt does not match its original chunk")
            db.execute("INSERT INTO research_answers (answer_id,session_id,source_refs,validation,answer_mode,created_at,context_metadata_json) VALUES(?,?,?,?,?,?,?)",
                       (answer_id, session_id, refs_raw, validation_raw, answer_mode, self.clock(), context_raw))
            self._require_storage(db)
            return {"answer_id": answer_id, "recorded": True}

    def answer_metadata(self, session_id: str, answer_ids: list[str]) -> dict:
        _session(session_id)
        if not isinstance(answer_ids, list) or len(answer_ids) > MAX_ANSWER_LOOKUP:
            raise ResearchLimitError("Too many answer references")
        for answer_id in answer_ids:
            _identifier(answer_id)
        if not answer_ids:
            return {}
        with self._read() as db:
            if db is None:
                return {}
            # Older empty research stores do not acquire a new schema on reads.
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='research_answers'").fetchone():
                return {}
            rows = db.execute("SELECT * FROM research_answers WHERE session_id=? AND answer_id IN (" + ",".join("?" for _ in answer_ids) + ")",
                              [session_id, *answer_ids]).fetchall()
            references = {row['answer_id']: json.loads(row['source_refs']) for row in rows}
            source_ids = list(dict.fromkeys(source['id'] for refs in references.values() for source in refs))
            if len(source_ids) > MAX_ANSWER_LOOKUP * MAX_ANSWER_SOURCES:
                raise ResearchLimitError('Too many answer source references')
            available = set()
            for offset in range(0, len(source_ids), ANSWER_READ_BATCH):
                batch = source_ids[offset:offset + ANSWER_READ_BATCH]
                found = db.execute("SELECT c.chunk_id FROM research_chunks c JOIN research_documents d ON d.document_id=c.document_id "
                                   "WHERE c.session_id=? AND d.session_id=? AND d.state='ready' AND c.chunk_id IN ("
                                   + ','.join('?' for _ in batch) + ')', [session_id, session_id, *batch])
                available.update(row['chunk_id'] for row in found)
            result = {}
            for row in rows:
                refs = references[row['answer_id']]
                for source in refs:
                    source['available'] = source['id'] in available
                result[row["answer_id"]] = {"answer_id": row["answer_id"], "source_refs": refs,
                                            "validation": json.loads(row["validation"]),
                                            "answer_mode": row["answer_mode"], "created_at": row["created_at"],
                                            **self._public_answer_context(self._answer_context_metadata(
                                                json.loads(row["context_metadata_json"]) if "context_metadata_json" in row.keys() else {}))}
            return result

    def answer_source_chunks(self, session_id: str, chunk_ids: list[str]) -> dict:
        """Read only this history request's live references, without a shared cache.

        This is not a corpus scan: at most 200 answers * 10 references, with
        scope/state checked again after metadata was loaded. Excerpt slicing
        and proof fingerprints remain the caller's existing hydration checks.
        """
        _session(session_id)
        if not isinstance(chunk_ids, list) or len(chunk_ids) > MAX_ANSWER_LOOKUP * MAX_ANSWER_SOURCES:
            raise ResearchLimitError('Too many answer source references')
        for identifier in chunk_ids:
            _identifier(identifier)
        identifiers = list(dict.fromkeys(chunk_ids))
        if not identifiers:
            return {}
        with self._read() as db:
            if db is None:
                return {}
            result = {}
            for offset in range(0, len(identifiers), ANSWER_READ_BATCH):
                batch = identifiers[offset:offset + ANSWER_READ_BATCH]
                rows = db.execute("SELECT c.*,d.filename,d.content_hash FROM research_chunks c JOIN research_documents d "
                                  "ON d.document_id=c.document_id WHERE c.session_id=? AND d.session_id=? "
                                  "AND d.state='ready' AND c.chunk_id IN (" + ','.join('?' for _ in batch) + ')',
                                  [session_id, session_id, *batch]).fetchall()
                for row in rows:
                    result[row['chunk_id']] = {'id': row['chunk_id'], 'chunk_id': row['chunk_id'],
                        'document_id': row['document_id'], 'filename': row['filename'],
                        'ordinal': row['ordinal'], 'page': row['page'], 'content_hash': row['content_hash'],
                        'text': row['text'], 'content': row['text']}
            return result

    def delete_document(self, session_id: str, document_id: str) -> dict:
        _session(session_id); _identifier(document_id)
        with self._write() as db:
            row = self._required_document(db, session_id, document_id)
            source_ids = {chunk[0] for chunk in db.execute("SELECT chunk_id FROM research_chunks WHERE document_id=?", (document_id,)).fetchall()}
            path = self.files_directory / (document_id + row["extension"])
            if self._check_regular(path):
                path.unlink()
            invalidated = []
            if source_ids:
                placeholders = ",".join("?" for _ in source_ids)
                parameters = [session_id, *source_ids]
                # Follow the reverse dependency index, not every saved memory.
                affected = db.execute("SELECT DISTINCT m.* FROM research_memory_sources s JOIN research_memories m "
                                      "ON m.memory_id=s.memory_id AND m.revision=s.revision WHERE s.session_id=? "
                                      "AND s.chunk_id IN (" + placeholders + ") AND m.status IN ('active','archived')",
                                      parameters).fetchall()
                revoked_rows = db.execute("SELECT DISTINCT v.rowid FROM research_memory_sources s "
                                          "JOIN research_memory_versions v ON v.memory_id=s.memory_id AND v.revision=s.revision "
                                          "WHERE s.session_id=? AND s.chunk_id IN (" + placeholders + ")", parameters).fetchall()
                for version in revoked_rows:
                    db.execute("DELETE FROM research_memory_search WHERE rowid=?", (version[0],))
                    db.execute("UPDATE research_memory_versions SET status='invalidated' WHERE rowid=?", (version[0],))
                for memory in affected:
                    db.execute("UPDATE research_memories SET status='invalidated',revision=revision+1,updated_at=? WHERE memory_id=?",
                               (self.clock(), memory["memory_id"]))
                    self._snapshot(db, self._required_memory(db, session_id, memory["memory_id"]))
                    db.execute("DELETE FROM research_memory_search WHERE rowid=(SELECT rowid FROM research_memory_versions WHERE memory_id=? AND revision=?)",
                               (memory["memory_id"], memory["revision"] + 1))
                    invalidated.append(memory["memory_id"])
            db.execute("DELETE FROM research_chunks WHERE document_id=?", (document_id,))
            db.execute("UPDATE research_documents SET state='deleted',chunk_count=0,text_chars=0,content_hash='',error_code=NULL,updated_at=? WHERE document_id=?",
                       (self.clock(), document_id))
            return {"document_id": document_id, "status": "deleted", "invalidated_memory_ids": invalidated}
