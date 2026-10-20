"""Small allowlisted facade for PRIVATE mobile development instances.

Never expose the legacy/admin API to customers based on this shared token.
Customer deployment still requires identity, owner checks and durable jobs.
"""
import asyncio
import math
import os
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from pydantic import BaseModel, ConfigDict, Field
from fastapi.responses import JSONResponse
from docthinker.harness_runtime import execute_run

from ..mobile_access import MobileRoute, require_mobile_access
from ..schemas import CreateSessionRequest, QueryRequest
from ..research_history import hydrate_research_history, hide_unverified_research_reply
from ..state import state
from ..model_response_error import ModelResponseError, model_response_failure
from ..research_source_contract import MAX_ANSWER_SOURCES

router = APIRouter(prefix="/mobile", dependencies=[Depends(require_mobile_access)], route_class=MobileRoute)
_active_queries: set[str] = set()
_query_boot_id = str(uuid.uuid4())
_query_tasks: dict[tuple[str, str, str], asyncio.Task] = {}
_starting_query_keys: set[tuple[str, str, str]] = set()
_query_background_tasks: set[asyncio.Task] = set()
QUERY_WAIT_SECONDS = 5
QUERY_EXECUTION_SECONDS = 110


def _manager():
    if state.session_manager is None:
        raise HTTPException(503, "服务尚未初始化")
    return state.session_manager


def _session(sid: str):
    if not re.fullmatch(r"#\d{5}", sid):
        raise HTTPException(400, "无效的会话编号")
    if not _manager().get_session(sid):
        raise HTTPException(404, "会话不存在")
    return sid


class MobileSessionRequest(BaseModel):
    title: str = Field(default="新对话", min_length=1, max_length=80)


class MobileQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    question: str = Field(min_length=1, max_length=8000)
    use_memory: bool = True
    knowledge_skill_mode: Literal['auto', 'off'] = 'auto'
    insight_mode: Literal['auto', 'off'] = 'auto'
    reasoning_mode: Literal["auto", "evidence", "analogy", "deduction", "induction"] = Field(
        default="auto", json_schema_extra={"deprecated": True},
        description="Legacy receipt compatibility; omit for normal chat and writing.")
    memory_as_of: float | None = Field(default=None, ge=0, le=253402300799, allow_inf_nan=False, strict=True)
    use_conversation_context: bool = True
    use_llm_cache: bool = True
    use_self_evolution: bool = False
    request_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


class MobileAnswerFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    session_id: str
    answer_id: str = Field(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
    rating: Literal["helpful", "inaccurate"]
    note: str = Field(default="", max_length=500)


@router.get("/capabilities")
async def capabilities():
    result = {"audience": "private-development", "customer_auth": False,
            "voice": "/mobile/voice/capabilities", "max_upload_bytes": 20 * 1024 * 1024,
            "realtime_voice": False,
            "safe_profile": bool(getattr(state, "mobile_safe_mode", False)),
            "document_rag": bool(getattr(state, "mobile_document_rag", False)),
            "chat_ready": bool(getattr(state, "mobile_chat_func", None) or state.rag_instance),
            "uploads": bool(state.ingestion_service),
            "answer_feedback": True,
            "workbench": {"brief": True, "artifacts": getattr(state, "research_service", None) is not None,
                          "max_artifacts_per_session": 100},
            "query_idempotency": True,
            "automatic_learning": not getattr(state, "mobile_safe_mode", False),
            "memory_write_enabled": not getattr(state, "mobile_safe_mode", False)}
    if getattr(state, "research_service", None) is not None:
        result['workbench']['tasks'] = True
        result['workbench']['long_projects'] = True
        result.update({"research_core": True, "uploads": True, "memory_write_enabled": True,
                       "automatic_learning": False, "research_memory_policy": "explicit_confirmation"})
    if os.getenv("DOCTHINKER_INVITE_WORKER_ID"):
        from ..invite_budget import INVITE_REQUEST_BUDGET
        result["request_budget"] = INVITE_REQUEST_BUDGET
    return result


@router.get("/sessions")
async def sessions():
    return {"sessions": [{"id": s["id"], "title": s.get("title", "新对话")}
                         for s in _manager().list_sessions()]}


@router.post("/sessions")
async def create_session(body: MobileSessionRequest):
    from .sessions import create_session as create
    result = await create(CreateSessionRequest(title=body.title))
    session = result["session"]
    return {"session": {"id": session["id"], "title": session.get("title", "新对话")}}


@router.get("/history")
async def history(session_id: str):
    sid = _session(session_id)
    messages = await asyncio.to_thread(_manager().get_history, sid)
    messages = await asyncio.to_thread(hydrate_research_history, messages, sid,
                                       getattr(state, "research_service", None))
    try:
        feedback = await asyncio.to_thread(_feedback_store().list_for_session, sid)
        by_answer = {item["answer_id"]: item for item in feedback}
        messages = [
            {**message, "feedback": by_answer[message["id"]]}
            if message.get("role") == "assistant" and message.get("id") in by_answer else message
            for message in messages
        ]
    except Exception:
        # A feedback-only outage must not hide users' real chat history.
        # Explicit feedback reads/writes still fail closed below.
        pass
    response = {"history": messages}
    try:
        journal = _query_journal()
        active_ids = {request_id for (path, session, request_id), task in _query_tasks.items()
                      if path == str(journal.path) and session == sid and not task.done()}
        active_ids.update(request_id for path, session, request_id in _starting_query_keys
                          if path == str(journal.path) and session == sid)
        requests = await asyncio.to_thread(journal.recent_unresolved, sid, active_ids=active_ids)
        if requests:
            response["query_requests"] = requests
    except Exception:
        # Receipt failure must not hide users' independently persisted history.
        pass
    return response


def _feedback_store():
    from ..answer_feedback import AnswerFeedbackStore

    return AnswerFeedbackStore(Path(_manager().base_storage_path))


@router.get("/feedback")
async def answer_feedback(session_id: str):
    sid = _session(session_id)
    try:
        items = await asyncio.to_thread(_feedback_store().list_for_session, sid)
        return {"items": items}
    except Exception:
        raise HTTPException(503, "Feedback is unavailable") from None


@router.post("/feedback")
async def submit_answer_feedback(body: MobileAnswerFeedback):
    from ..answer_feedback import FeedbackLimitError

    sid = _session(body.session_id)
    # Identity is resolved only from this worker's persisted session history,
    # never from an answer, role, user ID or an index supplied by the client.
    history = await asyncio.to_thread(_manager().get_history, sid)
    if not any(item.get("id") == body.answer_id and item.get("role") == "assistant" for item in history):
        raise HTTPException(404, "Assistant answer is unavailable")
    try:
        feedback = await asyncio.to_thread(
            _feedback_store().put, sid, body.answer_id, body.rating, body.note,
        )
        return {"feedback": feedback}
    except FeedbackLimitError:
        raise HTTPException(429, "Feedback storage limit reached") from None
    except Exception:
        raise HTTPException(503, "Feedback is unavailable") from None


@router.get("/files")
async def files(session_id: str):
    records = _manager().get_files(_session(session_id))
    allowed = {"id", "filename", "file_name", "file_size", "status", "processing_status", "created_at"}
    return {"files": [{key: value for key, value in record.items() if key in allowed} for record in records]}


def _memory_database_path() -> Path:
    if getattr(state, "mobile_safe_mode", False):
        if not state.long_horizon_db_path:
            raise RuntimeError("Mobile memory path is not initialized")
        return Path(state.long_horizon_db_path)
    # Use the existing path contract, but never instantiate the backend: its
    # constructor initializes schema and its normal connection configures WAL.
    from docthinker.memory_core.sqlite_backend import SQLiteLongHorizonBackend

    return SQLiteLongHorizonBackend.default_path().expanduser().resolve()


def _read_session_memories(session_id: str, limit: int, offset: int) -> dict:
    """Bounded SELECT-only view; no recall, usage update, schema or migration."""
    db_path = _memory_database_path()
    rows = []
    if db_path.exists():
        connection = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True, timeout=2.0)
        try:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, substr(kind, 1, 48) AS kind,
                       substr(summary, 1, 1500) AS summary,
                       substr(question, 1, 800) AS question,
                       length(kind) > 48 AS kind_truncated,
                       length(summary) > 1500 AS summary_truncated,
                       length(question) > 800 AS question_truncated,
                       created_at
                FROM long_horizon_memories
                WHERE session_id = ? AND scope = 'session'
                  AND scope_key = ? AND status = 'active'
                  AND typeof(id) = 'text' AND length(id) BETWEEN 1 AND 128
                ORDER BY last_seen_at DESC, id ASC
                LIMIT ? OFFSET ?
                """,
                (session_id, f"session:{session_id}", limit + 1, offset),
            ).fetchall()
        finally:
            connection.close()
    has_more = len(rows) > limit
    items = []
    for row in rows[:limit]:
        try:
            created_at = float(row["created_at"])
            if not math.isfinite(created_at) or not 0 <= created_at <= 32503680000:
                created_at = None
        except (TypeError, ValueError):
            created_at = None
        items.append({
            "id": row["id"], "kind": str(row["kind"] or ""),
            "summary": str(row["summary"] or ""), "question": str(row["question"] or ""),
            "created_at": created_at,
            "source_type": "assistant_summary", "question_source": "user_message",
            "truncated_fields": [name for name in ("kind", "summary", "question") if row[f"{name}_truncated"]],
        })
    next_offset = offset + limit if has_more and offset + limit <= 1000 else None
    return {"items": items, "scope": "session", "has_more": has_more,
            "next_offset": next_offset, "limit": limit, "offset": offset,
            "window_limited": has_more and next_offset is None}


@router.get("/memory")
async def memory(
    session_id: str,
    limit: Annotated[int, Query(ge=1, le=20)] = 20,
    offset: Annotated[int, Query(ge=0, le=1000)] = 0,
):
    sid = _session(session_id)
    try:
        return await asyncio.to_thread(_read_session_memories, sid, limit, offset)
    except Exception:
        # Do not expose database paths, content, configuration or driver errors.
        raise HTTPException(503, "Memory is unavailable") from None


def _query_journal():
    from ..request_journal import RequestJournal
    return RequestJournal(Path(_manager().base_storage_path), boot_id=_query_boot_id)


def _query_key(journal, sid: str, request_id: str) -> tuple[str, str, str]:
    return str(journal.path), sid, request_id


def _query_running(key) -> bool:
    task = _query_tasks.get(key)
    return key in _starting_query_keys or bool(task and not task.done())


def _query_concurrency_limit() -> int:
    return 1 if os.getenv("DOCTHINKER_INVITE_WORKER_ID", "").strip() else 2


def _query_response(result: dict) -> dict:
    response = {"answer": result.get("answer", ""), "sources": result.get("sources", [])}
    if result.get("context_windowed") is True:
        response["context_windowed"] = True
    answer_id = result.get("answer_id")
    if isinstance(answer_id, str) and re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", answer_id,
    ):
        response["answer_id"] = answer_id
    for name in ("research_core", "answer_mode", "citation_validation", "retrieval_gaps", "used_memory_count", "memory_selection", "reasoning", "claim_audit", "memory_as_of", "reasoning_mode", "knowledge_skill", "trial_delivery", "research_memory_required"):
        if name in result:
            response[name] = result[name]
    if result.get('research_provenance_required') is True:
        response['research_provenance_required'] = True
    return response


async def _hydrate_research_reply(result: dict):
    service = getattr(state, "research_service", None)
    if (result.get("research_core") is True or result.get("research_provenance_required") is True
            or result.get('research_memory_required') is True) and result.get("answer_id"):
        sid = result.get("_research_session_id")
        if sid:
            items = await asyncio.to_thread(hydrate_research_history,
                [{**result, "id": result["answer_id"], "role": "assistant"}], sid, service)
            # Hydration/redaction can deliberately remove proof fields; merging
            # the old response back would resurrect those discarded snippets.
            result = {key: value for key, value in items[0].items() if key not in {"id", "role"}}
        else:
            result = hide_unverified_research_reply(result)
    else:
        result = hide_unverified_research_reply(result)
    return {key: value for key, value in result.items() if key != "_research_session_id"}


async def _request_reply(result: dict):
    result = await _hydrate_research_reply(result)
    return JSONResponse(result, status_code=200 if result["request_status"] in {"completed", "failed"} else 202,
                        headers={"Cache-Control": "no-store"})


@router.get("/query/status")
async def query_status(session_id: str, request_id: Annotated[str, Query(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")]):
    sid = _session(session_id)
    try:
        journal = _query_journal()
        key = _query_key(journal, sid, request_id)
        result = await asyncio.to_thread(journal.read, sid, request_id, active=_query_running(key))
        return await _hydrate_research_reply(result)
    except Exception:
        raise HTTPException(503, "Request status is unavailable") from None


async def _run_journal_query(ask, body: MobileQuestion, journal, key):
    sid, request_id = body.session_id, body.request_id
    task_background = BackgroundTasks()
    try:
        result = await execute_run(
            lambda: ask(QueryRequest(**body.model_dump(exclude={"request_id"})), task_background),
            session_id=sid, run_id=request_id, timeout_seconds=QUERY_EXECUTION_SECONDS,
        )
        stored = _query_response(result)
        if stored.get("research_core") is True:
            # Only opaque refs survive in receipts; revoked raw quotes must not
            # be resurrected by replaying a previously completed request.
            stored["sources"] = [{"id": source["id"], "label": source["label"]} for source in stored["sources"]]
            stored.pop("reasoning", None)
            stored.pop("claim_audit", None)
            stored["_research_session_id"] = sid
        await asyncio.to_thread(journal.complete, sid, request_id, stored)
        response = await asyncio.to_thread(journal.read, sid, request_id)
        # Own task lifetime: FastAPI response background tasks cannot outlive a
        # disconnected 202 response. Existing enrichment remains best-effort,
        # separate from durable main-answer completion (safe workers add none).
        if task_background.tasks:
            task = asyncio.create_task(task_background())
            _query_background_tasks.add(task)
            def background_done(finished):
                _query_background_tasks.discard(finished)
                if not finished.cancelled():
                    finished.exception()
            task.add_done_callback(background_done)
        return response
    except ModelResponseError as error:
        # A response was received and rejected: unlike timeout/disconnection,
        # this outcome is known. Persist only its fixed code, not partial text.
        try:
            await asyncio.to_thread(journal.fail, sid, request_id, error.code)
            return await asyncio.to_thread(journal.read, sid, request_id)
        except Exception:
            # Storage failure must never imply durable completion or a free
            # retry. The accepted ID remains non-reusable even after restart.
            return {"request_id": request_id, "request_status": "unknown"}
    except (Exception, asyncio.CancelledError):
        # Cancellation/timeout/provider failures may already have incurred cost.
        # Never claim failure is safely retryable or automatically resubmit it.
        try:
            await asyncio.to_thread(journal.mark_unknown, sid, request_id)
        except Exception:
            pass
        return {"request_id": request_id, "request_status": "unknown"}


@router.post("/query")
async def query(body: MobileQuestion, background_tasks: BackgroundTasks):
    from .query import query as ask
    from ..request_journal import RequestConflictError, RequestJournalLimitError, request_fingerprint
    sid = _session(body.session_id)
    if not body.question.strip():
        raise HTTPException(422, "请输入问题")
    if os.getenv("DOCTHINKER_INVITE_WORKER_ID"):
        from ..invite_budget import InviteBudgetError, validate_invite_question
        try:
            validate_invite_question(body.question)
        except InviteBudgetError as error:
            raise HTTPException(error.status, "Question exceeds the invitation request budget") from None
    if body.request_id is None:
        if sid in _active_queries or len(_active_queries) >= _query_concurrency_limit():
            raise HTTPException(429, "已有问题正在处理中，请等待完成")
        _active_queries.add(sid)
        try:
            # Older clients remain compatible, but cannot claim reconnect dedup.
            result = await execute_run(
                lambda: ask(QueryRequest(**body.model_dump(exclude={"request_id"})), background_tasks),
                session_id=sid, timeout_seconds=QUERY_EXECUTION_SECONDS,
            )
            return _query_response(result)
        except ModelResponseError as error:
            return JSONResponse({"request_status": "failed", "request_error": model_response_failure(error.code)},
                                status_code=200, headers={"Cache-Control": "no-store"})
        except asyncio.TimeoutError:
            raise HTTPException(504, "处理超时，请先查看历史记录，避免重复提交") from None
        finally:
            _active_queries.discard(sid)
    try:
        journal = _query_journal()
        key = _query_key(journal, sid, body.request_id)
        fingerprint = request_fingerprint(body.model_dump())
        previous = await asyncio.to_thread(journal.read, sid, body.request_id,
                                           active=_query_running(key), fingerprint=fingerprint)
        if previous["request_status"] != "not_found":
            return await _request_reply(previous)
        if sid in _active_queries or len(_active_queries) >= _query_concurrency_limit():
            raise HTTPException(429, "已有问题正在处理中，请等待完成")
        _active_queries.add(sid)
        _starting_query_keys.add(key)
        try:
            first = await asyncio.to_thread(journal.begin, sid, body.request_id, fingerprint)
            if not first:
                _active_queries.discard(sid)
                return await _request_reply(await asyncio.to_thread(journal.read, sid, body.request_id))
            task = asyncio.create_task(_run_journal_query(ask, body, journal, key))
            _query_tasks[key] = task
            def completed(finished):
                if _query_tasks.get(key) is finished:
                    _query_tasks.pop(key, None)
                _active_queries.discard(sid)
                if not finished.cancelled():
                    finished.exception()
            task.add_done_callback(completed)
        except BaseException:
            _active_queries.discard(sid)
            raise
        finally:
            _starting_query_keys.discard(key)
        try:
            # Losing this HTTP client does not cancel the single model task.
            result = await asyncio.wait_for(asyncio.shield(task), QUERY_WAIT_SECONDS)
            return await _request_reply(result)
        except asyncio.TimeoutError:
            return await _request_reply(await asyncio.to_thread(journal.read, sid, body.request_id, active=not task.done()))
    except RequestConflictError:
        raise HTTPException(409, "Request ID belongs to different content") from None
    except RequestJournalLimitError:
        raise HTTPException(429, "Request storage capacity reached") from None
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(503, "Request processing is unavailable") from None


@router.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    session_id: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    original_filename: Annotated[str | None, Form()] = None,
):
    if getattr(state, "mobile_safe_mode", False) and not state.mobile_document_rag:
        await file.close()
        raise HTTPException(503, "Document processing requires separately configured embedding services")
    from .ingest import ingest_files
    _session(session_id)
    filename = Path((file.filename or "").replace("\\", "/")).name
    if Path(filename).suffix.lower() not in {".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg"}:
        await file.close()
        raise HTTPException(415, "支持 PDF、TXT、MD、DOCX 与图片")
    if original_filename is not None:
        name = original_filename
        if (
            not name or name != name.strip() or len(name.encode("utf-8")) > 255
            or "/" in name or "\\" in name or any(ord(char) < 32 or ord(char) == 127 for char in name)
            or Path(name).suffix.lower() != Path(filename).suffix.lower()
        ):
            await file.close()
            raise HTTPException(422, "Invalid original filename")
        # Display-name preservation only. This cannot turn an unknown/mismatched
        # upload type into an allowed document type.
        filename = name
    size = 0
    while chunk := await file.read(64 * 1024):
        size += len(chunk)
        if size > 20 * 1024 * 1024:
            await file.close()
            raise HTTPException(413, "单个文件不得超过 20 MB")
    if not size:
        await file.close()
        raise HTTPException(400, "不能上传空文件")
    await file.seek(0)
    file.filename = filename
    try:
        await ingest_files(background_tasks=background_tasks, files=[file], session_id=session_id, request=None)
        return {"status": "processing", "message": "已提交解析，请刷新文件状态"}
    finally:
        await file.close()


@router.get("/graph")
async def graph(session_id: str, focus_node: str | None = None):
    from .graph import get_graph_data
    result = await get_graph_data(session_id=_session(session_id), scope="summary", focus_node=focus_node)
    allowed = {"id", "label", "name", "type", "entity_type", "description", "degree", "is_discovered"}
    edges = {"id", "source", "target", "description", "keywords", "weight", "is_discovered"}
    return {"nodes": [{k: v for k, v in n.items() if k in allowed} for n in result.get("nodes", [])],
            "edges": [{k: v for k, v in e.items() if k in edges} for e in result.get("edges", [])]}


@router.get("/chunks")
async def chunks(session_id: str, entity_id: str):
    from .graph import get_entity_chunks
    result = await get_entity_chunks(session_id=_session(session_id), entity_id=entity_id, max_chunks=3)
    return {"chunks": [{"id": c.get("chunk_id", ""), "content": c.get("content", "")}
                       for c in result.get("chunks", [])]}


def _research():
    service = getattr(state, "research_service", None)
    if service is None:
        raise HTTPException(404, "Research profile is not enabled")
    return service


async def _research_call(function, *args, **kwargs):
    from ..research_store import (ResearchLimitError, ResearchNotFoundError,
                                  ResearchRevisionConflictError, ResearchStoreError)
    try:
        return await asyncio.to_thread(function, *args, **kwargs)
    except ResearchRevisionConflictError:
        raise HTTPException(409, "Record changed; refresh before editing") from None
    except ResearchNotFoundError:
        raise HTTPException(404, "Research record unavailable") from None
    except ResearchLimitError:
        raise HTTPException(429, "Research storage capacity reached") from None
    except ResearchStoreError:
        raise HTTPException(422, "Invalid research content") from None


def _public_document(document):
    return {key: document[key] for key in ("id", "session_id", "filename", "status", "chunk_count", "error_code")
            if key in document} | {"size_bytes": document["byte_size"]}


def _public_memory(memory):
    return ({key: memory[key] for key in ("id", "kind", "text", "status", "revision", "confirmed_by_user", "valid_from", "valid_until", "recorded_at", "recorded_until", "archived", "redacted", "historical", "version_status", "source_answer_id") if key in memory}
            | {"source_ids": memory["source_chunk_ids"]})


class ResearchDocumentReference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    session_id: str = Field(pattern=r"^#[0-9]{5}$")
    document_id: str = Field(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


class ResearchMemoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    session_id: str = Field(pattern=r"^#[0-9]{5}$")
    text: str = Field(min_length=1, max_length=2000)
    kind: Literal["preference", "decision", "finding"]
    source_ids: list[str] = Field(default_factory=list, max_length=20)
    memory_id: str | None = None
    expected_revision: int | None = Field(default=None, ge=1)
    source_answer_id: str | None = Field(default=None,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
    valid_from: float | None = Field(default=None, ge=0, le=253402300799, allow_inf_nan=False, strict=True)
    valid_until: float | None = Field(default=None, ge=0, le=253402300799, allow_inf_nan=False, strict=True)


class ResearchMemoryDelete(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    session_id: str = Field(pattern=r"^#[0-9]{5}$")
    memory_id: str
    expected_revision: int = Field(ge=1)


class ResearchMemoryArchive(ResearchMemoryDelete):
    archived: bool = True


@router.get("/research/capabilities")
async def research_capabilities():
    return {"enabled": getattr(state, "research_service", None) is not None,
            "profile": "bounded_research_v1", "supported_upload_types": ["pdf", "txt", "md"],
            "max_upload_bytes": 10 * 1024 * 1024, "max_documents_per_project": 20,
            "max_source_chunks": MAX_ANSWER_SOURCES, "max_source_context_bytes": 12000,
            "memory_write_enabled": getattr(state, "research_service", None) is not None,
            "knowledge_skills": getattr(state, "research_service", None) is not None,
            "uploads_allowed": getattr(state, "research_service", None) is not None,
            "memory_policy": "explicit_confirmation", "automatic_learning": False,
            "temporal_memory": True, "indexed_memory": True,
            "memory_scope": "session", "memory_answer_provenance": True,
            "memory_revision_revalidation": True,
            "context_assistance": True, "interaction": "natural_chat_and_writing",
            "background_algorithms": ["analogy", "deduction", "induction"],
            "background_max_operators": 1, "background_wait_seconds": 1.0,
            "ocr": False, "autonomous_tasks": False, "semantic_entailment_verified": False}


@router.get('/research/knowledge-skills')
async def research_knowledge_skills():
    _research()
    from ..knowledge_skills import knowledge_skill_catalog
    return {'items': knowledge_skill_catalog()}


@router.get("/research/documents")
async def research_documents(session_id: str):
    service, sid = _research(), _session(session_id)
    documents = await _research_call(service.store.list_documents, sid)
    return {"documents": [_public_document(doc) for doc in documents if doc["status"] != "deleted"]}


@router.get("/research/document")
async def research_document(session_id: str, document_id: str):
    service, sid = _research(), _session(session_id)
    doc = await _research_call(service.store.get_document, sid, document_id)
    if doc["status"] == "deleted":
        raise HTTPException(404, "Document unavailable")
    sources = await _research_call(service.store.source_chunks, sid, document_id=document_id, limit=20)
    return {"document": {**_public_document(doc), "chunks": sources,
                         "source_ids": [chunk["id"] for chunk in sources], "preview_limited": doc["chunk_count"] > 20}}


@router.get("/research/source")
async def research_source(session_id: str, chunk_id: str):
    service, sid = _research(), _session(session_id)
    sources = await _research_call(service.store.source_chunks, sid, [chunk_id])
    if not sources:
        raise HTTPException(404, "Source unavailable")
    return {"chunk": sources[0]}


@router.post("/research/upload")
async def research_upload(session_id: Annotated[str, Form()], file: Annotated[UploadFile, File()],
                          original_filename: Annotated[str | None, Form()] = None):
    try:
        service, sid = _research(), _session(session_id)
        filename = original_filename or file.filename or ""
        if not filename or Path(filename).suffix.lower() not in {".pdf", ".txt", ".md"}:
            raise HTTPException(415, "Only PDF/TXT/MD supported")
        if original_filename and Path(original_filename).suffix.lower() != Path(file.filename or "").suffix.lower():
            raise HTTPException(422, "Filename type mismatch")
        pieces, size = [], 0
        while chunk := await file.read(65536):
            size += len(chunk)
            if size > 10 * 1024 * 1024:
                raise HTTPException(413, "File exceeds 10 MB")
            pieces.append(chunk)
        document = await _research_call(service.store.accept_document, sid, filename, b"".join(pieces))
        service.notify()
        return {"document": _public_document(document), "deduplicated": document.get("deduplicated", False)}
    finally:
        await file.close()


@router.post("/research/documents/delete")
async def research_document_delete(body: ResearchDocumentReference):
    service, sid = _research(), _session(body.session_id)
    result = await _research_call(service.store.delete_document, sid, body.document_id)
    return {"removed": True, "invalidated_memory_count": len(result["invalidated_memory_ids"])}


@router.post("/research/documents/retry")
async def research_document_retry(body: ResearchDocumentReference):
    service, sid = _research(), _session(body.session_id)
    document = await _research_call(service.store.retry_document, sid, body.document_id)
    service.notify()
    return {"document": _public_document(document)}


@router.get("/research/memory")
async def research_memory(session_id: str, query: str = "", cursor: str | None = None,
                          limit: int = 20, as_of: float | None = None, include_archived: bool = False):
    service, sid = _research(), _session(session_id)
    page = await _research_call(service.store.page_memories, sid, query=query, cursor=cursor,
                               limit=limit, as_of=as_of, include_archived=include_archived)
    return {**page, "items": [_public_memory(item) for item in page["items"]]}


@router.get("/research/memory/history")
async def research_memory_history(session_id: str, memory_id: str, limit: int = 20, cursor: str | None = None):
    service, sid = _research(), _session(session_id)
    page = await _research_call(service.store.memory_history, sid, memory_id, limit=limit, cursor=cursor)
    return {**page, "items": [_public_memory(item) for item in page["items"]]}


@router.post("/research/memory/archive")
async def research_memory_archive(body: ResearchMemoryArchive):
    service, sid = _research(), _session(body.session_id)
    memory = await _research_call(service.store.archive_memory, sid, body.memory_id,
                                 expected_revision=body.expected_revision, archived=body.archived)
    return {"memory": _public_memory(memory)}


@router.post("/research/memory")
async def research_memory_save(body: ResearchMemoryRequest):
    service, sid = _research(), _session(body.session_id)
    temporal = {key: getattr(body, key) for key in ("valid_from", "valid_until") if key in body.model_fields_set}
    origin = ({'source_answer_id': body.source_answer_id}
              if 'source_answer_id' in body.model_fields_set else {})
    if body.memory_id is not None:
        if body.expected_revision is None:
            raise HTTPException(422, "Revision required")
        memory = await _research_call(service.store.update_memory, sid, body.memory_id,
                                     expected_revision=body.expected_revision, text=body.text,
                                     kind=body.kind, source_chunk_ids=body.source_ids if 'source_ids' in body.model_fields_set else None, **temporal, **origin)
    else:
        if body.expected_revision is not None:
            raise HTTPException(422, "Unexpected revision")
        memory = await _research_call(service.store.create_memory, sid, body.kind, body.text,
                                     source_chunk_ids=body.source_ids, **temporal, **origin)
    return {"memory": _public_memory(memory)}


@router.post("/research/memory/delete")
async def research_memory_delete(body: ResearchMemoryDelete):
    service, sid = _research(), _session(body.session_id)
    await _research_call(service.store.delete_memory, sid, body.memory_id, expected_revision=body.expected_revision)
    return {"deleted": True}


@router.get("/research/export")
async def research_export(session_id: str):
    service, sid = _research(), _session(session_id)
    docs = await _research_call(service.store.list_documents, sid)
    memories, cursor = [], None
    while True:
        page = await _research_call(service.store.page_memories, sid, cursor=cursor, limit=100, include_archived=True)
        memories.extend(page["items"])
        if sum(len(item["text"].encode("utf-8")) + 512 for item in memories) > 800000:
            raise HTTPException(429, "Export too large; narrow the project before exporting")
        cursor = page["next_cursor"]
        if not cursor:
            break
    lines = ["# 研究项目导出", "", "这是当前项目的资料清单与用户确认的记忆，不含全文或所有聊天记录。", "", "## 资料", ""]
    for document in docs:
        if document["status"] != "deleted":
            lines.append(f"- {document['filename']}（{document['status']}；{document['chunk_count']} 个片段）")
    lines += ["", "## 用户确认的记忆", ""]
    for memory in memories:
        if memory["status"] not in {"active", "archived"}:
            continue
        lines += [f"### {memory['kind']} · v{memory['revision']} · {memory['status']}", "", memory["text"], "",
                  "来源编号：" + (", ".join(memory["source_chunk_ids"]) or "用户自行输入，无原文引用"), ""]
    markdown = "\n".join(lines)
    if len(markdown.encode("utf-8")) > 800000:
        raise HTTPException(429, "Export too large")
    return {"markdown": markdown}


# Same protected facade, not a new public/admin server or execution engine.
from .workbench import router as workbench_router
router.include_router(workbench_router)
from .project_tasks import router as project_tasks_router
router.include_router(project_tasks_router)
from .personal_events import router as personal_events_router
router.include_router(personal_events_router)
from .long_projects import router as long_projects_router
router.include_router(long_projects_router)
