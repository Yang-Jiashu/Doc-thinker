"""Same-origin invitation gateway to fixed, isolated loopback workers.

Run one gateway process/worker: request/rate limits are process-local, while
credentials, revocation and daily usage quotas are persisted atomically.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException

from .invite_store import InviteError, InviteStore

PREFIX = "/api/v1"
AUTH_LIMIT = 8192
BUSINESS_LIMIT = 20 * 1024 * 1024 + 65536
RESPONSE_LIMIT = 8 * 1024 * 1024
UPSTREAM_TIMEOUT_SECONDS = 125
MOBILE_ROUTES = {
    ("GET", "capabilities"): set(),
    ("GET", "sessions"): set(),
    ("POST", "sessions"): set(),
    ("GET", "history"): {"session_id"},
    ("GET", "feedback"): {"session_id"},
    ("POST", "feedback"): set(),
    ("GET", "files"): {"session_id"},
    ("GET", "memory"): {"session_id", "limit", "offset"},
    ("GET", "graph"): {"session_id", "focus_node"},
    ("GET", "chunks"): {"session_id", "entity_id"},
    ("POST", "query"): set(),
    ("GET", "query/status"): {"session_id", "request_id"},
    ("POST", "upload"): set(),
    ("GET", "voice/capabilities"): set(),
    ("POST", "voice/transcribe"): set(),
    ("POST", "voice/speak"): set(),
    ("GET", "research/capabilities"): set(),
    ("GET", "research/knowledge-skills"): set(),
    ("GET", "research/documents"): {"session_id"},
    ("GET", "research/document"): {"session_id", "document_id"},
    ("GET", "research/source"): {"session_id", "chunk_id"},
    ("POST", "research/upload"): set(),
    ("POST", "research/documents/delete"): set(),
    ("POST", "research/documents/retry"): set(),
    ("GET", "research/memory"): {"session_id", "query", "cursor", "limit", "as_of", "include_archived"},
    ("GET", "research/memory/history"): {"session_id", "memory_id", "limit", "cursor"},
    ("POST", "research/memory/archive"): set(),
    ("POST", "research/memory"): set(),
    ("POST", "research/memory/delete"): set(),
    ("GET", "research/export"): {"session_id"},
    ("GET", "workbench/brief"): {"session_id"},
    ("POST", "workbench/brief"): set(),
    ("GET", "workbench/artifacts"): {"session_id"},
    ("GET", "workbench/artifact"): {"session_id", "answer_id"},
    ("POST", "workbench/artifacts"): set(),
    ("POST", "workbench/artifacts/delete"): set(),
    ("GET", "workbench/tasks"): {"session_id"},
    ("POST", "workbench/tasks"): set(),
    ("GET", "workbench/task"): {"session_id", "task_id"},
    ("POST", "workbench/tasks/update"): set(),
    ("POST", "workbench/tasks/replan/preview"): set(),
    ("POST", "workbench/tasks/replan"): set(),
    ("POST", "workbench/tasks/evidence/preview"): set(),
    ("POST", "workbench/tasks/run"): set(),
    ("POST", "workbench/tasks/control"): set(),
    ("GET", "workbench/task/artifact"): {"session_id", "task_id", "version", "step_id"},
    ("GET", "workbench/long-projects"): {"session_id", "cursor", "limit"},
    ("POST", "workbench/long-projects"): set(),
    ("GET", "workbench/long-project"): {"session_id", "project_id"},
    ("GET", "workbench/long-project/phases"): {"session_id", "project_id", "cursor", "limit"},
    ("POST", "workbench/long-project/phases"): set(),
    ("POST", "workbench/long-project/phases/review"): set(),
    ("POST", "workbench/long-project/review"): set(),
    ("POST", "research/entities"): set(), ("GET", "research/entities/resolve"): {"session_id", "kind", "name"},
    ("GET", "research/events"): {"session_id", "entity_id", "limit"}, ("POST", "research/events"): set(),
    ("POST", "research/events/revise"): set(), ("POST", "research/events/snapshots"): set(),
    ("GET", "research/event"): {"session_id", "event_id"}, ("GET", "research/event/history"): {"session_id", "event_id", "limit"},
    ("GET", "research/event-relations"): {"session_id", "event_id", "limit"}, ("POST", "research/event-relations"): set(),
}
JSON_FIELDS = {
    "sessions": {"title"},
    "feedback": {"session_id", "answer_id", "rating", "note"},
    "query": {"session_id", "question", "request_id", "use_memory", "use_conversation_context", "use_llm_cache", "use_self_evolution", "reasoning_mode", "memory_as_of", "knowledge_skill_mode", "insight_mode"},
    "voice/speak": {"text"},
    "research/documents/delete": {"session_id", "document_id"},
    "research/documents/retry": {"session_id", "document_id"},
    "research/memory": {"session_id", "text", "kind", "source_ids", "memory_id", "expected_revision", "valid_from", "valid_until"},
    "research/memory/archive": {"session_id", "memory_id", "expected_revision", "archived"},
    "research/memory/delete": {"session_id", "memory_id", "expected_revision"},
    "workbench/brief": {"session_id", "expected_revision", "goal", "output_format", "constraints"},
    "workbench/artifacts": {"session_id", "answer_id", "title"},
    "workbench/artifacts/delete": {"session_id", "answer_id"},
    "workbench/tasks": {"session_id", "goal", "constraints", "questions"},
    "workbench/tasks/update": {"session_id", "task_id", "expected_revision", "goal", "constraints", "questions"},
    "workbench/tasks/replan/preview": {"session_id", "task_id", "expected_revision", "goal", "constraints", "questions"},
    "workbench/tasks/replan": {"session_id", "task_id", "expected_revision", "goal", "constraints", "questions", "proposal_sha256"},
    "workbench/tasks/evidence/preview": {"session_id", "task_id", "expected_revision"},
    "workbench/tasks/run": {"session_id", "task_id", "expected_revision", "request_id", "max_calls"},
    "workbench/tasks/control": {"session_id", "task_id", "expected_revision", "action"},
    "workbench/long-projects": {"session_id", "request_id", "goal", "acceptance_criteria"},
    "workbench/long-project/phases": {"session_id", "project_id", "expected_revision", "request_id", "title", "acceptance_criteria", "questions", "constraints", "parent_refs"},
    "workbench/long-project/phases/review": {"session_id", "project_id", "phase_id", "expected_revision", "request_id", "report_version", "decision"},
    "workbench/long-project/review": {"session_id", "project_id", "expected_revision", "request_id", "phase_reports", "decision"},
    "research/entities": {"session_id", "confirmed", "kind", "name", "aliases", "identity_key"},
    "research/events": {"session_id", "confirmed", "event_type", "action", "actor_id", "object_id", "occurred_from", "occurred_until", "conditions", "origins"},
    "research/events/revise": {"session_id", "confirmed", "event_id", "expected_revision", "event_type", "action", "actor_id", "object_id", "occurred_from", "occurred_until", "conditions", "origins"},
    "research/events/snapshots": {"session_id", "event_ids"},
    "research/event-relations": {"session_id", "confirmed", "source_id", "target_id", "relation", "conditions", "origins"},
}


def validated_origin(value: str) -> tuple[str, str]:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise InviteError("Invalid invitation origin") from None
    if (not value or any(c.isspace() for c in value) or parsed.username is not None
            or parsed.password is not None or parsed.path or parsed.query or parsed.fragment
            or parsed.scheme not in {"http", "https"} or not parsed.hostname
            or not re.fullmatch(r"[A-Za-z0-9.-]+", parsed.hostname)
            or (parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost"})
            or (port is not None and not 1 <= port <= 65535)):
        raise InviteError("Invitation origin must be HTTPS or an explicit loopback development origin")
    normalized = f"{parsed.scheme}://{parsed.netloc.lower()}"
    if value != normalized:
        raise InviteError("Invitation origin must be canonical and contain no path")
    return normalized, parsed.netloc.lower()


class RateLimits:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.entries: OrderedDict[tuple, deque] = OrderedDict()

    def check(self, key: tuple, limit: int, seconds: int) -> None:
        now = self.clock()
        with self.lock:
            values = self.entries.setdefault(key, deque())
            self.entries.move_to_end(key)
            while values and values[0] <= now - seconds:
                values.popleft()
            if len(values) >= limit:
                raise InviteError("Too many requests; try again later", 429)
            values.append(now)
            while len(self.entries) > 4096:
                self.entries.popitem(last=False)


class ActiveRequests:
    def __init__(self):
        self.lock = threading.Lock()
        self.users: set[str] = set()

    def acquire(self, user_id: str) -> None:
        with self.lock:
            if user_id in self.users or len(self.users) >= 4:
                raise InviteError("Another request is in progress; wait before trying again", 429)
            self.users.add(user_id)

    def release(self, user_id: str) -> None:
        with self.lock:
            self.users.discard(user_id)


async def _bounded_body(request: Request, limit: int) -> bytes:
    if request.headers.get("content-encoding", "identity").lower() != "identity":
        raise InviteError("Compressed request bodies are not accepted", 415)
    announced = request.headers.get("content-length")
    if announced is not None:
        try:
            size = int(announced)
        except ValueError:
            raise InviteError("Invalid request size", 400) from None
        if size < 0 or size > limit:
            raise InviteError("Request body is too large", 413)

    async def collect():
        parts, size = [], 0
        async for part in request.stream():
            size += len(part)
            if size > limit:
                raise InviteError("Request body is too large", 413)
            parts.append(part)
        return b"".join(parts)
    try:
        return await asyncio.wait_for(collect(), timeout=15)
    except asyncio.TimeoutError:
        raise InviteError("Request body timed out", 408) from None


def _json_object(body: bytes, *, fields: set[str], required: set[str] = frozenset()) -> dict:
    try:
        value = json.loads(body)
    except (ValueError, UnicodeError):
        raise InviteError("Invalid JSON body", 422) from None
    if not isinstance(value, dict) or not required.issubset(value) or set(value) - fields:
        raise InviteError("Unexpected or missing request fields", 422)
    return value


def _token(request: Request) -> str:
    values = request.headers.getlist("authorization")
    if len(values) != 1 or not values[0].startswith("Bearer "):
        raise InviteError("Authentication required", 401)
    return values[0][7:]


def _upstream(user: dict) -> str:
    # Re-validate persisted bindings at use time, not just in the CLI.
    target = user.get("upstream_url", "")
    match = re.fullmatch(r"http://127\.0\.0\.1:([0-9]{4,5})", target)
    token = user.get("upstream_token", "")
    if not match or not 1024 <= int(match[1]) <= 65535 or not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", token):
        raise InviteError("User service is unavailable", 503)
    return target


def create_invite_app(*, store: InviteStore | None = None,
                      transport: httpx.AsyncBaseTransport | None = None,
                      origin: str | None = None) -> FastAPI:
    store = store if store is not None else InviteStore.from_environment()
    allowed_origin, allowed_host = validated_origin(origin if origin is not None else os.getenv("DOCTHINKER_INVITE_ORIGIN", ""))
    rates, active = RateLimits(), ActiveRequests()
    auth_slots = asyncio.Semaphore(2)
    reconnect_slots = asyncio.Semaphore(4)
    status_slots = asyncio.Semaphore(2)
    auth_jobs: set[asyncio.Task] = set()
    uploads_enabled = os.getenv("DOCTHINKER_INVITE_UPLOADS_ENABLED", "").lower() == "true"
    research_enabled = os.getenv("DOCTHINKER_RESEARCH_ENABLED", "").lower() in {"true", "1"}
    client = httpx.AsyncClient(transport=transport, trust_env=False, follow_redirects=False,
                              timeout=httpx.Timeout(120, connect=3), limits=httpx.Limits(max_connections=4))
    # Recovery reads must remain possible while all four POST streams wait.
    # A separate pool/slot protects status from model-query transport saturation.
    status_client = httpx.AsyncClient(transport=transport, trust_env=False, follow_redirects=False,
                                     timeout=httpx.Timeout(5, connect=2), limits=httpx.Limits(max_connections=2))

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            if auth_jobs:
                await asyncio.gather(*tuple(auth_jobs), return_exceptions=True)
            await client.aclose()
            await status_client.aclose()

    app = FastAPI(title="DocThinker Invitation Gateway", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None, redirect_slashes=False)
    app.state.invite_store = store
    app.state.invite_upstream_client = client
    app.state.invite_status_upstream_client = status_client

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        try:
            # Do not trust X-Forwarded-Host/Origin/IP. TLS reverse proxies must
            # preserve the configured Host and enforce HTTPS at their edge.
            if len(request.headers.getlist("host")) != 1 or request.headers["host"].lower() != allowed_host:
                raise InviteError("Host is not allowed", 400)
            origins = request.headers.getlist("origin")
            if (len(origins) > 1 or (origins and origins[0] != allowed_origin)
                    or (request.method not in {"GET", "HEAD"} and not origins)):
                raise InviteError("Same-origin requests are required", 403)
            if request.headers.get("sec-fetch-site") in {"cross-site", "same-site"}:
                raise InviteError("Same-origin requests are required", 403)
            response = await call_next(request)
        except InviteError as error:
            response = JSONResponse({"detail": str(error)}, status_code=error.status)
        except Exception:
            response = JSONResponse({"detail": "Service unavailable"}, status_code=503)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(InviteError)
    async def expected_error(_request, error):
        return JSONResponse({"detail": str(error)}, status_code=error.status)

    @app.exception_handler(HTTPException)
    async def http_error(_request, error):
        return JSONResponse({"detail": "Request unavailable"}, status_code=error.status_code)

    def no_query(request):
        if request.query_params:
            raise InviteError("Query parameters are not accepted here", 400)

    @app.get(PREFIX + "/invite/status")
    async def status(request: Request):
        no_query(request)
        return {"service": "docthinker-invite", "registration": "invitation-only"}

    async def credentials(request: Request, *, register: bool):
        no_query(request)
        ip = request.client.host if request.client else "unknown"
        rates.check(("auth-global",), 120, 60)
        rates.check(("auth-ip", ip), 20, 60)
        if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
            raise InviteError("JSON is required", 415)
        required = {"username", "password", "invite_code"} if register else {"username", "password"}
        data = _json_object(await _bounded_body(request, AUTH_LIMIT), fields=required, required=required)
        if not all(isinstance(value, str) for value in data.values()):
            raise InviteError("Credential fields must be strings", 422)
        rates.check(("auth-name", data["username"].lower()), 10, 900)
        try:
            await asyncio.wait_for(auth_slots.acquire(), timeout=0.1)
        except asyncio.TimeoutError:
            raise InviteError("Authentication service is busy", 429) from None
        method = store.register if register else store.login
        # A disconnected client must not free its scrypt slot while the thread
        # still runs. Keep a reference and observe every result at completion.
        job = asyncio.create_task(asyncio.to_thread(method, **data))
        auth_jobs.add(job)
        def completed(task):
            auth_jobs.discard(task)
            auth_slots.release()
            if not task.cancelled():
                task.exception()
        job.add_done_callback(completed)
        return await asyncio.shield(job)

    @app.post(PREFIX + "/invite/register")
    async def register(request: Request):
        return await credentials(request, register=True)

    @app.post(PREFIX + "/invite/login")
    async def login(request: Request):
        return await credentials(request, register=False)

    @app.get(PREFIX + "/invite/me")
    async def me(request: Request):
        no_query(request)
        user = await asyncio.to_thread(store.authenticate, _token(request))
        return {"user": {"id": user["id"], "username": user["username"]}}

    @app.post(PREFIX + "/invite/logout")
    async def logout(request: Request):
        no_query(request)
        token = _token(request)
        await asyncio.to_thread(store.authenticate, token)
        await _bounded_body(request, AUTH_LIMIT)
        await asyncio.to_thread(store.logout, token)
        return {"ok": True}

    async def upstream_request(user: dict, method: str, route: str, *, params=None, body=b"", content_type=None):
        headers = {"Authorization": "Bearer " + user["upstream_token"]}
        if content_type:
            headers["Content-Type"] = content_type
        upstream_client = status_client if method == "GET" and route == "query/status" else client
        try:
            async with upstream_client.stream(method, _upstream(user) + PREFIX + "/mobile/" + route,
                                     params=params, content=body, headers=headers) as upstream:
                # Never expose an upstream Location, cookie, provider error or
                # internal HTTP body when a worker is unavailable/misconfigured.
                if not 200 <= upstream.status_code < 300:
                    code = upstream.status_code if upstream.status_code in {400, 404, 409, 413, 415, 422, 429, 503, 504} else 502
                    raise InviteError("User service could not complete the request", code)
                pieces, size = [], 0
                async for chunk in upstream.aiter_bytes():
                    size += len(chunk)
                    if size > RESPONSE_LIMIT:
                        raise InviteError("User service response is too large", 502)
                    pieces.append(chunk)
                return b"".join(pieces), upstream.headers.get("content-type", "").split(";", 1)[0], upstream.status_code
        except httpx.HTTPError:
            raise InviteError("User service is unavailable; no automatic retry was sent", 502) from None

    @app.api_route(PREFIX + "/mobile/{route:path}", methods=["GET", "POST"])
    async def proxy(route: str, request: Request):
        allowed_params = MOBILE_ROUTES.get((request.method, route))
        if allowed_params is None:
            raise InviteError("Route unavailable", 404)
        # Authenticate before reading a body. Reconnect/status requests have a
        # bounded separate slot so they can inspect a currently running answer.
        user = await asyncio.to_thread(store.authenticate, _token(request))
        if route.startswith("research/") and route != "research/capabilities" and not research_enabled:
            raise InviteError("Research profile is not enabled", 403)
        if (route in {"workbench/artifacts", "workbench/artifact", "workbench/task", "workbench/task/artifact"}
                or route.startswith('workbench/tasks') or route.startswith('workbench/long-project')) and not research_enabled:
            raise InviteError("Artifact provenance service is not enabled", 403)
        params = list(request.query_params.multi_items())
        if len(params) != len({key for key, _ in params}) or any(key not in allowed_params for key, _ in params):
            raise InviteError("Unexpected query parameters", 400)
        if any(len(value) > 1000 for _, value in params):
            raise InviteError("Query parameter is too long", 422)
        rates.check(("business-user", user["id"]), 120, 60)
        rates.check(("business-global",), 300, 60)
        preparsed_body, query_data, fingerprint = None, None, None
        query_units = 1
        if request.method == "POST" and route == "query":
            if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
                raise InviteError("JSON is required", 415)
            preparsed_body = await _bounded_body(request, 65536)
            query_data = _json_object(preparsed_body, fields=JSON_FIELDS["query"])
            from .invite_budget import InviteBudgetError, validate_invite_question
            try:
                validate_invite_question(query_data.get("question"))
            except InviteBudgetError as error:
                raise InviteError("Question exceeds the invitation request budget", error.status) from None
            if query_data.get("request_id") is not None:
                from .request_journal import request_fingerprint, validate_request_reference
                try:
                    validate_request_reference(query_data.get("session_id"), query_data["request_id"])
                    fingerprint = request_fingerprint(query_data)
                except ValueError:
                    raise InviteError("Invalid query request", 422) from None
        elif request.method == 'POST' and route == 'workbench/tasks/run':
            if request.headers.get('content-type', '').split(';', 1)[0].strip() != 'application/json':
                raise InviteError('JSON is required', 415)
            preparsed_body = await _bounded_body(request, 65536)
            approval = _json_object(preparsed_body, fields=JSON_FIELDS[route])
            from .routers.project_tasks import RunTask
            try:
                validated = RunTask.model_validate(approval)
            except Exception:
                raise InviteError('Invalid project execution approval', 422) from None
            query_units = validated.max_calls
            query_data = {'session_id': validated.session_id, 'request_id': validated.request_id}
            fingerprint = hashlib.sha256(json.dumps({'kind': 'project-task-run', **validated.model_dump()},
                sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        status_read = route == "query/status"
        reconnect = status_read or fingerprint is not None
        reserved_slots = status_slots if status_read else reconnect_slots
        if reconnect:
            try:
                await asyncio.wait_for(reserved_slots.acquire(), timeout=0.1)
            except asyncio.TimeoutError:
                raise InviteError("Request status service is busy", 429) from None
        else:
            active.acquire(user["id"])
        try:
            if route in {"research/upload", "research/documents/retry"} and (not research_enabled or not user["allow_uploads"]):
                raise InviteError("Research uploads are disabled for this user", 403)
            if route == "upload":
                if not uploads_enabled or not user["allow_uploads"]:
                    raise InviteError("Uploads are disabled for this private instance", 403)
                raw, _, _ = await asyncio.wait_for(upstream_request(user, "GET", "capabilities"), UPSTREAM_TIMEOUT_SECONDS)
                try:
                    ready = json.loads(raw)
                except (ValueError, UnicodeError):
                    raise InviteError("Document service is unavailable", 503) from None
                if not isinstance(ready, dict) or ready.get("uploads") is not True or ready.get("document_rag") is not True:
                    raise InviteError("Document processing is not configured", 503)
            body, content_type = b"", None
            if request.method == "POST":
                limit = (10 * 1024 * 1024 + 65536 if route == "research/upload" else BUSINESS_LIMIT if route in {"upload", "voice/transcribe"}
                         else 4096 if route == "feedback" else 65536)
                body = preparsed_body if preparsed_body is not None else await _bounded_body(request, limit)
                content_type = request.headers.get("content-type", "")
                if route in JSON_FIELDS:
                    if content_type.split(";", 1)[0].strip() != "application/json":
                        raise InviteError("JSON is required", 415)
                    _json_object(body, fields=JSON_FIELDS[route])
                elif not content_type.startswith("multipart/form-data; boundary="):
                    raise InviteError("Multipart upload is required", 415)
                # Re-parsing spends the same finite extraction allowance as an
                # upload, not the much larger generic metadata-write allowance.
                # Its existing source is not a newly uploaded byte payload.
                kind = ("upload" if route in {"research/upload", "research/documents/retry"} else "workbench_write" if route.startswith("workbench/")
                        else "research_write" if route.startswith("research/")
                        else "voice" if route.startswith("voice/") else route)
                if fingerprint is not None:
                    await asyncio.to_thread(store.reserve_query, user["id"], query_data["session_id"],
                                            query_data["request_id"], fingerprint, body_bytes=len(body), units=query_units)
                else:
                    await asyncio.to_thread(store.reserve_action, user["id"], kind,
                                            body_bytes=0 if route == "research/documents/retry" else len(body))
            data, media_type, code = await asyncio.wait_for(
                upstream_request(user, request.method, route, params=params, body=body, content_type=content_type),
                UPSTREAM_TIMEOUT_SECONDS,
            )
            if route == "voice/speak":
                if media_type not in {"audio/mpeg", "audio/wav", "audio/x-wav", "audio/ogg", "audio/mp4", "audio/webm"}:
                    raise InviteError("Invalid audio response", 502)
                return Response(data, status_code=code, media_type=media_type)
            if media_type != "application/json":
                raise InviteError("Invalid user service response", 502)
            try:
                result = json.loads(data)
            except (ValueError, UnicodeError):
                raise InviteError("Invalid user service response", 502) from None
            if route == "research/capabilities" and isinstance(result, dict):
                result = {**result, "enabled": research_enabled and result.get("enabled") is True,
                          "knowledge_skills": research_enabled and result.get("knowledge_skills") is True,
                          "uploads_allowed": research_enabled and bool(user["allow_uploads"]),
                          "memory_write_enabled": research_enabled and result.get("enabled") is True and result.get("memory_write_enabled") is True}
            if route == "capabilities" and isinstance(result, dict) and isinstance(result.get("workbench"), dict):
                result["workbench"] = {**result["workbench"], "artifacts": research_enabled and result["workbench"].get("artifacts") is True}
                if 'tasks' in result['workbench']:
                    result['workbench']['tasks'] = research_enabled and result['workbench']['tasks'] is True
                if 'long_projects' in result['workbench']:
                    result['workbench']['long_projects'] = research_enabled and result['workbench']['long_projects'] is True
            return JSONResponse(result, status_code=code)
        except asyncio.TimeoutError:
            raise InviteError("User service timed out; inspect history before retrying", 504) from None
        finally:
            if reconnect:
                reserved_slots.release()
            else:
                active.release(user["id"])

    return app
