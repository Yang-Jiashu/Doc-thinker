"""Explicit user workbench metadata; no model, hidden prompt or new job queue."""
import asyncio
from pathlib import Path
import re

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..mobile_access import MobileRoute, require_mobile_access
from ..workbench_store import (ANSWER_ID, MAX_SESSION_ARTIFACTS, WorkbenchConflictError,
                               WorkbenchLimitError, WorkbenchStore)

router = APIRouter(prefix="/workbench", dependencies=[Depends(require_mobile_access)], route_class=MobileRoute)


class BriefRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    session_id: str = Field(pattern=r"^#[0-9]{5}$")
    expected_revision: int = Field(ge=0, lt=2_147_483_647)
    goal: str = Field(max_length=1200)
    output_format: str = Field(max_length=400)
    constraints: str = Field(max_length=1200)

    @field_validator("goal", "output_format", "constraints")
    @classmethod
    def clean_text(cls, value):
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", value):
            raise ValueError("Invalid brief text")
        value.encode("utf-8")
        return value


class ArtifactReference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    session_id: str = Field(pattern=r"^#[0-9]{5}$")
    answer_id: str = Field(pattern="^" + ANSWER_ID + "$")


class ArtifactRequest(ArtifactReference):
    title: str = Field(min_length=1, max_length=120)

    @field_validator("title")
    @classmethod
    def clean_title(cls, value):
        if not value.strip() or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", value):
            raise ValueError("Invalid artifact title")
        value.encode("utf-8")
        return value.strip()


def _context(session_id):
    from .mobile import _manager, _session
    sid = _session(session_id)
    manager = _manager()
    session = manager.get_session(sid)
    if not session or not isinstance(session.get("instance_id"), str) or not session["instance_id"]:
        raise HTTPException(503, "Stable project identity is unavailable")
    return sid, manager, WorkbenchStore(Path(manager.base_storage_path), session_instance_id=session["instance_id"])


async def _call(function, *args, **kwargs):
    try:
        return await asyncio.to_thread(function, *args, **kwargs)
    except WorkbenchConflictError:
        raise HTTPException(409, "Brief has changed; reload it before saving") from None
    except WorkbenchLimitError:
        raise HTTPException(429, "Workbench storage capacity reached") from None
    except Exception:
        raise HTTPException(503, "Workbench is unavailable") from None


def _research():
    from .mobile import state
    service = getattr(state, "research_service", None)
    if service is None:
        # Raw history can contain a withdrawn research answer. Never bypass
        # hydration just because a worker restarted without the research core.
        raise HTTPException(503, "Artifact provenance service is unavailable")
    return service


async def _answers(sid, manager, service, answer_ids):
    history = await _call(manager.get_history, sid)
    # Hydrate only requested assistant turns so bookmarks older than the normal
    # last-200 history window still receive current revocation checks.
    selected = [{**{key: item[key] for key in ("id", "role", "content", "created_at", "timestamp",
                                             "research_provenance_required", "research_memory_required") if key in item},
                 "research_core": True}
                for item in history if item.get("role") == "assistant" and item.get("id") in answer_ids]
    hydrated = await _call(service.hydrate_history, sid, selected) if selected else []
    return {item["id"]: item for item in hydrated}


def _artifact(record, answers):
    answer = answers.get(record["answer_id"])
    warnings = {warning.get("code") for warning in (answer or {}).get("citation_validation", {}).get("warnings", [])}
    if "source_revoked" in warnings or 'memory_context_revoked' in warnings:
        # A normal generated answer may directly quote withdrawn material even
        # without a recorded inference chain. Never export that raw answer.
        reason = 'memory_context_revoked' if 'memory_context_revoked' in warnings else 'source_revoked'
        return {**record, "available": False, "answer": None, "unavailable_reason": reason}
    if answer is not None and (not isinstance(answer.get("reasoning"), dict)
                              or not isinstance(answer.get("claim_audit"), dict)
                              or "provenance_unavailable" in warnings):
        # A lost metadata store cannot turn a potentially revoked historical
        # answer into a new raw-text export. Keep the bookmark, not its content.
        return {**record, "available": False, "answer": None, "unavailable_reason": "provenance_unavailable"}
    return {**record, "available": answer is not None, "answer": answer}


@router.get("/brief")
async def brief(session_id: str):
    sid, _, store = _context(session_id)
    return {"brief": await _call(store.get_brief, sid)}


@router.post("/brief")
async def save_brief(body: BriefRequest):
    sid, _, store = _context(body.session_id)
    return {"brief": await _call(store.save_brief, sid, **body.model_dump(exclude={"session_id"}))}


@router.get("/artifacts")
async def artifacts(session_id: str):
    sid, manager, store = _context(session_id)
    _research()
    records = await _call(store.list_artifacts, sid)
    # A gallery needs only pointers, not up to 100 full model responses plus
    # citations. Provenance is checked at detail read/export time, not cached.
    history = await _call(manager.get_history, sid) if records else []
    present = {item.get("id") for item in history if item.get("role") == "assistant"}
    return {"items": [{**record, "available": record["answer_id"] in present} for record in records],
            "limit": MAX_SESSION_ARTIFACTS}


@router.get("/artifact")
async def artifact(session_id: str, answer_id: str = Query(pattern="^" + ANSWER_ID + "$")):
    sid, manager, store = _context(session_id)
    service = _research()
    records = await _call(store.list_artifacts, sid)
    record = next((row for row in records if row["answer_id"] == answer_id), None)
    if record is None:
        raise HTTPException(404, "Artifact is unavailable")
    answers = await _answers(sid, manager, service, {answer_id})
    return {"artifact": _artifact(record, answers)}


@router.post("/artifacts")
async def save_artifact(body: ArtifactRequest):
    sid, manager, store = _context(body.session_id)
    answers = await _answers(sid, manager, _research(), {body.answer_id})
    if body.answer_id not in answers:
        raise HTTPException(404, "Assistant answer is unavailable")
    if not _artifact({"answer_id": body.answer_id}, answers)["available"]:
        raise HTTPException(503, "Artifact provenance is unavailable")
    record = await _call(store.save_artifact, sid, body.answer_id, body.title)
    return {"artifact": _artifact(record, answers)}


@router.post("/artifacts/delete")
async def delete_artifact(body: ArtifactReference):
    sid, _, store = _context(body.session_id)
    # Removing a pointer is safe even when research/model services are down.
    await _call(store.delete_artifact, sid, body.answer_id)
    return {"removed": True}
