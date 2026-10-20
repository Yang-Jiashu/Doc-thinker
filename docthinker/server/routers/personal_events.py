"""Explicit domain-event editing on the existing private mobile facade."""
from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..mobile_access import MobileRoute, require_mobile_access
from ..personal_events import PersonalEvents

router = APIRouter(prefix='/research', dependencies=[Depends(require_mobile_access)], route_class=MobileRoute)
UUID4 = r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
Identifier = Annotated[str, Field(pattern=UUID4)]
ShortText = Annotated[str, Field(min_length=1, max_length=256)]


class StrictBody(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class SessionBody(StrictBody):
    session_id: str = Field(pattern=r'^#[0-9]{5}$')


class ConfirmedBody(SessionBody):
    confirmed: Literal[True]

    @field_validator('confirmed', mode='before')
    @classmethod
    def explicit_confirmation(cls, value):
        if value is not True:
            raise ValueError('Explicit user confirmation required')
        return value


class MemoryOrigin(StrictBody):
    kind: Literal['memory']
    memory_id: Identifier
    revision: int = Field(ge=1, le=2147483647)


class SourceOrigin(StrictBody):
    kind: Literal['source']
    source_id: Identifier
    span: list[Annotated[int, Field(ge=0, le=8000)]] = Field(min_length=2, max_length=2)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


Origin = Annotated[MemoryOrigin | SourceOrigin, Field(discriminator='kind')]


class EntityMutation(ConfirmedBody):
    kind: str = Field(pattern=r'^[a-z][a-z0-9_]{0,63}$')
    name: str = Field(min_length=1, max_length=160)
    aliases: list[ShortText] = Field(default_factory=list, max_length=16)
    identity_key: ShortText | None = None


class EventMutation(ConfirmedBody):
    event_type: Literal['decision', 'observation', 'task', 'meeting', 'milestone', 'change', 'incident', 'preference']
    action: str = Field(min_length=1, max_length=1000)
    actor_id: Identifier | None = None
    object_id: Identifier | None = None
    occurred_from: float | None = Field(default=None, ge=-62135596800, le=253402300799, allow_inf_nan=False)
    occurred_until: float | None = Field(default=None, ge=-62135596800, le=253402300799, allow_inf_nan=False)
    conditions: list[ShortText] = Field(default_factory=list, max_length=16)
    origins: list[Origin] = Field(min_length=1, max_length=16)


class EventRevision(EventMutation):
    event_id: Identifier
    expected_revision: int = Field(ge=1, le=2147483647)


class RelationMutation(ConfirmedBody):
    source_id: Identifier
    target_id: Identifier
    relation: Literal['before', 'supersedes', 'depends_on', 'contradicts', 'related_to', 'reported_cause', 'hypothesis']
    conditions: list[ShortText] = Field(default_factory=list, max_length=16)
    origins: list[Origin] = Field(default_factory=list, max_length=32)


class EventSnapshots(SessionBody):
    event_ids: list[Identifier] = Field(max_length=100)


def _context(session_id):
    # Resolve the actual worker service per request, never cache another owner.
    from .mobile import _research, _session
    return PersonalEvents(_research().store), _session(session_id)


async def _call(operation, *args, **kwargs):
    from .mobile import _research_call
    return await _research_call(operation, *args, **kwargs)


@router.post('/entities')
async def register_entity(body: EntityMutation):
    events, sid = _context(body.session_id)
    row = await _call(events.register_entity, sid, **body.model_dump(exclude={'session_id', 'confirmed'}))
    return {'entity': row}


@router.get('/entities/resolve')
async def resolve_entity(session_id: str, kind: str = Query(pattern=r'^[a-z][a-z0-9_]{0,63}$'), name: str = Query(min_length=1, max_length=160)):
    events, sid = _context(session_id)
    return await _call(events.resolve_entity, sid, kind, name)


@router.post('/events')
async def record_event(body: EventMutation):
    events, sid = _context(body.session_id)
    return {'event': await _call(events.record_event, sid, **body.model_dump(exclude={'session_id', 'confirmed'}))}


@router.post('/events/revise')
async def revise_event(body: EventRevision):
    events, sid = _context(body.session_id)
    return {'event': await _call(events.revise_event, sid, body.event_id, expected_revision=body.expected_revision,
        **body.model_dump(exclude={'session_id', 'confirmed', 'event_id', 'expected_revision'}))}


@router.get('/events')
async def list_events(session_id: str, entity_id: str | None = Query(default=None, pattern=UUID4), limit: int = Query(default=50, ge=1, le=100)):
    events, sid = _context(session_id)
    return {'items': await _call(events.list_events, sid, entity_id=entity_id, limit=limit)}


@router.get('/event')
async def get_event(session_id: str, event_id: str = Query(pattern=UUID4)):
    events, sid = _context(session_id)
    return {'event': await _call(events.get_event, sid, event_id)}


@router.get('/event/history')
async def event_history(session_id: str, event_id: str = Query(pattern=UUID4), limit: int = Query(default=20, ge=1, le=100)):
    events, sid = _context(session_id)
    return {'items': await _call(events.history, sid, event_id, limit=limit)}


@router.post('/events/snapshots')
async def event_snapshots(body: EventSnapshots):
    events, sid = _context(body.session_id)
    return {'events': await _call(events.snapshots, sid, body.event_ids)}


@router.post('/event-relations')
async def relate_events(body: RelationMutation):
    events, sid = _context(body.session_id)
    return {'relation': await _call(events.relate, sid, body.source_id, body.target_id, body.relation,
        conditions=body.conditions, origins=[origin.model_dump() for origin in body.origins])}


@router.get('/event-relations')
async def list_event_relations(session_id: str, event_id: str | None = Query(default=None, pattern=UUID4), limit: int = Query(default=100, ge=1, le=100)):
    events, sid = _context(session_id)
    return {'items': await _call(events.list_relations, sid, event_id=event_id, limit=limit)}
