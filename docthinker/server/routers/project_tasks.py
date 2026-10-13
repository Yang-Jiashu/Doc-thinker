"""Durable, explicitly approved research projects; no arbitrary tool execution."""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from ..mobile_access import MobileRoute, require_mobile_access
from ..project_task_bridge import ProjectTaskBridge, step_question
from ..project_task_runtime import ProjectTaskRuntime
from ..project_task_store import (ProjectTaskStore, ProjectTaskConflictError,
    ProjectTaskNotFoundError, ProjectTaskLimitError, ProjectTaskError)

router = APIRouter(prefix='/workbench', dependencies=[Depends(require_mobile_access)], route_class=MobileRoute)
UUID4 = r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
_controllers = {}
_jobs: dict[tuple, asyncio.Task] = {}
_admission = asyncio.Lock()


class QuestionStep(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    id: str = Field(pattern=r'^[a-z][a-z0-9_-]{0,31}$')
    title: str | None = Field(default=None, min_length=1, max_length=120)
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] | None = Field(default=None, min_length=1, max_length=20)
    depends_on: list[str] = Field(default_factory=list, max_length=6)


class CreateTask(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    session_id: str = Field(pattern=r'^#[0-9]{5}$')
    goal: str = Field(min_length=1, max_length=1200)
    constraints: str = Field(default='', max_length=1200)
    questions: list[QuestionStep] = Field(min_length=1, max_length=6)


class UpdateTask(CreateTask):
    task_id: str = Field(pattern=UUID4)
    expected_revision: int = Field(ge=1, lt=2147483647)


class RunTask(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    session_id: str = Field(pattern=r'^#[0-9]{5}$')
    task_id: str = Field(pattern=UUID4)
    expected_revision: int = Field(ge=1, lt=2147483647)
    request_id: str = Field(pattern=UUID4)
    max_calls: int = Field(ge=1, le=6)


class ControlTask(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    session_id: str = Field(pattern=r'^#[0-9]{5}$')
    task_id: str = Field(pattern=UUID4)
    expected_revision: int = Field(ge=1, lt=2147483647)
    action: Literal['pause', 'resume', 'cancel', 'refresh', 'reconcile', 'retry_failed']


async def _safe(operation, *args, **kwargs):
    try:
        value = operation(*args, **kwargs)
        return await value if asyncio.iscoroutine(value) else value
    except ProjectTaskNotFoundError:
        raise HTTPException(404, 'Project task is unavailable') from None
    except ProjectTaskConflictError:
        raise HTTPException(409, 'Task changed or requires review; reload its current state') from None
    except ProjectTaskLimitError:
        raise HTTPException(429, 'Project task capacity reached') from None
    except ProjectTaskError:
        raise HTTPException(422, 'Invalid project task') from None
    except ValueError:
        raise HTTPException(422, 'Invalid or oversized task input') from None
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(503, 'Project task service is unavailable') from None


async def _context(session_id):
    from .workbench import _context as workbench_context, _research
    from .mobile import _query_boot_id, state
    sid, manager, _ = workbench_context(session_id)
    service = _research()
    session = manager.get_session(sid)
    instance = session['instance_id']
    key = (str(Path(manager.base_storage_path)), sid, instance, id(service), id(manager))
    if key not in _controllers:
        if len(_controllers) >= 64:
            idle = next((old for old in _controllers if not any(job[:5] == old for job in _jobs)), None)
            if idle is None:
                raise HTTPException(429, 'Project task service is busy')
            _controllers.pop(idle)
        store = ProjectTaskStore(Path(manager.base_storage_path), session_id=sid,
                                 instance_id=instance, boot_id=_query_boot_id)
        if store.path.exists():
            await _safe(store.recover)
        model = getattr(state, 'mobile_chat_func', None)
        bridge = ProjectTaskBridge(service, manager, sid, model)
        bridge.instance_id = instance
        runtime = ProjectTaskRuntime(store, execute_question=bridge.execute_question,
            read_sources=bridge.read_sources, reconcile_request=bridge.reconcile_request,
            load_answer=bridge.load_answer)
        _controllers[key] = (runtime, bridge)
    return key, *_controllers[key]


async def _current(runtime, task_id):
    task = await _safe(runtime.store.get_task, task_id)
    current = await _safe(runtime.read_sources)
    changed = set()
    for step in task['steps']:
        if step['status'] not in {'completed', 'running', 'blocked'}:
            continue
        before = step.get('input_snapshot', {})
        scope = step.get('document_ids')
        after = current if scope is None else {key: value for key, value in current.items() if key in scope}
        changed.update(key for key in set(before) | set(after) if before.get(key) != after.get(key))
    if changed:
        await _safe(runtime.store.invalidate_sources, sorted(changed), reason='source_changed')
    task = await _safe(runtime.reconcile, task_id)
    task['artifact_versions'] = await _safe(runtime.store.artifact_history, task_id, 'report')
    return task


@router.get('/tasks')
async def tasks(session_id: str):
    _, runtime, _ = await _safe(_context, session_id)
    items = await _safe(runtime.store.list_tasks)
    return {'items': [await _current(runtime, item['id']) for item in items]}


@router.get('/task')
async def task(session_id: str, task_id: str = Query(pattern=UUID4)):
    _, runtime, _ = await _safe(_context, session_id)
    return {'task': await _current(runtime, task_id)}


def _questions(body):
    rows = [question.model_dump(exclude_none=True) for question in body.questions]
    for row in rows:
        # Reject an impossible input BEFORE approving a run, never silently
        # truncate the user's requirements to fit the ordinary chat budget.
        step_question(body.goal, body.constraints, row['question'])
        if any(not re.fullmatch(UUID4, value) for value in row.get('document_ids', [])):
            raise ValueError('Invalid document scope')
    return rows


@router.post('/tasks')
async def create_task(body: CreateTask):
    _, runtime, _ = await _safe(_context, body.session_id)
    try:
        rows = _questions(body)
        task = await _safe(runtime.store.create_task, goal=body.goal,
                           constraints=body.constraints, questions=rows)
    except ValueError:
        raise HTTPException(422, 'Task inputs exceed the step budget or contain invalid text') from None
    return {'task': task}


@router.post('/tasks/update')
async def update_task(body: UpdateTask):
    _, runtime, _ = await _safe(_context, body.session_id)
    rows = await _safe(_questions, body)
    task = await _safe(runtime.store.revise_task, body.task_id, expected_revision=body.expected_revision,
                       goal=body.goal, constraints=body.constraints, questions=rows)
    return {'task': task}


async def _drive(key, runtime, body):
    try:
        await runtime.drive_approved(body.task_id, run_request_id=body.request_id)
    except asyncio.CancelledError:
        raise
    except Exception:
        # The durable store/journal, not a browser's optimistic animation,
        # determines outcome. Never dispatch another call to repair a failure.
        pass
    finally:
        _jobs.pop(key, None)


@router.post('/tasks/run')
async def run_task(body: RunTask):
    key, runtime, bridge = await _safe(_context, body.session_id)
    if bridge.model is None:
        raise HTTPException(503, 'Model service is unavailable')
    job_key = (*key, body.task_id)
    async with _admission:
        if job_key in _jobs and not _jobs[job_key].done():
            # Even while a driver is alive, a UUID is an immutable approval,
            # not permission to change its revision or maximum paid attempts.
            snapshot = await _safe(bridge.read_sources)
            await _safe(runtime.store.begin_run, body.task_id, expected_revision=body.expected_revision,
                        request_id=body.request_id, max_calls=body.max_calls, source_snapshots=snapshot)
            return {'task': await _current(runtime, body.task_id)}
        if any(not job.done() for job in _jobs.values()):
            raise HTTPException(429, 'Another project is running; pause or wait before starting')
        snapshot = await _safe(bridge.read_sources)
        task = await _safe(runtime.store.begin_run, body.task_id, expected_revision=body.expected_revision,
                           request_id=body.request_id, max_calls=body.max_calls, source_snapshots=snapshot)
        if task['status'] != 'completed' and task.get('run_replayed') is not True:
            _jobs[job_key] = asyncio.create_task(_drive(job_key, runtime, body))
    return {'task': task}


@router.post('/tasks/control')
async def control_task(body: ControlTask):
    _, runtime, _ = await _safe(_context, body.session_id)
    if body.action == 'reconcile':
        task = await _safe(runtime.reconcile, body.task_id)
    elif body.action == 'refresh':
        task = await _safe(runtime.refresh, body.task_id, expected_revision=body.expected_revision)
    else:
        function = getattr(runtime.store, 'reset_failed_task' if body.action == 'retry_failed' else body.action + '_task')
        task = await _safe(function, body.task_id, expected_revision=body.expected_revision)
    return {'task': task}


@router.get('/task/artifact')
async def artifact(session_id: str, task_id: str = Query(pattern=UUID4),
                   version: int = Query(ge=1, le=32), step_id: str = Query(default='report', pattern=r'^[a-z][a-z0-9_-]{0,31}$')):
    _, runtime, bridge = await _safe(_context, session_id)
    await _current(runtime, task_id)
    stored = await _safe(runtime.store.get_artifact, task_id, step_id, version=version)
    result = {**stored, 'content': None, 'sources': []}
    if stored['available']:
        if step_id == 'report':
            rendered = await _safe(runtime.render_report, task_id, stored)
            if rendered.get('complete') is True:
                result['content'] = rendered['content']
            else:
                result.update(available=False, references=None, status='withdrawn')
        else:
            refs = stored['references']
            answer = await bridge.load_answer(session_id=session_id, answer_id=refs.get('answer_id'))
            if answer:
                result['content'] = answer['content']
            else:
                result.update(available=False, references=None, status='withdrawn')
    return {'artifact': result}


async def shutdown_project_tasks():
    jobs = tuple(_jobs.values())
    for job in jobs:
        job.cancel()
    if jobs:
        await asyncio.gather(*jobs, return_exceptions=True)
    _jobs.clear()
    _controllers.clear()
