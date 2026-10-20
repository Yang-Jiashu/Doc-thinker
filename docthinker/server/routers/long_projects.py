"""Versioned personal projects; every paid phase uses the existing task lane."""
from __future__ import annotations

import json
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..mobile_access import MobileRoute, require_mobile_access
from ..project_task_bridge import step_question
from ..invite_budget import validate_invite_question
from .project_tasks import _context, _current, _questions, _safe, CreateTask, QuestionStep, UUID4

router = APIRouter(prefix='/workbench', dependencies=[Depends(require_mobile_access)], route_class=MobileRoute)
Identifier = Annotated[str, Field(pattern=UUID4)]
Criterion = Annotated[str, Field(min_length=1, max_length=1000)]


class Body(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    session_id: str = Field(pattern=r'^#[0-9]{5}$')


class NewProject(Body):
    request_id: Identifier
    goal: str = Field(min_length=1, max_length=1200)
    acceptance_criteria: list[Criterion] = Field(min_length=1, max_length=12)


class ParentReference(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    phase_id: Identifier
    report_version: int = Field(ge=1, le=32)
    step_ids: list[Annotated[str, Field(pattern=r'^[a-z][a-z0-9_-]{0,31}$')]] | None = Field(default=None, min_length=1, max_length=6)


class PhaseBody(Body):
    project_id: Identifier
    expected_revision: int = Field(ge=1, lt=2147483647)
    request_id: Identifier
    title: str = Field(min_length=1, max_length=120)
    acceptance_criteria: list[Criterion] = Field(min_length=1, max_length=12)
    questions: list[QuestionStep] = Field(min_length=1, max_length=6)
    constraints: str = Field(default='', max_length=1200)
    parent_refs: list[ParentReference] = Field(default_factory=list, max_length=6)


class PhaseReview(Body):
    project_id: Identifier
    phase_id: Identifier
    expected_revision: int = Field(ge=1, lt=2147483647)
    request_id: Identifier
    report_version: int = Field(ge=1, le=32)
    decision: Literal['accepted', 'rejected']


class ReportReference(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    phase_id: Identifier
    report_version: int = Field(ge=1, le=32)


class ProjectReview(Body):
    project_id: Identifier
    expected_revision: int = Field(ge=1, lt=2147483647)
    request_id: Identifier
    phase_reports: list[ReportReference] = Field(min_length=1, max_length=100)
    decision: Literal['accepted', 'rejected']


def _delivery(phase, task):
    expected = [{'phase_id': ref['phase_id'], 'task_id': ref['task_id'], 'report_version': ref['report_version'],
                 'step_id': key, 'version': version}
                for ref in phase['parent_refs'] for key, version in ref['step_versions'].items()]
    analysis = [row for row in task['steps'] if row['kind'] == 'analysis']
    delivered = [row for row in analysis if row['status'] == 'completed' and row.get('phase_handoff', {}).get('used') == len(expected)]
    status = ('no_parent_dependencies' if not expected else 'delivered' if len(delivered) == len(analysis)
              else 'partially_delivered' if delivered else 'references_only_not_delivered')
    return {**phase, 'handoff_status': status,
            'handoff_receipts': [{'step_id': row['id'], **row['phase_handoff']} for row in delivered]}


async def _fresh(runtime, project_id, phase_id):
    store = runtime.long_projects.projects
    phase = await _safe(store.get_phase, project_id, phase_id)
    task = await _current(runtime, phase['task_id'])
    phase = await _safe(store.get_phase, project_id, phase_id)
    return _delivery(phase, task), task


async def _project_view(runtime, project_id):
    store = runtime.long_projects.projects
    value = await _safe(store.get_project, project_id)
    # A previously accepted root is not currently accepted merely because its
    # task-store snapshot has not yet noticed a research-store withdrawal.
    for ref in (value.get('review') or {}).get('binding', {}).get('phases', []):
        await _fresh(runtime, project_id, ref['phase_id'])
    return await _safe(store.get_project, project_id)


@router.post('/long-projects')
async def create_project(body: NewProject):
    _, runtime, _ = await _safe(_context, body.session_id)
    return {'project': await _safe(runtime.long_projects.projects.create_project,
        **body.model_dump(exclude={'session_id'}))}


@router.get('/long-projects')
async def projects(session_id: str, cursor: str | None = Query(default=None, max_length=2048),
                   limit: int = Query(default=20, ge=1, le=100)):
    _, runtime, _ = await _safe(_context, session_id)
    return await _safe(runtime.long_projects.projects.list_projects, limit=limit, cursor=cursor)


@router.get('/long-project')
async def project(session_id: str, project_id: str = Query(pattern=UUID4)):
    _, runtime, _ = await _safe(_context, session_id)
    return {'project': await _project_view(runtime, project_id)}


@router.post('/long-project/phases')
async def add_phase(body: PhaseBody):
    _, runtime, _ = await _safe(_context, body.session_id)
    store = runtime.long_projects.projects
    project = await _safe(store.get_project, body.project_id)
    rows = await _safe(_questions, CreateTask(session_id=body.session_id, goal=project['goal'],
                                            constraints=body.constraints, questions=body.questions))
    context = {'project_goal': project['goal'], 'project_acceptance_criteria': project['acceptance_criteria'],
               'phase_id': '0' * 36, 'phase_title': body.title, 'phase_acceptance_criteria': body.acceptance_criteria}
    for row in rows:
        prompt = step_question(project['goal'], body.constraints, row['question'], personal=row.get('input_kind') == 'personal')
        await _safe(validate_invite_question, prompt + '\n长期项目与阶段验收标准（用户要求，不是事实或已完成声明）：\n'
                    + json.dumps(context, ensure_ascii=False, separators=(',', ':')))
    for ref in body.parent_refs:
        await _fresh(runtime, body.project_id, ref.phase_id)
    phase = await _safe(store.add_phase, body.project_id, expected_revision=body.expected_revision,
        request_id=body.request_id, title=body.title, acceptance_criteria=body.acceptance_criteria,
        constraints=body.constraints, questions=rows,
        parent_refs=[row.model_dump(exclude_none=True) for row in body.parent_refs])
    phase, task = await _fresh(runtime, body.project_id, phase['id'])
    return {'project': await _safe(store.get_project, body.project_id), 'phase': phase, 'task': task}


@router.get('/long-project/phases')
async def phases(session_id: str, project_id: str = Query(pattern=UUID4),
                 cursor: str | None = Query(default=None, max_length=2048), limit: int = Query(default=20, ge=1, le=100)):
    _, runtime, _ = await _safe(_context, session_id)
    page = await _safe(runtime.long_projects.projects.list_phases, project_id, limit=limit, cursor=cursor)
    return {**page, 'items': [(await _fresh(runtime, project_id, row['id']))[0] for row in page['items']]}


@router.post('/long-project/phases/review')
async def review_phase(body: PhaseReview):
    _, runtime, bridge = await _safe(_context, body.session_id)
    store = runtime.long_projects.projects
    phase, task = await _fresh(runtime, body.project_id, body.phase_id)
    try:
        rendered = await runtime.render_report(task['id'],
            runtime.store.get_artifact(task['id'], 'report', version=body.report_version))
    except ValueError:
        raise HTTPException(409, 'Report inputs changed; reload before review') from None
    if not rendered['complete'] or not bridge.validate_inputs([row for row in task['steps'] if row['kind'] == 'analysis']):
        raise HTTPException(409, 'Report cannot be reviewed against changed inputs')
    reviewed = await _safe(store.review_phase, body.project_id, body.phase_id,
        **body.model_dump(exclude={'session_id', 'project_id', 'phase_id'}))
    phase, _ = await _fresh(runtime, body.project_id, body.phase_id)
    return {'project': await _safe(store.get_project, body.project_id), 'phase': phase, 'review': reviewed['review']}


@router.post('/long-project/review')
async def review_project(body: ProjectReview):
    _, runtime, _ = await _safe(_context, body.session_id)
    store = runtime.long_projects.projects
    for ref in body.phase_reports:
        await _fresh(runtime, body.project_id, ref.phase_id)
    reviewed = await _safe(store.review_project, body.project_id, expected_revision=body.expected_revision,
        request_id=body.request_id, decision=body.decision,
        phase_reports=[row.model_dump() for row in body.phase_reports])
    current = await _project_view(runtime, body.project_id)
    return {'project': current, 'review': current['review']}
