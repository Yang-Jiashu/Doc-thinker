"""Native local-DAG task origins and strict lineage metadata; no provider I/O."""
import json
import uuid

import pytest

from docthinker.server.research_store import ResearchStoreError
from docthinker.server.routers.workbench import _answers, _artifact
from docthinker.server.routers import project_tasks
from tests.test_research_answer_dependencies_unit import opened, personal, work_source
from tests.test_research_service_unit import FakeModel, request, workspace
from tests.test_personal_tasks_mobile_unit import (
    mobile, research_mobile, mobile_client, plan, create, start, settled, SID,
)
from tests.test_research_mobile_unit import PREFIX


@pytest.mark.asyncio
async def test_refs_only_service_call_is_rejected_before_model_or_history(workspace):
    root, manager, sid, _ = workspace
    service, model = opened(root), FakeModel('This must never be generated.')
    ref = {'instance_id': manager.get_session(sid)['instance_id'], 'task_id': str(uuid.uuid4()),
           'step_id': 'draft', 'version': 1, 'answer_id': str(uuid.uuid4())}
    before = manager.get_history(sid)
    try:
        with pytest.raises(ValueError, match='Task origins require parent analysis references'):
            await service.answer(request(sid, 'Prepare a private follow-up.'), model, manager,
                                 project_parent_task_refs=[ref])
        assert model.prompts == [] and manager.get_history(sid) == before
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize('corrupt', ['{}', 'null', 'false', '0'])
async def test_falsey_nonlist_task_origin_rows_are_not_treated_as_legacy_empty_scope(workspace, corrupt):
    root, manager, sid, _ = workspace
    service = opened(root)
    try:
        _, first = await work_source(service, manager, sid)
        model = FakeModel('PRIVATE_DERIVED_DRAFT: not an executed action.')
        child = await personal(service, manager, sid, model, parents=[first['answer_id']])
        assert child['answer'] == model.answer
        assert service.answer_dependencies.states(sid, [child['answer_id']]) == {child['answer_id']: True}
        with service.store._write() as db:
            db.execute('UPDATE research_answer_parents SET task_refs=? WHERE answer_id=?', (corrupt, child['answer_id']))
        assert service.answer_dependencies.states(sid, [child['answer_id']]) == {child['answer_id']: False}
        assert not _artifact({'answer_id': child['answer_id']},
                             await _answers(sid, manager, service, {child['answer_id']}))['available']
        with pytest.raises(ResearchStoreError):
            service.store.create_memory(sid, 'finding', 'Do not reconfirm corrupt task-origin metadata.',
                                         source_answer_id=child['answer_id'])
        assert len(model.prompts) == 1
    finally:
        await service.stop()


async def test_native_standalone_local_handoff_binds_task_revision_and_cannot_reconfirm_stale_child(mobile):
    try:
        async with mobile_client(mobile) as browser:
            task = await create(browser, plan(chain=True, domain='私人的客户交接计划'))
            await start(browser, task, calls=2)
            task = await settled(browser, task['id'])
            assert task['status'] == 'completed' and mobile.model.await_count == 2
            _, runtime, _ = await project_tasks._context(SID)
            stored = runtime.store.get_artifact(task['id'], 'followup')
            child = stored['references']['answer_id']
            parent = runtime.store.get_artifact(task['id'], 'draft')['references']['answer_id']
            assert stored['references']['handoff']['required'] == 1
            assert 'phase_handoff' not in stored['references']
            with mobile.service.store._read() as db:
                origins = json.loads(db.execute('SELECT task_refs FROM research_answer_parents WHERE answer_id=?', (child,)).fetchone()[0])
            assert origins == [{'instance_id': 'instance-00001', 'task_id': task['id'], 'step_id': 'draft',
                                'version': 1, 'answer_id': parent}]
            saved = mobile.service.store.create_memory(SID, 'finding', 'A private follow-up from the earlier plan.', source_answer_id=child)
            revised = plan(chain=True, domain='私人的客户交接计划')['questions']
            revised[0]['question'] = '按修正后的客户要求起草，不再使用之前的交接安排；不得发送。'
            response = await browser.post(PREFIX + '/workbench/tasks/update', json={
                'session_id': SID, 'task_id': task['id'], 'expected_revision': task['revision'],
                'goal': task['goal'], 'constraints': task['constraints'], 'questions': revised})
            assert response.status_code == 200
            assert runtime.store.get_artifact(task['id'], 'followup')['available'] is False
            assert mobile.service.answer_dependencies.states(SID, [child]) == {child: False}
            assert not _artifact({'answer_id': child}, await _answers(SID, mobile.manager, mobile.service, {child}))['available']
            assert mobile.service.store.get_memory(SID, saved['id'])['redacted'] is True
            with pytest.raises(ResearchStoreError):
                mobile.service.store.create_memory(SID, 'finding', 'Cannot revive the superseded plan.', source_answer_id=child)
            assert mobile.model.await_count == 2
    finally:
        await project_tasks.shutdown_project_tasks()
