"""Two explicit supported adapter layouts; never probe another owned task DB."""
from pathlib import Path
import uuid

import pytest

from docthinker.server.project_task_store import ProjectTaskStore
from docthinker.server.research_store import ResearchNotFoundError, ResearchStoreError
from tests.test_research_answer_dependencies_unit import opened, personal
from tests.test_research_service_unit import SessionManager, FakeModel


async def parent_fixture(root, layout):
    directory = root / 'data' / '_system' if layout == 'worker_data_system' else root / '_system'
    manager = SessionManager(str(directory), str(root / 'data'), migrate_legacy=False)
    sid = manager.create_session('Synthetic worker layout handover')['id']
    service = opened(root)
    instance = manager.get_session(sid)['instance_id']
    tasks = ProjectTaskStore(Path(manager.base_storage_path), session_id=sid, instance_id=instance, boot_id='layout-test')
    questions = [{'id': 'draft', 'question': 'Prepare a private handover draft.', 'input_kind': 'personal', 'memory_ids': []}]
    task = tasks.create_task(goal='Private handover', questions=questions)
    task = tasks.begin_run(task['id'], expected_revision=task['revision'], max_calls=1, source_snapshots={})
    step = tasks.claim_next(task['id'], task['run_id'], context_snapshots={})
    task = tasks.complete_step(task['id'], step['id'], step['claim_token'], {'documents': []}, context_snapshots={})
    step = tasks.claim_next(task['id'], task['run_id'], context_snapshots={})
    answer = await personal(service, manager, sid, FakeModel('LOCAL_LAYOUT_PARENT: private draft only.'))
    task = tasks.complete_step(task['id'], step['id'], step['claim_token'],
        {'answer_id': answer['answer_id'], 'document_ids': [], 'source_ids': [], 'input_kind': 'personal', 'context_refs': {}},
        context_snapshots={})
    step = tasks.claim_next(task['id'], task['run_id'], context_snapshots={})
    task = tasks.complete_step(task['id'], step['id'], step['claim_token'], {'step_refs': [{'step_id': 'draft', 'version': 1}]}, context_snapshots={})
    ref = {'instance_id': instance, 'task_id': task['id'], 'step_id': 'draft', 'version': 1, 'answer_id': answer['answer_id']}
    return service, manager, sid, tasks, task, answer, ref, questions


@pytest.mark.asyncio
@pytest.mark.parametrize('layout', ['research_root_system', 'worker_data_system'])
async def test_exact_layout_keeps_current_origin_and_rejects_parent_revision(tmp_path, layout):
    service, manager, sid, tasks, task, first, ref, questions = await parent_fixture(tmp_path, layout)
    if layout == 'worker_data_system':
        ref['storage_layout'] = layout
    try:
        model = FakeModel('LOCAL_LAYOUT_CHILD: derived private draft.')
        child = await personal(service, manager, sid, model, parents=[first['answer_id']], origins=[ref])
        assert child['answer'] == model.answer and len(model.prompts) == 1
        assert service.answer_dependencies.states(sid, [child['answer_id']]) == {child['answer_id']: True}
        tasks.revise_task(task['id'], expected_revision=task['revision'], goal=task['goal'], constraints=task['constraints'],
                          questions=[{**questions[0], 'question': 'Use the corrected private handover requirements.'}])
        assert service.answer_dependencies.states(sid, [child['answer_id']]) == {child['answer_id']: False}
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize('marker', [None, 'research_root_system', '../data/_system', 'auto', '', False, 0, [], {}])
async def test_worker_layout_is_never_discovered_or_retagged_from_invalid_marker(tmp_path, marker):
    service, manager, sid, _, _, first, ref, _ = await parent_fixture(tmp_path, 'worker_data_system')
    if marker is not None:
        ref['storage_layout'] = marker
    try:
        model = FakeModel('Must not be called for another task layout.')
        with pytest.raises(ResearchNotFoundError):
            await personal(service, manager, sid, model, parents=[first['answer_id']], origins=[ref])
        assert model.prompts == []
        if marker not in (None, 'research_root_system'):
            with pytest.raises(ResearchStoreError):
                service.answer_dependencies.record(sid, str(uuid.uuid4()), [first['answer_id']], task_refs=[ref])
    finally:
        await service.stop()
