"""Real offline answer/history lineage across work and personal task phases."""
import json
from pathlib import Path
import uuid

import pytest

from docthinker.server.project_task_bridge import ProjectTaskBridge
from docthinker.server.project_task_store import ProjectTaskStore
from docthinker.server.research_answer_dependencies import AnswerDependencies, MAX_PARENTS
from docthinker.server.research_memory_dependencies import MemoryDependencies, source_answer_memory_live
from docthinker.server.research_service import ResearchService, WITHDRAWN_ANALYSIS
from docthinker.server.research_store import ResearchStoreError, ResearchNotFoundError
from docthinker.server.routers.workbench import _answers, _artifact
from tests.test_research_service_unit import FakeModel, request, text_extractor, workspace


def uid():
    return str(uuid.uuid4())


def opened(root):
    return ResearchService(root, extractor=text_extractor, context_assistance_enabled=False,
                           semantic_retriever=None, evidence_reranker=None, query_translator=None)


async def personal(service, manager, sid, model, *, parents=None, contexts=None, context='No external action; private draft only.', origins=None):
    return await service.answer(request(sid, 'Prepare a private work and family follow-up. Do not send or pay.'),
        model, manager, document_ids=[], project_context=context,
        project_instance_id=manager.get_session(sid)['instance_id'], task_context_refs=contexts or {},
        **({'project_parent_answer_ids': parents} if parents else {}),
        **({'project_parent_task_refs': origins} if origins is not None else {}))


async def work_source(service, manager, sid):
    doc = service.store.accept_document(sid, 'work-handover.txt',
        b'The work handover deadline is Thursday. Client documents must remain private. Do not send automatically.')
    await service.process(sid, doc['id'])
    model = FakeModel('The work handover deadline is Thursday; no automatic sending [S1].')
    result = await service.answer(request(sid, 'What is the work handover deadline and client privacy restriction?'), model, manager)
    assert len(model.prompts) == 1 and result['sources'] and result['answer'] == model.answer
    return doc, result


@pytest.mark.asyncio
@pytest.mark.parametrize('length', [3, 30])
@pytest.mark.parametrize('source_kind', ['work_document', 'family_memory'])
async def test_three_or_thirty_actual_steps_remain_revocable_through_history_workbench_and_notes(
        workspace, monkeypatch, length, source_kind):
    root, manager, sid, _ = workspace
    monkeypatch.setenv('DOCTHINKER_RESEARCH_SEMANTIC', 'off')
    service = opened(root)
    all_models, answers = [], []
    try:
        if source_kind == 'work_document':
            original, first = await work_source(service, manager, sid)
        else:
            original = service.store.create_memory(sid, 'decision', '家庭接送安排在周四，不能自动联系其他人。')
            root_model = FakeModel('家庭接送安排在周四，不能自动联系其他人。')
            first = await personal(service, manager, sid, root_model, contexts={original['id']: 1})
            assert original['text'] in root_model.prompts[0] and first['used_memory_count'] == 1
        answers.append(first['answer_id'])
        previous = first
        for index in range(1, length):
            model = FakeModel(f'PRIVATE_DERIVED_DRAFT_{index}: prepare a reviewable work/family follow-up; take no external action.')
            current = await personal(service, manager, sid, model, parents=[previous['answer_id']],
                                     context=json.dumps({'previous_analysis': previous['answer'], 'not_original_evidence': True}))
            assert len(model.prompts) == 1 and current['answer'] == model.answer
            assert current['answer_dependency_required'] is True
            all_models.append(model)
            answers.append(current['answer_id'])
            previous = current
        assert service.answer_dependencies.states(sid, answers) == {key: True for key in answers}
        saved = service.store.create_memory(sid, 'finding', 'A confirmed follow-up derived from the final draft.',
                                            source_answer_id=previous['answer_id'])
        bridge = ProjectTaskBridge(service, manager, sid, FakeModel())
        assert await bridge.load_answer(session_id=sid, answer_id=previous['answer_id'])
        if source_kind == 'work_document':
            service.store.delete_document(sid, original['id'])
        else:
            service.store.delete_memory(sid, original['id'], expected_revision=1)
        # Restart both the readers and the service: cached state is not the guard.
        await service.stop()
        service = opened(root)
        states = service.answer_dependencies.states(sid, answers)
        assert all(states[key] is False for key in answers[1:])
        hydrated = service.hydrate_history(sid, manager.get_history(sid))
        final = next(row for row in hydrated if row['id'] == previous['answer_id'])
        assert final['content'] == WITHDRAWN_ANALYSIS and final['sources'] == []
        result = await _answers(sid, manager, service, {previous['answer_id']})
        assert _artifact({'answer_id': previous['answer_id']}, result)['available'] is False
        assert await ProjectTaskBridge(service, manager, sid, FakeModel()).load_answer(
            session_id=sid, answer_id=previous['answer_id']) is None
        assert service.store.get_memory(sid, saved['id'])['redacted'] is True
        with pytest.raises(ResearchStoreError):
            service.store.create_memory(sid, 'finding', 'Cannot reconfirm withdrawn work.', source_answer_id=previous['answer_id'])
        next_model = FakeModel('The previous basis is unavailable; please confirm current instructions.')
        await service.answer(request(sid, 'Continue the follow-up, without taking action.', use_conversation_context=True),
                             next_model, manager)
        assert len(next_model.prompts) == 1 and f'PRIVATE_DERIVED_DRAFT_{length - 1}' not in next_model.prompts[0]
        assert sum(len(model.prompts) for model in all_models) == length - 1
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_memory_correction_invalidates_descendants_before_another_dispatch(workspace):
    root, manager, sid, _ = workspace
    service = opened(root)
    try:
        note = service.store.create_memory(sid, 'decision', 'Client deadline is Thursday; do not send.')
        first = await personal(service, manager, sid, FakeModel('Client deadline is Thursday; do not send.'), contexts={note['id']: 1})
        child = await personal(service, manager, sid, FakeModel('Prepare the private Thursday follow-up.'), parents=[first['answer_id']])
        service.store.update_memory(sid, note['id'], expected_revision=1, text='Corrected: client deadline is Wednesday; do not send.')
        model = FakeModel('Must not be dispatched.')
        with pytest.raises(ResearchNotFoundError):
            await personal(service, manager, sid, model, parents=[child['answer_id']])
        assert model.prompts == []
        assert service.answer_dependencies.states(sid, [child['answer_id']]) == {child['answer_id']: False}
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_history_only_reuse_keeps_parent_binding_even_without_current_sources(workspace):
    root, manager, sid, _ = workspace
    service = opened(root)
    try:
        doc, first = await work_source(service, manager, sid)
        child = await personal(service, manager, sid, FakeModel('PRIVATE_HANDOFF_ONLY: prepare the approved draft.'), parents=[first['answer_id']])
        model = FakeModel('PRIVATE_HISTORY_ONLY: use the earlier draft; do not send.')
        history_child = await service.answer(request(sid, 'Continue that draft, do not send.', use_conversation_context=True), model, manager)
        assert len(model.prompts) == 1 and 'PRIVATE_HANDOFF_ONLY' in model.prompts[0]
        assert history_child['answer_dependency_required'] is True
        with service.store._read() as db:
            raw = db.execute('SELECT parents FROM research_answer_parents WHERE answer_id=?', (history_child['answer_id'],)).fetchone()[0]
        assert child['answer_id'] in json.loads(raw)
        service.store.delete_document(sid, doc['id'])
        assert service.answer_dependencies.states(sid, [history_child['answer_id']]) == {history_child['answer_id']: False}
        final = next(row for row in service.hydrate_history(sid, manager.get_history(sid)) if row['id'] == history_child['answer_id'])
        assert final['content'] == WITHDRAWN_ANALYSIS
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize('failed_write', ['parent_binding', 'proof'])
async def test_partial_dependency_record_is_fail_closed_without_retry_or_reconfirmation(workspace, monkeypatch, failed_write):
    root, manager, sid, _ = workspace
    service = opened(root)
    try:
        _, first = await work_source(service, manager, sid)
        def broken_record(*args, **kwargs):
            raise OSError('Synthetic local storage outage')
        monkeypatch.setattr(service.answer_dependencies if failed_write == 'parent_binding' else service.evidence,
                            'record', broken_record)
        model = FakeModel('PRIVATE_UNCOMMITTED_DERIVATIVE: draft only.')
        result = await personal(service, manager, sid, model, parents=[first['answer_id']])
        assert len(model.prompts) == 1 and result['answer'] == WITHDRAWN_ANALYSIS
        metadata = service.store.answer_metadata(sid, [result['answer_id']])[result['answer_id']]
        assert metadata['answer_dependency_required'] is True
        assert service.answer_dependencies.states(sid, [result['answer_id']]) == {result['answer_id']: False}
        assert not _artifact({'answer_id': result['answer_id']}, await _answers(sid, manager, service, {result['answer_id']}))['available']
        with pytest.raises(ResearchStoreError):
            service.store.create_memory(sid, 'finding', 'Cannot confirm an uncommitted derivative.', source_answer_id=result['answer_id'])
        assert len(model.prompts) == 1
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_foreign_session_or_missing_parent_never_dispatches(workspace):
    root, manager, sid, other = workspace
    service = opened(root)
    try:
        _, first = await work_source(service, manager, sid)
        for target, parent in [(other, first['answer_id']), (sid, uid())]:
            model = FakeModel('Forbidden dispatch.')
            with pytest.raises(ResearchNotFoundError):
                await personal(service, manager, target, model, parents=[parent])
            assert model.prompts == []
        assert service.answer_dependencies.states(other, [first['answer_id']]) == {first['answer_id']: False}
    finally:
        await service.stop()


def bare_answer(service, sid, *, required=False):
    answer = uid()
    service.store.record_answer(sid, answer, [], {}, 'general', context_metadata={'answer_dependency_required': required})
    return answer


@pytest.mark.parametrize('raw', ['null', '{}', '[]', 'not-json', '[{}]', '[["bad"]]', '[true]', '["x","x"]'])
def test_corrupt_parent_rows_fail_closed_without_recursion_or_source_reconnection(tmp_path, raw):
    service = opened(tmp_path)
    parent, child = bare_answer(service, '#00001'), bare_answer(service, '#00001', required=True)
    service.answer_dependencies.record('#00001', child, [parent])
    with service.store._write() as db:
        db.execute('UPDATE research_answer_parents SET parents=? WHERE answer_id=?', (raw, child))
    assert service.answer_dependencies.states('#00001', [child]) == {child: False}


def test_cycle_and_diamond_checks_are_bounded_and_do_not_call_a_model(tmp_path):
    service = opened(tmp_path)
    sid = '#00001'
    a, b, c, d = [bare_answer(service, sid, required=bool(index)) for index in range(4)]
    # The proof records for reused parents are present, with no invented sources.
    with service.store._write() as db:
        service.evidence._schema(db)
    for identifier in [a, b, c, d]:
        service.evidence.record(sid, identifier, reasoning={'mode': 'context_assist', 'status': 'ready', 'trace': {}},
                                audit={'claims': []}, memory_as_of=None, reasoning_mode='auto')
    service.answer_dependencies.record(sid, b, [a])
    service.answer_dependencies.record(sid, c, [a])
    service.answer_dependencies.record(sid, d, [b, c])
    assert service.answer_dependencies.states(sid, [d]) == {d: True}
    service.answer_dependencies.record(sid, a, [d])
    assert service.answer_dependencies.states(sid, [d]) == {d: False}


@pytest.mark.parametrize('raw', ['[]', 'null', '0', '"metadata"'])
def test_unknown_metadata_shape_is_unavailable_not_an_exception(tmp_path, raw):
    service = opened(tmp_path)
    note = service.store.create_memory('#00001', 'decision', 'Keep the private work follow-up confidential.')
    child = bare_answer(service, '#00001', required=True)
    # Reach the answer-parent hook through a VALID memory lineage. With no
    # memory ledger, the older guard rejects corrupt metadata earlier and
    # would not exercise this independent new boundary.
    MemoryDependencies(service.store).record('#00001', child, [(note['id'], 1, None)])
    with service.store._write() as db:
        db.execute('UPDATE research_answers SET context_metadata_json=? WHERE answer_id=?', (raw, child))
    assert service.answer_dependencies.states('#00001', [child]) == {child: False}


def test_parent_decode_preserves_limits_and_exact_reference_identity():
    values = [uid() for _ in range(MAX_PARENTS)]
    assert AnswerDependencies.decode(json.dumps(values)) == values
    for invalid in [values + [uid()], values + values[:1], [], {'parents': values}]:
        with pytest.raises((ResearchStoreError, ValueError, TypeError)):
            AnswerDependencies.decode(json.dumps(invalid))


@pytest.mark.asyncio
async def test_missing_parent_proof_cannot_be_reintroduced_via_confirmed_memory(workspace):
    root, manager, sid, _ = workspace
    service = opened(root)
    try:
        _, first = await work_source(service, manager, sid)
        child = await personal(service, manager, sid, FakeModel('Draft based on the reviewed work note.'), parents=[first['answer_id']])
        with service.store._write() as db:
            db.execute('DELETE FROM research_answer_evidence WHERE answer_id=?', (first['answer_id'],))
        assert service.answer_dependencies.states(sid, [child['answer_id']]) == {child['answer_id']: False}
        with pytest.raises(ResearchStoreError):
            service.store.create_memory(sid, 'finding', 'Do not bypass lost parent proof.', source_answer_id=child['answer_id'])
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['revise', 'retire'])
@pytest.mark.parametrize('store_location', ['manager', 'other_owned_directory'])
async def test_explicit_task_origin_stales_history_and_confirmed_note_after_parent_plan_change(workspace, change, store_location):
    root, manager, sid, _ = workspace
    service = opened(root)
    instance = manager.get_session(sid)['instance_id']
    directory = Path(manager.base_storage_path).resolve()
    if store_location != 'manager':
        directory = root / 'other-owned-directory'
        directory.mkdir()
    tasks = ProjectTaskStore(directory.resolve(), session_id=sid, instance_id=instance, boot_id='lineage-test')
    rows = [{'id': 'work', 'question': 'Prepare a private work draft.', 'input_kind': 'personal', 'memory_ids': []}]
    task = tasks.create_task(goal='Coordinate a private work handover.', questions=rows)
    task = tasks.begin_run(task['id'], expected_revision=task['revision'], max_calls=1, source_snapshots={})
    try:
        inventory = tasks.claim_next(task['id'], task['run_id'], context_snapshots={})
        task = tasks.complete_step(task['id'], inventory['id'], inventory['claim_token'], {'documents': []}, context_snapshots={})
        claim = tasks.claim_next(task['id'], task['run_id'], context_snapshots={})
        first = await personal(service, manager, sid, FakeModel('PRIVATE_TASK_PARENT: a draft, not an executed action.'))
        task = tasks.complete_step(task['id'], claim['id'], claim['claim_token'],
            {'answer_id': first['answer_id'], 'source_ids': [], 'document_ids': [], 'input_kind': 'personal', 'context_refs': {}},
            context_snapshots={})
        report = tasks.claim_next(task['id'], task['run_id'], context_snapshots={})
        task = tasks.complete_step(task['id'], report['id'], report['claim_token'],
                                   {'step_refs': [{'step_id': 'work', 'version': 1}]}, context_snapshots={})
        origin = {'instance_id': instance, 'task_id': task['id'], 'step_id': 'work', 'version': 1, 'answer_id': first['answer_id']}
        if store_location != 'manager':
            # A matching answer in a different owned database does not grant
            # task lineage. The reader must not search/fallback across roots.
            model = FakeModel('Must never be dispatched for a foreign task database.')
            with pytest.raises(ResearchNotFoundError):
                await personal(service, manager, sid, model, parents=[first['answer_id']], origins=[origin])
            assert model.prompts == []
            return
        child = await personal(service, manager, sid, FakeModel('PRIVATE_TASK_CHILD: a derived draft.'),
                               parents=[first['answer_id']], origins=[origin])
        assert child['answer'] == 'PRIVATE_TASK_CHILD: a derived draft.'
        assert service.answer_dependencies.states(sid, [child['answer_id']]) == {child['answer_id']: True}
        saved = service.store.create_memory(sid, 'finding', 'A derived private handover draft.', source_answer_id=child['answer_id'])
        if change == 'revise':
            revised = [{**rows[0], 'question': 'Prepare the revised private work draft.'}]
            tasks.revise_task(task['id'], expected_revision=task['revision'], goal=task['goal'], constraints=task['constraints'], questions=revised)
        else:
            revised = [{**rows[0], 'id': 'replacement'}]
            proposal = tasks.preview_replan(task['id'], expected_revision=task['revision'], goal=task['goal'],
                                            constraints=task['constraints'], questions=revised)
            tasks.replan_task(task['id'], expected_revision=task['revision'], goal=task['goal'], constraints=task['constraints'],
                              questions=revised, proposal_sha256=proposal['proposal_sha256'])
        assert service.answer_dependencies.states(sid, [child['answer_id']]) == {child['answer_id']: False}
        assert service.store.get_memory(sid, saved['id'])['redacted'] is True
        assert not _artifact({'answer_id': child['answer_id']}, await _answers(sid, manager, service, {child['answer_id']}))['available']
        with pytest.raises(ResearchStoreError):
            service.store.create_memory(sid, 'finding', 'Cannot reconnect an old task version.', source_answer_id=child['answer_id'])
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_one_hundred_sixty_actual_service_steps_preserve_oldest_document_revocation(workspace):
    root, manager, sid, _ = workspace
    service = opened(root)
    try:
        original, previous = await work_source(service, manager, sid)
        identifiers = [previous['answer_id']]
        calls = []
        for index in range(1, 160):
            model = FakeModel(f'PRIVATE_LONG_DRAFT_{index}: a private work/family follow-up, not an executed action.')
            previous = await personal(service, manager, sid, model, parents=[previous['answer_id']],
                context=json.dumps({'previous_analysis': previous['answer'], 'not_original_evidence': True}))
            assert previous['answer'] == model.answer and len(model.prompts) == 1
            assert previous['reasoning']['trace']['context_budget']['input_bytes'] <= 24576
            identifiers.append(previous['answer_id'])
            calls.append(model)
        assert service.answer_dependencies.states(sid, [identifiers[-1]]) == {identifiers[-1]: True}
        assert len(service.store.list_memories(sid)) == 0
        saved = service.store.create_memory(sid, 'finding', 'A private follow-up based on the final long task draft.',
                                            source_answer_id=identifiers[-1])
        service.store.delete_document(sid, original['id'])
        await service.stop()
        service = opened(root)
        assert service.answer_dependencies.states(sid, [identifiers[-1]]) == {identifiers[-1]: False}
        selected = await _answers(sid, manager, service, {identifiers[-1]})
        assert _artifact({'answer_id': identifiers[-1]}, selected)['available'] is False
        assert selected[identifiers[-1]]['content'] == WITHDRAWN_ANALYSIS
        assert service.store.get_memory(sid, saved['id'])['redacted'] is True
        with pytest.raises(ResearchStoreError):
            service.store.create_memory(sid, 'finding', 'Cannot reconfirm the withdrawn long-task result.',
                                         source_answer_id=identifiers[-1])
        assert sum(len(model.prompts) for model in calls) == 159
    finally:
        await service.stop()
