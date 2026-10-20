"""No provider: personal tasks use explicit user instructions and memory refs."""
import asyncio
import uuid

import pytest

from docthinker.server.project_task_runtime import ProjectTaskRuntime, ProjectTaskRuntimeError
from docthinker.server.project_task_store import ProjectTaskStore, ProjectTaskConflictError


def uid():
    return str(uuid.uuid4())


class PersonalBridge:
    def __init__(self):
        self.memories, self.documents = {}, {}
        self.calls, self.context_reads, self.document_reads = [], [], []
        self.answers, self.receipts = {}, {}
        self.action = None

    async def read_sources(self, document_ids=None):
        self.document_reads.append(document_ids)
        return {key: digest for key, digest in self.documents.items() if document_ids is None or key in document_ids}

    async def read_context(self, memory_ids=None):
        self.context_reads.append(list(memory_ids or []))
        return {key: self.memories[key] for key in memory_ids or [] if key in self.memories}

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        identifier = uid()
        receipt = {'request_status': 'completed', 'answer_id': identifier,
                   'source_ids': [], 'document_ids': [], 'input_kind': 'personal',
                   'context_refs': dict(kwargs['context_snapshot'])}
        count = len(kwargs['dependency_artifacts'])
        receipt['handoff'] = {'required': count, 'used': count, 'full': count, 'excerpted': 0, 'omitted': 0}
        self.answers[identifier] = {'content': 'DRAFT ONLY: ' + kwargs['question'], 'sources': []}
        self.receipts[kwargs['request_id']] = receipt
        if self.action:
            response = await self.action(kwargs)
            if response is not None:
                return response
        return receipt

    async def reconcile(self, session_id, request_id):
        return self.receipts.get(request_id, {'request_status': 'unknown'})

    async def load_answer(self, session_id, answer_id):
        return self.answers.get(answer_id)


def setup(path, *, memories=(), questions=None, domain='Family appointment follow-up', boot='first'):
    store = ProjectTaskStore(path.resolve(), session_id='#00001', instance_id='private-agent', boot_id=boot)
    rows = questions or [{'id': 'draft', 'question': domain, 'input_kind': 'personal', 'memory_ids': list(memories)}]
    task = store.create_task(goal=domain, constraints='No sending, payments, file edits or external actions.', questions=rows)
    bridge = PersonalBridge()
    bridge.memories = {key: 1 for key in memories}
    return store, task, bridge, runtime(store, bridge)


def runtime(store, bridge):
    return ProjectTaskRuntime(store, execute_question=bridge.execute, read_sources=bridge.read_sources,
                              read_context=bridge.read_context, reconcile_request=bridge.reconcile,
                              load_answer=bridge.load_answer)


async def run(rt, task, rid=None):
    return await rt.run(task['id'], expected_revision=task['revision'], run_request_id=rid or uid(), max_calls=6)


def step(task, key='draft'):
    return next(row for row in task['steps'] if row['id'] == key)


@pytest.mark.parametrize('domain', ['Prepare a workplace handover', 'Plan a family appointment follow-up'])
def test_instruction_only_no_document_task_completes_but_goal_needs_review(tmp_path, domain):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path, domain=domain)
        done = await run(rt, task)
        assert done['status'] == 'completed' and len(bridge.calls) == 1
        call = bridge.calls[0]
        assert call['input_kind'] == 'personal' and call['document_ids'] == []
        assert call['input_snapshot'] == call['context_snapshot'] == {} and call['memory_ids'] == []
        report = await rt.render_report(task['id'], store.get_artifact(task['id'], 'report'))
        assert report['complete'] is True and report['model_calls'] == 0
        assert report['acceptance_status'] == 'review_required' and report['goal_verified'] is False
        assert report['semantic_verified'] is False and 'DRAFT ONLY' in report['content']
        assert '不是文档证据' in report['content']
        assert len(bridge.calls) == 1
    asyncio.run(scenario())


def test_selected_memory_ids_and_revisions_reach_execution_and_saved_report(tmp_path):
    async def scenario():
        selected, other = uid(), uid()
        store, task, bridge, rt = setup(tmp_path, memories=[selected])
        bridge.memories[other] = 9
        done = await run(rt, task)
        assert done['status'] == 'completed'
        assert bridge.calls[0]['memory_ids'] == [selected]
        assert bridge.calls[0]['context_snapshot'] == {selected: 1}
        assert all(other not in ids for ids in bridge.context_reads)
        refs = store.get_artifact(task['id'], 'draft')['references']
        assert refs['context_refs'] == {selected: 1} and refs['source_ids'] == []
        report = await rt.render_report(task['id'], store.get_artifact(task['id'], 'report'))
        assert selected in report['content'] and other not in report['content']
    asyncio.run(scenario())


def test_unrelated_document_backend_is_not_required_for_personal_instruction_only_task(tmp_path):
    async def scenario():
        _, task, bridge, rt = setup(tmp_path)
        async def unavailable(document_ids=None):
            raise OSError('unrelated document backend unavailable')
        rt.read_sources = unavailable
        done = await run(rt, task)
        assert done['status'] == 'completed' and len(bridge.calls) == 1
    asyncio.run(scenario())


@pytest.mark.parametrize('mutation', ['correct', 'delete'])
def test_changed_selected_memory_before_approved_drive_stales_without_a_call(tmp_path, mutation):
    async def scenario():
        selected = uid()
        store, task, bridge, rt = setup(tmp_path, memories=[selected])
        rid = uid()
        task = store.begin_run(task['id'], expected_revision=task['revision'], request_id=rid,
                               max_calls=1, source_snapshots={}, context_snapshots={selected: 1})
        if mutation == 'correct':
            bridge.memories[selected] = 2
        else:
            del bridge.memories[selected]
        changed = await rt.drive_approved(task['id'], run_request_id=rid)
        assert changed['status'] == 'stale' and len(bridge.calls) == 0
        assert changed['run_remaining'] == 1
    asyncio.run(scenario())


@pytest.mark.parametrize('mutation', ['correct', 'delete'])
def test_inflight_selected_change_never_publishes_or_automatically_repeats(tmp_path, mutation):
    async def scenario():
        selected = uid()
        store, task, bridge, rt = setup(tmp_path, memories=[selected])
        async def change(_):
            if mutation == 'correct':
                bridge.memories[selected] = 2
            else:
                del bridge.memories[selected]
        bridge.action = change
        changed = await run(rt, task)
        assert changed['status'] == 'stale'
        assert step(changed)['artifact_version'] == 0 and len(bridge.calls) == 1
        assert step(changed, 'report')['artifact_version'] == 0
        assert (await rt.drive_approved(task['id'], run_request_id=changed['run_id']))['status'] == 'stale'
        assert len(bridge.calls) == 1
    asyncio.run(scenario())


def test_unselected_memory_and_document_changes_do_not_invalidate_personal_result(tmp_path):
    async def scenario():
        selected, other = uid(), uid()
        store, task, bridge, rt = setup(tmp_path, memories=[selected])
        bridge.memories[other] = 1
        bridge.documents['unrelated'] = 'a' * 64
        async def change(_):
            bridge.memories[other] = 2
            bridge.documents['unrelated'] = 'b' * 64
        bridge.action = change
        done = await run(rt, task)
        assert done['status'] == 'completed' and len(bridge.calls) == 1
        assert (await rt.render_report(task['id'], store.get_artifact(task['id'], 'report')))['complete']
    asyncio.run(scenario())


@pytest.mark.parametrize('allowed', [True, False])
def test_memory_derived_predecessor_does_not_silently_expand_next_step_scope(tmp_path, allowed):
    async def scenario():
        selected = uid()
        rows = [{'id': 'prior', 'question': 'Review my confirmed decision.', 'input_kind': 'personal', 'memory_ids': [selected]},
                {'id': 'next', 'question': 'Draft follow-up.', 'depends_on': ['prior'], 'input_kind': 'personal',
                 'memory_ids': [selected] if allowed else []}]
        store, task, bridge, rt = setup(tmp_path, memories=[selected], questions=rows)
        done = await run(rt, task)
        if allowed:
            assert done['status'] == 'completed' and len(bridge.calls) == 2
            deps = bridge.calls[1]['dependency_artifacts']
            assert deps[0]['references']['context_refs'] == {selected: 1}
        else:
            assert done['status'] == 'blocked' and len(bridge.calls) == 1
            assert step(done, 'next')['reason'] == 'dependency_blocked'
    asyncio.run(scenario())


def test_unknown_personal_outcome_reconciles_without_repeat_then_requires_new_approval(tmp_path):
    async def scenario():
        selected = uid()
        store, task, bridge, rt = setup(tmp_path, memories=[selected])
        async def lost_reply(_):
            return {'request_status': 'unknown'}
        bridge.action = lost_reply
        blocked = await run(rt, task)
        assert step(blocked)['reason'] == 'provider_outcome_unknown'
        with pytest.raises(ProjectTaskConflictError):
            await run(rt, blocked)
        fresh = ProjectTaskStore(tmp_path.resolve(), session_id='#00001', instance_id='private-agent', boot_id='second')
        restored = runtime(fresh, bridge)
        settled = await restored.reconcile(task['id'])
        assert step(settled)['status'] == 'completed' and len(bridge.calls) == 1
        assert settled['status'] == 'paused'
        bridge.action = None
        done = await run(restored, settled)
        assert done['status'] == 'completed' and len(bridge.calls) == 1
    asyncio.run(scenario())


def test_report_read_after_memory_change_cannot_return_old_private_body(tmp_path):
    async def scenario():
        selected = uid()
        store, task, bridge, rt = setup(tmp_path, memories=[selected])
        done = await run(rt, task)
        bridge.memories[selected] = 2
        report = await rt.render_report(task['id'], store.get_artifact(task['id'], 'report'))
        assert report['complete'] is False and 'DRAFT ONLY' not in report['content']
        assert len(bridge.calls) == 1
    asyncio.run(scenario())


@pytest.mark.parametrize('value', [True, 0, -1, '1', 1.5])
def test_malformed_context_reader_never_dispatches(tmp_path, value):
    async def scenario():
        selected = uid()
        _, task, bridge, rt = setup(tmp_path, memories=[selected])
        async def broken(memory_ids=None):
            return {selected: value}
        rt.read_context = broken
        with pytest.raises(ProjectTaskRuntimeError):
            await run(rt, task)
        assert bridge.calls == []
    asyncio.run(scenario())


def test_missing_context_reader_does_not_use_implicit_global_memory(tmp_path):
    async def scenario():
        _, task, bridge, rt = setup(tmp_path, memories=[uid()])
        rt.read_context = None
        with pytest.raises(ProjectTaskRuntimeError):
            await run(rt, task)
        assert bridge.calls == []
    asyncio.run(scenario())


def test_context_reader_cannot_expand_explicit_selection_with_another_private_note(tmp_path):
    async def scenario():
        selected, unselected = uid(), uid()
        _, task, bridge, rt = setup(tmp_path, memories=[selected])
        async def expanded(memory_ids=None):
            return {selected: 1, unselected: 1}
        rt.read_context = expanded
        with pytest.raises(ProjectTaskRuntimeError):
            await run(rt, task)
        assert bridge.calls == []
    asyncio.run(scenario())
