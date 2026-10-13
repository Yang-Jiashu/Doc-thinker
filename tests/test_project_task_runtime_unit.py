"""Durable orchestration tests. Every provider is local; no model/network calls."""
import asyncio
import json
import os
import subprocess
import sys
import uuid

import pytest

from docthinker.server.project_task_runtime import (
    ProjectTaskRuntime, ProjectTaskRuntimeError, ProjectTaskBusyError,
)
from docthinker.server.project_task_store import ProjectTaskStore, ProjectTaskConflictError


def uid():
    return str(uuid.uuid4())


def step(task, key):
    return next(row for row in task['steps'] if row['id'] == key)


class LocalBridge:
    def __init__(self):
        self.documents = {'doc_a': 'a' * 64, 'doc_b': 'b' * 64}
        self.calls, self.reads, self.answers, self.receipts = [], [], {}, {}
        self.action = None

    async def read_sources(self, document_ids=None):
        return {key: value for key, value in self.documents.items()
                if document_ids is None or key in document_ids}

    async def execute_question(self, **kwargs):
        self.calls.append(kwargs)
        if self.action:
            value = await self.action(kwargs)
            if value is not None:
                return value
        answer_id = uid()
        document_ids = list(kwargs['input_snapshot'])[:1]
        sources = [{'id': 'source_' + str(len(self.calls)), 'document_id': document_ids[0],
                    'label': 'S1', 'filename': 'Synthetic Paper.pdf', 'page': 2, 'ordinal': 3,
                    'excerpt_start': 12, 'excerpt_end': 57,
                    'content': 'SOURCE QUOTE MUST NOT BE COPIED INTO REPORT'}] if document_ids else []
        receipt = {'request_status': 'completed', 'answer_id': answer_id,
                   'source_ids': [row['id'] for row in sources],
                   'document_ids': document_ids,
                   'answer': 'MUST NOT BE STORED IN TASK DATABASE', 'hidden_reasoning': 'NEVER STORE'}
        self.answers[answer_id] = {'content': 'Local answer for ' + kwargs['question'], 'sources': sources}
        self.receipts[kwargs['request_id']] = receipt
        return receipt

    async def reconcile_request(self, **kwargs):
        self.reads.append(kwargs)
        return self.receipts.get(kwargs['request_id'], {'request_status': 'unknown'})

    async def load_answer(self, **kwargs):
        return self.answers.get(kwargs['answer_id'])


def setup(tmp_path, questions=None, boot='boot_a'):
    store = ProjectTaskStore(tmp_path.resolve(), session_id='#00001', instance_id='worker_a', boot_id=boot)
    task = store.create_task(goal='Compare material and produce a report', constraints='No external actions',
        questions=questions or [{'id': 'q1', 'question': 'Find mechanisms'},
                                {'id': 'q2', 'question': 'Compare findings', 'depends_on': ['q1']}])
    bridge = LocalBridge()
    return store, task, bridge, runtime(store, bridge)


def runtime(store, bridge):
    return ProjectTaskRuntime(store, execute_question=bridge.execute_question,
        read_sources=bridge.read_sources, reconcile_request=bridge.reconcile_request,
        load_answer=bridge.load_answer)


async def run(rt, task, budget=6, request_id=None):
    return await rt.run(task['id'], expected_revision=task['revision'],
                        run_request_id=request_id or uid(), max_calls=budget)


def test_one_press_runs_dependency_chain_and_report_without_extra_model(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        result = await run(rt, task)
        assert result['status'] == 'completed'
        assert len(bridge.calls) == 2
        assert bridge.calls[0]['dependency_artifacts'] == []
        dep = bridge.calls[1]['dependency_artifacts'][0]
        assert dep['step_id'] == 'q1' and dep['version'] == 1
        assert set(dep['references']) == {'answer_id', 'source_ids', 'document_ids'}
        assert bridge.calls[1]['goal'] == task['goal']
        assert bridge.calls[1]['constraints'] == task['constraints']
        for call in bridge.calls:
            assert call['request_id'] == step(result, 'q1' if call is bridge.calls[0] else 'q2')['request_id']
        artifact = store.get_artifact(task['id'], 'report')
        report = await rt.render_report(task['id'], artifact)
        assert report['complete'] is True and report['model_calls'] == 0
        assert report['semantic_verified'] is False
        assert 'Local answer for Find mechanisms' in report['content']
        assert 'Local answer for Compare findings' in report['content']
        assert len(bridge.calls) == 2
        assert b'MUST NOT BE STORED' not in store.path.read_bytes()
        assert b'NEVER STORE' not in store.path.read_bytes()
    asyncio.run(scenario())


def test_model_budget_stops_and_new_explicit_run_only_executes_remaining(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        first_id = uid()
        paused = await run(rt, task, budget=1, request_id=first_id)
        assert paused['status'] == 'paused' and paused['reason'] == 'budget_exhausted'
        assert len(bridge.calls) == 1 and step(paused, 'q2')['status'] == 'ready'
        # Original request replay is a read, not a fresh allowance or resume.
        replay = await run(rt, task, budget=1, request_id=first_id)
        assert replay['status'] == 'paused' and len(bridge.calls) == 1
        resumed = store.resume_task(task['id'], expected_revision=paused['revision'])
        done = await run(rt, resumed, budget=1)
        assert done['status'] == 'completed' and len(bridge.calls) == 2
        assert step(done, 'q1')['artifact_version'] == 1
    asyncio.run(scenario())


def test_constructor_and_status_reads_never_start_work(tmp_path):
    store, task, bridge, rt = setup(tmp_path)
    assert store.get_task(task['id'])['status'] == 'ready'
    assert store.list_tasks()[0]['id'] == task['id']
    runtime(store, bridge)
    assert not bridge.calls


def test_pause_between_steps_finishes_current_accepted_call_only(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        async def action(_):
            current = store.get_task(task['id'])
            store.pause_task(task['id'], expected_revision=current['revision'])
        bridge.action = action
        paused = await run(rt, task)
        assert paused['status'] == 'paused'
        assert step(paused, 'q1')['status'] == 'completed'
        assert step(paused, 'q2')['status'] == 'ready' and len(bridge.calls) == 1
    asyncio.run(scenario())


def test_cancel_does_not_publish_late_completion_or_claim_rollback(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        async def action(_):
            current = store.get_task(task['id'])
            store.cancel_task(task['id'], expected_revision=current['revision'])
        bridge.action = action
        result = await run(rt, task)
        assert result['status'] == 'cancelled' and len(bridge.calls) == 1
        assert step(result, 'q1')['artifact_version'] == 0
        assert step(result, 'report')['status'] == 'cancelled'
        assert len(bridge.receipts) == 1  # provider/journal outcome may exist despite cancel
    asyncio.run(scenario())


def test_async_cancel_marks_accepted_call_unknown_and_no_retry(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        async def action(_):
            raise asyncio.CancelledError()
        bridge.action = action
        with pytest.raises(asyncio.CancelledError):
            await run(rt, task)
        blocked = store.get_task(task['id'])
        assert step(blocked, 'q1')['reason'] == 'provider_outcome_unknown'
        with pytest.raises(ProjectTaskConflictError):
            await run(rt, blocked)
        assert len(bridge.calls) == 1
    asyncio.run(scenario())


@pytest.mark.parametrize('outcome,reason', [('failed', 'provider_failed'), ('unknown', 'provider_outcome_unknown'),
                                          ('running', 'provider_outcome_unknown'), ('not_found', 'provider_outcome_unknown')])
def test_bridge_failure_stops_without_auto_retry_or_report(tmp_path, outcome, reason):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        async def action(_):
            return {'request_status': outcome}
        bridge.action = action
        result = await run(rt, task)
        assert result['status'] == 'blocked'
        assert step(result, 'q1')['reason'] == reason and len(bridge.calls) == 1
        assert step(result, 'report')['status'] == 'ready'
        assert step(result, 'q2')['status'] == 'ready'
    asyncio.run(scenario())


def test_exception_is_unknown_not_error_text_in_store(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        async def action(_):
            raise OSError('PRIVATE PROVIDER DETAIL')
        bridge.action = action
        result = await run(rt, task)
        assert step(result, 'q1')['reason'] == 'provider_outcome_unknown'
        assert b'PRIVATE PROVIDER DETAIL' not in store.path.read_bytes()
    asyncio.run(scenario())


def test_restart_reconciles_known_completed_without_call_then_requires_resume(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        approved = store.begin_run(task['id'], expected_revision=task['revision'], max_calls=2,
            source_snapshots=bridge.documents, request_id=uid())
        inv = store.claim_next(task['id'], approved['run_id'])
        store.complete_step(task['id'], inv['id'], inv['claim_token'],
            {'documents': [{'id': k, 'digest': v} for k, v in bridge.documents.items()]})
        claimed = store.claim_next(task['id'], approved['run_id'])
        answer_id = uid()
        bridge.receipts[claimed['request_id']] = {'request_status': 'completed', 'answer_id': answer_id,
            'source_ids': ['known_source'], 'document_ids': ['doc_a']}
        bridge.answers[answer_id] = {'content': 'Previously accepted result'}
        # New process, same private persistent DB. Recovery is state-only.
        fresh = ProjectTaskStore(tmp_path.resolve(), session_id='#00001', instance_id='worker_a', boot_id='boot_b')
        fresh.recover()
        newrt = runtime(fresh, bridge)
        recovered = fresh.get_task(task['id'])
        assert recovered['status'] == 'paused'
        assert step(recovered, 'q1')['reason'] == 'provider_outcome_unknown'
        result = await newrt.reconcile(task['id'])
        assert result['status'] == 'paused' and not bridge.calls
        assert step(result, 'q1')['status'] == 'completed'
        assert bridge.reads == [{'session_id': '#00001', 'request_id': claimed['request_id']}]
        resumed = fresh.resume_task(task['id'], expected_revision=result['revision'])
        done = await run(newrt, resumed, budget=1)
        assert done['status'] == 'completed' and len(bridge.calls) == 1
        assert bridge.calls[0]['question'] == 'Compare findings'
        assert bridge.calls[0]['dependency_artifacts'][0]['references']['answer_id'] == answer_id
    asyncio.run(scenario())


def test_unresolved_restart_is_not_replayed_even_after_explicit_resume(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        async def action(_):
            return {'request_status': 'unknown'}
        bridge.action = action
        first = await run(rt, task)
        fresh = ProjectTaskStore(tmp_path.resolve(), session_id='#00001', instance_id='worker_a', boot_id='boot_b')
        fresh.recover()
        newrt = runtime(fresh, bridge)
        unresolved = await newrt.reconcile(task['id'])
        assert step(unresolved, 'q1')['status'] == 'blocked'
        with pytest.raises(ProjectTaskConflictError):
            await run(newrt, unresolved)
        assert len(bridge.calls) == 1
    asyncio.run(scenario())


def test_changed_sources_during_call_stale_only_affected_and_dependent_steps(tmp_path):
    async def scenario():
        questions = [{'id': 'q1', 'question': 'A', 'document_ids': ['doc_a']},
                     {'id': 'q2', 'question': 'B', 'document_ids': ['doc_b']}]
        store, task, bridge, rt = setup(tmp_path, questions)
        # Pause after first answer; independent B then runs and changes its source.
        first = await run(rt, task, budget=1)
        async def action(_):
            bridge.documents['doc_b'] = 'c' * 64
        bridge.action = action
        resumed = store.resume_task(task['id'], expected_revision=first['revision'])
        stale = await run(rt, resumed, budget=1)
        assert stale['status'] == 'stale'
        assert step(stale, 'q1')['status'] == 'completed'
        assert step(stale, 'q2')['status'] == 'stale'
        assert step(stale, 'report')['status'] == 'stale'
        assert step(stale, 'q2')['artifact_version'] == 0
        bridge.action = None
        refreshed = await rt.refresh(task['id'], expected_revision=stale['revision'])
        assert step(refreshed, 'q1')['request_id'] == step(first, 'q1')['request_id']
        assert step(refreshed, 'q2')['request_id'] != step(first, 'q2')['request_id']
        done = await run(rt, refreshed, budget=1)
        assert done['status'] == 'completed' and len(bridge.calls) == 3
        assert step(done, 'q1')['artifact_version'] == 1
    asyncio.run(scenario())


def test_cross_scope_dependencies_are_not_exposed_to_narrower_step(tmp_path):
    async def scenario():
        questions = [{'id': 'q1', 'question': 'A', 'document_ids': ['doc_a']},
                     {'id': 'q2', 'question': 'B', 'document_ids': ['doc_b'], 'depends_on': ['q1']}]
        store, task, bridge, rt = setup(tmp_path, questions)
        done = await run(rt, task)
        assert done['status'] == 'blocked'
        assert step(done, 'q2')['reason'] == 'dependency_blocked'
        assert len(bridge.calls) == 1  # Missing declared dependency is no longer silently omitted.
    asyncio.run(scenario())


def test_indirect_dependencies_not_added_as_direct_context(tmp_path):
    async def scenario():
        questions = [{'id': 'q1', 'question': 'A'},
                     {'id': 'q2', 'question': 'B', 'depends_on': ['q1']},
                     {'id': 'q3', 'question': 'C', 'depends_on': ['q2']}]
        store, task, bridge, rt = setup(tmp_path, questions)
        done = await run(rt, task)
        assert done['status'] == 'completed'
        assert [item['step_id'] for item in bridge.calls[2]['dependency_artifacts']] == ['q2']
    asyncio.run(scenario())


def test_source_deleted_before_reconcile_cannot_publish_old_result(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        async def action(_):
            return {'request_status': 'unknown'}
        bridge.action = action
        blocked = await run(rt, task)
        request_id = step(blocked, 'q1')['request_id']
        bridge.receipts[request_id] = {'request_status': 'completed', 'answer_id': uid(),
            'source_ids': ['x'], 'document_ids': ['doc_a']}
        del bridge.documents['doc_a']
        result = await rt.reconcile(task['id'])
        assert result['status'] == 'stale' and step(result, 'q1')['artifact_version'] == 0
        assert len(bridge.calls) == 1
    asyncio.run(scenario())


@pytest.mark.parametrize('change', ['missing_answer', 'cross_scope', 'duplicate_source', 'invalid_answer_id'])
def test_invalid_completed_receipt_never_becomes_success(tmp_path, change):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path, [{'question': 'A', 'document_ids': ['doc_a']}])
        async def action(_):
            receipt = {'request_status': 'completed', 'answer_id': uid(), 'source_ids': ['x'], 'document_ids': ['doc_a']}
            if change == 'missing_answer':
                del receipt['answer_id']
            elif change == 'cross_scope':
                receipt['document_ids'] = ['doc_b']
            elif change == 'duplicate_source':
                receipt['source_ids'] = ['x', 'x']
            else:
                receipt['answer_id'] = 'not-an-answer-reference'
            return receipt
        bridge.action = action
        result = await run(rt, task)
        assert step(result, 'q1')['reason'] == 'invalid_result'
        assert step(result, 'q1')['artifact_version'] == 0
    asyncio.run(scenario())


def test_report_never_hydrates_withdrawn_or_changed_sources(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        await run(rt, task)
        artifact = store.get_artifact(task['id'], 'report')
        bridge.documents['doc_a'] = 'e' * 64
        result = await rt.render_report(task['id'], artifact)
        assert result['complete'] is False and 'Local answer' not in result['content']
        store.invalidate_sources(['doc_a'], reason='source_revoked')
        with pytest.raises(ProjectTaskRuntimeError):
            await rt.render_report(task['id'], artifact)
        assert len(bridge.calls) == 2
    asyncio.run(scenario())


def test_report_missing_body_uses_explicit_placeholder_not_new_generation(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        await run(rt, task)
        artifact = store.get_artifact(task['id'], 'report')
        bridge.answers.clear()
        report = await rt.render_report(task['id'], artifact['references'])
        assert report['complete'] is False and '未重新生成' in report['content']
        assert len(bridge.calls) == 2
    asyncio.run(scenario())


def test_forged_partial_report_references_rejected(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        await run(rt, task)
        with pytest.raises(ProjectTaskRuntimeError):
            await rt.render_report(task['id'], {'step_refs': [{'step_id': 'q1', 'version': 1}]})
    asyncio.run(scenario())


def test_concurrent_calls_to_same_runtime_do_not_duplicate_provider(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        entered, release = asyncio.Event(), asyncio.Event()
        async def action(_):
            entered.set()
            await release.wait()
        bridge.action = action
        running = asyncio.create_task(run(rt, task))
        await entered.wait()
        with pytest.raises(ProjectTaskBusyError):
            await run(rt, task)
        release.set()
        assert (await running)['status'] == 'completed'
        assert len(bridge.calls) == 2
    asyncio.run(scenario())


@pytest.mark.parametrize('value', [0, 7, True, -1, 1.5])
def test_invalid_run_budget_before_model(tmp_path, value):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        with pytest.raises(ProjectTaskRuntimeError):
            await run(rt, task, budget=value)
        assert not bridge.calls
    asyncio.run(scenario())


def test_sync_hooks_are_supported_without_changing_model_protocol(tmp_path):
    store, task, bridge, _ = setup(tmp_path, [{'question': 'A'}])
    calls = []
    def execute(**kwargs):
        calls.append(kwargs)
        return {'request_status': 'completed', 'answer_id': uid(), 'source_ids': [], 'document_ids': []}
    rt = ProjectTaskRuntime(store, execute_question=execute, read_sources=lambda **kw: bridge.documents)
    assert asyncio.run(run(rt, task))['status'] == 'completed'
    assert len(calls) == 1


def test_separate_runtime_replayed_run_never_dispatches_while_original_inflight(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        entered, release = asyncio.Event(), asyncio.Event()
        async def action(_):
            entered.set()
            await release.wait()
        bridge.action = action
        request_id = uid()
        running = asyncio.create_task(run(rt, task, request_id=request_id))
        await entered.wait()
        other_store = ProjectTaskStore(tmp_path.resolve(), session_id='#00001', instance_id='worker_a', boot_id='boot_a')
        other_runtime = runtime(other_store, bridge)
        current = await run(other_runtime, task, request_id=request_id)
        assert current['status'] == 'running' and len(bridge.calls) == 1
        release.set()
        assert (await running)['status'] == 'completed'
        assert len(bridge.calls) == 2
    asyncio.run(scenario())


def test_six_questions_maximum_has_only_six_model_attempts(tmp_path):
    async def scenario():
        rows = [{'id': 'q' + str(i), 'question': str(i),
                 'depends_on': ['q' + str(i - 1)] if i > 1 else []} for i in range(1, 7)]
        store, task, bridge, rt = setup(tmp_path, rows)
        done = await run(rt, task)
        assert done['status'] == 'completed' and len(bridge.calls) == 6
        assert len(done['steps']) == 8 and done['run_remaining'] == 0
    asyncio.run(scenario())


def test_cancel_after_completed_provider_during_snapshot_is_unknown_not_stuck_running(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        original_read = bridge.read_sources
        async def cancel_after_completion(**kwargs):
            if bridge.receipts:
                raise asyncio.CancelledError()
            return await original_read(**kwargs)
        rt.read_sources = cancel_after_completion
        with pytest.raises(asyncio.CancelledError):
            await run(rt, task)
        blocked = store.get_task(task['id'])
        assert step(blocked, 'q1')['status'] == 'blocked'
        assert step(blocked, 'q1')['reason'] == 'provider_outcome_unknown'
        assert len(bridge.calls) == 1 and len(bridge.receipts) == 1
    asyncio.run(scenario())


def test_report_rechecks_prior_sections_after_later_hydration_yields(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        await run(rt, task)
        artifact = store.get_artifact(task['id'], 'report')
        calls = []
        async def load(**kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                bridge.documents['doc_a'] = 'f' * 64
            return await bridge.load_answer(**kwargs)
        rt.load_answer = load
        with pytest.raises(ProjectTaskRuntimeError, match='changed while rendering'):
            await rt.render_report(task['id'], artifact)
        assert len(bridge.calls) == 2
    asyncio.run(scenario())


def test_old_report_version_stays_unavailable_after_explicit_refresh(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        done = await run(rt, task)
        old = store.get_artifact(task['id'], 'report')
        bridge.documents['doc_a'] = 'f' * 64
        store.invalidate_sources(['doc_a'], reason='source_changed')
        stale = store.get_task(task['id'])
        refreshed = await rt.refresh(task['id'], expected_revision=stale['revision'])
        newest = await run(rt, refreshed)
        assert newest['status'] == 'completed'
        assert step(newest, 'report')['artifact_version'] == 2
        with pytest.raises(ProjectTaskRuntimeError):
            await rt.render_report(task['id'], old)
        assert not store.get_artifact(task['id'], 'report', version=1)['available']
        assert (await rt.render_report(task['id'], store.get_artifact(task['id'], 'report')))['complete']
    asyncio.run(scenario())


def test_real_process_exit_and_durable_journal_completion_reconcile_without_duplicate_call(tmp_path):
    script = '''
import json, sys, uuid
from pathlib import Path
from docthinker.server.project_task_store import ProjectTaskStore
from docthinker.server.request_journal import RequestJournal
directory = Path(sys.argv[1]).resolve()
store = ProjectTaskStore(directory,session_id='#00001',instance_id='worker_a',boot_id='child')
task = store.create_task(goal='Durable research',questions=[{'id':'q1','question':'A'}, {'id':'q2','question':'B','depends_on':['q1']}])
docs = {'doc_a':'a'*64,'doc_b':'b'*64}
approved = store.begin_run(task['id'],expected_revision=task['revision'],max_calls=2,source_snapshots=docs,request_id=str(uuid.uuid4()))
inventory = store.claim_next(task['id'],approved['run_id'])
store.complete_step(task['id'],inventory['id'],inventory['claim_token'],{'documents':[{'id':k,'digest':v} for k,v in docs.items()]})
paid = store.claim_next(task['id'],approved['run_id'])
journal = RequestJournal(directory,boot_id='child')
journal.begin('#00001',paid['request_id'],'d'*64)
journal.complete('#00001',paid['request_id'],{'answer_id':str(uuid.uuid4()),'source_ids':['source_a'],'document_ids':['doc_a'],
    'project_handoff':{'required':0,'used':0,'full':0,'excerpted':0,'omitted':0}})
print(json.dumps({'task_id':task['id'],'request_id':paid['request_id']}))
# Exit before task.complete_step. The completed request is durably on disk.
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHON_DOTENV_DISABLED': '1'},
        text=True, capture_output=True, check=True, timeout=10)
    info = json.loads(result.stdout)
    async def scenario():
        from docthinker.server.request_journal import RequestJournal
        store = ProjectTaskStore(tmp_path.resolve(), session_id='#00001', instance_id='worker_a', boot_id='parent')
        journal = RequestJournal(tmp_path.resolve(), boot_id='parent')
        bridge = LocalBridge()
        async def read_journal(**kwargs):
            result = journal.read(kwargs['session_id'], kwargs['request_id'], active=False)
            assert result['project_handoff'] == {'required': 0, 'used': 0, 'full': 0, 'excerpted': 0, 'omitted': 0}
            # The production bridge restores this opaque, host-generated field
            # after hydrating the current answer, rather than trusting an LLM.
            return {**result, 'handoff': result['project_handoff']}
        rt = ProjectTaskRuntime(store, execute_question=bridge.execute_question,
            read_sources=bridge.read_sources, reconcile_request=read_journal)
        store.recover()
        assert not bridge.calls
        result = await rt.reconcile(info['task_id'])
        assert result['status'] == 'paused' and step(result, 'q1')['status'] == 'completed'
        assert step(result, 'q1')['handoff'] == {'required': 0, 'used': 0, 'full': 0, 'excerpted': 0, 'omitted': 0}
        assert step(result, 'q1')['request_id'] == info['request_id'] and not bridge.calls
        resumed = store.resume_task(info['task_id'], expected_revision=result['revision'])
        done = await run(rt, resumed, budget=1)
        assert done['status'] == 'completed' and len(bridge.calls) == 1
        assert bridge.calls[0]['request_id'] != info['request_id']
        assert bridge.calls[0]['question'] == 'B'
    asyncio.run(scenario())


def test_known_journal_failure_reconciliation_never_resets_or_retries_itself(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        async def action(_):
            return {'request_status': 'unknown'}
        bridge.action = action
        blocked = await run(rt, task)
        request_id = step(blocked, 'q1')['request_id']
        bridge.receipts[request_id] = {'request_status': 'failed', 'error': 'DO NOT STORE RAW ERROR'}
        reconciled = await rt.reconcile(task['id'])
        assert step(reconciled, 'q1')['status'] == 'blocked'
        assert step(reconciled, 'q1')['reason'] == 'provider_failed'
        assert step(reconciled, 'q1')['request_id'] == request_id
        assert len(bridge.calls) == 1 and b'DO NOT STORE RAW ERROR' not in store.path.read_bytes()
        reset = store.reset_failed_task(task['id'], expected_revision=reconciled['revision'])
        assert len(bridge.calls) == 1 and step(reset, 'q1')['request_id'] != request_id
        bridge.action = None
        done = await run(rt, reset)
        assert done['status'] == 'completed' and len(bridge.calls) == 3
    asyncio.run(scenario())


def test_known_journal_failure_for_changed_input_is_stale_not_resettable_failure(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        async def action(_):
            return {'request_status': 'unknown'}
        bridge.action = action
        blocked = await run(rt, task)
        request_id = step(blocked, 'q1')['request_id']
        bridge.receipts[request_id] = {'request_status': 'failed'}
        bridge.documents['doc_a'] = 'f' * 64
        result = await rt.reconcile(task['id'])
        assert result['status'] == 'stale' and step(result, 'q1')['reason'] == 'source_changed'
        assert len(bridge.calls) == 1
    asyncio.run(scenario())


def test_reconcile_does_not_read_or_change_same_boot_inflight_step(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        entered, release = asyncio.Event(), asyncio.Event()
        async def action(kwargs):
            bridge.receipts[kwargs['request_id']] = {'request_status': 'failed'}
            entered.set()
            await release.wait()
        bridge.action = action
        running = asyncio.create_task(run(rt, task))
        await entered.wait()
        result = await rt.reconcile(task['id'])
        assert result['status'] == 'running' and not bridge.reads
        release.set()
        assert (await running)['status'] == 'completed'
        assert len(bridge.calls) == 2
    asyncio.run(scenario())


def test_preapproved_http_grant_drives_without_second_begin_or_extra_allowance(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        request_id = uid()
        store.begin_run(task['id'], expected_revision=task['revision'], max_calls=2,
                        source_snapshots=bridge.documents, request_id=request_id)
        assert not bridge.calls
        # If the internal path incorrectly begins again, this test must fail.
        def no_second_begin(*args, **kwargs):
            raise AssertionError('Approval must not be repeated')
        store.begin_run = no_second_begin
        done = await rt.drive_approved(task['id'], run_request_id=request_id)
        assert done['status'] == 'completed' and len(bridge.calls) == 2
        assert done['run_remaining'] == 0 and done['run_max_calls'] == 2
        same = await rt.drive_approved(task['id'], run_request_id=request_id)
        assert same['status'] == 'completed' and len(bridge.calls) == 2
    asyncio.run(scenario())


def test_drive_cannot_refill_paused_grant_or_take_over_another_approval(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        request_id = uid()
        store.begin_run(task['id'], expected_revision=task['revision'], max_calls=1,
                        source_snapshots=bridge.documents, request_id=request_id)
        paused = await rt.drive_approved(task['id'], run_request_id=request_id)
        assert paused['status'] == 'paused' and len(bridge.calls) == 1
        assert (await rt.drive_approved(task['id'], run_request_id=request_id))['status'] == 'paused'
        store.resume_task(task['id'], expected_revision=paused['revision'])
        # Resume is explicit intent, but an old exhausted allowance is not new payment approval.
        no_grant = await rt.drive_approved(task['id'], run_request_id=request_id)
        assert step(no_grant, 'q2')['status'] == 'ready' and len(bridge.calls) == 1
        with pytest.raises(ProjectTaskConflictError):
            await rt.drive_approved(task['id'], run_request_id=uid())
        assert len(bridge.calls) == 1
    asyncio.run(scenario())


def test_duplicate_preapproved_drive_while_inflight_does_not_dispatch(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        request_id = uid()
        store.begin_run(task['id'], expected_revision=task['revision'], max_calls=2,
                        source_snapshots=bridge.documents, request_id=request_id)
        entered, release = asyncio.Event(), asyncio.Event()
        async def action(_):
            entered.set()
            await release.wait()
        bridge.action = action
        owner = asyncio.create_task(rt.drive_approved(task['id'], run_request_id=request_id))
        await entered.wait()
        with pytest.raises(ProjectTaskBusyError):
            await rt.drive_approved(task['id'], run_request_id=request_id)
        another = runtime(store, bridge)
        assert (await another.drive_approved(task['id'], run_request_id=request_id))['status'] == 'running'
        assert len(bridge.calls) == 1
        release.set()
        assert (await owner)['status'] == 'completed' and len(bridge.calls) == 2
    asyncio.run(scenario())


def test_report_has_section_local_current_source_locators_without_copying_quotes(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        await run(rt, task)
        for answer in bridge.answers.values():
            answer['content'] += ' [S1]'
        report = await rt.render_report(task['id'], store.get_artifact(task['id'], 'report'))
        assert report['complete'] is True
        assert report['content'].count('### 本节来源定位') == 2
        assert report['content'].count('S 标签仅对应本节来源，不跨节合并') == 2
        assert '第 2 页；块序号 3（从 0 计）；块内字符 [12, 57)' in report['content']
        assert '来源 `source_1`' in report['content'] and '来源 `source_2`' in report['content']
        assert 'SOURCE QUOTE MUST NOT BE COPIED INTO REPORT' not in report['content']
    asyncio.run(scenario())


@pytest.mark.parametrize('broken', ['missing_sources', 'missing_page_ordinal', 'missing_offset',
                                  'wrong_source', 'duplicate_label', 'unmapped_citation'])
def test_incomplete_source_metadata_refuses_complete_report(tmp_path, broken):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        await run(rt, task)
        answer = next(iter(bridge.answers.values()))
        if broken == 'missing_sources':
            answer.pop('sources')
        elif broken == 'missing_page_ordinal':
            answer['sources'][0].pop('ordinal')
        elif broken == 'missing_offset':
            answer['sources'][0].pop('excerpt_end')
        elif broken == 'wrong_source':
            answer['sources'][0]['id'] = 'source_other'
        elif broken == 'duplicate_label':
            answer['sources'].append({**answer['sources'][0], 'id': 'second_source'})
        else:
            answer['content'] += ' [S11]'
        report = await rt.render_report(task['id'], store.get_artifact(task['id'], 'report'))
        assert report['complete'] is False
        assert 'Local answer for Find mechanisms' not in report['content']
        assert 'Local answer for Compare findings' in report['content']
        assert len(bridge.calls) == 2
    asyncio.run(scenario())


def test_filename_markup_is_data_and_nonpaged_source_has_explicit_locator(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path, [{'question': 'A'}])
        await run(rt, task)
        source = next(iter(bridge.answers.values()))['sources'][0]
        source.update(filename='<script>x</script>\n[click](https://invalid.example)|x.md', page=None)
        report = await rt.render_report(task['id'], store.get_artifact(task['id'], 'report'))
        assert report['complete'] is True
        assert '<script>' not in report['content'] and '\n[click]' not in report['content']
        assert '&lt;script&gt;' in report['content']
        assert '\\[click\\]\\(' in report['content'] and '\\|x\\.md' in report['content']
        assert '无页码（非分页资料或页码不可用）' in report['content']
    asyncio.run(scenario())


def test_unreferenced_general_answer_is_not_presented_as_evidence_verified(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path, [{'question': 'A'}])
        bridge.documents = {}
        await run(rt, task)
        report = await rt.render_report(task['id'], store.get_artifact(task['id'], 'report'))
        assert report['complete'] is True and report['semantic_verified'] is False
        assert '本节没有可核查的原文来源' in report['content']
    asyncio.run(scenario())


def test_host_handoff_counts_persist_and_report_discloses_partial_reuse(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        original = bridge.execute_question
        async def with_counts(**kwargs):
            receipt = await original(**kwargs)
            count = len(kwargs['dependency_artifacts'])
            receipt['handoff'] = {'required': count, 'used': count, 'full': 0, 'excerpted': count, 'omitted': 0}
            return receipt
        rt.execute_question = with_counts
        done = await run(rt, task)
        expected = {'required': 1, 'used': 1, 'full': 0, 'excerpted': 1, 'omitted': 0}
        assert done['status'] == 'completed' and step(done, 'q2')['handoff'] == expected
        assert store.get_artifact(task['id'], 'q2')['references']['handoff'] == expected
        report = await rt.render_report(task['id'], store.get_artifact(task['id'], 'report'))
        assert report['complete'] is True and report['semantic_verified'] is False
        assert '需要 1 项，实际使用 1 项；完整复用 0 项、节选复用 1 项、未使用 0 项' in report['content']
        assert '本步只接收了前序产物的部分内容' in report['content']
        assert '前序分析仍不是原文证据' in report['content']
    asyncio.run(scenario())


def test_legacy_handoff_absence_is_not_claimed_as_full_reuse(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        done = await run(rt, task)
        assert 'handoff' not in step(done, 'q2')
        report = await rt.render_report(task['id'], store.get_artifact(task['id'], 'report'))
        assert report['complete'] is True
        assert '前序产物使用情况未记录（历史版本）' in report['content']
        assert '完整复用 1 项' not in report['content']
    asyncio.run(scenario())


@pytest.mark.parametrize('counts', [
    {'required': True, 'used': 0, 'full': 0, 'excerpted': 0, 'omitted': 0},
    {'required': 0, 'used': 1, 'full': 0, 'excerpted': 0, 'omitted': 0},
    {'required': 0, 'used': 0, 'full': 0, 'excerpted': 0, 'omitted': 0, 'raw_body': 'untrusted'},
    {'required': 7, 'used': 7, 'full': 7, 'excerpted': 0, 'omitted': 0},
    {'required': 1, 'used': 1, 'full': 1, 'excerpted': 0, 'omitted': 0},  # q1 has zero actual dependencies
])
def test_invalid_handoff_cannot_complete_step(tmp_path, counts):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        original = bridge.execute_question
        async def broken(**kwargs):
            return {**await original(**kwargs), 'handoff': counts}
        rt.execute_question = broken
        result = await run(rt, task)
        assert result['status'] == 'blocked' and step(result, 'q1')['reason'] == 'invalid_result'
        assert step(result, 'q1')['artifact_version'] == 0 and len(bridge.calls) == 1
    asyncio.run(scenario())


def test_dependency_unavailable_is_not_mislabeled_as_provider_failure(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        attempts = []
        original = bridge.execute_question
        async def unavailable(**kwargs):
            if kwargs['dependency_artifacts']:
                attempts.append(kwargs['request_id'])
                return {'request_status': 'failed', 'code': 'dependency_unavailable'}
            return await original(**kwargs)
        rt.execute_question = unavailable
        blocked = await run(rt, task)
        assert step(blocked, 'q2')['reason'] == 'dependency_blocked'
        assert len(attempts) == 1 and len(bridge.calls) == 1
        assert step(blocked, 'report')['artifact_version'] == 0
    asyncio.run(scenario())


def test_completed_journal_reconciliation_restores_host_handoff_counts(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path)
        original = bridge.execute_question
        async def lost_reply(**kwargs):
            receipt = await original(**kwargs)
            count = len(kwargs['dependency_artifacts'])
            receipt['handoff'] = {'required': count, 'used': count, 'full': count, 'excerpted': 0, 'omitted': 0}
            if count:
                return {'request_status': 'unknown'}
            return receipt
        rt.execute_question = lost_reply
        blocked = await run(rt, task)
        assert step(blocked, 'q2')['reason'] == 'provider_outcome_unknown'
        fresh = ProjectTaskStore(tmp_path.resolve(), session_id='#00001', instance_id='worker_a', boot_id='new_boot')
        recovered = await runtime(fresh, bridge).reconcile(task['id'])
        assert step(recovered, 'q2')['handoff'] == {'required': 1, 'used': 1, 'full': 1, 'excerpted': 0, 'omitted': 0}
        assert fresh.get_artifact(task['id'], 'q2')['references']['handoff'] == step(recovered, 'q2')['handoff']
        assert len(bridge.calls) == 2
    asyncio.run(scenario())
