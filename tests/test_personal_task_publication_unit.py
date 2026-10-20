"""Atomic final-input publication checks; no provider, paid call or repair."""
from contextlib import contextmanager
import sqlite3
import threading
import time
import uuid

import pytest

from tests.test_personal_tasks_mobile_unit import (
    mobile, research_mobile, mobile_client, memory, create, start, settled, artifact, plan, SID,
)
from tests.test_research_mobile_unit import upload_ready, PREFIX, OTHER_SID
from docthinker.server.routers import project_tasks


async def prepared_inputs(mobile, browser, kind):
    note = await memory(mobile, browser, '内部交接时间已确认，不得外发。') if kind != 'documents' else None
    document = await upload_ready(mobile, browser, text='Customer handover is Friday; do not send externally.') if kind != 'personal' else None
    rows = []
    if note:
        rows.append({'id': 'personal', 'question': '起草内部交接安排，不执行。',
                     'input_kind': 'personal', 'memory_ids': [note['id']]})
    if document:
        rows.append({'id': 'work', 'question': 'What is the handover date?', 'document_ids': [document]})
    task = await create(browser, {'session_id': SID, 'goal': 'Private work and family follow-up',
                                 'constraints': 'Do not send or write files.', 'questions': rows})
    _, runtime, bridge = await project_tasks._context(SID)
    documents, contexts = await runtime.approval_snapshots(task['id'])
    approved = runtime.store.begin_run(task['id'], expected_revision=task['revision'], max_calls=1,
        source_snapshots=documents, context_snapshots=contexts, request_id=str(uuid.uuid4()))
    steps = [row for row in approved['steps'] if row['kind'] == 'analysis']
    return steps, note, document, bridge


@pytest.mark.parametrize('kind', ['documents', 'personal', 'mixed'])
async def test_batch_barrier_accepts_exact_work_personal_and_mixed_inputs_without_model(mobile, kind):
    try:
        async with mobile_client(mobile) as browser:
            steps, _, _, bridge = await prepared_inputs(mobile, browser, kind)
            assert bridge.validate_inputs(steps) is True
            assert mobile.service.task_inputs_live(SID, steps) is True
            assert mobile.service.task_inputs_live(OTHER_SID, steps) is False
            mobile.model.assert_not_awaited()
    finally:
        await project_tasks.shutdown_project_tasks()


@pytest.mark.parametrize('mutation', ['document', 'memory'])
async def test_batch_barrier_rejects_one_withdrawn_input_in_mixed_packet(mobile, mutation):
    try:
        async with mobile_client(mobile) as browser:
            steps, note, document, bridge = await prepared_inputs(mobile, browser, 'mixed')
            if mutation == 'document':
                mobile.service.store.delete_document(SID, document)
            else:
                mobile.service.store.delete_memory(SID, note['id'], expected_revision=1)
            assert bridge.validate_inputs(steps) is False
            mobile.model.assert_not_awaited()
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_final_barrier_checks_current_session_instance_not_only_visible_session_number(mobile):
    try:
        async with mobile_client(mobile) as browser:
            steps, _, _, bridge = await prepared_inputs(mobile, browser, 'personal')
            mobile.manager.get_session.side_effect = lambda sid: {'id': sid, 'instance_id': 'recreated-private-instance'}
            assert bridge.validate_inputs(steps) is False
            mobile.model.assert_not_awaited()
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_busy_publication_guard_is_bounded_and_never_exports_or_generates(mobile):
    try:
        async with mobile_client(mobile) as browser:
            steps, _, _, bridge = await prepared_inputs(mobile, browser, 'mixed')
            with sqlite3.connect(mobile.service.store.path) as lock:
                lock.execute('BEGIN EXCLUSIVE')
                before = time.monotonic()
                assert bridge.validate_inputs(steps) is False
                elapsed = time.monotonic() - before
            assert elapsed < .5  # 50ms DB deadline plus generous CI scheduling margin.
            mobile.model.assert_not_awaited()
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_document_and_memory_check_share_one_sqlite_snapshot_not_mixed_instants(mobile, monkeypatch):
    try:
        async with mobile_client(mobile) as browser:
            steps, note, document, bridge = await prepared_inputs(mobile, browser, 'mixed')
            personal = next(row for row in steps if row.get('input_kind') == 'personal')
            personal['context_snapshot'] = {note['id']: 2}
            store = mobile.service.store
            original_read = store._read
            start_writer, updated = threading.Event(), threading.Event()
            failures = []
            def writer():
                try:
                    assert start_writer.wait(2)
                    with store._write() as db:
                        db.execute("UPDATE research_documents SET state='deleted' WHERE document_id=?", (document,))
                        db.execute('UPDATE research_memories SET revision=2,updated_at=? WHERE memory_id=?',
                                   (store.clock(), note['id']))
                        store._snapshot(db, store._required_memory(db, SID, note['id']))
                        updated.set()
                except BaseException as exc:
                    failures.append(type(exc).__name__)
                    updated.set()
            thread = threading.Thread(target=writer, daemon=True)
            thread.start()
            @contextmanager
            def tracked_read(*, timeout=5):
                with original_read(timeout=timeout) as db:
                    def traced(sql):
                        if sql.startswith('SELECT memory_id,revision FROM research_memories'):
                            start_writer.set()
                            assert updated.wait(2)
                    db.set_trace_callback(traced)
                    yield db
            monkeypatch.setattr(store, '_read', tracked_read)
            assert bridge.validate_inputs(steps) is False
            thread.join(2)
            assert not thread.is_alive() and not failures
            # The expected old-document/new-memory combination never existed
            # in a committed state. A per-SELECT reader could manufacture it.
            assert store.get_memory(SID, note['id'])['revision'] == 2
            mobile.model.assert_not_awaited()
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_step_artifact_withdrawn_during_answer_await_does_not_export_cached_body(mobile):
    secret = 'STEP_PRIVATE_BODY_WITHDRAWN_7331'
    try:
        async with mobile_client(mobile) as browser:
            note = await memory(mobile, browser, '家庭预算确认，禁止外发。')
            mobile.model.return_value = secret
            task = await create(browser, plan([note['id']]))
            await start(browser, task, calls=1)
            assert (await settled(browser, task['id']))['status'] == 'completed'
            _, _, bridge = await project_tasks._context(SID)
            original = bridge.load_answer
            state = {'withdrawn': False}
            async def load_then_withdraw(**kwargs):
                answer = await original(**kwargs)
                if not state['withdrawn']:
                    mobile.service.store.delete_memory(SID, note['id'], expected_revision=1)
                    state['withdrawn'] = True
                return answer
            bridge.load_answer = load_then_withdraw
            result = await artifact(browser, task, step_id='draft')
            assert state['withdrawn'] and not result['available']
            assert result['content'] is None and result['references'] is None
            assert secret not in str(result) and mobile.model.await_count == 1
    finally:
        await project_tasks.shutdown_project_tasks()


async def test_report_barrier_rechecks_prior_sections_after_all_async_hydration(mobile):
    secret = 'REPORT_PRIVATE_FIRST_SECTION_921'
    try:
        async with mobile_client(mobile) as browser:
            first = await memory(mobile, browser, '家庭预算已确认，禁止外发。')
            second = await memory(mobile, browser, '内部交接已确认，不执行。')
            payload = plan([first['id']])
            payload['questions'].append({'id': 'work', 'question': '起草内部交接。', 'input_kind': 'personal',
                                         'memory_ids': [second['id']]})
            mobile.model.side_effect = [secret, '第二节内部交接草稿。']
            task = await create(browser, payload)
            await start(browser, task, calls=2)
            assert (await settled(browser, task['id']))['status'] == 'completed'
            _, runtime, _ = await project_tasks._context(SID)
            original = runtime.validate_inputs
            state = {'calls': 0}
            def withdraw_at_final_barrier(steps):
                state['calls'] += 1
                assert len(steps) == 2
                mobile.service.store.delete_memory(SID, first['id'], expected_revision=1)
                return original(steps)
            runtime.validate_inputs = withdraw_at_final_barrier
            response = await browser.get(PREFIX + '/workbench/task/artifact', params={
                'session_id': SID, 'task_id': task['id'], 'version': 1})
            assert state['calls'] == 1 and secret not in response.text
            assert response.status_code in {200, 409, 422, 503}
            if response.status_code == 200:
                assert response.json()['artifact']['available'] is False
            assert mobile.model.await_count == 2
    finally:
        await project_tasks.shutdown_project_tasks()
