"""Independent durable-project boundary tests; every model is local synthetic code."""
import asyncio
import sqlite3
import uuid

import pytest
from fastapi import HTTPException

from docthinker.server.invite_store import InviteStore, InviteError
from docthinker.server.project_task_bridge import ProjectTaskBridge
from docthinker.server.request_journal import RequestJournal
from docthinker.server.research_service import ResearchService
from docthinker.server.routers import project_tasks
from docthinker.server.schemas import QueryRequest
from tests.test_project_task_runtime_unit import setup as runtime_setup
from tests.test_research_service_unit import SessionManager, text_extractor


def uid():
    return str(uuid.uuid4())


def test_real_router_first_approval_runs_once_and_replay_is_read_only(tmp_path, monkeypatch):
    async def scenario():
        store, created, bridge, runtime = runtime_setup(tmp_path)
        bridge.model = object()
        async def context(_):
            return ('isolated',), runtime, bridge
        monkeypatch.setattr(project_tasks, '_context', context)
        monkeypatch.setattr(project_tasks, '_jobs', {})
        monkeypatch.setattr(project_tasks, '_admission', asyncio.Lock())
        body = project_tasks.RunTask(session_id='#00001', task_id=created['id'],
            expected_revision=created['revision'], request_id=uid(), max_calls=2)
        await project_tasks.run_task(body)
        jobs = tuple(project_tasks._jobs.values())
        assert len(jobs) == 1
        await asyncio.gather(*jobs)
        assert store.get_task(created['id'])['status'] == 'completed'
        assert len(bridge.calls) == 2
        await project_tasks.run_task(body)
        assert len(bridge.calls) == 2 and not project_tasks._jobs
    asyncio.run(scenario())


@pytest.mark.parametrize('changed', [{'max_calls': 2}, {'expected_revision': 2}])
def test_active_approval_rejects_same_id_changed_semantics(tmp_path, monkeypatch, changed):
    async def scenario():
        store, created, bridge, runtime = runtime_setup(tmp_path)
        bridge.model = object()
        entered, release = asyncio.Event(), asyncio.Event()
        async def action(_):
            entered.set()
            await release.wait()
        bridge.action = action
        async def context(_):
            return ('isolated',), runtime, bridge
        monkeypatch.setattr(project_tasks, '_context', context)
        monkeypatch.setattr(project_tasks, '_jobs', {})
        monkeypatch.setattr(project_tasks, '_admission', asyncio.Lock())
        data = dict(session_id='#00001', task_id=created['id'],
                    expected_revision=created['revision'], request_id=uid(), max_calls=1)
        await project_tasks.run_task(project_tasks.RunTask(**data))
        await asyncio.wait_for(entered.wait(), 2)
        try:
            with pytest.raises(HTTPException) as error:
                await project_tasks.run_task(project_tasks.RunTask(**{**data, **changed}))
            assert error.value.status_code == 409
        finally:
            jobs = tuple(project_tasks._jobs.values())
            release.set()
            await asyncio.gather(*jobs)
        assert len(bridge.calls) == 1
    asyncio.run(scenario())


def test_invite_run_units_atomic_bound_and_replay_does_not_refill(tmp_path, monkeypatch):
    monkeypatch.setattr('docthinker.server.invite_store.QUERY_DAILY_LIMIT', 6)
    store = InviteStore.initialize((tmp_path / 'registry').resolve())
    invitation = store.add_user('alice', 8101)
    auth = store.register(invitation['invite_code'], 'alice', 'offline-test-password-123')
    user, rid = auth['user']['id'], uid()
    assert store.reserve_query(user, '#00001', rid, 'a' * 64, units=6)
    assert not store.reserve_query(user, '#00001', rid, 'a' * 64, units=6)
    with pytest.raises(InviteError) as conflict:
        store.reserve_query(user, '#00001', rid, 'a' * 64, units=5)
    assert conflict.value.status == 409
    with pytest.raises(InviteError) as exhausted:
        store.reserve_query(user, '#00001', uid(), 'b' * 64, units=1)
    assert exhausted.value.status == 429
    with sqlite3.connect(store.path) as db:
        assert db.execute('SELECT SUM(count) FROM usage WHERE kind=?', ('query',)).fetchone()[0] == 6


@pytest.mark.parametrize('units', [True, False, 0, 7, 1.0, '2'])
def test_invite_run_units_exact_types(tmp_path, units):
    store = InviteStore.initialize((tmp_path / 'registry').resolve())
    with pytest.raises(InviteError) as error:
        store.reserve_query(uid(), '#00001', uid(), 'a' * 64, units=units)
    assert error.value.status == 422


async def _workspace(tmp_path):
    manager = SessionManager(str(tmp_path / '_system'), str(tmp_path), migrate_legacy=False)
    sid = manager.create_session('Scoped project')['id']
    # This test instruments the literal candidate-derivation lane. Select it
    # explicitly instead of letting an installed local reranker change which
    # function is observed on different developer/CI machines.
    service = ResearchService(tmp_path, extractor=text_extractor, context_assistance_enabled=True,
                              semantic_retriever=None, evidence_reranker=None, query_translator=None)
    docs = []
    for filename, text in [('allowed.txt', 'Cedar cache preserves previous records under the retention rule. ALLOWED_SCOPE_MARKER.'),
                           ('excluded.txt', 'Cedar cache preserves previous records under the retention rule. PRIVATE_OTHER_DOCUMENT_MARKER.')]:
        row = service.store.accept_document(sid, filename, text.encode())
        await service.process(sid, row['id'])
        docs.append(row['id'])
    return manager, service, sid, docs


def _query(sid):
    return QueryRequest(session_id=sid, question='Explain how Cedar cache preserves previous records.',
                        use_memory=False, use_conversation_context=False, use_llm_cache=False,
                        use_self_evolution=False, knowledge_skill_mode='off', insight_mode='off')


def test_real_service_scope_filters_model_packet_and_candidate_derivations(tmp_path, monkeypatch):
    async def scenario():
        manager, service, sid, docs = await _workspace(tmp_path)
        calls, candidates_seen = [], []
        chunks = service.all_chunks(sid)
        allowed = next(row['id'] for row in chunks if row['document_id'] == docs[0])
        excluded = next(row['id'] for row in chunks if row['document_id'] == docs[1])
        monkeypatch.setattr(service.evidence, 'candidates', lambda *_: ([
            {'id': 'allowed', 'source_ids': [allowed]},
            {'id': 'excluded', 'source_ids': [excluded]},
            {'id': 'mixed', 'source_ids': [allowed, excluded]}], []))
        import docthinker.server.research_service as module
        original = module.build_context_assist
        def capture(question, sources, material, fetch, **kwargs):
            rows, gaps = fetch('deduction', [])
            candidates_seen.extend(rows)
            assert all(row['document_id'] == docs[0] for row in material)
            return original(question, sources, material, lambda *_: ([], gaps), **kwargs)
        monkeypatch.setattr(module, 'build_context_assist', capture)
        async def model(prompt, **_):
            calls.append(prompt)
            return 'The cache preserves previous records under the retention rule [S1].'
        try:
            result = await service.answer(_query(sid), model, manager, document_ids=[docs[0]])
            assert len(calls) == 1
            assert 'ALLOWED_SCOPE_MARKER' in calls[0]
            assert 'PRIVATE_OTHER_DOCUMENT_MARKER' not in calls[0]
            assert {row['id'] for row in candidates_seen} == {'allowed'}
            assert result['sources'] and all(row['document_id'] == docs[0] for row in result['sources'])
        finally:
            await service.stop()
    asyncio.run(scenario())


def test_old_instance_paid_completion_cannot_write_into_recreated_project(tmp_path, monkeypatch):
    async def scenario():
        manager, service, sid, docs = await _workspace(tmp_path)
        bridge = ProjectTaskBridge(service, manager, sid, None)
        bridge.instance_id = manager.get_session(sid)['instance_id']
        journal = RequestJournal(tmp_path, boot_id='acceptance')
        from docthinker.server.routers import mobile
        monkeypatch.setattr(mobile, '_query_journal', lambda: journal)
        monkeypatch.setattr(mobile, '_active_queries', set())
        monkeypatch.setattr(mobile, '_starting_query_keys', set())
        monkeypatch.setattr(mobile, '_query_tasks', {})
        calls, returned = [], []
        async def model(prompt, **_):
            calls.append(prompt)
            assert manager.delete_session(sid)
            replacement = manager.create_session('Replacement project')
            assert replacement['id'] == sid
            assert manager.get_session(sid)['instance_id'] != bridge.instance_id
            returned.append(True)
            return 'OLD_INSTANCE_PRIVATE_ANSWER [S1].'
        bridge.model = model
        snapshot = await bridge.read_sources([docs[0]])
        try:
            try:
                await bridge.execute_question(session_id=sid, request_id=uid(),
                    question='Explain how Cedar cache preserves previous records.', document_ids=[docs[0]],
                    input_snapshot=snapshot, goal='Review the cache', constraints='No external actions')
            except Exception:
                pass  # A safe rejection is permitted; leaking to the new instance is not.
            assert len(calls) == 1 and returned == [True]
            assert manager.get_history(sid) == []
        finally:
            await service.stop()
    asyncio.run(scenario())


def test_revoked_invited_worker_cannot_start_next_model_step(tmp_path, monkeypatch):
    async def scenario():
        registry = InviteStore.initialize((tmp_path / 'registry').resolve())
        invitation = registry.add_user('alice', 8101)
        auth = registry.register(invitation['invite_code'], 'alice', 'offline-test-password-123')
        user_id = auth['user']['id']
        registry.revoke_user('alice')
        monkeypatch.setenv('DOCTHINKER_INVITE_WORKER_ID', user_id)
        monkeypatch.setenv('DOCTHINKER_INVITE_WORKER_PORT', '8101')
        monkeypatch.setenv('DOCTHINKER_MOBILE_ACCESS_TOKEN', 'revoked-worker-token')
        monkeypatch.setenv('DOCTHINKER_MOBILE_DATA_DIR', str(registry.root / 'users' / user_id))
        calls = []
        async def model(*_args, **_kwargs):
            calls.append(True)
            return 'Should never be generated.'
        # Reauthorization precedes document access, request acceptance and any
        # provider work; these deliberately absent adapters cannot be used.
        bridge = ProjectTaskBridge(None, None, '#00001', model)
        result = await bridge.execute_question(session_id='#00001', request_id=uid(),
            question='Compare records.', document_ids=None, input_snapshot={},
            goal='Review records', constraints='No external actions')
        assert result == {'request_status': 'failed', 'code': 'worker_authorization_unavailable'}
        assert calls == []
    asyncio.run(scenario())


def test_report_versions_do_not_silently_change_after_title_edit(tmp_path, monkeypatch):
    async def scenario():
        store, created, bridge, runtime = runtime_setup(tmp_path)
        async def context(_):
            return ('isolated',), runtime, bridge
        monkeypatch.setattr(project_tasks, '_context', context)
        done = await runtime.run(created['id'], expected_revision=created['revision'],
                                 run_request_id=uid(), max_calls=2)
        original = await project_tasks.artifact('#00001', created['id'], version=1, step_id='report')
        assert original['artifact']['available']
        assert len(bridge.calls) == 2
        questions = [{key: row[key] for key in ('id', 'title', 'question', 'document_ids')}
                     | {'depends_on': [key for key in row['depends_on'] if key != 'inventory']}
                     for row in done['steps'] if row['kind'] == 'analysis']
        questions[0]['title'] = 'Revised human title'
        revised = store.revise_task(created['id'], expected_revision=done['revision'],
            goal=done['goal'], constraints=done['constraints'], questions=questions)
        unavailable = await project_tasks.artifact('#00001', created['id'], version=1, step_id='report')
        assert unavailable['artifact']['available'] is False
        assert unavailable['artifact']['content'] is None
        assert unavailable['artifact']['references'] is None
        refreshed = await runtime.refresh(created['id'], expected_revision=revised['revision'])
        await runtime.run(created['id'], expected_revision=refreshed['revision'],
                          run_request_id=uid(), max_calls=1)
        current = await project_tasks.artifact('#00001', created['id'], version=2, step_id='report')
        assert current['artifact']['available'] and 'Revised human title' in current['artifact']['content']
        assert len(bridge.calls) == 2  # A local report rebuild is not another model call.
        old = await project_tasks.artifact('#00001', created['id'], version=1, step_id='report')
        assert old['artifact']['content'] is None and not old['artifact']['available']
    asyncio.run(scenario())
