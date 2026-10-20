"""Real SQLite phase/input snapshots; no model, network or answer-quality claim."""
from contextlib import contextmanager
from types import SimpleNamespace
import sqlite3
import json
import time
import uuid

import pytest

from docthinker.server.long_project_context import LongProjectContext
from docthinker.server.project_task_bridge import ProjectTaskBridge
from docthinker.server.project_task_store import ProjectTaskConflictError
from docthinker.server.research_service import ResearchService
from tests.test_long_project_store_unit import opened, project, add, complete, questions


def context(tmp_path, tasks):
    service = ResearchService(tmp_path, context_assistance_enabled=False, semantic_retriever=None,
                              evidence_reranker=None, query_translator=None)
    manager = SimpleNamespace(get_session=lambda sid: {'instance_id': tasks.instance_id}
                              if sid == tasks.session_id else None)
    bridge = ProjectTaskBridge(service, manager, tasks.session_id, None)
    bridge.instance_id = tasks.instance_id
    return LongProjectContext(tasks, bridge), service


def chain(tasks, store, count, *, rows=None, contexts=None, sources=None):
    root, prior = project(store), None
    for index in range(count):
        refs = [{'phase_id': prior['id'], 'report_version': 1, 'step_ids': ['draft']}] if prior else []
        prior = add(store, root, parents=refs, rows=rows, title='Stage ' + str(index + 1))
        complete(tasks, prior['task_id'], contexts=contexts, sources=sources)
    return root, prior


@pytest.mark.parametrize('count', [30, 150])
def test_real_phase_chain_checks_one_task_and_one_research_snapshot_without_128_lifetime_wall(tmp_path, monkeypatch, count):
    tasks, store = opened(tmp_path)
    ctx, service = context(tmp_path, tasks)
    note = service.store.create_memory(tasks.session_id, 'decision', 'Owner-approved household budget.')
    _, last = chain(tasks, store, count, rows=[{**questions()[0], 'memory_ids': [note['id']]}],
                    contexts={note['id']: note['revision']})
    reads, research_reads, groups = [], [], []
    original_db, original_read = tasks._db, service.store._read
    original_groups = service.task_input_groups_live

    @contextmanager
    def db(*, write=False):
        reads.append(write)
        with original_db(write=write) as connection:
            yield connection

    @contextmanager
    def read(*, timeout=5):
        research_reads.append(timeout)
        with original_read(timeout=timeout) as connection:
            yield connection

    def batch(sid, value, **options):
        groups.append(value)
        return original_groups(sid, value, **options)

    monkeypatch.setattr(tasks, '_db', db)
    monkeypatch.setattr(service.store, '_read', read)
    monkeypatch.setattr(service, 'task_input_groups_live', batch)
    assert ctx.valid(last['task_id']) is True
    assert reads == [False]
    assert research_reads == [.05]
    assert len(groups) == 1 and len(groups[0]) == 1
    assert groups[0][0][0]['context_snapshot'] == {note['id']: 1}


def test_batch_checks_fresh_memory_revision_withdrawal_and_owner_instance_on_every_request(tmp_path):
    tasks, store = opened(tmp_path)
    ctx, service = context(tmp_path, tasks)
    note = service.store.create_memory(tasks.session_id, 'decision', 'Keep a 120-minute budget.')
    _, last = chain(tasks, store, 3, rows=[{**questions()[0], 'memory_ids': [note['id']]}],
                    contexts={note['id']: 1})
    assert ctx.valid(last['task_id'])
    service.store.update_memory(tasks.session_id, note['id'], expected_revision=1, text='New budget: 60 minutes.')
    assert not ctx.valid(last['task_id'])
    assert ctx.invalidate_if_stale(last['task_id'])['status'] == 'stale'


def test_service_grouping_captures_time_once_and_does_not_refresh_an_expired_note(tmp_path):
    tasks, _ = opened(tmp_path)
    _, service = context(tmp_path, tasks)
    service.store.clock = lambda: 100
    note = service.store.create_memory(tasks.session_id, 'preference', 'Keep private.', valid_until=101)
    step = {'input_kind': 'personal', 'input_snapshot': {}, 'context_snapshot': {note['id']: 1}}
    calls = []
    service.store.clock = lambda: calls.append(1) or 100.5
    assert service.task_input_groups_live(tasks.session_id, [[step]] * 150)
    assert calls == [1]
    service.store.clock = lambda: 101
    assert not service.task_input_groups_live(tasks.session_id, [[step]] * 150)


def test_document_scopes_keep_exact_source_digest_and_do_not_borrow_unselected_files(tmp_path):
    tasks, store = opened(tmp_path)
    ctx, service = context(tmp_path, tasks)
    selected = service.store.accept_document(tasks.session_id, 'selected.txt', b'Original source.')
    service.store.claim_document(tasks.session_id, selected['id'])
    service.store.complete_document(tasks.session_id, selected['id'], [{'text': 'Original source.'}])
    sources = service.store.document_snapshots(tasks.session_id, [selected['id']])
    rows = [{'id': 'draft', 'question': 'Read only the selected original.', 'document_ids': [selected['id']]}]
    _, last = chain(tasks, store, 3, rows=rows, sources=sources)
    assert ctx.valid(last['task_id'])
    service.store.accept_document(tasks.session_id, 'unselected.txt', b'An unrelated document.')
    assert ctx.valid(last['task_id'])
    service.store.delete_document(tasks.session_id, selected['id'])
    assert not ctx.valid(last['task_id'])


def test_selected_branch_remains_live_when_only_unselected_branch_changes(tmp_path):
    tasks, store = opened(tmp_path)
    ctx, service = context(tmp_path, tasks)
    a = service.store.create_memory(tasks.session_id, 'decision', 'Selected decision.')
    b = service.store.create_memory(tasks.session_id, 'decision', 'Unselected decision.')
    root = project(store)
    first = add(store, root, rows=[{**questions()[0], 'id': 'a', 'memory_ids': [a['id']]},
                                  {**questions()[0], 'id': 'b', 'memory_ids': [b['id']]}])
    complete(tasks, first['task_id'], contexts={a['id']: 1, b['id']: 1})
    second = add(store, root, parents=[{'phase_id': first['id'], 'report_version': 1, 'step_ids': ['a']}])
    assert ctx.valid(second['task_id'])
    service.store.delete_memory(tasks.session_id, b['id'], expected_revision=1)
    tasks.invalidate_memories([b['id']])
    assert ctx.valid(second['task_id'])
    dependencies = ctx.dependencies(task_id=second['task_id'])
    assert len(dependencies) == 1 and dependencies[0]['phase_ref']['step_id'] == 'a'
    service.store.delete_memory(tasks.session_id, a['id'], expected_revision=1)
    assert not ctx.valid(second['task_id'])


def test_dependency_references_and_execution_context_keep_exact_existing_shapes_after_restart(tmp_path):
    tasks, store = opened(tmp_path)
    root, last = chain(tasks, store, 3)
    tasks, store = opened(tmp_path, boot='restarted')
    ctx, _ = context(tmp_path, tasks)
    refs = store.parent_dependencies(last['task_id'])
    delivered = ctx.dependencies(task_id=last['task_id'])
    assert len(delivered) == 1
    assert delivered[0]['phase_ref'] == {'phase_id': refs[0]['phase_id'], 'task_id': refs[0]['task_id'],
        'report_version': refs[0]['report_version'], 'step_id': 'draft', 'version': 1}
    assert delivered[0]['references'] == tasks.get_artifact(refs[0]['task_id'], 'draft', version=1)['references']
    assert ctx.execution_context(task_id=last['task_id']) == {
        'project_goal': root['goal'], 'project_acceptance_criteria': root['acceptance_criteria'],
        'phase_id': last['id'], 'phase_title': last['title'],
        'phase_acceptance_criteria': last['acceptance_criteria']}
    ctx.bridge.manager.get_session = lambda sid: {'instance_id': 'different-owner-incarnation'}
    assert not ctx.valid(last['task_id'])


@pytest.mark.parametrize('limit', ['bytes', 'deadline'])
def test_work_budget_is_an_explicit_gap_not_a_source_change_or_silent_ancestor_skip(tmp_path, monkeypatch, limit):
    import docthinker.server.long_project_context as module
    tasks, store = opened(tmp_path)
    _, last = chain(tasks, store, 2)
    ctx, _ = context(tmp_path, tasks)
    monkeypatch.setattr(module, 'MAX_CHECK_BYTES' if limit == 'bytes' else 'MAX_CHECK_SECONDS', 1 if limit == 'bytes' else 0)
    assert not ctx.valid(last['task_id'])
    with pytest.raises(ProjectTaskConflictError, match='budget exceeded|deadline exceeded'):
        ctx.dependencies(task_id=last['task_id'])
    with pytest.raises(ProjectTaskConflictError, match='budget exceeded|deadline exceeded'):
        ctx.invalidate_if_stale(last['task_id'])
    assert tasks.get_task(last['task_id'])['status'] == 'completed'


def test_original_single_task_limits_remain_eight_steps_and_48_selected_memories(tmp_path):
    tasks, _ = opened(tmp_path)
    _, service = context(tmp_path, tasks)
    step = {'input_kind': 'personal', 'input_snapshot': {}, 'context_snapshot': {}}
    assert service.task_inputs_live(tasks.session_id, [step] * 8)
    for value in [[], [step] * 9]:
        with pytest.raises(ValueError, match='publication scope'):
            service.task_inputs_live(tasks.session_id, value)
    assert not service.task_inputs_live(tasks.session_id, [{**step, 'context_snapshot': {str(index): 1 for index in range(49)}}])
    assert not service.task_input_groups_live(tasks.session_id, [[step]], deadline=time.monotonic() - 1)


def test_brief_real_sqlite_writer_lock_retries_only_the_local_inspection(tmp_path, monkeypatch):
    tasks, store = opened(tmp_path)
    ctx, service = context(tmp_path, tasks)
    note = service.store.create_memory(tasks.session_id, 'decision', 'Keep the current household budget.')
    _, last = chain(tasks, store, 2, rows=[{**questions()[0], 'memory_ids': [note['id']]}],
                    contexts={note['id']: 1})
    before = tasks.get_task(last['task_id'])
    original_read, reads, failures = service.store._read, [], []
    with sqlite3.connect(service.store.path) as writer:
        writer.execute('BEGIN EXCLUSIVE')

        @contextmanager
        def read(*, timeout=5):
            reads.append(timeout)
            try:
                with original_read(timeout=timeout) as db:
                    yield db
            except sqlite3.OperationalError as error:
                failures.append(str(error))
                # The first real bounded SQLite read has observed contention.
                # End that commit window without timing-dependent sleeps.
                writer.rollback()
                raise

        monkeypatch.setattr(service.store, '_read', read)
        assert ctx.invalidate_if_stale(last['task_id'])['status'] == 'completed'
    assert failures == ['database is locked']
    assert reads == [.05, .05]
    assert tasks.get_task(last['task_id']) == before


def test_persistent_real_sqlite_lock_is_unavailable_not_current_or_permanently_stale(tmp_path):
    tasks, store = opened(tmp_path)
    ctx, service = context(tmp_path, tasks)
    note = service.store.create_memory(tasks.session_id, 'decision', 'Current private budget.')
    _, last = chain(tasks, store, 2, rows=[{**questions()[0], 'memory_ids': [note['id']]}],
                    contexts={note['id']: 1})
    before = tasks.get_task(last['task_id'])
    with sqlite3.connect(service.store.path) as writer:
        writer.execute('BEGIN EXCLUSIVE')
        assert not ctx.valid(last['task_id'])
        for operation in (lambda: ctx.dependencies(task_id=last['task_id']),
                          lambda: ctx.invalidate_if_stale(last['task_id'])):
            with pytest.raises(ProjectTaskConflictError, match='temporarily unavailable'):
                operation()
        assert tasks.get_task(last['task_id']) == before
        writer.rollback()
    assert ctx.valid(last['task_id'])
    assert tasks.get_task(last['task_id']) == before
    # A subsequently verified real revision change must still invalidate the
    # child. The unavailable result is not a cache or a freshness exemption.
    service.store.update_memory(tasks.session_id, note['id'], expected_revision=1, text='Corrected private budget.')
    assert ctx.invalidate_if_stale(last['task_id'])['status'] == 'stale'


def test_nested_answer_inspection_deadline_is_not_evidence_of_revocation(tmp_path, monkeypatch):
    import docthinker.server.research_answer_dependencies as dependencies
    from docthinker.server.research_memory_dependencies import ProvenanceInspectionUnavailable
    from tests.test_research_answer_dependencies_unit import bare_answer
    tasks, _ = opened(tmp_path)
    _, service = context(tmp_path, tasks)
    answer_id = bare_answer(service, tasks.session_id)
    step = {'input_kind': 'personal', 'input_snapshot': {}, 'context_snapshot': {}, 'answer_id': answer_id}
    assert service.task_input_groups_live(tasks.session_id, [[step]])
    clock = iter([0., 1.])
    monkeypatch.setattr(dependencies, 'time', SimpleNamespace(monotonic=lambda: next(clock, 1.)))
    with pytest.raises(ProvenanceInspectionUnavailable, match='temporarily unavailable'):
        service.task_input_groups_live(tasks.session_id, [[step]])


@pytest.mark.parametrize('failure', ['event_ledger_sql', 'nested_event_sql'])
def test_nested_event_io_failure_is_not_evidence_of_revocation(tmp_path, monkeypatch, failure):
    from docthinker.server.personal_events import PersonalEvents
    from docthinker.server.research_event_dependencies import EventDependencies
    from docthinker.server.research_memory_dependencies import ProvenanceInspectionUnavailable
    tasks, _ = opened(tmp_path)
    _, service = context(tmp_path, tasks)
    answer_id = str(uuid.uuid4())
    bindings = {'events': [], 'relations': [], 'source_scope_sha256': None, 'memory_scope_sha256': None}
    service.store.record_answer(tasks.session_id, answer_id, [], {}, 'general',
                               context_metadata=EventDependencies.context_metadata(bindings))
    with service.store._write() as db:
        db.execute('CREATE TABLE research_answer_event_dependencies '
                   '(answer_id TEXT PRIMARY KEY,session_id TEXT NOT NULL,bindings TEXT NOT NULL)')
        db.execute('INSERT INTO research_answer_event_dependencies VALUES(?,?,?)',
                   (answer_id, tasks.session_id, json.dumps(bindings)))
    step = {'input_kind': 'personal', 'input_snapshot': {}, 'context_snapshot': {}, 'answer_id': answer_id}
    assert service.task_input_groups_live(tasks.session_id, [[step]])
    if failure == 'event_ledger_sql':
        original_read = service.store._read

        class LedgerUnavailable:
            def __init__(self, db):
                self.db = db

            def execute(self, sql, *args):
                if 'research_answer_event_dependencies' in sql:
                    raise sqlite3.OperationalError('Synthetic nested event ledger IO failure')
                return self.db.execute(sql, *args)

        @contextmanager
        def read(*, timeout=5):
            with original_read(timeout=timeout) as db:
                yield LedgerUnavailable(db)

        monkeypatch.setattr(service.store, '_read', read)
    else:
        def unavailable(*args, **kwargs):
            raise sqlite3.OperationalError('Synthetic nested event binding IO failure')
        monkeypatch.setattr(PersonalEvents, '_event_bindings_live', unavailable)
    with pytest.raises(ProvenanceInspectionUnavailable, match='temporarily unavailable'):
        service.task_input_groups_live(tasks.session_id, [[step]])


def test_memory_dependency_depth_limit_is_unavailable_but_a_real_cycle_is_invalid():
    from docthinker.server.research_memory_dependencies import MAX_BINDINGS, source_answer_memory_live
    seen = {'active': set(), 'memo': {}, 'memory_depth': MAX_BINDINGS}
    assert not source_answer_memory_live(None, '#00001', 'answer', at=100, seen=seen)
    assert seen['inspection_unavailable'] is True
    cyclic = {'active': {('#00001', 'answer', 100)}, 'memo': {}, 'memory_depth': MAX_BINDINGS}
    assert not source_answer_memory_live(None, '#00001', 'answer', at=100, seen=cyclic)
    assert 'inspection_unavailable' not in cyclic
