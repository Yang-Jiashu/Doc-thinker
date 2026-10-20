"""Private-life/work memory lineage: offline SQLite and local lifecycle probes."""
import json
import uuid

import pytest

from docthinker.server.research_memory_dependencies import (
    MAX_BINDINGS, MemoryDependencies, source_answer_memory_live,
)
from docthinker.server.research_store import ResearchStore, ResearchStoreError
from docthinker.server.research_service import ResearchService
from tests.test_research_service_unit import FakeModel, request, text_extractor, workspace


SID = '#00001'


def recorded(store, deps, bindings=(), *, sid=SID):
    identifier = str(uuid.uuid4())
    store.record_answer(sid, identifier, [], {}, 'memory',
                        context_metadata={'used_memory_count': min(8, len(bindings))})
    deps.record(sid, identifier, list(bindings))
    return identifier


def test_as_of_event_binding_stays_historical_but_current_binding_is_revoked(tmp_path):
    now = [1000]
    store = ResearchStore(tmp_path, clock=lambda: now[0])
    deps = MemoryDependencies(store)
    note = store.create_memory(SID, 'decision', 'Family budget: 5000.', valid_from=1000)
    current = recorded(store, deps, [(note['id'], 1, None)])
    historical = recorded(store, deps, [(note['id'], 1, 1200)])
    now[0] = 2000
    store.update_memory(SID, note['id'], expected_revision=1, text='Family budget: 4500.', valid_from=1500)
    assert deps.states(SID, [current, historical]) == {current: False, historical: True}
    store.delete_memory(SID, note['id'], expected_revision=2)
    assert deps.states(SID, [current, historical]) == {current: False, historical: False}


def test_backdated_correction_invalidates_a_previously_historical_binding(tmp_path):
    now = [1000]
    store = ResearchStore(tmp_path, clock=lambda: now[0])
    deps = MemoryDependencies(store)
    note = store.create_memory(SID, 'decision', 'Client deadline: Friday.', valid_from=1000)
    answer = recorded(store, deps, [(note['id'], 1, 1200)])
    now[0] = 2000
    store.update_memory(SID, note['id'], expected_revision=1, text='Correction: Thursday.', valid_from=1100)
    assert deps.states(SID, [answer]) == {answer: False}


def diamond(store, deps):
    root = store.create_memory(SID, 'decision', 'Appointment is Tuesday; do not contact anyone.')
    branches = []
    for name in ('work calendar', 'family calendar'):
        answer = recorded(store, deps, [(root['id'], 1, None)])
        note = store.create_memory(SID, 'finding', name, source_answer_id=answer)
        branches.append(note)
    top = recorded(store, deps, [(row['id'], 1, None) for row in branches])
    return root, branches, top


def test_diamond_shared_ancestor_is_not_mistaken_for_a_cycle(tmp_path):
    store = ResearchStore(tmp_path, clock=lambda: 1000)
    deps = MemoryDependencies(store)
    root, branches, top = diamond(store, deps)
    assert deps.states(SID, [top]) == {top: True}
    store.delete_memory(SID, root['id'], expected_revision=1)
    assert deps.states(SID, [top]) == {top: False}
    assert all(store.get_memory(SID, row['id'])['redacted'] for row in branches)


def test_cyclic_corrupt_lineage_fails_closed_without_unbounded_recursion(tmp_path):
    store = ResearchStore(tmp_path, clock=lambda: 1000)
    deps = MemoryDependencies(store)
    root, _, top = diamond(store, deps)
    # Public confirmation cannot create a cycle; simulate an older/corrupt
    # durable row to test the read boundary, not a model-generated graph.
    with store._write() as db:
        for table in ('research_memories', 'research_memory_versions'):
            db.execute(f'UPDATE {table} SET source_answer_id=? WHERE memory_id=?', (top, root['id']))
    assert deps.states(SID, [top]) == {top: False}


def test_transitive_conversation_flattening_keeps_revision_and_historical_time(tmp_path):
    store = ResearchStore(tmp_path, clock=lambda: 2000)
    deps = MemoryDependencies(store)
    note = store.create_memory(SID, 'decision', 'Do not send client material.', valid_from=1000)
    one = recorded(store, deps, [(note['id'], 1, 1200)])
    bindings = deps.collect(SID, [], history_ids=[one])
    two = recorded(store, deps, bindings)
    assert deps.collect(SID, [], history_ids=[one, two]) == [(note['id'], 1, 1200)]
    store.delete_memory(SID, note['id'], expected_revision=1)
    assert deps.states(SID, [one, two]) == {one: False, two: False}


def test_wrong_tenant_and_other_private_database_cannot_reconnect_a_note(tmp_path):
    first, second = tmp_path / 'one', tmp_path / 'two'
    first.mkdir(); second.mkdir()
    store = ResearchStore(first, clock=lambda: 1000)
    deps = MemoryDependencies(store)
    note = store.create_memory(SID, 'preference', 'Keep family appointments private.')
    answer = recorded(store, deps, [(note['id'], 1, None)])
    assert deps.live('#00002', [(note['id'], 1, None)]) is False
    assert deps.states('#00002', [answer]) == {}
    assert MemoryDependencies(ResearchStore(second)).states(SID, [answer]) == {}
    with store._read() as db:
        assert source_answer_memory_live(db, '#00002', answer, at=1000) is False


@pytest.mark.parametrize('raw', ['null', '{}', '[]', 'not-json', '[["absent",0,null]]',
                                '[["absent",1,true]]', '[["absent",1,NaN]]'])
def test_malformed_dependency_metadata_never_becomes_live(tmp_path, raw):
    store = ResearchStore(tmp_path, clock=lambda: 1000)
    deps = MemoryDependencies(store)
    note = store.create_memory(SID, 'preference', 'Use concise handovers.')
    answer = recorded(store, deps, [(note['id'], 1, None)])
    with store._write() as db:
        db.execute('UPDATE research_answer_memory_dependencies SET bindings=? WHERE answer_id=?', (raw, answer))
    assert deps.states(SID, [answer]) == {answer: False}


def test_known_legacy_memory_count_without_lineage_cannot_reconfirm(tmp_path):
    store = ResearchStore(tmp_path)
    answer = str(uuid.uuid4())
    store.record_answer(SID, answer, [], {}, 'memory', context_metadata={'used_memory_count': 1})
    with pytest.raises(ResearchStoreError):
        store.create_memory(SID, 'finding', 'Cannot reconnect an unknown old family budget.', source_answer_id=answer)


def test_decode_enforces_global_binding_limit_and_does_not_truncate():
    with pytest.raises(ResearchStoreError):
        MemoryDependencies._decode(json.dumps([['note', 1, None]] * (MAX_BINDINGS + 1)))


@pytest.mark.parametrize('length,expected', [(8, True), (MAX_BINDINGS, True), (MAX_BINDINGS + 1, False)])
def test_transitive_answer_walk_has_an_explicit_depth_boundary(tmp_path, length, expected):
    store = ResearchStore(tmp_path, clock=lambda: 1000)
    deps = MemoryDependencies(store)
    notes = [store.create_memory(SID, 'finding', f'Private appointment record {index}.') for index in range(length)]
    answers = [recorded(store, deps, [(note['id'], 1, None)]) for note in notes]
    with store._write() as db:
        for index in range(length - 1):
            for table in ('research_memories', 'research_memory_versions'):
                db.execute(f'UPDATE {table} SET source_answer_id=? WHERE memory_id=?',
                           (answers[index + 1], notes[index]['id']))
    # The read-only walk must fail closed, rather than silently dropping the
    # oldest ancestor, when its bounded dependency walk cannot be completed.
    with store._read() as db:
        assert source_answer_memory_live(db, SID, answers[0], at=1000) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize('historical_writer_without_required_bit', [False, True])
async def test_partial_save_of_history_derived_answer_cannot_launder_forgotten_memory(
        workspace, monkeypatch, historical_writer_without_required_bit):
    root, manager, sid, _ = workspace
    monkeypatch.setenv('DOCTHINKER_RESEARCH_SEMANTIC', 'off')
    service = ResearchService(root, extractor=text_extractor, context_assistance_enabled=False,
                              semantic_retriever=None, evidence_reranker=None, query_translator=None)
    try:
        original = service.store.create_memory(sid, 'decision', '私人家庭预算为4827元，不得自动付款。')
        first_model = FakeModel('私人家庭预算为4827元，不得自动付款。')
        first = await service.answer(request(sid, '我已确认的决定是什么？', use_memory=True), first_model, manager)
        assert first['used_memory_count'] == 1 and original['text'] in first_model.prompts[0]
        def fail_lineage(*args, **kwargs):
            raise OSError('synthetic metadata persistence outage')
        monkeypatch.setattr(service.memory_dependencies, 'record', fail_lineage)
        if historical_writer_without_required_bit:
            # Reproduce the pre-fix writer's exact metadata projection. This
            # local historical fault adapter is not current acceptance and
            # does not alter source/answer text or dependency validation.
            normalize = service.store._answer_context_metadata
            monkeypatch.setattr(service.store, '_answer_context_metadata',
                lambda value: {key: item for key, item in normalize(value).items()
                               if key != 'memory_dependency_required'})
        second_model = FakeModel('继续按照已确认的4827元家庭预算安排，不执行付款。')
        second = await service.answer(request(sid, '继续这个决定，给一个不执行的安排。',
            use_memory=False, use_conversation_context=True),
            second_model, manager)
        assert second['used_memory_count'] == 0
        assert second['research_memory_required'] is True and '4827' in second_model.prompts[0]
        service.store.delete_memory(sid, original['id'], expected_revision=1)
        # UI markers must not be the only protection: reconfirmation resolves
        # the answer via ResearchStore and may happen after process restart.
        if historical_writer_without_required_bit:
            # This is the demonstrated historical bug, not an allowed current
            # behavior: the same valid count=0 history-derived answer loses its
            # ancestor when the required marker was not written atomically.
            unsafe = service.store.create_memory(sid, 'finding', 'Reconfirmed budget from the saved assistant answer.',
                                                  source_answer_id=second['answer_id'])
            assert service.store.get_memory(sid, unsafe['id'])['text']
            assert unsafe['source_answer_id'] == second['answer_id']
        else:
            with pytest.raises(ResearchStoreError):
                service.store.create_memory(sid, 'finding', 'Reconfirmed budget from the saved assistant answer.',
                                             source_answer_id=second['answer_id'])
    finally:
        await service.stop()
