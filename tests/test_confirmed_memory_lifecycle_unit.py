"""Real SQLite confirmed-memory lifecycle, without providers or model fixtures."""
import uuid
import sqlite3

import pytest

from docthinker.server.research_store import (
    ResearchNotFoundError, ResearchRevisionConflictError, ResearchStore, ResearchStoreError,
)


def ready(store, sid='#00001'):
    doc = store.accept_document(sid, 'meeting.txt', b'Customer budget is four staff hours.')
    store.claim_document(sid, doc['id'])
    store.complete_document(sid, doc['id'], [{'text': 'Customer budget is four staff hours.'}])
    return doc, store.source_chunks(sid)[0]


def answer(store, sid='#00001', refs=None):
    identifier = str(uuid.uuid4())
    store.record_answer(sid, identifier, refs or [], {}, 'general')
    return identifier


@pytest.mark.parametrize('foreign', [False, True])
def test_answer_provenance_must_exist_in_the_same_private_project(tmp_path, foreign):
    store = ResearchStore(tmp_path)
    identifier = answer(store, '#00002') if foreign else str(uuid.uuid4())
    with pytest.raises(ResearchStoreError, match='answer'):
        store.create_memory('#00001', 'decision', 'Budget is four hours.', source_answer_id=identifier)
    assert store.list_memories('#00001') == []


def test_answer_only_provenance_inherits_document_dependencies_and_revocation(tmp_path):
    store = ResearchStore(tmp_path, clock=lambda: 100)
    doc, chunk = ready(store)
    identifier = answer(store, refs=[{'id': chunk['id'], 'label': 'S1'}])
    memory = store.create_memory('#00001', 'finding', 'Private remembered budget.', source_answer_id=identifier)
    assert memory['source_chunk_ids'] == [chunk['id']]
    store.archive_memory('#00001', memory['id'], expected_revision=1)
    store.delete_document('#00001', doc['id'])
    restarted = ResearchStore(tmp_path, clock=lambda: 101)
    for item in (restarted.get_memory('#00001', memory['id']),
                 restarted.list_visible_memories('#00001')[0],
                 restarted.list_memories('#00001', include_deleted=True)[0]):
        assert item['text'] == '' and item['source_chunk_ids'] == []
        assert item['source_answer_id'] is None and item['redacted'] is True
    assert restarted.search_memories('#00001', 'remembered budget') == []
    assert all(item['text'] == '' for item in restarted.memory_history('#00001', memory['id'])['items'])
    with pytest.raises(ResearchRevisionConflictError):
        restarted.archive_memory('#00001', memory['id'], expected_revision=3, archived=False)


def test_correction_can_explicitly_detach_old_answer_without_reviving_old_fact(tmp_path):
    now = [100]
    store = ResearchStore(tmp_path, clock=lambda: now[0])
    doc, chunk = ready(store)
    identifier = answer(store, refs=[{'id': chunk['id'], 'label': 'S1'}])
    old = store.create_memory('#00001', 'decision', 'Budget is four staff hours.', source_answer_id=identifier)
    now[0] = 200
    updated = store.update_memory('#00001', old['id'], expected_revision=1,
        text='User corrected budget to two staff hours.', source_chunk_ids=[], source_answer_id=None)
    assert updated['source_answer_id'] is None and updated['source_chunk_ids'] == []
    store.delete_document('#00001', doc['id'])
    reopened = ResearchStore(tmp_path, clock=lambda: 201)
    assert [item['text'] for item in reopened.search_memories('#00001', 'budget')] == [updated['text']]
    assert reopened.search_memories('#00001', 'budget', as_of=150) == []
    history = reopened.memory_history('#00001', old['id'])['items']
    assert history[0]['revision'] == 2 and history[0]['text'] == updated['text']
    assert history[1]['redacted'] and history[1]['text'] == ''
    with pytest.raises(ResearchRevisionConflictError):
        reopened.update_memory('#00001', old['id'], expected_revision=1, text='Stale four-hour overwrite.')
    assert reopened.search_memories('#00002', 'budget') == []


def test_omitted_answer_preserves_provenance_and_cannot_drop_dependencies(tmp_path):
    store = ResearchStore(tmp_path)
    _, chunk = ready(store)
    identifier = answer(store, refs=[{'id': chunk['id'], 'label': 'S1'}])
    old = store.create_memory('#00001', 'finding', 'Budget observation.', source_answer_id=identifier)
    updated = store.update_memory('#00001', old['id'], expected_revision=1,
        text='Corrected budget observation.', source_chunk_ids=[])
    assert updated['source_answer_id'] == identifier
    assert updated['source_chunk_ids'] == [chunk['id']]


def test_revoked_answer_dependencies_cannot_create_a_new_memory(tmp_path):
    store = ResearchStore(tmp_path)
    doc, chunk = ready(store)
    identifier = answer(store, refs=[{'id': chunk['id'], 'label': 'S1'}])
    store.delete_document('#00001', doc['id'])
    with pytest.raises(ResearchStoreError):
        store.create_memory('#00001', 'finding', 'Cannot revive removed source.', source_answer_id=identifier)
    assert store.list_memories('#00001') == []


def test_other_worker_cannot_resolve_the_same_answer_identifier(tmp_path):
    first, second = tmp_path / 'first-user', tmp_path / 'second-user'
    first.mkdir(); second.mkdir()
    one, two = ResearchStore(first), ResearchStore(second)
    identifier = answer(one)
    memory = one.create_memory('#00001', 'preference', 'My private preference.', source_answer_id=identifier)
    with pytest.raises(ResearchStoreError):
        two.create_memory('#00001', 'preference', 'Not my provenance.', source_answer_id=identifier)
    with pytest.raises(ResearchNotFoundError):
        two.get_memory('#00001', memory['id'])
    assert two.search_memories('#00001', 'private preference') == []


def test_revocation_is_redacted_on_direct_getter_and_visible_list(tmp_path):
    store = ResearchStore(tmp_path)
    doc, chunk = ready(store)
    memory = store.create_memory('#00001', 'finding', 'Sensitive derived statement.', source_chunk_ids=[chunk['id']])
    store.delete_document('#00001', doc['id'])
    for item in (store.get_memory('#00001', memory['id']), store.list_visible_memories('#00001')[0]):
        assert item['status'] == 'invalidated'
        assert item['text'] == '' and item['source_chunk_ids'] == [] and item['redacted']


def test_legacy_answer_only_archive_cannot_resurrect_after_source_withdrawal(tmp_path):
    store = ResearchStore(tmp_path, clock=lambda: 100)
    doc, chunk = ready(store)
    identifier = answer(store, refs=[{'id': chunk['id'], 'label': 'S1'}])
    memory = store.create_memory('#00001', 'finding', 'Legacy private observation.', source_answer_id=identifier)
    archived = store.archive_memory('#00001', memory['id'], expected_revision=1)
    # Reproduce the actual pre-fix answer-only representation, not a synthetic
    # provider result: it lacks the materialized chunk dependency entirely.
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE research_memories SET source_chunk_ids='[]' WHERE memory_id=?", (memory['id'],))
        db.execute("UPDATE research_memory_versions SET source_chunk_ids='[]' WHERE memory_id=?", (memory['id'],))
        db.execute('DELETE FROM research_memory_sources WHERE memory_id=?', (memory['id'],))
    store.delete_document('#00001', doc['id'])
    reopened = ResearchStore(tmp_path, clock=lambda: 101)
    assert reopened.search_memories('#00001', 'legacy private') == []
    assert reopened.search_memories('#00001', 'legacy private', as_of=100) == []
    assert reopened.get_memory('#00001', memory['id'])['redacted']
    assert reopened.list_memories('#00001', include_deleted=True)[0]['text'] == ''
    assert reopened.page_memories('#00001', include_archived=True)['items'][0]['text'] == ''
    assert all(item['text'] == '' for item in reopened.memory_history('#00001', memory['id'])['items'])
    with pytest.raises(ResearchRevisionConflictError):
        reopened.archive_memory('#00001', memory['id'], expected_revision=archived['revision'], archived=False)
    with reopened._read() as db:
        row = reopened._required_memory(db, '#00001', memory['id'])
        assert reopened._memory_sources_live(db, row) is False


def test_two_hop_memory_only_answer_cannot_reconfirm_forgotten_ancestor(tmp_path):
    from docthinker.server.research_memory_dependencies import MemoryDependencies
    store = ResearchStore(tmp_path, clock=lambda: 100)
    original = store.create_memory('#00001', 'decision', 'Private budget is four hours.')
    derived_answer = answer(store)
    MemoryDependencies(store).record('#00001', derived_answer, [(original['id'], 1, None)])
    derived_note = store.create_memory('#00001', 'decision', 'Four-hour budget copied from assistant.',
                                       source_answer_id=derived_answer)
    assert store.search_memories('#00001', 'budget')
    store.delete_memory('#00001', original['id'], expected_revision=1)
    reopened = ResearchStore(tmp_path, clock=lambda: 101)
    assert reopened.get_memory('#00001', derived_note['id'])['redacted']
    assert reopened.search_memories('#00001', 'budget') == []
    with pytest.raises(ResearchStoreError):
        reopened.create_memory('#00001', 'decision', 'Reconfirm forgotten assistant fact.', source_answer_id=derived_answer)
