"""Actual legacy memory scope boundaries, not invented cross-chat endpoints.

User/project scopes are shared within one private backend database. Session
scope is not. These tests do not claim that a single database is multi-tenant,
or that the invitation research store implicitly shares its project memories.
"""
import importlib
import socket
import sqlite3

import pytest

from docthinker.memory_core import AgentMemoryBackends, AgentMemoryCore, MemoryPolicy, SQLiteLongHorizonBackend


@pytest.fixture(autouse=True)
def no_network_or_model_discovery(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('Network is not part of a memory scope test')
    monkeypatch.setattr(socket, 'create_connection', denied)
    monkeypatch.setattr(socket, 'getaddrinfo', denied)
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    core = importlib.import_module('docthinker.memory_core.core')
    monkeypatch.setattr(core, '_extract_entities_from_text', lambda *_args, **_kwargs: ['meeting', 'preference'])


def remember(backend, scope='user', sid='chat-one', summary=None):
    item = backend.consolidate(sid, 'Please remember my durable meeting preference.',
        summary or 'I prefer morning meetings because afternoons are reserved for focused engineering work.',
        concepts=['meeting', 'preference'], scope=scope, timestamp=100.0)
    assert item is not None
    return item


def recall(backend, sid='chat-two', scopes=('session', 'project', 'user')):
    return backend.retrieve(sid, 'meeting preference', scopes=scopes, top_k=10, min_confidence=0.35)


@pytest.mark.parametrize('scope', ['user', 'project'])
def test_real_cross_chat_memory_reopens_and_explicit_correction_supersedes_active_text(tmp_path, scope):
    path = tmp_path / 'one-private-principal.db'
    backend = SQLiteLongHorizonBackend(path)
    original = remember(backend, scope)
    assert [item['id'] for item in recall(backend, scopes=(scope,))] == [original['id']]
    correction = 'I prefer afternoon meetings because the morning is now reserved for focused engineering work.'
    changed = backend.update_insight(original['id'], {'summary': correction}, 'chat-two')
    assert changed['version'] == 2 and changed['summary'] == correction
    restarted = SQLiteLongHorizonBackend(path)
    current = recall(restarted, 'chat-three', (scope,))
    assert [(item['id'], item['version'], item['summary']) for item in current] == [(original['id'], 2, correction)]
    revisions = restarted.list_revisions(original['id'], 'chat-three')
    assert [item['action'] for item in revisions] == ['update', 'create']
    assert revisions[1]['snapshot']['summary'] == original['summary']
    # The old statement survives only as an explicit revision audit, not as
    # a second active memory or an authoritative current fact.
    assert restarted.list_insights('chat-one', scope=scope)[0]['summary'] == correction


def test_session_memory_cannot_be_read_edited_or_deleted_from_another_chat(tmp_path):
    backend = SQLiteLongHorizonBackend(tmp_path / 'private.db')
    item = remember(backend, scope='session')
    assert recall(backend, scopes=('session',)) == []
    assert backend.list_insights('chat-two', scope='session') == []
    assert backend.update_insight(item['id'], {'summary': 'Wrong chat mutation.'}, 'chat-two') is None
    assert backend.delete_insight(item['id'], 'chat-two') is False
    assert backend.list_revisions(item['id'], 'chat-two') == []
    assert recall(backend, 'chat-one', ('session',))[0]['id'] == item['id']


def test_scope_selection_does_not_promote_session_data_to_shared_memory(tmp_path):
    backend = SQLiteLongHorizonBackend(tmp_path / 'private.db')
    by_scope = {scope: remember(backend, scope) for scope in ('session', 'project', 'user')}
    assert len({item['id'] for item in by_scope.values()}) == 3
    for scope in ('user', 'project'):
        assert [item['id'] for item in recall(backend, scopes=(scope,))] == [by_scope[scope]['id']]
    assert recall(backend, scopes=('session',)) == []


def test_separate_private_worker_databases_are_the_principal_boundary(tmp_path):
    first = SQLiteLongHorizonBackend(tmp_path / 'worker-one' / 'memory.db')
    second = SQLiteLongHorizonBackend(tmp_path / 'worker-two' / 'memory.db')
    item = remember(first)
    revision = first.list_revisions(item['id'], 'chat-one')[0]
    assert recall(second) == []
    assert second.list_insights('chat-one') == []
    assert second.list_revisions(item['id'], 'chat-one') == []
    assert second.update_insight(item['id'], {'summary': 'Another principal.'}, 'chat-one') is None
    assert second.delete_insight(item['id'], 'chat-one') is False
    assert second.restore_revision(item['id'], revision['revision_id'], 'chat-one') is None
    assert recall(first)[0]['summary'] == item['summary']


def test_explicit_delete_stays_out_of_cross_chat_recall_after_restart(tmp_path):
    path = tmp_path / 'private.db'
    backend = SQLiteLongHorizonBackend(path)
    item = remember(backend)
    assert backend.delete_insight(item['id'], 'chat-two') is True
    restarted = SQLiteLongHorizonBackend(path)
    assert recall(restarted, 'chat-three') == []
    assert restarted.list_insights('chat-three', scope='user') == []
    # This backend intentionally supports explicit revision restoration. Its
    # retained audit is not a promise of physical erasure or automatic reuse.
    assert restarted.list_revisions(item['id'], 'chat-three')[0]['action'] == 'delete'


async def test_agent_core_explicit_user_write_cross_chat_and_skip_memory_control(tmp_path):
    backend = SQLiteLongHorizonBackend(tmp_path / 'private.db')
    core = AgentMemoryCore(backends=AgentMemoryBackends(long_horizon=backend),
        policy=MemoryPolicy(enabled_layers=('long_horizon',)))
    result = await core.after_response(session_id='chat-one',
        question='Please remember my durable meeting preference.',
        answer='I prefer morning meetings because afternoons are reserved for focused engineering work.',
        write_scope='user')
    assert result['long_horizon_insight_added'] is True
    bundle = await core.recall(session_id='chat-two', query='meeting preference', enable_cognition=False)
    assert bundle.long_horizon_matches[0]['id'] == result['long_horizon_insight']['id']
    assert 'morning meetings' in bundle.retrieval_instruction
    skipped = await core.recall(session_id='chat-two', query='meeting preference', base_instruction='Current instruction.', skip_memory=True)
    assert skipped.long_horizon_matches == [] and skipped.retrieval_instruction == 'Current instruction.'


async def test_agent_core_default_write_remains_session_scoped(tmp_path):
    backend = SQLiteLongHorizonBackend(tmp_path / 'private.db')
    core = AgentMemoryCore(backends=AgentMemoryBackends(long_horizon=backend),
        policy=MemoryPolicy(enabled_layers=('long_horizon',)))
    result = await core.after_response(session_id='chat-one', question='Please remember this meeting rule.',
        answer='Meetings should preserve focused engineering time and use a clear agenda for every planned discussion.')
    assert result['long_horizon_insight']['scope'] == 'session'
    assert (await core.recall(session_id='chat-two', query='meeting agenda', enable_cognition=False)).long_horizon_matches == []


def test_actual_main_query_factory_uses_shared_desktop_scope_but_restricts_mobile(monkeypatch, tmp_path):
    core_module = importlib.import_module('docthinker.memory_core.core')
    query = importlib.import_module('docthinker.server.routers.query')
    backend = SQLiteLongHorizonBackend(tmp_path / 'private.db')
    monkeypatch.setattr(core_module, 'get_default_long_horizon_backend', lambda: backend)
    monkeypatch.setattr(query.state, 'mobile_safe_mode', False)
    desktop = query._get_agent_memory_core()
    assert desktop.backends.long_horizon is backend
    assert tuple(desktop.policy.long_horizon_scopes) == ('session', 'project', 'user')
    assert desktop.policy.long_horizon_write_scope == 'session'
    monkeypatch.setattr(query.state, 'mobile_safe_mode', True)
    monkeypatch.setattr(query.state, 'long_horizon_backend', backend)
    mobile = query._get_agent_memory_core()
    assert mobile.backends.long_horizon is backend
    assert tuple(mobile.policy.long_horizon_scopes) == ('session',)
    assert mobile.policy.allow_memory_writes is False


@pytest.mark.parametrize('edit', ['update', 'delete'])
def test_stale_assistant_text_cannot_recreate_explicitly_retired_content(tmp_path, edit):
    path = tmp_path / 'private.db'
    backend = SQLiteLongHorizonBackend(path)
    item = remember(backend)
    if edit == 'update':
        backend.update_insight(item['id'], {'summary': 'I prefer afternoon meetings because mornings are reserved for focused engineering work.'}, 'chat-two')
    else:
        backend.delete_insight(item['id'], 'chat-two')
    reopened = SQLiteLongHorizonBackend(path)
    result = reopened.consolidate('chat-three', 'What is my meeting preference?', item['summary'],
        concepts=['meeting', 'preference'], scope='user', timestamp=200)
    assert result is None
    assert reopened.last_write_decision()['reason'] == 'explicit_memory_content_retired'
    assert all(row['summary'] != item['summary'] for row in reopened.list_insights('chat-four', scope='user'))
    assert reopened.consolidate('chat-four', 'What is my meeting preference?', item['summary'].upper().replace(' ', ' \n '),
        concepts=['meeting'], scope='user', timestamp=201) is None


def test_retirement_scope_and_worker_boundary_do_not_block_independent_notes(tmp_path):
    backend = SQLiteLongHorizonBackend(tmp_path / 'first.db')
    item = remember(backend, scope='session')
    backend.delete_insight(item['id'], 'chat-one')
    assert remember(backend, scope='session', sid='chat-two')['summary'] == item['summary']
    assert remember(backend, scope='user')['summary'] == item['summary']
    other = SQLiteLongHorizonBackend(tmp_path / 'second.db')
    assert remember(other, scope='session')['summary'] == item['summary']


def test_explicit_restore_and_edit_can_change_back_without_changing_recurrence_confidence(tmp_path):
    backend = SQLiteLongHorizonBackend(tmp_path / 'private.db')
    item = remember(backend)
    original_revision = backend.list_revisions(item['id'], 'chat-one')[0]
    backend.update_insight(item['id'], {'summary': 'Afternoon meeting preference is the newly confirmed choice.', 'confidence': 0.42}, 'chat-two')
    restored = backend.restore_revision(item['id'], original_revision['revision_id'], 'chat-three')
    assert restored['summary'] == item['summary']
    confidence = restored['confidence']
    repeat = remember(backend, sid='chat-four')
    assert repeat['id'] == item['id'] and repeat['confidence'] == confidence
    # A later explicit update is equally capable of choosing a retired value;
    # automatic consolidation alone is not granted that authority.
    backend.update_insight(item['id'], {'summary': 'A different durable meeting preference for focused work.'}, 'chat-four')
    changed_back = backend.update_insight(item['id'], {'summary': item['summary']}, 'chat-five')
    assert changed_back['summary'] == item['summary']
    assert remember(backend, sid='chat-six')['id'] == item['id']


def test_legacy_revision_backfill_is_once_only_and_stores_fingerprints_not_text(tmp_path, monkeypatch):
    path = tmp_path / 'private.db'
    backend = SQLiteLongHorizonBackend(path)
    item = remember(backend)
    backend.update_insight(item['id'], {'summary': 'Afternoon meetings now preserve the confirmed focus schedule.'}, 'chat-two')
    # Simulate an existing pre-guard database containing genuine edit history.
    with sqlite3.connect(path) as db:
        db.execute('DROP TABLE IF EXISTS long_horizon_retired_summaries')
        db.execute("DELETE FROM long_horizon_metadata WHERE key='retired_summary_fingerprints_v1'")
    reopened = SQLiteLongHorizonBackend(path)
    queries, connect = [], reopened._connect
    def traced_connect():
        connection = connect()
        connection.set_trace_callback(queries.append)
        return connection
    monkeypatch.setattr(reopened, '_connect', traced_connect)
    assert reopened.consolidate('chat-three', 'What is my meeting preference?', item['summary'],
        concepts=['meeting'], scope='user', timestamp=200) is None
    assert not any('SELECT' in sql.upper() and 'long_horizon_revisions' in sql for sql in queries)
    with sqlite3.connect(path) as db:
        columns = {row[1] for row in db.execute('PRAGMA table_info(long_horizon_retired_summaries)')}
        assert columns == {'scope_key', 'summary_sha256', 'retired_at'}
        assert db.execute('SELECT length(summary_sha256) FROM long_horizon_retired_summaries').fetchone()[0] == 64
        plan = ' '.join(row[3] for row in db.execute(
            "EXPLAIN QUERY PLAN SELECT id,summary FROM long_horizon_memories WHERE summary_sha256=''") )
        assert 'idx_lh_unhashed_summary' in plan


def test_existing_old_active_duplicate_is_not_recalled_reinforced_or_used_by_cognition(tmp_path):
    backend = SQLiteLongHorizonBackend(tmp_path / 'private.db')
    first = remember(backend)
    second = remember(backend, summary='A second durable meeting preference awaiting a user edit.')
    backend.update_insight(second['id'], {'summary': first['summary']}, 'chat-two')
    cognition = backend.create_cognition('chat-two', 'Morning meetings preserve focused engineering work.',
        evidence_memory_ids=[second['id']], scope='user')
    backend.update_insight(first['id'], {'summary': 'I prefer afternoon meetings because mornings now protect focused engineering work.'}, 'chat-three')
    assert second['id'] not in {row['id'] for row in recall(backend)}
    assert second['id'] not in {row['id'] for row in backend.list_insights('chat-four')}
    assert backend.consolidate('chat-four', 'What is my meeting preference?', first['summary'],
        concepts=['meeting'], scope='user', timestamp=200) is None
    assert cognition['id'] not in {row['id'] for row in backend.retrieve_cognitions('chat-four', 'meeting preference',
        scopes=('user',), top_k=10, min_confidence=0.35)}
    # No audit row is physically removed or rewritten to a made-up fact.
    assert backend.list_revisions(second['id'], 'chat-four')[0]['snapshot']['summary'] == first['summary']


@pytest.mark.parametrize('action', ['restore', 'update_back'])
def test_backfill_replays_later_explicit_authorization_in_revision_order(tmp_path, action):
    path = tmp_path / 'private.db'
    backend = SQLiteLongHorizonBackend(path)
    item = remember(backend)
    original_revision = backend.list_revisions(item['id'], 'chat-one')[0]
    if action == 'restore':
        backend.delete_insight(item['id'], 'chat-two')
        backend.restore_revision(item['id'], original_revision['revision_id'], 'chat-three')
    else:
        backend.update_insight(item['id'], {'summary': 'Afternoon meetings are the temporarily confirmed preference.'}, 'chat-two')
        backend.update_insight(item['id'], {'summary': item['summary']}, 'chat-three')
    with sqlite3.connect(path) as db:
        db.execute('DROP TABLE long_horizon_retired_summaries')
        db.execute("DELETE FROM long_horizon_metadata WHERE key='retired_summary_fingerprints_v1'")
    reopened = SQLiteLongHorizonBackend(path)
    assert recall(reopened)[0]['id'] == item['id']
    before = reopened.list_insights('chat-four', scope='user')[0]
    repeated = remember(reopened, sid='chat-four')
    assert repeated['id'] == item['id']
    assert repeated['confidence'] == before['confidence']
    assert repeated['use_count'] == before['use_count'] + 1
