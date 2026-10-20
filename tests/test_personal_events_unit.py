"""Source-bound real SQLite domain events, not paper or provider simulations."""
import hashlib
import uuid
import socket

import pytest

from docthinker.server.personal_events import PersonalEvents
from docthinker.server.research_memory_dependencies import MemoryDependencies
from docthinker.server.research_store import ResearchNotFoundError, ResearchRevisionConflictError, ResearchStore, ResearchStoreError

SID = '#00001'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('Domain-event operations must not use the network')
    monkeypatch.setattr(socket, 'getaddrinfo', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


def memory(store, text='The user confirmed the current plan.', sid=SID):
    row = store.create_memory(sid, 'decision', text)
    return row, {'kind': 'memory', 'memory_id': row['id'], 'revision': row['revision']}


def source(store, sid=SID):
    text = 'The meeting moved to Friday; execution still needs approval.'
    doc = store.accept_document(sid, 'message.txt', text.encode())
    store.claim_document(sid, doc['id'])
    store.complete_document(sid, doc['id'], [{'text': text}])
    row = store.source_chunks(sid)[0]
    return doc, {'kind': 'source', 'source_id': row['id'], 'span': [0, len(text)], 'sha256': hashlib.sha256(text.encode()).hexdigest()}


@pytest.mark.parametrize('domain,event_type,action', [
    ('work', 'decision', 'Customer rescheduled the delivery review.'),
    ('personal', 'meeting', 'The family appointment moved to Friday.'),
    ('learning', 'milestone', 'The user completed the first language lesson.'),
])
def test_three_domains_persist_with_distinct_occurrence_and_recording_times(tmp_path, domain, event_type, action):
    store = ResearchStore(tmp_path, clock=lambda: 200)
    events = PersonalEvents(store)
    _, origin = memory(store, action)
    actor = events.register_entity(SID, 'person', 'User', aliases=['Me'])
    subject = events.register_entity(SID, 'project', domain)
    row = events.record_event(SID, event_type=event_type, action=action, actor_id=actor['id'], object_id=subject['id'],
        occurred_from=100, occurred_until=110, origins=[origin], conditions=['Only under the confirmed resource limits.'])
    assert row['occurred_from'] == 100 and row['recorded_at'] == 200
    assert row['causality_verified'] is False and row['execution_authorized'] is False
    restarted = PersonalEvents(ResearchStore(tmp_path, clock=lambda: 300))
    assert restarted.get_event(SID, row['id']) == row
    assert restarted.list_events(SID, entity_id=actor['id']) == [row]
    assert restarted.resolve_entity(SID, 'person', 'me')['candidates'] == [actor]


def test_unknown_occurrence_is_not_recording_time_or_permanent_validity(tmp_path):
    store = ResearchStore(tmp_path, clock=lambda: 200)
    _, origin = memory(store)
    row = PersonalEvents(store).record_event(SID, event_type='observation', action='An observation with an unknown occurrence date.', origins=[origin])
    assert row['occurred_from'] is None and row['occurred_until'] is None and row['recorded_at'] == 200


def test_alias_ambiguity_is_preserved_without_automatic_merge(tmp_path):
    events = PersonalEvents(ResearchStore(tmp_path))
    a = events.register_entity(SID, 'person', 'Alex Chen', aliases=['Alex'])
    b = events.register_entity(SID, 'person', 'Alex Wang', aliases=['Alex'])
    found = events.resolve_entity(SID, 'person', ' ALEX ')
    assert found['status'] == 'ambiguous' and found['auto_merged'] is False
    assert {row['id'] for row in found['candidates']} == {a['id'], b['id']}
    assert events.register_entity(SID, 'person', 'alex chen')['id'] == a['id']
    different = events.register_entity(SID, 'person', 'Alex Chen', identity_key='confirmed-contact-2')
    assert different['id'] != a['id']
    assert events.resolve_entity(SID, 'project', 'Alex')['status'] == 'not_found'
    assert events.resolve_entity('#00002', 'person', 'Alex')['status'] == 'not_found'


def test_memory_correction_invalidates_event_and_explicit_revision_rebinds(tmp_path):
    now = [100]
    store = ResearchStore(tmp_path, clock=lambda: now[0])
    events = PersonalEvents(store)
    note, origin = memory(store, 'Budget is four hours.')
    first = events.record_event(SID, event_type='decision', action='Four-hour budget confirmed.', origins=[origin], occurred_from=10)
    other = events.record_event(SID, event_type='task', action='Plan the delivery within the budget.', origins=[origin])
    relation = events.relate(SID, first['id'], other['id'], 'depends_on', conditions=['Only while this budget remains current.'])
    now[0] = 200
    current = store.update_memory(SID, note['id'], expected_revision=1, text='Budget is two hours.')
    hidden = events.get_event(SID, first['id'])
    assert hidden['status'] == 'source_unavailable' and 'action' not in hidden
    revised = events.revise_event(SID, first['id'], expected_revision=1, event_type='decision',
        action='Two-hour budget confirmed.', origins=[{**origin, 'revision': current['revision']}], occurred_from=10)
    assert revised['revision'] == 2 and revised['occurred_from'] == 10 and revised['recorded_at'] == 200
    assert events.list_relations(SID)[0]['id'] == relation['id']
    assert events.list_relations(SID)[0]['status'] == 'stale_endpoint'
    with pytest.raises(ResearchRevisionConflictError):
        events.revise_event(SID, first['id'], expected_revision=1, event_type='decision', action='Stale overwrite.', origins=[origin])


def test_source_withdrawal_redacts_event_and_relationship_after_restart(tmp_path):
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    doc, origin = source(store)
    a = events.record_event(SID, event_type='meeting', action='Private meeting reschedule.', origins=[origin])
    b = events.record_event(SID, event_type='task', action='Private follow-up task.', origins=[origin])
    events.relate(SID, a['id'], b['id'], 'reported_cause', conditions=['Reported by the user; not independently established.'])
    store.delete_document(SID, doc['id'])
    reopened = PersonalEvents(ResearchStore(tmp_path))
    assert all(row['redacted'] and 'action' not in row and 'origins' not in row for row in reopened.list_events(SID))
    assert reopened.list_relations(SID)[0]['status'] == 'source_unavailable'
    assert 'conditions' not in reopened.list_relations(SID)[0]


def test_reported_cause_hypothesis_and_association_are_never_promoted(tmp_path):
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    _, origin = memory(store)
    a = events.record_event(SID, event_type='observation', action='Observed change.', origins=[origin])
    b = events.record_event(SID, event_type='observation', action='Another observed change.', origins=[origin])
    for relation in ('related_to', 'reported_cause', 'hypothesis', 'contradicts'):
        edge = events.relate(SID, a['id'], b['id'], relation, conditions=['Only if the reported context matches.'])
        assert edge['relation'] == relation and edge['causality_verified'] is False
        assert edge['conditions'] == ['Only if the reported context matches.']
    with pytest.raises(ResearchStoreError):
        events.relate(SID, a['id'], b['id'], 'verified_cause')


def test_cycles_and_known_time_contradictions_reject_without_new_edges(tmp_path):
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    _, origin = memory(store)
    a = events.record_event(SID, event_type='task', action='First task.', origins=[origin], occurred_from=10, occurred_until=20)
    b = events.record_event(SID, event_type='task', action='Second task.', origins=[origin], occurred_from=30, occurred_until=40)
    events.relate(SID, a['id'], b['id'], 'before')
    with pytest.raises(ResearchStoreError):
        events.relate(SID, b['id'], a['id'], 'before')
    events.relate(SID, a['id'], b['id'], 'depends_on')
    with pytest.raises(ResearchStoreError, match='Cyclic'):
        events.relate(SID, b['id'], a['id'], 'depends_on')
    assert len(events.list_relations(SID)) == 2


def test_source_origin_checks_current_scope_exact_span_and_hash(tmp_path):
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    _, origin = source(store)
    for bad in ({**origin, 'sha256': '0' * 64}, {**origin, 'span': [0, 7999]}, {**origin, 'span': [False, 3]}):
        with pytest.raises(ResearchStoreError):
            events.record_event(SID, event_type='observation', action='Must not forge provenance.', origins=[bad])
    with pytest.raises(ResearchStoreError):
        events.record_event('#00002', event_type='observation', action='Cannot reuse another project source.', origins=[origin])
    assert events.list_events(SID) == []


def test_private_worker_and_project_boundaries_apply_to_entities_and_events(tmp_path):
    one, two = tmp_path / 'one', tmp_path / 'two'
    one.mkdir(); two.mkdir()
    store = ResearchStore(one)
    events, other = PersonalEvents(store), PersonalEvents(ResearchStore(two))
    _, origin = memory(store)
    person = events.register_entity(SID, 'person', 'Private Person')
    row = events.record_event(SID, event_type='task', action='Private task.', actor_id=person['id'], origins=[origin])
    for surface, sid in ((events, '#00002'), (other, SID)):
        assert surface.list_events(sid) == []
        assert surface.resolve_entity(sid, 'person', 'Private Person')['status'] == 'not_found'
        with pytest.raises(ResearchNotFoundError):
            surface.get_event(sid, row['id'])


def test_transitive_memory_withdrawal_invalidates_event_without_model_call(tmp_path):
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    note, _ = memory(store)
    aid = str(uuid.uuid4())
    store.record_answer(SID, aid, [], {}, 'memory')
    MemoryDependencies(store).record(SID, aid, [(note['id'], note['revision'], None)])
    derived = store.create_memory(SID, 'finding', 'An assistant-derived private fact.', source_answer_id=aid)
    event = events.record_event(SID, event_type='observation', action='A source-bound derived observation.',
        origins=[{'kind': 'memory', 'memory_id': derived['id'], 'revision': derived['revision']}])
    store.delete_memory(SID, note['id'], expected_revision=1)
    assert events.get_event(SID, event['id'])['status'] == 'source_unavailable'


def test_empty_read_does_not_create_store_and_input_limits_are_closed(tmp_path):
    events = PersonalEvents(ResearchStore(tmp_path))
    assert events.list_events(SID) == [] and list(tmp_path.iterdir()) == []
    for bad in ([], {}, True, 'causality'):
        with pytest.raises(ResearchStoreError):
            events.record_event(SID, event_type=bad, action='Invalid type.', origins=[])


def test_explicit_snapshots_and_history_never_rebind_old_coordinates(tmp_path):
    now = [100]
    store = ResearchStore(tmp_path, clock=lambda: now[0])
    events = PersonalEvents(store)
    note, origin = memory(store)
    row = events.record_event(SID, event_type='decision', action='Original user decision.', origins=[origin])
    assert events.snapshots(SID, [row['id']])[row['id']]['revision'] == 1
    now[0] = 200
    changed = events.revise_event(SID, row['id'], expected_revision=1, event_type='decision', action='Corrected user decision.', origins=[origin])
    assert events.snapshots(SID, [row['id']])[row['id']]['revision'] == changed['revision'] == 2
    assert [item['historical'] for item in events.history(SID, row['id'])] == [False, True]
    with pytest.raises(ResearchNotFoundError):
        events.snapshots('#00002', [row['id']])
    store.delete_memory(SID, note['id'], expected_revision=1)
    assert all(item['redacted'] and 'action' not in item for item in events.history(SID, row['id']))
    assert events.snapshots(SID, [row['id']])[row['id']]['status'] == 'source_unavailable'


def test_stale_relation_does_not_fabricate_cycle_in_revised_event_graph(tmp_path):
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    _, origin = memory(store)
    first = events.record_event(SID, event_type='task', action='First plan.', origins=[origin])
    second = events.record_event(SID, event_type='task', action='Second plan.', origins=[origin])
    events.relate(SID, first['id'], second['id'], 'depends_on')
    events.revise_event(SID, first['id'], expected_revision=1, event_type='task', action='Explicitly corrected first plan.', origins=[origin])
    edge = events.relate(SID, second['id'], first['id'], 'depends_on')
    assert edge['status'] == 'active' and edge['target_revision'] == 2
