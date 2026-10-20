"""Indexed current event contexts and reference-only lineage, without providers."""
import copy
import json
import uuid

import pytest

from docthinker.server.personal_events import PersonalEvents
from docthinker.server.research_event_dependencies import EventDependencies, event_bindings_live
from docthinker.server.research_memory_dependencies import source_answer_memory_live
from docthinker.server.research_answer_dependencies import AnswerDependencies
from docthinker.server.research_store import ResearchStore, ResearchStoreError
from tests.test_personal_events_unit import memory, source, SID, no_network


def assert_context_budgets(packet):
    visible = {'events': packet['events'], 'relations': packet['relations']}
    evidence_bytes = len(json.dumps(visible, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())
    binding_bytes = len(json.dumps(packet['bindings'], ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())
    assert packet['trace']['evidence_bytes'] == evidence_bytes <= 6000
    assert packet['trace']['binding_bytes'] == binding_bytes <= 8192
    assert len(packet['events']) <= 8 and len(packet['relations']) <= 12


def event(store, *, action='Customer delivery moved to Friday.', origin=None, **fields):
    _, default = memory(store, action)
    return PersonalEvents(store).record_event(SID, event_type='decision', action=action,
        origins=[default if origin is None else origin], **fields)


def answer(store, bindings=None, *, parents=None):
    identifier = str(uuid.uuid4())
    metadata = EventDependencies.context_metadata(bindings) if bindings is not None else {}
    if parents:
        metadata['answer_dependency_required'] = True
    store.record_answer(SID, identifier, [], {'valid': True}, 'general', context_metadata=metadata)
    # Real parent traversal intentionally requires an existing proof record.
    # This fixture supplies only the table-level reference used by that guard;
    # it does not claim this synthetic answer has scientific evidence.
    with store._write() as db:
        db.execute('CREATE TABLE IF NOT EXISTS research_answer_evidence '
                   '(answer_id TEXT PRIMARY KEY,session_id TEXT NOT NULL,payload TEXT NOT NULL)')
        db.execute('INSERT INTO research_answer_evidence VALUES(?,?,?)', (identifier, SID, '{}'))
    if bindings is not None:
        EventDependencies(store).record(SID, identifier, bindings)
    if parents:
        AnswerDependencies(store).record(SID, identifier, parents)
    return identifier


def test_old_relevant_event_is_recalled_from_full_current_index_not_the_last_hundred(tmp_path):
    now = [100]
    store = ResearchStore(tmp_path, clock=lambda: now[0])
    events = PersonalEvents(store)
    old = event(store, action='Zephyr renewal decision: retain the original privacy restriction.')
    _, origin = memory(store)
    for index in range(125):
        now[0] += 1
        events.record_event(SID, event_type='observation', action=f'Unrelated household observation {index}', origins=[origin])
    assert old['id'] not in {row['id'] for row in events.list_events(SID, limit=100)}
    packet = events.retrieve_context(SID, 'What was the Zephyr renewal decision?')
    assert old['id'] in {row['id'] for row in packet['events']}
    assert EventDependencies(store).live(SID, packet['bindings'])


def test_index_upgrade_backfills_once_and_revision_replaces_search_terms(tmp_path):
    store = ResearchStore(tmp_path)
    row = event(store, action='Old Zephyr launch plan.')
    with store._write() as db:
        db.execute('DROP TABLE personal_event_search')
    events = PersonalEvents(ResearchStore(tmp_path))
    assert events.retrieve_context(SID, 'Zephyr')['events'][0]['id'] == row['id']
    events.revise_event(SID, row['id'], expected_revision=1, event_type='decision',
        action='New Borealis launch plan.', origins=row['origins'])
    assert not events.retrieve_context(SID, 'Zephyr')['events']
    assert events.retrieve_context(SID, 'Borealis')['events'][0]['revision'] == 2


def test_seed_then_adjacency_preserves_direction_conditions_types_and_identity_ids(tmp_path):
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    first = event(store, action='Zephyr delivery delay was reported.')
    second = event(store, action='Prepare the customer follow-up draft.')
    relation = events.relate(SID, first['id'], second['id'], 'reported_cause',
        conditions=['Reported by the user; not independently verified.'])
    packet = events.retrieve_context(SID, 'Zephyr')
    assert {row['id'] for row in packet['events']} == {first['id'], second['id']}
    edge = next(row for row in packet['relations'] if row['id'] == relation['id'])
    assert (edge['source_id'], edge['target_id'], edge['relation']) == (first['id'], second['id'], 'reported_cause')
    assert edge['conditions'] == relation['conditions']
    assert packet['trace']['causality_verified'] is False
    assert EventDependencies(store).live(SID, packet['bindings'])


def test_all_origin_scopes_must_be_allowed_and_explicit_event_ids_do_not_expand(tmp_path):
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    note, mem = memory(store)
    _, src = source(store)
    a = event(store, action='Zephyr memory-only decision.', origin=mem)
    b = event(store, action='Zephyr source-only decision.', origin=src)
    mixed = events.record_event(SID, event_type='decision', action='Zephyr mixed-origin decision.', origins=[mem, src])
    events.relate(SID, a['id'], b['id'], 'related_to')
    personal = events.retrieve_context(SID, 'Zephyr', allowed_source_ids=[], allowed_memory_ids=[note['id']])
    assert {row['id'] for row in personal['events']} == {a['id']}
    assert personal['relations'] == []
    docs = events.retrieve_context(SID, 'Zephyr', allowed_source_ids=[src['source_id']], allowed_memory_ids=[])
    assert {row['id'] for row in docs['events']} == {b['id']}
    assert not events.retrieve_context(SID, 'Zephyr', allowed_source_ids=[], allowed_memory_ids=[])['events']
    explicit = events.retrieve_context(SID, 'Zephyr', event_ids=[a['id']])
    assert {row['id'] for row in explicit['events']} == {a['id']}
    assert explicit['relations'] == []
    assert not events.retrieve_context('#00002', 'Zephyr')['events']
    assert mixed['id'] not in {row['id'] for row in personal['events']}


@pytest.mark.parametrize('origin_kind', ['source', 'memory'])
def test_scope_filter_precedes_candidate_limit_despite_forty_higher_ranked_events(tmp_path, origin_kind):
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    if origin_kind == 'source':
        _, allowed_origin = source(store)
        hidden_doc = store.accept_document(SID, 'other.txt', b'Unselected source record.')
        store.claim_document(SID, hidden_doc['id'])
        store.complete_document(SID, hidden_doc['id'], [{'text': 'Unselected source record.'}])
        hidden_chunk = next(row for row in store.source_chunks(SID) if row['document_id'] == hidden_doc['id'])
        import hashlib
        hidden_origin = {'kind': 'source', 'source_id': hidden_chunk['id'], 'span': [0, len(hidden_chunk['text'])],
            'sha256': hashlib.sha256(hidden_chunk['text'].encode()).hexdigest()}
        options = {'allowed_source_ids': [allowed_origin['source_id']], 'allowed_memory_ids': []}
    else:
        _, allowed_origin = memory(store, 'Selected current instruction.')
        _, hidden_origin = memory(store, 'Another current instruction outside this task.')
        options = {'allowed_source_ids': [], 'allowed_memory_ids': [allowed_origin['memory_id']]}
    valid = events.record_event(SID, event_type='decision',
        action='Zephyr ' + ('less specific surrounding context ' * 24), origins=[allowed_origin])
    for index in range(40):
        events.record_event(SID, event_type='observation', action='Zephyr Zephyr Zephyr.', origins=[hidden_origin])
    mixed = events.record_event(SID, event_type='observation', action='Zephyr Zephyr Zephyr.',
        origins=[allowed_origin, hidden_origin])
    unrestricted = events.retrieve_context(SID, 'Zephyr')
    assert valid['id'] not in {row['id'] for row in unrestricted['events']}
    scoped = events.retrieve_context(SID, 'Zephyr', **options)
    assert [row['id'] for row in scoped['events']] == [valid['id']]
    assert scoped['trace']['seed_candidates'] == 1
    explicit = events.retrieve_context(SID, 'Zephyr', event_ids=[valid['id'], mixed['id']], **options)
    assert [row['id'] for row in explicit['events']] == [valid['id']]


def test_exact_revision_and_hash_bindings_fail_after_revision_or_silent_row_tampering(tmp_path):
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    a = event(store, action='Zephyr delivery decision.')
    packet = events.retrieve_context(SID, 'Zephyr')
    events.revise_event(SID, a['id'], expected_revision=1, event_type='decision',
        action='Zephyr delivery decision was corrected.', origins=a['origins'])
    assert not EventDependencies(store).live(SID, packet['bindings'])
    current = events.retrieve_context(SID, 'Zephyr')
    with store._write() as db:
        raw = json.loads(db.execute('SELECT payload FROM personal_events WHERE event_id=?', (a['id'],)).fetchone()[0])
        raw['conditions'] = ['A silently changed restriction.']
        db.execute('UPDATE personal_events SET payload=? WHERE event_id=?', (json.dumps(raw), a['id']))
    assert not EventDependencies(store).live(SID, current['bindings'])


@pytest.mark.parametrize('change', ['memory_revision', 'memory_delete', 'memory_archive', 'source_delete'])
def test_current_origin_changes_hide_context_and_invalidate_exact_saved_bindings(tmp_path, change):
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    if change == 'source_delete':
        original, origin = source(store)
    else:
        original, origin = memory(store)
    event(store, action='Zephyr private planning restriction.', origin=origin)
    packet = events.retrieve_context(SID, 'Zephyr')
    assert packet['events']
    if change == 'memory_revision':
        store.update_memory(SID, original['id'], expected_revision=1, text='Corrected restriction.')
    elif change == 'memory_delete':
        store.delete_memory(SID, original['id'], expected_revision=1)
    elif change == 'memory_archive':
        store.archive_memory(SID, original['id'], expected_revision=1, archived=True)
    else:
        store.delete_document(SID, original['id'])
    restarted = PersonalEvents(ResearchStore(tmp_path))
    assert not restarted.retrieve_context(SID, 'Zephyr')['events']
    assert not EventDependencies(store).live(SID, packet['bindings'])


def test_relation_origins_and_endpoint_revisions_are_pinned_not_reinterpreted(tmp_path):
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    a = event(store, action='Zephyr launch decision.')
    b = event(store, action='Prepare the launch draft.')
    events.relate(SID, a['id'], b['id'], 'hypothesis', conditions=['Tentative only.'])
    packet = events.retrieve_context(SID, 'Zephyr')
    assert packet['relations'] and packet['relations'][0]['relation'] == 'hypothesis'
    events.revise_event(SID, b['id'], expected_revision=1, event_type='task', action='Corrected draft.', origins=b['origins'])
    assert not EventDependencies(store).live(SID, packet['bindings'])
    assert not events.retrieve_context(SID, 'Zephyr')['relations']


def test_homonymous_entities_are_not_merged_or_authorized_by_names(tmp_path):
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    left = events.register_entity(SID, 'person', 'Alex Chen', aliases=['Alex'])
    right = events.register_entity(SID, 'person', 'Alex Wang', aliases=['Alex'])
    event(store, action='First customer follow-up.', actor_id=left['id'])
    event(store, action='Second customer follow-up.', actor_id=right['id'])
    packet = events.retrieve_context(SID, 'Alex')
    assert packet['trace']['ambiguous_aliases'] >= 1
    assert {row['actor_id'] for row in packet['events']} == {left['id'], right['id']}
    assert {row['actor']['id'] for row in packet['events']} == {left['id'], right['id']}
    assert all(row['actor']['id'] == row['actor_id'] for row in packet['events'])
    assert events.retrieve_context(SID, 'Alex', allowed_memory_ids=[])['trace']['ambiguous_aliases'] == 0


def test_relevant_two_hop_bundle_precedes_twelve_isolated_lexical_seeds(tmp_path):
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    _, origin = memory(store)
    start = events.record_event(SID, event_type='decision', action='Zephyr renewal Zephyr renewal Zephyr renewal.', origins=[origin])
    middle = events.record_event(SID, event_type='task', action='Confirm the contractual restriction.', origins=[origin])
    end = events.record_event(SID, event_type='task', action='Prepare the compliant delivery draft.', origins=[origin])
    first = events.relate(SID, start['id'], middle['id'], 'depends_on', conditions=['Only after explicit approval.'])
    second = events.relate(SID, middle['id'], end['id'], 'before', conditions=['Keep the customer restriction.'])
    for index in range(12):
        events.record_event(SID, event_type='observation', action=f'Zephyr renewal isolated background {index}.', origins=[origin])
    packet = events.retrieve_context(SID, 'Zephyr renewal')
    assert {start['id'], middle['id'], end['id']} <= {row['id'] for row in packet['events']}
    assert {first['id'], second['id']} <= {row['id'] for row in packet['relations']}
    assert_context_budgets(packet)
    assert all(row['source_id'] in {event['id'] for event in packet['events']}
        and row['target_id'] in {event['id'] for event in packet['events']} for row in packet['relations'])


def test_bounded_packet_retains_whole_conditions_and_reports_omission_not_silent_truncation(tmp_path):
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    _, origin = memory(store)
    for index in range(12):
        events.record_event(SID, event_type='observation', action=f'Zephyr observation {index}.', origins=[origin], conditions=['Only under the original limitation.'])
    packet = events.retrieve_context(SID, 'Zephyr')
    assert 1 <= len(packet['events']) <= 8 and len(packet['relations']) <= 12
    assert_context_budgets(packet)
    assert packet['trace']['truncation']
    assert all(row['conditions'] == ['Only under the original limitation.'] for row in packet['events'])
    huge = events.record_event(SID, event_type='observation', action='巨大' * 500, origins=[origin],
        conditions=[str(index) + '限制' * 125 for index in range(16)])
    result = events.retrieve_context(SID, '巨大', event_ids=[huge['id']])
    assert not result['events'] and 'context_bytes' in result['trace']['truncation']


@pytest.mark.parametrize('kind', ['reported_cause', 'hypothesis', 'depends_on'])
def test_target_endpoint_bridge_keeps_intermediate_nodes_before_recent_distractions(tmp_path, kind):
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    _, origin = memory(store)
    actions = ['Zephyr change', 'Capacity rises', 'Queue grows', 'Slack vanishes', 'Orion delay']
    chain = [events.record_event(SID, event_type='observation', action=action, origins=[origin]) for action in actions]
    edges = []
    for left, right in zip(chain, chain[1:]):
        if kind == 'depends_on':
            left, right = right, left
        edges.append(events.relate(SID, left['id'], right['id'], kind))
    # Newer unrelated neighborhoods must not occupy all graph slots before
    # the only complete mechanism connecting the actual question endpoints.
    for index in range(10):
        noise = events.record_event(SID, event_type='observation', action=f'Unrelated note {index}', origins=[origin])
        events.relate(SID, chain[0]['id'], noise['id'], 'related_to')
    packet = events.retrieve_context(SID, 'Zephyr Orion')
    assert {row['id'] for row in chain} <= {row['id'] for row in packet['events']}
    assert {row['id'] for row in edges} <= {row['id'] for row in packet['relations']}
    assert packet['trace']['bridge_search']['retained_paths'] >= 1
    assert packet['trace']['bridge_search']['states'] <= 128
    assert packet['trace']['bridge_search']['edges'] <= 192
    assert_context_budgets(packet)
    assert EventDependencies(store).live(SID, packet['bindings'])


@pytest.mark.parametrize('blocked', ['scope', 'revision', 'withdrawn'])
def test_bridge_cannot_connect_through_a_disallowed_or_stale_middle_event(tmp_path, blocked):
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    _, allowed = memory(store, 'Permitted source')
    note, hidden = memory(store, 'Hidden source')
    chain = [events.record_event(SID, event_type='observation', action=action,
        origins=[hidden if index == 2 else allowed]) for index, action in enumerate(
            ['Zephyr input', 'First mechanism', 'Private mechanism', 'Last mechanism', 'Orion output'])]
    edges = [events.relate(SID, left['id'], right['id'], 'reported_cause') for left, right in zip(chain, chain[1:])]
    options = {}
    if blocked == 'scope':
        options = {'allowed_memory_ids': [allowed['memory_id']]}
    elif blocked == 'revision':
        events.revise_event(SID, chain[2]['id'], expected_revision=1, event_type='observation',
            action='Corrected private mechanism', origins=[hidden])
    else:
        store.delete_memory(SID, note['id'], expected_revision=1)
    packet = events.retrieve_context(SID, 'Zephyr Orion', **options)
    assert packet['trace']['bridge_search']['retained_paths'] == 0
    assert {row['id'] for row in edges} - {row['id'] for row in packet['relations']}
    assert all(row['source_revision'] == next(value['revision'] for value in packet['events'] if value['id'] == row['source_id'])
               for row in packet['relations'])


def test_oversized_bridge_is_not_reported_as_retained_complete_path(tmp_path):
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    _, origin = memory(store)
    chain = [events.record_event(SID, event_type='observation', action=(
        'Zephyr start ' if index == 0 else 'Orion finish ' if index == 6 else f'Mechanism {index} ')
        + 'Detailed constraint text. ' * 15, origins=[origin]) for index in range(7)]
    for left, right in zip(chain, chain[1:]):
        events.relate(SID, left['id'], right['id'], 'reported_cause')
    packet = events.retrieve_context(SID, 'Zephyr Orion')
    assert packet['trace']['bridge_search']['candidate_paths'] >= 1
    assert packet['trace']['bridge_search']['retained_paths'] == 0
    assert 'bridge_capacity' in packet['trace']['truncation']
    assert_context_budgets(packet)


def test_eight_event_seven_edge_chain_reaches_prompt_with_all_host_bindings(tmp_path):
    from docthinker.server.research_event_reasoning import (
        prepare_event_frontier, build_event_obligations, render_event_reasoning)
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    _, origin = memory(store)
    actions = ['Climate goal', 'Policy tool', 'Investment', 'Local supply', 'Imports fall',
               'Diversity', 'Risk falls', 'Energy security']
    chain = [events.record_event(SID, event_type='observation', action=action, origins=[origin],
        occurred_from=index * 2, occurred_until=index * 2 + 1) for index, action in enumerate(actions)]
    edges = [events.relate(SID, left['id'], right['id'], 'reported_cause') for left, right in zip(chain, chain[1:])]
    packet = events.retrieve_context(SID, 'Climate goal Energy security')
    assert [row['action'] for row in packet['events']] == actions
    assert {row['id'] for row in packet['bindings']['events']} == {row['id'] for row in chain}
    assert {row['id'] for row in packet['bindings']['relations']} == {row['id'] for row in edges}
    assert EventDependencies(store).live(SID, packet['bindings'])
    assert_context_budgets(packet)
    prepared = prepare_event_frontier('Climate goal Energy security', packet, [])
    rendered = render_event_reasoning(prepared, build_event_obligations(prepared))
    assert len(rendered.encode()) <= 7000
    prompt = json.loads(rendered)
    assert prompt['candidate_paths'][0]['event_ids'] == [row['id'] for row in chain]
    assert {row['id'] for row in prompt['needs']} == {
        *(f'L{index}' for index in range(1, 8)), *(f'E{index}' for index in range(1, 9)), 'P1', 'A1'}
    assert not packet['trace']['truncation']


def test_speculative_shortcut_does_not_hide_a_longer_recorded_mechanism(tmp_path):
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    _, origin = memory(store)
    chain = [events.record_event(SID, event_type='observation', action=action, origins=[origin])
             for action in ['Zephyr input', 'Capacity rise', 'Work queue', 'Slack loss', 'Orion delay']]
    recorded = [events.relate(SID, left['id'], right['id'], 'reported_cause') for left, right in zip(chain, chain[1:])]
    events.relate(SID, chain[0]['id'], chain[-1]['id'], 'hypothesis')
    parallel = events.relate(SID, chain[0]['id'], chain[1]['id'], 'hypothesis')
    for index in range(10):
        distractor = events.record_event(SID, event_type='observation', action=f'Unrelated recent note {index}', origins=[origin])
        events.relate(SID, chain[0]['id'], distractor['id'], 'related_to')
    packet = events.retrieve_context(SID, 'Zephyr Orion')
    assert {row['id'] for row in chain} <= {row['id'] for row in packet['events']}
    assert {row['id'] for row in recorded} <= {row['id'] for row in packet['relations']}
    assert parallel['id'] in {row['id'] for row in packet['relations']}
    assert packet['trace']['bridge_search']['candidate_paths'] >= 2
    assert_context_budgets(packet)


def test_host_binding_capacity_never_drops_revision_guards_to_keep_a_path(tmp_path, monkeypatch):
    import docthinker.server.personal_events as module
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    _, origin = memory(store)
    chain = [events.record_event(SID, event_type='observation', action=action, origins=[origin])
             for action in ['Zephyr start', 'Middle mechanism', 'Orion finish']]
    for left, right in zip(chain, chain[1:]):
        events.relate(SID, left['id'], right['id'], 'reported_cause')
    monkeypatch.setattr(module, 'MAX_CONTEXT_BINDING_BYTES', 400)
    packet = events.retrieve_context(SID, 'Zephyr Orion')
    assert packet['trace']['binding_bytes'] <= 400
    assert 'binding_bytes' in packet['trace']['truncation']
    assert packet['trace']['bridge_search']['retained_paths'] == 0
    assert {row['id'] for row in packet['events']} == {row['id'] for row in packet['bindings']['events']}
    assert {row['id'] for row in packet['relations']} == {row['id'] for row in packet['bindings']['relations']}


def test_optional_event_search_does_not_shrink_the_public_question_contract(tmp_path):
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    assert not events.retrieve_context(SID, '\\' * 8000)['events']
    original = event(store, action='Zephyr current decision.')
    question = 'Zephyr ' + ('补充检索上下文' * 3000)
    packet = events.retrieve_context(SID, question)
    assert packet['events'][0]['id'] == original['id']
    assert 'search_text_budget' in packet['trace']['truncation']


def test_answer_history_descendants_and_confirmed_notes_invalidate_when_an_event_is_corrected(tmp_path):
    store = ResearchStore(tmp_path)
    events = PersonalEvents(store)
    original = event(store, action='Zephyr private delivery instruction.')
    bindings = events.retrieve_context(SID, 'Zephyr')['bindings']
    first = answer(store, bindings)
    second = answer(store, parents=[first])
    note = store.create_memory(SID, 'finding', 'User-confirmed interpretation of the event.', source_answer_id=second)
    assert AnswerDependencies(store).states(SID, [first, second]) == {first: True, second: True}
    events.revise_event(SID, original['id'], expected_revision=1, event_type='decision',
        action='Corrected Zephyr instruction.', origins=original['origins'])
    assert AnswerDependencies(store).states(SID, [first, second]) == {first: False, second: False}
    assert store.get_memory(SID, note['id'])['redacted'] is True
    with pytest.raises(ResearchStoreError):
        store.create_memory(SID, 'finding', 'Cannot resurrect old event prose.', source_answer_id=second)


def test_event_lineage_metadata_is_bounded_private_and_missing_required_record_is_fail_closed(tmp_path):
    store = ResearchStore(tmp_path)
    original = event(store, action='Zephyr confirmed decision.')
    bindings = PersonalEvents(store).retrieve_context(SID, 'Zephyr')['bindings']
    metadata = EventDependencies.context_metadata(bindings)
    identifier = str(uuid.uuid4())
    store.record_answer(SID, identifier, [], {'valid': True}, 'general', context_metadata=metadata)
    assert EventDependencies(store).states(SID, [identifier]) == {identifier: False}
    EventDependencies(store).record(SID, identifier, bindings)
    assert EventDependencies(store).states(SID, [identifier]) == {identifier: True}
    public = store.answer_metadata(SID, [identifier])[identifier]
    assert public['used_event_count'] == 1 and public['used_event_relation_count'] == 0
    assert 'event_bindings_sha256' not in public and 'event_dependency_required' not in public
    with store._read() as db:
        raw = db.execute('SELECT bindings FROM research_answer_event_dependencies WHERE answer_id=?', (identifier,)).fetchone()[0]
        assert original['action'] not in raw
    bad = copy.deepcopy(bindings); bad['events'][0]['revision'] += 1
    assert not EventDependencies(store).live(SID, bad)
    with pytest.raises(ResearchStoreError):
        EventDependencies(store).record(SID, identifier, bad)


@pytest.mark.parametrize('bad', [None, [], {'events': []}, {'events': [{}], 'relations': [], 'source_scope_sha256': None, 'memory_scope_sha256': None}])
def test_untrusted_binding_shapes_never_authorize_an_event(tmp_path, bad):
    store = ResearchStore(tmp_path)
    with store._read() as db:
        assert event_bindings_live(db, SID, bad, store=store) is False


def test_legacy_answers_without_event_dependencies_keep_their_existing_behavior(tmp_path):
    store = ResearchStore(tmp_path)
    identifier = answer(store)
    with store._read() as db:
        assert source_answer_memory_live(db, SID, identifier, at=store.clock())
    assert EventDependencies(store).states(SID, [identifier]) == {identifier: True}
