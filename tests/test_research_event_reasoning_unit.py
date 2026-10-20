"""Directed event obligations and single-call, quote-bound model assessments."""
import hashlib
import json

import pytest

from docthinker.server.research_event_reasoning import (
    EVENT_RESPONSE_POLICY, build_event_obligations, event_response_contract,
    event_trial_response, hydrate_event_receipt, parse_event_response, prepare_event_frontier,
    render_event_reasoning, supplement_event_gaps, normalize_event_condition_schema, MAX_RECEIPT_BYTES,
)


def event(identifier, action, start=1, end=2, conditions=None):
    return {'id': identifier, 'revision': 1, 'status': 'active', 'action': action,
            'occurred_from': start, 'occurred_until': end, 'conditions': conditions or [], 'origins': []}


def relation(identifier, source, target, kind='reported_cause', conditions=None):
    return {'id': identifier, 'source_id': source, 'target_id': target, 'source_revision': 1,
            'target_revision': 1, 'relation': kind, 'conditions': conditions or [], 'status': 'active'}


def context(conditions=None):
    return {'events': [event('a', 'Customer added scope', conditions=conditions),
                       event('b', 'Workload increased', 3, 4), event('c', 'Delivery delayed', 5, 6)],
            'relations': [relation('r1', 'a', 'b'), relation('r2', 'b', 'c')], 'bindings': {}, 'trace': {}}


def source(text='Customer added scope increased workload, which delayed delivery.', identifier='s1', label='S1', start=0):
    return {'id': identifier, 'document_id': 'doc1', 'label': label, 'filename': 'worklog.txt',
            'content': text, 'excerpt_start': start, 'excerpt_end': start + len(text)}


def witness(row):
    return {'label': row['label'], 'quote': row['content'], 'start': row['excerpt_start']}


def catalog_witnesses(value):
    if isinstance(value, list):
        return [catalog_witnesses(item) for item in value]
    if isinstance(value, dict):
        return {key: ([{'label': item['label']} for item in raw] if key in {'evidence', 'satisfaction_evidence'}
                      else catalog_witnesses(raw)) for key, raw in value.items()}
    return value


def satisfied_condition(name, row):
    return {'condition': name, 'verdict': 'support', 'evidence': [witness(row)],
            'satisfaction': 'met', 'satisfaction_evidence': [witness(row)]}


def envelope(row, conditions=None):
    quote = [witness(row)]
    return {'answer': 'The records support the proposed chain, not a proven causal conclusion [S1].',
            'event_assessment': {
                'links': [{'obligation_id': identifier, 'verdict': 'support', 'evidence': quote,
                           'from_event': left, 'to_event': right,
                           'conditions': conditions or []} for identifier, left, right in (('L1', 'a', 'b'), ('L2', 'b', 'c'))],
                'events': [{'obligation_id': f'E{index}', 'verdict': 'support', 'evidence': quote,
                            'conditions': conditions or []} for index in range(1, 4)],
                'path': {'verdict': 'support', 'evidence': quote},
                'alternatives': [{'explanation': 'Staff leave rather than new scope',
                                  'verdict': 'counterevidence', 'evidence': quote}], 'missing_links': []}}


def parse(value, ctx=None, rows=None):
    rows = rows or [source()]
    return parse_event_response(json.dumps(value), ctx or context(), rows)


def statuses(receipt):
    return {row['id']: row['status'] for row in receipt['obligations']}


def test_directed_chain_and_alternative_are_explicit_unassessed_obligations():
    receipt = build_event_obligations(context())
    assert receipt['causality_verified'] is False and receipt['model_assessed'] is False
    assert statuses(receipt) == {name: 'unknown' for name in ('L1', 'L2', 'E1', 'E2', 'E3', 'P1', 'A1')}
    assert receipt['paths'] == [{'event_ids': ['a', 'b', 'c'], 'obligation_ids': ['L1', 'L2']}]
    assert receipt['suggested_queries'] and receipt['unknown'] == 7


def test_depends_on_is_inverted_not_traversed_as_effect_causing_prerequisite():
    ctx = context()
    ctx['relations'] = [relation('r1', 'b', 'a', 'depends_on'), relation('r2', 'c', 'b', 'depends_on')]
    receipt = build_event_obligations(ctx)
    assert [(row['source_id'], row['target_id']) for row in receipt['obligations'][:2]] == [('a', 'b'), ('b', 'c')]
    assert receipt['paths'] == []
    assert len(receipt['obligations']) == 5
    assert all(row['prerequisite_candidate'] and not row['causal_candidate'] for row in receipt['obligations'][:2])
    assert 'P1' not in statuses(receipt) and 'A1' not in statuses(receipt)


def test_before_and_related_to_never_become_causal_path():
    ctx = context()
    ctx['relations'] = [relation('r1', 'a', 'b', 'before'), relation('r2', 'b', 'c', 'related_to')]
    receipt = build_event_obligations(ctx)
    assert receipt['paths'] == []
    assert len(receipt['obligations']) == 5
    assert all(not row['causal_candidate'] for row in receipt['obligations'][:2])


def test_before_requires_nonoverlapping_interval_order_not_merely_ordered_starts():
    ctx = context()
    ctx['events'][0]['occurred_until'] = 4
    ctx['events'][1]['occurred_from'] = 3
    ctx['relations'] = [relation('r1', 'a', 'b', 'before')]
    receipt = build_event_obligations(ctx)
    assert 'temporal_order_conflict' in receipt['obligations'][0]['gaps']


def test_known_reverse_time_blocks_even_exact_quote_model_support():
    ctx = context()
    ctx['events'][0]['occurred_from'] = 10
    ctx['events'][0]['occurred_until'] = 11
    _, receipt = parse(envelope(source()), ctx)
    assert statuses(receipt)['L1'] == 'unknown'
    assert statuses(receipt)['P1'] == 'unknown'
    assert 'temporal_order_conflict' in receipt['obligations'][0]['gaps']
    assert receipt['causality_verified'] is False


def test_cause_after_effect_onset_is_not_accepted_from_overlapping_intervals():
    ctx, row = context(), source()
    ctx['events'][0].update(occurred_from=5, occurred_until=6)
    ctx['events'][1].update(occurred_from=3, occurred_until=8)
    _, receipt = parse(envelope(row), ctx, [row])
    assert statuses(receipt)['L1'] == statuses(receipt)['P1'] == 'unknown'
    assert 'temporal_order_conflict' in receipt['obligations'][0]['gaps']


def test_quote_bound_assessment_is_visible_but_not_truth_or_automatic_action():
    row = source()
    answer, receipt = parse(envelope(row))
    assert answer.startswith('The records')
    assert set(statuses(receipt).values()) == {'model_assessed_support'}
    assert receipt['status'] == 'assessed_with_caveats'
    assert receipt['model_assessed'] is True and receipt['causality_verified'] is False
    evidence = receipt['obligations'][0]['evidence'][0]
    assert evidence == {'label': 'S1', 'source_id': 's1', 'span': [0, len(row['content'])],
                        'quote_sha256': hashlib.sha256(row['content'].encode()).hexdigest()}
    assert 'quote' not in evidence and 'execution_authorized' not in receipt


@pytest.mark.parametrize('text,start', [
    ('Customer added scope increased workload, which delayed delivery.', 0),
    ('原文：客户追加需求后，记录到工作量增加。尚不能排除其他原因。', 173),
])
def test_catalog_label_binds_whole_current_window_with_character_offsets_and_utf8_hash(text, start):
    row = source(text, start=start)
    value = catalog_witnesses(envelope(row))
    answer, receipt = parse(value, rows=[row])
    assert answer == value['answer'] and receipt['causality_verified'] is False
    assert set(statuses(receipt).values()) == {'model_assessed_support'}
    expected = {'label': 'S1', 'source_id': row['id'], 'span': [start, start + len(text)],
                'quote_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()}
    assert receipt['obligations'][0]['evidence'] == [expected]
    assert hydrate_event_receipt(receipt, [row]) == receipt


@pytest.mark.parametrize('problem', ['unknown', 'duplicate', 'bad_end', 'negative', 'boolean', 'empty', 'whitespace', 'extra_key'])
def test_catalog_reference_cannot_accept_unknown_ambiguous_or_malformed_current_window(problem):
    row = source()
    value = catalog_witnesses(envelope(row))
    rows = [row]
    if problem == 'unknown':
        value['event_assessment']['links'][0]['evidence'][0]['label'] = 'S99'
    elif problem == 'duplicate':
        rows.append({**row, 'id': 'different-source'})
    elif problem == 'bad_end':
        row['excerpt_end'] += 1
    elif problem == 'negative':
        row.update(excerpt_start=-1, excerpt_end=len(row['content']) - 1)
    elif problem == 'boolean':
        row.update(excerpt_start=False, excerpt_end=len(row['content']))
    elif problem == 'empty':
        row.update(content='', excerpt_end=0)
    elif problem == 'whitespace':
        row.update(content=' \n\t', excerpt_end=3)
    else:
        value['event_assessment']['links'][0]['evidence'][0]['whole_window'] = True
    _, receipt = parse(value, rows=rows)
    assert statuses(receipt)['L1'] == statuses(receipt)['P1'] == 'unknown'


def test_catalog_labels_preserve_all_cross_source_premises_and_condition_axes():
    first = source('The scope expansion required additional critical-path work in this project.', 'scope', 'S1', 100)
    second = source('The rule requires explicit approval; written approval has NOT been given.', 'approval', 'S2', 250)
    ctx = context(['Explicit approval'])
    condition = {'condition': 'Explicit approval', 'verdict': 'support', 'evidence': [{'label': 'S2'}],
                 'satisfaction': 'unmet', 'satisfaction_evidence': [{'label': 'S2'}]}
    value = catalog_witnesses(envelope(first))
    value['event_assessment']['links'][0].update(
        evidence=[{'label': 'S1'}, {'label': 'S2'}], conditions=[condition])
    _, receipt = parse(value, ctx, [first, second])
    need = receipt['obligations'][0]
    assert need['status'] == 'unknown' and statuses(receipt)['P1'] == 'unknown'
    assert {ref['source_id'] for ref in need['evidence']} == {'scope', 'approval'}
    assert need['condition_checks'][0]['rule_status'] == 'model_assessed_support'
    assert need['condition_checks'][0]['satisfaction'] == 'unmet'
    assert 'conditions_not_met' in need['gaps']
    condition['satisfaction'] = 'unknown'
    condition['satisfaction_evidence'] = []
    _, uncertain = parse(value, ctx, [first, second])
    assert statuses(uncertain)['L1'] == statuses(uncertain)['P1'] == 'unknown'
    assert hydrate_event_receipt(receipt, [first]) is None


def test_catalog_counterevidence_is_not_upgraded_to_support_by_valid_geometry():
    row = source('The measured execution log does NOT support the alleged critical-path effect.')
    value = catalog_witnesses(envelope(row))
    value['event_assessment']['links'][0]['verdict'] = 'counterevidence'
    _, receipt = parse(value, rows=[row])
    assert statuses(receipt)['L1'] == 'model_assessed_counterevidence'
    assert statuses(receipt)['P1'] == 'unknown' and receipt['causality_verified'] is False


@pytest.mark.parametrize('change', ['removed', 'mutated', 'shrunk', 'moved', 'label'])
def test_catalog_bound_whole_window_receipt_revokes_when_current_source_geometry_changes(change):
    row = source('The whole retained original source includes both the rule and its qualifications.', start=200)
    _, receipt = parse(catalog_witnesses(envelope(row)), rows=[row])
    changed = dict(row)
    if change == 'removed':
        rows = []
    elif change == 'mutated':
        changed['content'] = changed['content'].replace('original', 'modified')
        rows = [changed]
    elif change == 'shrunk':
        changed.update(content=row['content'][:-1], excerpt_end=row['excerpt_end'] - 1)
        rows = [changed]
    elif change == 'moved':
        changed.update(excerpt_start=201, excerpt_end=row['excerpt_end'] + 1)
        rows = [changed]
    else:
        changed['label'] = 'S2'
        rows = [changed]
    assert hydrate_event_receipt(receipt, rows) is None


def test_catalog_and_legacy_exact_quote_protocols_can_share_a_joint_assessment():
    row = source('The original record includes a mechanism and all relevant conditions.', start=100)
    value = catalog_witnesses(envelope(row))
    value['event_assessment']['links'][0]['evidence'] = [witness(row)]
    _, receipt = parse(value, rows=[row])
    assert set(statuses(receipt).values()) == {'model_assessed_support'}
    assert hydrate_event_receipt(receipt, [row]) == receipt


@pytest.mark.parametrize('modification', ['label', 'quote', 'offset', 'duplicate', 'unhashable', 'oversize', 'verdict'])
def test_invalid_citation_binding_never_accepts_support(modification):
    value = envelope(source())
    proof = value['event_assessment']['links'][0]['evidence'][0]
    if modification == 'label':
        proof['label'] = 'S99'
    elif modification == 'quote':
        proof['quote'] = 'The project was definitely delayed by new scope.'
    elif modification == 'offset':
        proof['start'] = 1
    elif modification == 'duplicate':
        value['event_assessment']['links'].append(value['event_assessment']['links'][0])
    elif modification == 'unhashable':
        proof['label'] = ['S1']
    elif modification == 'oversize':
        proof['quote'] = 'x' * 601
    elif modification == 'verdict':
        value['event_assessment']['links'][0]['verdict'] = ['support']
    _, receipt = parse(value)
    assert statuses(receipt)['L1'] == 'unknown'
    assert statuses(receipt)['P1'] == 'unknown'


def test_absolute_quote_coordinate_must_fall_inside_final_retained_window():
    row = source(start=100)
    _, receipt = parse(envelope(row), rows=[row])
    assert statuses(receipt)['L1'] == 'model_assessed_support'
    value = envelope(row)
    value['event_assessment']['links'][0]['evidence'][0]['start'] = 0
    _, receipt = parse(value, rows=[row])
    assert statuses(receipt)['L1'] == 'unknown'


def test_unique_exact_quote_is_located_by_host_not_model_character_counting():
    row = source(start=150)
    value = envelope(row)
    for link in value['event_assessment']['links']:
        link['evidence'][0].pop('start', None)
    _, receipt = parse(value, rows=[row])
    assert statuses(receipt)['L1'] == 'model_assessed_support'
    assert receipt['obligations'][0]['evidence'][0]['span'] == [150, 150 + len(row['content'])]


def test_repeated_quote_needs_exact_disambiguating_coordinate():
    row = source('Workload increased. Workload increased.', start=100)
    value = envelope(row)
    proof = value['event_assessment']['links'][0]['evidence'][0]
    proof.update(quote='Workload increased.')
    proof.pop('start', None)
    _, receipt = parse(value, rows=[row])
    assert statuses(receipt)['L1'] == 'unknown'
    proof['start'] = 100
    _, receipt = parse(value, rows=[row])
    assert statuses(receipt)['L1'] == 'model_assessed_support'
    # A source present during recall but omitted by the fitter is not evidence.
    _, receipt = parse_event_response(json.dumps(envelope(row)), context(), [])
    assert statuses(receipt)['L1'] == 'unknown'


def test_duplicate_labels_fail_closed_even_if_first_quote_matches():
    rows = [source(), source('Unrelated staff holiday evidence exists.', identifier='s2')]
    _, receipt = parse(envelope(rows[0]), rows=rows)
    assert statuses(receipt)['L1'] == 'unknown'


def test_all_event_conditions_need_independent_grounded_assessment():
    ctx = context(['Only if the new scope was on the critical path'])
    row = source()
    _, receipt = parse(envelope(row), ctx)
    assert statuses(receipt)['L1'] == 'unknown'
    valid = [satisfied_condition(ctx['events'][0]['conditions'][0], row)]
    _, receipt = parse(envelope(row, valid), ctx)
    assert statuses(receipt)['L1'] == 'model_assessed_support'
    assert statuses(receipt)['P1'] == 'unknown'  # Individual checks do not bind conditions across the whole path.
    valid[0]['evidence'][0]['quote'] = 'Invented confirmation of the critical path'
    _, receipt = parse(envelope(row, valid), ctx)
    assert statuses(receipt)['L1'] == 'unknown'


def test_path_must_check_joint_conditions_not_just_each_link_separately():
    ctx, row = context(['Scope change was on the critical path']), source()
    condition = satisfied_condition(ctx['events'][0]['conditions'][0], row)
    value = envelope(row, [condition])
    _, before = parse(value, ctx)
    assert statuses(before)['P1'] == 'unknown'
    value['event_assessment']['path']['conditions'] = [condition]
    _, after = parse(value, ctx)
    assert statuses(after)['P1'] == 'model_assessed_support'


def test_counterevidence_is_not_converted_to_positive_path_support():
    value = envelope(source())
    value['event_assessment']['links'][0]['verdict'] = 'counterevidence'
    _, receipt = parse(value)
    assert statuses(receipt)['L1'] == 'model_assessed_counterevidence'
    assert statuses(receipt)['P1'] == 'unknown'
    assert receipt['counterevidence'] == 1 and receipt['status'] == 'evidence_gaps'


def test_reverse_model_direction_cannot_attach_support_to_a_host_link():
    value = envelope(source())
    link = value['event_assessment']['links'][0]
    link['from_event'], link['to_event'] = link['to_event'], link['from_event']
    _, receipt = parse(value)
    assert statuses(receipt)['L1'] == statuses(receipt)['P1'] == 'unknown'


def test_alternative_checks_cannot_be_skipped_or_supplied_without_sources():
    value = envelope(source())
    value['event_assessment']['alternatives'][0]['evidence'] = []
    _, receipt = parse(value)
    assert statuses(receipt)['A1'] == statuses(receipt)['P1'] == 'unknown'
    assert any(gap['code'] == 'alternative_explanation_unassessed' for gap in receipt['gaps'])


def test_supported_competing_explanation_blocks_preferred_unique_chain():
    value = envelope(source())
    value['event_assessment']['alternatives'][0]['verdict'] = 'support'
    _, receipt = parse(value)
    assert statuses(receipt)['A1'] == 'model_assessed_counterevidence'
    assert statuses(receipt)['P1'] == 'unknown'
    assert any(row['code'] == 'competing_explanation_supported' for row in receipt['gaps'])


def test_unrelated_isolated_event_has_occurrence_need_not_false_causal_success():
    ctx = {'events': [event('a', 'Meeting moved to Friday')], 'relations': []}
    _, receipt = parse_event_response(json.dumps({'answer': 'The meeting moved.', 'event_assessment': {}}), ctx, [])
    assert statuses(receipt) == {'E1': 'unknown'}
    assert receipt['model_assessed'] is False and receipt['status'] == 'not_assessed'
    _, empty = parse_event_response(json.dumps({'answer': 'Nothing here.', 'event_assessment': {}}), {}, [])
    assert empty['status'] == 'not_assessed' and empty['model_assessed'] is False


def test_isolated_event_memory_origin_is_context_not_a_source_quote():
    ctx = {'events': [event('a', 'Meeting moved to Friday')], 'relations': []}
    ctx['events'][0]['origins'] = [{'kind': 'memory', 'memory_id': 'm1', 'revision': 1}]
    value = {'answer': 'Your note says Friday.', 'event_assessment': {'events': [
        {'obligation_id': 'E1', 'verdict': 'support', 'evidence': [{'label': 'm1', 'quote': 'Meeting moved to Friday', 'start': 0}]}]}}
    _, receipt = parse_event_response(json.dumps(value), ctx, [])
    assert statuses(receipt)['E1'] == 'unknown'
    rendered = json.loads(render_event_reasoning(ctx, build_event_obligations(ctx)))
    assert rendered['events'][0]['origin_pointers_not_citations'] == [{'memory_id': 'm1', 'revision': 1}]


def test_missing_intermediate_stays_unknown_and_proposes_specific_next_query():
    value = envelope(source())
    value['event_assessment']['missing_links'] = [{'from_event': 'a', 'to_event': 'c',
        'reason': 'Workload records do not show that added scope hit the critical path.',
        'query': 'critical path worklog staff leave delivery schedule'}]
    _, receipt = parse(value)
    assert statuses(receipt)['P1'] == 'unknown'
    assert value['event_assessment']['missing_links'][0]['query'] in receipt['suggested_queries']
    assert any(row['code'] == 'missing_intermediate_link' for row in receipt['gaps'])


def test_all_distinct_model_missing_queries_survive_and_outrank_generic_queries():
    value = envelope(source())
    queries = ['Find actual extra work hours consumed on the critical path.',
               'Find whether the account outage overlapped the critical path or slack.']
    value['event_assessment']['missing_links'] = [
        {'from_event': 'b', 'to_event': 'c', 'reason': 'Different missing premise ' + str(index), 'query': query}
        for index, query in enumerate(queries)]
    _, receipt = parse(value)
    assert receipt['suggested_queries'][:2] == queries
    assert statuses(receipt)['P1'] == 'unknown'
    assert len(receipt['suggested_queries']) <= 8
    assert len(json.dumps(receipt, ensure_ascii=False, separators=(',', ':')).encode()) <= 6144


def test_oversized_missing_link_list_cannot_hide_gap_beyond_parser_bound():
    value = envelope(source())
    value['event_assessment']['missing_links'] = [None] * 4 + [{'from_event': 'a', 'to_event': 'c',
        'reason': 'A real gap after the allowed first four.', 'query': 'Find the missing mechanism'}]
    _, receipt = parse(value)
    assert statuses(receipt)['P1'] == 'unknown'


def test_disconnected_edges_never_form_complete_chain_from_model_claim_alone():
    ctx = context()
    ctx['events'].append(event('d', 'Staff leave affected delivery', 5, 6))
    ctx['relations'][1] = relation('r2', 'c', 'd')
    _, receipt = parse(envelope(source()), ctx)
    assert statuses(receipt)['P1'] == 'unknown'
    assert any(gap['code'] == 'disconnected_causal_components' for gap in receipt['gaps'])


def full_compact_review(ctx, row):
    checks = [{'id': need['id'], 'verdict': 'support', 'evidence': [witness(row)]}
              for need in build_event_obligations(ctx)['obligations'] if need['id'] != 'A1']
    checks.append({'id': 'A1', 'verdict': 'counterevidence', 'alternative': 'Another cause', 'evidence': [witness(row)]})
    return {'answer': 'A source-supported candidate mechanism, not verified causality [S1].',
            'event_assessment': {'checks': checks}}


def test_branching_and_converging_dag_requires_every_branch_occurrence_and_link():
    ctx = {'events': [event('a', 'Scope changed', 1, 1), event('b', 'Workload rose', 2, 2),
                      event('c', 'Testing expanded', 2, 2), event('d', 'Delivery delayed', 3, 3)],
           'relations': [relation('ab', 'a', 'b'), relation('ac', 'a', 'c'),
                         relation('bd', 'b', 'd'), relation('cd', 'c', 'd')]}
    row = source('The records support both workload and testing mechanisms in this project, with an independently recorded delay.')
    prepared = prepare_event_frontier('How did scope change delay delivery?', ctx, [row])
    packet = json.loads(render_event_reasoning(prepared, build_event_obligations(prepared)))
    assert len(packet['candidate_paths']) == 2
    assert next(need for need in packet['needs'] if need['id'] == 'P1')['required_link_ids'] == ['L1', 'L2', 'L3', 'L4']
    value = full_compact_review(ctx, row)
    _, accepted = parse(value, prepared, [row])
    assert statuses(accepted)['P1'] == 'model_assessed_support' and accepted['causality_verified'] is False
    for missing in ('L1', 'L2', 'L3', 'L4', 'E1', 'E2', 'E3', 'E4', 'A1'):
        partial = json.loads(json.dumps(value))
        partial['event_assessment']['checks'] = [check for check in partial['event_assessment']['checks'] if check['id'] != missing]
        _, receipt = parse(partial, prepared, [row])
        assert statuses(receipt)['P1'] == 'unknown'


def test_dag_branches_outside_four_illustrative_path_samples_still_need_occurrence():
    ctx = {'events': [event('a', 'Starting condition', 1, 1), event('z', 'Observed effect', 3, 3)]
                     + [event(f'm{index}', f'Intermediate mechanism {index}', 2, 2) for index in range(5)],
           'relations': [relation(f'{side}{index}', 'a' if side == 'a' else f'm{index}',
                                  f'm{index}' if side == 'a' else 'z') for index in range(5) for side in ('a', 'z')]}
    row = source('Five separate recorded intermediate mechanisms connect the starting condition with the observed effect.')
    assert len(build_event_obligations(ctx)['paths']) == 4
    value = full_compact_review(ctx, row)
    _, accepted = parse(value, ctx, [row])
    assert statuses(accepted)['P1'] == 'model_assessed_support'
    value['event_assessment']['checks'] = [check for check in value['event_assessment']['checks'] if check['id'] != 'E7']
    _, missing = parse(value, ctx, [row])
    assert statuses(missing)['P1'] == 'unknown'


def test_cyclic_claims_cannot_be_accepted_as_a_joint_dag_even_with_same_time_quotes():
    ctx = context()
    for item in ctx['events']:
        item.update(occurred_from=1, occurred_until=1)
    ctx['relations'].append(relation('cycle', 'c', 'a'))
    row = source()
    _, receipt = parse(full_compact_review(ctx, row), ctx, [row])
    assert statuses(receipt)['P1'] == 'unknown'
    assert any(gap['code'] == 'cyclic_causal_structure' for gap in receipt['gaps'])


@pytest.mark.parametrize('raw', ['normal prose [S1]', '{not json}', '[]', '[' * 2000 + '0' + ']' * 2000])
def test_bad_provider_envelope_has_no_repair_call_and_unknown_receipt(raw):
    answer, receipt = parse_event_response(raw, context(), [source()])
    assert answer == raw
    assert receipt['model_assessed'] is False and receipt['unknown'] == 7


def test_trial_spec_is_compatible_but_arbitrary_action_fields_are_not():
    value = envelope(source())
    value['trial_spec'] = {'hypothesis': 'This is separately controlled'}
    answer, receipt = parse(value)
    assert answer == value['answer'] and receipt['model_assessed'] is True
    value['execute'] = {'tool': 'send_email'}
    with pytest.raises(ValueError, match='^Invalid research response envelope$'):
        parse(value)


@pytest.mark.parametrize('raw', ['{"answer":"ok","answer":"evil"}',
                               '{"answer":"ok","event_assessment":NaN}'])
def test_unsafe_protocol_metadata_is_never_echoed_as_a_normal_answer(raw):
    with pytest.raises(ValueError, match='^Invalid research response envelope$'):
        parse_event_response(raw, context(), [source()])


def test_trial_envelope_is_canonical_only_after_exact_event_wrapper_validation():
    value = envelope(source())
    value['trial_spec'] = {'hypothesis': 'Needs the separate trial validator'}
    extracted = json.loads(event_trial_response('```json\n' + json.dumps(value) + '\n```'))
    assert extracted == {key: value[key] for key in ('answer', 'trial_spec')}
    duplicate = '{"answer":"first","answer":"second","trial_spec":{}}'
    assert event_trial_response(duplicate) == duplicate
    value['execute'] = 'not permitted'
    raw = json.dumps(value)
    assert event_trial_response(raw) == raw


def test_separate_host_task_assessment_survives_event_parser_but_not_trial_schema():
    value = envelope(source())
    value['task_assessment'] = {'host_criterion_review': []}
    answer, receipt = parse(value)
    assert answer == value['answer'] and receipt['model_assessed'] is True
    value['trial_spec'] = {'hypothesis': 'Separately validated'}
    assert set(json.loads(event_trial_response(json.dumps(value)))) == {'answer', 'trial_spec'}


def test_local_gap_pass_preserves_originals_and_adds_at_most_two_whole_windows():
    primary = source('Original primary evidence must stay unchanged.')
    chunks = [source('Customer added scope increased workload on the critical path.', 's2', 'S9'),
              source('Staff leave was an alternative explanation for delivery delay.', 's3', 'S9'),
              source('A completely unrelated astronomy background.', 's4', 'S9')]
    result = supplement_event_gaps('Why was delivery delayed?', [primary], chunks, build_event_obligations(context()))
    assert result['sources'][0] is primary
    assert result['trace']['api_calls'] == 0 and result['trace']['rounds'] == 1
    assert 1 <= len(result['added']) <= 2
    assert 's4' not in result['trace']['added_source_ids']
    assert [row['label'] for row in result['sources']] == ['S1', 'S2', 'S3']
    assert all(row['sentence_boundary_preserved'] for row in result['added'])


def test_gap_trace_records_actual_query_attempts_and_candidate_gains_not_semantic_support():
    primary = source('The original baseline remains unchanged.')
    candidate = source('The critical path record states scope increased workload.', 'candidate', 'S9')
    queries = ['critical path scope workload', 'galaxy telescope stars']
    result = supplement_event_gaps('scope', [primary], [candidate], {'suggested_queries': [None, *queries]})
    assert result['queries'] == queries
    assert result['trace']['attempted_query_indices'] == [0, 1]
    assert result['trace']['covered_query_indices'] == [0]
    assert result['trace']['corpus_attempted_query_indices'] == [0, 1]
    assert 'supported' not in result['trace'] and 'obligations' not in result
    missed = supplement_event_gaps('space', [primary], [candidate], {'suggested_queries': ['quasar cosmology']})
    assert missed['trace']['attempted_query_indices'] == [0]
    assert missed['trace']['covered_query_indices'] == []
    for original, worklist in (([primary], []), ([source(identifier=f'base-{index}', label=f'S{index + 1}')
                                               for index in range(10)], queries)):
        skipped = supplement_event_gaps('scope', original, [candidate], {'suggested_queries': worklist})
        assert skipped['trace']['covered_query_indices'] == []
        assert skipped['trace']['corpus_attempted_query_indices'] == []
        assert skipped['trace']['attempted_query_indices'] == (list(range(len(queries))) if worklist else [])
        assert skipped['trace']['scanned_chunks'] == 0


def test_gap_query_can_locate_an_existing_retained_window_without_adding_sources():
    retained = source('The critical path record states scope increased workload.')
    result = supplement_event_gaps('scope', [retained], [retained],
                                   {'suggested_queries': ['critical path scope workload', 'quasar cosmology']})
    assert result['sources'] == [retained] and result['added'] == []
    assert result['trace']['retained_scanned_windows'] == 1
    assert result['trace']['retained_candidate_query_indices'] == [0]
    assert result['trace']['attempted_query_indices'] == [0, 1]
    assert result['trace']['corpus_attempted_query_indices'] == []
    assert result['trace']['covered_query_indices'] == []
    full = [source(identifier=f'retained-{index}', label=f'S{index + 1}', text=retained['content']) for index in range(10)]
    capped = supplement_event_gaps('scope', full, [source('New unseen corpus.', 'outside', 'S99')],
                                   {'suggested_queries': ['critical path scope workload']})
    assert capped['trace']['status'] == 'capacity_reached'
    assert capped['trace']['retained_candidate_query_indices'] == [0]
    assert capped['trace']['attempted_query_indices'] == [0]
    assert capped['trace']['corpus_attempted_query_indices'] == []
    assert capped['trace']['scanned_chunks'] == 0


def test_every_obligation_keeps_its_own_lookup_including_late_path_and_alternatives():
    row, ctx = source(), context()
    value = {'answer': 'The evidence is not sufficient for a causal conclusion [S1].', 'event_assessment': {'checks': [
        {'id': 'P1', 'verdict': 'unknown', 'missing_query': 'Find the project critical path and actual work hours.'},
        {'id': 'A1', 'verdict': 'unknown', 'missing_query': 'Find actual staffing absences and supplier incidents.'}]}}
    _, receipt = parse(value, ctx, [row])
    by_id = {need['id']: need for need in receipt['obligations']}
    assert by_id['P1']['lookup_query'] == value['event_assessment']['checks'][0]['missing_query']
    assert by_id['A1']['lookup_query'] == value['event_assessment']['checks'][1]['missing_query']
    assert by_id['L1']['lookup_query'].startswith('Customer added scope | Workload increased')
    assert by_id['E1']['lookup_query'].startswith('Customer added scope | original event occurrence record')
    assert all(isinstance(need['lookup_query'], str) and 0 < len(need['lookup_query'].encode()) <= 400
               for need in receipt['obligations'])
    value['event_assessment']['checks'][0]['missing_query'] = '查找关键路径与工时' * 30
    _, bounded = parse(value, ctx, [row])
    assert len(next(need['lookup_query'] for need in bounded['obligations'] if need['id'] == 'P1').encode()) <= 400


def test_append_only_full_budget_never_drops_primary_for_supplement():
    primary = source('p' * 12000)
    result = supplement_event_gaps('delivery scope', [primary], [source(identifier='s2')], build_event_obligations(context()))
    assert result['sources'] == [primary] and result['added'] == []
    assert result['trace']['status'] == 'capacity_reached'


def test_exact_origin_geometry_prioritized_even_without_lexical_overlap():
    primary, target = source('Retain this first.'), source('原始排期表表明追加要求影响了关键路径。', 's2')
    ctx = context()
    ctx['events'][0]['origins'] = [{'kind': 'source', 'source_id': target['id'], 'span': [0, len(target['content'])],
                                  'sha256': hashlib.sha256(target['content'].encode()).hexdigest()}]
    result = supplement_event_gaps('Why delay?', [primary], [target], build_event_obligations(ctx), context=ctx)
    assert result['trace']['added_source_ids'] == ['s2']
    ctx['events'][0]['origins'][0]['sha256'] = '0' * 64
    result = supplement_event_gaps('Why delay?', [primary], [target], build_event_obligations(ctx), context=ctx)
    assert result['added'] == []


@pytest.mark.parametrize('background_count,background_chars', [(2001, 1000), (8193, 24)])
def test_known_origin_lookup_precedes_general_character_and_chunk_scan_limits(background_count, background_chars):
    original = source('The original primary evidence stays intact.')
    target = source('定点来源记录了缺失机制与关键约束。', 'pointed-after-window')
    ctx = context()
    ctx['relations'][0]['origins'] = [{'kind': 'source', 'source_id': target['id'], 'span': [0, len(target['content'])],
        'sha256': hashlib.sha256(target['content'].encode()).hexdigest()}]
    background = [source('z' * background_chars, f'background-{index}') for index in range(background_count)]
    result = supplement_event_gaps('Why delivery delay?', [original], [*background, target],
                                  build_event_obligations(ctx), context=ctx)
    assert result['trace']['added_source_ids'] == ['pointed-after-window']
    assert result['sources'][0] is original
    assert result['trace']['scanned_chars'] <= 2_000_000
    assert result['trace']['scanned_chunks'] <= 8192
    assert result['trace']['identity_lookup_rows'] <= 20000


def test_known_origin_identity_pass_rejects_conflicts_outside_lexical_window():
    original = source('The original primary evidence stays intact.')
    target = source('定点来源记录了缺失机制与关键约束。', 'ambiguous-origin')
    ctx = context()
    ctx['relations'][0]['origins'] = [{'kind': 'source', 'source_id': target['id'], 'span': [0, len(target['content'])],
        'sha256': hashlib.sha256(target['content'].encode()).hexdigest()}]
    background = [source('z' * 24, f'background-{index}') for index in range(8193)]
    conflict = source('另一份同ID来源不应拥有任何默认优先权。', target['id'])
    result = supplement_event_gaps('Why delivery delay?', [original], [target, *background, conflict],
                                  build_event_obligations(ctx), context=ctx)
    assert result['added'] == []


def test_stale_known_hash_does_not_promote_a_source_beyond_general_scope():
    original = source('The original primary evidence stays intact.')
    target = source('Customer added scope delayed delivery on the critical path.', 'stale-origin')
    ctx = context()
    ctx['relations'][0]['origins'] = [{'kind': 'source', 'source_id': target['id'],
        'span': [0, len(target['content'])], 'sha256': '0' * 64}]
    background = [source('z' * 24, f'background-{index}') for index in range(8193)]
    result = supplement_event_gaps('Why delivery delay?', [original], [*background, target],
                                  build_event_obligations(ctx), context=ctx)
    assert result['added'] == []


def test_known_origin_not_present_in_authorized_chunks_is_not_loaded_elsewhere():
    original = source('The original primary evidence stays intact.')
    ctx = context()
    ctx['relations'][0]['origins'] = [{'kind': 'source', 'source_id': 'outside-authorized-input',
        'span': [0, 20], 'sha256': '0' * 64}]
    result = supplement_event_gaps('Why delivery delay?', [original], [], build_event_obligations(ctx), context=ctx)
    assert result['added'] == []


def test_generated_sources_and_conflicting_ids_are_not_supplemented():
    original = source('Unchanged initial evidence.')
    generated = {**source(identifier='s2'), 'provenance': 'llm'}
    duplicate = [source(identifier='s3'), source('Customer added scope introduced workload only if approved.', 's3')]
    result = supplement_event_gaps('delivery delay', [original], [generated, *duplicate], build_event_obligations(context()))
    assert result['added'] == []


def test_relation_origin_beats_event_fact_when_only_one_extra_slot_remains():
    originals = [source(f'Original source number {i} remains intact.', f's{i}', f'S{i}') for i in range(1, 10)]
    event_origin = source('Demand increased at the recorded meeting.', 's10')
    relation_origin = source('关键路径工时与延期之间的明确机制记录。', 's11')
    ctx = context()
    for record, row in ((ctx['events'][0], event_origin), (ctx['relations'][0], relation_origin)):
        record['origins'] = [{'kind': 'source', 'source_id': row['id'], 'span': [0, len(row['content'])],
                              'sha256': hashlib.sha256(row['content'].encode()).hexdigest()}]
    result = supplement_event_gaps('Explain the delay', originals, [event_origin, relation_origin],
                                   build_event_obligations(ctx), context=ctx)
    assert result['trace']['added_source_ids'] == ['s11']
    assert result['sources'][:9] == originals


@pytest.mark.parametrize('primary_count,expected_added', [(8, 2), (9, 1), (10, 0)])
def test_existing_wire_contract_has_two_gap_slots_beyond_eight_primary_rows(primary_count, expected_added):
    originals = [source(f'Original source number {i} remains intact.', f's{i}', f'S{i}') for i in range(1, primary_count + 1)]
    missing = source('Customer added scope increased workload on the critical path.', 'mechanism')
    counter = source('Staff leave was an alternative explanation for delivery delay.', 'counter')
    result = supplement_event_gaps('Why delivery delay?', originals, [missing, counter], build_event_obligations(context()))
    assert result['sources'][:primary_count] == originals
    assert len(result['added']) == expected_added
    assert len(result['sources']) <= 10
    assert sum(len(row['content'].encode()) for row in result['sources']) <= 12000
    if primary_count == 8:
        assert {row['label'] for row in result['added']} == {'S9', 'S10'}
        value = envelope(result['added'][0])
        _, receipt = parse(value, rows=result['sources'])
        assert statuses(receipt)['L1'] == 'model_assessed_support'
    if primary_count == 10:
        assert result['trace']['status'] == 'capacity_reached'


def test_renderer_keeps_actions_conditions_direction_not_binding_secrets():
    ctx = context(['Only if newly approved'])
    ctx['bindings'] = {'private_key': 'must not be model data'}
    rendered = render_event_reasoning(ctx, build_event_obligations(ctx))
    packet = json.loads(rendered)
    assert packet['events'][0]['conditions'] == ['Only if newly approved']
    assert packet['needs'][0]['source_id'] == 'a'
    assert 'private_key' not in rendered and 'origins' not in rendered
    assert len(rendered.encode()) <= 7000
    assert EVENT_RESPONSE_POLICY == event_response_contract() and len(EVENT_RESPONSE_POLICY.encode()) < 3500


def test_renderer_does_not_lose_identity_when_entity_names_were_omitted():
    ctx = context()
    ctx['events'][0].update(actor_id='actor-id', object_id='project-id')
    rendered = json.loads(render_event_reasoning(ctx, build_event_obligations(ctx)))
    assert rendered['events'][0]['actor_id'] == 'actor-id'
    assert rendered['events'][0]['object_id'] == 'project-id'


def test_renderer_interns_shared_conditions_without_truncating_them():
    ctx = context(['Only if ' + 'a' * 240])
    ctx['relations'] = [relation(f'relation-{i}', 'a', 'b') for i in range(12)]
    rendered = render_event_reasoning(ctx, build_event_obligations(ctx))
    packet = json.loads(rendered)
    assert packet['required_conditions'] == {'C1': ctx['events'][0]['conditions'][0]}
    assert packet['events'][0]['conditions'] == ctx['events'][0]['conditions']
    assert all(row['condition_refs'] == ['C1'] for row in packet['needs'] if row['kind'] == 'relation')
    assert len(rendered.encode()) < 7000


def test_followup_query_keeps_both_long_non_ascii_event_anchors_in_byte_budget():
    ctx = context()
    ctx['events'][0]['action'] = '客户追加需求' + '变更范围与确认时间' * 20
    ctx['events'][1]['action'] = '工作量明显增加' + '工时记录与关键路径' * 20
    _, receipt = parse_event_response('No structural assessment.', ctx, [])
    query = receipt['suggested_queries'][0]
    assert '客户追加需求' in query and '工作量明显增加' in query
    assert len(query.encode()) <= 400


def test_relation_bound_to_old_event_revision_is_not_a_current_obligation():
    ctx = context()
    ctx['events'][0]['revision'] = 2
    receipt = build_event_obligations(ctx)
    assert len([row for row in receipt['obligations'] if row['kind'] == 'relation']) == 1
    assert receipt['obligations'][0]['relation_id'] == 'r2'


def test_maximal_feedback_keeps_all_occurrences_and_links_in_bounded_metadata():
    ctx = {'events': [event(f'event-{i}', '记录任务状态与关键约束' * 5, i * 2, i * 2 + 1) for i in range(8)],
           'relations': [relation(f'relation-{i}', f'event-{i % 7}', f'event-{i % 7 + 1}') for i in range(12)]}
    _, receipt = parse_event_response('A provider that ignored JSON.', ctx, [])
    assert len(receipt['obligations']) == 22
    assert len(json.dumps(receipt, ensure_ascii=False, separators=(',', ':')).encode()) <= MAX_RECEIPT_BYTES
    assert len(receipt['gaps']) <= 48 and len(receipt['suggested_queries']) <= 8


@pytest.mark.parametrize('change', ['deleted', 'mutated', 'shrunk', 'moved', 'wrong_label'])
def test_live_event_origin_a_cannot_preserve_an_assessment_quoted_from_stale_b(change):
    origin = source('The event origin A still exists unchanged.', 'origin-a', 'S1')
    proof = source('Source B directly supports the reported relationship.', 'proof-b', 'S2', 100)
    _, receipt = parse(envelope(proof), rows=[origin, proof])
    assert hydrate_event_receipt(receipt, [origin, proof]) == receipt
    rows = [origin, dict(proof)]
    if change == 'deleted':
        rows.pop()
    elif change == 'mutated':
        rows[1]['content'] = rows[1]['content'].replace('directly', 'possibly')
    elif change == 'shrunk':
        rows[1].update(content=proof['content'][:20], excerpt_end=120)
    elif change == 'moved':
        rows[1].update(excerpt_start=101, excerpt_end=101 + len(proof['content']))
    elif change == 'wrong_label':
        rows[1]['label'] = 'S3'
    assert hydrate_event_receipt(receipt, rows) is None
    from docthinker.server.research_evidence_index import hydrate_reasoning
    result = hydrate_reasoning({'mode': 'context_assist', 'claims': [], 'conclusions': [],
                               'trace': {'event_reasoning': receipt}}, rows)
    assert 'event_reasoning' not in result['trace']
    assert result['trace']['event_reasoning_revoked'] is True
    assert result['status'] == 'source_unavailable'


def test_single_hidden_witness_invalidates_receipt_not_just_visible_first_source():
    first = source('First proof remains available and correctly labelled.', 'first', 'S1')
    second = source('Second proof is later deleted from this answer.', 'second', 'S2')
    _, receipt = parse(envelope(first), rows=[first, second])
    target = receipt['obligations'][-1]['evidence']
    target[:] = [{'label': 'S2', 'source_id': second['id'], 'span': [0, len(second['content'])],
                   'quote_sha256': hashlib.sha256(second['content'].encode()).hexdigest()}]
    assert hydrate_event_receipt(receipt, [first, second]) == receipt
    assert hydrate_event_receipt(receipt, [first]) is None


def test_positive_assessment_without_persisted_witness_is_unavailable():
    _, receipt = parse(envelope(source()))
    receipt['obligations'][0]['evidence'] = []
    assert hydrate_event_receipt(receipt, [source()]) is None


def test_hydration_unknown_receipt_without_evidence_is_not_fabricated_proof():
    _, receipt = parse_event_response('No structured provider verdict.', context(), [])
    assert hydrate_event_receipt(receipt, []) == receipt


def test_joint_a_plus_b_witnesses_are_both_persisted_and_b_revocation_hides_judgment():
    first = source('Approval of additional scope was explicitly recorded.', 'a-source', 'S1')
    second = source('Additional scope required more critical-path engineering work.', 'b-source', 'S2')
    value = envelope(first)
    value['event_assessment']['links'][0]['evidence'] = [witness(first), witness(second)]
    _, receipt = parse(value, rows=[first, second])
    need = receipt['obligations'][0]
    assert need['status'] == 'model_assessed_support'
    assert {ref['source_id'] for ref in need['evidence']} == {'a-source', 'b-source'}
    assert hydrate_event_receipt(receipt, [first, second]) == receipt
    assert hydrate_event_receipt(receipt, [first]) is None


def test_condition_witness_dependency_is_not_lost_when_main_link_uses_another_source():
    first = source('A source describes the relationship between scope and work.', 'a-source', 'S1')
    second = source('The critical-path condition actually held for this work.', 'b-source', 'S2')
    ctx = context(['Critical-path condition held'])
    value = envelope(first, [satisfied_condition('Critical-path condition held', second)])
    _, receipt = parse(value, ctx, [first, second])
    assert statuses(receipt)['L1'] == 'model_assessed_support'
    assert {ref['source_id'] for ref in receipt['obligations'][0]['evidence']} == {'a-source', 'b-source'}
    assert hydrate_event_receipt(receipt, [first]) is None


def test_dense_receipt_downgrades_whole_needs_not_positive_partial_proofs():
    rows = [source(f'Independent supporting passage number {i} binds this need.', f'{i:0256}', f'S{i}') for i in range(1, 4)]
    ctx = context()
    ctx['relations'] = [relation(f'{index:036}', 'a', 'b') for index in range(12)]
    proof = [witness(row) for row in rows]
    value = envelope(rows[0])
    value['event_assessment']['links'] = [{'obligation_id': f'L{i + 1}', 'from_event': 'a', 'to_event': 'b',
        'verdict': 'support', 'evidence': proof} for i in range(12)]
    _, receipt = parse(value, ctx, rows)
    assert len(json.dumps(receipt, ensure_ascii=False, separators=(',', ':')).encode()) <= MAX_RECEIPT_BYTES
    assert any('assessment_receipt_budget' in row['gaps'] for row in receipt['obligations'])
    for row in receipt['obligations']:
        if row['kind'] == 'relation' and row['status'] == 'model_assessed_support':
            assert len(row['evidence']) == 3
        if 'assessment_receipt_budget' in row['gaps']:
            assert row['status'] == 'unknown' and row['evidence'] == []
    assert receipt['supported'] == sum(row['status'] == 'model_assessed_support' for row in receipt['obligations'])
    assert statuses(receipt)['P1'] == 'unknown'


def test_contained_condition_quote_is_interned_under_complete_larger_witness():
    row = source('Additional scope required more critical-path engineering work.')
    ctx = context(['Scope changed'])
    value = envelope(row, [satisfied_condition('Scope changed', row) | {
        'evidence': [{'label': 'S1', 'quote': 'critical-path engineering work'}],
        'satisfaction_evidence': [{'label': 'S1', 'quote': 'critical-path engineering work'}]}])
    _, receipt = parse(value, ctx, [row])
    assert statuses(receipt)['L1'] == 'model_assessed_support'
    assert len(receipt['obligations'][0]['evidence']) == 1


def anchored_context(kind='reported_cause'):
    ctx, row = context(), source()
    origin = {'kind': 'source', 'source_id': row['id'], 'span': [0, len(row['content'])],
              'sha256': hashlib.sha256(row['content'].encode()).hexdigest()}
    for event in ctx['events']:
        event['origins'] = [origin]
    for relation in ctx['relations']:
        relation['relation'] = kind
    return ctx, row


@pytest.mark.parametrize('kind,profile,derived', [
    ('reported_cause', 'reported_causality', True), ('hypothesis', 'reported_causality', True),
    ('depends_on', 'feasibility', False), ('before', 'event_context', False),
    ('related_to', 'event_context', False), ('supersedes', 'event_context', False),
])
def test_frontier_semantics_are_relation_types_not_case_or_model_keywords(kind, profile, derived):
    ctx, row = anchored_context(kind)
    prepared = prepare_event_frontier('Workload and delivery', ctx, [row])
    assert prepared['_semantic_profile'] == profile
    assert set(prepared['_active_need_ids']) == ({'L1', 'L2', 'E1', 'E2', 'E3', 'P1', 'A1'} if derived
                                               else {'L1', 'L2', 'E1', 'E2', 'E3'})
    receipt = build_event_obligations(prepared)
    assert ('P1' in statuses(receipt)) is derived
    if not derived:
        assert not {'P1', 'A1'} & set(prepared['_active_need_ids'])
    else:
        assert 'A1' in prepared['_active_need_ids']


def test_inactive_needs_cannot_be_promoted_even_with_valid_extra_model_proofs():
    ctx, row = anchored_context()
    prepared = prepare_event_frontier('Delivery delayed', ctx, [row])
    # Only a concrete capacity limit, not a permanent two-item policy, may
    # defer a need. The validator must still reject extra model judgments.
    prepared['_active_need_ids'].remove('L1')
    _, receipt = parse(envelope(row), prepared, [row])
    active = set(prepared['_active_need_ids'])
    assert len(active) == 6
    for need in receipt['obligations']:
        if need['id'] not in active:
            assert need['status'] == 'unknown'
            assert 'assessment_capacity_exceeded' in need['gaps']
    assert statuses(receipt)['P1'] == 'unknown'


def test_compact_checks_bind_host_direction_and_keep_joint_proof():
    ctx, row = anchored_context()
    prepared = prepare_event_frontier('Delivery delayed', ctx, [row])
    link = next(identifier for identifier in prepared['_active_need_ids'] if identifier.startswith('L'))
    value = {'answer': 'Only the reported relationship is supported [S1].', 'event_assessment': {'checks': [
        {'id': link, 'verdict': 'support', 'evidence': [witness(row)]},
        {'id': 'A1', 'verdict': 'counterevidence', 'alternative': 'Another possible mechanism', 'evidence': [witness(row)]}]}}
    _, receipt = parse(value, prepared, [row])
    assert statuses(receipt)[link] == statuses(receipt)['A1'] == 'model_assessed_support'
    assert statuses(receipt)['P1'] == 'unknown'


def test_complete_frontier_accepts_whole_chain_only_after_every_check_is_bound():
    ctx, row = anchored_context()
    prepared = prepare_event_frontier('Delivery delayed', ctx, [row])
    packet = json.loads(render_event_reasoning(prepared, build_event_obligations(prepared)))
    assert {need['id'] for need in packet['needs']} == {'L1', 'L2', 'E1', 'E2', 'E3', 'P1', 'A1'}
    assert packet['candidate_paths'] == [{'event_ids': ['a', 'b', 'c'], 'obligation_ids': ['L1', 'L2']}]
    checks = [{'id': name, 'verdict': 'support', 'evidence': [witness(row)]} for name in ('L1', 'L2', 'E1', 'E2', 'E3', 'P1')]
    checks.append({'id': 'A1', 'verdict': 'counterevidence', 'alternative': 'Staff leave', 'evidence': [witness(row)]})
    value = {'answer': 'A qualified account of the records [S1].', 'event_assessment': {'checks': checks}}
    _, receipt = parse(value, prepared, [row])
    assert set(statuses(receipt).values()) == {'model_assessed_support'}
    assert receipt['causality_verified'] is False
    for missing in ('L1', 'L2', 'E1', 'E2', 'E3', 'A1'):
        changed = json.loads(json.dumps(value))
        changed['event_assessment']['checks'] = [check for check in checks if check['id'] != missing]
        _, partial = parse(changed, prepared, [row])
        assert statuses(partial)['P1'] == 'unknown'


@pytest.mark.parametrize('problem', ['time_unknown', 'graph_truncated', 'competing_explanation', 'counteredge'])
def test_whole_chain_review_cannot_override_missing_time_capacity_or_counterevidence(problem):
    ctx, row = context(), source()
    value = envelope(row)
    if problem == 'time_unknown':
        ctx['events'][1].update(occurred_from=None, occurred_until=None)
    elif problem == 'graph_truncated':
        ctx['trace']['truncation'] = ['bridge_capacity']
    elif problem == 'competing_explanation':
        value['event_assessment']['alternatives'][0]['verdict'] = 'support'
    else:
        value['event_assessment']['links'][1]['verdict'] = 'counterevidence'
    prepared = prepare_event_frontier('Delivery delayed', ctx, [row])
    _, receipt = parse(value, prepared, [row])
    assert statuses(receipt)['P1'] == 'unknown'
    assert 'chain_not_supported_per_link' in next(need for need in receipt['obligations'] if need['id'] == 'P1')['gaps']


@pytest.mark.parametrize('kind', ['contradicts', 'depends_on'])
def test_causal_path_requires_explicit_conflict_and_prerequisite_resolution(kind):
    ctx, row = context(), source()
    ctx['events'].append(event('d', 'Explicit approval or conflicting project record', 0, 0))
    if kind == 'depends_on':
        ctx['relations'].append(relation('r3', 'b', 'd', kind))
        left, right, resolved_verdict = 'd', 'b', 'support'
    else:
        ctx['relations'].append(relation('r3', 'a', 'd', kind))
        left, right, resolved_verdict = 'a', 'd', 'counterevidence'
    value = envelope(row)
    # The additional endpoint's occurrence is kept even outside the main path.
    value['event_assessment']['events'].append({'obligation_id': 'E4', 'verdict': 'support', 'evidence': [witness(row)]})
    _, missing = parse(value, ctx, [row])
    assert statuses(missing)['P1'] == 'unknown'
    value['event_assessment']['links'].append({'obligation_id': 'L3', 'verdict': resolved_verdict,
        'from_event': left, 'to_event': right, 'evidence': [witness(row)]})
    _, resolved = parse(value, ctx, [row])
    assert statuses(resolved)['P1'] == 'model_assessed_support'
    if kind == 'depends_on':
        missing_occurrence = json.loads(json.dumps(value))
        missing_occurrence['event_assessment']['events'].pop()
        _, absent = parse(missing_occurrence, ctx, [row])
        assert statuses(absent)['E4'] == statuses(absent)['P1'] == 'unknown'
    value['event_assessment']['links'][-1]['verdict'] = 'support' if kind == 'contradicts' else 'counterevidence'
    _, contradicted = parse(value, ctx, [row])
    assert statuses(contradicted)['P1'] == 'unknown'


def test_general_mechanism_support_without_actual_occurrence_cannot_complete_path():
    row = source('Increasing project scope can increase workload and can cause delays in some projects.')
    value = envelope(row)
    value['event_assessment'].pop('events')
    _, receipt = parse(value, context(), [row])
    assert all(statuses(receipt)[name] == 'unknown' for name in ('E1', 'E2', 'E3', 'P1'))


def test_event_types_survive_review_so_a_planned_task_is_not_its_execution():
    ctx, row = anchored_context()
    ctx['events'][0]['event_type'] = 'decision'
    ctx['events'][1]['event_type'] = 'task'
    ctx['events'][2]['event_type'] = 'observation'
    prepared = prepare_event_frontier('Delivery delayed', ctx, [row])
    receipt = build_event_obligations(prepared)
    packet = json.loads(render_event_reasoning(prepared, receipt))
    assert next(item for item in packet['deferred_constraints'] if item['event_id'] == 'a')['event_type'] == 'decision'
    assert next(item for item in packet['events'] if item['id'] == 'b')['event_type'] == 'task'
    assert {item['event_id']: item['event_type'] for item in packet['needs'] if item['kind'] == 'event'} == {
        'a': 'decision', 'b': 'task', 'c': 'observation'}
    assert 'not that its intended action was executed' in EVENT_RESPONSE_POLICY


def test_old_path_receipt_without_occurrence_review_is_not_grandfathered_as_complete():
    row = source()
    _, receipt = parse(envelope(row))
    receipt.pop('causal_review_schema')
    receipt['obligations'] = [item for item in receipt['obligations'] if item['kind'] != 'event']
    migrated = hydrate_event_receipt(receipt, [row])
    assert migrated is not None
    assert migrated['causal_review_schema'] == 2 and statuses(migrated)['P1'] == 'unknown'
    assert 'legacy_path_review_unknown' in next(item for item in migrated['obligations'] if item['id'] == 'P1')['gaps']


def test_long_chain_topology_reaches_every_in_scope_node_not_six_node_prefix():
    ctx = {'events': [event(f'e{index}', f'Mechanism {index}', index * 2, index * 2 + 1) for index in range(8)],
           'relations': [relation(f'r{index}', f'e{index}', f'e{index + 1}') for index in range(7)]}
    prepared = prepare_event_frontier('Mechanism chain', ctx, [])
    receipt = build_event_obligations(prepared)
    packet = json.loads(render_event_reasoning(prepared, receipt))
    assert packet['candidate_paths'][0]['event_ids'] == [f'e{index}' for index in range(8)]
    assert len(packet['needs']) == 17
    assert {need['id'] for need in packet['needs']} == set(prepared['_active_need_ids'])


def test_review_packet_capacity_defers_complete_needs_and_blocks_their_claims(monkeypatch):
    import docthinker.server.research_event_reasoning as module
    ctx, row = anchored_context()
    prepared = prepare_event_frontier('Delivery delayed', ctx, [row])
    full = render_event_reasoning(prepared, build_event_obligations(prepared))
    monkeypatch.setattr(module, 'MAX_PACKET_BYTES', len(full.encode()) - 80)
    bounded = json.loads(render_event_reasoning(prepared, build_event_obligations(prepared)))
    assert bounded['capacity_deferred_need_ids']
    assert len(json.dumps(bounded, separators=(',', ':')).encode()) <= module.MAX_PACKET_BYTES
    _, receipt = parse(envelope(row), prepared, [row])
    for need in receipt['obligations']:
        if need['id'] in bounded['capacity_deferred_need_ids']:
            assert need['status'] == 'unknown' and 'assessment_capacity_exceeded' in need['gaps']
    assert statuses(receipt)['P1'] == 'unknown'


def test_all_compact_checks_are_processed_but_missing_alternative_blocks_chain():
    ctx, row = anchored_context()
    prepared = prepare_event_frontier('Delivery delayed', ctx, [row])
    value = {'answer': 'A cautious answer [S1].', 'event_assessment': {'checks': [
        {'id': identifier, 'verdict': 'support', 'evidence': [witness(row)]} for identifier in ('L1', 'L2', 'P1')]}}
    _, receipt = parse(value, prepared, [row])
    assert statuses(receipt)['L1'] == statuses(receipt)['L2'] == 'model_assessed_support'
    assert statuses(receipt)['P1'] == statuses(receipt)['A1'] == 'unknown'


def test_frontier_routes_to_relevant_bottleneck_and_prioritizes_known_conflict():
    ctx, row = anchored_context()
    prepared = prepare_event_frontier('Delivery delayed', ctx, [row])
    assert 'L2' in prepared['_active_need_ids']
    ctx['events'][0].update(occurred_from=10, occurred_until=11)
    prepared = prepare_event_frontier('Delivery delayed', ctx, [row])
    assert 'L1' in prepared['_active_need_ids']


def test_frontier_keeps_uncovered_personal_memory_facts_not_just_two_nodes():
    ctx = {'events': [event(f'e-{index}', f'User explicitly confirmed personal fact {index}', conditions=[f'Constraint {index}'])
                      for index in range(8)], 'relations': []}
    prepared = prepare_event_frontier('personal facts', ctx, [])
    receipt = build_event_obligations(prepared)
    packet = json.loads(render_event_reasoning(prepared, receipt))
    assert len(packet['needs']) == 8
    assert len(packet['events']) == 8
    assert len(receipt['obligations']) == 8
    assert all(event['action'] in json.dumps(packet) for event in ctx['events'])
    assert all(condition in packet['required_conditions'].values()
               for event in ctx['events'] for condition in event['conditions'])


def test_frontier_keeps_all_condition_and_time_constraints_while_deferring_source_backed_nodes():
    ctx, row = anchored_context()
    ctx['events'][0]['conditions'] = ['Special constraint for the earlier event']
    prepared = prepare_event_frontier('Delivery delayed', ctx, [row])
    packet = json.loads(render_event_reasoning(prepared, build_event_obligations(prepared)))
    assert {need['id'] for need in packet['needs']} == {'L1', 'L2', 'E1', 'E2', 'E3', 'P1', 'A1'}
    assert packet['candidate_paths'] == [{'event_ids': ['a', 'b', 'c'], 'obligation_ids': ['L1', 'L2']}]
    assert len(packet['events']) < len(ctx['events'])
    assert ctx['events'][0]['conditions'][0] in packet['required_conditions'].values()
    assert any(item['event_id'] == 'a' and item['time'] == [1, 2] for item in packet['deferred_constraints'])


def test_source_carried_action_deduplication_preserves_scope_identity_and_negation():
    ctx, row = anchored_context()
    ctx['events'][0].update(actor={'id': 'alice-project-a', 'name': 'Alice'},
                            object={'id': 'project-a', 'name': 'Project A'})
    ctx['events'][1].update(actor={'id': 'alice-project-b', 'name': 'Alice'},
                            object={'id': 'project-b', 'name': 'Project B'})
    # The node is omitted only because its complete assertion remains in its
    # guarded source; its actor/object/time still belong to that exact ID.
    prepared = prepare_event_frontier('Delivery delayed', ctx, [row])
    packet = json.loads(render_event_reasoning(prepared, build_event_obligations(prepared)))
    omitted = next(item for item in packet['deferred_constraints'] if item['event_id'] == 'a')
    assert omitted['actor']['id'] == 'alice-project-a' and omitted['object']['id'] == 'project-a'
    visible = next(item for item in packet['events'] if item['id'] == 'b')
    assert visible['actor']['id'] == 'alice-project-b' and visible['object']['id'] == 'project-b'
    negative = source('Project A scope increase did NOT affect its critical path; Project B suffered staff leave.')
    value = envelope(negative)
    value['event_assessment']['links'][0]['verdict'] = 'counterevidence'
    value['event_assessment']['alternatives'][0]['verdict'] = 'support'
    _, receipt = parse(value, prepared, [negative])
    assert statuses(receipt)['L1'] == statuses(receipt)['A1'] == 'model_assessed_counterevidence'
    assert statuses(receipt)['P1'] == 'unknown'


def test_an_origin_pointer_alone_does_not_erase_an_unquoted_personal_assertion():
    ctx, row = anchored_context()
    ctx['events'].append(event('private', 'The user personally decided to decline this optional project'))
    ctx['events'][-1]['origins'] = ctx['events'][0]['origins']
    prepared = prepare_event_frontier('Delivery delayed', ctx, [row])
    packet = json.loads(render_event_reasoning(prepared, build_event_obligations(prepared)))
    assert any(item['id'] == 'private' and item['action'] == ctx['events'][-1]['action'] for item in packet['events'])


def test_compact_condition_ids_are_resolved_by_host_not_model_rewording():
    ctx, row = anchored_context()
    ctx['events'][1]['conditions'] = ['Critical path condition']
    prepared = prepare_event_frontier('Delivery delayed', ctx, [row])
    packet = json.loads(render_event_reasoning(prepared, build_event_obligations(prepared)))
    need = next(item for item in packet['needs'] if item['kind'] == 'relation')
    value = {'answer': 'A reported relationship, not verified causality [S1].', 'event_assessment': {'checks': [
        {'id': need['id'], 'verdict': 'support', 'evidence': [witness(row)],
         'conditions': [satisfied_condition('ignored model name', row) | {'id': need['condition_refs'][0]}]}]}}
    _, receipt = parse(value, prepared, [row])
    assert statuses(receipt)[need['id']] == 'model_assessed_support'
    value['event_assessment']['checks'][0]['conditions'][0]['id'] = 'C999'
    _, receipt = parse(value, prepared, [row])
    assert statuses(receipt)[need['id']] == 'unknown'


def test_legacy_condition_support_only_supports_the_rule_not_its_fulfillment():
    row, ctx = source('The policy requires explicit approval before proceeding.'), context(['Explicit approval exists'])
    condition = {'condition': 'Explicit approval exists', 'verdict': 'support', 'evidence': [witness(row)]}
    _, receipt = parse(envelope(row, [condition]), ctx, [row])
    need = receipt['obligations'][0]
    assert need['status'] == 'unknown' and 'conditions_unresolved' in need['gaps']
    assert need['condition_count'] == 1
    assert need['condition_checks'][0]['rule_status'] == 'model_assessed_support'
    assert need['condition_checks'][0]['satisfaction'] == 'unknown'
    assert not need['condition_checks'][0]['satisfaction_refs']
    assert statuses(receipt)['P1'] == 'unknown'


@pytest.mark.parametrize('state', [None, 'unknown', 'supported', [], {}, True])
def test_condition_satisfaction_missing_unknown_or_invalid_does_not_promote(state):
    row, ctx = source(), context(['A required prerequisite'])
    check = satisfied_condition('A required prerequisite', row)
    check['satisfaction'] = state
    _, receipt = parse(envelope(row, [check]), ctx, [row])
    assert receipt['obligations'][0]['condition_checks'][0]['satisfaction'] == 'unknown'
    assert statuses(receipt)['L1'] == statuses(receipt)['P1'] == 'unknown'


@pytest.mark.parametrize('satisfaction_evidence', [[], [{'label': 'S99', 'quote': 'Unsupported material'}],
    [{'label': 'S1', 'quote': 'Fabricated prerequisite fulfillment'}],
    [{'label': 'S1', 'quote': 'A source says the prerequisite held.', 'start': 99}]])
def test_met_requires_independent_exact_satisfaction_witnesses(satisfaction_evidence):
    row, ctx = source('A source says the prerequisite held.'), context(['A required prerequisite'])
    check = satisfied_condition('A required prerequisite', row)
    check['satisfaction_evidence'] = satisfaction_evidence
    _, receipt = parse(envelope(row, [check]), ctx, [row])
    assert receipt['obligations'][0]['condition_checks'][0]['satisfaction'] == 'unknown'
    assert statuses(receipt)['L1'] == statuses(receipt)['P1'] == 'unknown'


def test_unmet_does_not_mean_the_condition_rule_itself_is_false():
    rule = source('The service requires account-owner authorization.', 'rule', 'S1')
    state = source('The account owner has deferred authorization for this request.', 'state', 'S2')
    ctx = context(['Account owner authorized this request'])
    condition = {'condition': ctx['events'][0]['conditions'][0], 'verdict': 'support',
        'evidence': [witness(rule)], 'satisfaction': 'unmet', 'satisfaction_evidence': [witness(state)]}
    value = envelope(rule, [condition])
    value['event_assessment']['path']['conditions'] = [condition]
    _, receipt = parse(value, ctx, [rule, state])
    need = receipt['obligations'][0]
    assert need['status'] == 'unknown' and 'conditions_not_met' in need['gaps']
    assert need['condition_checks'][0]['rule_status'] == 'model_assessed_support'
    assert need['condition_checks'][0]['satisfaction'] == 'unmet'
    assert receipt['counterevidence'] == 0 and statuses(receipt)['P1'] == 'unknown'
    assert hydrate_event_receipt(receipt, [rule, state]) == receipt
    assert hydrate_event_receipt(receipt, [rule]) is None


def test_joint_rule_and_satisfaction_dependencies_survive_normalization_and_hydration():
    relation_source = source('The work log supports the proposed scope relationship.', 'link', 'S1')
    rule = source('The workflow requires a recorded supervisor decision.', 'rule', 'S2')
    state = source('The supervisor recorded the decision for this specific request.', 'state', 'S3')
    ctx = context(['Supervisor decided this request'])
    condition = {'condition': ctx['events'][0]['conditions'][0], 'verdict': 'support', 'evidence': [witness(rule)],
                 'satisfaction': 'met', 'satisfaction_evidence': [witness(state)]}
    _, receipt = parse(envelope(relation_source, [condition]), ctx, [relation_source, rule, state])
    need = receipt['obligations'][0]
    assert need['status'] == 'model_assessed_support' and len(need['evidence']) == 3
    check = need['condition_checks'][0]
    assert {need['evidence'][index]['source_id'] for index in check['rule_refs']} == {'rule'}
    assert {need['evidence'][index]['source_id'] for index in check['satisfaction_refs']} == {'state'}
    assert normalize_event_condition_schema(receipt) == receipt
    assert hydrate_event_receipt(receipt, [relation_source, rule, state]) == receipt
    assert hydrate_event_receipt(receipt, [relation_source, rule]) is None


def test_four_independent_condition_premises_downgrade_whole_need_not_partial_proof():
    rows = [source(f'Independent evidence passage for premise {index}.', f'source-{index}', f'S{index + 1}')
            for index in range(4)]
    ctx = context(['Condition A', 'Condition B'])
    conditions = [satisfied_condition('Condition A', rows[1]),
                  satisfied_condition('Condition B', rows[2]) | {'satisfaction_evidence': [witness(rows[3])]}]
    _, receipt = parse(envelope(rows[0], conditions), ctx, rows)
    need = receipt['obligations'][0]
    assert need['status'] == 'unknown' and need['evidence'] == []
    assert 'assessment_receipt_budget' in need['gaps']
    assert all(check['satisfaction'] == 'unknown' and not check['satisfaction_refs']
               for check in need['condition_checks'])
    assert statuses(receipt)['P1'] == 'unknown'


def test_isolated_occurrence_cannot_bypass_its_conditions():
    row = source('A conditional personal decision was recorded.')
    ctx = {'events': [event('isolated', row['content'], conditions=['Required approval'])], 'relations': []}
    value = {'answer': 'The recorded decision is conditional [S1].', 'event_assessment': {'checks': [
        {'id': 'E1', 'verdict': 'support', 'evidence': [witness(row)]}]}}
    _, receipt = parse(value, ctx, [row])
    assert statuses(receipt)['E1'] == 'unknown' and receipt['obligations'][0]['condition_count'] == 1
    assert 'conditions_unresolved' in receipt['obligations'][0]['gaps']


def test_legacy_positive_receipts_migrate_idempotently_without_claiming_met():
    row = source()
    _, current = parse(envelope(row), rows=[row])
    legacy = json.loads(json.dumps(current))
    legacy.pop('condition_schema')
    for need in legacy['obligations']:
        need.pop('condition_count', None)
        need.pop('condition_checks', None)
    migrated = normalize_event_condition_schema(legacy)
    assert legacy['supported'] > 0 and migrated['supported'] == 0
    assert migrated['condition_schema'] == 2 and migrated['legacy_condition_projection'] is True
    assert all(need['status'] == 'unknown' and need['condition_count'] is None for need in migrated['obligations'])
    assert all('legacy_condition_state_unknown' in need['gaps'] for need in migrated['obligations'])
    assert normalize_event_condition_schema(migrated) == migrated
    assert hydrate_event_receipt(legacy, [row]) == migrated


def test_current_schema_cannot_forge_met_without_its_satisfaction_dependency():
    row, ctx = source(), context(['A required prerequisite'])
    _, receipt = parse(envelope(row, [satisfied_condition('A required prerequisite', row)]), ctx, [row])
    receipt['obligations'][0]['condition_checks'][0]['satisfaction_refs'] = []
    assert hydrate_event_receipt(receipt, [row]) is None
    with pytest.raises(ValueError, match='independent evidence'):
        normalize_event_condition_schema(receipt)


def test_duplicate_condition_states_cannot_choose_the_favorable_one():
    row, ctx = source(), context(['A required prerequisite'])
    met = satisfied_condition('A required prerequisite', row)
    _, receipt = parse(envelope(row, [met, met | {'satisfaction': 'unmet'}]), ctx, [row])
    assert statuses(receipt)['L1'] == 'unknown'
    assert receipt['obligations'][0]['condition_checks'][0]['satisfaction'] == 'unknown'


@pytest.mark.parametrize('rule_verdict', ['unknown', 'counterevidence'])
def test_fulfillment_does_not_supply_a_missing_or_contradicted_condition_rule(rule_verdict):
    row, ctx = source(), context(['A required prerequisite'])
    check = satisfied_condition('A required prerequisite', row) | {'verdict': rule_verdict}
    _, receipt = parse(envelope(row, [check]), ctx, [row])
    need = receipt['obligations'][0]
    assert need['condition_checks'][0]['satisfaction'] == 'met'
    assert need['status'] == 'unknown' and 'conditions_unresolved' in need['gaps']


def test_dense_condition_metadata_cannot_leave_positive_without_all_condition_states():
    row = source()
    names = [f'Independently required condition {index}' for index in range(16)]
    ctx = context(names)
    ctx['events'][1]['conditions'] = [f'Middle event condition {index}' for index in range(16)]
    ctx['events'][2]['conditions'] = [f'Final event condition {index}' for index in range(16)]
    names = [name for item in ctx['events'] for name in item['conditions']]
    value = envelope(row, [satisfied_condition(name, row) for name in names])
    value['event_assessment']['path']['conditions'] = [satisfied_condition(name, row) for name in names]
    _, receipt = parse(value, ctx, [row])
    assert len(json.dumps(receipt, ensure_ascii=False, separators=(',', ':')).encode()) <= MAX_RECEIPT_BYTES
    assert any('assessment_receipt_budget' in need['gaps'] for need in receipt['obligations'])
    for need in receipt['obligations']:
        if need['status'] == 'model_assessed_support' and need['condition_count']:
            assert len(need['condition_checks']) == need['condition_count']
        if 'assessment_receipt_budget' in need['gaps']:
            assert need['status'] == 'unknown' and need['evidence'] == []
    assert hydrate_event_receipt(receipt, [row]) == receipt


def test_contract_requires_complete_chain_review_without_equating_sources_to_truth():
    policy = event_response_contract()
    assert 'answer with [S#] citations' in policy
    assert 'host retains ALL listed event needs' in policy
    assert 'checks:[] is valid' in policy and 'no two-check limit' in policy
    assert 'Do not exhaustively prove' in policy
    assert 'Do not recopy quotes or calculate offsets' in policy
    assert '"evidence":[{"label":"S1"}]' in policy
    assert 'complete P1 chain' in policy and 'all required conditions' in policy
    assert 'Omitted/deferred items stay unknown' in policy
    assert 'rule support never implies met' in policy
    assert 'abstain rather than truncate' in policy


def test_empty_sparse_annotations_preserve_normal_answer_and_every_host_obligation():
    ctx, row = context(['A required prerequisite']), source()
    answer = 'This is a normal concise answer, not a proof worksheet [S1].'
    value = {'answer': answer, 'event_assessment': {'checks': []}}
    delivered, receipt = parse(value, ctx, [row])
    assert delivered == answer and receipt['model_assessed'] is False
    assert set(statuses(receipt)) == {'L1', 'L2', 'E1', 'E2', 'E3', 'P1', 'A1'}
    assert receipt['unknown'] == 7 and receipt['supported'] == 0
    assert receipt['obligations'][0]['condition_count'] == 1
    assert 'conditions_unresolved' in receipt['obligations'][0]['gaps']


def test_one_sparse_check_cannot_silently_discharge_other_links_or_a_whole_chain():
    row = source()
    value = {'answer': 'A useful sourced answer [S1].', 'event_assessment': {'checks': [
        {'id': 'L1', 'verdict': 'support', 'evidence': [witness(row)]}]}}
    _, receipt = parse(value, context(), [row])
    assert statuses(receipt)['L1'] == 'model_assessed_support'
    assert all(statuses(receipt)[identifier] == 'unknown' for identifier in ('L2', 'P1', 'A1'))


def test_sparse_unknown_condition_may_omit_witness_but_never_become_met():
    row, ctx = source(), context(['A required prerequisite'])
    value = {'answer': 'A useful sourced answer [S1].', 'event_assessment': {'checks': [
        {'id': 'L1', 'verdict': 'support', 'evidence': [witness(row)],
         'conditions': [{'condition': 'A required prerequisite', 'verdict': 'support',
                        'evidence': [witness(row)], 'satisfaction': 'unknown', 'satisfaction_evidence': []}]}]}}
    _, receipt = parse(value, ctx, [row])
    assert statuses(receipt)['L1'] == 'unknown' and receipt['supported'] == 0
    assert receipt['obligations'][0]['condition_checks'][0]['satisfaction'] == 'unknown'


@pytest.mark.parametrize('fence', [False, True])
def test_terminal_envelope_after_prose_uses_original_answer_and_original_proof_checks(fence):
    row, ctx = source(), context(['A required prerequisite'])
    check = satisfied_condition('A required prerequisite', row) | {'satisfaction': 'unmet'}
    value = envelope(row, [check])
    wire = json.dumps(value)
    raw = 'A prose version before the final response [S1].\n\n' + ('```json\n' + wire + '\n```' if fence else wire)
    answer, receipt = parse_event_response(raw, ctx, [row])
    assert answer == value['answer'] and 'event_assessment' not in answer
    assert statuses(receipt)['L1'] == 'unknown' and 'conditions_not_met' in receipt['obligations'][0]['gaps']
    assert statuses(receipt)['P1'] == 'unknown'
    value['event_assessment']['links'][0]['evidence'][0]['quote'] = 'Fabricated source proof'
    _, failed = parse_event_response('A prose preface.\n' + json.dumps(value), ctx, [row])
    assert statuses(failed)['L1'] == 'unknown' and 'assessment_quote_not_bound' in failed['obligations'][0]['gaps']
