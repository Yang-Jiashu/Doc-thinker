"""A compact frontier cannot lose its omitted facts during final packing."""
import hashlib
import json

import pytest

from docthinker.server.research_event_reasoning import (
    EVENT_RESPONSE_POLICY, build_event_obligations, event_source_guards_live,
    event_source_window_guards, prepare_event_frontier, render_event_reasoning,
)
from docthinker.server.research_prompt_budget import empty_assistance, fit_research_prompt
from docthinker.server.research_retrieval import build_retrieval_plan


def setup_packet():
    question = 'Delivery delayed'
    text = 'Customer added scope. Workload increased. Delivery delayed.'
    extra = {'id': 'event-source', 'document_id': 'event-doc', 'filename': 'events.txt',
             'content': text, 'label': 'S2', 'excerpt_start': 0, 'excerpt_end': len(text),
             'truncated': False, 'ordinal': 0}
    origin = {'kind': 'source', 'source_id': extra['id'], 'span': [0, len(text)],
              'sha256': hashlib.sha256(text.encode()).hexdigest()}
    events = [{'id': identifier, 'revision': 1, 'action': action, 'origins': [origin],
               'occurred_from': index * 2, 'occurred_until': index * 2 + 1, 'conditions': []}
              for index, (identifier, action) in enumerate((('a', 'Customer added scope'),
                  ('b', 'Workload increased'), ('c', 'Delivery delayed')))]
    relations = [{'id': f'r{index}', 'source_id': left, 'target_id': right, 'source_revision': 1,
                  'target_revision': 1, 'relation': 'reported_cause', 'conditions': []}
                 for index, (left, right) in enumerate((('a', 'b'), ('b', 'c')))]
    plan = build_retrieval_plan(question, [{'id': 'baseline', 'document_id': 'baseline-doc',
        'filename': 'baseline.txt', 'content': 'Delivery schedule requires confirmation.'}])
    context = prepare_event_frontier(question, {'events': events, 'relations': relations}, [*plan['sources'], extra])
    rendered = render_event_reasoning(context, build_event_obligations(context))
    guards = event_source_window_guards(context, rendered)
    assert len(guards) == 1 and guards[0]['action'] == 'Customer added scope'
    return question, plan, extra, context, rendered, guards


def fit(question, plan, assistance, rendered, guards, **kwargs):
    return fit_research_prompt(question, plan, [], assistance, system_policy=kwargs.get('system_policy', 'Answer faithfully.'),
        event_context=rendered, event_contract=EVENT_RESPONSE_POLICY, event_source_guards=guards)


def test_optional_assistance_is_shed_before_a_protected_omitted_event_source():
    question, plan, extra, _, rendered, guards = setup_packet()
    assistance = empty_assistance([*plan['sources'], extra])
    assistance['prompt'] = 'Optional auxiliary analysis ' * 2500
    fitted = fit(question, plan, assistance, rendered, guards)
    assert {row['id'] for row in fitted['plan']['sources']} == {'baseline', 'event-source'}
    assert guards[0]['action'] in fitted['prompt']
    assert 'Optional auxiliary analysis' not in fitted['prompt']
    assert event_source_guards_live(guards, fitted['plan']['sources'])
    assert fitted['budget']['event_source_guard_count'] == 1


def test_guard_does_not_change_an_already_fitting_wire_prompt():
    question, plan, extra, _, rendered, guards = setup_packet()
    assistance = empty_assistance([*plan['sources'], extra])
    guarded = fit(question, plan, assistance, rendered, guards)
    old = fit_research_prompt(question, plan, [], assistance, system_policy='Answer faithfully.',
        event_context=rendered, event_contract=EVENT_RESPONSE_POLICY)
    assert guarded['messages'] == old['messages']
    assert guarded['budget']['input_bytes'] == old['budget']['input_bytes']


@pytest.mark.parametrize('change', ['dropped', 'shrunk', 'moved', 'mutated'])
def test_same_id_or_visible_action_cannot_replace_all_origin_geometry(change):
    _, _, extra, _, _, guards = setup_packet()
    final = dict(extra)
    if change == 'dropped':
        rows = []
    elif change == 'shrunk':
        final.update(content='Customer added scope', excerpt_end=len('Customer added scope'))
        rows = [final]
    elif change == 'moved':
        final.update(excerpt_start=1, excerpt_end=1 + len(final['content']))
        rows = [final]
    else:
        final['content'] = final['content'].replace('Workload', 'Worktime')
        rows = [final]
    assert event_source_guards_live(guards, rows) is False


def test_all_sources_of_an_omitted_assertion_must_survive_not_just_one():
    _, _, extra, _, _, guards = setup_packet()
    other = {**extra, 'id': 'other-origin', 'label': 'S3', 'content': 'Second independent event origin.',
             'excerpt_end': len('Second independent event origin.')}
    guards[0]['origins'].append({'kind': 'source', 'source_id': other['id'], 'span': [0, len(other['content'])],
                                'sha256': hashlib.sha256(other['content'].encode()).hexdigest()})
    assert event_source_guards_live(guards, [extra, other])
    assert not event_source_guards_live(guards, [extra])


def test_action_outside_origin_keeps_its_own_original_source_window():
    question, plan, extra, context, _, _ = setup_packet()
    original = {key: value for key, value in context.items() if not key.startswith('_')}
    # The origin records the end of this source, while its asserted action is
    # elsewhere. Keeping only the origin must not silently remove the action.
    suffix = 'Delivery delayed.'
    original['events'][0]['origins'] = [{'kind': 'source', 'source_id': extra['id'],
        'span': [len(extra['content']) - len(suffix), len(extra['content'])],
        'sha256': hashlib.sha256(suffix.encode()).hexdigest()}]
    prepared = prepare_event_frontier(question, original, [*plan['sources'], extra])
    rendered = render_event_reasoning(prepared, build_event_obligations(prepared))
    guards = event_source_window_guards(prepared, rendered)
    assert event_source_guards_live(guards, [*plan['sources'], extra])
    suffix_window = {**extra, 'content': suffix,
        'excerpt_start': len(extra['content']) - len(suffix)}
    # A duplicate literal in an unrelated retained source cannot substitute
    # for the original action-bearing window's source/offset/hash either.
    duplicate = {**plan['sources'][0], 'content': 'Customer added scope',
                 'excerpt_start': 0, 'excerpt_end': len('Customer added scope')}
    assert not event_source_guards_live(guards, [duplicate, suffix_window])


def test_no_fitting_mandatory_packet_fails_before_any_model_call():
    question, plan, extra, _, rendered, guards = setup_packet()
    with pytest.raises(ValueError, match='Research prompt exceeds'):
        fit(question, plan, empty_assistance([*plan['sources'], extra]), rendered, guards, system_policy='x' * 24576)


def test_host_guards_cannot_be_attached_to_a_no_event_request():
    question, plan, _, _, _, guards = setup_packet()
    with pytest.raises(ValueError, match='require event context'):
        fit_research_prompt(question, plan, [], empty_assistance(plan['sources']),
                            system_policy='Answer faithfully.', event_source_guards=guards)
