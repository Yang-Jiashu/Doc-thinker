"""Actual service/SQLite compose→verify wiring; no real provider calls."""
from copy import deepcopy
import hashlib
import json

import pytest

from docthinker.server.research_review_stage import (COMPOSE_EVENT_POLICY, build_review_target,
    review_lineage, select_review_frontier, normalize_review_lineage)
from docthinker.server.research_store import ResearchNotFoundError
from docthinker.server.research_task_review import build_task_requirements
from docthinker.server.model_response_error import ModelResponseError
from tests.test_research_event_pipeline_unit import setup, material, req, packet


def options(manager, sid, *, task=True):
    return {'project_instance_id': manager.get_session(sid)['instance_id'],
            'project_context': 'Answer the original task using the retained records.',
            **({'task_requirements': build_task_requirements(req(sid).question)} if task else {})}


async def compose(service, manager, sid):
    seen = []
    async def writer(prompt, **kwargs):
        seen.append((prompt, kwargs))
        return 'The records describe a scope change and critical-path work [S1]. This is a candidate explanation.'
    answer = await service.answer(req(sid), writer, manager, **options(manager, sid))
    return answer, seen


@pytest.mark.asyncio
@pytest.mark.parametrize('fenced', [False, True])
async def test_compose_metadata_only_tail_and_round_citations_bind_final_body_before_review(setup, fenced):
    service, manager, sid = setup
    material(service, sid)
    original = {row['id']: row['text'] for row in service.all_chunks(sid)}
    calls = []
    body = '🌿 The records report added critical-path work（S1）。This is not a unique causal proof.'
    async def writer(prompt, **kwargs):
        calls.append(prompt)
        annotation = json.dumps({'task_assessment': {'requirements': [
            {'id': 'Q1', 'verdict': 'addressed', 'evidence': [{'label': 'S1'}],
             'answer_witness': {'whole_answer': True}}]}})
        return body + '\n\n' + ('```json\n' + annotation + '\n```' if fenced else annotation)
    try:
        answer = await service.answer(req(sid), writer, manager, **options(manager, sid))
        assert len(calls) == 1 and 'task_assessment' not in answer['answer']
        assert body.replace('（S1）', '[S1]') in answer['answer']
        assert answer['citation_validation']['valid'] and answer['citation_validation']['used_labels'] == ['S1']
        digest = hashlib.sha256(answer['answer'].encode()).hexdigest()
        assert answer['task_review']['answer_sha256'] == review_lineage(answer)['body_sha256'] == digest
        assert answer['task_review']['obligations'][0]['answer_witness'] == {
            'span': [0, len(answer['answer'])], 'quote_sha256': digest}
        assert answer['task_review']['goal_verified'] is False
        assert answer['event_reasoning']['unknown'] == len(answer['event_reasoning']['obligations'])
        assert {row['id']: row['text'] for row in service.all_chunks(sid)} == original
        reviewed = await verify(service, manager, sid, answer, Reviewer())
        assert reviewed['answer'] == answer['answer'] and 'task_assessment' not in reviewed['answer']
        assert reviewed['task_review']['answer_sha256'] == digest
        replay = service.hydrate_history(sid, manager.get_history(sid))[-1]
        assert replay['content'] == answer['answer'] and replay['citation_validation']['valid']
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_round_normalization_never_repairs_old_exact_answer_witness_or_controls(setup):
    service, manager, sid = setup
    material(service, sid)
    body = 'The work was reported（S1）。A control is not evidence [C1].'
    async def writer(prompt, **kwargs):
        return body + '\n' + json.dumps({'task_assessment': {'requirements': [
            {'id': 'Q1', 'verdict': 'addressed', 'evidence': [{'label': 'S1'}],
             'answer_witness': {'quote': body, 'start': 0}}]}})
    try:
        answer = await service.answer(req(sid), writer, manager, **options(manager, sid))
        assert '[C1]' in answer['answer'] and '(S1)' not in answer['answer']
        assert answer['citation_validation']['valid'] is False
        assert answer['citation_validation']['unknown_labels'] == ['C1']
        obligation = answer['task_review']['obligations'][0]
        assert obligation['status'] == 'unknown' and 'answer_witness' not in obligation
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_ordinary_chat_does_not_consume_unrequested_task_json_example(setup):
    service, manager, sid = setup
    raw = 'A schema example follows, not a task verdict.\n```json\n{"task_assessment": {"requirements": []}}\n```'
    async def writer(prompt, **kwargs):
        return raw
    try:
        answer = await service.answer(req(sid, 'Show a harmless JSON annotation schema.'), writer, manager)
        assert answer['answer'] == raw and answer.get('task_review') is None
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_ambiguous_task_tail_is_a_known_response_failure_not_an_unknown_dispatch(setup):
    service, manager, sid = setup
    material(service, sid)
    async def writer(prompt, **kwargs):
        return 'A draft.\n{"task_assessment":{}}\n{"task_assessment":{}}'
    try:
        with pytest.raises(ModelResponseError) as error:
            await service.answer(req(sid), writer, manager, **options(manager, sid))
        assert error.value.code == 'model_response_invalid'
        assert manager.get_history(sid) == []
    finally:
        await service.stop()


def target(answer, manager, sid):
    return build_review_target(answer, question=req(sid).question, session_id=sid,
                               instance_id=manager.get_session(sid)['instance_id'])


class Reviewer:
    def __init__(self, *, unknown=False, callback=None):
        self.calls = []
        self.unknown, self.callback = unknown, callback

    async def __call__(self, prompt, **kwargs):
        data = packet(prompt, 'FROZEN_REVIEW_DATA_JSON:\n')
        self.calls.append(data)
        if self.callback:
            self.callback()
        checks = []
        for need in (data['event_review'] or {}).get('needs', []):
            check = {'id': need['id'], 'verdict': 'unknown' if self.unknown else 'support',
                     'evidence': [] if self.unknown else [{'label': data['source_quotes'][0]['label']}]}
            if need['kind'] == 'alternatives':
                check.update(alternative='Unrecorded staff absence', verdict='unknown' if self.unknown else 'counterevidence')
            checks.append(check)
        return json.dumps({'answer': 'Review only; this text must not replace the frozen answer.',
            'event_assessment': {'checks': checks},
            'task_assessment': {'requirements': [{'id': row['id'],
                'verdict': 'unknown' if self.unknown else 'addressed',
                'evidence': [] if self.unknown else [{'label': data['source_quotes'][0]['label']}],
                'answer_witness': {'whole_answer': True}} for row in data['task_requirements']]}})


async def verify(service, manager, sid, answer, model):
    return await service.answer(req(sid), model, manager, **options(manager, sid),
        analysis_purpose='verify', review_target=target(answer, manager, sid),
        project_parent_answer_ids=[answer['answer_id']])


@pytest.mark.asyncio
async def test_writer_keeps_all_evidence_but_defers_formal_review_and_persists_draft(setup):
    service, manager, sid = setup
    material(service, sid)
    try:
        answer, seen = await compose(service, manager, sid)
        assert len(seen) == 1 and COMPOSE_EVENT_POLICY in seen[0][1]['system_prompt']
        assert '"event_assessment":{"checks"' not in seen[0][1]['system_prompt']
        event = packet(seen[0][0], 'EVENT_REASONING_DATA_JSON:\n')
        assert len(event['needs']) == len(answer['event_reasoning']['obligations']) == 7
        assert answer['event_reasoning']['supported'] == 0 and answer['event_reasoning']['unknown'] == 7
        lineage = review_lineage(answer)
        assert lineage['purpose'] == 'compose' and lineage['root_answer_id'] == answer['answer_id']
        assert lineage['assessed_need_ids'] == []
        replay = service.hydrate_history(sid, manager.get_history(sid))[-1]
        assert replay['content'] == answer['answer'] and target(replay, manager, sid) == target(answer, manager, sid)
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_explicit_review_uses_exact_frozen_windows_no_retrieval_and_never_rewrites_body(setup, monkeypatch):
    service, manager, sid = setup
    material(service, sid)
    try:
        answer, _ = await compose(service, manager, sid)
        def forbidden(*args, **kwargs):
            raise AssertionError('verify must not retrieve/rank new material')
        monkeypatch.setattr(service.events, 'retrieve_context', forbidden)
        monkeypatch.setattr(service, 'all_chunks', forbidden)
        reviewer = Reviewer()
        reviewed = await verify(service, manager, sid, answer, reviewer)
        assert len(reviewer.calls) == 1
        assert len(reviewer.calls[0]['active_event_ids']) + len(reviewer.calls[0]['active_task_ids']) <= 6
        assert 'P1' not in reviewer.calls[0]['active_event_ids']
        assert reviewed['answer'] == answer['answer']
        assert [(row['id'], row['label'], row['content'], row['excerpt_start'], row['excerpt_end']) for row in reviewed['sources']] == [
            (row['id'], row['label'], row['content'], row['excerpt_start'], row['excerpt_end']) for row in answer['sources']]
        assert reviewed['task_review']['answer_sha256'] == review_lineage(answer)['body_sha256']
        assert reviewed['event_reasoning']['supported'] > 0
        assert reviewed['event_reasoning']['unknown'] > 0
        assert review_lineage(reviewed)['root_answer_id'] == answer['answer_id']
        assert review_lineage(reviewed)['context_sha256'] == review_lineage(answer)['context_sha256']
        assert reviewed['reasoning']['trace']['review_batch']['new_retrieval'] is False
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_incremental_review_retains_support_and_only_then_admits_joint_path(setup):
    service, manager, sid = setup
    material(service, sid)
    try:
        answer, _ = await compose(service, manager, sid)
        reviewer = Reviewer()
        first = await verify(service, manager, sid, answer, reviewer)
        old_supported = {row['id'] for row in first['event_reasoning']['obligations'] if row['status'] == 'model_assessed_support'}
        unknown = await verify(service, manager, sid, first, Reviewer(unknown=True))
        assert old_supported <= {row['id'] for row in unknown['event_reasoning']['obligations'] if row['status'] == 'model_assessed_support'}
        second = await verify(service, manager, sid, unknown, reviewer)
        third = await verify(service, manager, sid, second, reviewer)
        assert 'P1' in reviewer.calls[-1]['active_event_ids']
        assert third['event_reasoning']['unknown'] == 0
        assert next(row for row in third['event_reasoning']['obligations'] if row['id'] == 'P1')['status'] == 'model_assessed_support'
        assert third['event_reasoning']['causality_verified'] is False
        assert third['answer'] == answer['answer']
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['empty', 'bad_protocol', 'changed_packet', 'withdrawn'])
async def test_failed_review_never_deletes_parent_and_invalid_target_never_spends(setup, failure):
    service, manager, sid = setup
    doc, _, _ = material(service, sid)
    try:
        answer, _ = await compose(service, manager, sid)
        frozen = target(answer, manager, sid)
        calls = []
        async def model(prompt, **kwargs):
            calls.append(prompt)
            return '' if failure == 'empty' else 'not a review envelope'
        if failure == 'changed_packet':
            frozen['content'] += ' invented edit'
        if failure == 'withdrawn':
            service.store.delete_document(sid, doc['id'])
        with pytest.raises((ValueError, ResearchNotFoundError)):
            await service.answer(req(sid), model, manager, **options(manager, sid), analysis_purpose='verify',
                review_target=frozen, project_parent_answer_ids=[answer['answer_id']])
        assert len(calls) == (0 if failure in {'changed_packet', 'withdrawn'} else 1)
        stored = next(row for row in manager.get_history(sid) if row['id'] == answer['answer_id'])
        assert stored['content'] == answer['answer']
        assert len(manager.get_history(sid)) == 2
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_task_review_without_event_graph_is_a_real_bounded_review(setup):
    service, manager, sid = setup
    doc, _, _ = material(service, sid)
    service.event_reasoning_enabled = False
    try:
        answer, _ = await compose(service, manager, sid)
        reviewer = Reviewer()
        reviewed = await verify(service, manager, sid, answer, reviewer)
        assert reviewer.calls[0]['event_review'] is None
        assert reviewer.calls[0]['active_task_ids'] == ['Q1']
        assert reviewed['task_review']['addressed'] == 1
        assert reviewed['task_review']['goal_verified'] is False
        assert reviewed['answer'] == answer['answer']
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_task_counterevidence_survives_a_later_unknown_same_body_review(setup):
    service, manager, sid = setup
    material(service, sid)
    try:
        answer, _ = await compose(service, manager, sid)
        reviewer = Reviewer()
        async def counter(prompt, **kwargs):
            value = json.loads(await reviewer(prompt, **kwargs))
            for row in value['task_assessment']['requirements']:
                row['verdict'] = 'counterevidence'
            return json.dumps(value)
        contradicted = await verify(service, manager, sid, answer, counter)
        assert contradicted['task_review']['counterevidence'] == 1
        later = await verify(service, manager, sid, contradicted, Reviewer(unknown=True))
        assert later['task_review']['counterevidence'] == 1
        assert later['task_review']['goal_verified'] is False
        assert later['answer'] == answer['answer']
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_personal_task_without_document_sources_reviews_delivery_not_external_truth(setup):
    service, manager, sid = setup
    question = 'Draft a short weekly planning checklist using my confirmed preferences.'
    memory = service.store.create_memory(sid, 'preference', 'Prefer one priority and a short checklist.')
    contexts = {memory['id']: memory['revision']}
    requirements = build_task_requirements(question, input_kind='personal')
    opts = {'document_ids': [], 'project_instance_id': manager.get_session(sid)['instance_id'],
            'project_context': 'Write a proposed weekly checklist, never claim tasks executed.',
            'task_context_refs': contexts, 'task_requirements': requirements}
    calls = []
    async def writer(prompt, **kwargs):
        calls.append('compose')
        return 'Proposed checklist: choose one priority; reserve an hour; review on Friday. Nothing has been scheduled.'
    async def reviewer(prompt, **kwargs):
        calls.append('verify')
        data = packet(prompt, 'FROZEN_REVIEW_DATA_JSON:\n')
        assert data['source_quotes'] == [] and data['active_task_ids'] == ['Q1']
        return json.dumps({'answer': 'Delivery coverage only; no external action.',
            'task_assessment': {'requirements': [{'id': 'Q1', 'verdict': 'addressed',
                'evidence': [], 'answer_witness': {'whole_answer': True}}]}})
    try:
        answer = await service.answer(req(sid, question), writer, manager, **opts)
        frozen = build_review_target(answer, question=question, session_id=sid,
                                     instance_id=opts['project_instance_id'])
        reviewed = await service.answer(req(sid, question), reviewer, manager, **opts,
            analysis_purpose='verify', review_target=frozen, project_parent_answer_ids=[answer['answer_id']])
        assert calls == ['compose', 'verify'] and reviewed['answer'] == answer['answer']
        assert reviewed['task_review']['addressed'] == 1 and reviewed['task_review']['supported'] == 0
        assert reviewed['task_review']['goal_verified'] is False and reviewed['sources'] == []
        assert reviewed['task_review']['obligations'][0]['basis'] == 'answer_fulfilment_only'
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_same_retained_window_cannot_hide_a_changed_approved_document_scope(setup):
    service, manager, sid = setup
    doc, _, _ = material(service, sid)
    try:
        answer, _ = await compose(service, manager, sid)
        additional = service.store.accept_document(sid, 'another-project-record.txt', b'Unrelated record.')
        reviewer = Reviewer()
        with pytest.raises(ValueError, match='input scope'):
            await verify(service, manager, sid, answer, reviewer)
        assert reviewer.calls == []
        # Explicitly keeping the originally approved scope remains allowed.
        reviewed = await service.answer(req(sid), reviewer, manager, **options(manager, sid),
            document_ids=[doc['id']], analysis_purpose='verify', review_target=target(answer, manager, sid),
            project_parent_answer_ids=[answer['answer_id']])
        assert reviewed['answer'] == answer['answer'] and len(reviewer.calls) == 1
    finally:
        await service.stop()


def test_new_counterevidence_blocks_merged_path_without_erasing_other_reviewed_links():
    from docthinker.server.research_event_reasoning import parse_event_response
    from tests.test_research_event_reasoning_unit import context, source, envelope, statuses
    ctx, row, sha = context(), source(), '1' * 64
    _, prior = parse_event_response(json.dumps(envelope(row)), ctx, [row], context_sha256=sha)
    current = {**ctx, '_active_need_ids': ['L1'], '_incremental_review': True}
    raw = json.dumps({'answer': 'Review', 'event_assessment': {'checks': [
        {'id': 'L1', 'verdict': 'counterevidence', 'evidence': [{'label': 'S1'}]}]}})
    _, reviewed = parse_event_response(raw, current, [row], previous=prior, context_sha256=sha)
    assert statuses(reviewed)['L1'] == 'model_assessed_counterevidence'
    assert statuses(reviewed)['L2'] == 'model_assessed_support'
    assert statuses(reviewed)['P1'] == 'unknown'
    _, omitted = parse_event_response('{"answer":"No new proof"}', current, [row],
                                      previous=reviewed, context_sha256=sha)
    assert statuses(omitted)['L1'] == 'model_assessed_counterevidence'
    _, changed = parse_event_response('{"answer":"No new proof"}', current, [row],
                                      previous=reviewed, context_sha256='2' * 64)
    assert all(value == 'unknown' for value in statuses(changed).values())


def test_new_supported_contradiction_is_adverse_not_hidden_by_old_refutation():
    from docthinker.server.research_event_reasoning import parse_event_response
    from tests.test_research_event_reasoning_unit import context, source, envelope, relation, statuses, witness
    ctx, row, sha = context(), source(), '3' * 64
    ctx['relations'].append(relation('r3', 'a', 'c', 'contradicts'))
    value = envelope(row)
    value['event_assessment']['links'].append({'obligation_id': 'L3', 'from_event': 'a', 'to_event': 'c',
                                              'verdict': 'counterevidence', 'evidence': [witness(row)]})
    _, prior = parse_event_response(json.dumps(value), ctx, [row], context_sha256=sha)
    assert statuses(prior)['P1'] == 'model_assessed_support'
    raw = json.dumps({'answer': 'Review', 'event_assessment': {'checks': [
        {'id': 'L3', 'verdict': 'support', 'evidence': [{'label': 'S1'}]}]}})
    _, reviewed = parse_event_response(raw, {**ctx, '_active_need_ids': ['L3'], '_incremental_review': True},
                                       [row], previous=prior, context_sha256=sha)
    assert statuses(reviewed)['L3'] == 'model_assessed_support'
    assert statuses(reviewed)['P1'] == 'unknown'


def test_new_unmet_condition_survives_later_unknown_or_conflicting_same_context_review():
    from docthinker.server.research_event_reasoning import parse_event_response
    from tests.test_research_event_reasoning_unit import context, source, envelope, statuses, satisfied_condition
    ctx, row, sha = context(['Manager approval applies']), source(), '4' * 64
    met = satisfied_condition('Manager approval applies', row)
    value = envelope(row, [met])
    value['event_assessment']['path']['conditions'] = [met]
    _, prior = parse_event_response(json.dumps(value), ctx, [row], context_sha256=sha)
    assert statuses(prior)['L1'] == 'model_assessed_support'
    scope = {**ctx, '_active_need_ids': ['L1'], '_incremental_review': True}
    def review(condition):
        return json.dumps({'answer': 'Review', 'event_assessment': {'checks': [
            {'id': 'L1', 'verdict': 'support', 'evidence': [{'label': 'S1'}], 'conditions': [condition]}]}})
    _, unmet = parse_event_response(review(met | {'satisfaction': 'unmet'}), scope, [row],
                                    previous=prior, context_sha256=sha)
    assert statuses(unmet)['L1'] == statuses(unmet)['P1'] == 'unknown'
    assert unmet['obligations'][0]['condition_checks'][0]['satisfaction'] == 'unmet'
    _, conflicting = parse_event_response(review(met), scope, [row], previous=unmet, context_sha256=sha)
    assert statuses(conflicting)['L1'] == 'unknown'
    assert conflicting['obligations'][0]['condition_checks'][0]['satisfaction'] == 'unmet'
