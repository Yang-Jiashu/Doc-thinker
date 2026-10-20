"""Bound parent evidence gaps feed the next authorized lookup; no paid model."""
import asyncio
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import uuid

import pytest

from docthinker.server.project_evidence_feedback import bind_feedback
from docthinker.server.project_task_bridge import ProjectTaskBridge, _dependency_followup
from docthinker.server.research_task_review import build_task_requirements, parse_task_response
from tests.test_research_task_review_unit import source, envelope
from tests.test_research_mobile_unit import mobile as research_mobile


def uid():
    return str(uuid.uuid4())


def parent():
    req = build_task_requirements('Compare the two causes.', ['Check the proposed alternative.'],
        project_goal='Explain the delay with evidence, not association.',
        project_acceptance_criteria=['Keep counterevidence visible.'])
    value = envelope(req)
    value['task_assessment']['requirements'][0].update(verdict='unknown',
        missing_query='Find staffing absence records and the critical path, not only the change request.')
    text, review = parse_task_response(json.dumps(value), req, [source()])
    task_id, answer_id = uid(), uid()
    binding = bind_feedback(review, task_id=task_id, task_revision=4, step_id='prior', artifact_version=2)
    refs = {'answer_id': answer_id, 'task_review': review, 'task_review_binding': binding}
    answer = {'id': answer_id, 'content': text, 'sources': [source()], 'task_review': review}
    return task_id, refs, answer


def collect(task_id, refs, answer, **changed):
    return _dependency_followup(refs, answer, **{'instance_id': 'private-owner', 'task_id': task_id,
        'step_id': 'prior', 'version': 2, **changed})


def test_current_bound_unknown_requirement_becomes_query_data_not_source_assertion():
    task_id, refs, answer = parent()
    rows = collect(task_id, refs, answer)
    assert len(rows) == 1 and rows[0]['review_kind'] == 'task'
    assert rows[0]['queries'] == refs['task_review']['suggested_queries']
    assert rows[0]['artifact_version'] == 2 and rows[0]['receipt_sha256'] == refs['task_review_binding']['receipt_sha256']
    assert rows[0]['gaps'] == refs['task_review']['gaps']
    assert 'content' not in rows[0] and 'sources' not in rows[0]
    assert source()['content'] not in json.dumps(rows)
    assert rows[0]['priority'] == 'missing_evidence'


@pytest.mark.parametrize('mutation', ['source', 'answer', 'answer_id', 'review', 'binding', 'task', 'step', 'version', 'missing'])
def test_parent_feedback_cannot_outlive_its_answer_witnesses_or_artifact_binding(mutation):
    task_id, refs, answer = parent()
    refs, answer = deepcopy(refs), deepcopy(answer)
    kwargs = {}
    if mutation == 'source':
        answer['sources'][0]['content'] = 'A changed source that no longer matches.'
    elif mutation == 'answer':
        answer['content'] = 'A different answer'
    elif mutation == 'answer_id':
        answer['id'] = uid()
    elif mutation == 'review':
        answer['task_review']['suggested_queries'] = ['A retagged query']
    elif mutation == 'binding':
        refs['task_review_binding']['receipt_sha256'] = 'f' * 64
    elif mutation == 'task':
        kwargs['task_id'] = uid()
    elif mutation == 'step':
        kwargs['step_id'] = 'other'
    elif mutation == 'version':
        kwargs['version'] = 1
    else:
        answer.pop('task_review')
    with pytest.raises(ValueError):
        _dependency_followup(refs, answer, **{'instance_id': 'private-owner', 'task_id': task_id,
            'step_id': 'prior', 'version': 2, **kwargs})


def test_legacy_unbound_review_does_not_gain_lookup_authorization_or_break_plain_handoff():
    task_id, refs, answer = parent()
    refs.pop('task_review_binding')
    assert collect(task_id, refs, answer) == []
    assert collect(task_id, {'answer_id': refs['answer_id']}, answer) == []


def test_current_event_alternative_gaps_are_carried_separately_without_promoting_relation_truth():
    task_id, refs, answer = parent()
    event = {'status': 'evidence_gaps', 'obligations': [{'id': 'A1', 'kind': 'alternatives',
        'status': 'unknown', 'evidence': [], 'gaps': ['alternative_not_found'],
        'condition_count': 0, 'condition_checks': []}],
        'supported': 0, 'counterevidence': 0, 'unknown': 1,
        'gaps': [{'obligation_id': 'A1', 'code': 'alternative_not_found'}],
        'suggested_queries': ['What other documented explanation could account for the delay?'],
        'condition_schema': 2, 'causality_verified': False, 'model_assessed': False}
    answer['event_reasoning'] = event
    refs['event_reasoning'] = event
    refs['evidence_feedback_binding'] = bind_feedback(event, task_id=task_id, task_revision=4,
                                                     step_id='prior', artifact_version=2)
    packets = collect(task_id, refs, answer)
    assert [row['review_kind'] for row in packets] == ['task', 'event']
    assert packets[1]['queries'] == event['suggested_queries']
    assert packets[1]['gaps'] == event['gaps']
    assert 'supported' not in packets[1] and 'causality_verified' not in packets[1]


def test_counterevidence_queries_precede_unassessed_criteria_in_the_same_parent():
    req = build_task_requirements('Compare alternatives', ['Do the records support the proposed cause?'])
    value = envelope(req)
    value['task_assessment']['requirements'][0].update(verdict='unknown', missing_query='Initial missing record')
    value['task_assessment']['requirements'][1].update(verdict='counterevidence',
        missing_query='Resolve the contradictory critical path record')
    _, receipt = parse_task_response(json.dumps(value), req, [source()])
    assert receipt['suggested_queries'][0] == 'Resolve the contradictory critical path record'


def event_parent(rows, *, suggested=()):
    current = source()
    proof = {'label': current['label'], 'source_id': current['id'],
             'span': [current['excerpt_start'], current['excerpt_end']],
             'quote_sha256': hashlib.sha256(current['content'].encode()).hexdigest()}
    obligations = []
    for value in rows:
        row = {'id': value['id'], 'kind': value.get('kind', 'relation'),
               'status': value.get('status', 'unknown'), 'gaps': value.get('gaps', []),
               'condition_count': 0, 'lookup_query': value['lookup_query'], **value}
        row['evidence'] = [] if row['status'] == 'unknown' else [proof]
        obligations.append(row)
    counts = {name: sum(row['status'] == name for row in obligations) for name in (
        'unknown', 'model_assessed_support', 'model_assessed_counterevidence')}
    event = {'status': 'evidence_gaps' if counts['unknown'] or counts['model_assessed_counterevidence'] else 'assessed_with_caveats',
        'obligations': obligations, 'supported': counts['model_assessed_support'],
        'counterevidence': counts['model_assessed_counterevidence'], 'unknown': counts['unknown'],
        'gaps': [{'obligation_id': row['id'], 'code': code} for row in obligations for code in row['gaps']][:48],
        'suggested_queries': list(suggested), 'condition_schema': 2, 'causal_review_schema': 2,
        'causality_verified': False, 'model_assessed': bool(counts['model_assessed_support'] + counts['model_assessed_counterevidence'])}
    task_id, answer_id = uid(), uid()
    refs = {'answer_id': answer_id, 'event_reasoning': event,
            'evidence_feedback_binding': bind_feedback(event, task_id=task_id, task_revision=4,
                step_id='prior', artifact_version=2)}
    answer = {'id': answer_id, 'content': 'A candidate explanation, not a verified cause [S1].',
              'sources': [current], 'event_reasoning': event}
    return task_id, refs, answer


@pytest.mark.parametrize('status', ['unknown', 'model_assessed_counterevidence'])
def test_no_gap_unknown_and_counterevidence_still_seed_their_own_parent_lookup(status):
    task_id, refs, answer = event_parent([{'id': 'L1', 'status': status,
        'lookup_query': 'Find the actual critical path record that tests this contrary evidence.'}])
    assert refs['event_reasoning']['gaps'] == []
    rows = collect(task_id, refs, answer)
    assert len(rows) == 1
    assert rows[0]['queries'] == [refs['event_reasoning']['obligations'][0]['lookup_query']]
    assert rows[0]['priority'] == ('counterevidence' if status == 'model_assessed_counterevidence' else 'missing_evidence')


@pytest.mark.parametrize('status,adverse', [('model_assessed_support', True), ('model_assessed_counterevidence', False)])
def test_explicit_conflict_uses_typed_polarity_not_the_receipt_counter_count(status, adverse):
    task_id, refs, answer = event_parent([{'id': 'L1', 'relation': 'contradicts', 'status': status,
        'lookup_query': 'Resolve the explicit conflict between the two project records.'}])
    rows = collect(task_id, refs, answer)
    if adverse:
        assert len(rows) == 1 and rows[0]['priority'] == 'counterevidence'
    else:
        assert rows == []  # Refuting a conflict does not create adverse work.


def test_resolved_conflict_does_not_take_priority_from_a_real_unknown_need():
    task_id, refs, answer = event_parent([
        {'id': 'L1', 'relation': 'contradicts', 'status': 'model_assessed_counterevidence',
         'lookup_query': 'Already refuted conflict should not be searched again.'},
        {'id': 'L2', 'status': 'unknown', 'lookup_query': 'Find the still missing work log.'}])
    rows = collect(task_id, refs, answer)
    assert rows[0]['priority'] == 'missing_evidence'
    assert rows[0]['queries'] == ['Find the still missing work log.']


def test_late_path_and_alternative_queries_survive_complete_handoff_before_fair_bounding():
    from docthinker.server.research_task_review import build_task_followup
    needs = [{'id': f'L{index}', 'lookup_query': f'Find the full original evidence for relationship {index}.',
              'gaps': ['semantic_support_unassessed']} for index in range(1, 13)]
    needs += [{'id': f'E{index}', 'kind': 'event', 'lookup_query': f'Find event occurrence record {index}.',
               'gaps': ['event_occurrence_unassessed']} for index in range(1, 9)]
    needs += [{'id': 'P1', 'kind': 'path', 'lookup_query': 'Find the missing intermediate mechanism for the entire chain.',
               'gaps': ['path_support_unassessed']},
              {'id': 'A1', 'kind': 'alternatives', 'lookup_query': 'Find the strongest competing explanation in staffing records.',
               'gaps': ['alternative_explanation_unassessed']}]
    task_id, refs, answer = event_parent(needs, suggested=[row['lookup_query'] for row in needs[:8]])
    rows = collect(task_id, refs, answer)
    assert len(rows[0]['queries']) == 22
    assert rows[0]['queries'][:2] == [needs[-2]['lookup_query'], needs[-1]['lookup_query']]
    delivered = build_task_followup(rows)['dependencies'][0]
    assert len(delivered['queries']) == 8 and delivered['omitted_query_count'] == 14
    assert delivered['queries'][:2] == rows[0]['queries'][:2]
    assert set(delivered['queries']) <= {row['lookup_query'] for row in needs}


def test_pure_task_counterevidence_is_not_dropped_when_its_gap_list_is_empty():
    req = build_task_requirements('Compare the two proposed causes.')
    value = envelope(req)
    value['task_assessment']['requirements'][0].update(verdict='counterevidence',
        missing_query='Resolve the signed critical-path correction before drafting.')
    text, receipt = parse_task_response(json.dumps(value), req, [source()])
    receipt['gaps'] = []
    for row in receipt['obligations']:
        row['gaps'] = []
    task_id, answer_id = uid(), uid()
    refs = {'answer_id': answer_id, 'task_review': receipt,
            'task_review_binding': bind_feedback(receipt, task_id=task_id, task_revision=4,
                step_id='prior', artifact_version=2)}
    answer = {'id': answer_id, 'content': text, 'sources': [source()], 'task_review': receipt}
    rows = collect(task_id, refs, answer)
    assert rows[0]['priority'] == 'counterevidence'
    assert rows[0]['queries'] == ['Resolve the signed critical-path correction before drafting.']


@pytest.mark.parametrize('change', ['source', 'query', 'artifact'])
def test_event_lookup_cannot_bypass_source_or_exact_receipt_binding(change):
    task_id, refs, answer = event_parent([{'id': 'L1', 'status': 'model_assessed_counterevidence',
        'lookup_query': 'Resolve the actual counterevidence.'}])
    kwargs = {}
    if change == 'source':
        answer['sources'][0]['content'] = 'The source has been replaced.'
    elif change == 'query':
        refs['event_reasoning']['obligations'][0]['lookup_query'] = 'A substituted lookup cannot borrow the old receipt hash.'
    else:
        kwargs['version'] = 1
    with pytest.raises(ValueError):
        collect(task_id, refs, answer, **kwargs)


def test_bridge_passes_validated_followup_and_full_top_level_requirements_to_service(tmp_path, monkeypatch):
    from docthinker.server.request_journal import RequestJournal
    from docthinker.server.routers import mobile
    from tests.test_project_task_acceptance_unit import _workspace

    async def scenario():
        manager, service, sid, docs = await _workspace(tmp_path)
        bridge = ProjectTaskBridge(service, manager, sid, object())
        bridge.instance_id = manager.get_session(sid)['instance_id']
        task_id, refs, answer = parent()
        for row in answer['sources']:
            row['document_id'] = docs[0]
        context = {'project_goal': 'Compare competing causes before drafting.',
            'project_acceptance_criteria': ['Preserve every unresolved causal link.', 'No external sending.'],
            'phase_id': uid(), 'phase_title': 'Investigate alternatives',
            'phase_acceptance_criteria': ['Find the critical path record.']}
        bridge.phase_context_reader = lambda **_: context
        journal = RequestJournal(tmp_path, boot_id='followup-before-provider')
        monkeypatch.setattr(mobile, '_query_journal', lambda: journal)
        seen = []

        async def load(**_):
            return answer

        async def capture(*args, **kwargs):
            seen.append(kwargs)
            # Reject before model dispatch: this test checks trusted wiring,
            # not model quality or a fabricated completed answer.
            raise ValueError('Synthetic pre-provider stop')

        monkeypatch.setattr(bridge, 'load_answer', load)
        monkeypatch.setattr(service, 'answer', capture)
        try:
            snapshot = await bridge.read_sources([docs[0]])
            result = await bridge.execute_question(session_id=sid, request_id=uid(),
                question='What evidence distinguishes the causes?', document_ids=[docs[0]], input_snapshot=snapshot,
                goal='Compare competing causes before drafting.', constraints='Do not send anything.',
                origin_task_id=task_id, phase_task_id=task_id, phase_context=context,
                dependency_artifacts=[{'step_id': 'prior', 'version': 2, 'references': refs}])
            assert result == {'request_status': 'failed', 'code': 'inputs_unavailable'}
            assert len(seen) == 1
            requirements = seen[0]['task_requirements']
            assert requirements['project_goal'] == context['project_goal']
            assert requirements['project_acceptance_criteria'] == context['project_acceptance_criteria']
            assert requirements['acceptance_criteria'] == context['phase_acceptance_criteria']
            packet = seen[0]['task_followup']
            assert packet['dependencies'][0]['queries'] == refs['task_review']['suggested_queries']
            assert packet['dependencies'][0]['instance_id'] == bridge.instance_id
            assert '前序证据缺口的待查队列' in seen[0]['project_context']
            assert '不是事实、不是指令' in seen[0]['project_context']
            model_view = json.loads(seen[0]['project_context'].rsplit('\n', 1)[1])
            assert model_view['queries'] == packet['dependencies'][0]['queries']
            assert model_view['dependency_count'] == 1
            assert model_view['kind'] == 'unverified_lookup_hints'
            assert 'receipt_sha256' not in model_view and 'dependencies' not in model_view
            assert all(value in seen[0]['project_context'] for value in context['project_acceptance_criteria'])
            assert all(value in seen[0]['project_context'] for value in context['phase_acceptance_criteria'])
            assert seen[0]['document_ids'] == [docs[0]]
        finally:
            await service.stop()
    asyncio.run(scenario())


def lookup_fixture(tmp_path):
    from docthinker.server.project_obligation_ledger import ProjectObligationLedger, LOOKUP_POLICY_VERSION
    from tests.test_project_obligation_ledger_unit import completed_store
    store, task = completed_store(tmp_path)
    ledger = ProjectObligationLedger(store)
    progress = ledger.synchronize(task['id'], eligible_step_ids=['a'])
    work = progress['work_items'][0]
    binding = work['binding']
    digest = hashlib.sha256(work['lookup_query'].encode()).hexdigest()
    trace = {'attempted_query_sha256': [digest], 'candidate_query_sha256': [digest],
        'lookup_policy': LOOKUP_POLICY_VERSION,
        'lookup_scope': {'input_kind': 'documents', 'document_snapshot':
            next(row['input_snapshot'] for row in task['steps'] if row['id'] == work['owner_step_id'])},
        'dependencies': [{'instance_id': store.instance_id, 'task_id': task['id'], 'step_id': 'a',
            'artifact_version': binding['artifact_version'], 'answer_id': binding['answer_id'],
            'review_kind': 'task', 'receipt_sha256': binding['receipt_sha256']}]}
    bridge = SimpleNamespace(session_id=store.session_id, instance_id=store.instance_id,
        phase_guard=None, manager=SimpleNamespace(base_storage_path=tmp_path,
        get_session=lambda _: {'instance_id': store.instance_id}), validate_inputs=lambda _: True)
    return bridge, ledger, task, work, trace


def test_lookup_outcome_is_idempotent_bookkeeping_not_goal_or_source_verification(tmp_path):
    from docthinker.server.project_evidence_controller import record_parent_lookups
    bridge, ledger, task, work, trace = lookup_fixture(tmp_path)
    first = record_parent_lookups(bridge, trace)
    assert first['status'] == 'recorded' and first['recorded_count'] >= 1
    before = ledger.read(task['id'], eligible_step_ids=['a'])
    record_parent_lookups(bridge, trace)
    after = ledger.read(task['id'], eligible_step_ids=['a'])
    assert after == before
    processed = next(row for row in after['work_items'] if row['id'] == work['id'])
    assert processed['attempts'] == 1 and processed['status'] == 'candidates_found'
    assert after['coverage']['ready_for_review'] == 0 and after['goal_verified'] is False


def test_already_attempted_parent_hints_cannot_starve_later_untried_evidence_needs(tmp_path):
    from docthinker.server.project_evidence_controller import record_parent_lookups, prioritize_parent_queries
    bridge, _, _, work, trace = lookup_fixture(tmp_path)
    record_parent_lookups(bridge, trace)
    row = {**trace['dependencies'][0], 'priority': 'missing_evidence',
        'queries': [work['lookup_query'], 'A different untried original record'], 'gaps': [],
        'omitted_query_count': 0, 'omitted_gap_count': 0}
    reordered = prioritize_parent_queries(bridge, [row])
    assert reordered[0]['queries'] == ['A different untried original record', work['lookup_query']]
    assert row['queries'][0] == work['lookup_query']


@pytest.mark.parametrize('change', ['source', 'instance', 'task', 'step', 'artifact', 'receipt', 'unattempted', 'candidate'])
def test_unknown_or_revoked_lookup_parent_cannot_receive_an_attempt(tmp_path, change):
    from docthinker.server.project_evidence_controller import record_parent_lookups
    bridge, ledger, task, work, trace = lookup_fixture(tmp_path)
    if change == 'source':
        bridge.validate_inputs = lambda _: False
    elif change == 'instance':
        trace['dependencies'][0]['instance_id'] = 'another-instance'
    elif change == 'task':
        trace['dependencies'][0]['task_id'] = uid()
    elif change == 'step':
        trace['dependencies'][0]['step_id'] = 'other'
    elif change == 'artifact':
        trace['dependencies'][0]['artifact_version'] = 2
    elif change == 'receipt':
        trace['dependencies'][0]['receipt_sha256'] = 'f' * 64
    elif change == 'unattempted':
        trace['attempted_query_sha256'] = trace['candidate_query_sha256'] = ['f' * 64]
    else:
        trace['candidate_query_sha256'] = ['f' * 64]
    result = record_parent_lookups(bridge, trace)
    assert result['recorded_count'] == 0
    progress = ledger.read(task['id'], eligible_step_ids=['a'])
    assert all(row['attempts'] == 0 for row in progress['work_items'])


@pytest.mark.asyncio
async def test_real_service_parent_lookup_survives_rehydration_without_new_model_calls(research_mobile):
    from docthinker.session_manager import SessionManager
    from docthinker.server.routers import project_tasks
    from tests.test_research_mobile_unit import mobile_client, upload_ready, SID
    from tests.test_personal_tasks_mobile_unit import create, start, settled
    mobile = research_mobile
    manager = SessionManager(str(mobile.root / '_system'), str(mobile.root), migrate_legacy=False)
    assert manager.create_session('Current parent lookup receipts')['id'] == SID
    mobile.manager = manager
    mobile.state.session_manager = manager
    lookup = 'Orion staffing absence roster and the approved critical path original records'
    calls = []

    async def response(prompt, **_):
        packet = json.JSONDecoder().raw_decode(prompt.split('TASK_REQUIREMENTS_DATA_JSON:\n', 1)[1].lstrip())[0]
        calls.append(prompt)
        return json.dumps({'answer': 'The Orion work log records additional work; its causal effect remains unproven [S1].',
            'task_assessment': {'requirements': [{'id': row['id'], 'verdict': 'unknown',
                'missing_query': lookup} for row in packet['requirements']]}})

    mobile.model.side_effect = response
    try:
        async with mobile_client(mobile) as browser:
            doc = await upload_ready(mobile, browser, filename='worklog.txt', text=
                'Orion delivery work log records added work. Staffing absence and critical path effects are not confirmed.')
            roster = await upload_ready(mobile, browser, filename='roster.txt', text=
                'Staffing absence roster registers two days of leave. The approved critical path record does not establish overlap.')
            task = await create(browser, {'session_id': SID, 'goal': 'Investigate Orion delay without sending a message.',
                'constraints': 'Keep alternatives and missing evidence visible.', 'questions': [
                    {'id': 'facts', 'question': 'Analyze the Orion delivery work log.',
                     'document_ids': [doc, roster]},
                    {'id': 'brief', 'question': 'Draft an Orion delivery brief, distinguishing facts and unknowns.',
                     'document_ids': [doc, roster], 'depends_on': ['facts']}]})
            await start(browser, task, calls=2)
            done = await settled(browser, task['id'])
            assert done['status'] == 'completed' and mobile.model.await_count == len(calls) == 2
            _, runtime, bridge = await project_tasks._context(SID)
            initial_sources = runtime.store.get_artifact(task['id'], 'facts')['references']['document_ids']
            supplemented_sources = runtime.store.get_artifact(task['id'], 'brief')['references']['document_ids']
            assert initial_sources == [doc]  # not available in the first model context
            assert roster in supplemented_sources  # recovered by the scoped gap pass
            completed = next(row for row in done['steps'] if row['id'] == 'brief')
            # Reconcile reads the real persisted answer/trace; it cannot call
            # the model or grant a new run. Both replays must be the same work.
            await bridge.reconcile_request(session_id=SID, request_id=completed['request_id'])
            before = runtime.review_current_feedback(runtime.store.get_task(task['id']))['evidence_progress']
            attempted = [row for row in before['work_items'] if row['owner_step_id'] == 'facts' and row['attempts']]
            assert attempted and all(row['attempts'] == 1 for row in attempted)
            assert any(row['status'] == 'candidates_found' for row in attempted)
            assert before['coverage']['ready_for_review'] == 0 and before['goal_verified'] is False
            await bridge.reconcile_request(session_id=SID, request_id=completed['request_id'])
            after = runtime.review_current_feedback(runtime.store.get_task(task['id']))['evidence_progress']
            assert after == before and mobile.model.await_count == 2
    finally:
        await project_tasks.shutdown_project_tasks()
        await mobile.service.stop()
