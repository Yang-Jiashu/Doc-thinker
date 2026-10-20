"""Evidence work changes actual scoped plans, not just advisory report text."""
from copy import deepcopy
import json

import pytest

from docthinker.server.project_evidence_controller import (
    retrieval_worklist, repair_plan, causal_delivery_state,
)
from docthinker.server.project_task_runtime import ProjectTaskRuntime
from docthinker.server.project_task_store import ProjectTaskConflictError
from tests.test_project_obligation_ledger_unit import completed_store
from tests.test_research_mobile_unit import mobile as research_mobile


def task():
    return {'id': 'owned-task', 'revision': 4, 'goal': 'Explain the delay without sending anything.',
        'constraints': 'Only selected internal records.', 'steps': [
            {'id': 'inventory', 'kind': 'inventory'},
            {'id': 'cause', 'kind': 'analysis', 'title': 'Compare explanations',
             'question': 'Was the change on the critical path?', 'document_ids': ['owned-document'],
             'depends_on': ['inventory']},
            {'id': 'draft', 'kind': 'analysis', 'title': 'Private draft', 'question': 'Draft a private note.',
             'input_kind': 'personal', 'memory_ids': ['owned-memory'], 'depends_on': []}]}


def worklist(value, *, owner='cause', query='Compare the approved critical path and absence roster.'):
    return {'task_id': value['id'], 'task_revision': value['revision'], 'work_items': [
        {'id': 'host-work', 'owner_step_id': owner, 'kind': 'event_path',
         'query': query, 'status': 'unresolved'}]}


def test_work_selection_fairly_consumes_parent_gaps_events_and_all_goals():
    packet, trace = retrieval_worklist({'suggested_queries': [f'event{i}' for i in range(8)]},
        followup_queries=[f'parent{i}' for i in range(16)],
        requirement_queries=[f'goal{i}' for i in range(26)])
    assert packet['suggested_queries'] == ['parent0', 'event0', 'goal0', 'parent1', 'event1', 'goal1', 'parent2', 'event2']
    assert trace['deferred_queries'] == 42
    assert trace['automatic_model_retry'] is False
    assert trace['scheduled_by_kind'] == {'parent_gap': 3, 'event_gap': 3, 'requirement': 2}


def test_queries_deduplicate_without_losing_original_requirement_or_claiming_completion():
    packet, trace = retrieval_worklist({'obligations': [{'id': 'P1', 'status': 'unknown'}],
        'suggested_queries': ['same', 'event']}, followup_queries=['same', 'parent'],
        requirement_queries=['same', 'goal'])
    assert len(packet['suggested_queries']) == len(set(packet['suggested_queries'])) == 4
    assert packet['obligations'] == [{'id': 'P1', 'status': 'unknown'}]
    assert trace['lookup_only'] is True


def test_repair_keeps_original_goal_scope_dependencies_and_independent_branch():
    value = task()
    before = deepcopy(value)
    repair = repair_plan(value, worklist(value))
    plan = repair['plan']
    assert value == before and plan['goal'] == value['goal'] and plan['constraints'] == value['constraints']
    original, independent, audit = plan['questions']
    assert original['question'] == value['steps'][1]['question']
    assert audit['question'] == original['question'] and audit['document_ids'] == ['owned-document']
    assert audit['depends_on'] == ['cause'] and audit['purpose'] == 'verify' and audit['review_of'] == 'cause'
    assert independent['input_kind'] == 'personal' and independent['memory_ids'] == ['owned-memory']
    assert independent['question'] == value['steps'][2]['question']
    assert repair['changed_steps'] == [audit['id']] and repair['execution_authorized'] is False


def test_repair_never_truncates_a_long_original_question_or_silently_drops_need():
    value = task()
    value['steps'][1]['question'] = '原' * 1800
    repair = repair_plan(value, worklist(value))
    assert repair['plan']['questions'][0]['question'] == '原' * 1800
    assert repair['plan']['questions'][-1]['question'] == '原' * 1800
    assert repair['plan']['questions'][-1]['purpose'] == 'verify'
    assert repair['changed_steps'] == [repair['plan']['questions'][-1]['id']]
    assert repair['capacity_steps'] == [] and repair['next_action'] == 'preview_replan'


def test_wrong_revision_work_cannot_change_a_plan():
    value = task()
    progress = worklist(value)
    progress['task_revision'] -= 1
    with pytest.raises(ValueError):
        repair_plan(value, progress)


def test_exhausted_same_version_lookup_requests_new_evidence_not_another_paid_pass():
    value = task()
    progress = worklist(value)
    progress['work_items'][0].update(status='exhausted', attempts=1, round=1)
    progress['next_action'] = 'collect_evidence'
    repair = repair_plan(value, progress)
    assert repair['changed_steps'] == [] and repair['next_action'] == 'collect_evidence'
    assert repair['deferred_work_count'] == 1
    assert repair['plan']['questions'][0]['question'] == value['steps'][1]['question']


@pytest.mark.parametrize('coverage', ['retained_only', 'limited_corpus', 'legacy_trace'])
def test_partial_or_unknown_scan_cannot_permanently_exhaust_parent_evidence(tmp_path, coverage):
    from docthinker.server.project_evidence_controller import record_parent_lookups
    from tests.test_project_task_followup_unit import lookup_fixture
    bridge, ledger, saved, work, trace = lookup_fixture(tmp_path)
    trace['candidate_query_sha256'] = []
    if coverage == 'legacy_trace':
        trace.pop('corpus_attempted_query_sha256', None)
        trace.pop('corpus_scan_limited', None)
    else:
        trace['corpus_attempted_query_sha256'] = (trace['attempted_query_sha256']
            if coverage == 'limited_corpus' else [])
        trace['corpus_scan_limited'] = coverage == 'limited_corpus'
    result = record_parent_lookups(bridge, trace)
    assert result['recorded_count'] >= 1
    current = next(row for row in ledger.read(saved['id'], eligible_step_ids=['a'])['work_items']
                   if row['id'] == work['id'])
    assert current['attempts'] == 1 and current['status'] == 'partially_searched'


def test_only_completed_current_corpus_scan_can_exhaust_an_unmatched_query(tmp_path):
    from docthinker.server.project_evidence_controller import record_parent_lookups
    from tests.test_project_task_followup_unit import lookup_fixture
    bridge, ledger, saved, work, trace = lookup_fixture(tmp_path)
    trace.update(candidate_query_sha256=[], corpus_scan_limited=False,
                 corpus_attempted_query_sha256=list(trace['attempted_query_sha256']))
    result = record_parent_lookups(bridge, trace)
    assert result['recorded_count'] >= 1
    current = next(row for row in ledger.read(saved['id'], eligible_step_ids=['a'])['work_items']
                   if row['id'] == work['id'])
    assert current['attempts'] == 1 and current['status'] == 'exhausted'


@pytest.mark.parametrize('state', ['unknown', 'model_assessed_counterevidence'])
def test_causal_counterevidence_or_missing_path_cannot_admit_definitive_conclusion(state):
    receipt = {'obligations': [
        {'id': 'L1', 'kind': 'relation', 'causal_candidate': True, 'status': 'model_assessed_support', 'gaps': []},
        {'id': 'P1', 'kind': 'path', 'status': state, 'gaps': ['missing_link']},
        {'id': 'A1', 'kind': 'alternatives', 'status': 'model_assessed_support', 'gaps': []}]}
    admission = causal_delivery_state(receipt)
    assert admission['status'] == ('counterevidence' if state.endswith('counterevidence') else 'needs_evidence')
    assert admission['policy'] == 'no_definitive_causal_conclusion'
    assert admission['definitive_causal_conclusion_allowed'] is False


def test_complete_model_assessment_still_requires_review_not_a_causal_truth_certificate():
    receipt = {'obligations': [{'id': key, 'kind': kind, 'causal_candidate': key == 'L1',
        'status': 'model_assessed_support', 'gaps': []} for key, kind in
        [('L1', 'relation'), ('P1', 'path'), ('A1', 'alternatives')]]}
    result = causal_delivery_state(receipt)
    assert result['status'] == 'ready_for_review' and result['semantic_verified'] is False
    assert result['causality_verified'] is False


def test_refuting_an_alleged_conflict_is_not_counterevidence_against_the_main_chain():
    rows = [{'id': key, 'kind': kind, 'causal_candidate': key == 'L1',
             'status': 'model_assessed_support', 'gaps': []} for key, kind in
            [('L1', 'relation'), ('P1', 'path'), ('A1', 'alternatives')]]
    rows.append({'id': 'L2', 'kind': 'relation', 'relation': 'contradicts',
                 'status': 'model_assessed_counterevidence', 'gaps': []})
    assert causal_delivery_state({'obligations': rows})['status'] == 'ready_for_review'
    rows[-1]['status'] = 'model_assessed_support'
    assert causal_delivery_state({'obligations': rows})['status'] == 'counterevidence'


def test_runtime_builds_real_cas_preview_from_live_persisted_gaps_without_dispatch(tmp_path):
    store, initial = completed_store(tmp_path)
    calls = []
    rt = ProjectTaskRuntime(store, execute_question=lambda **kw: calls.append(kw), read_sources=lambda: {},
        validate_inputs=lambda rows: True)
    current = rt.review_current_feedback(store.get_task(initial['id']))
    progress = current['evidence_progress']
    assert progress['coverage']['unresolved'] > 0
    # Finish the local report step to close the active approval, but do not
    # introduce any model call; only then can the normal CAS replan be applied.
    report = store.claim_next(initial['id'], initial['run_id'])
    assert report['kind'] == 'report'
    current = store.complete_step(initial['id'], report['id'], report['claim_token'],
                                  {'step_refs': [{'step_id': 'a', 'version': 1}]})
    preview = rt.preview_evidence_repair(current['id'], expected_revision=current['revision'])
    assert len(preview['changed_steps']) == 2
    assert preview['changed_steps'][0].startswith('refine_')
    assert preview['changed_steps'][1].startswith('verify_')
    assert preview['changed_steps'] == preview['proposal']['added'] and not preview['proposal']['blocked']
    assert preview['proposal']['modified'] is True and calls == []
    assert store.get_task(current['id'])['revision'] == current['revision']
    applied = store.replan_task(current['id'], expected_revision=current['revision'],
        proposal_sha256=preview['proposal']['proposal_sha256'], **preview['plan'])
    assert next(row for row in applied['steps'] if row['id'] == 'a')['status'] == 'completed'
    audit = next(row for row in applied['steps'] if row.get('purpose') == 'verify')
    refinement = next(row for row in applied['steps'] if row.get('refine_of') == 'a')
    assert refinement['question'] == next(row['question'] for row in applied['steps'] if row['id'] == 'a')
    assert refinement['depends_on'] == ['inventory', 'a']
    assert audit['review_of'] == refinement['id'] and audit['question'] == refinement['question']
    assert calls == []
    with pytest.raises(ProjectTaskConflictError):
        rt.preview_evidence_repair(current['id'], expected_revision=current['revision'])


def test_provenance_revocation_removes_old_ready_goal_projection(tmp_path):
    store, initial = completed_store(tmp_path)
    rt = ProjectTaskRuntime(store, execute_question=lambda **kw: None, read_sources=lambda: {},
        validate_inputs=lambda rows: False)
    progress = rt.review_current_feedback(store.get_task(initial['id']))['evidence_progress']
    assert progress['coverage']['ready_for_review'] == 0
    assert progress['coverage']['not_assessed'] == progress['coverage']['total']
    with store._db() as db:
        persisted = json.loads(db.execute('SELECT payload FROM project_obligation_ledgers WHERE task_id=?',
                                         (initial['id'],)).fetchone()[0])
    assert persisted['coverage']['ready_for_review'] == 0
    assert persisted['fingerprint'] == progress['fingerprint']


@pytest.mark.asyncio
async def test_actual_gateway_consumes_parent_gap_and_previews_scoped_repair_without_paid_retry(research_mobile):
    from docthinker.session_manager import SessionManager
    from docthinker.server.routers import project_tasks
    from tests.test_research_mobile_unit import mobile_client, upload_ready, SID, PREFIX
    from tests.test_personal_tasks_mobile_unit import create, start, settled, artifact
    mobile = research_mobile
    manager = SessionManager(str(mobile.root / '_system'), str(mobile.root), migrate_legacy=False)
    assert manager.create_session('Evidence-driven private work')['id'] == SID
    mobile.manager = manager
    mobile.state.session_manager = manager
    missing = 'Orion approved critical path and supplier outage overlap original records'
    prompts = []
    async def response(prompt, **kwargs):
        assert 'OUT_OF_SCOPE_SENTINEL' not in prompt
        if prompts:
            assert '前序证据缺口的待查队列' in prompt and missing in prompt
        prompts.append(prompt)
        payload = json.JSONDecoder().raw_decode(prompt.split('TASK_REQUIREMENTS_DATA_JSON:\n', 1)[1].lstrip())[0]
        return json.dumps({'answer': 'Orion has added work, but its critical-path effect needs checking [S1].',
            'task_assessment': {'requirements': [{'id': row['id'], 'verdict': 'unknown',
                'missing_query': missing} for row in payload['requirements']]}})
    mobile.model.side_effect = response
    try:
        async with mobile_client(mobile) as browser:
            owned = await upload_ready(mobile, browser, filename='worklog.txt', text=
                'Orion delivery work log records added work. The critical path and supplier overlap are not confirmed.')
            await upload_ready(mobile, browser, filename='private-other.txt', text=
                'OUT_OF_SCOPE_SENTINEL Orion critical path approval is private to another selected document.')
            task = await create(browser, {'session_id': SID, 'goal': 'Assess Orion delivery delay and draft a private update.',
                'constraints': 'No sending; do not assume the added work proves the cause.', 'questions': [
                    {'id': 'facts', 'question': 'Check Orion delivery work log for missing critical path evidence.',
                     'document_ids': [owned]},
                    {'id': 'brief', 'question': 'Draft the Orion delivery brief and disclose unresolved evidence.',
                     'document_ids': [owned], 'depends_on': ['facts']}]})
            await start(browser, task, calls=2)
            done = await settled(browser, task['id'])
            assert done['status'] == 'completed' and len(prompts) == mobile.model.await_count == 2
            assert done['evidence_progress']['coverage']['unresolved'] > 0
            final = await artifact(browser, task, step_id='brief')
            assert final['available'] and final['references']['document_ids'] == [owned]
            preview = await browser.post(PREFIX + '/workbench/tasks/evidence/preview', json={
                'session_id': SID, 'task_id': task['id'], 'expected_revision': done['revision']})
            assert preview.status_code == 200, preview.text
            repair = preview.json()['repair']
            assert repair['changed_steps'] and repair['proposal']['modified']
            assert repair['plan']['goal'] == task['goal'] and repair['execution_authorized'] is False
            assert all(row['document_ids'] == [owned] for row in repair['plan']['questions'])
            assert mobile.model.await_count == 2  # preview cannot spend or dispatch
            # Version changes revoke the prepared proposal instead of silently
            # approving a new plan or reusing a paid request identity.
            stale = await browser.post(PREFIX + '/workbench/tasks/evidence/preview', json={
                'session_id': SID, 'task_id': task['id'], 'expected_revision': done['revision'] - 1})
            assert stale.status_code == 409 and mobile.model.await_count == 2
    finally:
        await project_tasks.shutdown_project_tasks()
        await mobile.service.stop()
