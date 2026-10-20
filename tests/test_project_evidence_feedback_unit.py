"""Evidence feedback is durable advisory data, never another model/tool run."""
import asyncio
import hashlib
from copy import deepcopy
import json
from unittest.mock import AsyncMock, Mock

import pytest

from docthinker.server.project_evidence_feedback import (
    MAX_FEEDBACK_BYTES, normalize_event_reasoning, feedback_view, feedback_lines,
)
from docthinker.server.project_task_bridge import ProjectTaskBridge
from docthinker.server.project_task_runtime import ProjectTaskRuntimeError, _artifact
from docthinker.server.project_task_store import ProjectTaskStore, ProjectTaskError
from tests.test_project_task_runtime_unit import setup, run, step, runtime, uid
from tests.test_research_mobile_unit import mobile as research_mobile


def feedback(*, assessed=True, unknown=True):
    rows = [{'id': 'L1', 'kind': 'relation', 'status': 'model_assessed_support' if assessed else 'unknown',
             'evidence': ([{'label': 'S1', 'source_id': 'source_1', 'span': [12, 57],
                            'quote_sha256': 'a' * 64}] if assessed else []), 'gaps': []}]
    if unknown:
        rows.append({'id': 'A1', 'kind': 'alternatives', 'status': 'unknown',
                     'evidence': [], 'gaps': ['alternative_not_found']})
    for row in rows:
        row.update(condition_count=0, condition_checks=[])
    return {'status': 'evidence_gaps' if unknown or not assessed else 'assessed_with_caveats',
            'obligations': rows, 'supported': int(assessed), 'counterevidence': 0,
            'unknown': int(unknown) + int(not assessed),
            'gaps': [{'obligation_id': 'A1', 'code': 'alternative_not_found'}] if unknown else [],
            'suggested_queries': ['查找交接延期的其他解释及反例'] if unknown else [],
            'causality_verified': False, 'model_assessed': assessed, 'condition_schema': 2}


def test_compact_receipt_is_copied_without_quotes_or_semantic_promotion():
    value = feedback()
    checked = normalize_event_reasoning(value)
    assert checked == value and checked is not value
    value['obligations'][0]['evidence'][0]['span'][0] = 3
    assert checked['obligations'][0]['evidence'][0]['span'] == [12, 57]
    assert checked['causality_verified'] is False
    assert len(json.dumps(checked, ensure_ascii=False).encode()) < MAX_FEEDBACK_BYTES


@pytest.mark.parametrize('variant', ['chain', 'conditions', 'single_event', 'unknown', 'empty'])
def test_actual_event_controller_compact_receipt_is_accepted_without_reinterpretation(variant):
    from tests.test_research_event_reasoning_unit import context, source, envelope, event
    from docthinker.server.research_event_reasoning import parse_event_response
    ctx = context(['Only if scope was approved']) if variant == 'conditions' else context()
    if variant == 'single_event':
        ctx = {'events': [event('a', '客户交接时间变更')], 'relations': [], 'bindings': {}, 'trace': {}}
    if variant == 'empty':
        ctx = {'events': [], 'relations': [], 'bindings': {}, 'trace': {}}
    raw = 'Unstructured current answer' if variant == 'unknown' else json.dumps(envelope(source()))
    _, receipt = parse_event_response(raw, ctx, [source()])
    assert normalize_event_reasoning(receipt) == receipt


@pytest.mark.parametrize('mutation', [
    lambda d: d.update(causality_verified=True),
    lambda d: d.update(model_assessed=False),
    lambda d: d.update(supported=True),
    lambda d: d.update(unknown=0),
    lambda d: d.update(status=[]),
    lambda d: d['obligations'][0].update(status='verified'),
    lambda d: d['obligations'].append(deepcopy(d['obligations'][0])),
    lambda d: d['obligations'][0]['evidence'][0].update(quote='Do not store source text'),
    lambda d: d['obligations'][0]['evidence'][0].update(span=[True, 57]),
    lambda d: d['obligations'][0]['evidence'][0].update(span=[57, 12]),
    lambda d: d['obligations'][0]['evidence'][0].update(quote_sha256='not-a-digest'),
    lambda d: d['gaps'][0].update(obligation_id='OTHER'),
    lambda d: d.update(suggested_queries=['x' * 401]),
    lambda d: d.update(suggested_queries=['中' * 134]),
    lambda d: d.update(suggested_queries=['q'] * 9),
    lambda d: d.update(metadata={'semantic_verified': True}),
    lambda d: d.update(metadata={'content': 'A second answer body'}),
])
def test_unbounded_or_promoted_feedback_is_rejected(mutation):
    value = feedback()
    mutation(value)
    with pytest.raises(ValueError):
        normalize_event_reasoning(value)


def test_receipt_byte_limit_is_not_silently_truncated():
    value = feedback()
    value['metadata'] = {f'key_{i}': '中' * 300 for i in range(10)}
    with pytest.raises(ValueError, match='byte budget'):
        normalize_event_reasoning(value)


def test_boundary_query_and_gap_optional_query_are_preserved_as_safe_data():
    value = feedback()
    value['suggested_queries'] = ['x' * 400]
    value['gaps'][0]['query'] = '<img src=x onerror=alert(1)> [execute](https://private.invalid)'
    checked = normalize_event_reasoning(value)
    from docthinker.server.project_evidence_feedback import bind_feedback
    binding = bind_feedback(checked, task_id=uid(), task_revision=4, step_id='work', artifact_version=1)
    view = feedback_view(checked, binding)
    rendered = '\n'.join(feedback_lines(view))
    assert '<img' not in rendered and '\\[execute\\]' in rendered
    assert view['next_step_suggestion']['automatic_execution'] is False
    assert view['next_step_suggestion']['manual_replan_only'] is True
    assert view['event_reasoning']['gaps'][0]['query'] == value['gaps'][0]['query']
    binding['receipt_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='binding'):
        feedback_view(checked, binding)


@pytest.mark.parametrize('changed', ['task_id', 'step_id', 'artifact_version', 'task_revision'])
def test_feedback_cannot_be_retagged_to_another_task_artifact(changed):
    from docthinker.server.project_evidence_feedback import bind_feedback
    key = uid()
    value = feedback()
    binding = bind_feedback(value, task_id=key, task_revision=5, step_id='work', artifact_version=1)
    binding[changed] = {'task_id': uid(), 'step_id': 'another', 'artifact_version': 2, 'task_revision': 7}[changed]
    with pytest.raises(ValueError, match='different task artifact'):
        feedback_view(value, binding, task_id=key, step_id='work', artifact_version=1, current_task_revision=6)


def test_legacy_positive_receipt_is_downgraded_without_retagging_its_original_binding():
    from docthinker.server.project_evidence_feedback import bind_feedback
    legacy = feedback(unknown=False)
    legacy.pop('condition_schema')
    for row in legacy['obligations']:
        row.pop('condition_count')
        row.pop('condition_checks')
    before = deepcopy(legacy)
    binding = bind_feedback(legacy, task_id=uid(), task_revision=4, step_id='work', artifact_version=1)
    view = feedback_view(legacy, binding)
    assert legacy == before and view['binding'] == binding
    assert view['receipt_projection'] == 'legacy_condition_unknown'
    assert view['event_reasoning']['supported'] == 0 and view['event_reasoning']['unknown'] == 1
    assert 'legacy_condition_state_unknown' in view['event_reasoning']['obligations'][0]['gaps']
    assert view['event_reasoning']['obligations'][0]['evidence'] == legacy['obligations'][0]['evidence']
    assert view['content_review'] == 'needs_evidence'
    assert normalize_event_reasoning(view['event_reasoning']) == view['event_reasoning']


def test_v2_condition_metadata_depth_and_33_checks_fit_existing_byte_budget():
    value = feedback(assessed=False, unknown=False)
    row = value['obligations'][0]
    row['condition_count'] = 33
    row['condition_checks'] = [{'condition_sha256': hashlib.sha256(str(i).encode()).hexdigest(),
        'rule_status': 'unknown', 'satisfaction': 'unknown', 'rule_refs': [], 'satisfaction_refs': []}
        for i in range(33)]
    assert len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()) <= MAX_FEEDBACK_BYTES
    assert normalize_event_reasoning(value) == value


def test_legacy_durable_task_badge_and_report_keep_body_but_not_positive_condition_state(tmp_path):
    async def scenario():
        from docthinker.server.project_evidence_feedback import bind_feedback
        store, task, bridge, rt, done = await with_feedback(tmp_path, value=feedback(unknown=False))
        artifact = store.get_artifact(task['id'], 'work')
        payload = deepcopy(artifact['references'])
        legacy = payload['event_reasoning']
        legacy.pop('condition_schema')
        for row in legacy['obligations']:
            row.pop('condition_count')
            row.pop('condition_checks')
        previous = payload['evidence_feedback_binding']
        payload['evidence_feedback_binding'] = bind_feedback(legacy, **{key: previous[key] for key in
            ('task_id', 'task_revision', 'step_id', 'artifact_version')})
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        # Explicit sealed legacy fixture; no real answer or production record.
        with store._db(write=True) as db:
            db.execute('UPDATE project_task_artifacts SET payload=? WHERE task_id=? AND step_id=? AND version=1',
                       (raw, task['id'], 'work'))
        public = step(store.get_task(task['id']), 'work')['evidence_feedback']
        assert public['event_reasoning']['supported'] == 0 and public['content_review'] == 'needs_evidence'
        report = await rt.render_report(task['id'], store.get_artifact(task['id'], 'report'))
        assert report['complete'] and 'Local answer for' in report['content']
        assert report['evidence_feedback'][0]['event_reasoning']['supported'] == 0
        assert report['evidence_feedback'][0]['binding'] == payload['evidence_feedback_binding']
        assert report['evidence_feedback'][0]['receipt_projection'] == 'legacy_condition_unknown'
        assert store.get_artifact(task['id'], 'work')['references'] == payload
        assert len(bridge.calls) == 1
    asyncio.run(scenario())


async def with_feedback(tmp_path, *, value=None, barrier=None):
    store, task, bridge, rt = setup(tmp_path, questions=[{'id': 'work', 'question': '核对客户交接延期的原因与反例。'}])
    original = rt.execute_question
    async def execute(**kwargs):
        result = await original(**kwargs)
        result['event_reasoning'] = deepcopy(value if value is not None else feedback())
        return result
    rt.execute_question = execute
    rt.validate_inputs = barrier
    done = await run(rt, task, budget=1)
    return store, task, bridge, rt, done


def test_completed_execution_persists_bound_evidence_gaps_across_restart_without_retry(tmp_path):
    async def scenario():
        store, task, bridge, rt, done = await with_feedback(tmp_path)
        assert done['status'] == 'completed' and len(bridge.calls) == 1
        work = step(done, 'work')
        assert work['content_review'] == 'needs_evidence'
        saved = store.get_artifact(task['id'], 'work')
        binding = saved['references']['evidence_feedback_binding']
        assert binding['task_id'] == task['id'] and binding['step_id'] == 'work'
        assert binding['artifact_version'] == saved['version'] == 1
        assert 1 <= binding['task_revision'] <= done['revision']
        reopened = ProjectTaskStore(tmp_path.resolve(), session_id=store.session_id,
            instance_id=store.instance_id, boot_id='restart-after-feedback')
        assert reopened.get_artifact(task['id'], 'work') == saved
        rt2 = runtime(reopened, bridge)
        report = await rt2.render_report(task['id'], reopened.get_artifact(task['id'], 'report'))
        assert report['complete'] and report['content_review'] == 'needs_evidence'
        assert report['acceptance_status'] == 'review_required'
        assert report['semantic_verified'] is False and report['goal_verified'] is False
        assert report['evidence_feedback'][0]['binding'] == binding
        assert report['next_step_suggestions'][0]['gaps'] == feedback()['gaps']
        assert '查找交接延期的其他解释及反例' in report['content']
        assert '反证 0 项、未知 1 项' in report['content']
        assert len(bridge.calls) == 1 and report['model_calls'] == 0
    asyncio.run(scenario())


def test_supported_receipt_still_needs_review_not_semantic_or_goal_verification(tmp_path):
    async def scenario():
        store, task, bridge, rt, done = await with_feedback(tmp_path, value=feedback(unknown=False))
        assert step(done, 'work')['content_review'] == 'review_required'
        report = await rt.render_report(task['id'], store.get_artifact(task['id'], 'report'))
        assert report['content_review'] == 'review_required'
        assert report['goal_verified'] is False and report['semantic_verified'] is False
        assert report['next_step_suggestions'][0]['kind'] == 'review_evidence'
        assert done['status'] == 'completed' and len(bridge.calls) == 1
    asyncio.run(scenario())


def test_invalid_feedback_does_not_publish_or_repeat_known_model_attempt(tmp_path):
    async def scenario():
        invalid = feedback()
        invalid['causality_verified'] = True
        store, task, bridge, rt, done = await with_feedback(tmp_path, value=invalid)
        assert done['status'] == 'blocked' and step(done, 'work')['reason'] == 'invalid_result'
        assert step(done, 'work')['artifact_version'] == 0 and len(bridge.calls) == 1
        assert step(done, 'report')['status'] != 'completed'
    asyncio.run(scenario())


def test_final_sync_barrier_receives_actual_answer_before_commit_and_after_report_awaits(tmp_path):
    async def scenario():
        checks = []
        def barrier(rows):
            checks.append(deepcopy(rows))
            assert len(rows) == 1 and rows[0]['answer_id']
            return True
        store, task, bridge, rt, done = await with_feedback(tmp_path, barrier=barrier)
        saved = store.get_artifact(task['id'], 'work')['references']
        assert checks[0][0]['answer_id'] == saved['answer_id']
        await rt.render_report(task['id'], store.get_artifact(task['id'], 'report'))
        assert checks[-1][0]['answer_id'] == saved['answer_id'] and len(checks) == 2
        rt.validate_inputs = lambda rows: False
        with pytest.raises(ProjectTaskRuntimeError, match='publication'):
            await rt.render_report(task['id'], store.get_artifact(task['id'], 'report'))
        assert len(bridge.calls) == 1
    asyncio.run(scenario())


def test_event_change_at_final_completion_barrier_does_not_publish_artifact(tmp_path):
    async def scenario():
        store, task, bridge, rt, done = await with_feedback(tmp_path, barrier=lambda rows: False)
        assert step(done, 'work')['artifact_version'] == 0
        assert step(done, 'work')['reason'] == 'source_changed' and len(bridge.calls) == 1
    asyncio.run(scenario())


def test_withdrawal_hides_feedback_and_suggestion_with_original_artifact(tmp_path):
    async def scenario():
        store, task, bridge, rt, done = await with_feedback(tmp_path)
        store.invalidate_sources(['doc_a'])
        saved = store.get_artifact(task['id'], 'work')
        assert not saved['available'] and saved['references'] is None
        assert 'evidence_feedback' not in step(store.get_task(task['id']), 'work')
        assert len(bridge.calls) == 1
    asyncio.run(scenario())


def test_legacy_callback_keeps_exact_artifact_keys_and_no_new_feedback(tmp_path):
    async def scenario():
        store, task, bridge, rt = setup(tmp_path, questions=[{'id': 'work', 'question': 'Existing question'}])
        done = await run(rt, task)
        assert set(store.get_artifact(task['id'], 'work')['references']) == {'answer_id', 'source_ids', 'document_ids'}
        report = await rt.render_report(task['id'], store.get_artifact(task['id'], 'report'))
        assert 'evidence_feedback' not in report and 'content_review' not in step(done, 'work')
        assert len(bridge.calls) == 1
    asyncio.run(scenario())


@pytest.mark.parametrize('withdrawn,invalid', [(False, False), (False, True), (True, False)])
def test_bridge_hydrates_feedback_only_with_current_answer_provenance(withdrawn, invalid):
    async def scenario():
        bridge = ProjectTaskBridge(None, None, '#00001', None)
        value = feedback()
        if invalid:
            value['causality_verified'] = True
        answer = {'content': 'Synthetic handover draft', 'sources': [
            {'id': 'source_1', 'label': 'S1', 'document_id': 'doc_a'}]}
        bridge.load_answer = AsyncMock(return_value=None if withdrawn else answer)
        result = await bridge._hydrate_receipt({'request_status': 'completed', 'answer_id': uid(),
                                               'event_reasoning': value})
        assert result['request_status'] == ('failed' if invalid or withdrawn else 'completed')
        if invalid or withdrawn:
            assert 'event_reasoning' not in result
        else:
            assert result['event_reasoning'] == value
    asyncio.run(scenario())


@pytest.mark.asyncio
async def test_native_asgi_task_carries_real_event_receipt_and_withholds_after_event_revision(research_mobile):
    """Real session, event/research/task SQLite and ASGI; only the answer model is fake."""
    from tests.test_research_mobile_unit import mobile_client, PREFIX, SID
    from tests.test_personal_tasks_mobile_unit import create, start, settled, artifact
    from tests.test_research_event_pipeline_unit import material, EnvelopeModel
    from docthinker.session_manager import SessionManager
    from docthinker.server.personal_events import PersonalEvents
    from docthinker.server.routers import project_tasks
    mobile = research_mobile
    manager = SessionManager(str(mobile.root / '_system'), str(mobile.root), migrate_legacy=False)
    assert manager.create_session('Orion private handover')['id'] == SID
    mobile.manager = manager
    mobile.state.session_manager = manager
    document, events, _ = material(mobile.service, SID)
    model = EnvelopeModel()
    async def response(*args, **kwargs):
        return await model(*args, **kwargs)
    mobile.model.side_effect = response
    try:
        async with mobile_client(mobile) as browser:
            task = await create(browser, {'session_id': SID, 'goal': 'Explain the Orion delivery delay.',
                'constraints': 'No external actions; causal hypotheses need evidence and counterevidence.',
                'questions': [{'id': 'work', 'question': 'What caused the Orion delivery delay?',
                               'document_ids': [document['id']]}]})
            await start(browser, task, calls=1)
            done = await settled(browser, task['id'])
            assert done['status'] == 'completed' and mobile.model.await_count == 1
            assert model.packets and model.packets[0]['events']
            work = step(done, 'work')
            assert work['content_review'] == 'needs_evidence'
            saved = await artifact(browser, task, step_id='work')
            assert saved['available'] and saved['references']['event_reasoning']['unknown'] > 0
            binding = saved['references']['evidence_feedback_binding']
            assert binding['task_id'] == task['id'] and binding['artifact_version'] == 1
            report = await artifact(browser, task)
            assert report['available'] and report['content_review'] == 'needs_evidence'
            assert report['goal_verified'] is False and report['acceptance_status'] == 'review_required'
            assert report['evidence_feedback'][0]['binding'] == binding
            assert report['next_step_suggestions'][0]['manual_replan_only'] is True
            assert report['evidence_feedback'][0]['event_reasoning']['suggested_queries']
            snapshots = mobile.service.store.document_snapshots(SID, [document['id']])
            used = {row['id'] for row in model.packets[0]['events']}
            changed = next(row for row in events if row['id'] in used)
            PersonalEvents(mobile.service.store).revise_event(SID, changed['id'], expected_revision=1,
                event_type='observation', action='Corrected Orion event; the proposed chronology is unconfirmed.',
                origins=changed['origins'])
            assert mobile.service.store.document_snapshots(SID, [document['id']]) == snapshots
            for which in ('work', 'report'):
                hidden = await artifact(browser, task, step_id=which)
                assert hidden['available'] is False and hidden['content'] is None and hidden['references'] is None
                assert 'evidence_feedback' not in hidden
            assert mobile.model.await_count == 1
    finally:
        await project_tasks.shutdown_project_tasks()
        await mobile.service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize('personal', [False, True])
async def test_native_phase_without_events_reviews_every_original_criterion_and_binds_report(research_mobile, personal):
    from tests.test_research_mobile_unit import mobile_client, upload_ready, PREFIX, SID
    from tests.test_personal_tasks_mobile_unit import start, settled, artifact
    from tests.test_long_projects_mobile_unit import new_project
    from docthinker.session_manager import SessionManager
    from docthinker.server.routers import project_tasks
    mobile = research_mobile
    manager = SessionManager(str(mobile.root / '_system'), str(mobile.root), migrate_legacy=False)
    assert manager.create_session('Private fixed-criteria task')['id'] == SID
    mobile.manager = manager
    mobile.state.session_manager = manager
    text = 'Internal handover is Friday. Sending is not authorized.'
    question = 'Draft a private handover note and disclose what is missing.'
    criteria = ['Provide the internal date.', 'Do not pretend missing approval has been obtained.']
    packets = []
    async def response(prompt, **kwargs):
        assert 'EVENT_REASONING_DATA_JSON' not in prompt
        assert question in prompt and all(criterion in prompt for criterion in criteria)
        packet = json.JSONDecoder().raw_decode(prompt.split('TASK_REQUIREMENTS_DATA_JSON:\n', 1)[1].lstrip())[0]
        assert [row['id'] for row in packet['requirements']] == ['Q1', 'C1', 'C2']
        packets.append(packet)
        answer = 'Private draft: the handover is Friday; do not send it.' + ('' if personal else ' [S1].')
        evidence = [] if personal else [{'label': 'S1', 'quote': text, 'start': 0}]
        return json.dumps({'answer': answer, 'task_assessment': {'requirements': [
            {'id': key, 'verdict': 'addressed', 'evidence': evidence,
             'answer_witness': {'quote': answer, 'start': 0}} for key in ['Q1', 'C1']]}})
    mobile.model.side_effect = response
    try:
        async with mobile_client(mobile) as browser:
            document = None if personal else await upload_ready(mobile, browser, text=text, filename='handover.txt')
            project = await new_project(browser, SID)
            result = await browser.post(PREFIX + '/workbench/long-project/phases', json={
                'session_id': SID, 'project_id': project['id'], 'expected_revision': project['revision'],
                'request_id': uid(), 'title': 'Private handover', 'acceptance_criteria': criteria,
                'questions': [{'id': 'work', 'question': question,
                               **({'input_kind': 'personal', 'memory_ids': []} if personal else
                                  {'document_ids': [document]})}],
                'parent_refs': [], 'constraints': 'No sending or outside actions.'})
            assert result.status_code == 200, result.text
            task = result.json()['task']
            await start(browser, task, calls=1)
            done = await settled(browser, task['id'])
            assert done['status'] == 'completed' and mobile.model.await_count == 1
            assert step(done, 'work')['content_review'] == 'needs_evidence'
            current = await artifact(browser, task, step_id='work')
            review = current['references']['task_review']
            binding = current['references']['task_review_binding']
            assert current['available'] and review['addressed'] == 2 and review['unknown'] == 1
            assert review['supported'] == (0 if personal else 2)
            assert [row['id'] for row in review['obligations']] == ['Q1', 'C1', 'C2']
            assert review['obligations'][2]['status'] == 'unknown'
            assert review['requirements_sha256'] == packets[0]['requirements_sha256']
            assert binding['task_id'] == task['id'] and binding['artifact_version'] == current['version'] == 1
            report = await artifact(browser, task)
            assert report['available'] and report['task_review_feedback'][0]['binding'] == binding
            assert report['content_review'] == 'needs_evidence' and report['goal_verified'] is False
            assert report['task_review_feedback'][0]['task_review'] == review
            assert 'C2' in report['content'] and report['next_step_suggestions']
            assert all(row['manual_replan_only'] and not row['automatic_execution'] for row in report['next_step_suggestions'])
            assert mobile.model.await_count == 1
    finally:
        await project_tasks.shutdown_project_tasks()
        await mobile.service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['standalone', 'selected_parent', 'list_race', 'legacy_artifact'])
async def test_native_task_metadata_does_not_keep_old_support_after_event_revision(research_mobile, monkeypatch, surface):
    from tests.test_research_mobile_unit import mobile_client, PREFIX, SID
    from tests.test_personal_tasks_mobile_unit import create, start, settled, artifact
    from tests.test_research_event_pipeline_unit import material
    from docthinker.session_manager import SessionManager
    from docthinker.server.routers import project_tasks
    mobile = research_mobile
    manager = SessionManager(str(mobile.root / '_system'), str(mobile.root), migrate_legacy=False)
    assert manager.create_session('Current feedback only')['id'] == SID
    mobile.manager = manager
    mobile.state.session_manager = manager
    document, events, chunks = material(mobile.service, SID)
    async def response(prompt, **kwargs):
        packet = json.JSONDecoder().raw_decode(prompt.split('EVENT_REASONING_DATA_JSON:\n', 1)[1].lstrip())[0]
        need = next(row for row in packet['needs'] if row['kind'] == 'relation')
        source = next(row for row in chunks if 'because' in row['content'])
        label = next(row['label'] for row in packet['retained_sources'] if row['source_id'] == source['id'])
        return json.dumps({'answer': 'The record reports added critical-path work and delay [' + label + '].',
            'event_assessment': {'checks': [{'id': need['id'], 'verdict': 'support',
                'evidence': [{'label': label, 'quote': source['content']}]}]}})
    mobile.model.side_effect = response
    try:
        async with mobile_client(mobile) as browser:
            question_rows = [{'id': 'work', 'question': 'What caused the Orion delivery delay?',
                              'document_ids': [document['id']]}]
            other = None
            if surface == 'list_race':
                # The newer completed task is checked FIRST (the API sorts
                # newest first); the older task supplies the later await.
                other = await create(browser, {'session_id': SID, 'goal': 'Other task hydration boundary.',
                    'constraints': 'No actions.', 'questions': question_rows})
            if surface == 'selected_parent':
                from tests.test_long_projects_mobile_unit import new_project, phase
                project = await new_project(browser, SID)
                added = await phase(browser, SID, project, rows=question_rows)
                task = added['task']
            else:
                task = await create(browser, {'session_id': SID, 'goal': 'Review Orion delay records.',
                    'constraints': 'No external actions.', 'questions': question_rows})
            await start(browser, task, calls=1)
            done = await settled(browser, task['id'])
            assert done['status'] == 'completed'
            assert step(done, 'work')['evidence_feedback']['event_reasoning']['supported'] == 1
            if surface == 'legacy_artifact':
                from docthinker.server.project_evidence_feedback import bind_feedback
                _, runtime, _ = await project_tasks._context(SID)
                saved = runtime.store.get_artifact(task['id'], 'work')
                refs = deepcopy(saved['references'])
                legacy = refs['event_reasoning']
                legacy.pop('condition_schema')
                for row in legacy['obligations']:
                    row.pop('condition_count', None)
                    row.pop('condition_checks', None)
                old = refs['evidence_feedback_binding']
                refs['evidence_feedback_binding'] = bind_feedback(legacy, **{key: old[key] for key in
                    ('task_id', 'task_revision', 'step_id', 'artifact_version')})
                with runtime.store._db(write=True) as db:
                    db.execute('UPDATE project_task_artifacts SET payload=? WHERE task_id=? AND step_id=? AND version=1',
                        (json.dumps(refs, ensure_ascii=False, sort_keys=True, separators=(',', ':')), task['id'], 'work'))
                historical = await artifact(browser, task, step_id='work')
                assert historical['available'] and historical['content']
                assert historical['references'] == refs  # Original identity/hash remains immutable.
                assert historical['references']['event_reasoning']['supported'] == 1
                assert historical['evidence_feedback']['event_reasoning']['supported'] == 0
                assert historical['evidence_feedback']['receipt_projection'] == 'legacy_condition_unknown'
                assert historical['content_review'] == 'needs_evidence'
                assert historical['references_scope'] == 'immutable_historical_binding_not_current_assessment'
                assert historical['task_review_feedback']['task_review'] == refs['task_review']
                report = await artifact(browser, task)
                assert report['available'] and report['evidence_feedback'][0]['event_reasoning']['supported'] == 0
                assert mobile.model.await_count == 1
                return
            snapshots = mobile.service.store.document_snapshots(SID, [document['id']])
            changed = events[0]
            child = None
            if surface == 'selected_parent':
                child = await phase(browser, SID, project, rows=question_rows,
                    parents=[{'phase_id': added['phase']['id'], 'report_version': 1, 'step_ids': ['work']}])
                _, runtime, _ = await project_tasks._context(SID)
                assert runtime.long_projects.valid(child['task']['id']) is True
            def revise():
                mobile.service.events.revise_event(SID, changed['id'], expected_revision=1,
                    event_type='observation', action='Corrected scope chronology is unknown.', origins=changed['origins'])
            if surface == 'list_race':
                original = project_tasks._current
                state = {'withdrawn': False, 'positive_seen_before_later_await': False}
                async def yielding_current(runtime, task_id, **options):
                    value = await original(runtime, task_id, **options)
                    if task_id == task['id']:
                        state['positive_seen_before_later_await'] = (
                            step(value, 'work')['evidence_feedback']['event_reasoning']['supported'] == 1)
                    if task_id == other['id']:
                        revise()
                        state['withdrawn'] = True
                    return value
                monkeypatch.setattr(project_tasks, '_current', yielding_current)
                listed = await browser.get(PREFIX + '/workbench/tasks', params={'session_id': SID})
                assert listed.status_code == 200 and state['withdrawn'] and state['positive_seen_before_later_await']
                listed_task = next(row for row in listed.json()['items'] if row['id'] == task['id'])
                assert 'evidence_feedback' not in step(listed_task, 'work')
                assert 'task_review_feedback' not in step(listed_task, 'work')
                monkeypatch.setattr(project_tasks, '_current', original)
            else:
                revise()
            assert mobile.service.store.document_snapshots(SID, [document['id']]) == snapshots
            if child is not None:
                assert runtime.long_projects.valid(child['task']['id']) is False
            assert (await artifact(browser, task, step_id='work'))['available'] is False
            result = await browser.get(PREFIX + '/workbench/task', params={'session_id': SID, 'task_id': task['id']})
            assert result.status_code == 200
            public = step(result.json()['task'], 'work')
            assert 'evidence_feedback' not in public and 'task_review_feedback' not in public
            assert public['content_review'] == 'needs_evidence'
            assert public['feedback_status'] == 'provenance_unavailable'
            assert mobile.model.await_count == 1
    finally:
        await project_tasks.shutdown_project_tasks()
        await mobile.service.stop()
