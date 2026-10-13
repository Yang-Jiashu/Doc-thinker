"""Whole-unit predecessor handoff contracts; no provider/summary calls."""
import json

import pytest

from docthinker.server.project_task_handoff import (
    build_handoff, DependencyContextUnavailable, MAX_PRIOR_BYTES, MAX_SCAN_UNITS)
from docthinker.server.research_sentence_windows import evidence_units


def dep(text, step='q1'):
    return {'step_id': step, 'answer_id': '9712f829-e9e3-4298-bacf-62b3314a29f7', 'content': text}


def test_full_small_dependency_and_empty_roots_have_truthful_counts():
    packet, counts = build_handoff([dep('Renewable supply reduces imports [S1].')], 'imports')
    assert json.loads(packet)[0]['analysis_not_original_evidence'] == 'Renewable supply reduces imports [S1].'
    assert counts == {'required': 1, 'used': 1, 'full': 1, 'excerpted': 0, 'omitted': 0}
    assert build_handoff([], 'question') == ('', dict(required=0, used=0, full=0, excerpted=0, omitted=0))


def test_long_answer_selects_relevant_complete_unit_not_an_arbitrary_prefix():
    text = 'Unrelated background material. ' * 70 + 'Investment increases domestic supply [S1]. ' + 'Other background. ' * 70
    packet, counts = build_handoff([dep(text)], 'How does investment increase domestic supply?')
    body = json.loads(packet)[0]
    assert body['coverage'] == 'selected_analysis_units'
    assert any('Investment increases' in unit['text'] for unit in body['analysis_not_original_evidence'])
    assert counts['excerpted'] == 1 and counts['full'] == 0
    assert len(packet.encode()) <= MAX_PRIOR_BYTES
    whole = set(evidence_units(text))
    for unit in body['analysis_not_original_evidence']:
        assert (unit['start'], unit['end']) in whole
        assert unit['text'] == text[unit['start']:unit['end']]


def test_every_direct_dependency_survives_budget_competition():
    rows = [dep('先前分析背景。' * 180 + f'第{n}步的供应风险证据[S1]。', f'q{n}') for n in range(1, 7)]
    packet, counts = build_handoff(rows, '供应风险')
    parsed = json.loads(packet)
    assert [row['step_id'] for row in parsed] == [f'q{n}' for n in range(1, 7)]
    assert all(row['analysis_not_original_evidence'] for row in parsed)
    assert counts == dict(required=6, used=6, full=0, excerpted=6, omitted=0)
    assert len(packet.encode('utf-8')) <= MAX_PRIOR_BYTES


def test_no_half_condition_can_be_selected_to_force_a_long_claim_into_budget():
    text = 'Supply improves [S1]. However, ' + 'qualification ' * 240 + 'applies [S1].'
    assert len(evidence_units(text)) == 1
    with pytest.raises(DependencyContextUnavailable):
        build_handoff([dep(text)], 'supply')


def test_scanning_cap_is_total_and_does_not_claim_unscanned_content_read():
    rows = [dep('Prefix. ' * (MAX_SCAN_UNITS + 10) + 'TAIL_MUST_NOT_CLAIM_READ.', f'q{n}') for n in range(1, 3)]
    packet, counts = build_handoff(rows, 'TAIL_MUST_NOT_CLAIM_READ')
    assert counts['full'] == 0 and counts['excerpted'] == 2
    assert 'TAIL_MUST_NOT_CLAIM_READ' not in packet
    assert sum(len(row['analysis_not_original_evidence']) for row in json.loads(packet)) <= MAX_SCAN_UNITS


@pytest.mark.parametrize('rows', [None, {}, [None], [dep('')], [dep(' ' * 10)],
    [dep('x' * 131073)], [dict(dep('ok'), extra='not allowed')],
    [dict(dep('ok'), content=None)], [dict(dep('ok'), step_id='')], [dep('ok')] * 7])
def test_malformed_or_unusable_dependency_cannot_be_silently_skipped(rows):
    with pytest.raises(DependencyContextUnavailable):
        build_handoff(rows, 'question')


def test_project_context_is_mandatory_whole_data_but_not_a_retrieval_query(tmp_path, monkeypatch):
    import asyncio
    import docthinker.server.research_service as module
    from tests.test_research_service_unit import SessionManager, text_extractor, request
    from docthinker.server.research_service import ResearchService
    async def scenario():
        manager = SessionManager(str(tmp_path / '_system'), str(tmp_path), migrate_legacy=False)
        sid = manager.create_session('Handoff query separation')['id']
        service = ResearchService(tmp_path, extractor=text_extractor, context_assistance_enabled=False)
        seen, prompts = [], []
        original = module.build_retrieval_plan
        def capture(question, *args, **kwargs):
            seen.append(question)
            return original(question, *args, **kwargs)
        monkeypatch.setattr(module, 'build_retrieval_plan', capture)
        async def model(prompt, **_):
            prompts.append(prompt)
            return 'Renewable supply reduces imports [S1].'
        try:
            doc = service.store.accept_document(sid, 'source.txt', b'Renewable supply reduces imports under specified conditions.')
            await service.process(sid, doc['id'])
            question = 'How does renewable supply reduce imports?'
            context = 'User approved constraints. PRIOR_ANALYSIS_NOT_A_QUERY_TERM ' + 'private analysis. ' * 80
            result = await service.answer(request(sid, question), model, manager,
                                          document_ids=[doc['id']], project_context=context)
            assert seen and all('PRIOR_ANALYSIS_NOT_A_QUERY_TERM' not in query for query in seen)
            assert len(prompts) == 1 and 'PROJECT_TASK_DATA_JSON' in prompts[0]
            packet = json.JSONDecoder().raw_decode(prompts[0].split('PROJECT_TASK_DATA_JSON:\n', 1)[1])[0]
            assert packet['context'] == context
            assert result['reasoning']['trace']['context_budget']['input_bytes'] <= 24576
            assert result['reasoning']['trace']['context_budget']['project_context_bytes'] > len(context.encode())
            assert manager.get_history(sid)[0]['content'] == question
            before = len(prompts)
            with pytest.raises(ValueError):
                await service.answer(request(sid, question), model, manager,
                                     document_ids=[doc['id']], project_context='x' * 8193)
            assert len(prompts) == before
        finally:
            await service.stop()
    asyncio.run(scenario())


def test_handoff_that_cannot_be_preserved_fails_before_request_acceptance(tmp_path, monkeypatch):
    import asyncio
    from docthinker.server.project_task_bridge import ProjectTaskBridge
    from tests.test_project_task_acceptance_unit import _workspace
    from docthinker.server.request_journal import RequestJournal
    from docthinker.server.routers import mobile
    async def scenario():
        manager, service, sid, docs = await _workspace(tmp_path)
        calls = []
        async def model(*args, **kwargs):
            calls.append(args)
            raise AssertionError('No paid attempt should begin')
        bridge = ProjectTaskBridge(service, manager, sid, model)
        journal = RequestJournal(tmp_path, boot_id='handoff-before-dispatch')
        monkeypatch.setattr(mobile, '_query_journal', lambda: journal)
        async def load(**kwargs):
            return {'content': 'Huge indivisible analysis ' * 130, 'sources': [{'document_id': docs[0]}]}
        monkeypatch.setattr(bridge, 'load_answer', load)
        try:
            snapshot = await bridge.read_sources([docs[0]])
            rid = '2c7f0eb3-191a-4e49-9175-df9bf3210f46'
            result = await bridge.execute_question(session_id=sid, request_id=rid,
                question='Explain cache', document_ids=[docs[0]], input_snapshot=snapshot,
                goal='Review sources', constraints='Cite evidence', dependency_artifacts=[
                    {'step_id': 'q1', 'references': {'answer_id': dep('x')['answer_id']}}])
            assert result == {'request_status': 'failed', 'code': 'dependency_unavailable'}
            assert not calls and journal.read(sid, rid)['request_status'] == 'not_found'
        finally:
            await service.stop()
    asyncio.run(scenario())
