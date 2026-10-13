"""Normal writing + auxiliary evidence + one provider call, not mode reports."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from docthinker.server.research_scheduler import ContextWorkResult
from docthinker.server.research_service import ResearchService, WITHDRAWN_ANALYSIS
from docthinker.server.research_retrieval import build_retrieval_plan
from tests.test_research_service_unit import workspace, document, text_extractor, FakeModel, request


@pytest.mark.asyncio
async def test_normal_writing_receives_conditional_line_without_separate_mode(workspace):
    root, manager, sid, _ = workspace
    service = ResearchService(root, extractor=text_extractor)
    await document(service, sid, 'Observation: grid ready. If grid ready then supply grows.')
    model = FakeModel('这里是一段正常写作结果，依照给定前提讨论供给，而不是算法报告。[S1]')
    result = await service.answer(request(sid, '帮我写一段 grid ready 如何影响 supply 的论证'), model, manager)
    assert len(model.prompts) == 1
    assert result['answer'] == model.answer
    assert result['reasoning']['mode'] == 'context_assist'
    assert result['reasoning']['conclusions']
    assert 'supporting_material' in model.prompts[0]
    assert 'conditional_on_source_premises' in model.prompts[0]
    assert result['reasoning']['trace']['model_calls'] == 0
    assert result['reasoning']['trace']['answer_model_calls'] == 1
    assert service.store.list_memories(sid) == []


@pytest.mark.asyncio
async def test_legacy_mode_is_not_a_new_user_selected_report(workspace):
    root, manager, sid, _ = workspace
    service = ResearchService(root, extractor=text_extractor)
    await document(service, sid)
    body = request(sid, 'renewable supply')
    body.reasoning_mode = 'deduction'
    model = FakeModel()
    result = await service.answer(body, model, manager)
    assert result['answer'] == model.answer and len(model.prompts) == 1
    assert result['reasoning']['trace']['algorithm_modes'] == []
    assert result['answer_mode'] == 'evidence'


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['busy', 'timed_out', 'failed', 'closed'])
async def test_optional_scheduler_failure_never_blocks_writing_or_retries_model(workspace, status):
    root, manager, sid, _ = workspace
    service = ResearchService(root, extractor=text_extractor)
    await document(service, sid, 'Observation: grid ready. If grid ready then supply grows.')
    service._context_scheduler.run = AsyncMock(return_value=ContextWorkResult(status))
    model = FakeModel()
    result = await service.answer(request(sid, 'Explain grid ready and supply'), model, manager)
    assert result['answer'] == model.answer and len(model.prompts) == 1
    assert result['sources'] and result['reasoning']['conclusions'] == []
    assert result['reasoning']['trace']['scheduler_status'] == status
    assert result['reasoning']['trace']['skipped_reason'] == 'scheduler_' + status


@pytest.mark.asyncio
async def test_budget_pressure_does_not_rerank_or_rerun_operators(workspace):
    root, manager, sid, _ = workspace
    service = ResearchService(root, extractor=text_extractor)
    await document(service, sid)
    service.store.create_memory(sid, 'preference', '"' * 2000)
    manager.add_message(sid, 'user', '\\' * 2000)
    manager.add_message(sid, 'assistant', '\\' * 2000)
    model = FakeModel()
    with patch('docthinker.server.research_service.build_retrieval_plan', wraps=build_retrieval_plan) as rank:
        with patch.object(service._context_scheduler, 'run', wraps=service._context_scheduler.run) as lane:
            result = await service.answer(request(sid, 'Energy imports ' + '\\' * 8000,
                                                 use_memory=True, use_conversation_context=True), model, manager)
    assert rank.call_count == lane.call_count == len(model.prompts) == 1
    budget = result['reasoning']['trace']['context_budget']
    assert budget['input_bytes'] <= budget['max_input_bytes'] == 24576
    assert budget['automatic_retry'] is False


@pytest.mark.asyncio
async def test_withdrawn_background_premise_hides_generated_body_on_replay(workspace):
    root, manager, sid, _ = workspace
    service = ResearchService(root, extractor=text_extractor)
    doc = await document(service, sid, 'Observation: grid ready. If grid ready then supply grows.')
    model = FakeModel('PRIVATE_WRITING_SENTINEL grid ready implies supply growth [S1].')
    result = await service.answer(request(sid, 'Explain grid ready and supply'), model, manager)
    assert result['reasoning']['conclusions']
    service.store.delete_document(sid, doc['id'])
    restarted = ResearchService(root)
    saved = next(row for row in restarted.hydrate_history(sid, manager.get_history(sid)) if row['id'] == result['answer_id'])
    assert saved['content'] == WITHDRAWN_ANALYSIS
    assert 'PRIVATE_WRITING_SENTINEL' not in str(saved)
    assert saved['reasoning']['conclusions'] == []
    assert len(model.prompts) == 1
    await service.answer(request(sid, '你好', use_conversation_context=True), model, manager)
    assert 'PRIVATE_WRITING_SENTINEL' not in model.prompts[-1]
    assert 'grid ready implies supply growth' not in model.prompts[-1]


@pytest.mark.asyncio
async def test_withdrawn_ordinary_quote_is_not_resubmitted_in_next_history_context(workspace):
    root, manager, sid, _ = workspace
    service = ResearchService(root, extractor=text_extractor)
    doc = await document(service, sid)
    model = FakeModel('PRIVATE_QUOTE_IN_OLD_RESPONSE [S1].')
    await service.answer(request(sid), model, manager)
    service.store.delete_document(sid, doc['id'])
    await service.answer(request(sid, '请再解释一下', use_conversation_context=True), model, manager)
    assert len(model.prompts) == 2
    assert 'PRIVATE_QUOTE_IN_OLD_RESPONSE' not in model.prompts[-1]


@pytest.mark.asyncio
async def test_provider_failure_is_not_automatically_retried(workspace):
    root, manager, sid, _ = workspace
    service = ResearchService(root, extractor=text_extractor)
    await document(service, sid)
    model = AsyncMock(side_effect=RuntimeError('synthetic failure'))
    with pytest.raises(RuntimeError):
        await service.answer(request(sid), model, manager)
    model.assert_awaited_once()
    assert manager.get_history(sid) == []


@pytest.mark.asyncio
async def test_operator_ablation_changes_only_auxiliary_context_not_normal_generation(workspace):
    root, manager, sid, _ = workspace
    enabled = ResearchService(root, extractor=text_extractor)
    await document(enabled, sid, 'Observation: grid ready. If grid ready then supply grows.')
    disabled = ResearchService(root, extractor=text_extractor, context_assistance_enabled=False)
    on_model, off_model = FakeModel('normal writing [S1]'), FakeModel('normal writing [S1]')
    body = request(sid, 'Explain why grid ready supports supply')
    on = await enabled.answer(body, on_model, manager)
    off = await disabled.answer(body, off_model, manager)
    assert len(on_model.prompts) == len(off_model.prompts) == 1
    assert on['answer'] == off['answer'] == 'normal writing [S1]'
    assert on['reasoning']['conclusions'] and not off['reasoning']['conclusions']
    assert off['reasoning']['trace']['skipped_reason'] == 'operator_disabled'
    assert on_model.options[0]['system_prompt'] == off_model.options[0]['system_prompt']
    assert 'supporting_material' in on_model.prompts[0] and 'supporting_material' not in off_model.prompts[0]
    assert on['answer_mode'] == off['answer_mode'] == 'evidence'


@pytest.mark.parametrize('flag', ['false', None, 0, 1])
def test_ablation_flag_cannot_silently_coerce_non_boolean_values(tmp_path, flag):
    with pytest.raises(ValueError, match='boolean'):
        ResearchService(tmp_path, context_assistance_enabled=flag)


@pytest.mark.asyncio
@pytest.mark.parametrize('question', ['帮我写一封生日祝福邮件', '给我写一首关于冬天的诗', 'How do I write a thank-you letter?', '帮我起草一份产品使用文档'])
async def test_unrelated_uploaded_document_does_not_block_normal_writing(workspace, question):
    root, manager, sid, _ = workspace
    service = ResearchService(root, extractor=text_extractor)
    await document(service, sid)
    model = FakeModel('Here is your requested writing, not a claim from uploaded papers.')
    result = await service.answer(request(sid, question), model, manager)
    assert result['answer'] == model.answer and len(model.prompts) == 1
    assert result['answer_mode'] == 'general' and result['sources'] == []
    assert '不能声称研究结论来自论文' in model.prompts[0]


@pytest.mark.asyncio
async def test_short_followup_reuses_prior_user_query_only_when_history_enabled(workspace):
    root, manager, sid, _ = workspace
    service = ResearchService(root, extractor=text_extractor)
    await document(service, sid)
    manager.add_message(sid, 'user', '论文中的 renewable supply 和 energy imports 是什么关系？')
    manager.add_message(sid, 'assistant', 'UNSUPPORTED_ASSISTANT_PREMISE')
    model = FakeModel()
    on = await service.answer(request(sid, '请再解释一下', use_conversation_context=True), model, manager)
    assert on['sources'] and on['reasoning']['trace']['retrieval_context'] == 'previous_user_turn'
    off = await service.answer(request(sid, '请再解释一下', use_conversation_context=False), model, manager)
    assert not off['sources'] and off['reasoning']['trace']['retrieval_context'] == 'current_question'
    assert all('UNSUPPORTED_ASSISTANT_PREMISE' not in source['content'] for source in on['sources'])


@pytest.mark.asyncio
async def test_followup_to_missing_paper_does_not_fall_back_to_invented_evidence(workspace):
    root, manager, sid, _ = workspace
    service = ResearchService(root, extractor=text_extractor)
    manager.add_message(sid, 'user', '论文中的 experiment XYZ 结果呢？')
    model = FakeModel()
    result = await service.answer(request(sid, '为什么', use_conversation_context=True), model, manager)
    assert result['answer_mode'] == 'insufficient_evidence' and not model.prompts
