"""Personal-agent lifecycle contracts through real storage and answer plumbing.

Generation is a deterministic echo fixture, not an answer-quality benchmark.
No provider, pretrained local model, external service or paid API is used.
"""
from __future__ import annotations

import uuid

import pytest

from docthinker.server.research_service import ResearchService
from docthinker.server.research_store import ResearchStoreError
from tests.test_research_service_unit import (
    FakeModel, SessionManager, document, request, text_extractor, workspace,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('PYTHON_DOTENV_DISABLED', '1')
    monkeypatch.setenv('DOCTHINKER_RESEARCH_SEMANTIC', 'off')


def service_at(root):
    return ResearchService(root, extractor=text_extractor,
        context_assistance_enabled=False, semantic_retriever=None,
        evidence_reranker=None, query_translator=None)


def restart_manager(root):
    return SessionManager(str(root / '_system'), str(root), migrate_legacy=False)


def assistant(history, identifier):
    return next(row for row in history if row.get('id') == identifier)


@pytest.mark.asyncio
async def test_household_budget_correction_restart_and_new_chat_isolation(workspace):
    root, manager, first, second = workspace
    now = [1000.0]
    service = service_at(root)
    service.store.clock = lambda: now[0]
    try:
        budget = service.store.create_memory(first, 'decision',
            '家庭预算确认：每月生活开支上限为5000元；不得自动转账。', valid_from=1000)
        now[0] = 2000
        corrected = service.store.update_memory(first, budget['id'], expected_revision=1,
            text='家庭预算更正：每月生活开支上限为4500元；不得自动转账。', valid_from=1500)
        assert corrected['revision'] == 2 and corrected['recorded_at'] == 2000
    finally:
        await service.stop()

    service = service_at(root)
    service.store.clock = lambda: now[0]
    manager = restart_manager(root)
    try:
        model = FakeModel('目前已确认预算为4500元，未执行转账。')
        result = await service.answer(request(first, '我已确认的决定和家庭预算是什么？', use_memory=True), model, manager)
        assert result['used_memory_count'] == 1 and len(model.prompts) == 1
        assert corrected['text'] in model.prompts[0] and budget['text'] not in model.prompts[0]
        history = service.store.memory_history(first, budget['id'])['items']
        assert [row['revision'] for row in history] == [2, 1]
        assert history[1]['valid_until'] == 1500 and history[1]['recorded_until'] == 2000
        before = service.store.search_memories(first, '家庭预算', as_of=1499)
        at_change = service.store.search_memories(first, '家庭预算', as_of=1500)
        assert [row['revision'] for row in before] == [1]
        assert [row['revision'] for row in at_change] == [2]
        # A new chat is not implicit permission to disclose another chat's
        # private note. Personal-scope consent is a separate capability.
        other = FakeModel('这里没有已确认的家庭预算。')
        isolated = await service.answer(request(second, '我的记忆有哪些？', use_memory=True), other, manager)
        assert isolated['used_memory_count'] == 0 and len(other.prompts) == 1
        assert budget['text'] not in other.prompts[0] and corrected['text'] not in other.prompts[0]
    finally:
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize('origin', ['foreign_session', 'nonexistent'])
async def test_confirmed_work_decision_cannot_forge_answer_provenance(workspace, origin):
    root, manager, first, second = workspace
    service = service_at(root)
    try:
        await document(service, first, '客户交付截止日期为2026-10-30；客户数据不得外发。')
        result = await service.answer(request(first, '交付截止日期是什么？'),
            FakeModel('交付截止日期为2026-10-30 [S1]。'), manager)
        destination = second if origin == 'foreign_session' else first
        answer_id = result['answer_id'] if origin == 'foreign_session' else str(uuid.uuid4())
        with pytest.raises(ResearchStoreError):
            service.store.create_memory(destination, 'decision', '未经核对的截止日期确认。',
                source_answer_id=answer_id)
        assert service.store.list_memories(destination) == []
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_work_deadline_change_keeps_event_time_and_revokes_only_live_source_dependents(workspace):
    root, _, first, second = workspace
    now = [1000.0]
    service = service_at(root)
    service.store.clock = lambda: now[0]
    try:
        old_doc = await document(service, first, '客户通知：项目交付截止日期为2026-10-30。')
        old_chunk = next(row for row in service.all_chunks(first) if row['document_id'] == old_doc['id'])
        note = service.store.create_memory(first, 'decision', '客户确认交付截止日期为2026-10-30。',
            source_chunk_ids=[old_chunk['id']], valid_from=1000)
        now[0] = 2000
        new_doc = await document(service, first, '客户变更通知：交付截止日期改为2026-11-06；之前日期作废。')
        new_chunk = next(row for row in service.all_chunks(first) if row['document_id'] == new_doc['id'])
        revised = service.store.update_memory(first, note['id'], expected_revision=1,
            text='客户更正确认交付截止日期为2026-11-06，之前日期作废。',
            source_chunk_ids=[new_chunk['id']], valid_from=1500)
        assert revised['source_chunk_ids'] == [new_chunk['id']]
        service.store.delete_document(first, old_doc['id'])
        assert service.store.get_memory(first, note['id'])['status'] == 'active'
        revisions = service.store.memory_history(first, note['id'])['items']
        assert revisions[0]['text'] == revised['text']
        assert revisions[1]['text'] == '' and revisions[1]['redacted'] is True
        assert service.store.search_memories(first, '交付截止日期', as_of=1499) == []
        assert service.store.search_memories(first, '交付截止日期', as_of=1500)[0]['revision'] == 2
        assert service.store.search_memories(second, '交付截止日期') == []
        now[0] = 2100
        removed = service.store.delete_document(first, new_doc['id'])
        assert note['id'] in removed['invalidated_memory_ids']
    finally:
        await service.stop()
    restarted = service_at(root)
    restarted.store.clock = lambda: now[0]
    try:
        assert restarted.store.search_memories(first, '交付截止日期') == []
        assert all(row['text'] == '' and row['redacted'] for row in
            restarted.store.memory_history(first, note['id'])['items'])
    finally:
        await restarted.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize('mutation', ['delete', 'correct', 'expire'])
async def test_withdrawn_private_budget_cannot_return_through_saved_assistant_context(workspace, mutation):
    root, manager, first, _ = workspace
    now = [1000.0]
    secret = '私密预算4827元'
    old_body = '你确认的是私密预算4827元；我没有执行任何支付。'
    service = service_at(root)
    service.store.clock = lambda: now[0]
    try:
        note = service.store.create_memory(first, 'decision', secret + '，不得自动付款。',
            valid_from=1000, valid_until=1300 if mutation == 'expire' else None)
        original_model = FakeModel(old_body)
        original = await service.answer(request(first, '我已确认的决定是什么？', use_memory=True), original_model, manager)
        assert original['used_memory_count'] == 1 and secret in original_model.prompts[0]
        assert original['answer'] == old_body and len(original_model.prompts) == 1
        now[0] = 1400
        if mutation == 'delete':
            service.store.delete_memory(first, note['id'], expected_revision=1)
        elif mutation == 'correct':
            service.store.update_memory(first, note['id'], expected_revision=1,
                text='私密预算已经更正为4300元，不得自动付款。', valid_from=1400)
    finally:
        await service.stop()
    restarted = service_at(root)
    restarted.store.clock = lambda: now[0]
    manager = restart_manager(root)
    try:
        visible = restarted.hydrate_history(first, manager.get_history(first))
        old_answer = assistant(visible, original['answer_id'])
        assert secret not in old_answer['content'], 'Withdrawn premise remains visible through its saved assistant answer'
        next_model = FakeModel('根据目前有效记录继续，尚未执行任何支付。')
        await restarted.answer(request(first, '下一步呢？', use_memory=True, use_conversation_context=True), next_model, manager)
        assert len(next_model.prompts) == 1 and secret not in next_model.prompts[0]
        if mutation == 'correct':
            assert '私密预算已经更正为4300元' in next_model.prompts[0]
        else:
            assert restarted.store.search_memories(first, '预算') == []
    finally:
        await restarted.stop()


@pytest.mark.asyncio
async def test_memory_forgotten_during_generation_is_not_published_or_retried(workspace):
    root, manager, first, _ = workspace
    service = service_at(root)
    secret = '临时私人地址虹桥路777号'
    note = service.store.create_memory(first, 'decision', secret + '，仅本次送货使用。')
    calls = []
    async def model(prompt, **kwargs):
        calls.append(prompt)
        assert secret in prompt
        service.store.delete_memory(first, note['id'], expected_revision=1)
        return '将送货至' + secret + '，尚未执行。'
    try:
        result = await service.answer(request(first, '我已确认的决定是什么？', use_memory=True), model, manager)
        assert len(calls) == 1 and secret not in result['answer']
        visible = service.hydrate_history(first, manager.get_history(first))
        assert secret not in assistant(visible, result['answer_id'])['content']
        assert service.store.search_memories(first, '地址') == []
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_document_revocation_cannot_return_through_an_answer_using_derived_memory(workspace):
    root, manager, first, _ = workspace
    service = service_at(root)
    secret = '客户专属项目代号hiddenpine'
    try:
        source = await document(service, first, secret + '；仅用于本项目，不得向其他项目公开。')
        chunk = service.all_chunks(first)[0]
        note = service.store.create_memory(first, 'decision', secret + '需要内部审查后再交付。',
            source_chunk_ids=[chunk['id']])
        model = FakeModel(secret + '已确认需要内部审查，尚未执行。')
        result = await service.answer(request(first, '我已确认的决定是什么？', use_memory=True), model, manager)
        assert result['used_memory_count'] == 1 and secret in model.prompts[0]
        service.store.delete_document(first, source['id'])
        assert service.store.search_memories(first, 'hiddenpine') == []
        visible = service.hydrate_history(first, manager.get_history(first))
        assert secret not in assistant(visible, result['answer_id'])['content']
        assert service.store.memory_history(first, note['id'])['items'][0]['redacted'] is True
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_answer_only_memory_origin_is_a_live_dependency_not_a_decorative_pointer(workspace):
    root, manager, first, _ = workspace
    service = service_at(root)
    note = None
    try:
        source = await document(service, first, '客户确认：交付截止为2026-11-06；此通知替代之前日期。')
        answer = await service.answer(request(first, '交付截止是什么？'),
            FakeModel('客户确认截止为2026-11-06 [S1]。'), manager)
        assert answer['sources'] and answer['citation_validation']['valid']
        note = service.store.create_memory(first, 'finding',
            '客户截止日期来自刚才核对的通知，仍需要按原文核对。', source_answer_id=answer['answer_id'])
        assert service.store.search_memories(first, '客户截止日期')[0]['id'] == note['id']
        service.store.delete_document(first, source['id'])
        assert service.store.search_memories(first, '客户截止日期') == [], 'Answer-only provenance bypasses document revocation'
    finally:
        await service.stop()
    restarted = service_at(root)
    try:
        assert restarted.store.search_memories(first, '客户截止日期') == []
        assert all(row['text'] == '' and row['redacted'] for row in
            restarted.store.memory_history(first, note['id'])['items'])
    finally:
        await restarted.stop()


@pytest.mark.asyncio
async def test_forgotten_budget_cannot_reenter_as_a_second_generation_confirmed_note(workspace):
    root, manager, first, _ = workspace
    service = service_at(root)
    try:
        original = service.store.create_memory(first, 'decision', '私人预算为4827元，不得自动付款。')
        answer = await service.answer(request(first, '我已确认的决定是什么？', use_memory=True),
            FakeModel('私人预算为4827元，尚未执行。'), manager)
        assert answer['used_memory_count'] == 1 and not answer['sources']
        derived = service.store.create_memory(first, 'finding',
            '派生家庭决策：按之前确认的4827元预算安排支出，仅待用户批准。',
            source_answer_id=answer['answer_id'])
        assert derived['id'] in {row['id'] for row in service.store.search_memories(first, '派生家庭决策')}
        service.store.delete_memory(first, original['id'], expected_revision=1)
        assert service.store.search_memories(first, '派生家庭决策') == [], 'A second-generation note resurrects its forgotten premise'
    finally:
        await service.stop()
    restarted = service_at(root)
    try:
        assert restarted.store.search_memories(first, '派生家庭决策') == []
        assert all(row['redacted'] and row['text'] == '' for row in
            restarted.store.memory_history(first, derived['id'])['items'])
    finally:
        await restarted.stop()


@pytest.mark.asyncio
async def test_memory_dependency_follows_conversation_even_when_memory_toggle_is_off(workspace):
    root, manager, first, _ = workspace
    service = service_at(root)
    secret = '私人长期预算4827元'
    try:
        note = service.store.create_memory(first, 'decision', secret + '，所有付款另需批准。')
        first_model = FakeModel(secret + '是你当前确认记录；尚未付款。')
        first_answer = await service.answer(request(first, '我已确认的决定是什么？', use_memory=True), first_model, manager)
        second_model = FakeModel('根据上一轮，' + secret + '只用于规划，不执行付款。')
        second_answer = await service.answer(request(first, '继续，给一个不执行的安排。',
            use_memory=False, use_conversation_context=True), second_model, manager)
        assert first_answer['used_memory_count'] == 1
        assert second_answer['used_memory_count'] == 0 and secret in second_model.prompts[0]
        assert len(first_model.prompts) == len(second_model.prompts) == 1
        service.store.delete_memory(first, note['id'], expected_revision=1)
    finally:
        await service.stop()
    restarted = service_at(root)
    manager = restart_manager(root)
    try:
        visible = restarted.hydrate_history(first, manager.get_history(first))
        for result in (first_answer, second_answer):
            assert secret not in assistant(visible, result['answer_id'])['content']
        next_model = FakeModel('不再复用已经撤回的预算，尚未执行。')
        await restarted.answer(request(first, '还有什么要确认？',
            use_memory=False, use_conversation_context=True), next_model, manager)
        assert len(next_model.prompts) == 1 and secret not in next_model.prompts[0]
    finally:
        await restarted.stop()
