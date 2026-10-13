"""Real scoped chat/storage with original task knowledge and one fake writer.

These tests verify orchestration and provenance, not workplace advice quality.
No credentials, provider calls, profile, or network connection are required.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from docthinker.server.research_extract import extract_chunks
from docthinker.server.research_service import ResearchService, WITHDRAWN_ANALYSIS

with patch("dotenv.load_dotenv", return_value=False):
    from docthinker.session_manager import SessionManager


TASK = "我们的跨部门沟通总是卡住，负责人不明确，应该怎么安排分工？"
DOCUMENT_TASK = "根据上传的项目资料，分析项目延期原因，并建议怎样核实事实。"
ORIGINAL = "FAKE_WRITER_ORIGINAL_NOT_VERIFIED：先核实交付依赖，再提出可验证的方法建议。"


async def extractor(content, filename):
    return {"chunks": extract_chunks(content, ".txt")}


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    manager = SessionManager(str(tmp_path / "_system"), str(tmp_path), migrate_legacy=False)
    sid = manager.create_session("Own workplace question")["id"]
    service = ResearchService(tmp_path, extractor=extractor, context_assistance_enabled=False,
                              semantic_retriever=None, evidence_reranker=None, query_translator=None)
    return service, manager, sid


def request(sid, question, **controls):
    # No account profile or optional user model configuration: a plain research
    # request is enough, including clients predating knowledge_skill_mode.
    return SimpleNamespace(session_id=sid, question=question, reasoning_mode="auto",
                           use_memory=False, use_conversation_context=False, **controls)


async def upload(project):
    service, _, sid = project
    content = "项目延期两周。研发交付依赖设计确认。设计尚未确认，负责人未定。"
    document = service.store.accept_document(sid, "project-notes.txt", content.encode())
    assert await service.process(sid, document["id"])
    return document


async def ask(project, question=TASK, *, answer=None, callback=None, **controls):
    service, manager, sid = project
    calls = []

    async def writer(prompt, **kwargs):
        calls.append((prompt, kwargs))
        if callback:
            callback()
        return answer if answer is not None else ORIGINAL + " [K1]"

    result = await service.answer(request(sid, question, **controls), writer, manager)
    return result, calls


def restored(project):
    service, _, sid = project
    root = service.store.root
    manager = SessionManager(str(root / "_system"), str(root), migrate_legacy=False)
    service = ResearchService(root, extractor=extractor, semantic_retriever=None,
                              evidence_reranker=None, query_translator=None)
    return service, manager, sid


@pytest.mark.asyncio
@pytest.mark.parametrize("controls", [{}, {"knowledge_skill_mode": "auto"}])
async def test_current_work_task_auto_loads_original_methods_without_experimental_graph(project, controls):
    service, manager, sid = project
    assert service.context_assistance_enabled is False
    result, calls = await ask(project, **controls)
    assert len(calls) == 1
    skill = result["knowledge_skill"]
    assert skill["id"] == "mao-workplace" and skill["version"] == "2"
    assert skill["status"] == "ready"
    assert skill["authorship"] == "project_original_method_design"
    assert skill["validation_status"] == "design_unvalidated"
    assert 1 <= len(skill["sources"]) <= 3
    assert skill["sources"][0]["label"] == "K1"
    assert result["sources"] == [] and result["answer_mode"] == "general"
    assert result["citation_validation"]["valid"]
    assert result["citation_validation"]["entailment_verified"] is False
    assert result["research_provenance_required"] is True
    assert "KNOWLEDGE_SKILL_DATA_JSON" in calls[0][0]
    assert "decision_rule" in calls[0][0] and "stop_conditions" in calls[0][0]
    assert result["reasoning"]["trace"]["knowledge_skill_router"]["status"] == "selected"
    assert result["reasoning"]["trace"]["knowledge_skill_retrieval"]["model_calls"] == 0
    assert result["reasoning"]["trace"]["context_budget"]["knowledge_skill_bytes"] <= 3500
    assert result["reasoning"]["trace"]["context_budget"]["input_bytes"] <= 24576
    assert len(manager.get_history(sid)) == 2
    assert service.store.list_memories(sid) == []
    assert service.store.list_documents(sid) == [] and service.all_chunks(sid) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("question", "mode"), [
    (TASK, "off"), ("你好，今天心情不错。", "auto"),
    ("优化模型训练代码的GPU任务优先级", "auto"),
    ("不要用知识辅助，帮我协调跨部门沟通。", "auto"),
    ('请翻译“跨部门沟通卡住了，项目延期”，不要分析。', "auto"),
])
async def test_off_or_non_task_requests_do_not_read_pack_or_attach_card(project, monkeypatch, question, mode):
    monkeypatch.setattr("docthinker.server.research_knowledge.retrieve_knowledge_skill",
                        lambda *args, **kwargs: pytest.fail("No routed task must mean no pack read"))
    result, calls = await ask(project, question, answer="普通聊天回答。", knowledge_skill_mode=mode)
    assert len(calls) == 1
    assert "knowledge_skill" not in result
    assert "KNOWLEDGE_SKILL_DATA_JSON" not in calls[0][0]


@pytest.mark.asyncio
async def test_current_task_not_previous_chat_selects_pack(project, monkeypatch):
    service, manager, sid = project
    manager.add_message(sid, "user", TASK)
    manager.add_message(sid, "assistant", "此前讨论了团队沟通。")
    monkeypatch.setattr("docthinker.server.research_knowledge.retrieve_knowledge_skill",
                        lambda *args, **kwargs: pytest.fail("Old topic must not route current task"))
    calls = []

    async def writer(prompt, **kwargs):
        calls.append(prompt)
        return "普通交流。"

    req = request(sid, "今天晚上吃什么？", knowledge_skill_mode="auto")
    req.use_conversation_context = True
    result = await service.answer(req, writer, manager)
    assert len(calls) == 1 and "knowledge_skill" not in result
    assert "KNOWLEDGE_SKILL_DATA_JSON" not in calls[0]


@pytest.mark.asyncio
async def test_missing_uploaded_material_cannot_be_replaced_with_methods(project):
    result, calls = await ask(project, DOCUMENT_TASK, answer=ORIGINAL + " [K1]")
    assert not calls
    assert result["answer_mode"] == "insufficient_evidence"
    assert result["sources"] == []
    assert not result.get("knowledge_skill", {}).get("sources")
    assert result.get("knowledge_skill", {}).get("status") != "ready"
    assert "不能" in result["answer"]


@pytest.mark.asyncio
async def test_uploaded_s_and_method_k_stay_separate_in_response_and_store(project):
    await upload(project)
    service, _, sid = project
    result, calls = await ask(project, DOCUMENT_TASK, answer=ORIGINAL + " [S1] [K1]")
    assert len(calls) == 1
    assert result["answer_mode"] == "evidence"
    assert result["sources"][0]["label"] == "S1"
    assert result["knowledge_skill"]["sources"][0]["label"] == "K1"
    assert result["citation_validation"]["valid"]
    assert result["citation_validation"]["used_labels"] == ["S1", "K1"]
    assert set(source["id"] for source in result["sources"]).isdisjoint(
        source["id"] for source in result["knowledge_skill"]["sources"])
    metadata = service.store.answer_metadata(sid, [result["answer_id"]])[result["answer_id"]]
    assert all(source["label"].startswith("S") for source in metadata["source_refs"])
    assert len(service.store.list_documents(sid)) == 1
    assert service.store.list_memories(sid) == []


@pytest.mark.asyncio
async def test_method_citation_does_not_cover_missing_uploaded_citation(project):
    await upload(project)
    result, calls = await ask(project, DOCUMENT_TASK, answer=ORIGINAL + " [K1]")
    assert len(calls) == 1
    assert result["sources"] and result["knowledge_skill"]["status"] == "ready"
    assert result["citation_validation"]["missing_citations"] is True
    assert result["citation_validation"]["valid"] is False
    assert result["citation_validation"]["unknown_labels"] == []


@pytest.mark.asyncio
async def test_unselected_k_label_is_not_authorized_by_model_itself(project):
    result, calls = await ask(project, answer=ORIGINAL + " [K1]", knowledge_skill_mode="off")
    assert len(calls) == 1 and "knowledge_skill" not in result
    assert result["citation_validation"]["unknown_labels"] == ["K1"]
    assert not result["citation_validation"]["valid"]


@pytest.mark.asyncio
async def test_lowercase_known_k_is_normalized_in_both_receipts(project):
    result, calls = await ask(project, answer=ORIGINAL + " [k1]")
    assert len(calls) == 1
    assert result["citation_validation"]["valid"]
    assert result["knowledge_skill"]["citation_validation"]["unknown_labels"] == []
    assert result["knowledge_skill"]["citation_validation"]["missing_citations"] is False


@pytest.mark.asyncio
async def test_restart_preserves_exact_versioned_method_snapshot_without_retrieval(project, monkeypatch):
    result, _ = await ask(project)
    monkeypatch.setattr("docthinker.server.research_knowledge.retrieve_knowledge_skill",
                        lambda *args, **kwargs: pytest.fail("History must not rerun knowledge retrieval"))
    service, manager, sid = restored(project)
    answer = service.hydrate_history(sid, manager.get_history(sid))[-1]
    assert answer["content"] == result["answer"]
    assert answer["knowledge_skill"] == result["knowledge_skill"]
    assert answer["citation_validation"]["valid"]
    assert answer["sources"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize('missing', ['proof', 'all_metadata'])
async def test_status_replay_cannot_keep_cached_proof_text_after_metadata_loss(project, monkeypatch, missing):
    service, manager, sid = project
    result, _ = await ask(project)
    stale = {**result, 'id': result['answer_id'], 'role': 'assistant',
             'content': result['answer'],
             'reasoning': {'claims': [{'quote': 'STALE_PRIVATE_PROOF_TEXT'}]},
             'claim_audit': {'claims': [{'quote': 'STALE_PRIVATE_AUDIT_TEXT'}]}}
    if missing == 'proof':
        with service.store._write() as db:
            db.execute('DELETE FROM research_answer_evidence WHERE session_id=? AND answer_id=?',
                       (sid, result['answer_id']))
    else:
        monkeypatch.setattr(service.store, 'answer_metadata', lambda *args: {})
    refreshed = service.hydrate_history(sid, [stale])[0]
    assert refreshed['content'] == refreshed['answer'] == WITHDRAWN_ANALYSIS
    assert 'STALE_PRIVATE' not in str(refreshed)
    assert not refreshed.get('knowledge_skill', {}).get('sources')
    assert 'reasoning' not in refreshed


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["answer_store", "proof_store"])
async def test_method_metadata_outage_hides_marked_answer_after_restart(project, monkeypatch, failure):
    service, manager, sid = project

    def fail(*args, **kwargs):
        raise OSError("private metadata diagnostic must not leak")

    monkeypatch.setattr(service.store if failure == "answer_store" else service.evidence,
                        "record_answer" if failure == "answer_store" else "record", fail)
    result, calls = await ask(project)
    assert len(calls) == 1
    assert result["answer"] == WITHDRAWN_ANALYSIS
    assert result["sources"] == [] and result["research_provenance_required"] is True
    assert result["knowledge_skill"]["status"] == "unavailable"
    assert result["knowledge_skill"]["sources"] == []
    assert "private metadata" not in str(result)
    assert manager.get_history(sid)[-1]["research_provenance_required"] is True
    restarted, restarted_manager, sid = restored(project)
    answer = restarted.hydrate_history(sid, restarted_manager.get_history(sid))[-1]
    assert answer["content"] == WITHDRAWN_ANALYSIS
    assert not answer.get("knowledge_skill", {}).get("sources")
    assert answer["sources"] == []


@pytest.mark.asyncio
async def test_revoked_uploaded_premise_hides_derived_answer_and_method_card(project):
    document = await upload(project)
    service, manager, sid = project
    result, calls = await ask(project, DOCUMENT_TASK, answer=ORIGINAL + " [S1] [K1]",
                              callback=lambda: service.store.delete_document(sid, document["id"]))
    assert len(calls) == 1
    assert result["answer"] == WITHDRAWN_ANALYSIS and result["sources"] == []
    assert result["knowledge_skill"]["status"] == "unavailable"
    assert result["knowledge_skill"]["sources"] == []
    restored_service, restored_manager, sid = restored(project)
    answer = restored_service.hydrate_history(sid, restored_manager.get_history(sid))[-1]
    assert answer["content"] == WITHDRAWN_ANALYSIS
    assert not answer.get("knowledge_skill", {}).get("sources")


@pytest.mark.asyncio
async def test_later_source_deletion_hides_historical_derived_method_response(project):
    document = await upload(project)
    service, manager, sid = project
    result, _ = await ask(project, DOCUMENT_TASK, answer=ORIGINAL + " [S1] [K1]")
    assert result["answer"].startswith(ORIGINAL)
    service.store.delete_document(sid, document["id"])
    answer = service.hydrate_history(sid, manager.get_history(sid))[-1]
    assert answer["content"] == WITHDRAWN_ANALYSIS
    assert not answer.get("knowledge_skill", {}).get("sources")


@pytest.mark.asyncio
async def test_hidden_method_answer_cannot_leak_through_next_context(project, monkeypatch):
    service, manager, sid = project

    def fail(*args, **kwargs):
        raise OSError("synthetic outage")

    monkeypatch.setattr(service.evidence, "record", fail)
    await ask(project)
    restarted, restarted_manager, sid = restored(project)
    calls = []

    async def writer(prompt, **kwargs):
        calls.append(prompt)
        return "重新讨论当前问题。"

    req = request(sid, "今天晚上吃什么？", knowledge_skill_mode="auto")
    req.use_conversation_context = True
    await restarted.answer(req, writer, restarted_manager)
    assert len(calls) == 1 and ORIGINAL not in calls[0]
    assert "KNOWLEDGE_SKILL_DATA_JSON" not in calls[0]


@pytest.mark.asyncio
async def test_missing_pack_falls_back_to_one_normal_writer_without_claiming_methods(project, monkeypatch):
    def missing(_):
        raise FileNotFoundError("/private/pack/path")

    monkeypatch.setattr("docthinker.server.knowledge_skills._read_pack", missing)
    result, calls = await ask(project, answer="先弄清楚事实，再讨论工作安排。")
    assert len(calls) == 1
    assert not result.get("knowledge_skill", {}).get("sources")
    assert result.get("knowledge_skill", {}).get("status") != "ready"
    assert "KNOWLEDGE_SKILL_DATA_JSON" not in calls[0][0]
    assert "/private/pack/path" not in str(result)


@pytest.mark.asyncio
async def test_invalid_mode_is_rejected_before_model_or_history(project):
    service, manager, sid = project
    with pytest.raises(ValueError, match="knowledge skill mode"):
        await ask(project, knowledge_skill_mode="unbounded")
    assert manager.get_history(sid) == []


@pytest.mark.asyncio
async def test_method_snapshots_are_project_scoped(project):
    result, _ = await ask(project)
    service, manager, _ = project
    other = manager.create_session("Other project")["id"]
    assert service.store.answer_metadata(other, [result["answer_id"]]) == {}
    assert service.evidence.hydrate(other, result["answer_id"], []) == {}


@pytest.mark.asyncio
async def test_literal_document_mode_never_loads_method_pack_or_calls_writer(project, monkeypatch):
    await upload(project)
    service, manager, sid = project
    monkeypatch.setattr("docthinker.server.research_knowledge.retrieve_knowledge_skill",
                        lambda *args, **kwargs: pytest.fail("Literal excerpts must not load a method pack"))
    req = request(sid, DOCUMENT_TASK, knowledge_skill_mode="auto")
    req.reasoning_mode = "evidence"

    async def writer(*args, **kwargs):
        pytest.fail("Literal excerpt mode does not require a model")

    result = await service.answer(req, writer, manager)
    assert result["sources"] and result["answer_mode"] == "evidence"
    assert "knowledge_skill" not in result


@pytest.mark.asyncio
async def test_metadata_failure_clears_real_s_sources_as_well_as_method_card(project, monkeypatch):
    await upload(project)
    service, _, _ = project

    def fail(*args, **kwargs):
        raise OSError("synthetic provenance outage")

    monkeypatch.setattr(service.evidence, "record", fail)
    result, calls = await ask(project, DOCUMENT_TASK, answer=ORIGINAL + " [S1] [K1]")
    assert len(calls) == 1
    assert "项目延期两周" in calls[0][0]
    assert result["answer"] == WITHDRAWN_ANALYSIS
    assert result["sources"] == [] and result["knowledge_skill"]["sources"] == []


@pytest.mark.asyncio
async def test_optional_method_budget_rejection_preserves_original_sources(project, monkeypatch):
    await upload(project)
    baseline, _ = await ask(project, DOCUMENT_TASK, answer=ORIGINAL + " [S1]", knowledge_skill_mode="off")
    from docthinker.server import research_prompt_budget as budgeting
    validate = budgeting.validate_invite_messages
    attempts = []

    def limited_window(messages):
        attempts.append(messages)
        # Deterministically exercise the fit retry boundary, then delegate the
        # accepted candidate to the real input-budget validator.
        if "KNOWLEDGE_SKILL_DATA_JSON" in messages[1]["content"]:
            raise ValueError("Test-only smaller window rejects optional block")
        return validate(messages)

    monkeypatch.setattr(budgeting, "validate_invite_messages", limited_window)
    result, calls = await ask(project, DOCUMENT_TASK, answer=ORIGINAL + " [S1]")
    assert len(calls) == 1 and len(attempts) >= 2
    assert [source["id"] for source in result["sources"]] == [source["id"] for source in baseline["sources"]]
    assert result["knowledge_skill"]["status"] == "budget_omitted"
    assert result["knowledge_skill"]["sources"] == []
    assert "KNOWLEDGE_SKILL_DATA_JSON" not in calls[0][0]
    assert result["citation_validation"]["valid"]
