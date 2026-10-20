"""Offline method-plan contracts through real scoped storage and a fake writer.

These original snippets test orchestration and source geometry, not the quality
of a scientific conclusion. No credentials, provider, network or holdout data
are used. The writer always makes one explicit call or none for literal reads.
"""

from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from docthinker.server.research_service import ResearchService, WITHDRAWN_ANALYSIS

with patch("dotenv.load_dotenv", return_value=False):
    from docthinker.session_manager import SessionManager


TASK = "对比这两篇论文的指标和实验条件，整理冲突与证据缺口。"
WRITING_TASK = "根据上传的项目资料写项目报告，组织论点与证据，说明样本限制。"
CHUNK_SEPARATOR = "\n===SELF_AUTHORED_TEST_CHUNK===\n"
SNIPPETS = (
    "论文甲原始测试资料：比较检索系统，实验对象是数据集甲，指标为准确率0.82；"
    "实验条件为固定查询长度，样本限制是只有三个查询。",
    "论文乙原始测试资料：比较检索系统，实验对象是数据集乙，指标为召回率0.88；"
    "实验条件为可变查询长度，样本限制是未重复实验。",
)


async def _extractor(content, filename):
    return {"chunks": [{"text": text} for text in content.decode().split(CHUNK_SEPARATOR)]}


def _service(root):
    return ResearchService(root, extractor=_extractor, context_assistance_enabled=False,
                           semantic_retriever=None, evidence_reranker=None,
                           query_translator=None)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("DOCTHINKER_RESEARCH_SEMANTIC", "off")
    manager = SessionManager(str(tmp_path / "_system"), str(tmp_path), migrate_legacy=False)
    sid = manager.create_session("Synthetic method-plan project")["id"]
    return _service(tmp_path), manager, sid


async def _upload(project, *, texts=SNIPPETS, one_document=False):
    service, _, sid = project
    materials = [CHUNK_SEPARATOR.join(texts)] if one_document else list(texts)
    documents = []
    for index, text in enumerate(materials):
        document = service.store.accept_document(sid, f"self-authored-{index}.txt", text.encode())
        assert await service.process(sid, document["id"])
        documents.append(document)
    return documents


def _request(sid, question, **controls):
    return SimpleNamespace(session_id=sid, question=question,
                           reasoning_mode=controls.pop("reasoning_mode", "auto"),
                           knowledge_skill_mode=controls.pop("knowledge_skill_mode", "auto"),
                           use_memory=False, use_conversation_context=False, **controls)


async def _ask(project, question=TASK, *, callback=None, **controls):
    service, manager, sid = project
    calls = []

    async def writer(prompt, **kwargs):
        calls.append((prompt, kwargs))
        if callback is not None:
            callback()
        return "测试改写仍须人工核对，方法计划不是语义证明。[S1] [K1]"

    return await service.answer(_request(sid, question, **controls), writer, manager), calls


def _payload(prompt, marker):
    value, _ = json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])
    return value


def _assert_actual_plan(result, calls):
    from docthinker.server.method_planner import compile_method_plan

    assert len(calls) == 1
    card = result["knowledge_skill"]
    assert card["status"] == "ready"
    packet = _payload(calls[0][0], "KNOWLEDGE_SKILL_DATA_JSON:\n")
    assert packet["sources"] == card["sources"]
    assert packet["method_plan"] == card["method_plan"]
    assert card["method_plan"] == compile_method_plan(
        card["sources"], result["sources"], skill_id=card["id"])
    assert card["method_plan"]["source_labels"] == [row["label"] for row in result["sources"]]
    assert card["method_plan"]["model_calls"] == 0
    assert card["method_plan"]["semantic_entailment_verified"] is False
    assert card["method_plan"]["method_execution_claimed"] is False
    assert card["method_plan"]["normal_answer_allowed"] is True
    assert card["method_plan"]["methods"]
    assert all(row["requirements_status"] == row["stop_conditions_status"] == "pending_review"
               for row in card["method_plan"]["methods"])
    assert len(json.dumps(card["method_plan"], ensure_ascii=False,
                          separators=(",", ":")).encode()) <= 1600
    budget = result["reasoning"]["trace"]["context_budget"]
    assert 0 < budget["knowledge_skill_bytes"] <= 3500
    assert budget["input_bytes"] <= 24576
    assert result["citation_validation"]["entailment_verified"] is False
    assert result["claim_audit"]["entailment_verified"] is False
    return card["method_plan"]


@pytest.mark.asyncio
async def test_same_document_multiple_chunks_are_not_independent_documents(project):
    await _upload(project, one_document=True)
    result, calls = await _ask(project)
    assert len(result["sources"]) == 2
    assert len({row["document_id"] for row in result["sources"]}) == 1
    plan = _assert_actual_plan(result, calls)
    assert plan["evidence"]["distinct_documents"] == 1
    assert "insufficient_distinct_documents" in plan["structural_gaps"]


@pytest.mark.asyncio
async def test_final_budget_source_variant_owns_the_method_plan(project, monkeypatch):
    await _upload(project)
    baseline, baseline_calls = await _ask(project, knowledge_skill_mode="off")
    assert len(baseline_calls) == 1 and len(baseline["sources"]) == 2
    from docthinker.server import research_prompt_budget as budgeting

    original_validator = budgeting.validate_invite_messages
    rejected_counts = []

    def only_one_whole_excerpt(messages):
        packet = _payload(messages[1]["content"], "UNTRUSTED_RESEARCH_DATA_JSON:\n")
        if len(packet["source_quotes"]) > 1:
            rejected_counts.append(len(packet["source_quotes"]))
            raise ValueError("Synthetic provider window permits one complete source")
        return original_validator(messages)

    monkeypatch.setattr(budgeting, "validate_invite_messages", only_one_whole_excerpt)
    result, calls = await _ask(project)
    assert rejected_counts and len(result["sources"]) == 1
    plan = _assert_actual_plan(result, calls)
    assert plan["evidence"]["selected_sources"] == 1
    assert plan["evidence"]["distinct_documents"] == 1
    assert "insufficient_distinct_documents" in plan["structural_gaps"]
    assert any(row["code"] == "context_budget_omitted" for row in result["retrieval_gaps"])


@pytest.mark.asyncio
async def test_optional_method_plan_leaves_before_any_original_source(project, monkeypatch):
    await _upload(project)
    baseline, _ = await _ask(project, knowledge_skill_mode="off")
    from docthinker.server import research_prompt_budget as budgeting

    original_validator = budgeting.validate_invite_messages

    def reject_only_optional_method_blocks(messages):
        if "KNOWLEDGE_SKILL_DATA_JSON" in messages[1]["content"]:
            raise ValueError("Synthetic provider window omits the optional method block")
        return original_validator(messages)

    monkeypatch.setattr(budgeting, "validate_invite_messages", reject_only_optional_method_blocks)
    result, calls = await _ask(project)
    assert len(calls) == 1
    assert result["sources"] == baseline["sources"]
    assert result["knowledge_skill"]["status"] == "budget_omitted"
    assert result["knowledge_skill"]["sources"] == []
    assert "method_plan" not in result["knowledge_skill"]
    assert "KNOWLEDGE_SKILL_DATA_JSON" not in calls[0][0]
    assert result["reasoning"]["trace"]["context_budget"]["knowledge_skill_bytes"] == 0


@pytest.mark.asyncio
async def test_real_store_plan_snapshot_survives_restart_without_resource_reload(project, monkeypatch):
    await _upload(project)
    service, manager, sid = project
    result, calls = await _ask(project)
    _assert_actual_plan(result, calls)
    before = deepcopy(result["knowledge_skill"])
    monkeypatch.setattr("docthinker.server.knowledge_skills._read_pack",
                        lambda *args: pytest.fail("History must replay the saved plan snapshot"))
    restarted = _service(service.store.root)
    restarted_manager = SessionManager(str(service.store.root / "_system"),
                                       str(service.store.root), migrate_legacy=False)
    answer = restarted.hydrate_history(sid, restarted_manager.get_history(sid))[-1]
    assert answer["knowledge_skill"] == before
    assert [row["id"] for row in answer["sources"]] == [row["id"] for row in result["sources"]]
    assert answer["content"] == result["answer"]
    assert service.store.list_memories(sid) == []
    assert not (service.store.root / "answer_feedback.db").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("during_writer", [False, True])
async def test_source_deletion_removes_live_and_historical_method_plan(project, during_writer):
    documents = await _upload(project)
    service, manager, sid = project

    def delete_source():
        service.store.delete_document(sid, documents[0]["id"])

    result, calls = await _ask(project, callback=delete_source if during_writer else None)
    assert len(calls) == 1
    if not during_writer:
        _assert_actual_plan(result, calls)
        delete_source()
    else:
        assert result["answer"] == WITHDRAWN_ANALYSIS
        assert "method_plan" not in result["knowledge_skill"]
    restarted = _service(service.store.root)
    answer = restarted.hydrate_history(sid, manager.get_history(sid))[-1]
    assert answer["content"] == WITHDRAWN_ANALYSIS
    assert not answer.get("knowledge_skill", {}).get("sources")
    assert "method_plan" not in answer.get("knowledge_skill", {})


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["proof", "all_metadata"])
async def test_status_replay_cannot_retain_cached_method_plan_after_metadata_loss(project, monkeypatch, missing):
    await _upload(project)
    service, _, sid = project
    result, calls = await _ask(project)
    _assert_actual_plan(result, calls)
    stale = {**result, "id": result["answer_id"], "role": "assistant",
             "content": result["answer"],
             "knowledge_skill": {**result["knowledge_skill"],
                                 "method_plan": {"synthetic": "STALE_SYNTHETIC_METHOD_PLAN"}}}
    if missing == "proof":
        with service.store._write() as db:
            db.execute('DELETE FROM research_answer_evidence WHERE session_id=? AND answer_id=?',
                       (sid, result['answer_id']))
    else:
        monkeypatch.setattr(service.store, "answer_metadata", lambda *args: {})
    answer = service.hydrate_history(sid, [stale])[0]
    assert answer["content"] == answer["answer"] == WITHDRAWN_ANALYSIS
    assert "STALE_SYNTHETIC_METHOD_PLAN" not in str(answer)
    assert "method_plan" not in answer.get("knowledge_skill", {})


@pytest.mark.asyncio
@pytest.mark.parametrize("controls", [{"knowledge_skill_mode": "off"}, {"reasoning_mode": "evidence"}])
async def test_off_and_literal_paths_load_no_method_resource_or_extra_writer(project, monkeypatch, controls):
    await _upload(project)
    monkeypatch.setattr("docthinker.server.knowledge_skills._read_pack",
                        lambda *args: pytest.fail("Off and literal paths must not read a method pack"))
    result, calls = await _ask(project, **controls)
    assert len(calls) == (0 if controls.get("reasoning_mode") == "evidence" else 1)
    assert "knowledge_skill" not in result
    assert result["sources"]
    assert all("KNOWLEDGE_SKILL_DATA_JSON" not in prompt for prompt, _ in calls)


@pytest.mark.asyncio
async def test_missing_requested_document_cannot_be_replaced_by_method_plan(project):
    result, calls = await _ask(project, WRITING_TASK)
    assert calls == [] and result["answer_mode"] == "insufficient_evidence"
    assert result["sources"] == []
    assert result.get("knowledge_skill", {}).get("status") != "ready"
    assert "method_plan" not in result.get("knowledge_skill", {})


@pytest.mark.asyncio
async def test_real_project_isolation_keeps_the_plan_bound_to_its_own_sources(project):
    service, manager, sid = project
    second_sid = manager.create_session("Second synthetic method-plan project")["id"]
    second = service, manager, second_sid
    await _upload(project, texts=tuple(text + " PROJECT_ONE_ONLY" for text in SNIPPETS))
    await _upload(second, texts=tuple(text + " PROJECT_TWO_ONLY" for text in SNIPPETS))
    first_result, first_calls = await _ask(project)
    second_result, second_calls = await _ask(second)
    _assert_actual_plan(first_result, first_calls)
    _assert_actual_plan(second_result, second_calls)
    assert "PROJECT_ONE_ONLY" in first_calls[0][0]
    assert "PROJECT_TWO_ONLY" not in first_calls[0][0]
    assert "PROJECT_TWO_ONLY" in second_calls[0][0]
    assert "PROJECT_ONE_ONLY" not in second_calls[0][0]
    assert {row["id"] for row in first_result["sources"]}.isdisjoint(
        row["id"] for row in second_result["sources"])
    assert service.store.answer_metadata(second_sid, [first_result["answer_id"]]) == {}
    assert service.evidence.hydrate(second_sid, first_result["answer_id"], second_result["sources"]) == {}
    assert service.store.list_memories(sid) == service.store.list_memories(second_sid) == []


@pytest.mark.asyncio
async def test_repeated_display_id_cannot_cross_real_worker_storage(project, tmp_path):
    first_service, _, first_sid = project
    second_root = tmp_path / "second-worker"
    second_manager = SessionManager(str(second_root / "_system"), str(second_root), migrate_legacy=False)
    second_sid = second_manager.create_session("Independent worker, repeated display ID")["id"]
    second_service = _service(second_root)
    second = second_service, second_manager, second_sid
    assert first_sid == second_sid  # Tenant display IDs deliberately collide.
    await _upload(project, texts=tuple(text + " WORKER_ONE_ONLY" for text in SNIPPETS))
    await _upload(second, texts=tuple(text + " WORKER_TWO_ONLY" for text in SNIPPETS))
    first_result, first_calls = await _ask(project)
    second_result, second_calls = await _ask(second)
    _assert_actual_plan(first_result, first_calls)
    _assert_actual_plan(second_result, second_calls)
    assert "WORKER_TWO_ONLY" not in first_calls[0][0]
    assert "WORKER_ONE_ONLY" not in second_calls[0][0]
    foreign_ids = [row["id"] for row in first_result["sources"]]
    assert second_service.store.source_chunks(second_sid, foreign_ids) == []
    assert second_service.store.answer_metadata(second_sid, [first_result["answer_id"]]) == {}
    assert first_service.store.answer_metadata(first_sid, [second_result["answer_id"]]) == {}
    assert second_service.evidence.hydrate(second_sid, first_result["answer_id"], second_result["sources"]) == {}

