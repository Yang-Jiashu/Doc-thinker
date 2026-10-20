"""Multiple original packs through the real scoped research service.

The writer is deliberately fake. Paper snippets below are self-authored test
materials uploaded through the real document pipeline, not published research
or evidence of answer quality. No provider, credentials, or network is used.
"""

from __future__ import annotations

import json

import pytest

from tests.test_knowledge_skill_integration_unit import ask, project, request, restored


TASKS = {
    "research-ideation": "基于当前观察，帮我提出可检验的科研假设，并设计最小验证。",
    "literature-synthesis": "对比这两篇论文的指标和实验条件，整理冲突与证据缺口。",
    "evidence-writing": "帮我写一份给负责人看的项目报告，组织论点与证据，说明受众和验收要求。",
    "decision-review": "比较两个项目方案的预算、成本和期限，检查硬约束及回退条件。",
}

# Keep exact current versions: changing a pack must not loosen the registry or
# history contracts to accept an arbitrary version.
PACK_VERSIONS = {skill_id: ("2" if skill_id == "decision-review" else "1") for skill_id in TASKS}

MATERIALS = {
    "research-ideation": (
        "研究记录：研究对象为两个检索系统。观察到系统甲在查询较长时漏召回增加。"
        "候选假设为查询截断丢失关键词，需要检查截断位置并作最小对照验证。"
    ),
    "evidence-writing": (
        "项目报告交付要求：受众为项目负责人，需要决定是否继续试点。"
        "现有证据：试点耗时降低十分钟，参与者只有三人；报告必须披露样本限制。"
        "验收要求包括论点、证据出处与待确认项，不能声称已经证明普遍有效。"
    ),
    "decision-review": (
        "项目方案甲成本为八万元，期限四周，回退需两天。方案乙成本六万元，"
        "期限六周，回退需一周。硬约束为预算九万元、期限五周，方案乙未满足期限。"
    ),
}

PAPER_SNIPPETS = (
    ("paper-a.txt", "论文甲测试片段：研究比较检索系统，在数据集甲用准确率指标得到0.82。"
     "实验条件为单次检索、固定查询长度，未测试长文档，也未报告重复实验。"),
    ("paper-b.txt", "论文乙测试片段：研究比较检索系统，在数据集乙用召回率指标得到0.88。"
     "实验条件为两次检索，查询长度可变；数据集与指标不同，不能直接判定优于论文甲。"),
)


async def upload_material(project, skill_id):
    service, _, sid = project
    materials = PAPER_SNIPPETS if skill_id == "literature-synthesis" else (
        (f"{skill_id}-notes.txt", MATERIALS[skill_id]),)
    documents = []
    for filename, text in materials:
        document = service.store.accept_document(sid, filename, text.encode())
        assert await service.process(sid, document["id"])
        documents.append(document)
    return documents


def knowledge_payload(prompt):
    assert prompt.count("KNOWLEDGE_SKILL_DATA_JSON:\n") == 1
    text = prompt.split("KNOWLEDGE_SKILL_DATA_JSON:\n", 1)[1]
    payload, _ = json.JSONDecoder().raw_decode(text)
    return payload


@pytest.mark.asyncio
@pytest.mark.parametrize("skill_id", TASKS)
async def test_new_original_pack_auto_routes_with_one_writer_and_separate_s_evidence(project, skill_id):
    documents = await upload_material(project, skill_id)
    service, _, sid = project
    result, calls = await ask(project, TASKS[skill_id], answer="测试：依据原文提出有条件的建议。[S1] [K1]")
    assert len(calls) == 1
    receipt = result["knowledge_skill"]
    assert receipt["id"] == skill_id and receipt["version"] == PACK_VERSIONS[skill_id]
    assert receipt["status"] == "ready"
    assert result["sources"] and result["sources"][0]["label"] == "S1"
    assert receipt["sources"] and receipt["sources"][0]["label"] == "K1"
    assert all(source["reference_kind"] == "project_original" for source in receipt["sources"])
    assert all(source["source_url"] == source["excerpt"] == "" for source in receipt["sources"])
    assert all(source["source_title"].startswith("DocThinker · ") for source in receipt["sources"])
    assert {row["id"] for row in receipt["sources"]}.isdisjoint(row["id"] for row in result["sources"])
    assert result["citation_validation"]["valid"]
    assert result["citation_validation"]["entailment_verified"] is False
    assert receipt["authorship"] == "project_original_method_design"
    assert receipt["validation_status"] == "design_unvalidated"
    payload = knowledge_payload(calls[0][0])
    assert payload["skill"]["id"] == skill_id
    assert payload["sources"] == receipt["sources"]
    trace = result["reasoning"]["trace"]
    assert trace["knowledge_skill_router"]["model_calls"] == 0
    assert trace["knowledge_skill_retrieval"]["model_calls"] == 0
    assert trace["knowledge_skill_retrieval"]["network_calls"] == 0
    assert trace["context_budget"]["knowledge_skill_bytes"] <= 3500
    assert trace["context_budget"]["input_bytes"] <= 24576
    assert len(service.store.list_documents(sid)) == len(documents)
    assert service.store.list_memories(sid) == []
    metadata = service.store.answer_metadata(sid, [result["answer_id"]])[result["answer_id"]]
    assert all(source["label"].startswith("S") for source in metadata["source_refs"])


@pytest.mark.asyncio
@pytest.mark.parametrize("skill_id", ["research-ideation", "evidence-writing", "decision-review"])
async def test_general_task_can_use_original_methods_without_fabricated_user_documents(project, skill_id):
    result, calls = await ask(project, TASKS[skill_id], answer="以下是方法建议，不是已发生事实。[K1]")
    assert len(calls) == 1
    assert result["knowledge_skill"]["id"] == skill_id
    assert result["knowledge_skill"]["status"] == "ready"
    assert result["sources"] == []
    assert result["answer_mode"] == "general"
    assert result["citation_validation"]["valid"]


@pytest.mark.asyncio
@pytest.mark.parametrize("skill_id", TASKS)
async def test_disabled_multiple_pack_tasks_read_no_knowledge_resource(project, monkeypatch, skill_id):
    await upload_material(project, skill_id)
    monkeypatch.setattr(
        "docthinker.server.knowledge_skills._read_pack",
        lambda *args: pytest.fail("Disabling knowledge assistance must prevent all pack reads"),
    )
    result, calls = await ask(project, TASKS[skill_id], knowledge_skill_mode="off", answer="根据材料回答。[S1]")
    assert len(calls) == 1
    assert "knowledge_skill" not in result
    assert "KNOWLEDGE_SKILL_DATA_JSON" not in calls[0][0]
    assert result["sources"]


@pytest.mark.asyncio
@pytest.mark.parametrize("skill_id", TASKS)
async def test_each_pack_snapshot_survives_restart_without_retrieval(project, monkeypatch, skill_id):
    await upload_material(project, skill_id)
    original, _ = await ask(project, TASKS[skill_id], answer="测试原文与方法的分离。[S1] [K1]")
    monkeypatch.setattr(
        "docthinker.server.knowledge_skills._read_pack",
        lambda *args: pytest.fail("History must replay its snapshot rather than load today's pack"),
    )
    service, manager, sid = restored(project)
    replay = service.hydrate_history(sid, manager.get_history(sid))[-1]
    assert replay["content"] == original["answer"]
    assert replay["knowledge_skill"] == original["knowledge_skill"]
    assert replay["knowledge_skill"]["version"] == PACK_VERSIONS[skill_id]
    assert replay["citation_validation"]["valid"]
    assert [row["id"] for row in replay["sources"]] == [row["id"] for row in original["sources"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("question", [
    TASKS["literature-synthesis"],
    "根据上传的项目资料写报告，组织论点与证据。",
    "根据上传的项目资料，比较两个方案的预算与期限。",
    "根据这些论文提出可检验的科研假设。",
])
async def test_method_pack_cannot_substitute_for_missing_requested_source_facts(project, question):
    result, calls = await ask(project, question, answer="不应调用测试模型来虚构用户事实。[K1]")
    assert calls == []
    assert result["answer_mode"] == "insufficient_evidence"
    assert result["sources"] == []
    assert result.get("knowledge_skill", {}).get("status") != "ready"
    assert not result.get("knowledge_skill", {}).get("sources")


@pytest.mark.asyncio
async def test_long_input_drops_whole_optional_pack_before_original_evidence(project, monkeypatch):
    await upload_material(project, "evidence-writing")
    question = TASKS["evidence-writing"] + "\n附加交付背景：" + "先保留原文。" * 350
    assert 6000 < len(question.encode()) <= 8192
    baseline, _ = await ask(project, question, knowledge_skill_mode="off", answer="报告依据原材料。[S1]")
    from docthinker.server import research_prompt_budget as budgeting
    validate = budgeting.validate_invite_messages
    attempts = []

    def smaller_provider_window(messages):
        attempts.append(messages)
        # Exercise the whole-pack fit fallback at a narrower provider boundary,
        # then run the real 24 KiB validator on the accepted long input.
        if "KNOWLEDGE_SKILL_DATA_JSON" in messages[1]["content"]:
            raise ValueError("Test provider window omits the optional method block")
        return validate(messages)

    monkeypatch.setattr(budgeting, "validate_invite_messages", smaller_provider_window)
    result, calls = await ask(project, question, answer="报告依据原材料。[S1]")
    assert len(calls) == 1 and len(attempts) >= 2
    assert result["knowledge_skill"]["id"] == "evidence-writing"
    assert result["knowledge_skill"]["status"] == "budget_omitted"
    assert result["knowledge_skill"]["sources"] == []
    assert [row["id"] for row in result["sources"]] == [row["id"] for row in baseline["sources"]]
    assert "KNOWLEDGE_SKILL_DATA_JSON" not in calls[0][0]
    assert result["reasoning"]["trace"]["context_budget"]["knowledge_skill_bytes"] == 0
    assert result["citation_validation"]["valid"]


@pytest.mark.asyncio
async def test_switching_tasks_replaces_current_k_scope_but_keeps_historical_snapshot(project):
    service, manager, sid = project
    first, _ = await ask(project, TASKS["evidence-writing"], answer="先完成论点与证据映射。[K1]")
    calls = []

    async def writer(prompt, **kwargs):
        calls.append(prompt)
        return "当前先检查方案硬约束，而不是沿用上次写作方法。[K1]"

    req = request(sid, TASKS["decision-review"], knowledge_skill_mode="auto")
    req.use_conversation_context = True
    second = await service.answer(req, writer, manager)
    assert len(calls) == 1
    assert first["knowledge_skill"]["id"] == "evidence-writing"
    assert second["knowledge_skill"]["id"] == "decision-review"
    payload = knowledge_payload(calls[0])
    assert payload["skill"]["id"] == "decision-review"
    assert payload["sources"] == second["knowledge_skill"]["sources"]
    previous_ids = {row["id"] for row in first["knowledge_skill"]["sources"]}
    current_ids = {row["id"] for row in second["knowledge_skill"]["sources"]}
    assert previous_ids.isdisjoint(current_ids)
    assert second["citation_validation"]["valid"]
    history = [row for row in service.hydrate_history(sid, manager.get_history(sid)) if row["role"] == "assistant"]
    assert [row["knowledge_skill"]["id"] for row in history] == ["evidence-writing", "decision-review"]
    assert history[0]["knowledge_skill"] == first["knowledge_skill"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("question", "skill_id"), [
    ("Draft a report with claims and supporting evidence.", "evidence-writing"),
    ("Compare two product plans and their tradeoffs.", "decision-review"),
])
async def test_english_tasks_retrieve_original_methods_through_normal_service(project, question, skill_id):
    result, calls = await ask(project, question, answer="A bounded method, not a fact about the user's materials. [K1]")
    assert len(calls) == 1
    assert result["knowledge_skill"]["id"] == skill_id
    assert result["knowledge_skill"]["status"] == "ready"
    assert result["knowledge_skill"]["sources"]
    assert knowledge_payload(calls[0][0])["skill"]["id"] == skill_id
    assert result["citation_validation"]["valid"]
