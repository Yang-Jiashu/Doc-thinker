"""Independent task-shaped checks for product knowledge skills, without a model.

These examples exercise routing and usable method retrieval, not answer quality
or a claim that a lexical router understands arbitrary research/work requests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docthinker.server import knowledge_skills as skills


_TASKS = [
    (
        "我读了几篇关于智能体记忆的论文，想提出一个能被实验推翻的机制假设，怎么设计预测？",
        "research-ideation", "observation-to-testable-hypothesis",
    ),
    (
        "想把生态学里关于资源竞争的思路迁移到科研选题中，帮我构思一个研究问题，并指出不能类比的地方。",
        "research-ideation", "transfer-structure-not-vocabulary",
    ),
    (
        "帮我设计研究假设的最小实验，我的解释和替代解释都能说明现有结果，怎样最便宜地分开它们？",
        "research-ideation", "cheapest-discriminating-check",
    ),
    (
        "这几篇文献的结果用的指标和数据集不同，帮我对比哪些结论能放一起讲。",
        "literature-synthesis", "compare-like-with-like",
    ),
    (
        "不同研究结果矛盾，有两篇似乎复用了同一批样本，如何综合这些证据？",
        "literature-synthesis", "trace-conflicts-and-shared-evidence",
    ),
    (
        "写文献综述时，我想标出当前材料没有覆盖的问题，再安排补查，能梳理研究缺口吗？",
        "literature-synthesis", "bound-the-evidence-gap",
    ),
    (
        "帮我用这些访谈材料写一份给产品负责人的分析报告，主线要围绕他需要做的判断，不要堆背景。",
        "evidence-writing", "audience-claim-evidence",
    ),
    (
        "这份报告的论点似乎比证据强，请梳理每个结论的前提和反证，告诉我需要缩小到什么范围。",
        "evidence-writing", "claim-boundary-counterevidence",
    ),
    (
        "帮我梳理报告交付前的验收要求，并据现有结构标出引用、日期和关键数据哪些还需要核对。",
        "evidence-writing", "deliverable-readiness-check",
    ),
    (
        "我们要评估两个供应商方案，预算和上线期限是硬限制，怎么避免被功能总分误导？",
        "decision-review", "constraints-before-ranking",
    ),
    (
        "两个产品方案怎么选？假设的范围一变排序就反过来，请检查敏感性和回退代价。",
        "decision-review", "sensitivity-and-reversibility",
    ),
    (
        "请比较两套项目方案，先用试点验证关键假设，不达验收标准就停止，而不是直接扩大投入。",
        "decision-review", "stage-gates-before-scale",
    ),
]


@pytest.mark.parametrize(("question", "skill_id", "relevant_card"), _TASKS)
def test_independent_tasks_include_a_relevant_whole_method(question, skill_id, relevant_card):
    route = skills.route_knowledge_skill(question)
    assert route["skill_id"] == skill_id
    result = skills.retrieve_knowledge_skill(route["skill_id"], question)
    assert result["trace"]["status"] == "ready"
    assert relevant_card in {source["id"] for source in result["sources"]}
    assert 0 < len(result["prompt"].encode("utf-8")) <= 3_500
    payload = json.loads(result["prompt"])
    assert payload["authorship"] == "project_original_method_design"
    assert payload["validation_status"] == "design_unvalidated"
    originals = {
        entry["id"]: entry
        for entry in json.loads(skills._read_pack(skill_id))["entries"]
    }
    for source in result["sources"]:
        assert source["reference_kind"] == "project_original"
        assert source["source_url"] == source["excerpt"] == ""
        assert source["kind"] == "method_reference"
        assert source["knowledge_scope"] == "method_reference_not_user_evidence"
        assert source["label"].startswith("K")
        for field in (
            "commentary", "decision_rule", "required_facts", "action_template",
            "stop_conditions", "counterexample", "boundaries",
        ):
            assert source[field] == originals[source["id"]][field]
    for field in ("model_calls", "network_calls", "memory_writes", "graph_writes"):
        assert result["trace"][field] == 0
    assert result["trace"]["semantic_relevance_verified"] is False


@pytest.mark.parametrize(("question", "skill_id", "relevant_card"), [
    (
        "Develop a research hypothesis for why memory consolidation reduces retrieval noise; specify a falsifiable prediction.",
        "research-ideation", "observation-to-testable-hypothesis",
    ),
    (
        "Compare these papers: their datasets and metrics differ; identify which results are actually comparable.",
        "literature-synthesis", "compare-like-with-like",
    ),
    (
        "Draft a proposal for the product lead using interview evidence, with an outline focused on the decisions they must make.",
        "evidence-writing", "audience-claim-evidence",
    ),
    (
        "Compare two vendors under a fixed budget and delivery deadline; separate hard constraints from negotiable preferences.",
        "decision-review", "constraints-before-ranking",
    ),
])
def test_english_tasks_need_both_routing_and_retrievable_content(question, skill_id, relevant_card):
    route = skills.route_knowledge_skill(question)
    assert route["skill_id"] == skill_id
    result = skills.retrieve_knowledge_skill(route["skill_id"], question)
    assert result["trace"]["status"] == "ready"
    assert relevant_card in {source["id"] for source in result["sources"]}


@pytest.mark.parametrize("question", [
    "最近科研压力大，陪我聊会儿吧。",
    "这句话是什么意思：“比较两套方案，先设计研究假设，再写报告”？",
    "实现一个程序，读取 papers.json 然后 compare 两个 reports 数组。",
    "解释一下论文中的实验设置和统计指标是什么。",
    "同事发来一句“帮我起草报告，比较两个方案”，帮我数一下里面有多少字。",
    "Compare two medication treatment plans for me.",
    "我不要你写报告，先告诉我资料上传失败怎么办。",
    "我这不是项目方案取舍问题，只是想知道决策矩阵这个词的意思。",
    "我不需要你比较论文，只说说上传文献为什么卡住。",
    "不要使用知识辅助，比较这些论文的结果。",
])
def test_nearby_topics_quotes_and_explicit_exclusions_do_not_load_knowledge(monkeypatch, question):
    monkeypatch.setattr(skills, "_read_pack", lambda _: pytest.fail("Unselected task must not load a pack"))
    route = skills.route_knowledge_skill(question)
    assert route["skill_id"] is None
    result = skills.retrieve_knowledge_skill(route["skill_id"], question)
    assert result["prompt"] == ""
    assert result["sources"] == []


@pytest.mark.parametrize(("question", "skill_id", "relevant_card"), [
    (
        "把这些论文综合成一份给研究主管的报告，重点解释指标差异与结论冲突。",
        "literature-synthesis", "trace-conflicts-and-shared-evidence",
    ),
    (
        "比较这些论文的证据后，帮我提出一个可检验的研究假设，优先区分替代解释。",
        "research-ideation", "cheapest-discriminating-check",
    ),
    (
        "帮我写一份报告，对比两套方案的成本与约束，给项目负责人做决定。",
        "evidence-writing", "audience-claim-evidence",
    ),
])
def test_overlapping_domains_follow_bounded_deliverable_precedence(question, skill_id, relevant_card):
    route = skills.route_knowledge_skill(question)
    assert route["skill_id"] == skill_id
    result = skills.retrieve_knowledge_skill(skill_id, question)
    assert result["trace"]["status"] == "ready"
    assert relevant_card in {row["id"] for row in result["sources"]}
    assert result["skill"]["id"] == skill_id
    assert len(result["prompt"].encode("utf-8")) <= 3_500


@pytest.mark.parametrize(("question", "skill_id", "relevant_card"), [
    (
        "报告准备明天提交，帮我检查引用和数字是否一致、验收要求是否都出现在成品里。",
        "evidence-writing", "deliverable-readiness-check",
    ),
    (
        "Can you review this report for conflicting claims and unsupported causal conclusions before I submit it?",
        "evidence-writing", "claim-boundary-counterevidence",
    ),
    (
        "帮我把这些论文的样本条件、指标差异和共同缺陷列成表。",
        "literature-synthesis", "compare-like-with-like",
    ),
])
def test_review_and_comparison_deliverables_are_not_limited_to_drafting(question, skill_id, relevant_card):
    route = skills.route_knowledge_skill(question)
    assert route["skill_id"] == skill_id
    result = skills.retrieve_knowledge_skill(skill_id, question)
    assert result["trace"]["status"] == "ready"
    assert relevant_card in {source["id"] for source in result["sources"]}


def test_shared_sample_conflict_is_ranked_ahead_of_generic_evidence_gap():
    question = "不同研究结果矛盾，有两篇似乎复用了同一批样本，如何综合这些证据？"
    route = skills.route_knowledge_skill(question)
    result = skills.retrieve_knowledge_skill(route["skill_id"], question)
    assert result["sources"][0]["id"] == "trace-conflicts-and-shared-evidence"


def test_each_selection_only_opens_its_registered_resource_read_only(monkeypatch):
    original_open = Path.open
    opened = []

    def read_only_open(path, mode="r", *args, **kwargs):
        assert mode in ("r", "rb")
        opened.append(path.resolve())
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", read_only_open)
    for question, expected_id, _ in _TASKS:
        opened.clear()
        route = skills.route_knowledge_skill(question)
        assert opened == []
        result = skills.retrieve_knowledge_skill(route["skill_id"], question)
        assert result["trace"]["status"] == "ready"
        assert opened == [(skills._PACK_ROOT / expected_id / "knowledge.json").resolve()]


@pytest.mark.parametrize(("question", "skill_id", "relevant_card"), _TASKS)
def test_user_off_switch_wins_over_every_supported_task(monkeypatch, question, skill_id, relevant_card):
    monkeypatch.setattr(skills, "_read_pack", lambda _: pytest.fail("Disabled means no resource reads"))
    route = skills.route_knowledge_skill(question, mode="off")
    assert route["skill_id"] is None
    assert route["trace"]["status"] == "disabled"
    assert skills.retrieve_knowledge_skill(route["skill_id"], question)["prompt"] == ""
