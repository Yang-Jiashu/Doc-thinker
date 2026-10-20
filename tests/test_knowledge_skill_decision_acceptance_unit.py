"""Independent routing intent boundaries; synthetic tasks, no model or network."""

import json

import pytest

from docthinker.server import knowledge_skills as skills


@pytest.fixture
def route_without_pack_reads(monkeypatch):
    monkeypatch.setattr(skills, "_read_pack", lambda *_: pytest.fail("Routing must not read knowledge packs"))

    def route(question, **kwargs):
        result = skills.route_knowledge_skill(question, **kwargs)
        assert result["trace"]["model_calls"] == 0
        assert result["trace"]["network_calls"] == 0
        assert result["trace"]["memory_writes"] == 0
        assert question not in json.dumps(result, ensure_ascii=False)
        return result

    return route


@pytest.mark.parametrize("question", [
    "请制定内部文档服务的发布路线图，列明验收和停止条件。",
    "为知识搜索产品制定一份上线计划，并说明哪些功能暂缓。",
    "请给我事件管理产品的落地建议，区分现在能做和需要验证的部分。",
    "给出这个产品的路线建议，标明限制和后续验证。",
    "请提供新产品的交付计划，并列明阶段门槛。",
    "Draft an implementation plan for the customer support product with checkpoints.",
    "Please provide a rollout roadmap for the searchable archive service.",
    "For the customer support product, recommend which features to build first and what to defer.",
    "Can you recommend a launch plan for the document service?",
    "Recommend which service features to prioritize and which to defer.",
    "Draft a product roadmap for a JavaScript API service.",
    "请给出 API 服务的产品上线计划，列明验证和回退条件。",
])
def test_explicit_planning_deliverables_route_to_decision_review(route_without_pack_reads, question):
    result = route_without_pack_reads(question)
    assert result["skill_id"] == "decision-review"
    assert result["trace"]["task_type"] == "decision_review"
    assert result["trace"]["status"] == "selected"


@pytest.mark.parametrize("question", [
    '"Draft a product roadmap." Count the characters inside the quotation.',
    "请统计‘给出产品路线图’中的字符数量。",
    "```Create an implementation plan for the product.```",
    "我们聊过产品路线图；这不是一个交付请求。",
    "不要写产品上线计划，只列出三个文件名。",
    "Do not write a product roadmap. List three filenames.",
    "不需要落地建议，告诉我现在是什么时间。",
    "翻译：Draft an implementation plan for the document service.",
    "逐字引用这句话：给出产品路线图。",
    "解释产品路线图这个词是什么意思。",
    "What does product rollout guidance mean?",
    "Define a product roadmap.",
    "Give me some advice about this product.",
    "给我一点产品建议。",
    "这个服务不错，聊聊你的想法。",
    "给出去图书馆的路线建议。",
    "Draft a medication rollout plan for the clinical service.",
    "请给出投资产品的上线建议。",
    "Provide legal implementation advice for the service.",
    "Implement a Python program that creates a product roadmap.",
    "Write JavaScript code to create a product roadmap.",
    "Create a JavaScript function that renders the product roadmap.",
    "请编写代码，生成一个产品路线图页面。",
    "Write a poem about the product roadmap.",
    "给我介绍一下产品路线图。",
    "写一个关于产品路线图的故事。",
])
def test_mention_literal_negated_sensitive_and_code_tasks_do_not_route(route_without_pack_reads, question):
    assert route_without_pack_reads(question)["skill_id"] is None


@pytest.mark.parametrize("question,expected", [
    ("Write a report evaluating the product rollout and its evidence gaps.", "evidence-writing"),
    ("Write a report about JavaScript code quality and product rollout risks.", "evidence-writing"),
    ("请写一份项目上线报告，包括路线图和取舍依据。", "evidence-writing"),
    ("Propose research hypotheses and a product roadmap for testing them.", "research-ideation"),
    ("根据研究观察提出可检验假设，再给出产品落地建议。", "research-ideation"),
    ("Compare these papers and provide a product rollout recommendation.", "literature-synthesis"),
    ("比较这些论文的测量条件和证据冲突，再给出产品路线建议。", "literature-synthesis"),
    ("不要使用毛选，给出产品上线计划和验证门槛。", "decision-review"),
    ("Don't use Mao. Draft a product implementation plan.", "decision-review"),
    ("跨部门沟通不畅，怎么明确负责人和分工？", "mao-workplace"),
    ("不要用毛选，跨部门沟通不畅，怎么明确负责人？", None),
])
def test_specific_task_priority_and_scoped_mao_optout_are_preserved(route_without_pack_reads, question, expected):
    assert route_without_pack_reads(question)["skill_id"] == expected


@pytest.mark.parametrize("question", [
    "关闭知识辅助，请制定内部文档服务的发布路线图。",
    "Don't use knowledge skills. Draft a product rollout plan.",
])
def test_global_optout_remains_disabled(route_without_pack_reads, question):
    result = route_without_pack_reads(question)
    assert result["skill_id"] is None
    assert result["trace"]["status"] == "disabled"


def test_mode_off_precedes_explicit_deliverable(route_without_pack_reads):
    result = route_without_pack_reads("Draft a product rollout plan.", mode="off")
    assert result["skill_id"] is None
    assert result["trace"]["status"] == "disabled"


def test_query_window_does_not_route_from_unprocessed_tail(route_without_pack_reads):
    question = "x" * skills.MAX_QUERY_BYTES + "。请制定产品上线计划。"
    assert route_without_pack_reads(question)["skill_id"] is None


@pytest.mark.parametrize("question", [
    "请给出产品落地建议，包括预算约束、试点验证和回退条件。",
    "Draft a product rollout plan with budget constraints and pilot validation.",
])
def test_selected_deliverable_can_retrieve_bounded_method_data_without_claiming_evidence(question):
    route = skills.route_knowledge_skill(question)
    assert route["skill_id"] == "decision-review"
    support = skills.retrieve_knowledge_skill(route["skill_id"], question)
    assert support["trace"]["status"] == "ready"
    assert support["trace"]["model_calls"] == support["trace"]["network_calls"] == 0
    assert support["trace"]["memory_writes"] == support["trace"]["graph_writes"] == 0
    assert support["trace"]["semantic_relevance_verified"] is False
    assert 0 < len(support["sources"]) <= 3
    assert len(support["prompt"].encode("utf-8")) <= 3500
    assert all(row["kind"] == "method_reference" and
               row["knowledge_scope"] == "method_reference_not_user_evidence"
               for row in support["sources"])
