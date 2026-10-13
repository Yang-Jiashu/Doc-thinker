"""Packaged methods are optional references, not user evidence or live facts."""

from __future__ import annotations

import copy
import json

import pytest

from docthinker.server import knowledge_skills as skills


@pytest.fixture
def pack():
    return json.loads(skills._read_pack("mao-workplace"))


def install_pack(monkeypatch, pack):
    encoded = json.dumps(pack, ensure_ascii=False).encode("utf-8")
    monkeypatch.setattr(skills, "_read_pack", lambda _: encoded)


def test_catalog_is_cheap_safe_and_fresh(monkeypatch):
    monkeypatch.setattr(skills, "_read_pack", lambda _: pytest.fail("Catalog must not load resources"))
    catalog = skills.knowledge_skill_catalog()
    assert catalog[0] == {
        "id": "mao-workplace", "name": "职场方法 · 毛选启发", "version": "2",
        "description": "按任务查询原创的事实核实、优先级、试点与协作方法；历史原文只作启发出处。",
        "kind": "knowledge_skill", "knowledge_scope": skills.KNOWLEDGE_SCOPE,
        "authorship": "project_original_method_design", "validation_status": "design_unvalidated",
    }
    assert {item['id'] for item in catalog} == {
        'mao-workplace', 'research-ideation', 'literature-synthesis', 'evidence-writing', 'decision-review'}
    assert len(catalog) == 5
    catalog[0]["name"] = "mutated"
    assert skills.knowledge_skill_catalog()[0]["name"] == "职场方法 · 毛选启发"
    assert "excerpt" not in catalog[0] and "path" not in catalog[0]


@pytest.mark.parametrize("question", ["调查事实", "", None, "\ud800"])
def test_no_selection_never_reads_even_matching_question(monkeypatch, question):
    monkeypatch.setattr(skills, "_read_pack", lambda _: pytest.fail("Disabled means no I/O"))
    result = skills.retrieve_knowledge_skill(None, question)
    assert result["trace"]["status"] == "disabled"
    assert result["skill"] is None and not result["sources"] and not result["prompt"]


@pytest.mark.parametrize("identifier", ["", "MAO-WORKPLACE", "../../.env", "/etc/passwd",
                                       "mao-workplace/knowledge.json", "http://example.org", [], True, 1])
def test_unknown_id_never_becomes_a_path(monkeypatch, identifier):
    monkeypatch.setattr(skills, "_read_pack", lambda _: pytest.fail("Invalid IDs must not load"))
    with pytest.raises(ValueError, match="Unknown knowledge skill ID"):
        skills.retrieve_knowledge_skill(identifier, "调查")


@pytest.mark.parametrize("question", [None, 3, [], "", "\ud800", "毛选", "工作", "你好",
                                     "不要用毛选，分析项目延期原因", "毛选先别用，讨论试点",
                                     "Don't use Mao, analyze the project"])
def test_empty_generic_invalid_and_opted_out_queries_skip_loading(monkeypatch, question):
    # '你好' has tokens and may load; it remains unrelated rather than selecting
    # the skill because it was once chosen in a settings panel.
    if question != "你好":
        monkeypatch.setattr(skills, "_read_pack", lambda _: pytest.fail("No topical query"))
    result = skills.retrieve_knowledge_skill("mao-workplace", question)
    assert result["trace"]["status"] == "no_match"
    assert result["prompt"] == "" and result["sources"] == []


@pytest.mark.parametrize(("question", "first"), [
    ("同事说项目延期是我的问题，但信息不完整，先怎么调查？", "investigate-before-judgment"),
    ("资源不足，多个目标冲突，优先级如何取舍？", "identify-current-bottleneck"),
    ("怎么设计小范围试点，验证方案，再用反馈改进？", "test-and-revise"),
    ("跨部门沟通，协作分工怎么协调？", "coordinate-with-shared-context"),
])
def test_real_pack_retrieves_relevant_reference_without_full_dump(question, first):
    result = skills.retrieve_knowledge_skill("mao-workplace", question)
    assert result["trace"]["status"] == "ready"
    assert result["sources"][0]["id"] == first
    assert 1 <= len(result["sources"]) <= 3
    assert len(result["prompt"].encode("utf-8")) <= skills.MAX_PROMPT_BYTES
    assert result["trace"]["prompt_bytes"] == len(result["prompt"].encode("utf-8"))
    payload = json.loads(result["prompt"])
    assert payload["sources"] == result["sources"]
    assert payload["kind"] == "knowledge_skill_reference"
    assert payload["knowledge_scope"] == skills.KNOWLEDGE_SCOPE
    assert all(row["knowledge_scope"] == skills.KNOWLEDGE_SCOPE for row in result["sources"])
    assert [row["label"] for row in result["sources"]] == [f"K{i+1}" for i in range(len(result["sources"]))]
    assert "S1" not in result["prompt"]
    assert "retrieval_keywords" not in result["prompt"]
    assert "license_note" not in result["prompt"]
    assert result["trace"]["model_calls"] == result["trace"]["network_calls"] == 0
    assert result["trace"]["memory_writes"] == result["trace"]["graph_writes"] == 0
    assert result["trace"]["semantic_relevance_verified"] is False
    assert question not in result["prompt"]


def test_unrelated_questions_are_not_filled_with_three_methods():
    result = skills.retrieve_knowledge_skill("mao-workplace", "今晚吃饺子还是牛肉面？")
    assert result["trace"]["status"] == "no_match"
    assert result["sources"] == [] and result["prompt"] == ""


def test_whole_quote_commentary_and_boundaries_are_preserved(pack):
    result = skills.retrieve_knowledge_skill("mao-workplace", "调查原因，安排优先级，再用试点验证方案")
    originals = {entry["id"]: entry for entry in pack["entries"]}
    for source in result["sources"]:
        original = originals[source["id"]]
        for key in ("excerpt", "commentary", "boundaries", "source_title", "source_url", "published"):
            assert source[key] == original[key]
    # All four could overlap, but only three complete entries may be exposed.
    assert len(result["sources"]) <= 3


def test_oversized_reference_is_omitted_not_sliced(monkeypatch, pack):
    entry = pack["entries"][0]
    entry["commentary"] = "调查" * 300
    entry["boundaries"] = ["不能" * 150] * 4
    install_pack(monkeypatch, pack)
    result = skills.retrieve_knowledge_skill("mao-workplace", "调查")
    assert result["trace"]["status"] == "no_match"
    assert result["trace"]["budget_omitted_count"] == 1
    assert result["sources"] == [] and result["prompt"] == ""


def test_all_serialized_fields_count_toward_budget(monkeypatch, pack):
    # Two individually usable entries need too much space together once titles,
    # URLs and limits are counted. Their quotes cannot be silently abbreviated.
    for entry in pack["entries"]:
        entry["retrieval_keywords"] = ["查询测试"]
        entry["commentary"] = "查询测试" * 115
    install_pack(monkeypatch, pack)
    result = skills.retrieve_knowledge_skill("mao-workplace", "查询测试")
    assert result["trace"]["status"] == "ready"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["commentary"] == "查询测试" * 115
    assert result["trace"]["budget_omitted_count"] == 3
    assert len(result["prompt"].encode("utf-8")) <= skills.MAX_PROMPT_BYTES


def test_budget_skip_leaves_contiguous_labels(monkeypatch, pack):
    pack["entries"][0]["commentary"] = "调查" * 300
    pack["entries"][0]["boundaries"] = ["不能" * 150] * 4
    pack["entries"][1]["retrieval_keywords"].append("调查")
    install_pack(monkeypatch, pack)
    result = skills.retrieve_knowledge_skill("mao-workplace", "调查")
    assert result["trace"]["status"] == "ready"
    assert result["sources"][0]["id"] == "identify-current-bottleneck"
    assert result["sources"][0]["label"] == "K1"


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(schema_version=2),
    lambda p: p.update(schema_version=True),
    lambda p: p.update(version="1"),
    lambda p: p.update(id="foreign"),
    lambda p: p.update(kind="executable_skill"),
    lambda p: p.update(knowledge_scope="user_evidence"),
    lambda p: p.update(execute="rm -rf /"),
    lambda p: p.update(entries=[]),
    lambda p: p.update(entries=p["entries"] * 9),
    lambda p: p["entries"].append(copy.deepcopy(p["entries"][0])),
    lambda p: p["entries"][0].update(id="../../evil"),
    lambda p: p["entries"][0].update(script="print(42)"),
    lambda p: p["entries"][0].update(excerpt=""),
    lambda p: p["entries"][0].update(excerpt="x" * 121),
    lambda p: p["entries"][0].update(commentary="x" * 601),
    lambda p: p["entries"][0].update(title="x\x00"),
    lambda p: p["entries"][0].update(boundaries=[]),
    lambda p: p["entries"][0].update(retrieval_keywords="调查"),
    lambda p: p["entries"][0].update(source_url="http://www.12371.cn/a.shtml"),
    lambda p: p["entries"][0].update(source_url="file:///etc/passwd"),
    lambda p: p["entries"][0].update(source_url="https://evil.example/a"),
    lambda p: p["entries"][0].update(source_url="https://www.12371.cn.evil.example/a"),
    lambda p: p["entries"][0].update(source_url="https://user:secret@www.12371.cn/a"),
    lambda p: p["entries"][0].update(source_url="https://www.12371.cn:9000/a"),
    lambda p: p["entries"][0].update(source_url="https://www.12371.cn/a#execute"),
])
def test_malformed_untrusted_resources_fail_closed(monkeypatch, pack, mutation):
    mutation(pack)
    install_pack(monkeypatch, pack)
    result = skills.retrieve_knowledge_skill("mao-workplace", "调查")
    assert result["trace"]["status"] == "unavailable"
    assert result["sources"] == [] and result["prompt"] == ""
    assert "evil" not in json.dumps(result)


@pytest.mark.parametrize("raw", [b"not JSON", b"\xff", b"{}", b"[]",
                               b'{"schema_version":1,"schema_version":1}',
                               b'{"value":NaN}', b"x" * (skills.MAX_PACK_BYTES + 1)])
def test_bad_serialization_is_not_an_exception_or_a_prompt(monkeypatch, raw):
    monkeypatch.setattr(skills, "_read_pack", lambda _: raw)
    result = skills.retrieve_knowledge_skill("mao-workplace", "调查")
    assert result["trace"]["status"] == "unavailable"
    assert result["prompt"] == ""


def test_missing_resource_does_not_disclose_paths(monkeypatch):
    def missing(_):
        raise FileNotFoundError("/private/secret/location")
    monkeypatch.setattr(skills, "_read_pack", missing)
    result = skills.retrieve_knowledge_skill("mao-workplace", "调查")
    assert result["trace"]["status"] == "unavailable"
    assert "/private" not in json.dumps(result)


def test_symlink_pack_is_not_followed(monkeypatch, tmp_path, pack):
    root = tmp_path.resolve()
    outside = root / "outside.json"
    outside.write_text(json.dumps(pack), encoding="utf-8")
    resource = root / "packs" / "mao-workplace"
    resource.mkdir(parents=True)
    (resource / "knowledge.json").symlink_to(outside)
    monkeypatch.setattr(skills, "_PACK_ROOT", root / "packs")
    assert skills.retrieve_knowledge_skill("mao-workplace", "调查")["trace"]["status"] == "unavailable"


def test_resource_is_not_cached_across_replacement(monkeypatch, pack):
    original = skills.retrieve_knowledge_skill("mao-workplace", "调查")
    pack["entries"][0]["excerpt"] = "经核验替换后的短引文"
    install_pack(monkeypatch, pack)
    updated = skills.retrieve_knowledge_skill("mao-workplace", "调查")
    assert original["sources"][0]["excerpt"] != updated["sources"][0]["excerpt"]
    updated["sources"][0]["boundaries"].append("caller mutation")
    assert "caller mutation" not in skills.retrieve_knowledge_skill("mao-workplace", "调查")["prompt"]


def test_query_size_is_bounded_and_not_retained():
    result = skills.retrieve_knowledge_skill("mao-workplace", "调查 " + "机密项目" * 10_000)
    assert result["trace"]["status"] == "ready"
    assert "机密项目" not in json.dumps(result, ensure_ascii=False)


def test_ranking_deterministic_when_pack_order_changes(monkeypatch, pack):
    original = skills.retrieve_knowledge_skill("mao-workplace", "调查，优先级，试点，跨部门沟通")
    pack["entries"].reverse()
    install_pack(monkeypatch, pack)
    assert skills.retrieve_knowledge_skill("mao-workplace", "调查，优先级，试点，跨部门沟通") == original


@pytest.mark.parametrize(("question", "task_type"), [
    ("我们项目延期了，但信息不完整，我该怎么处理？", "fact_finding"),
    ("团队资源不足，多个任务优先级怎么取舍？", "prioritization"),
    ("跨部门沟通不畅，怎么安排分工？", "coordination"),
    ("团队要上线新的工作流程，怎么做小范围试点？", "bounded_trial"),
    ("算法团队跨部门沟通总是卡住，我该怎么办？", "coordination"),
    ("算法团队跨部门沟通总是卡住，怎么优化分工？", "coordination"),
    ("模型团队和测试团队交付代码前，如何协调负责人和沟通安排？", "coordination"),
    ("我们团队没有沟通就改了SQL上线排期，现在如何跟同事协调？", "coordination"),
    ("How should our team coordinate handoffs with other departments?", "coordination"),
    ("How to prioritize tasks given a project bottleneck?", "prioritization"),
])
def test_router_selects_concrete_workplace_tasks_without_model(monkeypatch, question, task_type):
    monkeypatch.setattr(skills, "_read_pack", lambda _: pytest.fail("Routing must not load content"))
    route = skills.route_knowledge_skill(question)
    assert route["skill_id"] == "mao-workplace"
    assert route["trace"]["status"] == "selected"
    assert route["trace"]["task_type"] == task_type
    assert route["trace"]["model_calls"] == route["trace"]["network_calls"] == 0
    assert question not in json.dumps(route, ensure_ascii=False)


@pytest.mark.parametrize("question", [
    "你好", "今天工作了", "这个模型如何运行", "我有个工作问题", "今天晚饭吃什么？",
    "优化这个项目的模型训练代码，调整GPU任务优先级", "总结论文里模型的主要瓶颈",
    "老板让我优化模型训练任务优先级，怎么实现调度算法？",
    "客户API请求的优先级队列怎么实现？",
    "这次不是沟通问题，我们团队要解决SQL查询性能瓶颈。",
    "我们团队不是沟通的问题，要解决SQL性能瓶颈。",
    "只帮我计算项目延期了3天还剩几天。",
    "My boss asked me to implement a priority queue for API requests.",
    "科研项目延期与算法性能有什么关系？", "实验试点反馈的统计假设怎么检验？",
    "我的程序里有个project delay变量，怎么计算？", "翻译：跨部门沟通不畅如何处理",
    "润色这句：我们项目延期需要调查", "作者对团队沟通有什么看法？",
    "原文建议的工作优先级是什么？", "总结这段关于项目延期的讨论",
    "解释这句：团队要做好跨部门协同", '请阅读“跨部门沟通不畅，项目延期”，然后计算2+2',
    '这里有代码 ```team project delay investigation```，帮我计数字符',
    '"How should a team coordinate projects?"', "Translate: How to prioritize workplace tasks?",
    "What does the author say about project delays?", None, [], 12, "\ud800",
])
def test_router_rejects_generic_technical_literal_and_quoted_requests(question):
    result = skills.route_knowledge_skill(question)
    assert result["skill_id"] is None
    assert result["trace"]["status"] == "not_applicable"
    assert result["trace"]["task_type"] is None


@pytest.mark.parametrize("question", [
    "不要用毛选，帮我处理项目延期", "不用知识辅助，跨部门怎么沟通", "关闭知识辅助，项目怎么做试点",
    "不需要skills，只说当前任务优先级", "方法辅助关闭，先讲工作交付瓶颈",
    "Don't use knowledge skills. Help our team coordinate.",
    "不要分析工作方法，只帮我计算项目延期了3天还剩几天。",
    "不用职场建议，项目怎么做试点只给统计计算。",
])
def test_router_respects_current_turn_opt_out(question):
    result = skills.route_knowledge_skill(question)
    assert result["skill_id"] is None and result["trace"]["status"] == "disabled"


def test_router_off_is_not_an_inference_and_unknown_modes_are_errors():
    assert skills.route_knowledge_skill("项目延期", mode="off")["trace"]["status"] == "disabled"
    with pytest.raises(ValueError, match="Unknown knowledge skill mode"):
        skills.route_knowledge_skill("项目延期", mode="force")


def test_original_method_fields_preserved_separately_from_historical_quote(pack):
    result = skills.retrieve_knowledge_skill("mao-workplace", "调查项目延期原因")
    assert result["trace"]["status"] == "ready"
    payload = json.loads(result["prompt"])
    assert payload["authorship"] == "project_original_method_design"
    assert payload["validation_status"] == "design_unvalidated"
    originals = {entry["id"]: entry for entry in pack["entries"]}
    for source in result["sources"]:
        for key in ("problem_pattern", "required_facts", "decision_rule", "action_template",
                    "stop_conditions", "counterexample", "reasoning_role"):
            assert source[key] == originals[source["id"]][key]
        assert source["excerpt"] != source["decision_rule"]
    assert "task_signals" not in result["prompt"]


@pytest.mark.parametrize(("key", "value"), [
    ("problem_pattern", ""), ("required_facts", []), ("decision_rule", []),
    ("action_template", "execute something"), ("stop_conditions", []),
    ("counterexample", ""), ("reasoning_role", "proven_truth"), ("task_signals", []),
])
def test_invalid_original_method_fields_fail_closed(monkeypatch, pack, key, value):
    pack["entries"][0][key] = value
    install_pack(monkeypatch, pack)
    assert skills.retrieve_knowledge_skill("mao-workplace", "调查")["trace"]["status"] == "unavailable"
