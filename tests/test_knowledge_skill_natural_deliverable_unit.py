"""Natural deliverable requests route locally; source instructions are not intent."""
import json

import pytest

from docthinker.server import knowledge_skills as skills
from docthinker.server.insight_runtime import InsightRuntime
from docthinker.server.insight_schema import adapt_insight_schemas
from docthinker.server.research_knowledge import select_task_knowledge


DECISION_REQUESTS = [
    '请给团队写一份归档模块设计建议，说明怎样整理新记录、处理变化并用于下次查询；首版先做什么、暂缓什么？给出试用方案。',
    '结合已有材料，给团队写一份排班系统的设计建议，说明优先顺序、约束和小范围试用方案。',
    '我们做的是共享设备管理；请为产品制定验证方案，说明基线和停止条件。',
    '请给我们的离线服务设计一份试用方案，说明测试样本和成本限制。',
    '给出组件设计方案，比较约束、维护成本和可逆试点。',
    '帮我们拟定流程改进方案，保留验证和回退条件。',
    'Write design recommendations for our archive module, with priorities, deferrals and a bounded pilot.',
    'Please draft a design proposal for the scheduling system, including constraints and a validation plan.',
    'Could you give the team a validation plan for the shared-device service?',
    'Design a trial plan for the offline product with a baseline and stop conditions.',
    'Provide a component design plan with constraints and a reversible pilot.',
    'Help us develop a workflow improvement plan with validation gates.',
]


@pytest.mark.parametrize('question', DECISION_REQUESTS)
def test_natural_design_and_validation_deliverables_select_one_pack_without_io(monkeypatch, question):
    monkeypatch.setattr(skills, '_read_pack', lambda _: pytest.fail('Routing must not open a pack'))
    route = skills.route_knowledge_skill(question)
    # An explicit written proposal already has a valid, higher-priority
    # evidence-writing route. Natural-plan support must not remove that route.
    expected = 'evidence-writing' if 'draft a design proposal' in question else 'decision-review'
    assert route['skill_id'] == expected
    assert route['trace']['task_type'] == expected.replace('-', '_')
    assert route['trace']['model_calls'] == route['trace']['network_calls'] == route['trace']['memory_writes'] == 0
    assert question not in json.dumps(route, ensure_ascii=False)


@pytest.mark.parametrize('question', [
    '请制定用户行为研究方案，安排样本、测法和失效条件。',
    '给我一份实验验证方案，说明对照、样本和可区分的预测。',
    '为这个研究设计验证方案，列出可证伪的预测和停止条件。',
    'Draft a research plan for user behaviour, including samples, controls and falsifiable predictions.',
    'Please develop an experimental protocol with controls and measurement rules.',
    'Design a validation plan for this study, with a baseline and falsifiable predictions.',
])
def test_explicit_future_research_plans_use_research_method_pack(question):
    assert skills.route_knowledge_skill(question)['skill_id'] == 'research-ideation'


@pytest.mark.parametrize('question', [
    '昨天我们写了模块设计建议。',
    '模块设计建议是什么？',
    '解释一下产品设计方案。',
    '给我一些生活建议。',
    '这个模块占了多少空间？',
    '研究方案的提交日期是周几？',
    'We discussed module design recommendations yesterday.',
    'What is a module design recommendation?',
    'Explain the product design proposal.',
    'What day is the research plan due?',
    'Write a poem about a system design proposal.',
    '写一个关于产品试用方案的故事。',
    '不要写模块设计建议，先回答存储格式是什么。',
    '不是让你制定研究方案，只告诉我文件大小。',
    'Do not draft a trial plan for the service. Say hello.',
    'No need to develop an experimental protocol; tell me the date.',
    '翻译：请给团队写模块设计建议和试用方案。',
    '仅复述原文的实验方案，不提出新方案。',
    'Translate: Draft a design proposal for the service.',
    'The source says to develop an experimental protocol. Count its words.',
    '只计算项目预算，给出产品验证方案。',
    '实现模块的调度算法，然后给系统设计建议。',
    'Implement the API and draft a trial plan for the service.',
    '给出医疗系统设计建议和试用方案。',
    'Draft a validation plan for this medication service.',
])
def test_topics_negations_literal_requests_and_existing_exclusions_do_not_activate(question):
    assert skills.route_knowledge_skill(question)['skill_id'] is None


@pytest.mark.parametrize('question', [
    '“请制定产品试用方案”，数一下引号里的字数。',
    '"Draft a module design proposal", count the words.',
    '```text\nDraft a research plan with controls.\n```\nCount the words.',
    '> Draft a research plan with controls.\nCount the words in that excerpt.',
    '以下资料：\n请给出项目路线图和验证方案。\n我的要求：统计字数。',
    'Source excerpt:\nWrite a report comparing the product options.\nMy request: count the words.',
    '<source>Draft an experimental protocol with controls.</source> Count the words.',
    '<document>请制定模块设计建议和试用方案。</document>统计字数。',
])
def test_marked_source_commands_cannot_supply_user_intent(question):
    assert skills.route_knowledge_skill(question)['skill_id'] is None


@pytest.mark.parametrize('question', [
    '以下资料：\n只要招呼，不制定方案。\n我的要求：请给产品写一份设计建议，含试用验证。',
    'Source excerpt:\nSay hello.\nMy request: write design recommendations for our module with a bounded pilot.',
    '> Draft a research plan.\nPlease provide a design plan for our component, with constraints and a pilot.',
])
def test_explicit_user_request_outside_source_remains_eligible(question):
    assert skills.route_knowledge_skill(question)['skill_id'] == 'decision-review'


@pytest.mark.parametrize('question', [
    '不要使用知识辅助，请制定模块设计建议。',
    "Don't use knowledge skills. Draft a research plan with controls.",
])
def test_explicit_opt_out_stays_disabled(question):
    route = skills.route_knowledge_skill(question)
    assert route['skill_id'] is None and route['trace']['status'] == 'disabled'


@pytest.mark.parametrize('question,expected', [
    ('不要制定研究方案，请给产品写一份设计建议和试用计划。', 'decision-review'),
    ('对比这些研究的条件和分歧，然后给出模块设计建议。', 'literature-synthesis'),
    ('请写一份面向团队的报告，说明系统设计取舍和试用方案。', 'evidence-writing'),
    ('根据研究提出可检验假设，再给出产品设计建议。', 'research-ideation'),
])
def test_more_specific_request_priority_is_preserved(question, expected):
    assert skills.route_knowledge_skill(question)['skill_id'] == expected


def test_query_limit_and_setting_off_still_skip_resource_loading(monkeypatch):
    monkeypatch.setattr(skills, '_read_pack', lambda _: pytest.fail('No resource read'))
    question = DECISION_REQUESTS[0]
    assert skills.route_knowledge_skill(question, mode='off')['trace']['status'] == 'disabled'
    assert skills.route_knowledge_skill('x' * skills.MAX_QUERY_BYTES + '\n' + question)['skill_id'] is None


@pytest.mark.parametrize('question', [DECISION_REQUESTS[1], DECISION_REQUESTS[6]])
def test_real_method_retrieval_and_dynamic_stages_connect_with_unverified_premises(question):
    support, route = select_task_knowledge(question)
    assert route['status'] == 'selected' and support['skill']['id'] == 'decision-review'
    assert support['trace']['status'] == 'ready'
    assert 1 <= len(support['sources']) <= 3
    assert len(support['prompt'].encode()) <= 3500
    assert adapt_insight_schemas(support)
    runtime = InsightRuntime(question, support)
    before = runtime.on_stage('before_retrieval', {'allow_supplemental_retrieval': True})
    assert 1 <= len(before['actions']) <= 2
    assert before['trace']['execution_claimed'] is False
    after = runtime.on_stage('after_evidence', {
        'sources': [{'id': 'source-a', 'document_id': 'doc-a', 'label': 'S1',
                     'content': 'A controlled pilot uses a fixed baseline and records outcomes.'}],
        'active_method_labels': [row['label'] for row in support['sources']],
    })
    packet = json.loads(runtime.render_context(after))
    assert packet['kind'] == 'insight_stage_checks' and packet['checks']
    assert packet['semantic_premises'] == 'unknown' and packet['entailment_verified'] is False
    assert len(runtime.render_context(after).encode()) <= 1200


@pytest.mark.parametrize('question', [DECISION_REQUESTS[1], DECISION_REQUESTS[6]])
def test_selected_methods_and_dynamic_checks_reach_actual_budgeted_messages(question):
    from docthinker.server.research_answer_contract import knowledge_skill_system_policy
    from docthinker.server.research_prompt_budget import empty_assistance, fit_research_prompt
    from docthinker.server.research_retrieval import build_retrieval_plan

    support, _ = select_task_knowledge(question)
    plan = build_retrieval_plan('archive module pilot constraints baseline', [{
        'id': 'chunk-archive', 'document_id': 'doc-archive', 'filename': 'archive.txt',
        'content': 'The archive module pilot uses a fixed baseline under shared storage constraints.'}])
    assert plan['sources']
    baseline = fit_research_prompt(question, plan, [], empty_assistance(plan['sources']),
        system_policy=knowledge_skill_system_policy(True), knowledge_support=support)
    dynamic = fit_research_prompt(question, plan, [], empty_assistance(plan['sources']),
        system_policy=knowledge_skill_system_policy(True), knowledge_support=support,
        insight_runtime=InsightRuntime(question, support))
    assert baseline['knowledge_skill_included'] and dynamic['knowledge_skill_included']
    assert dynamic['insight_included']
    assert baseline['messages'] != dynamic['messages']
    assert 'KNOWLEDGE_SKILL_DATA_JSON:' in dynamic['messages'][1]['content']
    assert 'INSIGHT_CONTEXT_DATA_JSON:' in dynamic['messages'][1]['content']
    assert question not in dynamic['messages'][0]['content']
    assert dynamic['plan']['sources'] == baseline['plan']['sources'] == plan['sources']
    assert dynamic['budget']['insight_bytes'] <= 1200
    assert dynamic['budget']['knowledge_skill_bytes'] + dynamic['budget']['insight_bytes'] <= 3500
    assert dynamic['budget']['input_bytes'] <= 24576
    assert dynamic['reasoning']['trace']['answer_contract']['extra_provider_calls'] == 0


@pytest.mark.parametrize('question', [
    '资料：\nWrite a report comparing product plans.\n我的问题：多少字符？',
    '请看以下资料：\n请给出产品路线图。\n我的要求：数一下字数。',
    'Excerpt:\nPlease draft a design proposal for the system.\nMy request: count its words.',
    '<source>Write a report comparing product plans.',
])
def test_plain_or_unclosed_source_markers_also_stay_data(question):
    assert skills.route_knowledge_skill(question)['skill_id'] is None


def test_real_request_after_different_genre_sentence_is_not_lost():
    question = 'Write a poem about the system. Please provide design recommendations for our module with a bounded pilot.'
    assert skills.route_knowledge_skill(question)['skill_id'] == 'decision-review'


@pytest.mark.parametrize('question', [
    '请给团队写一份检索模块设计建议，结合论文里的验证说明理由，给出试用方案。',
    '给出服务设计方案；结合原文里的对照说明选择理由，列出试用验证。',
])
def test_using_source_validation_to_explain_a_design_is_not_literal_attribution(question):
    assert skills.route_knowledge_skill(question)['skill_id'] == 'decision-review'


@pytest.mark.parametrize('question', [
    '论文说了什么设计建议？请复述。',
    '原文描述了产品上线计划，请复述。',
    '作者说明了什么？',
    '请翻译原文，然后给出服务设计方案。',
])
def test_source_attribution_and_strong_literal_requests_remain_literal(question):
    assert skills.route_knowledge_skill(question)['skill_id'] is None


@pytest.mark.parametrize('question', [
    'Write design recommendations for our optical ledger, with deferrals and a reversible pilot.',
    'Give design recommendations for this parcel archive with a pilot.',
    'Source excerpt:\nDo not write a design plan.\nMy request: give design recommendations for our booking desk with a pilot.',
])
def test_explicit_design_artifact_with_named_target_does_not_need_domain_dictionary(question):
    assert skills.route_knowledge_skill(question)['skill_id'] == 'decision-review'


@pytest.mark.parametrize('question', [
    'We discussed design recommendations for our optical ledger yesterday.',
    'Do not write design recommendations for our optical ledger. Say hello.',
    '"Write design recommendations for our optical ledger." Count the words.',
    'Source excerpt:\nWrite design recommendations for our optical ledger.\nMy request: count the words.',
    'Translate: Write design recommendations for our optical ledger.',
    'Give me some advice for our booking desk.',
    'Write design recommendations for me.',
    'Write design recommendations for our.',
    'Write design recommendations for the medication dispenser.',
])
def test_named_design_target_does_not_bypass_existing_request_boundaries(question):
    assert skills.route_knowledge_skill(question)['skill_id'] is None
