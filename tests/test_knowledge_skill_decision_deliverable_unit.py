"""Bounded decision-deliverable routing, not a semantic intent classifier."""
import json

import pytest

from docthinker.server import knowledge_skills as skills


@pytest.mark.parametrize('question', [
    '请给我产品首版的路线建议，说明应该先做哪些功能、暂缓哪些功能及验证步骤。',
    '帮我制定项目路线图，保留硬约束和可逆的试点。',
    '请根据已有资料为产品写一份落地建议。',
    '给我服务上线计划，包括回退条件和成本限制。',
    '项目方向还没定，推荐先做什么、暂缓什么。',
    '产品范围有限，请给出优先做与暂缓功能的建议。',
    'Write a product roadmap with priorities, deferrals and validation gates.',
    'Could you draft a rollout plan for our service?',
    'Provide implementation recommendations for this project under a fixed budget.',
    'Plan the product rollout with a reversible pilot.',
    'Recommend which product features to build first and which to defer.',
])
def test_explicit_decision_deliverables_route_without_model_or_profile(monkeypatch, question):
    monkeypatch.setattr(skills, '_read_pack', lambda _: pytest.fail('Router must not open knowledge'))
    result = skills.route_knowledge_skill(question)
    assert result['skill_id'] == 'decision-review'
    assert result['trace']['task_type'] == 'decision_review'
    assert result['trace']['model_calls'] == result['trace']['network_calls'] == result['trace']['memory_writes'] == 0
    assert question not in json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize('question', [
    '我们最近在聊产品路线图。',
    '昨天我写了产品路线图。',
    '产品首版看起来不错，聊聊吧。',
    '给我一点生活建议。',
    '给我去公园的路线建议。',
    'We talked about a product roadmap yesterday.',
    'I wrote a product roadmap yesterday.',
    'Could you give me some advice?',
    'Write a poem about the product roadmap.',
    '给我介绍一下产品路线图。',
    '写一个关于产品路线图的故事。',
    'Give me a brief explanation of the product rollout plan.',
    'Please write a short story about the service implementation plan.',
    '这个产品路线图是什么意思？',
    '解释这个术语：产品路线图。',
    '翻译：给出产品实施建议和暂缓项。',
    'Rewrite a product roadmap with priorities and deferrals.',
    '原文建议的项目路线是什么？',
    '"Write a product roadmap with deferrals." Count the words.',
    '请阅读“给出产品路线建议”，数一下有几个字。',
    '```请给我产品路线图```',
    '不要给我产品路线建议，先聊聊。',
    'Do not write a product roadmap. Just say hello.',
    '实现项目的调度算法，给出产品路线图。',
    'Implement an API for the product; write a rollout plan.',
    '只计算产品预算还剩多少，给出产品落地建议。',
    '给我医疗产品的实施建议。',
    'Draft a rollout plan for a medication treatment service.',
    '给我投资项目的路线图。',
    'Recommend which stock investment products to buy first.',
])
def test_nearby_topics_literal_negated_technical_and_high_stakes_stay_out(monkeypatch, question):
    monkeypatch.setattr(skills, '_read_pack', lambda _: pytest.fail('Rejected task must not open a pack'))
    result = skills.route_knowledge_skill(question)
    assert result['skill_id'] is None


@pytest.mark.parametrize('question,expected', [
    ('请写一份产品路线报告，推荐先做和暂缓项。', 'evidence-writing'),
    ('Write a report with a product roadmap and deferral recommendations.', 'evidence-writing'),
    ('Write a report about JavaScript code and its deployment tradeoffs.', 'evidence-writing'),
    ('Write a product roadmap for an API and JavaScript client.', 'decision-review'),
    ('为API服务制定上线计划，说明发布门槛。', 'decision-review'),
    ('Please create a rollout plan for the JavaScript service.', 'decision-review'),
    ('先写一个关于产品路线图的故事；请给出产品上线计划。', 'decision-review'),
    ('Write a product roadmap for a story-writing service.', 'decision-review'),
    ('Write a report about the product roadmap.', 'evidence-writing'),
    ('比较这些论文的冲突，给出产品落地建议。', 'literature-synthesis'),
    ('根据论文提出研究假设，给出项目路线图。', 'research-ideation'),
    ('不要用毛选，给我产品路线建议。', 'decision-review'),
    ('不要写报告，请给出产品路线建议。', 'decision-review'),
])
def test_existing_specific_deliverable_precedence_is_unchanged(question, expected):
    assert skills.route_knowledge_skill(question)['skill_id'] == expected


@pytest.mark.parametrize('question', [
    '不要使用知识辅助，请给出产品路线建议。',
    '关闭知识包，请帮我制定产品路线图。',
    "Don't use knowledge skills. Write a product roadmap.",
])
def test_explicit_off_remains_disabled(question):
    result = skills.route_knowledge_skill(question)
    assert result['skill_id'] is None and result['trace']['status'] == 'disabled'


def test_setting_off_precedes_even_explicit_action_and_no_extra_resource_reads(monkeypatch):
    monkeypatch.setattr(skills, '_read_pack', lambda _: pytest.fail('Off must not read packs'))
    result = skills.route_knowledge_skill('请给我产品路线建议。', mode='off')
    assert result['skill_id'] is None and result['trace']['status'] == 'disabled'


def test_query_bound_does_not_match_a_deliverable_beyond_accepted_window():
    question = 'x' * skills.MAX_QUERY_BYTES + '。请给我产品路线建议。'
    assert skills.route_knowledge_skill(question)['skill_id'] is None


@pytest.mark.parametrize('question,expected', [
    ('比较这些论文的冲突，给出产品落地建议。', 'literature-synthesis'),
    ('阅读资料原文；给出产品落地建议。', 'decision-review'),
    ('Read the source, then tell me what it says.', None),
    ('What does the author suggest for the product rollout?', None),
    ('作者建议的产品落地路线是什么？', None),
    ('原文描述了产品上线计划，请复述。', None),
    ('翻译原文，然后给出产品路线图。', None),
    ('Rewrite the source. Write a product roadmap.', None),
])
def test_source_attribution_is_clause_bounded_but_strong_literal_requests_stay_literal(question, expected):
    assert skills.route_knowledge_skill(question)['skill_id'] == expected


def test_routed_deliverable_retrieves_only_bounded_registered_method_data():
    question = '给出产品落地建议，区分硬约束、试点验证和回退条件。'
    route = skills.route_knowledge_skill(question)
    selected = skills.retrieve_knowledge_skill(route['skill_id'], question)
    assert selected['trace']['status'] == 'ready'
    assert 0 < len(selected['sources']) <= 3 and len(selected['prompt'].encode()) <= 3500
    assert all(row['knowledge_scope'] == 'method_reference_not_user_evidence' for row in selected['sources'])
    assert selected['trace']['semantic_relevance_verified'] is False
