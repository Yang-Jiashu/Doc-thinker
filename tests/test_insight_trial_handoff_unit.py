"""Trial checks are explicit design requirements, never evaluated outcomes."""
from __future__ import annotations

import copy
import json

import pytest

from docthinker.server.insight_runtime import InsightRuntime, MAX_CONTEXT_BYTES
from docthinker.server.insight_schema import adapt_insight_schemas
from docthinker.server.research_knowledge import select_task_knowledge


@pytest.mark.parametrize("question", [
    "为产品制定上线计划，先试点验证方案，列出验收和停止条件。",
    "Compare two project plans; pilot against a baseline before scaling.",
])
def test_trial_method_is_whole_bounded_and_explicitly_proposed(question):
    support, route = select_task_knowledge(question)
    assert route['status'] == 'selected'
    card = next(row for row in support['sources'] if row['id'] == 'stage-gates-before-scale')
    assert len(support['prompt'].encode()) <= 3500
    assert card['knowledge_scope'] == 'method_reference_not_user_evidence'
    for term in ('对照', '样本', '时间', '指标测法', '通过', '停止', '数据不足', '待确认'):
        assert term in card['decision_rule']
    assert any('拟议试点不是论文已验证' in text for text in card['boundaries'])
    runtime = InsightRuntime(question, support)
    proposal = runtime.on_stage('after_evidence', {
        'active_method_labels': [card['label']],
        'sources': [{'id': 'literal', 'document_id': 'doc', 'label': 'S1',
                     'content': 'The prototype was evaluated on a fixed test set.'}],
    })
    assert proposal['checks']['semantic_premises'] == 'unknown'
    assert proposal['checks']['entailment_verified'] is False
    assert proposal['trace']['model_calls'] == proposal['trace']['network_calls'] == 0
    text = proposal['actions'][0]['payload']['text']
    assert '不事后降低门槛' in text and '待确认' in text
    context = runtime.render_context(proposal)
    assert len(json.dumps(context, ensure_ascii=False, separators=(',', ':')).encode()) <= MAX_CONTEXT_BYTES


def test_revised_trial_requirements_invalidate_old_action_identity():
    question = '请比较两套项目方案，先用试点验证关键假设，不达验收标准就停止。'
    current, _ = select_task_knowledge(question)
    card = next(row for row in current['sources'] if row['id'] == 'stage-gates-before-scale')
    current = {**current, 'sources': [card]}
    old = copy.deepcopy(current)
    old['sources'][0]['required_facts'] = ['关键假设与现状基线', '验证权限及损失上限']
    assert adapt_insight_schemas(current)[0].version != adapt_insight_schemas(old)[0].version
    old_action = InsightRuntime(question, old).on_stage('after_evidence', {})['actions'][0]['id']
    updated = InsightRuntime(question, current).on_stage('after_evidence', {
        'executed_action_ids': [old_action],
    })
    assert updated['actions'] and updated['actions'][0]['id'] != old_action


@pytest.mark.parametrize('mode,literal', [('off', False), ('auto', True)])
def test_trial_constraints_do_not_bypass_disabled_or_literal_mode(mode, literal):
    question = '比较两套项目方案，先试点验证再扩大投入。'
    support, _ = select_task_knowledge(question)
    runtime = InsightRuntime(question, support, mode=mode, literal=literal)
    for stage in ('before_retrieval', 'after_evidence', 'before_delivery'):
        assert runtime.on_stage(stage, {'allow_supplemental_retrieval': True})['actions'] == []
