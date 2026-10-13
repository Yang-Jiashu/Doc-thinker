"""Five built-in task packs share one bounded data-only retrieval contract."""
from copy import deepcopy
import json

import pytest

from docthinker.server import knowledge_skills as skills
from docthinker.server.research_knowledge import select_task_knowledge


CASES = [
    ('research-ideation', '帮我提出可证伪的科研假设，并设计最小验证实验。'),
    ('literature-synthesis', '对比这些论文的条件与指标，整理证据冲突。'),
    ('evidence-writing', '帮我写一份面向管理层的报告，论点要有证据。'),
    ('decision-review', '比较这两个产品方案的硬约束和风险，给出取舍建议。'),
    ('mao-workplace', '跨部门沟通不畅，项目延期，怎么明确负责人和分工？'),
]


@pytest.mark.parametrize('identifier,question', CASES)
def test_one_selected_resource_read_and_complete_original_method_cards(monkeypatch, identifier, question):
    original = skills._read_pack
    reads = []

    def read(selected):
        reads.append(selected)
        return original(selected)

    monkeypatch.setattr(skills, '_read_pack', read)
    support, route = select_task_knowledge(question)
    assert reads == [identifier]  # Not scanning all five packs before selection.
    assert route['status'] == 'selected'
    assert route['model_calls'] == route['network_calls'] == 0
    assert support['skill']['id'] == identifier
    assert support['trace']['status'] == 'ready'
    assert len(support['prompt'].encode()) <= 3500
    assert 1 <= len(support['sources']) <= 3
    assert [row['label'] for row in support['sources']] == [f'K{i+1}' for i in range(len(support['sources']))]
    original_entries = {row['id']: row for row in json.loads(original(identifier))['entries']}
    for row in support['sources']:
        for field in ('commentary', 'decision_rule', 'required_facts', 'action_template',
                      'counterexample', 'stop_conditions', 'boundaries'):
            assert row[field] == original_entries[row['id']][field]
        if identifier != 'mao-workplace':
            assert row['reference_kind'] == 'project_original'
            assert row['source_url'] == row['excerpt'] == ''
        else:
            assert row['reference_kind'] == 'historical_inspiration'
    assert support['trace']['memory_writes'] == support['trace']['graph_writes'] == 0


@pytest.mark.parametrize('identifier,question', CASES)
def test_off_and_literal_paths_skip_all_resource_reads(monkeypatch, identifier, question):
    monkeypatch.setattr(skills, '_read_pack', lambda _: pytest.fail('No resources on off/literal path'))
    assert select_task_knowledge(question, mode='off')[0] is None
    assert select_task_knowledge(question, literal=True)[0] is None


@pytest.mark.parametrize('field,value', [
    ('reference_kind', 'historical_inspiration'), ('reference_kind', 'verified_fact'),
    ('source_url', 'https://www.12371.cn/fake'), ('excerpt', 'Invented original quotation'),
    ('source_title', 'Fabricated external paper citation'),
    ('action_template', []), ('required_facts', []), ('decision_rule', ''),
])
def test_original_methods_cannot_masquerade_as_quoted_evidence(monkeypatch, field, value):
    data = json.loads(skills._read_pack('research-ideation'))
    data['entries'][0][field] = value
    monkeypatch.setattr(skills, '_read_pack', lambda _: json.dumps(data, ensure_ascii=False).encode())
    result = skills.retrieve_knowledge_skill('research-ideation', '研究假设')
    assert result['trace']['status'] == 'unavailable'
    assert not result['sources'] and not result['prompt']


def test_original_pack_requires_explicit_original_marker(monkeypatch):
    data = json.loads(skills._read_pack('evidence-writing'))
    del data['entries'][0]['reference_kind']
    monkeypatch.setattr(skills, '_read_pack', lambda _: json.dumps(data, ensure_ascii=False).encode())
    assert skills.retrieve_knowledge_skill('evidence-writing', '写报告论点证据')['trace']['status'] == 'unavailable'


@pytest.mark.parametrize('question,expected', [
    ('不要提出研究假设，只比较这些论文的条件和冲突。', 'literature-synthesis'),
    ('不要用毛选，帮我写一份工作报告。', 'evidence-writing'),
    ('对比这些论文，设计可证伪的研究假设。', 'research-ideation'),
    ('围绕这个概念帮我提出可证伪的研究假设。', 'research-ideation'),
    ('写一份报告，比较团队的多个产品方案。', 'evidence-writing'),
    ('我不要你写报告，先告诉我资料上传失败怎么办。', None),
    ('这个研究假设是什么意思？', None),
    ('这不是项目方案取舍问题，只是想知道决策矩阵这个词的意思。', None),
    ('比较这两个手术方案的风险，该怎么选？', None),
    ('Compare two investment options and decide which stock to buy.', None),
    ('请翻译：写一份报告，比较方案。', None),
    ('"Propose research hypotheses"，这几个词有几个字符？', None),
])
def test_scope_negation_and_specific_deliverable_precedence(question, expected):
    assert skills.route_knowledge_skill(question)['skill_id'] == expected


@pytest.mark.parametrize('identifier,question', CASES)
def test_mutating_public_result_does_not_change_pack_or_other_tasks(identifier, question):
    result = skills.retrieve_knowledge_skill(identifier, question)
    pristine = deepcopy(result)
    result['sources'][0]['decision_rule'] = 'mutated caller text'
    assert skills.retrieve_knowledge_skill(identifier, question) == pristine
