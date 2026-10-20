"""Synthetic intent allocation tests; no benchmark paper facts or model calls."""
from docthinker.server.research_faceted_assist import _focused_requirement, _rerank_pairs, _content_requirements
from docthinker.server.research_query_facets import build_query_facets


def plan(question, documents=()):
    return build_query_facets(question, [dict(document_id=name, filename=name + '.pdf')
                                         for name in documents])['facets']


def test_content_need_selected_not_negative_writing_constraint():
    facets = plan('Atlas 更新哪些参数？说明使用什么监督、优化什么目标，以及为什么不改索引。不要把所有参数理解成全部更新。')
    selected = _focused_requirement(facets)
    assert selected['text'] == '说明使用什么监督、优化什么目标，以及为什么不改索引'
    assert not selected.get('derived_translation')


def test_single_need_and_anaphoric_limitation_remain_shared_query():
    facets = plan('Atlas 的 buffer、queue、dispatch 分别对应什么机制？这个类比有什么边界？请给出处。')
    assert _focused_requirement(facets) is None


def test_explicit_multi_document_comparison_uses_document_fairness_not_query_branches():
    facets = plan('Atlas 和 Boreal 如何处理索引？各自实验测量了什么？', ['Atlas', 'Boreal'])
    assert _focused_requirement(facets) is None


def test_translation_cannot_create_a_second_independent_requirement():
    facets = plan('Atlas 如何更新参数？')
    facets.append(dict(id='translated', text='What is optimized?', kind='clause', derived_translation=True))
    assert _focused_requirement(facets) is None


def test_48_total_pairs_source_balanced_and_no_source_derived_queries():
    question = 'Atlas 更新哪些参数？说明使用什么监督、优化什么目标，以及为什么不改索引。'
    facets = plan(question, ['Atlas'])
    focus = _focused_requirement(facets)
    original = next(row for row in facets if row['kind'] == 'original')
    candidates = [dict(id=f'w{i:02}', source_id=f's{i//2:02}', document_id='Atlas',
                       content='SOURCE_ONLY_NOT_A_QUERY',
                       facet_scores={focus['id']: 1-i/100, original['id']: .5}) for i in range(48)]
    pairs = _rerank_pairs(question, candidates, facets)
    full = [row for row in pairs if row['rerank_branch'] == 'full']
    focused = [row for row in pairs if row['rerank_branch'] == 'focused']
    assert len(pairs) == len({row['id'] for row in pairs}) == 48
    assert len(full) == len(focused) == 24
    assert len({row['source_id'] for row in focused}) == 24
    assert {row['query'] for row in full} == {question}
    assert {row['query'] for row in focused} == {focus['text']}
    assert all(row['facet_scores'][focus['id']] > 0 for row in focused)
    assert all(row['content'] == 'SOURCE_ONLY_NOT_A_QUERY' for row in pairs)
    assert all('query' not in row for row in candidates)


def test_absent_facet_support_does_not_create_rescue_branch():
    question = 'What does Atlas optimize? Why is the index fixed?'
    facets = plan(question)
    candidates = [dict(id=f'w{i}', source_id=f's{i}', document_id='Atlas', facet_scores={}) for i in range(48)]
    pairs = _rerank_pairs(question, candidates, facets)
    assert len(pairs) == 48
    assert all('rerank_branch' not in row for row in pairs)


def test_format_and_citation_questions_do_not_become_retrieval_needs():
    assert _focused_requirement(plan('What does Atlas optimize? How should I format the answer and what citations should I include?')) is None
    assert _focused_requirement(plan('Atlas 如何调度？请说明这个类比在哪些地方只是设计启发。')) is None


def test_demonstrative_experiment_and_selection_criteria_are_content():
    facets = plan('Atlas 如何训练？按哪些标准选择训练样本？这些实验如何排除数据泄漏？')
    selected = _focused_requirement(facets)
    assert selected is not None
    assert selected['text'] in {'按哪些标准选择训练样本', '这些实验如何排除数据泄漏'}


def test_named_comparison_scoring_removes_only_other_exact_names():
    question = 'Compare Atlas, GraphAtlas and Boreal: explain the mechanism, not a universal solution.'
    facets = plan(question, ['Atlas', 'GraphAtlas', 'Boreal'])
    original = next(row for row in facets if row['kind'] == 'original')
    candidates = [dict(id=name, source_id=name, document_id=name, content='PRIVATE SOURCE TERM',
                       facet_scores={original['id']: .4}) for name in ['Atlas', 'GraphAtlas', 'Boreal']]
    pairs = _rerank_pairs(question, candidates, facets)
    assert len(pairs) == 3 and len({row['query'] for row in pairs}) == 3
    graph = next(row for row in pairs if row['document_id'] == 'GraphAtlas')
    assert 'GraphAtlas' in graph['query'] and 'Boreal' not in graph['query']
    assert all('not a universal solution' in row['query'] for row in pairs)
    assert all('PRIVATE SOURCE TERM' not in row['query'] for row in pairs)
    assert all(row.get('rerank_branch') != 'focused' for row in pairs)


def test_coordinated_validation_request_is_distinct_without_paper_specific_words():
    question = 'Atlas在何处合并输入？请解释处理流程，它为何能扩展，以及实验中是否已经验证跨域泛化。'
    facets = plan(question)
    needs = _content_requirements(facets)
    assert any(row['text'] == '实验中是否已经验证跨域泛化' and row['requirement_kind'] == 'verification' for row in needs)
    assert _focused_requirement(facets)['text'] == '实验中是否已经验证跨域泛化'
    assert all(row['text'] in question for row in needs)


def test_imperative_limitations_and_experimental_controls_are_content_not_formatting():
    question = 'Explain the transport mechanism. Describe its limitations. Identify experimental controls.'
    needs = _content_requirements(plan(question))
    assert len(needs) == 3
    assert {row['requirement_kind'] for row in needs} >= {'limitation', 'experiment'}
    assert _focused_requirement(plan(question))['text'] in {'Describe its limitations', 'Identify experimental controls'}


def test_two_demands_share_24_pairs_and_never_become_48_pairs_each():
    question = 'Where does Atlas combine inputs? Explain how its controller scales. Determine whether the study tested joint tuning?'
    facets = plan(question)
    needs = _content_requirements(facets)
    assert len(needs) == 3
    candidates = [dict(id=f'w{i}',source_id=f's{i}',document_id='Atlas',content='SOURCE_ONLY',
                       facet_scores={row['id']: .9-i/100 for row in facets}) for i in range(48)]
    pairs = _rerank_pairs(question,candidates,facets)
    focused = [row for row in pairs if row.get('rerank_branch') == 'focused']
    assert len(pairs) == 48 and len(focused) == 24
    assert len({row['query'] for row in focused}) == 2
    assert len({row['id'] for row in pairs}) == 48
    assert all(sum(other['query'] == row['query'] for other in focused) == 12 for row in focused)
    for row in focused:
        assert row['query'] in question and row['query'] != row['content']
        assert row['retrieval_support'] == row['facet_scores'][row['rerank_facet_id']] > 0


def test_paraphrase_aliases_do_not_create_two_independent_requirements():
    assert _content_requirements(plan('How does Atlas learn? What does Atlas learn?')) == []


def test_unsupported_high_priority_requirement_gets_no_scoring_credit():
    question = 'How does Atlas route requests? Which experiments measure latency? Whether the study tested transfer?'
    facets = plan(question)
    needs = _content_requirements(facets)
    unsupported = next(row for row in needs if row['requirement_kind'] == 'verification')
    candidates = [dict(id=f'w{i}',source_id=f's{i}',document_id='Atlas',content='SOURCE_ONLY',
                       facet_scores={row['id']:.5 for row in facets if row['id'] != unsupported['id']}) for i in range(48)]
    pairs = _rerank_pairs(question,candidates,facets)
    assert all(row.get('rerank_facet_id') != unsupported['id'] for row in pairs)
    assert len(pairs) <= 48


def test_pair_hard_ceiling_also_applies_to_direct_callers_without_focus():
    question = 'Explain Atlas.'
    facets = plan(question)
    rows = [dict(id=f'w{i}',facet_scores={}) for i in range(100)]
    assert len(_rerank_pairs(question,rows,facets)) == 48
