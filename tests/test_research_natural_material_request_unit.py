"""Natural material requests must retain literal needs, without model calls."""
from copy import deepcopy

import pytest

from docthinker.server.research_query_facets import build_query_facets
from docthinker.server.research_retrieval import build_retrieval_plan
from docthinker.server.research_lexical_coverage import select_lexical_coverage
from docthinker.server.research_evidence_selection import prepare_evidence_candidates


def row(identifier, document, ordinal, text, filename):
    return {'id': identifier, 'document_id': document, 'ordinal': ordinal,
            'content': text, 'filename': filename, 'page': ordinal + 1}


def named_documents():
    return [row('quartz', 'doc-quartz', 0, 'Original text.', 'QuartzCache_2511.45678v2.pdf'),
            row('terra', 'doc-terra', 0, 'Original text.', 'TerraLedger_2026-08-21.pdf')]


def mentioned_ids(plan):
    return {doc for facet in plan['facets'] for doc in facet.get('document_ids', ())}


def test_technical_stem_plus_identifier_or_date_is_exact_document_metadata():
    chunks = named_documents()
    snapshot = deepcopy(chunks)
    plan = build_query_facets('请读 QuartzCache 和 TerraLedger，解释它们的方法。', chunks)
    assert mentioned_ids(plan) == {'doc-quartz', 'doc-terra'}
    assert plan['trace']['matched_document_count'] == 2
    assert not any(facet['kind'] == 'technical_phrase' and facet['text'] in {'QuartzCache', 'TerraLedger'}
                   for facet in plan['facets'])  # Names are scope, not evidence utility.
    assert chunks == snapshot and plan['trace']['provider_calls'] == 0


@pytest.mark.parametrize('name,question', [
    ('map_2511.45678v2.pdf', 'Explain map.'),
    ('cache_2026-08-21.pdf', 'Explain cache.'),
    ('report_2026-08-21.pdf', 'Explain report.'),
    ('QuartzCache_2511.45678v2.pdf', 'Explain Quartz.'),
    ('QuartzCache_notes.pdf', 'Explain QuartzCache.'),
])
def test_short_generic_substrings_and_non_identifier_suffixes_are_not_aliases(name, question):
    assert mentioned_ids(build_query_facets(question, [row('x', 'doc', 0, 'Body.', name)])) == set()


def test_ambiguous_short_alias_never_selects_one_arbitrary_version():
    chunks = named_documents()[:1]
    chunks.append(row('quartz-2', 'doc-other-version', 0, 'Different body.', 'QuartzCache_2512.12345v1.pdf'))
    ids = mentioned_ids(build_query_facets('Explain QuartzCache methods.', chunks))
    assert ids in (set(), {'doc-quartz', 'doc-other-version'})


def test_material_subquestions_receive_slots_before_personal_background_or_style():
    question = ('我们刚成立一个小团队，希望把资料用起来。请读 Aurelia 和 Zephyrus，说明怎样把历史资料用于回答。'
                '论文给了哪些实际验证？请解释方法在什么条件下不成立。团队暂时只有少量预算。请用简洁中文回答。')
    plan = build_query_facets(question)
    clauses = [facet['text'] for facet in plan['facets'] if facet['kind'] == 'clause']
    assert any('怎样把历史资料用于回答' in text for text in clauses)
    assert any('哪些实际验证' in text for text in clauses)
    assert any('什么条件下不成立' in text for text in clauses)
    assert len(plan['facets']) <= 8


@pytest.mark.parametrize('question', [
    '论文给了哪些实际验证？', '请说明这项研究的验证情况。',
    'What was empirically tested in these studies?',
])
def test_explicit_empirical_material_request_has_experiment_role_only_as_query_hint(question):
    plan = build_query_facets(question)
    assert any('experiment_setup' in facet['roles'] for facet in plan['facets'])
    assert plan['trace']['provider_calls'] == plan['trace']['llm_tokens'] == 0


def test_conditional_and_negative_request_is_never_stripped_to_expand_scope():
    question = '如果没有跨组对照，请说明 QuartzCache 的验证情况，不要把用户评价当因果证明。'
    plan = build_query_facets(question, named_documents())
    for facet in plan['facets']:
        if facet['kind'] == 'clause':
            assert '如果没有跨组对照' in facet['text'] and '不要把用户评价当因果证明' in facet['text']


def natural_corpus():
    chunks = []
    for name, filename in [('QuartzCache', 'QuartzCache_2511.45678v2.pdf'),
                           ('TerraLedger', 'TerraLedger_2026-08-21.pdf')]:
        for n in range(4):
            chunks.append(row(f'{name}-intro-{n}', name, n,
                f'1 Introduction\n{name} provides helpful assistance. We introduce a hierarchical memory system. '
                f'{name} promises convenient interactions for users in scenario {n}.', filename))
        chunks.append(row(name + '-method', name, 4,
            '2 Retrieval Methods\nStored records are encoded into vectors and indexed. '
            'The current conversation is encoded as a query; the nearest records are retrieved and supplied to the generator. '
            'Private records are excluded unless permission is granted.', filename))
        chunks.append(row(name + '-study', name, 5,
            '3 Experiments\nWe evaluated 37 prepared questions against a no-records baseline. '
            'The same model and output limit were used in both conditions. '
            'The result was not tested on real customers and does not establish long-term reliability.', filename))
    return chunks


def test_chinese_material_request_can_nominate_named_english_methods_and_actual_test_conditions():
    question = ('我们计划做一个资料助手。请读 QuartzCache 和 TerraLedger，写产品落地建议：'
                '两种方案各解决什么问题，怎样把历史资料用于回答，论文给了哪些实际验证？'
                '推荐先做什么、暂缓什么。')
    chunks = natural_corpus()
    original = deepcopy(chunks)
    baseline = build_retrieval_plan(question, chunks)['sources']
    result = select_lexical_coverage(question, baseline, chunks)
    selected = {source['id']: source for source in result['sources']}
    assert {'QuartzCache-method', 'TerraLedger-method', 'QuartzCache-study', 'TerraLedger-study'} <= selected.keys()
    originals = {source['id']: source['content'] for source in chunks}
    for source in result['sources']:
        assert originals[source['id']][source['excerpt_start']:source['excerpt_end']] == source['content']
    assert len(result['sources']) <= 8 and sum(len(source['content'].encode()) for source in result['sources']) <= 12000
    assert result['trace']['candidate_count'] <= 48 and result['trace']['provider_calls'] == 0
    assert result['trace']['semantic_entailment_verified'] is False and chunks == original


@pytest.mark.parametrize('question', [
    'Compare QuartzCache and TerraLedger methods and experiments using only these documents.',
    '只依据 QuartzCache 和 TerraLedger 两份文档比较方法和实验。',
])
def test_open_original_nomination_does_not_override_explicit_exclusive_scope(question):
    chunks = named_documents()
    chunks.append(row('third', 'third', 0,
        '3 Experiments\nQuartzCache TerraLedger comparison evaluates both methods on held-out data. '
        'Results show no improvement under fixed conditions.', 'OtherAudit.pdf'))
    facets = build_query_facets(question, chunks)
    assert facets['trace']['protected_clause_count'] > 0
    pool = prepare_evidence_candidates(question, chunks, [], facets, {})
    assert not any(item['document_id'] == 'third' for item in pool['candidates'])


def test_numeric_method_parent_exposes_plain_child_body_without_inheriting_into_experiments():
    chunks = []
    for name, filename in [('QuartzCache', 'QuartzCache_2511.45678v2.pdf'),
                           ('TerraLedger', 'TerraLedger_2026-08-21.pdf')]:
        chunks.extend([
            row(name + '-parent', name, 0, '2 Methods\nWe describe the proposed design below.', filename),
            row(name + '-method', name, 1,
                '2.1 Record Handling\nThe request is encoded as a vector. Matching entries are retrieved '
                'from the stored index and supplied to the decoder. Records without consent are never loaded.', filename),
            row(name + '-study', name, 2,
                '3 Experiments\nWe evaluated 48 independently held-out requests against a fixed baseline. '
                'The retrieved records were supplied to the same decoder. Results show improvement only with clean entries.', filename),
            row(name + '-overview', name, 3,
                '4 Conclusion\nWe propose a system with an index that retrieves stored records and passes '
                'memory from a cache to a decoder. We evaluated the model against several baselines.', filename),
            row(name + '-noise', name, 4, '5 Related Work\nThis is general background about useful assistants.', filename),
        ])
    question = '请读 QuartzCache 和 TerraLedger，说明如何利用记录，以及论文实际验证了什么。'
    baseline = build_retrieval_plan(question, chunks)['sources']
    result = select_lexical_coverage(question, baseline, chunks)
    selected = {source['id'] for source in result['sources']}
    assert {'QuartzCache-method', 'TerraLedger-method', 'QuartzCache-study', 'TerraLedger-study'} <= selected
    originals = {source['id']: source['content'] for source in chunks}
    for source in result['sources']:
        assert originals[source['id']][source['excerpt_start']:source['excerpt_end']] == source['content']


def test_numbered_parent_scope_is_document_local_and_explicit_child_roles_override_it():
    rows = [
        row('parent', 'doc-quartz', 0, '2 Methods\nDesign details follow.', 'QuartzCache_2511.45678v2.pdf'),
        row('child', 'doc-quartz', 1, '2.1 Record Handling\nEntries are encoded and retrieved from the index.', 'QuartzCache_2511.45678v2.pdf'),
        row('study', 'doc-quartz', 2, '2.2 Experiments\nWe evaluated a fixed sample against a baseline.', 'QuartzCache_2511.45678v2.pdf'),
        row('later', 'doc-quartz', 3, '3 Discussion\nEntries are encoded and retrieved from the index.', 'QuartzCache_2511.45678v2.pdf'),
        row('other', 'doc-terra', 0, '2.1 Record Handling\nEntries are encoded and retrieved from the index.', 'TerraLedger_2026-08-21.pdf'),
    ]
    question = '请读 QuartzCache 和 TerraLedger，说明如何工作及实际验证。'
    pool = prepare_evidence_candidates(question, rows, [], build_query_facets(question, rows), {})
    selected = {item['source_id']: item for item in pool['candidates']}
    assert selected['child']['outline_roles'] == ['mechanism']
    assert selected['study']['outline_roles'] == ['experiment_setup']
    assert 'later' not in selected and 'other' not in selected


def test_positive_unnumbered_experiment_heading_clears_numeric_method_parent_across_chunks():
    filename = 'QuartzCache_2511.45678v2.pdf'
    rows = [row('method', 'doc', 0, '2 Methods\nEntries are indexed and retrieved.', filename),
            row('study', 'doc', 1, 'Evaluation\nWe evaluated the index against a fixed baseline.', filename),
            row('study-next', 'doc', 2, 'The experiment retrieved entries for 48 held-out requests.', filename)]
    question = '请读 QuartzCache，说明如何工作和实际验证。'
    pool = prepare_evidence_candidates(question, rows, [], build_query_facets(question, rows), {})
    found = {item['source_id']: item for item in pool['candidates']}
    assert found['study']['outline_roles'] == ['experiment_setup']
    assert found['study-next']['outline_roles'] == ['experiment_setup']
    assert not found['study']['outline_path'] and not found['study-next']['outline_path']


def test_evaluation_goal_is_not_a_substitute_for_available_experiment_design():
    filename = 'QuartzCache_2511.45678v2.pdf'
    rows = [row('goal', 'doc', 0, '3 Experiments\nThe primary objective of our experiment is to evaluate the system.', filename),
            row('design', 'doc', 1, 'We evaluated 96 held-out requests against a fixed baseline. '
                'The same model and output limit were used. No generalization beyond these requests was tested.', filename)]
    rows.extend(row('noise-' + str(i), 'doc', i + 2, '4 Discussion\nThis background discusses popular applications.', filename)
                for i in range(8))
    q = '请说明 QuartzCache 实际验证了什么。'
    result = select_lexical_coverage(q, build_retrieval_plan(q, rows)['sources'], rows)
    assert 'design' in {source['id'] for source in result['sources']}
    selected = next(source for source in result['sources'] if source['id'] == 'design')
    assert 'No generalization beyond these requests was tested.' in selected['content']
