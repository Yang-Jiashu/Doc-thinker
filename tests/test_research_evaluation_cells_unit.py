"""Evaluation evidence complements: synthetic, fixed before implementation."""
from copy import deepcopy

import pytest

from docthinker.server.research_lexical_coverage import select_lexical_coverage, evaluation_cells
from docthinker.server.research_query_facets import build_query_facets
from docthinker.server.research_retrieval import build_retrieval_plan


def row(identifier, ordinal, text, *, doc='QuartzCache', status=None):
    value = {'id': identifier, 'document_id': doc, 'filename': doc + '_2502.12345.pdf',
             'ordinal': ordinal, 'page': ordinal + 1, 'content': text}
    if status:
        value['status'] = status
    return value


def corpus():
    return [
        row('method', 0, '2 Methods\nThe bounded queue stores the latest permitted records. '
            'The reader retrieves matching records and evicts the oldest entries when capacity is reached.'),
        row('setup', 1, '3 Experiments\nWe evaluated independently held-out requests against a fixed keyword baseline. '
            'The same model and output budget were used in both conditions.'),
        row('measure', 2, '3.1 Measurement\nWe define retrieval precision as the fraction of returned entries judged relevant. '
            'Annotators score answer correctness on a three-level scale; agreement is measured separately.'),
        row('outcome', 3, '3.2 Results\nResults show no improvement over the keyword baseline under changed workloads. '
            'Accuracy increases only when the workload remains stationary; concurrent edits were not tested.'),
        *[row('case-' + str(i), i + 4, '4 Case Examples\nWe evaluated one conversation against a baseline. '
              'The model retrieves stored memory and generates a response that appears helpful.') for i in range(6)],
    ]


def check_geometry(result, originals):
    by_id = {r['id']: r for r in originals}
    assert len(result['sources']) <= 8
    assert sum(len(r['content'].encode()) for r in result['sources']) <= 12000
    assert result['trace']['candidate_count'] <= 48
    assert result['trace']['provider_calls'] == result['trace']['model_calls'] == 0
    assert not result['trace']['semantic_entailment_verified']
    for s in result['sources']:
        assert by_id[s['id']]['content'][s['excerpt_start']:s['excerpt_end']] == s['content']
        assert s['document_id'] == by_id[s['id']]['document_id']


@pytest.mark.parametrize('question', [
    '请说明 QuartzCache 的方法，论文实际验证了什么。',
    'What was empirically tested in QuartzCache and what method was used?',
])
def test_requested_actual_evaluation_preserves_setup_measurement_and_negative_outcome(question):
    rows = corpus(); saved = deepcopy(rows)
    result = select_lexical_coverage(question, build_retrieval_plan(question, rows)['sources'], rows)
    found = {r['id']: r for r in result['sources']}
    assert {'method', 'setup', 'measure', 'outcome'} <= found.keys()
    assert 'no improvement' in found['outcome']['content'] and 'were not tested' in found['outcome']['content']
    assert rows == saved
    check_geometry(result, rows)


def test_measurement_window_in_same_source_gets_a_nomination_opportunity_after_early_case():
    rows = corpus()
    early = ('The example response retrieves records and generates helpful text. ' * 12)
    measurement = ('Measurement Protocol: Human annotators score response correctness against the recorded target. '
                   'We define precision as the fraction of retrieved records judged relevant. '
                   'Disagreements are resolved by a separate assessor; changed workloads were not tested.')
    rows[2] = row('measure', 2, early + measurement)
    rows.extend(row('noise-' + str(i), i + 20,
                    '3 Experiments\nWe evaluated a baseline and retrieved stored records for the example response.')
                for i in range(60))
    q = '请说明 QuartzCache 论文实际验证了什么。'
    result = select_lexical_coverage(q, build_retrieval_plan(q, rows)['sources'], rows)
    assert any('Human annotators score response correctness' in s['content'] and
               'changed workloads were not tested' in s['content'] for s in result['sources'])
    check_geometry(result, rows)


def test_numbers_and_planned_goals_do_not_substitute_for_reported_outcomes():
    rows = corpus()
    rows.extend([row('numbers', 30, '3 Results\nNumbers 11 22 33 44 55 66 77 88 99.'),
                 row('goal', 31, '3 Experiments\nOur goal is to evaluate performance on 999 requests.')])
    q = '请说明 QuartzCache 论文实际验证了什么。'
    result = select_lexical_coverage(q, build_retrieval_plan(q, rows)['sources'], rows)
    ids = {r['id'] for r in result['sources']}
    assert 'outcome' in ids and 'numbers' not in ids
    check_geometry(result, rows)


def test_missing_measurement_is_not_filled_from_an_unrelated_document_or_revoked_source():
    rows = [r for r in corpus() if r['id'] != 'measure']
    rows.extend([row('revoked', 20, '3 Metrics\nAnnotators score response accuracy on a fixed scale.', status='revoked'),
                 row('foreign', 0, '3 Metrics\nAnnotators score response accuracy on a fixed scale.', doc='CedarOther')])
    q = '只依据 QuartzCache 说明论文实际验证了什么。'
    result = select_lexical_coverage(q, build_retrieval_plan(q, rows)['sources'], rows)
    assert not {'revoked', 'foreign'} & {r['id'] for r in result['sources']}
    assert not result['trace']['semantic_entailment_verified']


def test_literal_only_task_does_not_activate_new_evaluation_coverage():
    rows = corpus(); q = '原样引用 QuartzCache 中的实验结果，不要分析。'
    baseline = build_retrieval_plan(q, rows)['sources']
    result = select_lexical_coverage(q, baseline, rows)
    assert result['sources'] == baseline
    assert result['trace']['reason'] == 'literal_or_editorial_request'


def test_actual_verification_is_query_side_outcome_intent_not_a_claim_of_success():
    facets = build_query_facets('这项论文实际验证了什么？')
    assert any({'experiment_setup', 'result'} <= set(f['roles']) for f in facets['facets'])
    assert facets['trace']['provider_calls'] == 0


def test_caption_supplement_cannot_drag_in_following_unrelated_background():
    caption = 'Figure 8: Accuracy measured against a fixed baseline under two workload conditions.'
    background = ('Other-transformers. Other systems cache keys and values. '
                  'Their sparse routing improves computational complexity in earlier publications.')
    text = caption + '\n' + background
    result = build_retrieval_plan('accuracy baseline workload', [row('caption', 0, text)], max_chunk_chars=800)
    selected = result['sources'][0]
    assert caption in selected['content']
    assert 'Other-transformers' not in selected['content']
    assert text[selected['excerpt_start']:selected['excerpt_end']] == selected['content']


@pytest.mark.parametrize('text, section, rejected', [
    ('Our planned experiment aims to outperform the keyword baseline.', '3 Experiments', 'outcome'),
    ('Prior studies outperform baselines on a different workload.', '2 Related Work', 'outcome'),
    ('No annotation criteria were reported; annotators did not score responses.', '3 Evaluation', 'measurement'),
])
def test_proposed_third_party_and_missing_measurement_are_not_completed_cells(text, section, rejected):
    assert rejected not in evaluation_cells({'content': text, 'section': section})


def test_proposed_method_with_reported_observation_is_not_a_planned_experiment():
    observed = ('The proposed method outperforms the sequential baseline on the held-out trial. '
                'The gain was observed only for stationary requests, not concurrent edits.')
    assert 'outcome' in evaluation_cells({'content': observed, 'section': '3 Results'})
    future = 'The proposed experiment will compare the method against a sequential baseline next year.'
    assert 'outcome' not in evaluation_cells({'content': future, 'section': '3 Experiments'})


def test_simulated_observation_without_control_is_a_setup_not_a_validity_claim():
    text = ('We generated histories for 27 virtual users over 6 days. '
            'The histories were synthesized from assigned profiles; 83 test questions assessed recall. '
            'No independent human participants or control group were included.')
    assert 'setup' in evaluation_cells({'content': text, 'section': '3 Experiments'})
    rows = [row('simulated', 1, '3 Experiments\n' + text)]
    q = 'QuartzCache 实际验证了什么？'
    result = select_lexical_coverage(q, [], rows)
    assert text in result['sources'][0]['content']
    assert not result['trace']['semantic_entailment_verified']
    check_geometry(result, rows)


def test_abstract_claim_cannot_replace_concrete_conditioned_findings():
    rows = corpus() + [row('abstract', 0, '1 Introduction\nWe show that our system outperforms strong baselines. '
                              'Our architecture stores records and retrieves memory to generate answers.')]
    q = 'QuartzCache 的方法和实际验证是什么？'
    result = select_lexical_coverage(q, build_retrieval_plan(q, rows)['sources'], rows)
    found = {r['id']: r for r in result['sources']}
    assert {'setup', 'measure', 'outcome'} <= found.keys()
    assert 'were not tested' in found['outcome']['content']
    check_geometry(result, rows)


def test_bounded_state_update_competes_with_unlimited_overview_without_inference():
    detail = ('The stored record queue retains at most the configured capacity of entries. '
              'After each retrieval, the oldest entry is removed and the new record is appended. '
              'Eviction does not preserve records beyond this retention limit.')
    rows = [row('claim', 0, '1 Introduction\nOur memory architecture can update unlimited memory and retrieve records.'),
            row('bounded', 1, '2 Methods\n' + detail),
            row('read', 2, '2.1 Retrieval\nThe reader encodes a request, searches stored records and selects matching entries.')]
    q = 'Explain the QuartzCache memory method and how it retrieves records.'
    result = select_lexical_coverage(q, build_retrieval_plan(q, rows[:1])['sources'], rows)
    assert any(detail in source['content'] for source in result['sources'])
    assert result['trace']['selection']['state_maintenance_cells'] == 1
    check_geometry(result, rows)


@pytest.mark.parametrize('text', [
    'Our system can update unlimited memory and retrieve stored records.',
    'The cache has 512 entries and the variable M is reported.',
    'We plan to retain at most 60 records and evict the oldest entry in a future queue.',
])
def test_maintenance_cell_needs_visible_operation_and_bound_not_a_promise(text):
    rows = [row('text', 0, '2 Methods\n' + text)]
    q = 'Explain the QuartzCache memory method.'
    result = select_lexical_coverage(q, [], rows)
    assert result['trace'].get('selection', {}).get('state_maintenance_cells', 0) == 0


def test_distinct_preparation_and_answer_operations_compete_with_evaluation_detail():
    prep = ('We first derive stable record identifiers from the source collection and then construct an offline index. '
            'The index is rebuilt only after an authorized source change.')
    answer = ('For retrieval we first retrieve candidates from the index and then generate a response from selected records. '
              'The response keeps links to those original records.')
    rows = [row('prepare', 0, '2 Methods\n' + prep, doc='PebbleVault'),
            row('answer', 1, '2.1 Retrieval\n' + answer, doc='PebbleVault'),
            row('answer-detail', 2, '2.2 Retrieval Details\nThe retrieval response combines retrieved records. '
                'The retrieval generator summarizes the selected records and links the retrieval output.', doc='PebbleVault')]
    for doc in ('PebbleVault', 'WillowMap', 'CedarLedger'):
        rows.extend([row(doc+'-setup', 3, '3 Experiments\nWe evaluated 70 held-out retrieval requests against a fixed baseline. '
                         'The model and output budget remained fixed.', doc=doc),
                     row(doc+'-metric', 4, '3.1 Metrics\nAnnotators score retrieval accuracy using the original records. '
                         'Accuracy is defined as the fraction of correctly returned records.', doc=doc),
                     row(doc+'-result', 5, '3.2 Results\nResults show no improvement on the held-out retrieval requests. '
                         'The observation does not establish equivalence on other workloads.', doc=doc)])
    q = 'Compare PebbleVault, WillowMap and CedarLedger retrieval methods and evaluations.'
    result = select_lexical_coverage(q, [], rows)
    found = {r['id']: r for r in result['sources']}
    assert {'prepare', 'answer'} <= found.keys()
    assert 'only after an authorized source change' in found['prepare']['content']
    check_geometry(result, rows)


def test_explicit_figure_data_request_retains_caption_opportunity():
    caption = 'Figure 2: QuartzCache retrieval architecture results and measured graph values under the fixed workload.'
    body = ('2.1 Storage\nExternal storage refers to a disk-backed area outside the processor window. '
            'Its records must be copied into the active window; it is not directly readable.')
    rows = [row('figure', 0, caption + '\n' + body)]
    q = 'Explain QuartzCache retrieval architecture and Figure 2 graph values.'
    result = select_lexical_coverage(q, [], rows)
    assert any(caption in source['content'] for source in result['sources'])
    check_geometry(result, rows)


def test_ordered_action_contract_and_experiment_guard_share_one_role():
    from docthinker.server.research_lexical_coverage import _role_strength
    from docthinker.server.research_component_coverage import ordered_method_steps
    text = 'The pipeline first extracts permitted record identifiers and then builds an exact-match tree.'
    method = {'content': text, 'section': '2 Method', 'outline_roles': ['mechanism']}
    experiment = {**method, 'section': '3 Experiments', 'outline_roles': ['experiment_setup']}
    pair = ordered_method_steps(method)
    assert pair == ('extracts', 'builds')
    assert _role_strength(method, pair)['mechanism'] >= .9
    assert _role_strength(experiment, pair)['mechanism'] < .4
    planned = {**method, 'content': 'We plan a pipeline that first extracts records and then builds a tree.'}
    assert _role_strength(planned, ordered_method_steps(planned))['mechanism'] < .9


@pytest.mark.parametrize('text', [
    'Latency is measured in milliseconds by a monotonic clock.',
    'Annotators score correctness on a three-level scale.',
    'Precision is calculated as the fraction of correctly retrieved records.',
])
def test_measurement_completeness_accepts_alternative_not_just_denominators(text):
    from docthinker.server.research_lexical_coverage import _evaluation_quality
    source = {'content': text, 'section': '3 Measurements'}
    assert 'measurement' in evaluation_cells(source)
    assert .5 <= _evaluation_quality(source, 'measurement') <= 1.


def test_noncoverage_matching_projection_preserves_constraints_and_literal_requests():
    from docthinker.server.research_evidence_selection import literal_query_surface
    text = ('The queue does not preserve records after eviction. '
            'The trial did not improve accuracy. '
            'This tuning does not describe the online answer stage.')
    projected = literal_query_surface(text, positive_evidence_request=True)
    assert 'does not preserve records' in projected
    assert 'did not improve accuracy' in projected
    assert 'online answer stage' not in projected
    assert literal_query_surface(text) == text
    quoted = 'The string is "This tuning does not describe the online answer stage."'
    assert 'online answer stage' in literal_query_surface(quoted, positive_evidence_request=True)
    rows = [row('literal', 0, text)]
    q = '逐字引用 QuartzCache 中 does not describe 的声明，不作分析。'
    result = select_lexical_coverage(q, build_retrieval_plan(q, rows)['sources'], rows)
    assert result['trace']['reason'] == 'literal_or_editorial_request'
    assert text in result['sources'][0]['content']


def test_outcome_scope_cannot_be_borrowed_from_unrelated_noncoverage_statement():
    from docthinker.server.research_lexical_coverage import _evaluation_quality
    simple = 'Results show unchanged accuracy for the stationary workload.'
    vague = {'content': simple + ' This run does not describe the online method.', 'section': '3 Results'}
    scoped = {'content': 'Results show unchanged accuracy against the baseline only at low load. '
                         'High-load conditions were not tested.', 'section': '3 Results'}
    assert _evaluation_quality(vague, 'outcome') == .5
    assert _evaluation_quality(scoped, 'outcome') > .5


@pytest.mark.parametrize('question, expected', [
    ('Separate preparation from online answering in the memory method.', True),
    ('Explain the memory method.', False),
    ('Explain only memory capacity and eviction maintenance.', False),
    ('Explain memory mechanisms and separate demonstrated facts from speculation.', False),
])
def test_explicit_stage_scope_requires_both_requested_heading_sides(question, expected):
    from docthinker.server.research_lexical_coverage import requested_stage_scopes
    rows = [{'document_id': 'jade', 'section': '2 Preparation',
             'content': 'The pipeline first extracts permitted identifiers and then builds the tree.'},
            {'document_id': 'jade', 'section': '2.1 Online Answering',
             'content': 'The reader first retrieves authorized entries and then generates the response.'},
            {'document_id': 'jade', 'section': '2.2 Memory Maintenance',
             'content': 'The queue first evicts expired records and then stores a new entry.'}]
    assert bool(requested_stage_scopes(question, rows)) is expected


def test_noncoverage_role_projection_does_not_remove_real_limits():
    from docthinker.server.research_lexical_coverage import _role_strength
    meta = {'content': 'This section does not describe the answer phase.', 'section': '2 Methods'}
    assert _role_strength(meta, positive_evidence_request=True)['limitation'] == 0
    assert _role_strength(meta)['limitation'] > 0
    for text in ('The answer phase cannot complete without original records.',
                 'The answer phase was not tested on changed data.',
                 'The queue does not preserve evicted records.',
                 'The method did not improve accuracy.'):
        assert _role_strength({'content': text, 'section': '2 Methods'},
                              positive_evidence_request=True)['limitation'] > 0


def test_full_facet_budget_does_not_restore_document_name_relevance_bonus():
    from docthinker.server.research_evidence_selection import prepare_evidence_candidates
    from docthinker.server.research_lexical_coverage import matching_tokens
    q = 'Explain the BirchCache retrieval method.'
    plan = {'facets': [{'id': 'clause-' + str(i), 'text': q, 'kind': 'clause', 'weight': 1.,
                       'roles': ['mechanism'], 'document_ids': ['BirchCache']} for i in range(8)]}
    rows = [row('without-name', 0, 'The retrieval method selects matching entries.', doc='BirchCache'),
            row('with-name', 1, 'BirchCache retrieval method selects matching entries.', doc='BirchCache')]
    result = prepare_evidence_candidates(q, rows, [], plan, {}, matching_tokens=matching_tokens,
                                        literal_requested_roles={'mechanism'})
    scores = {item['source_id']: item['pool_score'] for item in result['candidates']}
    assert scores['with-name'] == scores['without-name']
    for source in result['candidates']:
        original = next(r for r in rows if r['id'] == source['source_id'])
        assert source['content'] == original['content'][source['excerpt_start']:source['excerpt_end']]


def test_ordinary_cache_metadata_word_is_not_subtracted_as_a_document_name():
    from docthinker.server.research_evidence_selection import prepare_evidence_candidates
    from docthinker.server.research_lexical_coverage import matching_tokens
    q = 'Explain how the cache method retrieves records.'
    plan = {'facets': [{'id': 'q', 'text': q, 'kind': 'original', 'weight': 1.,
                       'roles': ['mechanism']}]}
    rows = [row('without', 0, 'The method retrieves records.', doc='cache'),
            row('with', 1, 'The cache method retrieves records.', doc='cache')]
    result = prepare_evidence_candidates(q, rows, [], plan, {}, matching_tokens=matching_tokens,
                                        literal_requested_roles={'mechanism'})
    scores = {item['source_id']: item['pool_score'] for item in result['candidates']}
    assert scores['with'] > scores['without']


@pytest.mark.parametrize('text, section, expected', [
    ('The candidate with lower velum score is selected as the label. '
     'Selection accuracy counts how often that label matches the held-out target.', '4 Evaluation', True),
    ('The option with the highest contour score is chosen as the answer.', '3 Evaluation', True),
    ('我们将轮廓得分最低的候选片段选为预测标签，随后与留出的目标标签比较。', '3 评测', True),
    ('We plan to run an evaluation where the candidate with lower velum score is selected as the label.',
     '4 Evaluation', False),
    ('The candidate with lower velum score is selected as the label, but no evaluation criteria were reported.',
     '4 Evaluation', False),
    ('The candidate with lower velum score is selected as the label.', '2 Related Work', False),
    ('The inventory has the lowest shelf and the highest door. No answer-selection procedure was evaluated.',
     '3 Evaluation', False),
])
def test_score_comparison_and_choice_rule_is_measurement_not_a_metric_name(text, section, expected):
    assert ('measurement' in evaluation_cells({'content': text, 'section': section})) is expected


@pytest.mark.parametrize('phrase, question, expected', [
    ('task limits', 'Explain the task limits.', True),
    ('study conditions', 'Describe study conditions.', True),
    ('experiment setup', 'Describe experiment setup.', True),
    ('failure conditions', 'Explain failure conditions.', True),
    ('memory limits', 'Explain memory limits.', False),
    ('sparse attention', 'Explain sparse attention.', False),
    ('索引 API conditions', '解释索引 API conditions。', False),
    ('task limits', 'Explain the explicitly named "task limits" operation.', False),
    ('study conditions', '说明术语“study conditions”的定义。', False),
])
def test_role_only_query_views_do_not_downgrade_unknown_or_explicit_terms(phrase, question, expected):
    from docthinker.server.research_query_facets import is_role_only_phrase
    assert is_role_only_phrase(phrase, question) is expected


def test_generic_short_role_view_cannot_outrank_its_request_view():
    from docthinker.server.research_evidence_selection import prepare_evidence_candidates
    from docthinker.server.research_lexical_coverage import matching_tokens
    q = 'Explain LarchVector actual archive evaluation and the task limits.'
    plan = {'facets': [
        {'id': 'request', 'kind': 'original', 'text': q, 'weight': 1.,
         'roles': ['experiment_setup', 'limitation'], 'document_ids': ['LarchVector']},
        {'id': 'generic', 'kind': 'technical_phrase', 'text': 'task limits', 'weight': .9,
         'roles': ['limitation'], 'anchors': ['task limits']},
    ]}
    rows = [row('setup', 0, 'The task used the same model and fixed time limits for all participants.', doc='LarchVector')]
    pool = prepare_evidence_candidates(q, rows, [], plan, {}, matching_tokens=matching_tokens,
                                       literal_requested_roles={'experiment_setup', 'limitation'})
    assert pool['candidates']
    scores = pool['candidates'][0]['facet_scores']
    assert scores['generic'] <= scores['request']


@pytest.mark.parametrize('text, section, expected', [
    ('Trace congruence: Determines whether the response preserves the assigned object '
     '(labels: {0 : divergent, 0.4 : partial, 1 : aligned}).', '3.1 Scoring', True),
    ('Link fidelity: Assesses whether the output names the correct relation '
     '(ratings: {A : absent, B : incomplete, C : complete}).', '3.1 Scoring', True),
    ('路径一致性：评定答复是否保留指定对象（标签：零：不一致；中：部分一致；全：一致）。', '3.1 评分', True),
    ('We plan to assess response congruence with labels: {0 : divergent, 1 : aligned}.',
     '3 Evaluation', False),
    ('Trace congruence: Determines whether the response preserves the assigned object '
     '(labels: {0 : divergent, 1 : aligned}).', '2 Related Work', False),
    ('The inventory records labels: {0 : 40, 1 : 70, 2 : 90}.', '3 Evaluation', False),
    ('The business router determines destinations from labels: {0 : customer, 1 : supplier}.',
     '2 Configuration', False),
    ('The request evaluates a business code mapping: {0 : customer, 1 : supplier}.',
     '2 Configuration', False),
    ('Trace congruence: Determines whether the response preserves the assigned object. '
     'A separate catalogue has labels: {0 : divergent, 1 : aligned}.', '3 Evaluation', False),
])
def test_local_value_meaning_scale_is_not_just_a_digit_or_generic_scoring_word(text, section, expected):
    from docthinker.server.research_lexical_coverage import _evaluation_quality
    item = {'content': text, 'section': section}
    assert ('measurement' in evaluation_cells(item)) is expected
    if expected:
        assert _evaluation_quality(item, 'measurement') >= .75


def test_explicit_unknown_scale_competes_above_generic_rater_transition():
    intro = row('intro', 0, '3 Evaluation\nWe ask human annotators to score the retrieved records '
                'and responses from CedarFlow and IrisFlow.', doc='CedarFlow')
    detail = row('scale', 1, '3.1 Scoring\nTrace congruence: Determines whether the response '
                 'preserves the assigned object (labels: {0 : divergent, 0.4 : partial, 1 : aligned}).', doc='CedarFlow')
    rows = [intro, detail,
            row('setup', 2, '3.2 Study\nWe evaluated 31 held-out requests using the same model '
                'and output budget in both conditions.', doc='CedarFlow'),
            row('outcome', 3, '3.3 Findings\nResults show improved accuracy against the baseline '
                'only with immutable records; changed records were not tested.', doc='CedarFlow')]
    question = 'Explain CedarFlow actual evaluation, scoring criteria, result and task limits.'
    result = select_lexical_coverage(question, [intro], rows)
    selected = {item['id']: item for item in result['sources']}
    assert {'scale', 'setup', 'outcome'} <= selected.keys()
    assert detail['content'].split('\n', 1)[1] in selected['scale']['content']
    check_geometry(result, rows)


@pytest.mark.parametrize('question, expected', [
    ('Explain CedarFlow actual evaluation, scoring criteria, result and task limits.', True),
    ('请读 CedarFlow，说明论文给了哪些实际验证。', True),
    ('Explain the actual evaluation, scoring criteria and results.', False),
    ('Explain CedarFlow actual evaluation of latency.', False),
    ('Explain only CedarFlow latency evaluation.', False),
    ('Explain CedarFlow actual evaluation; focus on latency, not classification.', False),
    ('说明 CedarFlow 延迟测试的实际验证。', False),
    ('只说明 CedarFlow 延迟的验证情况。', False),
    ('The document says "CedarFlow actual evaluation". Explain latency.', False),
    ('Do not describe CedarFlow actual evaluation; quote the title.', False),
])
def test_evaluation_relevance_sharing_requires_explicit_broad_document_request(question, expected):
    from docthinker.server.research_lexical_coverage import evaluation_relevance_documents
    rows = [row('request-scope', 0, 'An unrelated catalogue.', doc='CedarFlow')]
    assert bool(evaluation_relevance_documents(question, rows)) is expected


def test_broad_evaluation_scope_does_not_expand_to_another_clauses_architecture_document():
    from docthinker.server.research_lexical_coverage import evaluation_relevance_documents
    rows = [row('cedar', 0, 'An archive.', doc='CedarFlow'),
            row('iris', 0, 'A reader.', doc='IrisFlow')]
    assert evaluation_relevance_documents(
        'Explain CedarFlow actual evaluation; describe IrisFlow architecture.', rows) == {'CedarFlow'}
    assert evaluation_relevance_documents(
        'Explain CedarFlow and IrisFlow actual evaluation.', rows) == {'CedarFlow', 'IrisFlow'}
    assert evaluation_relevance_documents(
        'Read CedarFlow and IrisFlow, explain actual evaluation.', rows) == {'CedarFlow', 'IrisFlow'}
