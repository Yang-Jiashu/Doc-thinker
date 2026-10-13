"""Independent synthetic incremental-source acceptance, without any provider."""
from __future__ import annotations

from copy import deepcopy

import pytest

from docthinker.server.research_service import _supplement_insight_evidence


QUESTION = 'Compare the mechanisms and actual evaluation of AsterLatch retrieval, including conditions and results.'


def source(identifier, content, *, doc='aster', ordinal=0, label='S1', section=''):
    return {'id': identifier, 'document_id': doc, 'filename': doc + '.txt',
            'content': content, 'ordinal': ordinal, 'label': label, 'section': section,
            'excerpt_start': 0, 'excerpt_end': len(content),
            'original_content_length': len(content), 'truncated': False,
            'sentence_boundary_preserved': True}


def proposal(*dimensions):
    return {'actions': [{'id': 'bounded-lookup', 'kind': 'supplemental_query',
                         'payload': {'dimension_terms': list(dimensions or (
                             'conditions', 'results', 'evaluation', 'retrieval', 'method'))}}]}


def ids(result):
    return [row['id'] for row in result['sources']]


def test_neighbor_background_and_generic_dimensions_do_not_fill_a_missing_goal():
    base = source('base', 'AsterLatch retrieval stores verified records in a bounded archive.')
    background = source('background', 'AsterLatch retrieval is an important research topic. '
                        'Conditions and results are often discussed in the historical literature.', ordinal=1)
    result = _supplement_insight_evidence(QUESTION, [base], [base, background], proposal())
    assert ids(result) == ['base']
    assert result['receipt']['reason'] == 'no_incremental_evidence'


def test_equivalent_setup_in_another_chunk_is_not_an_incremental_goal():
    base = source('base', 'We evaluated AsterLatch retrieval on 24 held-out questions against a fixed baseline. '
                  'Both conditions used the same model and output budget.')
    duplicate = source('duplicate', 'The AsterLatch retrieval evaluation used the same model and output budget. '
                       'We evaluated 24 held-out questions against a fixed baseline.', ordinal=1)
    result = _supplement_insight_evidence(QUESTION, [base], [duplicate], proposal())
    assert ids(result) == ['base']


def test_missing_measurement_beats_repeated_background_before_the_two_source_cap():
    base = source('base', 'We evaluated AsterLatch retrieval on 24 held-out questions against a fixed baseline. '
                  'Results show no improvement under concurrent edits.')
    noise = [source(f'noise-{i}', 'AsterLatch retrieval conditions results evaluation method '
                    'are common topics in historical research.', ordinal=i + 1) for i in range(6)]
    measure = source('measure', 'For AsterLatch retrieval evaluation, annotators score correctness using verified targets. '
                     'Accuracy is defined as the fraction of correct answers; unanswered questions count as incorrect.', ordinal=8)
    result = _supplement_insight_evidence(QUESTION, [base], [*noise, measure], proposal())
    assert ids(result) == ['base', 'measure']
    assert result['receipt']['model_calls'] == 0


def test_unfinished_finding_cannot_saturate_outcome_before_complete_results():
    base = source('base', 'AsterLatch retrieval evaluation uses matched conditions. '
                  'Results show improved accuracy compared with the baseline on\n7')
    base['sentence_boundary_preserved'] = False
    complete = source('complete', 'Results show AsterLatch retrieval improves accuracy compared with the fixed baseline '
                      'only on the held-out archive; concurrent updates were not tested.', ordinal=1)
    result = _supplement_insight_evidence(QUESTION, [base], [complete], proposal())
    assert ids(result) == ['base', 'complete']
    assert result['sources'][0] == base


def test_url_footnote_does_not_complete_an_unfinished_outcome():
    from docthinker.server.research_insight_supplement import _profile

    row = source('tail', 'AsterLatch retrieval evaluation uses matched conditions. '
                 'Results show improved accuracy compared with the baseline on\n'
                 '1https://example.org/archive\n7')
    row['sentence_boundary_preserved'] = False
    profile = _profile(row, demand={'experiment_setup', 'result'}, structural=False,
                       topic={'asterlatch'}, explicit_background=False, explicit_planned=False)
    assert ('evaluation', 'outcome') not in profile['goals']


@pytest.mark.parametrize('separator', [
    '\n\n',
    ' A proposed experiment would measure a separate task. ',
    '\n4 Other Task\n',
])
def test_omitted_or_separate_task_cannot_lend_limits_to_an_outcome(separator):
    from docthinker.server.research_insight_supplement import _profile

    row = source('scope', 'Results show AsterLatch retrieval improves accuracy compared with the baseline.'
                 + separator + 'However, another task was only tested without concurrent edits.')
    profile = _profile(row, demand={'experiment_setup', 'result'}, structural=False,
                       topic={'asterlatch'}, explicit_background=False, explicit_planned=False)
    assert profile['goals']['evaluation', 'outcome'] == .75


def test_an_explicit_new_component_definition_is_not_covered_by_generic_retrieval():
    base = source('base', 'AsterLatch retrieval uses an archive index to access records. '
                  'The archive index stores stable lookup keys.')
    detail = source('detail', 'The bounded queue stores recent retrieved values for AsterLatch retrieval. '
                    'The queue has a fixed capacity and evicts the oldest values before appending new values.', ordinal=1)
    result = _supplement_insight_evidence(QUESTION, [base], [detail], proposal())
    assert ids(result) == ['base', 'detail']


def test_natural_function_request_reuses_literal_facet_mechanism_demand():
    question = '请读 AsterLatch，怎样把历史信息用进后续回答，论文给了哪些实际验证？'
    base = source('base', 'AsterLatch stores prior notes.')
    detail = source('detail', 'The archive index stores stable keys. '
                    'For AsterLatch, we first retrieve matching records, then generate the response using those records.', ordinal=1)
    result = _supplement_insight_evidence(question, [base], [detail], proposal())
    assert ids(result) == ['base', 'detail']


@pytest.mark.parametrize('question', [
    'Compare only the measured accuracy and latency of AsterLatch retrieval.',
    '比较 AsterLatch 的准确率和延迟，不要解释架构或机制。',
])
def test_numeric_or_explicitly_excluded_structure_does_not_nominate_a_new_component(question):
    base = source('base', 'AsterLatch retrieval accuracy was measured against the fixed baseline.')
    detail = source('detail', 'For AsterLatch retrieval, the archive index stores stable keys. '
                    'The bounded queue stores recent retrieved records.', ordinal=1)
    result = _supplement_insight_evidence(question, [base], [detail], proposal())
    assert ids(result) == ['base']


def test_new_measurement_handling_can_extend_an_already_present_cell():
    base = source('base', 'For AsterLatch retrieval evaluation, annotators score correctness on a three-level scale.')
    handling = source('handling', 'For AsterLatch retrieval evaluation, annotators score correctness on a three-level scale. '
                      'Unanswered requests and timeouts are scored as incorrect, not excluded from the denominator.', ordinal=1)
    result = _supplement_insight_evidence(QUESTION, [base], [handling], proposal())
    assert ids(result) == ['base', 'handling']


def test_second_slot_updates_coverage_instead_of_adding_duplicate_measurement():
    base = source('base', 'We evaluated AsterLatch retrieval on 24 held-out questions against a fixed baseline.')
    measurement = 'For AsterLatch retrieval evaluation, annotators score correctness on a three-level scale.'
    a = source('measure-a', measurement, ordinal=1)
    b = source('measure-b', measurement + ' This evaluation uses the same conditions.', ordinal=2)
    result = source('result', 'Results show AsterLatch retrieval has no improvement compared with the fixed baseline.', ordinal=3)
    selected = _supplement_insight_evidence(QUESTION, [base], [a, b, result], proposal())
    assert 'result' in ids(selected)
    assert len(set(ids(selected)) & {'measure-a', 'measure-b'}) == 1
    assert len(selected['sources']) == 3


def test_duplicate_text_from_another_id_does_not_consume_a_slot_or_receipt_gain():
    base = source('base', 'We evaluated AsterLatch retrieval on 24 held-out questions against a fixed baseline.')
    duplicate = source('copy', base['content'], doc='clone', ordinal=1)
    measurement = source('measurement', 'For AsterLatch retrieval evaluation, annotators score correctness on a three-level scale.', ordinal=2)
    result = _supplement_insight_evidence(QUESTION, [base], [duplicate, measurement], proposal())
    without_duplicate = _supplement_insight_evidence(QUESTION, [base], [measurement], proposal())
    assert ids(result) == ['base', 'measurement']
    assert result['receipt']['final_goal_cells'] == without_duplicate['receipt']['final_goal_cells']
    assert result['receipt']['rejection_counts']['duplicate_original_text'] >= 1


def test_new_source_projection_does_not_expose_internal_rank_or_query_fields():
    base = source('base', 'AsterLatch retrieval stores verified records.')
    detail = source('measurement', 'For AsterLatch retrieval evaluation, annotators score correctness on a three-level scale.')
    result = _supplement_insight_evidence(QUESTION, [base], [detail], proposal())
    assert ids(result) == ['base', 'measurement']
    assert set(result['sources'][1]) <= {
        'id', 'document_id', 'filename', 'content', 'ordinal', 'page', 'excerpt_start',
        'excerpt_end', 'offset_unit', 'original_content_length', 'sentence_boundary_preserved',
        'truncated', 'evidence_scope', 'label'}


def test_reported_explicit_multi_comparator_is_not_tied_with_a_generic_overview():
    from docthinker.server.research_insight_supplement import _profile

    settings = dict(demand={'result'}, structural=False, topic={'asterlatch'},
                    explicit_background=False, explicit_planned=False)
    detail = source('detail', 'AsterLatch outperforms both the fixed reader and the archive writer.', section='4 Results')
    overview = source('overview', 'Results show that AsterLatch is effective.', section='1 Introduction')
    planned = source('planned', 'A proposed experiment would show whether AsterLatch outperforms both the fixed reader and the archive writer.')
    assert _profile(detail, **settings)['goals']['evaluation', 'outcome'] >= .75
    assert _profile(overview, **settings)['goals']['evaluation', 'outcome'] <= .2
    assert ('evaluation', 'outcome') not in _profile(planned, **settings)['goals']


def test_an_extra_question_word_without_a_new_local_evidence_goal_does_not_add_a_source():
    question = 'Compare actual evaluation of AsterLatch retrieval for a private assistant.'
    base = source('base', 'For AsterLatch retrieval, annotators score correctness on a three-level scale.')
    repeated = source('repeated', base['content'] + ' This article concerns private assistants.', ordinal=1)
    result = _supplement_insight_evidence(question, [base], [repeated], proposal())
    assert ids(result) == ['base']


def test_an_explicit_new_measurement_subject_can_fill_another_requested_topic():
    question = 'Compare the actual accuracy and latency measurements for AsterLatch retrieval.'
    base = source('base', 'AsterLatch accuracy is defined as the fraction of correct answers. '
                  'Unanswered requests count as incorrect in the denominator.')
    latency = source('latency', 'AsterLatch latency is measured in milliseconds using a monotonic clock.', ordinal=1)
    result = _supplement_insight_evidence(question, [base], [latency], proposal())
    assert ids(result) == ['base', 'latency']


def test_explicit_background_request_may_keep_background_without_claiming_a_method_cell():
    question = 'Describe the historical background of AsterLatch retrieval and its inspiration from indexed archives.'
    base = source('base', 'AsterLatch retrieval is an archive assistant.')
    background = source('history', 'AsterLatch retrieval was inspired by indexed archives in earlier work. '
                        'These historical archives are the background of its design.', ordinal=1)
    result = _supplement_insight_evidence(question, [base], [background], proposal('background', 'historical'))
    assert ids(result) == ['base', 'history']
    assert result['receipt']['selection_is_semantic_verification'] is False


def test_third_party_planned_work_is_not_actual_result_but_explicit_request_can_retrieve_it():
    base = source('base', 'AsterLatch retrieval uses an archive index.')
    planned = source('planned', 'AsterLatch retrieval is discussed in prior work. '
                     'A proposed experiment would compare a fixed baseline and a new archive.', ordinal=1)
    rejected = _supplement_insight_evidence(QUESTION, [base], [planned], proposal())
    assert ids(rejected) == ['base']
    requested = _supplement_insight_evidence(
        'Describe the proposed experiment for AsterLatch retrieval discussed in prior work.',
        [base], [planned], proposal('proposed', 'experiment'))
    assert ids(requested) == ['base', 'planned']
    assert requested['receipt']['selection_is_semantic_verification'] is False


def test_scope_and_receipt_are_bounded_and_do_not_contain_source_prose():
    base = source('base', 'AsterLatch retrieval uses stable keys.')
    unrelated = source('unrelated', 'Medical conditions require fixed controls and dataset sizes.', doc='other')
    rows = [source(f'noise-{i}', 'Historical conditions and results are important in the research literature.',
                   ordinal=i + 1) for i in range(300)]
    snapshot = deepcopy([base, unrelated, *rows])
    result = _supplement_insight_evidence(QUESTION, [base], [unrelated, *rows], proposal())
    assert ids(result) == ['base']
    assert result['receipt']['scanned_chunks'] == 256
    assert result['receipt']['scan_limited'] is True
    assert result['receipt']['candidate_limit'] == 48
    assert result['receipt']['candidate_count'] <= 48
    assert result['receipt']['goal_cell_scope'] == 'retained_supplement_packet_before_prompt_fit'
    assert 'Historical conditions' not in str(result['receipt'])
    assert 'Medical conditions' not in str(result['receipt'])
    assert snapshot == [base, unrelated, *rows]


@pytest.mark.parametrize('initial_count', [1, 8, 9, 10])
def test_all_source_and_byte_budgets_preserve_original_geometry_and_sparse_labels(initial_count):
    base = [source(f'base-{i}', f'AsterLatch retrieval stores record {i}.', label=f'S{i + 1}')
            for i in range(initial_count)]
    extra = [source(f'extra-{i}', f'We evaluated AsterLatch retrieval on {i + 21} held-out questions '
                    'against a fixed baseline with the same model.', ordinal=i + 1, doc=f'doc-{i}')
             for i in range(4)]
    snapshot = deepcopy(base)
    result = _supplement_insight_evidence(QUESTION, base, extra, proposal())
    assert result['sources'][:initial_count] == snapshot
    assert len(result['sources']) <= min(initial_count + 2, 10)
    assert sum(len(row['content'].encode()) for row in result['sources'][initial_count:]) <= 2400
    assert sum(len(row['content'].encode()) for row in result['sources']) <= 12000
    for row in result['sources'][initial_count:]:
        original = next(item for item in extra if item['id'] == row['id'])
        assert original['content'][row['excerpt_start']:row['excerpt_end']] == row['content']


def test_one_request_reuses_profiles_across_nomination_selection_and_receipt(monkeypatch):
    from collections import Counter
    from docthinker.server import research_insight_supplement as module

    base = source('base', 'We evaluated AsterLatch retrieval on 24 held-out questions against a fixed baseline.')
    detail = source('detail', 'For AsterLatch retrieval evaluation, annotators score correctness on a three-level scale.')
    counts = Counter()
    original = module._profile

    def counted(row, **settings):
        counts[row['content'], row.get('section', ''), tuple(row.get('outline_roles', ()))] += 1
        return original(row, **settings)

    monkeypatch.setattr(module, '_profile', counted)
    result = _supplement_insight_evidence(QUESTION, [base], [base, detail], proposal())
    assert ids(result) == ['base', 'detail']
    assert counts and max(counts.values()) == 1
    first = dict(counts)
    repeated = _supplement_insight_evidence(QUESTION, [base], [base, detail], proposal())
    assert repeated == result
    assert counts == {key: value * 2 for key, value in first.items()}  # No cross-request memo.


@pytest.mark.parametrize('section,outline', [('', ()), ('1 Introduction', ()),
    ('4 Evaluation', ('experiment_setup',)), ('2 Method', ('mechanism',)),
    ('Background', ('mechanism',))])
def test_request_memo_preserves_profiles_and_source_boundaries(section, outline):
    from docthinker.server.research_insight_supplement import _RequestProfiles, _profile

    settings = dict(demand={'mechanism', 'experiment_setup', 'result', 'limitation'},
                    structural=True, topic={'asterlatch', 'accuracy'},
                    explicit_background=False, explicit_planned=False)
    rows = [
        source('a', 'AsterLatch accuracy is defined as the fraction of correct answers. '
               'Unanswered questions count as incorrect.'),
        source('b', 'Results show AsterLatch improves accuracy compared with the baseline.\n\n'
               'Another task was only tested without concurrent edits.'),
        source('c', 'Results show AsterLatch improves accuracy on\n1https://example.org/archive\n7'),
        source('d', 'The archive index stores stable keys. A proposed experiment would compare two models. '
               'For AsterLatch, first retrieve matching records, then generate the response from those records.'),
    ]
    memo = _RequestProfiles(settings)
    for row in rows:
        row.update(section=section, outline_roles=outline)
        snapshot = deepcopy(row)
        assert memo.profile(row) == _profile(row, **settings)
        assert memo.profile(row) == _profile(row, **settings)
        assert row == snapshot


def test_request_profile_memo_has_a_fixed_size_and_distinguishes_outline_scope():
    from docthinker.server.research_insight_supplement import _RequestProfiles, MAX_PROFILE_MEMO

    settings = dict(demand={'mechanism'}, structural=True, topic={'asterlatch'},
                    explicit_background=False, explicit_planned=False)
    memo = _RequestProfiles(settings)
    text = 'AsterLatch retrieves records and selects matching entries.'
    method = source('a', text)
    method['outline_roles'] = ('mechanism',)
    evaluation = {**method, 'outline_roles': ('experiment_setup',)}
    assert memo.profile(method) != memo.profile(evaluation)
    for index in range(MAX_PROFILE_MEMO + 5):
        memo.profile(source(str(index), f'AsterLatch record {index} is stored.'))
    assert memo._profile_cached.cache_info().currsize == MAX_PROFILE_MEMO


def test_missing_measurement_precedes_more_components_and_conditions():
    base = source('base', 'We evaluated AsterLatch retrieval on 24 held-out questions against a fixed baseline. '
                  'Results show no improvement compared with the baseline only under concurrent edits.')
    rows = [source('method-a', 'The archive index stores stable keys for AsterLatch retrieval. '
                   'The method uses the same model for every request.', ordinal=1),
            source('method-b', 'The bounded queue stores recent values for AsterLatch retrieval. '
                   'It uses the same output budget for every request.', ordinal=2),
            source('measurement', 'For AsterLatch retrieval evaluation, annotators score correctness on a three-level scale.', ordinal=3)]
    result = _supplement_insight_evidence(QUESTION, [base], rows, proposal())
    assert 'measurement' in ids(result)
    assert result['sources'][0] == base
    assert result['receipt']['required_coarse_goal_priority'] is True
    assert result['receipt']['coverage_is_complete'] is False


def test_missing_operation_precedes_more_evaluation_and_conditions():
    base = source('base', 'We evaluated AsterLatch retrieval on 24 held-out questions against a fixed baseline. '
                  'Annotators score correctness. Results show that accuracy increases.')
    rows = [source('measurement', 'AsterLatch correctness is measured in points using a three-level scale. '
                   'Unanswered requests count as incorrect; both conditions use the same model.', ordinal=1),
            source('outcome', 'Results show AsterLatch accuracy increases compared with the baseline only under immutable records. '
                   'Both conditions use the same output budget.', ordinal=2),
            source('method', 'For AsterLatch, first retrieve matching records from the archive, '
                   'then generate the response using those records.', ordinal=3)]
    result = _supplement_insight_evidence(QUESTION, [base], rows, proposal())
    assert 'method' in ids(result)
    assert len(result['sources']) == 3


def test_existing_required_cell_quality_precedes_extra_method_subtypes():
    base = source('base', 'We evaluated AsterLatch retrieval on 24 held-out questions against a fixed baseline. '
                  'For AsterLatch, first retrieve archived records, then generate the response using those records. '
                  'Annotators score correctness. Results show no improvement compared with the baseline only under concurrent edits. '
                  'Both conditions use the same model and output budget.')
    rows = [source('method-a', 'The archive index stores stable keys for AsterLatch retrieval.', ordinal=1),
            source('method-b', 'The bounded queue stores recent values for AsterLatch retrieval.', ordinal=2),
            source('measurement', 'For AsterLatch retrieval evaluation, accuracy is defined as the fraction of correct answers. '
                   'Unanswered requests count as incorrect.', ordinal=3)]
    result = _supplement_insight_evidence(QUESTION, [base], rows, proposal())
    assert 'measurement' in ids(result)


def test_comparison_and_schema_dimensions_do_not_require_unasked_evaluation():
    question = 'Compare the mechanisms of AsterLatch retrieval.'
    base = source('base', 'For AsterLatch, first retrieve archived records, then generate the response using those records.')
    evaluation = source('evaluation', 'We evaluated AsterLatch retrieval on 24 held-out questions against a fixed baseline. '
                        'Accuracy is defined as the fraction of correct answers. Results show improved accuracy.')
    result = _supplement_insight_evidence(question, [base], [evaluation], proposal('evaluation', 'results'))
    assert ids(result) == ['base']


def test_narrow_metric_requirement_does_not_borrow_a_different_metric_in_the_same_document():
    question = 'Describe only the latency measurements for AsterLatch.'
    base = source('base', 'AsterLatch has a retrieved archive.')
    wrong = source('accuracy', 'AsterLatch accuracy is defined as the fraction of correct answers. '
                   'Unanswered requests count as incorrect. Latency is discussed elsewhere.', ordinal=1)
    right = source('latency', 'AsterLatch latency is measured in milliseconds using a wall clock.', ordinal=2)
    result = _supplement_insight_evidence(question, [base], [wrong, right], proposal())
    assert ids(result) == ['base', 'latency']


def test_a_satisfied_requested_measurement_does_not_fill_spare_slots():
    question = 'Explain the measurement of AsterLatch latency.'
    base = source('base', 'AsterLatch latency is measured in milliseconds using a monotonic clock.')
    repeated = source('repeated', 'For AsterLatch, latency is measured in milliseconds using a monotonic clock.', ordinal=1)
    other = source('other', 'AsterLatch accuracy is defined as the fraction of correct answers.', ordinal=2)
    result = _supplement_insight_evidence(question, [base], [repeated, other], proposal())
    assert ids(result) == ['base']


def test_positive_remaining_gain_is_reported_as_addition_limit_not_no_gain():
    question = 'Explain the mechanisms of AsterLatch retrieval.'
    base = source('base', 'AsterLatch has archived records.')
    rows = [source(f'detail-{index}', f'The {name} stores stable records for AsterLatch retrieval.', ordinal=index + 1)
            for index, name in enumerate(('archive index', 'bounded queue', 'result cache'))]
    result = _supplement_insight_evidence(question, [base], rows, proposal())
    assert len(result['sources']) == 3
    assert result['receipt']['rejection_counts'].get('addition_limit') == 1
    assert not result['receipt']['rejection_counts'].get('no_remaining_incremental_goal')


def test_only_setup_does_not_create_measurement_or_outcome_obligations():
    question = 'Describe only the experimental setup for AsterLatch.'
    base = source('base', 'We evaluated AsterLatch on 24 held-out questions with the same model and output budget.')
    rows = [source('metric', 'AsterLatch accuracy is defined as the fraction of correct answers.', ordinal=1),
            source('outcome', 'Results show AsterLatch accuracy improves compared with the baseline.', ordinal=2)]
    result = _supplement_insight_evidence(question, [base], rows, proposal())
    assert ids(result) == ['base']


def test_an_unrecognized_non_evidence_request_does_not_gain_schema_requirements():
    base = source('base', 'AsterLatch is the uploaded title.')
    detail = source('detail', 'AsterLatch latency is measured in milliseconds using a monotonic clock.')
    result = _supplement_insight_evidence('Give the title of AsterLatch.', [base], [detail], proposal())
    assert ids(result) == ['base']
    assert result['receipt']['required_coarse_goal_priority'] is False


def test_narrow_metric_quality_cannot_borrow_a_neighboring_other_metric_protocol():
    from docthinker.server.research_insight_supplement import _profile

    row = source('mixed', 'AsterLatch latency is measured in milliseconds using a wall clock. '
                 'Accuracy is defined as the fraction of correct answers; unanswered items count as incorrect.')
    profile = _profile(row, demand={'experiment_setup'}, structural=False, topic={'latency'},
                       explicit_background=False, explicit_planned=False,
                       evaluation_requested={'measurement'}, evaluation_subjects=(frozenset({'latency'}),))
    assert profile['goals']['evaluation', 'measurement'] == .75
    assert ('condition', 'failure_handling') not in profile['goals']


def test_full_source_slots_are_not_reported_as_exhausted_quote_bytes():
    base = [source(str(i), f'AsterLatch record {i}.', label=f'S{i + 1}') for i in range(10)]
    result = _supplement_insight_evidence(QUESTION, base, [], proposal())
    assert result['sources'] == base
    assert result['receipt']['reason'] == 'addition_limit'


def test_remaining_useful_quote_over_the_byte_budget_is_reported_honestly():
    base = [source(str(i), 'x' * 1499, label=f'S{i + 1}') for i in range(8)]
    detail = source('detail', 'For AsterLatch evaluation, annotators score correctness on a three-level scale.')
    result = _supplement_insight_evidence(QUESTION, base, [detail], proposal())
    assert result['sources'] == base
    assert result['receipt']['rejection_counts'].get('evidence_budget') == 1
    assert not result['receipt']['rejection_counts'].get('no_remaining_incremental_goal')


@pytest.mark.parametrize('question', [
    'Explain AsterLatch mechanisms, but do not discuss experiments.',
    '解释 AsterLatch 的机制，不要展开评测。',
])
def test_explicitly_excluded_evaluation_does_not_become_a_required_gap(question):
    base = source('base', 'For AsterLatch, first retrieve matching records, then generate the response from those records.')
    detail = source('detail', 'We evaluated AsterLatch on 24 held-out questions against a fixed baseline. '
                    'Accuracy is defined as the fraction of correct answers. Results show an improvement.')
    result = _supplement_insight_evidence(question, [base], [detail], proposal())
    assert ids(result) == ['base']


@pytest.mark.parametrize('question', [
    'Describe only the latency measurements for AsterLatch, do not discuss accuracy.',
    'Describe only the evaluation of latency for AsterLatch, do not discuss accuracy.',
])
def test_real_facets_do_not_release_an_explicit_metric_scope_on_evaluation_word(question):
    from docthinker.server.research_query_facets import build_query_facets

    base = source('base', 'AsterLatch stores prior notes.')
    wrong = source('accuracy', 'AsterLatch accuracy is defined as the fraction of correct answers. '
                   'Unanswered requests count as incorrect. Latency is discussed elsewhere.', ordinal=1)
    right = source('latency', 'AsterLatch latency is measured in milliseconds using a wall clock.', ordinal=2)
    rows = [base, wrong, right]
    for row in rows:
        row['filename'] = 'AsterLatch_2504.12345.pdf'
    facets = build_query_facets(question, rows)
    assert {doc for item in facets['facets'] for doc in item.get('document_ids', ())} == {'aster'}
    if 'evaluation' in question:
        assert 'experiment_setup' in {role for item in facets['facets'] for role in item.get('roles', ())}
    result = _supplement_insight_evidence(question, [base], rows, proposal())
    assert ids(result) == ['base', 'latency']
    assert result['receipt']['literal_metric_scope_status'] == 'limited_unverified'


@pytest.mark.parametrize('wanted,excluded', [('延迟', '准确率'), ('准确率', '延迟')])
def test_real_chinese_facets_remove_negative_metric_from_positive_scope(wanted, excluded):
    from docthinker.server.research_query_facets import build_query_facets

    question = f'只解释 AsterLatch 的{wanted}评测，不讨论{excluded}。'
    base = source('base', 'AsterLatch 用于存储用户笔记。')
    rows = [source('准确率', '准确率指标通过核对标准答案进行评分，未回答样本计入错误。延迟会在其他章节讨论。', ordinal=1),
            source('延迟', '延迟指标通过单调时钟计算，以毫秒为单位记录。', ordinal=2)]
    for row in [base, *rows]:
        row['filename'] = 'AsterLatch_2504.12345.pdf'
    facets = build_query_facets(question, [base, *rows])
    assert 'experiment_setup' in {role for item in facets['facets'] for role in item.get('roles', ())}
    result = _supplement_insight_evidence(question, [base], rows, proposal())
    assert ids(result) == ['base', wanted]
    assert result['receipt']['literal_metric_scope_status'] == 'limited_unverified'


def test_a_literal_metric_scope_does_not_claim_an_implicit_translation_match():
    base = source('base', 'AsterLatch stores prior notes.')
    rows = [source('accuracy', 'AsterLatch accuracy is defined as the fraction of correct answers.', ordinal=1),
            source('latency', 'AsterLatch latency is measured in milliseconds using a monotonic clock.', ordinal=2)]
    for row in [base, *rows]:
        row['filename'] = 'AsterLatch_2504.12345.pdf'
    result = _supplement_insight_evidence('只解释 AsterLatch 的延迟评测，不讨论准确率。', [base], rows, proposal())
    assert ids(result) == ['base']
    assert result['receipt']['literal_metric_scope_status'] == 'limited_unverified'
    assert result['receipt']['coverage_is_complete'] is False


@pytest.mark.parametrize('separator', ['', ' ', '\t', '\n'])
def test_mixed_language_metric_exclusion_keeps_visible_whitespace_boundaries(separator):
    from docthinker.server.research_query_facets import build_query_facets

    question = f'仅说明 CedarFlow 的 latency 评测，不要讨论{separator}accuracy。'
    base = source('base', 'CedarFlow stores prior notes.', doc='cedar')
    wrong = source('accuracy', 'CedarFlow accuracy is defined as the fraction of correct answers. '
                   'Unanswered requests count as incorrect. Latency is discussed elsewhere.', doc='cedar', ordinal=1)
    right = source('latency', 'CedarFlow latency is measured in milliseconds using a monotonic clock.', doc='cedar', ordinal=2)
    rows = [base, wrong, right]
    for row in rows:
        row['filename'] = 'CedarFlow_2504.12345.pdf'
    facets = build_query_facets(question, rows)
    assert 'experiment_setup' in {role for item in facets['facets'] for role in item.get('roles', ())}
    snapshot = deepcopy(rows)
    result = _supplement_insight_evidence(question, [base], rows, proposal())
    assert ids(result) == ['base', 'latency']
    assert result['receipt']['literal_metric_scope_status'] == 'limited_unverified'
    assert rows == snapshot


def operation_source(identifier, content, *, doc='cedar', ordinal=0, label='S1'):
    row = source(identifier, content, doc=doc, ordinal=ordinal, label=label)
    row['filename'] = {'cedar': 'CedarCrate_2601.12345.pdf',
                       'iris': 'IrisLedger_2602.23456.pdf'}.get(doc, doc + '.txt')
    return row


@pytest.mark.parametrize('objects', ['inventory entries', 'subscription records'])
def test_local_operation_quality_adds_real_detail_to_ordinary_structural_request(objects):
    question = 'Explain the mechanism of CedarCrate.'
    base = operation_source('base', f'The service updates {objects}.')
    detail = operation_source('detail', f'When an approved correction arrives, the service invalidates '
                              f'obsolete {objects} by setting their status to superseded.', ordinal=1)
    result = _supplement_insight_evidence(question, [base], [base, detail], proposal())
    assert ids(result) == ['base', 'detail']
    assert result['receipt']['requested_operation_cells'] == 0
    assert result['receipt']['operation_quality_is_semantic_support'] is False
    assert result['receipt']['algorithm'] == 'source_local_incremental_supplement_v3'
    assert result['receipt']['coverage_is_complete'] is False
    assert result['sources'][0] == base


@pytest.mark.parametrize('missing,base_action,detail_action', [
    ('changes', 'The service writes incoming inventory entries. The worker retrieves current entries.',
     'The service updates obsolete inventory entries.'),
    ('retrieval', 'The service writes incoming subscription records. The service updates obsolete records.',
     'The worker retrieves current subscription records.'),
])
def test_requested_missing_stage_wins_the_last_slot_over_more_storage_detail(missing, base_action, detail_action):
    question = 'Explain CedarCrate ingestion, changes, and retrieval.'
    base = [operation_source('base', 'The system consists of an index and an archive. ' + base_action)]
    base.extend(operation_source(f'fixed-{i}', f'Unrelated retained citation {i}.', label=f'S{i + 1}')
                for i in range(1, 9))
    noise = [operation_source(f'noise-{i}', f'The {name} stores stable records for CedarCrate.', ordinal=i + 1)
             for i, name in enumerate(('archive index', 'bounded queue', 'result cache'))]
    needed = operation_source('needed', detail_action, ordinal=5)
    saved = deepcopy(base)
    result = _supplement_insight_evidence(question, base, [*base, *noise, needed], proposal())
    assert ids(result) == [*ids({'sources': base}), 'needed'], missing
    assert result['sources'][:9] == saved
    assert result['receipt']['requested_operation_cells'] == 3
    assert result['receipt']['coverage_is_complete'] is False


def test_explicit_stage_requirements_remain_document_bound():
    question = 'For CedarCrate, explain ingestion only; for IrisLedger, explain retrieval only.'
    base = [operation_source('cedar-base', 'The service writes inventory entries.'),
            operation_source('iris-base', 'IrisLedger is a subscription archive.', doc='iris', label='S2')]
    wrong = operation_source('wrong', 'When access is authorized, the worker retrieves current inventory '
                             'entries while preserving their earlier versions.', ordinal=1)
    right = operation_source('right', 'The worker retrieves current subscription records.', doc='iris', ordinal=1)
    result = _supplement_insight_evidence(question, base, [*base, wrong, right], proposal())
    assert ids(result) == ['cedar-base', 'iris-base', 'right']
    assert result['receipt']['requested_operation_cells'] == 2


@pytest.mark.parametrize('question', [
    'For UnknownCrate, explain revisions; for CedarCrate, give the title.',
    'Quote exactly "CedarCrate updates records".',
    'Explain CedarCrate ingestion; do not explain retrieval.',
])
def test_unresolved_literal_or_negated_stage_is_not_transferred_to_available_material(question):
    base = operation_source('base', 'The service writes inventory entries.')
    unasked = operation_source('unasked', 'When access is authorized, the worker retrieves inventory entries '
                               'while preserving their earlier versions.', ordinal=1)
    result = _supplement_insight_evidence(question, [base], [base, unasked], proposal())
    assert ids(result) == ['base']


@pytest.mark.parametrize('text,section,status', [
    ('Prompt template\nYou are an inventory manager.\nWhen corrections arrive, the service updates records. '
     'The worker retrieves records.', 'Appendix: Prompt template', ''),
    ('The service will update subscription records after approval.', '2 Methods', ''),
    ('The service updates subscription records after approval.', '2 Methods', 'revoked'),
    ('The service updates subscription records after approval.', '2 Methods', 'deleted'),
    ('This section does not describe updates to subscription records.', '2 Methods', ''),
])
def test_nonobserved_or_withdrawn_text_cannot_fill_operation_quality(text, section, status):
    from docthinker.server.research_insight_supplement import _RequestProfiles

    settings = dict(demand={'mechanism'}, structural=True, topic={'subscription'},
                    explicit_background=False, explicit_planned=False)
    row = source('candidate', text, section=section)
    if status:
        row['status'] = status
    result = _RequestProfiles(settings).profile(row)
    assert not any(family == 'mechanism' and key.startswith('stage:') for family, key in result['goals'])
    if status:
        assert result['goals'] == {}


@pytest.mark.parametrize('separator', [' ', '\n\n', ' We plan to study a different task. '])
def test_operation_quality_never_borrows_another_sentence_or_removed_body_condition(separator):
    from docthinker.server.research_insight_supplement import _profile

    row = source('candidate', 'The service updates inventory entries.' + separator +
                 'Another task starts only after approval and preserves previous versions.', section='2 Methods')
    profile = _profile(row, demand={'mechanism'}, structural=True, topic={'inventory'},
                       explicit_background=False, explicit_planned=False)
    assert profile['goals']['mechanism', 'stage:change'] == .5


def test_operation_quality_uses_the_request_local_bounded_memo(monkeypatch):
    from collections import Counter
    from docthinker.server import research_insight_supplement as module

    counts = Counter()
    original = module.source_operation_qualities

    def counted(row):
        counts[row['content'], row.get('section', ''), tuple(row.get('outline_roles', ()))] += 1
        return original(row)

    monkeypatch.setattr(module, 'source_operation_qualities', counted)
    settings = dict(demand={'mechanism'}, structural=True, topic={'inventory'},
                    explicit_background=False, explicit_planned=False)
    memo = module._RequestProfiles(settings)
    row = source('a', 'The service updates inventory entries.', section='2 Methods')
    assert memo.operations(row) == {'change': .5}
    assert memo.profile(row)['goals']['mechanism', 'stage:change'] == .5
    assert memo.operations(row) == {'change': .5}
    assert max(counts.values()) == 1
    assert memo.operations({**row, 'status': 'revoked'}) == {}
    assert memo.profile({**row, 'status': 'revoked'})['goals'] == {}
    for i in range(module.MAX_PROFILE_MEMO + 2):
        memo.operations(source(str(i), f'The service updates inventory item{i}.'))
    assert memo._operations_cached.cache_info().currsize == module.MAX_PROFILE_MEMO == 512


def test_three_operation_subcells_do_not_gain_three_family_votes_against_missing_measurement():
    question = ('Explain how CedarCrate ingests records, changes obsolete records, and retrieves current records; '
                'include the actual experiment, measurement, and results.')
    base = [operation_source('base', 'The system consists of an index and an archive. '
            'We evaluated CedarCrate on 24 held-out questions using the same model. '
            'Results show no improvement compared with the baseline only under concurrent edits.')]
    base.extend(operation_source(f'fixed-{i}', f'Unrelated retained citation {i}.', label=f'S{i + 1}')
                for i in range(1, 9))
    operations = operation_source('all-stages', 'When a delivery arrives, the service writes inventory entries. '
        'When a correction arrives, the service invalidates obsolete records by setting their status to superseded. '
        'The worker retrieves current records only after permission checks.', ordinal=1)
    measurement = operation_source('measurement', 'For CedarCrate evaluation, accuracy is defined as the '
        'fraction of correct answers. Unanswered requests count as incorrect.', ordinal=2)
    result = _supplement_insight_evidence(question, base, [*base, operations, measurement], proposal())
    assert ids(result)[-1] == 'measurement'
    assert len(result['sources']) == 10
    assert result['receipt']['requested_operation_cells'] == 3


def test_ordinary_retrieval_topic_does_not_force_any_operation_requirements():
    question = 'Write an evidence-backed report about CedarCrate retrieval and storage costs.'
    base = operation_source('base', 'The service updates inventory entries.')
    detail = operation_source('detail', 'When a correction arrives, the service invalidates obsolete inventory '
        'entries by setting their status to superseded.', ordinal=1)
    result = _supplement_insight_evidence(question, [base], [base, detail], proposal())
    assert result['receipt']['requested_operation_cells'] == 0
    assert result['receipt']['coverage_is_complete'] is False


def test_same_operation_quality_is_not_added_again_under_a_new_source_id():
    question = 'Explain how CedarCrate updates subscription records.'
    text = 'When an approved correction arrives, the service invalidates obsolete subscription records '
    text += 'by setting their status to superseded.'
    base = operation_source('base', text)
    repeated = operation_source('repeated', text + ' The archive is discussed in this document.', ordinal=1)
    result = _supplement_insight_evidence(question, [base], [base, repeated], proposal())
    assert ids(result) == ['base']


@pytest.mark.parametrize('question', [
    'Explain only the latency of ingestion for CedarCrate.',
    'Describe only CedarCrate retrieval accuracy, not its architecture.',
])
def test_a_metric_about_an_operation_does_not_override_the_measurement_only_guard(question):
    base = operation_source('base', 'CedarCrate has a measured archive.')
    detail = operation_source('detail', 'When a delivery arrives, the service writes current inventory records. '
        'The worker retrieves current records only after permission checks.', ordinal=1)
    result = _supplement_insight_evidence(question, [base], [base, detail], proposal())
    assert ids(result) == ['base']
    assert result['receipt']['requested_operation_cells'] == 0


@pytest.mark.parametrize('question,required_cells', [
    ('Explain the mechanism of CedarCrate.', 0),
    ('Explain how CedarCrate updates inventory records.', 1),
])
def test_real_bounded_candidate_nomination_receives_local_operation_quality(monkeypatch, question, required_cells):
    from docthinker.server import research_insight_supplement as module

    base = operation_source('base', 'The system consists of an index and an archive.')
    weak = [operation_source(f'weak-{i}', f'The CedarCrate service updates inventory records in archive{i}.',
                             ordinal=i + 1) for i in range(60)]
    detailed = operation_source('detailed', 'When an approved correction arrives, the service invalidates '
        'obsolete inventory records by setting their status to superseded.', ordinal=61)
    rows = [base, *weak, detailed]
    original = module.prepare_evidence_candidates
    captured = {}

    def observe(*args, **kwargs):
        # Run the real window/48-candidate machinery, not a fake result pool.
        captured['weak'] = kwargs['operation_profile'](weak[0])
        captured['detailed'] = kwargs['operation_profile'](detailed)
        pool = original(*args, **kwargs)
        captured['pool'] = pool
        return pool

    monkeypatch.setattr(module, 'prepare_evidence_candidates', observe)
    snapshot = deepcopy(rows)
    result = _supplement_insight_evidence(question, [base], rows, proposal())
    assert captured['weak'] == {'change': .5}
    assert captured['detailed'] == {'change': 1.}
    pool = captured['pool']['candidates']
    assert len(pool) == 48
    detail = next(row for row in pool if row['source_id'] == 'detailed')
    assert detail['_operation_quality'] == {'change': 1.}
    assert any(row['_operation_quality'] == {'change': .5} for row in pool)
    assert ids(result) == ['base', 'detailed']
    assert result['receipt']['requested_operation_cells'] == required_cells
    assert result['receipt']['model_calls'] == 0
    assert result['receipt']['coverage_is_complete'] is False
    assert result['sources'][0] == base
    assert rows == snapshot
