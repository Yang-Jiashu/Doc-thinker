"""Explicit operation needs must not be replaced by repeated storage detail."""
from copy import deepcopy

import pytest

from docthinker.server.research_retrieval import build_retrieval_plan
from docthinker.server.research_lexical_coverage import select_lexical_coverage


@pytest.mark.parametrize('question,expected', [
    ('For CedarFlow, explain ingestion only; for IrisLedger, explain retrieval only.',
     {'cedar': ('write',), 'iris': ('read',)}),
    ('对 CedarFlow 只说明写入；对 IrisLedger 只说明读取。',
     {'cedar': ('write',), 'iris': ('read',)}),
    ('Explain how CedarFlow and IrisLedger ingest records and retrieve them.',
     {'cedar': ('write', 'read'), 'iris': ('write', 'read')}),
    ('Explain CedarFlow ingestion; do not explain IrisLedger retrieval.',
     {'cedar': ('write',)}),
    ('For CedarFlow, explain ingestion; for UnknownCrate, explain retrieval.',
     {'cedar': ('write',)}),
    ('对 CedarFlow 只说明写入；对 UnknownCrate 只说明读取。',
     {'cedar': ('write',)}),
    ('Compare CedarFlow and IrisLedger. For each paper, explain ingestion and retrieval.',
     {'cedar': ('write', 'read'), 'iris': ('write', 'read')}),
    ('Compare CedarFlow and IrisLedger. For future use, explain how records are retrieved.',
     {'cedar': ('read',), 'iris': ('read',)}),
])
def test_operation_scope_uses_actual_named_material_in_each_positive_clause(question, expected):
    from docthinker.server.research_operation_stages import operation_stages_by_document
    chunks = [{'id': name, 'document_id': doc, 'filename': name + '_2601.12345.pdf',
               'ordinal': 0, 'page': 1, 'content': '2 Methods\nThe service stores records.'}
              for doc, name in [('cedar', 'CedarFlow'), ('iris', 'IrisLedger')]]
    assert operation_stages_by_document(question, chunks) == expected


def row(identifier, content, ordinal=0, doc='cedar'):
    return {'id': identifier, 'document_id': doc, 'filename': f'{doc}.pdf',
            'ordinal': ordinal, 'page': ordinal + 1, 'content': content}


@pytest.mark.parametrize('objects,condition', [
    ('subscriptions', 'the subscriber confirms the new plan'),
    ('inventory records', 'the warehouse acknowledges the correction'),
    ('archive revisions', 'the owner approves the replacement'),
])
def test_write_change_read_are_complementary_in_a_bounded_material_packet(objects, condition):
    chunks = [row('title', 'CedarFlow: A Bounded Record Processing System\nAlice Example\nAbstract\n'
                         'The system stores information for later use.'),
        row('write', f'2 Methods\n2.1 Ingestion\nThe service writes incoming {objects} into the journal. '
            'First the parser extracts fields. Then the indexer builds an index. Invalid inputs are rejected.', 1),
        row('change', f'2.2 Revisions\nThe service invalidates obsolete {objects} only after {condition}. '
            'The earlier version remains available for audit and cannot replace the current version.', 2),
        row('read', f'2.3 Retrieval\nThe worker retrieves current {objects} for the next request. '
            'First the encoder computes a query. Then the reader selects the matching record. Revoked records are excluded.', 3)]
    for index in range(4, 14):
        chunks.append(row(f'storage-{index}', f'2.4 Storage\nThe system contains tier {index} for stored records. '
            f'The buffer{index} stores indexed records. The queue{index} contains recent entries. '
            'First the encoder allocates slots. Then the worker caches entries. The capacity is fixed.', index))
    question = '请解释 CedarFlow 的方法：新记录到来后如何写入、处理变化，并在下次查询时用上？说明条件。'
    original = deepcopy(chunks)
    result = select_lexical_coverage(question, build_retrieval_plan(question, chunks)['sources'], chunks)
    selected = {s['id']: s for s in result['sources']}
    assert {'write', 'change', 'read'} <= set(selected)
    assert f'only after {condition}' in selected['change']['content']
    assert len(selected) <= 8 and result['trace']['candidate_count'] <= 48
    assert sum(len(s['content'].encode()) for s in selected.values()) <= 12000
    texts = {r['id']: r['content'] for r in chunks}
    assert all(texts[s['id']][s['excerpt_start']:s['excerpt_end']] == s['content'] for s in selected.values())
    assert chunks == original


@pytest.mark.parametrize('question,expected', [
    ('Explain the method: ingest new records, revise obsolete records, then retrieve them for the next request.', {'write', 'change', 'read'}),
    ('说明新信息写入、信息变更后如何使旧记录失效，并在下一次请求时读取。', {'write', 'change', 'read'}),
    ('Explain only how the archive stores records.', {'write'}),
    ('Explain only retrieval from the archive; do not discuss updates.', {'read'}),
    ('The document says "write, update, retrieve". Explain the measured accuracy only.', set()),
    ('Do not explain ingestion or revision. Compare accuracy scores only.', set()),
    ('Write a report about archive accuracy and revise the draft.', set()),
])
def test_requested_stages_are_explicit_user_operations_not_artifact_or_quoted_words(question, expected):
    from docthinker.server.research_operation_stages import requested_operation_stages
    assert set(requested_operation_stages(question)) == expected


@pytest.mark.parametrize('question', [
    'Write an evidence-backed report comparing two archival systems.',
    'Write a concise evidence-backed literature review about archive design.',
    'Compare these papers on graph-based retrieval and long-context limits.',
    'Propose one research hypothesis about retrieval and evidence placement.',
])
def test_writing_modifiers_and_retrieval_topics_are_not_operation_requests(question):
    from docthinker.server.research_operation_stages import requested_operation_stages
    assert requested_operation_stages(question) == ()


@pytest.mark.parametrize('text', [
    'Prompt template for the registry\nYou are a record manager.\n'
    'When the user changes {record}, the service updates the existing record.\n'
    'The worker retrieves {context} for later requests.',
    '系统提示模板\n你是档案管理助手。\n档案变化后，系统更新记录。\n输入：{record}',
])
def test_prompt_template_is_available_data_but_not_narrated_operation_evidence(text):
    from docthinker.server.research_operation_stages import source_operation_stages
    item = {'content': text, 'section': 'Appendix: Prompts', 'outline_roles': ['mechanism']}
    saved = deepcopy(item)
    assert source_operation_stages(item) == ()
    assert item == saved


def test_ordinary_method_mentions_of_prompt_templates_are_not_erased():
    from docthinker.server.research_operation_stages import source_operation_stages
    text = ('The service retrieves current records from the index only after permission checks. '
            'The resulting records are passed to the prompt template described in the appendix.')
    assert source_operation_stages({'content': text, 'section': '2 Methods'}) == ('read',)


@pytest.mark.parametrize('content,expected', [
    ('The service invalidates obsolete records only after approval.', {'change'}),
    ('The service cannot preserve the earlier revision after replacement.', {'change'}),
    ('We plan to update records in a future experiment.', set()),
    ('This section does not describe updates to stored records.', set()),
    ('Update operations are an important topic in historical literature.', set()),
])
def test_source_stage_is_visible_operation_not_a_future_promise_or_name(content, expected):
    from docthinker.server.research_operation_stages import source_operation_stages
    value = {'content': content, 'section': '2 Methods', 'outline_roles': ['mechanism']}
    assert set(source_operation_stages(value)) == expected


def test_unknown_or_revoked_material_does_not_acquire_a_stage():
    from docthinker.server.research_operation_stages import source_operation_stages
    assert not source_operation_stages({'content': 'Some processing happens here.', 'section': '2 Methods'})
    assert not source_operation_stages({'content': 'The service updates records.', 'status': 'revoked'})


@pytest.mark.parametrize('text', [
    'CedarFlow demonstrates substantial improvements in latency compared to the fixed baseline.',
    'IrisLedger achieved a 6.2 percent decrease in error relative to the baseline under the same conditions.',
    'The method showed no improvement over the baseline on held-out records.',
    'The system produced lower accuracy than the full-input baseline only under heavy load.',
])
def test_named_subject_reported_comparison_is_a_visible_outcome(text):
    from docthinker.server.research_lexical_coverage import evaluation_cells
    assert 'outcome' in evaluation_cells({'content': text, 'section': '3 Results'})


def test_reported_subject_variants_share_the_same_comparison_completeness_rule():
    from docthinker.server.research_lexical_coverage import _evaluation_quality
    first = {'content': 'We observed higher accuracy than the fixed baseline.', 'section': '3 Results'}
    named = {'content': 'CedarFlow achieved higher accuracy than the fixed baseline.', 'section': '3 Results'}
    assert _evaluation_quality(named, 'outcome') == _evaluation_quality(first, 'outcome') == .75


def test_operation_quality_prefers_local_lifecycle_conditions_without_borrowing_other_task_cues():
    from docthinker.server.research_operation_stages import source_operation_quality
    plain = 'The registry updates its display summary.'
    change = ('When a new subscription contradicts a previous valid record, the registry invalidates '
              'that previous record while preserving its earlier version for audit.')
    make = lambda text: {'content': text, 'section': '2 Methods'}
    assert source_operation_quality(make(change), 'change') > source_operation_quality(make(plain), 'change') > 0
    unrelated = plain + ' When another catalog changes, its old and new labels differ.'
    assert source_operation_quality(make(unrelated), 'change') == source_operation_quality(make(plain), 'change')


@pytest.mark.parametrize('text', [
    'We plan to invalidate old records after approval.',
    'System prompt: You are a registry. You update previous records when new records arrive.',
])
def test_operation_quality_is_zero_for_plans_and_embedded_instructions(text):
    from docthinker.server.research_operation_stages import source_operation_quality
    assert source_operation_quality({'content': text, 'section': '2 Methods'}, 'change') == 0


def test_action_local_assignment_explains_transition_without_interpreting_field_values():
    from docthinker.server.research_operation_stages import source_operation_quality
    plain = 'When a conflict is found, the registry invalidates the affected entry.'
    explicit = ('When a conflict is found, the registry invalidates the affected entry '
                'by setting expiry_flag to cutoff_value.')
    make = lambda text: {'content': text, 'section': '2 Methods'}
    assert source_operation_quality(make(explicit), 'change') > source_operation_quality(make(plain), 'change')


def test_assignment_from_another_action_does_not_complete_the_change_operation():
    from docthinker.server.research_operation_stages import source_operation_quality
    plain = 'The registry updates the display summary.'
    another = 'The registry updates the display summary while the decoder sets color to blue.'
    make = lambda text: {'content': text, 'section': '2 Methods'}
    assert source_operation_quality(make(another), 'change') == source_operation_quality(make(plain), 'change')


@pytest.mark.parametrize('connector', ['and', 'or', 'but not'])
def test_passive_state_labels_do_not_turn_coordination_into_a_write_object(connector):
    from docthinker.server.research_operation_stages import source_operation_stages
    text = f'The journal records when entries are created {connector} invalidated in the store.'
    assert 'write' not in source_operation_stages({'content': text, 'section': '2 Methods'})


def test_real_creation_with_a_visible_object_remains_an_operation():
    from docthinker.server.research_operation_stages import source_operation_stages
    assert 'write' in source_operation_stages({'content': 'The registry creates a record after validation.', 'section': '2 Methods'})


@pytest.mark.parametrize('verb', ['Explain', 'Describe'])
def test_explicit_comma_operation_list_retains_its_same_request(verb):
    from docthinker.server.research_operation_stages import requested_operation_stages, operation_stages_by_document
    question = f'{verb} CedarFlow ingestion, changes, and retrieval.'
    assert requested_operation_stages(question) == ('write', 'change', 'read')
    assert operation_stages_by_document(question, [row('title', 'CedarFlow methods.')]) == {'cedar': ('write', 'change', 'read')}


@pytest.mark.parametrize('question', [
    'Explain papers on ingestion, changes, and retrieval.',
    'Write a report about ingestion, changes, and retrieval.',
    'Explain CedarFlow ingestion. Retrieval is another topic.',
    'Explain CedarFlow ingestion, do not describe changes or retrieval.',
])
def test_operation_list_context_is_not_inherited_by_topics_negation_or_another_sentence(question):
    from docthinker.server.research_operation_stages import requested_operation_stages
    assert 'read' not in requested_operation_stages(question)


@pytest.mark.parametrize('question', [
    'Explain how CedarFlow ingests records; include the actual measurement and results.',
    'Describe CedarFlow ingestion, changes, and retrieval; compare accuracy scores.',
    '说明 CedarFlow 如何写入新记录；同时给出准确率指标。',
])
def test_explicit_operation_clause_is_not_cancelled_by_separate_measurement_clause(question):
    from docthinker.server.research_component_coverage import requests_measurement_without_structure
    from docthinker.server.research_query_facets import build_query_facets
    assert requests_measurement_without_structure(question) is False
    assert any('mechanism' in facet['roles'] for facet in build_query_facets(question)['facets'])


@pytest.mark.parametrize('question', [
    'Describe only CedarFlow retrieval accuracy, not its architecture.',
    'Explain only the latency of ingestion for CedarFlow.',
    'Report only the accuracy of new records.',
    'The paper says "explain how records are ingested"; report only accuracy scores.',
    'Compare the measurements; do not explain how CedarFlow ingests records.',
])
def test_metric_objects_quotes_and_explicit_exclusion_do_not_gain_operation_scope(question):
    from docthinker.server.research_component_coverage import requests_measurement_without_structure
    assert requests_measurement_without_structure(question) is True


@pytest.mark.parametrize('text,section', [
    ('CedarFlow will demonstrate improvements in latency compared to the baseline.', '3 Results'),
    ('Suppose CedarFlow achieved a latency improvement over the baseline.', '3 Results'),
    ('For example, CedarFlow achieved a latency improvement over the baseline.', '3 Results'),
    ('CedarFlow demonstrates substantial improvements in latency compared to the fixed baseline.', '2 Related Works'),
    ('CedarFlow achieved a high score. Another task uses a baseline.', '3 Results'),
])
def test_named_subject_shape_does_not_certify_plans_examples_or_borrow_comparators(text, section):
    from docthinker.server.research_lexical_coverage import evaluation_cells
    assert 'outcome' not in evaluation_cells({'content': text, 'section': section})


def test_single_read_stage_preserves_black_box_interface_and_training_preconditions():
    chunks = [row('title', 'CedarFlow: An External Record Reader\nAuthor\nAbstract\nWe describe retrieval.'),
        row('read', '2 Methods\nThe service retrieves current records from the index. '
            'First the reader computes a query. Then the encoder selects matching entries.', 1),
        row('external', '2.1 Interface\nExternal retrieval can use a separate encoder with a black-box generator API. '
            'The user must supply an index and access permission; hidden generator activations are not required.', 2),
        row('internal', '2.2 Internal Variant\nThe internal retrieval variant requires access to hidden model states '
            'and additional training. A black-box generator API cannot provide the required states.', 3)]
    chunks += [row(f'noise-{i}', '2.3 Retrieval\nThe worker retrieves indexed records for later use. '
        'The buffer stores records and the archive contains entries.', i) for i in range(4, 14)]
    question = 'Explain CedarFlow retrieval, including black-box API feasibility, training prerequisites and limitations.'
    result = select_lexical_coverage(question, build_retrieval_plan(question, chunks)['sources'], chunks)
    sources = {s['id']: s for s in result['sources']}
    assert {'read', 'external', 'internal'} <= set(sources)
    assert 'cannot provide the required states' in sources['internal']['content']
    assert 'hidden generator activations are not required' in sources['external']['content']


@pytest.mark.parametrize('question', ['Explain only the retrieval-score distribution.',
    '说明检索的准确率指标，不要说明内部流程。', 'Read the paper and report its accuracy.',
    'The document says "explain retrieval"; compare scores only.'])
def test_metric_or_document_reading_does_not_create_an_operation_mechanism_role(question):
    from docthinker.server.research_query_facets import build_query_facets
    assert not any('mechanism' in facet['roles'] for facet in build_query_facets(question)['facets'])


def test_supplement_recovers_abstract_role_from_reliable_initial_title_locator(monkeypatch):
    from docthinker.server import research_insight_supplement as module
    from docthinker.server.research_service import _supplement_insight_evidence
    header = 'CedarFlow: A Bounded Record Reader\nAuthor\ncontact@example.invalid\nAbstract\n'
    body = 'CedarFlow demonstrates higher accuracy compared to the baseline on held-out records.'
    raw = row('first', header + body)
    base = {**raw, 'label': 'S1', 'content': body, 'excerpt_start': len(header),
            'excerpt_end': len(header + body), 'original_content_length': len(header + body)}
    candidate = row('result', '3 Results\nCedarFlow achieved higher accuracy over the fixed baseline '
                    'only on held-out requests; concurrent updates were not tested.', 1)
    seen = []
    original = module._RequestProfiles.profile
    def profile(self, value):
        if value.get('content') == body:
            seen.append(value.get('section', ''))
        return original(self, value)
    monkeypatch.setattr(module._RequestProfiles, 'profile', profile)
    proposal = {'actions': [{'id': 'lookup', 'kind': 'supplemental_query',
        'payload': {'dimension_terms': ['results', 'evaluation']}}]}
    result = _supplement_insight_evidence('Explain CedarFlow actual evaluation results and limitations.',
                                         [base], [raw, candidate], proposal)
    assert seen and all('Abstract' in value for value in seen)
    assert result['sources'][0] == base
