"""Independent operation-stage acceptance, fixed before V13 implementation.

Stages are surface locators, never facts, validation, or external authority.
Synthetic subscriptions, stock records and archives do not encode paper IDs.
"""
from copy import deepcopy
import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Operation coverage must not make a network/model request')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)


@pytest.mark.parametrize('question,expected', [
    ('Explain how to write incoming subscription records, revise stale records, '
     'and retrieve the current records for subsequent requests.', {'write', 'change', 'read'}),
    ('请说明新档案如何写入、旧档案如何修订，以及下次查询时如何读取。', {'write', 'change', 'read'}),
    ('Describe only how new stock records are inserted.', {'write'}),
    ('只解释如何读取现有档案，不要讨论新增或修订。', {'read'}),
    ('Explain how to invalidate obsolete subscriptions; do not describe insertion or retrieval.', {'change'}),
])
def test_only_explicit_positive_user_operations_become_stage_requests(question, expected):
    from docthinker.server.research_operation_stages import requested_operation_stages
    found = requested_operation_stages(question)
    assert set(found) == expected
    assert len(found) == len(set(found)) <= 3


@pytest.mark.parametrize('question', [
    'Write a report about the archive.',
    'The team writes, revises and reads stock records. Count the verbs in this sentence.',
    'Translate: "Insert incoming records, revise obsolete entries and retrieve them later."',
    '“新增记录、修改旧记录、下次读取”这几个字分别是什么意思？',
    '> Insert incoming records and invalidate previous versions.\nCount the quoted words.',
    '不需要说明怎样新增、修订或读取档案，只报文件大小。',
    'What is a subscription?',
])
def test_mentions_quotes_negation_and_writing_artifact_do_not_expand_operations(question):
    from docthinker.server.research_operation_stages import requested_operation_stages
    assert requested_operation_stages(question) == ()


@pytest.mark.parametrize('text,expected', [
    ('When a subscription arrives, the registry inserts a new record into the archive.', {'write'}),
    ('On receipt of a stock event, the service writes the incoming record to the journal.', {'write'}),
    ('When an address changes, the registry replaces the old subscription record and invalidates its previous version.', {'change'}),
    ('The archive cannot preserve earlier revisions when it replaces an existing record.', {'change'}),
    ('For the next request, the service retrieves the currently valid subscription record from the index.', {'read'}),
    ('收到新增档案后，系统将新记录写入档案库。', {'write'}),
    ('档案修订后，系统更新记录并使旧版本失效。', {'change'}),
    ('处理下一次请求时，系统从档案库检索当前有效的记录。', {'read'}),
])
def test_local_actions_keep_original_objects_conditions_and_negative_behavior(text, expected):
    from docthinker.server.research_operation_stages import source_operation_stages
    row = {'content': text, 'section': '2 Methods', 'outline_roles': ['mechanism']}
    before = deepcopy(row)
    assert set(source_operation_stages(row)) == expected
    assert row == before


@pytest.mark.parametrize('text,section,extra', [
    ('The archive stores many subscription records.', '2 Methods', {}),
    ('Write, change and read are three terms.', '2 Methods', {}),
    ('Future work will invalidate obsolete subscription records.', '2 Methods', {}),
    ('We do not describe how to update subscription records.', '2 Methods', {}),
    ('Prior systems insert new records and invalidate older versions.', 'Related Works', {}),
    ('The service inserts incoming records into the archive.', '2 Methods', {'status': 'revoked'}),
    ('The publication was revised in 2021. The archive contains 52 records.', '2 Methods', {}),
    ('The rate changes each hour. An archive stores subscriptions.', '2 Methods', {}),
])
def test_labels_plans_absence_other_work_and_disjoint_subjects_are_not_completed_stages(text, section, extra):
    from docthinker.server.research_operation_stages import source_operation_stages
    row = {'content': text, 'section': section, **extra}
    before = deepcopy(row)
    assert source_operation_stages(row) == ()
    assert row == before


@pytest.mark.parametrize('domain', ['subscription', 'stock', 'archive'])
def test_three_requested_stages_compete_with_many_generic_definitions_in_existing_budget(domain):
    from docthinker.server.research_lexical_coverage import select_lexical_coverage
    from docthinker.server.research_retrieval import build_retrieval_plan
    name = 'CloverLedger'
    def row(identifier, ordinal, text):
        return {'id': identifier, 'document_id': 'live', 'filename': 'cloverledger_2602.12345.pdf',
                'ordinal': ordinal, 'page': ordinal + 1, 'content': text}
    records = f'{domain} records'
    chunks = [
        row('write', 0, f'2 Methods\nWhen incoming {records} arrive, the registry inserts each new record '
            'into the persistent journal. It rejects an insertion unless an owner is assigned.'),
        row('change', 1, f'2 Methods\nWhen existing {records} change, the registry replaces their current '
            'values and invalidates the previous version. It retains prior versions only for authorized audit.'),
        row('read', 2, f'2 Methods\nFor subsequent requests, the registry retrieves current {records} '
            'from the index. Invalidated versions are excluded unless the caller explicitly requests historical data.'),
    ]
    chunks += [row(f'generic-{n}', n, f'2 Methods\nThe {domain} archive stores many records. '
                    f'The {domain} index contains searchable names and the {domain} queue holds queued entries.')
               for n in range(3, 59)]
    question = (f'Explain {name}: how are incoming {records} inserted, how are changed records revised '
                'or invalidated, and how are current records retrieved for subsequent requests?')
    original = deepcopy(chunks)
    result = select_lexical_coverage(question, build_retrieval_plan(question, chunks)['sources'], chunks)
    selected = {s['id'] for s in result['sources']}
    assert {'write', 'change', 'read'} <= selected
    assert len(result['sources']) <= 8
    assert result['trace']['candidate_count'] <= 48
    assert sum(len(s['content'].encode()) for s in result['sources']) <= 12000
    selection = result['trace']['selection']
    assert selection['goal_family_caps']['mechanism'] == 1.3
    assert selection['local_exchange_attempts'] <= 384
    by_id = {s['id']: s['content'] for s in chunks}
    assert all(by_id[s['id']][s['excerpt_start']:s['excerpt_end']] == s['content'] for s in result['sources'])
    assert all('unless' in s['content'] for s in result['sources'] if s['id'] in {'write', 'read'})
    assert chunks == original


def split_stage_material():
    texts = {
        'write': 'The registry inserts incoming records into the journal only after owner approval.',
        'change': 'The registry invalidates obsolete records while retaining earlier revisions for audit.',
        'read': 'For the next request, the registry retrieves current records and excludes invalidated versions.',
    }
    chunks = []
    for doc, name in [('a', 'cloverledger'), ('b', 'hazelstore')]:
        for index, (stage, text) in enumerate(texts.items()):
            chunks.append({'id': f'{doc}-{stage}', 'document_id': doc,
                'filename': f'{name}_2602.12345.pdf', 'ordinal': index, 'page': index + 1,
                'content': '2 Methods\n' + text})
    return chunks


@pytest.mark.parametrize('question', [
    'For CloverLedger, explain only how incoming records are written; '
    'for HazelStore, explain only how current records are retrieved for subsequent requests.',
    '对 CloverLedger 只说明如何写入新记录；对 HazelStore 只说明后续查询如何读取当前记录。',
])
def test_named_documents_do_not_inherit_the_other_documents_requested_operations(question):
    from docthinker.server.research_lexical_coverage import select_lexical_coverage
    from docthinker.server.research_retrieval import build_retrieval_plan
    chunks = split_stage_material()
    original = deepcopy(chunks)
    result = select_lexical_coverage(question, build_retrieval_plan(question, chunks)['sources'], chunks)
    assert {'a-write', 'b-read'} <= {s['id'] for s in result['sources']}
    selection = result['trace']['selection']
    # This audits the request-scoped ledger, not a hard document permission
    # filter or a claim that any matching sentence is semantically sufficient.
    assert selection['operation_stage_cells_covered'] == 2
    assert selection['operation_stage_cells_unverified'] == 0
    assert selection['operation_stages_are_semantic_support'] is False
    assert selection['goal_family_caps']['mechanism'] == 1.3
    assert result['trace']['candidate_count'] <= 48 and len(result['sources']) <= 8
    assert sum(len(s['content'].encode()) for s in result['sources']) <= 12000
    assert chunks == original


@pytest.mark.parametrize('question,expected', [
    ('For CloverLedger, explain only how incoming records are written; '
     'do not explain how HazelStore retrieves records.', {'write'}),
    ('Explain only how CloverLedger writes incoming records. '
     'The source says "HazelStore retrieves records".', {'write'}),
    ('请解释 CloverLedger 如何写入新记录；不要讨论 HazelStore 的后续读取。', {'write'}),
])
def test_other_document_in_negated_or_quoted_material_does_not_add_a_global_stage(question, expected):
    from docthinker.server.research_operation_stages import requested_operation_stages
    assert set(requested_operation_stages(question)) == expected


def test_duplicate_material_name_does_not_choose_an_arbitrary_stage_scope_owner():
    from docthinker.server.research_query_facets import build_query_facets
    chunks = split_stage_material()
    chunks += [{**r, 'id': 'duplicate-' + r['id'], 'document_id': 'ambiguous'}
               for r in chunks if r['document_id'] == 'a']
    query = 'Explain only how CloverLedger writes incoming records.'
    facets = build_query_facets(query, chunks)
    owners = {doc for facet in facets['facets'] for doc in facet.get('document_ids', ())}
    assert owners in (set(), {'a', 'ambiguous'})


@pytest.mark.parametrize('question', [
    'For CloverLedger, explain only writes of incoming records; '
    'for UnknownCrate, explain only retrieval of current records.',
    '对 CloverLedger 只说明写入新记录；对 UnknownCrate 只说明后续读取。',
])
def test_explicit_unresolved_target_cannot_assign_its_stage_to_another_known_document(question):
    from docthinker.server.research_operation_stages import operation_stages_by_document
    assert operation_stages_by_document(question, split_stage_material()) == {'a': ('write',)}


def test_collective_paper_target_keeps_all_explicitly_named_owners():
    from docthinker.server.research_operation_stages import operation_stages_by_document
    question = ('Explain CloverLedger and HazelStore. For each paper, '
                'describe how incoming records are written and how current records are retrieved.')
    assert operation_stages_by_document(question, split_stage_material()) == {
        'a': ('write', 'read'), 'b': ('write', 'read')}


def test_purpose_preposition_is_not_misclassified_as_an_unknown_document_target():
    from docthinker.server.research_operation_stages import operation_stages_by_document
    question = 'Explain CloverLedger. For future use, describe how incoming records are written.'
    assert operation_stages_by_document(question, split_stage_material()) == {'a': ('write',)}


def test_ambiguous_target_pronoun_does_not_choose_a_known_owner():
    from docthinker.server.research_operation_stages import operation_stages_by_document
    question = 'Compare CloverLedger and HazelStore. For that system, explain how current records are retrieved.'
    assert operation_stages_by_document(question, split_stage_material()) == {}


@pytest.mark.parametrize('question', [
    'Write an evidence-backed report comparing the two archive designs.',
    'Write a concise technical literature review of the available archive methods.',
    'Write a short practical proposal for the catalog team.',
    'Compare these papers on retrieval and the limits of long-context evidence.',
    'Propose a testable research hypothesis about retrieval and evidence placement.',
    'Write an evidence-backed report about retrieval methods and reported results.',
])
def test_modified_writing_artifacts_and_retrieval_topics_are_not_lifecycle_requests(question):
    from docthinker.server.research_operation_stages import requested_operation_stages
    assert requested_operation_stages(question) == ()


@pytest.mark.parametrize('question,expected', [
    ('Write a concise report explaining how incoming archive records are inserted '
     'and how the service retrieves current records for subsequent requests.', {'write', 'read'}),
    ('Compare retrieval methods, then explain how the archive invalidates obsolete records.', {'change'}),
])
def test_artifact_or_topic_exclusion_keeps_separately_requested_real_operations(question, expected):
    from docthinker.server.research_operation_stages import requested_operation_stages
    assert set(requested_operation_stages(question)) == expected


@pytest.mark.parametrize('text,section', [
    ('The prompt template for catalog maintenance:\nYou are an archive maintenance agent. '
     'You update the current catalog records and retrieve previous records for comparison. '
     'Return the proposed operations in a list.', 'Appendix Prompt Templates'),
    ('System prompt:\nYou are the stock registry. You insert incoming records into the journal. '
     'You invalidate obsolete records when a conflict appears.', '4 Prompt Examples'),
    ('提示模板：\n你是档案管理系统。系统更新记录并使旧版本失效。'
     '系统从档案库检索当前有效的记录。请输出建议操作。', '附录 提示模板'),
])
def test_in_document_prompt_instructions_cannot_fill_reported_operation_cells(text, section):
    from docthinker.server.research_operation_stages import source_operation_stages
    row = {'content': text, 'section': section, 'outline_roles': ['mechanism']}
    original = deepcopy(row)
    assert source_operation_stages(row) == ()
    assert row == original


def test_description_of_using_a_prompt_is_not_blanket_excluded_as_prompt_data():
    from docthinker.server.research_operation_stages import source_operation_stages
    row = {'content': 'After the comparison prompt identifies a conflict, the registry invalidates '
           'the obsolete subscription record and preserves its previous version for audit.',
           'section': '2 Methods', 'outline_roles': ['mechanism']}
    assert source_operation_stages(row) == ('change',)


@pytest.mark.parametrize('domain', ['subscription', 'inventory'])
def test_record_validity_change_is_not_satisfied_by_only_refreshing_community_metadata(domain):
    from docthinker.server.research_lexical_coverage import select_lexical_coverage
    from docthinker.server.research_retrieval import build_retrieval_plan
    def row(identifier, ordinal, text):
        return {'id': identifier, 'document_id': 'live', 'filename': 'fernregistry_2603.12456.pdf',
                'ordinal': ordinal, 'page': ordinal + 1, 'content': text}
    chunks = [row('validity-change', 0,
        f'2 Record validity\nWhen a new {domain} record contradicts a previous record with an '
        'overlapping validity interval, the registry invalidates the previous record by setting its '
        'end time to the start time of the new record. The registry preserves earlier versions for '
        'historical audit rather than deleting the history.')]
    chunks += [row(f'community-{n}', n + 1,
        f'2 Community maintenance\nWhen a new {domain} node arrives, the registry updates the '
        'community summary and replaces its display label. The registry inserts the node into a '
        'neighboring community and retrieves the community summary for later display. '
        'These updates refresh the organizational metadata only.') for n in range(56)]
    question = (f'Explain FernRegistry: when {domain} records change, how are contradictory facts '
                'invalidated while previous valid states remain available for historical audit?')
    original = deepcopy(chunks)
    result = select_lexical_coverage(question, build_retrieval_plan(question, chunks)['sources'], chunks)
    selected = {s['id']: s for s in result['sources']}
    assert 'validity-change' in selected
    assert 'overlapping validity interval' in selected['validity-change']['content']
    assert 'rather than deleting the history' in selected['validity-change']['content']
    assert result['trace']['candidate_count'] <= 48 and len(selected) <= 8
    assert sum(len(s['content'].encode()) for s in selected.values()) <= 12000
    selection = result['trace']['selection']
    assert selection['goal_family_caps']['mechanism'] == 1.3
    assert selection['local_exchange_attempts'] <= 384
    by_id = {s['id']: s['content'] for s in chunks}
    assert all(by_id[s['id']][s['excerpt_start']:s['excerpt_end']] == s['content']
               for s in result['sources'])
    assert chunks == original


def test_operation_quality_distinguishes_local_validity_conditions_from_generic_refresh():
    from docthinker.server.research_operation_stages import source_operation_quality
    complete = {'content': 'When incoming subscription facts contradict previous valid records, '
                'the registry invalidates the previous versions while retaining their history for audit.',
                'section': '2 Methods', 'outline_roles': ['mechanism']}
    generic = {'content': 'The registry updates the community summary and replaces its display label.',
               'section': '2 Methods', 'outline_roles': ['mechanism']}
    before = deepcopy((complete, generic))
    assert 0 < source_operation_quality(generic, 'change') < source_operation_quality(complete, 'change') <= 1
    assert (complete, generic) == before


@pytest.mark.parametrize('separator', [' ', '\n\n'])
def test_operation_quality_cannot_borrow_conditions_or_old_new_states_from_another_sentence(separator):
    from docthinker.server.research_operation_stages import source_operation_quality
    plain = {'content': 'The registry updates the community summary.', 'section': '2 Methods'}
    unrelated = ('When previous visitors arrive before closing, the cafe serves new drinks '
                 'while retaining old receipts for audit.')
    decorated = {**plain, 'content': plain['content'] + separator + unrelated}
    assert source_operation_quality(decorated, 'change') == source_operation_quality(plain, 'change')


def test_operation_quality_preserves_explicit_failure_as_conditioned_material():
    from docthinker.server.research_operation_stages import source_operation_quality, source_operation_stages
    row = {'content': 'When a replacement arrives, the archive cannot preserve earlier versions '
           'because the previous record is overwritten.', 'section': '2 Methods'}
    original = deepcopy(row)
    assert source_operation_stages(row) == ('change',)
    assert 0 < source_operation_quality(row, 'change') <= 1
    assert row == original


@pytest.mark.parametrize('text,section', [
    ('Future work will invalidate previous versions when incoming stock records conflict, '
     'while retaining their history for audit.', '2 Methods'),
    ('System prompt:\nYou are a record maintenance agent. When records conflict, '
     'you invalidate previous versions while retaining old records for audit.', 'Appendix Prompt Templates'),
])
def test_future_or_template_conditions_cannot_create_nonzero_operation_quality(text, section):
    from docthinker.server.research_operation_stages import source_operation_quality
    row = {'content': text, 'section': section, 'outline_roles': ['mechanism']}
    assert source_operation_quality(row, 'change') == 0


def test_operation_quality_is_bounded_shape_metadata_not_a_new_inferred_stage():
    from docthinker.server.research_operation_stages import source_operation_quality, source_operation_qualities
    row = {'content': 'When incoming archive records arrive, the registry inserts new records '
           'only after an owner is assigned.', 'section': '2 Methods'}
    original = deepcopy(row)
    qualities = source_operation_qualities(row)
    assert set(qualities) <= {'write', 'change', 'read'}
    assert 0 < qualities['write'] <= 1
    assert all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in qualities.values())
    assert source_operation_quality(row, 'change') == 0
    assert source_operation_quality(row, 'invented-stage') == 0
    assert row == original


def test_same_operation_explicit_field_assignment_adds_local_state_shape():
    from docthinker.server.research_operation_stages import source_operation_quality
    plain = {'content': 'When entries conflict, the registry invalidates the affected records.',
             'section': '2 Methods'}
    assigned = {**plain, 'content': 'When entries conflict, the registry invalidates the affected '
                'records by setting cutoff_mark to arrival_mark.'}
    original = deepcopy((plain, assigned))
    assert 0 < source_operation_quality(plain, 'change') < source_operation_quality(assigned, 'change') <= 1
    assert (plain, assigned) == original


def test_another_actors_assignment_does_not_complete_the_registry_operation():
    from docthinker.server.research_operation_stages import source_operation_quality
    plain = {'content': 'The registry updates the summary.', 'section': '2 Methods'}
    mixed = {**plain, 'content': 'The registry updates the summary while decoder sets its color '
             'by assigning palette_mark to display_mark.'}
    original = deepcopy(mixed)
    assert source_operation_quality(mixed, 'change') == source_operation_quality(plain, 'change')
    assert mixed == original
