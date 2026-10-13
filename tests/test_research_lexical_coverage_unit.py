"""Default no-model evidence competition preserves exact live source spans."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from docthinker.server.research_lexical_coverage import select_lexical_coverage
from docthinker.server.research_retrieval import build_retrieval_plan
from docthinker.server.research_scheduler import ContextWorkResult
from docthinker.server.research_service import ResearchService
from docthinker.session_manager import SessionManager


@pytest.mark.parametrize('question, expected', [
    ('Explain how recall storage works.', True),
    ('Explain how the precision cache stores records.', True),
    ('Compare recall scores only.', False),
    ('Compare Cedar and Maple.', False),
    ('Compare methods for a research assistant.', True),
    ('Compare latency numbers; do not explain architecture.', False),
])
def test_component_scope_requires_positive_functional_intent(question, expected):
    from docthinker.server.research_component_coverage import requests_component_explanation
    assert requests_component_explanation(question) is expected


def test_ordered_method_proxy_is_bounded_original_process_not_arbitrary_first_then():
    from docthinker.server.research_component_coverage import ordered_method_steps
    row = {'content': 'We first extract a registry from the source files, then pre-\ngenerate a summary. '
           'This applies only when all files have passed the integrity check.', 'section': 'Methods'}
    original = deepcopy(row)
    assert ordered_method_steps(row) == ('extract', 'pre-\ngenerate')
    assert row == original
    assert not ordered_method_steps({'content': 'The first author wrote a note, then the reviewer responded.'})
    assert not ordered_method_steps({**row, 'section': 'References'})
    assert not ordered_method_steps({**row, 'content': 'Figure 4: ' + row['content']})


def test_ordered_method_pairs_have_per_document_cap_and_unknown_intent_does_not_activate():
    actions = [('build', 'select'), ('extract', 'store'), ('encode', 'retrieve'),
               ('partition', 'combine'), ('load', 'generate')]
    chunks = [{'id': f's{i}', 'document_id': 'Cedar', 'filename': 'Cedar.txt', 'ordinal': i,
               'content': f'2 Methods\nWe first {a} complete records from the permitted files, '
                          f'then {b} an output only after user consent. This operation does not alter input files.'}
              for i, (a, b) in enumerate(actions)]
    chunks += [{'id': f'noise{i}', 'document_id': 'Cedar', 'filename': 'Cedar.txt', 'ordinal': i + 5,
                'content': 'No measurements are reported in this unrelated note.'} for i in range(5)]
    question = 'Explain Cedar architecture and operation.'
    result = select_lexical_coverage(question, build_retrieval_plan(question, chunks)['sources'], chunks)
    assert 0 < result['trace']['selection']['ordered_method_cells'] <= 2
    _assert_exact_bounded_windows(result, chunks)
    question = 'Compare Cedar and Maple.'
    with patch('docthinker.server.research_lexical_coverage.ordered_method_steps',
               side_effect=AssertionError('unknown comparison intent must not acquire process obligations')):
        select_lexical_coverage(question, build_retrieval_plan(question, chunks)['sources'], chunks)


def corpus():
    method = ('2 Method\nThe Cedar retriever encodes each query into a dense vector. '
              'It selects five documents using inner-product search and supplies them to a generator. '
              'The generator integrates retrieved passages with the query; it does not update the source texts.')
    result = ('3 Evaluation\nWe compare the Cedar retrieval method against a fixed keyword baseline. '
              'The experiment uses 200 questions and evaluates exact-match accuracy. '
              'The observed advantage holds only for this question set.')
    background = ('1 Introduction\nHuman endeavors across a range of domains rely on our ability '
                  'to read and reason about large collections of documents. '
                  'Retrieval systems enable answers from private documents.')
    references = ('Alice Smith and Bob Doe. 2021. Dense retrieval survey. In Proc. of ACL.\n'
                  'Carol Jones and Dan Lee. 2022. Comparison of generation systems. Journal of AI, 4:1–9.')
    return [{'id': f'source-{index}', 'document_id': 'cedar', 'ordinal': index,
             'filename': 'cedar.txt', 'content': text}
            for index, text in enumerate([background, references, method, result,
                *[f'Background catalog {i}: retrieval generation systems are popular in the literature.' for i in range(8)]])]


def test_complete_method_and_condition_compete_without_semantic_assets():
    chunks = corpus()
    question = 'How does Cedar retrieval work, and what are the evaluation conditions and limitations?'
    baseline = build_retrieval_plan(question, chunks)['sources']
    snapshot = deepcopy(chunks)
    result = select_lexical_coverage(question, baseline, chunks)
    assert chunks == snapshot
    content = '\n'.join(row['content'] for row in result['sources'])
    assert 'inner-product search' in content and 'does not update' in content
    assert '200 questions' in content and 'only for this question set' in content
    assert 'Proc. of ACL' not in content
    assert 0 < len(result['sources']) <= 8
    assert sum(len(row['content'].encode()) for row in result['sources']) <= 12000
    originals = {row['id']: row['content'] for row in chunks}
    for row in result['sources']:
        assert originals[row['id']][row['excerpt_start']:row['excerpt_end']] == row['content']
    assert result['trace']['provider_calls'] == result['trace']['model_calls'] == 0
    assert result['trace']['score_kind'] == 'lexical_retrieval_utility'
    assert result['trace']['semantic_entailment_verified'] is False


def test_metadata_and_literal_requests_do_not_trigger_window_competition():
    chunks = corpus()
    for question in ('List the references in the Cedar paper.', 'Translate this method verbatim without analysis.'):
        baseline = build_retrieval_plan(question, chunks)['sources']
        with patch('docthinker.server.research_lexical_coverage.prepare_evidence_candidates',
                   side_effect=AssertionError('literal/metadata task must not rescan')):
            result = select_lexical_coverage(question, baseline, chunks)
        assert result['trace']['status'] == 'retained'
        assert result['trace']['reason'] in {'explicit_metadata_request', 'literal_or_editorial_request'}


def test_primary_document_coverage_hint_does_not_exclude_an_unretrieved_relevant_paper():
    question = 'Compare the cobalt retrieval methods and evaluation conditions across these papers.'
    def source(identifier, document, text):
        return {'id': identifier, 'document_id': document, 'filename': document + '.txt',
                'ordinal': 0, 'content': text}
    first = source('a', 'a', 'Methods: cobalt retrieval compares document performance.')
    second = source('b', 'b', 'Methods: cobalt retrieval evaluates basic document quality.')
    missing = source('c', 'c', 'Methods: the cobalt retrieval method computes calibrated dense scores. '
                     'Evaluation conditions use 500 independent questions and controls.')
    chunks = [first, second, missing, *[
        source(str(i), 'other', f'The unrelated orchard grows apples {i}.') for i in range(6)]]
    baseline = build_retrieval_plan(question, [first, second])['sources']
    assert {row['document_id'] for row in baseline} == {'a', 'b'}
    result = select_lexical_coverage(question, baseline, chunks)
    assert any(row['id'] == 'c' and 'calibrated dense scores' in row['content']
               and '500 independent questions' in row['content'] for row in result['sources'])
    assert len(result['sources']) <= 8
    assert sum(len(row['content'].encode()) for row in result['sources']) <= 12000
    originals = {row['id']: row['content'] for row in chunks}
    for row in result['sources']:
        assert originals[row['id']][row['excerpt_start']:row['excerpt_end']] == row['content']


@pytest.mark.asyncio
@pytest.mark.parametrize('scheduler_status', ['busy', 'timed_out', 'failed'])
async def test_no_model_assets_timeout_does_not_retry_or_use_late_coverage(tmp_path, scheduler_status):
    manager = SessionManager(str(tmp_path / '_system'), str(tmp_path), migrate_legacy=False)
    sid = manager.create_session('Cedar')['id']
    service = ResearchService(tmp_path, semantic_retriever=None, evidence_reranker=None,
                              query_translator=None)
    document = service.store.accept_document(sid, 'cedar.txt', b'public synthetic paper')
    service.store.claim_document(sid, document['id'])
    service.store.complete_document(sid, document['id'], [{'text': row['content']} for row in corpus()])
    calls = []
    async def writer(prompt, **kwargs):
        calls.append(prompt)
        return 'The Cedar method retrieves documents. [S1]'
    from unittest.mock import AsyncMock
    request = SimpleNamespace(session_id=sid, question='How does Cedar retrieval work?',
                              use_memory=False, use_conversation_context=False,
                              insight_mode='off', knowledge_skill_mode='off')
    try:
        with patch.object(service._context_scheduler, 'run', AsyncMock(return_value=ContextWorkResult(scheduler_status))) as run:
            result = await service.answer(request, writer, manager)
        assert run.await_count == len(calls) == 1
        assert result['reasoning']['trace']['lexical_evidence_coverage']['reason'] == 'scheduler_' + scheduler_status
        assert 'Proc. of ACL' not in calls[0]
    finally:
        await service.stop()


def _assert_exact_bounded_windows(result, chunks):
    originals = {row['id']: row['content'] for row in chunks}
    assert len(result['sources']) <= 8
    assert sum(len(row['content'].encode()) for row in result['sources']) <= 12000
    for source in result['sources']:
        assert originals[source['id']][source['excerpt_start']:source['excerpt_end']] == source['content']


def test_two_requested_mechanisms_in_one_paper_do_not_cover_each_other():
    question = 'Explain the cedar indexing mechanism and the maple paging mechanism.'
    def source(index, text):
        return {'id': str(index), 'document_id': 'paper', 'filename': 'paper.txt',
                'ordinal': index, 'content': text}
    first = source(0, '2 Methods\nThe cedar indexing mechanism encodes records and constructs an index. '
                   'It selects items and accesses stored identifiers.')
    second = source(1, '3 Methods\nThe maple paging mechanism loads blocks, evicts old blocks and swaps stored pages. '
                    'It accesses paged data through lookup calls.')
    chunks = [first, second, *[source(i, 'Unrelated apple orchard background.') for i in range(2, 10)]]
    baseline = build_retrieval_plan(question, chunks)['sources']
    assert {row['id'] for row in baseline} == {'0', '1'}
    result = select_lexical_coverage(question, baseline, chunks)
    text = '\n'.join(row['content'] for row in result['sources'])
    assert 'encodes records and constructs an index' in text
    assert 'loads blocks, evicts old blocks and swaps stored pages' in text
    assert 'accesses stored identifiers' in text and 'accesses paged data' in text
    _assert_exact_bounded_windows(result, chunks)


def test_empty_cleaned_primary_pack_still_searches_for_substantive_evidence():
    question = 'Explain retrieval methods and evaluation conditions.'
    def source(index, text):
        return {'id': str(index), 'document_id': 'paper', 'filename': 'paper.txt',
                'ordinal': index, 'content': text}
    bibliography = source(0, 'Alice Smith and Bob Doe. 2021. Retrieval methods. In Proc. of ACL.\n'
                          'Carol Jones and Dan Lee. 2022. Evaluation conditions. Journal of AI, 4:1–9.')
    method = source(1, 'Methods: the retriever computes calibrated dense scores. '
                    'Evaluation conditions require 500 questions.')
    chunks = [bibliography, method, *[source(i, 'Unrelated agricultural background about leaves.') for i in range(2, 10)]]
    baseline = build_retrieval_plan(question, [bibliography])['sources']
    result = select_lexical_coverage(question, baseline, chunks)
    assert result['trace']['metadata_filter']['retained_source_count'] == 0
    assert any(row['id'] == '1' and 'calibrated dense scores' in row['content']
               and '500 questions' in row['content'] for row in result['sources'])
    assert all(row['id'] != '0' for row in result['sources'])
    _assert_exact_bounded_windows(result, chunks)


def test_agent_noun_suffix_does_not_turn_flower_material_into_flow_evidence():
    question = 'Explain the flow mechanism and its operating conditions.'
    def source(identifier, document, text):
        return {'id': identifier, 'document_id': document, 'filename': document + '.txt',
                'ordinal': 0, 'content': text}
    flow = source('flow', 'fluid', 'The flow mechanism routes fluid through a pressure valve under operating conditions.')
    flower = source('flower', 'botany', 'Flower petals require sunlight for growth. The flower pollen is transported by bees.')
    chunks = [flow, flower, *[source(str(i), 'orchard', 'The orchard grows apples.') for i in range(7)]]
    baseline = build_retrieval_plan(question, [flow])['sources']
    result = select_lexical_coverage(question, baseline, chunks)
    assert {row['id'] for row in result['sources']} == {'flow'}
    assert result['sources'][0]['content'] == flow['content']
    _assert_exact_bounded_windows(result, chunks)


def _structured_source(identifier, document, ordinal, content):
    return {'id': identifier, 'document_id': document, 'filename': document + '.txt',
            'ordinal': ordinal, 'page': ordinal + 1, 'content': content}


def _select_exact_source_pack(question, chunks, baseline_chunks=None):
    snapshot = deepcopy(chunks)
    baseline = build_retrieval_plan(question, chunks if baseline_chunks is None else baseline_chunks)['sources']
    result = select_lexical_coverage(question, baseline, chunks)
    assert chunks == snapshot
    _assert_exact_bounded_windows(result, chunks)
    originals = {row['id']: row for row in chunks}
    assert all(row['page'] == originals[row['id']]['page']
               and row['ordinal'] == originals[row['id']]['ordinal'] for row in result['sources'])
    assert result['trace']['model_calls'] == result['trace']['provider_calls'] == 0
    return result


def test_high_overlap_appendix_does_not_crowd_out_low_overlap_main_method():
    question = 'Compare Cedar and Maple methods, evaluations, results and operating conditions.'
    cedar = _structured_source('cedar-core', 'cedar', 0,
        '2 Methods\nWe combine a dual encoder with an external index. '
        'Each input vector scores records by inner product, then passes the five highest records to a decoder. '
        'The decoder cannot update the stored records during inference.')
    maple = _structured_source('maple-core', 'maple', 0,
        '2 Methods\nThe memory is divided into a working context and archival storage. '
        'The controller evicts old blocks and loads archived blocks into the working context only after a lookup.')
    appendix = [_structured_source('appendix-' + str(index), 'cedar', index + 1,
        'Appendix A Additional Evaluation\nCedar and Maple methods, evaluations, results and operating conditions '
        'are compared in a participant training exercise. Participants select examples, retrieve records and '
        f'generate labels for catalog {index}. The evaluation reports accuracy only for these annotation exercises.')
        for index in range(10)]
    chunks = [cedar, *appendix, maple]
    # The candidate pass must recover the omitted method, not rely on an
    # already-favorable primary retrieval packet or an extra model call.
    result = _select_exact_source_pack(question, chunks, appendix + [maple])
    by_id = {row['id']: row['content'] for row in result['sources']}
    assert 'cedar-core' in by_id and 'maple-core' in by_id
    assert 'five highest records to a decoder' in by_id['cedar-core']
    assert 'cannot update the stored records' in by_id['cedar-core']
    assert 'only after a lookup' in by_id['maple-core']


def test_only_method_in_appendix_remains_eligible_for_normal_method_question():
    question = 'Explain the Cedar architecture and its operating conditions.'
    appendix = _structured_source('appendix-method', 'cedar', 12,
        'Appendix A Implementation Details\nThe Cedar memory is divided into a working context and external storage. '
        'The controller copies a block from storage into the context on a lookup miss; '
        'it does not evict a pinned block until the caller releases it.')
    other = [_structured_source('overview-' + str(index), 'cedar', index,
        f'1 Introduction\nThe Cedar architecture supports application catalog {index}. '
        'The operating conditions depend on the deployment environment. '
        'These introductory comments do not specify how the controller moves blocks.') for index in range(10)]
    result = _select_exact_source_pack(question, [*other, appendix])
    passage = next((row['content'] for row in result['sources'] if row['id'] == 'appendix-method'), '')
    assert 'copies a block from storage into the context on a lookup miss' in passage
    assert 'does not evict a pinned block until the caller releases it' in passage


def test_explicit_appendix_implementation_request_keeps_requested_details():
    question = 'Explain Appendix B implementation details of Cedar: give the bucket count and the condition for flushing.'
    appendix = _structured_source('appendix-details', 'cedar', 14,
        'Appendix B Implementation Details\nThe Cedar index contains exactly 257 buckets. '
        'It flushes a bucket only after 64 writes, unless the bucket is pinned; '
        'a pinned bucket must remain resident until its lock is released.')
    main = _structured_source('main-method', 'cedar', 0,
        '2 Methods\nWe combine the Cedar encoder with a searchable index. '
        'The main architecture description does not specify bucket count or the condition for flushing.')
    other = [_structured_source('main-extra-' + str(index), 'cedar', index + 1,
        f'3 Evaluation\nCedar implementation details vary in configuration {index}. '
        'We compare retrieval accuracy across benchmark questions and experimental settings.') for index in range(10)]
    result = _select_exact_source_pack(question, [main, *other, appendix])
    passage = next((row['content'] for row in result['sources'] if row['id'] == 'appendix-details'), '')
    assert 'exactly 257 buckets' in passage
    assert 'only after 64 writes, unless the bucket is pinned' in passage
    assert 'must remain resident until its lock is released' in passage


def test_third_party_related_work_cannot_replace_the_papers_own_result():
    question = 'Compare Cedar and Maple evaluation results and limitations: what did the authors demonstrate?'
    own = _structured_source('cedar-own-result', 'cedar', 0,
        '3 Results\nWe observed 61 correct answers out of 100 on the held-out set. '
        'This gain held only when the supporting record was present; '
        'without that record, accuracy did not exceed the baseline.')
    maple = _structured_source('maple-own-result', 'maple', 0,
        '3 Results\nWe observed 58 correct answers out of 100 on a separate held-out set. '
        'The experiment did not compare against Cedar, so these observations do not establish superiority.')
    related = [_structured_source('related-' + str(index), 'cedar', index + 1,
        '4 Related Work\nThe Cedar and Maple evaluation results and limitations are discussed in the literature. '
        f'Smith and colleagues reported that system Quartz{index} outperformed its baseline with 95 percent accuracy. '
        'Their benchmark results used different data and are not a measurement of the present system.')
        for index in range(10)]
    result = _select_exact_source_pack(question, [own, *related, maple], related + [maple])
    by_id = {row['id']: row['content'] for row in result['sources']}
    assert 'cedar-own-result' in by_id and 'maple-own-result' in by_id
    assert '61 correct answers out of 100' in by_id['cedar-own-result']
    assert 'only when the supporting record was present' in by_id['cedar-own-result']
    assert 'accuracy did not exceed the baseline' in by_id['cedar-own-result']
    assert 'did not compare against Cedar' in by_id['maple-own-result']


def test_reingestion_uuid_changes_do_not_change_frozen_evidence_selection():
    question = 'Compare Cedar and Maple methods, evaluation results and operating conditions.'
    chunks = []
    for document, mechanism in [('cedar', 'index'), ('maple', 'memory')]:
        chunks.append(_structured_source(document + '-method', document, 0,
            f'2 Methods\nWe combine an encoder with an external {mechanism}. '
            f'The {mechanism} passes five records to a generator only after an exact lookup. '
            'The stored records are not changed during inference.'))
        for ordinal in range(1, 36):
            chunks.append(_structured_source(document + '-' + str(ordinal), document, ordinal,
                '3 Evaluation\nCedar and Maple methods, evaluation results and operating conditions '
                f'were compared in panel {ordinal:02d}. We observed accuracy on the fixed benchmark, '
                'but these panel results do not establish performance on other tasks.'))

    def signature(result):
        return [(row['filename'], row['ordinal'], row['page'], row['excerpt_start'],
                 row['excerpt_end'], row['content']) for row in result['sources']]

    first = _select_exact_source_pack(question, chunks)
    assert first['trace']['candidate_count'] == 48  # Exercise the actual pool boundary.
    assert any(row['ordinal'] == 0 for row in first['sources'])
    import uuid
    for reverse in (False, True):
        document_ids = {document: str(uuid.UUID(int=number)) for document, number in
                        [('cedar', 2 if reverse else 1), ('maple', 1 if reverse else 2)]}
        cloned = [{**row, 'document_id': document_ids[row['document_id']],
                   'id': str(uuid.UUID(int=(len(chunks) - index if reverse else index + 1) + 100))}
                  for index, row in enumerate(chunks)]
        if reverse:
            cloned.reverse()
        result = _select_exact_source_pack(question, cloned)
        assert signature(result) == signature(first)


@pytest.mark.parametrize('extra_chunks', [0, 8])
def test_unfinished_source_condition_is_preserved_but_never_claimed_complete(extra_chunks):
    question = 'Explain the Cedar method, its evaluation results and operating conditions.'
    fragment = _structured_source('unfinished-result', 'cedar', 0,
        'We find that the Cedar retriever improves performance, especially when')
    chunks = [fragment, *[_structured_source('unrelated-' + str(index), 'orchard', index,
        'Plum branches bear green leaves.') for index in range(extra_chunks)]]
    result = _select_exact_source_pack(question, chunks)
    source = next(row for row in result['sources'] if row['id'] == fragment['id'])
    assert source['content'] == fragment['content']
    assert source['excerpt_start'] == 0 and source['excerpt_end'] == len(fragment['content'])
    assert source['sentence_boundary_preserved'] is False
    assert any(gap['code'] == 'source_boundary_unverified' for gap in result['gaps'])


def test_empty_candidate_pool_still_reports_the_4096_chunk_scan_limit():
    question = 'Explain the zirconium indexing mechanism and its operating conditions.'
    chunks = [_structured_source('orchard-' + str(index), 'orchard', index,
        'Plum branches bear green leaves.') for index in range(4097)]
    result = _select_exact_source_pack(question, chunks, [])
    assert result['sources'] == []
    assert result['trace']['reason'] == 'no_literal_window_candidates'
    assert result['trace']['corpus_scan_limited'] is True
    assert result['trace']['candidate_pool']['live_chunks_scanned'] == 4096
    assert [gap['code'] for gap in result['gaps']].count('literal_coverage_scan_limited') == 1


def test_named_papers_with_matching_passages_keep_each_sources_provenance():
    question = ('Compare Cedar and Maple methods and evaluation results, '
                'with evidence from both papers.')
    shared = ('We combine an encoder with an external index. '
              'The index passes five records to a generator only after an exact lookup. '
              'We observed 61 correct answers out of 100 on the held-out set; '
              'these observations apply only to that set.')
    chunks = [_structured_source(document + '-shared', document, 0, shared)
              for document in ('cedar', 'maple')]
    result = _select_exact_source_pack(question, chunks)
    by_document = {row['document_id']: row for row in result['sources']}
    # Both exact passages fit comfortably. A global similarity veto must not
    # remove one explicitly requested paper's provenance without explanation.
    # This tests attribution coverage, not two independent experiments: the
    # repeated words remain identical and receive no factual-support credit.
    assert set(by_document) == {'cedar', 'maple'}
    assert {row['id'] for row in result['sources']} == {'cedar-shared', 'maple-shared'}
    assert len({row['label'] for row in result['sources']}) == 2
    for document, source in by_document.items():
        assert source['filename'] == document + '.txt'
        assert source['content'] == shared
        assert source['excerpt_start'] == 0 and source['excerpt_end'] == len(shared)
    assert result['trace']['semantic_entailment_verified'] is False
    assert result['trace']['selection']['selection_proxy_is_verified_support'] is False


@pytest.mark.parametrize('text,section', [
    ('Figure 2: Drift storage, which stores records, is shown beside Beacon storage.', '2 Methods'),
    ('Drift storage, which stores records, supports retrieval.', 'References'),
    ('Drift storage, which stores records, was introduced by another system.', 'Related Work'),
    ('The system mentions Drift storage and Beacon storage in its architecture.', '2 Methods'),
    ('We discuss improving control flow and memory manage-\nment policies.', '5 Conclusion'),
])
def test_component_markers_exclude_mentions_metadata_and_split_nouns(text, section):
    from docthinker.server.research_component_coverage import functional_definitions
    assert functional_definitions({'content': text, 'section': section}) == []


def test_component_marker_spans_preserve_negation_and_chinese_without_claiming_truth():
    from docthinker.server.research_component_coverage import functional_definitions
    text = ('The Mirror buffer does not store payloads unless the owner opts in. '
            'The Beacon storage, which serves as a read-write datastore, cannot exceed its quota.')
    markers = functional_definitions({'content': text, 'section': '2 Methods'})
    assert {item['key'] for item in markers} == {'mirror buffer', 'beacon storage'}
    first = next(item for item in markers if item['key'] == 'mirror buffer')
    assert text[first['name_start']:first['name_end']] == 'Mirror buffer'
    assert text[first['predicate_start']:first['predicate_end']] == 'does not store'
    chinese = '环形缓冲区仅用于保存临时记录，未经确认不能持久化。'
    chinese_markers = functional_definitions({'content': chinese, 'section': '方法'})
    assert chinese_markers and chinese_markers[0]['key'] == '环形缓冲区'
    assert chinese[chinese_markers[0]['predicate_start']:chinese_markers[0]['predicate_end']] == '仅用于'


def test_named_component_definitions_compete_with_repeated_architecture_overviews():
    question = 'Compare Alder and Willow architectures and evaluation results for long-term assistants.'
    chunks = []
    for document in ('alder', 'willow'):
        chunks.append(_structured_source(document + '-overview', document, 0,
            f'1 Introduction\nWe combine a memory hierarchy with a controller in {document.title()}. '
            'The system evaluates long-term assistants and reports improved results. '
            'We compare architecture results with a fixed baseline only on the held-out set.'))
        for ordinal in range(1, 27):
            chunks.append(_structured_source(document + '-' + str(ordinal), document, ordinal,
                f'3 Evaluation\nAlder and Willow architectures and evaluation results for long-term assistants '
                f'were discussed in panel {ordinal}. The benchmark reports improvement only for this panel.'))
    body = _structured_source('alder-body', 'alder', 27,
        'Figure 2: Alder and Willow architectures and evaluation results for long-term assistants.\n'
        '2.2 Storage\nExternal storage refers to a disk-backed area outside the processor window. '
        'Its records must be copied into the active window before processing; it is not directly readable.')
    definitions = _structured_source('alder-components', 'alder', 28,
        '2.3 Components\nWe distinguish two types of storage: Tide storage, which holds immutable event history, '
        'and Beacon storage, which serves as a read-write datastore for overflow records. '
        'Neither component permits access unless the owner grants permission.')
    chunks += [body, definitions]
    result = _select_exact_source_pack(question, chunks, chunks[:-2])
    by_id = {row['id']: row['content'] for row in result['sources']}
    assert 'alder-components' in by_id
    assert 'Tide storage, which holds immutable event history' in by_id['alder-components']
    assert 'Beacon storage, which serves as a read-write datastore' in by_id['alder-components']
    assert 'unless the owner grants permission' in by_id['alder-components']
    assert 'alder-body' in by_id and 'not directly readable' in by_id['alder-body']
    assert result['trace']['candidate_count'] == 48
    assert result['trace']['selection']['defined_component_cells'] >= 3
    assert result['trace']['selection']['component_coverage_is_semantic_support'] is False
    # New component dimensions are source-bound; reingestion UUIDs cannot
    # silently change which exact definitions survive the same fixed budget.
    renamed = [{**row, 'id': f'clone-{len(chunks)-index}',
                'document_id': {'alder': 'z', 'willow': 'a'}[row['document_id']]}
               for index, row in enumerate(chunks)]
    repeated = _select_exact_source_pack(question, list(reversed(renamed)), list(reversed(renamed[:-2])))
    signature = lambda value: [(row['filename'], row['ordinal'], row['excerpt_start'], row['excerpt_end'], row['content'])
                               for row in value['sources']]
    assert signature(repeated) == signature(result)


def test_unnamed_unrelated_definitions_do_not_gain_mechanism_coverage():
    question = 'Explain the Alder indexing mechanism and its operating conditions.'
    own = _structured_source('alder-method', 'alder', 0,
        '2 Methods\nWe combine the Alder encoder with a searchable index. '
        'Each query passes five records to a generator only after an exact lookup.')
    unrelated = [_structured_source('glossary-' + str(index), 'orchard', index,
        f'2 Methods\nOrchard buffer refers to a bin for storing harvested apples in grove {index}. '
        'Harvest storage, which holds ripe pears, remains closed unless a farmer opens it.') for index in range(12)]
    result = _select_exact_source_pack(question, [own, *unrelated], [own])
    assert {row['document_id'] for row in result['sources']} == {'alder'}
    assert result['trace']['selection']['defined_component_cells'] == 0
