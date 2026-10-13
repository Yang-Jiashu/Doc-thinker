"""High precision, whole-source fallback filtering; unknown text is retained."""
from copy import deepcopy

import pytest

from docthinker.server.research_fallback_sources import filter_fallback_sources, source_filter_reason

REFERENCES = ("[41] Smith, A. The calibrated experiment. Journal of Trials 12, 31–39 (2021).\n"
              "[42] Doe, B. An alternative method. arXiv preprint arXiv:2408.12345 (2024).")
AFFILIATIONS = ("1 Example Research\n2 Institute of Physics, University of Example, London\n"
                "3 Department of Engineering, Sample University, Cambridge, MA\n4 Example Labs")
AUTHOR_YEAR = (
    "Chen, Maria Ruiz, Alex Martín, and Chris An-\nders. 2020. Calibrated retrieval in long documents.\n"
    "In Proc. of ACL.\n"
    "Ofir Sample, Noah A. Example, and Mike Test. 2021. Sparse evidence for comparison.\n"
    "Journal of Information Retrieval, 21(140):1–67.\n"
    "Ori Example and Jonathan Test. 2023. Relevance selection in documents. ArXiv:2306.12345.")
BACKGROUND = (
    "An open-source implementation of both methods is forthcoming at https://example.test/software.\n"
    "1 Introduction\n"
    "Computational systems across a range of domains rely on broad information collections. "
    "With the emergence of language models, we are witnessing attempts to automate document analysis.\n"
    "Preprint. Under review.\narXiv:2404.12345v1 [cs.CL] 24 Apr 2024")


def row(text, index=1):
    return {'id': f'id-{index}', 'label': f'S{index}', 'content': text, 'page': 5,
            'document_id': 'doc', 'excerpt_start': 37, 'excerpt_end': 37 + len(text),
            'offset_unit': 'characters_in_source_chunk', 'truncated': True}


@pytest.mark.parametrize('text,reason', [
    (REFERENCES, 'numbered_reference_list'),
    (REFERENCES + '\n19', 'numbered_reference_list'),
    ('19\nReferences\n' + REFERENCES + '\n20', 'numbered_reference_list'),
    (AFFILIATIONS, 'numbered_affiliation_list'),
    ('1 清华大学计算机系\n2 北京大学物理学院\n3 中国科学院', 'numbered_affiliation_list'),
    ('Email: author@example.edu', 'isolated_contact'),
    ('邮箱：author@example.edu', 'isolated_contact'),
])
def test_pure_metadata_is_filtered(text, reason):
    result = filter_fallback_sources('Explain the paper method and limitations.', [row(text)])
    assert result['sources'] == []
    assert result['trace']['filtered_count'] == 1
    assert result['trace']['reason_counts'] == {reason: 1}
    assert result['trace']['provider_calls'] == result['trace']['llm_tokens'] == 0
    assert result['trace']['semantic_entailment_verified'] is False


@pytest.mark.parametrize('text', [
    'We measured the error at 0.3%.\n' + REFERENCES,
    REFERENCES + '\nThese results establish the reported measurement.',
    REFERENCES + '\n[43] Smith, C. A partial unknown entry',
    'Nature\n536, 441–445 (2016).\n' + REFERENCES,  # Unknown excerpt prefix: preserve.
    '[1] Heat the sample to 300 K.\n[2] Measure the output after 5 minutes.',
    '1 Prepare the Sample\n2 Record the Result\n3 Repeat the Experiment',
    '1 Example Research released a sensor\n2 Sample University uses it\n3 Acme makes instruments',
    '1 Example Research Reports\n2 Sample University Tests\n3 Acme Corporation Results.',
    '1 University X Reports Positive Results\n2 Institute Y Proves Theorem\n3 University Z Discovers Particles',
    '1 University X Has New Equipment\n2 Institute Y Uses Sensors\n3 University Z Produces Batteries',
    '1 University X Simulates Motion\n2 Institute Y Constructs Devices\n3 University Z Evaluates Performance',
    'Our university measured this quantity.\n' + AFFILIATIONS,
    '1 该大学采用新方法\n2 另一学院证明理论\n3 研究院位于北京',
    'Send measured alerts to author@example.edu after calibration.',
    'Contact author@example.edu for details of the experimental controls.',
    'Lead Author <author@example.edu> computed the measurements.',
    'We use [1] and [2] to compare the methods.',
    '[1] Smith, A. One bibliography entry. Nature 12, 1–3 (2020).',
    'Results\n0.01\n0.02\n19',
    '',
])
def test_mixed_narrative_unknown_and_numbered_steps_are_unchanged(text):
    source = row(text)
    before = deepcopy(source)
    result = filter_fallback_sources('Explain the paper method.', [source])
    assert result['sources'] == [before]
    assert result['sources'][0] is source
    assert result['trace']['filtered_count'] == 0


@pytest.mark.parametrize('question', [
    'List the references of this paper.', 'List the paper references.', 'List\nthe paper\nreferences.', 'What are the author affiliations?',
    'Which institutions are these authors affiliated with?', 'List the citations.',
    'What is the author email address?', 'Who wrote this paper?',
    '请列出参考文献。', '作者的机构有哪些？', '请列出论文引用。', '作者邮箱是什么？',
    '著者の所属機関と連絡先は？', '참고문헌과 소속은?',
])
def test_explicit_metadata_queries_preserve_all_sources(question):
    sources = [row(REFERENCES), row(AFFILIATIONS, 2), row('author@example.edu', 3)]
    result = filter_fallback_sources(question, sources)
    assert result['sources'] == sources
    assert result['trace']['status'] == 'metadata_query_retained'


def test_order_labels_offsets_and_unicode_bytes_of_retained_sources_are_unchanged():
    sources = [row('方法保留其条件。', 1), row(REFERENCES, 2), row('Unknown factual text.', 3)]
    before = deepcopy(sources)
    result = filter_fallback_sources('依据论文解释方法。', sources)
    assert result['sources'] == [before[0], before[2]]
    assert sources == before
    assert result['sources'][0] is sources[0]
    assert sum(len(item['content'].encode()) for item in result['sources']) <= sum(
        len(item['content'].encode()) for item in sources)


def test_bounds_retain_unknown_pack_or_oversized_source_without_cropping():
    sources = [row(REFERENCES, index) for index in range(9)]
    assert filter_fallback_sources('Explain paper.', sources)['sources'] == sources
    oversized = [row(REFERENCES + ' ' * 16000)]
    assert filter_fallback_sources('Explain paper.', oversized)['sources'] == oversized
    assert filter_fallback_sources('x' * 8193, sources[:1])['sources'] == sources[:1]


@pytest.mark.parametrize('text', [AUTHOR_YEAR, 'References\n' + AUTHOR_YEAR + '\n12'])
def test_author_year_reference_blocks_without_numbers_are_filtered(text):
    assert source_filter_reason('Compare methods and evaluations.', text) == 'author_year_reference_list'


@pytest.mark.parametrize('text', [
    'We measured the accuracy before comparing models.\n' + AUTHOR_YEAR,
    AUTHOR_YEAR + '\nThese results establish a significant difference.',
    AUTHOR_YEAR + '\nHowever, the method does not improve precision.',
    AUTHOR_YEAR + '\nAn unfinished excerpt of the next reference',
    AUTHOR_YEAR + '\nA. Sample and B. Test. 2024. An incomplete reference.',
    AUTHOR_YEAR.split('Ofir Sample')[0],  # One entry alone is ambiguous.
    'Accuracy is 80%.\n' + AUTHOR_YEAR,
])
def test_bibliography_with_substantive_prose_or_incomplete_tail_is_kept(text):
    assert source_filter_reason('Compare methods and evaluations.', text) is None


@pytest.mark.parametrize('question', [
    'List the references.', 'Show the paper citations.', '请列出参考文献。',
    'What is the historical background of this method?', 'Explain the introduction.',
])
def test_task_intent_does_not_suppress_requested_material(question):
    text = AUTHOR_YEAR if 'references' in question or 'citations' in question or '文献' in question else BACKGROUND
    assert source_filter_reason(question, text) is None


def test_only_all_generic_background_is_removed_for_a_technical_task():
    assert source_filter_reason('Compare methods and evaluation limitations.', BACKGROUND) == 'generic_background'
    body = 'With the emergence of language models, broad applications have attracted considerable attention.'
    assert source_filter_reason('Explain the method.', body) is None
    assert source_filter_reason('Explain the method.', body, section='1 Introduction') == 'generic_background'
    assert source_filter_reason('Explain the method.', body, section=None) is None


@pytest.mark.parametrize('evidence', [
    'The method computes a phase offset from adjacent samples.',
    'Only under fixed controls does the method improve recall.',
    'However, the system does not improve precision.',
    'Unless the index is sparse, these conclusions cannot be transferred.',
    'We sampled 80 participants across independently randomized cohorts.',
    'The accuracy was 81%.',
    'Retrieval is defined as selection of original documents.',
    'The open-source implementation uses a sparse matrix and is available online.',
    'With the emergence of language models, this method causes a measurable reduction in errors.',
    'With the emergence of language models, the algorithm reports reliable classifications.',
    'An open-source implementation outputs sparse matrices and is available online.',
    '该方法仅在固定条件下提升召回，不能改善准确率。',
    'The quantized dictionary is a technical representation.',  # Unknown, not generic framing.
])
def test_introduction_with_mechanism_result_condition_or_negation_stays_whole(evidence):
    text = BACKGROUND + '\n' + evidence
    source = row(text)
    result = filter_fallback_sources('Compare methods and evaluations.', [source])
    assert result['sources'] == [source] and result['sources'][0] is source
    assert source_filter_reason('Compare methods and evaluations.', text) is None


def test_new_hygiene_filters_keep_original_rows_offsets_and_gaps():
    sources = [row('If the sensor is calibrated, it reports accurate readings. Otherwise, it does not.', 1),
               row(BACKGROUND, 2), row(AUTHOR_YEAR, 3), row('Unknown narrative must remain untouched.', 4)]
    before = deepcopy(sources)
    result = filter_fallback_sources('Compare methods and evaluation conditions.', sources)
    assert result['sources'] == [before[0], before[3]]
    assert result['sources'][0] is sources[0] and result['sources'][1] is sources[3]
    assert result['trace']['reason_counts'] == {'generic_background': 1, 'author_year_reference_list': 1}
    assert sources == before
