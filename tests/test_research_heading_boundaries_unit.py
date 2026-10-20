"""PDF typography must not turn numeric prose into discarded section titles."""
import pytest

from docthinker.server.research_sentence_windows import (
    evidence_units, is_section_heading, section_heading_spans, sentence_spans,
)
from docthinker.server.research_evidence_selection import prepare_evidence_candidates
from docthinker.server.research_query_facets import build_query_facets


@pytest.mark.parametrize('line', [
    '12 hours in advance. If corrected for transmission delays, the estimates would be',
    '7 samples from the control group were excluded because of contamination',
    '1.5 seconds per sample on the validation device',
    '80 percent of participants completed the intervention',
    '4 Patients were excluded from the study',
    '3.2 MAE [K]',
    '1 Results were not adjusted for missing observations',
    '42 次实验未能证明无误差',
    '10 ms 以下仅在固定负载时成立',
    '12 GPU hours under a fixed training budget',
    '32 GB of RAM with a fixed batch size',
    '8 CPU Hours Per Trial',
    '512 MiB of available memory',
    '4 GHz with disabled frequency scaling',
    '5 Patients received placebo in the control arm',
    '3 Operators ran the same workload',
    '3 Results improved under the same workload',
    '2.3 Results on the device were inconsistent',
    '2 Results on the benchmark. If adjusted for drift',
])
def test_numbered_pdf_prose_is_not_a_heading(line):
    assert not is_section_heading(line)


@pytest.mark.parametrize('line', [
    '2 Methods', '2.3 Experimental setup', 'IV. Evaluation',
    '3 limitations', '2. New Method', '## New Method', 'NEW METHOD', 'New Method',
    '3.2 Results on Model 1.5',
    '2.4 Experiments with U.S. data',
    '5.1 Analysis of error propagation',
    'IV. Training details',
    '3.2 Ablation study',
    '2 Methods for acoustic modelling',
    '4 Results on Fig. 2 measurements',
    '3.2 Generalization Performance',
])
def test_real_structural_boundaries_still_recognized(line):
    assert is_section_heading(line)


def test_numeric_line_wrap_does_not_cut_result_from_its_qualifier():
    text = ('The fictional service takes nine minutes to prepare all forecasts up to\n'
            '12 hours in advance. If corrected for transmission delays, the estimates would be\n'
            'different from those reported in the table. A separate analysis concerns accuracy.')
    spans = sentence_spans(text)
    units = evidence_units(text)
    assert all(text[left:right] for left, right in spans)
    assert any('takes nine minutes' in text[left:right] and 'different from those reported' in text[left:right]
               for left, right in units)
    assert ''.join(text[left:right] for left, right in units) == text


def test_candidate_builder_keeps_wrapped_latency_and_condition_as_source_text():
    text = ('The fictional service takes nine minutes to prepare all forecasts up to\n'
            '12 hours in advance. If corrected for transmission delays, the estimates would be\n'
            'different from those reported in the table.')
    chunks = [{'id': 'chunk', 'document_id': 'doc', 'filename': 'fictional.txt',
               'ordinal': 0, 'page': 1, 'content': text}]
    query = 'Explain how the fictional service prepares forecasts and corrects transmission delays.'
    plan = build_query_facets(query, chunks)
    result = prepare_evidence_candidates(query, chunks, [], plan, {})
    assert result['candidates']
    assert any('takes nine minutes' in row['content'] and 'different from those reported' in row['content']
               for row in result['candidates'])
    for row in result['candidates']:
        assert row['content'] == text[row['excerpt_start']:row['excerpt_end']]


def test_section_boundary_still_prevents_cross_section_condition_binding():
    text = 'The intervention improved accuracy.\n2.3 Experimental setup\nHowever, the next study used different participants.'
    assert not any('improved accuracy' in text[left:right] and 'However' in text[left:right]
                   for left, right in evidence_units(text))


@pytest.mark.parametrize('line', [
    '2 New Method', '2. New Method', 'IV. New Method', 'New Method', 'NEW METHOD', '2.3 Experimental setup',
])
@pytest.mark.parametrize('prefix', ['The observed quantity includes\n', 'The device was referenced as Fig.\n'])
def test_typography_after_unfinished_prose_is_not_a_boundary(prefix, line):
    # The isolated line looks structural. The actual soft-wrap context is not.
    assert is_section_heading(line)
    text = prefix + line + '\nand is evaluated under the same conditions.'
    assert section_heading_spans(text) == []
    assert evidence_units(text) == [(0, len(text))]


@pytest.mark.parametrize('prefix', ['', '\n', 'The first result was measured.\n', 'Unfinished prose\n\n'])
def test_real_numbered_heading_has_document_or_paragraph_context(prefix):
    heading = '2.3 Experimental setup\n'
    text = prefix + heading + 'The new study uses different participants.'
    assert (len(prefix), len(prefix) + len(heading)) in section_heading_spans(text)


def test_explicit_markdown_can_introduce_a_boundary_after_unfinished_prose():
    prefix = 'A parser left the earlier paragraph incomplete\n'
    heading = '## New Method\n'
    text = prefix + heading + 'The new study uses different participants.'
    assert section_heading_spans(text) == [(len(prefix), len(prefix) + len(heading))]


@pytest.mark.parametrize('line', [
    '12 GPU hours under a fixed training budget',
    '32 GB of RAM with a fixed batch size',
    '5 Patients received placebo in the control arm',
    '2 New Method',
    '2. New Method',
    'IV. New Method',
    'NEW METHOD',
])
def test_candidate_builder_does_not_drop_title_like_soft_wrap(line):
    text = 'The recorded observation includes\n' + line + '\nand remains subject to the same control conditions.'
    events = []
    chunks = [{'id': 'chunk', 'document_id': 'doc', 'filename': 'example.txt',
               'ordinal': 0, 'content': text}]
    query = 'What observation and control conditions were recorded?'
    result = prepare_evidence_candidates(query, chunks, [], build_query_facets(query, chunks), {},
                                         diagnostic_callback=events.append)
    assert result['candidates']
    assert any(row['content'] == text for row in result['candidates'])
    assert not any(row.get('reason') == 'section_heading' for row in events)


@pytest.mark.parametrize('line', ['2. New Method', 'IV. New Method'])
def test_rejected_soft_wrap_at_chunk_end_cannot_be_rediscovered_as_a_heading(line):
    text = 'The compared annotation is identified as\n' + line
    query = 'What annotation was compared?'
    chunks = [{'id': 'chunk', 'document_id': 'doc', 'filename': 'example.txt',
               'ordinal': 0, 'content': text}]
    events = []
    result = prepare_evidence_candidates(query, chunks, [], build_query_facets(query, chunks), {},
                                         diagnostic_callback=events.append)
    assert section_heading_spans(text) == []
    assert evidence_units(text) == [(0, len(text))]
    assert any(row['content'] == text for row in result['candidates'])
    assert not any(row.get('reason') == 'section_heading' for row in events)
    assert all(row['content'] == text[row['excerpt_start']:row['excerpt_end']]
               for row in result['candidates'])


@pytest.mark.parametrize('line', [
    '12 GPU hours under a fixed training budget',
    '32 GB of RAM with a fixed batch size',
    '5 Patients received placebo in the control arm',
])
def test_quantity_or_participant_fact_at_chunk_start_remains_evidence(line):
    text = line + '\nand the reported result applies only to that allocation.'
    chunks = [{'id': 'chunk', 'document_id': 'doc', 'filename': 'example.txt',
               'ordinal': 0, 'content': text}]
    query = 'What allocation limits the reported result?'
    result = prepare_evidence_candidates(query, chunks, [], build_query_facets(query, chunks), {})
    assert section_heading_spans(text) == []
    assert any(row['content'] == text for row in result['candidates'])


def test_original_review_reproduction_retains_training_and_evaluation_conditions():
    text = ('The method consumes\n'
            '12 GPU hours under a fixed training budget\n'
            'and requires no accelerator during evaluation.')
    chunks = [{'id': 'example', 'document_id': 'doc', 'filename': 'example.txt',
               'ordinal': 0, 'content': text}]
    query = 'What training budget and evaluation accelerator does the method require?'
    events = []
    result = prepare_evidence_candidates(query, chunks, [], build_query_facets(query, chunks), {},
                                         diagnostic_callback=events.append)
    assert evidence_units(text) == [(0, len(text))]
    assert any(row['content'] == text for row in result['candidates'])
    assert not any(row.get('reason') == 'section_heading' for row in events)


@pytest.mark.parametrize('transition', [
    'To overcome this', 'To address this', 'To resolve this',
    'To handle this', 'To avoid this', 'To mitigate this',
])
def test_referential_operation_stays_bound_to_its_named_task_not_the_next_task(transition):
    antecedent = ('AmberSet preprocessing uses pattern-valued labels, so a concrete answer string '
                  'is not directly available for training.')
    operation = (transition + ', we retrieve 2048 records and select the string matching the label pattern. '
                 'If no string matches, we enumerate permitted label variants without using held-out answers.')
    next_task = ('BirchSet evaluation uses a separate held-out split with written answer strings. '
                 'Its protocol measures exact-match accuracy without generating training labels.')
    text = antecedent + '\n' + operation + '\n' + next_task
    units = [text[left:right] for left, right in evidence_units(text)]
    assert any(antecedent in unit and operation in unit for unit in units)
    assert all(antecedent in unit for unit in units if 'we retrieve 2048 records' in unit)
    # A conservative unit may also include the following task. The necessary
    # invariant is that its name never replaces the operation's antecedent.
    assert next_task in ''.join(units)

    chunks = [{'id': 'two-tasks', 'document_id': 'source', 'filename': 'tasks.txt',
               'ordinal': 0, 'page': 7, 'content': text}]
    query = 'Explain AmberSet preprocessing and BirchSet evaluation conditions.'
    pool = prepare_evidence_candidates(query, chunks, [], build_query_facets(query, chunks), {})
    operations = [row for row in pool['candidates'] if 'we retrieve 2048 records' in row['content']]
    assert operations
    for row in operations:
        assert antecedent in row['content']
        assert 'If no string matches' in row['content']
        assert 'without using held-out answers' in row['content']
        assert row['page'] == 7 and row['source_id'] == 'two-tasks'
    assert all(row['content'] == text[row['excerpt_start']:row['excerpt_end']]
               for row in pool['candidates'])


def test_chinese_purpose_reference_keeps_the_prior_task_and_its_condition():
    antecedent = '青铜集的预处理标签是模式字符串，因此不能直接得到训练答案。'
    operation = '为此，我们检索2048条记录，选择符合标签模式的字符串。只有不存在匹配时才枚举允许的标签变体。'
    next_task = '白桦集的评测使用单独的留出集合，并以逐字匹配统计准确率。'
    text = antecedent + operation + next_task
    units = [text[left:right] for left, right in evidence_units(text)]
    assert any(antecedent in unit and operation in unit for unit in units)
    assert all(antecedent in unit for unit in units if '我们检索2048条记录' in unit)
    assert ''.join(units) == text


def test_leading_reference_without_antecedent_retains_source_and_a_gap_not_a_guessed_task():
    from docthinker.server.research_lexical_coverage import select_lexical_coverage
    from docthinker.server.research_retrieval import build_retrieval_plan

    fragment = ('To overcome this, we retrieve 2048 records and select a matching label string. '
                'If no string matches, we enumerate permitted variants without consulting held-out answers.')
    text = fragment + '\n## BirchSet Evaluation\nBirchSet reports exact-match accuracy on its held-out split.'
    chunks = [{'id': 'page-fragment', 'document_id': 'source', 'filename': 'tasks.txt',
               'ordinal': 4, 'page': 8, 'content': text}]
    query = 'Explain how the preprocessing retrieves 2048 records and selects a matching label string.'
    baseline = build_retrieval_plan(query, chunks)['sources']
    result = select_lexical_coverage(query, baseline, chunks)
    sources = result['sources']
    assert any(fragment in row['content'] for row in sources)
    assert all('AmberSet' not in row['content'] for row in sources)
    assert any(gap['code'] == 'source_antecedent_unverified' for gap in result['gaps'])
    assert all(row['sentence_boundary_preserved'] is True for row in sources)
    assert all(row['content'] == text[row['excerpt_start']:row['excerpt_end']]
               and row['id'] == 'page-fragment' and row['page'] == 8 for row in sources)
    # A later named section is not the missing antecedent of an earlier step.
    assert not any('we retrieve 2048 records' in text[left:right] and 'BirchSet reports' in text[left:right]
                   for left, right in evidence_units(text))


def test_oversized_antecedent_operation_unit_is_not_cut_to_remove_its_condition():
    antecedent = ('AmberSet preprocessing uses pattern-valued labels and records '
                  + ', '.join('annotation field ' + str(index) for index in range(100))
                  + ', so a concrete training answer is not directly available.')
    operation = ('To overcome this, we retrieve 2048 records and select a matching label string. '
                 'If no string matches, the method may enumerate variants only when no held-out answer is used.')
    text = antecedent + '\n' + operation + '\nBirchSet evaluation reports exact-match accuracy on its own split.'
    bound = next(text[left:right] for left, right in evidence_units(text) if 'we retrieve 2048 records' in text[left:right])
    assert len(bound) > 1600 and len(text) < 8000
    assert antecedent in bound and operation in bound
    chunks = [{'id': 'long-unit', 'document_id': 'source', 'filename': 'tasks.txt',
               'ordinal': 0, 'page': 2, 'content': text}]
    query = 'Explain AmberSet preprocessing and BirchSet evaluation conditions.'
    pool = prepare_evidence_candidates(query, chunks, [], build_query_facets(query, chunks), {})
    assert all('we retrieve 2048 records' not in row['content'] for row in pool['candidates'])
    assert all(len(row['content']) <= 1600
               and row['content'] == text[row['excerpt_start']:row['excerpt_end']]
               for row in pool['candidates'])
