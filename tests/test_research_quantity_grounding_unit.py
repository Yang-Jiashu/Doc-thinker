import hashlib
import json

import pytest

from docthinker.server.research_quantity_grounding import build_quantity_index, check_quantity_grounding


def source(content, label='S1', **extra):
    return {'id': 'current-' + label, 'label': label, 'content': content, **extra}


def test_real_decision_brief_quantity_regression_sections_do_not_ground_caps():
    quote = ('1. Actions should be simple and easy to understand. Simple commands with a few options '
             'and concise documentation are easier for agents to use. This is a defining principle '
             'for all SWE-agent commands covered in Section 3. 2. Actions should be efficient.')
    text = '命令选项≤3、文档≤2行、定位+查看合并单步，无效编辑要求重试 [S2]'
    result = check_quantity_grounding(text, [source(quote, 'S2')])
    assert result['status'] == 'needs_review'
    assert [row['unit'] for row in result['diagnostics']] == ['options', 'lines']
    assert all(row['code'] == 'quantity_not_directly_grounded' for row in result['diagnostics'])


@pytest.mark.parametrize('text,quote', [
    ('3 options [S1]', 'There are 3 options.'),
    ('三个选项 [S1]', 'There are 3 options.'),
    ('两个任务 [S1]', 'There are 2 tasks.'),
    ('10轮次 [S1]', '十轮次'),
    ('错引2次 [S1]', '2 citation errors'),
    ('文档不超过2行 [S1]', 'at most 2 lines'),
    ('选项≤3 [S1]', 'no more than 3 options'),
    ('12.5% [S1]', '12.50 percent'),
])
def test_direct_same_unit_value_and_comparator(text, quote):
    result = check_quantity_grounding(text, [source(quote)])
    assert result['status'] == 'direct_numeric_match'
    assert result['semantic_verified'] is False


@pytest.mark.parametrize('text,quote', [
    ('3 options [S1]', 'Section 3. There are few options.'),
    ('3 options [S1]', '3 tasks and 2 options.'),
    ('≤3 options [S1]', '3 options were used in this example.'),
    ('12.5% [S1]', '12.5 percentage points'),
    ('10.7个百分点 [S1]', '10.7 percent'),
    ('60 minutes [S1]', '1 hour'),
    ('2 lines [S1]', 'The list has 2 items and concise documentation.'),
])
def test_insufficient_direct_evidence_is_review_not_inference(text, quote):
    result = check_quantity_grounding(text, [source(quote)])
    assert result['status'] == 'needs_review'
    assert result['diagnostics'][0]['code'] == 'quantity_not_directly_grounded'


def test_wrong_source_or_different_sentence_cannot_launder_quantity():
    result = check_quantity_grounding('3 options [S2]. 2 tasks [S1].', [
        source('3 options and 2 tasks'), source('few options', 'S2')])
    assert [row['code'] for row in result['diagnostics']] == [
        'quantity_not_directly_grounded', 'direct_numeric_match']
    assert result['diagnostics'][0]['source_labels'] == ['S2']
    no_citation = check_quantity_grounding('3 options. See evidence [S1].', [source('3 options')])
    assert no_citation['checked_quantities'] == 0


def test_explicit_proposal_is_separate_not_paper_fact():
    result = check_quantity_grounding('我们拟定选项≤3、文档≤2行 [S2]', [source('few options', 'S2')])
    assert result['status'] == 'proposed_not_source_claim'
    assert all(row['code'] == 'proposed_not_source_claim' for row in result['diagnostics'])
    assert not any(row['matches'] for row in result['diagnostics'])
    ordinary = check_quantity_grounding('本文提出选项≤3 [S2]', [source('few options', 'S2')])
    assert ordinary['status'] == 'needs_review'
    assert check_quantity_grounding('我们拟定：选项≤3 [S2]', [source('few options', 'S2')])['status'] == 'proposed_not_source_claim'


def test_proposal_role_and_clause_boundary_are_not_fact_verification():
    assert check_quantity_grounding('≤3 options [S1]', [source('few options')], field_role='proposal')['status'] == 'proposed_not_source_claim'
    result = check_quantity_grounding('我们拟定3 options [S1]，论文报告2 tasks [S2]', [
        source('few options'), source('several tasks', 'S2')])
    assert [row['code'] for row in result['diagnostics']] == ['proposed_not_source_claim', 'quantity_not_directly_grounded']


def test_few_and_section_numbers_are_not_explicit_quantity_claims():
    result = check_quantity_grounding('A few options; concise documentation [S1]. Section 3 [S1].', [source('Section 3')])
    assert result['status'] == 'no_quantity_claims'
    assert result['diagnostics'] == []


def test_metadata_contains_no_private_text_and_input_is_not_modified():
    text = 'PRIVATE-TEXT 3 options [S1]'
    sources = [source('SECRET-QUOTE 2 options')]
    before = json.dumps(sources)
    result = check_quantity_grounding(text, sources)
    encoded = json.dumps(result)
    assert 'PRIVATE-TEXT' not in encoded and 'SECRET-QUOTE' not in encoded
    assert json.dumps(sources) == before
    assert result['text_modified'] is False and result['execution_authorized'] is False
    assert result['extra_model_calls'] == 0
    start, end = result['diagnostics'][0]['span']
    assert text[start:end].strip() == '3 options'


def test_current_quote_hash_change_and_revocation_do_not_pass():
    digest = hashlib.sha256(b'3 options').hexdigest()
    for row in (source('2 options', quote_sha256=digest), source('3 options', revoked=True),
                source('3 options', status='withdrawn'), source('3 options', status='deleted')):
        result = check_quantity_grounding('3 options [S1]', [row])
        assert result['status'] == 'needs_review'
        assert not result['diagnostics'][0]['matches']


def test_unicode_limits_and_diagnostic_limit_are_explicit():
    result = check_quantity_grounding(('🙂' * 24001) + '3 options [S1]', [source('3 options')])
    assert result['truncated'] is True and result['status'] == 'needs_review'
    result = check_quantity_grounding('3 options [S1]; 2 tasks [S1]', [source('3 options;2 tasks')], max_diagnostics=1)
    assert len(result['diagnostics']) == 1 and result['truncated'] is True
    assert result['status'] == 'needs_review'
    result = check_quantity_grounding('3 options [S1]', [source('秘' * 4097)])
    assert result['status'] == 'needs_review'


def test_adversarial_shapes_fail_closed_without_typeerror_or_echo():
    for kwargs in ({'field_role': []}, {'field_role': {}}, {'max_diagnostics': True}, {'max_diagnostics': 100}):
        assert check_quantity_grounding('3 options [S1]', [source('3 options')], **kwargs)['status'] == 'needs_review'
    for sources in (None, {}, [None], [source('3 options'), source('2 options')]):
        result = check_quantity_grounding('3 options [S1]', sources)
        assert result['status'] == 'needs_review'


def test_english_prefix_and_grouped_citations_use_current_source_bundle():
    result = check_quantity_grounding('According to [S1], 3 options are offered.', [source('3 options')])
    assert result['status'] == 'direct_numeric_match'
    result = check_quantity_grounding('3 options [S2][S1]', [source('3 options'), source('few options', 'S2')])
    assert result['diagnostics'][0]['source_labels'] == ['S2', 'S1']
    assert result['diagnostics'][0]['matches'][0]['source_label'] == 'S1'
    spaced = check_quantity_grounding('3 options [S2] \t [S1]', [source('3 options'), source('few options', 'S2')])
    assert spaced['diagnostics'][0]['source_labels'] == ['S2', 'S1']
    assert spaced['diagnostics'][0]['matches'][0]['source_label'] == 'S1'


def test_unsupported_ranges_and_negative_numbers_cannot_launder_numeric_tail():
    for quote in ('1-3 options', '-3 options', '1–3 options', '−3 options', '1—3 options'):
        result = check_quantity_grounding('3 options [S1]', [source(quote)])
        assert result['status'] == 'needs_review'
        assert result['diagnostics'][0]['matches'] == []


def test_local_index_reuse_is_equivalent_without_source_text_or_cross_packet_cache():
    sources = [source('3 options;2 tasks')]
    index = build_quantity_index(sources)
    for text in ('3 options [S1]', '2 tasks [S1]', '4 tasks [S1]'):
        assert check_quantity_grounding(text, index) == check_quantity_grounding(text, sources)
    sources[0]['content'] = '4 tasks'
    # The index is explicitly a snapshot, not a mutable live authorization.
    assert check_quantity_grounding('4 tasks [S1]', build_quantity_index(sources))['status'] == 'direct_numeric_match'
    assert check_quantity_grounding('4 tasks [S1]', index)['status'] == 'needs_review'


def test_terminal_punctuation_attached_citations_bind_the_preceding_claim():
    for suffix in ('。[S1]', '。 [S1] [S2]', '. [S1] \t[S2]'):
        result = check_quantity_grounding('论文要求命令选项≤3、文档≤2行' + suffix,
            [source('few options and concise documentation'), source('Section 3', 'S2')])
        assert result['status'] == 'needs_review'
        assert result['checked_quantities'] == 2
        assert all(row['code'] == 'quantity_not_directly_grounded' for row in result['diagnostics'])
    result = check_quantity_grounding('原文报告12.5%，不是本产品结果。[S1]', [source('12.5 percent')])
    assert result['status'] == 'direct_numeric_match'
    assert result['checked_quantities'] == 1


def test_attached_suffix_does_not_borrow_an_independent_sentence_citation():
    for text in ('论文声称3 options。另一个独立句子引用材料[S1]',
                 '论文声称3 options。第二句2 tasks [S1]'):
        result = check_quantity_grounding(text, [source('3 options;2 tasks')])
        assert all(row['unit'] != 'options' for row in result['diagnostics'])
    # A suffix attached to the previous claim is not also a prefix authority
    # for the independent uncited claim that follows it.
    result = check_quantity_grounding('3 options。[S1] 下一句另有2 tasks。', [source('3 options;2 tasks')])
    assert result['checked_quantities'] == 1
    assert result['diagnostics'][0]['unit'] == 'options'
