"""Scoped negation repair, not a general dependency/entailment parser."""
import pytest

from docthinker.server.research_query_facets import MAX_FACETS, build_query_facets


def clauses(query):
    result = build_query_facets(query)
    assert result['facets'][0]['kind'] == 'original'
    assert result['facets'][0]['text'] == query
    assert len(result['facets']) <= MAX_FACETS
    assert all(row['text'] in query for row in result['facets'])
    assert result['trace']['provider_calls'] == result['trace']['llm_tokens'] == 0
    return [row['text'] for row in result['facets'] if row['kind'] == 'clause']


@pytest.mark.parametrize('query,left,right', [
    ('Explain the queuing mechanism. Compare execution policies that do not share a scheduler.',
     'Explain the queuing mechanism', 'Compare execution policies that do not share a scheduler'),
    ('Explain why a measured delay is not a throughput limit. Compare the execution policies.',
     'Explain why a measured delay is not a throughput limit', 'Compare the execution policies'),
    ('Explain the transport mechanism. Describe experiments rather than speculating.',
     'Explain the transport mechanism', 'Describe experiments rather than speculating'),
    ('Compare protocols that never retry. Identify independent experimental controls.',
     'Compare protocols that never retry', 'Identify independent experimental controls'),
    ('Explain a controller at 0.25 Hz. Describe controls that do not change the load.',
     'Explain a controller at 0.25 Hz', 'Describe controls that do not change the load'),
])
def test_negation_in_one_complete_sentence_does_not_block_another(query, left, right):
    assert {left, right} <= set(clauses(query))


@pytest.mark.parametrize('query', [
    'Compare policy A rather than policy B, and explain why the costs differ.',
    'If samples are not independent, explain why the estimate changes, and compare the confidence intervals.',
    'Only under independent sampling. Explain transport. Describe limitations.',
    'Assume the records are not independent. Compare the estimators. Identify experimental controls.',
    'Assume all records share a cohort. Explain transport that does not use external inputs. Identify controls.',
    'Without matched samples. Compare the estimated effects. Describe experiments.',
])
def test_same_sentence_or_open_global_condition_is_not_removed(query):
    values = clauses(query)
    assert not any(value.startswith(('Compare the', 'Explain transport', 'Identify experimental', 'Describe experiments')) for value in values)
    assert len(values) <= 1


@pytest.mark.parametrize('query', [
    'Explain the actuator that does not use a spring. Explain why it remains stable.',
    'Describe the controller. Explain how its failures are not independent.',
    'Compare estimators that do not share samples. Describe these differences.',
    'Explain the protocol without retries. Describe the latter result.',
])
def test_newly_opened_boundary_does_not_detach_referential_request(query):
    assert clauses(query) == []  # Bound original span deduplicates the full view.


def test_repeated_explicit_noun_allows_a_new_boundary_without_inventing_subject():
    query = 'Explain the controller that does not retry. Describe the controller limitations.'
    assert 'Describe the controller limitations' in clauses(query)


def test_complete_local_conditional_request_does_not_block_later_named_request():
    query = 'If measurements are missing, explain uncertainty. Compare the calibration methods.'
    assert 'If measurements are missing, explain uncertainty' in clauses(query)
    assert 'Compare the calibration methods' in clauses(query)


@pytest.mark.parametrize('query,protected', [
    ('Explain drifts, e.g. Compare observations that do not agree. Identify controls.',
     'Explain drifts, e.g. Compare observations that do not agree'),
    ('Explain records from Dr. Why without using outside sources. Compare the data sets.',
     'Explain records from Dr. Why without using outside sources'),
    ('Explain effects at 0.25 Hz, i.e. Describe the measured response rather than a simulation. Identify controls.',
     'Explain effects at 0.25 Hz, i.e. Describe the measured response rather than a simulation'),
])
def test_new_guard_boundary_preserves_abbreviations_initials_and_decimals(query, protected):
    assert protected in clauses(query)


@pytest.mark.parametrize('query,left,right', [
    ('解释排队机制。比较不共享调度器的两种方法。', '解释排队机制', '比较不共享调度器的两种方法'),
    ('如果样本不独立，解释估计误差。比较两种校准方法。', '如果样本不独立，解释估计误差', '比较两种校准方法'),
    ('解释频率为0.25 Hz的系统。讨论尚未验证的限制。', '解释频率为0.25 Hz的系统', '讨论尚未验证的限制'),
])
def test_existing_chinese_sentence_boundaries_keep_local_negation(query, left, right):
    assert {left, right} <= set(clauses(query))


def test_preexisting_unguarded_literal_views_do_not_claim_resolved_coreference():
    query = 'Explain the transport mechanism. Describe its limitations. Identify experimental controls.'
    assert clauses(query) == ['Explain the transport mechanism', 'Describe its limitations', 'Identify experimental controls']
    result = build_query_facets(query)
    assert all('resolved_subject' not in row and 'semantic_independence_verified' not in row for row in result['facets'])


def test_original_chinese_question_mark_routing_is_unchanged():
    query = '索引如何合并输入？它为何能扩展？实验中是否已经验证泛化？'
    assert clauses(query) == ['索引如何合并输入', '它为何能扩展', '实验中是否已经验证泛化']
