"""Registered production review policy, without a provider call or paper IO."""
from docthinker.server.invite_budget import INVITE_REQUEST_BUDGET
from scripts.benchmark_goal_evidence import EXECUTION_SECONDS, execution_policy


def test_goal_evidence_runner_registers_the_production_review_allowance_and_no_override():
    policy = execution_policy()
    assert policy['timeout_seconds'] == EXECUTION_SECONDS == 240
    assert policy['public_request_budget'] == INVITE_REQUEST_BUDGET
    assert policy['public_request_budget']['max_output_tokens'] == 2048
    assert policy['public_request_budget']['max_grounded_output_tokens'] == 8192
    assert policy['public_request_budget']['max_reviewed_output_tokens'] == 16384
    assert policy['output_budget_policy'] == 'production_host_review_profile'
    assert policy['model_output_cap_override'] is policy['automatic_retry'] is False
    assert 'not a causal correctness guarantee' in policy['scope']
    policy['public_request_budget']['max_reviewed_output_tokens'] = 1
    assert INVITE_REQUEST_BUDGET['max_reviewed_output_tokens'] == 16384


def test_mobile_wall_allowance_matches_review_runner_without_changing_smaller_deadline_support():
    from docthinker.server.routers.mobile import QUERY_EXECUTION_SECONDS
    assert QUERY_EXECUTION_SECONDS == EXECUTION_SECONDS == 240
