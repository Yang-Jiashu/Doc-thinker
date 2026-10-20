"""Task-level allowances and endpoint-specific request fields, without networking."""
import asyncio

import pytest

from docthinker.providers.request_compat import sdk_chat_request_options
from docthinker.server.invite_budget import (
    completion_output_budget, current_output_tokens, has_output_budget,
)


def test_task_allowance_is_bounded_nested_and_restored_on_error():
    assert current_output_tokens() == 2048 and not has_output_budget()
    with pytest.raises(RuntimeError):
        with completion_output_budget(16384):
            assert current_output_tokens() == 16384 and has_output_budget()
            with completion_output_budget(1024):
                with completion_output_budget(16384):
                    assert current_output_tokens() == 1024
            assert current_output_tokens() == 16384
            raise RuntimeError('local test')
    assert current_output_tokens() == 2048 and not has_output_budget()


@pytest.mark.parametrize('limit', [0, -1, 16385, True, 4096.0, '16384', None])
def test_invalid_allowance_cannot_change_active_budget(limit):
    with pytest.raises(ValueError, match='Invalid completion budget'):
        with completion_output_budget(limit):
            pytest.fail('Invalid allowance accepted')
    assert current_output_tokens() == 2048


@pytest.mark.asyncio
async def test_parallel_users_do_not_share_output_budget_and_cancellation_restores():
    release = asyncio.Event()
    async def request(limit):
        with completion_output_budget(limit):
            await release.wait()
            return current_output_tokens()
    first = asyncio.create_task(request(16384))
    second = asyncio.create_task(request(2048))
    cancelled = asyncio.create_task(request(4096))
    await asyncio.sleep(0)
    assert current_output_tokens() == 2048
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    release.set()
    assert await asyncio.gather(first, second) == [16384, 2048]
    assert not has_output_budget()


@pytest.mark.parametrize('url', [
    'https://api.stepfun.com/v1', 'https://api.stepfun.com/v1/',
    'https://api.stepfun.com/step_plan/v1', 'https://api.stepfun.com:443/v1',
])
def test_known_step5_endpoint_shares_interactive_reasoning_policy(url):
    assert sdk_chat_request_options(url, 'step-5-preview', 8192, 0.1) == {
        'max_tokens': 8192, 'temperature': 0.1, 'reasoning_effort': 'low',
    }


@pytest.mark.parametrize('url,model', [
    ('https://api.stepfun.com/v1', 'step-3.5-flash'),
    ('https://api.stepfun.com/v1', 'some/step-5-preview'),
    ('https://gateway.example/v1', 'step-5-preview'),
    ('https://api.stepfun.com.attacker.invalid/v1', 'step-5-preview'),
    ('https://api.stepfun.com@attacker.invalid/v1', 'step-5-preview'),
    ('https://user:password@api.stepfun.com/v1', 'step-5-preview'),
    ('http://api.stepfun.com/v1', 'step-5-preview'),
    ('https://api.stepfun.com:9000/v1', 'step-5-preview'),
    ('https://api.stepfun.com/private', 'step-5-preview'),
    ('https://api.stepfun.com/v1?mode=proxy', 'step-5-preview'),
])
def test_vendor_fields_are_not_guessed_for_other_models_or_gateways(url, model):
    options = sdk_chat_request_options(url, model, 8192, 0.1)
    assert 'reasoning_effort' not in options and 'extra_body' not in options
    assert options['max_tokens'] == 8192


def test_other_provider_compatibility_is_preserved():
    assert sdk_chat_request_options('https://api.openai.com/v1', 'gpt-5.2', 8192, 0.1) == {
        'max_completion_tokens': 8192,
    }
    assert sdk_chat_request_options('https://dashscope.aliyuncs.com/compatible-mode/v1',
                                    'qwen-plus', 8192, 0.1) == {
        'max_tokens': 8192, 'temperature': 0.1, 'extra_body': {'enable_thinking': False},
    }


def test_provider_mapping_cannot_raise_small_nested_allowance():
    with completion_output_budget(256):
        for model in ('gpt-5.2', 'o3', 'step-5-preview', 'unknown-compatible-model'):
            options = sdk_chat_request_options('https://provider.example/v1', model,
                                                current_output_tokens(), 0.1)
            assert options.get('max_completion_tokens', options.get('max_tokens')) == 256


@pytest.mark.parametrize('model,field', [
    ('step-5-preview', 'max_tokens'), ('gpt-5.2', 'max_completion_tokens'),
    ('o3', 'max_completion_tokens'), ('unknown-compatible-model', 'max_tokens'),
])
def test_reviewed_profile_is_provider_neutral_and_cannot_raise_an_outer_grounded_cap(model, field):
    with completion_output_budget(16384):
        options = sdk_chat_request_options('https://provider.example/v1', model, current_output_tokens(), 0.1)
        assert options[field] == 16384
        assert 'reasoning_effort' not in options
    with completion_output_budget(8192):
        with completion_output_budget(16384):
            options = sdk_chat_request_options('https://provider.example/v1', model, current_output_tokens(), 0.1)
            assert options[field] == 8192
