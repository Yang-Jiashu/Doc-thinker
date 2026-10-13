"""Synthetic-only tests: never load a real key, dotenv or provider endpoint."""
import asyncio
import json
import sys

import httpx
import pytest

from scripts import check_step_model as check

KEY = "synthetic-test-key-must-not-appear-in-output"


def completion(**updates):
    return {
        "model": check.MODEL,
        "choices": [{"finish_reason": "stop", "message": {"content": "synthetic answer"}}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        **updates,
    }


@pytest.mark.parametrize("api", ["step-plan", "standard"])
async def test_exact_fixed_provider_and_two_requests_only(api, capsys):
    calls = []
    def provider(request):
        calls.append(request)
        assert request.url.scheme == "https" and request.url.host == "api.stepfun.com"
        assert request.headers["authorization"] == "Bearer " + KEY
        if request.method == "GET":
            assert str(request.url) == check.API_BASES[api] + "/models"
            return httpx.Response(200, json={"data": [{"id": check.MODEL}]})
        assert str(request.url) == check.API_BASES[api] + "/chat/completions"
        assert json.loads(request.content) == {
            "model": "step-5-preview", "messages": [{"role": "user", "content": "只回答：连接成功。"}],
            "max_tokens": 512, "stream": False, "reasoning_effort": "low",
        }
        return httpx.Response(200, json=completion())
    output = await check.check_model(api, KEY, transport=httpx.MockTransport(provider))
    assert len(calls) == 2 and [call.method for call in calls] == ["GET", "POST"]
    assert output["status"] == "ok" and output["model_available"] and output["answer_present"]
    assert output["model_matches"] and output["finish_reason"] == "stop"
    assert output["usage"] == {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}
    assert not capsys.readouterr().out
    assert KEY not in json.dumps(output) and "synthetic answer" not in json.dumps(output)


@pytest.mark.parametrize("model_status,body", [(404, b"not available"), (401, KEY.encode()), (200, b"invalid JSON")])
async def test_missing_models_list_does_not_retry_or_block_single_chat(model_status, body):
    calls = []
    def provider(request):
        calls.append(request.method)
        return httpx.Response(model_status, content=body) if request.method == "GET" else httpx.Response(200, json=completion())
    result = await check.check_model("step-plan", KEY, transport=httpx.MockTransport(provider))
    assert calls == ["GET", "POST"]
    assert result["status"] == "ok" and result["models_http_status"] == model_status
    assert KEY not in json.dumps(result)


@pytest.mark.parametrize("status", [302, 401, 403, 429, 500])
async def test_provider_http_error_is_sanitized_without_redirect_or_retry(status):
    calls = []
    def provider(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(status, content=KEY.encode(), headers={"Location": "https://attacker.invalid/" + KEY})
    result = await check.check_model("step-plan", KEY, transport=httpx.MockTransport(provider))
    assert len(calls) == 2 and result["status"] == "http_error"
    assert result["chat_http_status"] == status and not result["answer_present"]
    assert KEY not in json.dumps(result) and "attacker" not in json.dumps(result)


@pytest.mark.parametrize("error,expected", [
    ({"code": "permission_denied", "message": "Step Plan subscription does not allow this coding tool " + KEY},
     {"structured_error": True, "error_code": "permission_denied",
      "error_hints": ["client_restriction", "subscription_permission"]}),
    ({"code": KEY, "message": "Insufficient credit " + KEY},
     {"structured_error": True, "error_code": "other", "error_hints": ["credit_or_quota"]}),
    ({"code": {"nested": KEY}, "message": [KEY]},
     {"structured_error": True, "error_code": "other", "error_hints": []}),
    (KEY, {"structured_error": False, "error_code": None, "error_hints": []}),
])
async def test_error_diagnostics_are_fixed_labels_not_provider_content(error, expected):
    def provider(request):
        return httpx.Response(200, json={"data": []}) if request.method == "GET" else httpx.Response(403, json={"error": error})
    result = await check.check_model("step-plan", KEY, transport=httpx.MockTransport(provider))
    assert result["status"] == "http_error" and result["chat_http_status"] == 403
    assert all(result[name] == item for name, item in expected.items())
    assert KEY not in json.dumps(result) and "does not allow" not in json.dumps(result)


async def test_real_name_prerequisite_is_actionable_without_echoing_private_prose():
    def provider(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        return httpx.Response(403, json={"error": {
            "type": "real_name_required", "code": None,
            "message": "real-name verification is required for your free step plan. "
                       "Complete face verification " + KEY,
        }})
    result = await check.check_model("step-plan", KEY, transport=httpx.MockTransport(provider))
    assert result["error_type"] == "real_name_required"
    assert result["required_action"] == "complete_account_real_name_verification"
    assert "account_verification" in result["error_hints"]
    assert KEY not in json.dumps(result) and "free step plan" not in json.dumps(result)


def test_unregistered_error_type_is_not_emitted():
    result = check._safe_error({"error": {"type": KEY, "message": KEY}})
    assert "error_type" not in result and "required_action" not in result
    assert KEY not in json.dumps(result)


@pytest.mark.parametrize("body,status", [
    ({"model": "another-model"}, "unexpected_model"),
    ({"choices": [{"finish_reason": "length", "message": {"content": "some text"}}]}, "incomplete_answer"),
    ({"choices": [{"finish_reason": "length", "message": {"content": "", "reasoning_content": "thoughts only"}}]}, "empty_answer"),
    ({"choices": [{"finish_reason": "stop", "message": {"content": "  "}}]}, "empty_answer"),
    ({"choices": []}, "invalid_response"),
    ({"choices": [{"finish_reason": KEY, "message": {"content": KEY}}]}, "incomplete_answer"),
])
async def test_only_expected_model_complete_nonempty_answer_counts_as_success(body, status):
    def provider(request):
        return httpx.Response(200, json={"data": []} if request.method == "GET" else completion(**body))
    result = await check.check_model("standard", KEY, transport=httpx.MockTransport(provider))
    assert result["status"] == status
    assert KEY not in json.dumps(result) and "thoughts only" not in json.dumps(result)


async def test_usage_is_numeric_allowlist_only():
    def provider(request):
        return httpx.Response(200, json={"data": []} if request.method == "GET" else completion(usage={
            "prompt_tokens": -1, "completion_tokens": True, "total_tokens": 15,
            "extra": KEY, "nested": {"credential": KEY},
        }))
    result = await check.check_model("step-plan", KEY, transport=httpx.MockTransport(provider))
    assert result["usage"] == {"total_tokens": 15}
    assert KEY not in json.dumps(result)


async def test_client_disables_environment_proxies_redirects_and_configures_timeout(monkeypatch):
    original_client = httpx.AsyncClient
    configurations = []
    def client(**kwargs):
        configurations.append(kwargs)
        return original_client(**kwargs)
    monkeypatch.setattr(check.httpx, "AsyncClient", client)
    def provider(request):
        return httpx.Response(200, json={"data": []} if request.method == "GET" else completion())
    result = await check.check_model("step-plan", KEY, transport=httpx.MockTransport(provider))
    assert result["status"] == "ok" and len(configurations) == 1
    assert configurations[0]["trust_env"] is False
    assert configurations[0]["follow_redirects"] is False
    assert configurations[0]["timeout"].read == 60


@pytest.mark.parametrize("failure,status", [
    (httpx.ReadTimeout(KEY), "timeout"), (httpx.ConnectError(KEY), "network_error"),
])
async def test_network_errors_never_echo_messages_or_retry(failure, status):
    calls = []
    def provider(request):
        calls.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        raise failure
    result = await check.check_model("step-plan", KEY, transport=httpx.MockTransport(provider))
    assert calls == ["GET", "POST"] and result["status"] == status
    assert KEY not in json.dumps(result)


async def test_whole_check_has_hard_deadline(monkeypatch):
    monkeypatch.setattr(check, "TIMEOUT_SECONDS", 0.01)
    calls = []
    async def provider(request):
        calls.append(request.method)
        await asyncio.sleep(0.1)
        return httpx.Response(200, json={"data": []})
    result = await check.check_model("step-plan", KEY, transport=httpx.MockTransport(provider))
    assert result["status"] == "timeout" and calls == ["GET"]


async def test_chat_response_over_fixed_byte_limit_is_not_logged_or_retried():
    calls = []
    def provider(request):
        calls.append(request.method)
        return httpx.Response(200, json={"data": []}) if request.method == "GET" else httpx.Response(200, content=b"x" * (check.MAX_RESPONSE_BYTES + 1))
    result = await check.check_model("step-plan", KEY, transport=httpx.MockTransport(provider))
    assert calls == ["GET", "POST"] and result["status"] == "response_too_large"


@pytest.mark.parametrize("api,key", [("https://attacker.invalid", KEY), ("step-plan", ""), ("standard", KEY + "\n")])
async def test_invalid_destination_or_key_never_sends_request(api, key):
    def provider(_request):
        raise AssertionError("Invalid configuration must not send a key")
    result = await check.check_model(api, key, transport=httpx.MockTransport(provider))
    assert result["status"] == "configuration_error" and KEY not in json.dumps(result)


@pytest.mark.parametrize("argv", [[KEY], ["--api", KEY], ["--key", KEY], ["--api"]])
def test_cli_never_echoes_credential_pasted_into_arguments(argv, capsys):
    assert check.main(argv) == 1
    captured = capsys.readouterr()
    assert KEY not in captured.out + captured.err
    assert json.loads(captured.out)["status"] == "configuration_error"


def test_cli_refuses_noninteractive_key_input_and_ignores_environment(monkeypatch, capsys):
    monkeypatch.setenv("STEPFUN_API_KEY", KEY)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(check.getpass, "getpass", lambda _: pytest.fail("Must not prompt without TTY"))
    assert check.main([]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "configuration_error"


def test_cli_hidden_prompt_emits_only_sanitized_result(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    monkeypatch.setattr(check.getpass, "getpass", lambda _: KEY)
    async def checker(api, key):
        assert api == "step-plan" and key == KEY
        return {**check._result(), "status": "ok"}
    monkeypatch.setattr(check, "check_model", checker)
    assert check.main([]) == 0
    captured = capsys.readouterr()
    assert KEY not in captured.out + captured.err
    assert json.loads(captured.out)["status"] == "ok"
