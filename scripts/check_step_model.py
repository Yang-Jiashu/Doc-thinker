"""Private, one-shot StepFun connectivity check; no configuration is saved.

Run in an interactive terminal, then paste the key at the hidden prompt:
    python scripts/check_step_model.py --api step-plan
    python scripts/check_step_model.py --api standard

The key is never accepted through arguments or environment variables. Only the
two fixed StepFun API destinations below can receive it. This is a personal
synthetic smoke check, not permission to share a provider account or deploy it.
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys
import time
import warnings

import httpx

MODEL = "step-5-preview"
API_BASES = {
    "step-plan": "https://api.stepfun.com/step_plan/v1",
    "standard": "https://api.stepfun.com/v1",
}
MAX_RESPONSE_BYTES = 64 * 1024
TIMEOUT_SECONDS = 60
FINISH_REASONS = {"stop", "length", "tool_calls", "function_call", "content_filter"}
USAGE_FIELDS = {"prompt_tokens", "completion_tokens", "total_tokens"}
ERROR_CODES = {
    "invalid_api_key", "insufficient_credit", "insufficient_quota",
    "project_credit_limit_exceeded", "member_project_credit_limit_exceeded",
    "rate_limit_exceeded", "permission_denied", "access_denied",
    "model_not_found", "invalid_request_error", "authentication_error",
}


class ConfigurationError(ValueError):
    pass


class ResponseTooLarge(ValueError):
    pass


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message):
        # Do not echo unknown arguments: someone could accidentally paste a key.
        raise ConfigurationError("Invalid command options")


def _result(status: str = "not_checked") -> dict:
    return {
        "status": status, "model": MODEL,
        "models_http_status": None, "chat_http_status": None,
        "model_available": None, "model_matches": False,
        "answer_present": False, "finish_reason": None,
        "usage": {}, "latency_ms": 0,
    }


async def _request(client: httpx.AsyncClient, method: str, url: str, **kwargs):
    async with client.stream(method, url, **kwargs) as response:
        # Status is numeric; no body, URL, headers or exception text is printed.
        status = response.status_code
        pieces, size = [], 0
        async for piece in response.aiter_bytes():
            size += len(piece)
            if size > MAX_RESPONSE_BYTES:
                raise ResponseTooLarge("Provider response exceeded the fixed limit")
            pieces.append(piece)
        try:
            parsed = json.loads(b"".join(pieces))
        except (ValueError, UnicodeError, RecursionError):
            parsed = None
        return status, parsed


def _safe_usage(value) -> dict:
    if not isinstance(value, dict):
        return {}
    return {name: item for name, item in value.items() if name in USAGE_FIELDS
            and type(item) is int and 0 <= item <= 1_000_000_000}


def _safe_error(value) -> dict:
    """Classify a rejection without exposing provider prose or arbitrary codes.

    Hints are deliberately labelled as hints, not diagnoses. Providers sometimes
    return an unregistered code, HTML or a message containing request secrets.
    None of that content is emitted, even when the HTTP status is an error.
    """
    error = value.get("error") if isinstance(value, dict) else None
    if not isinstance(error, dict):
        return {"structured_error": False, "error_code": None, "error_hints": []}
    code = error.get("code")
    safe_code = code if isinstance(code, str) and code in ERROR_CODES else "other"
    message = error.get("message")
    message = message.lower() if isinstance(message, str) else ""
    patterns = {
        "credit_or_quota": ("insufficient credit", "insufficient balance", "quota", "额度", "余额"),
        "account_verification": ("real-name verification", "face verification", "实名认证", "人脸认证"),
        "model_permission": ("model permission", "model access", "模型权限", "模型访问"),
        "client_restriction": ("coding tool", "coding agent", "client not allowed", "unsupported client", "客户端", "编程工具"),
        "subscription_permission": ("subscription", "step plan", "step_plan", "订阅", "套餐"),
        "authentication": ("invalid api key", "invalid token", "authentication", "无效密钥"),
        "request_parameter": ("invalid parameter", "unsupported parameter", "invalid request", "参数"),
    }
    result = {"structured_error": True, "error_code": safe_code,
              "error_hints": [name for name, words in patterns.items()
                              if any(word in message for word in words)]}
    # Preserve an explicit, observed account prerequisite instead of flattening
    # it into a generic plan/permission hint. Arbitrary provider types, prose,
    # URLs and account identifiers must still never reach the output.
    if error.get("type") == "real_name_required":
        result.update(error_type="real_name_required",
                      required_action="complete_account_real_name_verification")
    return result


async def check_model(api: str, key: str, *, transport=None, clock=time.monotonic) -> dict:
    """Return sanitized metadata; tests may inject a local mock transport only."""
    result = _result()
    if (api not in API_BASES or not isinstance(key, str) or not key
            or len(key) > 4096 or any(ord(char) <= 32 or ord(char) == 127 for char in key)):
        result["status"] = "configuration_error"
        return result
    started = clock()

    async def run():
        async with httpx.AsyncClient(
            transport=transport, trust_env=False, follow_redirects=False,
            timeout=httpx.Timeout(TIMEOUT_SECONDS, connect=10),
            headers={"Authorization": "Bearer " + key},
        ) as client:
            try:
                code, models = await _request(client, "GET", API_BASES[api] + "/models")
                result["models_http_status"] = code
                if code == 200 and isinstance(models, dict) and isinstance(models.get("data"), list):
                    result["model_available"] = any(
                        isinstance(item, dict) and item.get("id") == MODEL for item in models["data"]
                    )
            except (httpx.HTTPError, ResponseTooLarge):
                # Some plans omit the model-list API; exactly one chat request
                # below is authoritative. This is not a retry of either call.
                pass
            code, chat = await _request(
                client, "POST", API_BASES[api] + "/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "只回答：连接成功。"}],
                    "max_tokens": 512, "stream": False, "reasoning_effort": "low",
                },
            )
            result["chat_http_status"] = code
            if code != 200:
                result["status"] = "http_error"
                result.update(_safe_error(chat))
                return
            if not isinstance(chat, dict):
                result["status"] = "invalid_response"
                return
            result["model_matches"] = chat.get("model") == MODEL
            if result["model_matches"]:
                result["model_available"] = True
            result["usage"] = _safe_usage(chat.get("usage"))
            choices = chat.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                result["status"] = "invalid_response"
                return
            first = choices[0]
            reason = first.get("finish_reason")
            result["finish_reason"] = reason if isinstance(reason, str) and reason in FINISH_REASONS else "other"
            message = first.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            result["answer_present"] = isinstance(content, str) and bool(content.strip())
            if not result["model_matches"]:
                result["status"] = "unexpected_model"
            elif not result["answer_present"]:
                result["status"] = "empty_answer"
            elif reason != "stop":
                result["status"] = "incomplete_answer"
            else:
                result["status"] = "ok"

    try:
        # A whole-check deadline bounds both requests and streamed response time.
        await asyncio.wait_for(run(), timeout=TIMEOUT_SECONDS)
    except (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException):
        result["status"] = "timeout"
    except ResponseTooLarge:
        result["status"] = "response_too_large"
    except httpx.HTTPError:
        result["status"] = "network_error"
    except Exception:
        result["status"] = "check_failed"
    result["latency_ms"] = max(0, int((clock() - started) * 1000))
    return result


def main(argv=None) -> int:
    parser = SafeArgumentParser(description=__doc__)
    parser.add_argument("--api", default="step-plan", help="step-plan (default) or standard")
    try:
        args, unknown = parser.parse_known_args(argv)
        if unknown or args.api not in API_BASES:
            raise ConfigurationError("Invalid command options")
        if not sys.stdin.isatty() or not sys.stderr.isatty():
            raise ConfigurationError("A private interactive terminal is required")
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            key = getpass.getpass("API key（隐藏输入，仅本次使用）：")
        result = asyncio.run(check_model(args.api, key))
    except (ConfigurationError, getpass.GetPassWarning, EOFError, KeyboardInterrupt):
        result = _result("configuration_error")
    except Exception:
        result = _result("check_failed")
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":")))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
