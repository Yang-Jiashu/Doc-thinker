"""Conservative request options for the supported OpenAI-compatible transports.

A shared URL does not imply every provider accepts every optional parameter.
Unknown embedding models use their server default; callers must validate its
actual dimension instead of silently padding/truncating vectors.
"""

import re
from urllib.parse import urlsplit


def _model_name(model: str) -> str:
    return str(model or "").lower().rsplit("/", 1)[-1]


def is_dashscope_endpoint(base_url: str) -> bool:
    host = urlsplit(str(base_url or "")).hostname or ""
    return host.endswith(".maas.aliyuncs.com") or host in {
        "dashscope.aliyuncs.com",
        "dashscope-intl.aliyuncs.com",
        "dashscope-us.aliyuncs.com",
    }


def chat_request_options(
    model: str, max_tokens: int, temperature: float | None = None
) -> dict:
    """Use the reasoning-model token field and omit unsupported sampling options."""
    name = _model_name(model)
    reasoning = name.startswith("gpt-5") or bool(re.match(r"^o[134](?:-|$)", name))
    budget = int(max_tokens)
    if budget <= 0:
        raise ValueError("max_tokens must be positive")
    # Respect the trusted cap exactly, including small budgets. A provider may
    # reject an unsuitable allowance; silently raising it would violate cost
    # and nested execution limits.
    options = {"max_completion_tokens" if reasoning else "max_tokens": budget}
    if temperature is not None and not reasoning:
        options["temperature"] = max(0.0, min(2.0, float(temperature)))
    return options


def thinking_request_options(base_url: str, model: str) -> dict:
    """Disable thinking only for explicitly recognized DashScope hybrid models.

    Do not send this vendor extension to arbitrary compatible endpoints or to
    thinking-only models. In particular, a model name containing Qwen is not enough.
    """
    if is_dashscope_endpoint(base_url) and re.fullmatch(
        r"(?:qwen-plus|qwen-turbo|qwen-flash|qwen3\.8-max)(?:-latest|-\d{4}-\d{2}-\d{2})?",
        _model_name(model),
    ):
        return {"enable_thinking": False}
    return {}


def sdk_chat_request_options(
    base_url: str, model: str, max_tokens: int, temperature: float | None = None
) -> dict:
    options = chat_request_options(model, max_tokens, temperature)
    # Verified vendor capability, deliberately not inferred from an arbitrary
    # OpenAI-compatible gateway or from a model name on an unrelated host.
    endpoint = urlsplit(str(base_url or ""))
    if (endpoint.scheme == "https" and endpoint.hostname == "api.stepfun.com"
            and endpoint.username is None and endpoint.password is None
            and endpoint.netloc in {"api.stepfun.com", "api.stepfun.com:443"}
            and endpoint.path.rstrip("/") in {"/v1", "/step_plan/v1"}
            and not endpoint.query and not endpoint.fragment
            and str(model).strip().lower() == "step-5-preview"):
        # Step 5 supports low/medium/high, not an assumed thinking-off flag.
        # https://platform.stepfun.com/docs/zh/guides/models/step-5-preview.md
        options["reasoning_effort"] = "low"
    extra = thinking_request_options(base_url, model)
    if extra:
        options["extra_body"] = extra
    return options


def embedding_request_options(base_url: str, model: str, dimension: int) -> dict:
    """Only request configurable dimensions on known official model APIs."""
    if type(dimension) is not int or dimension <= 0:
        raise ValueError("Embedding dimension must be a positive integer")
    name = _model_name(model)
    host = urlsplit(str(base_url or "")).hostname or ""
    if host == "api.openai.com" and name in {"text-embedding-3-small", "text-embedding-3-large"}:
        maximum = 1536 if name.endswith("small") else 3072
        if dimension > maximum:
            raise ValueError(f"{model} supports at most {maximum} embedding dimensions")
        return {"dimensions": dimension}
    if is_dashscope_endpoint(base_url) and name in {"text-embedding-v3", "text-embedding-v4"}:
        allowed = {64, 128, 256, 512, 768, 1024}
        if name == "text-embedding-v4":
            allowed |= {1536, 2048}
        if dimension not in allowed:
            raise ValueError(f"{model} embedding dimension must be one of {sorted(allowed)}")
        return {"dimensions": dimension}
    return {}


def embedding_batch_size(base_url: str, model: str) -> int:
    if is_dashscope_endpoint(base_url) and _model_name(model) in {"text-embedding-v3", "text-embedding-v4"}:
        return 10
    return 32
