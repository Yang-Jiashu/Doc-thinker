import hashlib
import hmac
import json
import os
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, Field

# 项目根目录（doc-thinker），确保 workdir 不随 CWD 变化
_DOC_THINKER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_WORKDIR = os.path.join(_DOC_THINKER_ROOT, "data", "_system")
CREDENTIAL_BINDING_ENV = {
    "llm": "DOCTHINKER_LLM_CREDENTIAL_BINDING",
    "embed": "DOCTHINKER_EMBED_CREDENTIAL_BINDING",
    "rerank": "DOCTHINKER_RERANK_CREDENTIAL_BINDING",
}


def normalized_api_base(value: str) -> tuple[str, str, int | None, str]:
    """Compare credential destinations, including the API path, without credentials."""
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(char.isspace() for char in value)
        or any(char in unquote(value) for char in "{}")
    ):
        raise ValueError(
            "API base must be an HTTP(S) URL without credentials, query or fragment"
        )
    port = parsed.port
    if port == {"http": 80, "https": 443}[parsed.scheme.lower()]:
        port = None
    return (
        parsed.scheme.lower(),
        parsed.hostname.lower(),
        port,
        parsed.path.rstrip("/"),
    )


def credential_binding(key: str, base: str) -> str:
    """Detect mixed .env/process overrides, not authenticate or encrypt the key."""
    payload = json.dumps([normalized_api_base(base), key], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _same_api_base(left: str, right: str) -> bool:
    try:
        return normalized_api_base(left) == normalized_api_base(right)
    except ValueError:
        return False


def _scoped_key(
    explicit: str | None, base: str, *fallbacks: tuple[str | None, str]
) -> str:
    """Never inherit a credential from a different API destination."""
    if explicit:
        return explicit
    for key, key_base in fallbacks:
        if key and _same_api_base(base, key_base):
            return key
    return "EMPTY"


class AppSettings(BaseModel):
    llm_api_key: str
    vlm_base_url: str
    llm_model: str
    llm_models: list[str] = Field(default_factory=list)
    llm_router_max_concurrency: int = 32
    llm_temperature: float = 0.1
    vlm_model: str

    keyword_llm_model: str = "qwen-turbo"
    """Lightweight model for keyword extraction (no extended thinking)."""

    entity_extraction_llm_model: str = "qwen-turbo"
    """Lightweight model for entity extraction during ingestion (no extended thinking)."""

    embed_api_key: str
    embed_base_url: str
    embed_model: str
    embed_dim: int = 1024
    rerank_api_key: str
    rerank_base_url: str
    rerank_model: str
    graphcore_llm_max_async: int = 16
    graphcore_embedding_max_async: int = 16
    graphcore_max_parallel_insert: int = 8
    graphcore_max_gleaning: int = 0
    extraction_llm_model: str = ""

    workdir: str = "./data/_system"
    timeout_seconds: int = 3600
    credential_binding_errors: list[str] = Field(default_factory=list)
    inherited_auxiliary_models: list[str] = Field(default_factory=list)


def load_settings() -> AppSettings:
    _dashscope_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    llm_model = os.getenv("LLM_MODEL") or "qwen-plus"
    llm_models_raw = os.getenv("LLM_MODELS") or llm_model
    llm_models = [x.strip() for x in llm_models_raw.split(",") if x.strip()]
    openai_base = os.getenv("OPENAI_BASE_URL") or _dashscope_base
    # OPENAI_API_KEY without an explicit custom base belongs to OpenAI, not the
    # legacy DashScope default. An explicit binding key remains authoritative.
    openai_key_base = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    vlm_base = os.getenv("LLM_VLM_HOST") or os.getenv("LLM_BINDING_HOST") or openai_base
    embed_base = (
        os.getenv("LLM_EMBED_HOST")
        or os.getenv("EMBEDDING_BINDING_HOST")
        or openai_base
    )
    rerank_base = (
        os.getenv("RERANK_HOST")
        or os.getenv("EMBEDDING_BINDING_HOST")
        or os.getenv("LLM_BINDING_HOST")
        or openai_base
    )
    binding_errors: list[str] = []

    def checked_key(service: str, key: str, base: str) -> str:
        binding = os.getenv(CREDENTIAL_BINDING_ENV[service])
        if binding:
            try:
                matches = hmac.compare_digest(binding, credential_binding(key, base))
            except (TypeError, ValueError):
                matches = False
            if not matches:
                binding_errors.append(service)
                # EMPTY also prevents downstream ``key or os.getenv(...)`` from
                # silently reintroducing the conflicting environment credential.
                return "EMPTY"
        return key

    llm_api_key = checked_key(
        "llm",
        _scoped_key(
            os.getenv("LLM_BINDING_API_KEY"),
            vlm_base,
            (os.getenv("OPENAI_API_KEY"), openai_key_base),
        ),
        vlm_base,
    )
    embed_api_key = checked_key(
        "embed",
        _scoped_key(
            os.getenv("EMBEDDING_BINDING_API_KEY"),
            embed_base,
            (os.getenv("OPENAI_API_KEY"), openai_key_base),
            (llm_api_key, vlm_base),
        ),
        embed_base,
    )
    rerank_api_key = checked_key(
        "rerank", os.getenv("RERANK_API_KEY") or "", rerank_base
    )
    primary_model = llm_models[0] if llm_models else llm_model
    inherited_auxiliary_models: list[str] = []

    def auxiliary_model(name: str, legacy_default: str) -> str:
        value = os.getenv(name)
        if value is None and _same_api_base(vlm_base, _dashscope_base):
            return legacy_default
        # Explicit blank means inherit the primary model, never resurrect qwen defaults.
        if not value or not value.strip():
            inherited_auxiliary_models.append(name.lower())
            return primary_model
        return value.strip()

    return AppSettings(
        llm_api_key=llm_api_key,
        vlm_base_url=vlm_base,
        llm_model=llm_model,
        llm_models=llm_models,
        llm_router_max_concurrency=int(os.getenv("LLM_ROUTER_MAX_CONCURRENCY") or 32),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE") or 0.1),
        vlm_model=auxiliary_model("VLM_MODEL", "qwen-vl-max"),
        keyword_llm_model=auxiliary_model("KEYWORD_LLM_MODEL", "qwen-turbo"),
        entity_extraction_llm_model=auxiliary_model(
            "ENTITY_EXTRACTION_LLM_MODEL", "qwen-turbo"
        ),
        embed_api_key=embed_api_key,
        embed_base_url=embed_base,
        embed_model=os.getenv("EMBEDDING_MODEL")
        or os.getenv("EMBED_MODEL")
        or "text-embedding-v3",
        embed_dim=int(os.getenv("EMBEDDING_DIM") or os.getenv("EMBED_DIM") or 1024),
        rerank_api_key=rerank_api_key,
        rerank_base_url=rerank_base,
        rerank_model=os.getenv("RERANK_MODEL") or "",
        graphcore_llm_max_async=int(
            os.getenv("GRAPHCORE_LLM_MAX_ASYNC") or os.getenv("MAX_ASYNC") or 16
        ),
        graphcore_embedding_max_async=int(
            os.getenv("GRAPHCORE_EMBEDDING_MAX_ASYNC")
            or os.getenv("EMBEDDING_FUNC_MAX_ASYNC")
            or 16
        ),
        graphcore_max_parallel_insert=int(
            os.getenv("GRAPHCORE_MAX_PARALLEL_INSERT")
            or os.getenv("MAX_PARALLEL_INSERT")
            or 8
        ),
        graphcore_max_gleaning=int(os.getenv("MAX_GLEANING") or 0),
        extraction_llm_model=os.getenv("EXTRACTION_LLM_MODEL") or "",
        workdir=os.getenv("RAG_WORKDIR") or _DEFAULT_WORKDIR,
        timeout_seconds=int(os.getenv("TIMEOUT") or 3600),
        credential_binding_errors=binding_errors,
        inherited_auxiliary_models=inherited_auxiliary_models,
    )
