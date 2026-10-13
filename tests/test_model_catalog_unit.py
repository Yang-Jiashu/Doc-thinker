"""Offline contract checks for manually verified provider presets.

These validate catalog integrity, not live API availability, paid-account access,
quality, latency, or model retirement. Source links and the checked date make the
manual verification boundary visible without contacting any provider in tests.
"""

import json
import re
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

import pytest

CATALOG_PATH = (
    Path(__file__).resolve().parents[1]
    / "docthinker"
    / "ui"
    / "static"
    / "model_catalog.json"
)
ROLE_ENDPOINT = {
    "chat": "llm",
    "vision": "llm",
    "keyword": "llm",
    "extraction": "llm",
    "embedding": "embedding",
    "rerank": "rerank",
}


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        assert key not in obj, f"Duplicate JSON key: {key}"
        obj[key] = value
    return obj


CATALOG = json.loads(
    CATALOG_PATH.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
)
PROVIDERS = CATALOG["providers"]
MODELS = [(provider, model) for provider in PROVIDERS for model in provider["models"]]


def _assert_public_link(value):
    assert isinstance(value, str) and value.strip() == value
    url = urlsplit(value)
    assert url.scheme == "https" and url.hostname
    assert url.username is None and url.password is None
    assert not url.query, "Do not put credentials or tracking tokens in source links"


def test_catalog_version_date_and_useful_breadth():
    assert set(CATALOG) == {"version", "checked_at", "providers"}
    assert type(CATALOG["version"]) is int and CATALOG["version"] == 1
    assert date.fromisoformat(CATALOG["checked_at"]) == date(2026, 9, 19)
    assert len(PROVIDERS) >= 10
    assert len(MODELS) >= 50
    assert len({p["id"] for p in PROVIDERS}) == len(PROVIDERS)
    assert set(ROLE_ENDPOINT).issubset(
        {role for _, model in MODELS for role in model["roles"]}
    )


@pytest.mark.parametrize("provider", PROVIDERS, ids=lambda p: p["id"])
def test_provider_contract_and_role_specific_base_urls(provider):
    assert set(provider) == {
        "id", "name", "description", "docs_url", "key_url", "local",
        "endpoints", "models", "notes",
    }
    assert re.fullmatch(r"[a-z][a-z0-9_-]*", provider["id"])
    assert type(provider["local"]) is bool
    for field in ("name", "description"):
        assert isinstance(provider[field], str) and provider[field].strip()
    _assert_public_link(provider["docs_url"])
    _assert_public_link(provider["key_url"])
    assert provider["models"] and provider["notes"]
    assert all(isinstance(note, str) and note.strip() for note in provider["notes"])
    assert set(provider["endpoints"]) <= {"llm", "embedding", "rerank"}
    assert len({m["id"] for m in provider["models"]}) == len(provider["models"])
    used_endpoints = {
        ROLE_ENDPOINT[role] for model in provider["models"] for role in model["roles"]
    }
    assert used_endpoints == set(provider["endpoints"])
    for endpoint in provider["endpoints"].values():
        url = urlsplit(endpoint)
        assert url.username is None and url.password is None
        assert not url.query and not url.fragment
        if provider["local"]:
            assert url.scheme == "http" and url.hostname == "localhost"
        else:
            assert url.scheme == "https" and url.hostname
        assert not url.path.rstrip("/").endswith(
            ("/chat/completions", "/embeddings", "/rerank", "/reranks", "/responses")
        ), "Presets supply base URLs, not complete operation URLs"
        if "{" in endpoint or "}" in endpoint:
            assert provider["id"] == "dashscope"
            assert endpoint.count("{WorkspaceId}") == 1
            assert any("{WorkspaceId}" in note for note in provider["notes"])


@pytest.mark.parametrize(
    ("provider", "model"), MODELS,
    ids=[f'{p["id"]}:{m["id"]}' for p, m in MODELS],
)
def test_model_contract_and_capability_boundaries(provider, model):
    required = {"id", "name", "roles", "description", "source_url"}
    assert required <= set(model)
    assert set(model) <= required | {"dimension", "requires_download"}
    for field in ("id", "name", "description"):
        assert isinstance(model[field], str) and model[field].strip() == model[field]
        assert model[field]
    assert not any(char.isspace() for char in model["id"])
    _assert_public_link(model["source_url"])
    assert model["roles"] and len(model["roles"]) == len(set(model["roles"]))
    assert set(model["roles"]) <= set(ROLE_ENDPOINT)
    if set(model["roles"]) & {"vision", "keyword", "extraction"}:
        assert "chat" in model["roles"]
    if set(model["roles"]) & {"embedding", "rerank"}:
        assert len(model["roles"]) == 1, "Do not conflate generation and retrieval APIs"
    if "dimension" in model:
        assert model["roles"] == ["embedding"]
        assert type(model["dimension"]) is int and 0 < model["dimension"] <= 65536
    if provider["local"]:
        assert model["requires_download"] is True
    else:
        assert "requires_download" not in model


@pytest.mark.parametrize(
    ("provider_id", "model_id", "dimension"),
    [
        ("openai", "text-embedding-3-small", 1536),
        ("openai", "text-embedding-3-large", 3072),
        ("dashscope", "text-embedding-v3", 1024),
        ("dashscope", "text-embedding-v4", 1024),
        ("bigmodel", "embedding-3", 2048),
        ("google", "gemini-embedding-001", 3072),
    ],
)
def test_verified_dimensions_are_defaults_not_arbitrary_truncations(
    provider_id, model_id, dimension
):
    model = next(m for p, m in MODELS if p["id"] == provider_id and m["id"] == model_id)
    assert model["dimension"] == dimension


def test_uncertain_service_dimensions_are_not_guessed():
    for provider, model in MODELS:
        if provider["id"] in {"siliconflow", "ollama"} and "embedding" in model["roles"]:
            assert "dimension" not in model
            assert "维度" in model["description"]


def test_thinking_models_not_suggested_for_low_budget_auxiliary_calls():
    for provider, model in MODELS:
        if provider["id"] in {"deepseek", "moonshot", "minimax"} or model["id"] in {
            "glm-5.3", "glm-5.3-flash", "qwen3.8-max",
            "deepseek-ai/DeepSeek-V4-Flash",
        }:
            assert not set(model["roles"]) & {"keyword", "extraction"}


@pytest.mark.parametrize(
    ("provider_id", "base_url"),
    [
        ("stepfun", "https://api.stepfun.com/v1"),
        ("stepfun-plan", "https://api.stepfun.com/step_plan/v1"),
    ],
)
def test_step5_presets_keep_distinct_official_channels_and_budget_roles(provider_id, base_url):
    provider = next(p for p in PROVIDERS if p["id"] == provider_id)
    assert provider["endpoints"] == {"llm": base_url}
    assert provider["key_url"] == "https://platform.stepfun.com"
    assert provider["docs_url"] == "https://platform.stepfun.com/docs/zh/api-reference/chat/chat-completion-create"
    assert len(provider["models"]) == 1
    model = provider["models"][0]
    assert model["id"] == "step-5-preview"
    assert model["roles"] == ["chat", "vision"]
    assert model["source_url"] == "https://platform.stepfun.com/docs/zh/guides/models/step-5-preview"
    assert any("2026-10-10" in note and "2026-09-19" in note for note in provider["notes"])
    assert any("Credit" in note and "独立" in note for note in provider["notes"])
    if provider_id == "stepfun-plan":
        assert any(
            "第三方" in note and "明确许可" in note
            and "https://platform.stepfun.com/docs/zh/step-plan/paid-service-agreement" in note
            for note in provider["notes"]
        )


def test_step_incremental_additions_do_not_redate_the_original_catalog():
    assert CATALOG["checked_at"] == "2026-09-19"
    assert len(PROVIDERS) == 13
    assert len(MODELS) == 73


def test_no_native_anthropic_or_unsupported_generated_media_endpoints():
    for provider, model in MODELS:
        if model["id"].startswith("anthropic/"):
            assert provider["id"] == "openrouter"
            assert provider["endpoints"]["llm"] == "https://openrouter.ai/api/v1"
        assert all("api.anthropic.com" not in v for v in provider["endpoints"].values())
        assert model["id"] not in {"deepseek-chat", "deepseek-reasoner"}
        assert not model["id"].startswith(("gpt-5-pro", "o3-pro", "veo-", "dall-e"))
    google = next(p for p in PROVIDERS if p["id"] == "google")
    assert not any("embedding-2" in m["id"] for m in google["models"])
    assert set(next(p for p in PROVIDERS if p["id"] == "jina")["endpoints"]) == {"rerank"}


def test_catalog_contains_no_credential_values():
    text = CATALOG_PATH.read_text(encoding="utf-8")
    assert not re.search(r"\\bsk-[a-zA-Z0-9_-]{12,}", text)
    assert not re.search(r"\\bAIza[a-zA-Z0-9_-]{20,}", text)
    assert not re.search(r'"(?:api_key|token|password)"\\s*:', text)
