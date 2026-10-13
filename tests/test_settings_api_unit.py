"""Offline settings transactions; only temporary .env files are ever accessed."""

import importlib
import os
from unittest.mock import Mock, patch

import pytest
from dotenv import dotenv_values
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Importing docthinker normally loads .env from cwd. This suite must not read the
# developer's credentials, even during collection.
with patch("dotenv.load_dotenv"):
    from docthinker.providers.settings import AppSettings, load_settings

    settings_api = importlib.import_module("docthinker.server.routers.settings")


_ENV_KEYS = (
    set(settings_api._ENV_MAP.values())
    | set(settings_api._ALIASES.values())
    | set(settings_api.CREDENTIAL_BINDING_ENV.values())
    | {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "LLM_MODELS",
        "MAX_GLEANING",
        "LLM_TEMPERATURE",
        "RAG_WORKDIR",
        "TIMEOUT",
    }
)


@pytest.fixture
def configured(tmp_path, monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    active = AppSettings(
        llm_api_key="secret-old-llm",
        vlm_base_url="https://old.example/v1",
        llm_model="old-main",
        llm_models=["old-main", "old-fallback"],
        vlm_model="old-vision",
        keyword_llm_model="old-keywords",
        entity_extraction_llm_model="old-entities",
        extraction_llm_model="old-override",
        embed_api_key="secret-old-embed",
        embed_base_url="https://embed.example/v1",
        embed_model="old-embed",
        embed_dim=1024,
        rerank_api_key="secret-old-rerank",
        rerank_base_url="https://rerank.example/v1",
        rerank_model="old-rerank",
    )
    monkeypatch.setattr(settings_api, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(settings_api.state, "settings", active)
    monkeypatch.setattr(settings_api, "_pending_for_runtime", None)
    monkeypatch.setattr(settings_api, "_pending_settings", None)
    app = FastAPI()
    app.include_router(settings_api.router, prefix="/api/v1")
    with TestClient(app) as client:
        yield client, active, tmp_path / ".env", monkeypatch


def _restart_from(env_path, monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in dotenv_values(env_path).items():
        if value is not None:
            monkeypatch.setenv(key, value)
    return load_settings()


@pytest.mark.parametrize("service", ["llm", "embed", "rerank"])
@pytest.mark.parametrize("key", [None, "", "   "])
def test_destination_change_requires_explicit_key(configured, service, key):
    client, active, env_path, _ = configured
    before = active.model_dump()
    payload = {f"{service}_base_url": "https://new.example/v1"}
    if key is not None:
        payload[f"{service}_api_key"] = key
    response = client.post("/api/v1/settings", json=payload)
    assert response.status_code == 400
    assert "requires an explicit new" in response.json()["detail"]
    assert "secret-old" not in response.text
    assert active.model_dump() == before
    assert not env_path.exists()


@pytest.mark.parametrize(
    "base", ["https://OLD.example:443/v1/", "https://old.example/v1"]
)
def test_equivalent_base_does_not_require_a_new_key(configured, base):
    client, _, _, _ = configured
    response = client.post("/api/v1/settings", json={"llm_base_url": base})
    assert response.status_code == 200


@pytest.mark.parametrize(
    "base",
    ["http://old.example/v1", "https://old.example:444/v1", "https://old.example/v2"],
)
def test_scheme_port_and_path_are_part_of_credential_boundary(configured, base):
    client, _, _, _ = configured
    assert (
        client.post("/api/v1/settings", json={"llm_base_url": base}).status_code == 400
    )


@pytest.mark.parametrize(
    "base",
    [
        "file:///tmp/api",
        "https://user:secret@new.example/v1",
        "https://new.example/v1?token=secret",
        "https://new.example/v1#fragment",
        "https://new.example:99999/v1",
        "https://new.example/{WorkspaceId}/v1",
        "https://new.example/%7BWorkspaceId%7D/v1",
        "",
    ],
)
def test_invalid_base_never_reflects_credentials(configured, base):
    client, _, env_path, _ = configured
    response = client.post(
        "/api/v1/settings", json={"llm_base_url": base, "llm_api_key": "secret-new"}
    )
    assert response.status_code == 400
    assert "secret" not in response.text
    assert not env_path.exists()


def test_save_is_pending_restart_and_preserves_other_destinations(configured):
    client, active, env_path, monkeypatch = configured
    before = active.model_dump()
    environment_before = dict(os.environ)
    response = client.post(
        "/api/v1/settings",
        json={
            "llm_base_url": "https://new.example/v1",
            "llm_api_key": "secret-new-llm",
            "llm_model": "new-main,new-fallback",
            "llm_max_async": 7,
        },
    )
    assert response.status_code == 200
    assert response.json()["requires_restart"] is True
    assert active.model_dump() == before
    assert dict(os.environ) == environment_before
    saved = client.get("/api/v1/settings").json()
    assert saved["llm_model"] == "new-main"
    assert saved["llm_models"] == ["new-main", "new-fallback"]
    assert saved["llm_max_async"] == 7
    assert saved["requires_restart"] is True
    assert saved["warnings"]
    for field in (
        "vlm_model",
        "keyword_llm_model",
        "entity_extraction_llm_model",
        "extraction_llm_model",
    ):
        assert saved[field] == ""
    assert "secret-new-llm" not in response.text + str(saved)
    restarted = _restart_from(env_path, monkeypatch)
    assert restarted.llm_models == ["new-main", "new-fallback"]
    assert restarted.llm_api_key == "secret-new-llm"
    assert restarted.vlm_base_url == "https://new.example/v1"
    assert (
        restarted.vlm_model
        == restarted.keyword_llm_model
        == restarted.entity_extraction_llm_model
        == "new-main"
    )
    assert restarted.extraction_llm_model == ""
    assert restarted.embed_base_url == active.embed_base_url
    assert restarted.embed_api_key == active.embed_api_key
    assert restarted.rerank_base_url == active.rerank_base_url
    assert restarted.rerank_api_key == active.rerank_api_key
    assert restarted.graphcore_llm_max_async == 7


def test_switch_host_without_main_model_is_rejected(configured):
    client, _, env_path, _ = configured
    response = client.post(
        "/api/v1/settings",
        json={"llm_base_url": "https://new.example/v1", "llm_api_key": "secret-new"},
    )
    assert response.status_code == 400
    assert not env_path.exists()


@pytest.mark.parametrize(
    "field", ["vlm_model", "keyword_llm_model", "entity_extraction_llm_model"]
)
def test_explicit_empty_auxiliary_model_inherits_primary_on_restart(configured, field):
    client, _, env_path, monkeypatch = configured
    assert client.post("/api/v1/settings", json={field: ""}).status_code == 200
    assert client.get("/api/v1/settings").json()[field] == ""
    assert getattr(_restart_from(env_path, monkeypatch), field) == "old-main"


@pytest.mark.parametrize("service", ["embed", "rerank"])
def test_explicit_key_authorizes_only_its_destination(configured, service):
    client, active, env_path, monkeypatch = configured
    response = client.post(
        "/api/v1/settings",
        json={
            f"{service}_base_url": "http://localhost:8000/v1",
            f"{service}_api_key": "EMPTY",
        },
    )
    assert response.status_code == 200
    restarted = _restart_from(env_path, monkeypatch)
    assert getattr(restarted, f"{service}_api_key") == "EMPTY"
    assert getattr(restarted, f"{service}_base_url") == "http://localhost:8000/v1"
    assert restarted.llm_api_key == active.llm_api_key


def test_dotenv_round_trip_removes_duplicate_alias_overrides(configured):
    client, _, env_path, monkeypatch = configured
    env_path.write_text(
        "# keep comment\nLLM_MODEL=stale\nexport LLM_MODELS='stale-first'\n"
        "LLM_MODELS=stale-last\nLLM_VLM_HOST=https://stale.example/v1\n"
        "LLM_EMBED_HOST=https://stale-embed.example/v1\n"
        "GRAPHCORE_LLM_MAX_ASYNC=99\nUNCHANGED='keep # data'\n",
        encoding="utf-8",
    )
    key = "new-key-with-'quote\\backslash#hash"
    response = client.post(
        "/api/v1/settings",
        json={"llm_model": "new-main", "llm_api_key": key, "llm_max_async": 5},
    )
    assert response.status_code == 200
    contents = env_path.read_text(encoding="utf-8")
    assert contents.count("LLM_MODELS=") == 1
    assert "# keep comment" in contents
    values = dotenv_values(env_path)
    assert values["UNCHANGED"] == "keep # data"
    assert values["LLM_BINDING_API_KEY"] == key
    restarted = _restart_from(env_path, monkeypatch)
    assert restarted.llm_models == ["new-main"]
    assert restarted.vlm_base_url == "https://old.example/v1"
    assert restarted.embed_base_url == "https://embed.example/v1"
    assert restarted.graphcore_llm_max_async == 5


def test_persistence_failure_does_not_mutate_active_pending_or_environment(configured):
    client, active, env_path, monkeypatch = configured
    assert (
        client.post("/api/v1/settings", json={"llm_model": "saved-model"}).status_code
        == 200
    )
    original_file = env_path.read_bytes()
    environment_before = dict(os.environ)
    before = active.model_dump()
    fail = Mock(side_effect=OSError("disk failure with secret-new-key"))
    monkeypatch.setattr(settings_api.os, "replace", fail)
    response = client.post(
        "/api/v1/settings",
        json={"llm_model": "unsaved", "llm_api_key": "secret-new-key"},
    )
    assert response.status_code == 500
    assert "secret-new-key" not in response.text
    assert active.model_dump() == before
    assert dict(os.environ) == environment_before
    assert env_path.read_bytes() == original_file
    assert client.get("/api/v1/settings").json()["llm_model"] == "saved-model"
    assert not list(env_path.parent.glob(".env.*"))


@pytest.mark.parametrize(
    "value",
    [
        "key\nRERANK_HOST=https://evil.example",
        "key\rnext",
        "key\x00next",
        "${OPENAI_API_KEY}",
        "****",
    ],
)
def test_key_injection_and_mask_are_rejected_without_echo(configured, value):
    client, _, env_path, _ = configured
    response = client.post("/api/v1/settings", json={"llm_api_key": value})
    assert response.status_code == 400
    assert "evil.example" not in response.text
    assert "OPENAI_API_KEY" not in response.text
    assert not env_path.exists()


@pytest.mark.parametrize(
    "payload",
    [
        {"llm_api_key": ["secret-invalid-key"]},
        {"embed_api_key": {"value": "secret-invalid-key"}},
        {"embed_dim": "secret-invalid-key"},
    ],
)
def test_validation_errors_do_not_reflect_invalid_secret_inputs(configured, payload):
    client, _, env_path, _ = configured
    response = client.post("/api/v1/settings", json=payload)
    assert response.status_code == 422
    assert "secret-invalid-key" not in response.text
    assert not env_path.exists()


@pytest.mark.parametrize(
    "field",
    [
        "embed_dim",
        "llm_max_async",
        "embedding_max_async",
        "max_parallel_insert",
        "llm_router_max_concurrency",
    ],
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8", 65537])
def test_numeric_values_require_bounded_integers(configured, field, value):
    client, _, env_path, _ = configured
    assert client.post("/api/v1/settings", json={field: value}).status_code == 422
    assert not env_path.exists()


def test_embedding_change_warns_but_does_not_rebuild_or_mutate_runtime(configured):
    client, active, _, _ = configured
    response = client.post(
        "/api/v1/settings", json={"embed_model": "new-embed", "embed_dim": 3072}
    )
    assert response.status_code == 200
    assert any("rebuild" in warning for warning in response.json()["warnings"])
    assert active.embed_model == "old-embed" and active.embed_dim == 1024
    assert (
        client.get("/api/v1/settings").json()["warnings"] == response.json()["warnings"]
    )


def test_followup_updates_use_pending_settings_and_reset_on_new_runtime(configured):
    client, active, _, monkeypatch = configured
    assert (
        client.post("/api/v1/settings", json={"llm_model": "next-model"}).status_code
        == 200
    )
    assert (
        client.post(
            "/api/v1/settings", json={"embed_dim": 2048, "llm_api_key": ""}
        ).status_code
        == 200
    )
    assert client.get("/api/v1/settings").json()["llm_model"] == "next-model"
    monkeypatch.setattr(settings_api.state, "settings", active.model_copy())
    assert client.get("/api/v1/settings").json()["requires_restart"] is False
    assert client.get("/api/v1/settings").json()["llm_model"] == "old-main"


def test_load_settings_does_not_inherit_keys_across_bases(configured):
    _, _, _, monkeypatch = configured
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example/v1")
    monkeypatch.setenv("LLM_BINDING_HOST", "https://llm.example/v1")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://embed.example/v1")
    loaded = load_settings()
    assert loaded.llm_api_key == loaded.embed_api_key == "EMPTY"
    monkeypatch.setenv("LLM_BINDING_API_KEY", "secret-llm")
    assert load_settings().embed_api_key == "EMPTY"
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "https://LLM.example:443/v1/")
    assert load_settings().embed_api_key == "secret-llm"


def test_openai_key_without_custom_base_is_not_sent_to_default_dashscope(configured):
    _, _, _, monkeypatch = configured
    monkeypatch.setenv("OPENAI_API_KEY", "secret-openai")
    assert load_settings().llm_api_key == "EMPTY"
    assert load_settings().embed_api_key == "EMPTY"


def test_non_dashscope_default_auxiliary_models_use_primary(configured):
    _, _, _, monkeypatch = configured
    monkeypatch.setenv("LLM_BINDING_HOST", "https://new.example/v1")
    monkeypatch.setenv("LLM_MODEL", "new-main")
    loaded = load_settings()
    assert (
        loaded.vlm_model
        == loaded.keyword_llm_model
        == loaded.entity_extraction_llm_model
        == "new-main"
    )


@pytest.mark.parametrize("service", ["llm", "embed", "rerank"])
@pytest.mark.parametrize("override", ["key", "host"])
def test_saved_binding_blocks_mixed_external_environment_overrides(
    configured, service, override
):
    client, _, env_path, monkeypatch = configured
    assert (
        client.post("/api/v1/settings", json={"llm_model": "saved-main"}).status_code
        == 200
    )
    _restart_from(env_path, monkeypatch)
    field = f"{service}_api_key" if override == "key" else f"{service}_base_url"
    env_name = settings_api._ENV_MAP[field]
    if override == "host":
        # The higher-precedence host alias is what the loader actually uses.
        env_name = settings_api._ALIASES.get(env_name, env_name)
    monkeypatch.setenv(
        env_name,
        (
            "secret-external-override"
            if override == "key"
            else "https://unexpected.example/v1"
        ),
    )
    blocked = load_settings()
    assert getattr(blocked, f"{service}_api_key") == "EMPTY"
    assert blocked.credential_binding_errors == [service]
    monkeypatch.setattr(settings_api.state, "settings", blocked)
    response = client.get("/api/v1/settings")
    assert any(
        f"{service} credentials were blocked" in warning
        for warning in response.json()["warnings"]
    )
    assert "secret-external-override" not in response.text
    assert not response.json()[f"{service}_api_key_masked"]
    before = env_path.read_bytes()
    assert client.post("/api/v1/settings", json={"llm_max_async": 3}).status_code == 400
    assert env_path.read_bytes() == before
    assert (
        client.post(
            "/api/v1/settings", json={f"{service}_api_key": "explicit-replacement"}
        ).status_code
        == 200
    )
    assert not any(
        "credentials were blocked" in warning
        for warning in client.get("/api/v1/settings").json()["warnings"]
    )


def test_inherit_choice_survives_restart_and_later_main_model_change(configured):
    client, _, env_path, monkeypatch = configured
    assert (
        client.post(
            "/api/v1/settings",
            json={
                "vlm_model": "",
                "keyword_llm_model": "",
                "entity_extraction_llm_model": "",
            },
        ).status_code
        == 200
    )
    restarted = _restart_from(env_path, monkeypatch)
    monkeypatch.setattr(settings_api.state, "settings", restarted)
    saved = client.get("/api/v1/settings").json()
    assert (
        saved["vlm_model"]
        == saved["keyword_llm_model"]
        == saved["entity_extraction_llm_model"]
        == ""
    )
    assert (
        client.post("/api/v1/settings", json={"llm_model": "second-main"}).status_code
        == 200
    )
    next_restart = _restart_from(env_path, monkeypatch)
    assert (
        next_restart.vlm_model
        == next_restart.keyword_llm_model
        == next_restart.entity_extraction_llm_model
        == "second-main"
    )


def test_explicit_auxiliary_choice_can_replace_inherit_even_when_equal(configured):
    client, _, env_path, monkeypatch = configured
    assert (
        client.post("/api/v1/settings", json={"keyword_llm_model": ""}).status_code
        == 200
    )
    restarted = _restart_from(env_path, monkeypatch)
    monkeypatch.setattr(settings_api.state, "settings", restarted)
    response = client.post("/api/v1/settings", json={"keyword_llm_model": "old-main"})
    assert response.status_code == 200
    assert "keyword_llm_model" in response.json()["changed"]
    assert (
        client.post("/api/v1/settings", json={"llm_model": "next-main"}).status_code
        == 200
    )
    assert _restart_from(env_path, monkeypatch).keyword_llm_model == "old-main"


def test_explicit_empty_can_rebind_an_unauthenticated_destination(configured):
    client, active, env_path, monkeypatch = configured
    blocked = active.model_copy(
        update={"llm_api_key": "EMPTY", "credential_binding_errors": ["llm"]}
    )
    monkeypatch.setattr(settings_api.state, "settings", blocked)
    response = client.post("/api/v1/settings", json={"llm_api_key": "EMPTY"})
    assert response.status_code == 200
    assert "llm_api_key" in response.json()["changed"]
    restarted = _restart_from(env_path, monkeypatch)
    assert restarted.llm_api_key == "EMPTY"
    assert restarted.credential_binding_errors == []
