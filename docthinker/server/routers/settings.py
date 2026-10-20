"""Settings API — atomically save model configuration for the next restart."""

import io
import os
import stat
import tempfile
from pathlib import Path

from dotenv.parser import parse_stream
from fastapi import APIRouter, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field

from docthinker.providers.settings import (
    CREDENTIAL_BINDING_ENV,
    AppSettings,
    credential_binding,
    normalized_api_base,
)

from ..state import state


class _SettingsRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def redacted_validation(request):
            try:
                return await handler(request)
            except RequestValidationError:
                # FastAPI's default 422 includes the invalid input, which may be
                # an API key or a malformed object containing credentials.
                raise HTTPException(
                    422,
                    "Invalid settings payload: check field types and numeric bounds",
                ) from None

        return redacted_validation


router = APIRouter(route_class=_SettingsRoute)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_pending_for_runtime: AppSettings | None = None
_pending_settings: AppSettings | None = None


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SettingsResponse(BaseModel):
    llm_api_key_masked: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    llm_models: list[str] = []
    vlm_model: str = ""
    keyword_llm_model: str = ""
    entity_extraction_llm_model: str = ""
    extraction_llm_model: str = ""

    embed_api_key_masked: str = ""
    embed_base_url: str = ""
    embed_model: str = ""
    embed_dim: int = 1024

    rerank_api_key_masked: str = ""
    rerank_base_url: str = ""
    rerank_model: str = ""

    llm_max_async: int = 16
    embedding_max_async: int = 16
    max_parallel_insert: int = 8
    llm_router_max_concurrency: int = 32

    workdir: str = ""
    requires_restart: bool = False
    warnings: list[str] = Field(default_factory=list)


class SettingsUpdate(BaseModel):
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    vlm_model: str | None = None
    keyword_llm_model: str | None = None
    entity_extraction_llm_model: str | None = None
    extraction_llm_model: str | None = None

    embed_api_key: str | None = None
    embed_base_url: str | None = None
    embed_model: str | None = None
    embed_dim: int | None = Field(default=None, ge=1, le=65536, strict=True)

    rerank_api_key: str | None = None
    rerank_base_url: str | None = None
    rerank_model: str | None = None

    llm_max_async: int | None = Field(default=None, ge=1, le=1024, strict=True)
    embedding_max_async: int | None = Field(default=None, ge=1, le=1024, strict=True)
    max_parallel_insert: int | None = Field(default=None, ge=1, le=1024, strict=True)
    llm_router_max_concurrency: int | None = Field(
        default=None, ge=1, le=1024, strict=True
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mask_key(key: str) -> str:
    if not key or key == "EMPTY":
        return ""
    if len(key) <= 8:
        return key[:2] + "****"
    return key[:4] + "****" + key[-4:]


_ENV_MAP: dict[str, str] = {
    "llm_api_key": "LLM_BINDING_API_KEY",
    "llm_base_url": "LLM_BINDING_HOST",
    "llm_model": "LLM_MODEL",
    "vlm_model": "VLM_MODEL",
    "keyword_llm_model": "KEYWORD_LLM_MODEL",
    "entity_extraction_llm_model": "ENTITY_EXTRACTION_LLM_MODEL",
    "extraction_llm_model": "EXTRACTION_LLM_MODEL",
    "embed_api_key": "EMBEDDING_BINDING_API_KEY",
    "embed_base_url": "EMBEDDING_BINDING_HOST",
    "embed_model": "EMBEDDING_MODEL",
    "embed_dim": "EMBEDDING_DIM",
    "rerank_api_key": "RERANK_API_KEY",
    "rerank_base_url": "RERANK_HOST",
    "rerank_model": "RERANK_MODEL",
    "llm_max_async": "MAX_ASYNC",
    "embedding_max_async": "EMBEDDING_FUNC_MAX_ASYNC",
    "max_parallel_insert": "MAX_PARALLEL_INSERT",
    "llm_router_max_concurrency": "LLM_ROUTER_MAX_CONCURRENCY",
}

_ATTR_MAP = {
    "llm_base_url": "vlm_base_url",
    "llm_max_async": "graphcore_llm_max_async",
    "embedding_max_async": "graphcore_embedding_max_async",
    "max_parallel_insert": "graphcore_max_parallel_insert",
}
_ALIASES = {
    "LLM_BINDING_HOST": "LLM_VLM_HOST",
    "EMBEDDING_BINDING_HOST": "LLM_EMBED_HOST",
    "EMBEDDING_MODEL": "EMBED_MODEL",
    "EMBEDDING_DIM": "EMBED_DIM",
    "MAX_ASYNC": "GRAPHCORE_LLM_MAX_ASYNC",
    "EMBEDDING_FUNC_MAX_ASYNC": "GRAPHCORE_EMBEDDING_MAX_ASYNC",
    "MAX_PARALLEL_INSERT": "GRAPHCORE_MAX_PARALLEL_INSERT",
}


def _configured_settings() -> AppSettings | None:
    if _pending_for_runtime is state.settings and _pending_settings is not None:
        return _pending_settings
    return state.settings


def _warnings(configured: AppSettings, active: AppSettings) -> list[str]:
    warnings = [
        f"{service} credentials were blocked because the saved API base/key binding "
        "does not match the effective environment. Check process environment overrides "
        "and explicitly save the correct key and API base before restarting."
        for service in configured.credential_binding_errors
    ]
    if (
        configured.embed_model != active.embed_model
        or configured.embed_dim != active.embed_dim
        or configured.embed_base_url != active.embed_base_url
    ):
        warnings.append(
            "Embedding configuration changed: rebuild the affected vector indexes before "
            "using the new model/dimensions. Indexes have not been rebuilt automatically."
        )
    if not configured.vlm_model or "vlm_model" in configured.inherited_auxiliary_models:
        warnings.append(
            "An empty VLM model inherits the primary LLM after restart. "
            "This does not guarantee image support; choose a vision-capable model for images."
        )
    return warnings


def _configured_value(settings: AppSettings, name: str):
    if name in settings.inherited_auxiliary_models:
        return ""
    return getattr(settings, _ATTR_MAP.get(name, name))


def _persist_to_dotenv(updates: dict[str, str]) -> None:
    """Atomically replace assignments, including duplicates/export and quoted values."""
    env_path = _PROJECT_ROOT / ".env"
    content = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    remaining = dict(updates)
    pieces: list[str] = []

    def assignment(key: str, value: str) -> str:
        if any(ord(char) < 32 or ord(char) == 127 for char in value) or "${" in value:
            raise ValueError("Unsafe value in saved configuration")
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"{key}='{escaped}'\n"

    for binding in parse_stream(io.StringIO(content)):
        if binding.key in updates:
            if binding.key in remaining:
                pieces.append(assignment(binding.key, remaining.pop(binding.key)))
        else:
            pieces.append(binding.original.string)
    rendered = "".join(pieces)
    if rendered and not rendered.endswith("\n"):
        rendered += "\n"
    rendered += "".join(assignment(key, value) for key, value in remaining.items())
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=_PROJECT_ROOT, prefix=".env.", delete=False
        ) as handle:
            temporary = handle.name
            if env_path.exists():
                os.fchmod(handle.fileno(), stat.S_IMODE(env_path.stat().st_mode))
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, env_path)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/settings")
async def get_settings() -> SettingsResponse:
    s = _configured_settings()
    if not s:
        return SettingsResponse()

    return SettingsResponse(
        llm_api_key_masked=_mask_key(s.llm_api_key),
        llm_base_url=s.vlm_base_url,
        llm_model=s.llm_model,
        llm_models=s.llm_models,
        vlm_model=_configured_value(s, "vlm_model"),
        keyword_llm_model=_configured_value(s, "keyword_llm_model"),
        entity_extraction_llm_model=_configured_value(s, "entity_extraction_llm_model"),
        extraction_llm_model=s.extraction_llm_model,
        embed_api_key_masked=_mask_key(s.embed_api_key),
        embed_base_url=s.embed_base_url,
        embed_model=s.embed_model,
        embed_dim=s.embed_dim,
        rerank_api_key_masked=_mask_key(s.rerank_api_key),
        rerank_base_url=s.rerank_base_url,
        rerank_model=s.rerank_model,
        llm_max_async=s.graphcore_llm_max_async,
        embedding_max_async=s.graphcore_embedding_max_async,
        max_parallel_insert=s.graphcore_max_parallel_insert,
        llm_router_max_concurrency=s.llm_router_max_concurrency,
        workdir=s.workdir,
        requires_restart=s is not state.settings,
        warnings=_warnings(s, state.settings),
    )


@router.post("/settings")
async def update_settings(body: SettingsUpdate) -> dict:
    global _pending_for_runtime, _pending_settings
    s = _configured_settings()
    if not s:
        raise HTTPException(status_code=503, detail="Settings not initialized")

    values = body.model_dump(exclude_none=True)
    for name, value in list(values.items()):
        if isinstance(value, str):
            if (
                any(ord(char) < 32 or ord(char) == 127 for char in value)
                or "${" in value
            ):
                raise HTTPException(
                    400,
                    f"{name} must not contain control characters or dotenv interpolation",
                )
            values[name] = value.strip()
            if name.endswith("api_key"):
                if "****" in value:
                    raise HTTPException(
                        400, f"{name}: submit a new key, not the displayed mask"
                    )
                if not values[name]:
                    # Blank key fields in the settings form mean keep the saved key.
                    values.pop(name)

    for service in s.credential_binding_errors:
        if f"{service}_api_key" not in values:
            raise HTTPException(
                400,
                f"Resolve the {service} credential binding conflict by explicitly supplying the correct key before saving",
            )

    host_changes: set[str] = set()
    for service in ("llm", "embed", "rerank"):
        host_field = f"{service}_base_url"
        if host_field not in values:
            continue
        try:
            new_base = normalized_api_base(values[host_field])
        except ValueError:
            raise HTTPException(
                400,
                f"{host_field} must be a valid HTTP(S) API base without credentials, query, fragment or template placeholders",
            ) from None
        try:
            old_base = normalized_api_base(
                getattr(s, _ATTR_MAP.get(host_field, host_field))
            )
        except ValueError:
            old_base = None
        if new_base != old_base:
            if f"{service}_api_key" not in values:
                raise HTTPException(
                    400,
                    f"Changing {host_field} requires an explicit new {service}_api_key; use EMPTY only for an unauthenticated endpoint",
                )
            host_changes.add(service)

    if "llm" in host_changes:
        if not values.get("llm_model"):
            raise HTTPException(400, "Changing llm_base_url also requires llm_model")
        for name in (
            "vlm_model",
            "keyword_llm_model",
            "entity_extraction_llm_model",
            "extraction_llm_model",
        ):
            values.setdefault(name, "")

    for name in ("llm_model", "embed_model"):
        if name in values and not values[name]:
            raise HTTPException(400, f"{name} must not be empty")
    candidate = s.model_copy(deep=True)
    for name, value in values.items():
        setattr(candidate, _ATTR_MAP.get(name, name), value)
        if name in ("vlm_model", "keyword_llm_model", "entity_extraction_llm_model"):
            candidate.inherited_auxiliary_models = [
                item for item in candidate.inherited_auxiliary_models if item != name
            ]
            if not value:
                candidate.inherited_auxiliary_models.append(name)
    for service in ("llm", "embed", "rerank"):
        if f"{service}_api_key" in values:
            candidate.credential_binding_errors = [
                item for item in candidate.credential_binding_errors if item != service
            ]
    if "llm_model" in values:
        models = list(
            dict.fromkeys(
                model.strip()
                for model in values["llm_model"].split(",")
                if model.strip()
            )
        )
        if not models:
            raise HTTPException(400, "llm_model must contain at least one model")
        candidate.llm_model, candidate.llm_models = models[0], models

    changed = [
        name
        for name in values
        if _configured_value(candidate, name) != _configured_value(s, name)
    ]
    if candidate.llm_models != s.llm_models and "llm_model" not in changed:
        changed.append("llm_model")
    for service in s.credential_binding_errors:
        key_field = f"{service}_api_key"
        if (
            service not in candidate.credential_binding_errors
            and key_field not in changed
        ):
            changed.append(key_field)
    if changed:
        # Freeze resolved endpoints and credentials, including previously implicit
        # fallbacks. Changing one service must not move another service's old key.
        env_updates = {
            env: str(_configured_value(candidate, name))
            for name, env in _ENV_MAP.items()
        }
        env_updates["LLM_MODELS"] = ",".join(
            candidate.llm_models or [candidate.llm_model]
        )
        env_updates.update({alias: env_updates[key] for key, alias in _ALIASES.items()})
        try:
            for service, binding_env in CREDENTIAL_BINDING_ENV.items():
                env_updates[binding_env] = credential_binding(
                    getattr(candidate, f"{service}_api_key"),
                    getattr(
                        candidate,
                        _ATTR_MAP.get(f"{service}_base_url", f"{service}_base_url"),
                    ),
                )
            _persist_to_dotenv(env_updates)
        except Exception:  # noqa: BLE001 -- persistence errors must never expose secret values
            # Do not disclose file contents, credentials or exception messages.
            raise HTTPException(
                500,
                "Could not persist settings; active and saved settings were not changed",
            ) from None
        _pending_for_runtime = state.settings
        _pending_settings = candidate

    configured = _configured_settings()
    requires_restart = configured is not state.settings

    return {
        "success": True,
        "message": f"Saved: {', '.join(changed)}" if changed else "No changes",
        "changed": changed,
        "requires_restart": requires_restart,
        "warnings": _warnings(configured, state.settings),
        "note": "Saved configuration is pending restart; active clients and environment are unchanged. External process environment variables can override .env on restart.",
    }
