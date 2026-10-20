from functools import lru_cache

from openai import AsyncOpenAI

from .settings import AppSettings


class CredentialBindingError(ValueError):
    """A saved credential no longer belongs to the effective API destination."""


def assert_credential_bindings(settings: AppSettings, *groups: str) -> None:
    """Fail before creating/reusing a client, without disclosing values or hashes."""
    allowed = {"llm", "embed", "rerank"}
    errors = getattr(settings, "credential_binding_errors", [])
    if not isinstance(errors, (list, tuple)) or any(
        not isinstance(group, str) or group not in allowed for group in errors
    ):
        raise CredentialBindingError("Invalid credential binding status; verify saved connection settings")
    if any(group not in allowed for group in groups):
        raise ValueError("Unknown provider group")
    blocked = sorted(set(errors).intersection(groups or allowed))
    if blocked:
        raise CredentialBindingError(
            "Provider requests blocked for " + ", ".join(blocked)
            + ": credential destination mismatch. Save the correct connection and key, then restart."
        )


@lru_cache(maxsize=8)
def _get_client(api_key: str, base_url: str, timeout_seconds: int, max_retries: int | None = None) -> AsyncOpenAI:
    options = {} if max_retries is None else {"max_retries": max_retries}
    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds, **options)


def get_vlm_client(settings: AppSettings, *, max_retries: int | None = None) -> AsyncOpenAI:
    assert_credential_bindings(settings, "llm")
    if max_retries is not None and (type(max_retries) is not int or not 0 <= max_retries <= 2):
        raise ValueError("Invalid model retry limit")
    return _get_client(settings.llm_api_key, settings.vlm_base_url, settings.timeout_seconds, max_retries)


def get_embed_client(settings: AppSettings) -> AsyncOpenAI:
    assert_credential_bindings(settings, "embed")
    return _get_client(settings.embed_api_key, settings.embed_base_url, settings.timeout_seconds)
