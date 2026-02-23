from functools import lru_cache
from openai import AsyncOpenAI

from .settings import AppSettings


@lru_cache(maxsize=8)
def _get_client(api_key: str, base_url: str, timeout_seconds: int) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)


def get_vlm_client(settings: AppSettings) -> AsyncOpenAI:
    return _get_client(settings.llm_api_key, settings.vlm_base_url, settings.timeout_seconds)


def get_embed_client(settings: AppSettings) -> AsyncOpenAI:
    return _get_client(settings.embed_api_key, settings.embed_base_url, settings.timeout_seconds)

