"""
Azure OpenAI compatibility layer.

This module provides backward compatibility by re-exporting Azure OpenAI functions
from the main openai module where the actual implementation resides.

All core logic for both OpenAI and Azure OpenAI now lives in graphcore.coregraph.llm.openai,
with this module serving as a thin compatibility wrapper for existing code that
imports from graphcore.coregraph.llm.azure_openai.
"""

from graphcore.coregraph.llm.openai import (
    azure_openai_complete_if_cache,
    azure_openai_complete,
    azure_openai_embed,
)

__all__ = [
    "azure_openai_complete_if_cache",
    "azure_openai_complete",
    "azure_openai_embed",
]
