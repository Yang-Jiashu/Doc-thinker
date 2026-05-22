"""HyperGraphRAG exports with lazy heavy dependencies."""

from __future__ import annotations


def __getattr__(name: str):
    if name in {"HyperGraphRAG", "QueryParam"}:
        from .hypergraphrag import HyperGraphRAG, QueryParam

        return HyperGraphRAG if name == "HyperGraphRAG" else QueryParam
    if name == "bltcy_gpt4o_mini_complete":
        from .bltcy_adapter import bltcy_gpt4o_mini_complete

        return bltcy_gpt4o_mini_complete
    raise AttributeError(name)

__all__ = ["HyperGraphRAG", "QueryParam", "bltcy_gpt4o_mini_complete"]

__version__ = "1.0.6"
__author__ = "Zirui Guo"
__url__ = "https://github.com/HKUDS/HyperGraphRAG"
