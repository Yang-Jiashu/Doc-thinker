"""Router exports are loaded lazily to keep lightweight imports cheap."""

from __future__ import annotations


def __getattr__(name: str):
    if name == "health_router":
        from .health import router
    elif name == "sessions_router":
        from .sessions import router
    elif name == "ingest_router":
        from .ingest import router
    elif name == "query_router":
        from .query import router
    elif name == "graph_router":
        from .graph import router
    elif name == "settings_router":
        from .settings import router
    else:
        raise AttributeError(name)
    return router

__all__ = [
    "health_router",
    "sessions_router",
    "ingest_router",
    "query_router",
    "graph_router",
    "settings_router",
]
