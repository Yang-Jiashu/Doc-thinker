"""Server package exports without importing the full FastAPI app eagerly."""

from __future__ import annotations


def __getattr__(name: str):
    if name in {"app", "create_app"}:
        from .app import app, create_app

        return app if name == "app" else create_app
    raise AttributeError(name)

__all__ = ["app", "create_app"]
