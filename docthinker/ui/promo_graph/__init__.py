"""Standalone promotional knowledge-graph experience."""

from __future__ import annotations

from flask import Blueprint, render_template


promo_graph_bp = Blueprint(
    "promo_graph",
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/promo-graph-assets",
)


def _api_prefix() -> str:
    try:
        from docthinker.api_config import api_config

        return str(api_config.api_prefix or "/api/v1")
    except (ImportError, AttributeError):
        return "/api/v1"


@promo_graph_bp.get("/gesture-experience")
def chooser():
    return render_template("promo_graph/chooser.html")


@promo_graph_bp.get("/promo-graph")
def experience():
    return render_template(
        "promo_graph/experience.html",
        api_prefix=_api_prefix(),
    )


__all__ = ["promo_graph_bp"]
