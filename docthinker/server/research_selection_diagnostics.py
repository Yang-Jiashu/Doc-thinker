"""Opt-in, in-process private receipts for offline retrieval diagnosis.

Nothing is persisted or attached to a public trace. Callers must explicitly
provide a callback and own any privacy-sensitive recording it performs.
Callback failures and mutation cannot affect evidence selection.
"""
from __future__ import annotations

from copy import deepcopy


def emit_diagnostic(callback, event, **fields):
    if callback is None:
        return
    try:
        callback(deepcopy({"event": event, **fields}))
    except Exception:
        # Observability must never become a retrieval dependency.
        pass


def window_identity(row):
    return {key: row.get(key) for key in (
        "id", "source_id", "document_id", "ordinal", "page",
        "excerpt_start", "excerpt_end", "section",
    )}
