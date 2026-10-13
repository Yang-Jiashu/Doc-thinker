"""Deterministic, text-only request bounds for invitation chat workers.

These are request-size and completion-token limits, not tokenizer estimates or
a currency/billing guarantee. They never summarize history with another model.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar

MAX_QUESTION_BYTES = 8192
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_MESSAGE_BYTES = 2048
MAX_HISTORY_BYTES = 8192
MAX_INPUT_BYTES = 24576
MAX_OUTPUT_TOKENS = 2048
# A provider's completion allowance can include reasoning as well as the
# delivered text. Evidence-backed synthesis gets headroom, not a second call.
# This is a task policy shared by every model, not a Step-only token override.
GROUNDED_OUTPUT_TOKENS = 8192
_OUTPUT_BUDGET: ContextVar[int | None] = ContextVar("docthinker_output_budget", default=None)
_TRUNCATED = "\n[历史片段已截断]\n"

INVITE_REQUEST_BUDGET = {
    "max_question_bytes": MAX_QUESTION_BYTES,
    "max_history_messages": MAX_HISTORY_MESSAGES,
    "max_history_bytes": MAX_HISTORY_BYTES,
    "max_input_bytes": MAX_INPUT_BYTES,
    "max_output_tokens": MAX_OUTPUT_TOKENS,
    "max_grounded_output_tokens": GROUNDED_OUTPUT_TOKENS,
    "automatic_retry": False,
    "history_policy": "recent_bounded",
}


def current_output_tokens() -> int:
    """Return the trusted execution budget; user prompt text cannot change it."""
    return _OUTPUT_BUDGET.get() or MAX_OUTPUT_TOKENS


def has_output_budget() -> bool:
    return _OUTPUT_BUDGET.get() is not None


@contextmanager
def completion_output_budget(limit: int):
    """Bind a bounded per-task allowance and restore it on failure/cancellation.

    Nested work may reduce but cannot raise the enclosing call allowance. The
    provider still receives one hard completion cap, with no automatic repair.
    """
    if type(limit) is not int or not 1 <= limit <= GROUNDED_OUTPUT_TOKENS:
        raise ValueError("Invalid completion budget")
    previous = _OUTPUT_BUDGET.get()
    token = _OUTPUT_BUDGET.set(min(previous, limit) if previous is not None else limit)
    try:
        yield
    finally:
        _OUTPUT_BUDGET.reset(token)


class InviteBudgetError(ValueError):
    """Diagnostics are fixed messages, never supplied user/provider content."""

    def __init__(self, *, invalid_text: bool = False):
        self.status = 422 if invalid_text else 413
        super().__init__("Invalid text encoding" if invalid_text else "Request exceeds the invitation chat budget")


def is_invite_worker() -> bool:
    return bool(os.getenv("DOCTHINKER_INVITE_WORKER_ID", "").strip())


def _encoded(text: str) -> bytes:
    if not isinstance(text, str):
        raise InviteBudgetError(invalid_text=True)
    try:
        return text.encode("utf-8", errors="strict")
    except UnicodeError:
        raise InviteBudgetError(invalid_text=True) from None


def validate_invite_question(question: str) -> None:
    """Reject an oversized or invalid current turn; never silently truncate it."""
    if len(_encoded(question)) > MAX_QUESTION_BYTES:
        raise InviteBudgetError()


def _clip(text: str, limit: int) -> str:
    raw = _encoded(text)
    if len(raw) <= limit:
        return text
    marker = _TRUNCATED.encode("utf-8")
    if limit <= len(marker):
        return raw[:limit].decode("utf-8", errors="ignore")
    allowance = limit - len(marker)
    head = allowance // 2
    tail = allowance - head
    # Dropping only split UTF-8 boundary bytes preserves valid output while
    # keeping the fixed byte cap. Retain both ends of a long historical turn.
    return raw[:head].decode("utf-8", errors="ignore") + _TRUNCATED + raw[-tail:].decode("utf-8", errors="ignore")


def bounded_invite_history(history: list[dict] | tuple[dict, ...] | None) -> list[dict[str, str]]:
    """Newest turns win; work, total bytes and individual turns are bounded."""
    if not isinstance(history, (list, tuple)):
        return []
    valid = []
    for item in history[-MAX_HISTORY_MESSAGES:]:
        if (not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}
                or not isinstance(item.get("content"), str)):
            continue
        try:
            _encoded(item["content"])
        except InviteBudgetError:
            continue  # An old corrupt turn is not a new user request.
        if item["content"].strip():
            valid.append({"role": item["role"], "content": item["content"]})
    kept, remaining = [], MAX_HISTORY_BYTES
    for item in reversed(valid[-MAX_HISTORY_MESSAGES:]):
        # Avoid inserting a meaningless few-byte tail after UTF-8 boundary
        # rounding. A clipped turn must have room for its disclosure marker.
        if remaining < len(_TRUNCATED.encode("utf-8")) + 8:
            break
        content = _clip(item["content"], min(MAX_HISTORY_MESSAGE_BYTES, remaining))
        remaining -= len(content.encode("utf-8"))
        kept.append({"role": item["role"], "content": content})
    return list(reversed(kept))


def validate_invite_messages(messages: list[dict]) -> None:
    """Check the complete adapter input, including system and prompt wrappers.

    Only plain text is supported: no media/tool payload can bypass this budget.
    The cap measures role/content UTF-8 bytes, not provider billed input tokens.
    """
    if not isinstance(messages, list) or not messages or len(messages) > MAX_HISTORY_MESSAGES + 2:
        raise InviteBudgetError(invalid_text=True)
    size = 0
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] not in {"system", "user", "assistant"}):
            raise InviteBudgetError(invalid_text=True)
        size += len(_encoded(message["role"])) + len(_encoded(message["content"]))
        if size > MAX_INPUT_BYTES:
            raise InviteBudgetError()
