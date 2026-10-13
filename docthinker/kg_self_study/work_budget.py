"""Per-session admission limits for self-study model calls.

The accounting is deliberately conservative and is not provider billing data:
UTF-8 input bytes plus an output reservation are charged before each attempt.
Optional provider usage is reported separately and is never required to proceed.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, NoReturn


class StudyBudgetStop(Exception):
    """A terminal control signal that intermediate P1-P6 handlers must re-raise."""

    def __init__(self, reason: str):
        self.reason = reason
        self.completed_items = 0
        super().__init__(reason)


@dataclass
class StudyWorkBudget:
    max_tokens: int
    max_calls: int
    output_tokens_per_call: int
    timeout_seconds: float
    calls: int = 0
    estimated_input_tokens: int = 0
    reserved_output_tokens: int = 0
    observed_output_bytes: int = 0
    charged_tokens: int = 0
    stop_reason: str = ""
    provider_limited_calls: int = 0
    provider_usage: dict[str, int] = field(default_factory=dict)
    provider_reported_calls: int = 0

    def __post_init__(self) -> None:
        for name in ("max_tokens", "max_calls", "output_tokens_per_call"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.output_tokens_per_call < 1:
            raise ValueError("output_tokens_per_call must be positive")
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")

    def stop(self, reason: str) -> NoReturn:
        self.stop_reason = self.stop_reason or reason
        raise StudyBudgetStop(self.stop_reason)

    def ensure_available(self) -> None:
        if self.stop_reason:
            raise StudyBudgetStop(self.stop_reason)
        if self.calls >= self.max_calls:
            self.stop("max_llm_calls")
        if self.charged_tokens >= self.max_tokens:
            self.stop("token_budget_exhausted")

    def report(self) -> dict[str, Any]:
        return {
            "llm_calls": self.calls,
            "max_llm_calls": self.max_calls,
            "max_tokens": self.max_tokens,
            "token_count_method": "utf8_input_bytes_upper_bound_plus_reserved_output",
            "estimated_input_tokens": self.estimated_input_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "observed_output_bytes": self.observed_output_bytes,
            "charged_tokens": self.charged_tokens,
            "provider_limited_calls": self.provider_limited_calls,  # Legacy name: requests only.
            "output_limit_requested_calls": self.provider_limited_calls,
            "provider_output_limit_verified": False,
            "provider_reported_calls": self.provider_reported_calls,
            "provider_usage": dict(self.provider_usage),
            "stop_reason": self.stop_reason,
        }

    @staticmethod
    def _response(raw: Any) -> tuple[str, dict]:
        if isinstance(raw, str):
            return raw, {}
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        if isinstance(raw, dict):
            text = raw.get("text", raw.get("content"))
            choices = raw.get("choices")
            if text is None and isinstance(choices, list) and choices:
                text = (choices[0].get("message") or {}).get("content")
            if isinstance(text, str):
                usage = raw.get("usage")
                return text, usage if isinstance(usage, dict) else {}
        raise ValueError("LLM must return text or a response with text content")

    async def call(self, llm_func: Callable, prompt: str) -> str:
        self.ensure_available()
        input_tokens = len(prompt.encode("utf-8"))
        reservation = self.output_tokens_per_call
        if input_tokens + reservation > self.max_tokens - self.charged_tokens:
            # Refuse the complete prompt; never silently remove source evidence.
            self.stop("token_budget_exhausted")
        kwargs: dict[str, Any] = {}
        try:
            parameters = inspect.signature(llm_func).parameters
            if "max_tokens" in parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
            ):
                kwargs["max_tokens"] = reservation
        except (TypeError, ValueError):
            pass

        # Failed/cancelled calls keep their reservation: provider-side spend may
        # already have occurred, so refunding it would permit additional work.
        self.calls += 1
        self.estimated_input_tokens += input_tokens
        self.reserved_output_tokens += reservation
        self.charged_tokens += input_tokens + reservation
        self.provider_limited_calls += int(bool(kwargs))
        call_started = time.monotonic()
        try:
            raw = await asyncio.wait_for(
                llm_func(prompt, **kwargs), timeout=self.timeout_seconds
            )
            returned_after_deadline = (
                time.monotonic() - call_started > self.timeout_seconds
            )
            text, usage = self._response(raw)
        except asyncio.CancelledError:
            self.stop_reason = "cancelled"
            raise
        except asyncio.TimeoutError:
            self.stop("llm_timeout")
        except StudyBudgetStop as exc:
            self.stop_reason = exc.reason
            raise
        except Exception:
            self.stop("llm_error")

        clean_usage = {
            name: value
            for name, value in usage.items()
            if name
            in {
                "input_tokens",
                "output_tokens",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            }
            and type(value) is int
            and value >= 0
        }
        if clean_usage:
            self.provider_reported_calls += 1
            for name, value in clean_usage.items():
                self.provider_usage[name] = self.provider_usage.get(name, 0) + value
        output_bytes = len(text.encode("utf-8"))
        self.observed_output_bytes += output_bytes
        # Usage may contain only one component, or both naming conventions.
        # Keep the estimate/reservation for missing fields and take the larger
        # alias if they disagree; absent metadata is not evidence of zero spend.
        reported_input = max(
            clean_usage.get("input_tokens", 0), clean_usage.get("prompt_tokens", 0)
        )
        reported_output_values = [
            clean_usage[name]
            for name in ("output_tokens", "completion_tokens")
            if name in clean_usage
        ]
        reported_output = (
            max(reported_output_values) if reported_output_values else None
        )
        oversized_output = output_bytes > reservation * 8
        output_estimate = (
            output_bytes
            if reported_output is None and (not kwargs or oversized_output)
            else 0
        )
        accounted_input = max(input_tokens, reported_input)
        accounted_output = max(reservation, reported_output or 0, output_estimate)
        # Total and components describe the same attempt. Reconcile with max,
        # never add total on top of its parts and never refund reserved cost.
        attempt_charge = max(
            input_tokens + reservation,
            accounted_input + accounted_output,
            clean_usage.get("total_tokens", 0),
        )
        self.charged_tokens += attempt_charge - input_tokens - reservation
        # Consume all reliable usage before raising any terminal output signal.
        # wait_for relies on cooperative cancellation: a callback may catch the
        # cancellation and return late. Account that response, but never accept
        # it as an in-budget success. This cannot hard-kill a stuck provider.
        if returned_after_deadline:
            self.stop("llm_timeout")
        if self.charged_tokens > self.max_tokens:
            self.stop(
                "provider_usage_exceeded_budget"
                if clean_usage
                else "output_budget_exceeded"
            )
        if oversized_output or accounted_output > reservation:
            self.stop("output_budget_exceeded")
        return text
