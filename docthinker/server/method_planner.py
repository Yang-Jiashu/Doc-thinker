"""Bounded structural method receipts and an offline paired evaluation gate.

Neither function reads a store, history, files or the network, calls a model,
or writes memory. A method receipt verifies excerpt *structure*, never whether
the natural-language premises or stop conditions of a method hold. Original
cards remain in the separate K-reference prompt, including their full wording.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from numbers import Real
from .research_source_contract import MAX_ANSWER_SOURCES, SOURCE_LABEL_PATTERN

MAX_PLAN_BYTES = 1_600
MAX_METHODS = 3
MAX_SOURCES = MAX_ANSWER_SOURCES
MAX_EXCERPT_CHARS = 64_000
_SOURCE_LABEL = SOURCE_LABEL_PATTERN
_METHOD_LABEL = re.compile(r"K[1-9][0-9]{0,3}\Z")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_ROLES = {"analogy", "deduction", "induction"}
_GAP_ORDER = (
    "invalid_sources", "source_limit_exceeded", "invalid_source_labels",
    "duplicate_source_labels", "empty_excerpts", "missing_document_ids",
    "invalid_source_ids", "invalid_excerpt_geometry", "no_usable_excerpts",
    "insufficient_distinct_documents",
)


def _text(value):
    if not isinstance(value, str) or len(value) > MAX_EXCERPT_CHARS:
        return ""
    try:
        value.encode("utf-8")
    except UnicodeError:
        return ""
    return value


def _sequence(value):
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _json_size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                          allow_nan=False).encode("utf-8"))


def compile_method_plan(methods, sources, *, skill_id):
    """Compile only the supplied selected K cards and current S excerpts.

    ``distinct_documents`` describes document-ID diversity after exact excerpt
    deduplication (whitespace normalized), not independent studies or support.
    Invalid/ambiguous source labels are never credited. Truncation is retained
    as a limitation even when a stale ``truncated=False`` contradicts geometry.
    A structural gap requests collection; it does not block the normal writer.
    """
    cards = []
    for raw in methods[:MAX_METHODS] if _sequence(methods) else ():
        if not isinstance(raw, Mapping):
            continue
        identifier, label = _text(raw.get("id")), _text(raw.get("label"))
        if not _IDENTIFIER.fullmatch(identifier) or not _METHOD_LABEL.fullmatch(label):
            continue
        facts = raw.get("required_facts", ())
        card = {
            "id": identifier, "label": label,
            "requirement_count": len(facts) if _sequence(facts) else 0,
            "requirements_status": "pending_review",
            "stop_conditions_status": "pending_review",
        }
        role = raw.get("reasoning_role")
        if isinstance(role, str) and role in _ROLES:
            card["reasoning_role"] = role
        cards.append(card)

    gaps = set()
    rows = list(sources[:MAX_SOURCES]) if _sequence(sources) else []
    if not _sequence(sources):
        gaps.add("invalid_sources")
    elif len(sources) > MAX_SOURCES:
        gaps.add("source_limit_exceeded")
    label_counts = {}
    for row in rows:
        if isinstance(row, Mapping):
            label = _text(row.get("label"))
            if _SOURCE_LABEL.fullmatch(label):
                label_counts[label] = label_counts.get(label, 0) + 1
    if any(count > 1 for count in label_counts.values()):
        gaps.add("duplicate_source_labels")

    labels, truncated, duplicates = [], [], []
    documents, excerpts = set(), set()
    usable = 0
    for row in rows:
        if not isinstance(row, Mapping):
            gaps.add("invalid_sources")
            continue
        label = _text(row.get("label"))
        if not _SOURCE_LABEL.fullmatch(label):
            gaps.add("invalid_source_labels")
            continue
        if label_counts[label] != 1:
            continue
        labels.append(label)
        content = _text(row.get("content"))
        document = _text(row.get("document_id")).strip()
        identifier = _text(row.get("id"))
        nonempty = bool(content.strip())
        if not nonempty:
            gaps.add("empty_excerpts")
        if not document.strip() or len(document) > 256:
            gaps.add("missing_document_ids")
        if not identifier.strip() or len(identifier) > 256:
            gaps.add("invalid_source_ids")
        start = row.get("excerpt_start", 0)
        end = row.get("excerpt_end", len(content) + start if type(start) is int else None)
        original_length = row.get("original_content_length", end)
        geometry_valid = (type(start) is int and start >= 0 and type(end) is int
                          and end == start + len(content) and type(original_length) is int
                          and original_length >= end)
        if not geometry_valid:
            gaps.add("invalid_excerpt_geometry")
        if row.get("truncated") is True or (geometry_valid and (start > 0 or original_length > end)):
            truncated.append(label)
        if (not nonempty or not document.strip() or len(document) > 256
                or not identifier.strip() or len(identifier) > 256 or not geometry_valid):
            continue
        usable += 1
        normalized = " ".join(content.split())
        if normalized in excerpts:
            duplicates.append(label)
            continue
        excerpts.add(normalized)
        documents.add(document)
    if not excerpts:
        gaps.add("no_usable_excerpts")
    if skill_id == "literature-synthesis" and len(documents) < 2:
        gaps.add("insufficient_distinct_documents")

    plan = {
        "kind": "method_plan", "version": 1,
        "skill_id": skill_id if isinstance(skill_id, str) and _IDENTIFIER.fullmatch(skill_id) else "",
        "methods": cards, "source_labels": labels,
        "evidence": {"selected_sources": len(rows), "usable_sources": usable,
                     "unique_excerpts": len(excerpts), "distinct_documents": len(documents),
                     "truncated_sources": truncated, "duplicate_excerpts": duplicates},
        "structural_gaps": [code for code in _GAP_ORDER if code in gaps],
        "decision": "collect_evidence" if gaps else "review_requirements",
        "semantic_entailment_verified": False, "model_calls": 0,
        "method_execution_claimed": False, "normal_answer_allowed": True,
    }
    # Original IDs are already present on the K cards. In adversarial inputs a
    # JSON-escaped ID can still be large; labels preserve exact card references.
    if _json_size(plan) > MAX_PLAN_BYTES:
        for card in cards:
            card.pop("id", None)
    if _json_size(plan) > MAX_PLAN_BYTES:
        for card in cards:
            card.pop("reasoning_role", None)
    assert _json_size(plan) <= MAX_PLAN_BYTES
    return plan


def _number(value, field, *, quality=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{field} must be a finite number") from None
    if not math.isfinite(result) or result < 0 or (quality and result > 1):
        raise ValueError(f"{field} is out of range")
    return result


def evaluate_method_revision(pairs, *, minimum_pairs=12):
    """Offline review eligibility, not deployment, novelty, or significance.

    Every task must have an independent, uniquely identified baseline/candidate
    pair supplied by the caller. Independence cannot be inferred from IDs and
    must be established by the evaluation protocol. No pooling can conceal an
    unsupported-claim regression on a task. Strictly positive average quality
    gain and a strict majority of quality wins are both required. Budget ratios
    use aggregate means; equal zero budgets have ratio 1, increases from zero
    fail with a JSON-safe ``None`` ratio. At least twelve pairs is mandatory.
    """
    if type(minimum_pairs) is not int or minimum_pairs < 12:
        raise ValueError("minimum_pairs must be an integer of at least 12")
    if not _sequence(pairs):
        raise ValueError("pairs must be a sequence of independent task pairs")
    seen, measurements = set(), []
    fields = ("quality", "unsupported_claims", "input_tokens", "latency_ms")
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ValueError("each pair must be an object")
        task_id = _text(pair.get("task_id")).strip()
        if not task_id or task_id in seen:
            raise ValueError("task_id must be nonempty and unique")
        seen.add(task_id)
        sides = []
        for name in ("baseline", "candidate"):
            raw = pair.get(name)
            if not isinstance(raw, Mapping):
                raise ValueError(f"{name} must contain all evaluation metrics")
            sides.append({key: _number(raw.get(key), f"{name}.{key}", quality=key == "quality")
                          for key in fields})
        measurements.append(tuple(sides))
    count = len(measurements)

    def mean(index, key):
        values = [pair[index][key] for pair in measurements]
        scale = max(values, default=0.0)
        # Scaling before summation keeps means of large but finite costs finite.
        return (math.fsum(value / scale for value in values) / count) * scale if scale else 0.0

    means = {side: {key: mean(index, key) for key in fields}
             for index, side in enumerate(("baseline", "candidate"))}
    wins = sum(candidate["quality"] > baseline["quality"] for baseline, candidate in measurements)
    regressions = sum(candidate["unsupported_claims"] > baseline["unsupported_claims"]
                      for baseline, candidate in measurements)
    gain = (math.fsum(candidate["quality"] - baseline["quality"]
                     for baseline, candidate in measurements) / count if count else 0.0)

    def ratio(key):
        base, candidate = means["baseline"][key], means["candidate"][key]
        if base == 0 and any(pair[0][key] > 0 for pair in measurements):
            # An average of subnormal, nonzero costs can round to zero. Compute
            # its ratio in a common scaled domain, not as a free zero budget.
            scale = max(pair[index][key] for pair in measurements for index in (0, 1))
            base = math.fsum(pair[0][key] / scale for pair in measurements)
            candidate = math.fsum(pair[1][key] / scale for pair in measurements)
        value = candidate / base if base else (1.0 if candidate == 0 else None)
        return value if value is None or math.isfinite(value) else None

    token_ratio, latency_ratio = ratio("input_tokens"), ratio("latency_ms")
    gates = {"minimum_pairs": count >= minimum_pairs,
             "mean_quality_improved": gain > 0,
             "majority_quality_wins": wins > count / 2,
             "no_unsupported_claim_regressions": regressions == 0,
             "input_token_budget": token_ratio is not None and token_ratio <= 1.1,
             "latency_budget": latency_ratio is not None and latency_ratio <= 1.2}
    return {"kind": "offline_method_revision_gate", "version": 1,
            "eligible_for_review": all(gates.values()), "pair_count": count,
            "gates": gates, "failed_gates": [name for name, passed in gates.items() if not passed],
            "thresholds": {"minimum_pairs": minimum_pairs, "mean_quality_gain_exclusive_min": 0,
                           "quality_win_fraction_exclusive_min": 0.5,
                           "unsupported_claim_regressions_max": 0,
                           "input_token_ratio_max": 1.1, "latency_ratio_max": 1.2},
            "metrics": {"baseline_means": means["baseline"], "candidate_means": means["candidate"],
                        "mean_quality_gain": gain, "quality_wins": wins,
                        "unsupported_claim_regressions": regressions,
                        "input_token_ratio": token_ratio, "latency_ratio": latency_ratio},
            "automatic_publication": False, "statistical_significance_proven": False,
            "novelty_proven": False, "pair_independence_verified": False, "model_calls": 0}
