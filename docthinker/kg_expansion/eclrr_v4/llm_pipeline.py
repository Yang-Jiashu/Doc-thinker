"""Independent Generator and Judge calls with strict JSON parsing."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Callable

from .models import (
    ECLRRConfig,
    EvidencePackage,
    JudgeDecision,
    Proposal,
)
from .prompts import RELATION_FAMILIES, fit_judge_prompt, plan_generator_batches


def _required_evidence_refs(package: EvidencePackage) -> set[str]:
    return {
        item.evidence_id
        for item in (*package.primary_evidence, *package.direct_evidence)
    }


def _strict_object(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise ValueError("llm_output_not_string")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("llm_output_not_object")
    return value


def parse_proposals(raw: Any, allowed_review_ids: set[str]) -> list[Proposal]:
    payload = _strict_object(raw)
    items = payload.get("proposals")
    if not isinstance(items, list):
        raise ValueError("missing_proposals_array")
    proposals: list[Proposal] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("invalid_proposal")
        review_id = str(item.get("review_id") or "").strip()
        family = str(item.get("relation_family") or "").strip().lower()
        if review_id not in allowed_review_ids or review_id in seen:
            raise ValueError("unknown_or_duplicate_review_id")
        if family not in RELATION_FAMILIES:
            raise ValueError("relation_family_outside_ontology")
        refs = item.get("evidence_refs")
        if not isinstance(refs, list):
            raise ValueError("invalid_evidence_refs")
        proposals.append(
            Proposal(
                review_id=review_id,
                source=str(item.get("source") or "").strip(),
                target=str(item.get("target") or "").strip(),
                relation=str(item.get("relation") or "").strip(),
                relation_family=family,
                direction=str(item.get("direction") or "").strip().lower(),
                description=str(item.get("description") or "").strip(),
                evidence_refs=tuple(
                    dict.fromkeys(str(ref).strip() for ref in refs if str(ref).strip())
                ),
            )
        )
        seen.add(review_id)
    return proposals


def parse_judge_decision(raw: Any, review_id: str) -> JudgeDecision:
    payload = _strict_object(raw)
    items = payload.get("decisions")
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
        raise ValueError("judge_requires_one_decision")
    item = items[0]
    if str(item.get("review_id") or "").strip() != review_id:
        raise ValueError("judge_review_id_mismatch")
    decision = str(item.get("decision") or "").strip().lower()
    if decision not in {"accept", "revise", "reject"}:
        raise ValueError("invalid_judge_decision")
    scores = {
        "evidence_coverage": (0, 4),
        "semantic_composability": (0, 3),
        "relation_direction": (0, 2),
        "uncertainty_calibration": (0, 1),
        "total": (0, 10),
    }
    parsed_scores: dict[str, int] = {}
    for key, (minimum, maximum) in scores.items():
        value = item.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= maximum
        ):
            raise ValueError(f"invalid_judge_score:{key}")
        parsed_scores[key] = value
    component_total = sum(
        parsed_scores[key]
        for key in (
            "evidence_coverage",
            "semantic_composability",
            "relation_direction",
            "uncertainty_calibration",
        )
    )
    if parsed_scores["total"] != component_total:
        raise ValueError("judge_total_mismatch")
    reason_codes = item.get("reason_codes")
    refs = item.get("verified_evidence_refs")
    if not isinstance(reason_codes, list) or not isinstance(refs, list):
        raise ValueError("invalid_judge_arrays")
    revised_relation = item.get("revised_relation")
    revised_family = item.get("revised_relation_family")
    if decision == "revise" and (
        not revised_relation or revised_family not in RELATION_FAMILIES
    ):
        raise ValueError("revise_missing_relation")
    return JudgeDecision(
        review_id=review_id,
        decision=decision,  # type: ignore[arg-type]
        evidence_coverage=parsed_scores["evidence_coverage"],
        semantic_composability=parsed_scores["semantic_composability"],
        relation_direction=parsed_scores["relation_direction"],
        uncertainty_calibration=parsed_scores["uncertainty_calibration"],
        total=parsed_scores["total"],
        reason_codes=tuple(
            str(code).strip() for code in reason_codes if str(code).strip()
        ),
        revised_relation=str(revised_relation).strip() if revised_relation else None,
        revised_relation_family=(
            str(revised_family).strip().lower() if revised_family else None
        ),
        verified_evidence_refs=tuple(
            dict.fromkeys(str(ref).strip() for ref in refs if str(ref).strip())
        ),
    )


async def run_generator_and_judge(
    packages: list[EvidencePackage],
    generator_func: Callable,
    judge_func: Callable,
    config: ECLRRConfig,
    *,
    audit: Any = None,
) -> tuple[list[Proposal], list[JudgeDecision], dict[str, str]]:
    proposals: list[Proposal] = []
    decisions: list[JudgeDecision] = []
    failures: dict[str, str] = {}
    package_by_id = {item.review_item.review_id: item for item in packages}
    try:
        batches = plan_generator_batches(packages, config)
    except ValueError as exc:
        return [], [], {"batching": str(exc)}

    for batch_index, (batch, prompt) in enumerate(batches):
        ids = {item.review_item.review_id for item in batch}
        try:
            raw = ""
            parsed: list[Proposal] | None = None
            last_error: Exception | None = None
            for attempt in range(config.max_llm_retries + 1):
                try:
                    raw = await generator_func(
                        prompt,
                        max_tokens=config.max_output_tokens,
                        temperature=0.0,
                    )
                    parsed = parse_proposals(raw, ids)
                    for proposal in parsed:
                        required = _required_evidence_refs(
                            package_by_id[proposal.review_id]
                        )
                        if set(proposal.evidence_refs) != required:
                            raise ValueError("generator_evidence_refs_incomplete")
                    break
                except Exception as exc:
                    last_error = exc
                    if audit:
                        audit.record_llm(
                            "generator-attempt",
                            batch_index * (config.max_llm_retries + 1) + attempt,
                            prompt,
                            raw,
                            {"error": str(exc)},
                        )
            if parsed is None:
                raise last_error or ValueError("generator_failed_without_error")
            proposals.extend(parsed)
            if audit:
                audit.record_llm(
                    "generator",
                    batch_index,
                    prompt,
                    raw,
                    [asdict(item) for item in parsed],
                )
        except Exception as exc:
            for review_id in ids:
                failures[review_id] = f"generator_failure:{type(exc).__name__}:{exc}"
            if audit:
                audit.record_llm(
                    "generator", batch_index, prompt, "", {"error": str(exc)}
                )

    for index, proposal in enumerate(proposals):
        package = package_by_id[proposal.review_id]
        try:
            prompt = fit_judge_prompt(package, proposal, config)
        except ValueError as exc:
            failures[proposal.review_id] = str(exc)
            continue
        try:
            raw = ""
            decision: JudgeDecision | None = None
            last_error: Exception | None = None
            for attempt in range(config.max_llm_retries + 1):
                try:
                    raw = await judge_func(
                        (
                            prompt
                            if last_error is None
                            else prompt
                            + "\nPREVIOUS_OUTPUT_VALIDATION_ERROR="
                            + str(last_error)
                            + ". Return corrected strict JSON only."
                        ),
                        max_tokens=config.max_output_tokens,
                        temperature=0.0,
                    )
                    decision = parse_judge_decision(raw, proposal.review_id)
                    if (
                        decision.decision == "reject"
                        and decision.total >= 8
                        and decision.evidence_coverage == 4
                        and decision.semantic_composability >= 2
                        and decision.relation_direction >= 1
                    ):
                        raise ValueError("judge_reject_conflicts_with_passing_scores")
                    if (
                        decision.evidence_coverage == 4
                        and any(
                            "missing_direct_evidence" in code.casefold()
                            for code in decision.reason_codes
                        )
                    ):
                        raise ValueError("judge_score_reason_conflict")
                    if (
                        decision.decision in {"accept", "revise"}
                        and set(decision.verified_evidence_refs)
                        != _required_evidence_refs(package)
                    ):
                        raise ValueError("judge_evidence_refs_incomplete")
                    break
                except Exception as exc:
                    last_error = exc
                    if audit:
                        audit.record_llm(
                            "judge-attempt",
                            index * (config.max_llm_retries + 1) + attempt,
                            prompt,
                            raw,
                            {"error": str(exc)},
                        )
            if decision is None:
                raise last_error or ValueError("judge_failed_without_error")
            decisions.append(decision)
            if audit:
                audit.record_llm("judge", index, prompt, raw, asdict(decision))
        except Exception as exc:
            failures[proposal.review_id] = f"judge_failure:{type(exc).__name__}:{exc}"
            if audit:
                audit.record_llm("judge", index, prompt, "", {"error": str(exc)})
    return proposals, decisions, failures
