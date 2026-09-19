"""Deterministic, multi-dimensional scoring for retrieval/self-evolution A/B tests."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional

_TOKEN_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9_+./-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+"
)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)、]|[（(]?\d+[）)])\s*")
_STOPWORDS = {
    "的", "了", "和", "与", "及", "是", "在", "为", "会", "也", "或", "而",
    "因此", "可以", "需要", "进行", "通过", "the", "a", "an", "and", "or",
    "of", "to", "in", "is", "are", "be", "for", "with", "that", "this",
}


def _tokens(text: str) -> List[str]:
    output: List[str] = []
    for raw in _TOKEN_RE.findall(str(text or "")):
        token = raw.lower()
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if len(token) == 1:
                if token not in _STOPWORDS:
                    output.append(token)
            else:
                output.extend(
                    token[index : index + 2]
                    for index in range(len(token) - 1)
                    if token[index : index + 2] not in _STOPWORDS
                )
        elif token not in _STOPWORDS:
            output.append(token)
    return output


def split_reference_points(reference_answer: str) -> List[str]:
    lines = []
    for raw in str(reference_answer or "").splitlines():
        clean = _BULLET_RE.sub("", raw).strip()
        if clean:
            lines.append(clean)
    if len(lines) > 1:
        return lines
    return [
        part.strip()
        for part in re.split(r"[。！？!?；;]+", str(reference_answer or ""))
        if part.strip()
    ]


def _split_claims(answer: str) -> List[str]:
    return [
        part.strip(" -\t")
        for part in re.split(r"[。！？!?；;\n]+", str(answer or ""))
        if len(_tokens(part)) >= 2
    ]


def _recall(candidate_tokens: Iterable[str], reference_tokens: Iterable[str]) -> float:
    candidate = set(candidate_tokens)
    reference = set(reference_tokens)
    if not reference:
        return 0.0
    return len(candidate & reference) / len(reference)


def _f1(left_tokens: Iterable[str], right_tokens: Iterable[str]) -> float:
    left = set(left_tokens)
    right = set(right_tokens)
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    precision = overlap / len(left)
    recall = overlap / len(right)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_answer(
    *,
    answer: str,
    reference_answer: str,
    evidence_chunks: Optional[Iterable[str]] = None,
    context_chunk_count: Optional[int] = None,
    context_tokens: Optional[int] = None,
    point_threshold: float = 0.55,
) -> Dict[str, Any]:
    """Score quality, grounding, focus, and retrieval cost without one-metric bias."""
    answer_tokens = _tokens(answer)
    reference_tokens = _tokens(reference_answer)
    reference_points = split_reference_points(reference_answer)
    point_scores = [
        _recall(answer_tokens, _tokens(point)) for point in reference_points
    ]
    hard_point_coverage = (
        sum(score >= point_threshold for score in point_scores) / len(point_scores)
        if point_scores
        else 0.0
    )
    soft_point_coverage = (
        sum(point_scores) / len(point_scores) if point_scores else 0.0
    )

    claims = _split_claims(answer)
    evidence = [str(item) for item in (evidence_chunks or []) if str(item).strip()]
    evidence_token_sets = [_tokens(item) for item in evidence]
    grounded_claim_rate: Optional[float] = None
    unsupported_claim_rate: Optional[float] = None
    if evidence_token_sets:
        supported = 0
        for claim in claims:
            claim_tokens = _tokens(claim)
            best_f1 = max((_f1(claim_tokens, item) for item in evidence_token_sets), default=0.0)
            best_recall = max(
                (_recall(item, claim_tokens) for item in evidence_token_sets), default=0.0
            )
            if best_f1 >= 0.25 or best_recall >= 0.45:
                supported += 1
        grounded_claim_rate = supported / len(claims) if claims else 0.0
        unsupported_claim_rate = 1.0 - grounded_claim_rate

    reference_claim_tokens = [_tokens(point) for point in reference_points]
    relevant_claims = 0
    for claim in claims:
        claim_tokens = _tokens(claim)
        if max((_f1(claim_tokens, point) for point in reference_claim_tokens), default=0.0) >= 0.20:
            relevant_claims += 1
    focus_rate = relevant_claims / len(claims) if claims else 0.0

    components = {
        "reference_point_coverage": (soft_point_coverage, 0.40),
        "reference_token_recall": (_recall(answer_tokens, reference_tokens), 0.20),
        "focus_rate": (focus_rate, 0.15),
    }
    if grounded_claim_rate is not None:
        components["grounded_claim_rate"] = (grounded_claim_rate, 0.25)
    weight_sum = sum(weight for _, weight in components.values())
    balanced_score = sum(value * weight for value, weight in components.values()) / weight_sum

    cost_penalty = None
    if context_tokens is not None and context_tokens >= 0:
        cost_penalty = min(1.0, math.log1p(context_tokens) / math.log1p(60000))

    return {
        "balanced_score": round(balanced_score, 4),
        "reference_point_coverage": round(soft_point_coverage, 4),
        "hard_reference_point_coverage": round(hard_point_coverage, 4),
        "reference_token_recall": round(_recall(answer_tokens, reference_tokens), 4),
        "grounded_claim_rate": (
            round(grounded_claim_rate, 4) if grounded_claim_rate is not None else None
        ),
        "unsupported_claim_rate": (
            round(unsupported_claim_rate, 4)
            if unsupported_claim_rate is not None
            else None
        ),
        "focus_rate": round(focus_rate, 4),
        "answer_chars": len(str(answer or "")),
        "claim_count": len(claims),
        "reference_point_count": len(reference_points),
        "context_chunk_count": context_chunk_count,
        "context_tokens": context_tokens,
        "context_cost_penalty": round(cost_penalty, 4) if cost_penalty is not None else None,
    }


def compare_answers(
    *,
    answer_a: str,
    answer_b: str,
    reference_answer: str,
    evidence_a: Optional[Iterable[str]] = None,
    evidence_b: Optional[Iterable[str]] = None,
    context_chunk_count_a: Optional[int] = None,
    context_chunk_count_b: Optional[int] = None,
    context_tokens_a: Optional[int] = None,
    context_tokens_b: Optional[int] = None,
) -> Dict[str, Any]:
    score_a = score_answer(
        answer=answer_a,
        reference_answer=reference_answer,
        evidence_chunks=evidence_a,
        context_chunk_count=context_chunk_count_a,
        context_tokens=context_tokens_a,
    )
    score_b = score_answer(
        answer=answer_b,
        reference_answer=reference_answer,
        evidence_chunks=evidence_b,
        context_chunk_count=context_chunk_count_b,
        context_tokens=context_tokens_b,
    )
    comparable = [
        "balanced_score",
        "reference_point_coverage",
        "hard_reference_point_coverage",
        "reference_token_recall",
        "grounded_claim_rate",
        "unsupported_claim_rate",
        "focus_rate",
        "answer_chars",
        "context_chunk_count",
        "context_tokens",
    ]
    deltas: Dict[str, Optional[float]] = {}
    for key in comparable:
        left, right = score_a.get(key), score_b.get(key)
        deltas[key] = round(float(right) - float(left), 4) if left is not None and right is not None else None
    margin = deltas["balanced_score"] or 0.0
    preferred = "B" if margin > 0.01 else "A" if margin < -0.01 else "tie"
    return {"answer_a": score_a, "answer_b": score_b, "delta_b_minus_a": deltas, "preferred": preferred}


def score_reasoning_paths(
    *,
    expected_chain: Iterable[str],
    paths: Iterable[Dict[str, Any]],
    source_chunks: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Measure ordered structure and citation coverage, not semantic truth.

    Citation presence is only a provenance proxy. Literal quotes are independently
    verified only against caller-supplied source_chunks; neither metric proves
    that the cited text entails a hop's relation or causal direction.
    """
    expected = [str(item).strip() for item in expected_chain if str(item).strip()]
    expected_hops = set(zip(expected, expected[1:]))

    def lcs_length(left: List[str], right: List[str]) -> int:
        previous = [0] * (len(right) + 1)
        for left_item in left:
            current = [0]
            for index, right_item in enumerate(right, 1):
                if left_item == right_item:
                    current.append(previous[index - 1] + 1)
                else:
                    current.append(max(current[-1], previous[index]))
            previous = current
        return previous[-1]

    best = {
        "ordered_node_coverage": 0.0,
        "expected_hop_coverage": 0.0,
        "endpoint_coverage": 0.0,
        "citation_hop_rate": 0.0,
        "verified_quote_hop_rate": 0.0,
        "continuous_hop_rate": 0.0,
        "candidate_edge_rate": 0.0,
        "path_score": 0.0,
    }
    for raw_path in paths:
        nodes = [str(item) for item in (raw_path.get("nodes") or [])]
        hops = [item for item in (raw_path.get("hops") or []) if isinstance(item, dict)]
        declared_hops = [
            (str(item.get("source") or ""), str(item.get("target") or ""))
            for item in hops
        ]
        consecutive = list(zip(nodes, nodes[1:]))
        actual_hops = {
            pair for index, pair in enumerate(declared_hops)
            if index < len(consecutive) and pair == consecutive[index]
        }
        continuous = sum(
            left == right for left, right in zip(declared_hops, consecutive)
        ) / max(1, len(declared_hops), len(consecutive))
        ordered = lcs_length(expected, nodes) / len(expected) if expected else 0.0
        hop_coverage = (
            len(expected_hops & actual_hops) / len(expected_hops)
            if expected_hops
            else 0.0
        )
        endpoint = (
            float(bool(nodes) and nodes[0] == expected[0])
            + float(bool(nodes) and nodes[-1] == expected[-1])
        ) / 2 if expected else 0.0
        cited = (
            sum(bool(str(item.get("source_id") or "").strip()) for item in hops) / len(hops)
            if hops
            else 0.0
        )
        candidate_rate = (
            sum(str(item.get("candidate", "")).lower() in {"true", "1"} for item in hops) / len(hops)
            if hops
            else 0.0
        )
        verified_quotes = 0
        for hop in hops:
            refs = hop.get("evidence") or []
            if not isinstance(refs, list):
                continue
            declared_sources = {
                value.strip() for value in str(hop.get("source_id") or "").split("<SEP>")
                if value.strip()
            }
            if any(
                isinstance(ref, dict)
                and str(ref.get("chunk_id")) in declared_sources
                and str(ref.get("quote") or "").strip()
                and str(ref.get("quote")) in (source_chunks or {}).get(str(ref.get("chunk_id")), "")
                for ref in refs
            ):
                verified_quotes += 1
        try:
            path_score = float(raw_path.get("score") or 0.0)
        except (TypeError, ValueError):
            path_score = 0.0
        if not math.isfinite(path_score):
            path_score = 0.0
        metrics = {
            "ordered_node_coverage": ordered,
            "expected_hop_coverage": hop_coverage,
            "endpoint_coverage": endpoint,
            "citation_hop_rate": cited,
            "verified_quote_hop_rate": verified_quotes / len(hops) if hops else 0.0,
            "continuous_hop_rate": continuous,
            "candidate_edge_rate": candidate_rate,
            "path_score": path_score,
        }
        if (
            metrics["expected_hop_coverage"],
            metrics["ordered_node_coverage"],
            metrics["path_score"],
        ) > (
            best["expected_hop_coverage"],
            best["ordered_node_coverage"],
            best["path_score"],
        ):
            best = metrics
    return {key: round(value, 4) for key, value in best.items()}
