"""Strict JSON prompts and prompt-budget planning for ECLRR-v4."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .models import ECLRRConfig, EvidencePackage, Proposal

RELATION_FAMILIES = (
    "kinship",
    "romantic",
    "friendship",
    "causation",
    "influence",
    "collaboration",
    "dependency",
    "ownership",
    "location",
    "participation",
    "identity",
    "communication",
    "protection",
    "opposition",
    "succession",
    "affiliation",
    "temporal",
    "other_specific",
)

GENERATOR_INSTRUCTION = """You are the Generator in ECLRR-v4.
Use only the supplied EvidencePackage objects. Propose a formal endpoint relation only
when every hop composes semantically. You do not approve graph writes. Do not use
outside knowledge, endpoint co-occurrence, node descriptions, or unlisted evidence.
The endpoint distance is at least three hops. Choose one relation_family from the
ontology and a precise snake_case relation; generic values such as related,
associated, connected, 联系, or 相关 are forbidden. Preserve direction. Cite every
primary evidence_id used by the path. Output strict JSON only, with no Markdown:
{"proposals":[{"review_id":"...","source":"...","target":"...",
"relation":"...","relation_family":"...","direction":"source_to_target|target_to_source|undirected",
"description":"...","evidence_refs":["ev-..."]}]}
Return {"proposals":[]} when no formal relation is justified.
"""

JUDGE_INSTRUCTION = """You are the independent Judge in ECLRR-v4.
Re-read the original EvidencePackage and the Generator Proposal. Reject relation
generalization, invented facts, missing hop evidence, direction conflicts, and
non-composable chains. Scores are decision scores, not probabilities. Output strict
JSON only, with no Markdown:
{"decisions":[{"review_id":"...","decision":"accept|revise|reject",
"evidence_coverage":0,"semantic_composability":0,"relation_direction":0,
"uncertainty_calibration":0,"total":0,"reason_codes":[],
"revised_relation":null,"revised_relation_family":null,
"verified_evidence_refs":["ev-..."]}]}
Ranges: evidence_coverage 0-4, semantic_composability 0-3,
relation_direction 0-2, uncertainty_calibration 0-1, total 0-10.
Use revise only when the evidence supports a more precise relation than the Proposal.
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def generator_prompt(packages: list[EvidencePackage], detail: int = 0) -> str:
    include_alternates = detail == 0
    include_context = detail <= 1
    payload = {
        "relation_family_ontology": list(RELATION_FAMILIES),
        "items": [
            package.to_prompt_dict(
                include_alternates=include_alternates,
                include_context=include_context,
            )
            for package in packages
        ],
    }
    if detail >= 2:
        for item in payload["items"]:
            item["supporting_paths"] = []
    return GENERATOR_INSTRUCTION + "\nINPUT=" + _json(payload)


def plan_generator_batches(
    packages: list[EvidencePackage],
    config: ECLRRConfig,
) -> list[tuple[list[EvidencePackage], str]]:
    """Fit batches by dropping optional evidence, never primary evidence."""

    def is_contiguous_slice(shorter: tuple[str, ...], longer: tuple[str, ...]) -> bool:
        width = len(shorter)
        return any(
            longer[index : index + width] == shorter
            for index in range(len(longer) - width + 1)
        )

    longest_paths = sorted(
        (package.review_item.primary_path.nodes for package in packages),
        key=lambda nodes: (-len(nodes), nodes),
    )
    chain_by_review: dict[str, tuple[str, ...]] = {}
    for package in packages:
        nodes = package.review_item.primary_path.nodes
        chain_by_review[package.review_item.review_id] = next(
            (
                candidate
                for candidate in longest_paths
                if is_contiguous_slice(nodes, candidate)
            ),
            nodes,
        )

    batches: list[tuple[list[EvidencePackage], str]] = []
    cursor = 0
    while cursor < len(packages):
        upper = cursor
        chains: set[tuple[str, ...]] = set()
        while upper < len(packages) and upper - cursor < config.max_generator_items:
            chain = chain_by_review[packages[upper].review_item.review_id]
            if chain not in chains and len(chains) >= config.max_generator_chains:
                break
            chains.add(chain)
            upper += 1
        accepted: tuple[list[EvidencePackage], str] | None = None
        while upper > cursor:
            candidate = packages[cursor:upper]
            for detail in range(3):
                prompt = generator_prompt(candidate, detail=detail)
                if len(prompt) <= config.max_prompt_chars:
                    accepted = (candidate, prompt)
                    break
            if accepted:
                break
            upper -= 1
        if accepted is None:
            single = [packages[cursor]]
            prompt = generator_prompt(single, detail=2)
            if len(prompt) > config.max_prompt_chars:
                raise ValueError(
                    f"primary_evidence_exceeds_prompt_budget:{packages[cursor].review_item.review_id}"
                )
            accepted = (single, prompt)
        batches.append(accepted)
        cursor += len(accepted[0])
    return batches


def judge_prompt(package: EvidencePackage, proposal: Proposal, detail: int = 0) -> str:
    payload = {
        "relation_family_ontology": list(RELATION_FAMILIES),
        "evidence_package": package.to_prompt_dict(
            include_alternates=detail == 0,
            include_context=detail <= 1,
        ),
        "proposal": asdict(proposal),
    }
    if detail >= 2:
        payload["evidence_package"]["supporting_paths"] = []
    return JUDGE_INSTRUCTION + "\nINPUT=" + _json(payload)


def fit_judge_prompt(
    package: EvidencePackage, proposal: Proposal, config: ECLRRConfig
) -> str:
    for detail in range(3):
        prompt = judge_prompt(package, proposal, detail=detail)
        if len(prompt) <= config.max_prompt_chars:
            return prompt
    raise ValueError(
        f"judge_primary_evidence_exceeds_prompt_budget:{proposal.review_id}"
    )
