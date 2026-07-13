"""Strict JSON prompts and prompt-budget planning for ECLRR-v4."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .models import ECLRRConfig, EvidencePackage, Proposal
from .clue_semantics import SYMMETRIC_RELATIONS, clue_contract

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

CANONICAL_RELATIONS = (
    "parent_child",
    "siblings",
    "spouses",
    "love",
    "liking",
    "jealousy",
    "hatred",
    "trust",
    "friendship",
    "collaboration",
    "participation",
    "influence",
    "causation",
    "teacher_student",
    "enemies",
    "colleagues",
    "protection",
    "communication",
    "dependency",
    "ownership",
    "succession",
    "identity",
    "affiliation",
    "location",
    "temporal_relation",
    "opposition",
)

GENERATOR_INSTRUCTION = """You are the Generator in ECLRR-v4.
Use only the supplied EvidencePackage objects. For refine items,
direct_endpoint_evidence is the primary hidden-relation clue and the 3-8 hop path is
the supporting context chain. The path establishes evidence-grounded narrative
connectivity and may contain heterogeneous relation families; it does not need to
entail or repeat the endpoint relation. Propose a formal endpoint relation when the
direct clue supports it and no path evidence contradicts it. You do not approve graph
writes. Do not use
outside knowledge, endpoint co-occurrence, node descriptions, or unlisted evidence.
The endpoint distance is at least three hops. Choose one relation_family and one
canonical_relation from the supplied ontologies. Copy the canonical relation spelling
exactly into relation; do not invent a narrower synonym. Generic values such as related,
associated, connected, 联系, or 相关 are forbidden. Preserve direction. Cite every
primary path evidence_id and every direct endpoint evidence_id. Output strict JSON
only, with no Markdown or trailing explanation:
{"proposals":[{"review_id":"...","source":"...","target":"...",
"relation":"...","relation_family":"...","direction":"source_to_target|target_to_source|undirected",
"description":"...","evidence_refs":["ev-..."]}]}
Return {"proposals":[]} when no formal relation is justified.
For every proposal, evidence_refs MUST exactly equal that item's
required_evidence_refs. Infer the latent endpoint relation, not the surface artifact:
relations such as evidence_linkage, shared_object_connection, route_alignment,
reserved_seat_ritual, or temporal_alignment are forbidden. If the clue cannot support
a precise endpoint relation such as influence, causation, kinship, romantic attachment,
collaboration, participation, protection, or opposition, return no proposal.
Interpret recurring clue semantics conservatively: paired ring halves can support a
spousal partnership and MUST use spouses; an intentionally reserved named seat can use
love or liking; matching birth tokens and childhood cloth MUST use parent_child;
matching half-bells MUST use siblings; a moved red pin beside a dated dossier can use
influence or causation, defaulting to influence unless the endpoint clue itself states
a resulting outcome; matching route and clock records can use collaboration or
participation; a repeated ink reaction to someone's name can use jealousy or hatred,
defaulting to jealousy unless the evidence contains explicit hostility; returned
invitations and torn notices MUST use enemies; layered lecture annotations MUST use
teacher_student; paired duty-damaged copper plates MUST use colleagues. Use the
supporting path to choose only between the explicitly allowed alternatives. These are
semantic constraints, not automatic answers; reject when the supplied quote does not
match.
When clue_semantic_contract is present, verify its template against the direct quote,
then choose only one of its allowed_relations and follow its direction_hint. Relations
listed in symmetric_relation_ontology must use direction=undirected.
"""

JUDGE_INSTRUCTION = """You are the independent Judge in ECLRR-v4.
Re-read the original EvidencePackage and the Generator Proposal. Treat
direct_endpoint_evidence as the hidden-relation clue and the path as supporting
context. The path may contain heterogeneous relation families. Reject relation
generalization, invented facts, missing direct or hop evidence, unsupported direction,
and chains that contradict the clue. Do not reject solely because the path does not
logically entail or repeat the endpoint relation. Scores are
decision scores, not probabilities. Output strict JSON only, with no Markdown or
trailing explanation:
{"decisions":[{"review_id":"...","decision":"accept|revise|reject",
"evidence_coverage":0,"semantic_composability":0,"relation_direction":0,
"uncertainty_calibration":0,"total":0,"reason_codes":[],
"revised_relation":null,"revised_relation_family":null,
"verified_evidence_refs":["ev-..."]}]}
Ranges: evidence_coverage 0-4, semantic_composability 0-3,
relation_direction 0-2, uncertainty_calibration 0-1, total 0-10.
Use revise only when the evidence supports a more precise relation than the Proposal.
When evidence_coverage is 4, verified_evidence_refs MUST exactly equal the supplied
required_evidence_refs. Surface-artifact relations such as evidence_linkage,
shared_object_connection, route_alignment, reserved_seat_ritual, or temporal_alignment
must be revised to the supported latent endpoint relation or rejected.
Apply the same clue semantics stated for the Generator, and use the supporting path to
resolve alternatives such as influence versus causation, collaboration versus
participation, love versus liking, and jealousy versus hatred.
The final or revised relation must obey those clue-to-relation constraints exactly;
never replace a constrained relation with a broader family label such as opposition,
romantic_attachment, affiliation, or collaboration when that label is not allowed for
the observed clue.
The EvidencePackage architecture is intentional and must be scored as follows:
- direct_endpoint_evidence is direct evidence of a latent endpoint relation when its
  quote contains both endpoints and matches one of the supplied clue semantics. Never
  emit missing_direct_evidence merely because the quote uses an artifact or behavior
  instead of explicitly naming the final relation.
- the primary 3-8 hop path proves evidence-grounded narrative continuity. Its hops need
  not repeat, entail, or share the family of the latent endpoint relation. For
  semantic_composability, score whether the path is continuous and non-contradictory,
  not whether transitive relation algebra derives the endpoint relation.
- an undirected relationship may receive full direction credit when the Proposal marks
  direction=undirected. Directional relations must follow the actor-to-affected order in
  the direct clue.
If all required quotes are present, the direct clue supports a canonical relation, and
the continuous path does not contradict it, do not reject for heterogeneous path
families or lack of a direct source-target hop. Prefer revise to the closest supplied
canonical relation over a bespoke synonym. A revised relation must exactly match the
canonical relation ontology.
Each evidence_package may include clue_semantic_contract. When present, it is a
deterministic ontology constraint derived from the quoted clue, not a proposed answer:
verify that the quote matches, then choose only an allowed relation and follow its
direction_hint. Relations listed in symmetric_relation_ontology must use undirected.
If evidence_coverage=4 and all numeric pass thresholds are met, decision=reject is
internally inconsistent unless a hard contradiction is identified and reflected by a
lower component score.
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def generator_prompt(packages: list[EvidencePackage], detail: int = 0) -> str:
    include_alternates = detail == 0
    include_context = detail <= 1
    payload = {
        "relation_family_ontology": list(RELATION_FAMILIES),
        "canonical_relation_ontology": list(CANONICAL_RELATIONS),
        "symmetric_relation_ontology": sorted(SYMMETRIC_RELATIONS),
        "items": [
            package.to_prompt_dict(
                include_alternates=include_alternates,
                include_context=include_context,
            )
            for package in packages
        ],
    }
    for package, item in zip(packages, payload["items"]):
        item["clue_semantic_contract"] = clue_contract(
            [evidence.quote for evidence in package.direct_evidence],
            package.review_item.source,
            package.review_item.target,
        )
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
        "canonical_relation_ontology": list(CANONICAL_RELATIONS),
        "symmetric_relation_ontology": sorted(SYMMETRIC_RELATIONS),
        "evidence_package": package.to_prompt_dict(
            include_alternates=detail == 0,
            include_context=detail <= 1,
        ),
        "proposal": asdict(proposal),
    }
    payload["evidence_package"]["clue_semantic_contract"] = clue_contract(
        [evidence.quote for evidence in package.direct_evidence],
        package.review_item.source,
        package.review_item.target,
    )
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
