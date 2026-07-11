"""Typed records used by the ECLRR-v4 discovery pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EdgeClass = Literal["fact", "fuzzy", "promoted", "legacy"]
TraversalDirection = Literal["forward", "inverse", "undirected"]
Decision = Literal["accept", "revise", "reject"]
WriteAction = Literal["create", "refine", "no-op"]


@dataclass(frozen=True)
class ECLRRConfig:
    min_hops: int = 3
    max_hops: int = 8
    beam_width: int = 12
    max_neighbours: int = 24
    max_review_items: int = 128
    max_paths_per_item: int = 3
    max_generator_chains: int = 2
    max_generator_items: int = 8
    max_prompt_chars: int = 14_000
    max_output_tokens: int = 4096
    alternate_evidence_per_hop: int = 2
    context_chars: int = 320
    artifact_dir: str | None = None
    algorithm_version: str = "eclrr_v4"


@dataclass(frozen=True)
class FactEdge:
    edge_id: str
    source: str
    target: str
    relation: str
    relation_family: str
    description: str
    source_ids: tuple[str, ...]
    edge_class: EdgeClass
    declared_direction: str
    specificity: float
    grounding: float
    raw: dict[str, Any] = field(compare=False, hash=False, repr=False)


@dataclass(frozen=True)
class PathStep:
    edge_id: str
    source: str
    target: str
    traversal_source: str
    traversal_target: str
    traversal_direction: TraversalDirection
    relation: str
    relation_family: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class SearchPath:
    nodes: tuple[str, ...]
    steps: tuple[PathStep, ...]
    score: float

    @property
    def hops(self) -> int:
        return len(self.steps)

    @property
    def signature(self) -> str:
        return "->".join(self.nodes)


@dataclass
class ReviewItem:
    review_id: str
    source: str
    target: str
    primary_path: SearchPath
    supporting_paths: list[SearchPath] = field(default_factory=list)
    fuzzy_edge: dict[str, Any] | None = None


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    hop_index: int
    edge_id: str
    source: str
    target: str
    chunk_id: str
    quote: str
    start: int
    end: int
    context: str = ""


@dataclass
class EvidencePackage:
    review_item: ReviewItem
    primary_evidence: list[EvidenceRef]
    alternate_evidence: list[EvidenceRef]
    node_types: dict[str, str]

    def to_prompt_dict(
        self,
        *,
        include_alternates: bool = True,
        include_context: bool = True,
    ) -> dict[str, Any]:
        def evidence_payload(item: EvidenceRef) -> dict[str, Any]:
            payload = asdict(item)
            if not include_context:
                payload.pop("context", None)
            return payload

        return {
            "review_id": self.review_item.review_id,
            "source": self.review_item.source,
            "target": self.review_item.target,
            "path": path_to_dict(self.review_item.primary_path),
            "supporting_paths": [],
            "node_types": self.node_types,
            "primary_evidence": [
                evidence_payload(item) for item in self.primary_evidence
            ],
            "alternate_evidence": (
                [evidence_payload(item) for item in self.alternate_evidence]
                if include_alternates
                else []
            ),
        }


@dataclass(frozen=True)
class Proposal:
    review_id: str
    source: str
    target: str
    relation: str
    relation_family: str
    direction: str
    description: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class JudgeDecision:
    review_id: str
    decision: Decision
    evidence_coverage: int
    semantic_composability: int
    relation_direction: int
    uncertainty_calibration: int
    total: int
    reason_codes: tuple[str, ...]
    revised_relation: str | None
    revised_relation_family: str | None
    verified_evidence_refs: tuple[str, ...]


@dataclass
class GateResult:
    action: WriteAction
    reason: str
    review_id: str
    source: str
    target: str
    relation: str = ""
    relation_family: str = ""
    direction: str = ""
    description: str = ""
    canonical_key: str = ""
    relation_id: str = ""
    evidence_chain: list[dict[str, Any]] = field(default_factory=list)
    evidence_chunk_ids: list[str] = field(default_factory=list)
    path_used: list[str] = field(default_factory=list)
    supporting_paths: list[dict[str, Any]] = field(default_factory=list)
    proposal: dict[str, Any] = field(default_factory=dict)
    judge_decision: dict[str, Any] = field(default_factory=dict)


@dataclass
class ECLRRRunResult:
    review_items: list[ReviewItem] = field(default_factory=list)
    proposals: list[Proposal] = field(default_factory=list)
    decisions: list[JudgeDecision] = field(default_factory=list)
    gate_results: list[GateResult] = field(default_factory=list)
    committed: list[GateResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def path_to_dict(path: SearchPath) -> dict[str, Any]:
    return {
        "nodes": list(path.nodes),
        "hops": path.hops,
        "score": round(path.score, 6),
        "steps": [asdict(step) for step in path.steps],
    }
