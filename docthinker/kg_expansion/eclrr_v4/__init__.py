"""ECLRR-v4: evidence-carrying long-chain relation refinement."""

from .beam_search import discover_review_items
from .evidence import build_evidence_package, build_evidence_packages
from .gate import deterministic_gate
from .graph_view import FactGraphView, canonical_relation_key, classify_edge
from .models import (
    ECLRRConfig,
    ECLRRRunResult,
    EvidencePackage,
    EvidenceRef,
    GateResult,
    JudgeDecision,
    Proposal,
    ReviewItem,
)
from .service import run_eclrr_v4

__all__ = [
    "ECLRRConfig",
    "ECLRRRunResult",
    "EvidencePackage",
    "EvidenceRef",
    "FactGraphView",
    "GateResult",
    "JudgeDecision",
    "Proposal",
    "ReviewItem",
    "build_evidence_package",
    "build_evidence_packages",
    "canonical_relation_key",
    "classify_edge",
    "deterministic_gate",
    "discover_review_items",
    "run_eclrr_v4",
]
