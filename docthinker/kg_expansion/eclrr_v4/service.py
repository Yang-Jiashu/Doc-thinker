"""Top-level ECLRR-v4 orchestration."""

from __future__ import annotations

from typing import Any, Callable

from .audit import AuditTrail
from .beam_search import discover_review_items
from .evidence import build_evidence_packages
from .gate import deterministic_gate
from .graph_view import FactGraphView
from .llm_pipeline import run_generator_and_judge
from .models import ECLRRConfig, ECLRRRunResult
from .writeback import commit_promotions


async def run_eclrr_v4(
    *,
    graph: Any,
    text_chunks: Any,
    generator_func: Callable,
    judge_func: Callable,
    relationships_vdb: Any = None,
    config: ECLRRConfig | None = None,
    write_graph: bool = True,
) -> ECLRRRunResult:
    cfg = config or ECLRRConfig()
    audit = AuditTrail(cfg.artifact_dir, cfg)
    nodes = await graph.get_all_nodes()
    edges = await graph.get_all_edges()
    view = FactGraphView.build(nodes, edges)
    review_items = discover_review_items(view, cfg)
    audit.record(
        "search",
        {
            "nodes": len(view.nodes),
            "fact_edges": len(view.fact_edges),
            "fuzzy_pairs": len(view.fuzzy_by_pair),
            "review_items": review_items,
        },
    )
    packages, evidence_rejected = await build_evidence_packages(
        review_items, view, text_chunks, cfg
    )
    audit.record(
        "evidence_packages",
        {"accepted": packages, "rejected": evidence_rejected},
    )
    proposals, decisions, llm_failures = await run_generator_and_judge(
        packages,
        generator_func,
        judge_func,
        cfg,
        audit=audit,
    )
    audit.record(
        "llm_results",
        {
            "proposals": proposals,
            "decisions": decisions,
            "failures": llm_failures,
        },
    )
    package_by_id = {item.review_item.review_id: item for item in packages}
    decision_by_id = {item.review_id: item for item in decisions}
    gate_results = []
    for proposal in proposals:
        decision = decision_by_id.get(proposal.review_id)
        if decision is None:
            continue
        gate_results.append(
            await deterministic_gate(
                package_by_id[proposal.review_id],
                proposal,
                decision,
                view,
                text_chunks,
                cfg,
            )
        )
    audit.record("gate", gate_results)
    committed = []
    write_failures: dict[str, str] = {}
    if write_graph:
        committed, write_failures = await commit_promotions(
            gate_results, graph, relationships_vdb
        )
    audit.record(
        "writeback",
        {"committed": committed, "failures": write_failures},
    )
    metrics = {
        "nodes": len(nodes),
        "edges": len(edges),
        "fact_edges": len(view.fact_edges),
        "review_items": len(review_items),
        "evidence_packages": len(packages),
        "evidence_rejected": evidence_rejected,
        "proposals": len(proposals),
        "decisions": len(decisions),
        "llm_failures": llm_failures,
        "create": sum(result.action == "create" for result in gate_results),
        "refine": sum(result.action == "refine" for result in gate_results),
        "no_op": sum(result.action == "no-op" for result in gate_results),
        "committed": len(committed),
        "write_failures": write_failures,
    }
    audit.finalize(metrics)
    return ECLRRRunResult(
        review_items=review_items,
        proposals=proposals,
        decisions=decisions,
        gate_results=gate_results,
        committed=committed,
        metrics=metrics,
    )
