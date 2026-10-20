#!/usr/bin/env python3
"""PRIVATE fixed-trace two-need nomination experiment, never runtime policy.

Saved facets and semantic scores are replayed unchanged. The sole in-memory
algorithm change is _focused_nominations profiles[:min(2,len(needs)-1)] to
profiles[:2]. Queries are not generated, answer providers are never called,
and watched source IDs affect diagnostic output only, never the selector/model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import uuid

from diagnose_rerank_length_bucketing_offline import (
    NETWORK_ATTEMPTS, ROOT, LocalEvidenceReranker, candidate_signature, code_receipt,
    digest, identity, prepare_fixed, read_chunks, sha, write_private,
)
from docthinker.server import research_faceted_assist as assist
from docthinker.server.research_evidence_selection import prepare_evidence_candidates, select_evidence


def receipt_rows(pairs, ranked):
    return [{**{key: row.get(key) for key in (
                "id", "source_id", "document_id", "page", "ordinal", "excerpt_start", "excerpt_end",
                "query", "content", "rerank_branch", "rerank_facet_id", "rerank_view", "rerank_view_of")},
             "input_signature": candidate_signature(row), "logit": ranked["scores"].get(row["id"]),
             "input_receipt": ranked["input_receipts"].get(row["id"])} for row in pairs]


def rebuild_pool(fixed):
    events = fixed["trace"]["events"]
    semantic = {row["facet_id"]: {"scores": row["scores"], "spans": row["spans"], "trace": {"status": row["status"]}}
                for row in events if row["event"] == "facet_semantic" and row["status"] == "ready"}
    question = next(row["text"] for row in fixed["plan"]["facets"] if row["kind"] == "original")
    chunks = read_chunks(fixed["database"])
    pool = prepare_evidence_candidates(question, chunks, fixed["baseline"], fixed["plan"], semantic)
    return question, chunks, pool["candidates"]


def score(scorer, fixed, pairs, arm, run_id):
    assert len(pairs) <= 48
    wall, cpu = time.perf_counter(), time.process_time()
    ranked = scorer.rank_pairs(pairs, session_id=f"private-two-focus::{run_id}::{fixed['trace']['question_id']}::{arm}")
    wall_ms, cpu_ms = (time.perf_counter() - wall) * 1000, (time.process_time() - cpu) * 1000
    assert ranked["trace"]["status"] == "ready", ranked["trace"]
    assert ranked["trace"]["cache_hits"] == 0
    events = []
    selected = select_evidence(deepcopy(pairs), ranked["scores"], fixed["plan"], fixed["baseline"],
                               diagnostic_callback=events.append, input_receipts=ranked["input_receipts"])
    assert len(selected["sources"]) <= 8
    assert sum(len(row["content"].encode()) for row in selected["sources"]) <= 12000
    allocation = {"total_pairs": len(pairs),
                  "full_pairs": sum(row.get("rerank_branch") not in {"focused", "quoted_audit"} for row in pairs),
                  "focused_pairs": sum(row.get("rerank_branch") == "focused" for row in pairs),
                  "quoted_audit_pairs": sum(row.get("rerank_branch") == "quoted_audit" for row in pairs),
                  "focused_counts_by_need": dict(Counter(row.get("rerank_view_of", row.get("rerank_facet_id"))
                                              for row in pairs if row.get("rerank_branch") == "focused"))}
    return {"arm": arm, "pair_allocation": allocation, "rank_wall_ms": wall_ms, "rank_cpu_ms": cpu_ms,
            "reranker_trace": ranked["trace"], "pair_receipts": receipt_rows(pairs, ranked),
            "selection_trace": selected["trace"], "selection_events": events,
            "final_sources": selected["sources"], "final_source_digest": digest(identity(selected["sources"])),
            "final_source_set_equals_saved_trace": {digest(row) for row in identity(selected["sources"])} ==
                {digest(row) for row in identity(fixed["trace"]["final_sources"])}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--watch-question")
    parser.add_argument("--watch-source", action="append", default=[])
    parser.add_argument("--private-diagnostics", action="store_true", required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use an empty/new output directory")
    os.umask(0o077)
    hashes = code_receipt()
    fixed_inputs = [prepare_fixed(path.resolve()) for path in args.trace]
    original = assist._focused_nominations
    original_text = inspect.getsource(original)
    old = "profiles = profiles[:min(2, len(needs) - 1)]"
    new = "profiles = profiles[:2]"
    assert original_text.count(old) == 1
    patched_text = original_text.replace(old, new)
    namespace = {}
    exec(compile(patched_text, "<private-two-focus-memory-patch>", "exec"), assist.__dict__, namespace)
    patched = namespace["_focused_nominations"]
    patch_receipt = {"function": "research_faceted_assist._focused_nominations", "old": old, "new": new,
                     "original_function_text": original_text, "original_function_sha256": sha(original_text),
                     "patched_function_text": patched_text, "patched_function_sha256": sha(patched_text),
                     "disk_runtime_modified": False, "only_function_binding_changed_in_memory": True}
    write_private(output / "memory-patch.json", patch_receipt)
    scorer = LocalEvidenceReranker()
    assets = scorer.prepare_assets()
    assert assets["trace"]["status"] == "ready"
    run_id, summaries = uuid.uuid4().hex, []
    try:
        for index, fixed in enumerate(fixed_inputs):
            qid = fixed["trace"]["question_id"]
            question, chunks, pool = rebuild_pool(fixed)
            assist._focused_nominations = patched
            patched_pairs = assist._rerank_pairs(question, deepcopy(pool), fixed["plan"]["facets"])
            assist._focused_nominations = original
            baseline_pairs = fixed["pairs"]
            variants = {"baseline": baseline_pairs, "two_focus": patched_pairs}
            assert len(baseline_pairs) == len(patched_pairs) <= 48
            assert [candidate_signature(row) for row in baseline_pairs if row.get("rerank_branch") != "focused"] == [
                candidate_signature(row) for row in patched_pairs if row.get("rerank_branch") != "focused"]
            # The binding is restored before any inference: patched nominations
            # are already fixed input data; model and selector stay unchanged.
            order = ["baseline", "two_focus"] if index % 2 == 0 else ["two_focus", "baseline"]
            results = {}
            for arm in order:
                measured = score(scorer, fixed, variants[arm], arm, run_id)
                write_private(output / f"{qid}.{arm}.json", measured)
                results[arm] = measured
                print(json.dumps({"question_id": qid, "arm": arm, "allocation": measured["pair_allocation"],
                                  "wall_ms": round(measured["rank_wall_ms"], 3)}), flush=True)
            left = {digest(row["input_signature"]): row for row in results["baseline"]["pair_receipts"]}
            right = {digest(row["input_signature"]): row for row in results["two_focus"]["pair_receipts"]}
            shared = left.keys() & right.keys()
            assert all(left[key]["input_receipt"] == right[key]["input_receipt"] for key in shared)
            before, after = identity(results["baseline"]["final_sources"]), identity(results["two_focus"]["final_sources"])
            info = {"question_id": qid, "input_trace_sha256": fixed["trace_sha256"], "execution_order": order,
                    "baseline_allocation": results["baseline"]["pair_allocation"],
                    "two_focus_allocation": results["two_focus"]["pair_allocation"],
                    "same_total_pair_budget": True, "same_nonfocus_pairs_order_bytes_queries": True,
                    "common_pairs": len(shared), "common_pair_receipts_identical": True,
                    "max_common_pair_logit_difference": max((abs(left[key]["logit"] - right[key]["logit"]) for key in shared), default=0),
                    "added_pairs": [right[key]["input_signature"] for key in sorted(right.keys() - left.keys())],
                    "removed_pairs": [left[key]["input_signature"] for key in sorted(left.keys() - right.keys())],
                    "final_source_order_and_spans_identical": before == after,
                    "final_source_set_identical": {digest(row) for row in before} == {digest(row) for row in after},
                    "baseline_final_equals_saved_trace": before == identity(fixed["trace"]["final_sources"])}
            if qid == args.watch_question:
                watched = set(args.watch_source)
                info["watch_output_only_no_input_influence"] = True
                info["watched_sources"] = [{"source_id": row["id"], "page": row.get("page"),
                                            "ordinal": row.get("ordinal"), "content": row["content"]}
                                           for row in chunks if row["id"] in watched]
                info["watched_pool_windows"] = [{key: row.get(key) for key in (
                    "id", "source_id", "page", "ordinal", "excerpt_start", "excerpt_end", "content", "facet_scores")}
                    for row in pool if row["source_id"] in watched]
                for arm in ("baseline", "two_focus"):
                    info[arm + "_watched_pairs"] = [row for row in results[arm]["pair_receipts"] if row["source_id"] in watched]
                    info[arm + "_watched_final_sources"] = [row for row in results[arm]["final_sources"] if row["id"] in watched]
            assert hashlib.sha256(fixed["database"].read_bytes()).hexdigest() == fixed["corpus_sha256"]
            write_private(output / f"{qid}.comparison.json", info)
            summaries.append({key: value for key, value in info.items() if "watched" not in key})
    finally:
        assist._focused_nominations = original
    after_hashes = code_receipt()
    assert not NETWORK_ATTEMPTS
    write_private(output / "summary.json", {
        "schema": "private-two-focus-nomination-v1", "private_diagnostics": True, "not_product_trace": True,
        "provider_calls": 0, "answer_model_calls": 0, "network_attempts_blocked": len(NETWORK_ATTEMPTS),
        "source_hashes_as_run": hashes, "source_hashes_after_run": after_hashes,
        "source_changes_during_run": [name for name, value in hashes.items() if after_hashes.get(name) != value],
        "frozen_runtime_claim": False, "memory_patch": patch_receipt,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "assets": assets,
        "input_policy": "Saved original facet and semantic receipts only; no translation or embedding inference. Watch IDs only filter diagnostics after scoring. No gold is read or used.",
        "numeric_caveat": "Changing focused pair membership/group sizes may change int8 batch logits even for shared pairs; common receipt equality is checked, numerical equality is measured rather than assumed.",
        "questions": summaries})


if __name__ == "__main__":
    main()
