#!/usr/bin/env python3
"""PRIVATE selector-only replay of saved baseline/two-focus model scores.

First reproduce each saved 12,000-byte selection exactly, then change only
max_source_bytes to min(12,000, UTF-8 bytes in that trace's baseline sources).
No model is initialized or inferred; no new query, gold or answer is read.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path

from diagnose_research_two_focus_offline import (
    NETWORK_ATTEMPTS, ROOT, assist, candidate_signature, code_receipt, digest,
    identity, prepare_fixed, rebuild_pool, receipt_rows, select_evidence, sha,
    write_private,
)


ARMS = ("baseline", "two_focus")
ORIGINAL_BUDGET = 12000


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def source_bytes(sources):
    assert all(isinstance(row["content"], str) for row in sources)
    return sum(len(row["content"].encode("utf-8")) for row in sources)


def source_comparison(before, after):
    left, right = identity(before), identity(after)
    left_set, right_set = {digest(row) for row in left}, {digest(row) for row in right}
    return {"source_order_and_spans_identical": left == right,
            "source_set_and_spans_identical": left_set == right_set,
            "before_source_count": len(before), "after_source_count": len(after),
            "before_source_bytes": source_bytes(before), "after_source_bytes": source_bytes(after),
            "added_source_ids": sorted({row["id"] for row in after} - {row["id"] for row in before}),
            "removed_source_ids": sorted({row["id"] for row in before} - {row["id"] for row in after})}


def saved_ranked(pairs, artifact):
    rows = artifact["pair_receipts"]
    assert 0 < len(rows) == len(pairs) <= 48
    assert len({row["id"] for row in rows}) == len(rows)
    assert [candidate_signature(row) for row in pairs] == [row["input_signature"] for row in rows]
    ranked = {"scores": {}, "input_receipts": {}}
    for pair, row in zip(pairs, rows):
        value, receipt = row["logit"], row["input_receipt"]
        assert type(value) in (int, float) and math.isfinite(value)
        assert receipt["submitted_query_sha256"] == sha(pair["query"])
        assert receipt["submitted_passage_sha256"] == sha(pair["content"])
        end = receipt["actual_query_character_end"]
        assert type(end) is int and 0 <= end <= len(pair["query"])
        assert receipt["actual_query_sha256"] == sha(pair["query"][:end])
        assert receipt["query_complete"] == (end == len(pair["query"]))
        assert receipt["actual_passage_sha256"] == (
            sha(pair["content"]) if receipt["passage_complete"] else None)
        ranked["scores"][row["id"]] = value
        ranked["input_receipts"][row["id"]] = receipt
    # Checks the saved literal inputs, geometry and branch/view metadata too.
    assert receipt_rows(pairs, ranked) == rows, "Rebuilt scored inputs differ from saved artifact"
    assert artifact["reranker_trace"]["status"] == "ready"
    return ranked


def replay(fixed, pairs, ranked, budget):
    events = []
    result = select_evidence(deepcopy(pairs), ranked["scores"], deepcopy(fixed["plan"]),
                             deepcopy(fixed["baseline"]), max_sources=8, max_source_bytes=budget,
                             input_receipts=deepcopy(ranked["input_receipts"]),
                             diagnostic_callback=events.append)
    assert len(result["sources"]) <= 8
    assert source_bytes(result["sources"]) <= budget
    assert result["trace"]["max_source_bytes"] == budget
    assert result["trace"]["source_bytes"] == source_bytes(result["sources"])
    return {"final_sources": result["sources"], "selection_trace": result["trace"],
            "selection_events": events, "gaps": result["gaps"],
            "final_source_digest": digest(identity(result["sources"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-diagnostics", action="store_true", required=True)
    args = parser.parse_args()
    score_dir, output = args.score_dir.resolve(), args.output_dir.resolve()
    if output == score_dir or score_dir in output.parents:
        parser.error("Output must be separate from the saved score directory")
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new/empty output directory; prior results are never replaced")
    os.umask(0o077)
    score_summary, score_summary_sha = read_json(score_dir / "summary.json")
    assert score_summary["schema"] == "private-two-focus-nomination-v1"
    assert not score_summary["source_changes_during_run"]
    hashes = code_receipt()
    assert hashes == score_summary["source_hashes_as_run"] == score_summary["source_hashes_after_run"]
    trace_rows = {row["question_id"]: row for row in score_summary["questions"]}
    fixed_inputs = [prepare_fixed(path.resolve()) for path in args.trace]
    assert len(fixed_inputs) == len(trace_rows) == 6
    assert {row["trace"]["question_id"] for row in fixed_inputs} == set(trace_rows)
    for fixed in fixed_inputs:
        assert fixed["trace_sha256"] == trace_rows[fixed["trace"]["question_id"]]["input_trace_sha256"]

    patch, patch_file_sha = read_json(score_dir / "memory-patch.json")
    assert patch == score_summary["memory_patch"]
    assert patch["function"] == "research_faceted_assist._focused_nominations"
    assert patch["disk_runtime_modified"] is False
    assert patch["only_function_binding_changed_in_memory"] is True
    original = assist._focused_nominations
    original_text = inspect.getsource(original)
    old = "profiles = profiles[:min(2, len(needs) - 1)]"
    new = "profiles = profiles[:2]"
    assert patch["old"] == old and patch["new"] == new and original_text.count(old) == 1
    patched_text = original_text.replace(old, new)
    assert original_text == patch["original_function_text"]
    assert patched_text == patch["patched_function_text"]
    assert sha(original_text) == patch["original_function_sha256"]
    assert sha(patched_text) == patch["patched_function_sha256"]
    namespace = {}
    # Execute only the checked local source plus the one whitelisted replacement.
    exec(compile(patched_text, "<private-source-budget-two-focus-rebuild>", "exec"), assist.__dict__, namespace)
    patched = namespace["_focused_nominations"]
    questions, artifact_hashes = [], {}
    for fixed in fixed_inputs:
        qid = fixed["trace"]["question_id"]
        question, _, pool = rebuild_pool(fixed)
        baseline_pairs = assist._rerank_pairs(question, deepcopy(pool), deepcopy(fixed["plan"]["facets"]))
        assert baseline_pairs == fixed["pairs"]
        try:
            assist._focused_nominations = patched
            two_focus_pairs = assist._rerank_pairs(question, deepcopy(pool), deepcopy(fixed["plan"]["facets"]))
        finally:
            assist._focused_nominations = original
        variants = {"baseline": baseline_pairs, "two_focus": two_focus_pairs}
        baseline_bytes = source_bytes(fixed["baseline"])
        budget = min(ORIGINAL_BUDGET, baseline_bytes)
        results, arm_summaries = {}, []
        for arm in ARMS:
            score_path = score_dir / f"{qid}.{arm}.json"
            artifact, artifact_sha = read_json(score_path)
            artifact_hashes[str(score_path)] = artifact_sha
            assert artifact["arm"] == arm
            pairs = variants[arm]
            ranked = saved_ranked(pairs, artifact)
            original_selection = replay(fixed, pairs, ranked, ORIGINAL_BUDGET)
            assert original_selection["final_sources"] == artifact["final_sources"], "12 KB source reproduction failed"
            assert original_selection["selection_trace"] == artifact["selection_trace"], "12 KB trace reproduction failed"
            assert original_selection["final_source_digest"] == artifact["final_source_digest"]
            capped = replay(fixed, pairs, ranked, budget)
            assert source_bytes(capped["final_sources"]) <= baseline_bytes
            comparison = source_comparison(original_selection["final_sources"], capped["final_sources"])
            payload = {"schema": "private-source-byte-budget-arm-v1", "private_diagnostics": True,
                       "not_product_trace": True, "question_id": qid, "arm": arm,
                       "provider_calls": 0, "added_model_calls": 0, "answer_model_calls": 0,
                       "score_artifact_path": str(score_path), "score_artifact_sha256": artifact_sha,
                       "score_summary_sha256": score_summary_sha, "input_trace_sha256": fixed["trace_sha256"],
                       "input_pair_digest": digest([candidate_signature(row) for row in pairs]),
                       "saved_pair_receipts_digest": digest(artifact["pair_receipts"]),
                       "pair_count": len(pairs), "pair_allocation": artifact["pair_allocation"],
                       "inputs_and_receipts_identical_to_saved_scores": True,
                       "original_12000_selection_reproduced": True,
                       "original_12000_events_identical": original_selection["selection_events"] == artifact["selection_events"],
                       "source_hashes_as_run": hashes, "memory_patch_file_sha256": patch_file_sha,
                       "patched_function_sha256": patch["patched_function_sha256"],
                       "baseline_source_bytes": baseline_bytes, "max_source_bytes": budget,
                       "budget_policy": "min(12000, sum(UTF8 baseline_sources content bytes))",
                       "original_12000": original_selection, **capped,
                       "comparison_to_original_12000": comparison}
            destination = output / f"{qid}.{arm}.json"
            write_private(destination, payload)
            results[arm] = payload
            arm_summaries.append({"arm": arm, "artifact_path": str(destination),
                                  "score_artifact_sha256": artifact_sha, "pair_count": len(pairs),
                                  "original_12000_selection_reproduced": True,
                                  "source_bytes_within_cap_and_baseline": True, **comparison})
        info = {"question_id": qid, "baseline_source_bytes": baseline_bytes, "max_source_bytes": budget,
                "input_trace_sha256": fixed["trace_sha256"], "arms": arm_summaries,
                "capped_baseline_vs_two_focus": source_comparison(results["baseline"]["final_sources"],
                                                                 results["two_focus"]["final_sources"])}
        write_private(output / f"{qid}.comparison.json", info)
        questions.append(info)
        assert hashlib.sha256(fixed["database"].read_bytes()).hexdigest() == fixed["corpus_sha256"]
        print(json.dumps({"question_id": qid, "max_source_bytes": budget,
                          "final_source_bytes": {arm: source_bytes(results[arm]["final_sources"]) for arm in ARMS}}), flush=True)
    after_hashes = code_receipt()
    assert hashes == after_hashes
    assert assist._focused_nominations is original
    assert not NETWORK_ATTEMPTS
    assert read_json(score_dir / "summary.json")[1] == score_summary_sha
    assert read_json(score_dir / "memory-patch.json")[1] == patch_file_sha
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == value for path, value in artifact_hashes.items())
    write_private(output / "summary.json", {
        "schema": "private-source-byte-budget-v1", "private_diagnostics": True, "not_product_trace": True,
        "provider_calls": 0, "added_model_calls": 0, "answer_model_calls": 0,
        "translation_calls": 0, "embedding_calls": 0, "reranker_calls": 0,
        "network_attempts_blocked": len(NETWORK_ATTEMPTS), "local_selector_only": True,
        "original_max_source_bytes": ORIGINAL_BUDGET,
        "budget_policy": "min(12000, sum(UTF8 baseline_sources content bytes))", "max_sources": 8,
        "score_summary_path": str(score_dir / "summary.json"), "score_summary_sha256": score_summary_sha,
        "score_artifact_sha256": artifact_hashes, "memory_patch_file_sha256": patch_file_sha,
        "memory_patch": patch, "source_hashes_as_run": hashes, "source_hashes_after_run": after_hashes,
        "source_changes_during_run": [], "frozen_runtime_claim": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_script_sha256": {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest()
                                 for name in ("diagnose_research_two_focus_offline.py",
                                              "diagnose_rerank_length_bucketing_offline.py",
                                              "diagnose_research_selection_offline.py")},
        "verified_arm_count": len(questions) * len(ARMS),
        "interpretation_limit": "Existing exposed traces and saved logits only. No new model evidence, quality judgment, matched excerpt-length claim, or end-to-end latency measurement.",
        "questions": questions})


if __name__ == "__main__":
    main()
