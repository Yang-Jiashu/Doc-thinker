#!/usr/bin/env python3
"""PRIVATE ORT intra-op thread experiment; no runtime edits or answer calls.

Only intra_op_num_threads changes (1/2/4); original pair order, exact query
groups, batch8, tokenizer and truncation remain fixed. Each trial/condition
gets a fresh ORT session and each score call a fresh namespace. Model setup
is separately metered; neither actual query nor dummy inference is prewarmed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import uuid

from diagnose_rerank_length_bucketing_offline import (
    NETWORK_ATTEMPTS, ROOT, LocalEvidenceReranker, code_receipt, compare,
    prepare_fixed, score_once, write_private,
)


OPTION_NAMES = (
    "intra_op_num_threads", "inter_op_num_threads", "execution_mode", "execution_order",
    "graph_optimization_level", "enable_cpu_mem_arena", "enable_mem_pattern", "enable_mem_reuse",
    "use_deterministic_compute", "enable_profiling", "optimized_model_filepath",
    "log_severity_level", "log_verbosity_level",
)


def option_receipt(options):
    result = {}
    for name in OPTION_NAMES:
        if hasattr(options, name):
            value = getattr(options, name)
            result[name] = value if type(value) in (bool, int, str, float) else str(value)
    return result


def cpu_receipt():
    values = {"os_cpu_count": os.cpu_count(), "load_average": list(os.getloadavg())}
    for key in ("machdep.cpu.brand_string", "hw.physicalcpu", "hw.logicalcpu", "hw.memsize",
                "hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu"):
        result = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=False)
        values[key] = result.stdout.strip() if result.returncode == 0 else None
    return values


def batch_signature(result):
    return [{key: row[key] for key in ("query_group", "batch_size", "padded_length", "nonpadding_lengths")}
            for row in result["actual_forward_batches"]]


def group_signature(result):
    return [{key: row[key] for key in ("query_sha256", "pair_count", "execution_order",
                                      "pair_tokens_before_truncation_original_order")}
            for row in result["groups"]]


def comparison(reference, treatment, pairs):
    result = compare(reference, treatment, pairs)
    result["wall_ms_delta_vs_threads2"] = result.pop("wall_ms_delta_sorted_minus_current")
    result["cpu_ms_delta_vs_threads2"] = result.pop("cpu_ms_delta_sorted_minus_current")
    result.update(scores_bitwise_equal=reference["scores"] == treatment["scores"],
                  actual_forward_batch_layout_identical=batch_signature(reference) == batch_signature(treatment),
                  group_and_pair_execution_order_identical=group_signature(reference) == group_signature(treatment))
    assert result["actual_forward_batch_layout_identical"]
    assert result["group_and_pair_execution_order_identical"]
    assert result["pair_receipts_identical"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-diagnostics", action="store_true", required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new/empty output directory; prior results are never replaced")
    os.umask(0o077)
    hashes = code_receipt()
    freeze = json.loads((ROOT / "dist/real-paper-ab-20261011/evidence-trace-v8-freeze/freeze.json").read_text())
    fixed_inputs = [prepare_fixed(path.resolve()) for path in args.trace]
    assert len({row["trace"]["question_id"] for row in fixed_inputs}) == len(fixed_inputs)
    import onnxruntime as ort
    scorer = LocalEvidenceReranker()
    started = time.perf_counter()
    asset_receipt = scorer.prepare_assets()
    asset_wall_ms = (time.perf_counter() - started) * 1000
    assert asset_receipt["trace"]["status"] == "ready", asset_receipt
    backend = scorer._model.model
    original_session = backend.model
    original_options = option_receipt(original_session.get_session_options())
    assert original_options["intra_op_num_threads"] == 2
    assert original_options["inter_op_num_threads"] == 2
    providers = original_session.get_providers()
    provider_options = original_session.get_provider_options()
    assert providers == ["CPUExecutionProvider"]
    from docthinker.server.research_reranker import MODEL_FILE
    model_path = scorer.model_path / MODEL_FILE
    run_id = uuid.uuid4().hex
    results, setup, sequence = {}, [], []
    before_cpu = cpu_receipt()
    for trial in (1, 2):
        thread_order = [1, 2, 4] if trial == 1 else [4, 2, 1]
        questions = fixed_inputs if trial == 1 else list(reversed(fixed_inputs))
        for threads in thread_order:
            # Fastembed's threads= parameter changes inter-op too. Clone the
            # original actual session options and override only intra-op here.
            options = original_session.get_session_options()
            options.intra_op_num_threads = threads
            receipt = option_receipt(options)
            assert {key: value for key, value in receipt.items() if key != "intra_op_num_threads"} == {
                key: value for key, value in original_options.items() if key != "intra_op_num_threads"}
            wall, cpu = time.perf_counter(), time.process_time()
            session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers,
                                           provider_options=[provider_options[name] for name in providers])
            setup.append({"trial": trial, "intra_threads": threads,
                          "session_setup_wall_ms": (time.perf_counter() - wall) * 1000,
                          "session_setup_cpu_ms": (time.process_time() - cpu) * 1000,
                          "actual_options": option_receipt(session.get_session_options()),
                          "providers": session.get_providers()})
            backend.model = session
            try:
                for position, fixed in enumerate(questions):
                    qid = fixed["trace"]["question_id"]
                    measured = score_once(scorer, fixed, "current", trial,
                                          run_id + "::intra" + str(threads))
                    assert all(group["execution_order"] == list(range(group["pair_count"]))
                               for group in measured["groups"])
                    measured.update(intra_threads=threads, inter_threads=2,
                                    query_position_in_fresh_session=position + 1,
                                    load_average=list(os.getloadavg()))
                    results[(qid, trial, threads)] = measured
                    write_private(output / f"{qid}.trial{trial}.threads{threads}.json", measured)
                    sequence.append({"question_id": qid, "trial": trial, "intra_threads": threads})
                    print(json.dumps({**sequence[-1], "wall_ms": round(measured["rank_pairs_wall_ms"], 3),
                                      "cpu_ms": round(measured["rank_pairs_cpu_ms"], 3)}), flush=True)
            finally:
                backend.model = original_session
                del session
                gc.collect()
    summaries = []
    for fixed in fixed_inputs:
        qid = fixed["trace"]["question_id"]
        conditions = []
        for threads in (1, 2, 4):
            samples = [results[(qid, trial, threads)] for trial in (1, 2)]
            conditions.append({"intra_threads": threads,
                               "median_wall_ms": statistics.median(row["rank_pairs_wall_ms"] for row in samples),
                               "median_cpu_ms": statistics.median(row["rank_pairs_cpu_ms"] for row in samples),
                               "comparisons_to_same_trial_threads2": [
                                   {"trial": trial, **comparison(results[(qid, trial, 2)], results[(qid, trial, threads)], fixed["pairs"])}
                                   for trial in (1, 2)],
                               "repeat_scores_bitwise_equal": samples[0]["scores"] == samples[1]["scores"],
                               "repeat_final_sources_identical": samples[0]["final_sources"] == samples[1]["final_sources"]})
        summaries.append({"question_id": qid, "pair_count": len(fixed["pairs"]),
                          "input_trace_sha256": fixed["trace_sha256"], "input_pair_digest": fixed["pair_digest"],
                          "recorded_source_set_reproduced_before_experiment": True, "conditions": conditions})
        assert hashlib.sha256(fixed["database"].read_bytes()).hexdigest() == fixed["corpus_sha256"]
    after_hashes = code_receipt()
    summary = {"schema": "private-reranker-intra-threads-v1", "private_diagnostics": True,
               "not_product_trace": True, "provider_calls": 0, "answer_model_calls": 0,
               "network_attempts_blocked": len(NETWORK_ATTEMPTS), "local_only": True,
               "fixed_pair_order": True, "fixed_query_groups": True, "batch_size": 8,
               "fixed_inter_threads": 2, "real_query_or_dummy_inference_prewarm": False,
               "conditions": [1, 2, 4], "trials": 2, "sequence": sequence,
               "cpu_before": before_cpu, "cpu_after": cpu_receipt(), "ort_version": ort.__version__,
               "original_session_options": original_options, "provider_options": provider_options,
               "asset_setup": {"receipt": asset_receipt, "wall_ms": asset_wall_ms},
               "session_setup": setup, "source_hashes_as_run": hashes,
               "source_hashes_differing_from_v8_freeze": [name for name, value in hashes.items()
                   if freeze["runtime_files"].get("docthinker/server/" + name) != value],
               "source_hashes_after_run": after_hashes,
               "source_changes_during_run": [name for name, value in hashes.items() if after_hashes.get(name) != value],
               "frozen_runtime_claim": False,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "helper_script_sha256": hashlib.sha256((ROOT / "scripts/diagnose_rerank_length_bucketing_offline.py").read_bytes()).hexdigest(),
               "interpretation_limit": "Fixed-pair local experiment, not service end-to-end latency or timeout proof. Two rounds, OS load/thermal scheduling uncontrolled. Equality is observed for these inputs, not guaranteed for all inputs/hardware.",
               "questions": summaries}
    assert not NETWORK_ATTEMPTS
    write_private(output / "summary.json", summary)


if __name__ == "__main__":
    main()
