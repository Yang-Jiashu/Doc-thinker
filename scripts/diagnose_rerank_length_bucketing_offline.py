#!/usr/bin/env python3
"""PRIVATE fixed-pair, network-denied reranker scheduling experiment.

Replays saved facet/semantic receipts, not translation, retrieval models, gold,
or answer generation. The only model work is the pinned local reranker. Runtime
files stay unchanged; a process-local proxy permutes passages within each exact
query, then restores logits to their original IDs. Dynamic int8 batch effects
are measured, never presumed absent. Results are not production latency claims.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import uuid

from diagnose_research_selection_offline import NETWORK_ATTEMPTS, ROOT, code_receipt, identity, read_chunks
from docthinker.server.research_evidence_selection import prepare_evidence_candidates, select_evidence
from docthinker.server.research_faceted_assist import _rerank_pairs
from docthinker.server.research_reranker import BATCH_SIZE, MAX_PAIR_TOKENS, LocalEvidenceReranker


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def write_private(destination, payload):
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(destination.parent, 0o700)
    with os.fdopen(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def candidate_signature(row):
    return {"id": row["id"], "source_id": row["source_id"], "document_id": row["document_id"],
            "excerpt_start": row["excerpt_start"], "excerpt_end": row["excerpt_end"],
            "query_sha256": sha(row["query"]), "content_sha256": sha(row["content"])}


def prepare_fixed(trace_path):
    trace = json.loads(trace_path.read_text())
    database = Path(trace["run_dir"]) / "corpora" / trace["corpus_digest"] / "_system/research/research.db"
    corpus_hash = hashlib.sha256(database.read_bytes()).hexdigest()
    assert corpus_hash == trace["corpus_sha256"], "Input corpus changed"
    events = trace["events"]
    facets = next(row["facets"] for row in events if row["event"] == "query_plan")
    question = next(row["text"] for row in facets if row["kind"] == "original")
    semantic = {row["facet_id"]: {"scores": row["scores"], "spans": row["spans"], "trace": {"status": row["status"]}}
                for row in events if row["event"] == "facet_semantic" and row["status"] == "ready"}
    plan = {"facets": deepcopy(facets)}
    baseline = trace["baseline_sources"]
    pool = prepare_evidence_candidates(question, read_chunks(database), baseline, plan, semantic)
    pairs = _rerank_pairs(question, pool["candidates"], facets)
    ranked = [row for row in events if row["event"] == "rerank_result"]
    assert 0 < len(pairs) == len(ranked) <= 48
    expected = [{"id": row["window"]["id"], "source_id": row["window"]["source_id"],
                 "document_id": row["window"]["document_id"],
                 "excerpt_start": row["window"]["excerpt_start"], "excerpt_end": row["window"]["excerpt_end"],
                 "query_sha256": sha(row["query"]),
                 "content_sha256": row["input_receipt"]["submitted_passage_sha256"]} for row in ranked]
    assert [candidate_signature(row) for row in pairs] == expected, "Rebuilt pairs/order differ from actual trace"
    recorded_receipts = {row["window"]["id"]: row["input_receipt"] for row in ranked}
    selection = select_evidence(deepcopy(pairs), {row["window"]["id"]: row["logit"] for row in ranked},
                                plan, baseline, input_receipts=recorded_receipts)
    assert identity(selection["sources"]) == identity(trace["final_sources"]), "Trace selection not reproduced"
    return {"trace": trace, "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            "database": database, "corpus_sha256": corpus_hash, "pairs": pairs, "plan": plan,
            "baseline": baseline, "recorded_receipts": recorded_receipts,
            "pair_digest": digest(expected)}


class ForwardReceipt:
    """Observe actual ONNX inputs after padding/truncation, never change them."""

    def __init__(self, session):
        self.session = session
        self.batches = []
        self.group_index = None

    def __getattr__(self, name):
        return getattr(self.session, name)

    def run(self, output_names, input_feed, *args, **kwargs):
        ids = input_feed["input_ids"]
        mask = input_feed.get("attention_mask")
        batch, width = map(int, ids.shape)
        start, cpu = time.perf_counter(), time.process_time()
        result = self.session.run(output_names, input_feed, *args, **kwargs)
        self.batches.append({"query_group": self.group_index, "batch_size": batch, "padded_length": width,
                             "forward_padded_tokens": batch * width,
                             "forward_attention_proxy": batch * width * width,
                             "nonpadding_tokens": int(mask.sum()) if mask is not None else None,
                             "nonpadding_lengths": list(map(int, mask.sum(axis=1))) if mask is not None else None,
                             "forward_wall_ms": (time.perf_counter() - start) * 1000,
                             "forward_cpu_ms": (time.process_time() - cpu) * 1000})
        return result


class GroupOrderProxy:
    def __init__(self, model, lengths, variant, forward):
        self.model, self.lengths, self.variant, self.forward = model, lengths, variant, forward
        self.groups = []

    def rerank(self, query, documents, batch_size=BATCH_SIZE, **kwargs):
        documents = list(documents)
        lengths = [self.lengths[(sha(query), sha(content))] for content in documents]
        order = list(range(len(documents)))
        if self.variant == "length_sorted":
            order.sort(key=lambda index: (min(lengths[index], MAX_PAIR_TOKENS), index))
        group = {"query_sha256": sha(query), "pair_count": len(documents),
                 "pair_tokens_before_truncation_original_order": lengths, "execution_order": order,
                 "ideal_original_padded_tokens": padding_proxy(lengths, batch_size),
                 "ideal_sorted_padded_tokens": padding_proxy(sorted(lengths), batch_size)}
        self.forward.group_index = len(self.groups)
        started, cpu = time.perf_counter(), time.process_time()
        values = list(self.model.rerank(query, [documents[index] for index in order], batch_size=batch_size, **kwargs))
        assert len(values) == len(documents)
        restored = [None] * len(values)
        for index, value in zip(order, values):
            restored[index] = value
        group.update(wall_ms=(time.perf_counter() - started) * 1000,
                     cpu_ms=(time.process_time() - cpu) * 1000)
        self.groups.append(group)
        return iter(restored)


def padding_proxy(lengths, batch_size):
    values = [min(length, MAX_PAIR_TOKENS) for length in lengths]
    return sum(len(batch) * max(batch) for offset in range(0, len(values), batch_size)
               if (batch := values[offset:offset + batch_size]))


def score_once(scorer, fixed, variant, trial, run_id):
    pairs = fixed["pairs"]
    lengths = {}
    # Reuse the exact prior pair receipt only after matching all input hashes.
    # rank_pairs independently retokenizes and must produce identical receipts.
    for row in pairs:
        receipt = fixed["recorded_receipts"][row["id"]]
        actual_query = row["query"][:receipt["actual_query_character_end"]]
        assert sha(actual_query) == receipt["actual_query_sha256"]
        lengths[(sha(actual_query), sha(row["content"]))] = receipt["pair_tokens_before_truncation"]
    model = scorer._model
    backend = model.model
    session = backend.model
    forward = ForwardReceipt(session)
    proxy = GroupOrderProxy(model, lengths, variant, forward)
    backend.model = forward
    scorer._model = proxy
    namespace = f"private-bucket-diagnostic::{run_id}::{fixed['trace']['question_id']}::{trial}::{variant}"
    try:
        started, cpu = time.perf_counter(), time.process_time()
        ranked = scorer.rank_pairs(pairs, session_id=namespace)
        wall_ms, cpu_ms = (time.perf_counter() - started) * 1000, (time.process_time() - cpu) * 1000
    finally:
        scorer._model = model
        backend.model = session
    assert ranked["trace"]["status"] == "ready", ranked["trace"]
    assert ranked["trace"]["cache_hits"] == 0, "Namespace cache contamination"
    assert ranked["input_receipts"] == fixed["recorded_receipts"], "Pair receipts or truncation changed"
    assert set(ranked["scores"]) == {row["id"] for row in pairs}
    selection = select_evidence(deepcopy(pairs), ranked["scores"], fixed["plan"], fixed["baseline"],
                                input_receipts=ranked["input_receipts"])
    return {"variant": variant, "trial": trial, "namespace_sha256": sha(namespace),
            "rank_pairs_wall_ms": wall_ms, "rank_pairs_cpu_ms": cpu_ms,
            "trace": ranked["trace"], "scores": ranked["scores"], "input_receipts": ranked["input_receipts"],
            "input_receipts_equal_prior_trace": True, "input_pair_digest": fixed["pair_digest"],
            "groups": proxy.groups, "actual_forward_batches": forward.batches,
            "actual_forward_padded_tokens": sum(row["forward_padded_tokens"] for row in forward.batches),
            "actual_nonpadding_tokens": sum(row["nonpadding_tokens"] for row in forward.batches),
            "actual_forward_attention_proxy": sum(row["forward_attention_proxy"] for row in forward.batches),
            "forward_wall_ms": sum(row["forward_wall_ms"] for row in forward.batches),
            "ideal_original_padded_tokens": sum(row["ideal_original_padded_tokens"] for row in proxy.groups),
            "ideal_sorted_padded_tokens": sum(row["ideal_sorted_padded_tokens"] for row in proxy.groups),
            "final_sources": identity(selection["sources"]), "final_source_digest": digest(identity(selection["sources"])),
            "selection_trace": selection["trace"]}


def compare(a, b, pairs):
    differences = [abs(a["scores"][key] - b["scores"][key]) for key in a["scores"]]
    inversions, ties_changed, rank_changes = 0, 0, 0
    groups = OrderedDict()
    for row in pairs:
        groups.setdefault(row["query"], []).append(row["id"])
    for ids in groups.values():
        for index, left in enumerate(ids):
            ar = 1 + sum(a["scores"][key] > a["scores"][left] for key in ids)
            br = 1 + sum(b["scores"][key] > b["scores"][left] for key in ids)
            rank_changes += ar != br
            for right in ids[index + 1:]:
                da, db = a["scores"][left] - a["scores"][right], b["scores"][left] - b["scores"][right]
                inversions += da * db < 0
                ties_changed += (da == 0) != (db == 0)
    return {"max_abs_logit_difference": max(differences), "mean_abs_logit_difference": statistics.mean(differences),
            "within_query_pair_order_inversions": inversions, "within_query_tie_changes": ties_changed,
            "within_query_competition_rank_changes": rank_changes,
            "pair_receipts_identical": a["input_receipts"] == b["input_receipts"],
            "final_source_order_and_spans_identical": a["final_sources"] == b["final_sources"],
            "final_source_set_identical": {digest(row) for row in a["final_sources"]} == {digest(row) for row in b["final_sources"]},
            "wall_ms_delta_sorted_minus_current": b["rank_pairs_wall_ms"] - a["rank_pairs_wall_ms"],
            "cpu_ms_delta_sorted_minus_current": b["rank_pairs_cpu_ms"] - a["rank_pairs_cpu_ms"],
            "actual_padding_token_reduction": a["actual_forward_padded_tokens"] - b["actual_forward_padded_tokens"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--private-diagnostics", action="store_true", required=True)
    args = parser.parse_args()
    if not 2 <= args.trials <= 4:
        parser.error("Use two to four alternating trials")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be new or empty; never replace prior results")
    os.umask(0o077)
    run_id = uuid.uuid4().hex
    hashes = code_receipt()
    freeze = json.loads((ROOT / "dist/real-paper-ab-20261011/evidence-trace-v8-freeze/freeze.json").read_text())
    changed = [name for name, value in hashes.items()
               if freeze["runtime_files"].get("docthinker/server/" + name) != value]
    fixed_inputs = [prepare_fixed(path.resolve()) for path in args.trace]
    scorer = LocalEvidenceReranker()
    setup_start = time.perf_counter()
    assets = scorer.prepare_assets()
    assert assets["trace"]["status"] == "ready", assets
    setup_ms = (time.perf_counter() - setup_start) * 1000
    summary = {"schema": "private-reranker-length-bucketing-v1", "private_diagnostics": True,
               "not_product_trace": True, "provider_calls": 0, "answer_model_calls": 0,
               "local_only": True, "network_attempts_blocked": len(NETWORK_ATTEMPTS),
               "source_hashes_as_run": hashes, "source_hashes_differing_from_v8_freeze": changed,
               "frozen_runtime_claim": False, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "runtime_policy": "No runtime edits; process-local ordering proxy only; no query prewarm or answer generation.",
               "pair_policy": "Exact saved queries, passage bytes, IDs, order, truncation receipts, and <=48 pair budget verified before/after; only within-query execution order changes.",
               "setup": {"asset_only": assets, "wall_ms": setup_ms, "real_query_inference_prewarm": False},
               "measurement": "Sequential alternating AB/BA, fresh score namespaces, one process/pinned CPU model. First inference may be cold; OS cache/load uncontrolled. Process CPU includes native threads. Padding measured from actual ONNX input tensors.",
               "numerical_warning": "Dynamic int8 batched inference may change logits/ranks/final sources; output equality is measured, not a mathematical guarantee.",
               "questions": []}
    for fixed in fixed_inputs:
        qid = fixed["trace"]["question_id"]
        trials, results = [], []
        for trial in range(args.trials):
            order = ["current", "length_sorted"] if trial % 2 == 0 else ["length_sorted", "current"]
            outputs = {}
            for variant in order:
                measured = score_once(scorer, fixed, variant, trial + 1, run_id)
                write_private(output / f"{qid}.trial{trial + 1}.{variant}.json", measured)
                outputs[variant] = measured
                print(json.dumps({"question_id": qid, "trial": trial + 1, "variant": variant,
                                  "wall_ms": round(measured["rank_pairs_wall_ms"], 3),
                                  "actual_padded_tokens": measured["actual_forward_padded_tokens"]}), flush=True)
            trials.append({"trial": trial + 1, "execution_order": order,
                           **compare(outputs["current"], outputs["length_sorted"], fixed["pairs"])})
            results.append(outputs)
        question_summary = {"question_id": qid, "input_trace_sha256": fixed["trace_sha256"],
                            "input_pair_digest": fixed["pair_digest"], "pair_count": len(fixed["pairs"]),
                            "recorded_final_selection_reproduced_before_experiment": True,
                            "trials": trials,
                            "current_repeat_comparison": compare(results[0]["current"], results[1]["current"], fixed["pairs"]),
                            "sorted_repeat_comparison": compare(results[0]["length_sorted"], results[1]["length_sorted"], fixed["pairs"]),
                            "median_current_wall_ms": statistics.median(row["current"]["rank_pairs_wall_ms"] for row in results),
                            "median_sorted_wall_ms": statistics.median(row["length_sorted"]["rank_pairs_wall_ms"] for row in results),
                            "median_current_cpu_ms": statistics.median(row["current"]["rank_pairs_cpu_ms"] for row in results),
                            "median_sorted_cpu_ms": statistics.median(row["length_sorted"]["rank_pairs_cpu_ms"] for row in results)}
        assert hashlib.sha256(fixed["database"].read_bytes()).hexdigest() == fixed["corpus_sha256"]
        write_private(output / f"{qid}.summary.json", question_summary)
        summary["questions"].append(question_summary)
    summary["source_hashes_after_run"] = code_receipt()
    summary["source_changes_during_run"] = [name for name, value in hashes.items()
                                            if summary["source_hashes_after_run"].get(name) != value]
    summary["network_attempts_blocked"] = len(NETWORK_ATTEMPTS)
    assert not NETWORK_ATTEMPTS, "Unexpected attempted network access"
    write_private(output / "summary.json", summary)


if __name__ == "__main__":
    main()
