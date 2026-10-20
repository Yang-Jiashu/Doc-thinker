"""Reproducible synthetic retrieval benchmark; no provider or private inputs.

Run from the repository root:
    PYTHON_DOTENV_DISABLED=1 python -m scripts.benchmark_graph_discovery

This compares the production lexical retrieval plan with the same plan plus
bounded graph expansion. Expected items are fixture source spans, NOT generated
answers or human quality labels. Timing measures graph.expand only: input
validation, cache lookup/index build and bounded search, not end-to-end latency.
The script reads no datasets, keys or configuration and emits JSON to stdout.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import random
from statistics import median
import time

from docthinker.server.research_graph_retrieval import BoundedEvidenceGraph
from docthinker.server.research_retrieval import build_retrieval_plan

SCHEMA = "graph-discovery-synthetic-retrieval-v1"
SEED = 20261011
QUESTION = "Find an evidence chain from Alpha to Delta."


def _chunk(identifier, content):
    return {"id": identifier, "document_id": "document-" + identifier,
            "filename": identifier + ".txt", "ordinal": 0, "content": content}


def _chain(bridge="Beta reduces Gamma."):
    return [_chunk("start", "Alpha increases Beta."),
            _chunk("bridge", bridge), _chunk("end", "Gamma improves Delta.")]


def fixtures():
    """Every baseline is computed, never manually populated with seed answers."""
    base = _chain()
    full = [*base, *[_chunk(f"background-{i}", f"Report {i} discussed Alpha and Delta.")
                    for i in range(6)]]
    definitions = [
        ("endpoint_bridge", QUESTION, base, ["bridge"], [], {}, "recall", []),
        ("two_hop_idea_neighborhood", "Propose research ideas about Alpha.", base,
         ["bridge", "end"], [], {}, "recall", []),
        ("unrelated_relation_words", QUESTION,
         [*base, _chunk("noise-1", "Omega increases Lemon."),
          _chunk("noise-2", "Betamax reduces Orange."),
          _chunk("noise-3", "Lemon improves Orange.")],
         ["bridge"], ["noise-1", "noise-2", "noise-3"], {}, "recall", []),
        ("negated_bridge", QUESTION, _chain("Beta does not reduce Gamma."),
         ["bridge"], [], {}, "recall", ["graph_counterevidence_in_path"]),
        ("conditional_bridge", QUESTION,
         _chain("Beta reduces Gamma under cold conditions. No trials were conducted under heat."),
         ["bridge"], [], {}, "recall", ["graph_path_conditions_differ"]),
        ("source_slots_full", QUESTION, full, ["bridge"], [], {}, "budget_guard",
         ["graph_capacity_reached"]),
        ("byte_budget_full", QUESTION, base, ["bridge"], [],
         {"max_source_bytes": len(base[0]["content"].encode()) + len(base[2]["content"].encode())},
         "budget_guard", ["graph_capacity_reached"]),
        ("relation_word_only_is_not_a_path", QUESTION,
         [_chunk("start", "Alpha increases Beta."), _chunk("noise", "Omega increases Lemon.")],
         [], ["noise"], {}, "negative_control", ["graph_no_connected_path"]),
    ]
    return [{"id": identifier, "question": question, "chunks": deepcopy(rows),
             "expected_bridge_ids": expected, "unrelated_ids": noise,
             "graph_options": options, "category": category,
             "required_gap_codes": gaps}
            for identifier, question, rows, expected, noise, options, category, gaps in definitions]


def _bytes(sources):
    return sum(len(row["content"].encode("utf-8")) for row in sources)


def _exact_sources(sources, corpus):
    live = {row["id"]: row for row in corpus}
    if len({row["id"] for row in sources}) != len(sources):
        return False
    for index, row in enumerate(sources):
        original = live.get(row["id"])
        if original is None or row["document_id"] != original["document_id"]:
            return False
        start, end = row.get("excerpt_start"), row.get("excerpt_end")
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(original["content"])
                or original["content"][start:end] != row["content"]
                or row["label"] != f"S{index + 1}"):
            return False
    return True


def _coverage(sources, expected_ids, corpus):
    # Short fixture bridges are whole statements including their qualifications.
    # An ID with a clipped condition does not count as a covered bridge.
    original = {row["id"]: row["content"] for row in corpus}
    supplied = {row["id"]: row["content"] for row in sources}
    matched = [identifier for identifier in expected_ids
               if original[identifier] in supplied.get(identifier, "")]
    return {"covered": len(matched), "total": len(expected_ids),
            "fraction": len(matched) / len(expected_ids) if expected_ids else None,
            "covered_ids": matched}


def benchmark_fixture(fixture):
    rows, question = fixture["chunks"], fixture["question"]
    unchanged_input = deepcopy(rows)
    plan = build_retrieval_plan(question, rows)
    baseline = plan["sources"]
    baseline_copy = deepcopy(baseline)
    result = BoundedEvidenceGraph().expand(
        question, baseline, rows, protected_source_ids=[row["id"] for row in baseline],
        **fixture["graph_options"])
    selected = result["sources"]
    added = selected[len(baseline):]
    gaps = [gap["code"] for gap in result["gaps"]]
    expected = fixture["expected_bridge_ids"]
    baseline_coverage, graph_coverage = (_coverage(sources, expected, rows)
                                         for sources in (baseline, selected))
    no_replacement = selected[:len(baseline)] == baseline_copy
    checks = {
        "seed_order_labels_and_text_preserved": no_replacement,
        "input_not_mutated": rows == unchanged_input and baseline == baseline_copy,
        "exact_current_source_spans": _exact_sources(selected, rows),
        "within_source_limit": len(selected) <= 8,
        "within_byte_limit": _bytes(selected) <= fixture["graph_options"].get("max_source_bytes", 12000),
        "zero_model_calls": result["trace"]["model_calls"] == 0,
        "not_a_semantic_proof": result["trace"]["semantic_entailment_verified"] is False,
        "no_unrelated_additions": not (set(fixture["unrelated_ids"]) & {row["id"] for row in added}),
        "required_limitations_reported": set(fixture["required_gap_codes"]) <= set(gaps),
    }
    if fixture["category"] == "recall":
        checks["complete_expected_bridges"] = graph_coverage["covered"] == len(expected)
    else:
        checks["no_additions_under_guard"] = not added
    if fixture["id"] == "two_hop_idea_neighborhood":
        checks["actual_two_hop_path"] = any(path["expansion_hops"] == 2 for path in result["trace"]["paths"])
    return {"id": fixture["id"], "category": fixture["category"],
            "corpus_chunks": len(rows), "question": question,
            "baseline_source_ids": [row["id"] for row in baseline],
            "graph_source_ids": [row["id"] for row in selected],
            "added_source_ids": [row["id"] for row in added],
            "baseline_bridge_coverage": baseline_coverage,
            "graph_bridge_coverage": graph_coverage,
            "baseline_source_bytes": _bytes(baseline), "graph_source_bytes": _bytes(selected),
            "added_source_bytes": _bytes(added),
            "model_calls": result["trace"]["model_calls"], "gap_codes": gaps,
            "checks": checks, "passed": all(checks.values())}


def synthetic_timing_corpus(count=1800):
    if type(count) is not int or not 3 <= count <= 2000:
        raise ValueError("Timing corpus size must be between 3 and 2000")
    randomizer = random.Random(SEED)
    relations = ("increases", "reduces", "improves")
    rows = _chain()
    for index in range(count - len(rows)):
        relation = relations[randomizer.randrange(len(relations))]
        # Alphabetic entities stay inside the production parser's conservative
        # grammar. Numeric names would benchmark unparseable text, not a graph.
        letters = "".join(chr(65 + (index // divisor) % 26) for divisor in (676, 26, 1))
        rows.append(_chunk(f"synthetic-{index:04d}",
                           f"Sector{letters} {relation} Outcome{letters}."))
    randomizer.shuffle(rows)
    return rows


def _times(samples):
    return {"samples_ms": [round(value, 6) for value in samples],
            "median_ms": round(median(samples), 6),
            "min_ms": round(min(samples), 6), "max_ms": round(max(samples), 6)}


def benchmark_graph_timing(*, count=1800, repeats=5):
    if type(repeats) is not int or not 1 <= repeats <= 30:
        raise ValueError("Timing repetitions must be between 1 and 30")
    rows = synthetic_timing_corpus(count)
    # Baseline ranking and fixture construction are deliberately OUTSIDE the
    # graph timing; no total RAG or answer latency is inferred from these runs.
    baseline = build_retrieval_plan(QUESTION, rows)["sources"]
    cold, warm, checks = [], [], []
    for _ in range(repeats):
        graph = BoundedEvidenceGraph()
        started = time.perf_counter_ns()
        first = graph.expand(QUESTION, baseline, rows)
        cold.append((time.perf_counter_ns() - started) / 1_000_000)
        started = time.perf_counter_ns()
        second = graph.expand(QUESTION, baseline, rows)
        warm.append((time.perf_counter_ns() - started) / 1_000_000)
        checks.append(first["trace"]["cache_hit"] is False and second["trace"]["cache_hit"] is True
                      and first["sources"] == second["sources"]
                      and "bridge" in second["trace"]["bridge_source_ids"]
                      and second["trace"].get("edge_count") == count
                      and first["trace"]["model_calls"] == second["trace"]["model_calls"] == 0)
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()
    return {"scope": "graph.expand_only_validation_index_or_cache_lookup_and_bounded_search",
            "excluded": ["baseline_retrieval", "model_inference", "network", "document_parsing",
                         "reranker", "hypothesis_scaffold", "end_to_end_response"],
            "corpus_origin": "fixed_seed_generated_statements_not_real_papers",
            "graph_shape": "three_edge_target_chain_plus_disjoint_one_edge_backgrounds",
            "seed": SEED, "corpus_chunks": count, "corpus_sha256": digest,
            "repeats": repeats, "indexed_chunks": second["trace"].get("indexed_chunks"),
            "edge_count": second["trace"].get("edge_count"),
            "cold": _times(cold), "warm": _times(warm),
            "model_calls": 0, "cache_and_bridge_checks_passed": all(checks)}


def benchmark(*, count=1800, repeats=5):
    cases = [benchmark_fixture(fixture) for fixture in fixtures()]
    eligible = [row for row in cases if row["category"] == "recall"]
    timing = benchmark_graph_timing(count=count, repeats=repeats)
    def coverage(key):
        return {"covered": sum(row[key]["covered"] for row in eligible),
                "total": sum(row[key]["total"] for row in eligible)}
    return {"schema": SCHEMA, "synthetic_only": True, "model_calls": 0,
            "measurement_limits": [
                "Expected bridges are authored source-span fixtures, not standard model answers.",
                "Coverage measures only these synthetic bridge statements, not semantic answer quality.",
                "No real papers, model inference, billed tokens or end-to-end speed are measured.",
                "Bytes are UTF-8 source bytes, not tokenizer or billing measurements.",
                "Graph associations, negative statements and conditions are not causal proof.",
                "The baseline is lexical production retrieval; embedding/reranker models are not loaded.",
                "Cold means a new graph instance, not a cold operating-system or Python process.",
                "Timing is hardware/load-dependent; fixture correctness is deterministic."],
            "fixtures": cases,
            "summary": {"passed": sum(row["passed"] for row in cases), "total": len(cases),
                        "bridge_coverage_scope": "recall_fixtures_only_budget_guards_reported_separately",
                        "baseline_bridge_coverage": coverage("baseline_bridge_coverage"),
                        "graph_bridge_coverage": coverage("graph_bridge_coverage"),
                        "added_source_bytes": sum(row["added_source_bytes"] for row in cases),
                        "budget_guard_cases": sum(row["category"] == "budget_guard" for row in cases)},
            "graph_stage_timing": timing,
            "passed": all(row["passed"] for row in cases) and timing["cache_and_bridge_checks_passed"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=1800, help="Synthetic timing chunks (3..2000)")
    parser.add_argument("--repeats", type=int, default=5, help="Cold/warm graph pairs (1..30)")
    args = parser.parse_args(argv)
    if not 3 <= args.chunks <= 2000 or not 1 <= args.repeats <= 30:
        parser.error("--chunks must be 3..2000; --repeats must be 1..30")
    result = benchmark(count=args.chunks, repeats=args.repeats)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
