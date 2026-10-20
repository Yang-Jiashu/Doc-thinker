#!/usr/bin/env python3
"""Private fixed-candidate query/metadata controls; never a production selector.

Queries come only from a prior literal/translation plan. Corpus text is used
only on the passage side. Imports reuse the offline runner's socket deny guard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from diagnose_research_selection_offline import NETWORK_ATTEMPTS, read_chunks
from docthinker.server.research_reranker import LocalEvidenceReranker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus-window", required=True)
    parser.add_argument("--query-clause-pair", nargs=2, action="append", metavar=("FIRST_FACET", "SECOND_FACET"),
                        help="Only score these ordered joins of two existing plan facets; may be repeated.")
    parser.add_argument("--private-diagnostics", action="store_true", required=True)
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text())
    database = Path(trace["run_dir"]) / "corpora" / trace["corpus_digest"] / "_system/research/research.db"
    corpus_hash = hashlib.sha256(database.read_bytes()).hexdigest()
    chunks = {row["id"]: row for row in read_chunks(database)}
    events = trace["events"]
    ranked_events = [row for row in events if row["event"] == "rerank_result"]
    facets = next(row["facets"] for row in events if row["event"] == "query_plan")
    rows = []
    for event in ranked_events:
        window = event["window"]
        chunk = chunks[window["source_id"]]
        content = chunk["content"][window["excerpt_start"]:window["excerpt_end"]]
        rows.append({**window, "filename": chunk["filename"], "content": content})
    assert len(rows) == len({row["id"] for row in rows}) == 48
    assert args.focus_window in {row["id"] for row in rows}
    original_queries = {event["query"] for event in ranked_events}
    assert len(original_queries) == 1, "This control requires a shared-query input trace"
    original_query = next(iter(original_queries))
    variants = [("baseline_exact_query", original_query, False),
                ("baseline_query_passage_metadata", original_query, True)]
    seen_queries = {original_query}
    for facet in facets:
        if facet["kind"] not in {"original", "clause"} or facet["text"] in seen_queries:
            continue
        seen_queries.add(facet["text"])
        variants.append(("query_" + facet["id"], facet["text"], False))
    if args.query_clause_pair:
        by_id = {facet["id"]: facet for facet in facets}
        variants = []
        for first, second in args.query_clause_pair:
            assert first in by_id and second in by_id
            assert all(by_id[key]["kind"] == "clause" for key in (first, second))
            # Delimit complete, already-recorded literal/translation clauses.
            # Do not invent source keywords or remove internal conditions.
            variants.append(("clause_order_" + first + "_" + second,
                             by_id[first]["text"] + "\n" + by_id[second]["text"], False))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    scorer = LocalEvidenceReranker()
    summaries = []
    baseline_scores = None
    for name, query, metadata in variants:
        pairs = [{**row, "query": query, "content":
                  (f"Document: {row['filename']}\nSection: {row.get('section', '')}\n\n" if metadata else "")
                  + row["content"]} for row in rows]
        result = scorer.rank_pairs(pairs, session_id="private-fixed-candidate-controls::" + trace["question_id"])
        if result["trace"].get("status") != "ready":
            raise RuntimeError("Local reranker unavailable: " + result["trace"].get("status", "missing"))
        scores = result["scores"]
        ranking = sorted(rows, key=lambda row: (-scores[row["id"]], row["id"]))
        ranked = [{key: row.get(key) for key in
                   ("id", "source_id", "document_id", "filename", "page", "section", "excerpt_start", "excerpt_end")}
                  | {"rank": index + 1, "logit": scores[row["id"]]}
                  for index, row in enumerate(ranking)]
        if baseline_scores is None:
            baseline_scores = scores
        receipt = {"schema": "private-fixed-candidate-rerank-control-v1",
                   "question_id": trace["question_id"], "variant": name,
                   "query": query, "query_source": "Prior question/facet plan; no corpus-derived query text",
                   "passage_metadata": metadata, "candidate_count": len(rows),
                   "candidate_content_unchanged_except_explicit_metadata_prefix": True,
                   "no_final_quote_or_selection_mutation": True,
                   "provider_api_calls": 0, "network_attempts_blocked": len(NETWORK_ATTEMPTS),
                   "corpus_sha256": corpus_hash, "trace": result["trace"],
                   "prior_shared_query_max_abs_logit_delta": max(abs(scores[event["window"]["id"]] - event["logit"])
                                                               for event in ranked_events) if name == "baseline_exact_query" else None,
                   "focus": next(row for row in ranked if row["id"] == args.focus_window), "ranking": ranked}
        destination = output / (name + ".json")
        destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        os.chmod(destination, 0o600)
        summary = {"variant": name, "focus_rank": receipt["focus"]["rank"],
                   "focus_logit": receipt["focus"]["logit"], "network_attempts_blocked": len(NETWORK_ATTEMPTS),
                   "output": str(destination)}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    assert hashlib.sha256(database.read_bytes()).hexdigest() == corpus_hash
    assert not NETWORK_ATTEMPTS
    destination = output / "summary.json"
    destination.write_text(json.dumps({"variants": summaries, "provider_api_calls": 0,
                                      "network_attempts_blocked": 0}, indent=2) + "\n")
    os.chmod(destination, 0o600)


if __name__ == "__main__":
    main()
