#!/usr/bin/env python3
"""Private controlled CE query-view diagnosis over an already exposed trace.

Four alternative scoring experiments, not four production scoring branches.
No new retrieval, source-derived queries, holdout access or provider calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_research_selection_offline import NETWORK_ATTEMPTS, read_chunks, LocalEvidenceReranker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--focus-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-diagnostics", action="store_true", required=True)
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text())
    events = trace["events"]
    facets = next(row["facets"] for row in events if row["event"] == "query_plan")
    original = next(row for row in facets if row["kind"] == "original")
    focus = next(row for row in facets if row["id"] == args.focus_id and not row.get("derived_translation"))
    aliases = [row for row in facets if row.get("query_alias_of") == focus["id"]
               and row.get("derived_translation") and row.get("query_only") and not row.get("translation_full_query")]
    subjects = list(dict.fromkeys(original.get("anchors", ())))
    if len(subjects) != 1:
        subjects = list(dict.fromkeys(row["text"] for row in facets if row["kind"] == "document"))
    if len(aliases) != 1 or len(subjects) != 1 or subjects[0] not in original["text"]:
        raise SystemExit("Need one retained literal subject and one guarded query-only alias")
    subject, alias = subjects[0], aliases[0]
    queries = {
        "literal": focus["text"],
        "subject_literal": subject + ": " + focus["text"],
        "translated_alias": alias["text"],
        "subject_translated_alias": subject + ": " + alias["text"],
    }
    windows = {row["window"]["id"]: row["window"] for row in events if row["event"] == "window_eligible"}
    pool = next(row["candidate_ids"] for row in events if row["event"] == "candidate_pool_final")
    assert len(pool) <= 48 and len(pool) == len(set(pool))
    corpus = Path(trace["run_dir"]) / "corpora" / trace["corpus_digest"] / "_system/research/research.db"
    before = hashlib.sha256(corpus.read_bytes()).hexdigest()
    chunks = {row["id"]: row for row in read_chunks(corpus)}
    candidates = []
    for identifier in pool:
        window = windows[identifier]
        source = chunks[window["source_id"]]["content"]
        content = source[window["excerpt_start"]:window["excerpt_end"]]
        candidates.append({**window, "content": content})
    model = LocalEvidenceReranker()
    outcomes = []
    for name, query in queries.items():
        result = model.rank_pairs([{**row, "query": query} for row in candidates], session_id="private-focus-query-diagnosis")
        assert result["trace"]["status"] == "ready" and result["trace"]["provider_calls"] == 0
        ranked = sorted(candidates, key=lambda row: (-result["scores"][row["id"]], row["id"]))
        outcomes.append({"variant": name, "query": query, "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                         "trace": result["trace"], "ranked": [dict(row, rank=index + 1, logit=result["scores"][row["id"]])
                                                               for index, row in enumerate(ranked)]})
    assert hashlib.sha256(corpus.read_bytes()).hexdigest() == before == trace["corpus_sha256"]
    payload = {"schema": "private-query-view-ce-diagnosis-v1", "private_diagnostics": True,
               "not_product_trace": True, "not_answer_quality_evaluation": True,
               "source_trace": str(args.trace.resolve()), "corpus_sha256": before,
               "query_subject_origin": "unique_literal_original_question_anchor_or_document",
               "literal_subject": subject, "original_question": original["text"],
               "alias_semantic_equivalence_verified": False, "candidate_count_per_experiment": len(candidates),
               "experiments": 4, "production_pair_budget_unchanged": 48,
               "provider_calls": 0, "network_attempts": len(NETWORK_ATTEMPTS), "outcomes": outcomes}
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(output.parent, 0o700)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.chmod(output, 0o600)
    print(json.dumps({"output": str(output), "candidate_count": len(candidates), "experiments": 4,
                      "provider_calls": 0, "network_attempts": len(NETWORK_ATTEMPTS)}), flush=True)


if __name__ == "__main__":
    main()
