#!/usr/bin/env python3
"""Private fixed-48 control deleting only explicitly named peer papers from query."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

from diagnose_research_selection_offline import NETWORK_ATTEMPTS, read_chunks
from docthinker.server.research_reranker import LocalEvidenceReranker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-diagnostics", action="store_true", required=True)
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text())
    assert trace["question_id"] == "q08_synthesis_memory_patterns"
    database = Path(trace["run_dir"]) / "corpora" / trace["corpus_digest"] / "_system/research/research.db"
    digest = hashlib.sha256(database.read_bytes()).hexdigest()
    chunks = {row["id"]: row for row in read_chunks(database)}
    events = [row for row in trace["events"] if row["event"] == "rerank_result"]
    original_queries = {row["query"] for row in events}
    assert len(original_queries) == 1
    original_query = original_queries.pop()
    names = {"rag.pdf": "RAG", "graphrag.pdf": "GraphRAG", "memgpt.pdf": "MemGPT"}
    assert all(re.search(r"\b" + re.escape(name) + r"\b", original_query, re.I) for name in names.values())
    queries = {filename: re.sub(r"\b(?:" + "|".join(re.escape(other) for other in names.values() if other != name)
                               + r")\b", "", original_query, flags=re.I)
               for filename, name in names.items()}
    rows = []
    for event in events:
        window = event["window"]
        chunk = chunks[window["source_id"]]
        assert chunk["filename"] in names
        rows.append({**window, "filename": chunk["filename"], "query": queries[chunk["filename"]],
                     "content": chunk["content"][window["excerpt_start"]:window["excerpt_end"]],
                     "prior_shared_query_logit": event["logit"]})
    assert len(rows) == len({row["id"] for row in rows}) == 48
    result = LocalEvidenceReranker().rank_pairs(rows, session_id="private-q08-peer-name-deletion")
    assert result["trace"].get("status") == "ready", result["trace"]
    rankings = {}
    top4 = {}
    for filename in names:
        candidates = sorted((row for row in rows if row["filename"] == filename),
                            key=lambda row: (-result["scores"][row["id"]], row["id"]))
        seen = set()
        ranked = []
        for rank, row in enumerate(candidates, 1):
            new_source = row["source_id"] not in seen
            seen.add(row["source_id"])
            ranked.append({**row, "rank_in_document": rank,
                           "distinct_source_rank": len(seen) if new_source else None,
                           "logit": result["scores"][row["id"]]})
        rankings[filename] = ranked
        top4[filename] = [row for row in ranked if row["distinct_source_rank"] is not None][:4]
    assert hashlib.sha256(database.read_bytes()).hexdigest() == digest
    assert not NETWORK_ATTEMPTS
    receipt = {"schema": "private-fixed48-peer-name-deletion-v1", "question_id": trace["question_id"],
               "private_diagnostics": True, "not_product_trace": True,
               "original_query": original_query, "queries_by_filename": queries,
               "query_mutation": "Delete only whole-word explicitly named peer papers; preserve every other character",
               "candidate_count": 48, "candidate_ids_order_offsets_content_unchanged": True,
               "no_selection_or_final_quote_mutation": True, "provider_api_calls": 0,
               "network_attempts_blocked": len(NETWORK_ATTEMPTS), "corpus_sha256": digest,
               "trace": result["trace"], "top4_distinct_sources": top4, "rankings_by_document": rankings}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output.parent, 0o700)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    os.chmod(args.output, 0o600)
    print(json.dumps({"output": str(args.output), "trace": result["trace"],
                      "provider_api_calls": 0, "network_attempts_blocked": 0}), flush=True)


if __name__ == "__main__":
    main()
