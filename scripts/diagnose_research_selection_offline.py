#!/usr/bin/env python3
"""Explicit private, network-denied replay of existing research answer records.

Does not read benchmark gold or holdout fixtures and never invokes an answer
provider. Frozen corpora are opened read-only. Output contains private query
facets, source identities/offsets and selection receipts: do not publish it as
the normal product trace. Local replay timings are NOT service benchmark times.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DOCTHINKER_RESEARCH_SEMANTIC"] = "local"

NETWORK_ATTEMPTS = []


def deny_network(*args, **kwargs):
    NETWORK_ATTEMPTS.append("blocked_connection_attempt")
    raise RuntimeError("Network access disabled for offline diagnosis")


socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network
socket.create_connection = deny_network

from docthinker.server.research_faceted_assist import build_faceted_assistance
from docthinker.server.research_query_translation import LocalQueryTranslator
from docthinker.server.research_reranker import LocalEvidenceReranker
from docthinker.server.research_semantic import LocalSemanticRetriever
from docthinker.server.research_retrieval import build_retrieval_plan


def identity(sources):
    return [{key: row.get(key) for key in ("id", "content", "excerpt_start", "excerpt_end")}
            for row in sources]


def read_chunks(database):
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        values = connection.execute(
            "SELECT c.*,d.filename,d.content_hash FROM research_chunks c "
            "JOIN research_documents d ON d.document_id=c.document_id "
            "WHERE c.session_id=d.session_id AND d.state='ready' "
            "ORDER BY d.created_at,d.document_id,c.ordinal"
        ).fetchall()
    return [{"id": row["chunk_id"], "chunk_id": row["chunk_id"], "document_id": row["document_id"],
             "filename": row["filename"], "ordinal": row["ordinal"], "page": row["page"],
             "content_hash": row["content_hash"], "text": row["text"], "content": row["text"]}
            for row in values]


def code_receipt():
    names = ("research_evidence_selection.py", "research_faceted_assist.py",
             "research_selection_diagnostics.py", "research_query_facets.py",
             "research_sentence_windows.py", "research_figure_axes.py", "research_semantic.py", "research_reranker.py",
             "research_query_translation.py", "research_retrieval.py")
    return {name: hashlib.sha256((ROOT / "docthinker/server" / name).read_bytes()).hexdigest() for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--question-id", action="append", required=True)
    parser.add_argument("--private-diagnostics", action="store_true", required=True)
    args = parser.parse_args()
    run_dir, output_dir = args.run_dir.resolve(), args.output_dir.resolve()
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise SystemExit("Refusing to write inside frozen run")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(output_dir, 0o700)
    frozen_code = code_receipt()  # Imports above pin this process while another worker edits source.
    semantic, reranker, translator = LocalSemanticRetriever(), LocalEvidenceReranker(), LocalQueryTranslator()
    for index, question_id in enumerate(args.question_id):
        if not question_id or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in question_id):
            raise SystemExit("Invalid question id")
        on = json.loads((run_dir / "answers" / f"{question_id}.r1.on.json").read_text())
        off = json.loads((run_dir / "answers" / f"{question_id}.r1.off.json").read_text())
        corpus = run_dir / "corpora" / on["corpus_digest"] / "_system/research/research.db"
        corpus_hash_before = hashlib.sha256(corpus.read_bytes()).hexdigest()
        chunks = read_chunks(corpus)
        namespace = "private-offline-v5-replay::" + question_id
        started = time.perf_counter()
        prewarm = semantic.prepare_documents(chunks, session_id=namespace)
        primary = semantic.semantic_scores(on["question"], chunks, session_id=namespace)
        if prewarm["trace"].get("status") != "ready" or primary["trace"].get("status") != "ready":
            raise RuntimeError("Pinned local semantic model unavailable")
        baseline = build_retrieval_plan(on["question"], chunks, [], mode="evidence",
                                        semantic_scores=primary["scores"], semantic_spans=primary["spans"])
        plain = build_faceted_assistance(on["question"], baseline["sources"], chunks, semantic, reranker,
                                          session_id=namespace, primary_semantic=primary, query_translator=translator)
        events = []
        observed = build_faceted_assistance(on["question"], baseline["sources"], chunks, semantic, reranker,
                                             session_id=namespace, primary_semantic=primary, query_translator=translator,
                                             diagnostic_callback=events.append)
        same_callback = identity(plain["sources"]) == identity(observed["sources"])
        same_v5 = identity(on["result"]["sources"]) == identity(observed["sources"])
        same_baseline = identity(off["result"]["sources"]) == identity(baseline["sources"])
        assert same_callback, "Diagnostic callback changed final sources"
        assert hashlib.sha256(corpus.read_bytes()).hexdigest() == corpus_hash_before, "Frozen corpus changed"
        traces = [plain["trace"], observed["trace"], prewarm["trace"], primary["trace"]]
        assert all(trace.get("provider_calls", 0) == 0 for trace in traces)
        payload = {
            "schema": "private-offline-selection-diagnostics-v1", "question_id": question_id,
            "private_diagnostics": True, "not_product_trace": True,
            "provider_api_calls": 0, "network_attempts_blocked": len(NETWORK_ATTEMPTS),
            "local_only": True, "run_dir": str(run_dir), "corpus_digest": on["corpus_digest"],
            "corpus_sha256": corpus_hash_before, "runtime_source_hashes_at_import": frozen_code,
            "comparison": {"callback_preserves_sources": same_callback,
                           "v5_on_sources_identical": same_v5, "v5_off_baseline_identical": same_baseline},
            "timing_scope": "Offline warmed direct-function replay; excludes service scheduler and answer model. Not V5 latency.",
            "elapsed_ms": (time.perf_counter()-started)*1000,
            "prewarm_trace": prewarm["trace"], "primary_semantic_trace": primary["trace"],
            "first_call_assistance_trace": plain["trace"],
            "assistance_trace": observed["trace"], "events": events,
            "final_sources": observed["sources"], "baseline_sources": baseline["sources"],
            "v5_on_source_identity": identity(on["result"]["sources"]),
        }
        destination = output_dir / f"{question_id}.json"
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        os.chmod(destination, 0o600)
        print(json.dumps({"question_id": question_id, "events": len(events), "comparison": payload["comparison"],
                          "network_attempts_blocked": len(NETWORK_ATTEMPTS), "output": str(destination)}), flush=True)


if __name__ == "__main__":
    main()
