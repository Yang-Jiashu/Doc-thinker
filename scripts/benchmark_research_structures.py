#!/usr/bin/env python3
"""Offline synthetic rank/index benchmark; no models, private data, or network.

Default: 600/6000/12000 chunks, five samples each, <= 6,000,000 text bytes.
Direct-index cold includes construction plus all queries. The service-policy
first-query arm uses lookup-only misses/full scan, with simulated post-answer
preparation timed separately; warm reuses corpus features, not answers. SQLite
setup is excluded and material reads are timed separately. OS page-cache state
is not controlled (this is NOT cold-disk I/O).
Prints content-free JSON to stdout; caller decides whether to save it.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
import uuid

# Prevent implicit project credential configuration. No provider/client imports.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["DOCTHINKER_RESEARCH_SEMANTIC"] = "off"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docthinker.server.research_corpus_index import PreparedCorpusCache, prepare_research_corpus
from docthinker.server.research_retrieval import build_retrieval_plan
from docthinker.server.research_store import ResearchStore


MAX_TEXT_BYTES = 6_000_000
QUERIES = (
    "How does archive retention update old revisions under validation?",
    "Compare retrieval latency and recall under a fixed query window.",
    "索引如何更新版本并保留限定条件？",
    "Summarize all papers",
)
REFERENCE = ROOT / "tests" / "fixtures" / "reference_research_rank_20261013.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_frozen_rank", REFERENCE)
_ORACLE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ORACLE)


def _uuid(value: int) -> str:
    return str(uuid.UUID(int=value + 1, version=4))


def synthetic_chunks(count: int) -> list[dict]:
    rows = []
    for index in range(count):
        document = _uuid(index // 600)
        text = (f"Entry {index}. Archive retention validates revision {index % 19} before replacing records. "
                f"The experiment uses cohort {index % 17} and a fixed query window. "
                "Retrieval latency differs from recall; measurements retain errors and successful attempts. "
                "Old revisions cannot be recovered after deletion. 索引更新版本时必须保留限定条件。")
        rows.append({"id": _uuid(1_000_000 + index), "document_id": document,
                     "filename": f"synthetic-{index // 600:02}.txt", "ordinal": index % 600,
                     "page": index % 600 + 1, "content": text})
    if sum(len(row["content"].encode("utf-8")) for row in rows) > MAX_TEXT_BYTES:
        raise ValueError("Synthetic corpus exceeds the registered 6,000,000-byte bound")
    return rows


def _summary(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {"samples_ms": [round(value, 3) for value in samples],
            "median_ms": round(statistics.median(samples), 3),
            "p95_ms": round(ordered[max(0, math.ceil(.95 * len(ordered)) - 1)], 3),
            "p95_method": "nearest_rank",
            "sample_count": len(samples)}


def _measure(function):
    start = time.perf_counter()
    result = function()
    return result, (time.perf_counter() - start) * 1000


def _queries(rows, prepared=None, *, oracle=False):
    if oracle:
        return [_ORACLE.build_retrieval_plan(question, rows) for question in QUERIES]
    return [build_retrieval_plan(question, rows, prepared_corpus=prepared) for question in QUERIES]


def _create_synthetic_store(root: Path, rows: list[dict]) -> ResearchStore:
    store = ResearchStore(root)
    documents = {}
    for row in rows:
        documents.setdefault(row["document_id"], []).append(row)
    with store._write() as db:
        for order, (identifier, members) in enumerate(documents.items()):
            digest = hashlib.sha256("".join(row["content"] for row in members).encode()).hexdigest()
            db.execute("INSERT INTO research_documents "
                       "(document_id,session_id,filename,extension,content_hash,byte_size,state,"
                       "chunk_count,text_chars,created_at,updated_at) VALUES(?,?,?,?,?,?,'ready',?,?,?,?)",
                       (identifier, "#00001", members[0]["filename"], ".txt", digest,
                        sum(len(row["content"].encode()) for row in members), len(members),
                        sum(len(row["content"]) for row in members), float(order), float(order)))
        db.executemany("INSERT INTO research_chunks VALUES(?,?,?,?,?,?)",
                       [(row["id"], "#00001", row["document_id"], row["ordinal"], row["content"], row["page"])
                        for row in rows])
    return store


def _material_read(store: ResearchStore) -> list[dict]:
    # Mirrors the original service read exactly; each page rechecks live state.
    chunks = []
    for offset in range(0, 12000, 600):
        page = store.source_chunks("#00001", limit=600, offset=offset)
        chunks.extend(page)
        if len(page) < 600:
            break
    return chunks


def _foundation_query(store: ResearchStore, question: str, cache=None, *, cold_snapshot=None):
    """Actual synchronous store/cache/rank building blocks, not full answer()."""
    if cache is None:
        return _ORACLE.build_retrieval_plan(question, _material_read(store))
    current = store.all_source_chunks("#00001")
    prepared = cache.lookup(current, namespace="#00001")
    if cold_snapshot is not None:
        if prepared is not None:
            raise AssertionError("First-query cache must be empty")
        cold_snapshot.update(cache=cache, rows=current)
    # Misses use full scan now; hits retain BOTH checks: lookup() and rank().
    return build_retrieval_plan(question, current, prepared_corpus=prepared)


def benchmark(count: int, repeats: int) -> dict:
    rows = synthetic_chunks(count)
    # Correctness is whole-plan equality (including gaps/scores/digest), not
    # merely identical top source IDs. These oracle evaluations are untimed.
    expected = _queries(rows, oracle=True)
    baseline_times, cold_times, warm_times, read_times = [], [], [], []
    foundation_times = {name: [] for name in (
        "baseline_first_single_query", "deferred_cold_first_single_query",
        "baseline_four_query_group", "indexed_warm_four_query_group", "material_read_one_select",
        "post_answer_background_prepare_wait")}
    prepared = prepare_research_corpus(rows)
    with tempfile.TemporaryDirectory(prefix="docthinker-structures-synthetic-") as directory, \
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="synthetic-corpus-warm") as warm_worker:
        store = _create_synthetic_store(Path(directory).resolve(), rows)
        service_cache = PreparedCorpusCache()
        # Prewarm for the warm-query arm. This setup is excluded from warm
        # query timings; the same work is separately measured for every cold
        # first query below using its actual captured source snapshot.
        warm_worker.submit(service_cache.get, store.all_source_chunks("#00001"), namespace="#00001").result()
        if _foundation_query(store, QUERIES[0], service_cache) != expected[0]:
            raise AssertionError("Foundation cache warm-up differs from frozen oracle")
        for repeat in range(repeats):
            # Alternate cold/baseline execution order to reduce a fixed-order
            # thermal bias. This is still an uncontrolled development machine.
            def cold():
                return _queries(rows, prepare_research_corpus(rows))
            functions = [("baseline", lambda: _queries(rows, oracle=True)), ("cold", cold)]
            if repeat % 2:
                functions.reverse()
            for name, function in functions:
                plans, elapsed = _measure(function)
                if plans != expected:
                    raise AssertionError(f"Full plan mismatch for {name}; performance result rejected")
                (baseline_times if name == "baseline" else cold_times).append(elapsed)
            plans, elapsed = _measure(lambda: _queries(rows, prepared))
            if plans != expected:
                raise AssertionError("Full plan mismatch for warm index; performance result rejected")
            warm_times.append(elapsed)
            materials, elapsed = _measure(lambda: _material_read(store))
            if [(x["id"], x["content"]) for x in materials] != [(x["id"], x["content"]) for x in rows]:
                raise AssertionError("Material read geometry/order mismatch")
            read_times.append(elapsed)
            # These additionally include actual materialization plus namespace
            # cache access; no source-body cache or shortened DB read is used.
            cold_snapshot = {}
            foundation_functions = [
                ("baseline_first_single_query", lambda: _foundation_query(store, QUERIES[0]), expected[0]),
                ("deferred_cold_first_single_query", lambda: _foundation_query(store, QUERIES[0], PreparedCorpusCache(), cold_snapshot=cold_snapshot), expected[0]),
                ("baseline_four_query_group", lambda: [_foundation_query(store, question) for question in QUERIES], expected),
                ("indexed_warm_four_query_group", lambda: [_foundation_query(store, question, service_cache) for question in QUERIES], expected),
            ]
            if repeat % 2:
                foundation_functions.reverse()
            for name, function, reference in foundation_functions:
                actual, elapsed = _measure(function)
                if actual != reference:
                    raise AssertionError(f"Full foundation plan mismatch for {name}; timing rejected")
                foundation_times[name].append(elapsed)
            if cold_snapshot["cache"].info()["snapshots"]:
                raise AssertionError("Cold request must not synchronously build corpus features")
            # Simulated post-answer lifecycle: wait for one actual local worker
            # in isolation to account for preparation cost, never inside the
            # first-query timer. This is not a fake model or a claim about
            # worker/model CPU overlap under real concurrent user traffic.
            def background_prepare():
                return warm_worker.submit(cold_snapshot["cache"].get, cold_snapshot["rows"],
                                          namespace="#00001").result()
            _, elapsed = _measure(background_prepare)
            foundation_times["post_answer_background_prepare_wait"].append(elapsed)
            if cold_snapshot["cache"].lookup(cold_snapshot["rows"], namespace="#00001") is None:
                raise AssertionError("Completed background preparation must be reusable")
            if _foundation_query(store, QUERIES[0], cold_snapshot["cache"]) != expected[0]:
                raise AssertionError("Background-prepared first-query replay differs from oracle")
            cold_snapshot["cache"].close()
            materials, elapsed = _measure(lambda: store.all_source_chunks("#00001"))
            if [(x["id"], x["content"]) for x in materials] != [(x["id"], x["content"]) for x in rows]:
                raise AssertionError("One-SELECT material read geometry/order mismatch")
            foundation_times["material_read_one_select"].append(elapsed)
        retained_cache = service_cache.info()
        service_cache.close()
    return {"chunk_count": count, "document_count": len({row["document_id"] for row in rows}),
            "text_bytes": sum(len(row["content"].encode()) for row in rows),
            "prepared_feature_bytes": prepared.feature_bytes,
            "query_count": len(QUERIES), "whole_plan_equal": True,
            "baseline_frozen_rank": _summary(baseline_times),
            "indexed_cold_prepare_plus_queries": _summary(cold_times),
            "indexed_warm_queries": _summary(warm_times),
            "material_read_paginated": _summary(read_times),
            "service_foundation_preparation": {
                "whole_plan_equal": True,
                "scope": "synchronous live material read + namespace corpus cache + rank; excludes event-loop scheduling, assistance, model and other answer stages",
                "cold_cache": "new empty cache; one live read + lookup miss + unchanged full scan, with no index construction",
                "warm_cache": "same private cache per corpus; each query still reads live data and validates both signatures",
                "background_wait": "separate simulated post-answer single-worker submit/get/wait; no model and no foreground overlap claim",
                "retained_cache": retained_cache,
                **{name: _summary(values) for name, values in foundation_times.items()},
            },
            "source_quotes_emitted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, nargs="+", default=[600, 6000, 12000])
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20 or not args.chunks or any(not 1 <= count <= 12000 for count in args.chunks):
        parser.error("Use 1–20 repeats and chunk counts in 1–12000")
    result = {"benchmark": "synthetic_research_structures_v3", "provider_calls": 0,
              "llm_tokens": 0, "private_corpus_used": False, "text_byte_cap": MAX_TEXT_BYTES,
              "python": platform.python_version(), "platform": platform.system(),
              "oracle_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
              "index_sha256": hashlib.sha256((ROOT / "docthinker/server/research_corpus_index.py").read_bytes()).hexdigest(),
              "rank_sha256": hashlib.sha256((ROOT / "docthinker/server/research_retrieval.py").read_bytes()).hexdigest(),
              "store_sha256": hashlib.sha256((ROOT / "docthinker/server/research_store.py").read_bytes()).hexdigest(),
              "service_sha256": hashlib.sha256((ROOT / "docthinker/server/research_service.py").read_bytes()).hexdigest(),
              "limits": ["No answer-quality or model-latency measurement", "No controlled cold-disk benchmark",
                         "Direct-index statistics exclude material reading and service cache lookup; service_foundation_preparation includes both",
                         "Direct eager index cold statistics are diagnostic only, not the current deferred-first-query service policy",
                         "With five samples, nearest-rank p95 is the observed maximum, not a tail-latency guarantee",
                         "Pure rank equality is not complete service/scheduler equivalence",
                         "Synthetic development timings; no universal speedup claim"],
              "results": [benchmark(count, args.repeats) for count in args.chunks]}
    result["four_production_files_unchanged"] = all(
        result[key] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for key, path in (("rank_sha256", "docthinker/server/research_retrieval.py"),
                          ("index_sha256", "docthinker/server/research_corpus_index.py"),
                          ("store_sha256", "docthinker/server/research_store.py"),
                          ("service_sha256", "docthinker/server/research_service.py")))
    if not result["four_production_files_unchanged"]:
        raise RuntimeError("Production source changed during measurement; timing run rejected")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
