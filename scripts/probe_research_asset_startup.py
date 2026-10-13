#!/usr/bin/env python3
"""Offline process-cold versus asset-only-warm ResearchService timing probe.

Uses copied frozen corpus storage and real local models. Network sockets are
blocked; the final generation adapter returns a fixed timing marker, not a
quality-evaluation answer. Every arm gets a new process and document-vector
preparation, with no query pre-execution. Never edits benchmark artifacts.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def source_digest(rows):
    return digest([{key: row.get(key) for key in ("id", "document_id", "page", "excerpt_start", "excerpt_end")}
                   | {"content_sha256": hashlib.sha256(row.get("content", "").encode()).hexdigest()}
                   for row in rows])


async def child(args):
    started = time.perf_counter()
    network_attempts = []
    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def guarded_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            network_attempts.append(1)
            raise RuntimeError("network_disabled_for_offline_probe")
        return original_connect(sock, address)

    def guarded_connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            network_attempts.append(1)
            raise RuntimeError("network_disabled_for_offline_probe")
        return original_connect_ex(sock, address)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    from docthinker.server.research_service import ResearchService
    from docthinker.session_manager import SessionManager
    from docthinker.server import research_faceted_assist as fa
    from docthinker.server import research_reranker as rr
    from docthinker.server import research_query_translation as qt

    fixture = json.loads(Path(args.fixture).read_text())
    question = next(row for row in fixture["questions"] if row["id"] == args.question_id)
    corpus = Path(args.corpus).resolve()
    metadata = json.loads((corpus / "corpus.json").read_text())
    if set(metadata["document_refs"].values()) != set(question["papers"]):
        raise ValueError("corpus_papers_do_not_match_question")
    events, phase = [], {"value": "setup"}
    event_lock = threading.Lock()

    def record(item):
        with event_lock:
            events.append(item)

    def timed(owner, name, component, trace=False):
        original = getattr(owner, name)

        def wrapped(*values, **options):
            before = time.perf_counter()
            current_phase = phase["value"]
            result = None
            try:
                result = original(*values, **options)
                return result
            finally:
                receipt = {"component": component, "phase": current_phase,
                           "duration_ms": (time.perf_counter() - before) * 1000}
                if trace and isinstance(result, dict):
                    receipt["trace"] = copy.deepcopy(result.get("trace", {}))
                if component == "helper" and isinstance(result, dict):
                    receipt["trace"] = copy.deepcopy(result["reasoning"]["trace"])
                    receipt["source_digest"] = source_digest(result.get("sources", []))
                record(receipt)

        setattr(owner, name, wrapped)

    with tempfile.TemporaryDirectory(prefix="docthinker-assets-probe-") as temporary:
        root = Path(temporary).resolve()
        manager = SessionManager(str(root / "_system"), str(root), migrate_legacy=False)
        sid = manager.create_session("Offline asset timing probe")["id"]
        if sid != "#00001":
            raise ValueError("fresh_session_required")
        clone_started = time.perf_counter()
        shutil.copytree(corpus / "_system" / "research", root / "_system" / "research")
        clone_ms = (time.perf_counter() - clone_started) * 1000
        service = ResearchService(root, context_assistance_enabled=True)
        if service._evidence_scheduler.timeout_seconds != 8.:
            raise ValueError("production_deadline_changed")
        timed(service._evidence_reranker, "_initialize", "reranker_initialize")
        timed(service._query_translator, "_load_runtime", "translator_initialize")
        timed(service._evidence_reranker, "rank_pairs", "reranker", trace=True)
        timed(service._query_translator, "translate", "translation", trace=True)
        timed(service._semantic, "semantic_scores", "semantic_query", trace=True)
        timed(fa, "prepare_evidence_candidates", "candidate_pool", trace=True)
        timed(fa, "select_evidence", "selection", trace=True)
        timed(fa, "build_faceted_assistance", "helper")
        setup_started = time.perf_counter()
        # Explicit baseline E5 preparation keeps this attribution arm stable
        # even if the public prepare_retrieval later also prepares auxiliaries.
        chunks = await asyncio.to_thread(service.all_chunks, sid)
        vector_preparation = await service._semantic_operation(sid, chunks)
        vector_setup_ms = (time.perf_counter() - setup_started) * 1000
        asset_setup_ms = 0.
        asset_scheduler_status = "not_requested"
        asset_preparation_receipt = None

        async def drain():
            wait_started = time.perf_counter()
            while service._evidence_scheduler.running:
                if time.perf_counter() - wait_started > 60:
                    raise TimeoutError("offline_probe_worker_tail_limit")
                await asyncio.sleep(.01)
            return (time.perf_counter() - wait_started) * 1000

        if args.condition == "asset_warm":
            phase["value"] = "asset_setup"

            def assets_only():
                if not rr._INFERENCE_GATE.acquire(blocking=False):
                    raise RuntimeError("unexpected_reranker_busy_in_fresh_process")
                try:
                    service._evidence_reranker._initialize()
                finally:
                    rr._INFERENCE_GATE.release()
                if not qt._INFERENCE_GATE.acquire(blocking=False):
                    raise RuntimeError("unexpected_translator_busy_in_fresh_process")
                try:
                    service._query_translator._runtime = service._query_translator._load_runtime()
                finally:
                    qt._INFERENCE_GATE.release()

            warm_started = time.perf_counter()
            if args.preparation_api == "public":
                asset_preparation_receipt = await service.prepare_assistance_assets()
                asset_scheduler_status = asset_preparation_receipt["status"]
            else:
                preparation = await service._evidence_scheduler.run(assets_only)
                asset_scheduler_status = preparation.status
            # This controlled experiment waits for actual preparation completion
            # and accounts for its whole cost, even if the wait deadline expired.
            await drain()
            asset_setup_ms = (time.perf_counter() - warm_started) * 1000
        if rr._SCORES or qt._CACHE:
            raise ValueError("asset_setup_must_not_populate_query_caches")
        phase["value"] = "answer"
        adapter_calls = []

        async def generation_marker(prompt, **kwargs):
            adapter_calls.append({"prompt_digest": hashlib.sha256(prompt.encode()).hexdigest(),
                                  "prompt_bytes": len(prompt.encode())})
            return "Offline startup timing marker; this is not an evaluated answer. [S1]"

        request = SimpleNamespace(session_id=sid, question=question["question"], reasoning_mode="auto",
                                  use_memory=False, use_conversation_context=False, use_llm_cache=False,
                                  use_self_evolution=False, memory_as_of=None)
        answer_started = time.perf_counter()
        result = await service.answer(request, generation_marker, manager)
        answer_ms = (time.perf_counter() - answer_started) * 1000
        scheduler_status = result.get("reasoning", {}).get("trace", {}).get("scheduler_status")
        tail_ms = await drain()
        answer_source_digest = source_digest(result.get("sources", []))
        await service.stop()
    record_value = {
        "schema": "research-asset-startup-probe-v1", "question_id": args.question_id,
        "question_sha256": hashlib.sha256(question["question"].encode()).hexdigest(),
        "corpus_digest": metadata["corpus_digest"], "condition": args.condition, "trial": args.trial,
        "actual_production_answer_path": True, "evidence_deadline_seconds": 8.,
        "provider_calls": 0, "generation_kind": "constant_offline_timing_marker_not_quality_data",
        "network_attempts": len(network_attempts), "new_process": True,
        "os_file_cache_cold_guaranteed": False, "query_prewarm": False,
        "corpus_clone_ms": clone_ms, "vector_setup_ms": vector_setup_ms,
        "vector_preparation_trace": vector_preparation["trace"],
        "asset_setup_ms": asset_setup_ms, "asset_scheduler_status": asset_scheduler_status,
        "preparation_api": args.preparation_api, "asset_preparation_receipt": asset_preparation_receipt,
        "answer_wall_ms": answer_ms, "scheduler_status": scheduler_status,
        "scheduler_timing": result.get("reasoning", {}).get("trace", {}).get("scheduler_timing"),
        "answer_contract": result.get("reasoning", {}).get("trace", {}).get("answer_contract"),
        "worker_tail_after_answer_ms": tail_ms,
        "setup_plus_answer_and_tail_ms": clone_ms + vector_setup_ms + asset_setup_ms + answer_ms + tail_ms,
        "child_total_ms": (time.perf_counter() - started) * 1000,
        "selected_source_count": len(result.get("sources", [])), "answer_source_digest": answer_source_digest,
        "adapter_calls": adapter_calls, "events": events,
        "source_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "docthinker/server/research_service.py", "docthinker/server/research_reranker.py",
            "docthinker/server/research_answer_contract.py", "docthinker/server/research_prompt_budget.py",
            "docthinker/server/research_scheduler.py", "docthinker/server/research_retrieval.py",
            "docthinker/server/research_query_translation.py", "docthinker/server/research_faceted_assist.py",
            "docthinker/server/research_evidence_selection.py")},
    }
    save(args.output, record_value)
    print(json.dumps({key: record_value[key] for key in ("condition", "trial", "scheduler_status", "answer_wall_ms",
                                                       "asset_setup_ms", "worker_tail_after_answer_ms", "network_attempts")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=str(ROOT / "benchmarks/research_real/research_faceted_ab_v1.json"))
    parser.add_argument("--question-id", default="q02_lost_middle_setup")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--preparation-api", choices=("private", "public"), default="private")
    parser.add_argument("--condition", choices=("process_cold", "asset_warm"))
    parser.add_argument("--trial", type=int, default=0)
    args = parser.parse_args()
    os.environ.update(PYTHON_DOTENV_DISABLED="1", DOCTHINKER_RESEARCH_SEMANTIC="local",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    if args.condition:
        asyncio.run(child(args))
        return
    if not 1 <= args.repetitions <= 5:
        parser.error("repetitions must be 1 through 5")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for trial in range(1, args.repetitions + 1):
        order = ("process_cold", "asset_warm") if trial % 2 else ("asset_warm", "process_cold")
        for condition in order:
            target = output / f"trial-{trial}-{condition}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--condition", condition,
                       "--trial", str(trial), "--fixture", args.fixture, "--question-id", args.question_id,
                       "--preparation-api", args.preparation_api,
                       "--corpus", args.corpus, "--output", str(target)]
            subprocess.run(command, check=True, timeout=180)
            records.append(json.loads(target.read_text()))
    save(output / "summary.json", {"schema": "research-asset-startup-probe-v1", "records": records,
                                   "provider_calls": 0, "scope": "process_startup_attribution_not_answer_quality"})


if __name__ == "__main__":
    main()
