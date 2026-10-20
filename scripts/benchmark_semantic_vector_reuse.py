"""Synthetic-only semantic cache oracle and CPU overhead benchmark.

No real documents, file stores, credentials, model assets or network are used.
The deterministic encoder checks selection equivalence; its elapsed time is
not a claim about production ONNX inference or full model response latency.
Run from the full checkout, which includes the frozen reference test fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import numpy as np

from docthinker.server import research_semantic as semantic


class VirtualAssets:
    def __truediv__(self, other):
        return SimpleNamespace(is_file=lambda: True)


class Tokenizer:
    def encode(self, text, **kwargs):
        return SimpleNamespace(offsets=[(index, index + 1) for index in range(len(text))])


class Encoder:
    def __init__(self):
        self.passages = 0
        self.queries = 0

    def embed(self, texts, **kwargs):
        for text in texts:
            if text.startswith("passage: "):
                self.passages += 1
            else:
                self.queries += 1
            digest = hashlib.sha256(text.encode()).digest()
            vector = np.zeros(semantic.MODEL_DIMENSION, dtype=np.float32)
            vector[:len(digest)] = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) + 1
            yield vector


def run(module, chunks, repeats):
    samples = []
    results = []
    counts = []
    states = [(None, chunks), ("Archive evidence", chunks), ("Archive evidence", chunks),
              (None, chunks), ("Archive evidence", chunks + [
                  {"id": "outside", "document_id": "late-doc", "content": "New document outside the bounded corpus."}])]
    for _ in range(repeats):
        module._VECTORS.clear()
        model = Encoder()
        subject = module.LocalSemanticRetriever(min_score=0)
        subject.model_path = VirtualAssets()
        subject._model, subject._tokenizer, subject.fingerprint = model, Tokenizer(), "synthetic-encoder"
        durations, packets = [], []
        for question, sources in states:
            started = time.perf_counter()
            packet = (subject.prepare_documents(sources, session_id="synthetic/project") if question is None else
                      subject.semantic_scores(question, sources, session_id="synthetic/project"))
            durations.append(round((time.perf_counter() - started) * 1000, 3))
            packets.append(packet)
        samples.append(durations)
        results.append(packets)
        counts.append({"passage_windows": model.passages, "queries": model.queries})
    return samples, results, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    if not 1 <= arguments.chunks <= 4096 or not 1 <= arguments.repeats <= 10:
        parser.error("chunks 1..4096 and repeats 1..10 are required")
    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/reference_semantic_cache_20261013.py"
    spec = importlib.util.spec_from_file_location("docthinker.server._benchmark_frozen_semantic_cache", fixture)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    chunks = [{"id": f"c{index:04}", "document_id": "synthetic-doc",
               "content": f"Archive evidence number {index}."} for index in range(arguments.chunks)]
    old_samples, old_results, old_counts = run(old, chunks, arguments.repeats)
    new_samples, new_results, new_counts = run(semantic, chunks, arguments.repeats)
    for old_round, new_round in zip(old_results, new_results):
        for expected, actual in zip(old_round, new_round):
            assert actual["scores"] == expected["scores"] and actual["spans"] == expected["spans"]
            for key in ("status", "cached_chunks", "query_cache_hit", "scanned_chunks", "corpus_scan_limited",
                        "limited_chunks", "accepted_chunks", "best_cosine", "query_tokens", "query_truncated"):
                if key in expected["trace"]:
                    assert actual["trace"][key] == expected["trace"][key]
            assert actual["trace"]["legacy_passage_window_budget_used"] == expected["trace"]["passage_windows_encoded"]
    stages = ["cold_prepare", "query_1", "query_2", "repeat_full_prepare", "query_after_late_document"]
    summary = [{"stage": name, "old_median_ms": statistics.median(row[index] for row in old_samples),
                "new_median_ms": statistics.median(row[index] for row in new_samples)} for index, name in enumerate(stages)]
    print(json.dumps({"scope": "synthetic_encoder_cpu_overhead_not_real_inference_or_end_to_end_answer",
                      "chunks": arguments.chunks, "repeats": arguments.repeats, "complete_scores_spans_and_selection_diagnostics_equal": True,
                      "new_provider_calls": 0, "old_encoding_counts": old_counts, "new_encoding_counts": new_counts,
                      "stage_medians": summary, "old_samples_ms": old_samples, "new_samples_ms": new_samples,
                      "new_final_retained_cache_bytes": new_results[-1][-1]["trace"]["retained_cache_bytes"],
                      "source_sha256": hashlib.sha256(Path(semantic.__file__).read_bytes()).hexdigest(),
                      "frozen_reference_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
