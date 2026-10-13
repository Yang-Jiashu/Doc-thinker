"""Lossless vector reuse against the frozen r6 semantic implementation.

All corpora, encoders, namespaces and databases in this file are synthetic.
The oracle has its own old LRU and model gate; no provider or network is used.
"""
import hashlib
import importlib.util
from pathlib import Path
import random
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from docthinker.server import research_semantic as semantic


_REFERENCE = Path(__file__).parent / "fixtures" / "reference_semantic_cache_20261013.py"
_SPEC = importlib.util.spec_from_file_location("docthinker.server._frozen_semantic_cache_20261013", _REFERENCE)
_OLD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_OLD)


def test_frozen_reference_fingerprint_has_not_changed():
    assert hashlib.sha256(_REFERENCE.read_bytes()).hexdigest() == "e65126f74f974be2c3e1e9219b3beeb132750e8f8ae73070d7fb7e413925fc60"


class Tokenizer:
    def encode(self, text, **kwargs):
        return SimpleNamespace(offsets=[(index, index + 1) for index in range(len(text))])


class Encoder:
    def __init__(self):
        self.calls = []

    def embed(self, texts, **kwargs):
        self.calls.append(list(texts))
        for text in texts:
            values = hashlib.sha256(text.encode()).digest()
            vector = np.zeros(semantic.MODEL_DIMENSION, dtype=np.float32)
            vector[:len(values)] = np.frombuffer(values, dtype=np.uint8).astype(np.float32) + 1
            yield vector


def row(text, identifier="c1", document="d1", **values):
    return {"id": identifier, "document_id": document, "content": text, **values}


@pytest.fixture
def pair(tmp_path, monkeypatch):
    models, searches = [], []
    for name, module in (("old", _OLD), ("new", semantic)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "docthinker-model.json").write_text("{}")
        model = Encoder()
        module._MODELS.clear()
        module._VECTORS.clear()
        monkeypatch.setattr(module, "_INFERENCE_GATE", threading.BoundedSemaphore(1))
        monkeypatch.setattr(module, "_model_manifest", lambda path: ({}, "fingerprint"))
        monkeypatch.setattr(module, "_load_model", lambda path, model=model: (model, Tokenizer()))
        searches.append(module.LocalSemanticRetriever(directory, min_score=0))
        models.append(model)
    return (*searches, *models)


def equal(old, new, question, rows, namespace="private/project"):
    method = "prepare_documents" if question is None else "semantic_scores"
    args = (rows,) if question is None else (question, rows)
    expected = getattr(old, method)(*args, session_id=namespace)
    actual = getattr(new, method)(*args, session_id=namespace)
    assert actual["scores"] == expected["scores"]
    assert actual["spans"] == expected["spans"]
    for key in ("status", "operation", "scanned_chunks", "corpus_scan_limited", "limited_chunks",
                "cached_chunks", "query_cache_hit", "query_tokens", "query_truncated",
                "accepted_chunks", "best_cosine", "query_embedded", "provider_calls", "llm_tokens"):
        if key in expected["trace"]:
            assert actual["trace"][key] == expected["trace"][key], key
    if expected["trace"]["status"] == "ready":
        assert actual["trace"]["legacy_passage_window_budget_used"] == expected["trace"]["passage_windows_encoded"]
        assert actual["trace"]["retained_cache_bytes"] <= new.cache_bytes
        assert actual["trace"]["retained_cache_entries"] <= new.cache_entries
    return expected, actual


def test_4096_complete_corpus_reuses_vectors_without_changing_legacy_selection(pair):
    old, new, old_model, new_model = pair
    rows = [row(f"Archive evidence number {index}.", f"c{index:04}") for index in range(4096)]
    equal(old, new, None, rows)
    expected, actual = equal(old, new, "Archive evidence", rows)
    assert expected["trace"]["passage_windows_encoded"] == 2048
    assert actual["trace"]["passage_windows_encoded"] == 0
    assert actual["trace"]["physical_cached_chunks"] == 4096
    expected, actual = equal(old, new, "Archive evidence", rows)
    assert expected["trace"]["passage_windows_encoded"] == 2049
    assert actual["trace"]["passage_windows_encoded"] == 0
    assert sum(len(call) for call in new_model.calls) == 4097
    assert sum(len(call) for call in old_model.calls) == 8195
    assert len(semantic._LOGICAL_VECTORS) == 2048


@pytest.mark.parametrize("entries", [1, 2, 3, 16, 2048])
@pytest.mark.parametrize("budget", [3, 7, 8192])
def test_long_windows_eviction_order_and_mixed_queries_equal_frozen(pair, monkeypatch, entries, budget):
    old, new, _, _ = pair
    old.cache_entries = entries
    new.cache_entries = entries if entries < 2048 else 4224
    for module in (_OLD, semantic):
        monkeypatch.setattr(module, "MAX_TOTAL_WINDOWS", budget)
    rows = [row((f"Source {index} distinct text. " * (index + 8)), f"c{index}") for index in range(8)]
    for question, indices in ((None, range(8)), ("Source 1", range(8)), (None, [7, 5, 3]),
                              ("Source 3", [6, 0, 7, 3, 2]), ("Source 1", range(8)),
                              (None, [0, 1, 0, 2]), ("Source 3", range(7, -1, -1))):
        equal(old, new, question, [rows[index] for index in indices])


def test_expanded_physical_cache_does_not_promote_out_of_budget_rows(pair, monkeypatch):
    old, new, _, _ = pair
    old.cache_entries = 2
    new.cache_entries = 8
    monkeypatch.setattr(semantic, "LEGACY_CACHE_ENTRIES", 2)
    for module in (_OLD, semantic):
        monkeypatch.setattr(module, "MAX_TOTAL_WINDOWS", 1)
    rows = [row(f"Source {name}.", name) for name in "abcdef"]
    for indices in ([0, 1], [2, 3], [0], [4, 5], [0, 1, 2, 3, 4, 5]):
        equal(old, new, None, [rows[index] for index in indices])
    expected, actual = equal(old, new, "Source b", rows)
    assert set(actual["scores"]) == set(expected["scores"])
    assert actual["trace"]["physical_cached_chunks"] >= actual["trace"]["cached_chunks"]


def test_byte_eviction_reencodes_logical_hits_never_removes_candidates(pair):
    old, new, _, _ = pair
    # Deliberately below a single vector + metadata, exercising a real physical
    # miss even though the compatibility LRU still declares a logical hit.
    new.cache_bytes = semantic.MODEL_DIMENSION * 4
    old.cache_entries = new.cache_entries = 2
    rows = [row(f"Archive evidence {index}.", f"c{index}") for index in range(12)]
    equal(old, new, None, rows)
    assert not semantic._VECTORS and semantic._LOGICAL_VECTORS
    expected, actual = equal(old, new, "Archive evidence", rows)
    assert expected["trace"]["cached_chunks"] == 2
    assert actual["trace"]["passage_windows_encoded"] == 12
    assert actual["trace"]["legacy_passage_window_budget_used"] == 10


def test_window_byte_estimate_retains_the_previous_worst_case_working_set():
    namespace = "𠀀" * 1024
    key = (namespace, "f" * 64, "passage", "h" * 64)
    value = (np.zeros((24, 384), dtype=np.float32), False, tuple((i, i + 1) for i in range(24)))
    assert (semantic._VectorCache._size(key, value) + semantic._IdentityLRU._size(key)) * 2048 < semantic.MAX_CACHE_BYTES
    assert semantic.MAX_CACHE_BYTES == 100 * 1024 * 1024


def test_previous_worst_case_namespace_window_working_set_is_really_retained():
    semantic._VECTORS.clear()
    try:
        for index in range(2048):
            key = ("𠀀" * 1024, "f" * 64, "passage", f"{index:064x}")
            value = (np.zeros((24, 384), dtype=np.float32), False, tuple((i, i + 1) for i in range(24)))
            semantic._VECTORS[key] = value
            semantic._LOGICAL_VECTORS[key] = None
        semantic._VECTORS.trim(4224, 100 * 1024 * 1024)
        assert len(semantic._VECTORS) == len(semantic._LOGICAL_VECTORS) == 2048
        charged = semantic._VECTORS.retained_bytes + semantic._LOGICAL_VECTORS.retained_bytes
        assert charged == 101216256 and charged < semantic.MAX_CACHE_BYTES
    finally:
        semantic._VECTORS.clear()


def test_default_long_window_budget_progression_and_byte_eviction_equal_frozen(pair):
    old, new, _, _ = pair
    class SharedOffsetTokenizer(Tokenizer):
        # Equivalent character-token offsets, memoized only inside this
        # synthetic fixture to avoid making acceptance a tokenizer benchmark.
        def __init__(self):
            self.offsets = {}
        def encode(self, text, **kwargs):
            if len(text) not in self.offsets:
                self.offsets[len(text)] = [(i, i + 1) for i in range(len(text))]
            return SimpleNamespace(offsets=self.offsets[len(text)])
    old._initialize()
    new._initialize()
    old._tokenizer = new._tokenizer = SharedOffsetTokenizer()
    rows = [row(f"Record {index:04}: " + "background evidence " * 260, f"c{index:04}") for index in range(2700)]
    for question in (None, "Record evidence", None, "Record evidence", None,
                     "Archive support", None, "Record evidence"):
        expected, actual = equal(old, new, question, rows)
        assert actual["trace"]["passage_windows_encoded"] <= expected["trace"]["passage_windows_encoded"]
        assert actual["trace"]["retained_cache_bytes"] <= 100 * 1024 * 1024
    assert len(semantic._LOGICAL_VECTORS) == 2048
    assert len(semantic._VECTORS) < 2700  # The byte cap really evicted vectors.


def test_duplicate_content_and_source_id_winners_unchanged(pair):
    old, new, _, _ = pair
    rows = [row("Archive evidence.", "a"), row("Archive evidence.", "b"),
            row("Changed winner must not replace first.", "a"),
            row("Generated fake evidence.", "g", is_generated=True),
            row("No source owner.", "orphan", document=None)]
    equal(old, new, None, rows)
    equal(old, new, "Archive", rows)
    equal(old, new, "Archive", rows[:1])


def test_scope_change_deletion_fingerprint_and_same_display_id_remain_isolated(pair):
    old, new, _, _ = pair
    first = [row("Private archive evidence.", "same")]
    for namespace in ("tenant-a/#00001", "tenant-b/#00001"):
        equal(old, new, None, first, namespace)
        equal(old, new, "Archive", first, namespace)
        equal(old, new, "Archive", [], namespace)
        equal(old, new, "Archive", [row("Different current source.", "same")], namespace)
    old.fingerprint = new.fingerprint = "different-model-fingerprint"
    equal(old, new, "Archive", first, "tenant-a/#00001")


def test_randomized_bounded_call_sequences_equal_independent_old_lru(pair, monkeypatch):
    old, new, _, _ = pair
    old.cache_entries = 7
    new.cache_entries = 17
    monkeypatch.setattr(semantic, "LEGACY_CACHE_ENTRIES", 7)
    for module in (_OLD, semantic):
        monkeypatch.setattr(module, "MAX_TOTAL_WINDOWS", 13)
    randomizer = random.Random(527199)
    rows = [row((f"Source {index}. " * (index + 8)), f"c{index}") for index in range(23)]
    for _ in range(100):
        selected = randomizer.sample(rows, randomizer.randrange(0, len(rows)))
        equal(old, new, randomizer.choice([None, "Source", "Archive", "Source 后半段"]),
              selected, randomizer.choice(["tenant-a/project", "tenant-b/project"]))


def test_distinct_retriever_instances_and_custom_limits_preserve_shared_legacy_timeline(pair):
    old, new, _, _ = pair
    old_small = _OLD.LocalSemanticRetriever(old.model_path, min_score=0, cache_entries=2)
    new_small = semantic.LocalSemanticRetriever(new.model_path, min_score=0, cache_entries=2)
    rows = [row(f"Archive record {i}.", f"c{i}") for i in range(37)]
    equal(old, new, None, rows)
    equal(old, new, "Archive", rows)
    equal(old_small, new_small, "Archive", rows[:1])
    equal(old, new, "Archive", rows)
    equal(old_small, new_small, "Other", rows[3:7], "other-tenant/project")
    equal(old, new, "Archive", rows)


def test_global_clear_and_accounting_reset_both_cache_layers(pair):
    old, new, _, _ = pair
    equal(old, new, "Archive", [row("Archive evidence.")])
    assert semantic._VECTORS.retained_bytes and semantic._LOGICAL_VECTORS
    _OLD._VECTORS.clear()
    semantic._VECTORS.clear()
    assert semantic._VECTORS.retained_bytes == 0 and not semantic._LOGICAL_VECTORS
    equal(old, new, "Archive", [row("Archive evidence.")])


@pytest.mark.parametrize("value", [True, 0, -1, "4096", 4225])
def test_invalid_physical_entry_budget_rejected(value):
    with pytest.raises(ValueError):
        semantic.LocalSemanticRetriever(cache_entries=value)


def test_clear_resets_shadow_metadata_accounting(pair):
    old, new, _, _ = pair
    equal(old, new, None, [row("Archive evidence.")])
    assert semantic._LOGICAL_VECTORS.retained_bytes > 0
    semantic._VECTORS.clear()
    assert semantic._LOGICAL_VECTORS.retained_bytes == 0
