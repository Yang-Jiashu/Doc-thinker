"""Local semantic retrieval boundaries; opt-in real ONNX integration at bottom."""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from docthinker.server import research_semantic as semantic


def row(content, identifier="one", document_id="doc", **extra):
    return {"id": identifier, "document_id": document_id, "content": content, **extra}


class Tokenizer:
    def encode(self, text, **kwargs):
        return SimpleNamespace(offsets=[(i, i + 1) for i in range(len(text))])


class Embedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts, **kwargs):
        self.calls.append(list(texts))
        for text in texts:
            vector = np.zeros(semantic.MODEL_DIMENSION, dtype=np.float32)
            vector[0 if "target" in text else 1] = 1.0
            yield vector


@pytest.fixture
def retriever(tmp_path, monkeypatch):
    (tmp_path / "docthinker-model.json").write_text("{}")
    model = Embedder()
    monkeypatch.setattr(semantic, "_model_manifest", lambda path: ({}, "fingerprint"))
    monkeypatch.setattr(semantic, "_load_model", lambda path: (model, Tokenizer()))
    semantic._MODELS.clear()
    semantic._VECTORS.clear()
    return semantic.LocalSemanticRetriever(tmp_path), model


def test_missing_model_does_not_import_or_initialize_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic, "_load_model", lambda _: pytest.fail("must not initialize"))
    result = semantic.LocalSemanticRetriever(tmp_path).semantic_scores("target", [row("target")], session_id="tenant-a/#00001")
    assert result["scores"] == {}
    assert result["trace"]["status"] == "model_not_installed"
    assert result["trace"]["provider_calls"] == 0


def test_live_scope_deleted_sources_content_updates_and_session_isolation(retriever):
    search, model = retriever
    first = search.semantic_scores("target", [row("target"), row("unrelated", "two")], session_id="tenant-a/#00001")
    assert first["scores"] == {"one": 1.0}
    assert first["spans"] == {"one": [0, 6]}
    again = search.semantic_scores("target", [row("unrelated", "two")], session_id="tenant-a/#00001")
    assert again["scores"] == {}  # Removed source cannot come back from cache.
    assert again["spans"] == {}
    assert again["trace"]["cached_chunks"] == 1
    assert again["trace"]["query_cache_hit"] is True
    changed = search.semantic_scores("target", [row("unrelated", "one")], session_id="tenant-a/#00001")
    assert changed["scores"] == {}  # Same ID must not reuse an earlier text vector.
    assert changed["spans"] == {}
    isolated = search.semantic_scores("target", [row("target")], session_id="tenant-b/#00001")
    assert isolated["scores"] == {"one": 1.0}
    assert isolated["spans"] == {"one": [0, 6]}
    assert isolated["trace"]["cached_chunks"] == 0
    assert isolated["trace"]["query_cache_hit"] is False


def test_model_reused_across_services_but_cache_includes_model_fingerprint(retriever):
    search, model = retriever
    search.semantic_scores("target", [row("target")], session_id="tenant/#00001")
    another = semantic.LocalSemanticRetriever(search.model_path)
    result = another.semantic_scores("target", [row("target")], session_id="tenant/#00001")
    assert result["trace"]["model_reused"] is True
    assert result["trace"]["cached_chunks"] == 1
    another.fingerprint = "new-model-fingerprint"
    changed = another.semantic_scores("target", [row("target")], session_id="tenant/#00001")
    assert changed["trace"]["cached_chunks"] == 0
    assert changed["trace"]["query_cache_hit"] is False


def test_generated_rows_do_not_gain_evidence_and_prefixes_are_correct(retriever):
    search, model = retriever
    result = search.semantic_scores("target", [row("target"), row("target", "fake", is_generated=True)], session_id="p")
    assert set(result["scores"]) == {"one"}
    assert model.calls == [["passage: target"], ["query: target"]]
    assert result["trace"]["llm_tokens"] == 0
    assert result["trace"]["semantic_entailment_verified"] is False


def test_local_cache_is_bounded(retriever):
    search, model = retriever
    search.cache_entries = 2
    search.semantic_scores("target", [row("target"), row("target two", "two"), row("target three", "three")], session_id="p")
    assert len(semantic._VECTORS) <= 2


def test_prepare_documents_only_encodes_passages_then_query_is_measured(retriever):
    search, model = retriever
    prepared = search.prepare_documents([row("target")], session_id="p")
    assert prepared["scores"] == {}
    assert prepared["spans"] == {}
    assert prepared["trace"]["query_embedded"] is False
    assert prepared["trace"]["query_embedding_ms"] == 0
    assert model.calls == [["passage: target"]]
    assert all(key[2] == "passage" for key in semantic._VECTORS)
    prepared_again = search.prepare_documents([row("target")], session_id="p")
    assert prepared_again["trace"]["cached_chunks"] == 1
    assert len(model.calls) == 1
    queried = search.semantic_scores("target", [row("target")], session_id="p")
    assert queried["trace"]["cached_chunks"] == 1
    assert queried["trace"]["query_cache_hit"] is False
    assert model.calls[-1] == ["query: target"]


def test_query_truncation_is_visible_and_does_not_truncate_original_cache_key(retriever):
    search, model = retriever
    question = "target " + "a" * 500
    result = search.semantic_scores(question, [row("target")], session_id="p")
    assert result["trace"]["query_truncated"] is True
    assert result["trace"]["query_tokens"] == len(question)
    assert model.calls[-1] == ["query: " + question[:448]]
    another = search.semantic_scores(question + "b", [row("target")], session_id="p")
    assert another["trace"]["query_cache_hit"] is False


def test_query_between_256_and_448_tokens_is_not_cut_to_passage_window(retriever):
    search, model = retriever
    question = "target " + "a" * 300
    result = search.semantic_scores(question, [row("target")], session_id="p")
    assert result["trace"]["query_truncated"] is False
    assert result["trace"]["query_max_tokens"] == 448
    assert result["trace"]["passage_window_tokens"] == 256
    assert model.calls[-1] == ["query: " + question]


def test_best_window_span_tracks_latter_half_and_is_stable_from_cache(retriever):
    search, model = retriever
    text = "无关背景。" * 180 + " target relevant evidence " + "其他结尾。" * 20
    first = search.semantic_scores("target", [row(text)], session_id="tenant-a/#00001")
    start, end = first["spans"]["one"]
    assert 0 <= start < end <= len(text)
    assert start > len(text) // 2
    assert "target" in text[start:end]
    assert end - start <= 256
    assert any(text[start:end] in passage for passage in model.calls[0])
    cached = search.semantic_scores("target", [row(text)], session_id="tenant-a/#00001")
    assert cached["scores"] == first["scores"]
    assert cached["spans"] == first["spans"]
    assert cached["trace"]["cached_chunks"] == 1
    moved = "target evidence " + "background " * 100
    revised = search.semantic_scores("target", [row(moved)], session_id="tenant-a/#00001")
    assert revised["spans"]["one"] == [0, 256]
    assert revised["trace"]["cached_chunks"] == 0
    other = search.semantic_scores("target", [row(moved)], session_id="tenant-b/#00001")
    assert other["trace"]["cached_chunks"] == 0
    assert other["spans"] == revised["spans"]


def test_semantic_spans_exactly_match_live_top40_scores(retriever):
    search, model = retriever
    rows = [row(f"target document {index}", f"source-{index:02d}") for index in range(45)]
    result = search.semantic_scores("target", rows, session_id="p")
    assert len(result["scores"]) == 40
    assert set(result["spans"]) == set(result["scores"])
    for item in rows:
        if item["id"] in result["spans"]:
            left, right = result["spans"][item["id"]]
            assert (left, right) == (0, len(item["content"]))
    current = search.semantic_scores("target", [rows[-1]], session_id="p")
    assert set(current["spans"]) == set(current["scores"]) == {rows[-1]["id"]}


def test_index_version_changes_vector_fingerprint(tmp_path, monkeypatch):
    hashes = {name: (semantic.MODEL_SHA256 if name == semantic.MODEL_FILE else "a" * 64) for name in semantic.MODEL_FILES}
    (tmp_path / "docthinker-model.json").write_text(json.dumps({"model": semantic.MODEL_NAME, "revision": semantic.MODEL_REVISION, "sha256": hashes}))
    for name in semantic.MODEL_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(semantic, "_sha256", lambda path: hashes[str(path.relative_to(tmp_path))])
    _, current = semantic._model_manifest(tmp_path)
    monkeypatch.setattr(semantic, "INDEX_VERSION", "e5-local-window-v1")
    _, old = semantic._model_manifest(tmp_path)
    assert current != old


def test_other_tenant_does_not_queue_behind_active_embedding_worker(retriever, monkeypatch):
    search, model = retriever
    entered, release = threading.Event(), threading.Event()
    original_embed = model.embed

    def paused_embed(texts, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        yield from original_embed(texts, **kwargs)

    monkeypatch.setattr(model, "embed", paused_embed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        active = pool.submit(search.prepare_documents, [row("target")], session_id="tenant-a")
        try:
            assert entered.wait(timeout=5)
            another = semantic.LocalSemanticRetriever(search.model_path)
            started = time.monotonic()
            result = another.semantic_scores("target", [row("target")], session_id="tenant-b")
            assert time.monotonic() - started < 0.5
            assert result["scores"] == {}
            assert result["spans"] == {}
            assert result["trace"]["status"] == "busy"
            assert result["trace"]["provider_calls"] == 0
        finally:
            release.set()
        assert active.result(timeout=5)["trace"]["status"] == "ready"
    assert search.semantic_scores("target", [row("target")], session_id="tenant-a")["trace"]["status"] == "ready"


def test_inference_failure_releases_process_gate_for_next_call(retriever, monkeypatch):
    search, model = retriever
    original_embed = model.embed

    def failed_embed(*args, **kwargs):
        raise RuntimeError("local inference failed")

    monkeypatch.setattr(model, "embed", failed_embed)
    with pytest.raises(RuntimeError, match="local inference failed"):
        search.prepare_documents([row("target")], session_id="tenant-a")
    monkeypatch.setattr(model, "embed", original_embed)
    result = search.semantic_scores("target", [row("target")], session_id="tenant-b")
    assert result["trace"]["status"] == "ready"
    assert result["scores"] == {"one": 1.0}


def test_long_passages_window_instead_of_silently_truncating():
    windows, limited = semantic._windows("a" * 1700, Tokenizer())
    assert windows[0] == (0, 256)
    assert windows[1][0] == 192
    assert windows[-1][1] == 1700
    assert all(right - left <= 256 for left, right in windows)
    assert not limited
    _, limited = semantic._windows("a" * 9000, Tokenizer())
    assert limited


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), True, "0.8"])
def test_bad_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        semantic.LocalSemanticRetriever(min_score=threshold)


def test_corrupted_model_fails_closed_without_loading(tmp_path, monkeypatch):
    (tmp_path / "docthinker-model.json").write_text(json.dumps({"model": "changed"}))
    monkeypatch.setattr(semantic, "_load_model", lambda _: pytest.fail("must not load corrupt assets"))
    result = semantic.LocalSemanticRetriever(tmp_path).semantic_scores("target", [row("target")], session_id="p")
    assert result["scores"] == {}
    assert result["trace"]["status"] == "model_unavailable"


@pytest.mark.skipif(os.getenv("DOCTHINKER_TEST_LOCAL_EMBEDDINGS") != "1", reason="explicit real local model test")
def test_real_e5_chinese_english_and_unrelated_query_without_network(monkeypatch):
    import socket
    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: pytest.fail("runtime must not use network"))
    search = semantic.LocalSemanticRetriever()
    rows = [row("Photovoltaic investment increases renewable electricity production and can reduce dependence on imported fossil fuels.", "energy"),
            row("Coral reefs support diverse marine species but suffer bleaching as ocean temperatures rise.", "coral"),
            row("The bakery prepares sourdough bread by fermenting flour and water before baking.", "bread")]
    first = search.semantic_scores("可再生能源投资如何降低进口依赖？", rows, session_id="real-smoke")
    assert first["trace"]["status"] == "ready", first
    assert next(iter(first["scores"])) == "energy", first
    english = search.semantic_scores("How does renewable electricity investment reduce fuel import dependence?", rows, session_id="real-smoke")
    assert next(iter(english["scores"])) == "energy", english
    unrelated = search.semantic_scores("月球背面的陨石坑直径是多少？", rows, session_id="real-smoke")
    assert unrelated["scores"] == {}, unrelated
    assert unrelated["spans"] == {}, unrelated


@pytest.mark.skipif(os.getenv("DOCTHINKER_TEST_LOCAL_EMBEDDINGS") != "1", reason="explicit real local model test")
def test_real_chinese_query_locates_latter_half_of_long_english_chunk(monkeypatch):
    import socket
    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: pytest.fail("runtime must not use network"))
    search = semantic.LocalSemanticRetriever()
    background = "The bakery prepares sourdough bread by fermenting flour and water before baking. " * 17
    target = "Photovoltaic investment increases renewable electricity production and can reduce dependence on imported fossil fuels. "
    text = background + target * 5
    result = search.semantic_scores("可再生能源投资如何降低进口依赖？", [row(text, "long")], session_id="real-long-window")
    assert result["trace"]["status"] == "ready", result
    assert "long" in result["scores"], result
    start, end = result["spans"]["long"]
    assert 0 < start < end <= len(text)
    assert start > len(text) // 2
    assert "Photovoltaic investment" in text[start:end]
    assert end > len(background)
    cached = search.semantic_scores("可再生能源投资如何降低进口依赖？", [row(text, "long")], session_id="real-long-window")
    assert cached["spans"] == result["spans"]
    assert cached["trace"]["cached_chunks"] == 1
