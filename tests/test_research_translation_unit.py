"""Offline contracts plus an explicit opt-in local runtime smoke (no corpus)."""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import string
import sys
import socket
import threading
from types import SimpleNamespace

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docthinker"
    / "server"
    / "research_query_translation.py"
)


@pytest.mark.skipif(os.getenv("DOCTHINKER_TEST_LOCAL_TRANSLATION") != "1",
                    reason="explicit real local translation model test")
def test_real_pinned_translation_is_offline_query_only_and_preserves_protected_spans(monkeypatch):
    from docthinker.server.research_query_translation import LocalQueryTranslator

    def deny_network(*args, **kwargs):
        pytest.fail("Query translation must never open a network connection")

    original_import = builtins.__import__

    def runtime_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "transformers", "huggingface_hub"}:
            pytest.fail("Conversion/download dependencies are not runtime dependencies")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(builtins, "__import__", runtime_import)
    question = "虚构温室中的 SolarMesh 如何控制 LED 照明和水流？比较 2 组幼苗，并解释为何不测量温度。"
    translator = LocalQueryTranslator()
    result = translator.translate(question, session_id="translation-public-smoke")
    assert result["trace"]["status"] == "ready"
    assert all(term in result["text"] for term in ("SolarMesh", "LED", "2"))
    assert result["trace"]["provider_calls"] == result["trace"]["llm_tokens"] == 0
    assert result["trace"]["semantic_preservation_verified"] is False
    assert len(result["trace"]["model_sha256"]) == 64
    cached = translator.translate(question, session_id="translation-public-smoke")
    assert cached["text"] == result["text"]
    assert cached["trace"]["cache_hit"] is True
    # This checks runtime isolation and literal spans, not semantic fidelity.


@pytest.fixture
def translation_module(monkeypatch):
    """Isolate module-level cache and gate without importing the application."""
    name = "_research_translation_unit_subject"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


class FakeProcessor:
    def __init__(self, pieces=None):
        self.pieces = pieces
        self.encoded = []
        self.decoded = []

    def encode(self, text, *, out_type):
        self.encoded.append((text, out_type))
        return list(self.pieces) if self.pieces is not None else [text]

    def decode(self, pieces):
        self.decoded.append(list(pieces))
        return " ".join(pieces)


class FakeTranslator:
    def __init__(self, output="Find supporting evidence", error=None):
        self.output = output
        self.error = error
        self.calls = []

    def translate_batch(self, batch, **kwargs):
        self.calls.append((batch, kwargs))
        if self.error is not None:
            raise self.error
        return [SimpleNamespace(hypotheses=[self.output.split()])]


@pytest.fixture
def make_translator(translation_module, monkeypatch, tmp_path):
    def make(*, output="Find supporting evidence", pieces=None, backend=None):
        instance = translation_module.LocalQueryTranslator(
            model_path=tmp_path / "offline-model"
        )
        backend = backend or FakeTranslator(output)
        source = FakeProcessor(pieces)
        target = FakeProcessor()
        monkeypatch.setattr(instance, "_load_runtime", lambda: (backend, source, target))
        return instance, backend, source, target

    return make


@pytest.fixture
def valid_model_assets(translation_module, tmp_path):
    """Build tiny synthetic assets; never read a downloaded model or test corpus."""
    # macOS can expose a symlinked /var temporary-directory prefix. Resolve the
    # fixture root so tests exercise only the deliberate symlinks they create.
    directory = tmp_path.resolve() / "synthetic-local-model"
    directory.mkdir()
    hashes = {}
    for filename in translation_module.MODEL_FILES:
        content = ("synthetic test asset: " + filename).encode("utf-8")
        (directory / filename).write_bytes(content)
        hashes[filename] = hashlib.sha256(content).hexdigest()
    manifest = {
        "model": translation_module.MODEL_NAME,
        "revision": translation_module.MODEL_REVISION,
        "format": translation_module.ASSET_FORMAT,
        "license": "CC-BY-4.0",
        "quantization": "int8",
        "sha256": hashes,
    }
    (directory / "docthinker-model.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory, manifest


def assert_result(result, *, status, text):
    assert result["text"] == text
    trace = result["trace"]
    assert trace["status"] == status
    assert trace["provider_calls"] == 0
    assert isinstance(trace["model"], str) and trace["model"]
    assert isinstance(trace["revision"], str) and trace["revision"]


def test_import_does_not_import_optional_inference_dependencies(monkeypatch):
    original_import = builtins.__import__
    forbidden = {"ctranslate2", "sentencepiece", "torch", "transformers", "huggingface_hub"}

    def guarded_import(name, *args, **kwargs):
        assert name.split(".", 1)[0] not in forbidden, name
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    spec = importlib.util.spec_from_file_location("_translation_lazy_import", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)


def test_english_query_skips_runtime_loading(translation_module, monkeypatch, tmp_path):
    instance = translation_module.LocalQueryTranslator(model_path=tmp_path)

    def forbidden_load():
        pytest.fail("English queries must not load the local model")

    monkeypatch.setattr(instance, "_load_runtime", forbidden_load)
    query = "Find evidence excluding cached results"
    assert_result(instance.translate(query, "english-session"), status="no_chinese", text=query)


def test_unencodable_session_id_falls_back_before_loading(make_translator):
    instance, backend, source, _ = make_translator()
    query = "请查找研究证据"
    assert_result(instance.translate(query, "session-\ud800"), status="invalid_input", text=query)
    assert not backend.calls
    assert not source.encoded


def test_translation_uses_one_deterministic_local_batch(make_translator):
    instance, backend, source, target = make_translator()
    query = "请查找支持这一结论的证据"
    result = instance.translate(query, "session")
    assert_result(result, status="ready", text="Find supporting evidence")
    assert source.encoded == [(query, str)]
    assert len(backend.calls) == 1
    batch, options = backend.calls[0]
    assert len(batch) == 1
    assert options["beam_size"] == 1
    assert options["max_decoding_length"] == 256
    assert target.decoded == [["Find", "supporting", "evidence"]]


def test_utf8_input_limit_is_bytes_not_characters(make_translator):
    instance, backend, source, _ = make_translator()
    query = "中" * 2731
    assert len(query) < 8192 < len(query.encode("utf-8"))
    assert_result(instance.translate(query, "session"), status="input_limit", text=query)
    assert not source.encoded
    assert not backend.calls


def test_input_at_8192_bytes_is_not_rejected(make_translator):
    instance, backend, _, _ = make_translator(output="ZXQ0")
    query = "中" * 2730 + "ab"
    assert len(query.encode("utf-8")) == 8192
    assert_result(instance.translate(query, "session"), status="ready", text="ab")
    assert len(backend.calls) == 1


@pytest.mark.parametrize("piece_count", [256, 257])
def test_source_piece_limit_accounts_for_end_token(make_translator, piece_count):
    instance, backend, _, _ = make_translator(pieces=["piece"] * piece_count)
    query = "请查找证据"
    assert_result(instance.translate(query, "session"), status="source_limit", text=query)
    assert not backend.calls


def test_255_source_pieces_are_accepted(make_translator):
    instance, backend, _, _ = make_translator(pieces=["piece"] * 255)
    assert_result(
        instance.translate("请查找证据", "session"),
        status="ready",
        text="Find supporting evidence",
    )
    assert len(backend.calls[0][0][0]) == 256


@pytest.mark.parametrize("token_count", [256, 257])
def test_output_hitting_or_exceeding_token_limit_falls_back(make_translator, token_count):
    instance, _, _, target = make_translator(output=" ".join(["evidence"] * token_count))
    query = "请查找证据"
    assert_result(instance.translate(query, "session"), status="output_limit", text=query)
    assert not target.decoded


@pytest.mark.parametrize("output", ["", "仍是中文", "x" * 8193])
def test_invalid_translation_falls_back_to_original_query(make_translator, output):
    instance, _, _, _ = make_translator(output=output)
    query = "请查找证据"
    assert_result(instance.translate(query, "session"), status="invalid_translation", text=query)


def test_runtime_is_loaded_once_for_distinct_queries(make_translator, monkeypatch):
    instance, backend, source, target = make_translator()
    loads = []

    def load_runtime():
        loads.append(True)
        return backend, source, target

    monkeypatch.setattr(instance, "_load_runtime", load_runtime)
    assert instance.translate("查询第一份资料", "session")["trace"]["status"] == "ready"
    assert instance.translate("查询第二份资料", "session")["trace"]["status"] == "ready"
    assert len(loads) == 1
    assert len(backend.calls) == 2


def test_successful_translation_cache_is_session_namespaced(make_translator):
    instance, backend, _, _ = make_translator()
    query = "请查找支持性证据"
    first = instance.translate(query, "private-session-alpha")
    repeated = instance.translate(query, "private-session-alpha")
    other_session = instance.translate(query, "private-session-beta")
    assert first["text"] == repeated["text"] == other_session["text"]
    assert len(backend.calls) == 2


def test_cache_is_separated_by_asset_directory(make_translator, tmp_path):
    first, first_backend, _, _ = make_translator()
    second, second_backend, _, _ = make_translator()
    second.model_path = tmp_path / "a-different-model-directory"
    query = "查找原始研究证据"
    first.translate(query, "same-session")
    second.translate(query, "same-session")
    assert len(first_backend.calls) == len(second_backend.calls) == 1


def test_cache_does_not_expose_raw_namespace_or_query(translation_module, make_translator):
    instance, _, _, _ = make_translator()
    query = "仅用于测试的私密问题标记"
    namespace = "private-namespace-sentinel"
    result = instance.translate(query, namespace)
    cache_repr = repr(translation_module._CACHE)
    trace_repr = json.dumps(result["trace"], ensure_ascii=False)
    for value in (query, namespace):
        assert value not in cache_repr
        assert value not in trace_repr


def test_default_lru_cache_evicts_oldest_entry_at_128(make_translator, translation_module):
    instance, backend, _, _ = make_translator()
    queries = ["查找证据" + chr(0x4E00 + index) for index in range(129)]
    for query in queries:
        assert instance.translate(query, "session")["trace"]["status"] == "ready"
    assert len(translation_module._CACHE) == 128
    assert len(backend.calls) == 129
    instance.translate(queries[-1], "session")
    assert len(backend.calls) == 129
    instance.translate(queries[0], "session")
    assert len(backend.calls) == 130
    assert len(translation_module._CACHE) == 128


def test_cache_hit_refreshes_lru_recency(make_translator):
    instance, backend, _, _ = make_translator()
    queries = ["查找资料" + chr(0x4E00 + index) for index in range(129)]
    for query in queries[:128]:
        instance.translate(query, "session")
    instance.translate(queries[0], "session")
    instance.translate(queries[128], "session")
    assert len(backend.calls) == 129
    instance.translate(queries[0], "session")
    assert len(backend.calls) == 129
    instance.translate(queries[1], "session")
    assert len(backend.calls) == 130


def test_global_gate_rejects_concurrent_inference_without_waiting(make_translator):
    started = threading.Event()
    release = threading.Event()
    second_done = threading.Event()
    results = {}
    errors = []

    class BlockingTranslator(FakeTranslator):
        def translate_batch(self, batch, **kwargs):
            started.set()
            if not release.wait(3):
                raise RuntimeError("test release timeout")
            return super().translate_batch(batch, **kwargs)

    first, _, _, _ = make_translator(backend=BlockingTranslator())
    second, second_backend, _, _ = make_translator()

    def run_first():
        try:
            results["first"] = first.translate("第一条中文查询", "first-session")
        except BaseException as exc:
            errors.append(exc)

    def run_second():
        try:
            results["second"] = second.translate("第二条中文查询", "second-session")
        except BaseException as exc:
            errors.append(exc)
        finally:
            second_done.set()

    first_thread = threading.Thread(target=run_first, daemon=True)
    second_thread = threading.Thread(target=run_second, daemon=True)
    first_thread.start()
    try:
        assert started.wait(1), "first inference never reached the fake runtime"
        second_thread.start()
        assert second_done.wait(0.5), "concurrent inference must fall back, not queue"
    finally:
        release.set()
        first_thread.join(2)
        if second_thread.ident is not None:
            second_thread.join(2)
    assert not errors
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert_result(results["second"], status="busy", text="第二条中文查询")
    assert results["first"]["trace"]["status"] == "ready"
    assert not second_backend.calls


@pytest.mark.parametrize(
    ("query", "output"),
    [
        ("不要使用缓存", "Use the cache"),
        ("没有对应证据", "There is supporting evidence"),
    ],
)
def test_lexical_guard_falls_back_when_constraints_disappear(make_translator, query, output):
    instance, _, _, _ = make_translator(output=output)
    result = instance.translate(query, "session")
    assert_result(result, status="lexical_guard", text=query)
    assert result["trace"]["guard_reasons"]
    trace = json.dumps(result["trace"], ensure_ascii=False)
    assert query not in trace
    assert output not in trace


def test_lexical_guard_accepts_preserved_constraints(make_translator):
    output = "Do not use BM25 with 42 results"
    instance, backend, _, _ = make_translator(output="Do not use ZXQ0 with ZXQ1 results")
    assert_result(
        instance.translate("不要使用 BM25 和 42 个结果", "session"),
        status="ready",
        text=output,
    )
    assert len(backend.calls) == 1


@pytest.mark.parametrize("output", [
    "How is the answer kept as long as possible?",
    "How is the temperature increased?",
    "How does the output change?",
])
def test_explicit_invariance_cannot_become_length_or_change(make_translator, output):
    query = "如何保持结果不变？"
    instance, backend, _, _ = make_translator(output=output)
    result = instance.translate(query, "invariance-example")
    assert_result(result, status="lexical_guard", text=query)
    assert "missing_invariance" in result["trace"]["guard_reasons"]
    assert len(backend.calls) == 1
    assert not instance.translate(query, "invariance-example")["trace"]["cache_hit"]


@pytest.mark.parametrize("output", [
    "How is the result kept unchanged?",
    "How is the result kept unaltered?",
    "How can the result be kept fixed?",
    "How can the result remain the same?",
    "How do we keep the result constant?",
    "How can we proceed without changing the result?",
    "How can we proceed without altering the result?",
    "How can we proceed and not alter the result?",
    "Do not use the cache, and keep the result unchanged.",
    "Do not keep the cache while keeping the result unchanged.",
])
def test_explicit_invariance_can_be_preserved_without_new_provider_calls(make_translator, output):
    instance, backend, _, _ = make_translator(output=output)
    query = "如何维持结果不变？"
    assert_result(instance.translate(query, "invariance-example"), status="ready", text=output)
    assert len(backend.calls) == 1
    assert instance.translate(query, "invariance-example")["trace"]["cache_hit"]


@pytest.mark.parametrize("output", [
    "The result is not unchanged.",
    "The result is no longer constant.",
    "The result is not necessarily fixed.",
    "Do not keep the result unchanged.",
    "The result must not be kept unaltered.",
    "The result cannot remain the same.",
    "Don't keep the result fixed.",
    "The result shouldn't stay constant.",
    "Stop keeping the result the same.",
    "Cease to maintain the result unchanged.",
    "Avoid keeping the result fixed.",
    "Fail to keep the result unchanged.",
    "Keep the sample constant but do not keep the result unchanged.",
    "The result is no longer constant while the sample remains unchanged.",
    "Proceed not without altering the result.",
])
def test_explicit_invariance_reversal_falls_back_without_caching(make_translator, output):
    query = "如何保持结果不变？"
    instance, backend, _, _ = make_translator(output=output)
    result = instance.translate(query, "invariance-polarity")
    assert_result(result, status="lexical_guard", text=query)
    assert "missing_invariance" in result["trace"]["guard_reasons"]
    assert result["trace"]["semantic_preservation_verified"] is False
    again = instance.translate(query, "invariance-polarity")
    assert_result(again, status="lexical_guard", text=query)
    assert again["trace"]["cache_hit"] is False
    assert len(backend.calls) == 2
    assert result["trace"]["provider_calls"] == result["trace"]["llm_tokens"] == 0


def test_invariance_surface_guard_does_not_claim_subject_binding(translation_module):
    # A different subject can still supply an invariance word. The guard is
    # deliberately lexical, not a parser or a semantic-preservation guarantee.
    assert translation_module._lexical_guard(
        "如何保持结果不变？",
        "Change the result while keeping processing time constant.",
    ) == []


@pytest.mark.parametrize("source,translated", [
    ("保持材料不变形", "Keep the material from deforming"),
    ("保持旋转不变性", "Preserve rotational invariance"),
    ("保持不变量", "Preserve the invariant quantity"),
    ("先保持记录，下一步说明不变的量", "Keep the record, then identify invariant quantities"),
])
def test_invariance_guard_does_not_capture_other_forms_or_cross_clauses(translation_module, source, translated):
    assert "missing_invariance" not in translation_module._lexical_guard(source, translated)


@pytest.mark.parametrize(
    ("query", "output"),
    [
        ("请保留 42 个结果", "Keep the results"),
        ("请解释 BM25 的作用", "Explain the ranking method"),
    ],
)
def test_missing_protected_number_or_term_falls_back(make_translator, query, output):
    instance, backend, _, _ = make_translator(output=output)
    assert_result(instance.translate(query, "session"), status="placeholder_guard", text=query)
    assert len(backend.calls) == 1


def test_generic_random_latin_phrase_and_number_are_restored_verbatim(make_translator):
    rng = random.Random(20261010)
    phrase = " ".join("".join(rng.choices(string.ascii_letters, k=11)) for _ in range(2))
    number = str(rng.randrange(10000, 99999))
    query = f"比较 {phrase} 与 {number} 的相关证据"
    instance, backend, source, _ = make_translator(output="Compare ZXQ0 with ZXQ1 evidence")
    result = instance.translate(query, "session")
    assert_result(result, status="ready", text=f"Compare {phrase} with {number} evidence")
    assert result["trace"]["protected_span_count"] == 2
    assert source.encoded == [("比较 ZXQ0 与 ZXQ1 的相关证据", str)]
    assert len(backend.calls) == 1
    batch, _ = backend.calls[0]
    assert phrase not in repr(batch)
    assert number not in repr(batch)


@pytest.mark.parametrize(
    "output",
    [
        "Compare ZXQ0 evidence",
        "Compare ZXQ0 ZXQ0 with ZXQ1 evidence",
        "Compare ZXQ1 with ZXQ0 evidence",
        "Compare zxq0 with ZXQ1 evidence",
    ],
)
def test_missing_duplicate_or_reordered_placeholders_fall_back(make_translator, output):
    query = "比较 RandomTerm 和 173 的证据"
    instance, backend, _, _ = make_translator(output=output)
    assert_result(instance.translate(query, "session"), status="placeholder_guard", text=query)
    assert len(backend.calls) == 1


def test_placeholder_restoration_does_not_relax_mixed_query_negation_guard(make_translator):
    query = "不要使用 RandomTerm 和 173 个结果"
    instance, backend, _, _ = make_translator(output="Use ZXQ0 with ZXQ1 results")
    result = instance.translate(query, "session")
    assert_result(result, status="lexical_guard", text=query)
    assert "missing_negation" in result["trace"]["guard_reasons"]
    assert result["trace"]["protected_span_count"] == 2
    assert len(backend.calls) == 1


def test_more_than_24_protected_spans_falls_back_without_inference(make_translator):
    query = "比较 " + "、".join(f"Term{index}" for index in range(25))
    instance, backend, _, _ = make_translator()
    assert_result(instance.translate(query, "session"), status="placeholder_limit", text=query)
    assert not backend.calls


def test_exactly_24_protected_spans_are_restored_in_order(make_translator):
    terms = [f"Term{index}" for index in range(24)]
    query = "比较 " + "、".join(terms)
    instance, backend, _, _ = make_translator(
        output="Compare " + " ".join(f"ZXQ{index}" for index in range(24))
    )
    result = instance.translate(query, "session")
    assert_result(result, status="ready", text="Compare " + " ".join(terms))
    assert result["trace"]["protected_span_count"] == 24
    assert len(backend.calls) == 1


@pytest.mark.parametrize("placeholder", ["ZXQ7", "zxq7"])
def test_user_supplied_placeholder_is_rejected_before_inference(make_translator, placeholder):
    query = f"请解释 {placeholder} 的含义"
    instance, backend, _, _ = make_translator()
    assert_result(instance.translate(query, "session"), status="placeholder_collision", text=query)
    assert not backend.calls


@pytest.mark.parametrize("failure_stage", ["load", "translate"])
def test_exception_trace_does_not_include_content_or_local_paths(
    make_translator, monkeypatch, tmp_path, failure_stage
):
    query = "私密异常查询标记"
    path = str(tmp_path / "private-model-name" / "credentials.bin")
    secret = "sensitive-error-detail"
    error = RuntimeError(f"{query}: {path}: {secret}")
    backend = FakeTranslator(error=error if failure_stage == "translate" else None)
    instance, _, _, _ = make_translator(backend=backend)
    if failure_stage == "load":
        def fail_load():
            raise error

        monkeypatch.setattr(instance, "_load_runtime", fail_load)
    result = instance.translate(query, "private-session")
    expected_status = "model_unavailable" if failure_stage == "load" else "translation_error"
    assert_result(result, status=expected_status, text=query)
    trace = json.dumps(result["trace"], ensure_ascii=False)
    for forbidden in (query, path, "private-model-name", "credentials.bin", secret):
        assert forbidden not in trace


def test_inference_gate_is_released_after_runtime_error(make_translator):
    failed, _, _, _ = make_translator(backend=FakeTranslator(error=RuntimeError("failure")))
    healthy, backend, _, _ = make_translator()
    assert failed.translate("第一条失败查询", "one")["trace"]["status"] == "translation_error"
    assert healthy.translate("第二条正常查询", "two")["trace"]["status"] == "ready"
    assert len(backend.calls) == 1


def test_manifest_accepts_complete_pinned_and_hash_verified_assets(
    translation_module, valid_model_assets
):
    directory, manifest = valid_model_assets
    assert translation_module._model_manifest(directory) == manifest


def test_manifest_rejects_modified_asset_bytes(translation_module, valid_model_assets):
    directory, _ = valid_model_assets
    (directory / "model.bin").write_bytes(b"changed after manifest creation")
    with pytest.raises(ValueError, match="integrity"):
        translation_module._model_manifest(directory)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("model", "another/model"),
        ("revision", "0" * 40),
        ("format", "unsupported-format"),
        ("license", "Apache-2.0"),
        ("quantization", "float32"),
    ],
)
def test_manifest_rejects_unpinned_or_wrong_metadata(
    translation_module, valid_model_assets, field, invalid_value
):
    directory, manifest = valid_model_assets
    manifest[field] = invalid_value
    (directory / "docthinker-model.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        translation_module._model_manifest(directory)


def test_manifest_rejects_missing_asset(translation_module, valid_model_assets):
    directory, _ = valid_model_assets
    (directory / "target.spm").unlink()
    with pytest.raises(ValueError, match="asset"):
        translation_module._model_manifest(directory)


def test_manifest_rejects_missing_hash_entry(translation_module, valid_model_assets):
    directory, manifest = valid_model_assets
    del manifest["sha256"]["ATTRIBUTION.txt"]
    (directory / "docthinker-model.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete"):
        translation_module._model_manifest(directory)


def test_manifest_rejects_symlinked_asset_even_when_hash_matches(
    translation_module, valid_model_assets
):
    directory, _ = valid_model_assets
    asset = directory / "source.spm"
    external = directory.parent / "external-source.spm"
    asset.rename(external)
    asset.symlink_to(external)
    with pytest.raises(ValueError, match="asset"):
        translation_module._model_manifest(directory)


def test_manifest_rejects_symlinked_model_directory(translation_module, valid_model_assets):
    directory, _ = valid_model_assets
    alias = directory.parent / "symlinked-model"
    alias.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match="directory"):
        translation_module._model_manifest(alias)


def test_manifest_rejects_symlinked_manifest(translation_module, valid_model_assets):
    directory, _ = valid_model_assets
    manifest_path = directory / "docthinker-model.json"
    external = directory.parent / "external-manifest.json"
    manifest_path.rename(external)
    manifest_path.symlink_to(external)
    with pytest.raises(ValueError, match="manifest"):
        translation_module._model_manifest(directory)
