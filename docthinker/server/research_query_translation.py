"""Optional bounded Chinese-to-English query translation; never downloads.

Translation is a retrieval hint, not an answer or a semantic-preservation proof.
Install assets explicitly with scripts/setup_research_translation.py. Runtime
needs only CTranslate2 and SentencePiece, not torch, transformers, or a provider.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any

MODEL_NAME = "Helsinki-NLP/opus-mt-zh-en"
MODEL_REVISION = "cf109095479db38d6df799875e34039d4938aaa6"
ASSET_FORMAT = "opus-zh-en-ct2-int8-query-v1"
INDEX_VERSION = "opus-zh-en-ct2-int8-query-v4-invariance-polarity"
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "docthinker" / "research-translation" / MODEL_REVISION
MODEL_FILES = ("model.bin", "config.json", "shared_vocabulary.json", "source.spm", "target.spm", "ATTRIBUTION.txt")
MAX_QUERY_BYTES = 8192
MAX_SOURCE_PIECES = 256
MAX_OUTPUT_TOKENS = 256
MAX_CACHE_ENTRIES = 128
MAX_PROTECTED_SPANS = 24
_INFERENCE_GATE = threading.BoundedSemaphore(1)
_CACHE: OrderedDict[tuple[str, ...], tuple[str, dict[str, Any]]] = OrderedDict()
_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_TERM = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)*")
_PROTECTED_SPAN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*(?:[ \t]+[A-Za-z][A-Za-z0-9_-]*)*|\d+(?:[.,]\d+)*")
_PLACEHOLDER = re.compile(r"(?<![A-Za-z0-9_])ZXQ\d+(?![A-Za-z0-9_])")
_NEG_ZH = re.compile(r"不能|不得|不要|没有|未曾|未能|并非|而非|无须|无需|从未|禁止|不(?:使用|允许|依赖|增加|减少|改变|包含|采用|支持|证明|等于|代表|是|会|应|能)")
_NEG_EN = re.compile(r"\b(?:no|not|never|without|neither|nor|cannot|unable|lack\w*|fail\w*|exclud\w*|forbid\w*)\b|\b\w+n['’]t\b|rather than|instead of", re.I)
# Keep an explicit invariance constraint from becoming a length/duration
# request. This is a conservative operator check, not subject/scope entailment.
# Do not treat noun/shape forms (不变量/不变性/不变形) as this predicate.
_INVARIANCE_ZH = re.compile(r"(?:保持|维持)[^。！？；，,\n]{0,24}不变(?![量性形])")
_INVARIANCE_EN = re.compile(
    r"\b(?:unchanged|unaltered|invariant|constant|fixed)\b|"
    r"\b(?:remain\w*|stay\w*|keep|kept|maintain\w*)\b[^.;!?\n]{0,60}\b(?:the\s+)?same\b|"
    r"\b(?:not|without)\b[^.;!?\n]{0,40}\b(?:chang(?:e|es|ed|ing)|alter(?:s|ed|ing)?)\b", re.I)
# Reject explicit reversals of those surface predicates, even if a different
# clause contains another positive invariance word. This does not bind subjects
# or prove that all constraints survived. Keep bridges bounded and do not carry
# a negation through a coordinating clause ("do not use X, keep Y unchanged").
_INVARIANCE_STATE_EN = r"(?:unchanged|unaltered|invariant|constant|fixed|(?:the\s+)?same)\b"
_INVARIANCE_LOCAL_BRIDGE_EN = r"(?:(?!\b(?:and|or|but|while|whereas|although|because|then)\b)[^,.;!?\n]){0,60}"
_REVERSED_INVARIANCE_EN = re.compile(
    r"\b(?:not|never|no\s+longer|cannot|\w+n['’]t)\s+"
    r"(?:(?:\w+ly|be|been|being)\s+){0,3}" + _INVARIANCE_STATE_EN + r"|"
    r"\b(?:not|never|no\s+longer|cannot|\w+n['’]t|"
    r"stop(?:s|ped|ping)?|ceas(?:e|es|ed|ing)|avoid(?:s|ed|ing)?|fail(?:s|ed|ing)?)\s+"
    r"(?:(?:to|be|been|being|\w+ly)\s+){0,3}"
    r"(?:keep(?:s|ing)?|kept|maintain\w*|remain\w*|stay\w*|leav(?:e|es|ing)|left|mak(?:e|es|ing)|made)\b"
    + _INVARIANCE_LOCAL_BRIDGE_EN + r"\b" + _INVARIANCE_STATE_EN + r"|"
    r"\b(?:not|never)\s+without\s+(?:changing|altering)\b", re.I)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_path(path: Path) -> bool:
    return not any(item.is_symlink() for item in (path, *path.parents))


def _model_manifest(path: Path) -> dict[str, Any]:
    if not _safe_path(path):
        raise ValueError("Invalid local model directory")
    manifest_path = path / "docthinker-model.json"
    if manifest_path.is_symlink() or manifest_path.stat().st_size > 65536:
        raise ValueError("Invalid local model manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("model"), manifest.get("revision"), manifest.get("format")) != (MODEL_NAME, MODEL_REVISION, ASSET_FORMAT):
        raise ValueError("Unsupported local model")
    if manifest.get("license") != "CC-BY-4.0" or manifest.get("quantization") != "int8":
        raise ValueError("Invalid local model metadata")
    hashes = manifest.get("sha256", {})
    if set(hashes) != set(MODEL_FILES):
        raise ValueError("Incomplete local model")
    for name in MODEL_FILES:
        asset = path / name
        if asset.is_symlink() or not asset.is_file() or asset.stat().st_size > 400_000_000:
            raise ValueError("Invalid local model asset")
        if not re.fullmatch(r"[a-f0-9]{64}", str(hashes[name])) or _sha256(asset) != hashes[name]:
            raise ValueError("Local model integrity check failed")
    return manifest


def _lexical_guard(source: str, translated: str) -> list[str]:
    """Conservative surface checks only: passing is not semantic verification."""
    reasons = []
    if Counter(_NUMBER.findall(source)) - Counter(_NUMBER.findall(translated)):
        reasons.append("missing_numbers")
    # All explicitly supplied Latin tokens are protected; no paper dictionaries.
    if Counter(x.casefold() for x in _TERM.findall(source)) - Counter(x.casefold() for x in _TERM.findall(translated)):
        reasons.append("missing_terms")
    if len(_NEG_ZH.findall(source)) + len(_NEG_EN.findall(source)) > len(_NEG_EN.findall(translated)):
        reasons.append("missing_negation")
    if _INVARIANCE_ZH.search(source) and (
            not _INVARIANCE_EN.search(translated) or _REVERSED_INVARIANCE_EN.search(translated)):
        reasons.append("missing_invariance")
    return reasons


class LocalQueryTranslator:
    """Shared nonblocking inference gate and 128-entry namespace-isolated LRU.

    session_id must contain the storage/project namespace, not a display-only ID.
    Text is returned only to the caller; traces never contain it or exception data.
    """

    def __init__(self, model_path: str | Path | None = None, *, cache_entries: int = MAX_CACHE_ENTRIES,
                 asset_dir: str | Path | None = None):
        if type(cache_entries) is not int or not 1 <= cache_entries <= MAX_CACHE_ENTRIES:
            raise ValueError("Invalid translation cache size")
        configured = model_path or asset_dir or os.getenv("DOCTHINKER_RESEARCH_TRANSLATION_DIR")
        self.model_path = Path(configured).expanduser().absolute() if configured else DEFAULT_MODEL_DIR
        self.cache_entries = cache_entries
        self._runtime = None
        self._asset_trace: dict[str, str] = {}

    def _load_runtime(self):
        manifest = _model_manifest(self.model_path)
        self._asset_trace = {
            "manifest_fingerprint": hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "model_sha256": manifest["sha256"]["model.bin"],
        }
        import ctranslate2
        import sentencepiece

        translator = ctranslate2.Translator(str(self.model_path), device="cpu", compute_type="int8",
                                            inter_threads=1, intra_threads=2)
        source = sentencepiece.SentencePieceProcessor(model_file=str(self.model_path / "source.spm"))
        target = sentencepiece.SentencePieceProcessor(model_file=str(self.model_path / "target.spm"))
        return translator, source, target

    def prepare_assets(self) -> dict[str, Any]:
        """Validate/load assets without translating or priming a query cache."""
        started = time.perf_counter()
        trace = {"status": "busy", "model": MODEL_NAME, "revision": MODEL_REVISION,
                 "asset_only": True, "query_inference": False, "provider_calls": 0,
                 "llm_tokens": 0, "model_reused": False, "initialization_ms": 0.0}

        def result(status):
            return {"trace": {**trace, "status": status,
                    "total_ms": round((time.perf_counter() - started) * 1000, 3)}}

        if not _INFERENCE_GATE.acquire(blocking=False):
            return result("busy")
        try:
            trace["model_reused"] = self._runtime is not None
            initialization = time.perf_counter()
            status = "ready"
            try:
                if self._runtime is None:
                    self._runtime = self._load_runtime()
            except Exception:
                status = "unavailable"
            finally:
                trace["initialization_ms"] = round((time.perf_counter() - initialization) * 1000, 3)
            return result(status)
        finally:
            _INFERENCE_GATE.release()

    def translate(self, question: str, session_id: str) -> dict[str, Any]:
        start = time.perf_counter()
        original = question if isinstance(question, str) else ""
        trace: dict[str, Any] = {"algorithm": INDEX_VERSION, "model": MODEL_NAME, "revision": MODEL_REVISION,
                               "provider_calls": 0, "llm_tokens": 0, "semantic_preservation_verified": False,
                               "source_limit": MAX_SOURCE_PIECES, "output_limit": MAX_OUTPUT_TOKENS,
                               "cpu_threads": 2, "beam_size": 1, "cache_hit": False}

        def result(status: str, text: str = original):
            return {"text": text, "trace": {**trace, "status": status,
                    "total_ms": round((time.perf_counter() - start) * 1000, 3)}}

        if not isinstance(question, str) or not question.strip() or not isinstance(session_id, str) or not session_id or len(session_id) > 1024:
            return result("invalid_input")
        try:
            session_id.encode("utf-8")
            if len(question.encode("utf-8")) > MAX_QUERY_BYTES:
                return result("input_limit")
        except UnicodeError:
            return result("invalid_input")
        if not _HAN.search(question):
            return result("no_chinese")
        if re.search(r"ZXQ\d+", question, re.I):
            return result("placeholder_collision")
        spans = list(_PROTECTED_SPAN.finditer(question))
        trace["protected_span_count"] = len(spans)
        if len(spans) > MAX_PROTECTED_SPANS:
            return result("placeholder_limit")
        preserved = [match.group() for match in spans]
        replacements = iter(range(len(preserved)))
        protected = _PROTECTED_SPAN.sub(lambda match: "ZXQ" + str(next(replacements)), question)
        if not _INFERENCE_GATE.acquire(blocking=False):
            return result("busy")
        try:
            try:
                initialization = time.perf_counter()
                reused = self._runtime is not None
                if self._runtime is None:
                    self._runtime = self._load_runtime()
                trace.update(self._asset_trace)
                trace.update(initialization_ms=round((time.perf_counter() - initialization) * 1000, 3), model_reused=reused)
            except Exception:
                return result("model_unavailable")
            # Artifact identity is included, so rebuilding a pinned source with
            # a different converter cannot reuse another build's translation.
            key = tuple(hashlib.sha256(value.encode()).hexdigest() for value in
                        (session_id, MODEL_NAME, MODEL_REVISION, INDEX_VERSION, str(self.model_path),
                         self._asset_trace.get("model_sha256", ""), question))
            if key in _CACHE:
                translated, details = _CACHE.pop(key)
                _CACHE[key] = translated, details
                trace.update(details, cache_hit=True)
                return result("ready", translated)
            try:
                translator, source, target = self._runtime
                pieces = source.encode(protected, out_type=str)
                trace["source_pieces"] = len(pieces) + 1
                if len(pieces) + 1 > MAX_SOURCE_PIECES:
                    return result("source_limit")
                # HF Marian input includes EOS; CT2 Transformers conversion does
                # not insert it. SentencePiece alone must append it explicitly.
                tokens = pieces + ["</s>"]
                inference = time.perf_counter()
                outputs = translator.translate_batch([tokens], beam_size=1, max_decoding_length=MAX_OUTPUT_TOKENS,
                                                     sampling_topk=1, return_scores=False)
                hypothesis = outputs[0].hypotheses[0]
                trace.update(inference_ms=round((time.perf_counter() - inference) * 1000, 3), output_tokens=len(hypothesis))
                if len(hypothesis) >= MAX_OUTPUT_TOKENS:
                    return result("output_limit")
                translated = target.decode(hypothesis).strip()
                if not translated or len(translated.encode()) > MAX_QUERY_BYTES or _HAN.search(translated):
                    return result("invalid_translation")
                expected = ["ZXQ" + str(index) for index in range(len(preserved))]
                if _PLACEHOLDER.findall(translated) != expected:
                    return result("placeholder_guard")
                # No fuzzy matching or correction: each exact placeholder must
                # occur once in original order before restoring user text.
                translated = _PLACEHOLDER.sub(lambda match: preserved[int(match.group()[3:])], translated)
                if len(translated.encode()) > MAX_QUERY_BYTES:
                    return result("invalid_translation")
                reasons = _lexical_guard(question, translated)
                if reasons:
                    trace["guard_reasons"] = reasons
                    return result("lexical_guard")
                details = {k: trace[k] for k in ("source_pieces", "output_tokens", "manifest_fingerprint", "model_sha256") if k in trace}
                _CACHE[key] = translated, details
                while len(_CACHE) > self.cache_entries:
                    _CACHE.popitem(last=False)
                return result("ready", translated)
            except Exception:
                return result("translation_error")
        finally:
            _INFERENCE_GATE.release()
