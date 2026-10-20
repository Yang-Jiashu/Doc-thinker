"""Optional local multilingual retrieval; never downloads or calls a provider.

Run ``scripts/setup_research_embeddings.py`` once to install the pinned model.
The model finds potentially useful source passages; its cosine scores are not
probabilities, entailment checks, or proof that the corpus answers a question.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any

MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_FILE = "onnx/model.onnx"
MODEL_SHA256 = "ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665"
MODEL_DIMENSION = 384
MODEL_FILES = (MODEL_FILE, "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
INDEX_VERSION = "e5-local-window-v2"
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "docthinker" / "research-embeddings" / MODEL_REVISION
DEFAULT_MIN_SCORE = 0.78
MAX_CHUNKS = 4096
MAX_CHUNK_CHARS = 8000
MAX_WINDOWS_PER_CHUNK = 24
PASSAGE_WINDOW_TOKENS = 256
PASSAGE_WINDOW_STRIDE = 192
QUERY_MAX_TOKENS = 448
# A complete bounded 4096-chunk corpus must not evict half of itself while
# preparing its first query. The extra 128 entries leave room for query vectors.
# This is a derived feature cache, not a larger retrieval/candidate window.
MAX_CACHE_ENTRIES = MAX_CHUNKS + 128
LEGACY_CACHE_ENTRIES = 2048
MAX_CACHE_BYTES = 100 * 1024 * 1024
MAX_TOTAL_WINDOWS = 8192
_LOCK = threading.RLock()
_INFERENCE_GATE = threading.BoundedSemaphore(1)
_MODELS: dict[str, tuple[tuple[Any, Any], str]] = {}
# Source selection formerly depended on which of 2048 vectors survived the
# mixed query/passage LRU: cache hits do not consume MAX_TOTAL_WINDOWS. Preserve
# that logical candidate contract separately from physically reusing vectors.
# Only hashed identities are retained here, never an additional vector copy.
class _IdentityLRU(OrderedDict):
    def __init__(self):
        super().__init__()
        self.retained_bytes = 0

    @staticmethod
    def _size(key):
        return sum(len(part.encode()) for part in key) + 512

    def __setitem__(self, key, value):
        if key not in self:
            self.retained_bytes += self._size(key)
        super().__setitem__(key, value)

    def popitem(self, last=True):
        key, value = super().popitem(last=last)
        self.retained_bytes -= self._size(key)
        return key, value

    def clear(self):
        super().clear()
        self.retained_bytes = 0


_LOGICAL_VECTORS = _IdentityLRU()

class _VectorCache(OrderedDict):
    """LRU with constant-time retained-feature accounting, not an RSS limit.

    Bytes include arrays and a conservative allowance for keys/window metadata.
    The model, tokenizers, current request and in-flight encoding matrix remain
    outside this cache budget. No source or question text is retained as a key.
    """

    def __init__(self):
        super().__init__()
        self.retained_bytes = 0

    @staticmethod
    def _size(key, value):
        vector = value[0] if isinstance(value, tuple) else value
        windows = len(value[2]) if isinstance(value, tuple) else 0
        return int(vector.nbytes) + sum(len(part.encode()) for part in key) + 512 + 128 * windows

    def __setitem__(self, key, value):
        if key in self:
            self.retained_bytes -= self._size(key, self[key])
        super().__setitem__(key, value)
        self.retained_bytes += self._size(key, value)

    def popitem(self, last=True):
        # Python 3.9/3.10 OrderedDict.popitem can dispatch to our __delitem__.
        # Subtracting here as well double-charges eviction on those versions,
        # producing a negative counter and letting the byte cap be exceeded.
        # Route every explicit removal through our own pop/del exactly once.
        if not self:
            raise KeyError("dictionary is empty")
        key = next(reversed(self)) if last else next(iter(self))
        return key, self.pop(key)

    def __delitem__(self, key):
        self.retained_bytes -= self._size(key, self[key])
        super().__delitem__(key)

    def pop(self, key):
        value = self[key]
        del self[key]
        return value

    def clear(self):
        super().clear()
        self.retained_bytes = 0
        _LOGICAL_VECTORS.clear()

    def trim(self, entries, byte_budget):
        def within_budget():
            return len(self) <= entries and self.retained_bytes + _LOGICAL_VECTORS.retained_bytes <= byte_budget
        if within_budget():
            return
        # The old logical working set has first claim on storage. One linear
        # pass avoids repeatedly scanning protected keys for every eviction.
        for key in tuple(self):
            if within_budget():
                return
            if key not in _LOGICAL_VECTORS:
                self.pop(key)
        # Defensive hard bound even after a deliberately tiny/adversarial byte
        # budget. A logical hit whose physical vector is missing is re-encoded,
        # not dropped or charged against the legacy pending-window budget.
        while self and not within_budget():
            self.popitem(last=False)


_VECTORS = _VectorCache()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_manifest(path: Path) -> tuple[dict[str, Any], str]:
    """Reject incomplete or changed assets before an inference library loads."""
    manifest = json.loads((path / "docthinker-model.json").read_text())
    if manifest.get("model") != MODEL_NAME or manifest.get("revision") != MODEL_REVISION:
        raise ValueError("Unsupported local embedding model")
    hashes = manifest.get("sha256", {})
    if set(hashes) != set(MODEL_FILES) or hashes.get(MODEL_FILE) != MODEL_SHA256:
        raise ValueError("Incomplete local embedding model manifest")
    for name in MODEL_FILES:
        if not (path / name).is_file() or _sha256(path / name) != hashes[name]:
            raise ValueError("Local embedding model files do not match the manifest")
    fingerprint = hashlib.sha256(json.dumps([MODEL_REVISION, INDEX_VERSION, hashes], sort_keys=True).encode()).hexdigest()
    return manifest, fingerprint


def _load_model(path: Path):
    # Specific local path plus local_files_only prevents HF/GCS network fallback.
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType
    from tokenizers import Tokenizer

    custom_name = "docthinker/multilingual-e5-small-pinned"
    if custom_name not in {item["model"] for item in TextEmbedding.list_supported_models()}:
        TextEmbedding.add_custom_model(model=custom_name, pooling=PoolingType.MEAN,
                                       normalization=True, sources=ModelSource(hf=MODEL_NAME),
                                       dim=MODEL_DIMENSION, model_file=MODEL_FILE)
    model = TextEmbedding(model_name=custom_name, specific_model_path=str(path),
                          local_files_only=True, threads=2, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    tokenizer.no_truncation()
    return model, tokenizer


def _windows(text: str, tokenizer) -> tuple[list[tuple[int, int]], bool]:
    scanned = text[:MAX_CHUNK_CHARS]
    encoding = tokenizer.encode(scanned, add_special_tokens=False)
    offsets = encoding.offsets
    if not offsets:
        return [], len(scanned) < len(text)
    windows = []
    for left in range(0, len(offsets), PASSAGE_WINDOW_STRIDE):
        right = min(left + PASSAGE_WINDOW_TOKENS, len(offsets))
        start, end = offsets[left][0], offsets[right - 1][1]
        if end > start:
            windows.append((start, end))
        if right == len(offsets) or len(windows) == MAX_WINDOWS_PER_CHUNK:
            break
    limited = len(scanned) < len(text) or (windows and windows[-1][1] < len(scanned.rstrip()))
    return windows, bool(limited)


def _normalized(vectors, expected: int):
    import numpy as np

    matrix = np.asarray(list(vectors), dtype=np.float32)
    if matrix.shape != (expected, MODEL_DIMENSION) or not np.isfinite(matrix).all():
        raise ValueError("Invalid local embedding vectors")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Invalid zero local embedding vector")
    return matrix / norms


def _passage_batch_identity(retriever, rows, pending, texts):
    """No plaintext persists; bind the exact original batch and transforms.

    Reusing individual windows from a different ONNX batch could change low
    floating bits. Cache only this complete ordered post-admission batch.
    """
    import importlib.metadata
    import marshal
    import numpy as np
    import sys

    versions = {}
    for name in ('fastembed', 'onnxruntime'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    manifest = retriever.model_path / 'docthinker-model.json'
    if manifest.is_symlink():
        raise ValueError('Passage cache requires a regular model manifest')
    return {'version': 'exact-normalized-passage-batch-v1',
        'model_fingerprint': retriever.fingerprint,
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'runtime': versions, 'numpy': np.__version__, 'byteorder': sys.byteorder,
        'encoder_type': type(retriever._model).__module__ + '.' + type(retriever._model).__qualname__,
        'normalization': hashlib.sha256(marshal.dumps(_normalized.__code__)).hexdigest(),
        'transform': {'index': INDEX_VERSION, 'prefix': 'passage: ', 'dtype': 'float32',
            'dimension': MODEL_DIMENSION, 'batch_size': 16, 'window_tokens': PASSAGE_WINDOW_TOKENS,
            'window_stride': PASSAGE_WINDOW_STRIDE, 'max_chars': MAX_CHUNK_CHARS,
            'max_windows_per_chunk': MAX_WINDOWS_PER_CHUNK, 'max_total_windows': MAX_TOTAL_WINDOWS},
        'ordered_live_rows_sha256': hashlib.sha256(json.dumps(rows, ensure_ascii=False,
            sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
        'ordered_admitted_windows': [
            [hashlib.sha256(key[-1].encode()).hexdigest(), left, right, cut, windows]
            for _, key, left, right, cut, windows, _, _ in pending],
        'ordered_encoder_text_hashes': [hashlib.sha256(text.encode()).hexdigest() for text in texts]}


class LocalSemanticRetriever:
    """Bounded process cache, scoped to the caller's storage/project namespace.

    ``session_id`` must include the storage namespace when separate tenants can
    use the same display project number. Only supplied live chunks can be returned.
    """

    def __init__(self, model_path: str | Path | None = None, *, min_score: float = DEFAULT_MIN_SCORE,
                 cache_entries: int = MAX_CACHE_ENTRIES, private_cache_root: str | Path | None = None):
        if isinstance(min_score, bool) or not isinstance(min_score, (int, float)) or not math.isfinite(min_score) or not 0 <= min_score <= 1:
            raise ValueError("Invalid local semantic threshold")
        if type(cache_entries) is not int or not 1 <= cache_entries <= MAX_CACHE_ENTRIES:
            raise ValueError("Invalid local semantic cache size")
        configured = model_path or os.getenv("DOCTHINKER_RESEARCH_EMBEDDING_DIR")
        self.model_path = Path(configured).expanduser().resolve() if configured else DEFAULT_MODEL_DIR
        self.min_score = float(min_score)
        self.cache_entries = cache_entries
        self.cache_bytes = MAX_CACHE_BYTES
        self._model = None
        self._tokenizer = None
        self.fingerprint = None
        # Opt-in workspace-owned storage only; no global/home cache or query
        # persistence. A missing root retains the original in-process path.
        self._private_cache_root = private_cache_root

    def _initialize(self) -> tuple[float, bool]:
        if self._model is not None:
            return 0.0, True
        start = time.perf_counter()
        with _LOCK:
            path_key = str(self.model_path)
            if path_key in _MODELS:
                (self._model, self._tokenizer), self.fingerprint = _MODELS[path_key]
                return (time.perf_counter() - start) * 1000, True
            _, self.fingerprint = _model_manifest(self.model_path)
            self._model, self._tokenizer = _load_model(self.model_path)
            # Two immutable model copies is already ~1 GB; never grow without limit.
            if len(_MODELS) >= 2:
                _MODELS.pop(next(iter(_MODELS)))
            _MODELS[path_key] = ((self._model, self._tokenizer), self.fingerprint)
        return (time.perf_counter() - start) * 1000, False

    def semantic_scores(self, question: str, chunks: Sequence[Mapping[str, Any]], *, session_id: str) -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip() or len(question.encode()) > 8192:
            raise ValueError("Invalid semantic search question")
        return self._run(question, chunks, session_id=session_id)

    def prepare_documents(self, chunks: Sequence[Mapping[str, Any]], *, session_id: str) -> dict[str, Any]:
        """Prepare only live passage vectors; never encode or cache a query.

        Call during ingestion, or outside timed answer generation in a benchmark.
        Initialization and passage encoding costs remain visible in the trace.
        """
        return self._run(None, chunks, session_id=session_id)

    def _run(self, question: str | None, chunks: Sequence[Mapping[str, Any]], *, session_id: str) -> dict[str, Any]:
        if not isinstance(session_id, str) or not session_id or len(session_id) > 1024:
            raise ValueError("A storage/project namespace is required for semantic retrieval")
        # Cancellation of asyncio.to_thread does not stop its worker. Never add
        # another waiting worker behind an active initializer or index builder.
        # The caller can answer lexically while this bounded local job finishes.
        if not _INFERENCE_GATE.acquire(blocking=False):
            return {"scores": {}, "spans": {}, "trace": {"algorithm": INDEX_VERSION, "model": MODEL_NAME,
                    "revision": MODEL_REVISION, "operation": "prepare_documents" if question is None else "semantic_scores",
                    "status": "busy", "provider_calls": 0, "llm_tokens": 0,
                    "semantic_entailment_verified": False, "total_ms": 0.0}}
        try:
            return self._run_exclusive(question, chunks, session_id=session_id)
        finally:
            _INFERENCE_GATE.release()

    def _run_exclusive(self, question: str | None, chunks: Sequence[Mapping[str, Any]], *, session_id: str) -> dict[str, Any]:
        start = time.perf_counter()
        trace: dict[str, Any] = {"algorithm": INDEX_VERSION, "model": MODEL_NAME, "revision": MODEL_REVISION,
                                 "provider_calls": 0, "llm_tokens": 0, "semantic_entailment_verified": False,
                                 "operation": "prepare_documents" if question is None else "semantic_scores",
                                 "passage_window_tokens": PASSAGE_WINDOW_TOKENS,
                                 "query_max_tokens": QUERY_MAX_TOKENS,
                                 "span_offset_unit": "characters_in_source_chunk",
                                 "threshold": self.min_score, "threshold_scope": "heuristic_not_universal_calibration"}
        if not chunks:
            return {"scores": {}, "spans": {}, "trace": {**trace, "status": "empty_corpus", "total_ms": 0.0}}
        if not (self.model_path / "docthinker-model.json").is_file():
            return {"scores": {}, "spans": {}, "trace": {**trace, "status": "model_not_installed", "total_ms": round((time.perf_counter() - start) * 1000, 3)}}
        try:
            initialization_ms, reused = self._initialize()
        except (ImportError, OSError, ValueError, KeyError):
            return {"scores": {}, "spans": {}, "trace": {**trace, "status": "model_unavailable", "total_ms": round((time.perf_counter() - start) * 1000, 3)}}
        trace.update(initialization_ms=round(initialization_ms, 3), model_reused=reused, model_fingerprint=self.fingerprint)
        # Filter again at the boundary: generated edges must never gain a source ID.
        from .research_retrieval import _source_row
        rows = []
        identities = set()
        for raw in chunks[:MAX_CHUNKS]:
            row = _source_row(raw)
            if row is not None and row["id"] not in identities:
                rows.append(row)
                identities.add(row["id"])
        documents_started = time.perf_counter()
        cached = 0
        physical_cached = 0
        newly_encoded = 0
        logical_windows = 0
        limited = 0
        vectors_by_id = {}
        windows_by_id = {}
        with _LOCK:
            logical_limit = min(self.cache_entries, LEGACY_CACHE_ENTRIES)
            pending = []
            pending_texts = []
            for row in rows:
                content_hash = hashlib.sha256(row["content"].encode()).hexdigest()
                key = (session_id, self.fingerprint, "passage", content_hash)
                logical_hit = key in _LOGICAL_VECTORS
                reusable = _VECTORS.get(key)
                if logical_hit:
                    _LOGICAL_VECTORS.move_to_end(key)
                    cached += 1
                    if reusable is not None:
                        _VECTORS.move_to_end(key)
                        vectors_by_id[row["id"]], cut, windows_by_id[row["id"]] = reusable
                        limited += int(cut)
                        physical_cached += 1
                        continue
                if reusable is None:
                    windows, cut = _windows(row["content"], self._tokenizer)
                else:
                    # The key includes the full current content hash and the
                    # model/window fingerprint. These exact source offsets are
                    # reusable even when the compatibility LRU forgot the key.
                    _, cut, windows = reusable
                limited += int(cut)
                if not windows:
                    continue
                if not logical_hit and logical_windows + len(windows) > MAX_TOTAL_WINDOWS:
                    limited += 1
                    continue
                if not logical_hit:
                    logical_windows += len(windows)
                # Reuse a physically retained vector only after deciding that
                # this row fits the unchanged logical candidate budget. An
                # expanded cache must not silently add/remove source windows.
                left = len(pending_texts)
                if reusable is None:
                    pending_texts.extend("passage: " + row["content"][a:b] for a, b in windows)
                else:
                    physical_cached += 1
                pending.append((row["id"], key, left, len(pending_texts), cut,
                                tuple(windows), reusable, logical_hit))
            if pending_texts:
                # Serial bounded batches avoid multiplying ONNX thread pools.
                def encode():
                    return _normalized(self._model.embed(pending_texts, batch_size=16), len(pending_texts))
                disk_status = None
                if self._private_cache_root is not None:
                    from .research_passage_cache import PrivatePassageCache
                    try:
                        identity = _passage_batch_identity(self, rows, pending, pending_texts)
                    except (OSError, ValueError, TypeError, UnicodeError):
                        matrix = encode()
                        disk_status = 'identity_unavailable'
                    else:
                        matrix, disk_status = PrivatePassageCache(self._private_cache_root).matrix(
                            session_id, identity, rows=len(pending_texts), dimension=MODEL_DIMENSION, encode=encode)
                    disk_hit = disk_status in ('hit', 'hit_maintenance_unavailable')
                    trace['private_passage_cache'] = {'status': disk_status,
                        'cached_windows': len(pending_texts) if disk_hit else 0,
                        'scope': 'exact_live_batch_derived_features_not_source_authority'}
                else:
                    matrix = encode()
                newly_encoded = 0 if disk_status in ('hit', 'hit_maintenance_unavailable') else len(pending_texts)
            if pending:
                for identifier, key, left, right, cut, windows, reusable, logical_hit in pending:
                    if reusable is None:
                        reusable = (matrix[left:right].copy(), cut, windows)
                    vectors_by_id[identifier], _, windows_by_id[identifier] = reusable
                    if _VECTORS.get(key) is not reusable:
                        _VECTORS[key] = reusable
                    _VECTORS.move_to_end(key)
                    if not logical_hit:
                        _LOGICAL_VECTORS[key] = None
                        _LOGICAL_VECTORS.move_to_end(key)
                if logical_windows:
                    while len(_LOGICAL_VECTORS) > logical_limit:
                        _LOGICAL_VECTORS.popitem(last=False)
                _VECTORS.trim(self.cache_entries, self.cache_bytes)
            document_ms = (time.perf_counter() - documents_started) * 1000
            if question is None:
                trace.update(status="ready", scanned_chunks=len(rows), corpus_scan_limited=len(chunks) > MAX_CHUNKS,
                             passage_windows_encoded=newly_encoded, cached_chunks=cached, limited_chunks=limited,
                             query_embedding_ms=0.0, query_embedded=False,
                             document_embedding_ms=round(document_ms, 3),
                             total_ms=round((time.perf_counter() - start) * 1000, 3))
                trace.update(retained_vector_bytes=_VECTORS.retained_bytes,
                             retained_logical_metadata_bytes=_LOGICAL_VECTORS.retained_bytes,
                             retained_cache_bytes=_VECTORS.retained_bytes + _LOGICAL_VECTORS.retained_bytes,
                             retained_cache_entries=len(_VECTORS), cache_entry_limit=self.cache_entries,
                             logical_cache_entries=len(_LOGICAL_VECTORS), logical_cache_entry_limit=logical_limit,
                             legacy_passage_window_budget_used=logical_windows,
                             physical_cached_chunks=physical_cached,
                             retained_cache_limit_bytes=self.cache_bytes,
                             cache_budget_scope="retained_derived_features_not_rss_or_build_peak")
                return {"scores": {}, "spans": {}, "trace": trace}
            query_started = time.perf_counter()
            query_tokens = self._tokenizer.encode(question, add_special_tokens=False).offsets
            query_truncated = len(query_tokens) > QUERY_MAX_TOKENS
            query_input = question[:query_tokens[QUERY_MAX_TOKENS - 1][1]] if query_truncated else question
            query_key = (session_id, self.fingerprint, "query", hashlib.sha256(question.encode()).hexdigest())
            query_cached = query_key in _LOGICAL_VECTORS
            physical_query_cached = query_key in _VECTORS
            if physical_query_cached:
                query_vector = _VECTORS[query_key]
                _VECTORS.move_to_end(query_key)
            else:
                query_vector = _normalized(self._model.embed(["query: " + query_input], batch_size=1), 1)[0]
                _VECTORS[query_key] = query_vector
            _LOGICAL_VECTORS[query_key] = None
            _LOGICAL_VECTORS.move_to_end(query_key)
            if not query_cached:
                while len(_LOGICAL_VECTORS) > logical_limit:
                    _LOGICAL_VECTORS.popitem(last=False)
            _VECTORS.trim(self.cache_entries, self.cache_bytes)
            query_ms = (time.perf_counter() - query_started) * 1000
        # Keep the location that produced the score. A score on the latter half
        # must not lead a downstream renderer to quote the unrelated beginning.
        # Equal-score ties choose the earliest window. These are retrieval spans,
        # not complete sentences, source claims or evidence of entailment.
        raw_scores, raw_spans = {}, {}
        for identifier, matrix in vectors_by_id.items():
            window_scores = matrix @ query_vector
            best_window = int(window_scores.argmax())
            raw_scores[identifier] = min(1.0, max(-1.0, float(window_scores[best_window])))
            raw_spans[identifier] = list(windows_by_id[identifier][best_window])
        accepted = {identifier: round(score, 8) for identifier, score in sorted(raw_scores.items(), key=lambda pair: (-pair[1], pair[0]))
                    if score >= self.min_score}
        scores = dict(list(accepted.items())[:40])
        spans = {identifier: raw_spans[identifier] for identifier in scores}
        trace.update(status="ready", scanned_chunks=len(rows), corpus_scan_limited=len(chunks) > MAX_CHUNKS,
                     passage_windows_encoded=newly_encoded, cached_chunks=cached, query_cache_hit=query_cached,
                     query_tokens=len(query_tokens), query_truncated=query_truncated,
                     limited_chunks=limited, accepted_chunks=len(scores), best_cosine=max(raw_scores.values(), default=None),
                     document_embedding_ms=round(document_ms, 3), query_embedding_ms=round(query_ms, 3),
                     total_ms=round((time.perf_counter() - start) * 1000, 3))
        trace.update(retained_vector_bytes=_VECTORS.retained_bytes,
                     retained_logical_metadata_bytes=_LOGICAL_VECTORS.retained_bytes,
                     retained_cache_bytes=_VECTORS.retained_bytes + _LOGICAL_VECTORS.retained_bytes,
                     retained_cache_entries=len(_VECTORS), cache_entry_limit=self.cache_entries,
                     logical_cache_entries=len(_LOGICAL_VECTORS), logical_cache_entry_limit=logical_limit,
                     legacy_passage_window_budget_used=logical_windows,
                     physical_cached_chunks=physical_cached, physical_query_cache_hit=physical_query_cached,
                     retained_cache_limit_bytes=self.cache_bytes,
                     cache_budget_scope="retained_derived_features_not_rss_or_build_peak")
        return {"scores": scores, "spans": spans, "trace": trace}
