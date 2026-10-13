"""Optional local multilingual cross-encoder for bounded evidence reranking.

Scores are raw relevance logits, not probabilities, entailment, or truth checks.
The runtime never downloads assets or calls a remote model. Install the pinned
public model explicitly with ``scripts/setup_research_reranker.py`` first.
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

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MODEL_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
MODEL_FILE = "onnx/model_qint8_arm64.onnx"
MODEL_SHA256 = "1825907d6c1a9001ff78124780bbde20a614a8c3df3b63409cf3c72c6fe5c8b4"
MODEL_HASHES = {
    MODEL_FILE: MODEL_SHA256,
    "config.json": "cc2cfe51aa3fd759d21d21acf5dfd6994aa67a3c9210636d22e143699d336c77",
    "tokenizer.json": "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626",
    "tokenizer_config.json": "e7fbfbfa6347b4e414c1cee50d142e2c2f9a895dad68b068ae83a8b564c3837e",
    "special_tokens_map.json": "378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d",
}
MODEL_FILES = tuple(MODEL_HASHES)
INDEX_VERSION = "mmarco-int8-evidence-v1"
DEFAULT_MODEL_DIR = Path.home() / ".cache" / "docthinker" / "research-reranker" / MODEL_REVISION
MAX_CANDIDATES = 48
MAX_CONTENT_CHARS = 2048
MAX_QUESTION_BYTES = 8192
MAX_QUERY_TOKENS = 192
MAX_PAIR_TOKENS = 512
BATCH_SIZE = 8
MAX_CACHE_ENTRIES = 4096
_LOCK = threading.RLock()
_INFERENCE_GATE = threading.BoundedSemaphore(1)
_MODELS: OrderedDict[str, tuple[Any, Any, str]] = OrderedDict()
_SCORES: OrderedDict[tuple[str, str, str, str], float] = OrderedDict()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_manifest(path: Path) -> tuple[dict[str, Any], str]:
    manifest = json.loads((path / "docthinker-model.json").read_text())
    if manifest.get("model") != MODEL_NAME or manifest.get("revision") != MODEL_REVISION:
        raise ValueError("Unsupported local reranker")
    if manifest.get("sha256") != MODEL_HASHES:
        raise ValueError("Unsupported local reranker assets")
    for name, expected in MODEL_HASHES.items():
        if not (path / name).is_file() or _sha256(path / name) != expected:
            raise ValueError("Changed or incomplete local reranker assets")
    fingerprint = hashlib.sha256(json.dumps([MODEL_REVISION, INDEX_VERSION, MODEL_HASHES], sort_keys=True).encode()).hexdigest()
    return manifest, fingerprint


def _load_model(path: Path):
    from fastembed.common.model_description import ModelSource
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    from tokenizers import Tokenizer

    custom_name = "docthinker/mmarco-minilm-int8-pinned"
    if custom_name not in {item["model"] for item in TextCrossEncoder.list_supported_models()}:
        TextCrossEncoder.add_custom_model(model=custom_name, sources=ModelSource(hf=MODEL_NAME),
                                          model_file=MODEL_FILE, license="apache-2.0")
    model = TextCrossEncoder(model_name=custom_name, specific_model_path=str(path),
                             local_files_only=True, threads=2, providers=["CPUExecutionProvider"])
    # The question is bounded separately, so only the passage may be truncated.
    model.model.tokenizer.enable_truncation(max_length=MAX_PAIR_TOKENS, strategy="only_second")
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    tokenizer.no_truncation()
    return model, tokenizer


class LocalEvidenceReranker:
    """Relevance-only scorer; callers keep exact source spans and final budgets.

    ``session_id`` must contain the storage namespace as well as the project ID.
    No input text, IDs, or exception messages are placed in diagnostic traces.
    """

    def __init__(self, model_path: str | Path | None = None, *, cache_entries: int = MAX_CACHE_ENTRIES):
        if type(cache_entries) is not int or not 1 <= cache_entries <= MAX_CACHE_ENTRIES:
            raise ValueError("Invalid reranker cache size")
        configured = model_path or os.getenv("DOCTHINKER_RESEARCH_RERANKER_DIR")
        self.model_path = Path(configured).expanduser().resolve() if configured else DEFAULT_MODEL_DIR
        self.cache_entries = cache_entries
        self._model = None
        self._tokenizer = None
        self.fingerprint = None

    def _initialize(self):
        if self._model is not None:
            return
        with _LOCK:
            key = str(self.model_path)
            cached = _MODELS.get(key)
            if cached is not None:
                self._model, self._tokenizer, self.fingerprint = cached
                _MODELS.move_to_end(key)
                return
            _, self.fingerprint = _model_manifest(self.model_path)
            self._model, self._tokenizer = _load_model(self.model_path)
            _MODELS[key] = (self._model, self._tokenizer, self.fingerprint)
            while len(_MODELS) > 2:
                _MODELS.popitem(last=False)

    def prepare_assets(self) -> dict[str, Any]:
        """Validate/load local assets only, sharing the real inference slot.

        Does not tokenize, score, populate query caches, or download. Busy
        callers skip immediately; a running loader keeps the gate until exit.
        """
        started = time.perf_counter()
        trace = {"status": "busy", "model": MODEL_NAME, "revision": MODEL_REVISION,
                 "asset_only": True, "query_inference": False, "provider_calls": 0,
                 "provider_tokens": 0, "model_reused": False, "initialization_ms": 0.0}

        def result(status):
            return {"trace": {**trace, "status": status,
                    "total_ms": round((time.perf_counter() - started) * 1000, 3)}}

        if not _INFERENCE_GATE.acquire(blocking=False):
            return result("busy")
        try:
            trace["model_reused"] = self._model is not None or str(self.model_path) in _MODELS
            initialization = time.perf_counter()
            status = "ready"
            try:
                self._initialize()
            except Exception:
                status = "unavailable"
            finally:
                trace["initialization_ms"] = round((time.perf_counter() - initialization) * 1000, 3)
            return result(status)
        finally:
            _INFERENCE_GATE.release()

    def rank(self, question: str, candidates: Sequence[Mapping[str, Any]], *, session_id: str) -> dict[str, Any]:
        """Compatibility API for a single shared question."""
        if not isinstance(question, str) or not question.strip() or len(question.encode()) > MAX_QUESTION_BYTES:
            raise ValueError("Invalid reranker question")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or len(candidates) > MAX_CANDIDATES:
            raise ValueError("Too many reranker candidates")
        if any(not isinstance(row, Mapping) for row in candidates):
            raise ValueError("Invalid reranker candidate")
        return self.rank_pairs([{**row, "query": question} for row in candidates], session_id=session_id)

    def rank_pairs(self, candidates: Sequence[Mapping[str, Any]], *, session_id: str) -> dict[str, Any]:
        """Score at most 48 total query/passage pairs, not 48 per query.

        Each row supplies its own literal question/facet in ``query``. All
        groups share one process slot and one total input limit. Grouping only
        batches equal queries; it never combines independent questions into a
        diluted long prompt or performs extra provider calls.
        """
        if not isinstance(session_id, str) or not session_id or len(session_id) > 512:
            raise ValueError("Invalid reranker namespace")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or len(candidates) > MAX_CANDIDATES:
            raise ValueError("Too many reranker candidates")
        rows = []
        identifiers = set()
        for row in candidates:
            if not isinstance(row, Mapping):
                raise ValueError("Invalid reranker candidate")
            identifier, content, question = row.get("id"), row.get("content"), row.get("query")
            if not isinstance(identifier, str) or not identifier or len(identifier) > 512 or identifier in identifiers:
                raise ValueError("Invalid reranker candidate ID")
            if not isinstance(content, str) or not content.strip() or len(content) > MAX_CONTENT_CHARS:
                raise ValueError("Invalid reranker passage")
            if not isinstance(question, str) or not question.strip() or len(question.encode()) > MAX_QUESTION_BYTES:
                raise ValueError("Invalid reranker question")
            identifiers.add(identifier)
            rows.append((identifier, question, content))
        started = time.perf_counter()
        trace = {"status": "empty", "model": MODEL_NAME, "revision": MODEL_REVISION,
                 "score_kind": "raw_relevance_logit", "candidate_count": len(rows),
                 "scored_count": 0, "cache_hits": 0, "inferred_count": 0,
                 "query_truncated": False, "passages_truncated": 0,
                 "query_count": len({row[1] for row in rows}), "queries_truncated": 0,
                 "provider_calls": 0, "provider_tokens": 0, "latency_ms": 0.0}
        input_receipts = {}

        def result(status: str, scores=None):
            trace["status"] = status
            trace["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            # Private pair receipts describe tokenizer scope, not attention or
            # entailment. Failed inference must not look like scored evidence.
            return {"scores": scores or {}, "trace": trace,
                    "input_receipts": input_receipts if status == "ready" else {}}

        if not rows:
            return result("empty")
        if not _INFERENCE_GATE.acquire(blocking=False):
            return result("busy")
        try:
            try:
                self._initialize()
            except Exception:
                return result("unavailable")
            namespace = hashlib.sha256(session_id.encode()).hexdigest()
            scores = {}
            pending: OrderedDict[tuple[str, str, str, str], tuple[str, list[str]]] = OrderedDict()
            queries = {}
            for identifier, question, content in rows:
                query_hash = hashlib.sha256(question.encode()).hexdigest()
                if query_hash not in queries:
                    encoded = self._tokenizer.encode(question, add_special_tokens=False)
                    query = question
                    if len(encoded.ids) > MAX_QUERY_TOKENS:
                        query = question[:encoded.offsets[MAX_QUERY_TOKENS - 1][1]]
                        trace["query_truncated"] = True
                        trace["queries_truncated"] += 1
                    queries[query_hash] = query
                query = queries[query_hash]
                pair = self._tokenizer.encode(query, content, add_special_tokens=True)
                if len(pair.ids) > MAX_PAIR_TOKENS:
                    trace["passages_truncated"] += 1
                query_complete = query == question
                passage_complete = len(pair.ids) <= MAX_PAIR_TOKENS
                input_receipts[identifier] = {
                    "submitted_query_sha256": query_hash,
                    "actual_query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                    "submitted_passage_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    # A truncated pair does not establish the exact passage
                    # prefix consumed inside the model. Never hash the whole
                    # submission and label it an observed complete passage.
                    "actual_passage_sha256": hashlib.sha256(content.encode()).hexdigest() if passage_complete else None,
                    "actual_query_character_end": len(query),
                    "query_complete": query_complete,
                    "passage_complete": passage_complete,
                    "pair_tokens_before_truncation": len(pair.ids),
                    "max_pair_tokens": MAX_PAIR_TOKENS,
                    "input_scope": ("complete_pair" if query_complete and passage_complete
                                    else "passage_prefix" if query_complete
                                    else "query_prefix" if passage_complete
                                    else "query_and_passage_prefix"),
                    "basis": "tokenizer_budget_check_not_semantic_verification",
                }
                key = (namespace, str(self.fingerprint), query_hash, hashlib.sha256(content.encode()).hexdigest())
                if key in _SCORES:
                    scores[identifier] = _SCORES[key]
                    trace["cache_hits"] += 1
                    _SCORES.move_to_end(key)
                elif key in pending:
                    pending[key][1].append(identifier)
                else:
                    pending[key] = (content, [identifier])
            if pending:
                groups = OrderedDict()
                for key, item in pending.items():
                    groups.setdefault(key[2], []).append((key, item))
                staged = []
                for query_hash, group in groups.items():
                    values = list(self._model.rerank(queries[query_hash], [item[0] for _, item in group], batch_size=BATCH_SIZE))
                    if len(values) != len(group) or any(isinstance(value, bool) or not math.isfinite(float(value)) for value in values):
                        raise ValueError("Invalid reranker outputs")
                    staged.extend((key, ids, float(value)) for (key, (_, ids)), value in zip(group, values))
                # No partial cache commits if a later query group fails.
                for key, ids, score in staged:
                    _SCORES[key] = score
                    for identifier in ids:
                        scores[identifier] = score
                trace["inferred_count"] = len(pending)
                while len(_SCORES) > self.cache_entries:
                    _SCORES.popitem(last=False)
            trace["scored_count"] = len(scores)
            return result("ready", scores)
        except Exception:
            # No partial scores on malformed model output or failed inference.
            return result("inference_error")
        finally:
            _INFERENCE_GATE.release()
