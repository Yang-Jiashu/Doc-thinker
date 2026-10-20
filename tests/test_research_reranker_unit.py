from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
import threading
from types import SimpleNamespace

import pytest

from docthinker.server import research_reranker as rr


@pytest.fixture(autouse=True)
def clean_globals():
    rr._SCORES.clear()
    rr._MODELS.clear()
    yield
    rr._SCORES.clear()
    rr._MODELS.clear()


class Tokens:
    def encode(self, query, content=None, add_special_tokens=False):
        text = query if content is None else query + content + "XXXX"
        return SimpleNamespace(ids=list(range(len(text))), offsets=[(i, i + 1) for i in range(len(text))])


class Model:
    def __init__(self):
        self.calls = []
        self.outputs = None

    def rerank(self, question, documents, *, batch_size):
        self.calls.append((question, list(documents), batch_size))
        return self.outputs if self.outputs is not None else [float(len(document)) for document in documents]


def ready(monkeypatch, **kwargs):
    model = Model()
    service = rr.LocalEvidenceReranker(**kwargs)
    service._model = model
    service._tokenizer = Tokens()
    service.fingerprint = "model-fingerprint"
    return service, model


def row(identifier="span-a", content="A relevant source."):
    return {"id": identifier, "content": content}


def test_raw_logits_scoped_cache_and_only_live_ids(monkeypatch):
    service, model = ready(monkeypatch)
    a = service.rank("Why?", [row()], session_id="root-one::session-1")
    assert a["scores"] == {"span-a": 18.0}
    assert a["trace"]["score_kind"] == "raw_relevance_logit"
    b = service.rank("Why?", [row("new-id")], session_id="root-one::session-1")
    assert b["scores"] == {"new-id": 18.0}
    assert b["trace"]["cache_hits"] == 1
    assert b["trace"]["inferred_count"] == 0
    service.rank("Why?", [row()], session_id="root-two::session-1")
    service.rank("When?", [row()], session_id="root-one::session-1")
    service.rank("Why?", [row(content="Changed source")], session_id="root-one::session-1")
    service.fingerprint = "changed-model"
    service.rank("Why?", [row()], session_id="root-one::session-1")
    assert len(model.calls) == 5


def test_duplicate_passages_infer_once_but_preserve_ids(monkeypatch):
    service, model = ready(monkeypatch)
    result = service.rank("Question", [row("a"), row("b")], session_id="project")
    assert set(result["scores"]) == {"a", "b"}
    assert result["trace"]["inferred_count"] == 1
    assert len(model.calls[0][1]) == 1


def test_cache_is_bounded(monkeypatch):
    service, _ = ready(monkeypatch, cache_entries=2)
    for number in range(5):
        service.rank(f"Query {number}", [row()], session_id="project")
    assert len(rr._SCORES) == 2


def test_missing_assets_fail_closed_without_paths_or_text(tmp_path):
    result = rr.LocalEvidenceReranker(tmp_path).rank("private-question", [row(content="private-source")], session_id="private-namespace")
    assert result["scores"] == {}
    assert result["trace"]["status"] == "unavailable"
    serialized = json.dumps(result)
    for secret in ("private-question", "private-source", "private-namespace", str(tmp_path)):
        assert secret not in serialized
    assert result["trace"]["provider_calls"] == result["trace"]["provider_tokens"] == 0


def test_busy_does_not_initialize_or_queue(monkeypatch):
    service, model = ready(monkeypatch)
    assert rr._INFERENCE_GATE.acquire(blocking=False)
    try:
        result = service.rank("Question", [row()], session_id="project")
        assert result["trace"]["status"] == "busy"
        assert not result["scores"] and not model.calls
    finally:
        rr._INFERENCE_GATE.release()
    assert service.rank("Question", [row()], session_id="project")["trace"]["status"] == "ready"


def test_running_worker_holds_process_wide_gate(monkeypatch):
    service, model = ready(monkeypatch)
    entered, release = threading.Event(), threading.Event()
    def blocked(question, documents, *, batch_size):
        entered.set()
        assert release.wait(2)
        return [1.0] * len(documents)
    model.rerank = blocked
    worker = threading.Thread(target=lambda: service.rank("Question", [row()], session_id="one"))
    worker.start()
    try:
        assert entered.wait(2)
        second, _ = ready(monkeypatch)
        assert second.rank("Question", [row()], session_id="two")["trace"]["status"] == "busy"
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()


@pytest.mark.parametrize("outputs", [[float("nan")], [float("inf")], [], [True], [1, 2]])
def test_invalid_outputs_fail_closed_and_release_gate(monkeypatch, outputs):
    service, model = ready(monkeypatch)
    model.outputs = outputs
    result = service.rank("Question", [row()], session_id="project")
    assert result["trace"]["status"] == "inference_error"
    assert result["scores"] == {} and rr._SCORES == {}
    model.outputs = [3.5]
    assert service.rank("Question", [row()], session_id="project")["scores"] == {"span-a": 3.5}


def test_query_and_pair_truncation_explicit(monkeypatch):
    service, model = ready(monkeypatch)
    result = service.rank("q" * 300, [row(content="d" * 1000)], session_id="project")
    assert result["trace"]["query_truncated"] is True
    assert result["trace"]["passages_truncated"] == 1
    assert len(model.calls[0][0]) == rr.MAX_QUERY_TOKENS
    assert model.calls[0][2] == 8
    receipt = result["input_receipts"]["span-a"]
    assert receipt["input_scope"] == "query_and_passage_prefix"
    assert receipt["query_complete"] is receipt["passage_complete"] is False
    assert receipt["actual_query_character_end"] == rr.MAX_QUERY_TOKENS
    assert receipt["actual_query_sha256"] == hashlib.sha256(("q" * rr.MAX_QUERY_TOKENS).encode()).hexdigest()
    assert receipt["pair_tokens_before_truncation"] == rr.MAX_QUERY_TOKENS + 1000 + 4
    assert receipt["submitted_passage_sha256"] == hashlib.sha256(("d" * 1000).encode()).hexdigest()
    assert receipt["actual_passage_sha256"] is None


@pytest.mark.parametrize("question,content,scope", [
    ("Q", "evidence", "complete_pair"),
    ("q" * 300, "evidence", "query_prefix"),
    ("Q", "d" * 1000, "passage_prefix"),
])
def test_private_input_scope_receipts_match_cache_without_public_text(monkeypatch, question, content, scope):
    service, model = ready(monkeypatch)
    first = service.rank(question, [row("first", content)], session_id="private namespace")
    cached = service.rank(question, [row("cached", content)], session_id="private namespace")
    receipt = first["input_receipts"]["first"]
    assert receipt["input_scope"] == scope
    assert cached["input_receipts"]["cached"] == receipt
    assert cached["trace"]["cache_hits"] == 1 and len(model.calls) == 1
    assert receipt["submitted_query_sha256"] == hashlib.sha256(question.encode()).hexdigest()
    passage_hash = hashlib.sha256(content.encode()).hexdigest()
    assert receipt["submitted_passage_sha256"] == passage_hash
    assert receipt["actual_passage_sha256"] == (passage_hash if receipt["passage_complete"] else None)
    assert receipt["basis"] == "tokenizer_budget_check_not_semantic_verification"
    assert "input_receipts" not in first["trace"]
    assert passage_hash not in json.dumps(first["trace"])
    assert "private namespace" not in json.dumps(first)


@pytest.mark.parametrize("question,candidates,namespace", [
    ("", [row()], "p"), ("x" * 8193, [row()], "p"),
    ("Q", [row()] * 49, "p"), ("Q", [row(), row()], "p"),
    ("Q", [row(content="a" * 2049)], "p"), ("Q", [row(content=" ")], "p"),
    ("Q", [{"id": 1, "content": "text"}], "p"), ("Q", [row()], ""),
    ("Q", "not-candidates", "p"),
])
def test_input_bounds(monkeypatch, question, candidates, namespace):
    service, _ = ready(monkeypatch)
    with pytest.raises(ValueError):
        service.rank(question, candidates, session_id=namespace)


def test_empty_does_not_load_model(tmp_path):
    result = rr.LocalEvidenceReranker(tmp_path).rank("Q", [], session_id="p")
    assert result["trace"]["status"] == "empty"


def test_all_asset_hashes_are_pinned_not_manifest_assertions(monkeypatch, tmp_path):
    expected = hashlib.sha256(b"pinned").hexdigest()
    monkeypatch.setattr(rr, "MODEL_HASHES", {"tiny-model": expected})
    (tmp_path / "tiny-model").write_bytes(b"pinned")
    manifest = {"model": rr.MODEL_NAME, "revision": rr.MODEL_REVISION, "sha256": {"tiny-model": expected}}
    (tmp_path / "docthinker-model.json").write_text(json.dumps(manifest))
    assert rr._model_manifest(tmp_path)[1]
    (tmp_path / "tiny-model").write_bytes(b"changed")
    with pytest.raises(ValueError):
        rr._model_manifest(tmp_path)
    manifest["sha256"]["tiny-model"] = hashlib.sha256(b"changed").hexdigest()
    (tmp_path / "docthinker-model.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        rr._model_manifest(tmp_path)


def test_real_pinned_model_chinese_to_english_offline(monkeypatch):
    if not (rr.DEFAULT_MODEL_DIR / "docthinker-model.json").is_file():
        pytest.skip("Explicit local reranker installation is optional")
    for dependency in ("onnxruntime", "tokenizers", "fastembed"):
        pytest.importorskip(dependency, reason="Real local reranker requires its optional inference dependencies")
    def deny(*args, **kwargs):
        raise AssertionError("Runtime must not open network connections")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    service = rr.LocalEvidenceReranker()
    candidates = [
        row("answer", "The fictional Orin trial randomly assigned participants to treatment or placebo and calculated confidence intervals to distinguish treatment effects from chance."),
        row("background", "The fictional Orin hospital has a parking lot and a cafeteria. It publishes its opening hours every year."),
        row("related", "Medical researchers develop treatments by studying biological mechanisms and manufacturing pharmaceutical compounds."),
    ]
    result = service.rank("研究如何判断治疗效果不是随机波动造成的？", candidates, session_id="offline-public-smoke")
    assert result["trace"]["status"] == "ready"
    assert max(result["scores"], key=result["scores"].get) == "answer"
    assert result["trace"]["provider_calls"] == 0


def test_pair_queries_group_inference_and_isolate_identical_passage_cache(monkeypatch):
    service, model = ready(monkeypatch)
    candidates = [
        {**row("a", "shared passage"), "query": "first query"},
        {**row("b", "shared passage"), "query": "second query"},
        {**row("c", "other passage"), "query": "first query"},
    ]
    result = service.rank_pairs(candidates, session_id="project")
    assert set(result['scores']) == {'a', 'b', 'c'}
    assert result['trace']['query_count'] == 2
    assert result['trace']['candidate_count'] == result['trace']['inferred_count'] == 3
    assert len(model.calls) == 2
    assert model.calls[0] == ('first query', ['shared passage', 'other passage'], 8)
    assert model.calls[1] == ('second query', ['shared passage'], 8)
    repeated = service.rank_pairs(candidates, session_id='project')
    assert repeated['trace']['cache_hits'] == 3
    assert repeated['trace']['inferred_count'] == 0
    assert len(model.calls) == 2
    service.rank_pairs([{**row('new', 'shared passage'), 'query': 'third query'}], session_id='project')
    assert len(model.calls) == 3
    # Single-question compatibility uses exactly the same scoped pair cache.
    cached = service.rank('first query', [row('new-id', 'shared passage')], session_id='project')
    assert cached['trace']['cache_hits'] == 1 and len(model.calls) == 3


def test_pair_limit_is_global_not_per_query(monkeypatch):
    service, model = ready(monkeypatch)
    candidates = [{**row(f'p{i}', f'passage {i}'), 'query': f'query {i % 8}'} for i in range(48)]
    result = service.rank_pairs(candidates, session_id='project')
    assert result['trace']['scored_count'] == 48
    assert sum(len(call[1]) for call in model.calls) == 48
    with pytest.raises(ValueError):
        service.rank_pairs(candidates + [{**row('extra'), 'query': 'ninth query'}], session_id='project')
    assert sum(len(call[1]) for call in model.calls) == 48


def test_pair_queries_truncate_independently_without_trace_text(monkeypatch):
    service, model = ready(monkeypatch)
    candidates = [{**row('a', 'private source'), 'query': 'q' * 300},
                  {**row('b', 'private source'), 'query': 'short query'}]
    result = service.rank_pairs(candidates, session_id='private namespace')
    assert result['trace']['query_truncated'] is True
    assert result['trace']['queries_truncated'] == 1
    assert len(model.calls[0][0]) == rr.MAX_QUERY_TOKENS
    assert model.calls[1][0] == 'short query'
    trace = json.dumps(result['trace'])
    assert 'short query' not in trace and 'private source' not in trace and 'private namespace' not in trace


def test_pair_api_busy_shares_single_question_gate(monkeypatch):
    service, model = ready(monkeypatch)
    assert rr._INFERENCE_GATE.acquire(blocking=False)
    try:
        result = service.rank_pairs([{**row(), 'query': 'one'}, {**row('b'), 'query': 'two'}], session_id='project')
        assert result['trace']['status'] == 'busy'
        assert result['scores'] == {} and model.calls == []
    finally:
        rr._INFERENCE_GATE.release()


def test_failed_later_pair_group_commits_no_partial_results_or_cache(monkeypatch):
    service, model = ready(monkeypatch)
    def score(question, documents, *, batch_size):
        return [1.0] if question == 'first' else [float('nan')]
    model.rerank = score
    result = service.rank_pairs([{**row('a'), 'query': 'first'}, {**row('b'), 'query': 'second'}], session_id='project')
    assert result['trace']['status'] == 'inference_error'
    assert result['scores'] == {} and rr._SCORES == {}
    assert result['input_receipts'] == {}


@pytest.mark.parametrize('query', [None, '', ' ', 'x' * 8193])
def test_pair_question_is_validated_before_inference(monkeypatch, query):
    service, model = ready(monkeypatch)
    with pytest.raises(ValueError):
        service.rank_pairs([{**row(), 'query': query}], session_id='project')
    assert model.calls == []


def test_real_pinned_pair_model_handles_two_cross_language_questions_offline(monkeypatch):
    if not (rr.DEFAULT_MODEL_DIR / 'docthinker-model.json').is_file():
        pytest.skip('Explicit local reranker installation is optional')
    for dependency in ("onnxruntime", "tokenizers", "fastembed"):
        pytest.importorskip(dependency, reason="Real local reranker requires its optional inference dependencies")
    def deny(*args, **kwargs):
        raise AssertionError('Runtime must not open network connections')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)
    service = rr.LocalEvidenceReranker()
    treatment = 'The fictional Orin trial randomly assigned participants to treatment or placebo and calculated confidence intervals to distinguish treatment effects from chance.'
    climate = 'Sea level can continue to rise after greenhouse gas emissions are reduced because the oceans retain heat and ice sheets respond slowly. Thermal expansion and ice loss have long response times.'
    study_query = '研究如何判断治疗效果不是随机波动造成的？'
    climate_query = '为什么减少温室气体排放不会立刻让海平面下降？'
    result = service.rank_pairs([
        {**row('study-answer', treatment), 'query': study_query},
        {**row('study-distractor', climate), 'query': study_query},
        {**row('climate-answer', climate), 'query': climate_query},
        {**row('climate-distractor', treatment), 'query': climate_query},
    ], session_id='offline-public-pairs')
    assert result['trace']['status'] == 'ready'
    assert result['scores']['study-answer'] > result['scores']['study-distractor']
    assert result['scores']['climate-answer'] > result['scores']['climate-distractor']
    assert result['trace']['query_count'] == 2
    assert result['trace']['provider_calls'] == 0
