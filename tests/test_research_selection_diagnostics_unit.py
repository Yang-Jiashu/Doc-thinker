"""Private diagnostic callbacks are observational and disabled by default."""
from copy import deepcopy
import hashlib
import json

from docthinker.server.research_evidence_selection import prepare_evidence_candidates, select_evidence
from docthinker.server.research_faceted_assist import build_faceted_assistance


QUESTION = "How does the cobalt controller update its buffer?"
CHUNKS = [
    {"id": "method", "document_id": "paper", "filename": "paper.pdf", "ordinal": 0,
     "content": "2 Methods\nThe cobalt controller updates its buffer through a bounded queue. "
                "The controller calls a parity check before updating its buffer."},
    {"id": "noise", "document_id": "paper", "filename": "paper.pdf", "ordinal": 1,
     "content": "References\n[1] Lane, A. 2024. Cobalt controller. Journal of fictional methods."},
]
PLAN = {"facets": [{"id": "mechanism", "kind": "clause", "text": QUESTION,
                    "weight": 1., "roles": ["mechanism"]}]}


def test_pool_and_selection_receipts_do_not_change_result_or_inputs():
    chunks, plan = deepcopy(CHUNKS), deepcopy(PLAN)
    events = []
    before = prepare_evidence_candidates(QUESTION, chunks, [], plan, {})
    after = prepare_evidence_candidates(QUESTION, chunks, [], plan, {}, diagnostic_callback=events.append)
    assert after == before
    assert chunks == CHUNKS and plan == PLAN
    scores = {row["id"]: 1. for row in before["candidates"]}
    plain = select_evidence(before["candidates"], scores, plan, [])
    observed = select_evidence(after["candidates"], scores, plan, [], diagnostic_callback=events.append)
    assert observed == plain
    names = {row["event"] for row in events}
    assert {"unit_filtered", "window_eligible", "candidate_nominated", "candidate_pool_final",
            "selection_candidate", "selection_option", "selection_chosen", "selection_stop"} <= names
    assert "diagnostic" not in repr(observed)


def test_callback_failure_and_mutation_cannot_change_selection():
    expected = prepare_evidence_candidates(QUESTION, CHUNKS, [], PLAN, {})

    def hostile(event):
        event.clear()
        raise RuntimeError("PRIVATE_CALLBACK_ERROR")

    actual = prepare_evidence_candidates(QUESTION, CHUNKS, [], PLAN, {}, diagnostic_callback=hostile)
    assert actual == expected
    scores = {row["id"]: 1. for row in expected["candidates"]}
    assert select_evidence(actual["candidates"], scores, PLAN, [], diagnostic_callback=hostile) == \
        select_evidence(expected["candidates"], scores, PLAN, [])


class Semantic:
    def semantic_scores(self, question, chunks, *, session_id):
        return {"scores": {row["id"]: .85 for row in chunks},
                "spans": {row["id"]: [0, len(row["content"])] for row in chunks},
                "trace": {"status": "ready"}}


class Reranker:
    def rank_pairs(self, candidates, *, session_id):
        return {"scores": {row["id"]: 1. for row in candidates},
                "trace": {"status": "ready", "provider_calls": 0}}


def stable(value):
    if isinstance(value, dict):
        return {key: stable(item) for key, item in value.items() if not key.endswith("_ms")}
    if isinstance(value, list):
        return [stable(item) for item in value]
    return value


def test_orchestration_callback_is_private_and_preserves_outputs():
    events = []
    plain = build_faceted_assistance(QUESTION, [], CHUNKS, Semantic(), Reranker(), session_id="offline::test")
    observed = build_faceted_assistance(QUESTION, [], CHUNKS, Semantic(), Reranker(), session_id="offline::test",
                                        diagnostic_callback=events.append)
    assert stable(plain) == stable(observed)
    names = {row["event"] for row in events}
    assert {"query_plan", "facet_semantic", "rerank_result", "sources_final"} <= names
    assert all(row["logit"] == 1. and row["query"] for row in events if row["event"] == "rerank_result")
    assert "query_plan" in observed["trace"]  # Existing aggregate trace remains.
    assert "facets" not in observed["trace"]["query_plan"]
    assert "diagnostics" not in observed and "events" not in observed["trace"]


class ReceiptReranker(Reranker):
    def rank_pairs(self, candidates, *, session_id):
        result = super().rank_pairs(candidates, session_id=session_id)
        self.receipts = {row["id"]: {
            "submitted_query_sha256": hashlib.sha256(row["query"].encode()).hexdigest(),
            "actual_query_sha256": hashlib.sha256(row["query"].encode()).hexdigest(),
            "actual_query_character_end": len(row["query"]),
            "query_complete": True, "passage_complete": True,
            "pair_tokens_before_truncation": 120, "max_pair_tokens": 512,
            "input_scope": "PRIVATE_MODEL_INPUT_SCOPE", "basis": "PRIVATE_TOKENIZER_RECEIPT",
        } for row in candidates}
        return {**result, "input_receipts": self.receipts}


def test_pair_input_receipts_are_private_observations_not_selection_inputs():
    events, reranker = [], ReceiptReranker()
    plain = build_faceted_assistance(QUESTION, [], CHUNKS, Semantic(), Reranker(), session_id="offline::test")
    observed = build_faceted_assistance(QUESTION, [], CHUNKS, Semantic(), reranker, session_id="offline::test",
                                        diagnostic_callback=events.append)
    assert stable(plain) == stable(observed)
    scored = [row for row in events if row["event"] == "rerank_result"]
    assert scored
    for event in scored:
        assert event["input_receipt"] == reranker.receipts[event["window"]["id"]]
        assert event["input_receipt"]["actual_query_sha256"] == hashlib.sha256(event["query"].encode()).hexdigest()
        assert event["input_receipt"]["query_complete"] is True
    public = json.dumps(observed)
    assert "input_receipt" not in public and "PRIVATE_MODEL_INPUT_SCOPE" not in public
    assert "PRIVATE_TOKENIZER_RECEIPT" not in public
    assert all(row["actual_query_sha256"] not in public for row in reranker.receipts.values())


def test_callback_cannot_mutate_reranker_input_receipt_or_selected_evidence():
    reranker = ReceiptReranker()
    plain = build_faceted_assistance(QUESTION, [], CHUNKS, Semantic(), Reranker(), session_id="offline::test")

    def hostile(event):
        if event["event"] == "rerank_result":
            event["input_receipt"]["actual_query_sha256"] = "PRIVATE_MUTATION"
            raise RuntimeError("PRIVATE_RECEIPT_ERROR")

    observed = build_faceted_assistance(QUESTION, [], CHUNKS, Semantic(), reranker, session_id="offline::test",
                                        diagnostic_callback=hostile)
    assert stable(plain) == stable(observed)
    assert all(row["actual_query_sha256"] != "PRIVATE_MUTATION" for row in reranker.receipts.values())


def test_missing_or_invalid_receipts_are_empty_private_diagnostics():
    class InvalidReceipts(Reranker):
        def rank_pairs(self, candidates, *, session_id):
            return {**super().rank_pairs(candidates, session_id=session_id), "input_receipts": None}

    events = []
    actual = build_faceted_assistance(QUESTION, [], CHUNKS, Semantic(), InvalidReceipts(), session_id="offline::test",
                                      diagnostic_callback=events.append)
    plain = build_faceted_assistance(QUESTION, [], CHUNKS, Semantic(), Reranker(), session_id="offline::test")
    assert stable(plain) == stable(actual)
    assert all(row["input_receipt"] == {} for row in events if row["event"] == "rerank_result")


def test_unready_scorer_cannot_publish_stale_pair_input_receipts():
    class UnreadyReceipts(ReceiptReranker):
        def rank_pairs(self, candidates, *, session_id):
            result = super().rank_pairs(candidates, session_id=session_id)
            return {**result, "trace": {"status": "busy"}, "scores": {}}

    events = []
    actual = build_faceted_assistance(QUESTION, [], CHUNKS, Semantic(), UnreadyReceipts(), session_id="offline::test",
                                      diagnostic_callback=events.append)
    assert actual["trace"]["skipped_reason"] == "reranker_unavailable"
    assert all(row["input_receipt"] == {} for row in events if row["event"] == "rerank_result")
    assert "PRIVATE_MODEL_INPUT_SCOPE" not in json.dumps(actual)
