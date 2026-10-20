"""Literal audit views use paid pairs, never independent facts or needs.

Entirely fictional questions/windows and synthetic logits: no model loading,
network access, paper-specific ranks or reference-answer vocabulary.
"""
from copy import deepcopy
import hashlib
import json

import pytest

from docthinker.server.research_evidence_selection import _window_row, select_evidence
from docthinker.server.research_faceted_assist import _content_requirements, _rerank_pairs, build_faceted_assistance
from docthinker.server.research_query_facets import MAX_FACET_BYTES, build_query_facets, quoted_audit_request, valid_quoted_audit_view


QUESTION = ("In English, revise this claim using Study Q: "
            "'Every sensor updates at 1 Hz immediately after sampling.' "
            "Explain which sampling, transmission and processing times the claim conflates, citing the paper.")
METADATA = [{"id": "metadata", "document_id": "study-q", "title": "Study Q", "filename": "study-q.pdf"}]


def plan_for(question=QUESTION):
    return build_query_facets(question, METADATA)


def fixture(count=48):
    plan = plan_for()
    original = plan["facets"][0]
    view = next(row for row in plan["facets"] if row.get("purpose") == "quoted_claim_audit")
    rows = []
    for index in range(count):
        # Distinct source words avoid a synthetic all-duplicate source pool.
        words = " ".join(f"unit{index}part{part}" for part in range(12))
        content = f"The sampling controller measures {words}. Transmission processing uses a bounded queue."
        source = {"id": f"source-{index}", "document_id": "study-q", "filename": "study-q.pdf",
                  "content": content, "ordinal": index}
        row = _window_row(source, 0, len(content), ["mechanism"], "Methods")
        row.update(facet_scores={original["id"]: .8, view["id"]: .9 - index / 100},
                   anchor_scores={}, pool_score=.8)
        rows.append(row)
    pairs = _rerank_pairs(QUESTION, rows, plan["facets"])
    return plan, rows, pairs


def receipt(row):
    digest = hashlib.sha256(row["query"].encode()).hexdigest()
    return {"submitted_query_sha256": digest, "actual_query_sha256": digest,
            "submitted_passage_sha256": hashlib.sha256(row["content"].encode()).hexdigest(),
            "actual_passage_sha256": hashlib.sha256(row["content"].encode()).hexdigest(),
            "actual_query_character_end": len(row["query"]),
            "query_complete": True, "passage_complete": True,
            "pair_tokens_before_truncation": 120, "max_pair_tokens": 512,
            "input_scope": "complete_pair", "basis": "tokenizer_budget_check_not_semantic_verification"}


def select(plan, pairs, *, receipts=None, cap=8, scale=1., shift=0.):
    events = []
    scores = {row["id"]: (-index - 1.) * scale + shift for index, row in enumerate(pairs)}
    result = select_evidence(pairs, scores, plan, [], max_sources=cap,
                             input_receipts=receipts, diagnostic_callback=events.append)
    return result, events


@pytest.mark.parametrize("opening,closing", [("'", "'"), ('"', '"'), ("“", "”"), ("‘", "’")])
def test_single_balanced_assertion_composes_exact_whole_prefix_and_request(opening, closing):
    question = QUESTION.replace("'Every", opening + "Every").replace("sampling.'", "sampling." + closing)
    plan = plan_for(question)
    view = next(row for row in plan["facets"] if row.get("query_only"))
    start, end = view["request_span"]
    assert question[start:end] == "Explain" + QUESTION.split("Explain", 1)[1]
    assert view["query_spans"] == [[0, len(question[:view["claim_span"][0]].rstrip())], [start, end]]
    assert view["text"] == " ".join(question[left:right] for left, right in view["query_spans"])
    assert view["text"].startswith("In English, revise this claim using Study Q: Explain")
    assert view["text"] not in question
    assert "Every sensor" not in view["text"]
    assert view["text"].endswith("citing the paper.")
    assert question[slice(*view["claim_span"])].endswith("sampling." + closing)
    assert plan["facets"][0]["text"] == question
    assert valid_quoted_audit_view(view, plan["facets"][0])
    assert view["query_only"] is True and view["counts_as_requirement"] is False
    assert not view.get("derived_translation") and not view.get("query_alias_of")
    assert len(plan["facets"]) <= 8
    assert _content_requirements(plan["facets"]) == _content_requirements([row for row in plan["facets"] if row is not view])


def test_outside_terminal_mark_and_literal_negations_numbers_are_not_deleted():
    question = QUESTION.replace("sampling.'", "sampling'.").replace("Every sensor updates", "Every sensor does not update")
    question = question.replace("processing times", "processing times at 7 ms, not 9 ms,")
    request = quoted_audit_request(question)
    assert request and "7 ms, not 9 ms" in request["text"]
    plan = plan_for(question)
    assert plan["facets"][0]["text"] == question
    assert "does not update" in question[slice(*request["claim_span"])]


def test_two_literal_spans_preserve_internal_whitespace_and_trim_only_join_edges():
    question = QUESTION.replace("this claim", "this deployment claim").replace("Study Q: ", "Study Q:\t  ")
    question = question.replace("' Explain", "'   Explain").replace("transmission and", "transmission\tand") + " \t"
    view = quoted_audit_request(question)
    assert view
    prefix, suffix = [question[left:right] for left, right in view["query_spans"]]
    assert prefix == "In English, revise this deployment claim using Study Q:"
    assert suffix.startswith("Explain") and suffix.endswith("paper.")
    assert "transmission\tand" in suffix
    assert view["text"] == prefix + " " + suffix
    assert view["query_spans"][1] == view["request_span"]
    assert view["query_spans"][0][1] < view["claim_span"][0] < view["claim_span"][1] < view["request_span"][0]


def test_complete_composed_query_not_only_request_must_fit_facet_byte_budget():
    base = QUESTION
    view = quoted_audit_request(base)
    assert view
    available = MAX_FACET_BYTES - len((view["text"] + " ").encode())
    at_limit = base[:-1] + " " + "x" * available + "."
    limited = quoted_audit_request(at_limit)
    assert limited and len(limited["text"].encode()) == MAX_FACET_BYTES
    # The suffix alone still fits; it must not allow the prefix to overflow.
    over_limit = at_limit[:-1] + "x."
    assert len(over_limit[limited["request_span"][0]:].encode()) < MAX_FACET_BYTES
    assert quoted_audit_request(over_limit) is None
    assert not any(row.get("query_only") for row in plan_for(over_limit)["facets"])


def test_same_suffix_different_literal_prefix_has_distinct_composed_query_identity():
    plans = [plan_for(QUESTION.replace("this claim", "this " + modifier + " claim"))
             for modifier in ("deployment", "performance")]
    views = [next(row for row in plan["facets"] if row.get("query_only")) for plan in plans]
    assert views[0]["id"] != views[1]["id"]
    assert all(valid_quoted_audit_view(view, plan["facets"][0]) for plan, view in zip(plans, views))
    assert not valid_quoted_audit_view(views[0], plans[1]["facets"][0])


@pytest.mark.parametrize("field,endpoint", [("query_spans", False), ("query_spans", 0.0),
                                            ("request_span", "float"), ("claim_span", "float")])
def test_literal_span_endpoints_must_be_real_integers(field, endpoint):
    plan = plan_for()
    view = next(row for row in plan["facets"] if row.get("query_only"))
    if field == "query_spans":
        view[field][0][0] = endpoint
    else:
        view[field][0] = float(view[field][0])
    assert not valid_quoted_audit_view(view, plan["facets"][0])


@pytest.mark.parametrize("modifier", ["deployment", "performance", "safety", "general numerical accuracy", "remote station timing"])
def test_ordinary_bounded_claim_modifiers_are_not_paper_specific(modifier):
    question = QUESTION.replace("this claim", "this " + modifier + " claim", 1)
    view = quoted_audit_request(question)
    assert view and "this " + modifier + " claim using Study Q:" in view["text"]
    assert any(row.get("purpose") == "quoted_claim_audit" for row in plan_for(question)["facets"])


@pytest.mark.parametrize("modifier", ["not", "never valid", "conditional", "only valid", "assumed", "hypothetical", "non-operational", "one two three four"])
def test_modified_claims_with_scope_negation_or_unbounded_modifiers_fail_closed(modifier):
    assert quoted_audit_request(QUESTION.replace("this claim", "this " + modifier + " claim", 1)) is None


@pytest.mark.parametrize("material_scope", [
    "Study Q Methods section, not the abstract",
    "Study Q when the station is online",
    "Study Q after excluding simulated measurements",
    "Study Q before removing failed measurements",
    "Study Q while the station is online",
    "Study Q until the calibration ends",
    "Study Q excluding simulated measurements",
    "Study Q ignoring failed measurements",
    "Study Q provided the station is online",
    "Study Q subject to a fixed sampling interval",
    "Study Q as long as the station is online",
    "Study Q in case the station is online",
    "Study Q with no failed measurements",
    "Study Q without simulated measurements",
])
def test_scope_anywhere_in_the_prefix_still_disables_the_audit_view(material_scope):
    question = QUESTION.replace("Study Q:", material_scope + ":")
    assert quoted_audit_request(question) is None
    plan = plan_for(question)
    assert plan["facets"][0]["text"] == question
    assert not any(row.get("purpose") == "quoted_claim_audit" for row in plan["facets"])


@pytest.mark.parametrize("question", [
    QUESTION.replace("revise this claim", "explain the term"),
    QUESTION.replace("revise this claim", "rewrite this sentence"),
    QUESTION.replace("revise this claim", "revise this claim for grammar"),
    "If the station is online, " + QUESTION,
    QUESTION.replace("Every sensor", "If the channel is clear, every sensor"),
    QUESTION.replace("Every sensor", "Only every sensor"),
    QUESTION.replace("Every sensor", "Every sensor's controller"),
    QUESTION.replace("Every sensor", 'Every "sensor"'),
    QUESTION.replace("sampling.'", "sampling."),
    QUESTION + " 'Another assertion is also quoted.'",
    QUESTION.replace("Study Q:", "Study Q and Study R:"),
    QUESTION.replace("Study Q:", "both papers:"),
    QUESTION.split("Explain")[0] + "Explain why it is wrong, citing the paper.",
    QUESTION.split("Explain")[0] + "Explain why the claim is wrong, citing the paper.",
    QUESTION.split("Explain")[0] + "Explain these sampling, transmission and processing times.",
    QUESTION.split("Explain")[0] + "Explain sampling, transmission and processing times in Study R.",
    QUESTION.split("Explain")[0] + "Explain sampling, transmission and processing times from another paper.",
    QUESTION.replace("processing times", "processing times only if the network is clear"),
    QUESTION + " Explain another experiment.",
    QUESTION + " " + "x" * 900,
    QUESTION + " " + "x" * 9000,
])
def test_ambiguous_conditional_multi_document_or_non_audit_questions_fail_closed(question):
    assert quoted_audit_request(question) is None
    assert not any(row.get("purpose") == "quoted_claim_audit" for row in plan_for(question)["facets"])


def test_ambiguous_document_metadata_disables_even_a_single_literal_paper_name():
    chunks = METADATA + [{**METADATA[0], "id": "other", "document_id": "different-doc"}]
    plan = build_query_facets(QUESTION, chunks)
    assert not any(row.get("purpose") == "quoted_claim_audit" for row in plan["facets"])


def test_paid_pair_budget_preserves_source_identity_bytes_and_original_full_query():
    plan, rows, pairs = fixture()
    original_rows = {row["id"]: row for row in rows}
    full = [row for row in pairs if row.get("rerank_branch") == "full"]
    audit = [row for row in pairs if row.get("rerank_branch") == "quoted_audit"]
    assert len(pairs) == 48 and len(full) == 36 and len(audit) == 12
    assert all(row["query"] == QUESTION for row in full)
    composed = quoted_audit_request(QUESTION)["text"]
    suffix = QUESTION[slice(*quoted_audit_request(QUESTION)["request_span"])]
    assert all(row["query"] == composed and row["query"] != suffix for row in audit)
    assert all(row["rerank_query_id"] == "rq_" + hashlib.sha256(composed.encode()).hexdigest()[:16]
               for row in audit)
    assert all(row["rerank_view_of"] == plan["facets"][0]["id"] for row in audit)
    for row in pairs:
        original = original_rows[row["id"].split("::")[0]]
        for key in ("document_id", "source_id", "content", "_source_content", "excerpt_start", "excerpt_end"):
            assert row[key] == original[key]
    assert all(row["facet_scores"] == original_rows[row["id"].split("::")[0]]["facet_scores"] for row in audit)


def test_audit_view_does_not_forge_full_query_nomination():
    plan, rows, _ = fixture(6)
    view = next(row for row in plan["facets"] if row.get("query_only"))
    for row in rows:
        row["facet_scores"] = {view["id"]: .8}
    pairs = _rerank_pairs(QUESTION, rows, plan["facets"])
    assert all(row["retrieval_support"] == 0 for row in pairs if row.get("rerank_branch") == "full")
    result, _ = select(plan, pairs, receipts={row["id"]: receipt(row) for row in pairs})
    assert result["sources"] == []  # No eligible full branch to own the shared slot.


def test_existing_independent_focus_owns_the_budget_without_extra_audit_branch():
    plan, rows, _ = fixture()
    clauses = [
        {"id": "mechanism", "kind": "clause", "text": "Explain how the chamber controller handles pressure", "weight": 1., "roles": []},
        {"id": "experiment", "kind": "clause", "text": "Describe the calibration experiment", "weight": 1., "roles": []},
    ]
    plan["facets"] = plan["facets"][:2] + clauses
    for row in rows:
        row["facet_scores"].update(mechanism=.8, experiment=.8)
    pairs = _rerank_pairs(QUESTION, rows, plan["facets"])
    assert any(row.get("rerank_branch") == "focused" for row in pairs)
    assert not any(row.get("rerank_branch") == "quoted_audit" for row in pairs)
    assert len(pairs) <= 48


def test_insufficient_view_nominees_do_not_create_a_tiny_champion_branch():
    plan, rows, _ = fixture(3)
    pairs = _rerank_pairs(QUESTION, rows, plan["facets"])
    assert len(pairs) == 3 and not any(row.get("rerank_branch") == "quoted_audit" for row in pairs)


def test_audit_reserved_source_earns_no_independent_requirement_or_anchor_credit():
    plan, _, pairs = fixture(6)
    before = deepcopy(pairs)
    scores = {row["id"]: float(row["source_id"].split("-")[-1]) *
              (1 if row.get("rerank_branch") == "quoted_audit" else -1) for row in pairs}
    events = []
    result = select_evidence(pairs, scores, plan, [], max_sources=1,
                             input_receipts={row["id"]: receipt(row) for row in pairs}, diagnostic_callback=events.append)
    trace = result["trace"]
    assert len(result["sources"]) == trace["quoted_audit_selected_sources"] == 1
    assert trace["focused_reservation_slots"] == trace["focused_source_limit"] == 1
    assert trace["independent_requirement_ids"] == [plan["facets"][0]["id"]]
    # The paired full submission legitimately earns its original relevance
    # credit; the audit submission adds no independent coverage dimension.
    assert set(trace["facet_coverage"].values()) == {.8}
    assert set(trace["anchor_coverage"].values()) <= {0.}
    assert trace["quoted_audit_requirement_credit"] is False
    assert trace["selection_proxy_is_verified_support"] is False
    audit_events = [event for event in events if event["event"] == "selection_candidate" and event.get("rerank_branch") == "quoted_audit"]
    assert audit_events and all(event["coverage"] == {} and event["anchor_proxy"] == {} for event in audit_events)
    assert all(event["credited_facet"] is None for event in audit_events)
    assert pairs == before


@pytest.mark.parametrize("failure", [
    "missing_receipt", "query_truncated", "passage_truncated", "wrong_submitted_hash", "wrong_actual_hash",
    "wrong_query_end", "wrong_scope_receipt", "over_budget", "bad_receipt_type", "wrong_passage_hash", "stale_passage_receipt",
    "query_mismatch", "query_hash", "mapping", "mapping_span", "query_spans_missing", "query_prefix_span",
    "query_request_span", "query_spans_contiguous", "query_spans_reversed", "suffix_receipt", "view_text", "view_id",
    "original_id", "alias", "translation", "query_only", "new_need", "wrong_document", "different_scope",
    "bad_scope_type", "source_identity", "content_identity", "view_nomination_zero", "forged_support",
    "alias_nomination", "full_branch", "unscored",
])
def test_invalid_view_or_observed_receipt_cannot_gain_reservation_or_coverage(failure):
    plan, _, pairs = fixture(6)
    full = next(row for row in pairs if row.get("rerank_branch") == "full")
    row = next(row for row in pairs if row.get("rerank_branch") == "quoted_audit")
    view = next(facet for facet in plan["facets"] if facet.get("purpose") == "quoted_claim_audit")
    receipts = {row["id"]: receipt(row)}
    observed = receipts[row["id"]]
    if failure == "missing_receipt": receipts.clear()
    if failure == "query_truncated": observed["query_complete"] = False
    if failure == "passage_truncated": observed["passage_complete"] = False
    if failure == "wrong_submitted_hash": observed["submitted_query_sha256"] = "0" * 64
    if failure == "wrong_actual_hash": observed["actual_query_sha256"] = "0" * 64
    if failure == "wrong_query_end": observed["actual_query_character_end"] -= 1
    if failure == "wrong_scope_receipt": observed["input_scope"] = "passage_prefix"
    if failure == "over_budget": observed["pair_tokens_before_truncation"] = 513
    if failure == "bad_receipt_type": receipts[row["id"]] = []
    if failure == "wrong_passage_hash": observed["actual_passage_sha256"] = "0" * 64
    if failure == "stale_passage_receipt":
        row["content"] = row["_source_content"] = "New sampling transmission processing material replaces the observed passage."
        row["excerpt_end"] = row["original_content_length"] = len(row["content"])
        identity = f"{row['document_id']}\0{row['source_id']}\0{row['excerpt_start']}:{row['excerpt_end']}\0{row['content']}"
        row["id"] = "ew_" + hashlib.sha256(identity.encode()).hexdigest()[:20] + "::" + view["id"]
        receipts = {row["id"]: observed}
    if failure == "query_mismatch": row["query"] += " unrelated"
    if failure == "query_hash": row["rerank_query_id"] = "rq_forged"
    if failure == "mapping": row["rerank_view_of"] = "missing"
    if failure == "mapping_span": view["request_span"][0] += 1
    if failure == "query_spans_missing": view.pop("query_spans")
    if failure == "query_prefix_span": view["query_spans"][0][1] -= 1
    if failure == "query_request_span": view["query_spans"][1][0] += 1
    if failure == "query_spans_contiguous": view["query_spans"] = [[0, len(QUESTION)]]
    if failure == "query_spans_reversed": view["query_spans"].reverse()
    if failure == "suffix_receipt":
        suffix = QUESTION[slice(*view["request_span"])]
        observed.update(submitted_query_sha256=hashlib.sha256(suffix.encode()).hexdigest(),
                        actual_query_sha256=hashlib.sha256(suffix.encode()).hexdigest(),
                        actual_query_character_end=len(suffix))
    if failure == "view_text": view["text"] = "Explain a completely different source requirement."
    if failure == "view_id": view["id"] = "forged"
    if failure == "original_id": plan["facets"][0]["id"] = "forged"
    if failure == "alias": view["query_alias_of"] = plan["facets"][0]["id"]
    if failure == "translation": view["derived_translation"] = True
    if failure == "query_only": view["query_only"] = False
    if failure == "new_need": view["counts_as_requirement"] = True
    if failure == "wrong_document": row["document_id"] = "another-document"
    if failure == "different_scope": view["document_ids"] = ["another-document"]
    if failure == "bad_scope_type": view["document_ids"] = [["study-q"]]
    if failure == "source_identity": row["source_id"] = "forged-source"
    if failure == "content_identity": row["content"] = row["_source_content"] = "The newly substituted content is different."
    if failure == "view_nomination_zero": row["facet_scores"][view["id"]] = 0
    if failure == "forged_support": row["retrieval_support"] = .01
    if failure == "alias_nomination": row["nomination_facet_ids"] = [view["id"]]
    if failure == "full_branch": row["rerank_branch"] = "full"
    pairs = [full] if failure == "unscored" else [full, row]
    result, _ = select(plan, pairs, cap=1, receipts=receipts)
    assert result["trace"]["quoted_audit_selected_sources"] == 0
    assert result["trace"]["quoted_audit_branch_enabled"] is False
    # A mismatched authoritative snapshot invalidates the whole same-source
    # set, including full, instead of guessing which supplied version is real.
    assert len(result["sources"]) == (0 if failure in {"original_id", "stale_passage_receipt"} else 1)


@pytest.mark.parametrize("scale,shift", [(1., 0.), (.2, -300.), (3., 100.)])
def test_one_view_source_hard_caps_and_negative_logits_are_rank_only(scale, shift):
    plan, rows, pairs = fixture()
    receipts = {row["id"]: receipt(row) for row in pairs}
    result, _ = select(plan, pairs, receipts=receipts, scale=scale, shift=shift)
    baseline, _ = select(plan, pairs, receipts=receipts)
    assert result["sources"] == baseline["sources"]
    assert len(result["sources"]) <= 8
    assert result["trace"]["quoted_audit_selected_sources"] == 0  # Identical ranks have no rescue gain.
    assert result["trace"]["source_bytes"] <= 12000
    original_sources = {row["source_id"]: row for row in rows}
    assert all(source["content"] == original_sources[source["id"]]["content"] for source in result["sources"])


@pytest.mark.parametrize("audit_first", [False, True])
def test_full_geometry_cannot_borrow_cached_audit_receipts_during_source_set_repair(audit_first):
    plan, _, pairs = fixture(20)
    pairs.sort(key=lambda row: (row.get("rerank_branch") != "quoted_audit") if audit_first else
               (row.get("rerank_branch") == "quoted_audit"))
    scores = {row["id"]: float(row["source_id"].split("-")[-1]) *
              (1 if row.get("rerank_branch") == "quoted_audit" else -1) for row in pairs}
    events = []
    result = select_evidence(pairs, scores, plan, [], input_receipts={row["id"]: receipt(row) for row in pairs},
                             diagnostic_callback=events.append)
    credits = next(event for event in events if event["event"] == "final_window_credits")
    pair_by_id = {row["id"]: row for row in pairs}
    quote_by_source = {quote["source_id"]: pair_by_id[quote["id"]] for quote in credits["final_quotes"]}
    audit_credit_sources = set()
    for window in credits["windows"]:
        scored = pair_by_id[window["id"]]
        final_quote = quote_by_source[window["source_id"]]
        if scored.get("rerank_branch") == "quoted_audit":
            audit_credit_sources.add(window["source_id"])
            assert final_quote.get("rerank_branch") == "quoted_audit"
    assert audit_credit_sources == {"source-11"}
    assert result["trace"]["quoted_audit_selected_sources"] == 1
    # The same-geometry audit score for source-10 must not evict full source-6.
    assert {row["id"] for row in result["sources"]} == {"source-11", *(f"source-{index}" for index in range(7))}


def test_private_observations_include_actual_query_and_receipt_but_public_trace_does_not():
    class Semantic:
        def semantic_scores(self, question, chunks, *, session_id):
            return {"scores": {row["id"]: .85 for row in chunks},
                    "spans": {row["id"]: [0, len(row["content"])] for row in chunks}, "trace": {"status": "ready"}}

    class Reranker:
        def rank_pairs(self, candidates, *, session_id):
            self.pairs = deepcopy(candidates)
            return {"scores": {row["id"]: -float(index) for index, row in enumerate(candidates)},
                    "input_receipts": {row["id"]: receipt(row) for row in candidates}, "trace": {"status": "ready"}}

    _, rows, _ = fixture(8)
    chunks = [{"id": row["source_id"], "document_id": "study-q", "title": "Study Q", "filename": "study-q.pdf",
               "content": row["content"], "ordinal": index} for index, row in enumerate(rows)]
    events, scorer = [], Reranker()
    result = build_faceted_assistance(QUESTION, [], chunks, Semantic(), scorer, session_id="synthetic::audit",
                                      diagnostic_callback=events.append)
    assert result["trace"]["pair_allocation"]["quoted_audit_pairs"] >= 4
    audited = [event for event in events if event["event"] == "rerank_result" and event.get("rerank_view") == "quoted_claim_audit"]
    assert audited and all(event["input_receipt"]["actual_query_sha256"] == hashlib.sha256(event["query"].encode()).hexdigest()
                           for event in audited)
    public = json.dumps(result["trace"])
    for event in audited:
        assert event["query"] not in public and event["query_id"] not in public
        assert event["input_receipt"]["actual_query_sha256"] not in public
        assert event["input_receipt"]["actual_passage_sha256"] not in public
    assert "input_receipt" not in public and "submitted_query_sha256" not in public

