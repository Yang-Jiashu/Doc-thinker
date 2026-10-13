"""Paired audit rank shifts are source-selection heuristics, not fact proof."""
from copy import deepcopy
import hashlib
import json

import pytest

from docthinker.server.research_evidence_selection import _window_row, select_evidence
from docthinker.server.research_faceted_assist import _rerank_pairs
from tests.test_research_quoted_audit_unit import QUESTION, fixture, receipt


def run(full_scores, audit_scores, *, cap=8, pairs=None, receipts=None, plan=None):
    if pairs is None:
        plan, _, pairs = fixture(len(full_scores))
    scores = {row["id"]: (audit_scores if row.get("rerank_branch") == "quoted_audit" else full_scores)[
        int(row["source_id"].split("-")[-1])] for row in pairs}
    if receipts is None:
        receipts = {row["id"]: receipt(row) for row in pairs}
    events = []
    result = select_evidence(pairs, scores, plan, [], max_sources=cap, input_receipts=receipts,
                             diagnostic_callback=events.append)
    return result, events


def shifts(events):
    return {event["window"]["source_id"]: (event["full_common_rank"], event["audit_common_rank"],
                                          event["rank_gain"], event["eligible"])
            for event in events if event["event"] == "audit_paired_rank"}


def reserved(events):
    credits = next(event for event in events if event["event"] == "final_window_credits")
    audit_ids = {event["window"]["id"] for event in events
                 if event["event"] == "selection_candidate" and event.get("rerank_branch") == "quoted_audit"}
    return {quote["source_id"] for quote in credits["final_quotes"] if quote["id"] in audit_ids}


def test_tail_rank_gain_loses_to_positive_upper_half_gain_on_fixed_common_set():
    # In this invented eight-source set, the tail improves by two positions
    # but remains below the median; the smaller upper-half gain owns the slot.
    full = list(range(8, 0, -1))
    audit = [8, 7, 6, 4, 5, 2, 1, 3]
    result, events = run(full, audit)
    assert shifts(events)["source-4"] == (5, 4, 1, True)
    assert shifts(events)["source-7"] == (8, 6, 2, False)
    assert reserved(events) == {"source-4"}
    assert result["trace"]["quoted_audit_common_sources"] == 8
    assert result["trace"]["quoted_audit_eligible_sources"] == 1
    assert result["trace"]["selection_proxy_is_verified_support"] is False


def test_gain_precedes_audit_rank_and_repair_cannot_replace_with_lower_gain():
    full = list(range(8, 0, -1))
    audit = [8, 5, 4, 7, 2, 6, 1, 3]
    _, events = run(full, audit)
    assert shifts(events)["source-3"] == (4, 2, 2, True)
    assert shifts(events)["source-5"] == (6, 3, 3, True)
    assert reserved(events) == {"source-5"}
    assert not any(event["event"] == "source_set_replacement"
                   and event["removed"]["source_id"] == "source-5" for event in events)


@pytest.mark.parametrize("full,audit", [
    ([8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 4, 3, 2, 1]),
    ([1] * 8, [1] * 8),
    ([1] * 8, [8, 7, 6, 5, 4, 3, 2, 1]),
    ([4, 3, 2], [2, 3, 4]),
])
def test_no_positive_shift_or_insufficient_independent_sources_means_no_reservation(full, audit):
    result, events = run(full, audit)
    assert reserved(events) == set()
    assert not result["trace"]["quoted_audit_branch_enabled"]
    assert result["trace"]["quoted_audit_eligible_sources"] == 0
    assert result["trace"]["focused_reservation_slots"] == 0


def test_competition_ties_do_not_get_source_id_or_dense_rank_advantage():
    _, events = run([9, 9, 8, 7, 6, 5], [9, 9, 8, 7, 6, 5])
    assert shifts(events)["source-0"][:3] == (1, 1, 0)
    assert shifts(events)["source-1"][:3] == (1, 1, 0)
    assert shifts(events)["source-2"][:3] == (3, 3, 0)
    assert reserved(events) == set()


@pytest.mark.parametrize("branch", ["full", "quoted_audit"])
@pytest.mark.parametrize("failure", ["missing", "query_incomplete", "passage_incomplete", "query_hash", "passage_hash", "query_end", "scope"])
def test_each_pair_side_requires_its_actual_complete_receipt(branch, failure):
    plan, _, pairs = fixture(4)
    receipts = {row["id"]: receipt(row) for row in pairs}
    target = next(row for row in pairs if row.get("rerank_branch") == branch and row["source_id"] == "source-0")
    value = receipts[target["id"]]
    if failure == "missing": receipts.pop(target["id"])
    if failure == "query_incomplete": value["query_complete"] = False
    if failure == "passage_incomplete": value["passage_complete"] = False
    if failure == "query_hash": value["actual_query_sha256"] = "0" * 64
    if failure == "passage_hash": value["submitted_passage_sha256"] = "0" * 64
    if failure == "query_end": value["actual_query_character_end"] -= 1
    if failure == "scope": value["input_scope"] = "passage_prefix"
    result, events = run([4, 3, 2, 1], [1, 2, 3, 4], plan=plan, pairs=pairs, receipts=receipts)
    assert result["trace"]["quoted_audit_common_sources"] == 3
    assert reserved(events) == set()


@pytest.mark.parametrize("failure", ["missing_full", "wrong_full_query", "full_alias", "full_window", "wrong_document", "unscored_full"])
def test_unmatched_full_original_or_geometry_cannot_join_common_population(failure):
    plan, _, pairs = fixture(4)
    target = next(row for row in pairs if row.get("rerank_branch") == "full" and row["source_id"] == "source-0")
    if failure in {"missing_full", "unscored_full"}: pairs.remove(target)
    if failure == "wrong_full_query":
        target["query"] = "An unrelated complete query"
        target["rerank_query_id"] = "rq_" + hashlib.sha256(target["query"].encode()).hexdigest()[:16]
    if failure == "full_alias": target["rerank_facet_id"] = next(row for row in plan["facets"] if row.get("kind") == "technical_phrase")["id"]
    if failure == "full_window":
        target["excerpt_start"] += 1
        target["content"] = target["content"][1:]
    if failure == "wrong_document": target["document_id"] = "elsewhere"
    result, events = run([4, 3, 2, 1], [1, 2, 3, 4], plan=plan, pairs=pairs)
    assert result["trace"]["quoted_audit_common_sources"] <= 3
    assert reserved(events) == set()


def test_present_full_pair_with_receipt_but_no_actual_score_does_not_match():
    plan, _, pairs = fixture(4)
    missing = next(row["id"] for row in pairs if row.get("rerank_branch") == "full" and row["source_id"] == "source-0")
    scores = {row["id"]: float(index) for index, row in enumerate(pairs) if row["id"] != missing}
    result = select_evidence(pairs, scores, plan, [], input_receipts={row["id"]: receipt(row) for row in pairs})
    assert result["trace"]["rerank_missing_or_invalid"] == 1
    assert result["trace"]["quoted_audit_common_sources"] == 3
    assert result["trace"]["quoted_audit_selected_sources"] == 0


def test_four_matched_windows_from_one_source_are_not_four_independent_sources():
    plan, rows, _ = fixture(4)
    content = "\n".join(row["content"] for row in rows)
    source = {"id": "shared", "document_id": "study-q", "filename": "study-q.pdf", "ordinal": 0, "content": content}
    windows, cursor = [], 0
    for row in rows:
        end = cursor + len(row["content"])
        window = _window_row(source, cursor, end, ["mechanism"], "Methods")
        window.update(facet_scores=row["facet_scores"], anchor_scores={}, pool_score=.8)
        windows.append(window)
        cursor = end + 1
    pairs = _rerank_pairs(QUESTION, windows, plan["facets"])
    assert sum(row.get("rerank_branch") == "quoted_audit" for row in pairs) == 4
    scores = {row["id"]: float(row["excerpt_start"]) *
              (1 if row.get("rerank_branch") == "quoted_audit" else -1) for row in pairs}
    result = select_evidence(pairs, scores, plan, [], input_receipts={row["id"]: receipt(row) for row in pairs})
    assert result["trace"]["quoted_audit_common_sources"] == 1
    assert result["trace"]["quoted_audit_eligible_sources"] == 0
    assert result["trace"]["quoted_audit_selected_sources"] == 0


def test_order_exact_duplicates_and_unmatched_audit_do_not_change_common_ranks():
    plan, _, pairs = fixture(9)
    full, audit = list(range(9, 0, -1)), [9, 8, 7, 5, 6, 3, 2, 4, 1000]
    # Source 8 has no full pair and an arbitrarily high audit logit.
    pairs = [row for row in pairs if not (row["source_id"] == "source-8" and row.get("rerank_branch") == "full")]
    base_result, base_events = run(full, audit, plan=plan, pairs=pairs)
    expected = shifts(base_events)
    variants = [list(reversed(pairs)), pairs + [deepcopy(row) for row in pairs],
                [row for row in pairs if row["source_id"] != "source-8"]]
    for changed in variants:
        result, events = run(full, audit, plan=plan, pairs=changed)
        assert shifts(events) == expected
        assert reserved(events) == reserved(base_events) == {"source-4"}
        assert result["trace"]["quoted_audit_common_sources"] == 8
    assert base_result["trace"]["quoted_audit_eligible_sources"] == 1


@pytest.mark.parametrize("full_scale,full_shift,audit_scale,audit_shift", [
    (1., 0., 1., 0.), (.1, -300., 9., 500.), (7., 100., .05, -1000.),
])
def test_separate_positive_affine_transforms_preserve_the_rescue(full_scale, full_shift, audit_scale, audit_shift):
    full = [value * full_scale + full_shift for value in range(8, 0, -1)]
    audit = [value * audit_scale + audit_shift for value in [8, 7, 6, 4, 5, 2, 1, 3]]
    result, events = run(full, audit)
    assert reserved(events) == {"source-4"}
    assert shifts(events)["source-4"] == (5, 4, 1, True)
    assert result["trace"]["source_bytes"] <= 12000 and len(result["sources"]) <= 8


def test_full_ranks_quality_and_coverage_are_not_overwritten_by_common_set_fields():
    plan, _, pairs = fixture(8)
    full, audit = list(range(8, 0, -1)), [8, 7, 6, 4, 5, 2, 1, 3]
    _, paired_events = run(full, audit, plan=plan, pairs=pairs)
    _, baseline_events = run(full, audit, plan=plan, pairs=[row for row in pairs if row.get("rerank_branch") == "full"])
    def full_fields(events):
        return {event["window"]["id"]: (event["rank"], event["quality"], event["coverage"])
                for event in events if event["event"] == "selection_candidate" and event.get("rerank_branch") == "full"}
    assert full_fields(paired_events) == full_fields(baseline_events)
    for event in paired_events:
        if event["event"] == "selection_candidate" and event.get("rerank_branch") == "full":
            assert event["audit_rank_gain"] is None


def test_ineligible_audits_do_not_reappear_in_final_receipts_or_public_diagnostics():
    result, events = run(list(range(8, 0, -1)), [8, 7, 6, 4, 5, 2, 1, 3])
    eligible_ids = {event["window"]["id"] for event in events if event["event"] == "audit_paired_rank" and event["eligible"]}
    credits = next(event for event in events if event["event"] == "final_window_credits")
    assert all("::" not in window["id"] or window["id"] in eligible_ids for window in credits["windows"])
    full_quote_sources = {quote["source_id"] for quote in credits["final_quotes"] if "::" not in quote["id"]}
    assert all("::" not in window["id"] or window["source_id"] not in full_quote_sources for window in credits["windows"])
    public = json.dumps(result["trace"])
    assert "audit_paired_rank" not in public and "full_common_rank" not in public
    assert result["trace"]["quoted_audit_selected_sources"] == 1
