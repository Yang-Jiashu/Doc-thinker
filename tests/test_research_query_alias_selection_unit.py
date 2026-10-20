"""Translated query nomination never substitutes for an actual literal pair."""
from copy import deepcopy
import hashlib

import pytest

from docthinker.server.research_evidence_selection import select_evidence
from tests.test_research_selection_regressions_unit import _candidate, _facet


def fixture():
    literal = _facet("check", "是否已经完成传感器联合训练", docs=("aster",))
    another = _facet("other", "Explain the chamber calibration", docs=("aster",))
    full = {**_facet("full", "Explain calibration and check joint sensor training", docs=("aster",)), "kind": "original"}
    alias = {**_facet("alias", "whether joint sensor training has been completed", docs=("aster",)),
             "kind": "clause", "derived_translation": True, "query_only": True,
             "translation_full_query": False, "query_alias_of": "check",
             "alias_alignment": "unique_query_only_requirement_kind"}
    baseline = {**_candidate("baseline", "Quartz chambers calibrate a pressure gauge against reference stations.", full),
                "rerank_branch": "full"}
    focused = {**_candidate("focused", "Joint sensor training remains planned work and was not implemented.", literal, nomination=0),
               "rerank_branch": "focused", "facet_scores": {"alias": .8},
               "nomination_facet_ids": ["alias"], "retrieval_support": .8,
               "nomination_alignment": "unique_query_only_requirement_kind"}
    return [baseline, focused], {baseline["id"]: 3., focused["id"]: 2.}, [full, literal, another, alias]


def select(rows, scores, facets):
    events = []
    result = select_evidence(rows, scores, {"facets": facets}, [], diagnostic_callback=events.append)
    return result, events


def test_only_actual_literal_pair_earns_rank_conditioned_alias_nomination_proxy():
    rows, scores, facets = fixture()
    untouched = deepcopy(rows)
    result, events = select(rows, scores, facets)
    assert {row["id"] for row in result["sources"]} == {"baseline", "focused"}
    assert result["trace"]["facet_coverage"]["check"] == .6
    assert result["trace"]["facet_coverage"]["other"] == 0
    assert "alias" not in result["trace"]["facet_coverage"]
    assert result["trace"]["query_alias_nominated_credits"] == 1
    assert result["trace"]["independent_requirement_ids"] == ["check", "other"]
    assert result["trace"]["scored_query_view_ids"] == ["full"]
    assert "alias" not in result["trace"]["uncovered_facets"]
    assert rows == untouched and "check" not in rows[1]["facet_scores"]
    receipt = next(row for row in events if row["event"] == "selection_candidate" and row["window"]["id"] == rows[1]["id"])
    assert receipt["nomination_facet_ids"] == ["alias"]
    assert receipt["coverage_semantics"] == "scored_literal_rank_times_alias_nomination_not_verified_support"


@pytest.mark.parametrize("failure", [
    "query_mismatch", "unscored", "full_branch", "different_need", "missing_alias",
    "not_translation", "not_query_only", "full_query_alias", "wrong_mapping", "wrong_alignment",
    "different_scope", "wrong_document", "zero_alias_support", "forged_support", "bad_alias_ids",
])
def test_invalid_alias_receipts_cannot_rescue_unsupported_literal_window(failure):
    rows, scores, facets = fixture()
    row, alias = rows[1], facets[-1]
    if failure == "query_mismatch": row["query"] = "An entirely different question"
    if failure == "unscored": scores.pop(row["id"])
    if failure == "full_branch": row["rerank_branch"] = "full"
    if failure == "different_need": row["rerank_facet_id"] = "other"
    if failure == "missing_alias": facets.pop()
    if failure == "not_translation": alias["derived_translation"] = False
    if failure == "not_query_only": alias["query_only"] = False
    if failure == "full_query_alias": alias["translation_full_query"] = True
    if failure == "wrong_mapping": alias["query_alias_of"] = "other"
    if failure == "wrong_alignment": row["nomination_alignment"] = "some_other_association"
    if failure == "different_scope": alias["document_ids"] = ["birch"]
    if failure == "wrong_document": row["document_id"] = "birch"
    if failure == "zero_alias_support": row["facet_scores"]["alias"] = 0
    if failure == "forged_support": row["retrieval_support"] = 1.
    if failure == "bad_alias_ids": row["nomination_facet_ids"] = [["alias"]]
    result, _ = select(rows, scores, facets)
    assert [item["id"] for item in result["sources"]] == ["baseline"]
    assert result["trace"]["facet_coverage"]["check"] == 0
    assert result["trace"]["query_alias_nominated_credits"] == 0


def test_alias_nomination_is_not_an_orphaned_focused_branch():
    rows, scores, facets = fixture()
    result, _ = select(rows[1:], scores, facets)
    assert result["sources"] == []
    assert result["trace"]["query_alias_nominated_credits"] == 0


@pytest.mark.parametrize("full_translation", [True, False])
def test_actually_scored_translated_clause_remains_query_view_without_creating_a_need(full_translation):
    rows, _, facets = fixture()
    alias = facets[-1]
    alias["translation_full_query"] = full_translation
    translated_pair = {**_candidate("translated", "The quartz chamber computes calibration offsets before each observation.", alias),
                       "rerank_branch": "full"}
    result, _ = select([translated_pair], {translated_pair["id"]: 2.}, facets)
    assert [row["id"] for row in result["sources"]] == ["translated"]
    assert result["trace"]["independent_requirement_ids"] == ["check", "other"]
    assert result["trace"]["scored_query_view_ids"] == ["alias"]
    assert result["trace"]["query_view_coverage"] == {"alias": .8}
    assert result["trace"]["uncovered_facets"] == ["check", "other"]
    assert result["trace"]["facet_coverage"]["check"] == 0
    assert result["trace"]["independent_document_requirements"] == 2


def test_translation_views_do_not_hide_original_need_when_no_literal_clauses_exist():
    _, _, facets = fixture()
    full, alias = facets[0], facets[-1]
    pair = _candidate("translated", "The chamber calibrates humidity before each measurement.", alias)
    result, _ = select([pair], {pair["id"]: 1.}, [full, alias])
    assert result["trace"]["independent_requirement_ids"] == ["full"]
    assert result["trace"]["uncovered_facets"] == ["full"]
    assert result["trace"]["query_view_coverage"] == {"alias": .8}


def scored_alias_fixture(native_support=False):
    rows, scores, facets = fixture()
    row, alias = rows[1], facets[-1]
    row.update(query=alias["text"], rerank_facet_id=alias["id"], rerank_view_of="check",
               rerank_view="guarded_alias",
               rerank_query_id="rq_" + hashlib.sha256(alias["text"].encode()).hexdigest()[:16])
    if native_support:
        row["facet_scores"]["check"] = .7
        row["retrieval_support"] = .7
        row.pop("nomination_facet_ids")
        row.pop("nomination_alignment")
    return rows, scores, facets


@pytest.mark.parametrize("native_support", [False, True])
def test_guarded_alias_pair_is_credited_as_a_view_of_one_need_not_literal_scoring(native_support):
    rows, scores, facets = scored_alias_fixture(native_support)
    untouched = deepcopy(rows)
    result, events = select(rows, scores, facets)
    assert {row["id"] for row in result["sources"]} == {"baseline", "focused"}
    expected = .525 if native_support else .6
    assert result["trace"]["facet_coverage"]["check"] == expected
    assert result["trace"]["facet_coverage"]["other"] == 0
    assert result["trace"]["query_view_coverage"]["alias"] == expected
    assert result["trace"]["query_alias_nominated_credits"] == 0
    assert result["trace"]["translated_query_view_credits"] == 1
    receipt = next(row for row in events if row["event"] == "selection_candidate" and row["window"]["id"] == rows[1]["id"])
    assert receipt["actual_scored_facet"] == "alias" and receipt["credited_facet"] == "check"
    assert receipt["coverage_semantics"] == "scored_query_view_rank_times_nomination_not_verified_support"
    assert rows == untouched


@pytest.mark.parametrize("field,value", [
    ("rerank_view", "unverified_alias"), ("rerank_view", "literal"),
    ("rerank_view_of", "other"), ("rerank_view_of", None),
    ("rerank_query_id", "forged_query_identity"), ("query", "Whether another experiment succeeded"),
    ("query", None), ("rerank_branch", "full"), ("document_id", "birch"),
])
def test_query_view_mapping_mismatch_cannot_fallback_to_literal_or_alias_credit(field, value):
    rows, scores, facets = scored_alias_fixture()
    rows[1][field] = value
    result, events = select(rows, scores, facets)
    assert [row["id"] for row in result["sources"]] == ["baseline"]
    assert result["trace"]["facet_coverage"]["check"] == 0
    assert result["trace"]["translated_query_view_credits"] == 0
    assert any(row.get("reason") == "invalid_scored_query_view_mapping" for row in events)


def test_explicit_literal_view_keeps_literal_identity_and_native_support():
    rows, scores, facets = fixture()
    rows[1].update(rerank_view="literal", rerank_view_of="check", facet_scores={"check": .8}, retrieval_support=.8)
    result, events = select(rows, scores, facets)
    assert result["trace"]["facet_coverage"]["check"] == .8
    assert result["trace"]["translated_query_view_credits"] == 0
    receipt = next(row for row in events if row["event"] == "selection_candidate" and row["window"]["id"] == rows[1]["id"])
    assert receipt["actual_scored_facet"] == receipt["credited_facet"] == "check"
