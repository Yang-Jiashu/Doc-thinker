"""Generic regressions for evidence lost after retrieval, not paper quality tests.

The fictional passages deliberately have independent wording.  These tests
exercise eligibility, ordinal ranking, redundancy and exact source spans; a
passing test does not establish factual sufficiency or answer quality.
"""

from __future__ import annotations

import hashlib

import pytest

from docthinker.server.research_evidence_selection import select_evidence
from docthinker.server.research_query_facets import build_query_facets


_PASSAGES = (
    "Aster telemetry uses sealed quartz chambers. Operators calibrate a pressure "
    "gauge against an independent reference before each observation. They record "
    "ambient humidity, room temperature and instrument serial numbers in a ledger. "
    "A copper enclosure shields the sensor from external electromagnetic fields. ",
    "Aster telemetry estimates drift with a sparse transition matrix. Training "
    "minimizes the squared residual between successive measurements. The optimizer "
    "updates coefficient vectors while retaining the original lookup table. "
    "A separate validation partition determines the stopping epoch. ",
    "Aster telemetry was evaluated on coastal stations during autumn storms. "
    "Researchers compared hourly forecasts with manually verified observations. "
    "Missing samples were excluded using a preregistered availability rule. "
    "Performance declined at remote locations beyond the equipment's operating range. ",
)


def _text(index: int, repetitions: int = 3) -> str:
    return (_PASSAGES[index] * repetitions).strip()


def _facet(identifier="need", text="Explain how Aster telemetry works", *, docs=()):
    return {"id": identifier, "text": text, "kind": "clause", "weight": 1.,
            "roles": ["mechanism"], "document_ids": list(docs)}


def _candidate(identifier, content, facet, *, nomination=.8, document="aster",
               source_id=None, source_content=None, start=0, end=None):
    source_content = content if source_content is None else source_content
    end = start + len(content) if end is None else end
    return {
        "id": "window_" + identifier, "source_id": source_id or identifier,
        "document_id": document, "filename": document + ".pdf", "ordinal": 0,
        "content": content, "_source_content": source_content,
        "excerpt_start": start, "excerpt_end": end,
        "original_content_length": len(source_content),
        "offset_unit": "characters_in_source_chunk", "sentence_boundary_preserved": True,
        "section": "Methods", "roles": ["mechanism"],
        "facet_scores": {facet["id"]: nomination} if nomination > 0 else {},
        "lexical_scores": {facet["id"]: nomination} if nomination > 0 else {},
        "semantic_facets": [facet["id"]] if nomination > 0 else [],
        "anchor_scores": {}, "pool_score": nomination, "fragment_penalty": 1.,
        "query": facet["text"], "rerank_facet_id": facet["id"],
        "rerank_query_id": "rq_" + hashlib.sha256(facet["text"].encode()).hexdigest()[:16],
    }


def _select(candidates, scores, facets, **limits):
    return select_evidence(candidates, scores, {"facets": facets}, [], **limits)


def _ids(result):
    return {source["id"] for source in result["sources"]}


def test_high_rerank_complement_is_not_multiplied_by_weak_nomination_and_byte_tax():
    facet = _facet()
    overview = _candidate("overview", _text(0), facet, nomination=.9)
    detail = _candidate("detail", _text(1), facet, nomination=.14)
    result = _select([overview, detail], {overview["id"]: 4., detail["id"]: 3.5}, [facet])
    assert _ids(result) == {"overview", "detail"}
    assert sum(len(row["content"].encode()) for row in result["sources"]) < 12_000


def test_third_independent_relevant_passage_is_not_removed_by_fixed_depth_saturation():
    facet = _facet()
    candidates = [_candidate(str(index), _text(index, 2), facet) for index in range(3)]
    scores = {row["id"]: 4. - .2 * index for index, row in enumerate(candidates)}
    assert _ids(_select(candidates, scores, [facet])) == {"0", "1", "2"}


@pytest.mark.parametrize("scale,shift", [(1., 25.), (4., 0.), (.2, -30.)])
def test_ordinal_selection_of_independently_supported_passages_is_affine_invariant(scale, shift):
    facet = _facet()
    candidates = [_candidate(str(index), _text(index), facet) for index in range(3)]
    scores = {row["id"]: 4. - index for index, row in enumerate(candidates)}
    baseline = _select(candidates, scores, [facet])
    transformed = _select(candidates, {key: scale * value + shift for key, value in scores.items()}, [facet])
    assert [(row["id"], row["content"]) for row in transformed["sources"]] == [
        (row["id"], row["content"]) for row in baseline["sources"]
    ]


def test_singleton_query_champion_without_retrieval_support_is_not_forced_by_document_name():
    supported = _facet("supported", docs=("aster",))
    unsupported = _facet("unsupported", "Explain the thermal control procedure", docs=("birch",))
    useful = _candidate("useful", _text(1), supported)
    unrelated = _candidate("unrelated", "A garden contains lilies and a wooden bench.",
                           unsupported, nomination=0., document="birch")
    # It is the sole scored item for this query, but relative rank alone cannot
    # create support that the independent retrieval stage did not supply.
    result = _select([useful, unrelated], {useful["id"]: 3., unrelated["id"]: -20.},
                     [supported, unsupported])
    assert _ids(result) == {"useful"}


def test_duplicate_background_is_not_added_to_fill_unused_budget():
    facet = _facet()
    candidates = [_candidate(str(index), _text(0, 1), facet) for index in range(12)]
    result = _select(candidates, {row["id"]: 3. for row in candidates}, [facet])
    assert len(result["sources"]) == 1
    assert result["trace"]["source_bytes"] < 12_000


def test_same_source_adjacent_complement_extends_an_exact_continuous_quote():
    facet = _facet()
    opening = _text(0, 2)
    detail = _text(1, 2)
    original = opening + "\n\n" + detail
    assert len(original) <= 1_600
    first = _candidate("opening", opening, facet, nomination=.9, source_id="single",
                       source_content=original)
    second = _candidate("detail", detail, facet, nomination=.14, source_id="single",
                        source_content=original, start=len(opening) + 2)
    result = _select([first, second], {first["id"]: 4., second["id"]: 3.5}, [facet])
    assert len(result["sources"]) == 1
    selected = result["sources"][0]
    assert selected["id"] == "single"
    assert selected["content"] == original[selected["excerpt_start"]:selected["excerpt_end"]]
    assert selected["content"] == original


def test_same_source_extension_never_breaches_hard_utf8_budget_or_stitches_a_quote():
    facet = _facet()
    opening, detail = _text(0, 2), _text(1, 2)
    original = opening + "\n\n" + detail
    first = _candidate("opening", opening, facet, source_id="single", source_content=original)
    second = _candidate("detail", detail, facet, source_id="single", source_content=original,
                        start=len(opening) + 2)
    budget = max(len(opening.encode()), len(detail.encode())) + 1
    result = _select([first, second], {first["id"]: 4., second["id"]: 3.5},
                     [facet], max_source_bytes=budget)
    assert sum(len(row["content"].encode()) for row in result["sources"]) <= budget
    assert all(row["content"] == original[row["excerpt_start"]:row["excerpt_end"]]
               for row in result["sources"])
    assert all(row["content"] != original for row in result["sources"])


def test_related_work_writing_instruction_does_not_create_related_technical_anchor():
    question = "综合 Aster 和 Birch，写一段 related work，比较它们如何校准传感器。请给出引用。"
    chunks = [{"id": name, "document_id": name, "filename": name + ".pdf", "content": "Metadata only."}
              for name in ("Aster", "Birch")]
    facets = build_query_facets(question, chunks)["facets"]
    technical = {row["text"].casefold() for row in facets if row.get("kind") == "technical_phrase"}
    anchors = {anchor.casefold() for row in facets for anchor in row.get("anchors", [])}
    assert not {"related", "work", "related work"} & (technical | anchors)


def test_merge_retries_a_right_window_after_its_middle_bridge_is_accepted():
    facet = _facet()
    left, middle, right = _text(0, 1), _text(1, 2), _text(2, 1)
    original = left + "\n\n" + middle + "\n\n" + right
    assert 240 < len(middle) and len(original) <= 1_600
    candidates = [
        _candidate("left", left, facet, nomination=1., source_id="single", source_content=original),
        _candidate("right", right, facet, nomination=.9, source_id="single", source_content=original,
                   start=len(left) + len(middle) + 4),
        _candidate("middle", middle, facet, nomination=.7, source_id="single", source_content=original,
                   start=len(left) + 2),
    ]
    events = []
    result = _select(candidates, {row["id"]: 4. - index for index, row in enumerate(candidates)},
                     [facet], diagnostic_callback=events.append)
    assert len(result["sources"]) == 1
    assert result["sources"][0]["content"] == original
    assert result["trace"]["same_chunk_disconnected_skipped"] == 0
    assert any(event["event"] == "merge_rejected" and event.get("reason") == "unsafe_bridge_or_window_limit"
               for event in events)
    assert sum(event["event"] == "merge_accepted" for event in events) == 2


def test_naming_one_document_does_not_exempt_unrequested_cross_document_duplicates():
    facet = _facet()
    original = {"id": "original", "text": "Explain Aster", "kind": "original",
                "weight": .5, "document_ids": ["aster"]}
    text = _text(1, 1)
    requested = _candidate("requested", text, facet)
    duplicate = _candidate("unrequested", text, facet, document="birch")
    result = _select([requested, duplicate], {requested["id"]: 4., duplicate["id"]: 3.},
                     [original, facet])
    assert _ids(result) == {"requested"}


def test_identical_evidence_from_two_explicitly_named_documents_keeps_both_origins():
    facet = _facet(docs=("aster", "birch"))
    text = _text(1, 1)
    candidates = [_candidate("a", text, facet), _candidate("b", text, facet, document="birch")]
    result = _select(candidates, {row["id"]: 3. for row in candidates}, [facet])
    assert _ids(result) == {"a", "b"}
    assert result["trace"]["named_documents_missing"] == 0


def test_same_source_id_with_conflicting_snapshots_is_rejected_as_a_group():
    facet = _facet()
    first_text, other_text = _text(0, 1), _text(2, 1)
    first = _candidate("first", first_text, facet, source_id="single")
    second = _candidate("second", other_text, facet, source_id="single",
                        source_content=" " * len(first_text) + other_text, start=len(first_text))
    events = []
    result = _select([first, second], {first["id"]: 4., second["id"]: 3.},
                     [facet], diagnostic_callback=events.append)
    assert result["sources"] == []
    assert any(event.get("reason") == "source_snapshot_mismatch" for event in events)


def test_same_source_merge_cannot_reintroduce_background_that_failed_the_selection_gate():
    facet = _facet()
    first_text = _text(1, 1)
    background = "Aster telemetry is important to historians and has become a topic of public discussion."
    original = first_text + "\n\n" + background
    first = _candidate("first", first_text, facet, source_id="single", source_content=original)
    second = _candidate("background", background, facet, source_id="single", source_content=original,
                        start=len(first_text) + 2)
    first["section"] = second["section"] = "Introduction"
    second["roles"] = []
    assert not _select([second], {second["id"]: 3.}, [facet])["sources"]
    result = _select([first, second], {first["id"]: 4., second["id"]: 3.}, [facet])
    assert len(result["sources"]) == 1
    assert result["sources"][0]["content"] == first_text


@pytest.mark.parametrize("rejection", ["unsupported", "wrong_scope", "generic_background"])
def test_ineligible_high_logits_cannot_occupy_ranks_and_displace_supported_evidence(rejection):
    facet = _facet(docs=("aster",))
    useful = _candidate("useful", _text(1, 1), facet)
    candidates = [useful]
    for index in range(40):
        candidate = _candidate("irrelevant" + str(index), "Garden information " + str(index) + ".", facet,
                               nomination=0. if rejection == "unsupported" else .8,
                               document="birch" if rejection == "wrong_scope" else "aster")
        if rejection == "generic_background":
            candidate.update(section="Introduction", roles=[])
        candidates.append(candidate)
    scores = {row["id"]: 100. - index for index, row in enumerate(candidates[1:])}
    scores[useful["id"]] = 3.
    assert _ids(_select(candidates, scores, [facet])) == {"useful"}


def test_explicit_multi_document_comparison_has_coverage_then_soft_balanced_slots():
    facet = _facet(docs=("aster", "birch", "cedar"))
    candidates = []
    for index in range(25):
        document = "birch" if index in (13, 14, 15) else "cedar" if index in (22, 23, 24) else "aster"
        # Independent vocabulary makes this a rank-allocation test, not a
        # learned relevance or paraphrase judgment. Top 9 all come from Aster.
        text = f"Mechanism{index} signal{index} register{index} condition{index} protocol{index}."
        candidates.append(_candidate(str(index), text, facet, document=document))
    scores = {row["id"]: 50. - index for index, row in enumerate(candidates)}
    result = _select(candidates, scores, [facet])
    documents = [row["document_id"] for row in result["sources"]]
    assert len(documents) == 8
    assert set(documents[:3]) == {"aster", "birch", "cedar"}
    assert all(2 <= documents.count(document) <= 3 for document in facet["document_ids"])
    assert result["trace"]["document_soft_quota"] == 3
    assert result["trace"]["fair_coverage_selections"] == 3
    assert result["trace"]["named_documents_missing"] == 0
    assert result["trace"]["source_bytes"] <= 12_000


def test_explicit_document_fairness_does_not_manufacture_support_for_a_named_document():
    facet = _facet(docs=("aster", "birch", "cedar"))
    candidates = [_candidate("a", _text(0, 1), facet),
                  _candidate("b", _text(1, 1), facet, document="birch", nomination=0.),
                  _candidate("c", _text(2, 1), facet, document="cedar")]
    scores = {candidates[0]["id"]: 3., candidates[1]["id"]: 100., candidates[2]["id"]: 2.}
    result = _select(candidates, scores, [facet])
    assert {row["document_id"] for row in result["sources"]} == {"aster", "cedar"}
    assert result["trace"]["named_documents_missing"] == 1
    assert any(gap["code"] == "documents_not_selected" and gap["count"] == 1 for gap in result["gaps"])


def test_document_fairness_cannot_exceed_byte_budget_and_reports_uncovered_documents():
    facet = _facet(docs=("aster", "birch", "cedar"))
    candidates = [_candidate("a", _text(0, 1), facet),
                  _candidate("b", _text(1, 3), facet, document="birch"),
                  _candidate("c", _text(2, 3), facet, document="cedar")]
    budget = len(candidates[0]["content"].encode()) + 1
    result = _select(candidates, {row["id"]: 3. - index for index, row in enumerate(candidates)},
                     [facet], max_source_bytes=budget)
    assert _ids(result) == {"a"}
    assert result["trace"]["source_bytes"] <= budget
    assert result["trace"]["named_documents_missing"] == 2
    assert any(gap["code"] == "documents_not_selected" and gap["count"] == 2 for gap in result["gaps"])


def test_multi_document_priority_respects_global_rank_floor_without_per_document_renormalization():
    facet = _facet(docs=("aster", "birch"))
    candidates = [_candidate(str(index), f"Signal{index} changes register{index} using mechanism{index}.",
                             facet, document="aster") for index in range(38)]
    candidates.append(_candidate("weak", _text(2, 1), facet, document="birch"))
    scores = {row["id"]: 100. - index for index, row in enumerate(candidates)}
    result = _select(candidates, scores, [facet])
    assert "weak" not in _ids(result)  # Rank 39 has utility 9 / 47 < .20.
    assert result["trace"]["named_documents_missing"] == 1
    assert len(result["sources"]) <= 8


def test_single_document_selection_does_not_enable_fairness_priority():
    facet = _facet(docs=("aster",))
    candidates = [_candidate(str(index), _text(index, 1), facet) for index in range(3)]
    result = _select(candidates, {row["id"]: 4. - index for index, row in enumerate(candidates)}, [facet])
    assert _ids(result) == {"0", "1", "2"}
    assert result["trace"]["document_slot_policy"] == "ordinary_rank_mmr"
    assert result["trace"]["document_soft_quota"] == 0
    assert result["trace"]["fair_coverage_selections"] == 0


def _branch_candidate(identifier, facet, branch, *, content=None, document="aster", nomination=.8,
                      source_id=None, source_content=None, start=0):
    content = content or f"Signal{identifier} register{identifier} condition{identifier} protocol{identifier}."
    candidate = _candidate(identifier, content, facet, document=document, nomination=nomination,
                           source_id=source_id, source_content=source_content, start=start)
    candidate["rerank_branch"] = branch
    return candidate


def _branch_facets(*, docs=("aster",)):
    return [_facet("full", "Describe the mechanism, training procedure and results of Aster", docs=docs),
            _facet("focused", "What supervision does the training procedure require?", docs=docs)]


def test_focused_branch_reserves_one_shared_slot_and_never_selects_more_than_two_sources():
    full, focused = _branch_facets()
    full_rows = [_branch_candidate("f" + str(index), full, "full") for index in range(8)]
    focused_rows = [_branch_candidate("g" + str(index), focused, "focused") for index in range(8)]
    candidates = full_rows + focused_rows
    scores = {row["id"]: 100. - index for rows in (full_rows, focused_rows) for index, row in enumerate(rows)}
    result = _select(candidates, scores, [full, focused])
    selected = result["sources"]
    assert selected[0]["id"].startswith("g")
    assert len(selected) == 8
    assert 1 <= sum(row["id"].startswith("g") for row in selected) <= 2
    assert result["trace"]["focused_selected_sources"] <= 2
    assert result["trace"]["focused_branch_enabled"] is True
    assert result["trace"]["focused_rank_weight"] == .75
    assert result["trace"]["source_bytes"] <= 12_000


def test_focused_reservation_never_admits_a_zero_support_candidate():
    full, focused = _branch_facets()
    first = _branch_candidate("full", full, "full")
    unsupported = _branch_candidate("focused", focused, "focused", nomination=0.)
    result = _select([first, unsupported], {first["id"]: 0., unsupported["id"]: 100.}, [full, focused])
    assert _ids(result) == {"full"}
    assert result["trace"]["focused_selected_sources"] == 0


def test_focused_branch_without_a_qualified_full_branch_cannot_fill_context():
    full, focused = _branch_facets()
    unsupported_full = _branch_candidate("full", full, "full", nomination=0.)
    focused_rows = [_branch_candidate(str(index), focused, "focused") for index in range(8)]
    candidates = [unsupported_full, *focused_rows]
    result = _select(candidates, {row["id"]: 1. for row in candidates}, [full, focused])
    assert result["sources"] == []
    assert result["trace"]["focused_branch_enabled"] is False


def test_document_coverage_precedes_focused_reservation_and_duplicate_source_does_not_take_a_slot():
    full, focused = _branch_facets(docs=("aster", "birch"))
    text = "Independent sparse gates normalize measurements."
    shared_full = _branch_candidate("shared_full", full, "full", content=text, source_id="shared")
    other_full = _branch_candidate("birch", full, "full", document="birch")
    shared_focused = _branch_candidate("shared_focused", focused, "focused", content=text, source_id="shared")
    new_focused = _branch_candidate("new_focused", focused, "focused")
    candidates = [shared_full, other_full, shared_focused, new_focused]
    scores = {shared_full["id"]: 4., other_full["id"]: 3., shared_focused["id"]: 4., new_focused["id"]: 3.}
    result = _select(candidates, scores, [full, focused])
    assert {row["document_id"] for row in result["sources"][:2]} == {"aster", "birch"}
    assert _ids(result) == {"shared", "birch", "new_focused"}
    assert len(result["sources"]) == 3
    assert result["trace"]["focused_selected_sources"] == 1


def test_focused_reservation_is_not_allowed_to_exceed_the_shared_byte_budget():
    full, focused = _branch_facets()
    first = _branch_candidate("full", full, "full")
    long_focused = _branch_candidate("focused", focused, "focused", content=_text(1, 3))
    budget = len(first["content"].encode()) + 1
    result = _select([first, long_focused], {first["id"]: 1., long_focused["id"]: 1.},
                     [full, focused], max_source_bytes=budget)
    assert _ids(result) == {"full"}
    assert result["trace"]["source_bytes"] <= budget
    assert result["trace"]["focused_selected_sources"] == 0


def test_weak_focused_group_cannot_fill_eight_sources_via_local_champions():
    full, focused = _branch_facets()
    first = _branch_candidate("full", full, "full")
    focused_rows = [_branch_candidate("g" + str(index), focused, "focused") for index in range(24)]
    candidates = [first, *focused_rows]
    scores = {first["id"]: 3., **{row["id"]: -100. - index for index, row in enumerate(focused_rows)}}
    result = _select(candidates, scores, [full, focused])
    assert "full" in _ids(result)
    assert 1 <= result["trace"]["focused_selected_sources"] <= 2
    assert len(result["sources"]) <= 3


def test_focused_merges_have_an_independent_bounded_extension_count():
    full, focused = _branch_facets()
    paragraphs = [f"Signal{index} condition{index} protocol{index}." for index in range(7)]
    original = "\n\n".join(paragraphs)
    first = _branch_candidate("full", full, "full", content=paragraphs[0], source_id="single",
                              source_content=original)
    focused_rows = []
    cursor = len(paragraphs[0]) + 2
    for index, paragraph in enumerate(paragraphs[1:]):
        focused_rows.append(_branch_candidate("g" + str(index), focused, "focused", content=paragraph,
                                              source_id="single", source_content=original, start=cursor))
        cursor += len(paragraph) + 2
    candidates = [first, *focused_rows]
    scores = {first["id"]: 3., **{row["id"]: 3. - index for index, row in enumerate(focused_rows)}}
    result = _select(candidates, scores, [full, focused])
    assert len(result["sources"]) == 1
    assert result["trace"]["focused_selected_sources"] == 1
    assert result["trace"]["focused_merge_extensions"] <= 2
    selected = result["sources"][0]
    assert selected["content"] == original[selected["excerpt_start"]:selected["excerpt_end"]]


def test_branch_labels_cannot_turn_the_same_actual_query_into_a_focused_rescue():
    full, focused = _branch_facets()
    first = _branch_candidate("full", full, "full")
    mislabeled = _branch_candidate("focused", focused, "focused")
    mislabeled["query"] = first["query"]
    result = _select([first, mislabeled], {first["id"]: 3., mislabeled["id"]: 4.}, [full, focused])
    assert _ids(result) == {"full"}
    assert result["trace"]["focused_branch_enabled"] is False


def test_ordinary_unbranched_selection_keeps_its_ranking_without_rescue_behavior():
    facet = _facet()
    candidates = [_candidate(str(index), _text(index, 1), facet) for index in range(3)]
    scores = {row["id"]: 3. - index for index, row in enumerate(candidates)}
    ordinary = _select(candidates, scores, [facet])
    full_only = _select([{**row, "rerank_branch": "full"} for row in candidates], scores, [facet])
    assert ordinary["sources"] == full_only["sources"]
    assert ordinary["trace"]["focused_selected_sources"] == 0
    assert ordinary["trace"]["focused_branch_enabled"] is False
