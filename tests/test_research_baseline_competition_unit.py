"""Original retrieval gets scoring eligibility, not protected answer slots."""
from docthinker.server.research_evidence_selection import prepare_evidence_candidates, select_evidence
from docthinker.server.research_faceted_assist import _rerank_pairs
from docthinker.server.research_query_facets import build_query_facets


def source(key, content, **extra):
    return {"id": key, "document_id": "d", "filename": "study.pdf", "ordinal": 0,
            "content": content, **extra}


def fixture():
    question = ("Explain how the cobalt controller processes measurements. "
                "Compare the controller accuracy against the offline baseline.")
    originals = [source(f"old-{i}", f"Cobalt processes measurements with bounded buffers for device {i}.")
                 for i in range(8)]
    alternatives = [source(f"new-{i}", f"The cobalt controller processes measurements and compares offline "
                           f"baseline accuracy in experiment {i} with independent memory.") for i in range(70)]
    plan = build_query_facets(question, originals + alternatives)
    return question, originals, alternatives, plan


def test_all_eight_eligible_original_sources_get_full_query_scoring_opportunity():
    question, originals, alternatives, plan = fixture()
    pool = prepare_evidence_candidates(question, alternatives + originals, originals, plan, {})
    assert len(pool["candidates"]) == 48
    assert pool["trace"]["baseline_nominations"] == 8
    pairs = _rerank_pairs(question, pool["candidates"], plan["facets"])
    assert len(pairs) <= 48
    full_ids = {row["source_id"] for row in pairs if row.get("rerank_branch") != "focused"}
    assert {row["id"] for row in originals} <= full_ids
    # Nomination is not correctness: the scorer can still discard every old source.
    scores = {row["id"]: (8. if row["source_id"].startswith("new-") else -12.) for row in pairs}
    selected = select_evidence(pairs, scores, plan, originals)
    assert not any(row["id"].startswith("old-") for row in selected["sources"])
    assert len(selected["sources"]) <= 8
    assert sum(len(row["content"].encode()) for row in selected["sources"]) <= 12000


def test_baseline_reserve_does_not_restore_ineligible_or_missing_text():
    question, originals, alternatives, plan = fixture()
    originals[1]["status"] = "revoked"
    originals[2]["is_generated"] = True
    originals[3]["content"] = "References\n[1] Writer A. Cobalt controller. Journal. 2020."
    pool = prepare_evidence_candidates(question, alternatives + originals[:7], originals, plan, {})
    ids = {row["source_id"] for row in pool["candidates"]}
    assert not ids & {"old-1", "old-2", "old-3", "old-7"}
    assert pool["trace"]["baseline_nominations_force_selection"] is False


def test_small_candidate_budget_still_leaves_room_for_new_material():
    question, originals, alternatives, plan = fixture()
    pool = prepare_evidence_candidates(question, alternatives + originals, originals, plan, {}, max_candidates=16)
    assert len(pool["candidates"]) == 16
    assert pool["trace"]["baseline_nominations"] <= 4
    assert any(row["source_id"].startswith("new-") for row in pool["candidates"])
