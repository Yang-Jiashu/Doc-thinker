"""Query-aware explicit project memory selection, without model calls or I/O."""
from copy import deepcopy
import random

import pytest

from docthinker.server.research_memory import classify_memory_query, select_confirmed_memories


def note(index, text, *, kind="decision", updated=None, status="active", confirmed=True, revision=1):
    return {"id": f"00000000-0000-4000-8000-{index:012d}", "kind": kind, "text": text,
            "updated_at": index if updated is None else updated, "status": status,
            "confirmed_by_user": confirmed, "revision": revision}


def test_eightieth_old_relevant_decision_beats_new_unrelated_notes():
    rows = [note(index, f"Prepare meeting logistics item {index}", updated=1000 - index) for index in range(100)]
    old = note(79, "Compare renewable investment and energy import dependence separately.", updated=20)
    rows[79] = old
    selected = select_confirmed_memories("How should we compare renewable investment and import dependence?", rows)
    assert selected["memories"] == [old]
    assert selected["omitted_count"] == 99
    assert selected["strategy"] == "lexical_bm25_with_preferences"


def test_preferences_do_not_displace_all_relevant_findings():
    findings = [note(index, "Renewable investment increases domestic supply.", kind="finding") for index in range(10)]
    preferences = [note(index + 20, "Keep answers brief and direct.", kind="preference") for index in range(10)]
    selected = select_confirmed_memories("Renewable investment and domestic supply", preferences + findings)
    assert sum(row["kind"] == "preference" for row in selected["memories"]) == 2
    assert sum(row["kind"] == "finding" for row in selected["memories"]) == 10
    single = select_confirmed_memories("Renewable investment and domestic supply", preferences + findings, max_candidates=1)
    assert single["memories"][0]["kind"] == "finding"


def test_zero_overlap_findings_and_decisions_are_not_context_padding():
    preference = note(1, "Use concise Chinese.", kind="preference")
    unrelated = [note(2, "Booking a room for Tuesday."), note(3, "Protein folding needs another assay.", kind="finding")]
    selected = select_confirmed_memories("Energy import dependence", unrelated + [preference])
    assert selected["memories"] == [preference]
    assert selected["omitted_count"] == 2


@pytest.mark.parametrize("question", ["项目进展怎么样？", "下一步做什么？", "下一步呢？", "接下来？", "上次确定的计划是什么？", "你还记得什么？", "What is our project progress?", "What did we decide?"])
def test_explicit_project_memory_questions_allow_recent_context(question):
    old = note(1, "Compare domestic supply and import demand.", updated=5)
    newest = note(2, "Run a second experiment after peer review.", updated=8)
    selected = select_confirmed_memories(question, [old, newest])
    assert selected["memories"] == [newest, old]
    assert selected["strategy"] == "recent_project_context"


def test_domain_memory_term_is_not_mistaken_for_personal_memory_request():
    unrelated = note(1, "Choose a venue for the meeting.")
    result = select_confirmed_memories("蛋白质的细胞记忆机制是什么？", [unrelated])
    assert result["memories"] == []
    assert result["strategy"] == "lexical_bm25_with_preferences"
    assert classify_memory_query("蛋白质的细胞记忆机制是什么？") == "lexical_bm25_with_preferences"


@pytest.mark.parametrize("question,strategy", [("What did we decide?", "recent_project_context"),
                                                ("下一步呢？", "recent_project_context"),
                                                ("我的偏好是什么？", "explicit_preferences"),
                                                ("Tell me about quantum memory", "lexical_bm25_with_preferences")])
def test_memory_intent_is_shareable_with_evidence_gate(question, strategy):
    assert classify_memory_query(question) == strategy


def test_explicit_preferences_question_can_review_more_than_two():
    rows = [note(index, "Use clear concise answers.", kind="preference") for index in range(10)]
    rows.append(note(20, "A microscopy result needs replication.", kind="finding"))
    selected = select_confirmed_memories("我的偏好是什么？", rows)
    assert len(selected["memories"]) == 10
    assert all(row["kind"] == "preference" for row in selected["memories"])
    assert selected["strategy"] == "explicit_preferences"


def test_only_active_explicit_confirmation_is_eligible():
    active = note(1, "Energy investment plan")
    rows = [active, note(2, "Energy finding", status="invalidated"),
            note(3, "Energy finding", status="deleted"), note(4, "Energy finding", status="candidate"),
            note(5, "Energy finding", confirmed=False)]
    selected = select_confirmed_memories("Energy investment", rows)
    assert selected["memories"] == [active]
    assert selected["omitted_count"] == 0  # Excluded statuses are not budget/relevance omissions.


def test_complete_long_note_is_preserved_for_later_prompt_budget_decision():
    content = "Do not share this project; only use approved internal sources. " * 30
    assert len(content) <= 2000 and len(content) > 600
    row = note(1, content, kind="preference")
    original = deepcopy(row)
    selected = select_confirmed_memories("Internal sources", [row])
    assert selected["memories"][0]["text"] == content
    assert row == original
    assert "confidence" not in selected["memories"][0]
    assert "score" not in selected["memories"][0]


def test_bm25_query_coverage_prefers_linked_concepts_over_repeated_single_word():
    narrow = note(1, "imports " * 60)
    broad = note(2, "Renewable investment and domestic supply affect energy imports.")
    selected = select_confirmed_memories("renewable investment domestic supply energy imports", [narrow, broad])
    assert selected["memories"][0] == broad


def test_newest_equal_relevance_then_id_are_stable_under_shuffle():
    rows = [note(index, "Energy imports depend on domestic supply.", updated=index % 4) for index in range(30)]
    expected = select_confirmed_memories("energy imports domestic supply", rows)
    shuffled = deepcopy(rows)
    random.Random(100).shuffle(shuffled)
    assert select_confirmed_memories("energy imports domestic supply", shuffled) == expected
    assert len(expected["memories"]) == 20
    assert expected["omitted_count"] == 10
    assert [row["updated_at"] for row in expected["memories"]] == sorted((row["updated_at"] for row in expected["memories"]), reverse=True)


def test_duplicate_current_notes_do_not_amplify_frequency_or_confidence():
    old = note(1, "Old energy import decision", updated=1, revision=1)
    corrected = note(1, "New energy import decision", updated=2, revision=2)
    other = note(2, "Energy import finding", kind="finding")
    result = select_confirmed_memories("energy import decision", [old, corrected, corrected, other])
    assert sum(row["id"] == old["id"] for row in result["memories"]) == 1
    assert old not in result["memories"] and corrected in result["memories"]
    assert result["omitted_count"] == 0


def test_preferences_and_findings_counts_remain_bounded_at_full_project_capacity():
    rows = [note(index, "Energy import investment.", kind="preference" if index < 50 else "finding") for index in range(100)]
    selected = select_confirmed_memories("Energy import investment", rows)
    assert len(selected["memories"]) == 20
    assert sum(row["kind"] == "preference" for row in selected["memories"]) == 2
    assert selected["omitted_count"] == 80


@pytest.mark.parametrize("row", [None, "notdict", {}, note(1, ""), note(1, "x" * 2001),
                                  note(1, "bad\ud800"), note(1, "bad\x00"), note(1, "ok", kind="hypothesis")])
def test_invalid_unconfirmed_or_oversized_notes_are_excluded(row):
    result = select_confirmed_memories("energy", [row])
    assert result["memories"] == [] and result["omitted_count"] == 0


@pytest.mark.parametrize("question,rows,limit", [("", [], 20), ("bad\ud800", [], 20),
                                                ("中" * 3000, [], 20), ("valid", [note(1, "x")] * 101, 20),
                                                ("valid", None, 20), ("valid", [], 0),
                                                ("valid", [], 21), ("valid", [], True)])
def test_algorithm_input_and_work_budgets(question, rows, limit):
    with pytest.raises(ValueError):
        select_confirmed_memories(question, rows, max_candidates=limit)


def test_invalid_timestamps_cannot_make_ranking_nondeterministic():
    rows = [note(1, "Energy imports", updated=float("nan")), note(2, "Energy imports", updated=float("inf")),
            note(3, "Energy imports", updated=True), note(4, "Energy imports", updated=2)]
    result = select_confirmed_memories("energy imports", rows)
    assert result["memories"][0]["id"] == rows[3]["id"]
    reverse = select_confirmed_memories("energy imports", list(reversed(rows)))
    assert [row["id"] for row in result["memories"]] == [row["id"] for row in reverse["memories"]]
