"""Method plans establish only structure; offline evaluations never deploy."""
from __future__ import annotations

import copy
import json
import math

import pytest

from docthinker.server.method_planner import (
    MAX_PLAN_BYTES, compile_method_plan, evaluate_method_revision,
)


def method(**changes):
    return {"id": "compare-like-with-like", "label": "K1", "reasoning_role": "induction",
            "required_facts": ["Objects, controls and measurement conditions", "Verified primary evidence"],
            "stop_conditions": ["The primary result cannot be checked"], **changes}


def source(number=1, *, document=None, content=None, **changes):
    return {"id": f"chunk-{number}", "label": f"S{number}",
            "document_id": document or f"document-{number}",
            "content": content if content is not None else f"Actual excerpt {number}.",
            "truncated": False, **changes}


def pair(number=1, *, quality=.7, tokens=100, latency=100, claims=0):
    return {"task_id": f"task-{number}",
            "baseline": {"quality": .6, "unsupported_claims": 0, "input_tokens": 100, "latency_ms": 100},
            "candidate": {"quality": quality, "unsupported_claims": claims,
                          "input_tokens": tokens, "latency_ms": latency}}


def encoded_size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def test_selected_cards_get_pending_semantic_requirements_not_keyword_verification():
    cards = [method()]
    rows = [source(1, content="Objects, controls and measurement conditions. Verified primary evidence."), source(2)]
    original = copy.deepcopy((cards, rows))
    plan = compile_method_plan(cards, rows, skill_id="literature-synthesis")
    assert (cards, rows) == original
    assert plan["decision"] == "review_requirements"
    assert plan["methods"][0]["requirement_count"] == 2
    assert plan["methods"][0]["requirements_status"] == "pending_review"
    assert plan["methods"][0]["stop_conditions_status"] == "pending_review"
    assert plan["semantic_entailment_verified"] is False
    assert plan["method_execution_claimed"] is False
    assert plan["normal_answer_allowed"] is True and plan["model_calls"] == 0
    assert plan["source_labels"] == ["S1", "S2"]
    assert plan["evidence"]["distinct_documents"] == 2
    assert "Verified primary evidence" not in json.dumps(plan)
    assert encoded_size(plan) <= 1_000


def test_same_document_chunks_never_satisfy_multiple_paper_structure():
    plan = compile_method_plan([method()], [source(1, document="one"), source(2, document="one")],
                               skill_id="literature-synthesis")
    assert plan["evidence"]["unique_excerpts"] == 2
    assert plan["evidence"]["distinct_documents"] == 1
    assert plan["structural_gaps"] == ["insufficient_distinct_documents"]
    assert plan["decision"] == "collect_evidence"
    assert plan["normal_answer_allowed"] is True


def test_whitespace_cannot_turn_one_document_id_into_two_documents():
    plan = compile_method_plan([method()], [source(1, document="one"), source(2, document=" one ")],
                               skill_id="literature-synthesis")
    assert plan["evidence"]["distinct_documents"] == 1
    assert plan["decision"] == "collect_evidence"


def test_duplicate_excerpt_across_documents_is_not_new_structural_support():
    rows = [source(1, content="The observed result is positive."),
            source(2, content="  The  observed result\nis positive.  ")]
    plan = compile_method_plan([method()], rows, skill_id="literature-synthesis")
    assert plan["evidence"]["usable_sources"] == 2
    assert plan["evidence"]["unique_excerpts"] == 1
    assert plan["evidence"]["distinct_documents"] == 1
    assert plan["evidence"]["duplicate_excerpts"] == ["S2"]
    assert "insufficient_distinct_documents" in plan["structural_gaps"]


@pytest.mark.parametrize("label", ["K1", "S0", "S01", "[S1]", "S1\n", "S11", "S10000", "", None, []])
def test_source_labels_must_be_current_s_labels(label):
    plan = compile_method_plan([method()], [source(label=label)], skill_id="evidence-writing")
    assert plan["source_labels"] == []
    assert "invalid_source_labels" in plan["structural_gaps"]
    assert plan["decision"] == "collect_evidence"


def test_ambiguous_labels_are_not_credited_even_if_documents_differ():
    plan = compile_method_plan([method()], [source(1), source(2, label="S1"), source(3)],
                               skill_id="literature-synthesis")
    assert plan["source_labels"] == ["S3"]
    assert plan["evidence"]["distinct_documents"] == 1
    assert "duplicate_source_labels" in plan["structural_gaps"]


@pytest.mark.parametrize("content", ["", " \n\t", None, 1, "\ud800"])
def test_only_real_nonempty_excerpts_count(content):
    row = source()
    row["content"] = content
    plan = compile_method_plan([method()], [row], skill_id="literature-synthesis")
    assert plan["evidence"]["unique_excerpts"] == 0
    assert "empty_excerpts" in plan["structural_gaps"]
    assert "no_usable_excerpts" in plan["structural_gaps"]


@pytest.mark.parametrize("changes,code", [({"document_id": ""}, "missing_document_ids"),
    ({"document_id": None}, "missing_document_ids"), ({"id": ""}, "invalid_source_ids"),
    ({"excerpt_start": -1}, "invalid_excerpt_geometry"),
    ({"excerpt_end": 999}, "invalid_excerpt_geometry"),
    ({"original_content_length": 1}, "invalid_excerpt_geometry"),
    ({"excerpt_start": True}, "invalid_excerpt_geometry")])
def test_invalid_source_identity_and_geometry_fail_structural_checks(changes, code):
    plan = compile_method_plan([method()], [source(**changes)], skill_id="evidence-writing")
    assert code in plan["structural_gaps"]
    assert plan["evidence"]["distinct_documents"] == 0


@pytest.mark.parametrize("changes", [{"truncated": True}, {"excerpt_start": 10},
    {"original_content_length": 1000}])
def test_truncation_is_retained_and_does_not_verify_semantics(changes):
    plan = compile_method_plan([method()], [source(**changes)], skill_id="evidence-writing")
    assert plan["evidence"]["truncated_sources"] == ["S1"]
    assert plan["methods"][0]["requirements_status"] == "pending_review"
    assert plan["semantic_entailment_verified"] is False
    assert plan["decision"] == "review_requirements"


def test_exact_json_byte_bound_method_source_caps_and_no_excerpt_leakage():
    methods = [method(id="m" + "x" * 63, label=f"K{number}",
                      required_facts=["事实" * 200] * 8) for number in range(1, 5)]
    sources = [source(number, content="原文" + str(number), truncated=True) for number in range(1, 14)]
    plan = compile_method_plan(methods, sources, skill_id="s" + "x" * 63)
    assert len(plan["methods"]) == 3 and len(plan["source_labels"]) == 10
    assert plan["evidence"]["selected_sources"] == 10
    assert "source_limit_exceeded" in plan["structural_gaps"]
    assert encoded_size(plan) <= MAX_PLAN_BYTES
    assert "原文" not in json.dumps(plan, ensure_ascii=False)
    assert all(card["requirements_status"] == "pending_review" for card in plan["methods"])


def test_plan_functions_have_no_io_side_effects(monkeypatch):
    import builtins
    import socket

    def forbidden(*args, **kwargs):
        pytest.fail("method planning and offline evaluation cannot perform I/O")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    plan = compile_method_plan([method()], [source()], skill_id="evidence-writing")
    result = evaluate_method_revision([pair(number) for number in range(12)])
    assert plan["model_calls"] == result["model_calls"] == 0


@pytest.mark.parametrize("sources", [None, {}, "S1", [None, {}, {"label": "S0"}]])
def test_malformed_input_is_json_safe_advice_not_a_chat_refusal(sources):
    plan = compile_method_plan(None, sources, skill_id=None)
    assert plan["decision"] == "collect_evidence"
    assert plan["normal_answer_allowed"] is True
    assert encoded_size(plan) <= MAX_PLAN_BYTES


def test_empty_current_window_does_not_gain_facts_from_card_or_title():
    plan = compile_method_plan([method()], [], skill_id="literature-synthesis")
    assert plan["source_labels"] == []
    assert plan["structural_gaps"] == ["no_usable_excerpts", "insufficient_distinct_documents"]
    assert plan["methods"][0]["requirements_status"] == "pending_review"


def test_offline_gate_requires_twelve_pairs_and_never_claims_publication_or_proof():
    pairs = [pair(number) for number in range(12)]
    original = copy.deepcopy(pairs)
    result = evaluate_method_revision(pairs)
    assert pairs == original
    assert result["eligible_for_review"] is True
    assert result["pair_count"] == 12 and not result["failed_gates"]
    assert result["metrics"]["quality_wins"] == 12
    assert result["automatic_publication"] is False
    assert result["statistical_significance_proven"] is False
    assert result["novelty_proven"] is False
    assert result["pair_independence_verified"] is False
    assert result["model_calls"] == 0
    assert json.dumps(result, allow_nan=False)
    assert evaluate_method_revision(pairs[:11])["failed_gates"] == ["minimum_pairs"]
    assert evaluate_method_revision(pairs, minimum_pairs=13)["failed_gates"] == ["minimum_pairs"]


@pytest.mark.parametrize("minimum", [0, 11, 12.0, True, None])
def test_caller_cannot_lower_minimum_pair_floor(minimum):
    with pytest.raises(ValueError, match="minimum_pairs"):
        evaluate_method_revision([], minimum_pairs=minimum)


def test_mean_quality_and_strict_majority_are_separate_gates():
    pairs = [pair(number, quality=.9 if number < 6 else .5) for number in range(12)]
    result = evaluate_method_revision(pairs)
    assert result["metrics"]["mean_quality_gain"] > 0
    assert result["failed_gates"] == ["majority_quality_wins"]
    pairs = [pair(number, quality=.601 if number < 7 else .1) for number in range(12)]
    result = evaluate_method_revision(pairs)
    assert result["metrics"]["quality_wins"] == 7
    assert result["failed_gates"] == ["mean_quality_improved"]
    tied = [pair(number, quality=.6) for number in range(12)]
    assert evaluate_method_revision(tied)["failed_gates"] == ["mean_quality_improved", "majority_quality_wins"]


def test_unsupported_claim_improvements_on_other_pairs_cannot_hide_one_regression():
    pairs = [pair(number) for number in range(12)]
    pairs[0]["baseline"]["unsupported_claims"] = 10
    pairs[1]["candidate"]["unsupported_claims"] = 1
    result = evaluate_method_revision(pairs)
    assert result["metrics"]["candidate_means"]["unsupported_claims"] < result["metrics"]["baseline_means"]["unsupported_claims"]
    assert result["failed_gates"] == ["no_unsupported_claim_regressions"]


@pytest.mark.parametrize("tokens,latency,failed", [(110, 120, []), (110.001, 120, ["input_token_budget"]),
    (110, 120.001, ["latency_budget"]), (111, 121, ["input_token_budget", "latency_budget"])])
def test_average_budget_ratios_have_inclusive_explicit_boundaries(tokens, latency, failed):
    result = evaluate_method_revision([pair(number, tokens=tokens, latency=latency) for number in range(12)])
    assert result["failed_gates"] == failed
    assert result["eligible_for_review"] is (not failed)


def test_budgets_use_ratio_of_means_not_mean_of_pair_ratios():
    pairs = [pair(number, tokens=110, latency=120) for number in range(12)]
    pairs[0]["baseline"]["input_tokens"] = 1
    pairs[0]["candidate"]["input_tokens"] = 11
    pairs[1]["baseline"]["input_tokens"] = 199
    pairs[1]["candidate"]["input_tokens"] = 209
    result = evaluate_method_revision(pairs)
    assert result["metrics"]["input_token_ratio"] == pytest.approx(1.1)
    assert result["eligible_for_review"] is True


def test_zero_budget_is_not_a_divide_by_zero_or_json_nan():
    pairs = [pair(number, tokens=0, latency=0) for number in range(12)]
    for row in pairs:
        row["baseline"].update(input_tokens=0, latency_ms=0)
    result = evaluate_method_revision(pairs)
    assert result["eligible_for_review"] is True
    assert result["metrics"]["input_token_ratio"] == result["metrics"]["latency_ratio"] == 1
    pairs[0]["candidate"]["input_tokens"] = 1
    result = evaluate_method_revision(pairs)
    assert result["failed_gates"] == ["input_token_budget"]
    assert result["metrics"]["input_token_ratio"] is None
    assert json.dumps(result, allow_nan=False)


def test_finite_extreme_costs_do_not_overflow_means_or_emit_json_infinity():
    pairs = [pair(number, tokens=1.7e308, latency=1.7e308) for number in range(12)]
    for row in pairs:
        row["baseline"].update(input_tokens=1.7e308, latency_ms=1.7e308)
    result = evaluate_method_revision(pairs)
    assert result["eligible_for_review"] is True
    assert json.dumps(result, allow_nan=False)
    for row in pairs:
        row["baseline"].update(input_tokens=5e-324, latency_ms=5e-324)
    result = evaluate_method_revision(pairs)
    assert result["failed_gates"] == ["input_token_budget", "latency_budget"]
    assert result["metrics"]["input_token_ratio"] is None
    assert json.dumps(result, allow_nan=False)


def test_subnormal_quality_and_costs_do_not_hide_actual_changes():
    pairs = [pair(number, quality=5e-324, tokens=0) for number in range(12)]
    for row in pairs:
        row["baseline"].update(quality=0, input_tokens=0)
    result = evaluate_method_revision(pairs)
    assert result["metrics"]["mean_quality_gain"] > 0
    assert result["eligible_for_review"] is True
    pairs[0]["baseline"]["input_tokens"] = 5e-324
    pairs[0]["candidate"]["input_tokens"] = 1e-323
    result = evaluate_method_revision(pairs)
    assert result["metrics"]["input_token_ratio"] == 2
    assert result["failed_gates"] == ["input_token_budget"]


@pytest.mark.parametrize("field,value", [("quality", -1), ("quality", 1.01), ("quality", math.nan),
    ("quality", math.inf), ("quality", True), ("quality", "0.8"),
    ("unsupported_claims", -1), ("input_tokens", -1), ("latency_ms", math.nan),
    ("latency_ms", None)])
def test_invalid_measurements_are_rejected_not_silently_ignored(field, value):
    pairs = [pair(number) for number in range(12)]
    pairs[0]["candidate"][field] = value
    with pytest.raises(ValueError, match=field):
        evaluate_method_revision(pairs)


@pytest.mark.parametrize("pairs", [None, {}, "pairs", [None], [{"task_id": "", "baseline": {}, "candidate": {}}]])
def test_invalid_pair_shapes_are_rejected(pairs):
    with pytest.raises(ValueError):
        evaluate_method_revision(pairs)


def test_duplicate_task_id_is_rejected_and_empty_sequence_is_ineligible():
    with pytest.raises(ValueError, match="task_id"):
        evaluate_method_revision([pair(1), pair(1)])
    result = evaluate_method_revision([])
    assert result["eligible_for_review"] is False
    assert result["pair_count"] == 0
    assert json.dumps(result, allow_nan=False)


def test_whitespace_cannot_evade_duplicate_task_identity_validation():
    pairs = [pair(1), pair(2)]
    pairs[1]["task_id"] = " task-1 "
    with pytest.raises(ValueError, match="task_id"):
        evaluate_method_revision(pairs)
