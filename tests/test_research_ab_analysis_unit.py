"""Accounting and blinding contracts; no network or real model calls."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.analyze_research_ab import (
    aggregate_scores, build_blind_packets, collect_pairs, eligibility, exposure, main, metrics,
    provider_accounting, retrieved_evidence_audit,
)


def fixture():
    dimensions = [{"id": name, "weight": weight} for name, weight in (
        ("factual_accuracy", 25), ("evidence_support", 25), ("key_fact_completeness", 20),
        ("scope_and_uncertainty", 15), ("task_usefulness", 15))]
    return {"papers": [{"id": "paper", "title": "Public Paper", "file": "paper.pdf"}],
            "rubric": {"dimensions": dimensions}, "questions": [
                {"id": "q1", "question": "Explain the public paper result.", "papers": ["paper"],
                 "key_facts": [{"id": "f1", "text": "Supply increases.", "paper_id": "paper", "pages": [1]}],
                 "forbidden_claims": ["Always perfect."]},
                {"id": "q2", "question": "Summarize the public paper result.", "papers": ["paper"],
                 "key_facts": [], "forbidden_claims": []},
                {"id": "control", "question": "纯中文对照", "papers": ["paper"],
                 "diagnostic_only": True, "include_in_primary_quality_average": False}]}


def answer(question, enabled, *, prompt="same", support=False, hints=0, usage=None, status="completed"):
    return {"call_id": f"{question}.r1.{'on' if enabled else 'off'}", "question_id": question,
            "repetition": 1, "assistance_enabled": enabled, "status": status, "corpus_digest": "same-corpus",
            "source_content_digest": "same-source", "ingestion_latency_ms": 100, "corpus_clone_latency_ms": 2,
            "total_answer_latency_ms": 1000, "post_model_latency_ms": 20,
            "provider": {"status": "ok", "model_actual": "step-5-preview", "finish_reason": "stop",
                         "prompt_digest": prompt, "parameters": {"temperature": 0, "max_tokens": 2048},
                         "pre_model_latency_ms": 30, "model_latency_ms": 950,
                         "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                                   "prompt_tokens_details": {"cached_tokens": 0},
                                   "completion_tokens_details": {"reasoning_tokens": 0}} if usage is None else usage},
            "assistance": {"requested": enabled, "has_actual_support": support},
            "result": {"answer": "Supply increases [S1].", "answer_mode": "evidence",
                       "retrieval_gaps": [{"code": "uncovered_query_terms", "message": "Some terms have no source match."}],
                       "sources": [{"id": "PRIVATE_CHUNK", "label": "S1", "paper_id": "paper", "page": 1,
                                    "content": "Supply increases.", "excerpt_start": 0, "excerpt_end": 17, "truncated": False}],
                       "reasoning": {"conclusions": [{"secret_algorithm": "do not expose"}] if support else [],
                                     "trace": {"assist_bytes": hints, "answer_model_calls": 1}}}}


def records(*values):
    return {(value["question_id"], value["repetition"], value["assistance_enabled"]): value for value in values}


def test_only_matched_successful_same_model_pairs_enter_primary_and_diagnostics_stay_separate():
    off, on = answer("q1", False), answer("q1", True)
    truncated = answer("q2", True, status="incomplete_answer")
    partial = answer("q2", False)
    control_off, control_on = answer("control", False), answer("control", True)
    control_off["status"] = control_on["status"] = "completed_without_provider"
    control_off["provider"] = control_on["provider"] = {}
    pairs, diagnostics, excluded = collect_pairs(fixture(), records(off, on, partial, truncated, control_off, control_on))
    assert len(pairs) == 1 and pairs[0]["question_id"] == "q1"
    assert len(diagnostics) == 1 and diagnostics[0]["question_id"] == "control"
    assert excluded[0]["arm_reasons"]["on"] == "incomplete_answer"
    on["provider"]["model_actual"] = "different-model"
    assert not collect_pairs(fixture(), records(off, on))[0]


def test_legacy_completed_records_are_rechecked_for_failed_actual_delivery():
    off, on = answer('q1', False), answer('q1', True)
    on['provider']['answer'] = 'Valid generated answer [S1].'
    on['result']['answer'] = 'Localized nonempty withdrawal notice.'
    on['result']['sources'] = []
    on['result']['citation_validation'] = {'valid': False, 'warnings': [{'code': 'provenance_unavailable'}]}
    assert on['status'] == 'completed'  # original artifact retained, not rewritten
    assert eligibility(on) == 'delivery_provenance_unavailable'
    pairs, _, excluded = collect_pairs(fixture(), records(off, on))
    assert pairs == []
    assert excluded[0]['arm_reasons'] == {'off': None, 'on': 'delivery_provenance_unavailable'}
    assert on['provider']['usage']['total_tokens'] == 120
    assert on['status'] == 'completed'


def test_expected_model_is_explicit_and_preserves_step5_default():
    off, on = answer("q1", False), answer("q1", True)
    values = records(off, on)
    default = metrics(fixture(), values, [])
    assert default["actual_model_required"] == "step-5-preview"
    assert default["primary"]["pair_count"] == 1
    for record in (off, on):
        record["provider"]["model_actual"] = "step-3.7-flash"
    assert eligibility(off) == "unexpected_actual_model"
    assert not collect_pairs(fixture(), values)[0]
    assert eligibility(off, expected_model="step-3.7-flash") is None
    result = metrics(fixture(), values, [], expected_model="step-3.7-flash")
    assert result["actual_model_required"] == "step-3.7-flash"
    assert result["primary"]["pair_count"] == 1


@pytest.mark.parametrize("actual_model", ["step-5-preview", "STEP-3.7-FLASH", "", None])
def test_expected_model_excludes_mixed_mismatched_or_missing_actual_models(actual_model):
    off, on = answer("q1", False), answer("q1", True)
    off["provider"]["model_actual"] = "step-3.7-flash"
    if actual_model is None:
        on["provider"].pop("model_actual")
    else:
        on["provider"]["model_actual"] = actual_model
    pairs, _, excluded = collect_pairs(fixture(), records(off, on), expected_model="step-3.7-flash")
    assert pairs == []
    assert excluded[0]["arm_reasons"] == {"off": None, "on": "unexpected_actual_model"}
    if actual_model is None:
        assert eligibility(on, expected_model=None) == "unexpected_actual_model"


@pytest.mark.parametrize("model_argument,expected_model", [
    ([], "step-5-preview"), (["--model", "step-3.7-flash"], "step-3.7-flash"),
])
def test_cli_model_is_used_for_metrics_and_blind_packets(tmp_path, capsys, model_argument, expected_model):
    spec = fixture()
    spec["questions"] = spec["questions"][:1]
    spec["questions"][0]["key_facts"] = []
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(spec))
    run_dir = tmp_path / "run"
    answers = run_dir / "answers"
    answers.mkdir(parents=True)
    for enabled in (False, True):
        record = answer("q1", enabled)
        record["provider"]["model_actual"] = expected_model
        # Synthetic source without page references: no PDF or model loading.
        record["result"]["sources"][0].pop("page")
        (answers / (record["call_id"] + ".json")).write_text(json.dumps(record))
    output = tmp_path / "analysis"
    assert main(["--fixture", str(fixture_path), "--run-dir", str(run_dir),
                 "--papers-dir", str(tmp_path), "--output", str(output), *model_argument]) == 0
    summary = json.loads(capsys.readouterr().out)
    report = json.loads((output / "metrics.json").read_text())
    inventory = json.loads((output / "packet_inventory.json").read_text())
    assert report["actual_model_required"] == expected_model
    assert report["primary"]["pair_count"] == summary["eligible_pairs"] == 1
    assert summary["blind_packets"] == len(inventory["packets"]) == 1
    packet = json.loads(Path(inventory["packets"][0]["path"]).read_text())
    assert len(packet["candidates"]) == 2
    assert packet["original_pdf_reference_pages"] == []
    assert expected_model not in json.dumps(packet)


def test_usage_unknown_is_null_and_paired_differences_are_not_differences_of_medians():
    first_off, first_on = answer("q1", False), answer("q1", True)
    second_off, second_on = answer("q2", False), answer("q2", True)
    first_on["provider"]["usage"]["total_tokens"] = 200
    second_off["provider"]["usage"]["total_tokens"] = 300
    second_on["provider"]["usage"]["total_tokens"] = 250
    second_on["provider"]["usage"].pop("completion_tokens")
    result = metrics(fixture(), records(first_off, first_on, second_off, second_on), [])
    totals = result["primary"]["usage"]["total_tokens"]
    assert totals["paired_delta_mean"] == 15 and totals["paired_delta_median"] == 15
    completions = result["primary"]["usage"]["completion_tokens"]
    assert completions["measured_pair_count"] == 1 and completions["unavailable_pair_count"] == 1
    assert result["pairs"][1]["usage"]["completion_tokens"]["on"] is None
    assert result["pairs"][1]["usage"]["completion_tokens"]["delta"] is None
    assert result["primary"]["latency_ms"]["model_latency_ms"]["paired_delta_mean"] == 0


def test_exposure_strata_and_vendor_cache_changes_are_reported_independently():
    off, on = answer("q1", False), answer("q1", True, prompt="changed", hints=400)
    on["provider"]["usage"]["prompt_tokens_details"]["cached_tokens"] = 50
    assert exposure(off, on) == "hint_only"
    result = metrics(fixture(), records(off, on), [])
    assert result["exposure_strata"]["hint_only"]["pair_count"] == 1
    assert result["cache_comparison"][0]["cache_hit_changed"] is True
    assert result["cache_comparison"][0]["on_cached_tokens"] == 50
    on["assistance"]["has_actual_support"] = True
    assert exposure(off, on) == "actual_support"
    on["provider"]["prompt_digest"] = off["provider"]["prompt_digest"]
    assert exposure(off, on) == "identical_prompt"  # actual input wins over a requested operator flag


def test_insight_pairs_require_identical_background_controls_and_explicit_modes():
    from scripts.benchmark_research_assist import arm_configuration

    off, on = answer("q1", False), answer("q1", True, prompt="changed")
    for value, enabled in ((off, False), (on, True)):
        value["contrast"] = "insight"
        value["arm_configuration"] = arm_configuration(enabled, "insight")
    pairs, _, excluded = collect_pairs(fixture(), records(off, on))
    assert len(pairs) == 1 and pairs[0]["contrast"] == "insight"
    result = metrics(fixture(), records(off, on), [])
    assert result["contrast_counts"] == {"insight": 1}
    on["arm_configuration"]["context_assistance_enabled"] = False
    assert collect_pairs(fixture(), records(off, on))[2][0]["reason"] == "insight_contrast_configuration_mismatch"
    on["contrast"] = "context-assistance"
    assert collect_pairs(fixture(), records(off, on))[2][0]["reason"] == "contrast_mismatch"


def test_insight_exposure_requires_actual_included_context_or_retained_changed_evidence():
    off, on = answer("q1", False), answer("q1", True, prompt="changed")
    on["contrast"] = "insight"
    trace = on["result"]["reasoning"]["trace"]
    trace["insight_runtime"] = {"analysis_included": True, "supplement": {"kept_source_ids": []}}
    assert exposure(off, on) == "insight_context"
    trace["insight_runtime"]["supplement"]["kept_source_ids"] = ["actual-S2"]
    assert exposure(off, on) == "insight_context"  # an ID alone does not prove different evidence
    on["source_content_digest"] = "different-source"
    assert exposure(off, on) == "insight_context_and_evidence"
    trace["insight_runtime"]["analysis_included"] = False
    assert exposure(off, on) == "insight_evidence"
    on["provider"]["prompt_digest"] = off["provider"]["prompt_digest"]
    assert exposure(off, on) == "identical_prompt"


def test_setup_costs_are_paired_separately_and_shared_ingestion_is_not_multiplied():
    first_off, first_on = answer("q1", False), answer("q1", True)
    second_off, second_on = answer("q2", False), answer("q2", True)
    for record, prewarm in ((first_off, 100), (first_on, 150)):
        record.update(retrieval_setup_latency_ms=prewarm, per_arm_setup_latency_ms=prewarm + 2,
                      upfront_setup_latency_ms=prewarm + 2 + 100)
    result = metrics(fixture(), records(first_off, first_on, second_off, second_on), [])
    setup = result["primary"]["setup_latency_ms"]
    assert setup["retrieval_setup_latency_ms"]["off_mean"] == 100
    assert setup["retrieval_setup_latency_ms"]["on_mean"] == 150
    assert setup["per_arm_setup_latency_ms"]["paired_delta_mean"] == 50
    assert setup["upfront_setup_latency_ms"]["off_mean"] == 202
    assert setup["upfront_setup_latency_ms"]["on_mean"] == 252
    assert all(value["measured_pair_count"] == 1 and value["unavailable_pair_count"] == 1
               for value in setup.values())
    assert result["pairs"][1]["setup_latency_ms"]["retrieval_setup_latency_ms"] == {
        "off": None, "on": None, "delta": None,
    }
    assert result["corpus_ingestion"] == {"same-corpus": {"ingestion_latency_ms": 100}}
    assert "never sum upfront_setup_latency_ms" in result["setup_cost_definition"]
    assert "Query encoding and retrieval remain inside" in result["latency_definition"]
    assert result["primary"]["latency_ms"]["total_answer_latency_ms"]["off_mean"] == 1000


def test_semantic_runtime_status_distinguishes_availability_timeout_busy_and_missing():
    first_off, first_on = answer("q1", False), answer("q1", True)
    second_off, second_on = answer("q2", False), answer("q2", True)
    control = answer("control", False)
    for record, status in ((first_off, "ready"), (first_on, "timed_out"), (second_off, "busy"), (control, "ready")):
        record["result"]["reasoning"]["trace"]["semantic_retrieval"] = {"status": status}
    result = metrics(fixture(), records(first_off, first_on, second_off, second_on, control), [])
    primary = result["primary"]["semantic_retrieval"]
    assert primary["off"]["status_counts"] == {"ready": 1, "busy": 1}
    assert primary["off"]["available_count"] == 1 and primary["off"]["busy_count"] == 1
    assert primary["on"]["timed_out_count"] == 1 and primary["on"]["missing_count"] == 1
    assert primary["on"]["available_count"] == 0
    assert result["pairs"][1]["semantic_retrieval_status"]["on"] == "missing"
    # A diagnostic operation is visible in all-record accounting, not promoted
    # into the matched primary answer-quality comparison.
    assert result["semantic_retrieval_all_answer_records"]["off"]["available_count"] == 2
    assert result["primary"]["pair_count"] == 2


def test_source_organization_and_formal_auxiliary_are_distinct_exposure_strata():
    first_off, first_on = answer("q1", False), answer("q1", True, prompt="organization", support=True)
    second_off, second_on = answer("q2", False), answer("q2", True, prompt="formal", support=True)
    first_on["result"]["reasoning"]["conclusions"] = [{"kind": "source_evidence", "epistemic_status": "source_organization"}]
    second_on["result"]["reasoning"]["conclusions"] = [{"kind": "deduced"}, {"kind": "analogy_hypothesis"}]
    result = metrics(fixture(), records(first_off, first_on, second_off, second_on), [])
    assert result["exposure_strata"]["source_organization"]["pair_count"] == 1
    assert result["exposure_strata"]["formal_auxiliary"]["pair_count"] == 1
    assert result["pairs"][0]["support_kind"]["on"] == {
        "category": "source_organization", "item_counts": {"source_organization": 1},
    }
    assert result["pairs"][1]["support_kind"]["on"]["item_counts"] == {"formal_auxiliary": 2}
    assert "not a deduction or causal proof" in result["support_kind_definition"]
    mixed = deepcopy(first_on)
    mixed["result"]["reasoning"]["conclusions"].append({"kind": "inductive_candidate"})
    assert exposure(first_off, mixed) == "mixed_source_and_formal"
    mixed["result"]["reasoning"]["conclusions"].append({"untyped_legacy": True})
    assert exposure(first_off, mixed) == "mixed_typed_and_unclassified_support"
    mixed["provider"]["prompt_digest"] = first_off["provider"]["prompt_digest"]
    assert exposure(first_off, mixed) == "identical_prompt"


def test_all_consumption_includes_failed_receipts_and_unconfirmed_dispatches(tmp_path):
    events = [
        {"call_id": "success", "event": "dispatch_started"},
        {"call_id": "success", "event": "provider_completed", "status": "ok", "usage": {"total_tokens": 100}},
        {"call_id": "truncated", "event": "dispatch_started"},
        {"call_id": "truncated", "event": "provider_completed", "status": "empty_provider_answer", "finish_reason": "length",
         "usage": {"total_tokens": 80}},
        {"call_id": "unknown", "event": "dispatch_started"}]
    (tmp_path / "calls.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n")
    result = provider_accounting(tmp_path)
    assert result["dispatched_call_count"] == 3 and result["indeterminate_call_count"] == 1
    total = result["usage_including_failed_and_incomplete_calls"]["total_tokens"]
    assert total["reported_total"] is None and total["known_reported_subtotal"] == 180
    assert total["unknown_call_count"] == 1
    assert result["non_ok_calls"][0]["usage"]["total_tokens"] == 80


def page_loader(identity, number):
    return {"paper_id": identity, "title": "Public Paper", "pdf_page": number, "text": "Supply increases in the controlled study."}


def test_blind_packets_hide_arm_and_internal_metrics_and_keep_random_order_stable(tmp_path):
    off, on = answer("q1", False), answer("q1", True, prompt="changed", support=True)
    pairs = collect_pairs(fixture(), records(off, on))[0]
    mapping, inventory = build_blind_packets(fixture(), pairs, tmp_path, tmp_path, page_loader=page_loader)
    assert len(inventory) == 1
    packet_path = tmp_path / "blind" / "batch_01" / (inventory[0]["packet_id"] + ".json")
    packet = json.loads(packet_path.read_text())
    text = packet_path.read_text()
    assert all(key not in text for key in ("assistance_enabled", "model_latency_ms", "prompt_tokens", "PRIVATE_CHUNK", "secret_algorithm", "call_id"))
    assert {row["candidate_id"] for row in packet["candidates"]} == set(mapping["packets"][packet["packet_id"]]["candidates"])
    assert packet["candidates"][0]["actually_supplied_sources"][0]["label"] == "S1"
    assert packet["candidates"][0]["supplied_retrieval_metadata"] == {
        "retrieval_gaps": [{"code": "uncovered_query_terms", "message": "Some terms have no source match."}]}
    assert "not paper evidence" in packet["review_instructions"]
    assert packet["original_pdf_reference_pages"][0]["pdf_page"] == 1
    assert packet["forbidden_claims"] == ["Always perfect."]
    _, second = build_blind_packets(fixture(), pairs, tmp_path, tmp_path, page_loader=page_loader)
    assert second == inventory and json.loads(packet_path.read_text()) == packet
    assert (tmp_path / "private_mapping.json").stat().st_mode & 0o077 == 0


def test_blind_quality_second_pass_unmasks_only_after_dimension_validation(tmp_path):
    off, on = answer("q1", False), answer("q1", True, prompt="changed", hints=100)
    pairs = collect_pairs(fixture(), records(off, on))[0]
    mapping, inventory = build_blind_packets(fixture(), pairs, tmp_path / "output", tmp_path, page_loader=page_loader)
    packet_id = inventory[0]["packet_id"]
    entry = mapping["packets"][packet_id]
    dimensions = [row["id"] for row in fixture()["rubric"]["dimensions"]]
    reviews = []
    for candidate_id, candidate in entry["candidates"].items():
        value = 4 if candidate["arm"] == "on" else 3
        reviews.append({"candidate_id": candidate_id, "dimensions": dict.fromkeys(dimensions, value), "critical_errors": [],
                        "rationale": "Cited fact checked against S1."})
    scores = tmp_path / "scores"
    scores.mkdir()
    (scores / "review.json").write_text(json.dumps({"packet_id": packet_id, "reviewer_id": "blind-reviewer",
                                                  "reviewer_kind": "agent_blind_review", "candidates": reviews}))
    quality = aggregate_scores(fixture(), mapping, pairs, scores)
    assert quality["off_mean"] == 75 and quality["on_mean"] == 100
    assert quality["paired_delta_mean"] == 25 and quality["wins_on"] == 1
    assert quality["exposure_strata"]["hint_only"]["paired_delta_mean"] == 25
    assert quality["reviewer_kinds"] == {"agent_blind_review": 2}
    reviews[0]["critical_errors"] = ["Central invented result"]
    (scores / "review.json").write_text(json.dumps({"packet_id": packet_id, "reviewer_id": "blind-reviewer", "candidates": reviews}))
    quality = aggregate_scores(fixture(), mapping, pairs, scores)
    assert quality["score_errors"][0]["error"] == "critical_error_requires_zero_fact_and_scope"


def test_blind_packet_carries_requested_language_without_changing_rubric(tmp_path):
    spec = fixture()
    spec['questions'][0]['language'] = 'en'
    original = deepcopy(spec)
    pairs = collect_pairs(spec, records(answer('q1', False), answer('q1', True)))[0]
    _, inventory = build_blind_packets(spec, pairs, tmp_path, tmp_path, page_loader=page_loader)
    packet = json.loads(Path(inventory[0]['path']).read_text())
    assert packet['question_language'] == 'en'
    assert 'language requested by the question' in packet['language_scoring_rule']
    assert packet['rubric'] == original['rubric']
    assert spec == original


def test_repetition_actual_dispatch_order_and_known_cache_match_strata(tmp_path):
    first_off, first_on = answer("q1", False), answer("q1", True)
    second_off, second_on = deepcopy(first_off), deepcopy(first_on)
    for record in (second_off, second_on):
        record["repetition"] = 2
        record["call_id"] = record["call_id"].replace(".r1.", ".r2.")
    second_on["provider"]["usage"]["prompt_tokens_details"]["cached_tokens"] = 30
    third_off, third_on = answer("q2", False), answer("q2", True)
    third_on["provider"]["usage"].pop("prompt_tokens_details")
    events = [
        {"event": "dispatch_started", "call_id": first_on["call_id"], "recorded_at": "2026-10-10T12:00:03+00:00"},
        # UTC clock went backwards; durable sequential line order is explicit.
        {"event": "dispatch_started", "call_id": first_off["call_id"], "recorded_at": "2026-10-10T12:00:02+00:00"},
        {"event": "dispatch_started", "call_id": second_off["call_id"], "recorded_at": "2026-10-10T12:01:00+00:00"},
        {"event": "dispatch_started", "call_id": second_on["call_id"], "recorded_at": "2026-10-10T12:01:01+00:00"}]
    (tmp_path / "calls.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n")
    result = metrics(fixture(), records(first_off, first_on, second_off, second_on, third_off, third_on), [], run_directory=tmp_path)
    assert result["primary"]["pair_count"] == 3 and result["primary"]["unique_question_count"] == 2
    assert result["by_repetition"]["1"]["pair_count"] == 2 and result["by_repetition"]["2"]["pair_count"] == 1
    assert result["by_first_called_arm"]["on"]["pair_count"] == 1
    assert result["by_first_called_arm"]["off"]["pair_count"] == 1
    assert result["by_first_called_arm"]["unknown"]["pair_count"] == 1
    assert result["pairs"][0]["call_order"]["first_called_arm"] == "on"
    assert result["pairs"][0]["call_order"]["dispatch_events"]["on"]["journal_line"] == 1
    assert result["pairs"][0]["call_order"]["dispatch_events"]["on"]["recorded_at"] == "2026-10-10T12:00:03+00:00"
    assert result["pairs"][2]["call_order"]["unknown_reason"] == "missing_actual_dispatch_event"
    assert all(result["cached_token_match_strata"][name]["pair_count"] == 1 for name in ("equal_known", "different_known", "unknown"))
    assert result["by_first_called_arm"]["on"]["latency_ms"]["total_answer_latency_ms"]["on_mean"] == 1000
    assert result["cached_token_match_strata"]["different_known"]["usage"]["cached_tokens"]["paired_delta_mean"] == 30


def test_quality_question_balancing_does_not_count_repetitions_as_new_questions(tmp_path):
    first_off, first_on = answer("q1", False), answer("q1", True)
    second_off, second_on = deepcopy(first_off), deepcopy(first_on)
    for record in (second_off, second_on):
        record["repetition"] = 2
        record["call_id"] = record["call_id"].replace(".r1.", ".r2.")
    third_off, third_on = answer("q2", False), answer("q2", True)
    pairs = collect_pairs(fixture(), records(first_off, first_on, second_off, second_on, third_off, third_on))[0]
    mapping, _ = build_blind_packets(fixture(), pairs, tmp_path / "output", tmp_path, page_loader=page_loader)
    dimensions = [row["id"] for row in fixture()["rubric"]["dimensions"]]
    scores = tmp_path / "scores"
    scores.mkdir()
    for packet_id, entry in mapping["packets"].items():
        candidates = []
        for candidate_id, candidate in entry["candidates"].items():
            value = (3 if candidate["arm"] == "off" else 4) if entry["pair_key"] != "q1.r2" else (4 if candidate["arm"] == "off" else 3)
            candidates.append({"candidate_id": candidate_id, "dimensions": dict.fromkeys(dimensions, value), "critical_errors": []})
        (scores / (packet_id + ".json")).write_text(json.dumps({"packet_id": packet_id, "reviewer_id": "blind-reviewer", "candidates": candidates}))
    quality = aggregate_scores(fixture(), mapping, pairs, scores)
    assert quality["scored_reviewer_pair_count"] == 3 and quality["unique_question_count"] == 2
    assert quality["by_repetition"]["1"]["unique_question_count"] == 2
    assert quality["by_repetition"]["2"]["unique_question_count"] == 1
    assert quality["by_question"]["q1"]["paired_delta_mean"] == 0
    assert quality["question_balanced"]["question_count"] == 2
    assert quality["question_balanced"]["paired_delta_mean"] == 12.5
    assert quality["paired_delta_mean"] == pytest.approx(25 / 3)


def test_reselection_requires_trace_and_changed_actual_quotes_not_support_or_digest():
    off, on = answer("q1", False), answer("q1", True, prompt="changed")
    on["result"]["reasoning"]["trace"]["retrieval_reselection"] = True
    on["source_content_digest"] = "claimed-change"
    assert exposure(off, on) == "changed_without_visible_assist"
    on["result"]["sources"][0]["content"] = "A different supported experiment."
    assert exposure(off, on) == "evidence_reselection"
    report = metrics(fixture(), records(off, on), [])
    assert report["exposure_strata"]["evidence_reselection"]["pair_count"] == 1
    assert report["pairs"][0]["support_kind"]["on"] == {"category": "none", "item_counts": {}}
    on["result"]["reasoning"]["trace"]["retrieval_reselection"] = False
    assert exposure(off, on) == "changed_without_visible_assist"
    on["result"]["reasoning"]["trace"]["retrieval_reselection"] = True
    on["provider"]["prompt_digest"] = off["provider"]["prompt_digest"]
    assert exposure(off, on) == "identical_prompt"


def test_blind_packet_retrieval_audit_contract_is_independent_of_answer_and_hides_split(tmp_path):
    spec = fixture()
    spec["questions"][0]["eval_split"] = "new_question_no_tuning"
    off, on = answer("q1", False), answer("q1", True)
    # Fixture metadata, never an arm-controlled record, defines the split.
    on["eval_split"] = "wrong-record-split"
    pairs = collect_pairs(spec, records(off, on))[0]
    mapping, inventory = build_blind_packets(spec, pairs, tmp_path, tmp_path, page_loader=page_loader)
    packet = json.loads((tmp_path / "blind" / "batch_01" / (inventory[0]["packet_id"] + ".json")).read_text())
    assert "new_question_no_tuning" not in json.dumps(packet)
    contract = packet["score_output_contract"]["candidates"][0]
    assert contract["retrieved_evidence_checks"][0]["fact_id"] == "f1"
    assert contract["retrieved_evidence_checks"][0]["status"] == "full | partial | absent"
    assert "irrelevant_source_labels" in contract
    assert "independently of whether the answer mentions it" in packet["review_instructions"]
    report = metrics(spec, records(off, on), [])
    assert report["by_eval_split"]["new_question_no_tuning"]["pair_count"] == 1
    assert report["pairs"][0]["eval_split"] == "new_question_no_tuning"


def test_retrieval_audit_missing_incomplete_and_partial_are_not_zero_or_full():
    question = fixture()["questions"][0]
    question["key_facts"].append({"id": "f2", "text": "A necessary qualification."})
    record = answer("q1", True)
    missing = retrieved_evidence_audit({}, question, record)
    assert missing["status"] == "missing" and missing["evidence_recall"] is None
    assert missing["full_fact_count"] is None and missing["irrelevant_source_label_rate"] is None
    item = {"retrieved_evidence_checks": [
        {"fact_id": "f1", "status": "full", "source_labels": ["S1"], "reason": "S1 provides the whole fact."}]}
    incomplete = retrieved_evidence_audit(item, question, record)
    assert incomplete["status"] == "incomplete" and incomplete["evidence_recall"] is None
    assert incomplete["checked_fact_count"] == 1 and incomplete["unreviewed_fact_count"] == 1
    item["retrieved_evidence_checks"].append({"fact_id": "f2", "status": "partial", "source_labels": ["S1"],
                                          "reason": "The premise is present but not its qualification."})
    item.update(irrelevant_source_labels=[], irrelevant_source_reason="The only source supports both facts in part.")
    audit = retrieved_evidence_audit(item, question, record)
    assert audit["status"] == "complete" and audit["evidence_recall"] == 0.5
    assert audit["partial_evidence_rate"] == 0.5 and audit["all_facts_full"] is False
    assert audit["irrelevant_source_label_rate"] == 0


@pytest.mark.parametrize("check", [
    {"fact_id": "unknown", "status": "full", "source_labels": ["S1"], "reason": "No gold fact."},
    {"fact_id": "f1", "status": "full", "source_labels": ["S99"], "reason": "No real label."},
    {"fact_id": "f1", "status": "full", "source_labels": [], "reason": "Remembered fact is not a source."},
    {"fact_id": "f1", "status": "absent", "source_labels": ["S1"], "reason": "Absent has no supporting label."},
    {"fact_id": "f1", "status": "covered", "source_labels": ["S1"], "reason": "Answer coverage is different."},
])
def test_retrieval_audit_rejects_unknown_facts_labels_and_answer_status(check):
    with pytest.raises(ValueError, match="invalid_retrieved_evidence_checks"):
        retrieved_evidence_audit({"retrieved_evidence_checks": [check]}, fixture()["questions"][0], answer("q1", True))


def test_wholly_irrelevant_labels_must_be_real_unique_and_not_supporting_a_fact():
    question, record = fixture()["questions"][0], answer("q1", True)
    for labels in (["S99"], ["S1", "S1"]):
        with pytest.raises(ValueError, match="invalid_irrelevant_source_labels"):
            retrieved_evidence_audit({"irrelevant_source_labels": labels, "irrelevant_source_reason": "Unused."}, question, record)
    with pytest.raises(ValueError, match="invalid_irrelevant_source_labels"):
        retrieved_evidence_audit({"retrieved_evidence_checks": [
            {"fact_id": "f1", "status": "partial", "source_labels": ["S1"], "reason": "Some support."}],
            "irrelevant_source_labels": ["S1"], "irrelevant_source_reason": "Contradictory annotation."}, question, record)


def test_quality_and_retrieval_audit_are_split_and_missing_reviews_count_as_unavailable(tmp_path):
    spec = fixture()
    spec["questions"][0]["eval_split"] = "development_regression"
    spec["questions"][1]["eval_split"] = "new_question_no_tuning"
    spec["questions"][1]["key_facts"] = deepcopy(spec["questions"][0]["key_facts"])
    values = [answer(q, enabled) for q in ("q1", "q2") for enabled in (False, True)]
    pairs = collect_pairs(spec, records(*values))[0]
    mapping, _ = build_blind_packets(spec, pairs, tmp_path / "output", tmp_path, page_loader=page_loader)
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    dimensions = [row["id"] for row in spec["rubric"]["dimensions"]]
    for packet_id, entry in mapping["packets"].items():
        candidates = []
        for candidate_id, candidate in entry["candidates"].items():
            item = {"candidate_id": candidate_id, "dimensions": dict.fromkeys(dimensions, 3), "critical_errors": []}
            if entry["question_id"] == "q1":
                full = candidate["arm"] == "on"
                item.update(retrieved_evidence_checks=[{"fact_id": "f1", "status": "full" if full else "absent",
                    "source_labels": ["S1"] if full else [], "reason": "The actual pack is sufficient." if full else "The required relation is absent."}],
                    irrelevant_source_labels=[] if full else ["S1"], irrelevant_source_reason="Relevance independently reviewed.")
            candidates.append(item)
        (score_dir / (packet_id + ".json")).write_text(json.dumps({"packet_id": packet_id, "reviewer_id": "blind", "candidates": candidates}))
    quality = aggregate_scores(spec, mapping, pairs, score_dir)
    assert quality["score_coverage"] == {"eligible_generation_pair_count": 2, "scored_generation_pair_count": 2,
                                         "unscored_generation_pair_count": 0}
    assert quality["off_mean"] == quality["on_mean"] == 75
    dev = quality["by_eval_split"]["development_regression"]
    fresh = quality["by_eval_split"]["new_question_no_tuning"]
    assert dev["retrieval_audit"]["evidence_recall"]["paired_delta_mean"] == 1
    assert dev["retrieval_audit"]["irrelevant_source_label_rate"]["paired_delta_mean"] == -1
    assert fresh["retrieval_audit"]["evidence_recall"]["off_mean"] is None
    assert fresh["retrieval_audit"]["evidence_recall"]["unavailable_pair_count"] == 1
    assert quality["retrieval_audit"]["evidence_recall"]["measured_pair_count"] == 1
    assert quality["retrieval_audit"]["candidate_audit_coverage"]["off"]["missing_fact_audit_count"] == 1
    assert not quality["score_errors"]


def test_quality_split_with_no_scores_has_null_means_and_explicit_unscored_coverage(tmp_path):
    spec = fixture()
    spec["questions"][0]["eval_split"] = "new_question_no_tuning"
    pairs = collect_pairs(spec, records(answer("q1", False), answer("q1", True)))[0]
    mapping, _ = build_blind_packets(spec, pairs, tmp_path / "out", tmp_path, page_loader=page_loader)
    report = aggregate_scores(spec, mapping, pairs, tmp_path / "not-created")
    assert report["score_coverage"]["unscored_generation_pair_count"] == 1
    assert report["by_eval_split"]["new_question_no_tuning"]["off_mean"] is None
    assert report["retrieval_audit"]["evidence_recall"]["off_mean"] is None
