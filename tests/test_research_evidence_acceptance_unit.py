"""Synthetic post-run accounting only; no real questions, models, or gold."""
from copy import deepcopy
import hashlib
import json
import sys

import pytest

from benchmarks.research_real import evaluate_evidence_acceptance as tool


def fixtures():
    protocol = {"frozen_descriptive_acceptance": {"quality": "nonnegative", "strict_evidence": "full only"},
                "model_and_budget": {"actual_returned_model_required": "model", "endpoint": "https://provider.example/v1"}}
    summary = {"schema": "evidence-trace-question-balanced-review-summary-v1", "run_finished_confirmed_by_caller": True,
               "planned_question_count": 2, "reviewer_ids": list(tool.REVIEWERS), "per_question": [],
               "frozen_acceptance": deepcopy(protocol["frozen_descriptive_acceptance"]),
               "by_language": {"en": {"preserved": True}, "zh": {"preserved": True}},
               **{key: [] for key in ["validation_issues", "upstream_artifact_errors", "upstream_score_errors",
                                      "upstream_partial_reviews", "excluded_or_incomplete_generations"]}}
    metrics = {"pairs": [], "artifact_errors": [], "excluded": [], "diagnostics": [],
               "quality": {"paired_reviews": [], "score_errors": [], "partial_reviews": []},
               "all_provider_consumption": {"dispatched_call_count": 4, "provider_receipt_count": 4,
                    "indeterminate_call_count": 0, "non_ok_calls": [], "journal_errors": []}}
    mapping = {"packets": {}}
    packets = []
    audit = {"schema": "evidence-answer-pipeline-provider-audit-v1", "run_finished": True,
             "main_calls": [], "diagnostic_calls": [], "dispatched_call_count": 4,
             "provider_receipt_count": 4, "indeterminate_call_count": 0}
    for index, split in enumerate(tool.SPLITS):
        qid, packet_id = f"q{index}", f"packet{index}"
        packet = {"packet_id": packet_id, "question": "A synthetic instrumentation question.",
                  "required_facts": [{"id": "f1"}, {"id": "f2"}], "original_pdf_reference_pages": [], "candidates": []}
        mapped = {"packet_id": packet_id, "question_id": qid, "repetition": 1, "candidates": {}}
        pair = {"question_id": qid, "repetition": 1, "eval_split": split,
                "on_assistance": {"scheduler_status": "timed_out" if index == 0 else "completed"},
                "usage": {name: {"off": value, "on": value - 10, "delta": -10}
                          for name, value in [("prompt_tokens", 100), ("total_tokens", 200)]}}
        row = {"question_id": qid, "eval_split": split, "language": "zh" if index == 0 else "en", "fact_count": 2,
               "pair": deepcopy(pair), "reviews": {}, "quality": {"off": 100., "on": 100., "delta": 0.},
               "active_selection": index == 1}
        for arm in tool.ARMS:
            cid, call_id = f"candidate_{arm}", f"{qid}.r1.{arm}"
            packet["candidates"].append({"candidate_id": cid, "actually_supplied_sources": [
                {"label": "S1", "excerpt": arm + " synthetic measured procedure."},
                {"label": "S2", "excerpt": arm + " synthetic control."}]})
            mapped["candidates"][cid] = {"arm": arm, "call_id": call_id}
            audit["main_calls"].append({"question_id": qid, "repetition": 1, "arm": arm, "call_id": call_id,
                "adapter_calls": 1, "auxiliary_provider_calls": 0, "endpoint_base": "https://provider.example/v1",
                "actual_model": "model", "http_status": 200, "finish_reason": "stop"})
        for reviewer in tool.REVIEWERS:
            raw_pair = {"question_id": qid, "reviewer_id": reviewer, "repetition": 1}
            row["reviews"][reviewer] = {}
            for arm in tool.ARMS:
                facts = {"f1": "full", "f2": "partial" if index == 1 and arm == "off" else "full"}
                reduced = {"weighted_score": 100., "dimensions": {key: 4 for key in tool.WEIGHTS}, "critical_errors": [],
                           "retrieved_facts": facts, "answer_facts": {"f1": "covered", "f2": "covered"},
                           "source_label_count": 2, "irrelevant_source_labels": []}
                row["reviews"][reviewer][arm] = reduced
                raw_pair[arm + "_review"] = {"packet_id": packet_id, "call_id": f"{qid}.r1.{arm}",
                    "reviewer_kind": "agent_blind_review", "weighted_score": 100.,
                    "dimensions": deepcopy(reduced["dimensions"]), "critical_errors": [],
                    "key_fact_checks": [{"fact_id": key, "status": value} for key, value in reduced["answer_facts"].items()],
                    "retrieval_audit": {"retrieved_evidence_checks": [{"fact_id": key, "status": value} for key, value in facts.items()],
                                        "source_label_count": 2, "irrelevant_source_labels": []}}
            metrics["quality"]["paired_reviews"].append(raw_pair)
        summary["per_question"].append(row); metrics["pairs"].append(pair)
        packets.append(packet); mapping["packets"][packet_id] = mapped
    return dict(summary=summary, metrics=metrics, mapping=mapping, blind_packets=packets,
                adjudication_packets=[], adjudication_scores=[], protocol=protocol, provider_audit=audit)


def changed_review(data, question, reviewer, arm, *, facts=None, noise=None):
    row = data["summary"]["per_question"][question]["reviews"][reviewer][arm]
    raw = next(r for r in data["metrics"]["quality"]["paired_reviews"] if r["question_id"] == f"q{question}" and r["reviewer_id"] == reviewer)[arm + "_review"]
    if facts is not None:
        row["retrieved_facts"] = deepcopy(facts)
        raw["retrieval_audit"]["retrieved_evidence_checks"] = [{"fact_id": key, "status": value} for key, value in facts.items()]
    if noise is not None:
        row["irrelevant_source_labels"] = deepcopy(noise)
        raw["retrieval_audit"]["irrelevant_source_labels"] = deepcopy(noise)


def adjudicate(data, question=1, arms=("on",), *, facts=("f2",), noise=False, status="full"):
    packet = data["blind_packets"][question]
    source = next(row["actually_supplied_sources"] for row in packet["candidates"] if row["candidate_id"] == "candidate_" + arms[0])
    eid = "evidence-" + tool.digest(source)[:16]
    data["adjudication_packets"] = [{"schema": "blind-evidence-conflict-packet-v1", "packet_id": packet["packet_id"],
        **{key: deepcopy(packet[key]) for key in ("question", "required_facts", "original_pdf_reference_pages")}, "evidence_packs": [{
        "evidence_id": eid, "candidate_ids": ["candidate_" + arm for arm in arms], "actually_supplied_sources": deepcopy(source),
        "fact_ids_to_check": list(facts), "check_irrelevance": noise}]}]
    data["adjudication_scores"] = [{"packet_id": packet["packet_id"], "reviewer_id": "blind_evidence_arbiter", "evidence_packs": [{
        "evidence_id": eid, "fact_checks": [{"fact_id": fact, "status": status, "source_labels": [] if status == "absent" else ["S1"],
                                             "reason": "The supplied excerpt directly contains this component."} for fact in facts],
        "irrelevant_source_labels": [] if noise else None, "irrelevance_reason": "Both excerpts are relevant." if noise else None}]}]


def test_complete_screen_passes_without_mutation_and_preserves_itt_fallback_and_languages():
    data = fixtures(); unchanged = deepcopy(data)
    result = tool.build(**data)
    assert result["descriptive_screen_passed"] and result["status"] == "passed"
    assert result["overall"]["runtime_exposure"]["itt_rows_removed"] == 0
    assert result["overall"]["runtime_exposure"]["nonactive_or_fallback_question_ids"] == ["q0"]
    assert result["original_language_summary"] == data["summary"]["by_language"]
    assert set(result["by_language"]) == {"en", "zh"}
    assert result["by_eval_split"]["new_document_holdout"]["strict_evidence"]["off"]["macro_full_rate"] == .5
    assert result["release_approval"] is False
    assert data == unchanged


def test_declared_arbiter_identity_does_not_rewrite_original_review_files():
    data = fixtures()
    changed_review(data, 1, "reviewer_b", "on", facts={"f1": "full", "f2": "partial"})
    adjudicate(data)
    data["adjudication_scores"][0]["reviewer_id"] = "blind_evidence_arbiter_v7"
    before = deepcopy(data)
    with pytest.raises(ValueError, match="invalid_evidence_arbiter_identity"):
        tool.build(**data)
    result = tool.build(**data, arbiter_id="blind_evidence_arbiter_v7")
    assert result["expected_arbiter_id"] == "blind_evidence_arbiter_v7"
    assert result["descriptive_screen_passed"] and data == before
    for identity in (None, "", " ", "reviewer_a", "reviewer_b"):
        with pytest.raises(ValueError, match="independent_arbiter_identity_required"):
            tool.build(**data, arbiter_id=identity)


def test_conflict_without_adjudication_blocks_instead_of_turning_disagreement_into_absent():
    data = fixtures()
    changed_review(data, 1, "reviewer_b", "on", facts={"f1": "full", "f2": "partial"})
    result = tool.build(**data)
    assert result["status"] == "blocked_incomplete_or_invalid"
    assert result["per_question"][1]["on"]["adjudicated_facts"]["f2"] is None
    assert result["by_eval_split"]["new_document_holdout"]["strict_evidence"]["on"]["macro_full_rate"] is None


def test_valid_evidence_adjudication_resolves_only_conflict_without_rewriting_quality():
    data = fixtures()
    changed_review(data, 1, "reviewer_b", "on", facts={"f1": "full", "f2": "partial"})
    adjudicate(data)
    before = deepcopy(data)
    result = tool.build(**data)
    assert result["status"] == "passed"
    assert result["per_question"][1]["original_reviews"]["reviewer_b"]["on"]["retrieved_facts"]["f2"] == "partial"
    assert result["per_question"][1]["quality"] == before["summary"]["per_question"][1]["quality"]
    assert data == before


def test_partial_adjudicated_fact_never_counts_as_full_or_positive_holdout_gain():
    data = fixtures()
    changed_review(data, 1, "reviewer_b", "on", facts={"f1": "full", "f2": "partial"})
    adjudicate(data, status="partial")
    result = tool.build(**data)
    assert result["status"] == "failed"
    assert result["gates"]["new_document_holdout"]["positive_strict_full_fact_gain"] == "fail"
    assert result["by_eval_split"]["new_document_holdout"]["strict_evidence"]["on"]["all_facts_full_questions"] == 0


def test_identical_source_packs_must_share_one_adjudication_across_both_arms():
    data = fixtures()
    data["blind_packets"][0]["candidates"][1]["actually_supplied_sources"] = deepcopy(data["blind_packets"][0]["candidates"][0]["actually_supplied_sources"])
    for reviewer in tool.REVIEWERS:
        changed_review(data, 0, reviewer, "on", facts={"f1": "full", "f2": "partial"})
    assert tool.build(**data)["status"] == "blocked_incomplete_or_invalid"
    adjudicate(data, question=0, arms=("off", "on"), status="partial")
    result = tool.build(**data)
    assert result["status"] == "passed"
    assert result["per_question"][0]["off"]["adjudicated_facts"] == result["per_question"][0]["on"]["adjudicated_facts"]


@pytest.mark.parametrize("broken", ["missing_score", "partial_facts", "wrong_labels", "empty_reason", "changed_pack", "wrong_reviewer"])
def test_incomplete_or_mismatched_adjudication_never_passes(broken):
    data = fixtures(); changed_review(data, 1, "reviewer_b", "on", facts={"f1": "full", "f2": "partial"}); adjudicate(data)
    score = data["adjudication_scores"][0]["evidence_packs"][0]
    if broken == "missing_score": data["adjudication_scores"] = []
    if broken == "partial_facts": score["fact_checks"] = []
    if broken == "wrong_labels": score["fact_checks"][0]["source_labels"] = ["S99"]
    if broken == "empty_reason": score["fact_checks"][0]["reason"] = ""
    if broken == "changed_pack": data["adjudication_packets"][0]["evidence_packs"][0]["actually_supplied_sources"][0]["excerpt"] = "replacement"
    if broken == "wrong_reviewer": data["adjudication_scores"][0]["reviewer_id"] = "reviewer_a"
    if broken == "wrong_reviewer":
        with pytest.raises(ValueError): tool.build(**data)
    else:
        assert tool.build(**data)["descriptive_screen_passed"] is False


def test_adjudication_cannot_overwrite_agreed_facts():
    data = fixtures(); adjudicate(data)
    with pytest.raises(ValueError, match="cannot_rewrite"):
        tool.build(**data)


def test_noise_conflict_requires_adjudication_and_reports_macro_micro_separately():
    data = fixtures(); changed_review(data, 1, "reviewer_b", "on", noise=["S2"])
    assert not tool.build(**data)["descriptive_screen_passed"]
    adjudicate(data, facts=(), noise=True)
    result = tool.build(**data)
    assert result["status"] == "passed"
    assert result["overall"]["source_noise"]["on"]["source_label_denominator"] == 4
    assert result["overall"]["source_noise"]["on"]["macro_rate"] == 0


@pytest.mark.parametrize("failure", ["review", "generation", "quality", "usage", "audit", "unknown_aux", "extra_call", "wrong_model", "indeterminate", "unknown_ledger", "partial_upstream"])
def test_missing_or_invalid_inputs_cannot_silently_pass(failure):
    data = fixtures()
    if failure == "review": data["summary"]["per_question"][1]["reviews"].pop("reviewer_b")
    if failure == "generation": data["metrics"]["pairs"].pop()
    if failure == "quality": data["summary"]["per_question"][1]["quality"]["on"] = 200
    if failure == "usage":
        data["metrics"]["pairs"][1]["usage"]["prompt_tokens"]["on"] = None
        data["summary"]["per_question"][1]["pair"] = deepcopy(data["metrics"]["pairs"][1])
    if failure == "audit": data["provider_audit"] = None
    if failure == "unknown_aux": data["provider_audit"]["main_calls"][0]["auxiliary_provider_calls"] = None
    if failure == "extra_call": data["provider_audit"]["main_calls"].append(deepcopy(data["provider_audit"]["main_calls"][0]))
    if failure == "wrong_model": data["provider_audit"]["main_calls"][0]["actual_model"] = "another-model"
    if failure == "indeterminate": data["provider_audit"]["indeterminate_call_count"] = 1
    if failure == "unknown_ledger": data["metrics"]["all_provider_consumption"].pop("journal_errors")
    if failure == "partial_upstream": data["metrics"]["quality"]["partial_reviews"] = [{"status": "pending"}]
    result = tool.build(**data)
    assert result["descriptive_screen_passed"] is False
    assert result["planned_question_count"] == 2 and len(result["per_question"]) == 2


def test_token_increase_fails_mean_gate_and_is_disclosed_without_counting_cache_twice():
    data = fixtures()
    for view in [data["summary"]["per_question"][1]["pair"], data["metrics"]["pairs"][1]]:
        view["usage"]["prompt_tokens"]["on"] = 101
        view["usage"]["cached_tokens"] = {"off": 90, "on": 0, "delta": -90}
    result = tool.build(**data)
    assert result["gates"]["new_document_holdout"]["prompt_tokens_mean_no_increase"] == "fail"
    assert result["by_eval_split"]["new_document_holdout"]["usage"]["prompt_tokens"]["off"]["mean"] == 100
    assert result["overall"]["individual_prompt_or_total_token_increases"][0]["question_id"] == "q1"


def test_declared_protocol_deviation_is_separate_from_quality_gates_and_never_changes_thresholds():
    data = fixtures()
    result = tool.build(**data, protocol_deviations=[{"code": "scope_prose_mismatch", "description": "Sealed per-question scope was used unchanged."}])
    assert result["status"] == "passed" and result["full_protocol_compliance"] is False
    assert result["release_approval"] is False


def test_no_finished_signal_or_mismatched_protocol_is_rejected():
    data = fixtures(); data["summary"]["run_finished_confirmed_by_caller"] = False
    with pytest.raises(ValueError): tool.build(**data)
    data = fixtures(); data["summary"]["frozen_acceptance"] = {}
    with pytest.raises(ValueError): tool.build(**data)


def test_changed_adjudication_fact_wording_cannot_reuse_the_same_id():
    data = fixtures(); changed_review(data, 1, "reviewer_b", "on", facts={"f1": "full", "f2": "partial"}); adjudicate(data)
    data["adjudication_packets"][0]["required_facts"][1]["text"] = "Changed condition"
    with pytest.raises(ValueError, match="context_changed"):
        tool.build(**data)


def test_unknown_provider_field_is_unavailable_not_an_invented_zero():
    data = fixtures(); data["provider_audit"]["main_calls"][0]["auxiliary_provider_calls"] = None
    result = tool.build(**data)
    assert result["provider_call_gate"]["status"] == "unavailable"
    assert result["status"] == "blocked_incomplete_or_invalid"


def test_on_critical_error_is_retained_and_fails_its_own_gate():
    data = fixtures()
    # Critical flags remain present in both original metrics and summary.
    for reviewer in tool.REVIEWERS:
        reduced = data["summary"]["per_question"][1]["reviews"][reviewer]["on"]
        raw = next(row for row in data["metrics"]["quality"]["paired_reviews"] if row["question_id"] == "q1" and row["reviewer_id"] == reviewer)["on_review"]
        for review in (reduced, raw):
            review["dimensions"]["factual_accuracy"] = 0
            review["dimensions"]["scope_and_uncertainty"] = 0
            review["weighted_score"] = 60.
            review["critical_errors"] = [{"sentence": "Unsupported universal claim.", "reason": "False condition."}]
    data["summary"]["per_question"][1]["quality"] = {"off": 100., "on": 60., "delta": -40.}
    result = tool.build(**data)
    assert result["gates"]["new_document_holdout"]["no_on_critical_errors"] == "fail"
    assert result["overall"]["critical_errors"][0]["errors"][0]["sentence"] == "Unsupported universal claim."


def test_holdout_full_count_gain_can_pass_with_exactly_flat_macro_recall():
    data = fixtures()
    # Two holdout questions: 2/4 -> 4/4 and 1/2 -> 0/2. Macro is flat,
    # but full facts increase 3 -> 4 and all-full questions increase 0 -> 1.
    clone = deepcopy(data["summary"]["per_question"][1]); clone["question_id"] = "q2"; clone["pair"]["question_id"] = "q2"
    data["summary"]["per_question"].append(clone); data["summary"]["planned_question_count"] = 3
    data["metrics"]["pairs"].append(deepcopy(clone["pair"]))
    packet = deepcopy(data["blind_packets"][1]); packet["packet_id"] = "packet2"; data["blind_packets"].append(packet)
    mapped = deepcopy(data["mapping"]["packets"]["packet1"]); mapped.update(packet_id="packet2", question_id="q2")
    for entry in mapped["candidates"].values(): entry["call_id"] = entry["call_id"].replace("q1", "q2")
    data["mapping"]["packets"]["packet2"] = mapped
    for review in list(data["metrics"]["quality"]["paired_reviews"]):
        if review["question_id"] == "q1":
            copied = deepcopy(review); copied["question_id"] = "q2"
            for arm in tool.ARMS:
                copied[arm + "_review"]["packet_id"] = "packet2"
                copied[arm + "_review"]["call_id"] = f"q2.r1.{arm}"
            data["metrics"]["quality"]["paired_reviews"].append(copied)
    for call in list(data["provider_audit"]["main_calls"]):
        if call["question_id"] == "q1":
            copied = deepcopy(call); copied["question_id"] = "q2"; copied["call_id"] = copied["call_id"].replace("q1", "q2")
            data["provider_audit"]["main_calls"].append(copied)
    for value in (data["provider_audit"], data["metrics"]["all_provider_consumption"]):
        value["dispatched_call_count"] = value["provider_receipt_count"] = 6
    data["blind_packets"][1]["required_facts"] += [{"id": "f3"}, {"id": "f4"}]
    data["summary"]["per_question"][1]["fact_count"] = 4
    for reviewer in tool.REVIEWERS:
        for arm in tool.ARMS:
            facts = {"f1": "full", "f2": "full", "f3": "partial" if arm == "off" else "full", "f4": "partial" if arm == "off" else "full"}
            changed_review(data, 1, reviewer, arm, facts=facts)
            data["summary"]["per_question"][1]["reviews"][reviewer][arm]["answer_facts"] = {key: "covered" for key in facts}
            raw = next(row for row in data["metrics"]["quality"]["paired_reviews"] if row["question_id"] == "q1" and row["reviewer_id"] == reviewer)[arm + "_review"]
            raw["key_fact_checks"] = [{"fact_id": key, "status": "covered"} for key in facts]
        changed_review(data, 2, reviewer, "on", facts={"f1": "partial", "f2": "partial"})
    result = tool.build(**data)
    assert result["status"] == "passed"
    assert result["by_eval_split"]["new_document_holdout"]["strict_evidence_delta"]["macro_full_rate"] == 0
    assert result["by_eval_split"]["new_document_holdout"]["strict_evidence_delta"]["full_fact_count"] == 1


def test_cli_writes_new_owner_only_output_and_never_overwrites_inputs_or_output(tmp_path, monkeypatch):
    data = fixtures(); paths = {}
    for name in ("summary", "metrics", "mapping", "protocol", "provider_audit"):
        paths[name] = tmp_path / (name + ".json")
    data["summary"]["inputs"] = {"metrics": {"sha256": hashlib.sha256(tool.encoded(data["metrics"])).hexdigest()}}
    for name, path in paths.items(): path.write_bytes(tool.encoded(data[name]))
    blind = tmp_path / "blind"; blind.mkdir()
    for index, packet in enumerate(data["blind_packets"]): (blind / f"packet{index}.json").write_bytes(tool.encoded(packet))
    output = tmp_path / "output.json"
    arguments = ["evaluate", "--run-finished", "--blind", str(blind), "--output", str(output)]
    for name, path in paths.items(): arguments += ["--" + name.replace("_", "-"), str(path)]
    originals = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    monkeypatch.setattr(sys, "argv", arguments)
    tool.main()
    assert json.loads(output.read_bytes())["status"] == "passed"
    assert output.stat().st_mode & 0o777 == 0o600
    saved = output.read_bytes()
    with pytest.raises(SystemExit) as error: tool.main()
    assert error.value.code == 2 and output.read_bytes() == saved
    assert all(path.read_bytes() == value for path, value in originals.items())
