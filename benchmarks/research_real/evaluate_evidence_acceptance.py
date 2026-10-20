"""Fail-closed post-run descriptive gates and evidence-only adjudication merge.

Standard library only; does not import runtime code, read fixtures/answers,
contact reviewers/providers, or rewrite any input. Inputs are the finished
question-balanced summary, its analyzer metrics/mapping/blind packets, frozen
protocol, and (where needed) blind evidence-only adjudication packets/scores.
Every planned summary row stays in intention-to-treat (ITT), including fallback
and missing generations. Missing/partial reviews or unresolved conflicts block
acceptance; a fact marked partial is valid data but never counts as full.

This is not deployment approval, a statistical test, or human evaluation.
Protocol deviations are reported separately and do not change quality gates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

ARMS = ("off", "on")
SPLITS = ("development_regression", "new_document_holdout")
REVIEWERS = ("reviewer_a", "reviewer_b")
WEIGHTS = {"factual_accuracy": 25, "evidence_support": 25,
           "key_fact_completeness": 20, "scope_and_uncertainty": 15, "task_usefulness": 15}
FACT_STATES = {"full", "partial", "absent"}
ANSWER_STATES = {"covered", "partial", "absent"}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(value):
    # Matches prepare_evidence_adjudication.py's source-pack identity.
    return hashlib.sha256(encoded(value)).hexdigest()


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def mean(values):
    return statistics.mean(values) if values else None


def indexed(rows, field, name):
    if not isinstance(rows, list) or any(not isinstance(row, dict) or not isinstance(row.get(field), str) for row in rows):
        raise ValueError("invalid_" + name)
    result = {row[field]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate_" + name)
    return result


def valid_labels(value, labels):
    return (isinstance(value, list) and all(isinstance(label, str) and label in labels for label in value)
            and len(set(value)) == len(value))


def fact_map(rows, expected, statuses):
    if not isinstance(rows, list):
        return None
    if any(not isinstance(row, dict) or row.get("fact_id") not in expected or row.get("status") not in statuses for row in rows):
        return None
    found = {row["fact_id"]: row["status"] for row in rows}
    return found if len(found) == len(rows) and set(found) == expected else None


def validate_review(review, facts, labels):
    if not isinstance(review, dict):
        return False
    dims = review.get("dimensions")
    if (not isinstance(dims, dict) or set(dims) != set(WEIGHTS)
            or any(type(v) is not int or not 0 <= v <= 4 for v in dims.values())):
        return False
    expected = sum(WEIGHTS[k] * dims[k] / 4 for k in WEIGHTS)
    critical = review.get("critical_errors")
    return (numeric(review.get("weighted_score")) and review["weighted_score"] == expected
            and isinstance(critical, list)
            and (not critical or (dims["factual_accuracy"] == 0 and dims["scope_and_uncertainty"] == 0))
            and isinstance(review.get("retrieved_facts"), dict) and set(review["retrieved_facts"]) == facts
            and set(review["retrieved_facts"].values()) <= FACT_STATES
            and isinstance(review.get("answer_facts"), dict) and set(review["answer_facts"]) == facts
            and set(review["answer_facts"].values()) <= ANSWER_STATES
            and type(review.get("source_label_count")) is int and review["source_label_count"] == len(labels)
            and valid_labels(review.get("irrelevant_source_labels"), labels))


def normalize_metric_review(review, facts):
    audit = review.get("retrieval_audit") or {}
    return {"weighted_score": review.get("weighted_score"), "dimensions": review.get("dimensions"),
            "critical_errors": review.get("critical_errors"),
            "retrieved_facts": fact_map(audit.get("retrieved_evidence_checks"), facts, FACT_STATES),
            "answer_facts": fact_map(review.get("key_fact_checks"), facts, ANSWER_STATES),
            "source_label_count": audit.get("source_label_count"),
            "irrelevant_source_labels": audit.get("irrelevant_source_labels")}


def fact_stats(rows, arm, field="adjudicated_facts", positive="full"):
    audits = [row[arm].get(field) for row in rows]
    states = ANSWER_STATES if positive == "covered" else FACT_STATES
    complete = [facts for facts in audits if isinstance(facts, dict) and facts
                and all(isinstance(value, str) and value in states for value in facts.values())]
    counts = Counter(status for facts in complete for status in facts.values())
    denominator = sum(len(facts) for facts in complete)
    fully_measured = bool(rows) and len(complete) == len(rows)
    return {"planned_question_count": len(rows), "complete_question_count": len(complete),
            "unavailable_question_count": len(rows) - len(complete), "status_counts": dict(counts),
            "fact_denominator": denominator, "full_fact_count": counts[positive] if fully_measured else None,
            "macro_full_rate": mean([sum(v == positive for v in facts.values()) / len(facts) for facts in complete]) if fully_measured else None,
            "micro_full_rate": counts[positive] / denominator if fully_measured and denominator else None,
            "all_facts_full_questions": sum(all(v == positive for v in facts.values()) for facts in complete) if fully_measured else None,
            "partial_counts_as_full": False}


def noise_stats(rows, arm):
    audits = [row[arm] for row in rows]
    complete = [row for row in audits if isinstance(row.get("irrelevant_source_labels"), list)
                and all(isinstance(value, str) for value in row["irrelevant_source_labels"])
                and len(set(row["irrelevant_source_labels"])) == len(row["irrelevant_source_labels"])
                and type(row.get("source_label_count")) is int
                and row["source_label_count"] >= len(row["irrelevant_source_labels"])]
    numerator = sum(len(row["irrelevant_source_labels"]) for row in complete)
    denominator = sum(row["source_label_count"] for row in complete)
    ready = bool(rows) and len(complete) == len(rows) and all(row["source_label_count"] > 0 for row in complete)
    return {"planned_question_count": len(rows), "complete_question_count": len(complete),
            "irrelevant_label_count": numerator, "source_label_denominator": denominator,
            "zero_source_questions": sum(row["source_label_count"] == 0 for row in complete),
            "macro_rate": mean([len(row["irrelevant_source_labels"]) / row["source_label_count"] for row in complete]) if ready else None,
            "micro_rate": numerator / denominator if ready and denominator else None,
            "definition": "Whole irrelevant source labels, not sentence-level precision or byte noise; empty packs are unavailable."}


def number_stats(values):
    measured = [value for value in values if numeric(value) and value >= 0]
    complete = bool(values) and len(measured) == len(values)
    return {"expected_count": len(values), "measured_count": len(measured),
            "unavailable_count": len(values) - len(measured), "known_subtotal": sum(measured),
            "total": sum(measured) if complete else None, "mean": mean(measured) if complete else None,
            "median": statistics.median(measured) if complete else None}


def aggregate(rows):
    ready_quality = [row for row in rows if row.get("quality") is not None]
    complete = bool(rows) and len(ready_quality) == len(rows)
    deltas = [row["quality"]["delta"] for row in ready_quality]
    quality = {"planned_question_count": len(rows), "scored_question_count": len(ready_quality),
               "off_mean": mean([row["quality"]["off"] for row in ready_quality]) if complete else None,
               "on_mean": mean([row["quality"]["on"] for row in ready_quality]) if complete else None,
               "paired_delta_mean": mean(deltas) if complete else None,
               "paired_delta_median": statistics.median(deltas) if complete else None,
               "wins_on": sum(v > 0 for v in deltas), "wins_off": sum(v < 0 for v in deltas),
               "ties": sum(v == 0 for v in deltas),
               "regression_question_ids": [row["question_id"] for row in ready_quality if row["quality"]["delta"] < 0]}
    quality["dimension_means_0_to_4"] = {arm: {key: mean([
        mean([row["original_reviews"][reviewer][arm]["dimensions"][key] for reviewer in REVIEWERS])
        for row in ready_quality]) if complete else None for key in WEIGHTS} for arm in ARMS}
    result = {"question_count": len(rows), "quality_intention_to_treat": quality,
              "critical_errors": [item for row in rows for item in row["critical_errors"]],
              "strict_evidence": {arm: fact_stats(rows, arm) for arm in ARMS},
              "source_noise": {arm: noise_stats(rows, arm) for arm in ARMS},
              "runtime_exposure": {"planned_on_count": len(rows),
                  "active_selection_count": sum(row["active_selection"] is True for row in rows),
                  "nonactive_or_fallback_question_ids": [row["question_id"] for row in rows if row["active_selection"] is not True],
                  "itt_rows_removed": 0},
              "original_review_statistics": {}}
    for reviewer in REVIEWERS:
        reviewed = [{arm: {"answer_facts": row["original_reviews"].get(reviewer, {}).get(arm, {}).get("answer_facts"),
                           "retrieved_facts": row["original_reviews"].get(reviewer, {}).get(arm, {}).get("retrieved_facts"),
                           "irrelevant_source_labels": row["original_reviews"].get(reviewer, {}).get(arm, {}).get("irrelevant_source_labels"),
                           "source_label_count": row["original_reviews"].get(reviewer, {}).get(arm, {}).get("source_label_count")}
                     for arm in ARMS} for row in rows]
        result["original_review_statistics"][reviewer] = {arm: {
            "answer_facts": fact_stats(reviewed, arm, "answer_facts", "covered"),
            "retrieved_facts": fact_stats(reviewed, arm, "retrieved_facts"),
            "source_noise": noise_stats(reviewed, arm)} for arm in ARMS}
    for category in ("usage", "latency_ms", "setup_latency_ms"):
        names = sorted({key for row in rows for key in (row.get("pair") or {}).get(category, {})})
        if category == "usage":
            names = sorted(set(names) | {"prompt_tokens", "total_tokens"})
        result[category] = {key: {arm: number_stats([(row.get("pair") or {}).get(category, {}).get(key, {}).get(arm)
                                                    for row in rows]) for arm in ARMS} for key in names}
    result["individual_prompt_or_total_token_increases"] = []
    for row in rows:
        for field in ("prompt_tokens", "total_tokens"):
            value = (row.get("pair") or {}).get("usage", {}).get(field, {})
            if all(numeric(value.get(arm)) for arm in ARMS) and value["on"] > value["off"]:
                result["individual_prompt_or_total_token_increases"].append({"question_id": row["question_id"], "field": field,
                    "off": value["off"], "on": value["on"], "delta": value["on"] - value["off"]})
    return result


def provider_gate(audit, metrics, mapping_rows, questions, protocol):
    errors = []
    if not isinstance(audit, dict) or audit.get("schema") != "evidence-answer-pipeline-provider-audit-v1" or audit.get("run_finished") is not True:
        return {"status": "unavailable", "issues": ["missing_or_unfinished_provider_audit"]}
    calls = audit.get("main_calls")
    diagnostic = audit.get("diagnostic_calls")
    if not isinstance(calls, list) or not isinstance(diagnostic, list):
        return {"status": "unavailable", "issues": ["missing_call_rows"]}
    expected = {(qid, arm) for qid in questions for arm in ARMS}
    identities = [(row.get("question_id"), row.get("arm")) for row in calls if isinstance(row, dict)]
    if len(identities) != len(calls) or len(set(identities)) != len(identities) or set(identities) != expected:
        errors.append("incomplete_duplicate_or_extra_main_call")
    all_calls = calls + diagnostic
    identifiers = [row.get("call_id") for row in all_calls if isinstance(row, dict)]
    if any(not isinstance(key, str) or not key for key in identifiers) or len(set(identifiers)) != len(all_calls):
        errors.append("invalid_duplicate_call_identity")
    model = protocol["model_and_budget"]["actual_returned_model_required"]
    endpoint = protocol["model_and_budget"]["endpoint"]
    for row in all_calls:
        if (not isinstance(row, dict) or type(row.get("repetition")) is not int or row["repetition"] != 1
                or type(row.get("adapter_calls")) is not int or row["adapter_calls"] != 1
                or type(row.get("auxiliary_provider_calls")) is not int or row["auxiliary_provider_calls"] != 0
                or row.get("actual_model") != model or row.get("endpoint_base") != endpoint
                or row.get("http_status") != 200 or row.get("finish_reason") != "stop"):
            errors.append("unknown_added_failed_or_mismatched_provider_call")
    for row in calls:
        if not isinstance(row, dict):
            continue
        mapped = mapping_rows.get(row.get("question_id"), {}).get("candidates", {})
        call_ids = {entry.get("call_id") for entry in mapped.values() if entry.get("arm") == row.get("arm")}
        if call_ids != {row.get("call_id")}:
            errors.append("call_mapping_mismatch")
    ledger = metrics.get("all_provider_consumption") or {}
    for key in ("dispatched_call_count", "provider_receipt_count"):
        if type(audit.get(key)) is not int or audit[key] != len(all_calls) or ledger.get(key) != audit[key]:
            errors.append("incomplete_or_extra_" + key)
    expected_diagnostic = {(row.get("question_id"), arm) for row in metrics.get("diagnostics", [])
                           for arm in ARMS if isinstance(row.get(arm), dict)}
    if {(row.get("question_id"), row.get("arm")) for row in diagnostic} != expected_diagnostic or len(diagnostic) != len(expected_diagnostic):
        errors.append("diagnostic_call_identity_mismatch")
    if (type(audit.get("indeterminate_call_count")) is not int or audit["indeterminate_call_count"] != 0
            or type(ledger.get("indeterminate_call_count")) is not int or ledger["indeterminate_call_count"] != 0):
        errors.append("indeterminate_calls")
    if ledger.get("non_ok_calls") != [] or ledger.get("journal_errors") != []:
        errors.append("provider_ledger_errors_or_missing_audit")
    unknown = any(not isinstance(row, dict) or any(row.get(key) is None for key in (
        "question_id", "arm", "call_id", "repetition", "adapter_calls", "auxiliary_provider_calls",
        "endpoint_base", "actual_model", "http_status", "finish_reason")) for row in all_calls)
    unknown |= any(audit.get(key) is None or ledger.get(key) is None for key in (
        "dispatched_call_count", "provider_receipt_count", "indeterminate_call_count"))
    unknown |= any(ledger.get(key) is None for key in ("journal_errors", "non_ok_calls"))
    return {"status": "unavailable" if unknown else "fail" if errors else "pass", "issues": sorted(set(errors)),
            "main_call_count": len(calls), "diagnostic_call_count": len(diagnostic),
            "dispatched_call_count": audit.get("dispatched_call_count"),
            "provider_receipt_count": audit.get("provider_receipt_count"),
            "expected_main_call_count": len(expected), "audit": deepcopy(audit),
            "scope": "Recorded calls and explicit auxiliary-call audit, not a claim about unobserved external activity."}


def build(summary, metrics, mapping, blind_packets, adjudication_packets, adjudication_scores, protocol,
          *, provider_audit=None, protocol_deviations=(), arbiter_id="blind_evidence_arbiter"):
    if not isinstance(arbiter_id, str) or not arbiter_id.strip() or arbiter_id in REVIEWERS:
        raise ValueError("independent_arbiter_identity_required")
    if (summary.get("schema") != "evidence-trace-question-balanced-review-summary-v1"
            or summary.get("run_finished_confirmed_by_caller") is not True):
        raise ValueError("finished_summary_required")
    if summary.get("frozen_acceptance") != protocol.get("frozen_descriptive_acceptance"):
        raise ValueError("frozen_threshold_mismatch")
    planned = indexed(summary.get("per_question"), "question_id", "summary_question")
    if len(planned) != summary.get("planned_question_count") or not planned:
        raise ValueError("invalid_planned_question_count")
    issues = []
    def issue(code, **context):
        issues.append({"code": code, **context})
    if summary.get("reviewer_ids") != list(REVIEWERS):
        issue("expected_reviewer_a_and_reviewer_b")
    for key in ("validation_issues", "upstream_artifact_errors", "upstream_score_errors",
                "upstream_partial_reviews", "excluded_or_incomplete_generations"):
        if summary.get(key) != []:
            issue("summary_" + key, details=deepcopy(summary.get(key)))
    for key, value in (("artifact_errors", metrics.get("artifact_errors")),
                       ("score_errors", metrics.get("quality", {}).get("score_errors")),
                       ("partial_reviews", metrics.get("quality", {}).get("partial_reviews")),
                       ("excluded", metrics.get("excluded"))):
        if value != []:
            issue("metrics_" + key, details=deepcopy(value))
    pairs = indexed(metrics.get("pairs"), "question_id", "metric_pair")
    packets = indexed(blind_packets, "packet_id", "blind_packet")
    adj_packets = indexed(adjudication_packets, "packet_id", "adjudication_packet")
    adj_scores = indexed(adjudication_scores, "packet_id", "adjudication_score")
    mapping_rows = {}
    for packet_id, row in mapping.get("packets", {}).items():
        if row.get("packet_id") != packet_id or row.get("repetition") != 1 or row.get("question_id") in mapping_rows:
            raise ValueError("invalid_mapping")
        mapping_rows[row["question_id"]] = row
    for name, identities in (("pairs", pairs), ("mapping", mapping_rows)):
        if set(identities) - set(planned):
            raise ValueError("unknown_" + name)
    if set(packets) - {row["packet_id"] for row in mapping_rows.values()} or set(adj_packets) - set(packets) or set(adj_scores) - set(adj_packets):
        raise ValueError("unknown_packet_or_adjudication")
    metric_reviews = {}
    for review in metrics.get("quality", {}).get("paired_reviews", []):
        key = review.get("question_id"), review.get("reviewer_id")
        if key in metric_reviews or key[0] not in planned or key[1] not in REVIEWERS or review.get("repetition") != 1:
            raise ValueError("invalid_metric_review_identity")
        metric_reviews[key] = review
    results, adjudicated_decisions = [], []
    for qid, original in planned.items():
        split = original.get("eval_split")
        language = original.get("language", "unspecified")
        if split not in SPLITS or not isinstance(language, str) or not language:
            raise ValueError("invalid_split_or_language")
        pair = pairs.get(qid)
        if pair is None or pair != original.get("pair") or pair.get("repetition") != 1:
            issue("missing_or_mismatched_pair", question_id=qid)
        packet_map = mapping_rows.get(qid, {})
        packet = packets.get(packet_map.get("packet_id"))
        row = {"question_id": qid, "eval_split": split, "language": language, "pair": deepcopy(pair),
               "quality": None, "original_reviews": deepcopy(original.get("reviews", {})),
               "active_selection": original.get("active_selection"), "critical_errors": [],
               **{arm: {"adjudicated_facts": None, "irrelevant_source_labels": None, "source_label_count": None} for arm in ARMS}}
        results.append(row)
        if packet is None:
            issue("missing_blind_packet", question_id=qid)
            continue
        facts = {fact["id"] for fact in packet["required_facts"]}
        if not facts or len(facts) != len(packet["required_facts"]) or len(facts) != original.get("fact_count"):
            raise ValueError("fact_identity_mismatch")
        candidates = indexed(packet["candidates"], "candidate_id", "candidate")
        mapped = packet_map.get("candidates", {})
        if set(candidates) != set(mapped) or len(mapped) != 2 or {entry.get("arm") for entry in mapped.values()} != set(ARMS):
            raise ValueError("candidate_mapping_mismatch")
        valid = set(row["original_reviews"]) == set(REVIEWERS)
        if not valid:
            issue("incorrect_question_reviewer_set", question_id=qid)
        groups = {}
        for candidate_id, candidate in candidates.items():
            arm = mapped[candidate_id]["arm"]
            sources = candidate["actually_supplied_sources"]
            labels = [source["label"] for source in sources]
            if any(not isinstance(label, str) for label in labels) or len(set(labels)) != len(labels):
                raise ValueError("invalid_source_labels")
            group = groups.setdefault(digest(sources), {"sources": sources, "labels": set(labels), "candidates": [], "arms": [], "reviews": []})
            group["candidates"].append(candidate_id); group["arms"].append(arm)
            row[arm]["source_label_count"] = len(labels)
            for reviewer in REVIEWERS:
                reduced = row["original_reviews"].get(reviewer, {}).get(arm)
                raw = metric_reviews.get((qid, reviewer), {}).get(arm + "_review", {})
                good = (validate_review(reduced, facts, set(labels))
                        and normalize_metric_review(raw, facts) == reduced
                        and raw.get("packet_id") == packet["packet_id"]
                        and raw.get("call_id") == mapped[candidate_id].get("call_id")
                        and raw.get("reviewer_kind") == "agent_blind_review")
                if not good:
                    valid = False
                    issue("missing_invalid_or_changed_review", question_id=qid, reviewer_id=reviewer, arm=arm)
                else:
                    group["reviews"].append(reduced)
                    if reduced["critical_errors"]:
                        row["critical_errors"].append({"question_id": qid, "reviewer_id": reviewer, "arm": arm,
                                                       "errors": deepcopy(reduced["critical_errors"])})
        if not valid:
            continue
        quality = {arm: mean([row["original_reviews"][reviewer][arm]["weighted_score"] for reviewer in REVIEWERS]) for arm in ARMS}
        quality["delta"] = quality["on"] - quality["off"]
        if quality != original.get("quality"):
            issue("changed_or_missing_quality_summary", question_id=qid)
        elif pair is not None:
            row["quality"] = quality
        packet_id = packet["packet_id"]
        task_packet = adj_packets.get(packet_id)
        if task_packet and (task_packet.get("schema") != "blind-evidence-conflict-packet-v1"
                or any(task_packet.get(key) != packet.get(key) for key in (
                    "question", "required_facts", "original_pdf_reference_pages"))):
            raise ValueError("adjudication_question_or_fact_context_changed")
        task_map = indexed(task_packet.get("evidence_packs", []), "evidence_id", "adjudication_task") if task_packet else {}
        score_packet = adj_scores.get(packet_id)
        score_map = indexed(score_packet.get("evidence_packs", []), "evidence_id", "adjudication_decision") if score_packet else {}
        if score_packet and score_packet.get("reviewer_id") != arbiter_id:
            raise ValueError("invalid_evidence_arbiter_identity")
        expected_tasks = set()
        for pack_hash, group in groups.items():
            evidence_id = "evidence-" + pack_hash[:16]
            agreed = {fact: {review["retrieved_facts"][fact] for review in group["reviews"]} for fact in facts}
            disputed = sorted(fact for fact in facts if len(agreed[fact]) != 1)
            noises = {tuple(sorted(review["irrelevant_source_labels"])) for review in group["reviews"]}
            noise_conflict = len(noises) != 1
            final_facts = {fact: next(iter(values)) if len(values) == 1 else None for fact, values in agreed.items()}
            final_noise = list(next(iter(noises))) if not noise_conflict else None
            if disputed or noise_conflict:
                expected_tasks.add(evidence_id)
                task, decision = task_map.get(evidence_id), score_map.get(evidence_id)
                if (not task or set(task.get("candidate_ids", [])) != set(group["candidates"])
                        or len(task.get("candidate_ids", [])) != len(group["candidates"])
                        or task.get("actually_supplied_sources") != group["sources"]
                        or task.get("fact_ids_to_check") != disputed or task.get("check_irrelevance") is not noise_conflict):
                    issue("missing_or_mismatched_adjudication_task", question_id=qid, evidence_id=evidence_id)
                elif decision is None:
                    issue("missing_evidence_adjudication", question_id=qid, evidence_id=evidence_id)
                else:
                    checks = decision.get("fact_checks")
                    values = fact_map(checks, set(disputed), FACT_STATES)
                    good = values is not None
                    for check in checks if good else []:
                        good &= (valid_labels(check.get("source_labels"), group["labels"])
                                 and isinstance(check.get("reason"), str) and bool(check["reason"].strip())
                                 and (bool(check["source_labels"]) if check.get("status") in {"full", "partial"} else check["source_labels"] == []))
                    if noise_conflict:
                        good &= (valid_labels(decision.get("irrelevant_source_labels"), group["labels"])
                                 and isinstance(decision.get("irrelevance_reason"), str) and bool(decision["irrelevance_reason"].strip()))
                    else:
                        good &= decision.get("irrelevant_source_labels") is None and decision.get("irrelevance_reason") is None
                    if not good:
                        issue("invalid_or_partial_adjudication", question_id=qid, evidence_id=evidence_id)
                    else:
                        final_facts.update(values)
                        if noise_conflict:
                            final_noise = list(decision["irrelevant_source_labels"])
                        adjudicated_decisions.append({"question_id": qid, "packet_id": packet_id,
                            "evidence_id": evidence_id, "arms": group["arms"], "candidate_ids": group["candidates"],
                            "source_pack_sha256": pack_hash, "decision": deepcopy(decision)})
            for arm in group["arms"]:
                row[arm].update(adjudicated_facts=deepcopy(final_facts), irrelevant_source_labels=deepcopy(final_noise),
                                source_pack_sha256=pack_hash)
        if set(task_map) - expected_tasks or set(score_map) - expected_tasks:
            raise ValueError("adjudication_cannot_rewrite_agreed_facts_or_unknown_packs")
    splits = {split: aggregate([row for row in results if row["eval_split"] == split]) for split in SPLITS}
    provider = provider_gate(provider_audit, metrics, mapping_rows, planned, protocol)
    gates = {}
    def comparison(left, right, direction):
        if not numeric(left) or not numeric(right):
            return "unavailable"
        return "pass" if (left >= right if direction == "at_least" else left <= right) else "fail"
    for split, values in splits.items():
        off, on = values["strict_evidence"]["off"], values["strict_evidence"]["on"]
        noise = values["source_noise"]
        delta = values["quality_intention_to_treat"]["paired_delta_mean"]
        gate = {"quality_mean_nonnegative": comparison(delta, 0, "at_least"),
                "no_on_critical_errors": ("unavailable" if delta is None else
                    "fail" if any(row["arm"] == "on" for row in values["critical_errors"]) else "pass"),
                "strict_full_macro_no_decrease": comparison(on["macro_full_rate"], off["macro_full_rate"], "at_least"),
                "all_facts_full_questions_no_decrease": comparison(on["all_facts_full_questions"], off["all_facts_full_questions"], "at_least"),
                "source_noise_macro_no_increase": comparison(noise["on"]["macro_rate"], noise["off"]["macro_rate"], "at_most"),
                "no_added_provider_calls": provider["status"]}
        for token in ("prompt_tokens", "total_tokens"):
            gate[token + "_mean_no_increase"] = comparison(values["usage"][token]["on"]["mean"], values["usage"][token]["off"]["mean"], "at_most")
        if split == "new_document_holdout":
            # Frozen clarification before looking at new scores: positive gain
            # is the count of FULL required facts on the same fixed denominator.
            # Macro recall must separately not decrease, not strictly increase.
            left, right = on["full_fact_count"], off["full_fact_count"]
            gate["positive_strict_full_fact_gain"] = "unavailable" if not numeric(left) or not numeric(right) else "pass" if left > right else "fail"
        values["strict_evidence_delta"] = {key: on[key] - off[key] if numeric(on[key]) and numeric(off[key]) else None
            for key in ("full_fact_count", "macro_full_rate", "micro_full_rate", "all_facts_full_questions")}
        gates[split] = gate
    unavailable = any(value == "unavailable" for gate in gates.values() for value in gate.values())
    passed = not issues and not unavailable and all(value == "pass" for gate in gates.values() for value in gate.values())
    status = "blocked_incomplete_or_invalid" if issues or unavailable else "passed" if passed else "failed"
    if not isinstance(protocol_deviations, (list, tuple)) or any(not isinstance(row, dict) for row in protocol_deviations):
        raise ValueError("invalid_protocol_deviations")
    languages = sorted({row["language"] for row in results})
    return {"schema": "evidence-trace-descriptive-acceptance-v1", "status": status,
            "descriptive_screen_passed": passed, "release_approval": False,
            "run_finished_confirmed_by_caller": True, "planned_question_count": len(results),
            "reviewer_ids": list(REVIEWERS), "validation_issues": issues,
            "expected_arbiter_id": arbiter_id,
            "gates": gates, "by_eval_split": splits, "overall": aggregate(results),
            "by_language": {language: aggregate([row for row in results if row["language"] == language]) for language in languages},
            "by_eval_split_language": {split: {language: aggregate([row for row in results if row["eval_split"] == split and row["language"] == language])
                for language in languages if any(row["eval_split"] == split and row["language"] == language for row in results)} for split in SPLITS},
            "original_language_summary": deepcopy(summary.get("by_language")),
            "original_split_language_summary": deepcopy(summary.get("by_eval_split_language")),
            "per_question": results, "adjudicated_decisions": adjudicated_decisions,
            "provider_call_gate": provider, "provider_accounting": deepcopy(metrics.get("all_provider_consumption")),
            "diagnostics": deepcopy(metrics.get("diagnostics")), "frozen_acceptance": deepcopy(summary["frozen_acceptance"]),
            "holdout_positive_gain_definition": "ON full_fact_count > OFF full_fact_count at fixed required-fact denominator; macro recall separately must not decrease.",
            "full_protocol_compliance": False if protocol_deviations else None,
            "protocol_deviations": deepcopy(list(protocol_deviations)),
            "protocol_compliance_scope": "False for declared deviations; null without a separate complete protocol audit. Deviations never alter quality thresholds.",
            "limitations": ["Descriptive bounded-run screen, not statistical significance, semantic truth, or release approval.",
                "All planned questions stay in ITT; active selection is exposure information, not a filter.",
                "Original answer-quality scores and critical errors are unchanged. Only disputed evidence labels are adjudicated.",
                "Full requires all material components; partial never counts as full. Missing data stays unavailable.",
                "Source-label noise is not sentence-level precision or byte noise; cached tokens are already part of prompt tokens.",
                "Language uses frozen summary metadata, never guessed from model answers."]}


def load(path):
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def load_directory(path):
    values, receipts = [], []
    if path is not None:
        for file in sorted(path.rglob("*.json")):
            value, receipt = load(file)
            values.append(value); receipts.append(receipt)
    return values, receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("summary", "metrics", "mapping", "blind", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("adjudication-packets", "adjudication-scores", "provider-audit", "protocol-deviations"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--run-finished", action="store_true", required=True)
    parser.add_argument("--arbiter-id", default="blind_evidence_arbiter",
                        help="Declared independent arbiter identity; must match the original review files.")
    args = parser.parse_args()
    try:
        inputs, receipts = {}, {}
        for name in ("summary", "metrics", "mapping", "protocol", "provider_audit", "protocol_deviations"):
            path = getattr(args, name)
            if path is not None:
                inputs[name], receipts[name] = load(path)
        for name in ("blind", "adjudication_packets", "adjudication_scores"):
            inputs[name], receipts[name] = load_directory(getattr(args, name))
        if inputs["summary"].get("inputs", {}).get("metrics", {}).get("sha256") != receipts["metrics"]["sha256"]:
            raise ValueError("summary_metrics_file_changed")
        deviations = inputs.get("protocol_deviations", [])
        if isinstance(deviations, dict):
            deviations = deviations.get("deviations")
        result = build(inputs["summary"], inputs["metrics"], inputs["mapping"], inputs["blind"],
                       inputs["adjudication_packets"], inputs["adjudication_scores"], inputs["protocol"],
                       provider_audit=inputs.get("provider_audit"), protocol_deviations=deviations,
                       arbiter_id=args.arbiter_id)
        result["inputs"] = receipts
        with os.fdopen(os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
            stream.write(encoded(result))
        print(json.dumps({"status": result["status"], "descriptive_screen_passed": result["descriptive_screen_passed"],
                          "validation_issue_count": len(result["validation_issues"]), "output": str(args.output.resolve())}))
    except (ValueError, TypeError, KeyError, OSError):
        parser.exit(2, "Cannot evaluate acceptance: validate schemas/identities and use a new output path. Inputs and scores were not changed.\n")


if __name__ == "__main__":
    main()
