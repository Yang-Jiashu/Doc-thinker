"""Post-run, standard-library-only question-balanced review accounting.

This helper does not import production code, load models, call providers, read
credentials, contact reviewers, or rewrite their scores. Run only after the
parent confirms that the answer run has finished and the analyzer has collected
the two independent reviews. Its output is an audit input, NOT release approval.
Original annotations remain in the SHA-identified analyzer metrics file.

Example (only after the run-finished signal):
  python summarize_evidence_trace_reviews.py --run-finished \
    --fixture research_evidence_trace_ab_v1.json \
    --protocol research_evidence_trace_ab_v1.protocol.json \
    --metrics /path/to/final-analysis/metrics.json \
    --output /path/to/new/question-balanced-summary.json

The output must not exist. Missing questions/reviews remain missing, not zeros.
Scheduler timeouts and other ON fallbacks remain in intention-to-treat results.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

ARMS = ("off", "on")
SPLITS = ("development_regression", "new_document_holdout")


def load(path):
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def mean(values):
    return statistics.mean(values) if values else None


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def numbers(values):
    measured = [value for value in values if numeric(value)]
    return {"expected_count": len(values), "measured_count": len(measured),
            "unavailable_count": len(values) - len(measured),
            "known_subtotal": sum(measured),
            "total": sum(measured) if len(measured) == len(values) else None,
            "mean": mean(measured), "median": statistics.median(measured) if measured else None}


def checked_facts(checks, expected, statuses):
    found = {}
    if isinstance(checks, list):
        for check in checks:
            if (not isinstance(check, dict) or check.get("fact_id") not in expected
                    or check.get("status") not in statuses or check["fact_id"] in found):
                return None
            found[check["fact_id"]] = check["status"]
    return found if set(found) == set(expected) else None


def reviewer_arm(review, facts, weights):
    dimensions = review.get("dimensions", {})
    valid = (set(dimensions) == set(weights)
             and all(type(value) is int and 0 <= value <= 4 for value in dimensions.values()))
    score = sum(weights[name] * dimensions[name] / 4 for name in weights) if valid else None
    if score != review.get("weighted_score"):
        score = None
    critical = review.get("critical_errors", [])
    if critical and (dimensions.get("factual_accuracy") != 0 or dimensions.get("scope_and_uncertainty") != 0):
        score = None
    audit = review.get("retrieval_audit") or {}
    evidence = checked_facts(audit.get("retrieved_evidence_checks"), facts, {"full", "partial", "absent"})
    answer = checked_facts(review.get("key_fact_checks"), facts, {"covered", "partial", "absent"})
    irrelevant = audit.get("irrelevant_source_labels")
    source_count = audit.get("source_label_count")
    if (not isinstance(irrelevant, list) or any(not isinstance(label, str) for label in irrelevant)
            or len(set(irrelevant)) != len(irrelevant) or type(source_count) is not int
            or source_count < len(irrelevant)):
        irrelevant = None
    return {"weighted_score": score, "dimensions": dimensions if valid else None,
            "critical_errors": critical, "retrieved_facts": evidence, "answer_facts": answer,
            "source_label_count": source_count, "irrelevant_source_labels": irrelevant}


def consensus(reviews, arm, field, positive, expected_reviewers):
    values = {reviewer: row[arm][field] for reviewer, row in reviews.items()}
    if len(values) != expected_reviewers or any(value is None for value in values.values()):
        return {"available": False, "positive_count": None, "fact_count": None,
                "macro_rate": None, "all_positive": None, "disagreements": []}
    fact_ids = sorted(next(iter(values.values())))
    disagreements, agreement = [], Counter()
    for fact_id in fact_ids:
        statuses = {reviewer: row[fact_id] for reviewer, row in values.items()}
        if len(set(statuses.values())) == 1:
            agreement[next(iter(statuses.values()))] += 1
        else:
            disagreements.append({"fact_id": fact_id, "statuses": statuses})
    positive_count = agreement[positive]
    return {"available": True, "positive_count": positive_count, "fact_count": len(fact_ids),
            "macro_rate": positive_count / len(fact_ids) if fact_ids else None,
            "all_positive": positive_count == len(fact_ids) if fact_ids else None,
            "agreed_status_counts": dict(agreement), "disagreement_count": len(disagreements),
            "disagreements": disagreements,
            "definition": "Positive only if both reviewers agree; disagreement is not recoded as absent. Not adjudication."}


def summarize_reviewer(rows, reviewer, arm, field, positive):
    audits = [row["reviews"].get(reviewer, {}).get(arm, {}) for row in rows]
    complete = [audit[field] for audit in audits if audit.get(field) is not None]
    counts = Counter(status for facts in complete for status in facts.values())
    denominator = sum(len(facts) for facts in complete)
    return {"planned_questions": len(rows), "complete_question_audits": len(complete),
            "unavailable_question_audits": len(rows) - len(complete), "status_counts": dict(counts),
            "fact_denominator": denominator, "micro_positive_rate": counts[positive] / denominator if denominator else None,
            "macro_positive_rate": mean([sum(status == positive for status in facts.values()) / len(facts)
                                         for facts in complete if facts]),
            "all_facts_positive_questions": sum(bool(facts) and all(status == positive for status in facts.values()) for facts in complete)}


def summarize_noise(rows, reviewer, arm):
    audits = [row["reviews"].get(reviewer, {}).get(arm, {}) for row in rows]
    complete = [audit for audit in audits if audit.get("irrelevant_source_labels") is not None]
    numerator = sum(len(audit["irrelevant_source_labels"]) for audit in complete)
    denominator = sum(audit["source_label_count"] for audit in complete)
    return {"planned_questions": len(rows), "complete_question_audits": len(complete),
            "irrelevant_label_count": numerator, "source_label_denominator": denominator,
            "micro_rate": numerator / denominator if denominator else None,
            "macro_rate": mean([len(audit["irrelevant_source_labels"]) / audit["source_label_count"]
                                for audit in complete if audit["source_label_count"]]),
            "zero_source_questions": sum(audit["source_label_count"] == 0 for audit in complete),
            "definition": "Wholly irrelevant labels only; not sentence-level precision or byte noise. Empty-pack rates remain unavailable."}


def quality_summary(rows):
    scored = [row for row in rows if row["quality"] is not None]
    deltas = [row["quality"]["delta"] for row in scored]
    dimensions = sorted({name for row in scored for review in row["reviews"].values()
                         for name in review["off"]["dimensions"]})
    return {"planned_question_count": len(rows), "scored_question_count": len(scored),
            "unscored_question_count": len(rows) - len(scored),
            "off_mean": mean([row["quality"]["off"] for row in scored]),
            "on_mean": mean([row["quality"]["on"] for row in scored]),
            "paired_delta_mean": mean(deltas), "paired_delta_median": statistics.median(deltas) if deltas else None,
            "wins_on": sum(value > 0 for value in deltas), "wins_off": sum(value < 0 for value in deltas),
            "ties": sum(value == 0 for value in deltas),
            "regression_question_ids": [row["question_id"] for row in scored if row["quality"]["delta"] < 0],
            "dimension_means_0_to_4": {arm: {name: mean([
                mean([review[arm]["dimensions"][name] for review in row["reviews"].values()])
                for row in scored]) for name in dimensions} for arm in ARMS},
            "definition": "Average the two independent weighted scores per arm per question, then weight questions equally. Wins are question-level, not reviewer-pair counts."}


def summarize(rows, reviewers):
    measured = [row for row in rows if row["pair"] is not None]
    complete_helper = [row for row in measured if row["helper_completed"]]
    active = [row for row in measured if row["active_selection"]]
    actual_change = [row for row in active if row["pair"].get("source_content_identical") is False]
    result = {"quality_intention_to_treat": quality_summary(rows),
              "runtime_exposure": {
                  "planned_on_count": len(rows), "measured_on_count": len(measured),
                  "scheduler_status_counts": dict(Counter(row["pair"].get("on_assistance", {}).get("scheduler_status") or "missing" for row in measured)),
                  "scheduler_completed_count": len(complete_helper), "active_selection_count": len(active),
                  "active_selection_changed_source_count": len(actual_change),
                  "nonactive_or_fallback_question_ids": [row["question_id"] for row in measured if not row["active_selection"]],
                  "identical_source_pack_count": sum(row["pair"].get("source_content_identical") is True for row in measured),
                  "identical_prompt_count": sum(row["pair"].get("exposure") == "identical_prompt" for row in measured),
                  "definition": "Active selection requires scheduler completed, evidence_selected, and no skipped_reason. Source change is counted separately. Fallbacks stay in ITT; same-source answer differences are not evidence of retrieval improvement."},
              "quality_active_selection_sensitivity_only": quality_summary(active),
              "quality_changed_source_sensitivity_only": quality_summary(actual_change),
              "by_reviewer": {reviewer: {arm: {
                  "retrieved_evidence": summarize_reviewer(rows, reviewer, arm, "retrieved_facts", "full"),
                  "answer_facts": summarize_reviewer(rows, reviewer, arm, "answer_facts", "covered"),
                  "irrelevant_sources": summarize_noise(rows, reviewer, arm),
              } for arm in ARMS} for reviewer in reviewers},
              "consensus_full_evidence": {}}
    for arm in ARMS:
        audits = [row["consensus_evidence"][arm] for row in rows if row["consensus_evidence"][arm]["available"]]
        numerator = sum(audit["positive_count"] for audit in audits)
        denominator = sum(audit["fact_count"] for audit in audits)
        result["consensus_full_evidence"][arm] = {
            "complete_question_count": len(audits), "full_fact_count": numerator,
            "fact_denominator": denominator, "micro_full_rate": numerator / denominator if denominator else None,
            "macro_full_rate": mean([audit["macro_rate"] for audit in audits if audit["macro_rate"] is not None]),
            "all_facts_full_questions": sum(audit["all_positive"] is True for audit in audits),
            "disputed_fact_count": sum(audit["disagreement_count"] for audit in audits)}
    for category in ("usage", "latency_ms", "setup_latency_ms"):
        names = sorted({name for row in measured for name in row["pair"].get(category, {})})
        result[category] = {name: {arm: numbers([row["pair"].get(category, {}).get(name, {}).get(arm)
                                               if row["pair"] else None for row in rows]) for arm in ARMS} for name in names}
    result["individual_prompt_or_total_token_increases"] = [
        {"question_id": row["question_id"], "field": field, **row["pair"]["usage"][field]}
        for row in measured for field in ("prompt_tokens", "total_tokens")
        if numeric(row["pair"].get("usage", {}).get(field, {}).get("delta")) and row["pair"]["usage"][field]["delta"] > 0]
    result["critical_errors"] = [{"question_id": row["question_id"], "reviewer_id": reviewer, "arm": arm, "errors": review[arm]["critical_errors"]}
                                 for row in rows for reviewer, review in row["reviews"].items() for arm in ARMS if review[arm]["critical_errors"]]
    return result


def build(fixture, protocol, metrics):
    planned = [row for row in fixture["questions"] if not row.get("diagnostic_only") and row.get("include_in_primary_quality_average", True)]
    expected_reviewers = protocol["quality_and_retrieval_review"]["independent_reviewers_per_main_pair"]
    weights = {row["id"]: row["weight"] for row in fixture["rubric"]["dimensions"]}
    if expected_reviewers != 2 or sum(weights.values()) != 100:
        raise ValueError("unsupported_protocol")
    quality = metrics.get("quality") or {}
    issues = []
    grouped, pairs = {}, {}
    for pair in metrics.get("pairs", []):
        identity = pair["question_id"]
        if identity in pairs or pair.get("repetition") != 1:
            raise ValueError("duplicate_or_repeated_generation")
        pairs[identity] = pair
    for review in quality.get("paired_reviews", []):
        identity, reviewer = review["question_id"], review["reviewer_id"]
        if review.get("repetition") != 1 or not isinstance(reviewer, str) or not reviewer:
            raise ValueError("invalid_review_identity")
        bucket = grouped.setdefault(identity, {})
        if reviewer in bucket:
            raise ValueError("duplicate_reviewer_pair")
        bucket[reviewer] = review
    if (set(pairs) | set(grouped)) - {row["id"] for row in planned}:
        raise ValueError("unknown_main_question")
    reviewers = sorted({reviewer for bucket in grouped.values() for reviewer in bucket})
    if len(reviewers) != expected_reviewers:
        issues.append({"code": "reviewer_count_not_two", "actual": len(reviewers)})
    rows, disputes, same_pack = [], [], []
    for question in planned:
        identity, split = question["id"], question.get("eval_split")
        if split not in SPLITS:
            raise ValueError("unexpected_eval_split")
        facts = {fact["id"] for fact in question.get("key_facts", [])}
        pair = pairs.get(identity)
        reviews = {reviewer: {arm: reviewer_arm(review[arm + "_review"], facts, weights) for arm in ARMS}
                   for reviewer, review in grouped.get(identity, {}).items()}
        for reviewer, review in grouped.get(identity, {}).items():
            if any(review[arm + "_review"].get("reviewer_kind") != "agent_blind_review" for arm in ARMS):
                issues.append({"question_id": identity, "reviewer_id": reviewer, "code": "unexpected_reviewer_kind"})
        if pair is None or len(reviews) != expected_reviewers:
            issues.append({"question_id": identity, "code": "incomplete_pair_or_reviews"})
        score = None
        if pair is not None and len(reviews) == expected_reviewers and all(numeric(review[arm]["weighted_score"]) for review in reviews.values() for arm in ARMS):
            score = {arm: mean([review[arm]["weighted_score"] for review in reviews.values()]) for arm in ARMS}
            score["delta"] = score["on"] - score["off"]
        else:
            issues.append({"question_id": identity, "code": "quality_unavailable"})
        consensus_evidence = {arm: consensus(reviews, arm, "retrieved_facts", "full", expected_reviewers) for arm in ARMS}
        for arm in ARMS:
            if not consensus_evidence[arm]["available"]:
                issues.append({"question_id": identity, "arm": arm, "code": "incomplete_retrieval_audit"})
            for disagreement in consensus_evidence[arm]["disagreements"]:
                disputes.append({"question_id": identity, "arm": arm, "kind": "required_fact", **disagreement})
            noise = {reviewer: review[arm]["irrelevant_source_labels"] for reviewer, review in reviews.items()}
            if len(noise) != expected_reviewers or any(value is None for value in noise.values()):
                issues.append({"question_id": identity, "arm": arm, "code": "incomplete_irrelevance_audit"})
            elif len({tuple(sorted(value)) for value in noise.values()}) > 1:
                disputes.append({"question_id": identity, "arm": arm, "kind": "irrelevant_labels", "labels_by_reviewer": noise})
            if any(review[arm]["answer_facts"] is None for review in reviews.values()):
                issues.append({"question_id": identity, "arm": arm, "code": "incomplete_answer_fact_audit"})
        if pair and pair.get("source_content_identical") is True:
            signatures = {(reviewer, arm): {"facts": review[arm]["retrieved_facts"],
                                          "irrelevant": sorted(review[arm]["irrelevant_source_labels"])
                                          if review[arm]["irrelevant_source_labels"] is not None else None}
                          for reviewer, review in reviews.items() for arm in ARMS}
            if len({json.dumps(value, sort_keys=True) for value in signatures.values()}) > 1:
                same_pack.append({"question_id": identity, "code": "identical_sources_inconsistent_retrieval_labels",
                                  "definition": "Requires same-pack evidence-only adjudication; answer scores are not copied."})
        assistance = pair.get("on_assistance", {}) if pair else {}
        completed = assistance.get("scheduler_status") == "completed"
        active = completed and assistance.get("status") == "evidence_selected" and not assistance.get("skipped_reason")
        language = question.get("language", "unspecified")
        if not isinstance(language, str) or not language:
            raise ValueError("invalid_question_language")
        rows.append({"question_id": identity, "eval_split": split, "language": language,
                     "fact_count": len(facts), "pair": pair,
                     "reviews": reviews, "quality": score, "consensus_evidence": consensus_evidence,
                     "helper_completed": completed, "active_selection": active})
    return {"schema": "evidence-trace-question-balanced-review-summary-v1",
            "status": "audit_input_not_release_approval", "run_finished_confirmed_by_caller": True,
            "planned_question_count": len(planned), "eligible_generation_pair_count": len(pairs),
            "scored_reviewer_pair_count": sum(len(bucket) for bucket in grouped.values()),
            "reviewer_ids": reviewers, "validation_issues": issues,
            "upstream_artifact_errors": metrics.get("artifact_errors", []),
            "upstream_score_errors": quality.get("score_errors", []),
            "upstream_partial_reviews": quality.get("partial_reviews", []),
            "excluded_or_incomplete_generations": metrics.get("excluded", []),
            "by_eval_split": {split: summarize([row for row in rows if row["eval_split"] == split], reviewers) for split in SPLITS},
            "by_language": {language: summarize([row for row in rows if row["language"] == language], reviewers)
                            for language in sorted({row["language"] for row in rows})},
            "by_eval_split_language": {split: {
                language: summarize([row for row in rows if row["eval_split"] == split and row["language"] == language], reviewers)
                for language in sorted({row["language"] for row in rows if row["eval_split"] == split})}
                for split in SPLITS},
            "overall": summarize(rows, reviewers), "per_question": rows,
            "evidence_label_disagreements": disputes, "identical_pack_consistency_issues": same_pack,
            "adjudication": {"status": "required_pending" if disputes or same_pack else "no_conflicts_detected",
                             "performed_by_this_helper": False,
                             "definition": "Frozen protocol requires separate blinded evidence-only adjudication of conflicts. Preserve originals and publish adjudicated counts/reasons separately; never overwrite weighted scores."},
            "provider_accounting": metrics.get("all_provider_consumption"),
            "setup_accounting": {"corpus_ingestion": metrics.get("corpus_ingestion"),
                                 "corpus_clone_latency_ms": metrics.get("corpus_clone_latency_ms"),
                                 "definition": metrics.get("setup_cost_definition"),
                                 "actual_ingestion_cost": None,
                                 "pending_audit": "Reconcile copied-corpus provenance with newly ingested corpora. Saved earlier-run timestamps/cost are not incurred again; never sum cold-start-equivalent allocations."},
            "diagnostics": metrics.get("diagnostics"),
            "latency_definition": metrics.get("latency_definition"),
            "frozen_acceptance": protocol["frozen_descriptive_acceptance"],
            "release_gate": {"status": "not_evaluated", "requires": ["complete runtime/usage audit", "two complete independent reviews per question", "blinded evidence adjudication where needed", "split-wise frozen acceptance checks"]},
            "limitations": [f"{len(planned)} questions, not {len(planned) * expected_reviewers} independent reviewer pairs; each arm generated once.",
                            "Timeout/fallback ON answers remain in ITT; active subsets are descriptive sensitivity analyses only.",
                            "Conservative full agreement is not human/expert ground truth and is not a substitute for adjudication.",
                            "Source-label noise is not sentence/byte noise. Partial fact coverage never counts as full.",
                            "Provider cache counts are a subset of prompt counts, not additional tokens; unknown usage remains null.",
                            "This small public-paper holdout does not establish unseen-domain or pretraining-free generalization.",
                            "Language strata use frozen fixture metadata; unspecified languages are not guessed from answers. Small strata are descriptive, not new independent trials."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "protocol", "metrics", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-finished", action="store_true", required=True,
                        help="Parent confirms the answer runner has finished; not a claim that every planned call succeeded.")
    args = parser.parse_args()
    try:
        fixture, fixture_receipt = load(args.fixture)
        protocol, protocol_receipt = load(args.protocol)
        metrics, metrics_receipt = load(args.metrics)
        result = build(fixture, protocol, metrics)
        result["inputs"] = {"fixture": fixture_receipt, "protocol": protocol_receipt, "metrics": metrics_receipt}
        encoded = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
        print(json.dumps({"status": result["status"], "planned_questions": result["planned_question_count"],
                          "eligible_pairs": result["eligible_generation_pair_count"],
                          "reviewer_pairs": result["scored_reviewer_pair_count"],
                          "validation_issue_count": len(result["validation_issues"]),
                          "evidence_disagreement_count": len(result["evidence_label_disagreements"])}))
    except (ValueError, TypeError, KeyError, OSError):
        parser.exit(2, "Unable to build summary: validate input schema and use a new output path. No input or reviewer scores were rewritten.\n")


if __name__ == "__main__":
    main()
