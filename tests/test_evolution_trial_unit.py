import copy
import json
import subprocess
import sys

import pytest

from docthinker.evaluation import assess_evolution_trial, score_answer


def trial(quality=0.7, tokens=1000):
    return {
        "metadata": {
            "dataset_id": "held-out-v1",
            "evidence_snapshot": "sha256:fixture",
            "model_id": "model-v1",
            "evaluator_id": "blind-panel-v1",
            "rubric_id": "rubric-v1",
            "fixed_config_id": "ablation-v1",
            "split": "held_out",
            "label_source": "human",
        },
        "cases": [
            {
                "case_id": f"{mode}-{i}",
                "mode": mode,
                "quality": quality,
                "unsupported_rate": 0.0,
                "total_tokens": tokens,
                "latency_ms": 100,
                "success": True,
            }
            for mode in ("faithful", "path", "explore")
            for i in range(5)
        ],
    }


@pytest.mark.parametrize("quality,tokens", [(0.75, 1000), (0.7, 850)])
def test_gain_only_recommends_review_without_mutating_runs(quality, tokens):
    baseline, candidate = trial(), trial(quality, tokens)
    original = copy.deepcopy((baseline, candidate))
    report = assess_evolution_trial(baseline, candidate)
    assert report["decision"] == "recommend_review"
    assert report["automatic_promotion"] is False
    assert (baseline, candidate) == original


def test_exploration_gain_cannot_hide_faithful_regression():
    candidate = trial(0.9)
    for row in candidate["cases"][:5]:
        row["quality"] = 0.65
    result = assess_evolution_trial(trial(), candidate)
    assert result["macro_quality_delta"] > 0
    assert result["decision"] == "reject"
    assert "faithful:quality_regression" in result["reasons"]


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("total_tokens", 1200, "total_tokens_increased"),
        ("latency_ms", 120, "latency_ms_increased"),
        ("unsupported_rate", 0.01, "unsupported_claims_increased"),
        ("quality", 0.4, "severe_case_regression"),
    ],
)
def test_quality_improvement_does_not_override_guardrails(field, value, reason):
    candidate = trial(0.8)
    for row in candidate["cases"][:5]:
        row[field] = value
    result = assess_evolution_trial(trial(), candidate)
    assert result["decision"] == "reject"
    assert f"faithful:{reason}" in result["reasons"]


def test_missing_question_family_and_small_sets_fail_closed():
    a, b = trial(), trial(0.8)
    a["cases"], b["cases"] = a["cases"][:8], b["cases"][:8]
    result = assess_evolution_trial(a, b)
    assert result["decision"] == "insufficient_evidence"
    assert "explore:insufficient_cases" in result["reasons"]


def test_unstable_small_gain_needs_more_evidence():
    candidate = trial(0.73)
    candidate["cases"][0]["quality"] = 0.6
    assert (
        assess_evolution_trial(trial(), candidate)["decision"]
        == "insufficient_evidence"
    )


def test_unchanged_trial_has_no_material_gain():
    assert assess_evolution_trial(trial(), trial())["decision"] == "reject"


def test_rounding_cannot_create_a_gain_at_the_admission_boundary():
    result = assess_evolution_trial(trial(), trial(0.7199996))
    assert result["decision"] == "reject"


def test_finite_individual_costs_cannot_overflow_aggregate_gate():
    with pytest.raises(ValueError, match="Aggregate"):
        assess_evolution_trial(trial(tokens=1e308), trial(tokens=1e308))


@pytest.mark.parametrize(
    "field,value",
    [
        ("quality", float("nan")),
        ("total_tokens", float("inf")),
        ("latency_ms", -1),
        ("quality", True),
        ("quality", 2),
        ("success", "false"),
        ("mode", "unknown"),
    ],
)
def test_invalid_observations_are_not_scores(field, value):
    candidate = trial(0.8)
    candidate["cases"][0][field] = value
    with pytest.raises(ValueError):
        assess_evolution_trial(trial(), candidate)


@pytest.mark.parametrize(
    "key",
    [
        "dataset_id",
        "evidence_snapshot",
        "model_id",
        "rubric_id",
        "fixed_config_id",
        "evaluator_id",
    ],
)
def test_mismatched_experiment_metadata_is_rejected(key):
    candidate = trial(0.8)
    candidate["metadata"][key] = "different"
    with pytest.raises(ValueError):
        assess_evolution_trial(trial(), candidate)


@pytest.mark.parametrize(
    "key,value",
    [
        ("split", "training"),
        ("label_source", "lexical_proxy"),
        ("label_source", "self_rating"),
    ],
)
def test_training_or_self_scores_cannot_admit_candidate(key, value):
    a, b = trial(), trial(0.8)
    a["metadata"][key] = b["metadata"][key] = value
    with pytest.raises(ValueError):
        assess_evolution_trial(a, b)


@pytest.mark.parametrize("change", ["drop", "duplicate", "mode"])
def test_case_pairing_is_strict(change):
    candidate = trial(0.8)
    if change == "drop":
        candidate["cases"].pop()
    elif change == "duplicate":
        candidate["cases"].append(candidate["cases"][0])
    else:
        candidate["cases"][0]["mode"] = "path"
    with pytest.raises(ValueError):
        assess_evolution_trial(trial(), candidate)


def test_new_failure_cannot_hide_behind_a_fixed_failure():
    a, b = trial(), trial(0.8)
    a["cases"][0].update(success=False, quality=0)
    a["cases"][1]["quality"] = 0.1
    b["cases"][1].update(success=False, quality=0)
    result = assess_evolution_trial(a, b)
    assert result["groups"]["faithful"]["additional_failures"] == 0
    assert "faithful:failures_increased" in result["reasons"]


def test_failed_case_cannot_claim_high_quality():
    b = trial(0.8)
    b["cases"][0]["success"] = False
    with pytest.raises(ValueError):
        assess_evolution_trial(trial(), b)


def test_lexical_metrics_are_explicitly_not_admission_labels():
    score = score_answer(answer="not safe", reference_answer="safe")
    assert score["scoring_method"] == "lexical_proxy"
    assert score["suitable_for_promotion"] is False


def test_cli_runs_offline_and_reports_invalid_input(tmp_path):
    a, b = tmp_path / "baseline.json", tmp_path / "candidate.json"
    a.write_text(json.dumps(trial()), encoding="utf-8")
    b.write_text(json.dumps(trial(0.8)), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "docthinker.evaluation",
        "--baseline",
        str(a),
        "--candidate",
        str(b),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["decision"] == "recommend_review"
    b.write_text("{}", encoding="utf-8")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert json.loads(result.stdout)["decision"] == "invalid_input"
