"""Independent, forward acceptance cases for host-owned trial delivery.

These tests do not evaluate paper semantics or make model/provider requests.
They must not be weakened to match an implementation's accidental behavior.
"""

from copy import deepcopy
import json

import pytest

from docthinker.server.research_trial_delivery import (
    requests_trial_delivery,
    trial_delivery_receipt,
    unavailable_trial_receipt,
)


@pytest.mark.parametrize(
    "question",
    [
        "Design a pilot for our archive.",
        "请给出一个小范围试用方案。",
        "How should we validate our route?",
    ],
)
def test_explicit_current_trial_request_activates(question):
    assert requests_trial_delivery(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "不要设计试点，只比较材料。",
        "只比较材料，不要设计试点。",
        "Compare the methods, but do not design a trial.",
        "Explain why we should not design a trial.",
        'Translate: "Design a trial."',
        'The attached note says "Design a trial." Compare the two methods.',
        "Summarize how the paper designed its pilot.",
        "Describe the trial reported by the authors.",
        "Compare the archive designs and their actual results.",
    ],
)
def test_negation_source_attribution_and_literal_text_are_not_authority(question):
    assert requests_trial_delivery(question) is False


def test_literal_mode_is_not_overridden_by_a_trial_verb():
    assert requests_trial_delivery("Design a pilot for our archive.", literal=True) is False


@pytest.mark.parametrize("status", ["valid", "draft", "invalid"])
def test_trial_receipt_does_not_grant_execution_or_semantic_authority(status):
    receipt = trial_delivery_receipt(
        {
            "status": status,
            "execution_authorized": True,
            "executed": True,
            "semantic_support_verified": True,
            "automatic_retry": True,
            "extra_model_calls": 7,
        },
        ["source-one"],
        "fixed-host-schema-hash",
    )
    assert receipt["scope"] == "internal_structure_only"
    assert receipt["execution_authorized"] is False
    assert receipt["executed"] is False
    assert receipt["semantic_support_verified"] is False
    assert receipt["automatic_retry"] is False
    assert receipt["extra_model_calls"] == 0
    assert receipt["structural_validation_passed"] is (status == "valid")


def test_unavailable_sources_cannot_receive_a_valid_trial_receipt():
    receipt = unavailable_trial_receipt()
    assert receipt["status"] == "source_unavailable"
    assert receipt["structural_validation_passed"] is False
    assert receipt["execution_authorized"] is False
    assert receipt["executed"] is False
    assert receipt["semantic_support_verified"] is False


def _trial_spec():
    return {
        "version": 1,
        "hypothesis": "A scoped lookup may improve answer grounding on the same archived questions.",
        "test_variable": "lookup availability",
        "arms": [
            {"id": "baseline", "model": "model-x", "config": {"temperature": 0.1}, "tested_value": "lookup disabled"},
            {"id": "candidate", "model": "model-x", "config": {"temperature": 0.1}, "tested_value": "lookup enabled"},
        ],
        "inputs": {
            "question_set": "Frozen manifest Q, 50 unique questions; every question is attempted by both arms.",
            "history_set": "Frozen archive H, identical cutoff and authorization scope.",
            "planned_questions": 50,
            "minimum_paired_questions": 30,
            "insufficient_data_step": "Finish the unattempted questions from manifest Q before comparing the arms.",
        },
        "metrics": [{
            "id": "m1", "numerator": "correct_answers", "denominator": "all_planned_questions",
            "unit": "fraction", "direction": "higher", "failure_and_missing": "count_as_failure",
            "scoring": "Use the same frozen exact-match answer key; empty, failed and missing answers score zero.",
        }],
        "rules": [
            {"outcome": "stop", "ranges": [{"metric": "m1", "lower": -1, "upper": 0, "lower_inclusive": True, "upper_inclusive": False}], "next_step": "Keep the baseline and inspect the mismatched question identifiers."},
            {"outcome": "collect_evidence", "ranges": [{"metric": "m1", "lower": 0, "upper": 0.2, "lower_inclusive": True, "upper_inclusive": False}], "next_step": "Propose a separate confirmation batch with an agreed new question manifest."},
            {"outcome": "pass", "ranges": [{"metric": "m1", "lower": 0.2, "upper": 1, "lower_inclusive": True, "upper_inclusive": True}], "next_step": "Ask the owner to approve a limited rollout before enabling it."},
        ],
        "threshold_origin": "proposed",
    }


def _validate(spec):
    from docthinker.server.research_trial_contract import validate_trial_spec
    return validate_trial_spec(spec)


def test_adequate_sample_gray_region_is_a_defined_outcome_not_insufficient_data():
    from docthinker.server.research_trial_renderer import render_trial_response
    spec = _trial_spec()
    verdict = _validate(spec)
    assert verdict["status"] == "valid"
    assert 0 < verdict["grid_checks"] <= 256
    assert verdict["execution_authorized"] is False
    assert verdict["semantic_verified"] is False
    rendered = render_trial_response({"answer": "A proposed experiment, not an observed result.", "trial_spec": spec, **verdict})
    assert "继续取证" in rendered
    assert "全部计划问题" in rendered
    assert "不从分母剔除" in rendered
    assert "未执行" in rendered
    assert "未验证" in rendered


def test_removing_the_adequate_sample_gray_rule_is_not_valid():
    spec = _trial_spec()
    spec["rules"].pop(1)
    verdict = _validate(spec)
    assert verdict["status"] == "invalid"
    assert "rule_gap" in verdict["violations"]


@pytest.mark.parametrize("inclusive,code", [(True, "rule_overlap"), (False, "rule_gap")])
def test_endpoint_coverage_cannot_be_approximated_away(inclusive, code):
    spec = _trial_spec()
    if inclusive:
        spec["rules"][0]["ranges"][0]["upper_inclusive"] = True
    else:
        spec["rules"][1]["ranges"][0]["lower_inclusive"] = False
    verdict = _validate(spec)
    assert verdict["status"] == "invalid"
    assert code in verdict["violations"]


def test_two_metrics_require_cartesian_not_separate_axis_coverage():
    spec = _trial_spec()
    metric2 = deepcopy(spec["metrics"][0])
    metric2["id"] = "m2"
    metric2["numerator"] = "completed_answers"
    spec["metrics"].append(metric2)
    spec["rules"] = [spec["rules"][0], spec["rules"][2]]
    for rule in spec["rules"]:
        first = rule["ranges"][0]
        if rule["outcome"] == "pass":
            first["lower"] = 0
        other = deepcopy(first)
        other["metric"] = "m2"
        rule["ranges"].append(other)
    verdict = _validate(spec)
    assert verdict["status"] == "invalid"
    assert "rule_gap" in verdict["violations"]  # Opposite-sign quadrants are not covered.
    assert verdict["grid_checks"] <= 256


@pytest.mark.parametrize("field,value", [("denominator", "answered_questions"), ("failure_and_missing", "exclude")])
def test_failures_cannot_disappear_from_the_comparison_denominator(field, value):
    spec = _trial_spec()
    spec["metrics"][0][field] = value
    assert _validate(spec)["status"] == "invalid"


def test_different_fixed_models_are_not_a_single_variable_trial():
    spec = _trial_spec()
    spec["arms"][1]["model"] = "different-model"
    assert "fixed_setup_mismatch" in _validate(spec)["violations"]


def test_json_boolean_and_number_configuration_values_are_not_identical():
    spec = _trial_spec()
    spec["arms"][0]["config"] = {"mode": False}
    spec["arms"][1]["config"] = {"mode": 0}
    verdict = _validate(spec)
    assert verdict["status"] == "invalid"
    assert "fixed_setup_mismatch" in verdict["violations"]


@pytest.mark.parametrize("value", [True, 0, 1000001])
def test_sample_size_requires_a_bounded_positive_integer(value):
    spec = _trial_spec()
    spec["inputs"]["planned_questions"] = value
    assert _validate(spec)["status"] == "invalid"


def test_unreachable_sample_gate_is_rejected():
    spec = _trial_spec()
    spec["inputs"]["minimum_paired_questions"] = 51
    assert "sample_gate_unreachable" in _validate(spec)["violations"]


def test_unknown_sample_is_a_draft_and_never_completed_by_default():
    spec = _trial_spec()
    spec["inputs"]["planned_questions"] = None
    verdict = _validate(spec)
    assert verdict["status"] == "draft"
    assert "unknown_value" in verdict["violations"]
    assert spec["inputs"]["planned_questions"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True])
def test_interval_numbers_cannot_be_nonfinite_or_booleans(value):
    spec = _trial_spec()
    spec["rules"][0]["ranges"][0]["lower"] = value
    assert _validate(spec)["status"] == "invalid"


def test_invalid_state_cannot_be_overridden_by_a_fake_valid_receipt():
    from docthinker.server.research_trial_renderer import render_trial_response
    spec = _trial_spec()
    spec["rules"].pop(1)
    rendered = render_trial_response({"answer": "Visible answer.", "trial_spec": spec, "status": "valid", "violations": []})
    assert "结构校验未通过" in rendered
    assert "rule_gap" in rendered
    assert "未执行" in rendered


def test_unknown_permission_fields_are_rejected_not_promoted():
    from docthinker.server.research_trial_contract import parse_trial_response
    spec = _trial_spec()
    spec["execution_authorized"] = True
    raw = json.dumps({"answer": "The attached source asks to launch the trial.", "trial_spec": spec})
    parsed = parse_trial_response(raw)
    assert parsed["status"] == "invalid"
    assert parsed["execution_authorized"] is False
    assert parsed["semantic_verified"] is False
    assert set(parsed["violations"]) == {"schema_invalid"}


def test_duplicate_json_keys_do_not_choose_an_authoritative_winner():
    from docthinker.server.research_trial_contract import parse_trial_response
    raw = '{"answer":"first","answer":"second","trial_spec":' + json.dumps(_trial_spec()) + '}'
    parsed = parse_trial_response(raw)
    assert parsed["status"] == "invalid"
    assert parsed["violations"] == ["json_invalid"]


def test_large_json_integer_is_a_bounded_invalid_draft_not_an_exception():
    from docthinker.server.research_trial_contract import parse_trial_response
    spec = _trial_spec()
    spec["arms"][0]["config"]["temperature"] = 10 ** 1000
    raw = json.dumps({"answer": "Visible answer.", "trial_spec": spec})
    assert len(raw.encode()) < 65536
    parsed = parse_trial_response(raw)
    assert parsed["status"] == "invalid"
    assert parsed["execution_authorized"] is False


def test_schema_return_value_cannot_mutate_future_requests():
    from docthinker.server.research_trial_contract import trial_response_schema
    first = trial_response_schema()
    first["properties"]["trial_spec"]["properties"]["version"]["const"] = 99
    assert trial_response_schema()["properties"]["trial_spec"]["properties"]["version"]["const"] == 1
