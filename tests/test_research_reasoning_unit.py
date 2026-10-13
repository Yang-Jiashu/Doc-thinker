"""Small real source corpora test proof, structural mapping and sampling errors."""

import copy
import json

import pytest

from docthinker.server.research_reasoning import (
    MAX_CLAIMS,
    MAX_CONCLUSIONS,
    MAX_DEPTH,
    MAX_MAPPING_EXPANSIONS,
    MAX_RESULT_BYTES,
    choose_mode,
    extract_claims,
    reason,
)


def chunk(content, identifier="c1", document_id="d1", **fields):
    return {"id": identifier, "document_id": document_id, "content": content, **fields}


def run(texts, mode):
    rows = [chunk(text, f"c{index}", f"d{index}") for index, text in enumerate(texts)]
    claims = extract_claims(rows)
    return reason("请检查原文再推理", claims, mode), claims


def codes(result):
    return {gap["code"] for gap in result["gaps"]}


def test_literal_claim_offsets_are_exact_with_unicode_labels_and_pages():
    text = "  案例：能源\n  可再生能源降低进口依赖。\n观察：政策工具增加可再生能源投资。"
    rows = extract_claims([chunk(text, page=4)])
    assert len(rows) == 2
    for row in rows:
        assert text[row["excerpt_start"]:row["excerpt_end"]] == row["quote"]
        assert row["page"] == 4
        assert row["case_label"] == "能源"
        assert row["source_ids"] == ["c1"]
    assert rows[0]["subject"] == "可再生能源"
    assert rows[0]["relation"] == "reduces"


def test_import_dependency_is_preserved_as_subject_not_mistaken_for_verb():
    rows = extract_claims([chunk("进口依赖降低能源安全。A降低B。B增加C。")])
    assert [(row["subject"], row["relation"], row["object"]) for row in rows] == [
        ("进口依赖", "reduces", "能源安全"), ("a", "reduces", "b"), ("b", "increases", "c")]


def test_multiline_english_case_labels_and_all_observations_are_read():
    rows = extract_claims([chunk("Case: alpha\nSolar reduces imports.\nCase: beta\nSolar does not reduce imports.")])
    assert len(rows) == 2
    assert [row["case_label"] for row in rows] == ["alpha", "beta"]
    assert [row["polarity"] for row in rows] == [1, -1]


def test_same_line_case_label_preserves_literal_quote_offsets():
    text = "Case: cloud: Redundancy reduces outage risk."
    row = extract_claims([chunk(text)])[0]
    assert row["quote"] == "Redundancy reduces outage risk."
    assert text[row["excerpt_start"]:row["excerpt_end"]] == row["quote"]
    assert row["case_label"] == "cloud"


def test_claim_ids_are_deterministic_across_source_input_order():
    rows = [chunk("Solar reduces imports.", "c2", "d2"), chunk("Demand increases imports.")]
    assert extract_claims(rows) == extract_claims(list(reversed(rows)))


@pytest.mark.parametrize("text", [
    "Solar may reduce imports.", "Solar might reduce imports.", "太阳能可能降低进口依赖。",
    "Solar reduces imports only if demand falls.", "Solar reduces imports unless drought occurs.",
    "All solar reduces imports.", "Solar reduces imports by 30%.",
    "Solar or wind reduces imports.", "Solar sometimes reduces imports.",
    "Solar can reduce imports.", "Solar should reduce imports.", "新能源不会降低进口依赖。",
    "Solar reduces imports but increases costs.", "They reduce imports.",
])
def test_uncertainty_and_unsupported_qualifiers_are_not_silently_facts(text):
    assert extract_claims([chunk(text)]) == []


@pytest.mark.parametrize("field,value", [("is_generated", True), ("is_discovered", "1"),
                                         ("provenance", "hypothesis"), ("state", "deleted")])
def test_generated_or_revoked_content_never_becomes_observation(field, value):
    assert extract_claims([chunk("Solar reduces imports.", **{field: value})]) == []


def test_excerpt_boundaries_cannot_drop_a_negation_or_conditional_prefix():
    excerpt = "Solar reduces imports. Demand increases imports. Grid reduces losses"
    original = "If " + excerpt + " when ready."
    rows = extract_claims([chunk(excerpt, excerpt_start=3, truncated=True, original_content_length=len(original))])
    assert len(rows) == 1
    assert rows[0]["subject"] == "demand"
    assert original[rows[0]["excerpt_start"]:rows[0]["excerpt_end"]] == rows[0]["quote"]


def test_exact_document_hash_overrides_weaker_chunk_fingerprint():
    rows = extract_claims([chunk("Solar reduces imports.", content_hash="a" * 64)])
    assert rows[0]["fingerprint_kind"] == "document"
    assert rows[0]["source_fingerprint"] == "a" * 64


@pytest.mark.parametrize("text,polarity", [("Solar reduces imports.", 1), ("Solar does not reduce imports.", -1),
                                         ("新能源不降低进口依赖。", -1), ("Grid is not ready.", -1)])
def test_negation_is_preserved(text, polarity):
    assert extract_claims([chunk(text)])[0]["polarity"] == polarity


def test_deduction_requires_every_explicit_premise_and_keeps_full_proof_dag():
    result, claims = run([
        "Observation: grid ready. If grid ready then supply grows.",
        "If supply grows and imports decrease then security improves. Observation: imports decrease.",
    ], "deduction")
    assert len(result["conclusions"]) == 2
    end = next(row for row in result["conclusions"] if row["subject"] == "security improves")
    assert end["depth"] == 2
    assert end["source_ids"] == ["c0", "c1"]
    assert {row["id"] for row in claims} <= {node["id"] for node in end["proof"]}
    assert any(item.startswith("deduction_") for item in end["premise_ids"])
    assert end["epistemic_status"] == "conditional_on_source_premises"
    assert result["trace"]["model_calls"] == 0


def test_missing_middle_premise_never_becomes_a_completed_causal_chain():
    result, _ = run(["Observation: grid ready. If grid ready and imports decrease then security improves."], "deduction")
    assert result["conclusions"] == []
    assert "missing_premises" in codes(result)


def test_ordinary_causal_edges_do_not_become_transitive_logical_rules():
    result, _ = run(["Investment increases supply. Supply reduces imports. Imports reduces security."], "deduction")
    assert result["conclusions"] == []
    assert "no_explicit_rules" in codes(result)


def test_chinese_conjunction_rule_requires_both_actual_observations():
    result, _ = run(["观察：电网就绪。观察：供给充足。如果电网就绪且供给充足，那么进口下降。"], "deduction")
    assert len(result["conclusions"]) == 1
    assert result["conclusions"][0]["subject"] == "进口"
    assert result["conclusions"][0]["relation"] == "state_decrease"
    assert len(result["conclusions"][0]["premise_ids"]) == 3


def test_natural_chinese_trends_satisfy_explicit_rule_without_observation_labels():
    result, _ = run(["国内供给增加。投资增加。如果国内供给增加且投资增加，那么进口依赖下降。"], "deduction")
    assert len(result["conclusions"]) == 1
    assert result["conclusions"][0]["subject"] == "进口依赖"
    assert result["conclusions"][0]["relation"] == "state_decrease"


def test_explicit_negative_premise_is_not_inferred_from_absence():
    rule = "If not grid ready then imports grow."
    missing, _ = run([rule], "deduction")
    present, _ = run([rule + " Observation: not grid ready."], "deduction")
    assert missing["conclusions"] == []
    assert present["conclusions"][0]["subject"] == "imports"


def test_conflicting_observations_stop_dependent_deduction_without_explosion():
    result, _ = run(["Observation: grid ready. Observation: not grid ready. If grid ready then supply grows."], "deduction")
    assert result["conclusions"] == []
    assert "contradictory_premises" in codes(result)


def test_opposing_observed_trends_stop_dependent_deduction():
    result, _ = run([
        "Solar reduces imports. Solar increases imports. If solar reduces imports then security improves.",
    ], "deduction")
    assert result["conclusions"] == []
    assert "contradictory_premises" in codes(result)


def test_opposing_derived_trends_retract_existing_descendants():
    result, _ = run([
        "Observation: start. Supply increases. If supply increases then ready. If start then supply decreases.",
    ], "deduction")
    assert result["conclusions"] == []
    assert "contradictory_premises" in codes(result)


def test_conflict_discovered_after_multiple_steps_removes_descendants():
    result, _ = run([
        "Observation: start. If start then ready. If ready then supply grows.",
        "If start then intermediate. If intermediate then not ready.",
    ], "deduction")
    assert all(row["subject"] not in {"ready", "supply"} for row in result["conclusions"])
    assert "contradictory_premises" in codes(result)


def test_qualified_observation_cannot_satisfy_unqualified_premise():
    result, _ = run(["Solar reduces imports under low demand. If solar reduces imports then security improves."], "deduction")
    assert result["conclusions"] == []
    assert "missing_premises" in codes(result)


def test_facts_from_different_named_cases_are_never_combined_into_one_proof():
    result, _ = run([
        "Case: north\nObservation: grid ready.",
        "Case: south\nIf grid ready then supply grows.",
    ], "deduction")
    assert result["conclusions"] == []


def test_cycle_terminates_without_duplicate_fact_or_unbounded_proof():
    result, _ = run(["Observation: start. If start then ready. If ready then start."], "deduction")
    assert len(result["conclusions"]) == 1
    assert result["trace"]["closure_iterations"] <= MAX_DEPTH


def test_unfinished_chain_at_depth_limit_is_disclosed_as_bounded():
    names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]
    text = "Observation: alpha. " + " ".join(f"If {a} then {b}." for a, b in zip(names, names[1:]))
    result, _ = run([text], "deduction")
    assert len(result["conclusions"]) == MAX_DEPTH
    assert all(row["subject"] != "kappa" for row in result["conclusions"])
    assert "reasoning_budget_reached" in codes(result)


def test_explicit_target_is_selected_before_eight_conclusion_display_cap():
    names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "ultimate"]
    text = "Observation: start. " + " ".join(f"If start then {name}." for name in names)
    claims = extract_claims([chunk(text)])
    result = reason("Prove ultimate", claims, "deduction")
    assert [row["subject"] for row in result["conclusions"]] == ["ultimate"]
    assert result["trace"]["derived_count"] == 11
    assert result["trace"]["unrelated_derivations_omitted"] == 10


def test_broad_derive_from_question_retains_forward_closure():
    claims = extract_claims([chunk("Observation: start. If start then ready. If ready then supply grows.")])
    result = reason("What can we derive from start?", claims, "deduction")
    assert {row["subject"] for row in result["conclusions"]} == {"ready", "supply"}


def test_cross_domain_analogy_matches_structure_with_no_shared_entity_names():
    result, _ = run([
        "Redundancy reduces outage risk. Outage risk reduces availability.",
        "Supplier diversity reduces concentration risk. Concentration risk reduces energy security.",
    ], "analogy")
    row = result["conclusions"][0]
    assert row["kind"] == "analogy_hypothesis"
    assert {item["base"]: item["target"] for item in row["mappings"]} == {
        "redundancy": "supplier diversity", "outage risk": "concentration risk", "availability": "energy security"}
    assert len(row["matched_relations"]) == 2
    assert row["source_ids"] == ["c0", "c1"]
    assert "analogy_is_not_entailment" in codes(result)
    assert row["epistemic_status"] == "hypothesis_requires_target_validation"


def test_isolated_keyword_similarity_cannot_make_an_analogy():
    result, _ = run(["Solar reduces imports.", "Redundancy reduces outages."], "analogy")
    assert result["conclusions"] == []


def test_disconnected_relations_cannot_make_a_systematic_analogy():
    result, _ = run([
        "Solar reduces imports. Batteries increases stability.",
        "Redundancy reduces outages. Monitoring increases availability.",
    ], "analogy")
    assert result["conclusions"] == []


def test_wrong_direction_or_many_to_one_mapping_is_rejected():
    result, _ = run([
        "Alpha reduces beta. Alpha increases gamma.",
        "Delta reduces epsilon. Zeta increases epsilon.",
    ], "analogy")
    assert result["conclusions"] == []


def test_opposite_sign_cannot_support_same_structural_analogy():
    result, _ = run([
        "Alpha reduces beta. Beta increases gamma.",
        "Delta reduces epsilon. Epsilon does not increase zeta.",
    ], "analogy")
    assert result["conclusions"] == []


def test_extra_opposite_relation_invalidates_an_otherwise_matching_analogy():
    result, _ = run([
        "Alpha reduces beta. Beta increases gamma.",
        "Delta reduces epsilon. Epsilon increases zeta. Delta does not reduce epsilon.",
    ], "analogy")
    assert result["conclusions"] == []
    assert "opposite_relation_rejected" in codes(result)


def test_opposite_observed_trend_invalidates_projected_analogy():
    result, _ = run([
        "Alpha reduces beta. Beta increases gamma. Alpha reduces gamma.",
        "Delta reduces epsilon. Epsilon increases zeta. Delta increases zeta.",
    ], "analogy")
    assert result["conclusions"] == []
    assert "opposite_relation_rejected" in codes(result)


def test_source_supplied_role_types_constrain_mapping():
    common = ["Alpha reduces beta. Beta increases gamma.", "Delta reduces epsilon. Epsilon increases zeta."]
    wrong, _ = run(["Role: alpha = intervention. " + common[0], "Role: delta = outcome. " + common[1]], "analogy")
    right, _ = run(["Role: alpha = intervention. " + common[0], "Role: delta = intervention. " + common[1]], "analogy")
    assert wrong["conclusions"] == []
    assert right["conclusions"]


def test_condition_differences_are_visible_not_silently_transferred():
    result, _ = run([
        "Alpha reduces beta under low load. Beta increases gamma.",
        "Delta reduces epsilon under high load. Epsilon increases zeta.",
    ], "analogy")
    assert any(row["code"] == "different_conditions" for row in result["conclusions"][0]["differences"])


def test_analogy_projects_only_explicit_base_relation_between_mapped_roles():
    result, claims = run([
        "Redundancy reduces outage risk. Outage risk reduces availability. Redundancy increases availability under low load.",
        "Supplier diversity reduces concentration risk. Concentration risk reduces energy security.",
    ], "analogy")
    row = result["conclusions"][0]
    candidate = row["candidate_inferences"][0]
    assert (candidate["subject"], candidate["relation"], candidate["object"]) == ("supplier diversity", "increases", "energy security")
    assert candidate["conditions"] == ["low load"]
    assert candidate["target_conditions_verified"] is False
    assert candidate["epistemic_status"] == "unverified_transfer_hypothesis"
    assert candidate["base_claim_id"] in row["premise_ids"]
    assert candidate["base_claim_id"] in {claim["id"] for claim in claims}
    assert candidate["source_ids"] == ["c0"]


@pytest.mark.parametrize("cloud_id,energy_id", [
    ("a", "z"), ("z", "a"),
    ("78f0c8b8-5c9d-4120-983c-11c49b5d98c3", "3269f407-0c62-430f-87bb-bcb128be00d4"),
    ("3269f407-0c62-430f-87bb-bcb128be00d4", "78f0c8b8-5c9d-4120-983c-11c49b5d98c3"),
])
def test_analogy_transfer_direction_is_independent_of_document_uuid_order(cloud_id, energy_id):
    sources = [
        chunk("Redundancy reduces outage risk. Outage risk reduces availability. Redundancy increases availability under low load.", "cloud-source", cloud_id),
        chunk("Supplier diversity reduces concentration risk. Concentration risk reduces energy security.", "energy-source", energy_id),
    ]
    result = reason("类比 redundancy", extract_claims(sources), "analogy")
    assert len(result["conclusions"]) == 1
    candidate = result["conclusions"][0]["candidate_inferences"][0]
    assert result["conclusions"][0]["base_case"] == cloud_id
    assert (candidate["subject"], candidate["relation"], candidate["object"]) == ("supplier diversity", "increases", "energy security")
    assert candidate["conditions"] == ["low load"]
    assert candidate["source_ids"] == ["cloud-source"]
    assert result["trace"]["mapping_directions_examined"] == 2


def test_analogy_does_not_invent_a_missing_causal_shortcut():
    result, _ = run([
        "Redundancy reduces outage risk. Outage risk reduces availability.",
        "Supplier diversity reduces concentration risk. Concentration risk reduces energy security.",
    ], "analogy")
    assert result["conclusions"][0]["candidate_inferences"] == []


def test_induction_reports_actual_support_and_counterexamples():
    result, _ = run([
        "Case: north\nSolar reduces imports.",
        "Case: south\nSolar reduces imports.",
        "Case: winter\nSolar does not reduce imports.",
    ], "induction")
    row = result["conclusions"][0]
    assert row["support_count"] == 2
    assert row["counterexample_count"] == 1
    assert row["source_group_count"] == 3
    assert row["source_ids"] == ["c0", "c1", "c2"]
    assert row["counterexample_ids"]
    assert row["epistemic_status"] == "sample_regularity_not_universal"
    assert "source_independence_not_verified" in codes(result)
    assert "confidence" not in row


def test_induction_human_conclusion_keeps_qualifying_conditions():
    result, _ = run([
        "Case: north\nSolar reduces imports under low demand.",
        "Case: south\nSolar reduces imports under low demand.",
    ], "induction")
    row = result["conclusions"][0]
    assert row["pattern"]["conditions"] == ["low demand"]
    assert "适用条件：low demand" in row["text"]


def test_observed_inverse_trend_counts_as_counterexample_to_inductive_pattern():
    result, _ = run([
        "Case: north\nSolar reduces imports.",
        "Case: south\nSolar reduces imports.",
        "Case: rebound\nSolar increases imports.",
    ], "induction")
    row = result["conclusions"][0]
    assert row["support_count"] == 2
    assert row["counterexample_count"] == 1
    assert row["counterexample_ids"]


def test_absence_of_increase_is_not_inductive_evidence_of_reduction():
    result, _ = run([
        "Case: north\nSolar does not increase imports.",
        "Case: south\nSolar does not increase imports.",
    ], "induction")
    assert result["conclusions"] == []


def test_duplicate_upload_and_duplicate_chunks_are_not_independent_support():
    claims = extract_claims([
        chunk("Case: north\nSolar reduces imports.", "c1", "d1"),
        chunk("Case: north\nSolar reduces imports.", "c2", "d2"),
        chunk("Case: second excerpt\nSolar reduces imports.", "c3", "d1"),
    ])
    result = reason("归纳", claims)
    assert result["conclusions"] == []
    assert "duplicate_observations_collapsed" in codes(result)


def test_identical_document_hash_collapses_different_excerpts_of_reupload():
    claims = extract_claims([
        chunk("Case: north\nSolar reduces imports.", "c1", "d1", content_hash="a" * 64),
        chunk("Case: south\nSolar reduces imports.", "c2", "d2", content_hash="a" * 64),
    ])
    assert reason("归纳", claims)["conclusions"] == []


def test_different_conditions_and_different_relations_are_not_pooled():
    result, _ = run([
        "Solar reduces imports under low demand.",
        "Solar reduces imports under high demand.",
        "Solar increases imports under low demand.",
    ], "induction")
    assert result["conclusions"] == []


def test_source_group_with_both_signs_is_counterexample_not_positive_support():
    claims = extract_claims([
        chunk("Case: north\nSolar reduces imports.", "c1", "d1"),
        chunk("Case: south\nSolar reduces imports.", "c2", "d2"),
        chunk("Case: mixed\nSolar reduces imports. Solar does not reduce imports.", "c3", "d3"),
    ])
    row = reason("归纳", claims)["conclusions"][0]
    assert row["support_count"] == 2 and row["counterexample_count"] == 1


def test_budgets_never_expand_from_caller_supplied_large_values():
    source = " ".join(f"Item{name} reduces outcome{name}." for name in ["alpha", "beta", "gamma"])
    rows = extract_claims([chunk(source)], max_claims=10_000)
    assert len(rows) <= MAX_CLAIMS
    result = reason("类比", rows, max_conclusions=10_000)
    assert result["trace"]["max_conclusions"] == MAX_CONCLUSIONS
    assert len(json.dumps(result)) < 50_000


def test_zero_limits_and_ordinary_evidence_mode_do_no_reasoning():
    rows = extract_claims([chunk("Observation: start. If start then ready.")])
    assert extract_claims([chunk("Solar reduces imports.")], max_claims=0) == []
    assert reason("演绎", rows, max_conclusions=0)["conclusions"] == []
    assert reason("这篇论文说了什么", rows)["status"] == "not_requested"


def test_malformed_stored_claims_are_ignored_without_crash_or_invented_fact():
    valid = extract_claims([chunk("Solar reduces imports.")])[0]
    malformed = [None, "text", {}, {"kind": "observation"}, {**valid, "scope": None},
                 {**valid, "conditions": "rain"}, {**valid, "polarity": True}, {**valid, "relation": []}]
    assert reason("归纳", malformed)["conclusions"] == []


def test_reasoning_does_not_mutate_claims_or_sources():
    rows = [chunk("Observation: start. If start then ready.")]
    before = copy.deepcopy(rows)
    claims = extract_claims(rows)
    old = copy.deepcopy(claims)
    reason("演绎", claims)
    assert rows == before and claims == old


@pytest.mark.parametrize("query,mode", [("找一个结构类比", "analogy"), ("根据规则演绎", "deduction"),
                                      ("归纳共同规律", "induction"), ("What does this say?", "evidence")])
def test_mode_routing_only_opts_in_on_explicit_reasoning_intent(query, mode):
    assert choose_mode(query) == mode


def test_explicit_mode_wins_and_unknown_mode_is_rejected():
    assert choose_mode("请类比", "induction") == "induction"
    with pytest.raises(ValueError):
        choose_mode("question", "invent")


def test_explicit_leading_negative_relation_does_not_become_positive_entity_name():
    rows = extract_claims([chunk("Observation: not solar reduces imports.")])
    assert rows[0]["subject"] == "solar"
    assert rows[0]["polarity"] == -1


def test_dense_graph_search_has_global_expansion_and_serialization_budgets():
    # Same typed topology with renamed roles creates many competing maps. The
    # bounded search must stop even when matching names cannot prune the search.
    names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    sources = []
    for letter in "abcdefgh":
        edges = [f"{letter}{a} reduces {letter}{b}." for a in names for b in names if a != b][:16]
        sources.append(chunk(" ".join(edges), f"c{letter}", f"d{letter}"))
    result = reason("analogy", extract_claims(sources))
    assert result["trace"]["mapping_expansions"] <= MAX_MAPPING_EXPANSIONS
    assert len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode()) <= MAX_RESULT_BYTES
    assert "reasoning_budget_reached" in codes(result)


@pytest.mark.parametrize("mode,text", [
    ("deduction", "If grid ready then supply grows."),
    ("analogy", "Solar reduces imports."),
    ("induction", "Solar reduces imports."),
])
def test_reasoning_gaps_have_human_messages_for_client_rendering(mode, text):
    result = reason("核对", extract_claims([chunk(text)]), mode)
    assert result["gaps"]
    assert all(isinstance(gap.get("message"), str) and len(gap["message"]) > 10 for gap in result["gaps"])
