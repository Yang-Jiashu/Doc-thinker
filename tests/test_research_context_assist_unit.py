"""Background material selection never becomes an independent answer mode."""
from __future__ import annotations

import copy
import json

import pytest

from docthinker.server.research_context_assist import (
    MAX_ASSIST_BYTES, build_context_assist, route_context_assist,
)
from docthinker.server.research_evidence_index import hydrate_reasoning
from docthinker.server.research_reasoning import extract_claims


def corpus(*texts):
    return [{"id": f"c{i}", "document_id": f"d{i}", "filename": f"paper-{i}.txt",
             "ordinal": 0, "content": text} for i, text in enumerate(texts)]


def refs(chunks):
    return [{**chunk, "label": f"S{i + 1}", "excerpt_start": 0, "excerpt_end": len(chunk["content"]),
             "truncated": False, "original_content_length": len(chunk["content"])} for i, chunk in enumerate(chunks)]


def assist(question, chunks, seeds=None, **kwargs):
    calls = []
    def loader(mode, sources):
        calls.append((mode, [row["id"] for row in sources]))
        return extract_claims(chunks), []
    result = build_context_assist(question, refs(chunks if seeds is None else seeds), chunks, loader, **kwargs)
    return result, calls


@pytest.mark.parametrize("question,mode", [
    ("Why does grid readiness improve security?", "deduction"),
    ("Explain security 的机制", "deduction"),
    ("分析 security 是否提高，写一段报告", "deduction"),
    ("从其他领域借鉴 grid 的设计", "analogy"),
    ("综合这些论文中 grid 的共同发现", "induction"),
    ("Write an analysis of grid security", "deduction"),
    ("Summarize grid security across papers", "induction"),
])
def test_natural_writing_intent_routes_without_a_user_mode(question, mode):
    rows = corpus("Grid supports security.")
    assert route_context_assist(question, refs(rows))["mode"] == mode


@pytest.mark.parametrize("question", [
    "hello", "今天开心吗", "What is grid?", "只引用 grid 原文", "翻译 grid improves security",
    "润色这句话：grid supports security", "Proofread grid supports security", "Quote only grid facts",
    "Analyse pineapple mechanism", "No analysis of grid please", "不要类比 grid",
])
def test_unneeded_or_unrelated_work_does_not_even_load_claims(question):
    result, calls = assist(question, corpus("Grid supports security."))
    assert calls == []
    assert result["prompt"] == ""
    assert result["trace"]["model_calls"] == 0
    assert result["trace"]["candidate_loads"] == 0


def test_no_sources_never_sends_arbitrary_corpus_to_reasoner():
    result, calls = assist("Why does security improve?", corpus("Observation: grid ready."), seeds=[])
    assert calls == []
    assert result["trace"]["skipped_reason"] == "no_grounded_sources"


def test_at_most_one_operator_for_multiple_intents_and_no_mode_argument():
    rows = corpus("Redundancy reduces outage risk. Outage risk reduces availability.")
    result, calls = assist("类比 redundancy 并归纳共同规律，再分析原因", rows)
    assert len(calls) == 1
    assert calls[0][0] == "analogy"
    assert result["reasoning"]["mode"] == "context_assist"


def test_deduction_supplies_material_and_complete_rule_sources_not_an_answer():
    rows = corpus("Observation: grid ready.", "If grid ready then security improves.")
    original = refs(rows[1:])
    result, calls = assist("Explain why security improves", rows, seeds=rows[1:])
    assert len(calls) == 1
    assert "answer" not in result
    assert result["reasoning"]["conclusions"][0]["kind"] == "deduced"
    assert result["reasoning"]["conclusions"][0]["method"] == "deduction"
    assert result["sources"][0] == original[0]
    assert [source["label"] for source in result["sources"]] == ["S1", "S2"]
    assert {source["id"] for source in result["sources"]} == {"c0", "c1"}
    assert "不是指令" in result["prompt"]
    assert "写作" in result["prompt"] and "前提" in result["prompt"]
    # Raw source paragraphs are not duplicated in the assist block.
    assert "If grid ready then security improves." not in result["prompt"]
    assert "source_chunk_characters" in result["prompt"]
    assert len(result["prompt"].encode()) <= MAX_ASSIST_BYTES
    assert result["trace"]["memory_writes"] == result["trace"]["model_calls"] == 0


def test_complete_multihop_premises_are_preserved_in_small_support_window():
    rows = corpus("Observation: grid ready. If grid ready then supply grows. "
                  "If supply grows then imports decrease. If imports decrease then security improves.")
    result, _ = assist("分析 security 是否提高", rows)
    conclusions = result["reasoning"]["conclusions"]
    assert len(conclusions) == 1
    assert conclusions[0]["depth"] == 3
    payload = json.loads(result["prompt"].split("\n")[-1])["supporting_material"][0]
    assert len(payload["premise_refs"]) == 4
    assert len(payload["steps"]) == 2
    known = {row["id"] for row in payload["premise_refs"]} | {row["id"] for row in payload["steps"]}
    assert set(payload["from"]) <= known
    for node in payload["steps"]:
        assert set(node["from"]) <= known


def test_missing_premise_is_a_writing_constraint_never_an_invented_conclusion():
    rows = corpus("Observation: grid ready. If grid ready and imports decrease then security improves.")
    result, _ = assist("Explain security mechanism", rows)
    assert result["reasoning"]["conclusions"] == []
    assert "缺少原文前提" in result["prompt"]
    assert result["sources"] == refs(rows)


def test_conflicting_premises_block_support_and_warn_the_writer():
    rows = corpus("Solar reduces imports. Solar increases imports. If solar reduces imports then security improves.")
    result, _ = assist("Explain solar security mechanism", rows)
    assert result["reasoning"]["conclusions"] == []
    assert "相互冲突" in result["prompt"]


def test_unrelated_incomplete_rule_does_not_pollute_complete_goal_explanation():
    rows = corpus("Observation: grid ready. If grid ready then security improves. "
                  "If grid ready and water decreases then harvest grows.")
    result, _ = assist("Explain why security improves", rows)
    assert result["reasoning"]["conclusions"]
    assert "missing_premises" not in {row["code"] for row in result["reasoning"]["gaps"]}
    assert "缺少原文前提" not in result["prompt"]
    assert result["trace"]["unrelated_gap_rules_omitted"] == 1


def test_two_named_goals_keep_the_incomplete_one_even_when_other_is_proved():
    rows = corpus("Observation: grid ready. If grid ready then security improves. "
                  "If grid ready and water decreases then harvest grows.")
    result, _ = assist("Explain security and harvest mechanisms", rows)
    assert result["reasoning"]["conclusions"]
    assert "缺少原文前提" in result["prompt"]
    assert result["trace"]["unrelated_gap_rules_omitted"] == 0


def test_missing_intermediate_rule_is_retained_by_backward_dependency_relevance():
    rows = corpus("Observation: grid ready. If grid ready and imports decrease then supply grows. "
                  "If supply grows then security improves. If water decreases then harvest grows.")
    result, _ = assist("Explain security mechanism", rows)
    missing = next(row for row in result["reasoning"]["gaps"] if row["code"] == "missing_premises")
    assert missing["rule_count"] == 2
    assert result["trace"]["unrelated_gap_rules_omitted"] == 1
    assert "缺少原文前提" in result["prompt"]


def test_no_named_consequent_does_not_claim_missing_rules_are_irrelevant():
    rows = corpus("Observation: grid ready. If grid ready and imports decrease then supply grows. "
                  "If water decreases then harvest grows.")
    result, _ = assist("Explain grid ready mechanism", rows)
    assert "缺少原文前提" in result["prompt"]
    assert result["trace"]["unrelated_gap_rules_omitted"] == 0


@pytest.mark.parametrize("code", ["index_window_reached", "case_budget_reached", "case_omitted_as_whole"])
def test_truncated_candidate_windows_keep_missing_premise_warnings(code):
    rows = corpus("Observation: grid ready. If grid ready then security improves. "
                  "If grid ready and water decreases then harvest grows.")
    result = build_context_assist("Explain security mechanism", refs(rows), rows,
                                  lambda *_: (extract_claims(rows), [{"code": code, "message": "Window limited"}]))
    assert "缺少原文前提" in result["prompt"]
    assert result["trace"]["unrelated_gap_rules_omitted"] == 0


def test_sampled_missing_rule_list_is_not_treated_as_all_failures():
    rows = corpus("Observation: grid ready. If grid ready then security improves. " + " ".join(
        f"If {entity} decreases then harvest{entity} grows." for entity in ("water", "rain", "sun", "snow", "wind")))
    result, _ = assist("Explain security mechanism", rows)
    missing = next(row for row in result["reasoning"]["gaps"] if row["code"] == "missing_premises")
    assert missing["rule_count"] == 5 and len(missing["rules"]) == 4
    assert result["trace"]["unrelated_gap_rules_omitted"] == 0


def test_unsupported_rule_grammar_is_not_described_as_absent_from_original():
    rows = corpus("If grid is ready under low load then supply increases under low load.")
    result, _ = assist("Explain supply mechanism", rows)
    assert "本次可解析范围内未找到完整规则" in result["prompt"]
    assert "不代表原文没有规则" in result["prompt"]
    assert "本次原文没有" not in result["prompt"]


def test_invalid_candidate_is_not_used_to_establish_gap_irrelevance():
    rows = corpus("Observation: grid ready. If grid ready then security improves. "
                  "If grid ready and water decreases then harvest grows.")
    claims = extract_claims(rows) + [{"id": "corrupt-rule", "kind": "rule"}]
    result = build_context_assist("Explain security mechanism", refs(rows), rows, lambda *_: (claims, []))
    assert result["reasoning"]["conclusions"]
    assert "缺少原文前提" in result["prompt"]
    assert result["trace"]["unrelated_gap_rules_omitted"] == 0


def test_analogy_material_includes_mapping_conditions_and_unverified_hypothesis():
    rows = corpus("Case: cloud\nRedundancy reduces outage risk. Outage risk reduces availability. "
                  "Redundancy increases availability when low load.",
                  "Case: energy\nDiversity reduces concentration risk. Concentration risk reduces security.")
    result, _ = assist("借鉴 redundancy 的类似案例，写一段分析", rows, seeds=rows[:1])
    data = json.loads(result["prompt"].split("\n")[-1])["supporting_material"][0]
    assert data["type"] == "analogy_hypothesis"
    assert data["mappings"]
    assert data["differences"]
    assert data["hypotheses"][0]["conditions"] == ["low load"]
    assert data["hypotheses"][0]["target_conditions_verified"] is False
    assert "待验证假设" in result["prompt"]


def test_induction_retains_counterexample_and_source_independence_caveats():
    rows = corpus("Case: north\nSolar reduces imports.", "Case: south\nSolar reduces imports.",
                  "Case: winter\nSolar does not reduce imports.")
    result, _ = assist("综合 solar imports 的共同发现", rows, seeds=rows[:1])
    data = json.loads(result["prompt"].split("\n")[-1])["supporting_material"][0]
    assert data["support_groups"] == 2
    assert data["counterexample_groups"] == 1
    assert data["counterexamples"]
    assert data["independence_verified"] is False
    assert data["selection_bias_excluded"] is False
    assert len(result["sources"]) == 3


def test_generic_summary_only_organizes_patterns_across_selected_documents():
    rows = corpus("Case: north\nSolar reduces imports.", "Case: south\nSolar reduces imports.")
    result, calls = assist("Summarize these papers", rows)
    assert calls[0][0] == "induction"
    assert result["reasoning"]["conclusions"]
    single, calls = assist("Summarize these papers", rows, seeds=rows[:1])
    assert calls == []
    assert single["reasoning"]["conclusions"] == []


def test_query_related_seed_prevents_unrelated_predicate_noise():
    rows = corpus("Case: north\nSolar reduces imports.", "Case: south\nSolar reduces imports.",
                  "Case: cloud\nRedundancy reduces failures.", "Case: server\nRedundancy reduces failures.")
    result, _ = assist("综合 solar imports 的共同发现", rows, seeds=rows[:1])
    assert len(result["reasoning"]["conclusions"]) == 1
    assert result["reasoning"]["conclusions"][0]["pattern"]["subject"] == "solar"
    assert result["trace"]["irrelevant_results_omitted"] == 1


def test_nonquery_pattern_in_same_source_is_not_accepted_as_relevant():
    rows = corpus("Case: north\nSolar supports policy. Redundancy reduces failures.",
                  "Case: south\nRedundancy reduces failures.")
    result, _ = assist("归纳 solar policy 的共同发现", rows, seeds=rows[:1])
    assert result["reasoning"]["conclusions"] == []
    assert result["trace"]["irrelevant_results_omitted"] == 1


def test_live_confirmed_memory_pointer_can_bridge_to_differently_named_source():
    rows = corpus("Observation: grid ready.", "If grid ready then security improves.")
    result, calls = assist("请分析项目蓝海的机制", rows, seeds=[], candidate_sources=rows[1:])
    assert calls[0][1] == ["c1"]
    assert result["reasoning"]["conclusions"]
    assert len(result["sources"]) == 2
    assert "项目蓝海" not in result["prompt"]  # Not manufactured as an axiom.


def test_original_material_is_not_evicted_for_extra_proof_sources():
    rows = corpus("Observation: grid ready.", "If grid ready then security improves.")
    result, _ = assist("Explain security mechanism", rows, seeds=rows[1:], max_sources=1)
    assert result["sources"] == refs(rows[1:])
    assert result["reasoning"]["conclusions"] == []
    assert {gap["code"] for gap in result["reasoning"]["gaps"]} & {"proof_budget_reached", "context_assist_source_budget"}


@pytest.mark.parametrize("options", [{"max_assist_bytes": 0}, {"max_assist_bytes": 800},
                                    {"max_sources": 0}, {"max_source_bytes": 1}])
def test_exhausted_budget_does_not_run_candidate_loader(options):
    rows = corpus("Observation: grid ready. If grid ready then security improves.")
    result, calls = assist("Explain security mechanism", rows, **options)
    assert calls == []
    assert result["trace"]["skipped_reason"] == "no_assist_budget"
    assert result["sources"] == refs(rows)


def test_prompt_budget_omits_whole_support_never_cuts_conditions():
    rows = corpus("Observation: grid ready. If grid ready then supply grows. "
                  "If supply grows then imports decrease. If imports decrease then security improves.")
    result, _ = assist("Explain security mechanism", rows, max_assist_bytes=900)
    assert len(result["prompt"].encode()) <= 900
    assert result["reasoning"]["conclusions"] == []
    assert result["sources"] == refs(rows)


def test_proof_source_bytes_are_bounded_together_with_original_material():
    rows = corpus("Observation: grid ready.", "If grid ready then security improves.")
    result, _ = assist("Explain security mechanism", rows, seeds=rows[1:], max_source_bytes=len(rows[1]["content"]))
    assert result["reasoning"]["conclusions"] == []
    assert result["sources"] == refs(rows[1:])


def test_loader_failure_is_one_safe_fallback_without_retry_or_private_diagnostics():
    rows = corpus("Grid improves security.")
    calls = []
    def unavailable(*args):
        calls.append(args)
        raise RuntimeError("PRIVATE_API_KEY_AND_DATABASE_PATH")
    result = build_context_assist("Explain security mechanism", refs(rows), rows, unavailable)
    assert len(calls) == 1
    assert result["sources"] == refs(rows)
    assert "PRIVATE" not in json.dumps(result)
    assert "暂不可用" in result["prompt"]


def test_revoked_source_is_never_used_even_if_loader_returned_stale_claims():
    rows = corpus("Observation: grid ready.", "If grid ready then security improves.")
    result = build_context_assist("Explain security mechanism", refs(rows[1:]), rows[1:],
                                  lambda mode, seeds: (extract_claims(rows), []))
    assert result["reasoning"]["conclusions"] == []
    assert result["sources"] == refs(rows[1:])


def test_persistable_background_metadata_hydrates_and_revokes_without_special_mode():
    rows = corpus("Observation: grid ready.", "If grid ready then security improves.")
    result, _ = assist("Explain security mechanism", rows, seeds=rows[1:])
    hydrated = hydrate_reasoning(result["reasoning"], result["sources"])
    assert hydrated["conclusions"] == result["reasoning"]["conclusions"]
    revoked = hydrate_reasoning(result["reasoning"], result["sources"][:1])
    assert revoked["conclusions"] == []
    assert revoked["status"] == "source_unavailable"


def test_helper_does_not_mutate_input_sources_claims_or_chunks():
    rows = corpus("Observation: grid ready.", "If grid ready then security improves.")
    original, claims = refs(rows[1:]), extract_claims(rows)
    snapshot = copy.deepcopy((original, rows, claims))
    build_context_assist("Explain security mechanism", original, rows, lambda *_: (claims, []))
    assert (original, rows, claims) == snapshot


def test_claim_order_does_not_change_assistance():
    rows = corpus("Observation: grid ready.", "If grid ready then security improves.")
    claims = extract_claims(rows)
    first = build_context_assist("Explain security mechanism", refs(rows[1:]), rows, lambda *_: (claims, []))
    second = build_context_assist("Explain security mechanism", refs(rows[1:]), rows, lambda *_: (list(reversed(claims)), []))
    assert first == second


def test_semantic_routing_requires_an_actual_qualified_source_and_preserves_literal_skip():
    rows = corpus("A finite active context provides a workspace for processing conversations.")
    question = "分析长期记忆怎样在新问题出现时调取资料"
    assert route_context_assist(question, refs(rows))["mode"] is None
    assert route_context_assist(question, refs(rows), semantic_scores={"c0": 0.82})["mode"] == "deduction"
    assert route_context_assist(question, refs(rows), semantic_scores={"unrelated": 0.99})["mode"] is None
    assert route_context_assist(question, refs(rows), semantic_scores={"c0": float("nan")})["mode"] is None
    assert route_context_assist("只引用原文，分析长期记忆", refs(rows), semantic_scores={"c0": 0.82})["mode"] is None


def test_natural_prose_fallback_adds_complete_conditions_without_claiming_deduction():
    rows = corpus("Adaptive storage maintains a finite active buffer for document analysis.",
                  "Adaptive storage evicts inactive records to a durable store. "
                  "However, deleted records cannot be retrieved from that store.")
    result, calls = assist("Explain adaptive storage limitations", rows, seeds=rows[:1])
    assert len(calls) == 1
    assert result["reasoning"]["status"] == "source_organized"
    assert result["trace"]["fallback_kind"] == "source_evidence"
    assert all(group["kind"] == "source_evidence" for group in result["reasoning"]["conclusions"])
    assert "deleted records cannot" in result["sources"][1]["content"]
    assert "no_explicit_rules" not in {gap["code"] for gap in result["reasoning"]["gaps"]}
    assert "未找到完整规则" not in result["prompt"]
    assert result["trace"]["memory_writes"] == result["trace"]["model_calls"] == 0
    assert result["trace"]["organizer_trace"]["added_sentence_count"] >= 1


def test_semantic_cross_language_routing_reaches_source_organization():
    rows = corpus("A finite active context provides a workspace for processing conversations.",
                  "The scheduler retrieves archived records into the workspace when a new request arrives. "
                  "However, deleted records cannot be retrieved from the archive.",
                  "The scheduler retrieves advertising records for auction pricing.")
    result, _ = assist("分析长期记忆怎样调取资料，有什么限制", rows, seeds=rows[:1],
                       semantic_scores={"c0": 0.84, "c1": 0.81},
                       semantic_spans={row["id"]: [0, len(row["content"])] for row in rows[:2]})
    assert result["reasoning"]["status"] == "source_organized"
    assert {row["id"] for row in result["sources"]} == {"c0", "c1"}
    assert result["trace"]["organizer_trace"]["lexical_anchor_unavailable"] is True
    assert not result["trace"]["semantic_entailment_verified"]


def test_semantic_locations_are_forwarded_and_missing_locations_do_not_expand():
    rows = corpus("A finite active context provides a workspace for processing conversations.",
                  "The scheduler retrieves archived records into the active workspace.")
    result, _ = assist("分析长期记忆如何调取资料", rows, seeds=rows[:1],
                       semantic_scores={"c0": 0.85, "c1": 0.83})
    assert result["sources"] == refs(rows[:1])
    assert not result["reasoning"]["conclusions"]
    assert result["trace"]["organizer_trace"]["semantic_span_count"] == 0


def test_expanded_antecedent_above_retrieval_window_still_hydrates_and_revokes():
    previous = "The earlier comparison system uses a vector index with " + "partitioned archival records " * 10 + ". "
    visible = "Queries are then embedded into the same vector space for " + "nearest neighbor retrieval " * 28 + "."
    rows = corpus(previous + visible)
    original = [{**refs(rows)[0], "content": visible, "excerpt_start": len(previous),
                 "excerpt_end": len(rows[0]["content"]), "truncated": True}]
    result = build_context_assist("Explain vector nearest neighbor retrieval", original, rows,
                                  lambda *_: (extract_claims(rows), []))
    assert result["reasoning"]["status"] == "source_organized"
    assert 1000 < len(result["sources"][0]["content"]) <= 1600
    assert result["sources"][0]["content"] == rows[0]["content"]
    hydrated = hydrate_reasoning(result["reasoning"], result["sources"])
    assert hydrated["conclusions"]
    assert all(row["epistemic_status"] == "source_organization" for row in hydrated["conclusions"])
    revoked = hydrate_reasoning(result["reasoning"], [])
    assert revoked["conclusions"] == []
    assert revoked["status"] == "source_unavailable"


def test_fallback_preserves_six_baseline_sources_and_backfills_with_unique_labels():
    rows = corpus(*["Adaptive storage archives records for workspace handling." for i in range(8)],
                  "Adaptive storage retrieves archived records with an index. "
                  "However, adaptive storage cannot recover records that were deleted from the archive.")
    for row in rows:
        row["document_id"] = "same-paper"
    original = refs(rows[:8])
    result, _ = assist("Explain adaptive storage retrieval limitations", rows, seeds=rows[:8])
    assert result["reasoning"]["status"] == "source_organized"
    assert result["sources"][:6] == original[:6]
    assert len(result["sources"]) == len({row["label"] for row in result["sources"]}) == 8
    assert result["trace"]["baseline_source_count"] == 8
    assert result["trace"]["protected_source_count"] == 6
    assert result["trace"]["added_source_count"] == result["trace"]["replaced_source_count"] == 1
    assert result["trace"]["new_source_span_count"] >= 1
    assert result["sources"][6]["id"] == "c8"
    assert result["sources"][7]["id"] == "c6"
    assert result["sources"][7]["label"] == "S8"
    data = json.loads(result["prompt"].split("\n")[-1])
    assert all(ref["source"] == "S7" for item in data["supporting_material"] for ref in item["source_spans"])
    assert sum(len(row["content"].encode()) for row in result["sources"]) <= 12000


def test_no_real_new_material_preserves_all_eight_baseline_sources_and_no_generic_hint():
    rows = corpus(*[f"Adaptive storage archives records for workspace handling in deployment number {i}." for i in range(8)])
    result, _ = assist("Explain adaptive storage retrieval limitations", rows)
    assert result["sources"] == refs(rows)
    assert result["reasoning"]["conclusions"] == []
    assert result["prompt"] == ""
    assert result["trace"]["organizer_trace"]["support_item_count"] == 0


def test_organized_source_spans_hydrate_and_revoke_with_original_provenance_checks():
    rows = corpus("Adaptive storage maintains a finite active buffer for documents.",
                  "Adaptive storage retrieves archived records through explicit index operations.")
    result, _ = assist("Explain adaptive storage retrieval", rows, seeds=rows[:1])
    assert result["reasoning"]["status"] == "source_organized"
    hydrated = hydrate_reasoning(result["reasoning"], result["sources"])
    assert [{key: value for key, value in item.items() if key != "proof"}
            for item in hydrated["conclusions"]] == result["reasoning"]["conclusions"]
    assert all(not item.get("proof") for item in hydrated["conclusions"])
    revoked = hydrate_reasoning(result["reasoning"], result["sources"][:1])
    assert revoked["conclusions"] == []
    assert revoked["status"] == "source_unavailable"


def test_fallback_budget_failure_does_not_drop_original_or_shorten_condition():
    rows = corpus("Adaptive storage maintains a finite active buffer for documents.",
                  "Adaptive storage retrieves archived records with an index, "
                  "but cannot recover records after their original source has been deleted.")
    original = refs(rows[:1])
    result, _ = assist("Explain adaptive storage retrieval limitations", rows, seeds=rows[:1],
                       max_source_bytes=len(original[0]["content"].encode()))
    assert result["sources"] == original
    assert result["reasoning"]["conclusions"] == []
    assert result["trace"]["organizer_trace"]["support_item_count"] == 0


def test_complete_formal_support_does_not_run_the_natural_evidence_fallback(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("A complete formal result must not start a second evidence expansion")
    monkeypatch.setattr("docthinker.server.research_context_assist.organize_source_evidence", unexpected)
    rows = corpus("Observation: grid ready.", "If grid ready then security improves.")
    result, _ = assist("Explain security mechanism", rows, seeds=rows[1:])
    assert result["reasoning"]["conclusions"][0]["kind"] == "deduced"


def test_natural_fallback_does_not_erase_an_unfinished_formal_premise_warning():
    rows = corpus("Observation: grid ready. If grid ready and imports decrease then security improves.",
                  "Grid security improves with managed reserves in our evaluation. "
                  "However, this outcome was measured only under low demand.")
    result, _ = assist("Explain grid security mechanism", rows, seeds=rows[:1])
    assert result["reasoning"]["status"] == "source_organized"
    assert "missing_premises" in {row["code"] for row in result["reasoning"]["gaps"]}
    assert "缺少原文前提" in result["prompt"]
    assert all(item["kind"] == "source_evidence" for item in result["reasoning"]["conclusions"])


def test_replacement_cannot_remove_the_only_viewpoint_of_a_document():
    rows = corpus(*["Adaptive storage maintains records in the working buffer." for _ in range(6)],
                  "Adaptive storage fails when the retrieval index is stale.",
                  "Adaptive storage indexes records before serving each retrieval request.",
                  "Adaptive storage improves retrieval performance in our evaluation.")
    for row in rows[:6] + rows[7:]:
        row["document_id"] = "main-paper"
    rows[6]["document_id"] = "counterexample-paper"
    result, _ = assist("Explain adaptive storage retrieval performance", rows, seeds=rows[:8])
    assert result["sources"] == refs(rows[:8])
    assert result["reasoning"]["conclusions"] == []
    assert result["trace"]["replacement_guard"]["status"] == "rejected"


def test_same_document_summary_cannot_displace_a_unique_experimental_condition():
    rows = corpus(*["Adaptive storage maintains records in the working buffer." for _ in range(5)],
                  "Adaptive storage was examined in a separate experimental study.",
                  "Adaptive storage performance decreases when the relevant record appears between distractor records.",
                  "Adaptive storage uses a persistent archive for its records.",
                  "Adaptive storage improves the ability to maintain a large virtual workspace. "
                  "However, a finite working buffer limits the active records.")
    for row in rows:
        row["document_id"] = "architecture-paper"
    rows[5]["document_id"] = rows[6]["document_id"] = "experimental-paper"
    result, _ = assist("Explain adaptive storage performance", rows, seeds=rows[:8])
    assert result["sources"] == refs(rows[:8])
    assert result["trace"]["replacement_guard"]["status"] == "rejected"
    assert result["reasoning"]["conclusions"] == []


def test_generic_summary_cannot_displace_the_explicitly_requested_procedure():
    rows = corpus(*["Adaptive storage maintains records in the working buffer." for _ in range(6)],
                  "Function chaining retrieves archive pages sequentially before returning control to the user.",
                  "Adaptive storage uses a persistent archive for its records.",
                  "Adaptive storage improves the ability to maintain a large virtual workspace. "
                  "However, a finite working buffer limits the active records.")
    for row in rows:
        row["document_id"] = "same-paper"
    result, _ = assist("Explain adaptive storage function chaining", rows, seeds=rows[:8])
    assert result["sources"] == refs(rows[:8])
    assert result["trace"]["replacement_guard"]["status"] == "rejected"
    assert result["reasoning"]["conclusions"] == []


def test_rejected_replacement_can_still_complete_a_clipped_existing_source_at_full_cap():
    first = "Adaptive storage improves retrieval performance, but only under low demand."
    rows = corpus(first, *[f"Adaptive storage maintains records for deployment {i}." for i in range(7)],
                  "Adaptive storage uses a durable archive for retrieval operations. "
                  "However, deleted records cannot be recovered from the archive.")
    original = refs(rows[:8])
    original[0].update(content=first[:first.index(",")], excerpt_end=first.index(","), truncated=True)
    result = build_context_assist("Explain adaptive storage retrieval limitations", original, rows,
                                  lambda *_: (extract_claims(rows), []))
    assert result["reasoning"]["status"] == "source_organized"
    assert len(result["sources"]) == 8
    assert {row["id"] for row in result["sources"]} == {row["id"] for row in original}
    assert result["sources"][0]["content"] == first
    assert result["trace"]["replaced_source_count"] == 0
    assert result["trace"]["expanded_source_count"] == 1
