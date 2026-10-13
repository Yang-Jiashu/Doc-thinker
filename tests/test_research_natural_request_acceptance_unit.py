"""Independent literal cross-language request coverage, not answer-quality evals.

All documents are synthetic. No production benchmark question, paper, answer,
translation model or provider is consulted by these acceptance cases.
"""

from copy import deepcopy

import pytest

from docthinker.server.research_query_facets import build_query_facets
from docthinker.server.research_lexical_coverage import select_lexical_coverage
from docthinker.server.research_retrieval import build_retrieval_plan, refresh_query_coverage
from docthinker.server.research_evidence_selection import prepare_evidence_candidates
from docthinker.server.research_sentence_windows import evidence_units, opening_reference_unverified


def chunk(identifier, document, filename, text, ordinal=0):
    return {"id": identifier, "document_id": document, "filename": filename,
            "content": text, "ordinal": ordinal, "page": ordinal + 1}


def doc_ids(result):
    return {identifier for facet in result["facets"]
            for identifier in facet.get("document_ids", ())}


def assert_literal_facets(result, question):
    assert len(result["facets"]) <= 8
    assert all(facet["text"] in question for facet in result["facets"])
    assert result["trace"]["provider_calls"] == result["trace"]["llm_tokens"] == 0
    assert result["trace"]["generated_expansions"] is False


@pytest.mark.parametrize("filename,name", [
    ("quartzcache_2511.45678v2.pdf", "QuartzCache"),
    ("QuartzCache-2502.12345.pdf", "QuartzCache"),
    ("terraledger_2026-08-21.pdf", "TerraLedger"),
    ("archive/terraledger_2026-08-21.txt", "TerraLedger"),
])
def test_exact_technical_filename_prefix_with_recognized_suffix_is_only_a_scope_hint(filename, name):
    question = f"请说明 {name} 如何工作，以及实际验证了什么？"
    rows = [chunk("c", "target", filename, "Do not convert this hidden body phrase into a query.")]
    saved = deepcopy(rows)
    result = build_query_facets(question, rows)
    assert doc_ids(result) == {"target"}
    assert_literal_facets(result, question)
    assert "hidden body phrase" not in str(result)
    assert rows == saved


@pytest.mark.parametrize("filename,question", [
    ("map_2502.12345.pdf", "Explain map accuracy and evaluation conditions."),
    ("art_2026-08-21.pdf", "Explain art classification experiments."),
    ("cache_2502.12345.pdf", "Explain cache behavior and evaluation conditions."),
    ("plan_2026-08-21.pdf", "Explain the product plan and evaluation conditions."),
    ("quartzcache_2502.12345.pdf", "请说明 Quartz 的方法和实验。"),
    ("quartzcache_2502.12345.pdf", "Explain SuperQuartzCache methods and experiments."),
    ("QuartzCache_internal_notes_final.pdf", "请说明 QuartzCache 的方法和实验。"),
])
def test_short_words_substrings_or_arbitrary_filename_suffixes_do_not_invent_aliases(filename, question):
    rows = [chunk("c", "target", filename, "Synthetic method and experiment text.")]
    result = build_query_facets(question, rows)
    assert doc_ids(result) == set()
    assert_literal_facets(result, question)


def test_ambiguous_technical_prefix_never_picks_an_arbitrary_single_document():
    question = "请说明 QuartzCache 的方法和实验。"
    rows = [chunk("a", "earlier", "quartzcache_2502.12345v1.pdf", "Earlier document."),
            chunk("b", "later", "quartzcache_2511.45678v2.pdf", "Different document.")]
    result = build_query_facets(question, rows)
    assert doc_ids(result) in (set(), {"earlier", "later"})
    assert build_query_facets(question, rows[::-1]) == result
    assert_literal_facets(result, question)


def test_no_named_scope_is_manufactured_from_document_body_or_generic_request():
    question = "请基于这些材料说明系统如何工作，以及实际验证了什么。"
    rows = [chunk("a", "q", "quartzcache_2502.12345.pdf", "These materials explain QuartzCache."),
            chunk("b", "t", "terraledger_2026-08-21.pdf", "These materials explain TerraLedger.")]
    result = build_query_facets(question, rows)
    assert not doc_ids(result)
    assert_literal_facets(result, question)


def test_explicit_content_subquestions_have_priority_over_many_background_clauses():
    background = ["我们的工作台准备更新", "页面风格已经确定", "读者希望更容易阅读",
                  "材料用于产品讨论", "内部文档已经整理", "展示布局稍后处理",
                  "团队会再讨论命名", "颜色可以以后再定", "图标仅供参考"]
    needs = ["QuartzCache 如何保存记录", "TerraLedger 的查询流程是什么",
             "两者实际验证了哪些能力", "只在离线数据可用时，比较失败条件，不要把未验证当作失败"]
    question = "。".join([*background, *needs]) + "。"
    rows = [chunk("q", "q", "quartzcache_2502.12345.pdf", "Private source vocabulary must not become a query."),
            chunk("t", "t", "terraledger_2026-08-21.pdf", "Other private source vocabulary.")]
    result = build_query_facets(question, rows)
    clauses = [facet["text"] for facet in result["facets"] if facet["kind"] == "clause"]
    assert all(any(need in clause for clause in clauses) for need in needs)
    assert all(facet["text"] != "未验证当作失败" for facet in result["facets"])
    assert_literal_facets(result, question)


@pytest.mark.parametrize("question,guard,detached", [
    ("只有在原始记录仍可访问时，说明 QuartzCache 的验证情况，不要把未验证说成失败。",
     "只有在原始记录仍可访问时", "说明 QuartzCache 的验证情况"),
    ("Only under independent sampling. Explain how QuartzCache stores its index. Describe its evaluation conditions.",
     "Only under independent sampling", "Describe its evaluation conditions"),
    ("Unless the workload changes, QuartzCache uses a frozen index. Explain its evaluation conditions.",
     "Unless the workload changes", "Explain its evaluation conditions"),
])
def test_priority_does_not_strip_conditions_negation_or_bound_referential_scope(question, guard, detached):
    result = build_query_facets(question)
    derived_clauses = [facet["text"] for facet in result["facets"] if facet["kind"] == "clause"]
    assert all(guard in clause for clause in derived_clauses)
    assert not any(facet["text"].strip(".。") == detached for facet in result["facets"])
    assert not any(facet["kind"] == "technical_phrase" for facet in result["facets"])
    assert_literal_facets(result, question)


@pytest.mark.parametrize("question", [
    "QuartzCache 实际验证了什么？",
    "QuartzCache 的验证情况如何？",
    "What did QuartzCache empirically test?",
    "What was empirically tested in QuartzCache?",
])
def test_verification_language_requests_experimental_material_not_an_entailment_verdict(question):
    rows = [chunk("q", "q", "quartzcache_2502.12345.pdf", "This source claims a result without providing evidence.")]
    result = build_query_facets(question, rows)
    assert any("experiment_setup" in facet["roles"] for facet in result["facets"])
    assert_literal_facets(result, question)
    assert "claims a result" not in str(result)
    assert all(not facet.get("entailment_verified", False) for facet in result["facets"])


def synthetic_corpus():
    rows = []
    specs = [
        ("quartz", "quartzcache_2502.12345v2.pdf",
         "2 Architecture\nThe access buffer stores validated records and the background table keeps immutable identifiers. "
         "A lookup first reads the buffer, then retrieves matching table entries. This operation does not alter the original records.",
         "3 Experiments\nWe evaluated the design on 128 held-out requests with a fixed keyword baseline. "
         "We observed an accuracy increase only under the stationary workload; no conclusion is reported for changed workloads."),
        ("terra", "terraledger_2026-08-21.pdf",
         "2 Method\nThe routing layer stores timestamps and uses a separate index to retrieve complete records. "
         "The lookup combines the query with permitted entries; rejected entries are never loaded.",
         "4 Evaluation\nA controlled experiment used 96 requests and compared two fixed load settings. "
         "Results show reduced latency in the low-load setting only. The study did not test concurrent edits."),
    ]
    for document, filename, method, experiment in specs:
        rows.extend([chunk(document + "-method", document, filename, method, 1),
                     chunk(document + "-experiment", document, filename, experiment, 2)])
        for index in range(12):
            # Repeated names and product words must not substitute for actual
            # method/experiment passages that omit the method name entirely.
            rows.append(chunk(f"{document}-background-{index}", document, filename,
                              f"1 Introduction\nQuartzCache TerraLedger 产品路线 工作台 系统设计 功能。 "
                              f"Background note {index} discusses the popularity of research products.", index + 3))
    return rows


def assert_exact_bounded_sources(result, rows):
    assert 0 < len(result["sources"]) <= 8
    assert sum(len(row["content"].encode("utf-8")) for row in result["sources"]) <= 12000
    assert result["trace"]["candidate_limit"] == 48
    assert result["trace"].get("candidate_count", 0) <= 48
    assert result["trace"]["model_calls"] == result["trace"]["provider_calls"] == result["trace"]["llm_tokens"] == 0
    assert result["trace"]["semantic_entailment_verified"] is False
    assert result["trace"]["selection"]["document_scope_is_hard_filter"] is False
    assert result["trace"]["selection"]["selection_proxy_is_verified_support"] is False
    originals = {row["id"]: row for row in rows}
    for selected in result["sources"]:
        original = originals[selected["id"]]
        assert original["document_id"] == selected["document_id"]
        assert original["filename"] == selected["filename"]
        assert original["content"][selected["excerpt_start"]:selected["excerpt_end"]] == selected["content"]


def test_chinese_product_request_recovers_each_named_english_method_and_actual_experiment():
    rows = synthetic_corpus()
    original = deepcopy(rows)
    question = ("请基于 QuartzCache 和 TerraLedger 给工作台提供产品路线建议。"
                "这两个系统如何保存和查询记录？论文实际验证了哪些能力，结果有哪些适用条件？")
    background = [row for row in rows if "-background-" in row["id"]]
    baseline = build_retrieval_plan(question, background)["sources"]
    result = select_lexical_coverage(question, baseline, rows)
    selected = {row["id"]: row["content"] for row in result["sources"]}
    assert {"quartz-method", "quartz-experiment", "terra-method", "terra-experiment"} <= set(selected)
    assert "does not alter the original records" in selected["quartz-method"]
    assert "only under the stationary workload" in selected["quartz-experiment"]
    assert "never loaded" in selected["terra-method"]
    assert "did not test concurrent edits" in selected["terra-experiment"]
    assert_exact_bounded_sources(result, rows)
    assert rows == original


def test_named_document_coverage_is_not_a_hard_filter_for_other_relevant_material():
    rows = synthetic_corpus()
    rows.append(chunk("third-experiment", "other", "DeltaAudit.pdf",
                      "3 Controlled Evaluation\nQuartzCache TerraLedger comparison uses 256 requests with identical timestamps. "
                      "The experiment compares both routing mechanisms against the same baseline. "
                      "Results show no improvement when the load changes."))
    question = "比较 QuartzCache 和 TerraLedger 的方法、实际验证结果及适用条件。"
    baseline = build_retrieval_plan(question, rows[:2])["sources"]
    result = select_lexical_coverage(question, baseline, rows)
    assert "third-experiment" in {row["id"] for row in result["sources"]}
    assert_exact_bounded_sources(result, rows)


def test_numbered_role_hint_is_local_to_document_and_current_parent_not_old_method_sibling():
    rows = [
        chunk("parent", "orchid", "OrchidIndex.pdf", "2 Methods\nEntry storage keeps immutable keys.", 0),
        chunk("child", "orchid", "OrchidIndex.pdf", "2.1 Buffer Layout\nEntry storage keeps offsets in a separate buffer.", 1),
        chunk("child-study", "orchid", "OrchidIndex.pdf", "2.2 Experiments\nEntry storage was evaluated against a fixed baseline.", 2),
        chunk("new-root", "orchid", "OrchidIndex.pdf", "3 Discussion\nEntry storage has several possible applications.", 3),
        chunk("new-child", "orchid", "OrchidIndex.pdf", "3.1 Error Catalog\nEntry storage rejects malformed records.", 4),
        chunk("other-child", "ember", "EmberStore.pdf", "2.1 Buffer Layout\nEntry storage keeps immutable snapshots.", 0),
    ]
    question = "Explain entry storage and evaluation conditions."
    saved = deepcopy(rows)
    pool = prepare_evidence_candidates(question, rows, [], build_query_facets(question, rows), {})
    by_source = {row["source_id"]: row for row in pool["candidates"]}
    assert set(by_source) == {row["id"] for row in rows}
    assert by_source["child"]["outline_roles"] == ["mechanism"]
    assert by_source["child-study"]["outline_roles"] == ["experiment_setup"]
    assert all("mechanism" not in by_source[key]["outline_roles"]
               for key in ("new-root", "new-child", "other-child"))
    assert len(pool["candidates"]) <= 48
    originals = {row["id"]: row["content"] for row in rows}
    for selected in pool["candidates"]:
        assert originals[selected["source_id"]][selected["excerpt_start"]:selected["excerpt_end"]] == selected["content"]
    assert rows == saved


def test_unnumbered_evaluation_resets_parent_before_following_chunk_without_heading():
    rows = [
        chunk("parent", "orchid", "OrchidIndex.pdf", "2 Methods\nEntry storage preserves complete records.", 0),
        chunk("study", "orchid", "OrchidIndex.pdf", "Evaluation\nEntry storage was evaluated against a fixed baseline.", 1),
        chunk("continuation", "orchid", "OrchidIndex.pdf", "Entry storage was tested on 72 held-out requests, not changing workloads.", 2),
    ]
    question = "Explain entry storage and evaluation conditions."
    pool = prepare_evidence_candidates(question, rows, [], build_query_facets(question, rows), {})
    by_source = {row["source_id"]: row for row in pool["candidates"]}
    for key in ("study", "continuation"):
        assert by_source[key]["outline_roles"] == ["experiment_setup"]
        assert not by_source[key]["outline_path"]
    assert "not changing workloads" in by_source["continuation"]["content"]


@pytest.mark.parametrize("antecedent,continuation", [
    ("The trial enrolled 42 participants for six days.",
     "These participants completed one fixed questionnaire each day."),
    ("The laboratory produced two ceramic materials.",
     "Those materials remained intact only in the low-temperature chamber."),
    ("The structure contains three balanced trees.",
     "These trees are traversed without changing their original branches."),
    ("The archive retained 72 authorized records.",
     "Such records were never transmitted outside the workspace."),
])
def test_unfamiliar_demonstrative_nouns_keep_the_whole_adjacent_antecedent(antecedent, continuation):
    text = antecedent + "\n" + continuation
    assert evidence_units(text) == [(0, len(text))]
    assert opening_reference_unverified(continuation)
    assert text[evidence_units(text)[0][0]:evidence_units(text)[0][1]] == text


@pytest.mark.parametrize("text", [
    "This is a definition of the bounded archive.",
    "These are independently defined storage classes.",
    "That was the stated objective.",
    "This section presents the experimental protocol.",
    "It is important to retain the measurement conditions.",
    "These materials named Orchid and Ember were compared under matched conditions.",
])
def test_standalone_copular_or_locally_named_statements_are_not_nominal_gap_warnings(text):
    assert opening_reference_unverified(text) is False


@pytest.mark.parametrize("noun", ["participants", "materials", "trees", "records"])
def test_source_start_missing_antecedent_is_preserved_but_explicitly_unresolved(noun):
    text = f"These {noun} were tested only under fixed conditions; no other setting was measured."
    original = chunk("start", "orchid", "OrchidIndex.pdf", text)
    saved = deepcopy(original)
    plan = build_retrieval_plan(f"Explain {noun} and experimental conditions.", [original])
    assert plan["sources"] and plan["sources"][0]["content"] == text
    assert any(gap["code"] == "source_antecedent_unverified" for gap in plan["gaps"])
    refreshed = refresh_query_coverage({"gaps": [], "query_terms": []}, plan["sources"])
    assert any(gap["code"] == "source_antecedent_unverified" for gap in refreshed["gaps"])
    assert original == saved


@pytest.mark.parametrize("boundary", [
    "\n4 Evaluation\n",
    "\n## Separate Evaluation\n",
    "\nFigure 6: Distribution of sampled objects.\n",
])
def test_demonstrative_binding_cannot_cross_heading_or_figure_caption(boundary):
    antecedent = "The trial enrolled 42 participants for six days."
    continuation = "These participants completed the fixed questionnaire."
    text = antecedent + boundary + continuation
    parts = [text[left:right] for left, right in evidence_units(text)]
    containing = [part for part in parts if continuation in part]
    assert len(containing) == 1
    assert antecedent not in containing[0]
    assert boundary.strip() not in containing[0]
    assert opening_reference_unverified(containing[0])


def test_filtered_bibliographic_gap_cannot_be_silently_stitched_into_a_contiguous_study_quote():
    first = "The trial enrolled 42 participants for six days."
    bibliography = "[12] Hale et al., 2024, Controlled participant questionnaires, Journal of Measurement."
    continuation = "These participants completed the fixed questionnaire."
    text = first + "\n" + bibliography + "\n" + continuation
    row = chunk("gap", "orchid", "OrchidIndex.pdf", text)
    question = "Explain the participants and questionnaire protocol."
    pool = prepare_evidence_candidates(question, [row], [], build_query_facets(question, [row]), {})
    assert pool["trace"]["bibliographic_units_filtered"] > 0
    for selected in pool["candidates"]:
        assert selected["content"] == text[selected["excerpt_start"]:selected["excerpt_end"]]
        assert bibliography not in selected["content"]
        assert not (first in selected["content"] and continuation in selected["content"])


def test_over_budget_bound_demonstrative_unit_is_skipped_whole_and_omission_is_disclosed():
    antecedent = "The experiment collected " + "carefully labeled materials " * 65 + "."
    continuation = " These materials were tested only under dry conditions; wet-condition effects were not measured."
    text = "2 Experiments\n" + antecedent + continuation
    row = chunk("long", "orchid", "OrchidIndex.pdf", text)
    question = "Explain OrchidIndex materials and experimental conditions."
    assert len(antecedent + continuation) > 1600
    assert any(text[left:right].strip() == antecedent + continuation for left, right in evidence_units(text))
    pool = prepare_evidence_candidates(question, [row], [], build_query_facets(question, [row]), {})
    assert pool["candidates"] == []
    plan = build_retrieval_plan(question, [row])
    assert plan["sources"] == []
    assert any(gap["code"] == "complete_statement_exceeds_budget" for gap in plan["gaps"])
    assert row["content"] == text
