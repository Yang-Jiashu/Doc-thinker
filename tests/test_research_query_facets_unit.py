"""Facets are literal user intent, not paper-specific answer expansions."""

import json

import pytest

from docthinker.server.research_query_facets import (
    MAX_FACETS, MAX_QUERY_BYTES, build_query_facets,
)


def row(document_id, filename, **extra):
    return {"id": document_id + "-chunk", "document_id": document_id,
            "filename": filename, "content": "Never turn this source into a query.", **extra}


def texts(result, kind=None):
    return [facet["text"] for facet in result["facets"] if kind is None or facet["kind"] == kind]


def test_chinese_english_mixed_clauses_and_literal_technical_phrases():
    query = "请帮我解释 flux-aware scheduling 如何工作？实验怎么设置，结果是什么？有什么局限？"
    result = build_query_facets(query)
    assert texts(result)[0] == query
    assert "flux-aware scheduling" in texts(result, "technical_phrase")
    assert "实验怎么设置" in texts(result, "clause")
    assert "结果是什么" in texts(result, "clause")
    assert any("experiment_setup" in facet["roles"] for facet in result["facets"])
    assert any("limitation" in facet["roles"] for facet in result["facets"])
    assert all(facet["text"] in query for facet in result["facets"])
    assert result["trace"]["provider_calls"] == result["trace"]["llm_tokens"] == 0


@pytest.mark.parametrize("query", [
    "如果数据分布不变，实验结果是否提升，为什么？",
    "只有低湿度条件下，flux-aware scheduling 才有效吗？",
    "Explain flux-aware scheduling only if the boundary is closed, how does it fail?",
    "Do not equate correlation with causality, why does the estimate improve?",
    "Without replicated samples, what evidence supports the claim?",
    "Only when the sample is independent; how does flux-aware scheduling behave?",
    "仅当输入独立；flux-aware scheduling 才有效吗？",
    "Do results contradict flux-aware scheduling, how shouldn't it be applied?",
])
def test_negation_and_conditions_not_removed_to_get_shorter_queries(query):
    result = build_query_facets(query)
    assert result["trace"]["protected_clause_count"] >= 1
    assert texts(result, "technical_phrase") == []
    assert all("flux-aware scheduling" != value for value in texts(result))
    assert len(result["facets"]) <= 2


def test_english_safe_interrogative_conjunctions_split():
    query = "How does the zephyr index work and what experiments measure drift? Why can it fail?"
    result = build_query_facets(query)
    assert "How does the zephyr index work" in texts(result, "clause")
    assert "what experiments measure drift" in texts(result, "clause")
    assert all(facet["text"] in query for facet in result["facets"])


def test_english_commands_split_without_breaking_decimal_or_initialism():
    query = "Explain transport at 0.5 Hz, e.g. where the valve opens. Describe limitations. Identify controls."
    result = build_query_facets(query)
    assert "Explain transport at 0.5 Hz, e.g. where the valve opens" in texts(result, "clause")
    assert "Describe limitations" in texts(result, "clause")
    assert "Identify controls" in texts(result, "clause")
    assert not any(value.startswith("where the valve") for value in texts(result, "clause"))


def test_english_period_does_not_detach_a_conditional_scope():
    query = "Only under independent sampling. Explain transport. Describe limitations."
    result = build_query_facets(query)
    assert result["trace"]["protected_clause_count"] == 1
    assert not any(value in {"Explain transport", "Describe limitations"} for value in texts(result))
    assert texts(result, "technical_phrase") == []


def test_explicit_quoted_user_technical_phrase_not_invented_from_source():
    query = 'Compare "oscillation binding" with "local channel indexing".'
    result = build_query_facets(query, [row("a", "Example.pdf", content="invented extra search words")])
    assert "oscillation binding" in texts(result, "technical_phrase")
    assert "local channel indexing" in texts(result, "technical_phrase")
    assert "invented extra search words" not in json.dumps(result)


def test_document_mentions_match_functionword_normalization_and_cover_ids():
    query = "比较 Signals in the Valley 与 Frost Ledger 的方法和实验。"
    rows = [row("a", "signals_valley.pdf"), row("b", "frost-ledger.pdf"),
            row("c", "unmentioned.pdf")]
    result = build_query_facets(query, rows)
    assert result["facets"][0]["document_ids"] == ["a", "b"]
    documents = [facet for facet in result["facets"] if facet["kind"] == "document"]
    assert {facet["text"] for facet in documents} == {"Signals in the Valley", "Frost Ledger"}
    assert not any(value in texts(result, "technical_phrase") for value in ["Signals in the Valley", "Frost Ledger"])
    assert result["trace"]["matched_document_count"] == 2


def test_no_guessed_acronym_or_partial_english_name_match():
    rows = [row("a", "Valley Gradient Map.pdf"), row("b", "art.pdf"), row("c", "of.pdf")]
    result = build_query_facets("What does VGM imply for cartography?", rows)
    assert result["trace"]["matched_document_count"] == 0
    assert "document_ids" not in result["facets"][0]


def test_title_metadata_and_chinese_embedded_name_match():
    rows = [row("a", "download-2024.pdf", title="星河索引"),
            row("b", "other.pdf", document_title="Frozen Meadows")]
    result = build_query_facets("解释星河索引的实验；对比 Frozen Meadows 的结果", rows)
    assert result["facets"][0]["document_ids"] == ["a", "b"]
    assert {"星河索引", "Frozen Meadows"} <= set(texts(result, "document"))


def test_repeat_chunks_and_input_order_do_not_change_facets():
    query = "Compare Crimson Atlas and Glacial Loom: how are they evaluated?"
    rows = [row("b", "Glacial_Loom.pdf"), row("a", "Crimson_Atlas.pdf")]
    assert build_query_facets(query, rows) == build_query_facets(query, rows[::-1] * 10)


def test_trace_never_contains_question_or_metadata_values():
    query = "user-private-question-4928 如何做实验？"
    result = build_query_facets(query, [row("private-document-345", "Confidential.pdf")])
    trace = json.dumps(result["trace"])
    assert "private" not in trace and "Confidential" not in trace and "4928" not in trace
    assert result["trace"]["generated_expansions"] is False


def test_bounded_utf8_query_and_facet_count_without_source_expansion():
    query = "怎么做？" * 1000 + "界" * 4000
    result = build_query_facets(query)
    assert len(result["facets"]) <= MAX_FACETS
    assert result["trace"]["query_truncated"] is True
    assert len(result["facets"][0]["text"].encode("utf-8")) <= MAX_QUERY_BYTES
    assert "\ufffd" not in result["facets"][0]["text"]
    assert all(facet["text"] in query for facet in result["facets"])


def test_limits_many_different_subquestions():
    result = build_query_facets("；".join(f"实验{i}有什么结果" for i in range(50)))
    assert len(result["facets"]) == MAX_FACETS
    assert len({facet["id"] for facet in result["facets"]}) == MAX_FACETS


@pytest.mark.parametrize("question", [None, {}, "", "  ", "\ud800"])
def test_empty_or_invalid_question_is_safe(question):
    result = build_query_facets(question)
    assert result["facets"] == []
    assert result["trace"]["status"] == "empty_query"


def test_metadata_fields_are_validated_and_arbitrary_content_is_ignored():
    result = build_query_facets("How is Bright Moss evaluated?", [
        None, {"document_id": []}, row("a", "Bright_Moss.pdf", title={}),
        row("b", "Elsewhere.pdf", content="Bright Moss generated unsupported expansion"),
    ])
    assert result["facets"][0]["document_ids"] == ["a"]


def test_single_plain_question_deduplicates_punctuation_only_clause():
    result = build_query_facets("How does the cold index work?")
    assert texts(result) == ["How does the cold index work?"]


def test_chinese_how_and_variable_control_intents_are_generic():
    result = build_query_facets("冷雾索引怎样工作，怎样检查控制变量的影响？")
    assert "冷雾索引怎样工作" in texts(result, "clause")
    assert any("mechanism" in facet["roles"] for facet in result["facets"])
    assert any("experiment_setup" in facet["roles"] for facet in result["facets"])


def test_long_multi_need_task_retains_tail_anchors_without_meta_facets():
    concepts = [f"{color}-gate scheduling" for color in ("amber", "blue", "coral", "dune", "ember", "frost", "green", "haze", "ivory", "jade")]
    question = "请依据论文回答；请给引用；写一段 related work，" + "；".join(f"解释 {concept} 的机制" for concept in concepts)
    result = build_query_facets(question)
    assert len(result["facets"]) <= 8
    assert set(concepts) <= set(result["facets"][0]["anchors"])
    assert result["trace"]["technical_anchor_count"] == len(concepts)
    assert result["trace"]["meta_clauses_removed"] >= 2
    assert "related work" not in result["facets"][0]["anchors"]
    assert not any(item["text"] in {"依据论文回答", "给引用", "related work"} for item in result["facets"][1:])
    assert concepts[-1] in texts(result, "technical_phrase")
    assert all(item["text"] in question for item in result["facets"])


def test_document_introduction_does_not_consume_high_weight_facet():
    question = "依据论文回答：介绍 Signals in the Valley；解释 frost-lock dispatch 如何运行；请给出引用"
    result = build_query_facets(question, [row("v", "signals_valley.pdf")])
    assert result["facets"][0]["document_ids"] == ["v"]
    assert "Signals in the Valley" not in result["facets"][0]["anchors"]
    assert not any(item["kind"] == "clause" and "介绍" in item["text"] for item in result["facets"])
    assert "frost-lock dispatch" in result["facets"][0]["anchors"]


def test_more_than_anchor_cap_is_balanced_and_explicitly_reported():
    concepts = [f"gate-{index} scheduling" for index in range(40)]
    result = build_query_facets("；".join(f"解释 {item}" for item in concepts))
    anchors = result["facets"][0]["anchors"]
    assert len(anchors) == 32
    assert concepts[0] in anchors and concepts[-1] in anchors
    assert result["trace"]["technical_anchors_omitted"] == 8


def test_audience_intro_is_output_constraint_when_specific_evidence_need_follows():
    question = "我想向非技术读者介绍 Crimson Atlas。coral-lock dispatch 如何控制队列？实验怎样设置？"
    result = build_query_facets(question, [row("a", "Crimson_Atlas.pdf")])
    assert not any(facet["kind"] == "clause" and "非技术读者" in facet["text"] for facet in result["facets"])
    assert any(facet["kind"] == "clause" and "如何控制队列" in facet["text"] for facet in result["facets"])
    assert result["trace"]["meta_clauses_removed"] >= 1
    assert all(facet["text"] in question for facet in result["facets"])


def test_standalone_audience_intro_remains_a_substantive_request():
    question = "我想向非技术读者介绍 Crimson Atlas。"
    result = build_query_facets(question, [row("a", "Crimson_Atlas.pdf")])
    assert result["trace"]["meta_clauses_removed"] == 0
    assert any("非技术读者" in facet["text"] for facet in result["facets"])
