"""Independent component-retrieval acceptance; synthetic sources, no model calls."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from docthinker.server.research_component_coverage import functional_definitions
from docthinker.server.research_lexical_coverage import select_lexical_coverage
from docthinker.server.research_retrieval import build_retrieval_plan


def source(identifier, document, ordinal, text, **extra):
    return {"id": identifier, "document_id": document, "filename": document + ".txt",
            "ordinal": ordinal, "page": ordinal + 1, "content": text, **extra}


def pair_corpus():
    chunks = []
    for document, qualifier in (("Birch", "a permit is granted"), ("Elm", "a checksum matches")):
        for number in range(5):
            chunks.append(source(f"{document}-intro-{number}", document, number,
                f"1 Introduction\n{document} architecture uses hierarchical memory and context. "
                "We compare evaluation conditions and results for long-running workflows. "
                f"Performance improves on the selected benchmark in panel {number}."))
        chunks.append(source(document + "-definition", document, 6,
            "2 Architecture\nThe amber queue stores immutable events only after " + qualifier + ". "
            "The cobalt cache does not return expired values. "
            "Both components remain unavailable until the integrity check succeeds."))
        chunks.append(source(document + "-result", document, 7,
            "3 Results\nWe observed 61 correct answers out of 100 independent questions. "
            "The experiment held the model and token budget fixed; no cross-system superiority was tested."))
    return chunks


def assert_source_contract(result, chunks):
    original = {(row["document_id"], row["id"]): row["content"] for row in chunks}
    assert len(result["sources"]) <= 8
    assert sum(len(row["content"].encode()) for row in result["sources"]) <= 12000
    for row in result["sources"]:
        assert original[row["document_id"], row["id"]][row["excerpt_start"]:row["excerpt_end"]] == row["content"]
    assert result["trace"]["provider_calls"] == result["trace"]["model_calls"] == 0
    assert result["trace"]["semantic_entailment_verified"] is False


def test_functional_markers_preserve_exact_unicode_offsets_negation_and_source():
    row = {"section": "2 Architecture", "content":
           "🌿 The amber queue stores immutable records only after consent. "
           "The cobalt cache does not return expired values."}
    snapshot = deepcopy(row)
    markers = functional_definitions(row)
    assert {item["key"] for item in markers} == {"amber queue", "cobalt cache"}
    for item in markers:
        name = row["content"][item["name_start"]:item["name_end"]]
        assert " ".join(name.casefold().split()) == item["key"]
        assert 0 <= item["name_start"] < item["name_end"] <= item["predicate_start"] < item["predicate_end"] <= len(row["content"])
    cache = next(item for item in markers if item["key"] == "cobalt cache")
    assert "not" in row["content"][cache["predicate_start"]:cache["predicate_end"]]
    assert row == snapshot


def test_definition_markers_are_bounded_before_they_can_expand_candidate_obligations():
    names = ("amber", "cobalt", "violet", "silver", "crimson", "golden", "indigo", "azure")
    row = {"section": "2 Architecture", "content": " ".join(
        "The " + name + " queue stores event records." for name in names)}
    markers = functional_definitions(row)
    assert len(markers) == 4
    assert len({item["key"] for item in markers}) == 4
    for item in markers:
        assert row["content"][item["name_start"]:item["name_end"]].casefold() == item["key"]


@pytest.mark.parametrize("row", [
    {"section": "Related Work", "content": "The amber queue stores records."},
    {"section": "References", "content": "The amber queue stores records."},
    {"section": "2 Architecture", "content": "Figure 3: The amber queue stores records."},
    {"section": "2 Architecture", "content": "Table 2: The amber queue stores records."},
    {"section": "2 Architecture", "content": "The amber queue and cobalt cache are mentioned."},
    {"section": "2 Architecture", "content": "Memory stores records."},
])
def test_caption_reference_mention_and_generic_head_are_not_definitions(row):
    assert functional_definitions(row) == []


def test_component_cells_do_not_let_one_paper_cover_another_papers_definition():
    chunks = pair_corpus()
    question = "Compare Birch and Elm architecture, evaluation conditions and results."
    snapshot = deepcopy(chunks)
    baseline = build_retrieval_plan(question, chunks)["sources"]
    result = select_lexical_coverage(question, baseline, chunks)
    by_id = {row["id"]: row["content"] for row in result["sources"]}
    assert "Birch-definition" in by_id and "Elm-definition" in by_id
    assert "only after a permit is granted" in by_id["Birch-definition"]
    assert "only after a checksum matches" in by_id["Elm-definition"]
    assert all("does not return expired" in by_id[key] for key in ("Birch-definition", "Elm-definition"))
    assert chunks == snapshot
    assert_source_contract(result, chunks)


def test_unrelated_unnamed_glossary_does_not_gain_component_obligations():
    chunks = pair_corpus()
    chunks += [source("orchard-definitions", "Orchard", 0,
        "2 Definitions\nThe violet cache stores apple deliveries. "
        "The plum queue holds harvest labels. The pear buffer returns tree counts.")]
    question = "Compare Birch and Elm architecture and evaluation results."
    result = select_lexical_coverage(question, build_retrieval_plan(question, chunks)["sources"], chunks)
    assert not any(row["id"] == "orchard-definitions" for row in result["sources"])
    assert_source_contract(result, chunks)


@pytest.mark.parametrize("question", [
    "What were the exact-match accuracy scores for Birch?",
    "Compare Birch and Elm exact-match accuracy scores only.",
    "Compare the measured latency results for Birch and Elm; do not explain architecture.",
])
def test_numeric_only_request_does_not_invoke_component_marker_extraction(question):
    chunks = pair_corpus()
    baseline = build_retrieval_plan(question, chunks)["sources"]
    with patch("docthinker.server.research_lexical_coverage.functional_definitions",
               side_effect=AssertionError("pure numerical task must not request structural definitions")):
        result = select_lexical_coverage(question, baseline, chunks)
    assert_source_contract(result, chunks)


def test_numeric_comparison_packet_keeps_each_papers_result_and_its_conditions():
    chunks = pair_corpus()
    question = "Compare Birch and Elm exact-match accuracy scores only."
    result = select_lexical_coverage(question, build_retrieval_plan(question, chunks)["sources"], chunks)
    selected = {row["id"]: row["content"] for row in result["sources"]}
    assert {"Birch-result", "Elm-result"} <= selected.keys()
    assert not any(identifier.endswith("-definition") for identifier in selected)
    for identifier in ("Birch-result", "Elm-result"):
        assert "61 correct answers out of 100 independent questions" in selected[identifier]
        assert "held the model and token budget fixed" in selected[identifier]
        assert "no cross-system superiority was tested" in selected[identifier]
    assert_source_contract(result, chunks)


def test_explicit_structure_exclusion_applies_to_final_packet_not_only_markers():
    chunks = pair_corpus()
    question = "Compare the measured latency results for Birch and Elm; do not explain architecture."
    result = select_lexical_coverage(question, build_retrieval_plan(question, chunks)["sources"], chunks)
    assert not any(row["id"].endswith("-definition") for row in result["sources"])
    assert_source_contract(result, chunks)
