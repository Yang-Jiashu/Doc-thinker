"""Natural research prose may add evidence, never invented logical claims."""
from __future__ import annotations

import copy
import json

import pytest

from docthinker.server.research_evidence_organizer import organize_source_evidence, sentence_spans


def chunk(identifier, text, document="paper-a", ordinal=0, **extras):
    return {"id": identifier, "document_id": document, "filename": document + ".pdf",
            "page": ordinal + 1, "ordinal": ordinal, "content": text, **extras}


def source(raw, label="S1", start=0, end=None):
    end = len(raw["content"]) if end is None else end
    return {**raw, "content": raw["content"][start:end], "excerpt_start": start, "excerpt_end": end,
            "label": label, "truncated": start > 0 or end < len(raw["content"])}


def assert_spans_bound(result, rows):
    live = {row["id"]: row for row in rows}
    sources = {row["id"]: row for row in result["sources"]}
    for quote in result["claims"]:
        raw = live[quote["source_ids"][0]]
        assert raw["content"][quote["excerpt_start"]:quote["excerpt_end"]] == quote["quote"]
        visible = sources[raw["id"]]
        assert visible["excerpt_start"] <= quote["excerpt_start"] < quote["excerpt_end"] <= visible["excerpt_end"]
    for row in result["sources"]:
        assert row["content"] == live[row["id"]]["content"][row["excerpt_start"]:row["excerpt_end"]]


def test_natural_method_and_condition_add_real_material_without_horn_rules():
    intro = chunk("intro", "Adaptive storage maintains a bounded active buffer for document analysis.")
    methods = chunk("methods", "Adaptive storage evicts inactive records to a durable store. "
                    "However, storage eviction requires explicit retrieval before the records can influence a response.", ordinal=1)
    unrelated = chunk("noise", "Storage prices decrease during summer promotions for personal phones.", document="sales")
    rows = [intro, methods, unrelated]
    result = organize_source_evidence("Explain adaptive storage eviction conditions", [source(intro)], rows)
    assert result["trace"]["support_item_count"] >= 1
    assert "methods" in {row["id"] for row in result["sources"]}
    assert "noise" not in {row["id"] for row in result["sources"]}
    assert result["trace"]["added_source_bytes"] > 0
    assert result["sources"][0] == source(intro)
    assert all(group["kind"] == "source_evidence" and not group.get("proof") for group in result["supporting_material"])
    assert all(not group["entailment_verified"] for group in result["supporting_material"])
    assert any("requires explicit retrieval" in row["quote"] for row in result["claims"])
    assert_spans_bound(result, rows)


def test_clipped_positive_result_restores_negation_and_full_condition():
    row = chunk("measure", "Adaptive storage improves response accuracy in our experiment, "
                "but does not improve accuracy when the retrieval index is stale.")
    initial = source(row, end=row["content"].index(","))
    result = organize_source_evidence("Explain adaptive storage accuracy", [initial], [row])
    assert result["trace"]["completed_sentence_count"] == 1
    assert result["sources"][0]["content"] == row["content"]
    assert result["sources"][0]["label"] == "S1"
    assert "does not improve" in result["claims"][0]["quote"]
    assert_spans_bound(result, [row])


def test_cross_language_seed_can_complete_a_clipped_sentence_without_new_document():
    row = chunk("measure", "Adaptive storage improves accuracy, but only when retrieved records are current.")
    outsider = chunk("outsider", "Adaptive storage improves accuracy under all conditions.", document="other")
    initial = source(row, end=row["content"].index(","))
    result = organize_source_evidence("这种长期记忆有什么限制？", [initial], [row, outsider])
    assert result["trace"]["lexical_anchor_unavailable"] is True
    assert result["trace"]["completed_sentence_count"] == 1
    assert [row["id"] for row in result["sources"]] == ["measure"]
    assert "only when" in result["sources"][0]["content"]


def test_multilingual_search_terms_are_only_retrieval_hints():
    intro = chunk("intro", "Adaptive storage keeps the current interaction small.")
    method = chunk("method", "Adaptive storage loads archived records through an explicit retrieval operation.")
    result = organize_source_evidence("长期记忆怎么调出资料？", [source(intro)], [intro, method],
                                      query_terms=["adaptive", "storage", "retrieval"])
    assert result["trace"]["support_item_count"] == 1
    assert result["claims"][0]["quote"] == method["content"]
    assert "长期记忆" not in result["claims"][0]["quote"]


def test_cross_language_semantic_seed_adds_only_backend_gated_new_source_sentences():
    seed = chunk("seed", "A finite active context provides a workspace for processing conversations.")
    method = chunk("method", "The scheduler retrieves archived records into the workspace when a new request arrives. "
                   "However, deleted records cannot be retrieved from the archive.", ordinal=1)
    unrelated = chunk("unrelated", "The scheduler retrieves advertising records for auction pricing. "
                      "However, auction pricing increases under heavy demand.", ordinal=2)
    rows = [seed, method, unrelated]
    result = organize_source_evidence("长期记忆怎样在新问题出现时调取资料，有什么限制？", [source(seed)], rows,
                                      semantic_scores={"seed": 0.84, "method": 0.81},
                                      semantic_spans={row["id"]: [0, len(row["content"])] for row in (seed, method)})
    assert result["trace"]["lexical_anchor_unavailable"] is True
    assert result["trace"]["semantic_anchor_count"] == 1
    assert result["trace"]["support_item_count"] >= 1
    assert {row["id"] for row in result["sources"]} == {"seed", "method"}
    assert any("deleted records cannot" in row["quote"] for row in result["claims"])
    assert result["trace"]["semantic_entailment_verified"] is False
    assert_spans_bound(result, rows)


def test_semantic_candidates_without_a_scored_seed_do_not_ground_an_unrelated_question():
    seed = chunk("seed", "A finite active context provides a workspace for processing conversations.")
    method = chunk("method", "The scheduler retrieves archived records when a new request arrives.")
    result = organize_source_evidence("长期记忆如何调取资料？", [source(seed)], [seed, method],
                                      semantic_scores={"method": 0.82})
    assert result["trace"]["support_item_count"] == 0
    assert result["trace"]["skipped_reason"] == "no_grounded_query_anchor"


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -0.4, 2.0, "0.8"])
def test_invalid_similarity_values_cannot_create_a_cross_language_anchor(bad):
    seed = chunk("seed", "A finite active context provides a workspace for processing conversations.")
    method = chunk("method", "The scheduler retrieves archived records when a new request arrives.")
    result = organize_source_evidence("长期记忆如何调取资料？", [source(seed)], [seed, method],
                                      semantic_scores={"seed": bad, "method": 0.82})
    assert result["trace"]["support_item_count"] == 0


def test_repackaging_any_already_supplied_sentence_is_not_support():
    intro = chunk("intro", "Adaptive storage maintains active records for retrieval.")
    existing = chunk("existing", "Adaptive storage retrieves archived records using a lexical index.", ordinal=1)
    original = [source(intro), source(existing, "S2")]
    result = organize_source_evidence("Explain adaptive storage retrieval", original[:1], [intro, existing],
                                      already_supplied_sources=original)
    assert result["trace"]["support_item_count"] == 0
    assert result["prompt"] == ""
    assert result["sources"] == original[:1]


def test_duplicate_document_ids_do_not_create_fake_independent_support():
    intro = chunk("intro", "Adaptive storage retrieves archived records using a lexical index.")
    duplicate = chunk("dup", intro["content"], document="copied-paper")
    result = organize_source_evidence("Explain adaptive storage retrieval", [source(intro)], [intro, duplicate])
    assert not result["supporting_material"]
    assert result["sources"] == [source(intro)]


def test_counterexample_is_quoted_and_never_turned_into_consensus():
    positive = chunk("positive", "Adaptive storage improves retrieval accuracy on curated records.")
    negative = chunk("negative", "Adaptive storage does not improve retrieval accuracy on noisy records.", document="negative-study")
    result = organize_source_evidence("Compare adaptive storage retrieval accuracy", [source(positive)], [positive, negative])
    assert result["trace"]["support_item_count"] == 1
    assert "does not improve" in result["claims"][0]["quote"]
    assert "limitation" in result["claims"][0]["role_hints"]
    assert not any(key in result["supporting_material"][0] for key in ("support_count", "depth", "hypotheses"))


def test_generic_rhetorical_words_cannot_pull_in_unrelated_papers():
    seed = chunk("seed", "Thermal storage reduces the cooling load in the monitored building.")
    noise = chunk("noise", "The method improves performance under controlled conditions. "
                  "However, our experimental results show limitations.", document="random")
    result = organize_source_evidence("Explain thermal storage method results and limitations", [source(seed)], [seed, noise])
    assert not result["supporting_material"]


def test_bibliographic_titles_do_not_become_evidence_for_a_cited_papers_findings():
    seed = chunk("seed", "Adaptive storage maintains an active buffer for documents.")
    references = chunk("references", "A. Author. Adaptive storage improves retrieval accuracy. "
                       "arXiv preprint arXiv:1234.5678, 2020. "
                       "B. Author. Adaptive storage for perfect long-term recall. "
                       "arXiv preprint arXiv:1234.1234, 2021.", ordinal=2)
    result = organize_source_evidence("Explain adaptive storage retrieval accuracy", [source(seed)], [seed, references])
    assert not result["supporting_material"]
    assert result["trace"]["bibliographic_chunks_omitted"] == 1


def test_shortlist_prefers_relevant_qualifier_over_many_lexical_near_duplicates():
    seed = chunk("seed", "Thermal storage reduces cooling demand in the monitored building.")
    condition = chunk("condition", "Thermal storage reduces cooling demand only under low humidity. "
                      "However, the savings disappear in humid weather.", ordinal=1)
    duplicate_rows = [chunk(f"duplicate-{i}", seed["content"], document=f"copy-{i}") for i in range(100)]
    result = organize_source_evidence("Explain thermal storage cooling demand limitations", [source(seed)],
                                      [seed, *duplicate_rows, condition])
    assert "condition" in {row["id"] for row in result["sources"]}
    assert len(result["sources"]) <= 2
    assert any("savings disappear" in row["quote"] for row in result["claims"])


@pytest.mark.parametrize("options", [{"max_added_source_bytes": 1}, {"max_source_bytes": 1},
                                    {"max_sources": 1}, {"max_groups": 0}, {"max_prompt_bytes": 1}])
def test_budget_failure_preserves_original_and_never_sends_half_sentence(options):
    intro = chunk("intro", "Adaptive storage maintains an active buffer for documents.")
    addition = chunk("addition", "Adaptive storage moves inactive material into a durable searchable archive.")
    original = [source(intro)]
    result = organize_source_evidence("Explain adaptive storage", original, [intro, addition], **options)
    assert result["sources"] == original
    assert result["supporting_material"] == []
    assert result["prompt"] == ""


def test_stale_or_generated_sources_cannot_become_new_quotes():
    intro = chunk("intro", "Adaptive storage maintains an active buffer for documents.")
    generated = chunk("generated", "Adaptive storage gives perfect retrieval accuracy.", provenance="llm")
    result = organize_source_evidence("Explain adaptive storage retrieval", [source(intro)], [intro, generated])
    assert not result["supporting_material"]
    modified = source(intro)
    modified["content"] = "Adaptive storage guarantees eternal memory."
    invalid = organize_source_evidence("Explain adaptive storage", [modified], [intro])
    assert invalid["trace"]["skipped_reason"] == "source_binding_invalid"
    assert invalid["supporting_material"] == []


def test_input_order_does_not_change_selection_and_inputs_are_untouched():
    intro = chunk("intro", "Thermal storage reduces cooling demand in the monitored building.")
    rows = [intro, chunk("a", "Thermal storage improves cooling efficiency under low humidity.", "second"),
            chunk("b", "Thermal storage does not reduce cooling demand during humid weather.", "third")]
    frozen = copy.deepcopy(rows)
    first = organize_source_evidence("Compare thermal storage cooling demand limitations", [source(intro)], rows)
    second = organize_source_evidence("Compare thermal storage cooling demand limitations", [source(intro)], list(reversed(rows)))
    assert first == second
    assert rows == frozen
    assert first["trace"]["model_calls"] == first["trace"]["memory_writes"] == 0
    payload = json.loads(first["prompt"].split("\n")[-1])
    assert payload["supporting_material"]
    assert all(row["content"] not in first["prompt"] for row in first["sources"])


def test_sentence_boundaries_preserve_decimal_and_abbreviation_and_skip_unfinished_tail():
    text = "We measured 3.14 units in Fig. 2 under low load. However, high load failed. An unfinished claim"
    spans = sentence_spans(text)
    assert [text[left:right] for left, right in spans] == [
        "We measured 3.14 units in Fig. 2 under low load.", "However, high load failed."]


def test_oversized_sentence_is_not_clipped_to_a_misleading_positive_clause():
    row = chunk("long", "Adaptive storage improves retrieval accuracy " + "in a controlled setting " * 100 +
                ", but cannot recover deleted source documents.")
    result = organize_source_evidence("Explain adaptive storage retrieval accuracy", [source(row, end=120)], [row])
    assert result["supporting_material"] == []
    assert result["sources"] == [source(row, end=120)]


def test_section_headers_are_excluded_from_sentences_and_break_adjacency():
    text = ("Adaptive storage improves archival retrieval.\n4 Related Work\n"
            "However, adaptive storage was not used in the earlier comparison system.")
    spans = sentence_spans(text)
    assert [text[left:right] for left, right in spans] == [
        "Adaptive storage improves archival retrieval.",
        "However, adaptive storage was not used in the earlier comparison system."]
    seed = chunk("seed", "Adaptive storage maintains an active buffer for documents.")
    material = chunk("material", text)
    result = organize_source_evidence("Explain adaptive storage retrieval", [source(seed)], [seed, material])
    for group in result["supporting_material"]:
        quoted = [row for row in result["claims"] if row["id"] in group["premise_ids"]]
        assert len(quoted) == 1
        assert "Related Work" not in quoted[0]["quote"]


def test_semantic_window_in_later_half_cannot_qualify_unrelated_first_half():
    seed = chunk("seed", "A finite active context provides a workspace for processing conversations.")
    unrelated = "The scheduler retrieves advertising records for auction pricing. "
    relevant = ("The scheduler retrieves archived records into the workspace when a request arrives. "
                "However, deleted records cannot be retrieved from the archive.")
    material = chunk("material", unrelated + relevant)
    result = organize_source_evidence("分析长期记忆如何调取資料", [source(seed)], [seed, material],
        semantic_scores={"seed": 0.85, "material": 0.83},
        semantic_spans={"seed": [0, len(seed["content"])],
                        "material": [len(unrelated), len(material["content"])]})
    assert result["supporting_material"]
    assert "auction" not in " ".join(row["content"] for row in result["sources"])
    assert "deleted records cannot" in result["sources"][-1]["content"]
    assert_spans_bound(result, [seed, material])


@pytest.mark.parametrize("bad", [None, {}, {"seed": [0, 99999]}, {"seed": [True, 50]},
                                     {"seed": [0.0, 50]}, {"seed": [0, 0]}, {"seed": "0,50"}])
def test_missing_or_invalid_semantic_locations_cannot_promote_unseen_prose(bad):
    seed = chunk("seed", "A finite active context provides a workspace for processing conversations.")
    material = chunk("material", "The scheduler retrieves archived records into the active workspace.")
    result = organize_source_evidence("分析长期记忆如何调取资料", [source(seed)], [seed, material],
        semantic_scores={"seed": 0.85, "material": 0.83}, semantic_spans=bad)
    assert result["sources"] == [source(seed)]
    assert not result["supporting_material"]


def test_semantic_anchor_must_overlap_the_actual_visible_seed():
    first = "The scheduler retrieves advertising records for auction pricing. "
    second = "A finite active context provides a workspace for processing conversations."
    seed = chunk("seed", first + second)
    material = chunk("material", "The scheduler retrieves archived records into the active workspace.")
    result = organize_source_evidence("分析长期记忆如何调取资料", [source(seed, end=len(first) - 1)], [seed, material],
        semantic_scores={"seed": 0.85, "material": 0.83},
        semantic_spans={"seed": [len(first), len(seed["content"])], "material": [0, len(material["content"])]})
    assert not result["supporting_material"]
    assert result["trace"]["semantic_anchor_count"] == 0


def test_same_source_expansion_retains_generic_procedural_antecedent():
    previous = "The earlier comparison system uses a vector index. "
    visible = "Queries are then embedded into the same vector space for nearest neighbor retrieval."
    raw = chunk("method", previous + visible)
    original = source(raw, start=len(previous))
    result = organize_source_evidence("Explain vector nearest neighbor retrieval", [original], [raw])
    assert result["sources"][0]["content"] == raw["content"]
    assert "earlier comparison system" in result["sources"][0]["content"]
    assert result["sources"][0]["label"] == "S1"
    assert_spans_bound(result, [raw])


def test_same_source_expansion_cannot_bridge_a_section_header_or_unrelated_paragraph():
    first = "Adaptive storage improves archival retrieval."
    for gap in ("\n4 Related Work\n", " Unrelated auction prices increased last year. "):
        raw = chunk("method", first + gap + "Adaptive storage uses nearest neighbor retrieval for current queries.")
        original = source(raw, end=len(first))
        result = organize_source_evidence("Explain adaptive storage retrieval", [original], [raw])
        assert result["sources"] == [original]
        assert not result["supporting_material"]
