"""Real source-text graph routes, bounded budgets and revocable cache views."""
import copy
import json

import pytest

from docthinker.server import research_graph_retrieval as module
from docthinker.server.research_graph_retrieval import BoundedEvidenceGraph, graph_retrieval_intent


QUESTION = "Find an evidence chain from Alpha to Delta."


def chunk(content, identifier="c1", document="d1", **fields):
    return {"id": identifier, "document_id": document, "content": content,
            "filename": document + ".txt", "ordinal": 0, **fields}


def source(row, label="S1", start=0, end=None):
    end = len(row["content"]) if end is None else end
    return {**row, "content": row["content"][start:end], "label": label,
            "excerpt_start": start, "excerpt_end": end, "truncated": bool(start or end < len(row["content"])),
            "original_content_length": len(row["content"])}


def chain():
    return [chunk("Alpha increases Beta.", "a", "a"),
            chunk("Beta reduces Gamma.", "bridge", "b"),
            chunk("Gamma improves Delta.", "d", "d")]


def codes(result):
    return {gap["code"] for gap in result["gaps"]}


def test_rescues_literal_middle_edge_between_separately_retrieved_endpoint_sources():
    rows = chain()
    seeds = [source(rows[0]), source(rows[2], "S2")]
    original = copy.deepcopy((rows, seeds))
    result = BoundedEvidenceGraph().expand(QUESTION, seeds, rows, protected_source_ids=["a", "d"])
    assert result["sources"][:2] == seeds
    assert result["sources"][2]["content"] == rows[1]["content"]
    assert result["sources"][2]["label"] == "S3"
    assert result["trace"]["bridge_source_ids"] == ["bridge"]
    assert result["trace"]["kept_source_ids"] == ["a", "d", "bridge"]
    assert result["trace"]["status"] == "expanded"
    assert result["trace"]["max_hops"] == 2
    assert result["trace"]["paths"][0]["source_ids"] == ["a", "bridge", "d"]
    assert result["trace"]["paths"][0]["expansion_hops"] == 1
    assert result["trace"]["semantic_entailment_verified"] is False
    assert result["trace"]["model_calls"] == 0
    assert "graph_association_not_proof" in codes(result)
    assert (rows, seeds) == original


def test_two_missing_edges_are_selected_as_a_whole_path():
    rows = chain()
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0])], rows)
    assert set(result["trace"]["bridge_source_ids"]) == {"bridge", "d"}
    assert result["trace"]["paths"][0]["expansion_hops"] == 2
    assert all(row["truncated"] is False for row in result["sources"][1:])


def test_no_partial_path_when_second_missing_edge_exceeds_capacity():
    rows = chain()
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0])], rows, max_sources=2)
    assert result["sources"] == [source(rows[0])]
    assert result["trace"]["paths"] == []
    assert "graph_capacity_reached" in codes(result)


def test_more_than_two_new_edges_are_not_silently_walked():
    rows = chain()
    rows.append(chunk("Delta improves Epsilon.", "e", "e"))
    result = BoundedEvidenceGraph().expand("Find a causal chain from Alpha to Epsilon.", [source(rows[0])], rows)
    assert result["trace"]["bridge_source_ids"] == []
    assert "graph_no_connected_path" in codes(result)


def test_shared_relation_words_and_substring_entity_matches_do_not_make_edges():
    rows = [chunk("Alpha increases Beta.", "a", "a"),
            chunk("Omega increases Delta.", "noise", "n"),
            chunk("BetaMax increases Delta.", "substring", "s")]
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0])], rows)
    assert result["trace"]["bridge_source_ids"] == []


def test_unrelated_background_is_not_added_after_the_bridge():
    rows = chain() + [chunk("Beta increases Lemon.", "noise", "z"),
                      chunk("Lemon increases Orange.", "noise2", "z2")]
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0]), source(rows[2], "S2")], rows)
    assert result["trace"]["bridge_source_ids"] == ["bridge"]


def test_negated_relation_is_retained_as_counterevidence_not_promoted_to_fact():
    rows = chain()
    rows[1]["content"] = "Beta does not reduce Gamma."
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0]), source(rows[2], "S2")], rows)
    assert result["sources"][-1]["content"] == rows[1]["content"]
    assert result["trace"]["paths"][0]["edges"][0]["polarity"] == -1
    assert "graph_counterevidence_in_path" in codes(result)


def test_entire_conditional_original_and_condition_mismatch_warning_are_preserved():
    rows = chain()
    rows[1]["content"] = "Beta reduces Gamma under cold conditions. No trials were conducted under heat."
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0]), source(rows[2], "S2")], rows)
    assert result["sources"][-1]["content"] == rows[1]["content"]
    assert result["trace"]["paths"][0]["edges"][0]["conditional"] is True
    assert "graph_path_conditions_differ" in codes(result)


@pytest.mark.parametrize("bridge", ["Beta may reduce Gamma.", "Beta reduces Gamma only if snow falls.",
                                    "Beta reduces Gamma but raises cost."])
def test_unsupported_qualifiers_are_not_stripped_to_invent_a_connection(bridge):
    rows = chain()
    rows[1]["content"] = bridge
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0]), source(rows[2], "S2")], rows)
    assert result["trace"]["bridge_source_ids"] == []


def test_explicit_different_case_scopes_do_not_join_across_documents():
    rows = chain()
    rows[0]["content"] = "Case: cold\n" + rows[0]["content"]
    rows[1]["content"] = "Case: warm\n" + rows[1]["content"]
    rows[2]["content"] = "Case: cold\n" + rows[2]["content"]
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0]), source(rows[2], "S2")], rows)
    assert result["trace"]["bridge_source_ids"] == []


def test_matching_named_case_scopes_can_bridge_cross_document_sources():
    rows = chain()
    for row in rows:
        row["content"] = "Case: cold\n" + row["content"]
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0]), source(rows[2], "S2")], rows)
    assert result["trace"]["bridge_source_ids"] == ["bridge"]


def test_an_unseen_claim_in_the_seed_chunk_cannot_be_used_as_a_visible_edge():
    rows = chain()
    rows[0]["content"] += " Beta reduces Gamma."
    rows = [rows[0], rows[2]]
    seed = source(rows[0], end=len("Alpha increases Beta."))
    result = BoundedEvidenceGraph().expand(QUESTION, [seed], rows)
    assert result["trace"]["bridge_source_ids"] == []


def test_literal_seed_with_no_parseable_statement_is_not_an_arbitrary_graph_root():
    rows = [chunk("A report about Alpha and Delta.", "start", "start"), *chain()]
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0])], rows)
    assert result["sources"] == [source(rows[0])]


def test_explicit_idea_request_can_discover_connected_material_without_fixed_question():
    rows = chain()
    result = BoundedEvidenceGraph().expand("请根据我的资料提出新的研究假设", [source(rows[0])], rows)
    assert result["trace"]["status"] == "expanded"
    assert "bridge" in result["trace"]["bridge_source_ids"]
    assert all(path["epistemic_status"] == "related_sources_not_entailment" for path in result["trace"]["paths"])


@pytest.mark.parametrize("question", ["What does Alpha mean?", "Explain Alpha and Delta.",
    "总结全部论文", "不要扩展因果链，逐字引用", "Translate this evidence chain.",
    "Find a causal chain without graph expansion.", "Do not perform multi-hop reasoning.",
    "Don't use graph expansion for this evidence chain.", "不要提出假设，解释这段原文", "Alpha 和 Delta 是否有关？"])
def test_ordinary_literal_or_declined_requests_do_not_build_the_graph(question, monkeypatch):
    graph = BoundedEvidenceGraph()
    monkeypatch.setattr(graph, "_prepare", lambda *_: pytest.fail("graph should not run"))
    rows = chain()
    assert graph_retrieval_intent(question) is False
    result = graph.expand(question, [source(rows[0])], rows)
    assert result["sources"] == [source(rows[0])]
    assert result["trace"]["reason"] == "not_requested"


@pytest.mark.parametrize("question", ["研究 Alpha 到 Delta 的因果链", "Find a multi-hop path from Alpha to Delta.",
                                     "Propose research ideas about Alpha.", "请提出可检验的新假设"])
def test_explicit_intents_are_enabled_without_a_user_mode_switch(question):
    assert graph_retrieval_intent(question)


def test_full_capacity_keeps_every_literal_and_protected_source():
    rows = chain()
    seeds = [source(rows[0]), source(rows[2], "S2")]
    for index in range(6):
        row = chunk(f"Unique background {index}.", f"n{index}", f"n{index}")
        rows.append(row)
        seeds.append(source(row, f"S{len(seeds) + 1}"))
    result = BoundedEvidenceGraph().expand(QUESTION, seeds, rows, protected_source_ids=[row["id"] for row in seeds])
    assert result["sources"] == seeds
    assert result["trace"]["bridge_source_ids"] == []
    assert "graph_capacity_reached" in codes(result)


def test_byte_limit_is_exact_and_does_not_clip_a_qualifying_tail():
    rows = chain()
    seeds = [source(rows[0]), source(rows[2], "S2")]
    size = sum(len(row["content"].encode()) for row in rows)
    graph = BoundedEvidenceGraph()
    assert graph.expand(QUESTION, seeds, rows, max_source_bytes=size)["trace"]["status"] == "expanded"
    result = graph.expand(QUESTION, seeds, rows, max_source_bytes=size - 1)
    assert result["sources"] == seeds
    assert "graph_capacity_reached" in codes(result)


def test_oversized_bound_evidence_unit_is_omitted_instead_of_cutting_qualifications():
    rows = chain()
    rows[1]["content"] += " However, this observation is limited to " + "a narrow testing context " * 60 + "."
    seeds = [source(rows[0]), source(rows[2], "S2")]
    result = BoundedEvidenceGraph().expand(QUESTION, seeds, rows)
    assert result["sources"] == seeds
    assert "graph_capacity_reached" in codes(result)


def long_bridge(main):
    before = "The archive describes sample preparation and laboratory equipment. " * 15
    after = "The appendix provides additional historical background for the reader. " * 14
    return before + main + after


def test_two_thousand_character_chunk_yields_exact_complete_claim_window():
    rows = chain()
    rows[1]["content"] = long_bridge("Beta reduces Gamma. ")
    assert 1800 < len(rows[1]["content"]) < 2300
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0]), source(rows[2], "S2")], rows)
    bridge = result["sources"][-1]
    assert bridge["id"] == "bridge"
    assert bridge["content"].strip() == "Beta reduces Gamma."
    assert bridge["truncated"] is True
    assert bridge["evidence_scope"] == "excerpt_only"
    assert bridge["original_content_length"] == len(rows[1]["content"])
    assert rows[1]["content"][bridge["excerpt_start"]:bridge["excerpt_end"]] == bridge["content"]
    assert 0 < bridge["excerpt_start"] < bridge["excerpt_end"] < len(rows[1]["content"])
    assert "graph_excerpt_windowed" in codes(result)


@pytest.mark.parametrize("statement", [
    "Beta reduces Gamma under cold conditions. No trials were conducted under heat. ",
    "Under the assumption that every device has calibration. Beta reduces Gamma. ",
    "The experiment did not measure the heat response. Beta reduces Gamma. ",
    "Beta reduces Gamma. However, this finding is limited to winter. ",
    "Beta reduces Gamma. No evidence supports\nextrapolation to summer. ",
    "Beta reduces\nGamma. No evidence supports extrapolation to summer. ",
])
def test_long_source_window_keeps_immediately_adjacent_conditions_and_negations(statement):
    rows = chain()
    rows[1]["content"] = long_bridge(statement)
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0]), source(rows[2], "S2")], rows)
    bridge = result["sources"][-1]
    assert bridge["id"] == "bridge"
    assert statement.strip() in bridge["content"]
    assert len(bridge["content"]) <= 1000
    assert rows[1]["content"][bridge["excerpt_start"]:bridge["excerpt_end"]] == bridge["content"]


def test_path_requiring_distant_claims_in_one_long_chunk_is_omitted_whole():
    rows = chain()
    rows[1]["content"] = ("Beta reduces Gamma. " + "Additional historical context is described here. " * 25
                           + "Gamma improves Delta.")
    rows = rows[:2]
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0])], rows)
    assert result["sources"] == [source(rows[0])]
    assert result["trace"]["paths"] == []
    assert "graph_capacity_reached" in codes(result)


def test_same_chunk_window_covers_all_route_claims_not_only_the_first_match():
    rows = chain()
    required = "Beta reduces Gamma. The equipment was calibrated beforehand. Gamma improves Delta. "
    rows[1]["content"] = long_bridge(required)
    rows = rows[:2]
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0])], rows)
    assert result["trace"]["bridge_source_ids"] == ["bridge"]
    added = result["sources"][-1]
    assert required.strip() in added["content"]
    assert len(result["trace"]["paths"][0]["edges"]) == 2
    for edge in result["trace"]["paths"][0]["edges"]:
        assert added["excerpt_start"] <= edge["excerpt_start"] < edge["excerpt_end"] <= added["excerpt_end"]


def test_negative_context_beyond_window_cap_blocks_whole_two_edge_addition():
    rows = chain()
    rows[1]["content"] = long_bridge("Beta reduces Gamma. ")
    rows[2]["content"] = long_bridge("Gamma improves Delta. However, " + "the study does not establish generalization " * 25 + ". ")
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0])], rows)
    assert result["sources"] == [source(rows[0])]
    assert result["trace"]["bridge_source_ids"] == []
    assert "graph_capacity_reached" in codes(result)


def test_canonical_chunk_already_cut_across_a_statement_cannot_enter_graph():
    rows = chain()
    rows[1].update(content=long_bridge("Beta reduces Gamma. "), excerpt_start=7, truncated=True)
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0]), source(rows[2], "S2")], rows)
    assert result["trace"]["bridge_source_ids"] == []


def test_long_excerpt_byte_budget_uses_exact_selected_window_not_entire_chunk():
    rows = chain()
    rows[1]["content"] = long_bridge("Beta reduces Gamma. ")
    seeds = [source(rows[0]), source(rows[2], "S2")]
    graph = BoundedEvidenceGraph()
    first = graph.expand(QUESTION, seeds, rows)
    size = sum(len(row["content"].encode()) for row in first["sources"])
    assert graph.expand(QUESTION, seeds, rows, max_source_bytes=size)["trace"]["status"] == "expanded"
    assert graph.expand(QUESTION, seeds, rows, max_source_bytes=size - 1)["sources"] == seeds


def test_source_window_receipt_binds_exact_offsets_and_digest_for_final_fit_checks():
    import hashlib
    rows = chain()
    rows[1]["content"] = long_bridge("Beta reduces Gamma. ")
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0]), source(rows[2], "S2")], rows)
    by_id = {row["id"]: row for row in result["sources"]}
    receipt = result["trace"]["paths"][0]["source_windows"]
    assert {window["source_id"] for window in receipt} == {"a", "bridge", "d"}
    for window in receipt:
        retained = by_id[window["source_id"]]
        assert window["content_sha256"] == hashlib.sha256(retained["content"].encode()).hexdigest()
        assert window["excerpt_start"] == retained["excerpt_start"]
        assert window["excerpt_end"] == retained["excerpt_end"]


def test_candidate_and_cache_result_are_invariant_to_corpus_permutation():
    rows = chain()
    seeds = [source(rows[0]), source(rows[2], "S2")]
    first = BoundedEvidenceGraph().expand(QUESTION, seeds, rows)
    second = BoundedEvidenceGraph().expand(QUESTION, seeds, list(reversed(rows)))
    assert first == second


def test_same_content_reuses_sparse_graph_without_parsing_again(monkeypatch):
    rows = chain()
    seeds = [source(rows[0]), source(rows[2], "S2")]
    graph = BoundedEvidenceGraph()
    graph.prepare(rows)
    monkeypatch.setattr(module, "extract_document_claims", lambda *_a, **_kw: pytest.fail("cache must reuse parsed graph"))
    result = graph.expand(QUESTION, seeds, list(reversed(rows)))
    assert result["trace"]["cache_hit"] is True
    assert result["trace"]["bridge_source_ids"] == ["bridge"]


def test_current_live_membership_revokes_cached_bridge_source():
    rows = chain()
    seeds = [source(rows[0]), source(rows[2], "S2")]
    graph = BoundedEvidenceGraph()
    assert graph.expand(QUESTION, seeds, rows)["trace"]["status"] == "expanded"
    result = graph.expand(QUESTION, seeds, [rows[0], rows[2]])
    assert result["trace"]["status"] == "unchanged"
    assert result["trace"]["bridge_source_ids"] == []


def test_same_ids_with_changed_content_cannot_reuse_old_tenant_or_revision_edges():
    rows = chain()
    seeds = [source(rows[0]), source(rows[2], "S2")]
    graph = BoundedEvidenceGraph()
    graph.expand(QUESTION, seeds, rows)
    other_view = copy.deepcopy(rows)
    other_view[1]["content"] = "Unrelated increases Orange."
    result = graph.expand(QUESTION, seeds, other_view)
    assert result["trace"]["cache_hit"] is False
    assert result["trace"]["bridge_source_ids"] == []


def test_cached_graph_never_returns_old_filename_or_source_text():
    rows = chain()
    seeds = [source(rows[0]), source(rows[2], "S2")]
    graph = BoundedEvidenceGraph()
    graph.expand(QUESTION, seeds, rows)
    rows[1]["filename"] = "current-name.txt"
    result = graph.expand(QUESTION, seeds, rows)
    assert result["trace"]["cache_hit"] is True
    assert result["sources"][-1]["filename"] == "current-name.txt"


@pytest.mark.parametrize("fields", [{"is_generated": True}, {"is_discovered": "1"},
                                    {"provenance": "hypothesis"}, {"state": "deleted"}])
def test_generated_or_revoked_bridge_cannot_enter_graph_or_sources(fields):
    rows = chain()
    rows[1].update(fields)
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0]), source(rows[2], "S2")], rows)
    assert result["trace"]["bridge_source_ids"] == []


def test_ambiguous_duplicate_ids_fail_closed():
    rows = chain()
    rows.append(chunk("Unrelated reduces Gamma.", "bridge", "other"))
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0]), source(rows[2], "S2")], rows)
    assert result["trace"]["bridge_source_ids"] == []


def test_revoked_or_modified_seed_is_not_used_even_when_previously_cached():
    rows = chain()
    graph = BoundedEvidenceGraph()
    seeds = [source(rows[0])]
    graph.expand(QUESTION, seeds, rows)
    result = graph.expand(QUESTION, seeds, rows[1:])
    assert result["sources"] == []
    assert "graph_source_unavailable" in codes(result)
    assert "graph_no_source_seed" in codes(result)


def test_partial_revocation_cannot_make_new_citation_labels_collide():
    rows = chain()
    missing = source(chunk("Old source text.", "revoked", "revoked"))
    remaining = source(rows[2], "S2")
    result = BoundedEvidenceGraph().expand(QUESTION, [missing, remaining], rows)
    labels = [row["label"] for row in result["sources"]]
    assert len(labels) == len(set(labels))
    assert result["sources"][0] == remaining


def test_out_of_range_seed_offsets_are_rejected_instead_of_becoming_source_proof():
    rows = chain()
    invalid = source(rows[0])
    invalid["excerpt_end"] += 100
    result = BoundedEvidenceGraph().expand(QUESTION, [invalid], rows)
    assert result["sources"] == []
    assert "graph_source_unavailable" in codes(result)


def test_high_degree_hub_is_not_used_to_expand_background():
    rows = chain() + [chunk(f"Beta increases Entity{chr(65 + index)}.", f"n{index}", f"n{index}")
                      for index in range(module.MAX_DEGREE)]
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0]), source(rows[2], "S2")], rows)
    assert result["trace"]["bridge_source_ids"] == []
    assert "graph_hub_omitted" in codes(result)
    assert result["trace"]["search_states"] <= module.MAX_PATH_STATES


def test_corpus_window_omits_a_document_whole_and_reports_limitation(monkeypatch):
    rows = chain()
    for row in rows:
        row["document_id"] = "one-document"
    monkeypatch.setattr(module, "MAX_GRAPH_CHUNKS", 2)
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0]), source(rows[2], "S2")], rows)
    assert result["trace"]["edge_count"] == 0
    assert "graph_window_limited" in codes(result)


def test_input_cap_bypasses_graph_build_without_displacing_existing_evidence(monkeypatch):
    rows = chain()
    monkeypatch.setattr(module, "MAX_INPUT_CHUNKS", 2)
    graph = BoundedEvidenceGraph()
    monkeypatch.setattr(graph, "_prepare", lambda *_: pytest.fail("over-cap input should not be parsed"))
    result = graph.expand(QUESTION, [source(rows[0])], rows)
    assert result["sources"] == [source(rows[0])]
    assert "graph_window_limited" in codes(result)


def test_graph_metadata_does_not_copy_source_quotations_or_claim_entailment():
    rows = chain()
    graph = BoundedEvidenceGraph()
    result = graph.expand(QUESTION, [source(rows[0]), source(rows[2], "S2")], rows)
    trace = json.dumps(result["trace"], ensure_ascii=False)
    for row in rows:
        assert row["content"] not in trace
    assert all(not hasattr(edge, "quote") for edge in graph._graph.edges)
    assert all(path["epistemic_status"] != "proved" for path in result["trace"]["paths"])


@pytest.mark.parametrize("question,entity,expected", [
    ("ALPHA connects to Delta", "alpha", True),
    ("alphabets connect", "alpha", False),
    ("prealpha Alpha", "alpha", True),
    ("Alpha_1 connects", "alpha", False),
    ("Alpha2 connects", "alpha", False),
    ("2Alpha connects", "alpha", False),
    ("_Alpha connects", "alpha", False),
    ("中文Alpha connects", "alpha", False),
    ("Alpha中文 connects", "alpha", False),
    ("中文，Alpha。", "alpha", True),
    ("Solar-panel connectivity", "solar-panel", True),
    ("Solar panels connectivity", "solar panel", False),
    ("Solar Panel connectivity", "solar panel", True),
    ("a causal chain from A to D", "a", True),
    ("a causal chain from alpha to delta", "a", False),
    ("A_1 to D", "a", False),
    ("mİto and mıto", "mito", True),
    ("ſolar and Kelvin", "solar", True),
    ("ſolar and Kelvin", "kelvin", True),
    ("groß", "gross", False),
    ("能源安全提高", "能源安全", True),
    ("Β受体", "β受体", True),
])
def test_anchor_substring_search_preserves_unicode_word_and_identifier_boundaries(question, entity, expected):
    assert module._question_mentions(question, entity) is expected
    # Explicitly guard compatibility with the pre-optimization matcher.
    import re
    if entity.isascii():
        needle, flags = (entity.upper(), 0) if len(entity) == 1 and entity.isalpha() else (entity, re.I)
        previous = bool(re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", question, flags))
        assert previous is expected


def test_anchor_matching_does_not_compile_per_entity_patterns(monkeypatch):
    monkeypatch.setattr(module.re, "search", lambda *_args, **_kwargs: pytest.fail("per-entity regex compilation"))
    assert module._question_mentions("Alpha to Delta", "alpha")
    assert module._question_mentions("A to D", "a")


def test_new_long_window_can_be_used_as_a_seed_on_subsequent_requests():
    rows = chain()
    rows[1]["content"] = long_bridge("Beta reduces Gamma. No trials studied heat. ")
    graph = BoundedEvidenceGraph()
    first = graph.expand(QUESTION, [source(rows[0]), source(rows[2], "S2")], rows)
    second = graph.expand(QUESTION, first["sources"], rows)
    assert second["sources"] == first["sources"]
    assert "graph_source_unavailable" not in codes(second)
    assert second["trace"]["cache_hit"]


def test_pdf_soft_wrapped_relation_is_found_with_original_offsets_and_bytes():
    rows = chain()
    rows[1]["content"] = long_bridge("Beta reduces\r\nGamma under cold conditions. No trial measured heat. ")
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0]), source(rows[2], "S2")], rows)
    assert result["trace"]["bridge_source_ids"] == ["bridge"]
    added = result["sources"][-1]
    assert "Beta reduces\r\nGamma under cold conditions." in added["content"]
    assert "No trial measured heat." in added["content"]
    assert added["content"] == rows[1]["content"][added["excerpt_start"]:added["excerpt_end"]]


def test_soft_wrap_normalization_keeps_case_and_heading_boundaries():
    original = "Methods\nBeta reduces\nGamma.\nCase: cold\nGamma improves Delta."
    prepared = module._parser_rows([chunk(original)])[0]["content"]
    assert len(prepared) == len(original)
    assert prepared.startswith("Methods\nBeta reduces Gamma.")
    assert "Case: cold\nGamma improves Delta." in prepared


def test_soft_wrapped_negation_does_not_become_a_positive_relation():
    rows = chain()
    rows[1]["content"] = long_bridge("Beta does not\nreduce Gamma. ")
    result = BoundedEvidenceGraph().expand(QUESTION, [source(rows[0]), source(rows[2], "S2")], rows)
    assert result["trace"]["bridge_source_ids"] == ["bridge"]
    assert result["trace"]["paths"][0]["edges"][0]["polarity"] == -1
    assert "Beta does not\nreduce Gamma." in result["sources"][-1]["content"]
