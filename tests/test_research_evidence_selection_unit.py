"""Generic evidence-window retrieval and bounded coverage, no paper fixtures."""
import copy
import pytest

from docthinker.server.research_evidence_selection import prepare_evidence_candidates, select_evidence


def row(identifier, content, doc="d", ordinal=0, **extra):
    return {"id": identifier, "document_id": doc, "filename": doc + ".pdf",
            "content": content, "ordinal": ordinal, **extra}


def plan(*facets, docs=()):
    values = [{"id": key, "text": text, "kind": "clause", "weight": 1., "roles": roles,
               "document_ids": list(docs)} for key, text, roles in facets]
    return {"facets": values}


def prepared(question, chunks, facets, semantics=None):
    return prepare_evidence_candidates(question, chunks, [], facets, semantics or {})["candidates"]


def score_all(candidates, useful=()):
    return {row["id"]: (3. if any(word in row["content"] for word in useful) else 0.) for row in candidates}


def test_key_method_and_experiment_retrieved_over_dominant_background_and_references():
    facets = plan(("mechanism", "zephyr oscillator mechanism", ["mechanism"]),
                  ("setup", "independent sensor evaluation experiment", ["experiment_setup"]))
    chunks = [row("intro", "1 Introduction\nZephyr oscillator systems have wide applications. " * 12),
              row("method", "2 Methods\nThe zephyr oscillator mechanism computes a phase offset from adjacent samples."),
              row("setup", "3 Experiments\nThe independent sensor evaluation experiment sampled 60 sealed chambers."),
              row("refs", "References\n[12] Lane, A. 2021. Zephyr oscillator mechanism and sensor experiment.")]
    candidates = prepared("Explain zephyr oscillator mechanism and independent sensor evaluation experiment", chunks, facets)
    assert not any(item["source_id"] == "refs" for item in candidates)
    result = select_evidence(candidates, score_all(candidates, ("phase offset", "60 sealed")), facets, [])
    assert {item["id"] for item in result["sources"]} >= {"method", "setup"}
    assert all(item["id"] != "intro" for item in result["sources"])


def test_semantic_multi_facet_locates_different_exact_windows_in_large_chunk():
    text = "Broad unrelated background occupies this first sentence. " + "Filler context occupies another sentence. " * 10
    text += "The controller rotates its buffer clockwise. " + "Generic prose unrelated to either question. " * 8
    text += "The experiment samples 80 independent basins."
    first, second = text.index("The controller"), text.index("The experiment")
    facets = plan(("mechanism", "控制器如何工作", ["mechanism"]), ("setup", "实验怎么采样", ["experiment_setup"]))
    semantics = {"mechanism": {"scores": {"long": .9}, "spans": {"long": [first, first + 43]}},
                 "setup": {"scores": {"long": .91}, "spans": {"long": [second, len(text)]}}}
    candidates = prepared("控制器和实验", [row("long", text)], facets, semantics)
    assert any("rotates" in item["content"] and item["facet_scores"].get("mechanism", 0) > 0 for item in candidates)
    assert any("80 independent" in item["content"] and item["facet_scores"].get("setup", 0) > 0 for item in candidates)
    for item in candidates:
        assert item["content"] == text[item["excerpt_start"]:item["excerpt_end"]]
        assert 0 <= item["excerpt_start"] < item["excerpt_end"] <= len(text)
        assert len(item["content"]) <= 1600
        assert "rotates" in item["content"] or "80 independent" in item["content"]


def test_qualification_negation_units_never_cut():
    text = "The cobalt filter improves recall. However, it does not improve precision unless the index is sparse."
    facets = plan(("result", "cobalt filter recall precision", ["result", "limitation"]))
    candidates = prepared("cobalt filter recall precision", [row("claim", text)], facets)
    assert candidates
    assert all("does not" in item["content"] and "unless" in item["content"] for item in candidates)


def test_named_three_document_coverage_when_relevant_not_three_copies_of_intro():
    facets = plan(("method", "buffer rotation algorithm", ["mechanism"]), docs=("a", "b", "c"))
    chunks = [row(doc, f"Methods\nThe buffer rotation algorithm uses {detail}.", doc=doc)
              for doc, detail in [("a", "cyclic phase offsets"), ("b", "a parity gate"), ("c", "a sparse lookup")]]
    candidates = prepared("compare buffer rotation algorithm", chunks, facets)
    result = select_evidence(candidates, {item["id"]: 3. for item in candidates}, facets, [])
    assert {item["document_id"] for item in result["sources"]} == {"a", "b", "c"}
    assert result["trace"]["named_documents_missing"] == 0


def test_no_matching_material_is_empty_not_context_padding():
    facets = plan(("m", "cobalt phase estimation", ["mechanism"]))
    candidates = prepared("cobalt phase estimation", [row("x", "The orchard produces apples in spring.")], facets)
    assert candidates == []
    result = select_evidence(candidates, {}, facets, [])
    assert result["sources"] == []
    assert result["gaps"][0]["code"] == "insufficient_evidence"


def test_noise_and_same_evidence_repetition_do_not_fill_eight_sources():
    facets = plan(("m", "cobalt phase estimation", ["mechanism"]))
    chunks = [row(str(i), "The cobalt phase estimation algorithm uses a bounded Fourier transform.") for i in range(12)]
    candidates = prepared("cobalt phase estimation", chunks, facets)
    result = select_evidence(candidates, {item["id"]: 3. for item in candidates}, facets, [])
    assert len(result["sources"]) == 1


def test_baseline_noise_can_be_replaced_not_forcibly_retained():
    facets = plan(("m", "glacial buffer algorithm", ["mechanism"]))
    chunks = [row("new", "The glacial buffer algorithm hashes its clock to select the correct ring.")]
    old = [row("old", "Generic research may be useful to many readers.")]
    candidates = prepare_evidence_candidates("glacial buffer algorithm", chunks, old, facets, {})["candidates"]
    result = select_evidence(candidates, score_all(candidates), facets, old)
    assert result["trace"]["removed_source_ids"] == ["old"]
    assert result["trace"]["added_source_ids"] == ["new"]


def test_generated_deleted_and_revoked_sources_never_return_from_old_semantics():
    facets = plan(("m", "glacial buffer algorithm", ["mechanism"]))
    chunks = [row("live", "Glacial buffer algorithm uses a parity gate."),
              row("fake", "Glacial buffer algorithm uses magic.", is_generated=True),
              row("deleted", "Glacial buffer algorithm uses forgotten code.", status="deleted"),
              row("revoked", "Glacial buffer algorithm uses revoked code.", status="revoked")]
    semantic = {"m": {"scores": {"removed": .99, "fake": .99}, "spans": {"removed": [0, 20], "fake": [0, 20]}}}
    candidates = prepared("glacial buffer algorithm", chunks, facets, semantic)
    assert {item["source_id"] for item in candidates} == {"live"}


def test_section_role_carries_within_document_not_to_other_document():
    facets = plan(("m", "crimson sampling", ["experiment_setup"]))
    chunks = [row("head", "4 Experiments", ordinal=0),
              row("body", "Crimson sampling selects ten independent chambers.", ordinal=1),
              row("other", "Crimson sampling controls a pressure valve.", doc="other", ordinal=2)]
    candidates = prepared("crimson sampling", chunks, facets)
    body = next(item for item in candidates if item["source_id"] == "body")
    other = next(item for item in candidates if item["source_id"] == "other")
    assert "experiment_setup" in body["roles"]
    assert other["section"] == ""


def test_references_explicit_request_bypasses_reference_filter():
    facets = plan(("r", "references coral index", []))
    text = "References\n[12] Lane, A. 2024. Coral index. Journal of fictional methods."
    assert prepared("Show reference list for coral index", [row("refs", text)], facets)
    assert not prepared("Explain coral index", [row("refs", text)], facets)


def test_numeric_table_fragments_penalized_unless_requested():
    facets = plan(("m", "coral index", []))
    text = "coral index 0.82 0.45 0.35 0.82 0.45 0.35 0.82 0.45 0.35"
    ordinary = prepared("coral index", [row("table", text)], facets)
    requested = prepared("Show coral index table numbers", [row("table", text)], facets)
    assert ordinary[0]["pool_score"] < requested[0]["pool_score"]


def test_adjacent_model_labels_are_not_a_method_explanation():
    facets = plan(("m", "ALPHA-Seq ALPHA-Tok", ["mechanism"]))
    chunk = row("labels", "ALPHA-Seq\nALPHA-Tok")
    assert not prepared("Explain ALPHA-Seq ALPHA-Tok mechanisms", [chunk], facets)
    assert prepared("Show ALPHA-Seq ALPHA-Tok table numbers", [chunk], facets)


def test_same_chunk_outputs_unique_original_id_and_exact_contiguous_quote():
    text = "The copper algorithm rotates the gate. The sensor experiment samples independent basins."
    facets = plan(("m", "copper algorithm rotates gate", ["mechanism"]),
                  ("e", "sensor experiment independent basins", ["experiment_setup"]))
    candidates = prepared("copper algorithm sensor experiment", [row("one", text)], facets)
    result = select_evidence(candidates, {item["id"]: 2. for item in candidates}, facets, [])
    assert len(result["sources"]) == 1
    source = result["sources"][0]
    assert source["id"] == "one"
    assert source["content"] == text[source["excerpt_start"]:source["excerpt_end"]]
    assert "rotates" in source["content"] and "basins" in source["content"]


def test_invalid_span_or_tampered_candidate_cannot_gain_a_citation():
    facets = plan(("m", "coral algorithm", []))
    candidates = prepared("coral algorithm", [row("one", "The coral algorithm rotates its index.")], facets)
    tampered = copy.deepcopy(candidates)
    tampered[0]["content"] = "A fabricated conclusion."
    assert select_evidence(tampered, {}, facets, [])["sources"] == []
    semantic = {"m": {"scores": {"one": .99}, "spans": {"one": [-2, 999]}}}
    assert prepared("珊瑚算法", [row("one", "Unrelated apple orchard.")], plan(("m", "珊瑚算法", [])), semantic) == []


def test_hard_utf8_budget_and_stable_order():
    facets = plan(("m", "量子索引检索方法", ["mechanism"]), docs=("a", "b"))
    chunks = [row("a", "量子索引检索方法首先计算键值，然后执行循环。", doc="a"),
              row("b", "量子索引检索方法使用另一种奇偶校验。", doc="b")]
    first = prepared("量子索引检索方法", chunks, facets)
    second = prepared("量子索引检索方法", chunks[::-1], facets)
    assert first == second
    scores = {item["id"]: 1. for item in first}
    result = select_evidence(first, scores, facets, [], max_source_bytes=70)
    assert sum(len(item["content"].encode("utf-8")) for item in result["sources"]) <= 70
    assert result == select_evidence(first, scores, facets, [], max_source_bytes=70)


def test_candidate_cap_and_named_document_nominations():
    facets = plan(("m", "coral algorithm", ["mechanism"]), docs=("other",))
    chunks = [row(str(i), f"Coral algorithm version {i} computes a hash.") for i in range(80)]
    chunks.append(row("last", "Coral algorithm computes an independent gate.", doc="other"))
    result = prepare_evidence_candidates("coral algorithm", chunks, [], facets, {}, max_candidates=8)
    assert len(result["candidates"]) <= 8
    assert any(item["document_id"] == "other" for item in result["candidates"])


def test_oversized_indivisible_sentence_is_omitted_not_cut():
    text = "The coral algorithm " + "x " * 1000 + "does not improve recall."
    facets = plan(("m", "coral algorithm", []))
    assert prepared("coral algorithm", [row("one", text)], facets) == []


def test_requesting_citations_does_not_admit_bibliography_as_answer_evidence():
    facets = plan(("m", "coral buffer mechanism", ["mechanism"]))
    chunks = [row("refs", "References\n[12] Lane, A. 2024. Coral buffer mechanism. Journal of fictional methods.")]
    for query in ("Explain coral buffer mechanism with citations", "Explain coral buffer mechanism and give references",
                  "解释 coral buffer mechanism，请给引用", "解释 coral buffer mechanism，提供参考文献"):
        assert prepared(query, chunks, facets) == []
    for query in ("List references about coral buffer mechanism", "Show bibliography for coral buffer mechanism",
                  "列出参考文献 coral buffer mechanism", "查看参考文献列表 coral buffer mechanism"):
        assert prepared(query, chunks, facets)


def test_multiple_windows_cannot_crowd_out_distinct_source_nominations():
    facets = plan(("m", "coral buffer mechanism", ["mechanism"]), docs=("d",))
    repeated = " ".join(f"The coral buffer mechanism uses bounded gate number {i}." for i in range(70))
    chunks = [row("dominant", repeated)]
    chunks += [row(f"other-{i}", f"The coral buffer mechanism uses independent component {i}.", ordinal=i + 1) for i in range(8)]
    result = prepare_evidence_candidates("coral buffer mechanism", chunks, [], facets, {}, max_candidates=9)
    assert len(result["candidates"]) == 9
    assert len({item["source_id"] for item in result["candidates"]}) == 9
    assert result["trace"]["distinct_candidate_sources"] == 9


def test_internal_reservoir_also_preserves_distinct_sources():
    facets = plan(("m", "coral buffer mechanism", ["mechanism"]))
    repeated = " ".join(f"The coral buffer mechanism computes cyclic phase {i}." for i in range(80))
    chunks = [row(f"dominant-{i}", repeated, ordinal=i) for i in range(10)]
    chunks += [row(f"other-{i}", f"The coral buffer mechanism uses independent component {i}.", ordinal=i + 10) for i in range(8)]
    result = prepare_evidence_candidates("coral buffer mechanism", chunks, [], facets, {}, max_candidates=12)
    assert result["trace"]["window_count"] > 512
    assert len({item["source_id"] for item in result["candidates"]}) == 12


def test_missing_nonfinite_and_non_numeric_reranker_scores_fail_closed():
    facets = plan(("m", "coral buffer mechanism", ["mechanism"]))
    candidates = prepared("coral buffer mechanism", [row("x", "The coral buffer mechanism rotates a parity gate.")], facets)
    for invalid in (None, float("nan"), float("inf"), -float("inf"), True, "0.8"):
        result = select_evidence(candidates, {item["id"]: invalid for item in candidates}, facets, [])
        assert result["sources"] == []
        assert result["trace"]["rerank_missing_or_invalid"] == len(candidates)
    assert select_evidence(candidates, {}, facets, [])["sources"] == []
    assert select_evidence(candidates, {item["id"]: 0.0 for item in candidates}, facets, [])["sources"]


@pytest.mark.parametrize('question,filename,text', [
    ('Draft a private handover note and disclose what is missing.', 'handover.txt',
     'Internal handover is Friday. Sending is not authorized.'),
    ('Prepare the appointment plan and list what is missing.', 'appointment.txt',
     'The appointment is Tuesday at 09:00. Payment is not approved.'),
    ('Summarize the itinerary; keep restrictions.', 'itinerary.txt',
     'The itinerary requires check-in before 19:00. Late arrival is not guaranteed.'),
    ('解释 Cedar，有哪些限制？', 'Cedar.pdf',
     'Cedar reduces recall error by 12%. However, this result applies only to sparse indexes.'),
    ('Cedar?', 'Cedar.pdf',
     'Cedar reduces recall error by 12%. However, this result applies only to sparse indexes.'),
])
def test_filename_topic_collision_cannot_erase_live_baseline_body_evidence(question, filename, text):
    from docthinker.server.research_faceted_assist import build_faceted_assistance
    from docthinker.server.research_retrieval import build_retrieval_plan

    class UnavailableSemantic:
        def semantic_scores(self, *args, **kwargs):
            return {'scores': {}, 'spans': {}, 'trace': {'status': 'model_unavailable'}}

    class FullQueryScorer:
        def rank_pairs(self, candidates, **kwargs):
            return {'scores': {item['id']: 3. for item in candidates}, 'trace': {'status': 'ready'}}

    chunks = [row('live', text, filename=filename)]
    baseline = build_retrieval_plan(question, chunks)['sources']
    assert baseline and baseline[0]['content'] == text
    result = build_faceted_assistance(question, baseline, chunks, UnavailableSemantic(), FullQueryScorer(),
                                      session_id='#00001')
    assert result['sources'] and result['sources'][0]['id'] == 'live'
    assert result['sources'][0]['content'] == text
    assert result['trace']['candidate_pool']['candidate_count'] >= 1
    assert 0 <= result['trace']['candidate_pool']['baseline_witness_nominations'] <= result['trace']['candidate_pool']['candidate_count']
    assert result['trace']['model_calls'] == result['trace']['provider_calls'] == 0
    assert sum(len(item['content'].encode()) for item in result['sources']) <= 12000


@pytest.mark.parametrize('text', [
    'References\n[12] Lane, A. 2024. Cedar architecture. Journal of fictional methods.',
    'Cedar',
    'Cedar University\ncontact: cedar@example.com',
    'The orchard produces apples in spring.',
])
def test_filename_baseline_rescue_never_bypasses_body_or_metadata_hygiene(text):
    from docthinker.server.research_query_facets import build_query_facets
    question = 'Explain the Cedar architecture.'
    chunk = row('old', text, filename='Cedar.pdf')
    facets = build_query_facets(question, [chunk])
    baseline = [{**chunk, 'excerpt_start': 0, 'excerpt_end': len(text)}]
    result = prepare_evidence_candidates(question, [chunk], baseline, facets, {})
    assert not any(item.get('baseline_witness_nomination') for item in result['candidates'])


@pytest.mark.parametrize('mutation', ['body', 'offset', 'document', 'revoked'])
def test_baseline_witness_rescue_requires_exact_current_scoped_original(mutation):
    from docthinker.server.research_query_facets import build_query_facets
    question = 'Draft the handover note.'
    text = 'The handover is Friday. Sending is not authorized.'
    live = row('live', text, filename='handover.txt')
    baseline = {**live, 'excerpt_start': 0, 'excerpt_end': len(text)}
    if mutation == 'body':
        baseline['content'] = 'A corrupted baseline quote.'
    elif mutation == 'offset':
        baseline['excerpt_start'] = 1
    elif mutation == 'document':
        baseline['document_id'] = 'another'
    else:
        live['status'] = 'revoked'
    facets = build_query_facets(question, [live])
    result = prepare_evidence_candidates(question, [live], [baseline], facets, {})
    assert not any(item.get('baseline_witness_nomination') for item in result['candidates'])


def test_explicit_anchor_detail_coverage_adds_mechanisms_after_generic_summary():
    facets = plan(("broad", "amber-lock routing frost-gate scheduling", ["mechanism"]))
    facets["facets"][0]["anchors"] = ["amber-lock routing", "frost-gate scheduling"]
    chunks = [row("summary", "This report presents amber-lock routing and frost-gate scheduling as major topics."),
              row("routing", "Amber-lock routing computes a parity bit before it selects the destination ring."),
              row("schedule", "Frost-gate scheduling executes the waiting callback only when both queues are idle.")]
    candidates = prepared("Explain amber-lock routing and frost-gate scheduling", chunks, facets)
    result = select_evidence(candidates, {item["id"]: 3. for item in candidates}, facets, [])
    assert {item["id"] for item in result["sources"]} >= {"routing", "schedule"}
    assert len(result["trace"]["anchor_coverage"]) == 2
    assert all(value >= .8 for value in result["trace"]["anchor_coverage"].values())


def test_document_scoped_facets_cannot_be_satisfied_by_another_paper():
    facets = plan(("a-method", "coral buffer mechanism", ["mechanism"]))
    facets["facets"][0]["document_ids"] = ["a"]
    chunks = [row("a-source", "Coral buffer mechanism rotates a parity gate.", doc="a"),
              row("b-source", "Coral buffer mechanism rotates a parity gate and updates its index.", doc="b")]
    candidates = prepared("coral buffer mechanism", chunks, facets)
    assert {item["document_id"] for item in candidates} == {"a"}


def test_relative_rerank_scores_are_grouped_by_query_not_global_scale():
    facets = plan(("a-method", "coral buffer mechanism", ["mechanism"]),
                  ("b-method", "amber clock update", ["mechanism"]))
    chunks = [row("a", "Coral buffer mechanism rotates a gate.", doc="a"),
              row("b", "Amber clock update computes a parity bit.", doc="b")]
    candidates = prepared("coral buffer mechanism and amber clock update", chunks, facets)
    for item in candidates:
        item["rerank_facet_id"] = item["document_id"]
    scores = {item["id"]: 12. if item["document_id"] == "a" else -12. for item in candidates}
    result = select_evidence(candidates, scores, facets, [])
    assert {item["id"] for item in result["sources"]} == {"a", "b"}
    assert result["trace"]["rerank_query_group_count"] == 2


def test_missing_named_document_and_proxy_limits_are_always_explicit():
    facets = plan(("m", "coral buffer mechanism", ["mechanism"]), docs=("a", "missing"))
    candidates = prepared("compare coral buffer mechanism", [row("a", "Coral buffer mechanism rotates a gate.", doc="a")], facets)
    result = select_evidence(candidates, score_all(candidates), facets, [])
    assert {gap["code"] for gap in result["gaps"]} >= {"documents_not_selected", "evidence_coverage_proxy"}
    assert result["trace"]["named_documents_missing"] == 1


def test_document_titles_do_not_become_technical_anchor_gain():
    facets = plan(("m", "Crimson Atlas coral routing", ["mechanism"]))
    facets["facets"][0]["anchors"] = ["Crimson Atlas", "coral routing"]
    facets["facets"].append({"id": "doc", "text": "Crimson Atlas", "kind": "document", "weight": .45, "document_ids": ["a"]})
    candidates = prepared("Crimson Atlas coral routing", [row("a", "Coral routing computes a parity bit.", doc="a")], facets)
    result = select_evidence(candidates, score_all(candidates), facets, [])
    assert len(result["trace"]["anchor_coverage"]) == 1


def test_adjacent_whitespace_gap_can_merge_but_quote_remains_exact():
    text = "Coral routing computes a parity bit.   Frost gating executes the pending callback."
    facets = plan(("a", "coral routing", ["mechanism"]), ("b", "frost gating", ["mechanism"]))
    facets["facets"][0]["anchors"] = ["coral routing"]
    facets["facets"][1]["anchors"] = ["frost gating"]
    # Simulate independently localized complete statements. Candidate building
    # now prefers their maximal combined window; selection must still safely
    # accept separately localized windows from a scorer/cached candidate pool.
    spans = [(0, text.index(".") + 1), (text.index("Frost"), len(text))]
    singles = []
    for facet, (start, end) in zip(facets["facets"], spans):
        item = prepared(facet["text"], [row("same", text[start:end])], {"facets": [facet]})[0]
        item.update(_source_content=text, excerpt_start=start, excerpt_end=end, original_content_length=len(text))
        singles.append(item)
    result = select_evidence(singles, score_all(singles), facets, [])
    assert len(result["sources"]) == 1
    assert result["sources"][0]["content"] == text


def test_each_named_document_requires_its_own_method_and_experiment_evidence():
    facets = plan(("method", "coral buffer mechanism", ["mechanism"]),
                  ("experiment", "independent basin experiment", ["experiment_setup"]), docs=("a", "b"))
    chunks = [row("a-method", "Coral buffer mechanism rotates a parity gate.", doc="a"),
              row("b-method", "Coral buffer mechanism computes a cyclic offset.", doc="b"),
              row("a-experiment", "The independent basin experiment sampled 20 enclosed chambers.", doc="a"),
              row("b-experiment", "The independent basin experiment sampled 40 sealed containers.", doc="b")]
    candidates = prepared("compare methods and experiments", chunks, facets)
    result = select_evidence(candidates, score_all(candidates), facets, [])
    assert {item["id"] for item in result["sources"]} == {item["id"] for item in chunks}
    assert result["trace"]["independent_document_requirements"] == 4
    assert not result["trace"]["uncovered_facets"]


def test_missing_requirement_in_one_document_is_not_hidden_by_another():
    facets = plan(("method", "coral buffer mechanism", ["mechanism"]), docs=("a", "b"))
    candidates = prepared("compare coral buffer mechanism", [row("a", "Coral buffer mechanism rotates a gate.", doc="a")], facets)
    result = select_evidence(candidates, score_all(candidates), facets, [])
    assert result["trace"]["facet_coverage"]["method"] == 0.
    assert result["trace"]["uncovered_facets"] == ["method"]


def test_reference_section_stays_locked_across_author_and_title_like_headings():
    facets = plan(("method", "coral buffer mechanism", ["mechanism"]))
    chunks = [row("reference-start", "References\nAnne Frost and Bruno Moss\nCoral Buffer Mechanism\nA study of coral buffer mechanism. Published in 2024.", ordinal=0),
              row("reference-continuation", "Dina Snow\nCoral Buffer Mechanism Revisited\nCoral buffer mechanism rotates a magic wheel.", ordinal=1),
              row("appendix", "Appendix A Implementation\nThe coral buffer mechanism computes a cyclic index.", ordinal=2)]
    candidates = prepared("explain coral buffer mechanism with citations", chunks, facets)
    assert {item["source_id"] for item in candidates} == {"appendix"}


def test_numbered_chapter_can_end_reference_lock_but_other_document_is_independent():
    facets = plan(("method", "coral buffer mechanism", ["mechanism"]))
    chunks = [row("reference-start", "References\nCoral Buffer Mechanism\nCoral buffer mechanism journal article.", doc="a", ordinal=0),
              row("new-section", "8 Additional Experiments\nThe coral buffer mechanism sampled independent chambers.", doc="a", ordinal=1),
              row("other-document", "The coral buffer mechanism computes a parity bit.", doc="b", ordinal=0)]
    candidates = prepared("coral buffer mechanism", chunks, facets)
    assert {item["source_id"] for item in candidates} == {"new-section", "other-document"}


def test_maximal_context_window_keeps_mechanism_after_cheap_definition():
    text = ("Coral routing is a message forwarding mechanism. "
            "A parity flag selects the next destination ring. "
            "Otherwise the queue yields control to the caller. "
            "The design has a bounded pending queue.")
    facets = plan(("method", "coral routing mechanism", ["mechanism"]))
    candidates = prepared("how does coral routing work", [row("mechanism", text)], facets)
    starts = [item["excerpt_start"] for item in candidates]
    assert len(starts) == len(set(starts))
    first = next(item for item in candidates if item["excerpt_start"] == 0)
    assert "parity flag" in first["content"] and "yields control" in first["content"]
    assert "bounded pending" in first["content"]
    assert first["content"] == text
    assert first["content"] == text[first["excerpt_start"]:first["excerpt_end"]]


def test_context_windows_shorten_at_heading_and_size_boundary_without_truncation():
    facets = plan(("method", "coral routing mechanism", ["mechanism"]))
    text = ("1 Methods\nCoral routing mechanism computes a parity flag.\n"
            "2 Results\nCoral routing mechanism sampled independent chambers.")
    result = prepare_evidence_candidates("coral routing mechanism", [row("one", text)], [], facets, {})
    assert result["trace"]["window_context_is_verified_support"] is False
    assert all(not ("parity flag" in item["content"] and "sampled" in item["content"]) for item in result["candidates"])
    lengthy = "Coral routing mechanism " + "x " * 500 + ". " + "Coral routing mechanism " + "y " * 500 + "."
    candidates = prepared("coral routing mechanism", [row("two", lengthy)], facets)
    assert candidates and all(len(item["content"]) <= 1600 for item in candidates)
    assert all(item["content"] == lengthy[item["excerpt_start"]:item["excerpt_end"]] for item in candidates)


def exact_candidate(identifier, content, scores, scored_facet, *, doc="d"):
    """Synthetic exact spans isolate packing from learned-model judgments."""
    return {**row(identifier, content, doc=doc), "source_id": identifier,
            "_source_content": content, "excerpt_start": 0, "excerpt_end": len(content),
            "original_content_length": len(content), "offset_unit": "characters_in_source_chunk",
            "sentence_boundary_preserved": True, "facet_scores": scores, "anchor_scores": {},
            "query": "Literal question for " + scored_facet, "rerank_facet_id": scored_facet}


def test_broad_overview_cannot_zero_out_complementary_scored_evidence():
    facets = plan(("how", "How does Atlas operate?", ["mechanism"]),
                  ("when", "What conditions restrict Atlas?", ["limitation"]))
    candidates = [
        exact_candidate("overview", "Atlas has an efficient method and documented operating conditions.",
                        {"how": .95, "when": .95}, "how"),
        exact_candidate("procedure", "Encode each segment, sort the keys, and scan adjacent entries.",
                        {"how": .85, "when": .10}, "how"),
        exact_candidate("conditions", "Unique sorted keys are required; duplicates must first be reduced.",
                        {"how": .10, "when": .85}, "when"),
    ]
    result = select_evidence(candidates, score_all(candidates), facets, [])
    assert {item["id"] for item in result["sources"]} == {"overview", "procedure", "conditions"}
    assert result["trace"]["depth_discounts"] == []
    assert result["trace"]["length_policy"] == "hard_budget_only"
    assert result["trace"]["pair_credit"] == "one_scored_query_facet_not_broadcast"
    assert result["trace"]["selection_proxy_is_verified_support"] is False
    for source in result["sources"]:
        original = next(item for item in candidates if item["id"] == source["id"])
        assert source["content"] == original["_source_content"][source["excerpt_start"]:source["excerpt_end"]]


def test_unscored_clause_remains_uncovered_even_when_lexical_proxy_is_high():
    facets = plan(("how", "How does Atlas operate?", []), ("when", "What conditions restrict Atlas?", []))
    candidate = exact_candidate("overview", "Atlas includes methods and operating conditions.",
                                {"how": .95, "when": .95}, "how")
    result = select_evidence([candidate], {candidate["id"]: 0.}, facets, [])
    assert result["trace"]["facet_coverage"] == {"how": .95, "when": 0.}
    assert result["trace"]["uncovered_facets"] == ["when"]


def test_duplicate_and_near_duplicate_windows_do_not_earn_additional_rank_utility():
    facets = plan(("how", "How does Atlas operate?", []))
    content = "Encode sort scan compare reduce normalize align merge compute rotate append return."
    candidates = [exact_candidate("first", content, {"how": .9}, "how"),
                  exact_candidate("copy", content.upper(), {"how": .9}, "how"),
                  exact_candidate("near", content + " Finish.", {"how": .9}, "how")]
    result = select_evidence(candidates, score_all(candidates), facets, [])
    assert len(result["sources"]) == 1
    assert result["trace"]["depth_objective"] == {"how": 1.}
    assert result["trace"]["depth_objective_semantics"] == "sum_rank_utility_not_selection_cap"


def test_bridging_near_duplicate_cannot_reduce_depth_objective():
    from docthinker.server.research_evidence_selection import _depth_value, _duplicate_window

    common = ["token" + str(index) for index in range(18)]
    specifications = [("left", common + ["alpha", "beta"], .8),
                      ("right", common + ["gamma", "delta"], .8),
                      ("bridge", common + ["alpha", "beta", "gamma", "delta"], .85),
                      ("independent", ["separate", "construction", "procedure"], .7)]
    candidates = []
    for identifier, tokens, score in specifications:
        candidate = exact_candidate(identifier, " ".join(tokens), {"how": score}, "how")
        candidate.update(_tokens=set(tokens), _coverage={"how": score})
        candidates.append(candidate)
    left, right, bridge, independent = candidates
    assert not _duplicate_window(left, right)
    assert _duplicate_window(left, bridge) and _duplicate_window(right, bridge)
    for discounts, original in [((1., .4), [left, right]),
                                ((1., .4, .2), [left, right, independent])]:
        before = _depth_value(original, "how", discounts)
        assert _depth_value([*original, bridge], "how", discounts) == before
        assert _depth_value([bridge, *reversed(original)], "how", discounts) == before


def test_complementary_ranked_evidence_is_bounded_by_sources_and_bytes_not_fixed_depth():
    facets = plan(("how", "How does Atlas operate?", []))
    contents = ["Encode segments using binary keys.", "Rotate the destination ring clockwise.",
                "A sparse queue stores pending messages.", "Discard duplicate handles before sorting.",
                "Independent workers maintain local counters."]
    candidates = [exact_candidate(str(i), text, {"how": .8}, "how") for i, text in enumerate(contents)]
    result = select_evidence(candidates, score_all(candidates), facets, [])
    assert {source["id"] for source in result["sources"]} == {candidate["id"] for candidate in candidates}
    assert result["trace"]["depth_objective"] == {"how": 5.}
    limited = select_evidence(candidates, score_all(candidates), facets, [], max_sources=1, max_source_bytes=50)
    assert len(limited["sources"]) <= 1
    assert limited["trace"]["source_bytes"] <= 50


def test_eligible_baseline_nominations_stay_inside_cap_without_forcing_output():
    facets = plan(("method", "cobalt phase estimation", ["mechanism"]))
    chunks = [row(str(i), f"The cobalt phase estimation algorithm computes independent key {i}.") for i in range(80)]
    baseline = [row("baseline-" + str(i), "Cobalt applies sparse gating to a stable ring with an independent path.")
                for i in range(3)]
    chunks.extend(baseline)
    chunks.append(row("revoked", "The cobalt phase estimation algorithm computes a key.", status="revoked"))
    pool = prepare_evidence_candidates("cobalt phase estimation", chunks, baseline + [chunks[-1]], facets, {})
    assert len(pool["candidates"]) == 48
    assert pool["trace"]["baseline_nominations"] == 3
    assert pool["trace"]["baseline_nominations_force_selection"] is False
    identifiers = {item["source_id"] for item in pool["candidates"]}
    assert {"baseline-0", "baseline-1", "baseline-2"} <= identifiers
    assert "revoked" not in identifiers
    # Candidate nomination is not final relevance. Give the fake reranker an
    # explicit preference for the actual mechanism instead of requiring a
    # rank-based selector to infer semantic distinctions from identical logits.
    result = select_evidence(pool["candidates"], score_all(pool["candidates"], ("independent key",)), facets, baseline)
    assert not any(item["id"].startswith("baseline-") for item in result["sources"])


def test_anchor_diagnostics_remain_separate_from_scored_facet_depth():
    facets = plan(("how", "How does amber-lock routing work?", []))
    facets["facets"][0]["anchors"] = ["amber-lock routing"]
    candidate = exact_candidate("weak", "A topical overview does not state the operation.", {"how": .8}, "how")
    result = select_evidence([candidate], {candidate["id"]: 0.}, facets, [])
    assert result["trace"]["uncovered_anchors"]
    assert set(result["trace"]["depth_objective"]) == {"how"}
    assert result["trace"]["anchor_coverage_semantics"] == "lexical_detail_proxy_not_scored_coverage"
