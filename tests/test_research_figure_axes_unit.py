"""Fictional PDF geometry fixtures; no model, API, paper values or scoring tune."""
import pytest

from docthinker.server.research_evidence_selection import prepare_evidence_candidates, select_evidence
from docthinker.server.research_figure_axes import pure_axis_ticks, requests_graph_data
from docthinker.server.research_query_facets import build_query_facets
from docthinker.server.research_sentence_windows import evidence_units, is_section_heading, sentence_spans


TICKS = "Time (s)\n" + "\n".join(str(value) for value in range(12)) + "\n"
CAPTION = "Figure 3: The cobalt controller uses a bounded sampling queue."
BODY = "The controller transmits the queued measurements after processing completes."
QUESTION = "Explain how the cobalt controller processes sampling measurements."


def pool(content, question=QUESTION):
    chunks = [{"id": "chunk", "document_id": "fictional", "filename": "fictional.pdf",
               "ordinal": 0, "content": content}]
    plan = build_query_facets(question, chunks)
    # Synthetic retrieval scores only nominate exact untouched source spans.
    semantic = {facet["id"]: {"scores": {"chunk": .9}, "spans": {"chunk": [0, len(content)]}}
                for facet in plan["facets"]}
    events = []
    result = prepare_evidence_candidates(question, chunks, [], plan, semantic, diagnostic_callback=events.append)
    return result, plan, events


@pytest.mark.parametrize("marker", ["Figure 3:", "Figure 3.", "Fig. 3:", "Fig. 3.", "Figure S3:", "Fig. A.3:"])
def test_line_leading_caption_is_a_boundary_but_never_deleted_as_heading(marker):
    caption = marker + " Cobalt Controller Queue"
    text = TICKS + caption + "\n" + BODY
    spans = sentence_spans(text)
    units = evidence_units(text)
    assert "".join(text[start:end] for start, end in spans) == text
    assert "".join(text[start:end] for start, end in units) == text
    assert any(start == len(TICKS) for start, _ in units)
    assert not is_section_heading(caption)
    assert not any(text[start:end].strip() == marker for start, end in units)


def test_inline_figure_reference_is_not_an_axis_or_caption_boundary():
    text = "The measurements in Figure 3: calibration detail remain valid at 0.25 units."
    assert evidence_units(text) == [(0, len(text))]
    dotted_reference = "Figure 3.5 shows the controller at 0.25 units."
    assert evidence_units(dotted_reference) == [(0, len(dotted_reference))]


@pytest.mark.parametrize("ticks", [
    TICKS,
    "Time (s)\n0 1 2 3 4 5 6 7 8 9\n",
    "Time (s)\n-8\n-7\n-6\n-5\n-4\n-3\n-2\n-1\n",
    "Error (%)\n0.0\n0.1\n0.2\n0.3\n0.4\n0.5\n0.6\n0.7\n",
])
def test_dense_pure_ticks_are_filtered_only_in_explicit_caption_context(ticks):
    assert pure_axis_ticks(ticks, immediately_before_caption=True)
    assert not pure_axis_ticks(ticks, immediately_before_caption=False)
    content = ticks + CAPTION + "\n" + BODY
    result, plan, events = pool(content)
    assert result["trace"]["axis_tick_units_filtered"] == 1
    assert result["candidates"]
    assert all(row["excerpt_start"] >= len(ticks) for row in result["candidates"])
    assert any(CAPTION in row["content"] and BODY in row["content"] for row in result["candidates"])
    assert any(event.get("reason") == "pure_axis_ticks_before_caption" for event in events)
    selected = select_evidence(result["candidates"], {row["id"]: 1. for row in result["candidates"]}, plan, [])
    assert selected["sources"]
    for source in selected["sources"]:
        assert source["label"].startswith("S") and CAPTION in source["content"]
        assert content[source["excerpt_start"]:source["excerpt_end"]] == source["content"]


@pytest.mark.parametrize("data", [
    "Temperature Pressure\n" + "\n".join(f"{i} {i + 1}" for i in range(12)),
    "Table 2: Numerical observations\n" + "\n".join(str(i) for i in range(12)),
    "\n".join(f"v{i} = {i / 10}" for i in range(12)),
    "\n".join(f"{i} ms" for i in range(12)),
    "\n".join(f"-{i} kg" for i in range(12)),
    "\n".join(f"{i}. Calibrate the sensor" for i in range(1, 10)),
    "\n".join(f"-{i} -{i + 1}" for i in range(12)),
    "0 2 1 3 7 4 9 5 8",
    "0 1 2 3 4 5 6 7\n8 9 10 11 12 13 14 15",
    "The measured delay was 0.25 ms and variance was -0.125 units.",
    TICKS + "The sample contains independent observed values.",
    TICKS + "0.12Metric [units/h]",  # Unknown glued PDF text is deliberately retained.
])
def test_tables_variables_quantities_lists_prose_and_unknown_mixtures_are_not_axis_noise(data):
    assert not pure_axis_ticks(data, immediately_before_caption=True)
    result, _, _ = pool(data + "\n" + CAPTION)
    assert result["trace"]["axis_tick_units_filtered"] == 0


@pytest.mark.parametrize("question", [
    "Read the x-axis ticks in Figure 3.",
    "What are the graph data values for the cobalt controller?",
    "Read the plotted values in the figure.",
    "请读取图3的坐标刻度和数据。",
    "What coordinates are shown for the sampling measurements?",
])
def test_explicit_graph_data_requests_keep_the_numeric_units(question):
    assert requests_graph_data(question)
    content = TICKS + CAPTION + "\n" + BODY
    result, plan, _ = pool(content, question)
    assert result["trace"]["graph_data_request_exempt"] is True
    assert result["trace"]["axis_tick_units_filtered"] == 0
    assert any(row["excerpt_start"] == 0 and TICKS in row["content"] for row in result["candidates"])
    assert all("_axis_tick_exclusions" not in row for row in result["candidates"])
    selected = select_evidence(result["candidates"], {row["id"]: 1. for row in result["candidates"]}, plan, [])
    assert any(TICKS.strip() in source["content"] for source in selected["sources"])


def test_dense_numeric_data_without_a_caption_is_retained():
    result, _, _ = pool(TICKS + "\n" + BODY)
    assert result["trace"]["axis_tick_units_filtered"] == 0
    assert any(row["excerpt_start"] == 0 for row in result["candidates"])


def test_filtered_ticks_cannot_be_bridged_between_caption_and_previous_body():
    before = "The cobalt controller stores each sampling measurement in a queue.\n"
    content = before + TICKS + CAPTION + "\n" + BODY
    result, plan, _ = pool(content)
    assert result["trace"]["axis_tick_units_filtered"] == 1
    for row in result["candidates"]:
        assert content[row["excerpt_start"]:row["excerpt_end"]] == row["content"]
        assert row["excerpt_end"] <= len(before) or row["excerpt_start"] >= len(before + TICKS)
        assert len(row["content"]) <= 1600
    assert len(result["candidates"]) <= 48
    selected = select_evidence(result["candidates"], {row["id"]: 1. for row in result["candidates"]}, plan, [])
    assert all(TICKS.strip() not in source["content"] for source in selected["sources"])
    for source in selected["sources"]:
        assert content[source["excerpt_start"]:source["excerpt_end"]] == source["content"]
        assert "_axis_tick_exclusions" not in source


def test_partial_extension_and_source_set_repair_cannot_restore_excluded_intervals():
    before = ("The cobalt controller stores sampling measurements in the calibrated buffer. " * 12).strip() + "\n"
    after = (CAPTION + "\n" + (BODY + " ") * 12).strip()
    content = before + TICKS + after
    result, plan, _ = pool(content)
    assert result["trace"]["axis_tick_units_filtered"] == 1
    excluded_start, excluded_end = len(before), len(before + TICKS)
    for direction in (1., -1.):
        scores = {row["id"]: direction * row["excerpt_start"] for row in result["candidates"]}
        events = []
        selected = select_evidence(result["candidates"], scores, plan, [], diagnostic_callback=events.append)
        assert selected["sources"]
        for source in selected["sources"]:
            assert not (source["excerpt_start"] < excluded_end and source["excerpt_end"] > excluded_start)
            assert content[source["excerpt_start"]:source["excerpt_end"]] == source["content"]
            assert len(source["content"]) <= 1600
        assert any(event.get("reason") == "excluded_axis_interval" for event in events)

