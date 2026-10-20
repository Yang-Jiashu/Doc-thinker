"""Source-set geometry regressions, with fictional prose and no paper gold."""
from copy import deepcopy

from docthinker.server.research_evidence_selection import select_evidence
from docthinker.server.research_sentence_windows import bounded_evidence_extension, evidence_units
from tests.test_research_selection_regressions_unit import _candidate, _facet


def sentence(index, size=180):
    return f" Measurement {index} records " + (f"signal{index} " * 30)[:size] + "."


def run(rows, scores, facets, **limits):
    events = []
    result = select_evidence(rows, scores, {"facets": facets}, [], diagnostic_callback=events.append, **limits)
    return result, events


def test_overlong_union_extends_through_whole_units_without_claiming_partial_pair_credit():
    a, b = _facet("method"), _facet("condition", "What control condition was held constant?")
    parts = [sentence(index) for index in range(9)]
    text = "".join(parts)
    end_first = sum(map(len, parts[:5]))
    start_second = sum(map(len, parts[:3]))
    end_second = sum(map(len, parts[:8]))
    first = _candidate("a", text[:end_first], a, source_id="shared", source_content=text)
    second = _candidate("b", text[start_second:end_second], b, source_id="shared", source_content=text, start=start_second)
    result, events = run([first, second], {first["id"]: 5., second["id"]: 5.}, [a, b], max_sources=1)
    quote = result["sources"][0]
    assert quote["excerpt_start"] == 0
    assert end_first < quote["excerpt_end"] < end_second
    assert quote["excerpt_end"] in {end for _, end in evidence_units(text)}
    assert len(quote["content"]) <= 1600
    assert quote["content"] == text[quote["excerpt_start"]:quote["excerpt_end"]]
    assert result["trace"]["complete_unit_partial_extensions"] == 1
    assert result["trace"]["facet_coverage"]["condition"] == 0
    credits = next(item for item in events if item["event"] == "final_window_credits")
    assert [row["id"] for row in credits["windows"]] == [first["id"]]
    assert credits["semantics"] == "fully_retained_submitted_windows_only"
    assert credits["retained_submission_implies_complete_model_input"] is False
    assert result["trace"]["credit_geometry"] == "fully_retained_submitted_windows_only"
    assert result["trace"]["retained_submission_implies_complete_model_input"] is False


def test_bound_qualifier_is_not_cut_to_make_extension_fit():
    initial = "Measurements record phase angles."
    claim = " The prism shows improved retrieval performance" + " detail" * 50 + "."
    qualifier = " However, the improvement only holds with calibrated input and fails otherwise."
    text = initial + claim + qualifier
    limit = len(initial + claim) + 2
    assert bounded_evidence_extension(text, (0, len(initial)), (len(initial), len(text)), limit) is None


def test_complete_context_extension_can_grow_left_and_stays_in_original_slice():
    parts = [sentence(index, 80) for index in range(5)]
    text = "".join(parts)
    retained = (sum(map(len, parts[:3])), len(text))
    supplement = (0, sum(map(len, parts[:4])))
    limit = len(text) - len(parts[0])
    span = bounded_evidence_extension(text, retained, supplement, limit)
    assert span == (len(parts[0]), len(text))


def test_same_source_window_can_be_replaced_when_another_source_preserves_old_requirement():
    method, control = _facet("method"), _facet("control", "Which control is held constant?")
    opening = "The bronze apparatus turns a wheel using a rotating spring."
    later = "Independent control chambers maintain pressure and humidity at identical settings during every trial."
    source = opening + " Unrelated distant material fills a large gap." * 18 + later
    first = _candidate("a", opening, method, source_id="shared", source_content=source)
    second = _candidate("b", later, control, source_id="shared", source_content=source, start=len(source)-len(later))
    backup = _candidate("c", "Rotating spirals transmit torque through a bronze axle at a fixed angular rate.", method)
    result, events = run([first, second, backup], {first["id"]: 5., second["id"]: 5., backup["id"]: 4.8},
                         [method, control], max_sources=2)
    retained = next(row for row in result["sources"] if row["id"] == "shared")
    assert retained["content"] == later
    assert result["trace"]["source_set_replacements"] >= 1
    assert result["trace"]["facet_coverage"] == {"method": .8, "control": .8}
    assert result["trace"]["same_chunk_disconnected_skipped"] == 0
    assert not any(gap["code"] == "complete_statement_exceeds_budget" for gap in result["gaps"])
    credits = next(row for row in events if row["event"] == "final_window_credits")["windows"]
    assert first["id"] not in {row["id"] for row in credits}
    assert second["id"] in {row["id"] for row in credits}


def test_lower_rank_disconnected_window_without_new_requirement_does_not_displace_or_jitter():
    facet = _facet()
    opening = "A crystal instrument measures angular drift at each station."
    later = "A telemetry receiver computes lagged offsets from retained calibration records."
    source = opening + " Unrelated distant material fills a large gap." * 18 + later
    first = _candidate("a", opening, facet, source_id="shared", source_content=source)
    second = _candidate("b", later, facet, source_id="shared", source_content=source, start=len(source)-len(later))
    scores = {first["id"]: 5., second["id"]: 4.}
    results = [run(rows, scores, [facet], max_sources=1)[0] for rows in ([first, second], [second, first])]
    assert all(result["sources"][0]["content"] == opening for result in results)
    assert all(result["trace"]["source_set_replacements"] == 0 for result in results)
    assert results[0]["sources"] == results[1]["sources"]


def test_refinement_respects_utf8_byte_budget_and_does_not_mutate_inputs():
    facet = _facet()
    parts = [(f"观测{index}记录传感器的相位与压力。" * 12) for index in range(8)]
    source = "".join(parts)
    first = _candidate("a", "".join(parts[:4]), facet, source_id="same", source_content=source)
    second = _candidate("b", "".join(parts[2:6]), facet, source_id="same", source_content=source, start=len("".join(parts[:2])))
    rows = [first, second]
    before = deepcopy(rows)
    budget = len(first["content"].encode())
    result, _ = run(rows, {first["id"]: 5., second["id"]: 4.}, [facet], max_source_bytes=budget)
    assert sum(len(row["content"].encode()) for row in result["sources"]) <= budget
    assert result["trace"]["source_bytes"] == sum(len(row["content"].encode()) for row in result["sources"])
    assert rows == before


def test_oversized_candidate_cannot_bypass_window_cap_via_replacement():
    facet = _facet()
    large = _candidate("oversized", "The instrument records waveforms. " * 60, facet)
    result, _ = run([large], {large["id"]: 10.}, [facet])
    assert result["sources"] == []


def test_same_source_merged_context_cannot_be_cropped_just_to_reduce_redundancy():
    facet = _facet()
    parts = [sentence(index) for index in range(8)]
    # The added complete sentence shares words with another valid source.
    # This is not authorization to delete it and game a redundancy penalty.
    text = "".join(parts)
    end_first = sum(map(len, parts[:5]))
    start_second = sum(map(len, parts[:3]))
    first = _candidate("a", text[:end_first], facet, source_id="shared", source_content=text)
    second = _candidate("b", text[start_second:], facet, source_id="shared", source_content=text, start=start_second)
    other = _candidate("c", "Distinct calibration measurements record signal5 signal6 signal7 phase pressure offsets.", facet)
    result, _ = run([first, second, other], {first["id"]: 5., second["id"]: 4.9, other["id"]: 4.8}, [facet], max_sources=2)
    quote = next(row for row in result["sources"] if row["id"] == "shared")
    assert quote["excerpt_start"] == 0 and quote["excerpt_end"] > end_first
    assert quote["excerpt_end"] in {end for _, end in evidence_units(text)}
