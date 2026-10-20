"""The local benchmark measures source retrieval, never generated-answer quality."""
import json

import pytest

from scripts.benchmark_graph_discovery import (
    benchmark_fixture, benchmark_graph_timing, fixtures, main, synthetic_timing_corpus,
)


def test_fixture_results_are_computed_reproducibly_and_include_budget_refusals():
    first = [benchmark_fixture(item) for item in fixtures()]
    second = [benchmark_fixture(item) for item in fixtures()]
    assert first == second
    assert len(first) == 8 and all(item["passed"] for item in first)
    by_id = {item["id"]: item for item in first}
    bridge = by_id["endpoint_bridge"]
    assert bridge["baseline_bridge_coverage"]["covered"] == 0
    assert bridge["graph_bridge_coverage"]["covered"] == 1
    assert bridge["added_source_ids"] == ["bridge"]
    assert by_id["two_hop_idea_neighborhood"]["checks"]["actual_two_hop_path"]
    assert "graph_counterevidence_in_path" in by_id["negated_bridge"]["gap_codes"]
    assert "graph_path_conditions_differ" in by_id["conditional_bridge"]["gap_codes"]
    for name in ("source_slots_full", "byte_budget_full"):
        row = by_id[name]
        assert row["baseline_source_ids"] == row["graph_source_ids"]
        assert row["graph_bridge_coverage"]["covered"] == row["added_source_bytes"] == 0
        assert "graph_capacity_reached" in row["gap_codes"]


def test_timing_corpus_is_seeded_and_measures_actual_parsed_graph_edges():
    assert synthetic_timing_corpus(30) == synthetic_timing_corpus(30)
    timing = benchmark_graph_timing(count=30, repeats=2)
    assert timing["corpus_chunks"] == timing["indexed_chunks"] == timing["edge_count"] == 30
    assert timing["cache_and_bridge_checks_passed"]
    assert timing["model_calls"] == 0
    assert "end_to_end_response" in timing["excluded"]
    assert len(timing["cold"]["samples_ms"]) == len(timing["warm"]["samples_ms"]) == 2
    assert timing["cold"]["min_ms"] >= 0 and timing["warm"]["min_ms"] >= 0


@pytest.mark.parametrize("count", [None, True, 2, 2001, 20.5])
def test_timing_corpus_rejects_unbounded_or_invalid_sizes(count):
    with pytest.raises(ValueError):
        synthetic_timing_corpus(count)


@pytest.mark.parametrize("repeats", [None, True, 0, 31, 1.5])
def test_timing_repetitions_are_bounded(repeats):
    with pytest.raises(ValueError):
        benchmark_graph_timing(count=3, repeats=repeats)


def test_cli_emits_json_with_explicit_non_quality_scope(capsys):
    assert main(["--chunks", "10", "--repeats", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["synthetic_only"] and result["passed"]
    assert result["model_calls"] == 0
    assert result["summary"]["total"] == 8
    assert result["summary"]["baseline_bridge_coverage"] == {"covered": 0, "total": 6}
    assert result["summary"]["graph_bridge_coverage"] == {"covered": 6, "total": 6}
    assert any("not semantic answer quality" in item for item in result["measurement_limits"])
    assert any("not tokenizer or billing" in item for item in result["measurement_limits"])
