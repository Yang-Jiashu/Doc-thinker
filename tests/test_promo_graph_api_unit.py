import json

from docthinker.server.routers.graph import (
    _expand_graph_edge_records,
    _limit_chunk_source_ids,
    _select_graph_nodes,
    _split_chunk_source_ids,
)


def test_promo_full_scope_keeps_all_nodes_while_summary_stays_capped():
    nodes = [{"id": f"node-{index}"} for index in range(260)]
    degree = {node["id"]: 260 - index for index, node in enumerate(nodes)}

    summary = _select_graph_nodes(nodes, degree, scope="summary")
    full = _select_graph_nodes(nodes, degree, scope="full")

    assert len(summary) == 200
    assert len(full) == 260
    assert full == nodes


def test_zero_chunk_limit_returns_all_deduplicated_source_ids():
    source_ids = _split_chunk_source_ids("chunk-a<SEP>chunk-b<SEP>chunk-a")

    assert source_ids == ["chunk-a", "chunk-b"]
    assert _limit_chunk_source_ids(source_ids, 0) == source_ids
    assert _limit_chunk_source_ids(source_ids, 1) == ["chunk-a"]


def test_graph_api_exposes_fact_and_eclrr_relations_as_separate_edges():
    promoted = {
        "relation_id": "rel-eclrr-1",
        "source": "A",
        "target": "D",
        "relation": "间接影响",
        "description": "A 经由 B、C 间接影响 D。",
        "source_id": "chunk-1<SEP>chunk-2<SEP>chunk-3",
        "review_status": "promoted",
        "provenance": "eclrr_v4",
        "algorithm_version": "eclrr_v4",
        "is_discovered": "1",
    }
    physical = {
        "source": "A",
        "target": "D",
        "keywords": "同场出现",
        "description": "原始事实关系",
        "source_id": "chunk-fact",
        "eclrr_relations": json.dumps([promoted], ensure_ascii=False),
    }

    records = _expand_graph_edge_records(physical, 0)

    assert len(records) == 2
    assert records[0]["edge_kind"] == "original"
    assert records[0]["label"] == "同场出现"
    assert records[1]["edge_kind"] == "eclrr_v4"
    assert records[1]["id"] == "rel-eclrr-1"
    assert records[1]["source_id"] == "chunk-1<SEP>chunk-2<SEP>chunk-3"


def test_promoted_physical_edge_is_not_duplicated_by_its_nested_record():
    promoted = {
        "source": "A",
        "target": "D",
        "relation_id": "rel-eclrr-1",
        "relation": "间接影响",
        "review_status": "promoted",
        "provenance": "eclrr_v4",
        "algorithm_version": "eclrr_v4",
    }
    promoted["eclrr_relations"] = json.dumps([promoted])

    records = _expand_graph_edge_records(promoted, 0)

    assert len(records) == 1
    assert records[0]["edge_kind"] == "eclrr_v4"
