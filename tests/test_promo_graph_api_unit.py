from docthinker.server.routers.graph import (
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
