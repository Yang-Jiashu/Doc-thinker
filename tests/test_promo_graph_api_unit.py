import json
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI

with patch("dotenv.load_dotenv"):
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


def test_summary_keeps_exact_focused_node_inside_existing_cap():
    nodes = [{"id": f"node-{index}"} for index in range(260)]
    original = list(nodes)
    degree = {node["id"]: 260 - index for index, node in enumerate(nodes)}
    selected = _select_graph_nodes(nodes, degree, focus_node="node-259")
    assert len(selected) == 200
    assert selected[:-1] == nodes[:199]
    assert selected[-1] is nodes[-1]
    assert nodes == original
    assert _select_graph_nodes(nodes, degree, focus_node="node-0") == nodes[:200]
    assert _select_graph_nodes(nodes, degree, focus_node="node-259", scope="full") == nodes


def test_summary_focus_cannot_inject_foreign_nodes_or_expand_candidate_budget():
    nodes = [{"id": f"candidate-{index}", "is_expanded": "1"} for index in range(240)]
    nodes.append({"entity_id": "source-node"})
    selected = _select_graph_nodes(nodes, {}, focus_node="source-node")
    assert len(selected) == 200
    assert selected[-1] == {"entity_id": "source-node"}
    assert _select_graph_nodes(nodes, {}, focus_node="foreign-node") == nodes[:200]
    assert _select_graph_nodes(nodes, {}, focus_node="source", max_nodes=1) == nodes[:1]
    assert _select_graph_nodes(nodes, {}, focus_node="source-node", max_nodes=0) == []
    assert _select_graph_nodes([], {}, focus_node="source-node") == []


async def test_graph_api_focus_query_stays_in_the_requested_session(monkeypatch):
    graph_api = importlib.import_module("docthinker.server.routers.graph")
    session_nodes = {
        "#one": [{"id": f"node-{index}"} for index in range(260)],
        "#two": [{"id": "other-node"}],
    }
    rags = {
        sid: SimpleNamespace(
            graphcore=SimpleNamespace(chunk_entity_relation_graph=SimpleNamespace(
                get_all_nodes=AsyncMock(return_value=nodes),
                get_all_edges=AsyncMock(return_value=[]),
            )),
            _ensure_graphcore_initialized=AsyncMock(),
        )
        for sid, nodes in session_nodes.items()
    }
    monkeypatch.setattr(graph_api.state, "rag_instance", SimpleNamespace(
        config=object(), graphcore_kwargs={}, llm_model_func=None, embedding_func=None,
    ))
    monkeypatch.setattr(graph_api.state, "session_manager", SimpleNamespace(
        get_session_rag=lambda sid, *_args: rags[sid],
    ))
    app = FastAPI()
    app.include_router(graph_api.router, prefix="/api/v1")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/knowledge-graph/data", params={"session_id": "#one", "focus_node": "node-259"})
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["nodes"]) == 200
        assert payload["nodes"][-1]["id"] == "node-259"
        assert payload["metadata"]["focus_node_found"] is True
        assert payload["metadata"]["truncated"] is True
        other = await client.get("/api/v1/knowledge-graph/data", params={"session_id": "#two", "focus_node": "node-259"})
        assert [node["id"] for node in other.json()["nodes"]] == ["other-node"]
        assert other.json()["metadata"]["focus_node_found"] is False
        full = await client.get("/api/v1/knowledge-graph/data", params={"session_id": "#one", "focus_node": "node-259", "scope": "full"})
        assert len(full.json()["nodes"]) == 260


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
