"""Answer mentions are activity signals, not independent document evidence."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from docthinker.kg_expansion.manager import ExpandedNodeManager
from docthinker.memory_core import AgentMemoryBackends, AgentMemoryCore, MemoryPolicy
from docthinker.memory_core.adapters import (
    ExpandedNodeBackend,
    GraphCorePromotionBackend,
)
from graphcore.coregraph.kg.networkx_impl import NetworkXStorage
from graphcore.coregraph.kg.shared_storage import (
    finalize_share_data,
    initialize_share_data,
)


@pytest.fixture
async def source_graph(tmp_path):
    finalize_share_data()
    initialize_share_data()
    graph = NetworkXStorage(
        namespace="chunk_entity_relation",
        workspace="",
        global_config={"working_dir": str(tmp_path / "knowledge")},
        embedding_func=None,
    )
    await graph.initialize()
    for name in ("Policy", "Imports"):
        await graph.upsert_node(
            name,
            {
                "entity_id": name,
                "entity_type": "source_concept",
                "description": f"Original observation about {name}.",
                "source_id": f"chunk-{name}",
            },
        )
    await graph.upsert_edge(
        "Policy",
        "Imports",
        {"description": "An original source relation.", "source_id": "chunk-1"},
    )
    await graph.index_done_callback()
    yield graph
    finalize_share_data()


def candidate_manager(tmp_path):
    # Even thresholds chosen by older clients cannot authorize a fact write.
    manager = ExpandedNodeManager(
        tmp_path / "expanded.json",
        promote_score_threshold=0.1,
        promote_use_threshold=1,
    )
    manager.upsert_candidates(
        [
            {
                "entity": "Policy",
                "description": "An unsupported explanation about imports.",
                "reason": "The model inferred a causal mechanism.",
                "validation_score": 1.0,
            }
        ],
        default_root_ids=["Imports"],
    )
    return manager


def test_repeated_mentions_only_raise_candidate_activity(tmp_path):
    manager = candidate_manager(tmp_path)
    assert manager.get("Policy")["status"] == "candidate"
    backend = ExpandedNodeBackend(lambda _sid: manager)
    for _ in range(5):
        matches = backend.match("#00003", "Policy", top_k=2, min_score=0.0)
        result = manager.record_response_usage(
            answer="Policy might influence Imports.",
            matches=matches,
            attached_entities=["Imports"],
        )
        assert result == {"used": ["Policy"], "promoted": []}

    record = manager.get("Policy")
    assert record["status"] == "active"
    assert record["use_count"] == record["hit_count"] == 5
    assert record["promotion_score"] >= 2.0
    assert record["last_used_at"] and record["last_hit_at"]
    assert record["attached_entities"] == ["Imports"]
    assert record["source"] == "llm_expansion"
    assert ExpandedNodeManager(manager.storage_path).get("Policy") == record
    instruction = backend.build_instruction("#00003", matches, limit=2)
    assert "Policy" in instruction and "未验证" in instruction


def test_unused_candidate_stays_candidate_and_legacy_adapter_does_not_claim_promotion(
    tmp_path,
):
    manager = candidate_manager(tmp_path)
    matches = [{"entity": "Policy", "score": 0.9}]
    assert manager.record_response_usage(
        answer="The document does not answer this question.", matches=matches
    ) == {"used": [], "promoted": []}
    assert manager.get("Policy")["status"] == "candidate"
    assert manager.get("Policy")["use_count"] == 0

    # Old custom managers may emit names; the built-in adapter only records usage.
    old_manager = SimpleNamespace(
        record_response_usage=Mock(
            return_value={"used": ["Policy"], "promoted": ["Policy"]}
        )
    )
    expanded = ExpandedNodeBackend(lambda _sid: old_manager)
    assert (
        expanded.record_usage(
            "#00003", "Policy", matches, attached_entities=["Imports"]
        )
        == []
    )
    old_manager.record_response_usage.assert_called_once_with(
        answer="Policy", matches=matches, attached_entities=["Imports"]
    )


@pytest.mark.asyncio
async def test_after_response_records_usage_without_rewriting_source(
    tmp_path, source_graph
):
    manager = candidate_manager(tmp_path)
    expanded = ExpandedNodeBackend(lambda _sid: manager)
    get_rag = AsyncMock(
        return_value=SimpleNamespace(
            graphcore=SimpleNamespace(chunk_entity_relation_graph=source_graph)
        )
    )
    core = AgentMemoryCore(
        backends=AgentMemoryBackends(
            expanded=expanded,
            graph=GraphCorePromotionBackend(get_rag),
        ),
        policy=MemoryPolicy(enabled_layers=("expanded", "graph")),
    )
    original_nodes = await source_graph.get_all_nodes()
    original_edges = await source_graph.get_all_edges()
    graph_path = next((tmp_path / "knowledge").glob("*.graphml"))
    original_bytes = graph_path.read_bytes()
    for _ in range(5):
        matches = expanded.match("#00003", "Policy", top_k=2, min_score=0.0)
        result = await core.after_response(
            session_id="#00003",
            question="How does Policy affect Imports?",
            answer="Policy could lower Imports, but this needs source verification.",
            matched_expanded=matches,
        )
        assert result["expanded_promoted"] == []

    assert manager.get("Policy")["use_count"] == 5
    assert manager.get("Policy")["status"] == "active"
    assert await source_graph.get_all_nodes() == original_nodes
    assert await source_graph.get_all_edges() == original_edges
    assert graph_path.read_bytes() == original_bytes
    get_rag.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_promoted_records_cannot_rewrite_source(tmp_path, source_graph):
    path = tmp_path / "legacy_expanded.json"
    payload = {
        "nodes": [
            {
                "entity": "Policy",
                "status": "promoted",
                "description": "Historical unsupported expansion.",
                "reason": "Do not overwrite the original source with this.",
                "source": "llm_expansion",
                "root_ids": ["Imports"],
                "use_count": 50,
                "promotion_score": 50.0,
                "validation_score": 1.0,
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    saved_bytes = path.read_bytes()
    manager = ExpandedNodeManager(path)
    record = manager.get("Policy")
    assert record["status"] == "promoted"
    assert path.read_bytes() == saved_bytes  # Loading is not a historical migration.
    expanded = ExpandedNodeBackend(lambda _sid: manager)
    get_rag = AsyncMock(
        return_value=SimpleNamespace(
            graphcore=SimpleNamespace(chunk_entity_relation_graph=source_graph)
        )
    )
    backend = GraphCorePromotionBackend(get_rag)
    original_nodes = await source_graph.get_all_nodes()
    original_edges = await source_graph.get_all_edges()

    # Protect the adapter even when an old caller explicitly supplies names.
    assert (
        await backend.promote(
            "#00003",
            ["Policy"],
            answer_entities=["Imports", "Unsubstantiated"],
            expanded_backend=expanded,
        )
        == []
    )
    assert path.read_bytes() == saved_bytes
    assert await source_graph.get_all_nodes() == original_nodes
    assert await source_graph.get_all_edges() == original_edges
    get_rag.assert_not_awaited()

    # Historical status is preserved, but repeated mentions never request re-writes.
    for _ in range(2):
        usage = manager.record_response_usage(
            answer="Policy might lower Imports.",
            matches=[{"entity": "Policy", "score": 0.9}],
            attached_entities=["Imports"],
        )
        assert usage == {"used": ["Policy"], "promoted": []}
    assert manager.get("Policy")["status"] == "promoted"
    assert manager.get("Policy")["use_count"] == 52
    assert manager.get("Policy")["description"] == record["description"]
