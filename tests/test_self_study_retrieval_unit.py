import json
from unittest.mock import AsyncMock

from docthinker.server.self_study_retrieval import SelfStudySnapshotRetriever


async def test_snapshot_retrieval_is_bounded_and_reads_referenced_chunks_only():
    chunks = AsyncMock()
    chunks.get_by_ids.side_effect = lambda ids: [
        {"content": "source evidence " * 1000, "file_path": "source.txt"} for _ in ids
    ]
    nodes = [
        {
            "id": f"Policy{index}",
            "description": "Policy reduces imports. " * 40,
            "source_id": f"chunk-{index}",
            "large_unused_field": "x" * 10000,
        }
        for index in range(40)
    ]
    edges = [
        {
            "source": f"Policy{index}",
            "target": "Imports",
            "description": "Policy may reduce imports.",
            "source_id": f"chunk-{index}",
        }
        for index in range(40)
    ]
    retriever = SelfStudySnapshotRetriever(nodes, edges, chunks)
    result = await retriever.query("Policy imports")
    assert 0 < len(result["entities"]) <= 12
    assert len(result["relations"]) == 12
    assert (
        sum(len(json.dumps(item, ensure_ascii=False)) for item in result["entities"])
        <= 6000
    )
    assert len(result["chunks"]) == 5
    chunks.get_by_ids.assert_awaited_once()
    assert len(chunks.get_by_ids.call_args.args[0]) == 5
    assert all(
        item["description"] == nodes[0]["description"] for item in result["entities"]
    )
    assert all("large_unused_field" not in item for item in result["entities"])
    assert all(
        item["evidence_scope"] == "graph_summary_not_source_quote"
        for item in result["entities"]
    )
    assert all(len(item["content"]) <= 2000 for item in result["chunks"])
    for item in result["chunks"]:
        assert item["source_id"] == item["chunk_id"]
        assert item["truncated"] is True
        assert item["excerpt_start"] == 0 and item["excerpt_end"] == 2000
        assert item["original_content_length"] == len("source evidence " * 1000)
        assert item["offset_unit"] == "characters_in_source_chunk"
        assert item["evidence_scope"] == "excerpt_only"
    assert len(nodes[0]["description"]) > 800  # No snapshot mutation.


async def test_snapshot_retrieval_ignores_expanded_records_and_preserves_chunk_identity():
    chunks = AsyncMock()
    chunks.get_by_ids.return_value = [None, {"content": "A source passage."}]
    nodes = [
        {
            "id": "Climate",
            "description": "气候政策降低进口依赖",
            "source_id": "missing<SEP>chunk-2",
        },
        {"id": "Policy", "source_id": "promoted_expansion"},
        {"id": "Policy", "source_id": "chunk-3", "is_expanded": "1"},
    ]
    edges = [
        {
            "source": "Policy",
            "target": "Imports",
            "is_discovered": "1",
            "source_id": "chunk-4",
        },
        {
            "source": "Policy",
            "target": "Imports",
            "provenance": "eclrr_v4",
            "source_id": "chunk-5",
        },
    ]
    retriever = SelfStudySnapshotRetriever(nodes, edges, chunks)
    result = await retriever.query("气候政策和进口的联系")
    assert len(result["entities"]) == 1
    assert result["entities"][0]["id"] == "Climate"
    assert result["relations"] == []
    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["chunk_id"] == "chunk-2"
    assert result["chunks"][0]["source_id"] == "chunk-2"
    assert result["chunks"][0]["content"] == "A source passage."
    assert result["chunks"][0]["truncated"] is False
    assert result["chunks"][0]["evidence_scope"] == "source_chunk"
    chunks.get_by_ids.assert_awaited_once_with(["missing", "chunk-2"])

    chunks.get_by_ids.reset_mock()
    assert await retriever.query("unrelated") == {
        "entities": [],
        "relations": [],
        "chunks": [],
    }
    chunks.get_by_ids.assert_not_awaited()


async def test_oversized_graph_records_are_omitted_not_semantically_truncated():
    description = "Policy reduces imports " * 150 + "except under supply constraints."
    source_ids = "<SEP>".join(f"chunk-{index}" for index in range(300))
    nodes = [{"id": "Policy", "description": description, "source_id": "chunk-1"}]
    edges = [{"source": "Policy", "target": "Imports", "source_id": source_ids}]
    chunks = AsyncMock()
    retriever = SelfStudySnapshotRetriever(nodes, edges, chunks)
    assert await retriever.query("Policy imports") == {
        "entities": [],
        "relations": [],
        "chunks": [],
    }
    assert nodes[0]["description"] == description
    assert edges[0]["source_id"] == source_ids
    chunks.get_by_ids.assert_not_awaited()
