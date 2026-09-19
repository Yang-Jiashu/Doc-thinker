"""Exercise the upload hook with real source stores and no external model calls."""

import asyncio
import importlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import BackgroundTasks, UploadFile

from docthinker.kg_self_study.orchestrator import (
    SelfStudyOrchestrator,
    StudySessionResult,
)
from docthinker.server.background_learning import SessionLearningRunner
from graphcore.coregraph.kg.json_kv_impl import JsonKVStorage
from graphcore.coregraph.kg.networkx_impl import NetworkXStorage
from graphcore.coregraph.kg.shared_storage import (
    finalize_share_data,
    initialize_share_data,
)

ingest_router = importlib.import_module("docthinker.server.routers.ingest")


@pytest.fixture
async def source_session(tmp_path, monkeypatch):
    finalize_share_data()
    initialize_share_data()
    knowledge_dir = tmp_path / "data" / "#00003" / "knowledge"
    global_config = {"working_dir": str(knowledge_dir)}
    graph = NetworkXStorage(
        namespace="chunk_entity_relation",
        workspace="",
        global_config=global_config,
        embedding_func=None,
    )
    chunks = JsonKVStorage(
        namespace="text_chunks",
        workspace="",
        global_config=global_config,
        embedding_func=None,
    )
    await graph.initialize()
    await chunks.initialize()
    for name in "ABCDE":
        await graph.upsert_node(
            name,
            {
                "description": f"Original observation about {name}.",
                "source_id": f"chunk-{name}",
            },
        )
    await chunks.upsert({"chunk-A": {"content": "Original observation about A."}})
    await graph.index_done_callback()
    await chunks.index_done_callback()
    llm = AsyncMock(side_effect=AssertionError("This regression must not call an LLM"))
    rag = SimpleNamespace(
        config=object(),
        graphcore_kwargs={},
        llm_model_func=llm,
        embedding_func=None,
        graphcore=SimpleNamespace(
            chunk_entity_relation_graph=graph,
            text_chunks=chunks,
            working_dir=str(knowledge_dir),
            workspace="",
        ),
        _ensure_graphcore_initialized=AsyncMock(),
    )
    manager = SimpleNamespace(
        get_session=lambda sid: {
            "id": sid,
            "metadata": {"knowledge_dir": str(knowledge_dir)},
        },
        get_session_rag=lambda *_args: rag,
        allocate_session_file_path=lambda _sid, filename: (
            knowledge_dir.parent / "content" / filename
        ),
        add_document_record=Mock(),
        set_document_status=Mock(),
    )
    service = SimpleNamespace(ingest_text=AsyncMock())
    monkeypatch.setattr(ingest_router.state, "session_manager", manager)
    monkeypatch.setattr(ingest_router.state, "rag_instance", rag)
    monkeypatch.setattr(ingest_router.state, "ingestion_service", service)
    yield SimpleNamespace(
        graph=graph,
        chunks=chunks,
        knowledge_dir=knowledge_dir,
        llm=llm,
        rag=rag,
        service=service,
        manager=manager,
    )
    finalize_share_data()


def _proposals():
    return {
        "entity_updates": [
            {
                "entity": "A",
                "action": "enrich_description",
                "new_content": "Speculative explanation unsupported by the source.",
                "confidence": 0.99,
            }
        ],
        "new_edges": [],
        "contradiction_flags": [{"entity": "A", "reason": "requires review"}],
    }


def _mock_study(monkeypatch, operations=None):
    async def propose(self):
        await self.kg_write_func(_proposals() if operations is None else operations)
        return StudySessionResult(total_entity_updates=1, total_contradictions=1)

    monkeypatch.setattr(SelfStudyOrchestrator, "run_session", propose)


@pytest.mark.parametrize(
    "operations",
    [
        _proposals(),
        {
            **_proposals(),
            "new_edges": [{"source": "A", "target": "B", "confidence": 0.99}],
            "summary_nodes": [
                {"entity": "Candidate summary", "member_entities": ["A", "B", "C"]}
            ],
        },
    ],
)
async def test_self_study_only_appends_candidate_annotations(
    source_session, monkeypatch, operations
):
    session = source_session
    _mock_study(monkeypatch, operations)
    graph_file = Path(session.graph._graphml_xml_file)
    chunk_file = Path(session.chunks._file_name)
    graph_before, chunks_before = graph_file.read_bytes(), chunk_file.read_bytes()
    audit_file = session.knowledge_dir / "self_study_audit.jsonl"
    previous = {
        "review_status": "audit_only",
        "new_edges": [{"source": "B", "target": "C"}],
    }
    audit_file.write_text(json.dumps(previous) + "\n", encoding="utf-8")

    await ingest_router._background_self_study("#00003")
    await ingest_router._background_self_study("#00003")

    assert (await session.graph.get_node("A"))[
        "description"
    ] == "Original observation about A."
    assert (await session.chunks.get_by_id("chunk-A"))[
        "content"
    ] == "Original observation about A."
    assert graph_file.read_bytes() == graph_before
    assert chunk_file.read_bytes() == chunks_before
    records = [
        json.loads(line) for line in audit_file.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0] == previous
    assert len(records) == 3
    for record in records[1:]:
        assert record["session_id"] == "#00003"
        assert record["review_status"] == "audit_only"
        assert record["query_eligible"] is False
        for field, proposed in operations.items():
            if proposed:
                assert record[field] == proposed
    session.llm.assert_not_awaited()


async def test_upload_study_reuses_snapshot_without_implicit_query_llm(
    source_session, monkeypatch
):
    session = source_session
    graph = session.graph
    node_reads = AsyncMock(wraps=graph.get_all_nodes)
    edge_reads = AsyncMock(wraps=graph.get_all_edges)
    monkeypatch.setattr(graph, "get_all_nodes", node_reads)
    monkeypatch.setattr(graph, "get_all_edges", edge_reads)
    hidden_query = AsyncMock(side_effect=AssertionError("Do not invoke keyword LLMs"))
    monkeypatch.setattr(
        session.rag.graphcore, "aquery_data", hidden_query, raising=False
    )
    completed = []

    async def inspect_study(self):
        assert self.config.audit_only is True
        assert self.config.max_tokens == 30000
        for _ in range(3):
            assert len(await self.kg_read_nodes()) == 5
            assert await self.kg_read_edges() == []
            result = await self.kg_query_func("A")
            assert [item["id"] for item in result["entities"]] == ["A"]
            assert result["chunks"][0]["chunk_id"] == "chunk-A"
            assert result["chunks"][0]["content"] == "Original observation about A."
            completed.append(result)
        return StudySessionResult()

    monkeypatch.setattr(SelfStudyOrchestrator, "run_session", inspect_study)
    await ingest_router._background_self_study("#00003")
    assert len(completed) == 3
    node_reads.assert_awaited_once()
    edge_reads.assert_awaited_once()
    hidden_query.assert_not_awaited()
    session.llm.assert_not_awaited()


async def test_consecutive_uploads_schedule_only_one_learning_rerun(
    source_session, monkeypatch
):
    session = source_session
    for name in (
        "_update_local_knowledge_graph",
        "_update_local_knowledge_base",
        "_write_session_code_snapshot",
        "_sync_kg_entity_ids",
        "_run_density_clustering",
    ):
        monkeypatch.setattr(ingest_router, name, AsyncMock())
    started = asyncio.Event()
    release = asyncio.Event()
    runs = []

    async def learning(sid):
        runs.append(sid)
        started.set()
        await release.wait()

    runner = SessionLearningRunner(learning)
    monkeypatch.setattr(ingest_router.state, "post_upload_learning", runner)
    try:
        for index in range(5):
            background = BackgroundTasks()
            response = await ingest_router.upload_files(
                background_tasks=background,
                files=[
                    UploadFile(
                        filename=f"source-{index}.txt", file=io.BytesIO(b"Source")
                    )
                ],
                session_id="#00003",
            )
            assert response["success"] is True
            # Source processing finishes while learning remains blocked.
            await asyncio.wait_for(background(), 2)
            await asyncio.wait_for(started.wait(), 2)
        assert session.service.ingest_text.await_count == 5
        assert runs == ["#00003"]
        release.set()
        await asyncio.wait_for(runner.wait_idle(), 2)
        assert runs == ["#00003", "#00003"]
    finally:
        await runner.shutdown()
    session.llm.assert_not_awaited()


async def test_upload_runs_self_study_without_changing_source_graph(
    source_session, monkeypatch
):
    session = source_session
    _mock_study(monkeypatch)
    for name in (
        "_update_local_knowledge_graph",
        "_update_local_knowledge_base",
        "_write_session_code_snapshot",
        "_sync_kg_entity_ids",
        "_run_density_clustering",
        "_background_path_edge_discovery",
    ):
        monkeypatch.setattr(ingest_router, name, AsyncMock())

    runner = SessionLearningRunner(ingest_router._run_post_upload_learning)
    monkeypatch.setattr(ingest_router.state, "post_upload_learning", runner)
    background = BackgroundTasks()
    response = await ingest_router.upload_files(
        background_tasks=background,
        files=[
            UploadFile(
                filename="source.txt", file=io.BytesIO(b"Original observation about A.")
            )
        ],
        session_id="#00003",
    )
    assert response["success"] is True
    try:
        await background()
        await asyncio.wait_for(runner.wait_idle(), 2)
    finally:
        await runner.shutdown()

    session.service.ingest_text.assert_awaited_once_with(
        "Original observation about A.",
        session_id="#00003",
        file_path="source.txt",
    )
    assert (await session.graph.get_node("A"))[
        "description"
    ] == "Original observation about A."
    assert not await session.graph.has_edge("A", "B")
    records = [
        json.loads(line)
        for line in (session.knowledge_dir / "self_study_audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records[0]["entity_updates"] == _proposals()["entity_updates"]
    assert records[0]["query_eligible"] is False
    session.llm.assert_not_awaited()
