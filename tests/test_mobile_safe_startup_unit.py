"""Private mobile startup/queries use only temporary stores and fake models."""

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException, UploadFile


@pytest.fixture
def safe_mobile(monkeypatch, tmp_path):
    with patch("dotenv.load_dotenv"):
        server = importlib.import_module("docthinker.server.app")
        entry = importlib.import_module("docthinker.server.mobile_app")
        mobile = importlib.import_module("docthinker.server.routers.mobile")
        query = importlib.import_module("docthinker.server.routers.query")
        ingest = importlib.import_module("docthinker.server.routers.ingest")
        from docthinker.providers.settings import AppSettings
        from docthinker.server.state import AppState

    state = AppState()
    for module in (server, mobile, query, ingest):
        monkeypatch.setattr(module, "state", state)
    desktop = tmp_path / "desktop" / "data" / "_system"
    root = tmp_path / "phone"
    settings = AppSettings(
        llm_api_key="test-only-chat-credential", llm_model="test-chat",
        vlm_model="test-chat", vlm_base_url="https://chat.example.invalid/v1",
        embed_api_key="EMPTY", embed_model="default-not-authorized",
        embed_base_url="https://embed.example.invalid/v1", rerank_api_key="",
        rerank_model="", rerank_base_url="https://rerank.example.invalid/v1",
        workdir=str(desktop),
    )
    monkeypatch.setattr(server, "load_settings", Mock(return_value=settings))
    monkeypatch.setattr(server, "APIConfig", Mock())
    warmup = AsyncMock(side_effect=AssertionError("No automatic model probe"))
    initialize = AsyncMock(side_effect=AssertionError("No implicit embedding/RAG initialization"))
    chat = AsyncMock(return_value="这是一条离线测试回答")
    monkeypatch.setattr(server, "_warmup_llm_connection", warmup)
    monkeypatch.setattr(server, "_initialize_rag", initialize)
    monkeypatch.setattr(server, "_get_llm_model_func", AsyncMock(return_value=chat))
    monkeypatch.setattr(server, "_get_embedding_func", AsyncMock(side_effect=AssertionError("No embedding")))
    monkeypatch.setattr(server, "_cleanup_global_graphcore_artifacts", Mock(side_effect=AssertionError("No desktop cleanup")))
    for name in ("EMBEDDING_BINDING_API_KEY", "EMBEDDING_MODEL", "EMBED_MODEL", "LLM_EMBED_HOST", "EMBEDDING_BINDING_HOST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("DOCTHINKER_RESEARCH_ENABLED", raising=False)
    monkeypatch.delenv("DOCTHINKER_RESEARCH_CONTEXT_ASSIST", raising=False)
    monkeypatch.setenv("DOCTHINKER_MOBILE_ENABLED", "true")
    monkeypatch.setenv("DOCTHINKER_MOBILE_ACCESS_TOKEN", "test-only-phone-access-code-with-enough-characters")
    monkeypatch.setenv("DOCTHINKER_MOBILE_DATA_DIR", str(root))
    # An explicit desktop singleton path must not influence the mobile store.
    monkeypatch.setenv("LONG_HORIZON_DB_PATH", str(desktop / "desktop-memory.db"))
    monkeypatch.setenv("RAG_WORKDIR", str(desktop))
    return SimpleNamespace(server=server, entry=entry, mobile=mobile, query=query,
                           ingest=ingest, state=state, root=root, desktop=desktop,
                           settings=settings, warmup=warmup, initialize=initialize,
                           chat=chat)


@pytest.mark.parametrize("name,value", [
    ("DOCTHINKER_MOBILE_ENABLED", "false"),
    ("DOCTHINKER_MOBILE_ACCESS_TOKEN", "short"),
    ("DOCTHINKER_MOBILE_DATA_DIR", ""),
    ("DOCTHINKER_MOBILE_DATA_DIR", "relative/store"),
])
async def test_unsafe_startup_fails_before_loading_settings_or_creating_stores(safe_mobile, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    app = safe_mobile.entry.create_mobile_app()
    with pytest.raises(ValueError):
        async with app.router.lifespan_context(app):
            pytest.fail("Unsafe startup accepted")
    safe_mobile.server.load_settings.assert_not_called()
    safe_mobile.warmup.assert_not_awaited()
    safe_mobile.initialize.assert_not_awaited()
    assert not safe_mobile.root.exists()


@pytest.mark.parametrize("location", ["same", "child", "parent"])
async def test_mobile_directory_cannot_overlap_desktop_data(safe_mobile, monkeypatch, location):
    old = safe_mobile.desktop.parent
    target = {"same": old, "child": old / "phone", "parent": old.parent}[location]
    monkeypatch.setenv("DOCTHINKER_MOBILE_DATA_DIR", str(target))
    with pytest.raises(ValueError, match="overlap"):
        async with safe_mobile.entry._safe_mobile_lifespan(None):
            pytest.fail("Overlapping storage accepted")
    safe_mobile.warmup.assert_not_awaited()
    safe_mobile.server._get_llm_model_func.assert_not_awaited()
    assert not old.exists()


async def test_mobile_startup_cannot_follow_storage_symlink_outside_its_root(safe_mobile, monkeypatch, tmp_path):
    safe_mobile.root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (safe_mobile.root / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escape"):
        async with safe_mobile.entry._safe_mobile_lifespan(None):
            pytest.fail("Symlink escape accepted")
    assert not list(outside.iterdir())
    safe_mobile.server._get_llm_model_func.assert_not_awaited()


async def test_chat_only_startup_is_separate_without_migration_or_model_calls(safe_mobile, monkeypatch):
    from docthinker.session_manager import SessionManager
    migrate = Mock(side_effect=AssertionError("Do not inspect/move legacy stores"))
    for name in ("_migrate_from_legacy_rag_storage", "_migrate_sessions_to_numbered_format", "_cleanup_legacy_uuid_dirs"):
        monkeypatch.setattr(SessionManager, name, migrate)
    async with safe_mobile.entry._safe_mobile_lifespan(None):
        state = safe_mobile.state
        assert state.mobile_safe_mode and not state.mobile_document_rag
        assert state.settings.workdir == str(safe_mobile.root / "data" / "_system")
        assert state.session_manager.data_root == safe_mobile.root / "data"
        assert state.long_horizon_backend.db_path == safe_mobile.root / "data" / "_system" / "long_horizon_memory.db"
        assert safe_mobile.mobile._memory_database_path() == state.long_horizon_backend.db_path
        assert state.session_manager.list_sessions() == []
        assert state.post_upload_learning is None and state.ingestion_service is None
        assert state.memory_engine_factory is None and state.claw_manager_factory is None
        caps = await safe_mobile.mobile.capabilities()
        assert caps["chat_ready"] and not caps["document_rag"] and not caps["uploads"]
        assert not caps["automatic_learning"] and not caps["memory_write_enabled"]
    migrate.assert_not_called()
    safe_mobile.warmup.assert_not_awaited()
    safe_mobile.initialize.assert_not_awaited()
    safe_mobile.chat.assert_not_awaited()
    assert not safe_mobile.desktop.exists()


async def test_mobile_memory_injection_never_creates_or_uses_desktop_singleton(safe_mobile, monkeypatch):
    from docthinker.memory_core import core
    monkeypatch.setattr(core, "get_default_long_horizon_backend", Mock(side_effect=AssertionError("No desktop singleton")))
    async with safe_mobile.entry._safe_mobile_lifespan(None):
        memory = safe_mobile.query._get_agent_memory_core()
        assert memory.backends.long_horizon is safe_mobile.state.long_horizon_backend
        assert memory.policy.long_horizon_scopes == ("session",)
        assert memory.policy.enabled_layers == ("long_horizon",)
        assert not memory.policy.allow_memory_writes
        assert memory.backends.conversation is None and memory.backends.episodic is None


async def test_chat_only_query_remains_real_query_without_embedding_and_respects_context(safe_mobile):
    from docthinker.server.schemas import QueryRequest
    async with safe_mobile.entry._safe_mobile_lifespan(None):
        sid = safe_mobile.state.session_manager.create_session("测试")["id"]
        safe_mobile.state.session_manager.add_message(sid, "user", "之前的专属话题")
        background = BackgroundTasks()
        request = QueryRequest(session_id=sid, question="今天有什么写作建议", use_memory=False, use_conversation_context=False)
        result = await safe_mobile.query.query(request, background)
        assert result["answer"] == "这是一条离线测试回答" and result["sources"] == []
        persisted = safe_mobile.state.session_manager.get_history(sid)
        assert result["answer_id"] == persisted[-1]["id"]
        assert result["answer_mode"] == "chat_only" and not result["document_rag"]
        prompt = safe_mobile.chat.call_args.args[0]
        assert "之前的专属话题" not in prompt and "不会写入新的长期记忆" in prompt
        assert not background.tasks
        request = request.model_copy(update={"question": "继续给一点写作建议", "use_conversation_context": True})
        await safe_mobile.query.query(request, background)
        assert "之前的专属话题" in safe_mobile.chat.call_args.args[0]
        assert len(safe_mobile.state.session_manager.get_history(sid)) == 5
    safe_mobile.initialize.assert_not_awaited()


async def test_feedback_on_real_persisted_turn_stays_in_its_session_without_learning(safe_mobile):
    from docthinker.server.schemas import QueryRequest
    from docthinker.server.routers.mobile import MobileAnswerFeedback
    async with safe_mobile.entry._safe_mobile_lifespan(None):
        sid = safe_mobile.state.session_manager.create_session("first")["id"]
        other_sid = safe_mobile.state.session_manager.create_session("second")["id"]
        background = BackgroundTasks()
        answer = await safe_mobile.query.query(
            QueryRequest(session_id=sid, question="synthetic question", use_memory=False), background,
        )
        calls_before_feedback = safe_mobile.chat.await_count
        payload = MobileAnswerFeedback(session_id=sid, answer_id=answer["answer_id"], rating="inaccurate", note="明确记录，不自动改写")
        response = await safe_mobile.mobile.submit_answer_feedback(payload)
        assert response["feedback"]["rating"] == "inaccurate"
        restored = await safe_mobile.mobile.history(sid)
        assert restored["history"][-1]["feedback"] == response["feedback"]
        assert restored["history"][-1]["content"] == answer["answer"]
        with pytest.raises(HTTPException) as error:
            await safe_mobile.mobile.submit_answer_feedback(payload.model_copy(update={"session_id": other_sid}))
        assert error.value.status_code == 404
        assert (await safe_mobile.mobile.answer_feedback(other_sid))["items"] == []
        assert safe_mobile.chat.await_count == calls_before_feedback
        assert not background.tasks
        assert safe_mobile.state.post_upload_learning is None
        assert safe_mobile.state.session_manager.base_storage_path.joinpath("answer_feedback.db").exists()
    assert not safe_mobile.desktop.exists()
    safe_mobile.initialize.assert_not_awaited()


async def test_chat_only_document_question_and_upload_do_not_call_models(safe_mobile):
    from io import BytesIO
    from docthinker.server.schemas import QueryRequest
    async with safe_mobile.entry._safe_mobile_lifespan(None):
        sid = safe_mobile.state.session_manager.create_session("测试")["id"]
        response = await safe_mobile.query.query(QueryRequest(session_id=sid, question="根据文档原文回答"), BackgroundTasks())
        assert "证据" in response["answer"] and response["sources"] == []
        upload = UploadFile(file=BytesIO(b"not a real document"), filename="example.txt")
        with pytest.raises(HTTPException) as error:
            await safe_mobile.mobile.upload(BackgroundTasks(), sid, upload)
        assert error.value.status_code == 503 and upload.file.closed
    safe_mobile.chat.assert_not_awaited()
    safe_mobile.initialize.assert_not_awaited()


async def test_safe_profile_defensively_blocks_automatic_learning_entrypoints(safe_mobile, monkeypatch):
    async with safe_mobile.entry._safe_mobile_lifespan(None):
        eclrr = AsyncMock(side_effect=AssertionError("No background ECLRR"))
        study = AsyncMock(side_effect=AssertionError("No background self-study"))
        monkeypatch.setattr(safe_mobile.ingest, "_background_path_edge_discovery", eclrr)
        monkeypatch.setattr(safe_mobile.ingest, "_background_self_study", study)
        monkeypatch.setattr(safe_mobile.query, "_get_agent_memory_core", Mock(side_effect=AssertionError("No background enrichment")))
        await safe_mobile.ingest._run_post_upload_learning("#00001")
        await safe_mobile.query._background_post_query_enrichment("#00001", "question", "answer")
        eclrr.assert_not_awaited()
        study.assert_not_awaited()


def test_desktop_session_manager_migration_defaults_are_preserved(tmp_path, monkeypatch):
    from docthinker.session_manager import SessionManager
    migrations = [Mock(), Mock(), Mock()]
    for name, method in zip(("_migrate_from_legacy_rag_storage", "_migrate_sessions_to_numbered_format", "_cleanup_legacy_uuid_dirs"), migrations):
        monkeypatch.setattr(SessionManager, name, method)
    SessionManager(str(tmp_path / "desktop" / "data" / "_system"))
    for method in migrations:
        method.assert_called_once()


async def test_explicit_embedding_enables_only_requested_rag_without_extra_startup_models(safe_mobile, monkeypatch):
    for name, value in {"EMBEDDING_BINDING_API_KEY": "offline-test-embedding-key", "EMBEDDING_MODEL": "test-embedding", "LLM_EMBED_HOST": "https://embed.example.invalid/v1"}.items():
        monkeypatch.setenv(name, value)
    safe_mobile.settings.embed_api_key = "offline-test-embedding-key"
    rag = SimpleNamespace(llm_model_func=AsyncMock(), embedding_func=AsyncMock(), finalize_storages=AsyncMock())
    monkeypatch.setattr(safe_mobile.server, "_initialize_rag", AsyncMock(return_value=rag))
    monkeypatch.setattr(safe_mobile.server, "CognitiveProcessor", Mock())
    monkeypatch.setattr(safe_mobile.server, "IngestionService", Mock())
    async with safe_mobile.entry._safe_mobile_lifespan(None):
        assert safe_mobile.state.mobile_document_rag
        assert safe_mobile.state.ingestion_service is not None
        assert safe_mobile.state.orchestrator is None
        assert safe_mobile.state.post_upload_learning is None
    safe_mobile.warmup.assert_not_awaited()
    safe_mobile.chat.assert_not_awaited()
    rag.llm_model_func.assert_not_awaited()
    rag.embedding_func.assert_not_awaited()
    rag.finalize_storages.assert_awaited_once()


async def test_research_and_legacy_embedding_profiles_fail_before_any_research_runner(safe_mobile, monkeypatch):
    research_module = importlib.import_module("docthinker.server.research_service")
    research = Mock(side_effect=AssertionError("Mixed profiles must not create a research runner"))
    monkeypatch.setattr(research_module, "ResearchService", research)
    monkeypatch.setenv("DOCTHINKER_RESEARCH_ENABLED", "true")
    for name, value in {"EMBEDDING_BINDING_API_KEY": "offline-test-embedding-key", "EMBEDDING_MODEL": "test-embedding", "LLM_EMBED_HOST": "https://embed.example.invalid/v1"}.items():
        monkeypatch.setenv(name, value)
    safe_mobile.settings.embed_api_key = "offline-test-embedding-key"
    with pytest.raises(ValueError, match="separate mobile profiles"):
        async with safe_mobile.entry._safe_mobile_lifespan(None):
            pytest.fail("Mixed research and legacy RAG profiles accepted")
    research.assert_not_called()
    assert safe_mobile.state.research_service is None
    safe_mobile.warmup.assert_not_awaited()
    safe_mobile.initialize.assert_not_awaited()
    safe_mobile.chat.assert_not_awaited()
    safe_mobile.server._get_embedding_func.assert_not_awaited()


@pytest.fixture
def research_adapter(monkeypatch):
    with patch("dotenv.load_dotenv", return_value=False):
        server = importlib.import_module("docthinker.server.app")
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="offline bounded research answer"))])
    create = AsyncMock(return_value=response)
    client = SimpleNamespace(base_url="https://provider.example.invalid/v1",
                             chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    factory = Mock(return_value=client)
    settings = SimpleNamespace(llm_model="step-5-preview", llm_models=["must-not-fallback-a", "must-not-fallback-b"],
                               llm_router_max_concurrency=2, llm_temperature=0.1)
    monkeypatch.setattr(server, "get_vlm_client", factory)
    monkeypatch.setattr(server, "state", SimpleNamespace(settings=settings, mobile_safe_mode=True))
    monkeypatch.setenv("DOCTHINKER_RESEARCH_ENABLED", "true")
    monkeypatch.delenv("DOCTHINKER_INVITE_WORKER_ID", raising=False)
    return SimpleNamespace(server=server, factory=factory, settings=settings, create=create)


async def test_private_research_adapter_has_zero_sdk_retries_and_fixed_completion_budget(research_adapter):
    from docthinker.server.invite_budget import MAX_OUTPUT_TOKENS
    adapter = research_adapter
    complete = await adapter.server._get_llm_model_func()
    adapter.factory.assert_called_once_with(adapter.settings, max_retries=0)
    assert await complete("synthetic research question", system_prompt="trusted synthetic policy", max_tokens=10**12) == "offline bounded research answer"
    assert adapter.create.await_count == 1
    payload = adapter.create.call_args.kwargs
    assert payload["model"] == "step-5-preview"
    assert payload["max_tokens"] == MAX_OUTPUT_TOKENS
    assert payload["messages"][0] == {"role": "system", "content": "trusted synthetic policy"}
    assert payload["stream"] is False


async def test_private_research_adapter_does_not_retry_or_fall_back_after_provider_failure(research_adapter):
    adapter = research_adapter
    adapter.create.side_effect = RuntimeError("synthetic provider failure")
    complete = await adapter.server._get_llm_model_func()
    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        await complete("synthetic research question")
    assert adapter.create.await_count == 1
    assert adapter.create.call_args.kwargs["model"] == "step-5-preview"


async def test_private_research_adapter_rejects_assembled_prompt_before_sdk_request(research_adapter):
    from docthinker.server.invite_budget import InviteBudgetError, MAX_INPUT_BYTES
    adapter = research_adapter
    complete = await adapter.server._get_llm_model_func()
    with pytest.raises(InviteBudgetError):
        await complete("x" * MAX_INPUT_BYTES, system_prompt="must be counted too")
    adapter.create.assert_not_awaited()
