import asyncio
import logging
import os
import numpy as np
from contextlib import asynccontextmanager
from typing import Any, List
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from graphcore.coregraph.utils import EmbeddingFunc

from docthinker import DocThinker, DocThinkerConfig
from docthinker.api_config import APIConfig
from docthinker.cognitive import CognitiveProcessor
from docthinker.providers import load_settings, get_embed_client, get_vlm_client
from docthinker.providers.clients import CredentialBindingError, assert_credential_bindings
from docthinker.providers.request_compat import (
    embedding_batch_size,
    embedding_request_options,
    sdk_chat_request_options,
)
from docthinker.utils import create_bltcy_rerank_func

_log = logging.getLogger("docthinker.server.app")
from docthinker.services import IngestionService
from docthinker.session_manager import SessionManager

from .state import state
from .memory import save_all_memory_engines
from .background_learning import SessionLearningRunner


class AsyncModelRouter:
    """Round-robin model router with bounded concurrency and fallback."""

    def __init__(self, client: Any, models: List[str], max_concurrency: int = 32):
        self.client = client
        self.models = [m for m in models if m]
        self.max_concurrency = max(1, int(max_concurrency))
        self._cursor = 0
        self._cursor_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

    async def _next_start_index(self) -> int:
        async with self._cursor_lock:
            if not self.models:
                return 0
            idx = self._cursor % len(self.models)
            self._cursor += 1
            return idx

    async def chat_completion(
        self,
        *,
        messages: List[dict],
        max_tokens: int = 2048,
        stream: bool = False,
        temperature: float = 0.1,
    ) -> Any:
        if not self.models:
            raise ValueError("No LLM models configured for routing")

        start = await self._next_start_index()
        last_err: Exception | None = None
        async with self._semaphore:
            for offset in range(len(self.models)):
                model = self.models[(start + offset) % len(self.models)]
                try:
                    token_kwargs = sdk_chat_request_options(
                        str(getattr(self.client, "base_url", "")), model, max_tokens, temperature
                    )
                    return await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **token_kwargs,
                        stream=stream,
                    )
                except Exception as e:
                    last_err = e
                    continue
        if last_err:
            raise last_err
        raise RuntimeError("Model router failed without explicit exception")


def _make_vision_model_func(vlm_client: Any):
    """Build a vision model adapter that supports both image_data and OpenAI-style messages."""

    async def vision_model_func(
        prompt: str,
        *,
        system_prompt: str | None = None,
        image_data: Any = None,
        messages: List[dict] | None = None,
        **kwargs: Any,
    ) -> str:
        images = None
        if image_data:
            if isinstance(image_data, (list, tuple)):
                images = list(image_data)
            else:
                images = [image_data]

        return await vlm_client.generate(
            prompt or "",
            images=images,
            system_prompt=system_prompt,
            max_tokens=int(kwargs.get("max_tokens", 4096)),
            temperature=float(kwargs.get("temperature", 0.2)),
            extra_messages=messages,
        )

    return vision_model_func


def _cleanup_global_graphcore_artifacts(workdir: str) -> None:
    """Remove root-level GraphCore artifacts so knowledge remains session-scoped."""
    root = Path(workdir)
    if not root.exists():
        return

    removed: list[str] = []
    for pattern in ("graph_*.graphml", "vdb_*.json", "kv_store_*.json"):
        for path in root.glob(pattern):
            try:
                path.unlink()
                removed.append(path.name)
            except Exception:
                continue

    if removed:
        _log.info(
            "Removed global GraphCore artifacts for session isolation: %s",
            ", ".join(sorted(removed)),
        )


def _create_rag_config() -> DocThinkerConfig:
    return DocThinkerConfig(
        working_dir=state.settings.workdir,
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )


async def _get_embedding_func() -> Any:
    embed_client = get_embed_client(state.settings)
    # Keep model and dimensions consistent with this client's/index's lifetime.
    # Editing settings requires reinitialization, not half-changing a live index.
    model = state.settings.embed_model
    dimension = state.settings.embed_dim
    base_url = state.settings.embed_base_url
    options = embedding_request_options(base_url, model, dimension)
    batch_size = embedding_batch_size(base_url, model)
    import logging as _logging
    _embed_log = _logging.getLogger("docthinker.embedding")

    async def _embed_batch(texts: List[str]) -> Any:
        max_retries = 8
        for attempt in range(max_retries):
            try:
                resp = await embed_client.embeddings.create(
                    model=model,
                    input=texts,
                    **options,
                )
                items = list(resp.data)
                if len(items) != len(texts):
                    raise ValueError("Embedding response count does not match the input batch")
                indices = [getattr(item, "index", None) for item in items]
                if any(index is not None for index in indices):
                    if any(type(index) is not int for index in indices) or set(indices) != set(range(len(texts))):
                        raise ValueError("Embedding response contains invalid or duplicate indices")
                    items.sort(key=lambda item: item.index)
                vectors: List[List[float]] = []
                for item in items:
                    emb = getattr(item, "embedding", None)
                    if not isinstance(emb, list) or len(emb) != dimension:
                        actual = len(emb) if isinstance(emb, list) else "invalid"
                        raise ValueError(
                            f"Embedding dimension mismatch for {model}: configured {dimension}, returned {actual}. "
                            "Use the provider's actual dimension and rebuild the affected index; vectors are never resized."
                        )
                    vectors.append(emb)
                result = np.array(vectors, dtype=np.float32)
                if not np.isfinite(result).all():
                    raise ValueError("Embedding response contains non-finite values")
                return result
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate" in err_str.lower() or "quota" in err_str.lower():
                    wait = min(3 * (2 ** attempt), 60)
                    _embed_log.warning(
                        "[embedding] 429 rate limit (attempt %d/%d, %d texts), waiting %.0fs",
                        attempt + 1, max_retries, len(texts), wait,
                    )
                    await asyncio.sleep(wait)
                    if attempt == max_retries - 1:
                        raise
                else:
                    raise

    async def embedding_func_impl(texts: List[str]) -> Any:
        if not texts:
            return np.empty((0, dimension), dtype=np.float32)
        batches = [await _embed_batch(texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
        return np.concatenate(batches, axis=0)

    return EmbeddingFunc(
        embedding_dim=dimension,
        max_token_size=8192,
        func=embedding_func_impl,
    )


async def _get_llm_model_func() -> Any:
    from .invite_budget import (
        bounded_invite_history, current_output_tokens, has_output_budget,
        is_invite_worker, validate_invite_messages,
    )
    from .model_response_error import ModelResponseError

    invitation = is_invite_worker() or (
        getattr(state, "mobile_safe_mode", False)
        and os.getenv("DOCTHINKER_RESEARCH_ENABLED", "").lower() in {"true", "1"}
    )
    # The invitation path is one explicit model call, not a hidden SDK retry or
    # a failover sweep across an operator-configured model pool.
    vlm_client = (get_vlm_client(state.settings, max_retries=0) if invitation
                  else get_vlm_client(state.settings))
    models = [state.settings.llm_model] if invitation else (state.settings.llm_models or [state.settings.llm_model])
    model_router = AsyncModelRouter(
        client=vlm_client,
        models=models,
        max_concurrency=state.settings.llm_router_max_concurrency,
    )

    async def chat_complete(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: List[dict] | None = None,
        **kwargs: Any,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(bounded_invite_history(history_messages) if invitation else history_messages)
        messages.append({"role": "user", "content": prompt})
        if invitation:
            validate_invite_messages(messages)

        resp = await model_router.chat_completion(
            messages=messages,
            max_tokens=(current_output_tokens() if invitation or has_output_budget()
                        else int(kwargs.get("max_tokens", 2048) or 2048)),
            stream=False,
            temperature=state.settings.llm_temperature,
        )
        # HTTP success is not answer completion. A token-limited or malformed
        # response must not become a saved answer merely because it contains
        # some text (or because an SDK object's repr is nonempty). Validate
        # after dispatch so this cannot trigger the router's model failover.
        choices = getattr(resp, "choices", None)
        choice = choices[0] if isinstance(choices, (list, tuple)) and len(choices) == 1 else None
        content = getattr(getattr(choice, "message", None), "content", None)
        if getattr(choice, "finish_reason", None) == "length":
            raise ModelResponseError("model_response_truncated")
        if getattr(choice, "finish_reason", None) != "stop" or not isinstance(content, str):
            raise ModelResponseError("model_response_invalid")
        if not content.strip():
            raise ModelResponseError("model_response_empty")
        return content

    return chat_complete


async def _get_keyword_llm_func() -> Any:
    """Lightweight LLM function for keyword extraction using a fast model (no extended thinking)."""
    vlm_client = get_vlm_client(state.settings)
    kw_model = state.settings.keyword_llm_model

    async def keyword_complete(prompt: str, system_prompt: str | None = None, **_: Any) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = await vlm_client.chat.completions.create(
                model=kw_model,
                messages=messages,
                **sdk_chat_request_options(str(getattr(vlm_client, "base_url", "")), kw_model, 512),
                stream=False,
            )
            if hasattr(resp, "choices") and resp.choices:
                return resp.choices[0].message.content
            return str(resp)
        except Exception:
            return ""

    return keyword_complete


async def _get_entity_extraction_llm_func() -> Any:
    """Lightweight LLM for entity extraction during ingestion (no extended thinking)."""
    vlm_client = get_vlm_client(state.settings)
    model = state.settings.entity_extraction_llm_model

    async def entity_extract_complete(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: List[dict] | None = None,
        **_: Any,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        try:
            resp = await vlm_client.chat.completions.create(
                model=model,
                messages=messages,
                **sdk_chat_request_options(str(getattr(vlm_client, "base_url", "")), model, 4096),
                stream=False,
            )
            if hasattr(resp, "choices") and resp.choices:
                return resp.choices[0].message.content
            return str(resp)
        except Exception as e:
            import logging
            logging.getLogger("docthinker.entity_extract").warning(f"Entity extraction LLM failed: {e}")
            return ""

    return entity_extract_complete


async def _warmup_llm_connection():
    """Send a lightweight probe to establish API connection pool early."""
    import logging
    log = logging.getLogger("docthinker.warmup")
    try:
        assert_credential_bindings(state.settings)
    except CredentialBindingError as exc:
        log.warning("Skipping provider warmup: %s", exc)
        return
    try:
        vlm_client = get_vlm_client(state.settings)
        kw_model = state.settings.keyword_llm_model
        log.info(f"Warming up LLM connection with model: {kw_model}")
        resp = await vlm_client.chat.completions.create(
            model=kw_model,
            messages=[{"role": "user", "content": "hi"}],
            **sdk_chat_request_options(str(getattr(vlm_client, "base_url", "")), kw_model, 1),
            stream=False,
        )
        log.info("LLM warmup completed successfully")
    except Exception as e:
        log.warning(f"LLM warmup failed (non-fatal): {e}")


async def _initialize_rag() -> DocThinker | None:
    try:
        assert_credential_bindings(state.settings)
    except CredentialBindingError as exc:
        _log.warning("Model services are unavailable until settings are repaired: %s", exc)
        return None
    from docthinker.auto_thinking.vlm_client import VLMClient as AutoThinkingVLMClient

    embedding_func = await _get_embedding_func()
    chat_complete = await _get_llm_model_func()
    keyword_complete = await _get_keyword_llm_func()
    entity_extract_complete = await _get_entity_extraction_llm_func()
    rerank_func = create_bltcy_rerank_func(
        api_key=state.settings.rerank_api_key,
        base_url=state.settings.rerank_base_url,
        model_name=state.settings.rerank_model,
    )
    config = _create_rag_config()
    vlm_client = AutoThinkingVLMClient(
        api_key=state.settings.llm_api_key,
        api_base=state.settings.vlm_base_url,
        model=state.settings.vlm_model,
    )
    state.vision_vlm_client = vlm_client
    vision_model_func = _make_vision_model_func(vlm_client)

    graphcore_kwargs = {
        "llm_model_max_async": state.settings.graphcore_llm_max_async,
        "embedding_func_max_async": min(state.settings.graphcore_embedding_max_async, 2),
        "max_parallel_insert": state.settings.graphcore_max_parallel_insert,
        "entity_extract_max_gleaning": state.settings.graphcore_max_gleaning,
    }
    if rerank_func:
        graphcore_kwargs["rerank_model_func"] = rerank_func

    # Use a dedicated fast model for entity extraction if configured
    extraction_model = state.settings.extraction_llm_model
    if extraction_model and extraction_model != state.settings.llm_model:
        extraction_router = AsyncModelRouter(
            client=get_vlm_client(state.settings),
            models=[extraction_model],
            max_concurrency=state.settings.llm_router_max_concurrency,
        )

        async def extraction_chat_complete(prompt: str, system_prompt: str | None = None, **_: Any) -> str:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            resp = await extraction_router.chat_completion(
                messages=messages,
                max_tokens=2048,
                stream=False,
                temperature=0.0,
            )
            if not hasattr(resp, "choices") or not resp.choices:
                return str(resp)
            return resp.choices[0].message.content

        graphcore_kwargs["entity_extraction_llm_model_func"] = extraction_chat_complete

    return DocThinker(
        config=config,
        llm_model_func=chat_complete,
        keyword_llm_model_func=keyword_complete,
        entity_extraction_llm_model_func=entity_extract_complete,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        graphcore_kwargs=graphcore_kwargs,
    )


@asynccontextmanager
async def lifespan(app: FastAPI, *, mobile_data_dir: Path | None = None):
    context_assist = os.getenv("DOCTHINKER_RESEARCH_CONTEXT_ASSIST", "false").strip().lower()
    if context_assist not in {"true", "false", "1", "0"}:
        raise ValueError("DOCTHINKER_RESEARCH_CONTEXT_ASSIST must be true, false, 1 or 0")
    state.settings = load_settings()
    state.mobile_safe_mode = mobile_data_dir is not None
    state.long_horizon_backend = None
    state.long_horizon_db_path = None
    state.mobile_document_rag = False
    state.mobile_chat_func = None
    state.research_service = None
    if state.mobile_safe_mode:
        root = Path(mobile_data_dir).expanduser().resolve()
        desktop_workdir = Path(state.settings.workdir).expanduser().resolve()
        desktop_data = SessionManager._infer_data_root(desktop_workdir).resolve()
        # A differently named child of desktop data is not an independent store.
        if any(root == old or root.is_relative_to(old) or old.is_relative_to(root)
               for old in (desktop_workdir, desktop_data)):
            raise ValueError("Mobile data directory must not overlap desktop storage")
        data_root = (root / "data").resolve()
        workdir = (data_root / "_system").resolve()
        if not data_root.is_relative_to(root) or not workdir.is_relative_to(root):
            raise ValueError("Mobile storage must not escape its dedicated directory")
        state.settings = state.settings.model_copy(update={"workdir": str(workdir)})
    state.api_config = APIConfig()
    Path(state.settings.workdir).mkdir(parents=True, exist_ok=True)
    if state.mobile_safe_mode:
        state.session_manager = SessionManager(
            base_storage_path=state.settings.workdir,
            data_root_path=str(data_root), migrate_legacy=False,
        )
        from docthinker.memory_core import SQLiteLongHorizonBackend

        state.long_horizon_db_path = str(workdir / "long_horizon_memory.db")
        state.long_horizon_backend = SQLiteLongHorizonBackend(state.long_horizon_db_path)
    else:
        state.session_manager = SessionManager(base_storage_path=state.settings.workdir)

    if not state.mobile_safe_mode:
        await _warmup_llm_connection()
    else:
        # A personal chat subscription is not an embedding subscription. Do not
        # infer permission from legacy provider/model defaults or fallback keys.
        llm_ready = bool(state.settings.llm_api_key not in {"", "EMPTY"})
        try:
            assert_credential_bindings(state.settings)
        except CredentialBindingError:
            llm_ready = False
        if llm_ready:
            state.mobile_chat_func = await _get_llm_model_func()
        research_enabled = os.getenv("DOCTHINKER_RESEARCH_ENABLED", "").lower() in {"1", "true"}
        state.mobile_document_rag = bool(
            llm_ready and state.settings.embed_api_key not in {"", "EMPTY"}
            and os.getenv("EMBEDDING_BINDING_API_KEY", "").strip()
            and (os.getenv("EMBEDDING_MODEL", "").strip() or os.getenv("EMBED_MODEL", "").strip())
            and (os.getenv("LLM_EMBED_HOST", "").strip() or os.getenv("EMBEDDING_BINDING_HOST", "").strip())
        )
        if research_enabled and state.mobile_document_rag:
            raise ValueError("Bounded research and legacy embedding RAG are separate mobile profiles")
        if research_enabled:
            from .research_service import ResearchService

            # Real-paper evaluation has not established a quality improvement.
            # Keep experimental source expansion behind an operator opt-in.
            state.research_service = ResearchService(
                root, context_assistance_enabled=context_assist in {"true", "1"},
            )
            state.research_service.start()
        if not state.mobile_document_rag:
            state.rag_instance = None
            for name in ("cognitive_processor", "ingestion_service", "orchestrator",
                         "memory_engine", "memory_engine_factory", "claw_manager_factory",
                         "post_upload_learning", "auto_thinking_vlm_client", "vision_vlm_client"):
                setattr(state, name, None)
            state.memory_engines = {}
            state.claw_managers = {}
            state.expanded_node_managers = {}
            try:
                yield
            finally:
                from .routers.project_tasks import shutdown_project_tasks
                await shutdown_project_tasks()
                if state.research_service is not None:
                    await state.research_service.stop()
                    state.research_service = None
            return
    state.rag_instance = await _initialize_rag()
    # Session isolation: do not initialize global GraphCore.
    state.kg_entity_ids = set()
    if state.rag_instance is None:
        state.mobile_document_rag = False
        # Keep settings/session APIs available for recovery, but do not create
        # ingestion, memory or background clients that could transmit documents.
        for name in (
            "cognitive_processor", "ingestion_service", "orchestrator",
            "memory_engine", "memory_engine_factory", "claw_manager_factory",
            "post_upload_learning", "auto_thinking_vlm_client", "vision_vlm_client",
        ):
            setattr(state, name, None)
        state.memory_engines = {}
        state.claw_managers = {}
        yield
        return

    if not state.mobile_safe_mode:
        _cleanup_global_graphcore_artifacts(state.settings.workdir)

    state.cognitive_processor = CognitiveProcessor(
        llm_func=state.rag_instance.llm_model_func,
        embedding_func=state.rag_instance.embedding_func,
        knowledge_graph=None,
    )
    if state.mobile_safe_mode:
        # Only user-requested query/ingestion models are available. Do not create
        # optional memory model factories or the post-upload learning runner.
        state.memory_engine = None
        state.memory_engine_factory = None
        state.memory_engines = {}
        state.claw_manager_factory = None
        state.claw_managers = {}
        state.expanded_node_managers = {}
        state.orchestrator = None
        state.auto_thinking_vlm_client = None
        state.post_upload_learning = None
        state.ingestion_service = IngestionService(
            rag_global=state.rag_instance, session_manager=state.session_manager,
            create_rag_config=_create_rag_config, get_llm_model_func=_get_llm_model_func,
            get_embedding_func=_get_embedding_func,
        )
        try:
            yield
        finally:
            client = getattr(state, "vision_vlm_client", None)
            try:
                if client is not None and hasattr(client, "close"):
                    await client.close()
            finally:
                await state.rag_instance.finalize_storages()
        return
    try:
        from neuro_memory import MemoryEngine

        _embed = getattr(state.rag_instance.embedding_func, "func", state.rag_instance.embedding_func)

        async def _neuro_embed(texts):
            if isinstance(texts, str):
                texts = [texts]
            out = await _embed(texts)
            if hasattr(out, "tolist"):
                return out.tolist()
            return list(out) if out else []

        def _kg_entity_resolver(entity_id: str) -> bool:
            return entity_id in getattr(state, "kg_entity_ids", set())

        def _session_memory_factory(session_id: str):
            if not state.session_manager:
                raise ValueError("Session manager not initialized")
            talk_dir = state.session_manager.get_session_talk_dir(session_id)
            engine = MemoryEngine(
                embedding_func=_neuro_embed,
                llm_func=state.rag_instance.llm_model_func,
                working_dir=str(talk_dir),
                kg_entity_resolver=_kg_entity_resolver,
            )
            engine.load()
            return engine

        state.memory_engine_factory = _session_memory_factory
        state.memory_engines = {}
        state.memory_engine = None
        _log.info("Neuro memory engine initialized in session-scoped mode.")
    except Exception as e:
        _log.warning("Neuro memory engine not initialized: %s", e)
        state.memory_engine = None
        state.memory_engine_factory = None
        state.memory_engines = {}

    # ── Claw tiered memory system ──
    try:
        from claw import ClawMemoryManager, MemoryConfig

        _embed_raw = getattr(state.rag_instance.embedding_func, "func", state.rag_instance.embedding_func)

        async def _claw_embed(texts):
            if isinstance(texts, str):
                texts = [texts]
            out = await _embed_raw(texts)
            return out

        def _claw_manager_factory(session_id: str):
            if not state.session_manager:
                return None
            talk_dir = state.session_manager.get_session_talk_dir(session_id)
            return ClawMemoryManager(
                talk_dir=str(talk_dir),
                llm_func=state.rag_instance.llm_model_func,
                embedding_func=_claw_embed,
                config=MemoryConfig(),
            )

        state.claw_manager_factory = _claw_manager_factory
        state.claw_managers = {}
        _log.info("Claw tiered memory system initialized.")
    except Exception as e:
        _log.warning("Claw memory system not initialized: %s", e)
        state.claw_manager_factory = None
        state.claw_managers = {}

    state.ingestion_service = IngestionService(
        rag_global=state.rag_instance,
        session_manager=state.session_manager,
        create_rag_config=_create_rag_config,
        get_llm_model_func=_get_llm_model_func,
        get_embedding_func=_get_embedding_func,
    )

    # Initialize Auto-Thinking Orchestrator
    try:
        from docthinker.auto_thinking.classifier import ComplexityClassifier
        from docthinker.auto_thinking.decomposer import QuestionDecomposer
        from docthinker.auto_thinking.orchestrator import HybridRAGOrchestrator
        from docthinker.auto_thinking.vlm_client import VLMClient as AutoThinkingVLMClient
        from docthinker.hypergraph import HyperGraphRAG

        at_client = AutoThinkingVLMClient(
            api_key=state.settings.llm_api_key,
            api_base=state.settings.vlm_base_url,
            model=state.settings.vlm_model,
        )
        state.auto_thinking_vlm_client = at_client
        classifier = ComplexityClassifier(vlm_client=at_client)
        decomposer = QuestionDecomposer(vlm_client=at_client)

        # Initialize HyperGraphRAG
        hyper_system = HyperGraphRAG(
            working_dir=state.settings.workdir,
            llm_model_func=state.rag_instance.llm_model_func,
            embedding_func=state.rag_instance.embedding_func,
            vlm_client=at_client,
            graph_construction_mode=state.rag_instance.config.graph_construction_mode,
            spacy_model=state.rag_instance.config.spacy_model,
        )
        
        state.orchestrator = HybridRAGOrchestrator(
            rag_system=state.rag_instance,
            hyper_system=hyper_system,
            classifier=classifier,
            vlm_client=at_client,
            decomposer=decomposer,
            enable_multi_step=True,
            sync_mode="eager", # Sync immediately for demo purposes
        )
        _log.info("Auto-Thinking Orchestrator (Hybrid) initialized.")
    except Exception as e:
        _log.warning("Failed to initialize Auto-Thinking Orchestrator: %s", e)

    from .routers.ingest import _run_post_upload_learning

    learning_runner = SessionLearningRunner(_run_post_upload_learning)
    state.post_upload_learning = learning_runner
    try:
        yield
    finally:
        # Stop tasks while their model clients and graph stores are still alive.
        await learning_runner.shutdown()
        if state.post_upload_learning is learning_runner:
            state.post_upload_learning = None
        save_all_memory_engines()
        for client_attr in ("auto_thinking_vlm_client", "vision_vlm_client"):
            client = getattr(state, client_attr, None)
            if client is not None and hasattr(client, "close"):
                try:
                    await client.close()
                except Exception:
                    pass
        if state.rag_instance:
            await state.rag_instance.finalize_storages()


def create_app() -> FastAPI:
    api_config = APIConfig()
    app = FastAPI(
        title="Multi-Document Enhanced RAG API",
        description="API service for multi-document enhanced RAG system using knowledge graph",
        version="1.0.0",
        lifespan=lifespan,
    )

    if api_config.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=api_config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    routers = []
    try:
        from .routers import health_router, sessions_router, ingest_router, query_router, graph_router, settings_router

        routers = [health_router, sessions_router, ingest_router, query_router, graph_router, settings_router]
    except RuntimeError as exc:
        if "python-multipart" not in str(exc):
            raise
        from .routers.health import router as health_router
        from .routers.settings import router as settings_router

        routers = [health_router, settings_router]
        _log.warning("ingest routes disabled because python-multipart is not installed.")

    for r in routers:
        app.include_router(r, prefix=api_config.api_prefix)

    return app


app = create_app()
