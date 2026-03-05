import numpy as np
from contextlib import asynccontextmanager
from typing import Any, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from graphcore.coregraph.utils import EmbeddingFunc

from docthinker import DocThinker, DocThinkerConfig
from docthinker.api_config import APIConfig
from docthinker.cognitive import CognitiveProcessor
from docthinker.providers import load_settings, get_embed_client, get_vlm_client
from docthinker.utils import create_bltcy_rerank_func
from docthinker.services import IngestionService
from docthinker.session_manager import SessionManager
from docthinker.auto_thinking.orchestrator import HybridRAGOrchestrator
from docthinker.auto_thinking.classifier import ComplexityClassifier
from docthinker.auto_thinking.decomposer import QuestionDecomposer
from docthinker.auto_thinking.vlm_client import VLMClient as AutoThinkingVLMClient
from docthinker.hypergraph import HyperGraphRAG

from .state import state
from .routers import health_router, sessions_router, ingest_router, query_router, graph_router


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

    async def embedding_func_impl(texts: List[str]) -> Any:
        resp = await embed_client.embeddings.create(
            model=state.settings.embed_model,
            input=texts,
        )
        vectors: List[List[float]] = []
        for item in resp.data:
            emb = getattr(item, "embedding", None)
            if isinstance(emb, list):
                vectors.append(emb)
        return np.array(vectors, dtype=np.float32)

    return EmbeddingFunc(
        embedding_dim=state.settings.embed_dim,
        max_token_size=8192,
        func=embedding_func_impl,
    )


async def _get_llm_model_func() -> Any:
    vlm_client = get_vlm_client(state.settings)

    async def chat_complete(prompt: str, system_prompt: str | None = None, **_: Any) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = await vlm_client.chat.completions.create(
            model=state.settings.llm_model,
            messages=messages,
            max_tokens=2048,
            stream=False,
        )
        if not hasattr(resp, "choices") or not resp.choices:
            return str(resp)
        return resp.choices[0].message.content

    return chat_complete


async def _initialize_rag() -> DocThinker:
    embedding_func = await _get_embedding_func()
    chat_complete = await _get_llm_model_func()
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

    async def vision_model_func(
        prompt: str,
        *,
        system_prompt: str | None = None,
        image_data: Any = None,
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
            max_tokens=int(kwargs.get("max_tokens", 350)),
            temperature=float(kwargs.get("temperature", 0.2)),
        )

    graphcore_kwargs = {}
    if rerank_func:
        graphcore_kwargs["rerank_model_func"] = rerank_func

    return DocThinker(
        config=config,
        llm_model_func=chat_complete,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
        graphcore_kwargs=graphcore_kwargs,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.settings = load_settings()
    state.api_config = APIConfig()
    state.session_manager = SessionManager(base_storage_path=state.settings.workdir)

    state.rag_instance = await _initialize_rag()
    try:
        await state.rag_instance._ensure_graphcore_initialized()
    except Exception:
        pass

    # 预先加载主 KG 实体 ID，供记忆桥接边 (MENTIONS) 判断
    try:
        if state.rag_instance and state.rag_instance.graphcore:
            G = state.rag_instance.graphcore.chunk_entity_relation_graph
            nodes = await G.get_all_nodes()
            state.kg_entity_ids = {n.get("id", "") for n in nodes if n.get("id")}
    except Exception:
        state.kg_entity_ids = set()

    state.cognitive_processor = CognitiveProcessor(
        llm_func=state.rag_instance.llm_model_func,
        embedding_func=state.rag_instance.embedding_func,
        knowledge_graph=state.rag_instance.knowledge_graph,
    )
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

        state.memory_engine = MemoryEngine(
            embedding_func=_neuro_embed,
            llm_func=state.rag_instance.llm_model_func,
            working_dir=state.settings.workdir,
            kg_entity_resolver=_kg_entity_resolver,
        )
        state.memory_engine.load()
        print("INFO: Neuro memory engine (brain-like association) initialized.")
    except Exception as e:
        print(f"WARNING: Neuro memory engine not initialized: {e}")
        state.memory_engine = None

    state.ingestion_service = IngestionService(
        rag_global=state.rag_instance,
        session_manager=state.session_manager,
        create_rag_config=_create_rag_config,
        get_llm_model_func=_get_llm_model_func,
        get_embedding_func=_get_embedding_func,
    )

    # Initialize Auto-Thinking Orchestrator
    try:
        at_client = AutoThinkingVLMClient(
            api_key=state.settings.llm_api_key,
            api_base=state.settings.vlm_base_url,
            model=state.settings.vlm_model,
        )
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
        print("INFO: Auto-Thinking Orchestrator (Hybrid) initialized.")
    except Exception as e:
        print(f"WARNING: Failed to initialize Auto-Thinking Orchestrator: {e}")

    yield

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

    for r in [health_router, sessions_router, ingest_router, query_router, graph_router]:
        app.include_router(r, prefix=api_config.api_prefix)

    return app


app = create_app()
