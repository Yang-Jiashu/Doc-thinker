import os
from pydantic import BaseModel

# 项目根目录（doc-thinker），确保 workdir 不随 CWD 变化
_DOC_THINKER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_WORKDIR = os.path.join(_DOC_THINKER_ROOT, "rag_storage_api")


class AppSettings(BaseModel):
    llm_api_key: str
    vlm_base_url: str
    llm_model: str
    vlm_model: str

    embed_api_key: str
    embed_base_url: str
    embed_model: str
    embed_dim: int = 1024
    rerank_api_key: str
    rerank_base_url: str
    rerank_model: str

    workdir: str = "./rag_storage_api"
    timeout_seconds: int = 3600


def load_settings() -> AppSettings:
    llm_api_key = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
    embed_api_key = os.getenv("EMBEDDING_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY") or llm_api_key or "EMPTY"

    return AppSettings(
        llm_api_key=llm_api_key,
        vlm_base_url=os.getenv("LLM_VLM_HOST") or os.getenv("LLM_BINDING_HOST") or "https://api.bltcy.ai/v1",
        llm_model=os.getenv("LLM_MODEL") or "qwen3-8b",
        vlm_model=os.getenv("VLM_MODEL") or "qwen3-vl-235b-a22b",
        embed_api_key=embed_api_key,
        embed_base_url=os.getenv("LLM_EMBED_HOST") or os.getenv("EMBEDDING_BINDING_HOST") or "https://api.bltcy.ai/v1",
        embed_model=os.getenv("EMBEDDING_MODEL") or os.getenv("EMBED_MODEL") or "qwen3-embedding-4b",
        embed_dim=int(os.getenv("EMBEDDING_DIM") or os.getenv("EMBED_DIM") or 1024),
        rerank_api_key=os.getenv("RERANK_API_KEY") or llm_api_key or "EMPTY",
        rerank_base_url=os.getenv("RERANK_HOST") or os.getenv("EMBEDDING_BINDING_HOST") or os.getenv("LLM_BINDING_HOST") or "https://api.bltcy.ai/v1",
        rerank_model=os.getenv("RERANK_MODEL") or "qwen3-reranker-8b",
        workdir=os.getenv("RAG_WORKDIR") or _DEFAULT_WORKDIR,
        timeout_seconds=int(os.getenv("TIMEOUT") or 3600),
    )

