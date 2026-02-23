"""
Example demonstrating the multi-document enhanced query functionality

This example shows how to use the new multi-document query enhancement feature
that leverages the knowledge graph to find related documents and provide more
comprehensive answers.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List

import httpx
import numpy as np
from openai import AsyncOpenAI
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc

# ====== Config (adjust as needed) ======
API_KEY = os.getenv("LLM_BINDING_API_KEY") or "<YOUR_API_KEY>"
BASE_URL = os.getenv("LLM_BINDING_HOST") or "https://api.bltcy.ai/v1"
# VLM 服务（本地 vLLM 转发）的 OpenAI 兼容地址
VLM_BASE_URL = os.getenv("LLM_VLM_HOST") or "http://127.0.0.1:22004/v1"
# VLM 模型名（启动 vLLM 时的名称/路径）
LLM_MODEL = os.getenv("LLM_MODEL") or "/home/yjs/robot/VLM/32B"

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 4096  # change if your provider uses a different dimension
# 本地 embedding 服务
EMBED_BASE_URL = os.getenv("LLM_EMBED_HOST") or "http://127.0.0.1:8808/v1"
EMBED_MODEL = os.getenv("EMBEDDING_MODEL") or "/home/yjs/robot/VLM/8Bembedding"

WORKDIR = "./rag_storage_multi_doc"  # isolated working dir
# ======================================


def build_rag() -> RAGAnything:
    """Build and configure RAGAnything instance"""
    vlm_client = AsyncOpenAI(api_key="EMPTY", base_url=VLM_BASE_URL, timeout=3600)
    embed_client = AsyncOpenAI(api_key="EMPTY", base_url=EMBED_BASE_URL, timeout=3600)

    async def chat_complete(prompt: str, system_prompt: str | None = None, **_: Any) -> str:
        """Chat completion function"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = await vlm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=2048,
            stream=False,
        )
        if not hasattr(resp, "choices") or not resp.choices:
            return str(resp)
        return resp.choices[0].message.content

    async def embedding_func_impl(texts: List[str]) -> Any:
        """Embedding function implementation"""
        resp = await embed_client.embeddings.create(
            model=EMBED_MODEL,
            input=texts,
        )
        if not hasattr(resp, "data"):
            raise RuntimeError(f"Unexpected embedding response: {resp}")
        vectors: List[List[float]] = []
        for item in resp.data:
            emb = getattr(item, "embedding", None)
            if isinstance(emb, list):
                vectors.append(emb)
        return np.array(vectors, dtype=np.float32)

    embedding_func = EmbeddingFunc(
        embedding_dim=EMBED_DIM,
        max_token_size=8192,
        func=embedding_func_impl,
    )

    config = RAGAnythingConfig(
        working_dir=WORKDIR,
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    return RAGAnything(
        config=config,
        llm_model_func=chat_complete,
        embedding_func=embedding_func,
    )


async def add_sample_documents(rag: RAGAnything):
    """Add sample documents to RAG with entities and relationships"""
    # Example: Add documents with entities and relationships
    # In a real scenario, you would parse actual documents and extract entities/relationships
    
    # Sample document 1: Machine Learning Basics
    doc1_content = """
    Machine learning is a subset of artificial intelligence that allows systems to learn from data.
    Key concepts include supervised learning, unsupervised learning, and reinforcement learning.
    
    Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data.
    Reinforcement learning involves agents learning through trial and error.
    """
    
    # Sample document 2: Deep Learning Techniques
    doc2_content = """
    Deep learning is a type of machine learning that uses neural networks with many layers.
    Convolutional Neural Networks (CNNs) are used for image recognition.
    Recurrent Neural Networks (RNNs) are used for sequential data like text.
    
    Machine learning algorithms often serve as the foundation for deep learning models.
    """
    
    # Add sample entities and relationships to knowledge graph
    # This would normally be done automatically during document processing
    rag.knowledge_graph.add_entity("Machine Learning", "Field", document_id="doc_1")
    rag.knowledge_graph.add_entity("Artificial Intelligence", "Field", document_id="doc_1")
    rag.knowledge_graph.add_entity("Deep Learning", "Field", document_id="doc_2")
    rag.knowledge_graph.add_entity("Neural Networks", "Concept", document_id="doc_2")
    rag.knowledge_graph.add_entity("CNNs", "Algorithm", document_id="doc_2")
    rag.knowledge_graph.add_entity("RNNs", "Algorithm", document_id="doc_2")
    
    # Add relationships
    ml_entity = rag.knowledge_graph.get_entity_by_name("Machine Learning")
    ai_entity = rag.knowledge_graph.get_entity_by_name("Artificial Intelligence")
    dl_entity = rag.knowledge_graph.get_entity_by_name("Deep Learning")
    nn_entity = rag.knowledge_graph.get_entity_by_name("Neural Networks")
    cnn_entity = rag.knowledge_graph.get_entity_by_name("CNNs")
    rnn_entity = rag.knowledge_graph.get_entity_by_name("RNNs")
    
    if ml_entity and ai_entity:
        rag.knowledge_graph.add_relationship(
            ml_entity.id, ai_entity.id, "is_subset_of", document_id="doc_1"
        )
    
    if dl_entity and ml_entity:
        rag.knowledge_graph.add_relationship(
            dl_entity.id, ml_entity.id, "is_type_of", document_id="doc_2"
        )
    
    if dl_entity and nn_entity:
        rag.knowledge_graph.add_relationship(
            dl_entity.id, nn_entity.id, "uses", document_id="doc_2"
        )
    
    if cnn_entity and nn_entity:
        rag.knowledge_graph.add_relationship(
            cnn_entity.id, nn_entity.id, "is_type_of", document_id="doc_2"
        )
    
    if rnn_entity and nn_entity:
        rag.knowledge_graph.add_relationship(
            rnn_entity.id, nn_entity.id, "is_type_of", document_id="doc_2"
        )
    
    # Save knowledge graph
    rag.knowledge_graph.save(str(Path(WORKDIR) / "knowledge_graph.json"))
    
    print("Added sample documents and knowledge graph relationships")


async def run_multi_document_query(rag: RAGAnything):
    """Run a multi-document enhanced query"""
    print("\n=== Running Multi-Document Enhanced Query ===")
    
    # Example question that spans multiple documents
    question = "What is the relationship between deep learning and artificial intelligence?"
    
    print(f"Question: {question}")
    
    # Use the new multi-document query enhancement
    result = await rag.aquery_multi_document_enhanced(question, mode="hybrid")
    
    print(f"\nAnswer: {result['answer']}")
    print(f"\nRelated Documents: {result['related_documents']}")
    print(f"Extracted Entities: {result['extracted_entities']}")
    
    # Compare with regular query
    print("\n=== Running Regular Query for Comparison ===")
    regular_result = await rag.aquery(question, mode="hybrid")
    print(f"Regular Answer: {regular_result}")


async def main():
    """Main function"""
    print("=== Initializing RAG System ===")
    rag = build_rag()
    
    # Add sample documents with entities and relationships
    await add_sample_documents(rag)
    
    # Run multi-document query
    await run_multi_document_query(rag)
    
    # Finalize storages
    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
