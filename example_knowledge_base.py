"""
Example demonstrating the multi-layer knowledge base system

This example shows how to use the new knowledge base functionality to:
1. Create and manage knowledge bases
2. Add entries to knowledge bases
3. Query knowledge bases
4. Use document-specific knowledge bases
5. Use task/question-specific knowledge bases
"""

import os
import asyncio
from pathlib import Path
from typing import Any, Dict, List

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

WORKDIR = "./rag_storage_kb_example"  # isolated working dir
# ======================================


def build_rag() -> RAGAnything:
    """Build and configure RAGAnything instance with knowledge base functionality"""
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


async def example_knowledge_base_basics(rag: RAGAnything):
    """Example demonstrating basic knowledge base functionality"""
    print("\n=== Example 1: Basic Knowledge Base Operations ===")
    
    # Add entries to global knowledge base
    entry1_id = rag.add_knowledge_entry(
        content="Machine learning is a subset of artificial intelligence that allows systems to learn from data.",
        entry_type="concept",
        metadata={"domain": "AI", "source": "example"}
    )
    print(f"Added entry 1 with ID: {entry1_id}")
    
    entry2_id = rag.add_knowledge_entry(
        content="Deep learning is a type of machine learning that uses neural networks with many layers.",
        entry_type="concept",
        metadata={"domain": "AI", "source": "example"}
    )
    print(f"Added entry 2 with ID: {entry2_id}")
    
    entry3_id = rag.add_knowledge_entry(
        content="Supervised learning uses labeled data to train models.",
        entry_type="concept",
        metadata={"domain": "AI", "source": "example"}
    )
    print(f"Added entry 3 with ID: {entry3_id}")
    
    # Query global knowledge base
    print("\nQuerying knowledge base for 'machine learning':")
    results = rag.query_knowledge_base("machine learning")
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result.content[:100]}...")
    
    print("\nQuerying knowledge base for 'deep learning':")
    results = rag.query_knowledge_base("deep learning")
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result.content[:100]}...")


async def example_document_knowledge_base(rag: RAGAnything):
    """Example demonstrating document-specific knowledge bases"""
    print("\n=== Example 2: Document-Specific Knowledge Base ===")
    
    # Simulate document processing
    doc_id = "doc_example_001"
    doc_content = """
    Document Title: Introduction to Artificial Intelligence
    
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence of humans and other animals.
    AI research has been highly successful in developing effective techniques for solving a wide range of problems.
    
    Key areas of AI include:
    - Machine learning
    - Natural language processing
    - Computer vision
    - Robotics
    - Expert systems
    """
    
    # Add document content to document-specific knowledge base
    doc_entry_id = rag.add_document_knowledge_entry(
        doc_id=doc_id,
        content=doc_content,
        entry_type="document",
        metadata={"title": "Introduction to Artificial Intelligence", "author": "Example Author"}
    )
    print(f"Added document entry with ID: {doc_entry_id}")
    
    # Add more entries to the same document knowledge base
    section1_id = rag.add_document_knowledge_entry(
        doc_id=doc_id,
        content="Machine learning is a method of data analysis that automates analytical model building.",
        entry_type="section",
        metadata={"section": "Machine Learning", "page": 1}
    )
    print(f"Added section entry with ID: {section1_id}")
    
    # Query the document-specific knowledge base
    print(f"\nQuerying document '{doc_id}' knowledge base for 'machine learning':")
    doc_kb = rag.knowledge_base_manager.get_document_kb(doc_id)
    if doc_kb:
        results = doc_kb.query("machine learning")
        for i, result in enumerate(results):
            print(f"Result {i+1}: {result.content[:100]}...")


async def example_question_answer_knowledge_base(rag: RAGAnything):
    """Example demonstrating question-answer knowledge base"""
    print("\n=== Example 3: Question-Answer Knowledge Base ===")
    
    # Create a task-specific knowledge base
    task_id = "qa_task_001"
    rag.knowledge_base_manager.create_task_kb(task_id, metadata={"purpose": "QA Evaluation"})
    
    # Add question-answer pairs to the task knowledge base
    qa_kb_name = f"task_{task_id}"
    
    # Question 1
    q1 = "What is artificial intelligence?"
    a1 = "Artificial intelligence is intelligence demonstrated by machines, as opposed to intelligence of humans and other animals."
    
    q1_id = rag.add_knowledge_entry(
        content=q1,
        entry_type="question",
        metadata={"task_id": task_id, "category": "Definition"},
        kb_name=qa_kb_name
    )
    
    a1_id = rag.add_knowledge_entry(
        content=a1,
        entry_type="answer",
        metadata={"task_id": task_id, "question_id": q1_id},
        kb_name=qa_kb_name
    )
    print(f"Added QA pair 1 with IDs: Q={q1_id}, A={a1_id}")
    
    # Question 2
    q2 = "What are the key areas of AI?"
    a2 = "Key areas of AI include machine learning, natural language processing, computer vision, robotics, and expert systems."
    
    q2_id = rag.add_knowledge_entry(
        content=q2,
        entry_type="question",
        metadata={"task_id": task_id, "category": "List"},
        kb_name=qa_kb_name
    )
    
    a2_id = rag.add_knowledge_entry(
        content=a2,
        entry_type="answer",
        metadata={"task_id": task_id, "question_id": q2_id},
        kb_name=qa_kb_name
    )
    print(f"Added QA pair 2 with IDs: Q={q2_id}, A={a2_id}")
    
    # Query all knowledge bases
    print("\nQuerying all knowledge bases for 'key areas of AI':")
    results = rag.query_all_knowledge_bases("key areas of AI")
    for i, result in enumerate(results):
        print(f"Result {i+1} (KB: {result['kb_name']}): {result['entry'].content}")


async def example_integrated_rag_usage(rag: RAGAnything):
    """Example demonstrating integrated RAG and knowledge base usage"""
    print("\n=== Example 4: Integrated RAG and Knowledge Base Usage ===")
    
    # Add some knowledge base entries first
    rag.add_knowledge_entry(
        content="Python is a high-level, interpreted programming language known for its readability and simplicity.",
        entry_type="concept",
        metadata={"domain": "Programming", "source": "example"}
    )
    
    rag.add_knowledge_entry(
        content="NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices.",
        entry_type="library",
        metadata={"domain": "Programming", "source": "example"}
    )
    
    rag.add_knowledge_entry(
        content="Pandas is a software library written for the Python programming language for data manipulation and analysis.",
        entry_type="library",
        metadata={"domain": "Programming", "source": "example"}
    )
    
    # Now use RAG to query
    print("\nUsing RAG to answer: 'What is NumPy and how is it related to Python?'")
    result = await rag.aquery("What is NumPy and how is it related to Python?")
    print(f"Answer: {result}")
    
    # The query and answer should be automatically added to the knowledge base
    print("\nChecking if query and answer were added to knowledge base:")
    results = rag.query_knowledge_base("NumPy")
    for i, result in enumerate(results):
        print(f"Result {i+1} (type: {result.entry_type}): {result.content[:150]}...")


async def main():
    """Main function"""
    print("=== Initializing RAG System with Knowledge Base ===")
    rag = build_rag()
    
    # Run examples
    await example_knowledge_base_basics(rag)
    await example_document_knowledge_base(rag)
    await example_question_answer_knowledge_base(rag)
    await example_integrated_rag_usage(rag)
    
    # Finalize storages
    await rag.finalize_storages()
    print("\n=== All Examples Completed ===")
    print("Knowledge base data has been saved to:")
    print(f"- SQLite database: {Path(WORKDIR) / 'knowledge_base.db'}")
    print(f"- JSON knowledge bases: {Path(WORKDIR) / 'knowledge_bases'}")


if __name__ == "__main__":
    asyncio.run(main())
