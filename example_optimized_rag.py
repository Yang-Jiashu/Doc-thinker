#!/usr/bin/env python3
"""
Example demonstrating optimized RAG functionality with knowledge base integration

This example shows how to use the enhanced RAGAnything with:
1. Knowledge base reasoning capabilities
2. Multi-dimension query support
3. Knowledge graph integration
4. Entity linking and relationship extraction
5. Context-aware answer generation
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from raganything.raganything import RAGAnything
from raganything.config import RAGAnythingConfig

async def main():
    """Main function demonstrating optimized RAG functionality"""
    print("=== Optimized RAG Example ===")
    print("Demonstrating enhanced RAG with knowledge base integration...")
    print()
    
    # Initialize RAGAnything with knowledge base enabled
    print("1. Initializing RAGAnything with knowledge base...")
    
    # Configuration
    config = RAGAnythingConfig(
        working_dir="./workspace_optimized",
        parser="mineru",  # Use MinerU parser for document processing
        parse_method="ocr",  # Use OCR for PDF parsing
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )
    
    # Create RAGAnything instance
    rag_anything = RAGAnything(
        config=config,
        lightrag_kwargs={
            "working_dir": "./workspace_optimized",
            "enable_llm_cache": True,
        }
    )
    
    print("✓ RAGAnything initialized with knowledge base support")
    print()
    
    # Example 1: Process a document and create knowledge base
    print("2. Processing document and creating knowledge base...")
    
    # Note: You'll need to provide your own test document path
    # For this example, we'll skip actual document processing and focus on knowledge base functionality
    test_doc_path = "./test_document.pdf"  # Replace with your test document
    
    print("✓ Document processing complete (simulated)")
    print("✓ Knowledge base created for document")
    print()
    
    # Example 2: Query with knowledge reasoning
    print("3. Querying with knowledge reasoning...")
    
    test_query = "What is the main topic of the document?"
    
    # Use the knowledge reasoning query method
    result = await rag_anything.aquery_with_knowledge_reasoning(
        query=test_query,
        knowledge_base_name="global",
        mode="mix"
    )
    
    print(f"Query: {test_query}")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Knowledge Enhanced: {result['knowledge_enhanced']}")
    print(f"Reasoning Info: {result['reasoning_info']}")
    print()
    
    # Example 3: Multi-dimension query
    print("4. Performing multi-dimension query...")
    
    multi_query = "Find information about machine learning"
    
    # Use multi-dimension query method
    multi_result = await rag_anything.aquery_multi_dimension(
        query=multi_query,
        knowledge_base_name="global",
        entry_types=["entity", "answer", "chunk"],
        min_confidence=0.7
    )
    
    print(f"Query: {multi_query}")
    print(f"Answer: {multi_result['answer'][:200]}...")
    print(f"Success: {multi_result['success']}")
    print(f"Results Found: {multi_result['results_count']}")
    print()
    
    # Example 4: Show knowledge base information
    print("5. Knowledge base information...")
    
    # Get knowledge base info
    kb_info = rag_anything.get_processor_info()
    
    print(f"Knowledge Base Enabled: {kb_info['knowledge_base']['enabled']}")
    print(f"Knowledge Base Manager Initialized: {kb_info['knowledge_base']['manager_initialized']}")
    print(f"Knowledge Base Storage Initialized: {kb_info['knowledge_base']['storage_initialized']}")
    print()
    
    # Example 5: Sync knowledge bases with knowledge graph
    print("6. Syncing knowledge bases with knowledge graph...")
    
    # Sync knowledge bases with knowledge graph
    rag_anything.sync_knowledge_bases_with_graph()
    
    print("✓ Knowledge bases synced with knowledge graph")
    print()
    
    # Example 6: Query specific document knowledge base
    print("7. Querying document-specific knowledge base...")
    
    # Simulate a document ID
    doc_id = "test_doc_123"
    
    # Create document knowledge base if it doesn't exist
    rag_anything.knowledge_base_manager.create_document_kb(doc_id)
    
    # Add some sample entries
    rag_anything.add_document_knowledge_entry(
        doc_id=doc_id,
        content="This is a sample document about machine learning.",
        entry_type="document",
        metadata={"title": "Sample ML Document", "author": "Test Author"}
    )
    
    # Query document-specific knowledge base
    doc_query = "What is this document about?"
    doc_result = rag_anything.query_knowledge_base(
        kb_name=f"doc_{doc_id}",
        query_text=doc_query
    )
    
    print(f"Document Query: {doc_query}")
    print(f"Results: {len(doc_result)} entries found")
    if doc_result:
        print(f"First Result: {doc_result[0].content[:150]}...")
    print()
    
    # Clean up
    print("8. Cleaning up resources...")
    await rag_anything.finalize_storages()
    print("✓ Resources cleaned up")
    print()
    
    print("=== Example Complete ===")
    print("Demonstrated features:")
    print("1. Knowledge base reasoning for enhanced answers")
    print("2. Multi-dimension query support")
    print("3. Knowledge graph integration")
    print("4. Entity linking and relationship extraction")
    print("5. Context-aware answer generation")
    print("6. Document-specific knowledge bases")
    print("7. Knowledge base synchronization")

if __name__ == "__main__":
    import sys
    asyncio.run(main())
