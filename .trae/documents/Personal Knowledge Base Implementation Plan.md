# Implementation Plan: Personal Knowledge Base with Dual-Layer Memory

This plan addresses your vision for a system with **separated session storage**, **integrated global memory**, **selective retrieval**, and **visual graph editing**.

## 1. Core Architecture: Dual-Layer Storage (Refinement)
**Goal**: Ensure every conversation (Session) has its own isolated knowledge graph while contributing to the global "Super Large Hypergraph".

*   **Dual Ingestion Verification**: 
    *   The current backend (`/ingest`) already triggers indexing in both `session_rag` (local) and `rag_instance` (global).
    *   **Task**: Verify this by creating a new session, uploading a distinct file, and confirming it appears in the Session Graph but not in other sessions.
*   **File Management**:
    *   **Task**: Enhance `SessionManager` to track not just file names but their storage paths and status.
    *   **Task**: Add a "Session Files" list in the UI so users can clearly see "This conversation has 4 files".

## 2. Selective Memory & Hybrid Query
**Goal**: Implement the mechanism where "Users choose memory via prompt" and support "Session," "Global," and "Hybrid" modes.

*   **Query Strategies**:
    *   **Current**: Hardcoded "Hybrid" logic merges Session and Global results.
    *   **Improvement**: Expose a **"Memory Scope"** selector in the UI (Current Session vs. Global vs. Hybrid).
*   **Prompt-Driven Selection**:
    *   **Task**: Update the `QueryRequest` to accept a `retrieval_instruction` (e.g., "Only use the marketing PDF").
    *   **Task**: Inject this instruction into the LLM's "Merge Prompt" so the final answer respects the user's constraint on which information source to prioritize.

## 3. Interactive Knowledge Graph (Visual Editing)
**Goal**: Allow users to "View and Edit" the knowledge graph, persisting changes to the backend.

*   **Backend Persistence**:
    *   **Task**: Implement a `save_graph()` method in `api_multi_document.py`. The current mock endpoints modify the graph in memory but do not save to disk (LightRAG storage).
    *   **Task**: Update `PUT /knowledge-graph/entity` and `DELETE /knowledge-graph/relationship` to trigger this save.
*   **Frontend Interaction**:
    *   **Task**: Add an **"Edit Mode"** toggle in the Graph View.
    *   **Task**: When a node is clicked, show a "Node Details" panel allowing modification of properties (Type, Description).
    *   **Task**: Allow right-clicking an edge to "Delete Relationship".

## 4. Visualization of "Hypergraph"
**Goal**: Make the "Super Large Graph" navigable.

*   **Graph Filtering**:
    *   **Task**: Add client-side filters in `knowledge_graph.html` to show/hide nodes by type (e.g., "Show only 'Person' and 'Project' entities").
    *   **Task**: Implement a "Focus" feature: Click a node to expand only its immediate neighbors, keeping the rest of the huge graph hidden.

## Execution Sequence
1.  **Verify & Fix Dual-Write**: Ensure uploading files to a session works flawlessly.
2.  **Enable Graph Editing**: Make the "Edit" buttons real (Backend persistence).
3.  **UI for Memory Selection**: Add the "Memory Scope" dropdown and "Files List" to the chat interface.
4.  **Prompt Customization**: Connect the user's prompt instructions to the retrieval logic.
