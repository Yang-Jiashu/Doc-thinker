# Implementation Plan: Selective Memory & Graph Visualization

To fully realize your product vision of a "Personal Knowledge Base" with separate-yet-linked storage and selective memory, I will focus on the following key areas:

## 1. Fix & Stabilize Data Storage
We encountered a database error (`datatype mismatch`) due to the Full-Text Search (FTS) index incompatibility with UUIDs.
- **Action**: I have already updated the schema to disable the problematic FTS triggers and deleted the corrupted database. The system will auto-reinitialize on the next start.
- **Goal**: Ensure every Q&A and file upload is correctly persisted without errors.

## 2. Implement "Selective Memory" (Hybrid Query)
You requested that users can "selectively call relevant memory" across multiple sessions.
- **Current State**: We have `session` (isolated) and `global` (all-encompassing) modes.
- **New Logic**: Implement the `hybrid` mode in `api_multi_document.py`.
  - **Step 1**: Query the **Current Session Graph** for immediate context.
  - **Step 2**: Query the **Global Hypergraph** for broader historical knowledge.
  - **Step 3**: Use the LLM to synthesize an answer that prioritizes the session context but supplements it with global knowledge, effectively "selecting" relevant past information.

## 3. Realize "Super Large Hypergraph" Visualization
Users need to "see" the knowledge graph. Currently, the UI shows mock data.
- **Backend**: Create a real `/knowledge-graph/data` endpoint in `api_multi_document.py` that extracts nodes (entities) and edges (relations) from the actual `LightRAG` NetworkX storage.
- **Frontend**: Update `knowledge_graph.html` to fetch this real data, allowing users to switch views between their **Current Session Graph** and the **Global Hypergraph**.

## 4. Enable "Edit" Functionality (Foundation)
Users want to "edit" the graph.
- **Action**: Implement basic API endpoints (`/entities/update`, `/relationships/delete`) that modify the underlying Graph/Vector stores.
- **Note**: This allows users to correct the knowledge graph if the automatic extraction was inaccurate.

## Execution Roadmap
1.  **Restart & Verify**: Ensure DB writes are working.
2.  **Backend Logic**: Implement `hybrid` query merging and `get_graph_data` extraction.
3.  **UI Connection**: Connect the visualization page to real data.
4.  **Testing**: Run a flow creating multiple sessions and verifying the "Separate" vs "Global" storage structure.
