# Comprehensive Implementation Plan: Personal Knowledge Base

To fully realize your vision of a personal knowledge base that features "Separated & Integrated Storage," "Selective Memory," and "Hypergraph Editing," I have broken down the remaining work into three product-focused phases.

## Phase 1: Perfecting the "Separated Storage" (Session Isolation)
*Goal: Ensure "0 files, 1 file, 4 files" are strictly isolated per dialogue.*

1.  **Fix Conversation UI & Context**:
    *   **Action**: Modify `query.html` to inject the `currentSessionId` into every message and file upload.
    *   **Why**: Currently, uploads are "floating" and not tied to the active session. This fixes the "div not clickable" issue by ensuring the session state is valid.
2.  **Verify "Dual-Write" Ingestion**:
    *   **Action**: Test the `ingest` endpoint to ensure files are saved twice:
        1.  Into `rag_storage_api/sessions/<id>/` (for the "Separated" isolated graph).
        2.  Into `rag_storage_api/` (for the "Integrated" super hypergraph).
    *   **Outcome**: User sees "File A" only in Session A, but "File A" is also available in the Global Search.

## Phase 2: "Selective Memory" & Global Intelligence
*Goal: Allow users to "call relevant memories" via prompt.*

1.  **Implement "Scope Control" in UI**:
    *   **Action**: In the chat interface, refine the "Memory Mode" selector:
        *   **"Focus (Current Session)"**: Only searches the local session graph.
        *   **"Recall (Global Memory)"**: Searches the entire history.
        *   **"Auto (Hybrid)"**: (Default) The system intelligently merges both.
2.  **Enable Natural Language Selection**:
    *   **Action**: Optimize the `hybrid` search logic in `api_multi_document.py`.
    *   **Logic**: When a user prompts "Based on the project plan from last week...", the Hybrid mode will detect the lack of context in the *Current Session*, automatically query the *Global Graph*, and retrieve the relevant node, effectively "selecting" that memory.

## Phase 3: "Super Large Hypergraph" Visualization & Editing
*Goal: "User can view and edit" the knowledge structure.*

1.  **Real-Time Graph Visualization**:
    *   **Action**: Update `knowledge_graph.html` to fetch real data from `/knowledge-graph/data`.
    *   **Feature**: Add a dropdown to switch views: "View Current Dialogue Graph" vs. "View Global Hypergraph".
2.  **Graph Editing Features**:
    *   **Action**: Implement the frontend UI for editing:
        *   **Click Node** -> Open "Edit Entity" Modal -> Rename or Change Type -> Call API.
        *   **Click Edge** -> "Delete Relationship" -> Call API.
    *   **Backend**: Upgrade the mock `update_entity` endpoints to actually persist changes to the `LightRAG` storage (GraphML/JSON files).

## Execution Order (Immediate Next Steps)
1.  **Fix UI**: Modify `query.html` to fix the session binding (Solving the "can't click" and "upload not separated" bugs).
2.  **Visualize**: Connect the Visualization page to real data so you can *see* the graph growing.
3.  **Edit**: Add the "Edit" buttons to the graph view.
