# Implementation Plan for Personal Knowledge Base with Session Memory & Dual-Graph Storage

To realize your vision of a personal knowledge base that supports dynamic inputs, persistent multi-turn memory, and isolated-yet-integrated storage (Global + Session Graphs), I will implement the following architecture:

## 1. Data Architecture: "Session" + "Global" Dual Storage
We will adopt a structure that isolates session data while maintaining a unified global view.

- **Global Knowledge Graph (The "Hypergraph")**:
  - **Storage**: `rag_storage_api/` (Existing root storage).
  - **Function**: Stores the merged knowledge from *all* documents and interactions. This is the "super large hypergraph".

- **Session Knowledge Graphs (Isolated Memories)**:
  - **Storage**: `rag_storage_api/sessions/<session_id>/` (New directories).
  - **Function**: Each conversation session (Dialogue) has its own independent LightRAG storage.
  - **Content**: Only contains the 0, 1, or 4 files uploaded specifically for that dialogue.

- **Metadata & History (SQLite)**:
  - **Storage**: `rag_storage_api/knowledge_base.db`.
  - **Function**:
    - Track **Sessions** (as `knowledge_bases` of type 'session').
    - Store **Chat History** (User Q & System A) as `knowledge_entries`.
    - Link **Files** to Sessions.

## 2. Backend Implementation Steps

### A. Session Manager (`raganything/session_manager.py`)
Create a new module to handle the lifecycle of sessions:
- `create_session(name)`: Creates DB entry and physical directory.
- `get_session_rag(session_id)`: Initializes a `RAGAnything` instance pointing to the session's specific storage path.
- `save_interaction(session_id, query, response)`: Persists chat history to SQLite.

### B. API Enhancements (`api_multi_document.py`)
Update the API to be session-aware:
- **New Endpoints**:
  - `POST /sessions`: Create a new dialogue session.
  - `GET /sessions`: List all past conversations.
  - `GET /sessions/{id}/history`: Retrieve chat history for the UI.
- **Modified Endpoints**:
  - `POST /upload`: Now accepts `session_id`.
    - **Dual-Write**: Files are processed into the **Session Graph** (for isolation) AND the **Global Graph** (for integration).
  - `POST /query`: Now accepts `session_id` and `memory_mode`.
    - **Local Mode**: Query only the Session Graph (context of current files).
    - **Global Mode**: Query the Global Graph.
    - **Hybrid Mode**: Query both and merge results (Selective Memory).

## 3. Frontend/UI Updates
- **Sidebar**: Add a "Conversation History" list to switch between sessions.
- **Chat Interface**:
  - Display persistent history when loading a session.
  - Show which files are attached to the current session.
- **Graph Editor**:
  - Allow users to toggle between viewing the **Current Session Graph** and the **Global Hypergraph**.
  - (Future) Add edit capabilities for nodes/edges.

## 4. Execution Roadmap
1.  **Database**: Verify `knowledge_base.db` schema (confirmed compatible).
2.  **Backend Logic**: Implement `SessionManager` and update `RAGAnything` initialization to support dynamic paths.
3.  **API**: Refactor `api_multi_document.py` to support sessions.
4.  **UI**: Update `index.html` and `app.py` to support the new session-based workflow.

This approach ensures that "Dialogue A with 1 file" is physically isolated from "Dialogue B with 4 files", while the "Global Graph" accumulates everything for cross-session intelligence.
