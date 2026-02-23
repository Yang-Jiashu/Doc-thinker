import os
import uuid
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

from raganything.knowledge_base_storage import KnowledgeBaseStorage
from raganything.knowledge_base import KnowledgeEntry, KnowledgeBase
from raganything.raganything import RAGAnything, RAGAnythingConfig

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manages user sessions, including storage creation and history tracking.
    """
    
    def __init__(self, base_storage_path: str = "./rag_storage_api"):
        self.base_storage_path = Path(base_storage_path)
        self.sessions_dir = self.base_storage_path / "sessions"
        self.kb_storage = KnowledgeBaseStorage(str(self.base_storage_path / "knowledge_base.db"))
        
        # Ensure sessions directory exists
        if not self.sessions_dir.exists():
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            
    def create_session(self, title: Optional[str] = None) -> Dict[str, Any]:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        name = f"session_{session_id}"
        # Default to "New Chat" if no title provided
        display_title = title or "新会话"
        
        metadata = {
            "title": display_title,
            "session_id": session_id,
            "path": str(self.sessions_dir / session_id)
        }
        
        # Create physical directory
        session_path = self.sessions_dir / session_id
        session_path.mkdir(parents=True, exist_ok=True)
        
        # Register in DB
        kb = self.kb_storage.create_knowledge_base(
            name=name,
            kb_type="session",
            metadata=metadata
        )
        
        return {
            "id": session_id,
            "title": display_title,
            "created_at": kb.created_at.isoformat()
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions"""
        kbs = self.kb_storage.list_knowledge_bases(kb_type="session")
        sessions = []
        for kb in kbs:
            metadata = kb.metadata or {}
            title = metadata.get("title", kb.name)
            
            # Sanitize ugly titles (e.g. raw session IDs)
            if title.startswith("session_") or title.startswith("test_session_"):
                title = "未命名会话"
                
            sessions.append({
                "id": metadata.get("session_id", kb.name.replace("session_", "")),
                "title": title,
                "created_at": kb.created_at.isoformat(),
                "file_count": len([e for e in kb.entries.values() if e.entry_type == "document"])
            })
        
        # Sort by creation time desc
        sessions.sort(key=lambda x: x["created_at"], reverse=True)
        return sessions

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session details"""
        name = f"session_{session_id}"
        kb = self.kb_storage.get_knowledge_base(name)
        if not kb:
            return None
            
        metadata = kb.metadata or {}
        return {
            "id": session_id,
            "title": metadata.get("title", kb.name),
            "created_at": kb.created_at.isoformat(),
            "path": metadata.get("path")
        }

    def update_session(self, session_id: str, title: str) -> bool:
        """Update session details"""
        name = f"session_{session_id}"
        kb = self.kb_storage.get_knowledge_base(name)
        if not kb:
            return False
            
        if not kb.metadata:
            kb.metadata = {}
        kb.metadata["title"] = title
        
        return self.kb_storage.update_knowledge_base(kb)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its data"""
        name = f"session_{session_id}"
        
        # Get path before deleting DB entry
        kb = self.kb_storage.get_knowledge_base(name)
        if not kb:
            return False
            
        # Delete from DB
        self.kb_storage.delete_knowledge_base(name)
        
        # Delete physical directory
        session_path = self.sessions_dir / session_id
        if session_path.exists():
            shutil.rmtree(session_path)
            
        return True

    def add_message(self, session_id: str, role: str, content: str) -> str:
        """Add a chat message to session history"""
        name = f"session_{session_id}"
        entry_id = str(uuid.uuid4())
        
        entry = KnowledgeEntry(
            id=entry_id,
            content=content,
            entry_type="question" if role == "user" else "answer",
            metadata={"role": role}
        )
        
        self.kb_storage.add_entry(name, entry)
        return entry_id

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        name = f"session_{session_id}"
        
        # Get all entries
        entries = self.kb_storage.query_entries(name, "")
        
        # Filter for chat messages and sort by time
        history = []
        for entry in entries:
            if entry.entry_type in ["question", "answer"]:
                history.append({
                    "role": entry.metadata.get("role", "user"),
                    "content": entry.content,
                    "created_at": entry.created_at.isoformat()
                })
        
        history.sort(key=lambda x: x["created_at"])
        return history

    def get_files(self, session_id: str) -> List[Dict[str, Any]]:
        """Get files uploaded in a session with processing status"""
        name = f"session_{session_id}"
        
        # Get all entries
        entries = self.kb_storage.query_entries(name, "")
        
        # Load doc_status if available
        doc_status_map = {}
        try:
            session_path = self.sessions_dir / session_id
            status_file = session_path / "kv_store_doc_status.json"
            if status_file.exists():
                import json
                with open(status_file, "r", encoding="utf-8") as f:
                    doc_status_data = json.load(f)
                    # Map filename to status
                    for doc_id, data in doc_status_data.items():
                        if "file_path" in data:
                            # file_path might be full path or basename, try to match basename
                            fname = os.path.basename(data["file_path"])
                            doc_status_map[fname] = data.get("status", "unknown")
        except Exception as e:
            logger.warning(f"Failed to load doc status for session {session_id}: {e}")

        files = []
        for entry in entries:
            if entry.entry_type == "document":
                filename = entry.metadata.get("filename", "unknown")
                status = doc_status_map.get(filename, "pending")
                
                # If we have status map but file is not in it, it might be still processing (not yet indexed)
                # or failed before indexing. Default to "pending" is safe.
                # However, if doc_status_map is empty, it means no processing started or file missing.
                
                files.append({
                    "id": entry.id,
                    "filename": filename,
                    "summary": entry.metadata.get("summary", ""),
                    "created_at": entry.created_at.isoformat(),
                    "status": status,
                    "file_path": entry.metadata.get("file_path"),
                    "file_size": entry.metadata.get("file_size"),
                    "file_ext": entry.metadata.get("file_ext"),
                })
        
        files.sort(key=lambda x: x["created_at"], reverse=True)
        return files

    def add_document_record(
        self,
        session_id: str,
        filename: str,
        content_summary: str = "",
        file_path: Optional[str] = None,
        file_size: Optional[int] = None,
        file_ext: Optional[str] = None,
    ):
        """Record a file upload in the session history"""
        name = f"session_{session_id}"
        entry_id = str(uuid.uuid4())
        
        metadata = {
            "filename": filename,
            "summary": content_summary,
        }
        if file_path:
            metadata["file_path"] = file_path
        if file_size is not None:
            metadata["file_size"] = file_size
        if file_ext:
            metadata["file_ext"] = file_ext

        entry = KnowledgeEntry(
            id=entry_id,
            content=f"Uploaded file: {filename}",
            entry_type="document",
            metadata=metadata,
        )
        
        self.kb_storage.add_entry(name, entry)

    def get_session_rag(self, session_id: str, config: RAGAnythingConfig, lightrag_kwargs: Optional[Dict[str, Any]] = None) -> RAGAnything:
        """Get a RAGAnything instance for a specific session"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
            
        # Create a config for this session
        session_config = RAGAnythingConfig()
        # Copy relevant config from global config
        for key, value in config.__dict__.items():
            if hasattr(session_config, key):
                setattr(session_config, key, value)
        
        # Override working directory
        session_config.working_dir = session["path"]
        
        # Initialize RAG
        return RAGAnything(config=session_config, lightrag_kwargs=lightrag_kwargs or {})
