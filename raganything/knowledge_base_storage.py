"""
Knowledge Base Storage Engine

This module provides a SQLite-based storage engine for the knowledge base system.
"""

import sqlite3
import json
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from datetime import datetime
from uuid import uuid4

from raganything.knowledge_base import KnowledgeBase, KnowledgeEntry


class KnowledgeBaseStorage:
    """SQLite-based storage engine for knowledge bases"""
    
    def __init__(self, db_path: str = "./knowledge_base.db"):
        """Initialize the storage engine"""
        self.db_path = Path(db_path)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the database with schema"""
        # Check if database exists
        db_exists = self.db_path.exists()
        
        # Connect to database
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        
        try:
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Read schema from file
            schema_path = Path(__file__).parent / "knowledge_base_schema.sql"
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = f.read()
            
            # Execute schema
            conn.executescript(schema)
            conn.commit()
            
            if not db_exists:
                print(f"Created new knowledge base database at {self.db_path}")
            else:
                print(f"Connected to existing knowledge base database at {self.db_path}")
        finally:
            conn.close()
    
    def _get_connection(self):
        """Get a database connection"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    
    def create_knowledge_base(self, name: str, kb_type: str, metadata: Optional[Dict[str, Any]] = None) -> KnowledgeBase:
        """Create a new knowledge base"""
        conn = self._get_connection()
        try:
            # Insert knowledge base
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO knowledge_bases (name, type, metadata) VALUES (?, ?, ?)",
                (name, kb_type, json.dumps(metadata) if metadata else None)
            )
            conn.commit()
            
            # Get the created knowledge base
            cursor.execute("SELECT * FROM knowledge_bases WHERE id = ?", (cursor.lastrowid,))
            kb_row = cursor.fetchone()
            
            if not kb_row:
                raise ValueError(f"Failed to create knowledge base '{name}'")
            
            return self._row_to_knowledge_base(kb_row)
        finally:
            conn.close()
    
    def get_knowledge_base(self, name: str) -> Optional[KnowledgeBase]:
        """Get a knowledge base by name"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM knowledge_bases WHERE name = ?", (name,))
            kb_row = cursor.fetchone()
            
            if not kb_row:
                return None
            
            return self._row_to_knowledge_base(kb_row, conn)
        finally:
            conn.close()
    
    def update_knowledge_base(self, kb: KnowledgeBase) -> bool:
        """Update a knowledge base"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE knowledge_bases SET type = ?, metadata = ? WHERE name = ?",
                (kb.kb_type, json.dumps(kb.metadata), kb.name)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def delete_knowledge_base(self, name: str) -> bool:
        """Delete a knowledge base"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM knowledge_bases WHERE name = ?", (name,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def list_knowledge_bases(self, kb_type: Optional[str] = None) -> List[KnowledgeBase]:
        """List all knowledge bases, optionally filtered by type"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            if kb_type:
                cursor.execute("SELECT * FROM knowledge_bases WHERE type = ?", (kb_type,))
            else:
                cursor.execute("SELECT * FROM knowledge_bases")
            
            kb_rows = cursor.fetchall()
            return [self._row_to_knowledge_base(row) for row in kb_rows]
        finally:
            conn.close()
    
    def add_entry(self, kb_name: str, entry: KnowledgeEntry) -> str:
        """Add an entry to a knowledge base"""
        conn = self._get_connection()
        try:
            # Get knowledge base ID
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM knowledge_bases WHERE name = ?", (kb_name,))
            kb_row = cursor.fetchone()
            
            if not kb_row:
                raise ValueError(f"Knowledge base '{kb_name}' not found")
            
            kb_id = kb_row["id"]
            
            # Insert entry
            cursor.execute(
                "INSERT INTO knowledge_entries (id, kb_id, content, type, metadata) VALUES (?, ?, ?, ?, ?)",
                (entry.id, kb_id, entry.content, entry.entry_type, json.dumps(entry.metadata) if entry.metadata else None)
            )
            
            # Insert relations
            for relation in entry.relations:
                cursor.execute(
                    "INSERT INTO knowledge_relations (source_id, target_id, relation_type) VALUES (?, ?, ?)",
                    (entry.id, relation["target_id"], relation["type"])
                )
            
            conn.commit()
            return entry.id
        finally:
            conn.close()
    
    def get_entry(self, kb_name: str, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get an entry from a knowledge base"""
        conn = self._get_connection()
        try:
            # Get knowledge base ID
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM knowledge_bases WHERE name = ?", (kb_name,))
            kb_row = cursor.fetchone()
            
            if not kb_row:
                return None
            
            kb_id = kb_row["id"]
            
            # Get entry
            cursor.execute(
                "SELECT * FROM knowledge_entries WHERE id = ? AND kb_id = ?",
                (entry_id, kb_id)
            )
            entry_row = cursor.fetchone()
            
            if not entry_row:
                return None
            
            return self._row_to_entry(entry_row, conn)
        finally:
            conn.close()
    
    def update_entry(self, kb_name: str, entry: KnowledgeEntry) -> bool:
        """Update an entry in a knowledge base"""
        conn = self._get_connection()
        try:
            # Get knowledge base ID
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM knowledge_bases WHERE name = ?", (kb_name,))
            kb_row = cursor.fetchone()
            
            if not kb_row:
                return False
            
            kb_id = kb_row["id"]
            
            # Update entry
            cursor.execute(
                "UPDATE knowledge_entries SET content = ?, type = ?, metadata = ? WHERE id = ? AND kb_id = ?",
                (entry.content, entry.entry_type, json.dumps(entry.metadata) if entry.metadata else None, entry.id, kb_id)
            )
            
            # Delete existing relations
            cursor.execute("DELETE FROM knowledge_relations WHERE source_id = ?", (entry.id,))
            
            # Insert new relations
            for relation in entry.relations:
                cursor.execute(
                    "INSERT INTO knowledge_relations (source_id, target_id, relation_type) VALUES (?, ?, ?)",
                    (entry.id, relation["target_id"], relation["type"])
                )
            
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def delete_entry(self, kb_name: str, entry_id: str) -> bool:
        """Delete an entry from a knowledge base"""
        conn = self._get_connection()
        try:
            # Get knowledge base ID
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM knowledge_bases WHERE name = ?", (kb_name,))
            kb_row = cursor.fetchone()
            
            if not kb_row:
                return False
            
            kb_id = kb_row["id"]
            
            # Delete entry (relations will be deleted by foreign key constraint)
            cursor.execute(
                "DELETE FROM knowledge_entries WHERE id = ? AND kb_id = ?",
                (entry_id, kb_id)
            )
            
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def query_entries(self, kb_name: str, query_text: str, entry_types: Optional[List[str]] = None) -> List[KnowledgeEntry]:
        """Query entries in a knowledge base"""
        conn = self._get_connection()
        try:
            # Get knowledge base ID
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM knowledge_bases WHERE name = ?", (kb_name,))
            kb_row = cursor.fetchone()
            
            if not kb_row:
                return []
            
            kb_id = kb_row["id"]
            
            # Build query
            if entry_types:
                placeholders = ",".join(["?"] * len(entry_types))
                cursor.execute(
                    f"SELECT * FROM knowledge_entries WHERE kb_id = ? AND type IN ({placeholders}) AND content LIKE ?",
                    [kb_id] + entry_types + [f"%{query_text}%"]
                )
            else:
                cursor.execute(
                    "SELECT * FROM knowledge_entries WHERE kb_id = ? AND content LIKE ?",
                    (kb_id, f"%{query_text}%")
                )
            
            entry_rows = cursor.fetchall()
            return [self._row_to_entry(row, conn) for row in entry_rows]
        finally:
            conn.close()
    
    def query_entries_fulltext(self, kb_name: str, query_text: str, entry_types: Optional[List[str]] = None) -> List[KnowledgeEntry]:
        """Query entries in a knowledge base using full-text search"""
        conn = self._get_connection()
        try:
            # Get knowledge base ID
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM knowledge_bases WHERE name = ?", (kb_name,))
            kb_row = cursor.fetchone()
            
            if not kb_row:
                return []
            
            kb_id = kb_row["id"]
            
            # Full-text search
            cursor.execute(
                "SELECT ke.* FROM knowledge_entries ke JOIN knowledge_entries_fts kef ON ke.id = kef.rowid WHERE ke.kb_id = ? AND kef.content MATCH ?",
                (kb_id, query_text)
            )
            
            entry_rows = cursor.fetchall()
            
            # Filter by entry type if specified
            if entry_types:
                filtered_rows = [row for row in entry_rows if row["type"] in entry_types]
            else:
                filtered_rows = entry_rows
            
            return [self._row_to_entry(row, conn) for row in filtered_rows]
        finally:
            conn.close()
    
    def get_entries_by_type(self, kb_name: str, entry_type: str) -> List[KnowledgeEntry]:
        """Get all entries of a specific type"""
        conn = self._get_connection()
        try:
            # Get knowledge base ID
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM knowledge_bases WHERE name = ?", (kb_name,))
            kb_row = cursor.fetchone()
            
            if not kb_row:
                return []
            
            kb_id = kb_row["id"]
            
            # Get entries
            cursor.execute(
                "SELECT * FROM knowledge_entries WHERE kb_id = ? AND type = ?",
                (kb_id, entry_type)
            )
            
            entry_rows = cursor.fetchall()
            return [self._row_to_entry(row, conn) for row in entry_rows]
        finally:
            conn.close()
    
    def _row_to_knowledge_base(self, kb_row: sqlite3.Row, conn: Optional[sqlite3.Connection] = None) -> KnowledgeBase:
        """Convert a database row to a KnowledgeBase object"""
        # Create knowledge base without entries first
        metadata = json.loads(kb_row["metadata"]) if kb_row["metadata"] else {}
        kb = KnowledgeBase(
            name=kb_row["name"],
            kb_type=kb_row["type"],
            metadata=metadata,
            created_at=datetime.fromisoformat(kb_row["created_at"]),
            updated_at=datetime.fromisoformat(kb_row["updated_at"])
        )
        
        # If connection is provided, load entries
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM knowledge_entries WHERE kb_id = ?", (kb_row["id"],))
            entry_rows = cursor.fetchall()
            
            for entry_row in entry_rows:
                entry = self._row_to_entry(entry_row, conn)
                kb.entries[entry.id] = entry
        
        return kb
    
    def _row_to_entry(self, entry_row: sqlite3.Row, conn: sqlite3.Connection) -> KnowledgeEntry:
        """Convert a database row to a KnowledgeEntry object"""
        # Get relations
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM knowledge_relations WHERE source_id = ?", (entry_row["id"],))
        relation_rows = cursor.fetchall()
        
        relations = [
            {"type": row["relation_type"], "target_id": row["target_id"]}
            for row in relation_rows
        ]
        
        # Create entry
        metadata = json.loads(entry_row["metadata"]) if entry_row["metadata"] else {}
        return KnowledgeEntry(
            id=entry_row["id"],
            content=entry_row["content"],
            entry_type=entry_row["type"],
            metadata=metadata,
            relations=relations,
            created_at=datetime.fromisoformat(entry_row["created_at"]),
            updated_at=datetime.fromisoformat(entry_row["updated_at"])
        )
    
    def get_all_knowledge_bases(self) -> List[KnowledgeBase]:
        """Get all knowledge bases"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM knowledge_bases")
            kb_rows = cursor.fetchall()
            return [self._row_to_knowledge_base(row) for row in kb_rows]
        finally:
            conn.close()
    
    def get_relations(self, entry_id: str) -> List[Dict[str, str]]:
        """Get all relations for an entry"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM knowledge_relations WHERE source_id = ?",
                (entry_id,)
            )
            relation_rows = cursor.fetchall()
            return [
                {"type": row["relation_type"], "target_id": row["target_id"]}
                for row in relation_rows
            ]
        finally:
            conn.close()
    
    def add_relation(self, source_id: str, target_id: str, relation_type: str) -> bool:
        """Add a relation between two entries"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO knowledge_relations (source_id, target_id, relation_type) VALUES (?, ?, ?)",
                (source_id, target_id, relation_type)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def remove_relation(self, source_id: str, target_id: str) -> bool:
        """Remove a relation between two entries"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM knowledge_relations WHERE source_id = ? AND target_id = ?",
                (source_id, target_id)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def backup_database(self, backup_path: str):
        """Backup the database to a file"""
        import shutil
        shutil.copy2(str(self.db_path), backup_path)
        print(f"Database backed up to {backup_path}")
    
    def restore_database(self, backup_path: str):
        """Restore the database from a backup file"""
        import shutil
        shutil.copy2(backup_path, str(self.db_path))
        print(f"Database restored from {backup_path}")
