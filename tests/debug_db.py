import sqlite3
import json
import uuid
from pathlib import Path

DB_PATH = "./rag_storage_api/knowledge_base.db"

def test_insert():
    print(f"Connecting to {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # 1. Create KB
    kb_name = f"test_session_{uuid.uuid4()}"
    print(f"Creating KB: {kb_name}")
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO knowledge_bases (name, type, metadata) VALUES (?, ?, ?)",
            (kb_name, "session", json.dumps({"test": True}))
        )
        kb_id = cursor.lastrowid
        print(f"Created KB ID: {kb_id} (Type: {type(kb_id)})")
        conn.commit()
    except Exception as e:
        print(f"Error creating KB: {e}")
        return

    # 2. Add Entry
    entry_id = str(uuid.uuid4())
    content = "Test content"
    entry_type = "question"
    metadata = {"role": "user"}
    
    print(f"Inserting Entry: {entry_id}")
    try:
        cursor.execute(
            "INSERT INTO knowledge_entries (id, kb_id, content, type, metadata) VALUES (?, ?, ?, ?, ?)",
            (entry_id, kb_id, content, entry_type, json.dumps(metadata))
        )
        print("Insert successful")
        conn.commit()
    except Exception as e:
        print(f"Error inserting entry: {e}")
        
    conn.close()

if __name__ == "__main__":
    test_insert()
