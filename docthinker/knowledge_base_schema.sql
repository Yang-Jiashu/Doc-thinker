-- Knowledge Base Schema
-- This file defines the database schema for the hierarchical knowledge base system

-- Create knowledge_bases table to store knowledge base information
CREATE TABLE IF NOT EXISTS knowledge_bases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL, -- global, document, task, user
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT -- JSON string for additional metadata
);

-- Create index on knowledge_bases.type for faster queries by type
CREATE INDEX IF NOT EXISTS idx_knowledge_bases_type ON knowledge_bases(type);

-- Create index on knowledge_bases.name for faster lookups by name
CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_bases_name ON knowledge_bases(name);

-- Create knowledge_entries table to store individual knowledge entries
CREATE TABLE IF NOT EXISTS knowledge_entries (
    id TEXT PRIMARY KEY, -- UUID as string
    kb_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    type TEXT NOT NULL, -- generic, document, question, answer, entity, relationship
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT, -- JSON string for additional metadata
    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
);

-- Create index on knowledge_entries.kb_id for faster queries within a knowledge base
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_kb_id ON knowledge_entries(kb_id);

-- Create index on knowledge_entries.type for faster queries by entry type
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_type ON knowledge_entries(type);

-- Create index on knowledge_entries.created_at for time-based queries
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_created_at ON knowledge_entries(created_at);

-- Create full-text search index for content
-- DISABLED due to UUID primary key incompatibility with FTS5 rowid
-- CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_entries_fts USING fts5(
--    content,
--    content=knowledge_entries, 
--    content_rowid=id
-- );

-- Create knowledge_relations table to store relationships between entries
CREATE TABLE IF NOT EXISTS knowledge_relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES knowledge_entries(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES knowledge_entries(id) ON DELETE CASCADE
);

-- Create index on knowledge_relations.source_id for faster lookups by source
CREATE INDEX IF NOT EXISTS idx_knowledge_relations_source ON knowledge_relations(source_id);

-- Create index on knowledge_relations.target_id for faster lookups by target
CREATE INDEX IF NOT EXISTS idx_knowledge_relations_target ON knowledge_relations(target_id);

-- Create index on knowledge_relations.relation_type for faster queries by relation type
CREATE INDEX IF NOT EXISTS idx_knowledge_relations_type ON knowledge_relations(relation_type);

-- Create knowledge_base_metadata table for flexible metadata storage
CREATE TABLE IF NOT EXISTS knowledge_base_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kb_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    UNIQUE(kb_id, key) -- Ensure only one value per key per knowledge base
);

-- Create index on knowledge_base_metadata.kb_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_kb_metadata_kb_id ON knowledge_base_metadata(kb_id);

-- Create index on knowledge_base_metadata.key for faster queries by key
CREATE INDEX IF NOT EXISTS idx_kb_metadata_key ON knowledge_base_metadata(key);

-- Create knowledge_entry_metadata table for flexible metadata storage
CREATE TABLE IF NOT EXISTS knowledge_entry_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (entry_id) REFERENCES knowledge_entries(id) ON DELETE CASCADE,
    UNIQUE(entry_id, key) -- Ensure only one value per key per entry
);

-- Create index on knowledge_entry_metadata.entry_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_entry_metadata_entry_id ON knowledge_entry_metadata(entry_id);

-- Create index on knowledge_entry_metadata.key for faster queries by key
CREATE INDEX IF NOT EXISTS idx_entry_metadata_key ON knowledge_entry_metadata(key);

-- Create trigger to update updated_at timestamp when knowledge base is modified
CREATE TRIGGER IF NOT EXISTS update_knowledge_base_timestamp
AFTER UPDATE ON knowledge_bases
BEGIN
    UPDATE knowledge_bases SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Create trigger to update updated_at timestamp when knowledge entry is modified
CREATE TRIGGER IF NOT EXISTS update_knowledge_entry_timestamp
AFTER UPDATE ON knowledge_entries
BEGIN
    UPDATE knowledge_entries SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Create trigger to update full-text search index when entry is inserted
-- CREATE TRIGGER IF NOT EXISTS knowledge_entries_ai AFTER INSERT ON knowledge_entries BEGIN
--    INSERT INTO knowledge_entries_fts(rowid, content) VALUES (NEW.id, NEW.content);
-- END;

-- Create trigger to update full-text search index when entry is updated
-- CREATE TRIGGER IF NOT EXISTS knowledge_entries_au AFTER UPDATE ON knowledge_entries BEGIN
--    UPDATE knowledge_entries_fts SET content = NEW.content WHERE rowid = NEW.id;
-- END;

-- Create trigger to delete from full-text search index when entry is deleted
-- CREATE TRIGGER IF NOT EXISTS knowledge_entries_ad AFTER DELETE ON knowledge_entries BEGIN
--    DELETE FROM knowledge_entries_fts WHERE rowid = OLD.id;
-- END;

-- Create view to get all entries with their knowledge base information
CREATE VIEW IF NOT EXISTS v_entries_with_kb AS
SELECT 
    ke.id, 
    ke.content, 
    ke.type AS entry_type,
    ke.created_at AS entry_created_at,
    ke.updated_at AS entry_updated_at,
    ke.metadata AS entry_metadata,
    kb.id AS kb_id,
    kb.name AS kb_name,
    kb.type AS kb_type,
    kb.metadata AS kb_metadata
FROM knowledge_entries ke
JOIN knowledge_bases kb ON ke.kb_id = kb.id;

-- Create view to get all relations with source and target entry information
CREATE VIEW IF NOT EXISTS v_relations_with_entries AS
SELECT 
    kr.id AS relation_id,
    kr.source_id,
    kr.target_id,
    kr.relation_type,
    kr.created_at AS relation_created_at,
    s.content AS source_content,
    s.type AS source_type,
    t.content AS target_content,
    t.type AS target_type,
    s.kb_id AS source_kb_id,
    t.kb_id AS target_kb_id
FROM knowledge_relations kr
JOIN knowledge_entries s ON kr.source_id = s.id
JOIN knowledge_entries t ON kr.target_id = t.id;
