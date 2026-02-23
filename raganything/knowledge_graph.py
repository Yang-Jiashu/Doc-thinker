"""Knowledge Graph module for multi-document relationship management"""
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from uuid import uuid4
from pathlib import Path
from functools import lru_cache
import hashlib
import re
import json
import time
import sqlite3

# Import entity extractor
from raganything.entity_extractor import EntityLinker


@dataclass
class Entity:
    """Entity in the knowledge graph"""
    id: str
    name: str
    type: str
    properties: Dict = field(default_factory=dict)
    document_ids: Set[str] = field(default_factory=set)
    aliases: Set[str] = field(default_factory=set)  # Alternative names for the entity
    description: str = field(default="")  # Detailed description of the entity
    confidence: float = field(default=1.0)  # Confidence score for the entity
    created_at: float = field(default_factory=lambda: time.time())  # Creation timestamp
    updated_at: float = field(default_factory=lambda: time.time())  # Last update timestamp
    sources: Set[str] = field(default_factory=set)  # Sources that mention this entity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "properties": self.properties,
            "document_ids": list(self.document_ids),
            "aliases": list(self.aliases),
            "description": self.description,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sources": list(self.sources)
        }
    
    def update(self, updates: Dict[str, Any]):
        """Update entity properties"""
        if "name" in updates:
            self.name = updates["name"]
        if "type" in updates:
            self.type = updates["type"]
        if "properties" in updates:
            self.properties.update(updates["properties"])
        if "description" in updates:
            self.description = updates["description"]
        if "confidence" in updates:
            self.confidence = updates["confidence"]
        if "aliases" in updates:
            self.aliases.update(updates["aliases"])
        if "sources" in updates:
            self.sources.update(updates["sources"])
        self.updated_at = time.time()
    
    def add_alias(self, alias: str):
        """Add an alias to the entity"""
        self.aliases.add(alias)
        self.updated_at = time.time()
    
    def add_source(self, source: str):
        """Add a source to the entity"""
        self.sources.add(source)
        self.updated_at = time.time()


@dataclass
class Relationship:
    """Relationship between entities in the knowledge graph"""
    id: str
    source_id: str
    target_id: str
    type: str
    properties: Dict = field(default_factory=dict)
    document_ids: Set[str] = field(default_factory=set)
    description: str = field(default="")  # Detailed description of the relationship
    confidence: float = field(default=1.0)  # Confidence score for the relationship
    created_at: float = field(default_factory=lambda: time.time())  # Creation timestamp
    updated_at: float = field(default_factory=lambda: time.time())  # Last update timestamp
    sources: Set[str] = field(default_factory=set)  # Sources that mention this relationship
    is_validated: bool = field(default=False)  # Whether the relationship has been validated
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary"""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "properties": self.properties,
            "document_ids": list(self.document_ids),
            "description": self.description,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sources": list(self.sources),
            "is_validated": self.is_validated
        }
    
    def update(self, updates: Dict[str, Any]):
        """Update relationship properties"""
        if "type" in updates:
            self.type = updates["type"]
        if "properties" in updates:
            self.properties.update(updates["properties"])
        if "description" in updates:
            self.description = updates["description"]
        if "confidence" in updates:
            self.confidence = updates["confidence"]
        if "sources" in updates:
            self.sources.update(updates["sources"])
        if "is_validated" in updates:
            self.is_validated = updates["is_validated"]
        self.updated_at = time.time()
    
    def validate(self):
        """Mark relationship as validated"""
        self.is_validated = True
        self.updated_at = time.time()
    
    def add_source(self, source: str):
        """Add a source to the relationship"""
        self.sources.add(source)
        self.updated_at = time.time()


# Storage abstraction layer
class KnowledgeGraphStorage:
    """Abstract base class for knowledge graph storage backends"""
    
    def save_entity(self, entity: Entity) -> None:
        """Save an entity to storage"""
        pass
    
    def save_relationship(self, relationship: Relationship) -> None:
        """Save a relationship to storage"""
        pass
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity from storage"""
        pass
    
    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Get a relationship from storage"""
        pass
    
    def get_all_entities(self) -> Dict[str, Entity]:
        """Get all entities from storage"""
        pass
    
    def get_all_relationships(self) -> Dict[str, Relationship]:
        """Get all relationships from storage"""
        pass
    
    def delete_entity(self, entity_id: str) -> None:
        """Delete an entity from storage"""
        pass
    
    def delete_relationship(self, relationship_id: str) -> None:
        """Delete a relationship from storage"""
        pass
    
    def bulk_save_entities(self, entities: Dict[str, Entity]) -> None:
        """Bulk save entities to storage"""
        pass
    
    def bulk_save_relationships(self, relationships: Dict[str, Relationship]) -> None:
        """Bulk save relationships to storage"""
        pass


class InMemoryKnowledgeGraphStorage(KnowledgeGraphStorage):
    """In-memory storage backend for knowledge graph"""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
    
    def save_entity(self, entity: Entity) -> None:
        self.entities[entity.id] = entity
    
    def save_relationship(self, relationship: Relationship) -> None:
        self.relationships[relationship.id] = relationship
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)
    
    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        return self.relationships.get(relationship_id)
    
    def get_all_entities(self) -> Dict[str, Entity]:
        return self.entities
    
    def get_all_relationships(self) -> Dict[str, Relationship]:
        return self.relationships
    
    def delete_entity(self, entity_id: str) -> None:
        if entity_id in self.entities:
            del self.entities[entity_id]
    
    def delete_relationship(self, relationship_id: str) -> None:
        if relationship_id in self.relationships:
            del self.relationships[relationship_id]
    
    def bulk_save_entities(self, entities: Dict[str, Entity]) -> None:
        self.entities.update(entities)
    
    def bulk_save_relationships(self, relationships: Dict[str, Relationship]) -> None:
        self.relationships.update(relationships)


class FileKnowledgeGraphStorage(KnowledgeGraphStorage):
    """File-based storage backend for knowledge graph"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
    
    def save_entity(self, entity: Entity) -> None:
        # Load all data, update, and save back
        data = self._load_data()
        data["entities"][entity.id] = entity.to_dict()
        self._save_data(data)
    
    def save_relationship(self, relationship: Relationship) -> None:
        # Load all data, update, and save back
        data = self._load_data()
        data["relationships"][relationship.id] = relationship.to_dict()
        self._save_data(data)
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        data = self._load_data()
        entity_dict = data["entities"].get(entity_id)
        if entity_dict:
            return Entity(
                id=entity_dict["id"],
                name=entity_dict["name"],
                type=entity_dict["type"],
                properties=entity_dict["properties"],
                document_ids=set(entity_dict["document_ids"]),
                aliases=set(entity_dict.get("aliases", [])),
                description=entity_dict.get("description", ""),
                confidence=entity_dict.get("confidence", 1.0),
                created_at=entity_dict.get("created_at", time.time()),
                updated_at=entity_dict.get("updated_at", time.time()),
                sources=set(entity_dict.get("sources", []))
            )
        return None
    
    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        data = self._load_data()
        rel_dict = data["relationships"].get(relationship_id)
        if rel_dict:
            return Relationship(
                id=rel_dict["id"],
                source_id=rel_dict["source_id"],
                target_id=rel_dict["target_id"],
                type=rel_dict["type"],
                properties=rel_dict["properties"],
                document_ids=set(rel_dict["document_ids"]),
                description=rel_dict.get("description", ""),
                confidence=rel_dict.get("confidence", 1.0),
                created_at=rel_dict.get("created_at", time.time()),
                updated_at=rel_dict.get("updated_at", time.time()),
                sources=set(rel_dict.get("sources", [])),
                is_validated=rel_dict.get("is_validated", False)
            )
        return None
    
    def get_all_entities(self) -> Dict[str, Entity]:
        data = self._load_data()
        entities = {}
        for entity_id, entity_dict in data["entities"].items():
            entities[entity_id] = Entity(
                id=entity_dict["id"],
                name=entity_dict["name"],
                type=entity_dict["type"],
                properties=entity_dict["properties"],
                document_ids=set(entity_dict["document_ids"]),
                aliases=set(entity_dict.get("aliases", [])),
                description=entity_dict.get("description", ""),
                confidence=entity_dict.get("confidence", 1.0),
                created_at=entity_dict.get("created_at", time.time()),
                updated_at=entity_dict.get("updated_at", time.time()),
                sources=set(entity_dict.get("sources", []))
            )
        return entities
    
    def get_all_relationships(self) -> Dict[str, Relationship]:
        data = self._load_data()
        relationships = {}
        for rel_id, rel_dict in data["relationships"].items():
            relationships[rel_id] = Relationship(
                id=rel_dict["id"],
                source_id=rel_dict["source_id"],
                target_id=rel_dict["target_id"],
                type=rel_dict["type"],
                properties=rel_dict["properties"],
                document_ids=set(rel_dict["document_ids"]),
                description=rel_dict.get("description", ""),
                confidence=rel_dict.get("confidence", 1.0),
                created_at=rel_dict.get("created_at", time.time()),
                updated_at=rel_dict.get("updated_at", time.time()),
                sources=set(rel_dict.get("sources", [])),
                is_validated=rel_dict.get("is_validated", False)
            )
        return relationships
    
    def delete_entity(self, entity_id: str) -> None:
        data = self._load_data()
        if entity_id in data["entities"]:
            del data["entities"][entity_id]
            self._save_data(data)
    
    def delete_relationship(self, relationship_id: str) -> None:
        data = self._load_data()
        if relationship_id in data["relationships"]:
            del data["relationships"][relationship_id]
            self._save_data(data)
    
    def bulk_save_entities(self, entities: Dict[str, Entity]) -> None:
        data = self._load_data()
        for entity_id, entity in entities.items():
            data["entities"][entity_id] = entity.to_dict()
        self._save_data(data)
    
    def bulk_save_relationships(self, relationships: Dict[str, Relationship]) -> None:
        data = self._load_data()
        for rel_id, rel in relationships.items():
            data["relationships"][rel_id] = rel.to_dict()
        self._save_data(data)
    
    def _load_data(self) -> Dict[str, Any]:
        """Load data from file"""
        if not self.file_path.exists():
            return {"entities": {}, "relationships": {}}
        
        with open(self.file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _save_data(self, data: Dict[str, Any]) -> None:
        """Save data to file"""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)


class SQLiteKnowledgeGraphStorage(KnowledgeGraphStorage):
    """SQLite-based storage backend for knowledge graph"""
    
    def __init__(self, db_path: str = "./knowledge_graph.db"):
        """Initialize SQLite storage"""
        self.db_path = Path(db_path)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        
        try:
            # Create entities table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                properties TEXT NOT NULL DEFAULT '{}',
                document_ids TEXT NOT NULL DEFAULT '[]',
                aliases TEXT NOT NULL DEFAULT '[]',
                description TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 1.0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                sources TEXT NOT NULL DEFAULT '[]'
            )
            """)
            
            # Create relationships table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                type TEXT NOT NULL,
                properties TEXT NOT NULL DEFAULT '{}',
                document_ids TEXT NOT NULL DEFAULT '[]',
                description TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 1.0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                sources TEXT NOT NULL DEFAULT '[]',
                is_validated INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (source_id) REFERENCES entities(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES entities(id) ON DELETE CASCADE
            )
            """)
            
            # Create index for faster entity lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(type)")
            
            conn.commit()
        finally:
            conn.close()
    
    def _get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    
    def save_entity(self, entity: Entity) -> None:
        """Save an entity to SQLite database"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO entities (id, name, type, properties, document_ids, aliases, description, confidence, created_at, updated_at, sources) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entity.id,
                    entity.name,
                    entity.type,
                    json.dumps(entity.properties),
                    json.dumps(list(entity.document_ids)),
                    json.dumps(list(entity.aliases)),
                    entity.description,
                    entity.confidence,
                    entity.created_at,
                    entity.updated_at,
                    json.dumps(list(entity.sources))
                )
            )
            conn.commit()
        finally:
            conn.close()
    
    def save_relationship(self, relationship: Relationship) -> None:
        """Save a relationship to SQLite database"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO relationships (id, source_id, target_id, type, properties, document_ids, description, confidence, created_at, updated_at, sources, is_validated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    relationship.id,
                    relationship.source_id,
                    relationship.target_id,
                    relationship.type,
                    json.dumps(relationship.properties),
                    json.dumps(list(relationship.document_ids)),
                    relationship.description,
                    relationship.confidence,
                    relationship.created_at,
                    relationship.updated_at,
                    json.dumps(list(relationship.sources)),
                    1 if relationship.is_validated else 0
                )
            )
            conn.commit()
        finally:
            conn.close()
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity from SQLite database"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return Entity(
                id=row["id"],
                name=row["name"],
                type=row["type"],
                properties=json.loads(row["properties"]),
                document_ids=set(json.loads(row["document_ids"])),
                aliases=set(json.loads(row["aliases"])),
                description=row["description"],
                confidence=row["confidence"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                sources=set(json.loads(row["sources"]))
            )
        finally:
            conn.close()
    
    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Get a relationship from SQLite database"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM relationships WHERE id = ?", (relationship_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return Relationship(
                id=row["id"],
                source_id=row["source_id"],
                target_id=row["target_id"],
                type=row["type"],
                properties=json.loads(row["properties"]),
                document_ids=set(json.loads(row["document_ids"])),
                description=row["description"],
                confidence=row["confidence"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                sources=set(json.loads(row["sources"])),
                is_validated=bool(row["is_validated"])
            )
        finally:
            conn.close()
    
    def get_all_entities(self) -> Dict[str, Entity]:
        """Get all entities from SQLite database"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM entities")
            rows = cursor.fetchall()
            
            entities = {}
            for row in rows:
                entity = Entity(
                    id=row["id"],
                    name=row["name"],
                    type=row["type"],
                    properties=json.loads(row["properties"]),
                    document_ids=set(json.loads(row["document_ids"])),
                    aliases=set(json.loads(row["aliases"])),
                    description=row["description"],
                    confidence=row["confidence"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    sources=set(json.loads(row["sources"]))
                )
                entities[entity.id] = entity
            
            return entities
        finally:
            conn.close()
    
    def get_all_relationships(self) -> Dict[str, Relationship]:
        """Get all relationships from SQLite database"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM relationships")
            rows = cursor.fetchall()
            
            relationships = {}
            for row in rows:
                relationship = Relationship(
                    id=row["id"],
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    type=row["type"],
                    properties=json.loads(row["properties"]),
                    document_ids=set(json.loads(row["document_ids"])),
                    description=row["description"],
                    confidence=row["confidence"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    sources=set(json.loads(row["sources"])),
                    is_validated=bool(row["is_validated"])
                )
                relationships[relationship.id] = relationship
            
            return relationships
        finally:
            conn.close()
    
    def delete_entity(self, entity_id: str) -> None:
        """Delete an entity from SQLite database"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
            conn.commit()
        finally:
            conn.close()
    
    def delete_relationship(self, relationship_id: str) -> None:
        """Delete a relationship from SQLite database"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM relationships WHERE id = ?", (relationship_id,))
            conn.commit()
        finally:
            conn.close()
    
    def bulk_save_entities(self, entities: Dict[str, Entity]) -> None:
        """Bulk save entities to SQLite database"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Start transaction for bulk insert
            conn.execute("BEGIN TRANSACTION")
            
            for entity in entities.values():
                cursor.execute(
                    "INSERT OR REPLACE INTO entities (id, name, type, properties, document_ids, aliases, description, confidence, created_at, updated_at, sources) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        entity.id,
                        entity.name,
                        entity.type,
                        json.dumps(entity.properties),
                        json.dumps(list(entity.document_ids)),
                        json.dumps(list(entity.aliases)),
                        entity.description,
                        entity.confidence,
                        entity.created_at,
                        entity.updated_at,
                        json.dumps(list(entity.sources))
                    )
                )
            
            conn.commit()
        finally:
            conn.close()
    
    def bulk_save_relationships(self, relationships: Dict[str, Relationship]) -> None:
        """Bulk save relationships to SQLite database"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Start transaction for bulk insert
            conn.execute("BEGIN TRANSACTION")
            
            for relationship in relationships.values():
                cursor.execute(
                    "INSERT OR REPLACE INTO relationships (id, source_id, target_id, type, properties, document_ids, description, confidence, created_at, updated_at, sources, is_validated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        relationship.id,
                        relationship.source_id,
                        relationship.target_id,
                        relationship.type,
                        json.dumps(relationship.properties),
                        json.dumps(list(relationship.document_ids)),
                        relationship.description,
                        relationship.confidence,
                        relationship.created_at,
                        relationship.updated_at,
                        json.dumps(list(relationship.sources)),
                        1 if relationship.is_validated else 0
                    )
                )
            
            conn.commit()
        finally:
            conn.close()


@dataclass
class KnowledgeGraph:
    """Knowledge Graph for managing entities and relationships across documents"""
    entities: Dict[str, Entity] = field(default_factory=dict)
    relationships: Dict[str, Relationship] = field(default_factory=dict)
    entity_name_map: Dict[str, List[str]] = field(default_factory=dict)  # Map from name/alias to entity IDs
    storage: Optional[KnowledgeGraphStorage] = field(default=None)  # Storage backend
    use_storage: bool = field(default=False)  # Whether to use external storage
    reasoning_rules: List[Dict[str, Any]] = field(default_factory=list)  # Reasoning rules for inference
    
    @staticmethod
    @lru_cache(maxsize=4096)
    def _tokenize_for_minhash(text: str, ngram_size: int) -> Tuple[str, ...]:
        text_lower = (text or "").lower()
        tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text_lower)
        chinese_segments = re.findall(r"[\u4e00-\u9fff]+", text_lower)
        for segment in chinese_segments:
            if len(segment) >= ngram_size:
                for i in range(len(segment) - ngram_size + 1):
                    tokens.append(segment[i : i + ngram_size])
        return tuple(t for t in tokens if t)

    @staticmethod
    @lru_cache(maxsize=4096)
    def _minhash_signature(text: str, num_hashes: int, ngram_size: int) -> Tuple[int, ...]:
        tokens = KnowledgeGraph._tokenize_for_minhash(text, ngram_size)
        if not tokens:
            return tuple([0] * num_hashes)
        token_set = set(tokens)
        signature: List[int] = []
        for seed in range(num_hashes):
            min_val = None
            for token in token_set:
                digest = hashlib.blake2b(
                    f"{seed}:{token}".encode("utf-8"),
                    digest_size=8,
                ).digest()
                value = int.from_bytes(digest, "big")
                if min_val is None or value < min_val:
                    min_val = value
            signature.append(min_val or 0)
        return tuple(signature)

    @staticmethod
    def _minhash_similarity(sig1: Tuple[int, ...], sig2: Tuple[int, ...]) -> float:
        if not sig1 or not sig2 or len(sig1) != len(sig2):
            return 0.0
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    def _calculate_name_similarity(self, entity1: Entity, entity2: Entity) -> float:
        name1 = " ".join([entity1.name] + sorted(entity1.aliases))
        name2 = " ".join([entity2.name] + sorted(entity2.aliases))
        sig1 = self._minhash_signature(name1, 64, 2)
        sig2 = self._minhash_signature(name2, 64, 2)
        return self._minhash_similarity(sig1, sig2)

    def add_entity(self, name: str, type: str, properties: Optional[Dict] = None, document_id: Optional[str] = None, 
                  description: str = "", confidence: float = 1.0, aliases: Optional[List[str]] = None, 
                  source: Optional[str] = None) -> Entity:
        """Add entity to knowledge graph with entity disambiguation support"""
        # Normalize name for consistent matching
        normalized_name = name.lower().strip()
        
        # Check if entity with same name already exists
        existing_entity = self.get_entity_by_name(name)
        
        if existing_entity:
            # Update existing entity
            updates = {}
            if document_id:
                existing_entity.document_ids.add(document_id)
            if properties:
                updates["properties"] = properties
            if description:
                updates["description"] = description
            if confidence > existing_entity.confidence:
                updates["confidence"] = confidence
            if aliases:
                updates["aliases"] = aliases
            if source:
                updates["sources"] = [source]
            
            existing_entity.update(updates)
            
            # Save to storage if enabled
            if self.use_storage and self.storage:
                self.storage.save_entity(existing_entity)
            elif not self.use_storage:
                # Update in-memory store
                self.entities[existing_entity.id] = existing_entity
            
            return existing_entity
        
        # Create new entity
        entity_id = str(uuid4())
        entity = Entity(
            id=entity_id,
            name=name,
            type=type,
            properties=properties or {},
            document_ids={document_id} if document_id else set(),
            description=description,
            confidence=confidence,
            aliases=set(aliases) if aliases else set(),
            sources={source} if source else set()
        )
        
        # Add to in-memory cache and entity name map
        self.entities[entity_id] = entity
        
        # Update entity name map for efficient lookup
        if normalized_name not in self.entity_name_map:
            self.entity_name_map[normalized_name] = []
        self.entity_name_map[normalized_name].append(entity_id)
        
        # Add aliases to name map
        for alias in entity.aliases:
            normalized_alias = alias.lower().strip()
            if normalized_alias not in self.entity_name_map:
                self.entity_name_map[normalized_alias] = []
            self.entity_name_map[normalized_alias].append(entity_id)
        
        # Save to storage if enabled
        if self.use_storage and self.storage:
            self.storage.save_entity(entity)
        
        return entity
    
    def add_relationship(self, source_id: str, target_id: str, type: str, properties: Optional[Dict] = None, document_id: Optional[str] = None, 
                        description: str = "", confidence: float = 1.0, source: Optional[str] = None, 
                        validate: bool = False) -> Relationship:
        """Add relationship between entities with validation support"""
        # Validate source and target entities exist
        source_entity = self.get_entity_by_id(source_id)
        target_entity = self.get_entity_by_id(target_id)
        
        if not source_entity:
            raise ValueError(f"Source entity not found: {source_id}")
        if not target_entity:
            raise ValueError(f"Target entity not found: {target_id}")
        
        # Check if similar relationship already exists
        existing_relation = self.get_relationship(source_id, target_id, type)
        if existing_relation:
            # Update existing relationship
            updates = {}
            if document_id:
                existing_relation.document_ids.add(document_id)
            if properties:
                updates["properties"] = properties
            if description:
                updates["description"] = description
            if confidence > existing_relation.confidence:
                updates["confidence"] = confidence
            if source:
                updates["sources"] = [source]
            if validate:
                updates["is_validated"] = True
            
            existing_relation.update(updates)
            
            # Save to storage if enabled
            if self.use_storage and self.storage:
                self.storage.save_relationship(existing_relation)
            elif not self.use_storage:
                # Update in-memory store
                self.relationships[existing_relation.id] = existing_relation
            
            return existing_relation
        
        # Create new relationship
        relationship_id = str(uuid4())
        relationship = Relationship(
            id=relationship_id,
            source_id=source_id,
            target_id=target_id,
            type=type,
            properties=properties or {},
            document_ids={document_id} if document_id else set(),
            description=description,
            confidence=confidence,
            sources={source} if source else set(),
            is_validated=validate
        )
        
        # Add to in-memory cache
        self.relationships[relationship_id] = relationship
        
        # Save to storage if enabled
        if self.use_storage and self.storage:
            self.storage.save_relationship(relationship)
        
        return relationship
    
    def get_relationship(self, source_id: str, target_id: str, type: str) -> Optional[Relationship]:
        """Get relationship by source, target, and type"""
        for rel in self.relationships.values():
            if rel.source_id == source_id and rel.target_id == target_id and rel.type == type:
                return rel
        return None
    
    def get_entity_by_name(self, name: str, type: Optional[str] = None) -> Optional[Entity]:
        """Get entity by name or alias, with optional type filtering"""
        normalized_name = name.lower().strip()
        
        # Check entity_name_map for quick lookup
        if normalized_name in self.entity_name_map:
            entity_ids = self.entity_name_map[normalized_name]
            # Filter by type if specified
            for entity_id in entity_ids:
                entity = self.entities.get(entity_id)
                if entity:
                    if type and entity.type != type:
                        continue
                    return entity
        
        # If not found in name map, fall back to full scan (for backwards compatibility)
        for entity in self.entities.values():
            if (entity.name.lower() == normalized_name or normalized_name in [alias.lower() for alias in entity.aliases]):
                if type and entity.type != type:
                    continue
                return entity
        
        return None
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID"""
        # Check in-memory cache first
        if entity_id in self.entities:
            return self.entities[entity_id]
        
        # If not in cache and storage is enabled, get from storage
        if self.use_storage and self.storage:
            entity = self.storage.get_entity(entity_id)
            if entity:
                # Cache the entity in memory
                self.entities[entity_id] = entity
            return entity
        
        return None
    
    def search_entities(self, keyword: str, limit: int = 10) -> List[Entity]:
        """Search entities by fuzzy keyword matching against names and aliases"""
        key = (keyword or "").strip().lower()
        if not key:
            return []
        results: List[Entity] = []
        for e in self.entities.values():
            if key in e.name.lower() or any(key in a.lower() for a in e.aliases):
                results.append(e)
                if len(results) >= limit:
                    break
        return results
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type"""
        return [entity for entity in self.entities.values() if entity.type == entity_type]
    
    def get_entities_by_confidence(self, min_confidence: float = 0.0) -> List[Entity]:
        """Get all entities with confidence >= min_confidence"""
        return [entity for entity in self.entities.values() if entity.confidence >= min_confidence]
    
    def get_relationships_by_entity(self, entity_id: str, validated_only: bool = False) -> List[Relationship]:
        """Get all relationships involving an entity, optionally filtered by validation status"""
        relationships = [rel for rel in self.relationships.values() 
                        if rel.source_id == entity_id or rel.target_id == entity_id]
        
        if validated_only:
            relationships = [rel for rel in relationships if rel.is_validated]
        
        return relationships
    
    def get_relationships_by_type(self, relationship_type: str) -> List[Relationship]:
        """Get all relationships of a specific type"""
        return [rel for rel in self.relationships.values() if rel.type == relationship_type]
    
    def get_relationships_by_confidence(self, min_confidence: float = 0.0) -> List[Relationship]:
        """Get all relationships with confidence >= min_confidence"""
        return [rel for rel in self.relationships.values() if rel.confidence >= min_confidence]
    
    def get_relationships_by_document(self, document_id: str, validated_only: bool = False) -> List[Relationship]:
        """Get all relationships from a specific document, optionally filtered by validation status"""
        relationships = [rel for rel in self.relationships.values() 
                        if document_id in rel.document_ids]
        
        if validated_only:
            relationships = [rel for rel in relationships if rel.is_validated]
        
        return relationships
    
    def get_entities_by_document(self, document_id: str) -> List[Entity]:
        """Get all entities from a specific document"""
        return [entity for entity in self.entities.values() 
                if document_id in entity.document_ids]
    
    # ------------------------
    # Enhanced Multi-dimension Query Methods
    # ------------------------
    
    def query_entities(self, filters: Optional[Dict[str, Any]] = None, sort_by: Optional[str] = None, 
                      sort_order: str = "desc", limit: int = 100, offset: int = 0) -> List[Entity]:
        """Query entities with complex filters, sorting, and pagination"""
        # Start with all entities
        result = list(self.entities.values())
        
        # Apply filters if provided
        if filters:
            result = self._apply_entity_filters(result, filters)
        
        # Apply sorting if requested
        if sort_by:
            result = self._sort_entities(result, sort_by, sort_order)
        
        # Apply pagination
        result = result[offset:offset + limit]
        
        return result
    
    def _apply_entity_filters(self, entities: List[Entity], filters: Dict[str, Any]) -> List[Entity]:
        """Apply filters to a list of entities"""
        filtered = entities
        
        # Filter by type
        if "type" in filters:
            entity_type = filters["type"]
            filtered = [e for e in filtered if e.type == entity_type]
        
        # Filter by types (list)
        if "types" in filters:
            entity_types = filters["types"]
            filtered = [e for e in filtered if e.type in entity_types]
        
        # Filter by confidence
        if "min_confidence" in filters:
            min_conf = filters["min_confidence"]
            filtered = [e for e in filtered if e.confidence >= min_conf]
        
        if "max_confidence" in filters:
            max_conf = filters["max_confidence"]
            filtered = [e for e in filtered if e.confidence <= max_conf]
        
        # Filter by property
        if "properties" in filters:
            prop_filters = filters["properties"]
            for prop_name, prop_value in prop_filters.items():
                filtered = [e for e in filtered if prop_name in e.properties and e.properties[prop_name] == prop_value]
        
        # Filter by document ID
        if "document_id" in filters:
            doc_id = filters["document_id"]
            filtered = [e for e in filtered if doc_id in e.document_ids]
        
        # Filter by source
        if "source" in filters:
            source = filters["source"]
            filtered = [e for e in filtered if source in e.sources]
        
        return filtered
    
    def _sort_entities(self, entities: List[Entity], sort_by: str, sort_order: str = "desc") -> List[Entity]:
        """Sort entities based on a specific attribute"""
        reverse = sort_order.lower() == "desc"
        
        if sort_by == "confidence":
            return sorted(entities, key=lambda e: e.confidence, reverse=reverse)
        elif sort_by == "created_at":
            return sorted(entities, key=lambda e: e.created_at, reverse=reverse)
        elif sort_by == "updated_at":
            return sorted(entities, key=lambda e: e.updated_at, reverse=reverse)
        elif sort_by == "name":
            return sorted(entities, key=lambda e: e.name.lower(), reverse=reverse)
        elif sort_by == "type":
            return sorted(entities, key=lambda e: e.type.lower(), reverse=reverse)
        else:
            # Default to confidence
            return sorted(entities, key=lambda e: e.confidence, reverse=reverse)
    
    def query_relationships(self, filters: Optional[Dict[str, Any]] = None, sort_by: Optional[str] = None, 
                          sort_order: str = "desc", limit: int = 100, offset: int = 0) -> List[Relationship]:
        """Query relationships with complex filters, sorting, and pagination"""
        # Start with all relationships
        result = list(self.relationships.values())
        
        # Apply filters if provided
        if filters:
            result = self._apply_relationship_filters(result, filters)
        
        # Apply sorting if requested
        if sort_by:
            result = self._sort_relationships(result, sort_by, sort_order)
        
        # Apply pagination
        result = result[offset:offset + limit]
        
        return result
    
    def _apply_relationship_filters(self, relationships: List[Relationship], filters: Dict[str, Any]) -> List[Relationship]:
        """Apply filters to a list of relationships"""
        filtered = relationships
        
        # Filter by type
        if "type" in filters:
            rel_type = filters["type"]
            filtered = [r for r in filtered if r.type == rel_type]
        
        # Filter by types (list)
        if "types" in filters:
            rel_types = filters["types"]
            filtered = [r for r in filtered if r.type in rel_types]
        
        # Filter by confidence
        if "min_confidence" in filters:
            min_conf = filters["min_confidence"]
            filtered = [r for r in filtered if r.confidence >= min_conf]
        
        if "max_confidence" in filters:
            max_conf = filters["max_confidence"]
            filtered = [r for r in filtered if r.confidence <= max_conf]
        
        # Filter by source entity
        if "source_id" in filters:
            source_id = filters["source_id"]
            filtered = [r for r in filtered if r.source_id == source_id]
        
        # Filter by target entity
        if "target_id" in filters:
            target_id = filters["target_id"]
            filtered = [r for r in filtered if r.target_id == target_id]
        
        # Filter by validation status
        if "validated" in filters:
            validated = filters["validated"]
            filtered = [r for r in filtered if r.is_validated == validated]
        
        # Filter by property
        if "properties" in filters:
            prop_filters = filters["properties"]
            for prop_name, prop_value in prop_filters.items():
                filtered = [r for r in filtered if prop_name in r.properties and r.properties[prop_name] == prop_value]
        
        # Filter by document ID
        if "document_id" in filters:
            doc_id = filters["document_id"]
            filtered = [r for r in filtered if doc_id in r.document_ids]
        
        return filtered
    
    def _sort_relationships(self, relationships: List[Relationship], sort_by: str, sort_order: str = "desc") -> List[Relationship]:
        """Sort relationships based on a specific attribute"""
        reverse = sort_order.lower() == "desc"
        
        if sort_by == "confidence":
            return sorted(relationships, key=lambda r: r.confidence, reverse=reverse)
        elif sort_by == "created_at":
            return sorted(relationships, key=lambda r: r.created_at, reverse=reverse)
        elif sort_by == "updated_at":
            return sorted(relationships, key=lambda r: r.updated_at, reverse=reverse)
        elif sort_by == "type":
            return sorted(relationships, key=lambda r: r.type.lower(), reverse=reverse)
        else:
            # Default to confidence
            return sorted(relationships, key=lambda r: r.confidence, reverse=reverse)
    
    def complex_query(self, entity_filters: Optional[Dict[str, Any]] = None, 
                     relationship_filters: Optional[Dict[str, Any]] = None, 
                     sort_by: Optional[str] = None, sort_order: str = "desc", 
                     limit: int = 100) -> Dict[str, Any]:
        """Perform a complex query that returns both entities and relationships"""
        # Query entities
        entities = self.query_entities(entity_filters, sort_by, sort_order, limit)
        
        # Query relationships
        relationships = self.query_relationships(relationship_filters, sort_by, sort_order, limit)
        
        # If we have entity filters but no relationship filters, find relationships between matching entities
        if entity_filters and not relationship_filters:
            entity_ids = {e.id for e in entities}
            relationships = [r for r in relationships if r.source_id in entity_ids or r.target_id in entity_ids]
        
        return {
            "entities": entities,
            "relationships": relationships,
            "total_entities": len(entities),
            "total_relationships": len(relationships)
        }
    
    def search_entities(self, query: str, entity_types: Optional[List[str]] = None, 
                       min_confidence: float = 0.0, limit: int = 20) -> List[Entity]:
        """Search entities by name, alias, or description"""
        query_lower = query.lower()
        results = []
        
        for entity in self.entities.values():
            # Check if entity meets confidence threshold
            if entity.confidence < min_confidence:
                continue
            
            # Check if entity type matches
            if entity_types and entity.type not in entity_types:
                continue
            
            # Check name, aliases, and description
            if (query_lower in entity.name.lower() or
                any(query_lower in alias.lower() for alias in entity.aliases) or
                query_lower in entity.description.lower()):
                results.append(entity)
        
        # Sort by confidence and limit results
        results.sort(key=lambda e: e.confidence, reverse=True)
        return results[:limit]
    
    def _tokenize_text(self, text: str) -> List[str]:
        text_lower = (text or "").lower()
        # 1. Basic tokens (English words + Chinese chars)
        tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text_lower)
        
        # 2. Add bigrams for Chinese characters to capture order info
        # Find continuous Chinese segments first
        chinese_segments = re.findall(r"[\u4e00-\u9fff]+", text_lower)
        for segment in chinese_segments:
            if len(segment) > 1:
                for i in range(len(segment) - 1):
                    tokens.append(segment[i:i+2])
        
        return [t for t in tokens if t]
    
    def _ensure_entity_token_index(self) -> None:
        version = getattr(self, "_entity_token_index_version", None)
        if version == len(self.entities):
            return
        token_index: Dict[str, Set[str]] = {}
        entity_tokens: Dict[str, Set[str]] = {}
        for entity_id, entity in self.entities.items():
            tokens = set(self._tokenize_text(entity.name))
            for alias in entity.aliases:
                tokens.update(self._tokenize_text(alias))
            entity_tokens[entity_id] = tokens
            for token in tokens:
                token_index.setdefault(token, set()).add(entity_id)
        self._entity_token_index = token_index
        self._entity_tokens_map = entity_tokens
        self._entity_token_index_version = len(self.entities)
    
    def fast_match_entities(
        self,
        query: str,
        max_entities: int = 10,
        min_score: float = 0.2,
    ) -> List[Tuple[Entity, float]]:
        if not query:
            return []
        self._ensure_entity_token_index()
        query_lower = query.lower()
        query_tokens = set(self._tokenize_text(query_lower))
        if not query_tokens:
            return []
        candidate_ids: Set[str] = set()
        token_index = getattr(self, "_entity_token_index", {})
        for token in query_tokens:
            candidate_ids.update(token_index.get(token, set()))
        if not candidate_ids:
            return []
        scored: List[Tuple[Entity, float]] = []
        for entity_id in candidate_ids:
            entity = self.entities.get(entity_id)
            if not entity:
                continue
            names = [entity.name] + list(entity.aliases)
            exact_hit = any(name.lower() in query_lower for name in names if name)
            if exact_hit:
                score = 1.0
            else:
                entity_tokens = getattr(self, "_entity_tokens_map", {}).get(entity_id, set())
                overlap = len(query_tokens & entity_tokens)
                union = len(query_tokens | entity_tokens)
                score = overlap / union if union > 0 else 0.0
            if score >= min_score:
                scored.append((entity, score))
        scored.sort(key=lambda x: (x[1], x[0].confidence), reverse=True)
        return scored[:max_entities]
    
    def fast_related_documents(
        self,
        query: str,
        hops: int = 1,
        max_docs: int = 50,
        max_entities: int = 12,
        min_score: float = 0.2,
    ) -> Tuple[Set[str], List[str]]:
        matched = self.fast_match_entities(query, max_entities=max_entities, min_score=min_score)
        if not matched:
            return set(), []
        seed_entities = [entity for entity, _ in matched]
        related_docs: Set[str] = set()
        for entity in seed_entities:
            related_docs.update(entity.document_ids)
        if hops <= 0:
            return related_docs, [e.name for e in seed_entities]
        adjacency: Dict[str, Set[str]] = {}
        for rel in self.relationships.values():
            adjacency.setdefault(rel.source_id, set()).add(rel.target_id)
            adjacency.setdefault(rel.target_id, set()).add(rel.source_id)
            related_docs.update(rel.document_ids)
        visited: Set[str] = set()
        frontier: Set[str] = {e.id for e in seed_entities}
        for _ in range(hops):
            next_frontier: Set[str] = set()
            for node_id in frontier:
                if node_id in visited:
                    continue
                visited.add(node_id)
                for neighbor in adjacency.get(node_id, set()):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        for entity_id in visited:
            entity = self.entities.get(entity_id)
            if entity:
                related_docs.update(entity.document_ids)
        if max_docs and len(related_docs) > max_docs:
            related_docs = set(list(related_docs)[:max_docs])
        return related_docs, [e.name for e in seed_entities]
    
    def get_related_documents(self, document_id: str) -> Set[str]:
        """Get all documents related to a given document"""
        related_docs = set()
        
        # Get entities from this document
        entities = self.get_entities_by_document(document_id)
        
        # Find relationships involving these entities
        for entity in entities:
            relationships = self.get_relationships_by_entity(entity.id)
            for relationship in relationships:
                # Add all documents that mention these relationships
                related_docs.update(relationship.document_ids)
        
        # Remove the original document
        related_docs.discard(document_id)
        
        return related_docs
    
    def merge_entities(self, entity_id1: str, entity_id2: str) -> Entity:
        """Merge two entities into one, keeping the entity with higher confidence"""
        entity1 = self.entities.get(entity_id1)
        entity2 = self.entities.get(entity_id2)
        
        if not entity1 or not entity2:
            raise ValueError("One or both entities not found")
        
        # Determine which entity to keep (higher confidence wins)
        keep_entity = entity1 if entity1.confidence >= entity2.confidence else entity2
        merge_entity = entity2 if keep_entity == entity1 else entity1
        
        # Update keep_entity with merge_entity's data
        keep_entity.document_ids.update(merge_entity.document_ids)
        keep_entity.properties.update(merge_entity.properties)
        keep_entity.aliases.update(merge_entity.aliases)
        keep_entity.sources.update(merge_entity.sources)
        
        # Update description if merge_entity has a better one
        if merge_entity.description and not keep_entity.description:
            keep_entity.description = merge_entity.description
        
        # Update relationships to point to keep_entity
        for rel_id, rel in self.relationships.items():
            if rel.source_id == merge_entity.id:
                rel.source_id = keep_entity.id
            if rel.target_id == merge_entity.id:
                rel.target_id = keep_entity.id
        
        # Remove merge_entity from entities and entity_name_map
        del self.entities[merge_entity.id]
        
        # Clean up entity_name_map
        for name, entity_ids in self.entity_name_map.items():
            if merge_entity.id in entity_ids:
                entity_ids.remove(merge_entity.id)
                # If no more entities for this name, remove the entry
                if not entity_ids:
                    del self.entity_name_map[name]
        
        return keep_entity
    
    def validate_relationship(self, relationship_id: str, validator: str = "user", confidence: Optional[float] = None) -> bool:
        """Validate a relationship with optional validator information"""
        if relationship_id not in self.relationships:
            return False
        
        relationship = self.relationships[relationship_id]
        relationship.validate()
        
        # Update confidence if provided
        if confidence is not None:
            relationship.confidence = max(0.0, min(1.0, confidence))
        
        # Add validation metadata
        if "validation_history" not in relationship.properties:
            relationship.properties["validation_history"] = []
        
        relationship.properties["validation_history"].append({
            "validator": validator,
            "timestamp": time.time(),
            "action": "validated",
            "confidence": relationship.confidence
        })
        
        # Save to storage if enabled
        if self.use_storage and self.storage:
            self.storage.save_relationship(relationship)
        
        return True
    
    def invalidate_relationship(self, relationship_id: str, validator: str = "user", reason: Optional[str] = None) -> bool:
        """Invalidate a relationship with optional reason"""
        if relationship_id not in self.relationships:
            return False
        
        relationship = self.relationships[relationship_id]
        relationship.is_validated = False
        
        # Add validation metadata
        if "validation_history" not in relationship.properties:
            relationship.properties["validation_history"] = []
        
        validation_entry = {
            "validator": validator,
            "timestamp": time.time(),
            "action": "invalidated"
        }
        
        if reason:
            validation_entry["reason"] = reason
        
        relationship.properties["validation_history"].append(validation_entry)
        
        # Save to storage if enabled
        if self.use_storage and self.storage:
            self.storage.save_relationship(relationship)
        
        return True
    
    def validate_all_relationships(self, validator: str = "system") -> int:
        """Validate all relationships"""
        count = 0
        for relationship in self.relationships.values():
            if not relationship.is_validated:
                self.validate_relationship(relationship.id, validator)
                count += 1
        return count
    
    def auto_validate_relationships(self, validation_rules: Optional[List[Dict[str, Any]]] = None) -> int:
        """Automatically validate relationships based on rules"""
        validated_count = 0
        
        # Use provided rules or default rules
        rules = validation_rules or [
            # Default rule: high confidence relationships are automatically validated
            {"type": "confidence_threshold", "threshold": 0.9}
        ]
        
        for relationship in self.relationships.values():
            if relationship.is_validated:
                continue
            
            # Check if relationship matches any validation rule
            for rule in rules:
                if self._match_validation_rule(relationship, rule):
                    self.validate_relationship(relationship.id, validator="auto", confidence=relationship.confidence)
                    validated_count += 1
                    break
        
        return validated_count
    
    def _match_validation_rule(self, relationship: Relationship, rule: Dict[str, Any]) -> bool:
        """Check if a relationship matches a validation rule"""
        rule_type = rule.get("type")
        
        if rule_type == "confidence_threshold":
            # Validate based on confidence score
            threshold = rule.get("threshold", 0.8)
            return relationship.confidence >= threshold
        
        elif rule_type == "relationship_type":
            # Validate based on relationship type
            allowed_types = rule.get("allowed_types", [])
            return relationship.type in allowed_types
        
        elif rule_type == "source_count":
            # Validate based on number of sources
            min_sources = rule.get("min_sources", 2)
            return len(relationship.sources) >= min_sources
        
        elif rule_type == "property_match":
            # Validate based on property values
            required_properties = rule.get("properties", {})
            for prop_name, prop_value in required_properties.items():
                if prop_name not in relationship.properties or relationship.properties[prop_name] != prop_value:
                    return False
            return True
        
        return False
    
    def get_validation_stats(self) -> Dict[str, int]:
        """Get validation statistics"""
        total = len(self.relationships)
        validated = sum(1 for r in self.relationships.values() if r.is_validated)
        invalidated = total - validated
        
        # Calculate by validator type
        by_validator = {}
        for relationship in self.relationships.values():
            if relationship.is_validated:
                validation_history = relationship.properties.get("validation_history", [])
                if validation_history:
                    validator = validation_history[-1].get("validator", "unknown")
                    by_validator[validator] = by_validator.get(validator, 0) + 1
        
        return {
            "total_relationships": total,
            "validated": validated,
            "invalidated": invalidated,
            "validation_rate": validated / total if total > 0 else 0,
            "by_validator": by_validator
        }
    
    def get_relationships_by_validation_status(self, validated: Optional[bool] = None) -> List[Relationship]:
        """Get relationships by validation status"""
        if validated is None:
            return list(self.relationships.values())
        return [r for r in self.relationships.values() if r.is_validated == validated]
    
    def add_validation_rule(self, rule: Dict[str, Any]) -> None:
        """Add a validation rule"""
        # Store validation rules in a dedicated list
        if not hasattr(self, "validation_rules"):
            self.validation_rules = []
        self.validation_rules.append(rule)
    
    def apply_validation_rules(self) -> int:
        """Apply all stored validation rules"""
        if not hasattr(self, "validation_rules"):
            self.validation_rules = []
        return self.auto_validate_relationships(self.validation_rules)
    
    def get_relationship_validation_history(self, relationship_id: str) -> List[Dict[str, Any]]:
        """Get validation history for a relationship"""
        if relationship_id not in self.relationships:
            return []
        return self.relationships[relationship_id].properties.get("validation_history", [])
    
    def disambiguate_entity(self, entity: Entity, context: Optional[str] = None) -> Optional[Entity]:
        """Disambiguate entity using context, returning the most relevant existing entity if found"""
        # Get all entities with similar names/aliases
        potential_matches = []
        normalized_name = entity.name.lower().strip()
        
        # Find all entities with matching name or aliases
        for existing_entity in self.entities.values():
            if existing_entity.id == entity.id:
                continue
            
            # Check name match
            existing_name_lower = existing_entity.name.lower().strip()
            if normalized_name == existing_name_lower:
                potential_matches.append(existing_entity)
                continue
            
            # Check alias match
            for alias in existing_entity.aliases:
                if normalized_name == alias.lower().strip():
                    potential_matches.append(existing_entity)
                    break
            else:
                name_score = self._calculate_name_similarity(entity, existing_entity)
                if name_score >= 0.6:
                    potential_matches.append(existing_entity)
        
        if not potential_matches:
            return None
        
        # Score potential matches based on various criteria
        scored_matches = []
        for match in potential_matches:
            score = self._calculate_entity_similarity(entity, match, context)
            scored_matches.append((match, score))
        
        # Sort by similarity score (descending)
        scored_matches.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best match if similarity is above threshold
        best_match, best_score = scored_matches[0]
        if best_score >= 0.6:  # Similarity threshold
            return best_match
        
        return None
    
    def _calculate_entity_similarity(self, entity1: Entity, entity2: Entity, context: Optional[str] = None) -> float:
        """Calculate similarity score between two entities (0.0 to 1.0)"""
        score = 0.0
        
        name_score = self._calculate_name_similarity(entity1, entity2)
        score += name_score * 0.35
        
        if entity1.type == entity2.type:
            score += 0.2
        
        prop_score = self._calculate_property_similarity(entity1.properties, entity2.properties)
        score += prop_score * 0.2
        
        doc_overlap = self._calculate_document_overlap(entity1, entity2)
        score += doc_overlap * 0.1
        
        rel_similarity = self._calculate_relationship_similarity(entity1, entity2)
        score += rel_similarity * 0.1
        
        if context:
            context_score = self._calculate_context_similarity(entity1, entity2, context)
            score += context_score * 0.05
        
        return min(1.0, max(0.0, score))  # Ensure score is between 0 and 1
    
    def _calculate_property_similarity(self, props1: Dict, props2: Dict) -> float:
        """Calculate similarity between two property dictionaries"""
        if not props1 and not props2:
            return 1.0
        
        common_keys = set(props1.keys()) & set(props2.keys())
        if not common_keys:
            return 0.0
        
        # Calculate matching properties
        matching = 0
        for key in common_keys:
            if props1[key] == props2[key]:
                matching += 1
        
        # Use Jaccard similarity
        total_keys = len(set(props1.keys()) | set(props2.keys()))
        return matching / total_keys
    
    def _calculate_document_overlap(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate overlap between documents mentioning the entities"""
        if not entity1.document_ids or not entity2.document_ids:
            return 0.0
        
        overlap = len(entity1.document_ids & entity2.document_ids)
        total = len(entity1.document_ids | entity2.document_ids)
        
        return overlap / total if total > 0 else 0.0
    
    def _calculate_relationship_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate similarity between entities based on their relationships"""
        # Get relationships for both entities
        rels1 = self.get_relationships_by_entity(entity1.id)
        rels2 = self.get_relationships_by_entity(entity2.id)
        
        if not rels1 or not rels2:
            return 0.0
        
        # Create relationship type sets
        rel_types1 = {rel.type for rel in rels1}
        rel_types2 = {rel.type for rel in rels2}
        
        # Calculate Jaccard similarity of relationship types
        common_types = rel_types1 & rel_types2
        total_types = rel_types1 | rel_types2
        
        if not total_types:
            return 0.0
        
        return len(common_types) / len(total_types)
    
    def _calculate_context_similarity(self, entity1: Entity, entity2: Entity, context: str) -> float:
        """Calculate similarity based on context"""
        # Simple implementation: check if entity descriptions match context
        context_lower = context.lower()
        
        # Check entity1 description
        entity1_match = 0
        if entity1.description and entity1.description.lower() in context_lower:
            entity1_match = 1
        
        # Check entity2 description
        entity2_match = 0
        if entity2.description and entity2.description.lower() in context_lower:
            entity2_match = 1
        
        if entity1_match == 0 and entity2_match == 0:
            return 0.0
        
        return 1.0 if entity2_match > 0 else 0.0
    
    def merge_similar_entities(self, similarity_threshold: float = 0.7) -> int:
        """Merge similar entities based on similarity score"""
        merged_count = 0
        processed_entities = set()
        
        # Iterate through all entities
        for entity_id, entity in self.entities.items():
            if entity_id in processed_entities:
                continue
            
            # Find similar entities
            similar_entities = []
            for other_id, other_entity in self.entities.items():
                if other_id == entity_id or other_id in processed_entities:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_entity_similarity(entity, other_entity)
                if similarity >= similarity_threshold:
                    similar_entities.append(other_entity)
            
            # Merge similar entities
            for similar_entity in similar_entities:
                self.merge_entities(entity_id, similar_entity.id)
                processed_entities.add(similar_entity.id)
                merged_count += 1
        
        return merged_count
    
    def get_similar_entities(self, entity_id: str, similarity_threshold: float = 0.5) -> List[Tuple[Entity, float]]:
        """Get entities similar to the given entity, sorted by similarity score"""
        entity = self.entities.get(entity_id)
        if not entity:
            return []
        
        similar_entities = []
        for other_id, other_entity in self.entities.items():
            if other_id == entity_id:
                continue
            
            similarity = self._calculate_entity_similarity(entity, other_entity)
            if similarity >= similarity_threshold:
                similar_entities.append((other_entity, similarity))
        
        # Sort by similarity (descending)
        similar_entities.sort(key=lambda x: x[1], reverse=True)
        
        return similar_entities
    
    def bulk_add_entities(self, entities: List[Dict[str, Any]]) -> List[Entity]:
        """Bulk add entities for better performance"""
        added_entities = []
        for entity_data in entities:
            entity = self.add_entity(
                name=entity_data["name"],
                type=entity_data["type"],
                properties=entity_data.get("properties"),
                document_id=entity_data.get("document_id"),
                description=entity_data.get("description", ""),
                confidence=entity_data.get("confidence", 1.0),
                aliases=entity_data.get("aliases"),
                source=entity_data.get("source")
            )
            added_entities.append(entity)
        return added_entities
    
    def bulk_add_relationships(self, relationships: List[Dict[str, Any]]) -> List[Relationship]:
        """Bulk add relationships for better performance"""
        added_relationships = []
        for rel_data in relationships:
            try:
                relationship = self.add_relationship(
                    source_id=rel_data["source_id"],
                    target_id=rel_data["target_id"],
                    type=rel_data["type"],
                    properties=rel_data.get("properties"),
                    document_id=rel_data.get("document_id"),
                    description=rel_data.get("description", ""),
                    confidence=rel_data.get("confidence", 1.0),
                    source=rel_data.get("source"),
                    validate=rel_data.get("validate", False)
                )
                added_relationships.append(relationship)
            except ValueError as e:
                print(f"Error adding relationship: {e}")
                continue
        return added_relationships
    
    def save(self, path: str):
        """Save knowledge graph to file"""
        if self.use_storage and self.storage:
            print("Warning: Using external storage, save() method is not supported. Use storage-specific methods instead.")
            return
        
        data = {
            "entities": [entity.to_dict() for entity in self.entities.values()],
            "relationships": [rel.to_dict() for rel in self.relationships.values()],
            "entity_name_map": self.entity_name_map
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))
    
    @classmethod
    def load(cls, path: str) -> 'KnowledgeGraph':
        """Load knowledge graph from file"""
        if not Path(path).exists():
            return cls()
        
        data = json.loads(Path(path).read_text())
        graph = cls()
        
        # Load entities
        for entity_data in data["entities"]:
            entity = Entity(
                id=entity_data["id"],
                name=entity_data["name"],
                type=entity_data["type"],
                properties=entity_data["properties"],
                document_ids=set(entity_data["document_ids"]),
                aliases=set(entity_data.get("aliases", [])),
                description=entity_data.get("description", ""),
                confidence=entity_data.get("confidence", 1.0),
                created_at=entity_data.get("created_at", time.time()),
                updated_at=entity_data.get("updated_at", time.time()),
                sources=set(entity_data.get("sources", []))
            )
            graph.entities[entity.id] = entity
        
        # Load relationships
        for rel_data in data["relationships"]:
            rel = Relationship(
                id=rel_data["id"],
                source_id=rel_data["source_id"],
                target_id=rel_data["target_id"],
                type=rel_data["type"],
                properties=rel_data["properties"],
                document_ids=set(rel_data["document_ids"]),
                description=rel_data.get("description", ""),
                confidence=rel_data.get("confidence", 1.0),
                created_at=rel_data.get("created_at", time.time()),
                updated_at=rel_data.get("updated_at", time.time()),
                sources=set(rel_data.get("sources", [])),
                is_validated=rel_data.get("is_validated", False)
            )
            graph.relationships[rel.id] = rel
        
        # Load entity_name_map
        if "entity_name_map" in data:
            graph.entity_name_map = data["entity_name_map"]
        else:
            # Build entity_name_map if it doesn't exist in the file
            for entity in graph.entities.values():
                # Add name to map
                normalized_name = entity.name.lower().strip()
                if normalized_name not in graph.entity_name_map:
                    graph.entity_name_map[normalized_name] = []
                if entity.id not in graph.entity_name_map[normalized_name]:
                    graph.entity_name_map[normalized_name].append(entity.id)
                
                # Add aliases to map
                for alias in entity.aliases:
                    normalized_alias = alias.lower().strip()
                    if normalized_alias not in graph.entity_name_map:
                        graph.entity_name_map[normalized_alias] = []
                    if entity.id not in graph.entity_name_map[normalized_alias]:
                        graph.entity_name_map[normalized_alias].append(entity.id)
        
        return graph
    
    def initialize_storage(self, storage: KnowledgeGraphStorage, load_from_storage: bool = True) -> None:
        """Initialize storage backend and optionally load existing data"""
        self.storage = storage
        self.use_storage = True
        
        if load_from_storage:
            # Load entities from storage
            self.entities = self.storage.get_all_entities()
            self.relationships = self.storage.get_all_relationships()
            
            # Rebuild entity_name_map
            self.entity_name_map = {}
            for entity in self.entities.values():
                # Add name to map
                normalized_name = entity.name.lower().strip()
                if normalized_name not in self.entity_name_map:
                    self.entity_name_map[normalized_name] = []
                self.entity_name_map[normalized_name].append(entity.id)
                
                # Add aliases to map
                for alias in entity.aliases:
                    normalized_alias = alias.lower().strip()
                    if normalized_alias not in self.entity_name_map:
                        self.entity_name_map[normalized_alias] = []
                    self.entity_name_map[normalized_alias].append(entity.id)
    
    def merge(self, other: 'KnowledgeGraph', document_id: Optional[str] = None):
        """Merge another knowledge graph into this one"""
        # Merge entities
        for entity in other.entities.values():
            merged_entity = self.add_entity(
                name=entity.name,
                type=entity.type,
                properties=entity.properties,
                document_id=document_id,
                description=entity.description,
                confidence=entity.confidence,
                aliases=list(entity.aliases),
                source=None  # Source should be specific to original document
            )
            
            # Update document_ids if document_id is provided
            if document_id:
                merged_entity.document_ids.add(document_id)
        
        # Merge relationships
        for rel in other.relationships.values():
            # Find corresponding entities in this graph
            source_entity = self.get_entity_by_name(
                other.entities[rel.source_id].name,
                other.entities[rel.source_id].type
            )
            target_entity = self.get_entity_by_name(
                other.entities[rel.target_id].name,
                other.entities[rel.target_id].type
            )
            
            if source_entity and target_entity:
                self.add_relationship(
                    source_id=source_entity.id,
                    target_id=target_entity.id,
                    type=rel.type,
                    properties=rel.properties,
                    document_id=document_id,
                    description=rel.description,
                    confidence=rel.confidence,
                    source=None,  # Source should be specific to original document
                    validate=rel.is_validated
                )
    
    # ------------------------
    # Knowledge Reasoning Methods
    # ------------------------
    
    def add_reasoning_rule(self, rule: Dict[str, Any]) -> None:
        """Add a reasoning rule to the knowledge graph"""
        self.reasoning_rules.append(rule)
    
    def apply_reasoning_rules(self) -> List[Relationship]:
        """Apply all reasoning rules to infer new relationships"""
        inferred_relationships = []
        
        for rule in self.reasoning_rules:
            inferred = self._apply_rule(rule)
            inferred_relationships.extend(inferred)
        
        return inferred_relationships
    
    def _apply_rule(self, rule: Dict[str, Any]) -> List[Relationship]:
        """Apply a single reasoning rule"""
        inferred_relationships = []
        
        # Get rule components
        pattern = rule.get("pattern", [])
        conclusion = rule.get("conclusion", {})
        
        if not pattern or not conclusion:
            return inferred_relationships
        
        # Simple implementation: find matching patterns and infer new relationships
        # This would be extended with more sophisticated rule matching in production
        for source_entity in self.entities.values():
            for target_entity in self.entities.values():
                if source_entity.id == target_entity.id:
                    continue
                
                # Check if entities match the pattern
                if self._match_pattern(source_entity, target_entity, pattern):
                    # Create new relationship based on conclusion
                    new_rel = self.add_relationship(
                        source_id=source_entity.id,
                        target_id=target_entity.id,
                        type=conclusion.get("type", "related_to"),
                        properties=conclusion.get("properties", {}),
                        description=f"Inferred using rule: {rule.get('name', 'unnamed')}",
                        confidence=conclusion.get("confidence", 0.8),
                        validate=False
                    )
                    inferred_relationships.append(new_rel)
        
        return inferred_relationships
    
    def _match_pattern(self, source_entity: Entity, target_entity: Entity, pattern: List[Dict[str, Any]]) -> bool:
        """Check if two entities match a pattern"""
        # Simple pattern matching implementation
        # This would be extended with more sophisticated matching in production
        for condition in pattern:
            # Check if relationship exists between entities
            if "relationship" in condition:
                rel_type = condition["relationship"]
                if not self.get_relationship(source_entity.id, target_entity.id, rel_type):
                    return False
        
        return True
    
    def find_path(self, source_id: str, target_id: str, max_depth: int = 3, relationship_types: Optional[List[str]] = None) -> List[List[Relationship]]:
        """Find paths between two entities"""
        paths = []
        
        def dfs(current_id: str, visited: Set[str], current_path: List[Relationship], depth: int):
            if current_id == target_id:
                paths.append(current_path.copy())
                return
            
            if depth >= max_depth:
                return
            
            # Get all outgoing relationships from current entity
            for rel in self.relationships.values():
                if rel.source_id == current_id and rel.target_id not in visited:
                    # Check if relationship type is allowed
                    if not relationship_types or rel.type in relationship_types:
                        visited.add(rel.target_id)
                        current_path.append(rel)
                        dfs(rel.target_id, visited, current_path, depth + 1)
                        current_path.pop()
                        visited.remove(rel.target_id)
        
        # Start DFS from source entity
        visited = {source_id}
        dfs(source_id, visited, [], 0)
        
        return paths
    
    def infer_relationships(self, entity_id: str, relationship_type: str, max_depth: int = 2) -> List[Tuple[Entity, float]]:
        """Infer new relationships for an entity using multi-step reasoning"""
        inferred = []
        
        # Get all direct relationships
        direct_relationships = self.get_relationships_by_entity(entity_id)
        
        # Explore indirect relationships
        for rel in direct_relationships:
            # Get target entity
            target_entity = self.entities.get(rel.target_id)
            if not target_entity:
                continue
            
            # Check if we can infer the desired relationship type
            if rel.type == relationship_type:
                # Direct relationship already exists
                inferred.append((target_entity, rel.confidence))
            else:
                # Check indirect relationships
                indirect_relationships = self.get_relationships_by_entity(target_entity.id)
                for indirect_rel in indirect_relationships:
                    if indirect_rel.type == relationship_type:
                        indirect_target = self.entities.get(indirect_rel.target_id)
                        if indirect_target and indirect_target.id != entity_id:
                            # Calculate confidence for inferred relationship
                            confidence = (rel.confidence * indirect_rel.confidence) / 2
                            inferred.append((indirect_target, confidence))
        
        return inferred
    
    def get_entity_connections(self, entity_id: str, hops: int = 1) -> Dict[str, List[Relationship]]:
        """Get all entities connected to a given entity within N hops"""
        connections = {}
        
        def explore(current_id: str, current_hop: int, path: List[Relationship]):
            if current_hop > hops:
                return
            
            for rel in self.relationships.values():
                if rel.source_id == current_id:
                    target_id = rel.target_id
                    if target_id not in connections:
                        connections[target_id] = []
                    connections[target_id].append(rel)
                    
                    if current_hop < hops:
                        explore(target_id, current_hop + 1, path + [rel])
                elif rel.target_id == current_id:
                    source_id = rel.source_id
                    if source_id not in connections:
                        connections[source_id] = []
                    connections[source_id].append(rel)
                    
                    if current_hop < hops:
                        explore(source_id, current_hop + 1, path + [rel])
        
        explore(entity_id, 0, [])
        return connections
    
    def reason_over_query(self, query: str, entity_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Reason over a query to enhance results"""
        # This is a simplified implementation
        # In production, this would use NLP to parse the query and extract entities/relations
        
        # Extract entities from query (simplified)
        entities = []
        for entity in self.entities.values():
            if entity.name.lower() in query.lower():
                if not entity_types or entity.type in entity_types:
                    entities.append(entity)
        
        # Find relationships between extracted entities
        relationships = []
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                rel = self.get_relationship(entity1.id, entity2.id, "related_to")
                if rel:
                    relationships.append(rel)
                else:
                    # Try to find any relationship between them
                    for rel in self.relationships.values():
                        if (rel.source_id == entity1.id and rel.target_id == entity2.id) or \
                           (rel.source_id == entity2.id and rel.target_id == entity1.id):
                            relationships.append(rel)
                            break
        
        # Infer new relationships using reasoning
        inferred_relationships = []
        for entity in entities:
            inferred = self.infer_relationships(entity.id, "related_to")
            for inferred_entity, confidence in inferred:
                inferred_relationships.append((entity, inferred_entity, confidence))
        
        return {
            "entities": entities,
            "relationships": relationships,
            "inferred_relationships": inferred_relationships
        }
    
    # ------------------------
    # Visualization Methods
    # ------------------------
    
    def get_visualization_data(self, entity_ids: Optional[Set[str]] = None, 
                              relationship_types: Optional[List[str]] = None, 
                              max_entities: int = 100) -> Dict[str, Any]:
        """Get data for visualization in format compatible with most graph visualization libraries"""
        nodes = []
        edges = []
        processed_entity_ids = set()
        
        # Get entities to include
        entities_to_include = []
        if entity_ids:
            # Include specific entities
            for entity_id in entity_ids:
                if entity_id in self.entities:
                    entities_to_include.append(self.entities[entity_id])
        else:
            # Include all entities up to max_entities
            entities_to_include = list(self.entities.values())[:max_entities]
        
        # Create nodes
        for entity in entities_to_include:
            if entity.id in processed_entity_ids:
                continue
            
            # Add node
            node = {
                "id": entity.id,
                "label": entity.name,
                "type": entity.type,
                "properties": entity.properties,
                "confidence": entity.confidence,
                "size": 10 + (entity.confidence * 10),  # Size based on confidence
                "color": self._get_entity_color(entity.type)
            }
            nodes.append(node)
            processed_entity_ids.add(entity.id)
        
        # Create edges
        for relationship in self.relationships.values():
            # Check if both source and target are in processed entities
            if relationship.source_id not in processed_entity_ids or \
               relationship.target_id not in processed_entity_ids:
                continue
            
            # Check if relationship type is allowed
            if relationship_types and relationship.type not in relationship_types:
                continue
            
            # Add edge
            edge = {
                "id": relationship.id,
                "source": relationship.source_id,
                "target": relationship.target_id,
                "label": relationship.type,
                "type": relationship.type,
                "properties": relationship.properties,
                "confidence": relationship.confidence,
                "validated": relationship.is_validated,
                "width": 1 + (relationship.confidence * 3),  # Width based on confidence
                "color": self._get_relationship_color(relationship.type, relationship.is_validated)
            }
            edges.append(edge)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_entities": len(entities_to_include),
                "total_relationships": len(edges),
                "generated_at": time.time()
            }
        }
    
    def export_to_dot(self, file_path: str, entity_ids: Optional[Set[str]] = None, 
                     relationship_types: Optional[List[str]] = None, 
                     max_entities: int = 100) -> None:
        """Export knowledge graph to Graphviz DOT format"""
        viz_data = self.get_visualization_data(entity_ids, relationship_types, max_entities)
        
        # Generate DOT content
        dot_content = "digraph KnowledgeGraph {\n"
        dot_content += "    // Graph settings\n"
        dot_content += "    graph [rankdir=LR, bgcolor=white, fontname='Arial', fontsize=12];\n"
        dot_content += "    node [shape=box, style=filled, fontname='Arial', fontsize=10, margin=0.2];\n"
        dot_content += "    edge [fontname='Arial', fontsize=8];\n\n"
        
        # Add nodes
        dot_content += "    // Nodes\n"
        for node in viz_data["nodes"]:
            dot_content += f"    {node['id'].replace('-', '_')} [label='{node['label']}', fillcolor='{node['color']}', tooltip='Type: {node['type']}\nConfidence: {node['confidence']:.2f}']\n"
        
        dot_content += "\n    // Edges\n"
        for edge in viz_data["edges"]:
            source_id = edge['source'].replace('-', '_')
            target_id = edge['target'].replace('-', '_')
            dot_content += f"    {source_id} -> {target_id} [label='{edge['label']}', color='{edge['color']}', penwidth={edge['width']:.1f}, tooltip='Confidence: {edge['confidence']:.2f}\nValidated: {edge['validated']}']\n"
        
        dot_content += "}\n"
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(dot_content)
    
    def export_to_json(self, file_path: str, entity_ids: Optional[Set[str]] = None, 
                      relationship_types: Optional[List[str]] = None, 
                      max_entities: int = 100) -> None:
        """Export knowledge graph to JSON format for web visualization"""
        viz_data = self.get_visualization_data(entity_ids, relationship_types, max_entities)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(viz_data, f, ensure_ascii=False, indent=2)
    
    def _get_entity_color(self, entity_type: str) -> str:
        """Get color for entity based on type"""
        # Simple color mapping based on entity type
        color_map = {
            "person": "#3498db",  # Blue
            "organization": "#2ecc71",  # Green
            "location": "#e74c3c",  # Red
            "event": "#f39c12",  # Orange
            "concept": "#9b59b6",  # Purple
            "numeric": "#1abc9c",  # Teal
            "document": "#7f8c8d",  # Gray
            "default": "#95a5a6"   # Light gray
        }
        return color_map.get(entity_type.lower(), color_map["default"])
    
    def _get_relationship_color(self, relationship_type: str, validated: bool) -> str:
        """Get color for relationship based on type and validation status"""
        if validated:
            return "#27ae60"  # Green for validated
        
        # Color based on relationship type for non-validated
        color_map = {
            "part_of": "#e67e22",  # Orange
            "related_to": "#95a5a6",  # Light gray
            "causes": "#e74c3c",  # Red
            "has_property": "#3498db",  # Blue
            "owns": "#2ecc71",  # Green
            "default": "#bdc3c7"  # Light gray
        }
        return color_map.get(relationship_type.lower(), color_map["default"])
    
    # ------------------------
    # Version Control Methods
    # ------------------------
    
    def create_snapshot(self, description: str = "", user: str = "system") -> str:
        """Create a snapshot of the current knowledge graph state"""
        snapshot_id = str(uuid4())
        
        # Create snapshot data
        snapshot = {
            "id": snapshot_id,
            "description": description,
            "user": user,
            "timestamp": time.time(),
            "entities": {entity_id: entity.to_dict() for entity_id, entity in self.entities.items()},
            "relationships": {rel_id: rel.to_dict() for rel_id, rel in self.relationships.items()},
            "entity_name_map": self.entity_name_map,
            "metadata": {
                "total_entities": len(self.entities),
                "total_relationships": len(self.relationships)
            }
        }
        
        # Save snapshot
        self._save_snapshot(snapshot)
        
        return snapshot_id
    
    def _save_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Save a snapshot to storage"""
        # For file storage, we'll save snapshots to a snapshots directory
        # For database storage, we'd need a snapshots table
        snapshots_dir = Path("./snapshots")
        snapshots_dir.mkdir(exist_ok=True)
        
        snapshot_path = snapshots_dir / f"snapshot_{snapshot['id']}.json"
        with open(snapshot_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
    
    def get_snapshots(self) -> List[Dict[str, Any]]:
        """Get all available snapshots"""
        snapshots_dir = Path("./snapshots")
        if not snapshots_dir.exists():
            return []
        
        snapshots = []
        
        # Read all snapshot files
        for snapshot_file in snapshots_dir.glob("snapshot_*.json"):
            try:
                with open(snapshot_file, 'r', encoding='utf-8') as f:
                    snapshot = json.load(f)
                    snapshots.append(snapshot)
            except (json.JSONDecodeError, IOError):
                continue
        
        # Sort snapshots by timestamp (newest first)
        snapshots.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return snapshots
    
    def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific snapshot by ID"""
        snapshots_dir = Path("./snapshots")
        snapshot_path = snapshots_dir / f"snapshot_{snapshot_id}.json"
        
        if not snapshot_path.exists():
            return None
        
        try:
            with open(snapshot_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def restore_snapshot(self, snapshot_id: str) -> bool:
        """Restore the knowledge graph from a snapshot"""
        snapshot = self.get_snapshot(snapshot_id)
        if not snapshot:
            return False
        
        # Clear current state
        self.entities.clear()
        self.relationships.clear()
        self.entity_name_map.clear()
        
        # Restore entities
        for entity_dict in snapshot["entities"].values():
            entity = Entity(
                id=entity_dict["id"],
                name=entity_dict["name"],
                type=entity_dict["type"],
                properties=entity_dict["properties"],
                document_ids=set(entity_dict["document_ids"]),
                aliases=set(entity_dict.get("aliases", [])),
                description=entity_dict.get("description", ""),
                confidence=entity_dict.get("confidence", 1.0),
                created_at=entity_dict.get("created_at", time.time()),
                updated_at=entity_dict.get("updated_at", time.time()),
                sources=set(entity_dict.get("sources", []))
            )
            self.entities[entity.id] = entity
        
        # Restore relationships
        for rel_dict in snapshot["relationships"].values():
            relationship = Relationship(
                id=rel_dict["id"],
                source_id=rel_dict["source_id"],
                target_id=rel_dict["target_id"],
                type=rel_dict["type"],
                properties=rel_dict["properties"],
                document_ids=set(rel_dict["document_ids"]),
                description=rel_dict.get("description", ""),
                confidence=rel_dict.get("confidence", 1.0),
                created_at=rel_dict.get("created_at", time.time()),
                updated_at=rel_dict.get("updated_at", time.time()),
                sources=set(rel_dict.get("sources", [])),
                is_validated=rel_dict.get("is_validated", False)
            )
            self.relationships[relationship.id] = relationship
        
        # Restore entity_name_map
        self.entity_name_map = snapshot["entity_name_map"]
        
        return True
    
    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot"""
        snapshots_dir = Path("./snapshots")
        snapshot_path = snapshots_dir / f"snapshot_{snapshot_id}.json"
        
        if not snapshot_path.exists():
            return False
        
        try:
            snapshot_path.unlink()
            return True
        except IOError:
            return False
    
    def compare_snapshots(self, snapshot_id1: str, snapshot_id2: str) -> Dict[str, Any]:
        """Compare two snapshots and return the differences"""
        snapshot1 = self.get_snapshot(snapshot_id1)
        snapshot2 = self.get_snapshot(snapshot_id2)
        
        if not snapshot1 or not snapshot2:
            return {"error": "One or both snapshots not found"}
        
        # Get entity IDs from both snapshots
        entities1 = set(snapshot1["entities"].keys())
        entities2 = set(snapshot2["entities"].keys())
        
        # Get relationship IDs from both snapshots
        relationships1 = set(snapshot1["relationships"].keys())
        relationships2 = set(snapshot2["relationships"].keys())
        
        # Calculate differences
        added_entities = entities2 - entities1
        removed_entities = entities1 - entities2
        common_entities = entities1 & entities2
        
        added_relationships = relationships2 - relationships1
        removed_relationships = relationships1 - relationships2
        common_relationships = relationships1 & relationships2
        
        # Check for modified entities
        modified_entities = []
        for entity_id in common_entities:
            entity1 = snapshot1["entities"][entity_id]
            entity2 = snapshot2["entities"][entity_id]
            if entity1 != entity2:
                modified_entities.append(entity_id)
        
        # Check for modified relationships
        modified_relationships = []
        for rel_id in common_relationships:
            rel1 = snapshot1["relationships"][rel_id]
            rel2 = snapshot2["relationships"][rel_id]
            if rel1 != rel2:
                modified_relationships.append(rel_id)
        
        return {
            "snapshot1": {
                "id": snapshot1["id"],
                "timestamp": snapshot1["timestamp"],
                "description": snapshot1["description"]
            },
            "snapshot2": {
                "id": snapshot2["id"],
                "timestamp": snapshot2["timestamp"],
                "description": snapshot2["description"]
            },
            "entities": {
                "added": list(added_entities),
                "removed": list(removed_entities),
                "modified": modified_entities,
                "total1": len(entities1),
                "total2": len(entities2)
            },
            "relationships": {
                "added": list(added_relationships),
                "removed": list(removed_relationships),
                "modified": modified_relationships,
                "total1": len(relationships1),
                "total2": len(relationships2)
            }
        }
    
    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get the latest snapshot"""
        snapshots = self.get_snapshots()
        if snapshots:
            return snapshots[0]
        return None
    
    def create_auto_snapshot(self, interval_seconds: int = 3600) -> bool:
        """Create an automatic snapshot if enough time has passed since last snapshot"""
        snapshots = self.get_snapshots()
        if snapshots:
            last_snapshot_time = snapshots[0]["timestamp"]
            if time.time() - last_snapshot_time < interval_seconds:
                return False
        
        self.create_snapshot(description="Auto-snapshot", user="system")
        return True
