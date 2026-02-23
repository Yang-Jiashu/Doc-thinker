"""
Knowledge Base Management Module

This module provides a hierarchical knowledge base system for storing and managing
information related to tasks, questions, and documents.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from uuid import uuid4
from datetime import datetime
from pathlib import Path
import json

# Forward declaration for knowledge graph integration
class KnowledgeGraph:
    pass


@dataclass
class KnowledgeEntry:
    """A single entry in the knowledge base"""
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    entry_type: str = "generic"  # generic, document, question, answer, entity, relationship
    metadata: Dict[str, Any] = field(default_factory=dict)
    relations: List[Dict[str, str]] = field(default_factory=list)  # list of {"type": str, "target_id": str}
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    entity_links: List[str] = field(default_factory=list)  # Links to knowledge graph entities
    confidence_score: float = field(default=1.0)  # Confidence score for the entry content
    source: str = field(default="manual")  # Source of the entry (manual, auto-extracted, etc.)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "type": self.entry_type,
            "metadata": self.metadata,
            "relations": self.relations,
            "entity_links": self.entity_links,
            "confidence_score": self.confidence_score,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEntry":
        """Create entry from dictionary"""
        return cls(
            id=data["id"],
            content=data["content"],
            entry_type=data["type"],
            metadata=data.get("metadata", {}),
            relations=data.get("relations", []),
            entity_links=data.get("entity_links", []),
            confidence_score=data.get("confidence_score", 1.0),
            source=data.get("source", "manual"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )
    
    def add_relation(self, relation_type: str, target_id: str):
        """Add a relation to another entry"""
        self.relations.append({"type": relation_type, "target_id": target_id})
        self.updated_at = datetime.now()
    
    def remove_relation(self, target_id: str):
        """Remove a relation to another entry"""
        self.relations = [rel for rel in self.relations if rel["target_id"] != target_id]
        self.updated_at = datetime.now()
    
    def update_content(self, new_content: str):
        """Update entry content"""
        self.content = new_content
        self.updated_at = datetime.now()
    
    def update_metadata(self, updates: Dict[str, Any]):
        """Update entry metadata"""
        self.metadata.update(updates)
        self.updated_at = datetime.now()
    
    def add_entity_link(self, entity_id: str):
        """Add a link to a knowledge graph entity"""
        if entity_id not in self.entity_links:
            self.entity_links.append(entity_id)
            self.updated_at = datetime.now()
    
    def remove_entity_link(self, entity_id: str):
        """Remove a link to a knowledge graph entity"""
        if entity_id in self.entity_links:
            self.entity_links.remove(entity_id)
            self.updated_at = datetime.now()
    
    def set_confidence_score(self, score: float):
        """Set confidence score for the entry"""
        self.confidence_score = max(0.0, min(1.0, score))  # Ensure score is between 0 and 1
        self.updated_at = datetime.now()


@dataclass
class KnowledgeBase:
    """A knowledge base containing multiple entries"""
    name: str
    kb_type: str  # global, document, task, user
    entries: Dict[str, KnowledgeEntry] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    knowledge_graph: Optional[KnowledgeGraph] = field(default=None, init=False)  # Linked knowledge graph
    reasoning_history: List[Dict[str, Any]] = field(default_factory=list, init=False)  # Reasoning history records
    entity_mapping: Dict[str, List[str]] = field(default_factory=dict, init=False)  # Entity to entry mappings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert knowledge base to dictionary"""
        return {
            "name": self.name,
            "type": self.kb_type,
            "entries": {entry_id: entry.to_dict() for entry_id, entry in self.entries.items()},
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeBase":
        """Create knowledge base from dictionary"""
        kb = cls(
            name=data["name"],
            kb_type=data["type"],
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )
        
        # Add entries
        for entry_id, entry_data in data.get("entries", {}).items():
            kb.entries[entry_id] = KnowledgeEntry.from_dict(entry_data)
        
        return kb
    
    def add_entry(self, entry: KnowledgeEntry) -> str:
        """Add an entry to the knowledge base"""
        self.entries[entry.id] = entry
        self.updated_at = datetime.now()
        return entry.id
    
    def add_entry_content(self, content: str, entry_type: str = "generic", 
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a new entry with content"""
        entry = KnowledgeEntry(
            content=content,
            entry_type=entry_type,
            metadata=metadata or {}
        )
        return self.add_entry(entry)
    
    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get an entry by ID"""
        return self.entries.get(entry_id)
    
    def update_entry(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update an entry"""
        entry = self.get_entry(entry_id)
        if not entry:
            return False
        
        if "content" in updates:
            entry.update_content(updates["content"])
        if "type" in updates:
            entry.entry_type = updates["type"]
        if "metadata" in updates:
            entry.update_metadata(updates["metadata"])
        if "relations" in updates:
            entry.relations = updates["relations"]
        
        entry.updated_at = datetime.now()
        self.updated_at = datetime.now()
        return True
    
    def delete_entry(self, entry_id: str) -> bool:
        """Delete an entry"""
        if entry_id in self.entries:
            del self.entries[entry_id]
            self.updated_at = datetime.now()
            return True
        return False
    
    def query(self, query_text: str, entry_types: Optional[List[str]] = None) -> List[KnowledgeEntry]:
        """Query the knowledge base for relevant entries"""
        results = []
        
        # Simple full-text search for now
        query_lower = query_text.lower()
        
        for entry in self.entries.values():
            # Filter by entry type if specified
            if entry_types and entry.entry_type not in entry_types:
                continue
            
            # Check if query is in content or metadata
            if query_lower in entry.content.lower():
                results.append(entry)
            else:
                # Check metadata for matches
                for key, value in entry.metadata.items():
                    if isinstance(value, str) and query_lower in value.lower():
                        results.append(entry)
                        break
                    elif isinstance(value, (list, dict)) and query_lower in str(value).lower():
                        results.append(entry)
                        break
        
        return results
    
    def get_entries_by_type(self, entry_type: str) -> List[KnowledgeEntry]:
        """Get all entries of a specific type"""
        return [entry for entry in self.entries.values() if entry.entry_type == entry_type]
    
    def get_entries_by_metadata(self, key: str, value: Any) -> List[KnowledgeEntry]:
        """Get all entries with a specific metadata key-value pair"""
        return [entry for entry in self.entries.values() 
                if entry.metadata.get(key) == value]
    
    def get_related_entries(self, entry_id: str) -> List[KnowledgeEntry]:
        """Get all entries related to a specific entry"""
        entry = self.get_entry(entry_id)
        if not entry:
            return []
        
        related_entries = []
        for relation in entry.relations:
            related_entry = self.get_entry(relation["target_id"])
            if related_entry:
                related_entries.append(related_entry)
        
        return related_entries
    
    def set_knowledge_graph(self, knowledge_graph: KnowledgeGraph):
        """Set the knowledge graph for this knowledge base"""
        self.knowledge_graph = knowledge_graph
        
    def link_entity_to_entry(self, entity_id: str, entry_id: str):
        """Link a knowledge graph entity to a knowledge entry"""
        entry = self.get_entry(entry_id)
        if entry:
            entry.add_entity_link(entity_id)
            # Update entity mapping
            if entity_id not in self.entity_mapping:
                self.entity_mapping[entity_id] = []
            if entry_id not in self.entity_mapping[entity_id]:
                self.entity_mapping[entity_id].append(entry_id)
        
    def get_entries_by_entity(self, entity_id: str) -> List[KnowledgeEntry]:
        """Get all entries linked to a specific knowledge graph entity"""
        entry_ids = self.entity_mapping.get(entity_id, [])
        return [self.entries[entry_id] for entry_id in entry_ids if entry_id in self.entries]
    
    def query_with_reasoning(self, query_text: str, entry_types: Optional[List[str]] = None) -> Tuple[List[KnowledgeEntry], Dict[str, Any]]:
        """Query the knowledge base with reasoning capabilities"""
        # Perform basic query first
        basic_results = self.query(query_text, entry_types)
        
        # If knowledge graph is available, enhance results with reasoning
        reasoning_info = {}
        if self.knowledge_graph:
            # 1. Extract entities from query
            # 2. Find related entities in knowledge graph
            # 3. Expand query with related entities
            # 4. Get additional entries based on related entities
            # 5. Score results based on relevance and relationships
            
            # For now, we'll implement a simple version that enhances results
            # with entries linked to related entities
            enhanced_results = basic_results.copy()
            processed_entry_ids = {entry.id for entry in basic_results}
            
            # Find all entities linked to basic results
            related_entities = set()
            for entry in basic_results:
                related_entities.update(entry.entity_links)
            
            # Find additional entries linked to these entities
            for entity_id in related_entities:
                entity_entries = self.get_entries_by_entity(entity_id)
                for entry in entity_entries:
                    if entry.id not in processed_entry_ids:
                        enhanced_results.append(entry)
                        processed_entry_ids.add(entry.id)
            
            # Update reasoning info
            reasoning_info = {
                "enhanced_with_knowledge_graph": True,
                "basic_result_count": len(basic_results),
                "enhanced_result_count": len(enhanced_results),
                "related_entities_count": len(related_entities)
            }
            
            # Add reasoning record
            self.add_reasoning_record(
                query=query_text,
                query_type="reasoning_query",
                results_count=len(enhanced_results),
                reasoning_info=reasoning_info
            )
            
            return enhanced_results, reasoning_info
        
        # If no knowledge graph, return basic results
        self.add_reasoning_record(
            query=query_text,
            query_type="basic_query",
            results_count=len(basic_results)
        )
        
        return basic_results, reasoning_info
    
    def multi_dimension_query(self, 
                             query_text: Optional[str] = None, 
                             entry_types: Optional[List[str]] = None,
                             entities: Optional[List[str]] = None,
                             metadata_filters: Optional[Dict[str, Any]] = None,
                             min_confidence: float = 0.0) -> List[KnowledgeEntry]:
        """Multi-dimension query support"""
        # Start with all entries
        results = list(self.entries.values())
        
        # Apply entry type filter
        if entry_types:
            results = [entry for entry in results if entry.entry_type in entry_types]
        
        # Apply confidence filter
        results = [entry for entry in results if entry.confidence_score >= min_confidence]
        
        # Apply entity filter
        if entities:
            filtered_results = []
            for entry in results:
                if any(entity in entry.entity_links for entity in entities):
                    filtered_results.append(entry)
            results = filtered_results
        
        # Apply metadata filters
        if metadata_filters:
            filtered_results = []
            for entry in results:
                match = True
                for key, value in metadata_filters.items():
                    if entry.metadata.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_results.append(entry)
            results = filtered_results
        
        # Apply text search if query_text is provided
        if query_text:
            query_lower = query_text.lower()
            filtered_results = []
            for entry in results:
                if query_lower in entry.content.lower():
                    filtered_results.append(entry)
                else:
                    # Check metadata for matches
                    for meta_value in entry.metadata.values():
                        if isinstance(meta_value, str) and query_lower in meta_value.lower():
                            filtered_results.append(entry)
                            break
                        elif isinstance(meta_value, (list, dict)) and query_lower in str(meta_value).lower():
                            filtered_results.append(entry)
                            break
            results = filtered_results
        
        return results
    
    def add_reasoning_record(self, query: str, query_type: str, results_count: int, reasoning_info: Optional[Dict[str, Any]] = None):
        """Add a reasoning record to the history"""
        record = {
            "query": query,
            "query_type": query_type,
            "results_count": results_count,
            "reasoning_info": reasoning_info or {},
            "timestamp": datetime.now().isoformat()
        }
        self.reasoning_history.append(record)
        
        # Keep only the last 100 records to avoid memory issues
        if len(self.reasoning_history) > 100:
            self.reasoning_history = self.reasoning_history[-100:]
    
    def sync_with_knowledge_graph(self):
        """Sync knowledge base entries with knowledge graph entities"""
        if not self.knowledge_graph:
            return
        
        # For each entry, extract entities and link to knowledge graph
        for entry in self.entries.values():
            if entry.entry_type in ["document", "answer", "generic"]:
                # Simple entity extraction (in production, this would use NER)
                # For now, we'll simulate entity extraction by looking for entity names in content
                for entity in self.knowledge_graph.entities.values():
                    if entity.name.lower() in entry.content.lower():
                        self.link_entity_to_entry(entity.id, entry.id)
    
    def save(self, path: Optional[str] = None):
        """Save knowledge base to file"""
        save_path = Path(path) if path else Path(f"{self.name}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, default=str)
    
    @classmethod
    def load(cls, path: str) -> "KnowledgeBase":
        """Load knowledge base from file"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class KnowledgeBaseManager:
    """Manager for multiple knowledge bases"""
    
    def __init__(self, storage_path: str = "./knowledge_bases"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.knowledge_bases: Dict[str, KnowledgeBase] = {}
        
        # Load existing knowledge bases
        self._load_all_knowledge_bases()
    
    def _load_all_knowledge_bases(self):
        """Load all knowledge bases from storage"""
        for kb_file in self.storage_path.glob("*.json"):
            try:
                kb = KnowledgeBase.load(str(kb_file))
                self.knowledge_bases[kb.name] = kb
            except Exception as e:
                print(f"Error loading knowledge base {kb_file}: {e}")
    
    def create_knowledge_base(self, name: str, kb_type: str, metadata: Optional[Dict[str, Any]] = None) -> KnowledgeBase:
        """Create a new knowledge base"""
        if name in self.knowledge_bases:
            raise ValueError(f"Knowledge base '{name}' already exists")
        
        kb = KnowledgeBase(
            name=name,
            kb_type=kb_type,
            metadata=metadata or {}
        )
        
        self.knowledge_bases[name] = kb
        kb.save(str(self.storage_path / f"{name}.json"))
        return kb
    
    def get_knowledge_base(self, name: str) -> Optional[KnowledgeBase]:
        """Get a knowledge base by name"""
        return self.knowledge_bases.get(name)
    
    def delete_knowledge_base(self, name: str) -> bool:
        """Delete a knowledge base"""
        if name not in self.knowledge_bases:
            return False
        
        # Remove from memory
        del self.knowledge_bases[name]
        
        # Remove from disk
        kb_file = self.storage_path / f"{name}.json"
        if kb_file.exists():
            kb_file.unlink()
        
        return True
    
    def list_knowledge_bases(self, kb_type: Optional[str] = None) -> List[KnowledgeBase]:
        """List all knowledge bases, optionally filtered by type"""
        if kb_type:
            return [kb for kb in self.knowledge_bases.values() if kb.kb_type == kb_type]
        return list(self.knowledge_bases.values())
    
    def add_entry_to_kb(self, kb_name: str, entry: KnowledgeEntry) -> str:
        """Add an entry to a knowledge base"""
        kb = self.get_knowledge_base(kb_name)
        if not kb:
            raise ValueError(f"Knowledge base '{kb_name}' not found")
        
        entry_id = kb.add_entry(entry)
        kb.save(str(self.storage_path / f"{kb_name}.json"))
        return entry_id
    
    def add_entry_content_to_kb(self, kb_name: str, content: str, entry_type: str = "generic", 
                               metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a new entry with content to a knowledge base"""
        entry = KnowledgeEntry(
            content=content,
            entry_type=entry_type,
            metadata=metadata or {}
        )
        return self.add_entry_to_kb(kb_name, entry)
    
    def update_entry_in_kb(self, kb_name: str, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update an entry in a knowledge base"""
        kb = self.get_knowledge_base(kb_name)
        if not kb:
            return False
        
        success = kb.update_entry(entry_id, updates)
        if success:
            kb.save(str(self.storage_path / f"{kb_name}.json"))
        return success
    
    def delete_entry_from_kb(self, kb_name: str, entry_id: str) -> bool:
        """Delete an entry from a knowledge base"""
        kb = self.get_knowledge_base(kb_name)
        if not kb:
            return False
        
        success = kb.delete_entry(entry_id)
        if success:
            kb.save(str(self.storage_path / f"{kb_name}.json"))
        return success
    
    def query_knowledge_base(self, kb_name: str, query_text: str, 
                            entry_types: Optional[List[str]] = None) -> List[KnowledgeEntry]:
        """Query a knowledge base"""
        kb = self.get_knowledge_base(kb_name)
        if not kb:
            return []
        
        return kb.query(query_text, entry_types)
    
    def query_all_knowledge_bases(self, query_text: str, kb_types: Optional[List[str]] = None, 
                                 entry_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Query all knowledge bases"""
        results = []
        
        for kb in self.knowledge_bases.values():
            # Filter by knowledge base type if specified
            if kb_types and kb.kb_type not in kb_types:
                continue
            
            # Query the knowledge base
            kb_results = kb.query(query_text, entry_types)
            
            # Add results with knowledge base info
            for entry in kb_results:
                results.append({
                    "kb_name": kb.name,
                    "kb_type": kb.kb_type,
                    "entry": entry
                })
        
        return results
    
    def create_document_kb(self, doc_id: str, metadata: Optional[Dict[str, Any]] = None) -> KnowledgeBase:
        """Create a document-specific knowledge base"""
        kb_name = f"doc_{doc_id}"
        return self.create_knowledge_base(kb_name, "document", metadata)
    
    def create_task_kb(self, task_id: str, metadata: Optional[Dict[str, Any]] = None) -> KnowledgeBase:
        """Create a task-specific knowledge base"""
        kb_name = f"task_{task_id}"
        return self.create_knowledge_base(kb_name, "task", metadata)
    
    def get_document_kb(self, doc_id: str) -> Optional[KnowledgeBase]:
        """Get a document-specific knowledge base"""
        return self.get_knowledge_base(f"doc_{doc_id}")
    
    def get_task_kb(self, task_id: str) -> Optional[KnowledgeBase]:
        """Get a task-specific knowledge base"""
        return self.get_knowledge_base(f"task_{task_id}")
    
    def set_knowledge_graph_for_kb(self, kb_name: str, knowledge_graph: KnowledgeGraph):
        """Set knowledge graph for a specific knowledge base"""
        kb = self.get_knowledge_base(kb_name)
        if kb:
            kb.set_knowledge_graph(knowledge_graph)
    
    def query_knowledge_base_with_reasoning(self, kb_name: str, query_text: str, 
                                           entry_types: Optional[List[str]] = None) -> Tuple[List[KnowledgeEntry], Dict[str, Any]]:
        """Query a knowledge base with reasoning capabilities"""
        kb = self.get_knowledge_base(kb_name)
        if not kb:
            return [], {}
        
        return kb.query_with_reasoning(query_text, entry_types)
    
    def multi_dimension_query_kb(self, kb_name: str, 
                                query_text: Optional[str] = None, 
                                entry_types: Optional[List[str]] = None,
                                entities: Optional[List[str]] = None,
                                metadata_filters: Optional[Dict[str, Any]] = None,
                                min_confidence: float = 0.0) -> List[KnowledgeEntry]:
        """Perform multi-dimension query on a specific knowledge base"""
        kb = self.get_knowledge_base(kb_name)
        if not kb:
            return []
        
        return kb.multi_dimension_query(
            query_text=query_text,
            entry_types=entry_types,
            entities=entities,
            metadata_filters=metadata_filters,
            min_confidence=min_confidence
        )
    
    def multi_dimension_query_all_kbs(self, 
                                     query_text: Optional[str] = None, 
                                     kb_types: Optional[List[str]] = None,
                                     entry_types: Optional[List[str]] = None,
                                     entities: Optional[List[str]] = None,
                                     metadata_filters: Optional[Dict[str, Any]] = None,
                                     min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """Perform multi-dimension query on all knowledge bases"""
        results = []
        
        for kb in self.knowledge_bases.values():
            # Filter by knowledge base type if specified
            if kb_types and kb.kb_type not in kb_types:
                continue
            
            # Query the knowledge base
            kb_results = kb.multi_dimension_query(
                query_text=query_text,
                entry_types=entry_types,
                entities=entities,
                metadata_filters=metadata_filters,
                min_confidence=min_confidence
            )
            
            # Add results with knowledge base info
            for entry in kb_results:
                results.append({
                    "kb_name": kb.name,
                    "kb_type": kb.kb_type,
                    "entry": entry
                })
        
        return results
    
    def sync_all_kbs_with_knowledge_graph(self, knowledge_graph: KnowledgeGraph):
        """Sync all knowledge bases with the given knowledge graph"""
        for kb in self.knowledge_bases.values():
            kb.set_knowledge_graph(knowledge_graph)
            kb.sync_with_knowledge_graph()
    
    def link_entity_to_entry(self, kb_name: str, entity_id: str, entry_id: str):
        """Link a knowledge graph entity to a knowledge entry in a specific knowledge base"""
        kb = self.get_knowledge_base(kb_name)
        if kb:
            kb.link_entity_to_entry(entity_id, entry_id)
            kb.save(str(self.storage_path / f"{kb_name}.json"))
    
    def add_reasoning_record(self, kb_name: str, query: str, query_type: str, 
                            results_count: int, reasoning_info: Optional[Dict[str, Any]] = None):
        """Add a reasoning record to a specific knowledge base"""
        kb = self.get_knowledge_base(kb_name)
        if kb:
            kb.add_reasoning_record(query, query_type, results_count, reasoning_info)
            kb.save(str(self.storage_path / f"{kb_name}.json"))
