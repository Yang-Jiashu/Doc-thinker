#!/usr/bin/env python3
"""
Advanced Entity Extractor Module

This module provides advanced entity extraction capabilities using various NER models.
Supports both rule-based and model-based entity extraction.
"""

import re
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod

try:
    import spacy

    class SpacyNER:
        def __init__(self, model_name: str):
            self._nlp = spacy.load(model_name)

        def extract(self, text: str) -> List[Dict[str, Any]]:
            doc = self._nlp(text)
            return [
                {
                    "entity_name": ent.text,
                    "entity_type": ent.label_,
                    "start_pos": ent.start_char,
                    "end_pos": ent.end_char,
                    "confidence": 0.85,
                    "source": "spacy",
                    "text": text[ent.start_char:ent.end_char],
                }
                for ent in doc.ents
            ]

    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False

# Forward declaration for KnowledgeGraph
class KnowledgeGraph:
    pass


class EntityExtractor(ABC):
    """Abstract base class for entity extractors"""
    
    @abstractmethod
    def extract_entities(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Extract entities from text
        
        Args:
            text: Text to extract entities from
            **kwargs: Additional parameters
            
        Returns:
            List of extracted entities with metadata
        """
        pass
    
    @abstractmethod
    def supported_entity_types(self) -> List[str]:
        """Get supported entity types
        
        Returns:
            List of supported entity types
        """
        pass
    
    @abstractmethod
    def register_entity_type(self, entity_type: str, config: Dict[str, Any]):
        """Register a new entity type
        
        Args:
            entity_type: Name of the entity type to register
            config: Configuration for the entity type
        """
        pass


class RuleBasedEntityExtractor(EntityExtractor):
    """Rule-based entity extractor using regex patterns"""
    
    def __init__(self):
        # Predefined regex patterns for common entity types
        self.patterns = {
            "number": r"\b\d+(?:\.\d+)?\b",
            "percentage": r"\b\d+(?:\.\d+)?%\b",
            "date": r"\b(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\w+ \d{1,2}, \d{4})\b",
            "time": r"\b(?:\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?|\d{1,2}\s*[AP]M)\b",
            "currency": r"\b(?:\$|￥|€|£)\s*\d+(?:\.\d+)?\b",
        }
    
    def extract_entities(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Extract entities using regex patterns"""
        entities = []
        
        for entity_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = {
                    "entity_name": match.group().strip(),
                    "entity_type": entity_type,
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                    "confidence": 0.8,  # Fixed confidence for rule-based extraction
                    "source": "rule-based",
                    "text": text[match.start():match.end()]
                }
                entities.append(entity)
        
        return entities
    
    def supported_entity_types(self) -> List[str]:
        """Get supported entity types"""
        return list(self.patterns.keys())
    
    def register_entity_type(self, entity_type: str, config: Dict[str, Any]):
        """Register a new entity type with regex pattern
        
        Args:
            entity_type: Name of the entity type to register
            config: Configuration for the entity type, must include 'pattern' key
        """
        if "pattern" not in config:
            raise ValueError("Entity type configuration must include 'pattern' key")
        
        self.patterns[entity_type] = config["pattern"]


class ModelBasedEntityExtractor(EntityExtractor):
    """Model-based entity extractor using advanced NER models"""
    
    def __init__(self, model_name: str = "default"):
        """Initialize model-based entity extractor
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
        # Store additional entity types registered by users
        self.additional_entity_types = []
    
    def _load_model(self):
        """Load the entity extraction model"""
        # In a real implementation, this would load a pre-trained NER model
        # For now, we'll use a mock implementation
        self.model = MockNERModel()
    
    def extract_entities(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Extract entities using a pre-trained model"""
        if not self.model:
            self._load_model()
        
        return self.model.extract_entities(text, **kwargs)
    
    def supported_entity_types(self) -> List[str]:
        """Get supported entity types"""
        base_types = [
            "person", "organization", "location", "date", "time",
            "number", "percentage", "currency", "event", "product",
            "work_of_art", "law", "language", "disease", "symptom"
        ]
        return base_types + self.additional_entity_types
    
    def register_entity_type(self, entity_type: str, config: Dict[str, Any]):
        """Register a new entity type for the model-based extractor
        
        Args:
            entity_type: Name of the entity type to register
            config: Configuration for the entity type
        """
        # In a real implementation, this would require retraining or fine-tuning the model
        # For now, we'll just add it to the supported types list
        if entity_type not in self.additional_entity_types:
            self.additional_entity_types.append(entity_type)


class MockNERModel:
    """Mock NER model for demonstration purposes"""
    
    def extract_entities(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Mock entity extraction"""
        # Simple mock implementation that extracts common entities
        entities = []
        
        # Extract person names (simple heuristic)
        person_pattern = r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b"
        matches = re.finditer(person_pattern, text)
        for match in matches:
            entities.append({
                "entity_name": match.group().strip(),
                "entity_type": "person",
                "start_pos": match.start(),
                "end_pos": match.end(),
                "confidence": 0.9,
                "source": "mock-model",
                "text": text[match.start():match.end()]
            })
        
        # Extract organization names (simple heuristic)
        org_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+(?:Inc|Corp|Ltd|LLC|Group|Company)\b"
        matches = re.finditer(org_pattern, text)
        for match in matches:
            entities.append({
                "entity_name": match.group().strip(),
                "entity_type": "organization",
                "start_pos": match.start(),
                "end_pos": match.end(),
                "confidence": 0.85,
                "source": "mock-model",
                "text": text[match.start():match.end()]
            })
        
        # Extract dates
        date_pattern = r"\b(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\w+ \d{1,2}, \d{4})\b"
        matches = re.finditer(date_pattern, text)
        for match in matches:
            entities.append({
                "entity_name": match.group().strip(),
                "entity_type": "date",
                "start_pos": match.start(),
                "end_pos": match.end(),
                "confidence": 0.95,
                "source": "mock-model",
                "text": text[match.start():match.end()]
            })
        
        return entities


class HybridEntityExtractor(EntityExtractor):
    """Hybrid entity extractor combining rule-based and model-based approaches"""
    
    def __init__(self):
        self.rule_extractor = RuleBasedEntityExtractor()
        self.model_extractor = ModelBasedEntityExtractor()
    
    def extract_entities(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Extract entities using hybrid approach"""
        # Extract entities using both methods
        rule_entities = self.rule_extractor.extract_entities(text, **kwargs)
        model_entities = self.model_extractor.extract_entities(text, **kwargs)
        
        # Combine and deduplicate entities
        combined_entities = []
        seen_entities = set()
        
        for entity in rule_entities + model_entities:
            entity_key = (entity["entity_name"], entity["start_pos"], entity["end_pos"])
            if entity_key not in seen_entities:
                seen_entities.add(entity_key)
                combined_entities.append(entity)
        
        return combined_entities
    
    def supported_entity_types(self) -> List[str]:
        """Get supported entity types"""
        return list(set(
            self.rule_extractor.supported_entity_types() + 
            self.model_extractor.supported_entity_types()
        ))
    
    def register_entity_type(self, entity_type: str, config: Dict[str, Any]):
        """Register a new entity type with both extractors
        
        Args:
            entity_type: Name of the entity type to register
            config: Configuration for the entity type
        """
        # Register with both extractors
        self.rule_extractor.register_entity_type(entity_type, config)
        self.model_extractor.register_entity_type(entity_type, config)


class EntityLinker:
    """Entity linker for linking extracted entities to knowledge graph"""
    
    def __init__(self, knowledge_graph: Optional[KnowledgeGraph] = None):
        self.knowledge_graph = knowledge_graph
    
    def link_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Link extracted entities to knowledge graph
        
        Args:
            entities: List of extracted entities
            
        Returns:
            List of entities with knowledge graph links
        """
        if not self.knowledge_graph:
            # If no knowledge graph provided, return entities as-is
            return entities
        
        linked_entities = []
        
        for entity in entities:
            entity_name = entity["entity_name"]
            
            # Try to find existing entity in knowledge graph
            kg_entity = self.knowledge_graph.get_entity_by_name(entity_name)
            
            if kg_entity:
                # Entity found in knowledge graph
                linked_entity = {
                    **entity,
                    "knowledge_graph_id": kg_entity.id,
                    "is_existing_entity": True
                }
            else:
                # Entity not found in knowledge graph
                linked_entity = {
                    **entity,
                    "knowledge_graph_id": None,
                    "is_existing_entity": False
                }
            
            linked_entities.append(linked_entity)
        
        return linked_entities


class SpacyEntityExtractor(EntityExtractor):
    """Entity extractor using spaCy"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize Spacy-based entity extractor
        
        Args:
            model_name: Name of the Spacy model to use (e.g., en_core_web_sm, zh_core_web_sm)
        """
        self.model_name = model_name
        self.ner = None
        if _SPACY_AVAILABLE:
            try:
                self.ner = SpacyNER(model_name)
            except Exception as e:
                print(f"Error loading Spacy model {model_name}: {e}")
    
    def extract_entities(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Extract entities using Spacy"""
        if not self.ner:
            return []
        
        # Use Spacy's question_ner for single text extraction
        # Since question_ner returns lowercased strings, we might want to use a more detailed method
        # for general extraction if needed.
        doc = self.ner.spacy_model(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in ["ORDINAL", "CARDINAL"]:
                continue
            entities.append({
                "entity_name": ent.text.strip(),
                "entity_type": ent.label_,
                "start_pos": ent.start_char,
                "end_pos": ent.end_char,
                "confidence": 1.0,
                "source": "spacy",
                "text": ent.text
            })
        return entities
    
    def supported_entity_types(self) -> List[str]:
        """Get supported entity types from Spacy model"""
        if not self.ner:
            return []
        return self.ner.spacy_model.pipe_labels.get("ner", [])
    
    def register_entity_type(self, entity_type: str, config: Dict[str, Any]):
        """Not supported for Spacy pre-trained models without fine-tuning"""
        pass


class EntityExtractorManager:
    """Manager for entity extractors"""
    
    def __init__(self):
        self.extractors: Dict[str, EntityExtractor] = {
            "rule-based": RuleBasedEntityExtractor(),
            "model-based": ModelBasedEntityExtractor(),
            "hybrid": HybridEntityExtractor(),
            "spacy": SpacyEntityExtractor()
        }
        self.default_extractor = "hybrid"
    
    def get_extractor(self, extractor_type: str = "default") -> EntityExtractor:
        """Get an entity extractor by type
        
        Args:
            extractor_type: Type of extractor to get
            
        Returns:
            Entity extractor instance
        """
        if extractor_type == "default":
            extractor_type = self.default_extractor
        
        if extractor_type not in self.extractors:
            raise ValueError(f"Unknown extractor type: {extractor_type}")
        
        return self.extractors[extractor_type]
    
    def register_extractor(self, name: str, extractor: EntityExtractor):
        """Register a new entity extractor
        
        Args:
            name: Name of the extractor
            extractor: Entity extractor instance
        """
        self.extractors[name] = extractor
    
    def extract_entities(self, text: str, extractor_type: str = "default", **kwargs) -> List[Dict[str, Any]]:
        """Extract entities from text using specified extractor
        
        Args:
            text: Text to extract entities from
            extractor_type: Type of extractor to use
            **kwargs: Additional parameters
            
        Returns:
            List of extracted entities
        """
        extractor = self.get_extractor(extractor_type)
        return extractor.extract_entities(text, **kwargs)
    
    def supported_extractors(self) -> List[str]:
        """Get supported extractor types
        
        Returns:
            List of supported extractor types
        """
        return list(self.extractors.keys())
    
    def register_entity_type(self, entity_type: str, config: Dict[str, Any]):
        """Register a new entity type with all extractors
        
        Args:
            entity_type: Name of the entity type to register
            config: Configuration for the entity type
        """
        for extractor in self.extractors.values():
            extractor.register_entity_type(entity_type, config)
    
    def supported_entity_types(self) -> List[str]:
        """Get all supported entity types from all extractors
        
        Returns:
            List of supported entity types
        """
        all_types = set()
        for extractor in self.extractors.values():
            all_types.update(extractor.supported_entity_types())
        return list(all_types)


# Global entity extractor manager instance
global_entity_extractor_manager = EntityExtractorManager()


def get_entity_extractor(extractor_type: str = "default") -> EntityExtractor:
    """Get global entity extractor instance
    
    Args:
        extractor_type: Type of extractor to get
        
    Returns:
        Entity extractor instance
    """
    return global_entity_extractor_manager.get_extractor(extractor_type)


def extract_entities(text: str, extractor_type: str = "default", **kwargs) -> List[Dict[str, Any]]:
    """Extract entities from text using global extractor
    
    Args:
        text: Text to extract entities from
        extractor_type: Type of extractor to use
        **kwargs: Additional parameters
        
    Returns:
        List of extracted entities
    """
    return global_entity_extractor_manager.extract_entities(text, extractor_type, **kwargs)


def register_entity_type(entity_type: str, config: Dict[str, Any]):
    """Register a new entity type with all extractors
    
    Args:
        entity_type: Name of the entity type to register
        config: Configuration for the entity type
    """
    global_entity_extractor_manager.register_entity_type(entity_type, config)


def get_supported_entity_types() -> List[str]:
    """Get all supported entity types from all extractors
    
    Returns:
        List of supported entity types
    """
    return global_entity_extractor_manager.supported_entity_types()


def get_supported_extractors() -> List[str]:
    """Get supported extractor types
    
    Returns:
        List of supported extractor types
    """
    return global_entity_extractor_manager.supported_extractors()
