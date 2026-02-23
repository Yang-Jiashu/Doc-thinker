#!/usr/bin/env python3
"""
Advanced Relation Extractor Module

This module provides advanced relation extraction capabilities using various methods.
Supports both rule-based and model-based relation extraction.
"""

from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod

# Forward declaration for KnowledgeGraph
class KnowledgeGraph:
    pass


class RelationExtractor(ABC):
    """Abstract base class for relation extractors"""
    
    @abstractmethod
    def extract_relations(self, text: str, entities: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Extract relations from text
        
        Args:
            text: Text to extract relations from
            entities: List of pre-extracted entities
            **kwargs: Additional parameters
            
        Returns:
            List of extracted relations with metadata
        """
        pass
    
    @abstractmethod
    def supported_relation_types(self) -> List[str]:
        """Get supported relation types
        
        Returns:
            List of supported relation types
        """
        pass
    
    @abstractmethod
    def register_relation_type(self, relation_type: str, config: Dict[str, Any]):
        """Register a new relation type
        
        Args:
            relation_type: Name of the relation type to register
            config: Configuration for the relation type
        """
        pass


class RuleBasedRelationExtractor(RelationExtractor):
    """Rule-based relation extractor using regex patterns and entity proximity"""
    
    def __init__(self):
        # Predefined relation types and their extraction rules
        self.relation_types: Dict[str, Dict[str, Any]] = {
            "located_in": {
                "description": "Entity is located in another entity",
                "patterns": [r"(\w+) is located in (\w+)", r"(\w+) is in (\w+)", r"(\w+), (\w+)"]
            },
            "works_for": {
                "description": "Person works for organization",
                "patterns": [r"(\w+) works for (\w+)", r"(\w+) is employed by (\w+)", r"(\w+) is at (\w+)"]
            },
            "founded_by": {
                "description": "Organization is founded by person",
                "patterns": [r"(\w+) was founded by (\w+)", r"(\w+) founder is (\w+)"]
            },
            "parent_of": {
                "description": "Person is parent of another person",
                "patterns": [r"(\w+) is the parent of (\w+)", r"(\w+) is (\w+)'s parent"]
            },
            "part_of": {
                "description": "Entity is part of another entity",
                "patterns": [r"(\w+) is part of (\w+)", r"(\w+) belongs to (\w+)"]
            }
        }
    
    def extract_relations(self, text: str, entities: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Extract relations using regex patterns and entity proximity"""
        relations = []
        
        # Extract relations using patterns
        for relation_type, config in self.relation_types.items():
            for pattern in config.get("patterns", []):
                matches = self._find_relation_matches(text, pattern, entities, relation_type)
                relations.extend(matches)
        
        # Extract relations based on entity proximity
        proximity_relations = self._extract_relations_by_proximity(text, entities, **kwargs)
        relations.extend(proximity_relations)
        
        return relations
    
    def _find_relation_matches(self, text: str, pattern: str, entities: List[Dict[str, Any]], 
                              relation_type: str) -> List[Dict[str, Any]]:
        """Find relation matches using regex pattern"""
        import re
        matches = []
        pattern_matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in pattern_matches:
            # Get the matched groups
            groups = match.groups()
            if len(groups) < 2:
                continue
            
            source_entity_name = groups[0].strip()
            target_entity_name = groups[1].strip()
            
            # Find corresponding entities in the pre-extracted entities list
            source_entity = self._find_entity_by_name(source_entity_name, entities)
            target_entity = self._find_entity_by_name(target_entity_name, entities)
            
            if source_entity and target_entity:
                relation = {
                    "source_entity": source_entity,
                    "target_entity": target_entity,
                    "relation_type": relation_type,
                    "confidence": 0.8,  # Fixed confidence for rule-based extraction
                    "source": "rule-based",
                    "text": text[match.start():match.end()],
                    "start_pos": match.start(),
                    "end_pos": match.end()
                }
                matches.append(relation)
        
        return matches
    
    def _find_entity_by_name(self, entity_name: str, entities: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find entity by name in the pre-extracted entities list"""
        for entity in entities:
            if entity["entity_name"].lower() == entity_name.lower():
                return entity
        return None
    
    def _extract_relations_by_proximity(self, text: str, entities: List[Dict[str, Any]], 
                                      max_distance: int = 50, **kwargs) -> List[Dict[str, Any]]:
        """Extract relations based on entity proximity in text"""
        relations = []
        
        # Only extract relations between entities of compatible types
        for i, source_entity in enumerate(entities):
            for j, target_entity in enumerate(entities):
                if i >= j:  # Avoid duplicate relations
                    continue
                
                # Calculate distance between entities in text
                distance = abs(source_entity["start_pos"] - target_entity["start_pos"])
                if distance > max_distance:
                    continue
                
                # Extract context between entities
                start = min(source_entity["end_pos"], target_entity["end_pos"])
                end = max(source_entity["start_pos"], target_entity["start_pos"])
                context = text[start:end].strip()
                
                # Simple relation type detection based on context
                relation_type = self._detect_relation_type(context, source_entity["entity_type"], target_entity["entity_type"])
                
                if relation_type:
                    relation = {
                        "source_entity": source_entity,
                        "target_entity": target_entity,
                        "relation_type": relation_type,
                        "confidence": 0.6,  # Lower confidence for proximity-based extraction
                        "source": "proximity-based",
                        "text": context,
                        "start_pos": start,
                        "end_pos": end
                    }
                    relations.append(relation)
        
        return relations
    
    def _detect_relation_type(self, context: str, source_type: str, target_type: str) -> Optional[str]:
        """Detect relation type based on context and entity types"""
        context_lower = context.lower()
        
        # Simple heuristic for relation type detection
        if "located in" in context_lower or "is in" in context_lower:
            return "located_in"
        elif "works for" in context_lower or "employed by" in context_lower:
            return "works_for"
        elif "founded by" in context_lower or "founder" in context_lower:
            return "founded_by"
        elif "parent" in context_lower:
            return "parent_of"
        elif "part of" in context_lower or "belongs to" in context_lower:
            return "part_of"
        
        # Type-based relation detection
        if source_type == "organization" and target_type == "location":
            if "headquartered" in context_lower or "based" in context_lower:
                return "located_in"
        elif source_type == "person" and target_type == "organization":
            if "works" in context_lower or "employee" in context_lower:
                return "works_for"
        elif source_type == "person" and target_type == "person":
            if "family" in context_lower or "relative" in context_lower:
                return "related_to"
        
        return None
    
    def supported_relation_types(self) -> List[str]:
        """Get supported relation types"""
        return list(self.relation_types.keys())
    
    def register_relation_type(self, relation_type: str, config: Dict[str, Any]):
        """Register a new relation type
        
        Args:
            relation_type: Name of the relation type to register
            config: Configuration for the relation type, including:
                - description: Description of the relation type
                - patterns: List of regex patterns to match the relation
        """
        self.relation_types[relation_type] = {
            "description": config.get("description", ""),
            "patterns": config.get("patterns", [])
        }


class ModelBasedRelationExtractor(RelationExtractor):
    """Model-based relation extractor using advanced ML models"""
    
    def __init__(self, model_name: str = "default"):
        """Initialize model-based relation extractor
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        self.model = None
        self.relation_types = self._get_default_relation_types()
        self._load_model()
    
    def _get_default_relation_types(self) -> Dict[str, Dict[str, Any]]:
        """Get default relation types"""
        return {
            "located_in": {"description": "Entity is located in another entity"},
            "works_for": {"description": "Person works for organization"},
            "founded_by": {"description": "Organization is founded by person"},
            "parent_of": {"description": "Person is parent of another person"},
            "part_of": {"description": "Entity is part of another entity"},
            "related_to": {"description": "Entities are related to each other"},
            "owns": {"description": "Entity owns another entity"},
            "invested_in": {"description": "Entity invested in another entity"},
            "collaborates_with": {"description": "Entities collaborate with each other"},
            "develops": {"description": "Entity develops another entity"}
        }
    
    def _load_model(self):
        """Load the relation extraction model"""
        # In a real implementation, this would load a pre-trained relation extraction model
        # For now, we'll use a mock implementation
        self.model = MockRelationModel()
    
    def extract_relations(self, text: str, entities: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Extract relations using a pre-trained model"""
        if not self.model:
            self._load_model()
        
        return self.model.extract_relations(text, entities, **kwargs)
    
    def supported_relation_types(self) -> List[str]:
        """Get supported relation types"""
        return list(self.relation_types.keys())
    
    def register_relation_type(self, relation_type: str, config: Dict[str, Any]):
        """Register a new relation type
        
        Args:
            relation_type: Name of the relation type to register
            config: Configuration for the relation type
        """
        self.relation_types[relation_type] = {
            "description": config.get("description", "")
        }


class MockRelationModel:
    """Mock relation model for demonstration purposes"""
    
    def extract_relations(self, text: str, entities: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Mock relation extraction"""
        # Simple mock implementation that extracts relations based on entity pairs
        relations = []
        
        # Generate some mock relations
        for i, source_entity in enumerate(entities):
            for j, target_entity in enumerate(entities):
                if i >= j:
                    continue
                
                # Skip if entities are the same
                if source_entity["entity_name"] == target_entity["entity_name"]:
                    continue
                
                # Simple heuristic for relation type
                relation_type = self._get_mock_relation_type(source_entity, target_entity)
                if relation_type:
                    relation = {
                        "source_entity": source_entity,
                        "target_entity": target_entity,
                        "relation_type": relation_type,
                        "confidence": 0.7,
                        "source": "mock-model",
                        "text": text,
                        "start_pos": source_entity["start_pos"],
                        "end_pos": target_entity["end_pos"]
                    }
                    relations.append(relation)
        
        return relations
    
    def _get_mock_relation_type(self, source_entity: Dict[str, Any], target_entity: Dict[str, Any]) -> Optional[str]:
        """Get mock relation type based on entity types"""
        source_type = source_entity["entity_type"]
        target_type = target_entity["entity_type"]
        
        if source_type == "person" and target_type == "organization":
            return "works_for"
        elif source_type == "organization" and target_type == "location":
            return "located_in"
        elif source_type == "person" and target_type == "person":
            return "related_to"
        elif source_type == "organization" and target_type == "organization":
            return "collaborates_with"
        elif source_type == "product" and target_type == "organization":
            return "developed_by"
        
        return None


class HybridRelationExtractor(RelationExtractor):
    """Hybrid relation extractor combining rule-based and model-based approaches"""
    
    def __init__(self):
        self.rule_extractor = RuleBasedRelationExtractor()
        self.model_extractor = ModelBasedRelationExtractor()
        self.relation_types = {
            **self.rule_extractor.relation_types,
            **self.model_extractor.relation_types
        }
    
    def extract_relations(self, text: str, entities: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Extract relations using hybrid approach"""
        # Extract relations using both methods
        rule_relations = self.rule_extractor.extract_relations(text, entities, **kwargs)
        model_relations = self.model_extractor.extract_relations(text, entities, **kwargs)
        
        # Combine and deduplicate relations
        combined_relations = []
        seen_relations = set()
        
        for relation in rule_relations + model_relations:
            relation_key = (
                relation["source_entity"]["entity_name"].lower(),
                relation["target_entity"]["entity_name"].lower(),
                relation["relation_type"]
            )
            if relation_key not in seen_relations:
                seen_relations.add(relation_key)
                combined_relations.append(relation)
        
        return combined_relations
    
    def supported_relation_types(self) -> List[str]:
        """Get supported relation types"""
        return list(set(
            self.rule_extractor.supported_relation_types() + 
            self.model_extractor.supported_relation_types()
        ))
    
    def register_relation_type(self, relation_type: str, config: Dict[str, Any]):
        """Register a new relation type with both extractors"""
        self.rule_extractor.register_relation_type(relation_type, config)
        self.model_extractor.register_relation_type(relation_type, config)
        self.relation_types[relation_type] = {
            "description": config.get("description", "")
        }


class RelationExtractorManager:
    """Manager for relation extractors"""
    
    def __init__(self):
        self.extractors: Dict[str, RelationExtractor] = {
            "rule-based": RuleBasedRelationExtractor(),
            "model-based": ModelBasedRelationExtractor(),
            "hybrid": HybridRelationExtractor()
        }
        self.default_extractor = "hybrid"
        self.relation_types: Dict[str, Dict[str, Any]] = {}
        self._sync_relation_types()
    
    def _sync_relation_types(self):
        """Sync relation types from all extractors"""
        for extractor in self.extractors.values():
            for relation_type in extractor.supported_relation_types():
                if relation_type not in self.relation_types:
                    self.relation_types[relation_type] = {
                        "description": f"Relation type: {relation_type}"
                    }
    
    def get_extractor(self, extractor_type: str = "default") -> RelationExtractor:
        """Get a relation extractor by type
        
        Args:
            extractor_type: Type of extractor to get
            
        Returns:
            Relation extractor instance
        """
        if extractor_type == "default":
            extractor_type = self.default_extractor
        
        if extractor_type not in self.extractors:
            raise ValueError(f"Unknown extractor type: {extractor_type}")
        
        return self.extractors[extractor_type]
    
    def register_extractor(self, name: str, extractor: RelationExtractor):
        """Register a new relation extractor
        
        Args:
            name: Name of the extractor
            extractor: Relation extractor instance
        """
        self.extractors[name] = extractor
        self._sync_relation_types()
    
    def extract_relations(self, text: str, entities: List[Dict[str, Any]], 
                        extractor_type: str = "default", **kwargs) -> List[Dict[str, Any]]:
        """Extract relations from text using specified extractor
        
        Args:
            text: Text to extract relations from
            entities: List of pre-extracted entities
            extractor_type: Type of extractor to use
            **kwargs: Additional parameters
            
        Returns:
            List of extracted relations
        """
        extractor = self.get_extractor(extractor_type)
        return extractor.extract_relations(text, entities, **kwargs)
    
    def supported_extractors(self) -> List[str]:
        """Get supported extractor types
        
        Returns:
            List of supported extractor types
        """
        return list(self.extractors.keys())
    
    def supported_relation_types(self) -> List[str]:
        """Get supported relation types
        
        Returns:
            List of supported relation types
        """
        return list(self.relation_types.keys())
    
    def register_relation_type(self, relation_type: str, config: Dict[str, Any]):
        """Register a new relation type with all extractors
        
        Args:
            relation_type: Name of the relation type to register
            config: Configuration for the relation type
        """
        for extractor in self.extractors.values():
            extractor.register_relation_type(relation_type, config)
        
        self.relation_types[relation_type] = {
            "description": config.get("description", "")
        }
    
    def get_relation_type_config(self, relation_type: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a relation type
        
        Args:
            relation_type: Name of the relation type
            
        Returns:
            Configuration for the relation type, or None if not found
        """
        return self.relation_types.get(relation_type)


# Global relation extractor manager instance
global_relation_extractor_manager = RelationExtractorManager()


def get_relation_extractor(extractor_type: str = "default") -> RelationExtractor:
    """Get global relation extractor instance
    
    Args:
        extractor_type: Type of extractor to get
        
    Returns:
        Relation extractor instance
    """
    return global_relation_extractor_manager.get_extractor(extractor_type)


def extract_relations(text: str, entities: List[Dict[str, Any]], extractor_type: str = "default", **kwargs) -> List[Dict[str, Any]]:
    """Extract relations from text using global extractor
    
    Args:
        text: Text to extract relations from
        entities: List of pre-extracted entities
        extractor_type: Type of extractor to use
        **kwargs: Additional parameters
        
    Returns:
        List of extracted relations
    """
    return global_relation_extractor_manager.extract_relations(text, entities, extractor_type, **kwargs)


def register_relation_type(relation_type: str, config: Dict[str, Any]):
    """Register a new relation type
    
    Args:
        relation_type: Name of the relation type to register
        config: Configuration for the relation type
    """
    global_relation_extractor_manager.register_relation_type(relation_type, config)


def get_supported_relation_types() -> List[str]:
    """Get supported relation types
    
    Returns:
        List of supported relation types
    """
    return global_relation_extractor_manager.supported_relation_types()
