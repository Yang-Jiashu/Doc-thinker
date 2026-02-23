"""
API Configuration for RAGAnything

Contains centralized API configuration for all endpoints
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from lightrag.utils import get_env_value


@dataclass
class APIConfig:
    """Centralized API configuration for RAGAnything"""
    
    # Base API configuration
    # ---
    api_prefix: str = field(default=get_env_value("API_PREFIX", "/api/v1", str))
    """Prefix for all API endpoints"""
    
    enable_cors: bool = field(default=get_env_value("ENABLE_CORS", True, bool))
    """Enable CORS for API endpoints"""
    
    cors_origins: List[str] = field(default_factory=lambda: get_env_value(
        "CORS_ORIGINS", "*", str).split(","))
    """Allowed CORS origins"""
    
    # Knowledge Graph API Configuration
    # ---
    knowledge_graph_api_enabled: bool = field(default=get_env_value("KNOWLEDGE_GRAPH_API_ENABLED", True, bool))
    """Enable knowledge graph API endpoints"""
    
    kg_api_timeout: int = field(default=get_env_value("KG_API_TIMEOUT", 30, int))
    """Timeout for knowledge graph API requests (seconds)"""
    
    kg_api_max_results: int = field(default=get_env_value("KG_API_MAX_RESULTS", 100, int))
    """Maximum results for knowledge graph API responses"""
    
    # Entity API Configuration
    # ---
    entity_api_enabled: bool = field(default=get_env_value("ENTITY_API_ENABLED", True, bool))
    """Enable entity API endpoints"""
    
    entity_search_max_results: int = field(default=get_env_value("ENTITY_SEARCH_MAX_RESULTS", 50, int))
    """Maximum results for entity search"""
    
    entity_batch_size: int = field(default=get_env_value("ENTITY_BATCH_SIZE", 20, int))
    """Batch size for entity operations"""
    
    # Relationship API Configuration
    # ---
    relationship_api_enabled: bool = field(default=get_env_value("RELATIONSHIP_API_ENABLED", True, bool))
    """Enable relationship API endpoints"""
    
    relationship_validation_enabled: bool = field(default=get_env_value("RELATIONSHIP_VALIDATION_ENABLED", True, bool))
    """Enable relationship validation API"""
    
    # Query API Configuration
    # ---
    query_api_enabled: bool = field(default=get_env_value("QUERY_API_ENABLED", True, bool))
    """Enable query API endpoints"""
    
    query_timeout: int = field(default=get_env_value("QUERY_TIMEOUT", 60, int))
    """Timeout for query requests (seconds)"""
    
    enable_multi_document_query: bool = field(default=get_env_value("ENABLE_MULTI_DOCUMENT_QUERY", True, bool))
    """Enable multi-document enhanced query"""
    
    enable_knowledge_reasoning: bool = field(default=get_env_value("ENABLE_KNOWLEDGE_REASONING", True, bool))
    """Enable knowledge reasoning in queries"""
    
    # Visualization API Configuration
    # ---
    visualization_api_enabled: bool = field(default=get_env_value("VISUALIZATION_API_ENABLED", True, bool))
    """Enable visualization API endpoints"""
    
    viz_max_entities: int = field(default=get_env_value("VIZ_MAX_ENTITIES", 100, int))
    """Maximum entities in visualization response"""
    
    viz_max_relationships: int = field(default=get_env_value("VIZ_MAX_RELATIONSHIPS", 200, int))
    """Maximum relationships in visualization response"""
    
    enable_export_formats: List[str] = field(default_factory=lambda: get_env_value(
        "ENABLE_EXPORT_FORMATS", "json,dot", str).split(","))
    """Enabled export formats for visualization"""
    
    # Version Control API Configuration
    # ---
    version_control_api_enabled: bool = field(default=get_env_value("VERSION_CONTROL_API_ENABLED", True, bool))
    """Enable version control API endpoints"""
    
    max_snapshots: int = field(default=get_env_value("MAX_SNAPSHOTS", 100, int))
    """Maximum number of snapshots to keep"""
    
    snapshot_auto_cleanup: bool = field(default=get_env_value("SNAPSHOT_AUTO_CLEANUP", True, bool))
    """Enable automatic snapshot cleanup"""
    
    # Reasoning API Configuration
    # ---
    reasoning_api_enabled: bool = field(default=get_env_value("REASONING_API_ENABLED", True, bool))
    """Enable reasoning API endpoints"""
    
    enable_rule_based_reasoning: bool = field(default=get_env_value("ENABLE_RULE_BASED_REASONING", True, bool))
    """Enable rule-based reasoning"""
    
    enable_path_reasoning: bool = field(default=get_env_value("ENABLE_PATH_REASONING", True, bool))
    """Enable path-based reasoning"""
    
    # Authentication Configuration (optional)
    # ---
    enable_auth: bool = field(default=get_env_value("ENABLE_AUTH", False, bool))
    """Enable authentication for API endpoints"""
    
    api_key: Optional[str] = field(default=get_env_value("API_KEY", None, str))
    """API key for authentication (if enabled)"""
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Ensure cors_origins is properly formatted
        self.cors_origins = [origin.strip() for origin in self.cors_origins]
        
        # Ensure enable_export_formats is properly formatted
        self.enable_export_formats = [format.strip().lower() for format in self.enable_export_formats]
    
    @property
    def all_api_enabled(self) -> bool:
        """Check if all APIs are enabled"""
        return (
            self.knowledge_graph_api_enabled and
            self.entity_api_enabled and
            self.relationship_api_enabled and
            self.query_api_enabled and
            self.visualization_api_enabled and
            self.version_control_api_enabled and
            self.reasoning_api_enabled
        )


@dataclass
class APIRoutes:
    """API route configurations"""
    
    # Knowledge Graph Routes
    kg_base: str = "/knowledge-graph"
    kg_entities: str = "/entities"
    kg_relationships: str = "/relationships"
    kg_query: str = "/query"
    kg_stats: str = "/stats"
    
    # Entity Routes
    entity_base: str = "/entities"
    entity_get: str = "/{entity_id}"
    entity_search: str = "/search"
    entity_add: str = "/add"
    entity_update: str = "/{entity_id}/update"
    entity_delete: str = "/{entity_id}/delete"
    entity_similar: str = "/{entity_id}/similar"
    
    # Relationship Routes
    rel_base: str = "/relationships"
    rel_get: str = "/{rel_id}"
    rel_add: str = "/add"
    rel_update: str = "/{rel_id}/update"
    rel_delete: str = "/{rel_id}/delete"
    rel_validate: str = "/{rel_id}/validate"
    rel_invalidate: str = "/{rel_id}/invalidate"
    
    # Query Routes
    query_base: str = "/query"
    query_text: str = "/text"
    query_multimodal: str = "/multimodal"
    query_knowledge: str = "/knowledge"
    query_multidim: str = "/multidim"
    
    # Visualization Routes
    viz_base: str = "/visualization"
    viz_data: str = "/data"
    viz_export: str = "/export"
    
    # Version Control Routes
    vc_base: str = "/version-control"
    vc_snapshots: str = "/snapshots"
    vc_snapshot_create: str = "/snapshots/create"
    vc_snapshot_get: str = "/snapshots/{snapshot_id}"
    vc_snapshot_restore: str = "/snapshots/{snapshot_id}/restore"
    vc_snapshot_delete: str = "/snapshots/{snapshot_id}/delete"
    vc_snapshots_compare: str = "/snapshots/compare/{snapshot_id1}/{snapshot_id2}"
    
    # Reasoning Routes
    reasoning_base: str = "/reasoning"
    reasoning_rules: str = "/rules"
    reasoning_apply: str = "/apply"
    reasoning_infer: str = "/infer"
    reasoning_path: str = "/path/{source_id}/{target_id}"
    
    def get_full_route(self, prefix: str, route: str) -> str:
        """Get full route with prefix"""
        return f"{prefix}{route}"


# Create global API config instance
api_config = APIConfig()
api_routes = APIRoutes()

# API endpoint metadata
def get_api_endpoint_metadata() -> Dict[str, Dict[str, Any]]:
    """Get metadata for all API endpoints"""
    return {
        # Knowledge Graph endpoints
        f"{api_config.api_prefix}{api_routes.kg_base}{api_routes.kg_stats}": {
            "method": "GET",
            "description": "Get knowledge graph statistics",
            "enabled": api_config.knowledge_graph_api_enabled
        },
        f"{api_config.api_prefix}{api_routes.kg_base}{api_routes.kg_query}": {
            "method": "POST",
            "description": "Query knowledge graph with complex filters",
            "enabled": api_config.knowledge_graph_api_enabled
        },
        
        # Entity endpoints
        f"{api_config.api_prefix}{api_routes.entity_base}": {
            "method": "GET",
            "description": "Get all entities",
            "enabled": api_config.entity_api_enabled
        },
        f"{api_config.api_prefix}{api_routes.entity_base}{api_routes.entity_search}": {
            "method": "GET",
            "description": "Search entities",
            "enabled": api_config.entity_api_enabled
        },
        f"{api_config.api_prefix}{api_routes.entity_base}{api_routes.entity_add}": {
            "method": "POST",
            "description": "Add new entity",
            "enabled": api_config.entity_api_enabled
        },
        
        # Relationship endpoints
        f"{api_config.api_prefix}{api_routes.rel_base}": {
            "method": "GET",
            "description": "Get all relationships",
            "enabled": api_config.relationship_api_enabled
        },
        f"{api_config.api_prefix}{api_routes.rel_base}{api_routes.rel_add}": {
            "method": "POST",
            "description": "Add new relationship",
            "enabled": api_config.relationship_api_enabled
        },
        f"{api_config.api_prefix}{api_routes.rel_base}{api_routes.rel_validate}": {
            "method": "POST",
            "description": "Validate relationship",
            "enabled": api_config.relationship_api_enabled
        },
        
        # Query endpoints
        f"{api_config.api_prefix}{api_routes.query_base}{api_routes.query_text}": {
            "method": "POST",
            "description": "Text query",
            "enabled": api_config.query_api_enabled
        },
        f"{api_config.api_prefix}{api_routes.query_base}{api_routes.query_knowledge}": {
            "method": "POST",
            "description": "Knowledge-enhanced query",
            "enabled": api_config.query_api_enabled
        },
        
        # Visualization endpoints
        f"{api_config.api_prefix}{api_routes.viz_base}{api_routes.viz_data}": {
            "method": "GET",
            "description": "Get visualization data",
            "enabled": api_config.visualization_api_enabled
        },
        f"{api_config.api_prefix}{api_routes.viz_base}{api_routes.viz_export}": {
            "method": "GET",
            "description": "Export visualization data",
            "enabled": api_config.visualization_api_enabled
        },
        
        # Version Control endpoints
        f"{api_config.api_prefix}{api_routes.vc_base}{api_routes.vc_snapshots}": {
            "method": "GET",
            "description": "Get all snapshots",
            "enabled": api_config.version_control_api_enabled
        },
        f"{api_config.api_prefix}{api_routes.vc_base}{api_routes.vc_snapshot_create}": {
            "method": "POST",
            "description": "Create new snapshot",
            "enabled": api_config.version_control_api_enabled
        },
        f"{api_config.api_prefix}{api_routes.vc_base}{api_routes.vc_snapshot_restore}": {
            "method": "POST",
            "description": "Restore snapshot",
            "enabled": api_config.version_control_api_enabled
        },
        
        # Reasoning endpoints
        f"{api_config.api_prefix}{api_routes.reasoning_base}{api_routes.reasoning_rules}": {
            "method": "GET",
            "description": "Get all reasoning rules",
            "enabled": api_config.reasoning_api_enabled
        },
        f"{api_config.api_prefix}{api_routes.reasoning_base}{api_routes.reasoning_apply}": {
            "method": "POST",
            "description": "Apply reasoning rules",
            "enabled": api_config.reasoning_api_enabled
        },
    }
