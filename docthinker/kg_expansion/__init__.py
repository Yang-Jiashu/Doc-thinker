"""KG expansion: density clustering + two-part LLM expansion with rich edges."""

from .expander import KGExpander
from .clustering import (
    build_cluster_summaries,
    cluster_nodes,
    ClusterSummary,
    save_cluster_summaries,
    load_cluster_summaries,
)
from .manager import ExpandedNodeManager, extract_entities_from_text

__all__ = [
    "KGExpander",
    "ExpandedNodeManager",
    "extract_entities_from_text",
    "build_cluster_summaries",
    "cluster_nodes",
    "ClusterSummary",
    "save_cluster_summaries",
    "load_cluster_summaries",
]
