# neuro_memory/__init__.py
"""类人脑联想与记忆重连：Episode、扩散激活、巩固、类比检索。"""

from .models import Episode, EdgeType, MemoryEdge, EDGE_TYPE_DECAY, get_decay_for_edge_type
from .graph_store import MemoryGraphStore
from .episode_store import InMemoryEpisodeStore, EpisodeVectorStore
from .spreading_activation import spreading_activation, top_k_activated
from .consolidation import consolidate, build_structure_description, infer_cross_episode_relations
from .analogical_retrieval import retrieve_analogies, score_episode, structure_description_from_triples
from .engine import MemoryEngine

__all__ = [
    "Episode",
    "EdgeType",
    "MemoryEdge",
    "EDGE_TYPE_DECAY",
    "get_decay_for_edge_type",
    "MemoryGraphStore",
    "InMemoryEpisodeStore",
    "EpisodeVectorStore",
    "spreading_activation",
    "top_k_activated",
    "consolidate",
    "build_structure_description",
    "infer_cross_episode_relations",
    "retrieve_analogies",
    "score_episode",
    "structure_description_from_triples",
    "MemoryEngine",
]
