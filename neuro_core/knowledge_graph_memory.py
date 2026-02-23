"""
知识图谱记忆架构 (KG-Based Memory Architecture)

核心创新：以 KG 为骨架，Episode 为节点，实现自动联想记忆

架构:
    Episode Node ──语义相似──▶ Episode Node
         │                           │
         ├──包含──▶ Entity Node ◀──包含──┤
         │              │              │
         └────关系────▶ Relation ◀───关系────┘
                        │
                   同主题/同文档关联
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

from .models import Episode, EdgeType, MemoryEdge
from .memory_graph import MemoryGraphStore


class KGMemoryArchitecture:
    """
    KG 作为记忆架构的核心实现
    
    不是 "KG + Vector DB" 的双系统，而是 "KG 即记忆":
    - Episode 是 KG 的节点（带向量化内容）
    - Entity/Relation 是 KG 的子图结构
    - 所有检索都基于图遍历（扩散激活）而非单纯向量匹配
    """
    
    def __init__(self, graph: Optional[MemoryGraphStore] = None):
        self.graph = graph or MemoryGraphStore()
        self._episode_index: Dict[str, Episode] = {}  # episode_id -> Episode
        self._entity_episodes: Dict[str, Set[str]] = defaultdict(set)  # entity -> episodes
    
    def add_episode_to_kg(self, episode: Episode) -> None:
        """
        将 Episode 作为节点加入 KG，并建立自动联想边
        
        自动建立的关系:
        1. 与相似 Episode 的语义边 (SEMANTIC_SIMILARITY)
        2. 与包含 Entity 的概念边 (CONCEPT_LINK)
        3. 与同一文档的上下文边 (SAME_DOCUMENT)
        """
        # 1. Episode 作为主节点
        self.graph.add_node(
            episode.episode_id,
            node_type="episode",
            data={
                "episode_id": episode.episode_id,
                "timestamp": episode.timestamp,
                "source_type": episode.source_type,
                "summary": episode.summary,
                "content_embedding": episode.content_embedding,
            }
        )
        self._episode_index[episode.episode_id] = episode
        
        # 2. 自动建立与 Entities 的关联
        for entity_id in episode.entity_ids:
            self._link_episode_to_entity(episode.episode_id, entity_id)
        
        # 3. 自动建立与 Relations 的关联
        for source, rel, target in episode.relation_triples:
            self._link_episode_to_relation(episode.episode_id, source, rel, target)
        
        # 4. 自动联想：找到语义相似的 Episode 并建边
        self._auto_associate_episodes(episode)
    
    def _link_episode_to_entity(self, episode_id: str, entity_id: str) -> None:
        """建立 Episode 与 Entity 的关联"""
        # Entity 作为独立节点
        self.graph.add_node(entity_id, node_type="entity", data={"name": entity_id})
        
        # 双向边：Episode <-> Entity
        self.graph.add_edge(
            episode_id, entity_id,
            EdgeType.CONCEPT_LINK,
            weight=0.8,
            metadata={"relation": "contains_entity"}
        )
        self.graph.add_edge(
            entity_id, episode_id,
            EdgeType.CONCEPT_LINK,
            weight=0.8,
            metadata={"relation": "appears_in"}
        )
        
        # 记录 Entity 出现的 Episodes（用于快速检索）
        self._entity_episodes[entity_id].add(episode_id)
    
    def _link_episode_to_relation(
        self, episode_id: str, source: str, relation: str, target: str
    ) -> None:
        """建立 Episode 与 Relation 的关联"""
        relation_id = f"rel:{source}:{relation}:{target}"
        
        # Relation 作为边节点
        self.graph.add_node(relation_id, node_type="relation", data={
            "source": source,
            "relation": relation,
            "target": target,
        })
        
        # Episode 连接到 Relation
        self.graph.add_edge(
            episode_id, relation_id,
            EdgeType.INFERRED_RELATION,
            weight=0.9,
            metadata={"relation_type": relation}
        )
        
        # Relation 连接到 Entity
        self.graph.add_edge(relation_id, source, EdgeType.INFERRED_RELATION, weight=1.0)
        self.graph.add_edge(relation_id, target, EdgeType.INFERRED_RELATION, weight=1.0)
    
    def _auto_associate_episodes(self, new_episode: Episode) -> None:
        """
        自动联想：将新 Episode 与相似 Episode 建立关联
        
        基于:
        - 共享 Entity（共现实体）
        - 共享 Concept（共现概念）
        - 时间邻近性
        """
        # 1. 基于共享 Entity 建立关联
        for entity_id in new_episode.entity_ids:
            related_eps = self._entity_episodes.get(entity_id, set())
            for related_ep_id in related_eps:
                if related_ep_id != new_episode.episode_id:
                    # 共享实体越多的 Episode 权重越高
                    self.graph.add_edge(
                        new_episode.episode_id, related_ep_id,
                        EdgeType.EPISODE_SIMILARITY,
                        weight=0.6,
                        metadata={"reason": f"shared_entity:{entity_id}"}
                    )
        
        # 2. 基于共享 Concept 建立关联
        for concept in new_episode.concepts:
            concept_node_id = f"concept:{concept}"
            self.graph.add_node(concept_node_id, node_type="concept", data={"name": concept})
            
            # Episode -> Concept
            self.graph.add_edge(
                new_episode.episode_id, concept_node_id,
                EdgeType.CONCEPT_LINK,
                weight=0.7
            )
            
            # 找到同一概念的其他 Episode
            for node_id, edges in self.graph._out_edges.items():
                if node_id.startswith("ep-") and node_id != new_episode.episode_id:
                    for edge in edges:
                        if edge.target_id == concept_node_id:
                            self.graph.add_edge(
                                new_episode.episode_id, node_id,
                                EdgeType.SAME_THEME,
                                weight=0.5,
                                metadata={"reason": f"shared_concept:{concept}"}
                            )
    
    def spreading_recall(
        self,
        seed_episode_ids: List[str],
        max_hops: int = 3,
        min_activation: float = 0.1
    ) -> List[Tuple[str, float, List[str]]]:
        """
        基于 KG 的扩散联想检索
        
        Returns:
            [(node_id, activation_score, path), ...]
            path: 联想路径（如 ["ep-1", "entity-A", "ep-2"]）
        """
        from .spreading_activation import spreading_activation
        
        # 执行扩散激活
        activated = spreading_activation(
            self.graph,
            seed_episode_ids,
            max_hops=max_hops,
            min_activation=min_activation
        )
        
        # 追溯联想路径
        results = []
        for node_id, score in activated:
            if node_id not in seed_episode_ids:
                path = self._trace_association_path(seed_episode_ids[0], node_id)
                results.append((node_id, score, path))
        
        return sorted(results, key=lambda x: -x[1])
    
    def _trace_association_path(self, source: str, target: str) -> List[str]:
        """追溯从 source 到 target 的联想路径（简化版 BFS）"""
        from collections import deque
        
        queue = deque([(source, [source])])
        visited = {source}
        
        while queue:
            current, path = queue.popleft()
            if current == target:
                return path
            
            for neighbor, edge in self.graph.get_neighbors_with_edges(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return [source, "...", target]
    
    def get_episode_subgraph(self, episode_id: str, depth: int = 2) -> Dict[str, Any]:
        """获取 Episode 的局部记忆子图（用于可视化）"""
        if episode_id not in self._episode_index:
            return {}
        
        subgraph_nodes = {episode_id}
        subgraph_edges = []
        
        # BFS 扩展
        current_layer = {episode_id}
        for _ in range(depth):
            next_layer = set()
            for node_id in current_layer:
                for neighbor, edge in self.graph.get_neighbors_with_edges(node_id):
                    if neighbor not in subgraph_nodes:
                        subgraph_nodes.add(neighbor)
                        next_layer.add(neighbor)
                    subgraph_edges.append({
                        "source": edge.source_id,
                        "target": edge.target_id,
                        "type": edge.edge_type.value,
                        "weight": edge.weight,
                    })
            current_layer = next_layer
        
        return {
            "center": episode_id,
            "nodes": list(subgraph_nodes),
            "edges": subgraph_edges,
            "episodes": {
                nid: self._episode_index[nid].summary
                for nid in subgraph_nodes
                if nid in self._episode_index
            }
        }
