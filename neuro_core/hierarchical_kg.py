"""
层级化知识图谱 (Hierarchical KG)

实现: 高阶抽象 → 中阶概念 → 低阶具体实例

层级结构:
Level 3 (高阶抽象): Domain/Field (领域)
    └── 如: "人工智能", "医学", "法律"
    
Level 2 (中阶概念): Topic/Technology (主题/技术)
    └── 如: "深度学习", "CNN", "RNN"
    
Level 1 (低阶具体): Episode/Instance (具体实例)
    └── 如: "某篇论文", "一次对话", "一个笔记"

自动联想在这个层级间上下流动:
- 向上抽象: 从具体实例归纳出高阶概念
- 向下具体化: 从高阶概念联想相关实例
- 横向关联: 同层级间的相似关联
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import time

from .models import Episode, EdgeType, MemoryEdge
from .memory_graph import MemoryGraphStore


class HierarchyLevel:
    """层级枚举"""
    ABSTRACT = 3    # 高阶抽象: Domain/Field
    CONCEPT = 2     # 中阶概念: Topic/Technology
    INSTANCE = 1    # 低阶具体: Episode/Instance


class HierarchicalKG:
    """
    层级化知识图谱
    
    核心能力:
    1. 层级建模: 抽象-概念-实例三层结构
    2. 向上抽象: 从 Episode 提取/关联高阶概念
    3. 向下具体化: 从概念联想相关实例
    4. 层级间联想: 跨层级的激活传播
    """
    
    def __init__(self, graph: MemoryGraphStore):
        self.graph = graph
        self._episode_index: Dict[str, Episode] = {}
        
        # 层级索引
        self._level_nodes: Dict[int, Set[str]] = {
            HierarchyLevel.ABSTRACT: set(),
            HierarchyLevel.CONCEPT: set(),
            HierarchyLevel.INSTANCE: set(),
        }
        
        # 概念层级映射
        self._concept_hierarchy: Dict[str, str] = {}  # concept -> parent_concept
        self._domain_concepts: Dict[str, Set[str]] = defaultdict(set)  # domain -> concepts
    
    def add_episode_with_hierarchy(self, episode: Episode) -> Dict[str, Any]:
        """
        添加 Episode 并建立层级关联
        
        流程:
        1. Episode 作为 INSTANCE 层节点
        2. 从 Episode 提取/关联 CONCEPT 层节点
        3. 从 CONCEPT 关联/生成 ABSTRACT 层节点
        4. 建立层级间的边
        """
        results = {
            "episode_id": episode.episode_id,
            "concepts_linked": [],
            "domains_linked": [],
            "new_edges": [],
        }
        
        # 1. 添加 Episode 到 INSTANCE 层
        self._add_instance_node(episode)
        
        # 2. 向上关联: Episode → Concepts
        for concept in episode.concepts:
            concept_id = f"concept:{concept}"
            self._link_instance_to_concept(episode.episode_id, concept_id, concept)
            results["concepts_linked"].append(concept)
            
            # 3. 继续向上: Concept → Domain (抽象)
            domain_id = self._infer_domain_for_concept(concept)
            if domain_id:
                self._link_concept_to_domain(concept_id, domain_id)
                results["domains_linked"].append(domain_id)
        
        # 4. 从 Entities 提取更细粒度的概念
        for entity in episode.entity_ids:
            entity_concept_id = f"concept:{entity}"
            self._link_instance_to_concept(
                episode.episode_id, 
                entity_concept_id, 
                entity,
                edge_type=EdgeType.CONCEPT_LINK,
                weight=0.7
            )
        
        return results
    
    def _add_instance_node(self, episode: Episode) -> None:
        """添加实例层节点 (Episode)"""
        self.graph.add_node(
            episode.episode_id,
            node_type="instance",
            data={
                "level": HierarchyLevel.INSTANCE,
                "episode_id": episode.episode_id,
                "summary": episode.summary,
                "timestamp": episode.timestamp,
                "source_type": episode.source_type,
            }
        )
        self._episode_index[episode.episode_id] = episode
        self._level_nodes[HierarchyLevel.INSTANCE].add(episode.episode_id)
    
    def _link_instance_to_concept(
        self,
        instance_id: str,
        concept_id: str,
        concept_name: str,
        edge_type: EdgeType = EdgeType.CONCEPT_LINK,
        weight: float = 0.8
    ) -> None:
        """连接实例到概念 (向上关联)"""
        # 确保概念节点存在
        if not self.graph.has_node(concept_id):
            self.graph.add_node(
                concept_id,
                node_type="concept",
                data={
                    "level": HierarchyLevel.CONCEPT,
                    "name": concept_name,
                    "first_seen": time.time(),
                }
            )
            self._level_nodes[HierarchyLevel.CONCEPT].add(concept_id)
        
        # 建立向上边: Instance → Concept
        self.graph.add_edge(
            instance_id, concept_id,
            edge_type,
            weight=weight,
            metadata={"relation": "instance_of", "concept": concept_name}
        )
        
        # 建立向下边: Concept → Instance (用于快速检索)
        self.graph.add_edge(
            concept_id, instance_id,
            EdgeType.SAME_THEME,
            weight=weight * 0.9,
            metadata={"relation": "has_instance"}
        )
    
    def _infer_domain_for_concept(self, concept: str) -> Optional[str]:
        """
        推断概念所属的领域 (高阶抽象)
        
        简化实现: 基于概念关键词映射
        实际应该用 LLM 或预训练模型
        """
        domain_mappings = {
            # AI/CS 领域
            "深度学习": "人工智能",
            "CNN": "人工智能",
            "RNN": "人工智能",
            "神经网络": "人工智能",
            "机器学习": "人工智能",
            "自然语言处理": "人工智能",
            "计算机视觉": "人工智能",
            "算法": "计算机科学",
            "数据结构": "计算机科学",
            
            # 医学领域
            "诊断": "医学",
            "治疗": "医学",
            "症状": "医学",
            "药物": "医学",
            
            # 法律领域
            "合同": "法律",
            "诉讼": "法律",
            "法规": "法律",
        }
        
        return domain_mappings.get(concept)
    
    def _link_concept_to_domain(self, concept_id: str, domain_name: str) -> str:
        """连接概念到领域 (向上抽象)"""
        domain_id = f"domain:{domain_name}"
        
        # 确保领域节点存在
        if not self.graph.has_node(domain_id):
            self.graph.add_node(
                domain_id,
                node_type="domain",
                data={
                    "level": HierarchyLevel.ABSTRACT,
                    "name": domain_name,
                    "first_seen": time.time(),
                }
            )
            self._level_nodes[HierarchyLevel.ABSTRACT].add(domain_id)
        
        # 建立向上边: Concept → Domain
        self.graph.add_edge(
            concept_id, domain_id,
            EdgeType.CONCEPT_LINK,
            weight=0.9,
            metadata={"relation": "belongs_to", "domain": domain_name}
        )
        
        # 建立向下边: Domain → Concept
        self.graph.add_edge(
            domain_id, concept_id,
            EdgeType.SAME_THEME,
            weight=0.85,
            metadata={"relation": "contains_concept"}
        )
        
        # 记录层级关系
        self._concept_hierarchy[concept_id] = domain_id
        self._domain_concepts[domain_id].add(concept_id)
        
        return domain_id
    
    def upward_abstraction(self, instance_id: str, max_levels: int = 2) -> List[Dict]:
        """
        向上抽象: 从具体实例找到高阶概念
        
        用途:
        - "这篇论文属于哪个领域?"
        - "这个笔记涉及什么主题?"
        """
        if instance_id not in self._episode_index:
            return []
        
        results = []
        current_level = HierarchyLevel.INSTANCE
        current_nodes = [instance_id]
        
        for level in range(max_levels):
            next_level_nodes = []
            next_level = current_level + 1
            
            for node_id in current_nodes:
                # 找到向上连接的节点
                for edge in self.graph.get_out_edges(node_id):
                    if edge.metadata.get("relation") in ["instance_of", "belongs_to"]:
                        parent_id = edge.target_id
                        parent_node = self.graph.get_node(parent_id)
                        
                        if parent_node and parent_node.get("level") == next_level:
                            results.append({
                                "level": next_level,
                                "node_id": parent_id,
                                "name": parent_node.get("name", ""),
                                "path": [instance_id, parent_id],
                                "edge_type": edge.edge_type.value,
                                "weight": edge.weight,
                            })
                            next_level_nodes.append(parent_id)
            
            current_nodes = next_level_nodes
            current_level = next_level
            
            if not current_nodes:
                break
        
        return results
    
    def downward_concretization(
        self,
        concept_id: str,
        max_depth: int = 2,
        min_activation: float = 0.3
    ) -> List[Dict]:
        """
        向下具体化: 从高阶概念找到相关实例
        
        用途:
        - "关于人工智能，我有哪些笔记?"
        - "深度学习的具体例子有哪些?"
        """
        results = []
        visited = {concept_id}
        queue = [(concept_id, 0, 1.0)]  # (node_id, depth, activation)
        
        while queue:
            current_id, depth, activation = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            # 找到向下连接的节点
            for edge in self.graph.get_out_edges(current_id):
                if edge.metadata.get("relation") in ["has_instance", "contains_concept"]:
                    child_id = edge.target_id
                    
                    if child_id in visited:
                        continue
                    
                    visited.add(child_id)
                    child_activation = activation * edge.weight
                    
                    if child_activation < min_activation:
                        continue
                    
                    child_node = self.graph.get_node(child_id)
                    if child_node:
                        level = child_node.get("level", 0)
                        
                        # 如果是实例层，获取 Episode
                        if level == HierarchyLevel.INSTANCE and child_id in self._episode_index:
                            ep = self._episode_index[child_id]
                            results.append({
                                "level": level,
                                "node_id": child_id,
                                "episode": ep,
                                "activation": child_activation,
                                "depth": depth + 1,
                            })
                        else:
                            results.append({
                                "level": level,
                                "node_id": child_id,
                                "name": child_node.get("name", ""),
                                "activation": child_activation,
                                "depth": depth + 1,
                            })
                        
                        queue.append((child_id, depth + 1, child_activation))
        
        # 按激活强度排序
        results.sort(key=lambda x: -x["activation"])
        return results
    
    def hierarchical_spreading(
        self,
        seed_id: str,
        max_hops: int = 3,
        allow_cross_hierarchy: bool = True
    ) -> List[Tuple[str, float, List[str]]]:
        """
        层级扩散激活
        
        特点:
        - 可以在同层级横向扩散
        - 可以跨层级上下扩散
        - 模拟人脑的多层次联想
        """
        from .spreading_activation import spreading_activation
        
        # 基础扩散激活
        activated = spreading_activation(
            self.graph,
            [seed_id],
            max_hops=max_hops,
            min_activation=0.1
        )
        
        results = []
        for node_id, score in activated:
            if node_id != seed_id:
                # 追溯路径
                path = self._trace_hierarchical_path(seed_id, node_id)
                results.append((node_id, score, path))
        
        return sorted(results, key=lambda x: -x[1])
    
    def _trace_hierarchical_path(self, source: str, target: str) -> List[str]:
        """追溯层级路径"""
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
                    queue.append((neighbor, path + [f"{neighbor}({edge.edge_type.value[:3]})"]))
        
        return [source, "...", target]
    
    def get_concept_hierarchy(self, domain_id: Optional[str] = None) -> Dict[str, Any]:
        """获取概念层级结构 (用于可视化)"""
        if domain_id:
            # 获取特定领域的层级
            concepts = self._domain_concepts.get(domain_id, set())
            return {
                "domain": domain_id,
                "concepts": [
                    {
                        "id": c,
                        "name": self.graph.get_node(c, {}).get("name", ""),
                        "instances": self._get_concept_instances(c),
                    }
                    for c in concepts
                ]
            }
        else:
            # 获取所有层级
            return {
                "domains": [
                    {
                        "id": d,
                        "name": self.graph.get_node(d, {}).get("name", ""),
                        "concepts": list(concepts),
                    }
                    for d, concepts in self._domain_concepts.items()
                ]
            }
    
    def _get_concept_instances(self, concept_id: str) -> List[str]:
        """获取概念下的所有实例"""
        instances = []
        for edge in self.graph.get_out_edges(concept_id):
            if edge.metadata.get("relation") == "has_instance":
                instances.append(edge.target_id)
        return instances
