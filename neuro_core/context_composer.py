"""
上下文编排器 (Context Composer)

核心功能：
1. 指令理解：将自然语言指令解析为检索策略
2. 层级选择：根据策略选择特定层级的节点
3. 相关性过滤：动态平衡相关性与多样性
4. 上下文压缩：在 token 限制内优化信息密度

使用场景：
- "总结一下" → 提取高层抽象（父节点）
- "详细说说" → 提取低层细节（子节点）
- "对比这两篇" → 提取相关实体的对比视图
- "给我相关的所有信息" → 全层级扩散
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

from .models import Episode, EdgeType
from .hierarchical_kg import HierarchicalKG, HierarchyLevel


class ContextStrategy(Enum):
    """上下文提取策略"""
    ABSTRACT_ONLY = "abstract_only"          # 仅高层抽象（总结模式）
    CONCRETE_ONLY = "concrete_only"          # 仅具体细节（细节模式）
    BALANCED = "balanced"                    # 平衡模式（各层代表）
    DEPTH_FIRST = "depth_first"              # 深度优先（沿一条路径深入）
    BREADTH_FIRST = "breadth_first"          # 广度优先（多路径浅层）
    ENTITY_FOCUSED = "entity_focused"        # 实体聚焦（围绕关键实体）
    RELATION_FOCUSED = "relation_focused"    # 关系聚焦（突出关联）
    ADAPTIVE = "adaptive"                    # 自适应（基于查询动态调整）


@dataclass
class CompositionConfig:
    """上下文编排配置"""
    strategy: ContextStrategy = ContextStrategy.ADAPTIVE
    max_tokens: int = 4000                   # 上下文 token 限制
    max_nodes: int = 20                      # 最大节点数
    min_relevance: float = 0.3               # 最小相关性阈值
    hierarchy_weights: Dict[int, float] = field(default_factory=lambda: {
        HierarchyLevel.ABSTRACT: 0.2,
        HierarchyLevel.CONCEPT: 0.5,
        HierarchyLevel.INSTANCE: 0.3,
    })
    include_paths: bool = True               # 是否包含联想路径
    deduplicate: bool = True                 # 是否去重


@dataclass
class ContextNode:
    """上下文节点"""
    node_id: str
    content: str                             # 节点内容（摘要或详情）
    level: int                               # 层级
    relevance: float                         # 相关性分数
    source: str                              # 来源（检索路径）
    path: List[str] = field(default_factory=list)  # 联想路径
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntentParser:
    """
    意图解析器：从自然语言指令提取检索策略
    
    支持的指令模式：
    - "总结一下" → ABSTRACT_ONLY
    - "详细说说" → CONCRETE_ONLY  
    - "对比分析" → RELATION_FOCUSED
    - "相关的都给我" → BREADTH_FIRST
    - "深入讲讲这个" → DEPTH_FIRST
    """
    
    # 指令关键词映射
    STRATEGY_PATTERNS = {
        ContextStrategy.ABSTRACT_ONLY: [
            "总结", "概要", "概述", "大纲", "要点",
            "summary", "overview", "outline", "key points"
        ],
        ContextStrategy.CONCRETE_ONLY: [
            "详细", "细节", "具体", "深入", "例子",
            "detail", "specific", "deep dive", "example"
        ],
        ContextStrategy.DEPTH_FIRST: [
            "深入", "深挖", "追踪", "溯源",
            "deep", "trace", "follow"
        ],
        ContextStrategy.BREADTH_FIRST: [
            "相关", "所有", "全面", "广泛",
            "related", "all", "comprehensive", "broad"
        ],
        ContextStrategy.ENTITY_FOCUSED: [
            "关于", "围绕", "聚焦", "主体",
            "about", "focus on", "center on"
        ],
        ContextStrategy.RELATION_FOCUSED: [
            "对比", "关系", "联系", "比较",
            "compare", "contrast", "relationship", "connection"
        ],
    }
    
    # 层级关键词
    LEVEL_PATTERNS = {
        HierarchyLevel.ABSTRACT: ["领域", "主题", "方向", "domain", "field"],
        HierarchyLevel.CONCEPT: ["概念", "技术", "方法", "concept", "technique"],
        HierarchyLevel.INSTANCE: ["例子", "案例", "具体", "example", "case"],
    }
    
    def parse_instruction(self, instruction: str) -> CompositionConfig:
        """
        解析用户指令，生成编排配置
        
        Args:
            instruction: 用户指令，如 "详细说说深度学习的具体应用"
            
        Returns:
            CompositionConfig: 编排配置
        """
        instruction_lower = instruction.lower()
        
        # 1. 识别策略
        strategy = self._detect_strategy(instruction_lower)
        
        # 2. 识别层级偏好
        level_weights = self._detect_level_preference(instruction_lower)
        
        # 3. 识别数量限制
        max_nodes = self._detect_quantity_limit(instruction_lower)
        
        return CompositionConfig(
            strategy=strategy,
            hierarchy_weights=level_weights,
            max_nodes=max_nodes,
        )
    
    def _detect_strategy(self, instruction: str) -> ContextStrategy:
        """检测策略类型"""
        scores = {strategy: 0 for strategy in ContextStrategy}
        
        for strategy, patterns in self.STRATEGY_PATTERNS.items():
            for pattern in patterns:
                if pattern in instruction:
                    scores[strategy] += 1
        
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return ContextStrategy.ADAPTIVE
    
    def _detect_level_preference(self, instruction: str) -> Dict[int, float]:
        """检测层级偏好"""
        weights = {
            HierarchyLevel.ABSTRACT: 0.2,
            HierarchyLevel.CONCEPT: 0.5,
            HierarchyLevel.INSTANCE: 0.3,
        }
        
        # 检测偏好
        for level, patterns in self.LEVEL_PATTERNS.items():
            for pattern in patterns:
                if pattern in instruction:
                    # 提升该层级权重
                    weights[level] = 0.6
                    # 降低其他层级
                    for other in weights:
                        if other != level:
                            weights[other] = 0.2
                    return weights
        
        return weights
    
    def _detect_quantity_limit(self, instruction: str) -> int:
        """检测数量限制"""
        import re
        
        # 匹配 "前N个"、"最多N个" 等
        patterns = [
            r"前(\d+)个",
            r"最多(\d+)个",
            r"top\s*(\d+)",
            r"limit\s*(\d+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, instruction)
            if match:
                return int(match.group(1))
        
        return 20  # 默认


class ContextComposer:
    """
    上下文编排器
    
    核心算法：
    1. 种子选择：基于查询选择初始种子节点
    2. 层级过滤：根据策略过滤特定层级
    3. 相关性排序：动态计算节点相关性
    4. 多样性保证：避免信息冗余
    5. 路径追溯：记录联想路径用于可解释性
    """
    
    def __init__(self, hierarchical_kg: HierarchicalKG):
        self.hkg = hierarchical_kg
        self.intent_parser = IntentParser()
    
    def compose(
        self,
        query: str,
        instruction: Optional[str] = None,
        config: Optional[CompositionConfig] = None,
        existing_context: Optional[List[str]] = None,
    ) -> List[ContextNode]:
        """
        编排上下文
        
        Args:
            query: 用户查询
            instruction: 显式指令（可选）
            config: 编排配置（可选）
            existing_context: 已有上下文（用于去重）
            
        Returns:
            List[ContextNode]: 编排好的上下文节点
        """
        # 1. 解析配置
        if config is None:
            if instruction:
                config = self.intent_parser.parse_instruction(instruction)
            else:
                config = CompositionConfig()
        
        # 2. 选择种子节点
        seed_nodes = self._select_seeds(query, config)
        
        # 3. 根据策略检索
        candidates = self._retrieve_by_strategy(seed_nodes, config)
        
        # 4. 层级过滤和加权
        filtered = self._apply_hierarchy_weights(candidates, config)
        
        # 5. 多样性保证（避免冗余）
        diverse = self._ensure_diversity(filtered, config)
        
        # 6. 去重（与已有上下文）
        if existing_context and config.deduplicate:
            diverse = [n for n in diverse if n.node_id not in existing_context]
        
        # 7. 截断到最大数量
        final = diverse[:config.max_nodes]
        
        return final
    
    def _select_seeds(
        self,
        query: str,
        config: CompositionConfig
    ) -> List[Tuple[str, float]]:
        """选择种子节点"""
        # 这里简化实现，实际应该基于向量检索
        # 返回 (node_id, relevance) 列表
        
        seeds = []
        
        # 1. 从 Instance 层匹配关键词
        for ep_id, ep in self.hkg._episode_index.items():
            relevance = self._compute_query_relevance(query, ep)
            if relevance >= config.min_relevance:
                seeds.append((ep_id, relevance))
        
        # 2. 从 Concept 层匹配
        for concept_id in self.hkg._level_nodes[HierarchyLevel.CONCEPT]:
            node = self.hkg.graph.get_node(concept_id)
            if node:
                name = node.get("name", "")
                if any(kw in name for kw in query.split()):
                    seeds.append((concept_id, 0.8))
        
        # 按相关性排序
        seeds.sort(key=lambda x: -x[1])
        return seeds[:5]  # 最多 5 个种子
    
    def _compute_query_relevance(self, query: str, episode: Episode) -> float:
        """计算查询与 Episode 的相关性"""
        score = 0.0
        query_lower = query.lower()
        
        # 1. 标题匹配
        if query_lower in episode.summary.lower():
            score += 0.5
        
        # 2. 概念匹配
        for concept in episode.concepts:
            if concept.lower() in query_lower:
                score += 0.3
        
        # 3. 实体匹配
        for entity in episode.entity_ids:
            if entity.lower() in query_lower:
                score += 0.2
        
        return min(1.0, score)
    
    def _retrieve_by_strategy(
        self,
        seed_nodes: List[Tuple[str, float]],
        config: CompositionConfig
    ) -> List[ContextNode]:
        """根据策略检索候选节点"""
        candidates = []
        
        if config.strategy == ContextStrategy.ABSTRACT_ONLY:
            # 只向上检索到抽象层
            candidates = self._retrieve_upward(seed_nodes, max_level=HierarchyLevel.ABSTRACT)
            
        elif config.strategy == ContextStrategy.CONCRETE_ONLY:
            # 只向下检索到实例层
            candidates = self._retrieve_downward(seed_nodes, max_depth=2)
            
        elif config.strategy == ContextStrategy.DEPTH_FIRST:
            # 深度优先：选择最相关的一条路径深入
            candidates = self._retrieve_depth_first(seed_nodes[0] if seed_nodes else None)
            
        elif config.strategy == ContextStrategy.BREADTH_FIRST:
            # 广度优先：多路径浅层
            candidates = self._retrieve_breadth_first(seed_nodes)
            
        elif config.strategy == ContextStrategy.ENTITY_FOCUSED:
            # 实体聚焦：围绕关键实体
            candidates = self._retrieve_entity_focused(seed_nodes)
            
        elif config.strategy == ContextStrategy.RELATION_FOCUSED:
            # 关系聚焦：突出关联
            candidates = self._retrieve_relation_focused(seed_nodes)
            
        else:  # ADAPTIVE 或 BALANCED
            # 平衡模式：上下兼顾
            candidates = self._retrieve_balanced(seed_nodes)
        
        return candidates
    
    def _retrieve_upward(
        self,
        seed_nodes: List[Tuple[str, float]],
        max_level: int = HierarchyLevel.ABSTRACT
    ) -> List[ContextNode]:
        """向上检索到指定层级"""
        candidates = []
        
        for seed_id, seed_score in seed_nodes:
            # 获取向上路径
            abstractions = self.hkg.upward_abstraction(seed_id, max_levels=3)
            
            for abs_item in abstractions:
                if abs_item["level"] <= max_level:
                    node = ContextNode(
                        node_id=abs_item["node_id"],
                        content=abs_item.get("name", ""),
                        level=abs_item["level"],
                        relevance=abs_item["weight"] * seed_score,
                        source="upward_abstraction",
                        path=abs_item.get("path", []),
                    )
                    candidates.append(node)
        
        return candidates
    
    def _retrieve_downward(
        self,
        seed_nodes: List[Tuple[str, float]],
        max_depth: int = 2
    ) -> List[ContextNode]:
        """向下检索"""
        candidates = []
        
        for seed_id, seed_score in seed_nodes:
            # 获取向下实例
            instances = self.hkg.downward_concretization(seed_id, max_depth=max_depth)
            
            for inst in instances:
                if "episode" in inst:
                    ep = inst["episode"]
                    node = ContextNode(
                        node_id=ep.episode_id,
                        content=ep.summary,
                        level=HierarchyLevel.INSTANCE,
                        relevance=inst["activation"] * seed_score,
                        source="downward_concretization",
                        metadata={
                            "key_points": ep.key_points,
                            "concepts": ep.concepts,
                        }
                    )
                    candidates.append(node)
                else:
                    node = ContextNode(
                        node_id=inst.get("node_id", ""),
                        content=inst.get("name", ""),
                        level=inst.get("level", HierarchyLevel.CONCEPT),
                        relevance=inst.get("activation", 0.5) * seed_score,
                        source="downward_concretization",
                    )
                    candidates.append(node)
        
        return candidates
    
    def _retrieve_depth_first(
        self,
        seed_node: Optional[Tuple[str, float]],
        max_depth: int = 4
    ) -> List[ContextNode]:
        """深度优先检索"""
        if not seed_node:
            return []
        
        candidates = []
        seed_id, seed_score = seed_node
        
        # 执行层级扩散
        results = self.hkg.hierarchical_spreading(
            seed_id,
            max_hops=max_depth,
            allow_cross_hierarchy=True
        )
        
        for node_id, score, path in results[:10]:
            # 获取节点内容
            content = self._get_node_content(node_id)
            level = self._get_node_level(node_id)
            
            node = ContextNode(
                node_id=node_id,
                content=content,
                level=level,
                relevance=score * seed_score,
                source="hierarchical_spreading",
                path=path,
            )
            candidates.append(node)
        
        return candidates
    
    def _retrieve_breadth_first(
        self,
        seed_nodes: List[Tuple[str, float]]
    ) -> List[ContextNode]:
        """广度优先检索"""
        candidates = []
        
        for seed_id, seed_score in seed_nodes[:3]:  # 限制种子数
            # 浅层扩散
            results = self.hkg.hierarchical_spreading(
                seed_id,
                max_hops=2,
            )
            
            for node_id, score, path in results[:5]:
                content = self._get_node_content(node_id)
                level = self._get_node_level(node_id)
                
                node = ContextNode(
                    node_id=node_id,
                    content=content,
                    level=level,
                    relevance=score * seed_score,
                    source="breadth_first",
                    path=path,
                )
                candidates.append(node)
        
        return candidates
    
    def _retrieve_entity_focused(
        self,
        seed_nodes: List[Tuple[str, float]]
    ) -> List[ContextNode]:
        """实体聚焦检索"""
        candidates = []
        
        # 找到种子周围的实体
        for seed_id, seed_score in seed_nodes:
            for neighbor, edge in self.hkg.graph.get_neighbors_with_edges(seed_id):
                if edge.edge_type == EdgeType.CONCEPT_LINK:
                    node = ContextNode(
                        node_id=neighbor,
                        content=self._get_node_content(neighbor),
                        level=self._get_node_level(neighbor),
                        relevance=edge.weight * seed_score,
                        source="entity_focused",
                        path=[seed_id, neighbor],
                    )
                    candidates.append(node)
        
        return candidates
    
    def _retrieve_relation_focused(
        self,
        seed_nodes: List[Tuple[str, float]]
    ) -> List[ContextNode]:
        """关系聚焦检索"""
        candidates = []
        
        # 找到种子之间的关系路径
        if len(seed_nodes) >= 2:
            for i, (seed1, score1) in enumerate(seed_nodes[:3]):
                for seed2, score2 in seed_nodes[i+1:4]:
                    # 找连接路径
                    path = self._find_connection_path(seed1, seed2)
                    if path:
                        for node_id in path:
                            node = ContextNode(
                                node_id=node_id,
                                content=self._get_node_content(node_id),
                                level=self._get_node_level(node_id),
                                relevance=(score1 + score2) / 2,
                                source="relation_focused",
                                path=path,
                            )
                            candidates.append(node)
        
        return candidates
    
    def _retrieve_balanced(
        self,
        seed_nodes: List[Tuple[str, float]]
    ) -> List[ContextNode]:
        """平衡检索（各层兼顾）"""
        candidates = []
        
        # 向上（抽象层）
        upward = self._retrieve_upward(seed_nodes, max_level=HierarchyLevel.ABSTRACT)
        candidates.extend(upward[:3])
        
        # 同级（概念层）
        for seed_id, seed_score in seed_nodes[:2]:
            level = self._get_node_level(seed_id)
            if level == HierarchyLevel.INSTANCE:
                # 获取父概念
                abstractions = self.hkg.upward_abstraction(seed_id, max_levels=1)
                for abs_item in abstractions[:2]:
                    node = ContextNode(
                        node_id=abs_item["node_id"],
                        content=abs_item.get("name", ""),
                        level=HierarchyLevel.CONCEPT,
                        relevance=abs_item["weight"] * seed_score,
                        source="balanced_concept",
                    )
                    candidates.append(node)
        
        # 向下（实例层）
        downward = self._retrieve_downward(seed_nodes, max_depth=1)
        candidates.extend(downward[:5])
        
        return candidates
    
    def _apply_hierarchy_weights(
        self,
        candidates: List[ContextNode],
        config: CompositionConfig
    ) -> List[ContextNode]:
        """应用层级权重"""
        for node in candidates:
            weight = config.hierarchy_weights.get(node.level, 0.5)
            node.relevance *= weight
        
        # 按加权后的相关性排序
        candidates.sort(key=lambda x: -x.relevance)
        return candidates
    
    def _ensure_diversity(
        self,
        candidates: List[ContextNode],
        config: CompositionConfig
    ) -> List[ContextNode]:
        """确保多样性（避免同一层级的信息冗余）"""
        selected = []
        level_counts = {HierarchyLevel.ABSTRACT: 0, 
                       HierarchyLevel.CONCEPT: 0, 
                       HierarchyLevel.INSTANCE: 0}
        
        max_per_level = config.max_nodes // 3
        
        for node in candidates:
            if level_counts[node.level] < max_per_level:
                selected.append(node)
                level_counts[node.level] += 1
            
            if len(selected) >= config.max_nodes:
                break
        
        return selected
    
    def _get_node_content(self, node_id: str) -> str:
        """获取节点内容"""
        # Instance 层
        if node_id in self.hkg._episode_index:
            ep = self.hkg._episode_index[node_id]
            return ep.summary
        
        # 其他层
        node = self.hkg.graph.get_node(node_id)
        if node:
            return node.get("name", node_id)
        
        return node_id
    
    def _get_node_level(self, node_id: str) -> int:
        """获取节点层级"""
        if node_id in self.hkg._level_nodes[HierarchyLevel.ABSTRACT]:
            return HierarchyLevel.ABSTRACT
        if node_id in self.hkg._level_nodes[HierarchyLevel.CONCEPT]:
            return HierarchyLevel.CONCEPT
        if node_id in self.hkg._level_nodes[HierarchyLevel.INSTANCE]:
            return HierarchyLevel.INSTANCE
        return HierarchyLevel.CONCEPT  # 默认
    
    def _find_connection_path(self, start: str, end: str) -> List[str]:
        """找到两个节点之间的连接路径"""
        from collections import deque
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            if current == end:
                return path
            
            for neighbor, _ in self.hkg.graph.get_neighbors_with_edges(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []
    
    def format_context(self, nodes: List[ContextNode], format_type: str = "text") -> str:
        """格式化上下文为字符串"""
        if format_type == "text":
            parts = []
            for i, node in enumerate(nodes, 1):
                level_name = {3: "领域", 2: "概念", 1: "实例"}.get(node.level, "未知")
                part = f"[{i}] [{level_name}] {node.content}"
                if node.path and len(node.path) > 1:
                    part += f" (路径: {' -> '.join(node.path[-3:])})"
                parts.append(part)
            return "\n".join(parts)
        
        elif format_type == "json":
            import json
            return json.dumps([{
                "id": n.node_id,
                "content": n.content,
                "level": n.level,
                "relevance": n.relevance,
                "source": n.source,
            } for n in nodes], ensure_ascii=False, indent=2)
        
        return ""
