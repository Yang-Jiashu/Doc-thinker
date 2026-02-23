"""
自动联想机制 (Auto-Association)

核心功能：在记忆写入时自动触发联想，无需等待查询

触发时机:
1. On-Insert Association: 新记忆写入时立即联想
2. Background Association: 后台定时联想（巩固阶段）
3. Query-Driven Association: 查询时动态联想（已有）
"""

from typing import Any, Dict, List, Optional
import asyncio

from .models import Episode, EdgeType
from .knowledge_graph_memory import KGMemoryArchitecture


class AutoAssociator:
    """自动联想器"""
    
    def __init__(self, kg_architecture: KGMemoryArchitecture):
        self.kg = kg_architecture
        self.association_hooks: List[callable] = []  # 联想触发钩子
    
    def register_hook(self, hook: callable):
        """注册联想触发后的回调函数"""
        self.association_hooks.append(hook)
    
    async def on_insert_association(self, episode: Episode) -> Dict[str, Any]:
        """
        写入时自动联想
        
        当新 Episode 被添加时，自动：
        1. 找到语义相似的已有 Episode
        2. 找到共享 Entity/Concept 的 Episode
        3. 推断潜在关系
        4. 触发钩子通知（如：提示用户"这让你想起了..."）
        """
        associations = {
            "semantic_similar": [],
            "shared_entities": [],
            "shared_concepts": [],
            "inferred_relations": [],
        }
        
        # 1. 语义相似联想（基于向量）
        # 这里简化处理，实际应该查询向量存储
        for existing_id, existing_ep in self.kg._episode_index.items():
            if existing_id == episode.episode_id:
                continue
            
            # 计算相似度（简化版，实际应基于 embedding）
            shared_entities = set(episode.entity_ids) & set(existing_ep.entity_ids)
            shared_concepts = set(episode.concepts) & set(existing_ep.concepts)
            
            if shared_entities:
                associations["shared_entities"].append({
                    "episode_id": existing_id,
                    "shared": list(shared_entities),
                    "similarity": len(shared_entities) / max(len(episode.entity_ids), len(existing_ep.entity_ids), 1)
                })
            
            if shared_concepts:
                associations["shared_concepts"].append({
                    "episode_id": existing_id,
                    "shared": list(shared_concepts),
                    "similarity": len(shared_concepts) / max(len(episode.concepts), len(existing_ep.concepts), 1)
                })
        
        # 2. 触发钩子
        for hook in self.association_hooks:
            try:
                await hook(episode, associations)
            except Exception:
                pass
        
        return associations
    
    async def background_association(self, episodes: List[Episode]) -> None:
        """
        后台自动联想（巩固阶段）
        
        在系统空闲时：
        1. 重放近期记忆
        2. 发现潜在的跨 Episode 关系
        3. 强化高频联想路径
        """
        # 这里应该调用 consolidation.py 的逻辑
        pass
    
    def get_spontaneous_recall(
        self,
        trigger_entity: str,
        activation_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        自发回忆：当某个实体被激活时，自动浮现相关记忆
        
        模拟：看到"苹果" → 自动想起"昨天吃的苹果派"
        """
        if trigger_entity not in self.kg._entity_episodes:
            return []
        
        related_episodes = []
        for ep_id in self.kg._entity_episodes[trigger_entity]:
            ep = self.kg._episode_index.get(ep_id)
            if ep:
                # 计算激活强度
                activation = self._calculate_entity_activation(ep, trigger_entity)
                if activation >= activation_threshold:
                    related_episodes.append({
                        "episode": ep,
                        "activation": activation,
                        "trigger": trigger_entity,
                    })
        
        return sorted(related_episodes, key=lambda x: -x["activation"])
    
    def _calculate_entity_activation(self, episode: Episode, entity: str) -> float:
        """计算实体在 Episode 中的激活强度"""
        # 简化版：基于实体出现频率和位置
        base_activation = 0.5
        
        # 如果是 key entity（在 entity_ids 中）
        if entity in episode.entity_ids:
            base_activation += 0.3
        
        # 如果出现在 summary 中
        if entity in episode.summary:
            base_activation += 0.2
        
        return min(1.0, base_activation)


class AssociativeMemoryTrigger:
    """
    联想记忆触发器
    
    实现被动回忆（非主动查询时的记忆浮现）
    """
    
    def __init__(self, auto_associator: AutoAssociator):
        self.associator = auto_associator
        self.active_triggers: Dict[str, float] = {}  # 当前激活的触发器
    
    def activate(self, trigger: str, intensity: float = 1.0):
        """激活一个触发器（如用户提到某个关键词）"""
        self.active_triggers[trigger] = intensity
        
        # 检查是否有自发回忆
        recalls = self.associator.get_spontaneous_recall(trigger, 0.3)
        return recalls
    
    def decay(self, decay_factor: float = 0.9):
        """触发器衰减（模拟工作记忆的遗忘）"""
        self.active_triggers = {
            k: v * decay_factor
            for k, v in self.active_triggers.items()
            if v * decay_factor > 0.1
        }
