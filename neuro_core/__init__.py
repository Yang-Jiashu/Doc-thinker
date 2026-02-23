"""
Neuro Core - 类人脑记忆核心

ToC 个人知识助手场景:
1. 文档感知器 - 用户知识输入的主要方式
2. 层级化 KG - 从高阶抽象到低阶具体
3. 自动联想 - 生成新边/节点，构建知识网络

核心组件:
- HierarchicalKG: 层级化知识图谱 (高阶→低阶)
- KGMemoryArchitecture: KG 架构基础
- AutoAssociator: 自动联想器
- MemoryEngine: 记忆引擎
"""

# 基础模型
from .models import Episode, EdgeType, MemoryEdge, get_decay_for_edge_type

# 存储
from .memory_graph import MemoryGraphStore
from .episode_store import InMemoryEpisodeStore, EpisodeVectorStore

# 核心引擎
from .engine import MemoryEngine

# KG 架构
from .knowledge_graph_memory import KGMemoryArchitecture
from .hierarchical_kg import HierarchicalKG, HierarchyLevel

# 自动联想
from .auto_association import AutoAssociator, AssociativeMemoryTrigger

# 上下文编排 (ToC 对话核心)
from .context_composer import (
    ContextComposer,
    ContextStrategy,
    CompositionConfig,
    ContextNode,
    IntentParser,
)

# 算法
from .spreading_activation import spreading_activation, top_k_activated
from .analogical_retrieval import retrieve_analogies, score_episode

__all__ = [
    # 基础模型
    "Episode",
    "EdgeType", 
    "MemoryEdge",
    "get_decay_for_edge_type",
    
    # 存储
    "MemoryGraphStore",
    "InMemoryEpisodeStore",
    "EpisodeVectorStore",
    
    # 核心引擎
    "MemoryEngine",
    
    # KG 架构 (ToC 核心)
    "KGMemoryArchitecture",
    "HierarchicalKG",
    "HierarchyLevel",
    
    # 自动联想
    "AutoAssociator",
    "AssociativeMemoryTrigger",
    
    # 上下文编排
    "ContextComposer",
    "ContextStrategy",
    "CompositionConfig",
    "ContextNode",
    "IntentParser",
    
    # 算法
    "spreading_activation",
    "top_k_activated",
    "retrieve_analogies",
    "score_episode",
]

__version__ = "1.0.0"
__description__ = "类人脑记忆系统：层级化 KG + 自动联想，ToC 个人知识助手核心"
