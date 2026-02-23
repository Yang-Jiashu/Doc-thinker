# NeuroAgent 架构设计

## 核心创新

### 1. KG 作为记忆架构 (KG-Based Memory Architecture)

传统 RAG 的架构：
```
Vector DB (文档分片) + KG (实体关系)  ← 两个独立系统
```

本系统的架构：
```
统一 KG 架构:
├── Episode Node (情节记忆) ──▶ 包含实体/概念
├── Entity Node (知识实体) ──▶ 出现在多个 Episode
├── Relation Node (关系) ──▶ 连接实体
└── Concept Node (概念) ──▶ 主题聚合

所有节点都在同一张图中，边类型定义联想方式
```

**关键区别**：
- Episode 不是独立存储，而是 KG 的节点
- 检索不是向量匹配，而是图遍历（扩散激活）
- Entity/Relation 不是抽离的，而是记忆的一部分

### 2. 自动联想机制 (Auto-Association)

不是被动响应查询，而是主动建立关联：

**On-Insert Association (写入时)**：
```
新 Episode "CNN 图像识别" 写入
    ↓
自动检测共享 Entity "深度学习"
    ↓
自动建立边: ep-cnn-002 ──[EPISODE_SIMILARITY]──▶ ep-dl-001
    ↓
触发联想通知: "这与之前的'深度学习基础'相关"
```

**Spreading Activation (查询时)**：
```
用户查询 "神经网络"
    ↓
种子激活: "神经网络" Entity 节点
    ↓
扩散传播: 神经网络 ──▶ 深度学习 ──▶ CNN/RNN
              │
              └──▶ 反向传播 ──▶ ep-dl-001
    ↓
多路径联想结果浮现
```

**Spontaneous Recall (自发回忆)**：
```
用户提到 "深度学习"（非查询，只是对话中的词）
    ↓
触发器激活
    ↓
相关记忆自动浮现（无需用户主动检索）
    ↓
系统提示: "这让我想起之前讨论过 CNN..."
```

## 架构分层

```
┌─────────────────────────────────────────────────────────────┐
│                      应用层 (Applications)                     │
│  Chat Interface │ Memory Visualization │ Task Execution       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      认知层 (Cognition)                        │
│  Intent Understanding │ Reasoning │ Reflection                 │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    记忆核心层 (Memory Core) ⭐                  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           KG Memory Architecture                      │  │
│  │                                                      │  │
│  │   Episode ──contains──▶ Entity                       │  │
│  │       │                   │                          │  │
│  │       └──related_to◀──────┘                          │  │
│  │                                                      │  │
│  │   Concept ──same_theme──▶ Episode                    │  │
│  │                                                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                              │                               │
│  ┌──────────────────────────┼──────────────────────────┐    │
│  │                          │                          │    │
│  ▼                          ▼                          ▼    │
│ ┌─────────────┐    ┌────────────────┐    ┌──────────────┐  │
│ │Spreading    │    │Auto Association│    │Consolidation │  │
│ │Activation   │    │                │    │               │  │
│ │(扩散激活)   │    │(自动联想)      │    │(记忆巩固)     │  │
│ └─────────────┘    └────────────────┘    └──────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      感知层 (Perception)                       │
│  Document │ Chat │ API │ Tools                                │
└─────────────────────────────────────────────────────────────┘
```

## 数据流

### 写入流程 (Perceive → Memorize)
```
Input (Document/Chat/API)
    ↓
Perceiver 解析为 Episode
    ↓
KGMemoryArchitecture.add_episode_to_kg()
    ├── 添加 Episode 节点
    ├── 添加 Entity 节点
    ├── 添加 Relation 边
    └── AutoAssociator 建立相似关联
    ↓
扩散索引更新
```

### 检索流程 (Recall)
```
Query
    ↓
HybridRetriever
    ├── Vector Search (语义相似)
    ├── Spreading Activation (关联联想)
    └── Analogical Retrieval (结构类比)
    ↓
Ranked Episodes with association paths
```

### 巩固流程 (Consolidate)
```
Background Task
    ↓
Consolidation
    ├── 重放近期记忆
    ├── 发现跨 Episode 关系
    ├── 强化高频联想路径
    └── 更新主题聚类
    ↓
Memory Graph Update
```

## 边类型与联想语义

| 边类型 | 语义 | 衰减系数 | 使用场景 |
|--------|------|----------|----------|
| SEMANTIC_SIMILARITY | 语义相似 | 0.85 | 向量相似的 Episode |
| CONCEPT_LINK | 概念关联 | 0.75 | Episode ↔ Entity/Concept |
| EPISODE_SIMILARITY | 情节相似 | 0.85 | 共享实体的 Episode |
| SAME_THEME | 同主题 | 0.80 | 同一概念的 Episode |
| INFERRED_RELATION | 推断关系 | 0.75 | LLM 推断的跨 Episode 关系 |
| SAME_DOCUMENT | 同文档 | 0.60 | 同一文档内的分片 |
| ANALOGOUS_TO | 结构类比 | 0.82 | 结构相似的 Episode |
| CO_ACTIVATED | 共激活 | 0.70 | 巩固时共同激活 |

## 核心算法

### 1. 扩散激活 (Spreading Activation)
```python
def spreading_activation(graph, seeds, max_hops=3):
    activation = {seed: 1.0 for seed in seeds}
    
    for hop in range(max_hops):
        for node in current_activated:
            for neighbor, edge in graph.get_neighbors(node):
                # 激活传递: 强度 × 边权重 × 衰减
                transfer = activation[node] * edge.weight * decay[edge.type]^(hop+1)
                activation[neighbor] += transfer
    
    return sorted(activation.items(), key=lambda x: -x[1])
```

### 2. 自动联想 (Auto Association)
```python
def auto_associate(new_episode, existing_episodes):
    associations = []
    
    for ep in existing_episodes:
        # 共享实体
        shared_entities = set(new_episode.entities) & set(ep.entities)
        if shared_entities:
            associations.append({
                "type": "shared_entities",
                "target": ep.id,
                "shared": shared_entities,
                "weight": len(shared_entities) / max(len(entities))
            })
            
        # 语义相似（向量）
        similarity = cosine_sim(new_episode.embedding, ep.embedding)
        if similarity > threshold:
            associations.append({
                "type": "semantic_similarity",
                "target": ep.id,
                "similarity": similarity
            })
    
    return associations
```

## 与文档问答的关系

文档问答只是输入源之一：

```
输入源多样性:
├── Document (PDF/Word/Markdown) ──▶ Episode
├── Chat (对话历史) ──────────────▶ Episode
├── API (结构化数据) ─────────────▶ Episode
└── Tools (工具执行结果) ─────────▶ Episode

所有输入统一为 Episode 进入 KG 记忆架构
```

因此：
- **文档问答** = 文档感知器 + 记忆检索 + 生成
- **对话系统** = 对话感知器 + 记忆检索 + 生成
- **智能助手** = 工具感知器 + 记忆检索 + 规划

记忆系统是核心，应用层是可选的封装。

## 未来优化方向

1. **时序联想**：加入时间衰减，越久远的记忆激活阈值越高
2. **情感标记**：Episode 加入情感维度，影响联想强度
3. **注意力机制**：工作记忆的容量限制，只保留高激活节点
4. **梦境重放**：离线时的随机巩固，模拟睡眠记忆整理
5. **分层记忆**：短期/长期/永久记忆的迁移机制
