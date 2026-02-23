# 上下文编排器设计文档

## 概述

上下文编排器 (Context Composer) 是 NeuroAgent 的核心组件，负责根据对话意图智能选择和组合知识图谱 (KG) 中的信息作为 LLM 的上下文。

**核心问题**：KG 可能包含数万节点，如何为每次对话选择最有价值的上下文？

**解决方案**：通过自然语言指令动态控制上下文提取策略。

---

## 核心功能

### 1. 指令理解 (Intent Parsing)

将用户的自然语言指令解析为机器可执行的检索策略。

**支持的指令模式**：

| 指令示例 | 解析策略 | 说明 |
|---------|---------|------|
| "总结一下" | ABSTRACT_ONLY | 提取高层抽象概念 |
| "详细说说" | CONCRETE_ONLY | 提取具体细节实例 |
| "对比分析" | RELATION_FOCUSED | 提取关联关系 |
| "相关的都给我" | BREADTH_FIRST | 多路径广度检索 |
| "深入讲讲" | DEPTH_FIRST | 单路径深度追踪 |

**层级关键词识别**：
- "领域"、"主题" → 偏重 ABSTRACT 层
- "概念"、"技术" → 偏重 CONCEPT 层
- "例子"、"具体" → 偏重 INSTANCE 层

### 2. 策略执行 (Strategy Execution)

六种核心检索策略：

#### ABSTRACT_ONLY (总结模式)
```
用户: "总结一下深度学习"

检索逻辑:
  - 向上抽象到 Domain 层
  - 提取核心概念 (Concept)
  - 忽略具体论文 (Instance)

返回: ["人工智能", "机器学习", "深度学习", "神经网络架构"]
```

#### CONCRETE_ONLY (细节模式)
```
用户: "详细说说 ResNet"

检索逻辑:
  - 找到 ResNet 相关论文
  - 提取技术细节和实现
  - 包含 key_points 和 entities

返回: ["ResNet论文", "残差连接实现", "实验结果细节"]
```

#### BALANCED (平衡模式)
```
用户: "介绍一下 Transformer"

检索逻辑:
  - 向上: Transformer 所属领域
  - 同级: 相关技术 (BERT, GPT)
  - 向下: 具体实现论文

返回: [NLP领域, Attention概念, Transformer论文, BERT应用]
```

#### DEPTH_FIRST (深度模式)
```
用户: "深入讲讲注意力机制的原理"

检索逻辑:
  - 选择最相关的种子
  - 沿单一路径深入 4-5 跳
  - 追踪原理 → 实现 → 应用 → 优化

返回: [Attention原理, Self-Attention, Multi-Head, BERT实现]
```

#### BREADTH_FIRST (广度模式)
```
用户: "给我相关的所有信息"

检索逻辑:
  - 多种子并行
  - 每种子浅层扩散 (2-3 跳)
  - 覆盖更多相关概念

返回: [CNN, RNN, Transformer, GAN, 各种变体...]
```

#### RELATION_FOCUSED (关系模式)
```
用户: "对比 CNN 和 Transformer"

检索逻辑:
  - 找到两个种子的连接路径
  - 提取中间节点 (共同祖先/差异点)
  - 突出关联和区别

返回: [共同:深度学习, 差异:架构设计, 应用:CV vs NLP]
```

### 3. 层级控制 (Hierarchy Control)

精细控制每层级的权重和数量：

```python
CompositionConfig(
    hierarchy_weights={
        HierarchyLevel.ABSTRACT: 0.5,  # 高权重
        HierarchyLevel.CONCEPT: 0.3,
        HierarchyLevel.INSTANCE: 0.2,  # 低权重
    },
    max_nodes=10,
)
```

**应用场景**：
- **概览模式**: 领域 50% + 概念 30% + 实例 20%
- **研究模式**: 领域 10% + 概念 20% + 实例 70%
- **概念模式**: 领域 0% + 概念 100% + 实例 0%

### 4. 多样性保证 (Diversity)

避免信息冗余，确保上下文覆盖不同视角：

```
限制每层最大数量:
  - 领域层: 最多 2 个
  - 概念层: 最多 3 个
  - 实例层: 最多 5 个

避免: 10 篇相似的 ResNet 论文
确保: 1 篇 ResNet + 1 篇 Transformer + 1 篇对比分析
```

### 5. 去重机制 (Deduplication)

维护对话历史，避免重复提供相同上下文：

```
第一轮: 提取 ["深度学习", "ResNet", "CNN"]
  ↓ 标记为已使用
第二轮: 查询 "神经网络"
  ↓ 排除已使用
  提取 ["Transformer", "Attention"] (不再提取 ResNet)
```

---

## 使用示例

### 基础使用

```python
from neuro_core import ContextComposer, HierarchicalKG

# 初始化
composer = ContextComposer(hkg)

# 基础检索（自适应策略）
nodes = composer.compose(query="深度学习")
```

### 指令控制

```python
# 通过自然语言指令控制
nodes = composer.compose(
    query="ResNet",
    instruction="详细说说技术细节"
)
# 自动使用 CONCRETE_ONLY 策略
```

### 精细配置

```python
from neuro_core import CompositionConfig, ContextStrategy

# 自定义配置
config = CompositionConfig(
    strategy=ContextStrategy.BALANCED,
    max_tokens=4000,
    max_nodes=15,
    hierarchy_weights={
        3: 0.2,  # 领域
        2: 0.5,  # 概念
        1: 0.3,  # 实例
    }
)

nodes = composer.compose(
    query="Transformer",
    config=config
)
```

### 多轮对话

```python
# 维护已使用上下文
existing_context = []

# 第一轮
nodes1 = composer.compose(
    query="深度学习",
    instruction="总结一下",
    existing_context=existing_context
)
existing_context.extend([n.node_id for n in nodes1])

# 第二轮（自动去重）
nodes2 = composer.compose(
    query="神经网络",
    instruction="详细说说",
    existing_context=existing_context
)
```

### 格式化输出

```python
# 文本格式
context_text = composer.format_context(nodes, format_type="text")
print(context_text)
# 输出:
# [1] [领域] 人工智能
# [2] [概念] 深度学习
# [3] [实例] ResNet论文 (路径: 深度学习 -> CNN -> ResNet)

# JSON 格式
context_json = composer.format_context(nodes, format_type="json")
```

---

## 算法流程

```
用户输入
  ├── 查询: "深度学习"
  └── 指令: "总结一下"
      ↓
意图解析器 (IntentParser)
  ├── 策略: ABSTRACT_ONLY
  ├── 层级权重: {领域:0.5, 概念:0.4, 实例:0.1}
  └── 数量限制: 10
      ↓
种子选择 (Seed Selection)
  ├── 向量检索匹配 "深度学习"
  └── 返回 Top-3 种子节点
      ↓
策略执行 (Strategy Execution)
  └── ABSTRACT_ONLY:
      ├── 向上抽象到 Domain 层
      └── 提取相关 Concept
      ↓
层级加权 (Hierarchy Weighting)
  ├── 领域层节点 × 0.5
  ├── 概念层节点 × 0.4
  └── 实例层节点 × 0.1
      ↓
多样性保证 (Diversity)
  ├── 领域: 选择 2 个
  ├── 概念: 选择 3 个
  └── 实例: 选择 5 个
      ↓
去重 (Deduplication)
  └── 排除已使用的节点
      ↓
格式化输出
  └── List[ContextNode]
```

---

## 架构集成

```python
class NeuroAgent:
    def __init__(self):
        self.memory = MemoryEngine()
        self.kg = HierarchicalKG(self.memory.graph)
        self.composer = ContextComposer(self.kg)  # 新增
    
    async def respond(self, query, instruction=None):
        # 1. 编排上下文
        context_nodes = self.composer.compose(
            query=query,
            instruction=instruction
        )
        
        # 2. 格式化上下文
        context_text = self.composer.format_context(context_nodes)
        
        # 3. 构建 Prompt
        prompt = f"""基于以下上下文回答用户问题。

上下文:
{context_text}

用户问题: {query}
"""
        
        # 4. 调用 LLM
        response = await self.llm_func(prompt)
        
        return response
```

---

## 性能优化

### 1. 缓存机制

```python
# 缓存种子检索结果
@lru_cache(maxsize=1000)
def select_seeds(query):
    ...

# 缓存层级扩散结果
@lru_cache(maxsize=500)
def hierarchical_spreading(seed_id, max_hops):
    ...
```

### 2. 增量更新

```python
# 只重新计算变化的节点
changed_nodes = get_changed_nodes_since(timestamp)
candidates = [n for n in candidates if n.id in changed_nodes]
```

### 3. 并行计算

```python
# 多种子并行检索
with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(retrieve_for_seed, seed)
        for seed in seeds
    ]
    results = [f.result() for f in futures]
```

---

## 应用场景

### 场景 1: 智能客服

```
用户: "我的订单怎么还没发货？"
指令: "查询订单状态"
策略: CONCRETE_ONLY
提取: [具体订单信息, 物流详情, 预计时间]
```

### 场景 2: 学习助手

```
用户: "给我讲讲深度学习的核心概念"
指令: "概念梳理"
策略: BALANCED (偏重概念层)
提取: [领域概述, 核心概念, 1-2个经典论文]
```

### 场景 3: 研究分析

```
用户: "对比 Transformer 和 CNN 在图像任务上的表现"
指令: "对比分析"
策略: RELATION_FOCUSED
提取: [共同领域, 架构差异, 应用场景, 性能对比]
```

---

## 扩展方向

### 1. 多模态上下文

```python
# 支持图片、表格、代码
ContextNode(
    content="",
    content_type="image",  # text/image/table/code
    data=image_bytes,
)
```

### 2. 时序感知

```python
# 考虑时间因素
CompositionConfig(
    temporal_decay=True,  # 越新的记忆权重越高
    time_window="last_30_days",
)
```

### 3. 个性化

```python
# 基于用户历史偏好
user_profile = {
    "preferred_depth": "detailed",  # 偏好详细内容
    "favorite_domains": ["AI", "CV"],
}
```

---

## 总结

上下文编排器实现了 **"按需提取，精准投喂"** 的目标：

1. **指令理解**：用户用自然语言控制上下文
2. **策略多样**：6 种策略应对不同场景
3. **层级精细**：控制每层权重和数量
4. **质量保障**：多样性 + 去重 + 相关性平衡

核心价值：**让 KG 根据对话需求灵活变形，为 LLM 提供最相关的上下文**。
