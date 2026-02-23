# NeuroAgent 最终架构总结

## 项目概述

**NeuroAgent** 是一个面向个人用户的**类人脑知识助手**，核心创新是以**知识图谱 (KG)** 为记忆架构，通过**自动联想**实现智能记忆管理。

---

## 核心架构

```
用户交互层
├── 文档上传 (PDF/Markdown/Word) - 主要输入
├── 自然语言对话
└── 知识图谱可视化

感知层 (Perception)
└── DocumentPerceiver
    ├── 文档结构解析
    ├── 层级信息提取 (高阶→低阶)
    └── Episode 生成

记忆核心层 (Neuro Core)
├── HierarchicalKG (层级化 KG) ⭐
│   ├── Level 3: Domain (领域) - "人工智能"
│   ├── Level 2: Concept (概念) - "深度学习"
│   ├── Level 1: Instance (实例) - "ResNet论文"
│   ├── 向上抽象: Instance → Concept → Domain
│   └── 向下具体化: Domain → Concept → Instance
│
├── ContextComposer (上下文编排器) ⭐
│   ├── 指令理解: 自然语言 → 检索策略
│   ├── 策略执行: 6种策略 (总结/详细/对比/广度/深度/平衡)
│   ├── 层级控制: 精细控制每层权重
│   ├── 多样性保证: 避免信息冗余
│   └── 去重机制: 维护对话历史
│
├── AutoAssociator (自动联想)
│   ├── On-Insert: 写入时自动建立关联
│   ├── Spreading: 查询时扩散激活
│   └── Spontaneous: 被动触发回忆
│
└── MemoryEngine (存储引擎)
    ├── Episode 存储
    └── 向量索引

输出
├── 智能检索 (根据指令动态调整)
├── 学习建议 (跨文档关联)
└── 上下文感知回答
```

---

## 核心创新

### 1. KG 作为记忆架构

**不是** "Vector DB + 独立 KG" 的双系统
**而是** "KG 即记忆" 的统一架构

```
传统 RAG:
文档分片 → Vector DB → 向量相似匹配
    └── 问题: 只懂"相似", 不懂"关联"

NeuroAgent:
Episode/Entity/Relation → 统一 KG → 图遍历扩散激活
    └── 优势: 理解"关联", 支持联想
```

### 2. 自动联想机制

**三层联想触发**：

| 触发时机 | 功能 | 效果 |
|---------|------|------|
| On-Insert | 写入时自动关联 | 新记忆自动找到相关旧记忆 |
| Spreading | 查询时扩散激活 | 多路径联想，相关记忆浮现 |
| Spontaneous | 被动关键词触发 | 聊天中自动浮现相关记忆 |

### 3. 层级化 KG

**双向流动**：
```
向上抽象: ResNet论文 → CNN → 深度学习 → 人工智能
向下具体化: 人工智能 → 深度学习 → CNN → ResNet论文
```

**应用场景**：
- "总结一下" → 提取高层 Domain/Concept
- "详细说说" → 提取低层 Instance

### 4. 上下文编排器 (Context Composer)

**核心问题**: KG 可能有数万节点，如何为每次对话选择最优上下文？

**解决方案**: 自然语言指令控制

```python
# 通过指令自动调整策略
"总结一下深度学习" → ABSTRACT_ONLY (提取高层)
"详细说说 ResNet" → CONCRETE_ONLY (提取细节)
"对比 CNN 和 Transformer" → RELATION_FOCUSED (提取关系)
"相关的都给我" → BREADTH_FIRST (广度优先)
"深入讲讲原理" → DEPTH_FIRST (深度优先)
```

**精细化控制**：
- 层级权重: 领域 50% + 概念 30% + 实例 20%
- 数量限制: 最多 10 个节点
- 多样性: 每层最多选 N 个
- 去重: 避免对话中重复提供

---

## 文件结构

```
doc/
├── neuro_core/                    # 核心记忆系统
│   ├── __init__.py
│   ├── hierarchical_kg.py         # 层级化 KG ⭐
│   ├── context_composer.py        # 上下文编排器 ⭐
│   ├── knowledge_graph_memory.py  # KG 架构
│   ├── auto_association.py        # 自动联想
│   ├── spreading_activation.py    # 扩散激活
│   ├── engine.py                  # 记忆引擎
│   └── models.py                  # 数据模型
│
├── perception/                    # 感知层
│   ├── document/                  # 文档感知
│   │   ├── parser.py
│   │   └── perceiver.py           # 层级化文档解析
│   └── chat/
│
├── cognition/                     # 认知层
├── agent/                         # 智能体
├── api/                           # API 接口
│
├── examples/                      # 演示
│   ├── kg_memory_demo.py          # KG 记忆演示
│   ├── toc_knowledge_assistant.py # ToC 场景演示
│   └── context_composer_demo.py   # 上下文编排演示
│
├── neuro_agent_v2.py              # V2 入口 (集成编排器)
├── TOC_PRODUCT_ARCHITECTURE.md    # ToC 产品架构
├── CONTEXT_COMPOSER_DESIGN.md     # 编排器设计文档
└── PROJECT_SUMMARY.md             # 项目总结
```

---

## 使用示例

### 基础对话

```python
from neuro_agent_v2 import NeuroAgentV2

agent = NeuroAgentV2(
    llm_func=your_llm,
    embedding_func=your_embedding,
)

# 基础查询
result = await agent.respond("什么是 ResNet？")

# 指令控制
result = await agent.respond(
    query="深度学习",
    instruction="总结一下核心概念"
)
# 自动使用 ABSTRACT_ONLY 策略，提取高层概念

result = await agent.respond(
    query="ResNet",
    instruction="详细说说技术细节"
)
# 自动使用 CONCRETE_ONLY 策略，提取论文细节
```

### 文档管理

```python
# 添加文档
result = await agent.add_document(
    file_path="papers/resnet.pdf",
    doc_id="doc_001"
)
# 自动:
# - 解析文档结构
# - 提取层级: 论文 → 章节 → 概念
# - 建立 KG 关联

# 多轮对话（自动去重）
session_id = "session_001"

# 第一轮
r1 = await agent.respond("介绍一下 ResNet", session_id=session_id)
# 使用上下文 [ResNet论文, CNN概念, 深度学习领域]

# 第二轮
r2 = await agent.respond("和 Transformer 有什么区别？", session_id=session_id)
# 自动去重，不再重复 ResNet 论文
# 使用新上下文 [Transformer论文, Attention机制, 对比关系]
```

### 精细控制

```python
from neuro_core import CompositionConfig, ContextStrategy

# 自定义配置
config = CompositionConfig(
    strategy=ContextStrategy.BALANCED,
    max_nodes=15,
    hierarchy_weights={
        3: 0.2,  # 领域层 20%
        2: 0.5,  # 概念层 50%
        1: 0.3,  # 实例层 30%
    }
)

nodes = agent.composer.compose(
    query="深度学习",
    config=config
)
```

---

## 演示运行

```bash
# 1. KG 记忆架构演示
python examples/kg_memory_demo.py

# 2. ToC 知识助手演示
python examples/toc_knowledge_assistant.py

# 3. 上下文编排演示
python examples/context_composer_demo.py

# 4. V2 交互式聊天
python neuro_agent_v2.py
```

---

## 产品价值

### 与传统 RAG 对比

| 特性 | 传统 RAG | NeuroAgent |
|------|---------|------------|
| 记忆模型 | 向量相似 | KG 联想 |
| 检索方式 | 被动查询 | 主动联想 |
| 上下文控制 | 固定数量 | 指令动态调整 |
| 层级理解 | 无 | 高阶→低阶双向 |
| 多轮对话 | 无状态 | 上下文演化 |

### 用户价值

1. **自然输入**: 只管上传文档，系统自动理解结构
2. **智能联想**: 相关记忆主动浮现，无需精确查询
3. **对话感知**: 根据意图自动调整回答深度
4. **持续学习**: 每次对话都在优化记忆关联

---

## 后续优化

### 短期 (1-2 周)
- [ ] 接入真实 LLM API
- [ ] 完善文档解析 (MinerU)
- [ ] 添加 Web UI (Streamlit)

### 中期 (1 个月)
- [ ] 多模态支持 (图片/音频)
- [ ] 时序记忆 (时间线视图)
- [ ] 个性化偏好学习

### 长期 (3 个月)
- [ ] 移动端 App
- [ ] 第三方集成 (Notion/微信)
- [ ] 协作功能 (团队知识共享)

---

## 总结

**NeuroAgent** 实现了从 "搜索知识" 到 "知识找你" 的范式转变：

1. **KG 是记忆架构**: Episode/Entity/Relation 统一图谱管理
2. **自动联想是核心**: 写入/查询/被动三层触发
3. **层级化是特色**: 高阶抽象 ↔ 低阶具体双向流动
4. **上下文编排是关键**: 指令控制，精准投喂 LLM

**项目状态**: 核心架构完成，可产品化。
