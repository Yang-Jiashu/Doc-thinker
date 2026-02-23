# NeuroAgent - 类人脑智能体

基于人脑记忆机制的智能体系统，文档问答只是其中一个输入源。

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    应用层 (Applications)                   │
│  Chat UI │ API Server │ 文档问答 │ 任务执行 │ 记忆可视化   │
└─────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────┐
│                    认知层 (Cognition)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │  意图理解    │  │  推理规划    │  │  反思与归纳      │ │
│  │(Intent)     │  │(Reasoning)  │  │(Reflection)     │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────┐
│                 记忆核心层 (Memory Core)                    │
│                                                         │
│   ┌──────────────────────────────────────────────┐     │
│   │        Episode（情节记忆）                     │     │
│   │   一次对话 / 一个文档 / 一个事件 = 一个 Episode  │     │
│   └──────────────────────────────────────────────┘     │
│                      │                                  │
│   ┌──────────────────┼──────────────────┐              │
│   ▼                  ▼                  ▼              │
│ ┌─────────┐    ┌──────────┐    ┌──────────────┐       │
│ │工作记忆  │    │联想记忆图 │    │向量语义记忆  │       │
│ │(Hot)    │    │(Graph)   │    │(Vector DB)   │       │
│ └─────────┘    └──────────┘    └──────────────┘       │
│                                                         │
│   ┌──────────────────────────────────────────────┐     │
│   │   记忆维护：扩散激活 → 巩固重放 → 类比检索       │     │
│   └──────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────┐
│                 感知层 (Perception)                        │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐  │
│  │文档解析 │  │对话理解  │  │API数据  │  │工具结果  │  │
│  │(MinerU) │  │(Chat)    │  │(JSON)   │  │(Tools)   │  │
│  └─────────┘  └──────────┘  └─────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 目录结构

```
doc/
├── neuro_core/              # 核心记忆系统
│   ├── engine.py            # MemoryEngine 记忆引擎
│   ├── models.py            # Episode 数据模型
│   ├── memory_graph.py      # 联想记忆图
│   ├── episode_store.py     # 情节存储
│   ├── spreading_activation.py  # 扩散激活
│   ├── consolidation.py     # 记忆巩固
│   └── analogical_retrieval.py  # 类比检索
│
├── perception/              # 感知层
│   ├── base.py              # 感知器基类
│   ├── document/            # 文档感知
│   │   ├── parser.py        # 文档解析
│   │   └── perceiver.py     # 文档感知器
│   └── chat/                # 对话感知
│       └── perceiver.py     # 对话感知器
│
├── cognition/               # 认知层
│   └── processor.py         # 认知处理器
│
├── retrieval/               # 检索层
│   └── hybrid_retriever.py  # 混合检索器
│
├── agent/                   # 智能体
│   ├── agent.py             # NeuroAgent 主类
│   └── session.py           # 会话管理
│
├── api/                     # API 接口
│   └── server.py            # FastAPI 服务
│
├── main.py                  # 主入口
├── examples/                # 示例代码
│   └── basic_usage.py       # 基础使用示例
│
└── config/                  # 配置文件
    └── settings.yaml
```

## 快速开始

### 1. 基础使用

```python
import asyncio
from neuro_core import MemoryEngine
from cognition import CognitiveProcessor
from agent import NeuroAgent
from perception.chat import ChatPerceiver

# 创建 Agent
agent = NeuroAgent(
    llm_func=your_llm_func,
    embedding_func=your_embedding_func,
    working_dir="./data",
)

# 注册感知器
chat_perceiver = ChatPerceiver(cognitive_processor=agent.cognition)
agent.register_perceiver("chat", chat_perceiver)

# 感知并存储对话
messages = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你？"}
]
episode = await agent.perceive(messages, source_type="chat")

# 检索相关记忆
memories = await agent.recall("人工智能", top_k=5)

# 生成回答
response = await agent.respond("什么是深度学习？")
print(response["answer"])
```

### 2. 启动交互式聊天

```bash
python main.py
```

### 3. 启动 API 服务

```bash
python main.py --server
```

## 核心概念

### Episode（情节记忆）

一切输入（文档、对话、API 数据）都被封装为 Episode：

```python
Episode(
    episode_id="ep-xxx",
    source_type="chat",  # document/chat/api/event
    summary="用户询问AI问题",
    key_points=["要点1", "要点2"],
    concepts=["AI", "深度学习"],
    entity_ids=["神经网络", "机器学习"],
    relation_triples=[("AI", "包含", "机器学习")],
    content_embedding=[...],  # 向量表示
)
```

### 记忆维护机制

1. **扩散激活 (Spreading Activation)**
   - 从种子节点出发，沿图传播激活
   - 不同边类型有不同衰减系数
   - 多路径激活可叠加

2. **记忆巩固 (Consolidation)**
   - 定期重放近期和高显著性记忆
   - 推断跨事件关系
   - 更新主题/图式

3. **类比检索 (Analogical Retrieval)**
   - 内容相似度
   - 结构相似度
   - 显著性（检索频率、新近度）

## 与旧架构对比

| 旧架构 | 新架构 | 说明 |
|--------|--------|------|
| `neuro_memory/` | `neuro_core/` | 核心记忆系统 |
| 核心库 cognitive/ | `cognition/` | 认知层独立 |
| 核心库 parser | `perception/document/` | 感知层 |
| `LinearRAG/` | `retrieval/` | 整合到检索层 |
| 分散的测试文件 | `tests/` | 统一测试目录 |

## 迁移的组件

- ✅ `neuro_memory/` → `neuro_core/`
- ✅ 核心库 cognitive/ → `cognition/`
- ✅ 核心库 parser → `perception/document/`
- ✅ `neuro_memory/` 全部功能 → 新架构

## 未迁移的组件

- `Autothink-RAG/` - 保留在项目根目录，作为文档理解 RAG 架构
- `LinearRAG/` - 保留在项目根目录，用于优化 Graph RAG
- 图存储目录 - 保留在项目根目录，本地图/向量实现修改版

## 下一步优化

1. **整合 Autothink-RAG** - 将文档理解能力深度整合到感知层
2. **整合 LinearRAG** - 将图优化算法整合到检索层
3. **添加工具调用** - 在 agent/tools/ 添加工具集
4. **完善 API** - 添加更多 RESTful 接口
5. **添加可视化** - 记忆图谱可视化界面
