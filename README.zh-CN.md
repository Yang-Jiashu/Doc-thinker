<div align="center">

<img src="docs/assets/banner.png" alt="DocThinker Banner" width="820" />

# DocThinker

**Agentic Memory Framework · 自进化知识图谱 · 文档推理**

*语言记录了认知过程的结果，而认知过程包含感知，经验，推理的过程。*

[![Paper](https://img.shields.io/badge/arXiv-2603.05551-b31b1b.svg)](https://arxiv.org/pdf/2603.05551)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Demo](https://img.shields.io/badge/Demo-Live-orange)](http://localhost:5000)
[![LightRAG](https://img.shields.io/badge/LightRAG-Based-8B5CF6)](https://github.com/HKUDS/LightRAG)
[![OpenClaw](https://img.shields.io/badge/OpenClaw-Integration-E74C3C)](https://github.com/letta-ai/letta)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Flask](https://img.shields.io/badge/Flask-UI-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![NetworkX](https://img.shields.io/badge/NetworkX-KG-4C72B0)](https://networkx.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector-3B5998?logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)

[English](README.md) | [中文](README.zh-CN.md)

</div>

<br>

**DocThinker** 是一个面向文档理解的 agentic memory 框架。它把文档、对话轮次、检索轨迹和图谱扩展转化为可召回、可推理、可巩固的多层记忆系统。

这个名字有两层含义：**Doc** 既是 document，也是 doctor。DocThinker 的目标是让 agent 基于文档形成记忆，并具备博士级的研究深度和推理能力。

与传统“检索后回答”的 RAG 管线不同，DocThinker 将知识视为持续演化的记忆底座：session-scoped 知识图谱承担语义记忆，Claw 承担分层对话记忆，Neuro Memory 承担情节类比记忆，KG expansion 则维护可被使用和晋升的图谱假设。

---

## 📑 目录 (Index)

- [🚀 快速安装 (Quick Install)](#-快速安装)
- [🔥 快速开始 (Quick Start)](#-快速开始)
  - [1. Web UI & 服务端](#1-web-ui--服务端)
  - [2. Python API 极简调用](#2-python-api-极简调用)
  - [3. Memory Layer API](#3-memory-layer-api)
- [🧬 核心贡献 (Key Contributions)](#-核心贡献)
  - [1. Agentic Memory Core](#1--agentic-memory-core)
  - [2. 会话级知识图谱](#2--会话级知识图谱)
  - [3. 自进化 KG 扩展](#3--自进化-kg-扩展)
  - [4. 分层对话记忆 (Claw)](#4--分层对话记忆-claw)
  - [5. 情节类比记忆](#5--情节类比记忆)
  - [6. 多模态检索信号](#6--多模态检索信号)
- [💡 使用场景 (Use Cases)](#-使用场景)
- [⚡ 查询模式与文档处理](#-查询模式)
- [📡 API 参考](#-api-参考)
- [❓ 常见问题 (FaQ)](#-faq)

---

## 🚀 快速安装

推荐使用 Python 3.10 或更高版本。

```bash
# 1. 克隆代码仓库
git clone https://github.com/Yang-Jiashu/doc-thinker.git
cd doc-thinker

# 2. 创建虚拟环境
conda create -n docthinker python=3.11 -y
conda activate docthinker

# 3. 安装依赖
pip install -r requirements.txt
pip install -e .
```

---

## 🔥 快速开始

### 1. Web UI & 服务端

最直观的体验方式是使用 Web 控制台：

```bash
# 1. 配置文件（填入大模型 API Keys）
cp env.example .env

# 2. 启动后端 API（FastAPI）
python -m uvicorn docthinker.server.app:app --host 0.0.0.0 --port 8000

# 3. 启动前端 UI（Flask）
python run_ui.py
```
> 打开 `http://localhost:5000` — 上传 PDF，提出问题，探索不断生长的知识图谱。

### 2. Python API 极简调用

你也可以用极简的 Python API 快速集成 DocThinker：

```python
import asyncio
from docthinker import DocThinker, DocThinkerConfig

async def main():
    # 1. 初始化配置
    config = DocThinkerConfig(working_dir="./my_knowledge_base")
    
    # 2. 实例化 (需要预先配置 LLM 和 Embedding 模型)
    dt = DocThinker(config=config, ...) 
    
    # 3. 摄入文档 (解析 + 构建知识图谱)
    await dt.process_document_complete("your_document.pdf")
    
    # 4. 查询会话知识图谱
    response = await dt.aquery("这篇文档的核心思想是什么？", mode="mix")
    print(response)

asyncio.run(main())
```

### 3. Memory Layer API

记忆层可以脱离完整 Web App 被第三方项目嵌入使用。外部项目只要实现 backend protocols，就能接入 `AgentMemoryCore`。

```python
from docthinker.memory_core import AgentMemoryBackends, AgentMemoryCore

memory = AgentMemoryCore(
    backends=AgentMemoryBackends(
        conversation=my_conversation_memory,
        episodic=my_episode_store,
        expanded=my_candidate_graph,
        graph=my_semantic_graph,
    )
)

recall = await memory.recall(
    session_id="research-session",
    query="回答前 agent 应该记住什么？",
    enable_thinking=True,
)

answer = await my_agent.run(query, context=recall.retrieval_instruction)

await memory.after_response(
    session_id="research-session",
    question=query,
    answer=answer,
    matched_expanded=recall.expanded_matches,
)
```

---

## 🧬 核心贡献

DocThinker 将检索和记忆组织成一个面向 agent 的框架，而不是把记忆逻辑分散在 API handler 里。

<div align="center">
<img src="docs/assets/pipeline.png" alt="DocThinker Pipeline" width="820" />
<p><b>图 1.</b> DocThinker 端到端管线 — 从文档输入到知识图谱构建、分层记忆管理、混合检索推理，最终输出并反馈回图谱。</p>
</div>

### 1. 🧠 Agentic Memory Core
`docthinker.memory_core.AgentMemoryCore` 是当前稳定的 agent 记忆门面。它显式定义了 conversation memory、episodic memory、expanded KG hypotheses、graph promotion 和可选 chat-turn ingestion 的 backend protocols。回答生成前，`recall()` 会合并：

* Claw 的 working/core/archive 对话记忆。
* Neuro Memory 的相似情节与类比 episode。
* KG 扩展节点匹配结果和强制检索指令。

回答生成后，`after_response()` 会把本轮问答巩固回记忆层，写入 chat episode，可选回写到图谱，并推动有用的扩展节点晋升。

### 2. 🧩 会话级知识图谱
每个 session 拥有自己的 GraphCore 知识图谱和文档状态。上传文件会在对应 session 内解析、写入和查询，从而隔离用户上下文，同时允许每个会话的图谱持续生长。

### 3. 🔀 自进化 KG 扩展
扩展以两条互补路径执行：
* **A — 聚类驱动：** HDBSCAN 聚类实体 embedding → LLM 生成聚类摘要 → 基于摘要主题扩展新实体。
* **B — Top-N 多角度：** 取连接度最高的 50 个节点，从 6 个认知维度（层级、因果、类比、对立、时序、应用）扩展。

扩展节点不会直接成为正式知识。它们先作为候选节点进入生命周期管理，在 query-time 被匹配，并在回答中反复产生有效使用后晋升为正式图谱节点。

<div align="center">
<img src="docs/assets/multi_agent_evolution.png" alt="记忆与图谱反馈架构" width="820" />
<p><sub><b>图 2.</b> 记忆与图谱反馈闭环。</sub></p>
</div>

### 4. 🗃️ 分层对话记忆 (Claw)
Claw 提供三层对话记忆：热层 working memory、温层核心摘要、冷层语义归档，用于支撑长时间会话。

### 5. 🧠 情节类比记忆
Neuro Memory 将对话和文档经历存为 episode，并在后续问题中检索相似情节作为类比线索。这些结果通过 `episodic_matches` 暴露，并作为推理提示注入，而不是直接当作事实来源。

<div align="center">
<img src="docs/assets/sparql_cot.png" alt="结构化推理" width="680" />
</div>

### 6. 🖼️ 多模态检索信号
DocThinker 会记录文档解析出的图片资产，并在 deep UI 查询中激活相关视觉证据。

---

## 💡 使用场景

<table>
<tr>
<td width="50%" valign="top">

> *"上传小说，探索自动构建的知识图谱"*

<img src="docs/assets/usecase_kg.gif" width="100%"/>

</td>
<td width="50%" valign="top">

> *"深度模式对话 — 情节记忆 + KG 扩展匹配 + 分层记忆"*

<img src="docs/assets/usecase_chat.gif" width="100%"/>

</td>
</tr>
</table>

---

## ⚡ 查询模式

| 模式 | UI 映射 | 策略 | 深度 |
|------|---------|------|------|
| **快速** | `naive` | 轻量检索，关闭 rerank | 浅 |
| **标准** | `local` | 会话 KG 检索 + rerank | 中 |
| **深度** | `mix` | KG + 向量检索、Claw 记忆、情节类比、扩展节点匹配、图像激活、查询后巩固 | 完整 |

<details>
<summary><b>深度模式管线（7 步）</b></summary>

1. Claw 召回 working/core/archive 对话记忆。
2. Neuro Memory 检索相似 episode 作为类比线索。
3. ExpandedNodeManager 将候选扩展节点与查询匹配。
4. `AgentMemoryCore.recall()` 合并为统一 retrieval instruction。
5. GraphCore 执行 KG + 向量混合检索。
6. LLM 基于问题、检索结果和记忆上下文生成回答。
7. `AgentMemoryCore.after_response()` 写入 episode、更新 Claw，并推动扩展节点生命周期。

</details>

## 📄 PDF 处理

| 模式 | 引擎 | 适用场景 |
|------|------|---------|
| `auto`（默认） | VLM（短文档）/ MinerU（长文档） | 通用 |
| `vlm` | 云端 VLM（Qwen-VL） | 图片密集文档 |
| `mineru` | MinerU 布局引擎 | 含复杂表格的长文档 |

<details>
<summary><b>📡 API 参考</b></summary>

| 类别 | 端点 | 方法 | 说明 |
|------|------|------|------|
| 会话 | `/api/v1/sessions` | GET / POST | 列出 / 创建会话 |
| | `/api/v1/sessions/{id}/history` | GET | 聊天历史 |
| | `/api/v1/sessions/{id}/files` | GET | 已上传文件 |
| 上传 | `/api/v1/ingest` | POST | 上传 PDF / TXT |
| | `/api/v1/ingest/stream` | POST | 流式文本上传 |
| 查询 | `/api/v1/query/stream` | POST | SSE 流式查询 |
| | `/api/v1/query` | POST | 非流式查询 |
| | `/api/v1/query/text` | POST | 非流式查询别名 |
| KG | `/api/v1/knowledge-graph/data` | GET | 可视化节点/边 |
| | `/api/v1/knowledge-graph/expand` | POST | 触发 KG 扩展 |
| | `/api/v1/knowledge-graph/stats` | GET | KG 统计 |
| | `/api/v1/knowledge-graph/expanded-nodes` | GET | 扩展节点生命周期 |
| 记忆 | `/api/v1/memory/stats` | GET | 情节 + Claw 记忆统计 |
| 设置 | `/api/v1/settings` | GET / POST | 运行时配置 |

</details>

<details>
<summary><b>📂 项目结构</b></summary>

| 目录 | 说明 |
|------|------|
| `docthinker/` | 核心：PDF 解析、KG 构建、agentic memory 门面（`memory_core/`）、查询路由、KG 扩展（`kg_expansion/`）、自动思考（`auto_thinking/`）、HyperGraphRAG（`hypergraph/`）、服务端（`server/`）、UI（`ui/`）。 |
| `graphcore/` | 图 RAG 引擎：KG 存储（NetworkX / FAISS / Qdrant / PG）、实体抽取、混合检索、重排序。 |
| `neuro_memory/` | 情节记忆：扩散激活、情节存储、类比检索、记忆固化。 |
| `claw/` | 分层记忆：热层（工作记忆）、温层（核心 / MEMORY.md）、冷层（语义档案）。 |
| `config/` | `settings.yaml` — PDF、记忆、检索、认知参数。 |

</details>

---

## 📝 引用

如果 DocThinker 对您的研究有帮助，请引用：

```bibtex
@article{yang2026autothinkrag,
  title={AutothinkRAG: Complexity-Aware Control of Retrieval-Augmented Reasoning for Image-Text Interaction},
  author={Yang, Jiashu and Zhang, Chi and Wuerkaixi, Abudukelimu and Cheng, Xuxin and Liu, Cao and Zeng, Ke and Jia, Xu and Cai, Xunliang},
  journal={arXiv preprint arXiv:2603.05551},
  year={2026}
}
```

## 🤝 贡献

欢迎 PR 和 Issue！详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📜 协议

[MIT](LICENSE)
