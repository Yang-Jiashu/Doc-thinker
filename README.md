<div align="center">

<img src="docs/assets/banner.png" alt="DocThinker Banner" width="820" />

# DocThinker

**Self-Evolving Knowledge Graphs · Tiered Memory · Structured Reasoning**

*Language captures the results of cognition, while cognition itself encompasses perception, experience, and reasoning.*

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

**DocThinker** is a document-driven RAG system that constructs self-evolving knowledge graphs from uploaded documents. Unlike conventional retrieve-then-respond pipelines, DocThinker treats knowledge as a **dynamic graph**.

### 🎬 Explore our Tutorial!!

<!-- TODO: Replace with demo video -->
> [▶️ **Watch the YouTube Tutorial**](#) | [🚀 **Use DocThinker in HuggingFace Space**](#) | [📝 **Try Colab Tutorial**](#)

---

## 📑 Index

- [🚀 Quick Install](#-quick-install)
- [🔥 Quick Start](#-quick-start)
  - [1. Web UI & Server](#1-web-ui--server)
  - [2. Python API Usage](#2-python-api-usage)
- [🧬 Key Contributions (Pipeline)](#-key-contributions)
  - [1. Test-Time Scaling & Agentic Memory](#1--test-time-scaling--agentic-memory)
  - [2. Two-Path KG Self-Expansion](#2--two-path-kg-self-expansion)
  - [3. Self-Evolving Knowledge Graph](#3--self-evolving-knowledge-graph)
  - [4. Multi-Agent Co-Evolution](#4--multi-agent-co-evolution)
  - [5. Tiered Conversation Memory (Claw)](#5--tiered-conversation-memory-claw)
  - [6. SPARQL Chain-of-Thought Reasoning](#6--sparql-chain-of-thought-cot-reasoning)
- [💡 Use Cases](#-use-cases)
- [⚡ Query Modes & PDF Processing](#-query-modes)
- [📡 API Reference](#-api-reference)
- [❓ FaQ](#-faq)

---

## 🚀 Quick Install

We recommend using Python version 3.10 or higher for DocThinker.

```bash
# 1. Clone the repository
git clone https://github.com/Yang-Jiashu/doc-thinker.git
cd doc-thinker

# 2. Create a virtual environment
conda create -n docthinker python=3.11 -y
conda activate docthinker

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .
```

---

## 🔥 Quick Start

### 1. Web UI & Server

The easiest way to experience DocThinker is through its web dashboard.

```bash
# 1. Configure environment variables (LLM API Keys)
cp env.example .env

# 2. Start the Backend API (FastAPI)
python -m uvicorn docthinker.server.app:app --host 0.0.0.0 --port 8000

# 3. Start the Frontend UI (Flask)
python run_ui.py
```
> Open `http://localhost:5000` — upload a PDF, ask questions, and explore the evolving knowledge graph.

### 2. Python API Usage

You can also use DocThinker programmatically with just a few lines of code.

```python
import asyncio
from docthinker import DocThinker, DocThinkerConfig

async def main():
    # 1. Configuration
    config = DocThinkerConfig(working_dir="./my_knowledge_base")
    
    # 2. Initialize (Requires LLM and Embedding models setup)
    dt = DocThinker(config=config, ...) 
    
    # 3. Ingest Document (Parsing & Knowledge Graph Construction)
    await dt.process_document_complete("your_document.pdf")
    
    # 4. Trigger Test-Time Scaling (Self-Study Loop) to enhance KG density
    await dt.run_self_study_loop(max_rounds=5)
    
    # 5. Query with SPARQL CoT Reasoning
    response = await dt.aquery("What is the core idea of the document?", mode="deep")
    print(response)

asyncio.run(main())
```

---

## 🧬 Key Contributions

DocThinker splits the monolithic pipeline into autonomous agents and introduces graph-based cognitive reasoning.

<div align="center">
<img src="docs/assets/pipeline.png" alt="DocThinker Pipeline" width="820" />
<p><b>Figure 1.</b> DocThinker end-to-end pipeline — from document input to knowledge graph construction, tiered memory management, hybrid retrieval & reasoning, and output with feedback back to the graph.</p>
</div>

### 1. 🧠 Test-Time Scaling & Agentic Memory
Between document ingestion and user querying, DocThinker runs a **background self-study loop** (Test-Time Scaling on KG). The LLM autonomously analyzes existing subgraphs, generates questions, retrieves answers, performs continuous deductive reasoning, and writes back new knowledge and methodological experiences (`entity_type="experience"`). This significantly increases graph density and reasoning capability *without* requiring additional user prompts.

### 2. 🔀 Two-Path KG Self-Expansion
Expansion operates in two complementary passes:
* **Path A (Cluster-based):** HDBSCAN clusters entity embeddings → LLM generates cluster summaries → expands new entities grounded in cluster themes.
* **Path B (Top-N multi-angle):** Top-50 highest-degree nodes expanded across 6 cognitive dimensions (hierarchy, causation, analogy, contrast, temporal, application).

### 3. 🔄 Self-Evolving Knowledge Graph
Newly expanded nodes do not immediately become authoritative knowledge — they enter the graph as `candidates`. Only when users repeatedly adopt a node in actual conversations do its usage count and score accumulate; once thresholds are met, the node is promoted to a formal part of the graph.

### 4. 🤖 Multi-Agent Co-Evolution
DocThinker splits the traditional RAG monolithic pipeline into three specialized Agents:
* **Retrieval Agent:** Maximizes retrieval hit rate.
* **Extraction Agent:** Maximizes extraction coverage.
* **Answering Agent:** Generates final answers and triggers node promotion/decay feedback.

<div align="center">
<img src="docs/assets/multi_agent_evolution.png" alt="Multi-Agent Co-Evolution Architecture" width="820" />
<p><sub><b>Figure 2.</b> DocThinker multi-Agent co-evolution architecture.</sub></p>
</div>

### 5. 🗃️ Tiered Conversation Memory (Claw)
Inspired by the [OpenClaw / Letta](https://github.com/letta-ai/letta) architecture, Claw implements a **three-layer memory hierarchy** (Hot, Warm, Cold) for unbounded conversation length.

### 6. 🧠 SPARQL Chain-of-Thought (CoT) Reasoning
Complex queries are internally decomposed into **SPARQL-style triple-pattern chains** before answer generation. The LLM binds variables against KG context via shared-variable chaining.

<div align="center">
<img src="docs/assets/sparql_cot.png" alt="SPARQL CoT Reasoning" width="680" />
</div>

---

## 💡 Use Cases

<table>
<tr>
<td width="50%" valign="top">

> *"Upload a novel and explore its knowledge graph"*

<img src="docs/assets/usecase_kg.gif" width="100%"/>

</td>
<td width="50%" valign="top">

> *"Deep-mode conversation with SPARQL CoT reasoning and tiered memory"*

<img src="docs/assets/usecase_chat.gif" width="100%"/>

</td>
</tr>
</table>

---

## ⚡ Query Modes

| Mode | Strategy | Latency | Depth |
|------|----------|---------|-------|
| **Fast** | Vector similarity | ~1 s | Shallow |
| **Standard** | Hybrid KG + vector + reranking | ~3 s | Medium |
| **Deep** | SPARQL CoT + spreading activation + episodic memory + expansion matching + post-query feedback | ~8 s | Full |

---

## 📄 PDF Processing

| Mode | Engine | Best for |
|------|--------|----------|
| `auto` (default) | VLM (short) / MinerU (long) | General use |
| `vlm` | Cloud VLM (Qwen-VL) | Image-heavy documents |
| `mineru` | MinerU layout engine | Long documents with complex tables |

---

## 📡 API Reference

<details>
<summary><b>Click to expand API endpoints</b></summary>

| Category | Endpoint | Method | Description |
|----------|----------|--------|-------------|
| Sessions | `/sessions` | GET / POST | List / create sessions |
| | `/sessions/{id}/history` | GET | Chat history |
| | `/sessions/{id}/files` | GET | Ingested files |
| Ingest | `/ingest` | POST | Upload PDF / TXT |
| | `/ingest/stream` | POST | Stream raw text |
| Query | `/query/stream` | POST | SSE streaming query |
| | `/query` | POST | Non-streaming query |
| KG | `/knowledge-graph/data` | GET | Nodes + edges for visualization |
| | `/knowledge-graph/expand` | POST | Trigger 2-path expansion |
| | `/knowledge-graph/stats` | GET | KG statistics |
| Memory | `/memory/stats` | GET | Episode + Claw memory stats |
| | `/memory/consolidate` | POST | Run episodic consolidation |
| Settings | `/settings` | GET / POST | Runtime config |

</details>

---

## 📝 Citation

If you find DocThinker useful in your research, please cite:

```bibtex
@article{yang2026autothinkrag,
  title={AutothinkRAG: Complexity-Aware Control of Retrieval-Augmented Reasoning for Image-Text Interaction},
  author={Yang, Jiashu and Zhang, Chi and Wuerkaixi, Abudukelimu and Cheng, Xuxin and Liu, Cao and Zeng, Ke and Jia, Xu and Cai, Xunliang},
  journal={arXiv preprint arXiv:2603.05551},
  year={2026}
}
```

## 🤝 Contributing

PRs and issues welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## 📜 License

[MIT](LICENSE)
