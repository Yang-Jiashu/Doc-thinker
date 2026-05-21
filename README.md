<div align="center">

<img src="docs/assets/banner.png" alt="DocThinker Banner" width="820" />

# DocThinker

**Agentic Memory Framework · Self-Evolving Knowledge Graphs · Document Reasoning**

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

**DocThinker** is an agentic memory framework for document understanding. It turns documents, chat turns, retrieval traces, and graph expansions into a multi-layer memory system that can recall, reason, and consolidate knowledge over time.

The name has two meanings: **Doc** as in documents, and **Doc** as in doctor-level depth. DocThinker is built for document-grounded memory with research-grade reasoning.

Unlike a conventional retrieve-then-respond RAG pipeline, DocThinker treats knowledge as an evolving memory substrate: session-scoped knowledge graphs provide semantic memory, Claw provides tiered conversation memory, Neuro Memory provides episodic analogies, and KG expansion tracks hypotheses that can be promoted through use.

---

## 📑 Index

- [🚀 Quick Install](#-quick-install)
- [🔥 Quick Start](#-quick-start)
  - [1. Web UI & Server](#1-web-ui--server)
  - [2. Python API Usage](#2-python-api-usage)
  - [3. Memory Layer API](#3-memory-layer-api)
- [🧬 Key Contributions](#-key-contributions)
  - [1. Agentic Memory Core](#1--agentic-memory-core)
  - [2. Session-Scoped Knowledge Graphs](#2--session-scoped-knowledge-graphs)
  - [3. Self-Evolving KG Expansion](#3--self-evolving-kg-expansion)
  - [4. Tiered Conversation Memory (Claw)](#4--tiered-conversation-memory-claw)
  - [5. Episodic Analogy Memory](#5--episodic-analogy-memory)
  - [6. Multimodal Retrieval Signals](#6--multimodal-retrieval-signals)
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
    
    # 4. Query the session knowledge graph
    response = await dt.aquery("What is the core idea of the document?", mode="mix")
    print(response)

asyncio.run(main())
```

### 3. Memory Layer API

The memory layer can be embedded without using the full web app. Third-party projects can implement the backend protocols and plug them into `AgentMemoryCore`.

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
    query="What should the agent remember before answering?",
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

## 🧬 Key Contributions

DocThinker organizes retrieval and memory as an agent-facing framework instead of scattering memory logic across API handlers.

<div align="center">
<img src="docs/assets/pipeline.png" alt="DocThinker Pipeline" width="820" />
<p><b>Figure 1.</b> DocThinker end-to-end pipeline — from document input to knowledge graph construction, tiered memory management, hybrid retrieval & reasoning, and output with feedback back to the graph.</p>
</div>

### 1. 🧠 Agentic Memory Core
`docthinker.memory_core.AgentMemoryCore` is the stable facade for agent memory work. It exposes explicit backend protocols for conversation memory, episodic memory, expanded KG hypotheses, graph promotion, and optional chat-turn ingestion. Before generation, `recall()` merges:

* Claw working/core/archive conversation memory.
* Neuro Memory episodic analogy matches.
* KG expanded-node matches and forced retrieval instructions.

After generation, `after_response()` consolidates the turn back into memory layers, writes chat episodes, optionally feeds the Q&A back into the graph, and promotes useful expanded nodes.

### 2. 🧩 Session-Scoped Knowledge Graphs
Each session owns its own GraphCore-backed knowledge graph and document state. Uploaded files are parsed, inserted, and queried within that session, which keeps user context isolated while still allowing the graph to grow over time.

### 3. 🔀 Self-Evolving KG Expansion
Expansion operates in two complementary passes:
* **Path A (Cluster-based):** HDBSCAN clusters entity embeddings → LLM generates cluster summaries → expands new entities grounded in cluster themes.
* **Path B (Top-N multi-angle):** Top-50 highest-degree nodes expanded across 6 cognitive dimensions (hierarchy, causation, analogy, contrast, temporal, application).

Newly expanded nodes do not immediately become authoritative knowledge. They enter as candidates, are matched during query time, and only become formal graph nodes after repeated useful adoption in assistant responses.

<div align="center">
<img src="docs/assets/multi_agent_evolution.png" alt="Memory and graph feedback architecture" width="820" />
<p><sub><b>Figure 2.</b> Memory and graph feedback loop.</sub></p>
</div>

### 4. 🗃️ Tiered Conversation Memory (Claw)
Claw implements a three-layer memory hierarchy for long-running conversations: hot working memory, warm core summaries, and cold semantic archives.

### 5. 🧠 Episodic Analogy Memory
Neuro Memory stores chat/document experiences as episodes and retrieves similar past situations as analogy context. These matches are surfaced through `episodic_matches` and injected as guidance rather than treated as direct factual sources.

<div align="center">
<img src="docs/assets/sparql_cot.png" alt="Structured reasoning" width="680" />
</div>

### 6. 🖼️ Multimodal Retrieval Signals
DocThinker tracks image assets extracted from documents and can activate relevant visual evidence during deep UI queries.

---

## 💡 Use Cases

<table>
<tr>
<td width="50%" valign="top">

> *"Upload a novel and explore its knowledge graph"*

<img src="docs/assets/usecase_kg.gif" width="100%"/>

</td>
<td width="50%" valign="top">

> *"Deep-mode conversation with episodic memory, expanded KG matching, and tiered memory"*

<img src="docs/assets/usecase_chat.gif" width="100%"/>

</td>
</tr>
</table>

---

## ⚡ Query Modes

| Mode | UI mapping | Strategy | Depth |
|------|------------|----------|-------|
| **Quick** | `naive` | Lightweight vector-style retrieval, rerank disabled | Shallow |
| **Standard** | `local` | Session KG retrieval with reranking | Medium |
| **Deep** | `mix` | KG + vector retrieval, Claw memory, episodic analogies, expanded-node matching, image activation, post-query consolidation | Full |

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
| Sessions | `/api/v1/sessions` | GET / POST | List / create sessions |
| | `/api/v1/sessions/{id}/history` | GET | Chat history |
| | `/api/v1/sessions/{id}/files` | GET | Ingested files |
| Ingest | `/api/v1/ingest` | POST | Upload PDF / TXT |
| | `/api/v1/ingest/stream` | POST | Stream raw text |
| Query | `/api/v1/query/stream` | POST | SSE streaming query |
| | `/api/v1/query` | POST | Non-streaming query |
| | `/api/v1/query/text` | POST | Alias for non-streaming query |
| KG | `/api/v1/knowledge-graph/data` | GET | Nodes + edges for visualization |
| | `/api/v1/knowledge-graph/expand` | POST | Trigger KG expansion |
| | `/api/v1/knowledge-graph/stats` | GET | KG statistics |
| | `/api/v1/knowledge-graph/expanded-nodes` | GET | Expanded-node lifecycle state |
| Memory | `/api/v1/memory/stats` | GET | Episode + Claw memory stats |
| Settings | `/api/v1/settings` | GET / POST | Runtime config |

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
