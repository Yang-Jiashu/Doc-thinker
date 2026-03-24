<div align="center">

<img src="docs/assets/banner.png" alt="DocThinker Banner" width="820" />

# DocThinker

**Self-Evolving Knowledge Graphs with Tiered Memory and Structured Reasoning**

*Build a living knowledge graph from documents — it grows, restructures, and reasons on its own.*

[![Paper](https://img.shields.io/badge/arXiv-2603.05551-b31b1b.svg)](https://arxiv.org/abs/2603.05551)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Demo](https://img.shields.io/badge/Demo-Live-orange)](http://localhost:5000)
[![LightRAG](https://img.shields.io/badge/LightRAG-Based-8B5CF6)](https://github.com/HKUDS/LightRAG)
[![OpenClaw](https://img.shields.io/badge/OpenClaw-Integration-E74C3C)](https://github.com/letta-ai/letta)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Flask](https://img.shields.io/badge/Flask-UI-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![NetworkX](https://img.shields.io/badge/NetworkX-KG-4C72B0)](https://networkx.org/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector-3B5998?logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![D3.js](https://img.shields.io/badge/D3.js-Visualization-F9A03C?logo=d3dotjs&logoColor=white)](https://d3js.org/)
[![HDBSCAN](https://img.shields.io/badge/HDBSCAN-Clustering-27AE60)](https://hdbscan.readthedocs.io/)
[![SpaCy](https://img.shields.io/badge/SpaCy-NER-09A3D5?logo=spacy&logoColor=white)](https://spacy.io/)

[English](README.md) | [中文](README.zh-CN.md)

[Quick Start](#-quick-start) · [Key Contributions](#-key-contributions) · [Pipeline](#-overview) · [Use Cases](#-use-cases) · [API Reference](#api-reference)

</div>

## 📖 Overview

**DocThinker** is a document-grounded RAG system that constructs a living, self-evolving knowledge graph from uploaded documents. Unlike conventional retrieve-and-generate pipelines, DocThinker treats the knowledge graph as a **dynamic cognitive substrate** — it grows through ingestion, restructures itself through usage-driven feedback, and reasons over structured triples via SPARQL-style chain-of-thought decomposition. Built on [LightRAG](https://github.com/HKUDS/LightRAG) with integrated [OpenClaw / Letta](https://github.com/letta-ai/letta) tiered memory.

<!-- TODO: Replace with demo video -->
<!-- https://github.com/user-attachments/assets/YOUR_VIDEO_ID -->

<div align="center">
<img src="docs/assets/pipeline.png" alt="DocThinker Pipeline" width="820" />
<p><b>Figure 1.</b> DocThinker end-to-end pipeline — five-layer architecture spanning input perception, query cognition,<br/>dynamic memory core (KG self-expansion, Claw tiered memory, episodic memory), hybrid retrieval & reasoning, and output feedback loop.</p>
</div>

### ✨ Highlights

- **Self-evolving knowledge graph** — LLM-expanded candidate nodes are validated against real user queries; only those adopted in answers survive and get promoted to the formal KG
- **Tiered episodic memory (Claw)** — An OpenClaw-inspired three-layer memory hierarchy (hot / warm / cold) that mirrors human short-term, working, and long-term memory, enabling unbounded conversation length
- **SPARQL Chain-of-Thought reasoning** — Complex queries are decomposed into triple-pattern chains with variable binding against KG context, replacing "find relevant info" with systematic graph traversal
- **Autonomous edge discovery** — A background pipeline scans entity windows to surface latent relationships the original extraction missed, then validates edge plausibility
- **Two-path KG self-expansion** — HDBSCAN clustering + top-N multi-angle cognitive expansion, with LLM self-validation and semantic deduplication
- **Interactive KG visualization** — D3.js force-directed graph with color-coded nodes (original / expanded / promoted), discovered edges (dashed red), and real-time exploration

---

## 🧬 Key Contributions

### 1. 🔀 Two-Path KG Self-Expansion

Expansion operates in two complementary passes:

| Path | Strategy | Grounding |
|------|----------|-----------|
| **A — Cluster-based** | HDBSCAN clusters entity embeddings; each cluster receives an LLM summary; expansion generates entities grounded in cluster themes | Density structure |
| **B — Top-N multi-angle** | Top-50 highest-degree nodes are expanded across 6 cognitive dimensions (hierarchy, causation, analogy, contrast, temporal, application) | Graph topology |

All candidates pass through **LLM self-validation** (factuality, non-redundancy, edge validity, specificity scoring) and **semantic deduplication** before admission.

### 2. 🔄 Expanded Node Lifecycle

<div align="center">
<img src="docs/assets/lifecycle.png" alt="Expanded Node Lifecycle" width="680" />
<p><sub><b>Figure 2.</b> Expanded node lifecycle — candidates are validated by real user queries; only adopted nodes survive promotion to the formal KG.</sub></p>
</div>

The graph **earns** its knowledge — only query-validated expansions persist.

### 3. 🧠 SPARQL Chain-of-Thought (CoT) Reasoning

<div align="center">
<img src="docs/assets/sparql_cot.png" alt="SPARQL CoT Reasoning" width="680" />
<p><sub><b>Figure 3.</b> SPARQL Chain-of-Thought reasoning pipeline — queries are decomposed into triple-pattern chains with variable binding against KG context.</sub></p>
</div>

Complex queries are internally decomposed into **SPARQL-like triple-pattern chains** before answer generation. The LLM binds variables against KG context via shared-variable chaining, constructs a variable binding table, then synthesizes the final answer. This replaces unstructured "find relevant info" with **systematic graph traversal reasoning**.

### 4. 🗃️ Tiered Episodic Memory (Claw)

Inspired by the [OpenClaw / MemGPT / Letta](https://github.com/letta-ai/letta) architecture, Claw implements a **three-layer memory hierarchy**:

| Layer | Temperature | Mechanism | Injection |
|-------|------------|-----------|-----------|
| **Hot** — Working Memory | Immediate | Recent *N* conversation turns | Always prepended |
| **Warm** — Core Memory | Session | LLM-compressed `MEMORY.md` (user preferences, facts, instructions) | Always prepended |
| **Cold** — Semantic Archive | Long-term | Older turns chunked, embedded, and vector-indexed | Top-*k* retrieval per query |

After each Q&A turn, older conversations are automatically archived to the cold layer, and the warm layer is periodically re-compressed by the LLM — enabling **unbounded conversation length** without context window overflow.

### 5. 🔍 Background Edge Discovery & Validation

After entity extraction, many cross-chunk relationships are missed. DocThinker runs a **background edge discovery pipeline**:

1. **Windowed scanning** — Entities are grouped into overlapping windows (size 30, overlap 10).
2. **LLM relationship inference** — Each window is analyzed for 6 relationship types: hierarchical, causal, contrastive, temporal, application, and collaborative.
3. **Deduplication & validation** — Discovered edges are checked against existing edges and validated for plausibility.
4. **Visual distinction** — Discovered edges are persisted with `is_discovered=1` and rendered distinctly (e.g., dashed red) in the KG visualization.

### 6. 🌊 Episodic Memory with Spreading Activation

The `neuro_memory` module implements a brain-inspired episodic memory system:

- **Episode store** — Every Q&A interaction is stored as a structured episode with entities, triples, and embeddings.
- **Spreading activation** — Retrieval propagates activation from seed nodes along typed edges with edge-type-specific decay, simulating human associative recall.
- **Analogical retrieval** — Past episodes are scored by content similarity (0.6), structural similarity (0.25), and salience (0.15) for experience transfer.
- **Consolidation** — Periodic consolidation strengthens frequently co-activated edges and infers cross-episode relations.

---

## 🚀 Quick Start

```bash
git clone https://github.com/Yang-Jiashu/doc-thinker.git && cd doc-thinker
conda create -n docthinker python=3.11 -y && conda activate docthinker
pip install -r requirements.txt && pip install -e .
cp env.example .env   # ← fill in API keys (OpenAI / DashScope / SiliconFlow)
```

**Launch:**

```bash
# Terminal 1 — Backend (FastAPI)
python -m uvicorn docthinker.server.app:app --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend (Flask UI)
python run_ui.py
```

Open `http://localhost:5000` — upload a PDF, ask questions, and explore the evolving knowledge graph.

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

<details>
<summary><b>Deep Mode Pipeline (7 steps)</b></summary>

1. Retrieve analogous episodes from episodic memory via spreading activation.
2. Match expanded candidate nodes against the query (token-overlap + embedding).
3. Inject matched expansions as forced retrieval instructions.
4. Decompose query into SPARQL CoT triple-pattern chain.
5. Hybrid KG + vector retrieval with spreading activation.
6. LLM generates answer with full context and variable binding.
7. Post-query feedback: validate expanded nodes, store episode, co-activate links, update Claw memory layers.

</details>

## 📄 PDF Processing

| Mode | Engine | Best for |
|------|--------|----------|
| `auto` (default) | VLM (short) / MinerU (long) | General use |
| `vlm` | Cloud VLM (Qwen-VL) | Image-heavy documents |
| `mineru` | MinerU layout engine | Long documents with complex tables |

<details>
<summary><b>📡 API Reference</b></summary>

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

<details>
<summary><b>📂 Project Structure</b></summary>

| Directory | Description |
|-----------|-------------|
| `docthinker/` | Core: PDF parsing, KG construction, query routing, 2-path expansion (`kg_expansion/`), auto-thinking (`auto_thinking/`), HyperGraphRAG (`hypergraph/`), server (`server/`), UI (`ui/`). |
| `graphcore/` | Graph RAG engine: KG storage (NetworkX / FAISS / Qdrant / PG), SPARQL CoT prompting, entity extraction, reranking. |
| `neuro_memory/` | Episodic memory: spreading activation, episode store, analogical retrieval, consolidation. |
| `claw/` | Tiered memory: hot (working), warm (core / MEMORY.md), cold (semantic archive). |
| `config/` | `settings.yaml` — PDF, memory, retrieval, cognition parameters. |

</details>

---

## 📝 Citation

If you find DocThinker useful in your research, please cite:

```bibtex
@article{docthinker2026,
  title={DocThinker: Self-Evolving Knowledge Graphs with Tiered Memory and Structured Reasoning for Document Understanding},
  author={Yang, Jiashu},
  journal={arXiv preprint arXiv:2603.05551},
  year={2026}
}
```

## 🤝 Contributing

PRs and issues welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## 📜 License

[MIT](LICENSE)
