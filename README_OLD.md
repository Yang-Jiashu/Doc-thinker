<div align="center">

# 🐕 Doc Thinker

**文档即思考** — 像小狗一样拿着放大镜读懂每一份文档

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![GitHub](https://img.shields.io/badge/GitHub-Yang--Jiashu%2Fdoc--thinker-black?logo=github)](https://github.com/Yang-Jiashu/doc-thinker)

<img src="logo.png" alt="Doc Thinker Logo" width="220" />

**基于 AutoThink 架构** — 多模态文档解析 · 图 RAG 检索 · 一键测评

[快速开始](#-快速开始) · [特性](#-特性) · [项目结构](#-项目结构) · [测评](#-测评)

</div>

---

## ✨ 特性

- **📄 多模态解析** — MinerU / Docling 解析 PDF，支持文本、图片、表格、公式
- **🕸️ 图 RAG** — AutoThink 图引擎 + 图遍历（BFS 多跳），实体 / 关系检索更准
- **🔀 超图检索** — 超图存储与检索，适合复杂关系与多跳推理
- **📊 批量测评** — MMLongBench 等数据集一键评测，输出准确率与明细

---

## 🚀 快速开始

```bash
git clone https://github.com/Yang-Jiashu/doc-thinker.git
cd doc-thinker

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -U pip && pip install -r requirements.txt && pip install -e .
```

复制 `env.example` 为 `.env`，填入 LLM / Embedding 的 API 地址与 Key。

```bash
python scripts/mmtest_bai.py    # 主测评（需配置 DASHSCOPE_API_KEY、SILICONFLOW_API_KEY）
```

---

## 📁 项目结构

| 目录 | 说明 |
|------|------|
| 核心库目录 | **AutoThink 核心**：解析、入库、查询、知识图谱、自动思考、超图 |
| `scripts/` | 入库脚本、单题/批量/MMLongBench 评测 |
| `neuro_memory/` | 类脑记忆引擎（扩散激活、巩固、类比检索） |
| `docs/` | 项目说明与文档 |

---

## ⚙️ 配置

在 `.env` 中配置：`WORKING_DIR`、`LLM_BINDING_HOST` / `LLM_BINDING_API_KEY`、`EMBEDDING_*`、`RERANK_*`。完整项见 `env.example`。

---

## 📊 测评

| 脚本 | 用途 |
|------|------|
| `scripts/run_qa_eval.py` | 单问题集评测 |
| `scripts/run_batch_eval.py` | 按数据集批量评测（如 `--dataset aca`） |
| `scripts/mmtest_bai.py` | MMLongBench 云端 API 批量评测 |

数据约定：PDF 与 QA 放 `data/raw/...`，MinerU 输出 `*_content_list.json` 放 `data/mineru_output/` 或 `data/mineru_results/`。

---

## 📄 License & Contributing

- **License**：[MIT](LICENSE)
- **Contributing**：[CONTRIBUTING.md](CONTRIBUTING.md)
