<div align="center">

<img src="logo.png" width="64" height="64" alt="Original DocThinker logo" />

# DocThinker 2.0

**A personal agent with a memory you can inspect — and work you can continue.**

Remember what matters. See what changed. Pick up where you left off.

[English](README.md) · [中文](README.zh-CN.md) · [Try it](#quick-start) · [Architecture](docs/ARCHITECTURE.md) · [Acceptance](docs/RELEASE_ACCEPTANCE_2026-10-20.md)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB)](pyproject.toml)
[![CI](https://github.com/Yang-Jiashu/Doc-thinker/actions/workflows/ci.yml/badge.svg)](https://github.com/Yang-Jiashu/Doc-thinker/actions/workflows/ci.yml)
[![Paper](https://img.shields.io/badge/arXiv-2603.05551-b31b1b)](https://arxiv.org/abs/2603.05551)
[![License](https://img.shields.io/badge/License-PolyForm_Shield-orange)](LICENSE)

</div>

[![DocThinker overview: confirmed memory, version-bound evidence and approved project phases](docs/assets/docthinker-v2-overview.svg)](docs/ARCHITECTURE.md)

*Conceptual workflow, not a product screenshot or benchmark. Dotted connections remain unverified.*

## Not another conversation that starts from zero

DocThinker connects your materials, confirmed memories and ongoing projects. Use it for work decisions, everyday planning, research or writing — not just papers. Chat normally; inspect the context and evidence when you need to.

- **Memory with an owner.** Confirm, search, correct, archive or delete what the agent remembers. Storage can grow; each answer still uses bounded context.
- **Connections with a history.** Trace answers to sources, events and versions. Correcting or withdrawing an input invalidates dependent results instead of silently recycling them.
- **Projects with continuity.** Approve a plan, pause, restart and continue valid checkpoints. Add a phase without replaying every previous conversation.

## See it, then try your own task

<p align="center">
  <img src="docs/assets/usecase_chat.gif" width="48%" alt="Original demo: ask questions grounded in your material" />
  <img src="docs/assets/usecase_kg.gif" width="48%" alt="Original demo: explore related concepts in the knowledge graph" />
</p>

*Original chat and graph demonstrations, preserved unchanged. Available capabilities differ by runtime profile below.*

Try a task that changes: “Help me prepare next week's project review.” Add notes and constraints, confirm a preference, then introduce a corrected deadline. Inspect the sources, review which results became stale, and continue an approved phase. The [long-project guide](docs/LONG_PROJECT_RUNTIME.md) explains the real workflow and its limits; offline examples only demonstrate interaction.

## What happens behind the conversation

One **Harness** controls scope, permissions, versions, retrieval budgets and execution. Models interpret material and draft responses; algorithms handle indexing, deduplication, bounded graph/path search, scheduling and source invalidation.

Optional **Insight Schema** knowledge and methods help find supporting material and organize evidence in the background. Analogy, deduction and induction are aids to analysis — not separate user-facing modes or permission to invent facts. Original evidence takes priority.

For approved long tasks, the loop is **compose → check → find missing evidence → revise → check again**. Each executed node consumes its own approved model-call budget. Event directions, prerequisites, counterevidence and unknowns remain explicit; a connected graph is not proof of causation.

[Runtime architecture](docs/ARCHITECTURE.md) · [Memory and project lifecycle](docs/LONG_PROJECT_RUNTIME.md) · [Insight Schema design](docs/INSIGHT_RUNTIME_DESIGN.md)

## Quick start

### 1 · Explore the interface locally

From a repository checkout, use the shared mobile/desktop-browser client:

Requires Node.js 22 and Python 3 for the local static preview.

```sh
git clone https://github.com/Yang-Jiashu/Doc-thinker.git
cd Doc-thinker/clients/mobile
npm ci --ignore-scripts
npm run build:h5
npm run preview:h5
```

Open [localhost:5175](http://127.0.0.1:5175). The default demo uses labelled sample data, makes no model calls and does not request the microphone. Installing dependencies requires network access; the demo is not a real-answer evaluation. Use the static preview, not a publicly exposed development server.

### 2 · Invite a few people to a private instance

Use the [Docker Compose invitation deployment guide](deploy/invite/README.md): build the service, configure your model privately, create separate invited accounts, and deploy behind HTTPS. API keys stay on the server; users open a webpage.

The default profile provides chat and history. Explicit research enablement adds PDF/TXT/MD import, source-grounded answers, confirmed memory and approved project phases. Full graph growth, gesture control and voice are **not** included in this invitation profile. A deployment bundle does not create a public URL or grant permission to resell a model subscription.

Downloaded an invitation source ZIP? Follow its deployment guide; full-workspace development commands below do not apply to the restricted bundle.

### 3 · Run the full graph workbench

<details>
<summary>Desktop / graph developer installation</summary>

From the repository root, with Python 3.11 recommended:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all]"
cp -n env.example .env
```

Configure accessible chat, embedding and optional vision models in `.env` before starting. Never commit credentials. Then use two terminals with the environment activated:

```sh
# Terminal 1
python -m uvicorn docthinker.server.app:app --host 127.0.0.1 --port 8000

# Terminal 2
UI_HOST=127.0.0.1 UI_PORT=5001 python run_ui.py
```

Open [localhost:5001/query](http://127.0.0.1:5001/query). Start with a TXT file before heavier multimodal parsing. This original desktop service must not be exposed as the authenticated invitation gateway.

[Model configuration](docs/MODEL_CONFIGURATION.md) · [Graph and gestures](docs/GESTURE_INTERACTION.md) · [Optional voice services](docs/VOICE_MODELS.md) · [Mobile and mini-program builds](docs/MOBILE_APP.md)

</details>

## What has actually been checked

The integrated software snapshot is [main `0cda681`](https://github.com/Yang-Jiashu/Doc-thinker/commit/0cda681). Software contracts and answer quality are reported separately:

| Check | Recorded result |
|---|---|
| [Backend CI](https://github.com/Yang-Jiashu/Doc-thinker/actions/runs/39445691276) | Python 3.10 and 3.11: **7,140 passed each**, 9 optional skips each |
| [Client / product contracts](https://github.com/Yang-Jiashu/Doc-thinker/actions/runs/39445691569) | **999 client tests passed**; invitation and preservation contracts passed |
| [Long-project integration](docs/RELEASE_ACCEPTANCE_2026-10-20.md) | **30 phases, 3 restarts**; source withdrawal, dependent-result invalidation and tenant isolation, using local model doubles |
| [Frozen real-model acceptance](benchmarks/runtime_algorithms/goal_evidence_staged_result_20261020.json) | **8 Step 5 Preview calls delivered** across work, home and a real paper; **accepted with caveats** |

The real-model run belongs to the earlier frozen production fingerprint `88f08e28…`, not a new model run on the merged commit. Review found two overstrong paraphrases; it is neither a held-out quality win nor evidence of universal causal reliability. [Full acceptance and historical results](docs/RELEASE_ACCEPTANCE_2026-10-20.md) retain failures as well as successes, including [earlier helper regressions](benchmarks/research_real/result_evidence_trace_20261010.json).

### Current boundary

This is a **self-hosted preview**, not a certified production service. Memory and project storage have explicit capacity limits; event reasoning remains evidence-conditioned and may be wrong. Public HTTPS operation, security/dependency review, signed apps, real-phone experience and sustained customer use need deployment-specific acceptance. Operators and configured model providers may process submitted content; this is not end-to-end encryption. RSI is the research direction: validated strategy adoption and rollback are not yet a complete autonomous improvement loop. See the [RSI roadmap](docs/RSI_ROADMAP.md).

## Original memory and graph gallery

<details>
<summary>Expand the original demos, memory interface and architecture artwork</summary>

<img src="docs/assets/banner.png" width="900" alt="Original DocThinker banner" />
<img src="docs/assets/memory_kg_observability_demo.gif" width="900" alt="Original memory and knowledge-graph observability demo" />
<img src="docs/assets/gpu_knowledge_graph_demo.gif" width="900" alt="Original GPU knowledge-graph demo" />
<img src="docs/assets/ui_memory_editor_screenshot.png" width="900" alt="Original editable memory interface" />
<img src="docs/assets/agentic_memory_architecture.png" width="900" alt="Original layered personal-agent memory architecture" />
<img src="docs/assets/pipeline.png" width="900" alt="Original document-processing pipeline" />
<img src="docs/assets/lifecycle.png" width="900" alt="Original memory lifecycle" />
<img src="docs/assets/multi_agent_evolution.png" width="900" alt="Original multi-agent evolution framework" />
<img src="docs/assets/sparql_cot.png" width="900" alt="Original graph-query reasoning illustration" />

Historical illustrations show the project's design lineage; they do not establish every pictured mechanism as an accepted capability of every runtime profile. The shared client retains its existing <img src="clients/mobile/src/static/brand/logo-32.png" width="24" height="24" alt="Evidence Window mobile mark" /> Evidence Window identity.

</details>

## Go deeper

[Architecture and algorithms](docs/ARCHITECTURE.md) · [Persistent projects](docs/LONG_PROJECT_RUNTIME.md) · [Insight Schema](docs/INSIGHT_RUNTIME_DESIGN.md) · [Source-grounded retrieval](docs/RESEARCH_GROUNDED_DESIGN.md) · [Performance measurements](docs/PERFORMANCE_STRUCTURES_2026-10-13.md)

[Deployment](deploy/invite/README.md) · [Mobile](docs/MOBILE_APP.md) · [Memory plugin](docs/MEMORY_PLUGIN_GUIDE.md) · [Evaluation protocol](docs/SELF_EVOLUTION_EVALUATION.md) · [Contributing](CONTRIBUTING.md)

## Citation and license

```bibtex
@article{yang2026autothinkrag,
  title={AutothinkRAG: Complexity-Aware Control of Retrieval-Augmented Reasoning for Image-Text Interaction},
  author={Yang, Jiashu and Zhang, Chi and Wuerkaixi, Abudukelimu and Cheng, Xuxin and Liu, Cao and Zeng, Ke and Jia, Xu and Cai, Xunliang},
  journal={arXiv preprint arXiv:2603.05551},
  year={2026}
}
```

Current releases use the [PolyForm Shield License 1.0.0](LICENSE). Previously MIT-licensed releases retain their original license.
