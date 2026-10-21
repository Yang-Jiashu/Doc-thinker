<div align="center">

<img src="logo.png" width="64" height="64" alt="Original DocThinker logo" />

# DocThinker 2.0

**A personal agent with a memory you can inspect — and work you can continue.**

[English](README.md) · [中文](README.zh-CN.md) · [Try it](#quick-start) · [Architecture](docs/ARCHITECTURE.md) · [Acceptance](docs/RELEASE_ACCEPTANCE_2026-10-20.md) · [History](docs/HISTORY.md)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB)](pyproject.toml)
[![CI](https://github.com/Yang-Jiashu/Doc-thinker/actions/workflows/ci.yml/badge.svg)](https://github.com/Yang-Jiashu/Doc-thinker/actions/workflows/ci.yml)
[![Paper](https://img.shields.io/badge/arXiv-2603.05551-b31b1b)](https://arxiv.org/abs/2603.05551)
[![License](https://img.shields.io/badge/License-PolyForm_Shield-orange)](LICENSE)

</div>

## Your memory, in view. Your evidence, in reach.

The updated H5 workbench connects memory observability with **two data views in the same interactive canvas**: **Memory & sources**, and **Knowledge relations · full service**. Research-enabled invitations expose the bounded source/event projection; a research-enabled private full-service connection can switch between that view and the GraphCore knowledge window. Server capabilities determine what is available.

**Inspect context:** confirmed memories, source links and versions stay distinguishable from model-generated hypotheses. **Navigate evidence:** open the accessible record behind a node, not an invented relationship.

### Bring the request. Keep the thread.

![Current workspace: keep a draft, open project tools and append a continuation recipe without sending](docs/assets/ui-v2-workspace.gif)

Your draft stays yours: open **Project tools → Continue project → Append to draft**. An actual isolated service is connected, but this interaction neither sends the question nor calls a model.

### See what stays — and where it came from

![Current memory observatory: inspect layers, source links, explicit events and an existing memory version history](docs/assets/ui-v2-memory-observatory.gif)

Switch between **layers, sources and events**, then inspect a saved memory's versions. These are synthetic records in real isolated storage; counts describe this session, missing metrics remain unknown, and a bounded snapshot is not a growth benchmark.

### Follow the evidence, in the same workspace

![Current interactive source and event canvas: pan, zoom, select a node and open its synthetic original passage](docs/assets/ui-v2-graph-canvas.gif)

Pan and zoom the canvas, select a node and open its accessible passage. This clip uses the bounded research source/event projection; private full-service users can switch the **same canvas** to a GraphCore knowledge window and request a selected node's neighborhood. Neither view is an unbounded graph export or proof of causation; the older GPU renderer and camera gestures are not presented as migrated features.

All three clips use synthetic data and **zero external model calls**. They show actual browser interactions, not mockups or answer-quality, camera-gesture, real-phone or voice acceptance. [Recording method and boundaries →](docs/UI_SHOWCASE.md)

Earlier recordings and the original framework remain unchanged in the [visible history archive](docs/HISTORY.md), rather than standing in for the current product.

## Remember what matters. Pick up where you left off.

Your materials, confirmed memories and ongoing projects belong in one conversation — for work, everyday planning, research and writing.

[![DocThinker overview: confirmed memory, version-bound evidence and approved project phases](docs/assets/docthinker-v2-overview.svg)](docs/assets/docthinker-v2-overview.svg)

*Conceptual workflow, not a screenshot or benchmark; dotted connections remain unverified.*

## 01 · One Harness, a clear division of work

[![System architecture: the shared Harness controls retrieval, context, models and delivery; the original full graph service remains a separate path](docs/assets/architecture-system.svg)](docs/assets/architecture-system.svg)

Code handles scope, search and execution budgets. Models interpret and write, with Insight Schema assisting evidence gathering in the background.

## 02 · A memory you can correct, not just accumulate

[![Memory architecture: original passages, confirmed personal context and conditional event relations enter answers in separate layers; changes invalidate dependent results](docs/assets/architecture-memory.svg)](docs/assets/architecture-memory.svg)

**Sources are evidence; confirmed memories are personal context; relationships retain their conditions.** Correct or withdraw an input, and dependent results become stale; storage can expand while each retrieval and prompt stays bounded.

## 03 · Continue the project, not the entire transcript

[![Long-task loop: plan, explicit approval, composition, independent checking and delivery; gaps lead to local lookup and user-approved revisions, not an automatic paid loop](docs/assets/architecture-task-loop.svg)](docs/assets/architecture-task-loop.svg)

**Approve → compose → check → fill evidence gaps → revise → deliver.** Restart restores state, not permission to spend: further paid work needs approval, and changed sources invalidate only affected dependency branches.

## Design history, preserved

[Original demos, framework artwork and previous UI recordings →](docs/HISTORY.md). These remain unchanged and directly visible in the archive, not presented as current product screenshots.

## Quick start

**Explore the interface:** [build the local browser preview](#local-interface-preview) — labelled samples, no model calls or microphone access.

**Try your own material:** [deploy a private invitation instance](deploy/invite/README.md) — separate accounts, server-held API keys, HTTPS required for remote use.

**Explore the full graph:** [run the desktop workbench](#full-graph-workbench) — a separate developer profile with optional multimodal, gesture and voice components.

The invitation default is chat/history; explicitly enabling its research core adds PDF/TXT/MD, confirmed memory, approved phases and the permitted bounded source/event projection. That projection is not the separate full GraphCore service, automatic graph growth, gesture or voice; private hosting is not end-to-end encryption, and operators/model providers may process content.

Try a changing task: “Help me prepare next week's project review.” Add notes, confirm a preference, correct a deadline, inspect stale results, then continue an approved phase; the [long-project guide](docs/LONG_PROJECT_RUNTIME.md) explains the real workflow.

## Technical details, installation and acceptance

The integration baseline passed **7,140 backend tests on each Python version** and **999 client tests**; frozen real-model acceptance remains **accepted with caveats**, not a universal quality claim.

<details>
<summary id="local-interface-preview">Local interface preview · commands</summary>

Requires Node.js 22 and Python 3; dependency installation accesses npm.

```sh
git clone https://github.com/Yang-Jiashu/Doc-thinker.git
cd Doc-thinker/clients/mobile
npm ci --ignore-scripts
npm run build:h5
npm run preview:h5
```

Open [localhost:5175](http://127.0.0.1:5175). The default demo uses labelled sample data, does not read real documents and is not an answer-quality evaluation; use the static preview rather than a publicly exposed development server.

For invited users, follow the [Docker Compose guide](deploy/invite/README.md), including configuration, user creation and HTTPS; a source bundle does not create a public URL or grant permission to resell a model subscription. Downloaded invitation ZIPs use that guide, not full-workspace development commands.

</details>

<details>
<summary id="full-graph-workbench">Full graph workbench · developer installation</summary>

From the repository root, with Python 3.11 recommended:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all]"
cp -n env.example .env
```

Configure accessible chat, embedding and optional vision models in `.env` before starting; never commit credentials. Then use two terminals with the environment activated:

```sh
# Terminal 1
python -m uvicorn docthinker.server.app:app --host 127.0.0.1 --port 8000

# Terminal 2
UI_HOST=127.0.0.1 UI_PORT=5001 python run_ui.py
```

Open [localhost:5001/query](http://127.0.0.1:5001/query) and start with TXT before heavier multimodal parsing; do not expose this original desktop service as an authenticated invitation gateway.

[Model configuration](docs/MODEL_CONFIGURATION.md) · [Graph and gestures](docs/GESTURE_INTERACTION.md) · [Optional voice](docs/VOICE_MODELS.md) · [Mobile / mini-program builds](docs/MOBILE_APP.md)

</details>

<details>
<summary>What was tested · software, real-model results and retained failures</summary>

Integration baseline: [main `0cda681`](https://github.com/Yang-Jiashu/Doc-thinker/commit/0cda681); software contracts and answer quality are different checks.

| Check | Recorded result |
|---|---|
| [Backend CI](https://github.com/Yang-Jiashu/Doc-thinker/actions/runs/39445691276) | Python 3.10 / 3.11: **7,140 passed each**, 9 optional skips each |
| [Client / product contracts](https://github.com/Yang-Jiashu/Doc-thinker/actions/runs/39445691569) | **999 client tests passed**; invitation and preservation contracts passed |
| [Long-project integration](docs/RELEASE_ACCEPTANCE_2026-10-20.md) | **30 phases, 3 restarts**; withdrawal, dependent-result invalidation and tenant isolation, using local model doubles |
| [Frozen real-model acceptance](benchmarks/runtime_algorithms/goal_evidence_staged_result_20261020.json) | **8 Step 5 Preview calls delivered** across work, home and a real paper; **accepted with caveats** |

The real-model run belongs to frozen production fingerprint `88f08e28…`, not a new run on the merged commit; reviewers found two overstrong paraphrases. It is not a held-out quality win or causal-reliability proof, and [full acceptance](docs/RELEASE_ACCEPTANCE_2026-10-20.md) retains failures, including [earlier helper regressions](benchmarks/research_real/result_evidence_trace_20261010.json).

</details>

<details>
<summary>Deployment boundaries, privacy and the research roadmap</summary>

This is a self-hosted preview, not a certified production service; memory and project storage have explicit capacity limits. Public HTTPS operation, dependency/security review, signed apps, real-phone experience and sustained customer use require deployment-specific acceptance.

Models may misinterpret sources or event relations; inspect evidence before relying on consequential conclusions. Source withdrawal suppresses future use and dependent results, but cannot erase already copied content, provider records or backups; operators and configured model providers may process submitted content, and self-hosting is not end-to-end encryption.

RSI is the research direction: independent evaluation, validated strategy adoption and complete rollback are not a finished autonomous improvement loop. See the [RSI roadmap](docs/RSI_ROADMAP.md), [runtime contract](docs/LONG_PROJECT_RUNTIME.md), [evaluation protocol](docs/SELF_EVOLUTION_EVALUATION.md) and [performance measurements](docs/PERFORMANCE_STRUCTURES_2026-10-13.md).

</details>

The app uses the original <img src="clients/mobile/src/static/original-brand/logo-32.png" width="24" height="24" alt="DocThinker original dog logo" /> DocThinker dog identity. Original source bytes and all historical assets remain protected; app icon exports only resize and letterbox the original.

[Architecture and algorithms](docs/ARCHITECTURE.md) · [Insight Schema](docs/INSIGHT_RUNTIME_DESIGN.md) · [Source-grounded retrieval](docs/RESEARCH_GROUNDED_DESIGN.md) · [Memory plugin](docs/MEMORY_PLUGIN_GUIDE.md) · [Contributing](CONTRIBUTING.md)

<details>
<summary>Citation</summary>

```bibtex
@article{yang2026autothinkrag,
  title={AutothinkRAG: Complexity-Aware Control of Retrieval-Augmented Reasoning for Image-Text Interaction},
  author={Yang, Jiashu and Zhang, Chi and Wuerkaixi, Abudukelimu and Cheng, Xuxin and Liu, Cao and Zeng, Ke and Jia, Xu and Cai, Xunliang},
  journal={arXiv preprint arXiv:2603.05551},
  year={2026}
}
```

</details>

Current releases use the [PolyForm Shield License 1.0.0](LICENSE); previously MIT-licensed releases retain their original license.
