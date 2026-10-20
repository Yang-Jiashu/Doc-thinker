<div align="center">

<img src="logo.png" width="72" height="72" alt="DocThinker 原 Logo" />

# DocThinker 2.0

### 记得你，也接得住你的事。

**一个以长期记忆、可追溯证据和持续任务为核心的私人助手。**

不是每次从头聊起。让你确认过的偏好、做过的决定和积累的材料，在下一件事中接着发挥作用。

[English](README.md) · [开始体验](#开始体验) · [核心架构](docs/ARCHITECTURE.md) · [验收记录](docs/RELEASE_ACCEPTANCE_2026-10-20.md)

[![CI](https://github.com/Yang-Jiashu/Doc-thinker/actions/workflows/ci.yml/badge.svg)](https://github.com/Yang-Jiashu/Doc-thinker/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](pyproject.toml)
[![License](https://img.shields.io/badge/License-PolyForm_Shield-orange)](LICENSE)
[![Paper](https://img.shields.io/badge/arXiv-2603.05551-b31b1b)](https://arxiv.org/abs/2603.05551)

</div>

[![长期记忆、可追溯证据和持续任务如何连接](docs/assets/docthinker-v2-overview.svg)](docs/ARCHITECTURE.md)

*概念示意，非实测截图。联系保留来源与版本；虚线表示待核实，不把关联当成因果。*

## 先看它如何工作

<p align="center">
  <img src="docs/assets/usecase_chat.gif" width="48%" alt="原文档问答 Demo：对话中使用材料" />
  <img src="docs/assets/usecase_kg.gif" width="48%" alt="原交互图谱 Demo：查看知识之间的联系" />
</p>

*原始演示，素材保持不变。当前移动端与邀请内测的实际能力见下方入口；原 Demo 不代表所有部署模式都开放图谱。*

### 01　记住，而不是堆上下文

保存你**明确确认**的偏好、约束与决定，按任务检索保留期内的记忆；更正、来源、有效期和历史版本一起保留。无需把全部聊天塞进模型，也不把一次随口表达自动当成永久偏好。

### 02　给答案，也给检查它的线索

原文片段、事件关系与确认记忆进入同一条证据链。回答能定位来源；材料更正或撤回，会让相关派生内容失效。类比、演绎、归纳在后台辅助找材料与组织论证，用户照常聊天和写作。

### 03　事情没做完，下次接着做

项目保存总目标，阶段保存计划与检查点。你批准本批工作后才调用模型；重启后可明确续做，保留有效成果，重做受影响分支。草稿、核对和报告分开，**执行完成不冒充事实正确或目标达成**。

工作协作、生活安排、个人写作、论文研究共用这套底座。研究是应用场景，不是产品边界。

## 开始体验

### 先看交互：不需要 Key

需要 Node.js 22 与 Python 3。克隆仓库，构建并在本机预览移动网页：

```bash
git clone https://github.com/Yang-Jiashu/Doc-thinker.git
cd Doc-thinker/clients/mobile
npm ci --ignore-scripts
npm run build:h5
npm run preview:h5
```

打开 **[本机体验入口](http://127.0.0.1:5175/)**。默认离线演示可以体验导航、资料、记忆与图谱交互；这是本地样例，**不调用真实模型，不验证回答质量**。同一工程可构建微信游客和 App 资源，见[移动端指南](docs/MOBILE_APP.md)。

### 再用自己的材料：私密邀请内测

给少量受邀用户使用网页，试用者无需安装开发环境，也无需填写模型 Key。部署者需要 Docker Compose、允许该用途的模型服务，以及对外访问时的可信 HTTPS。

[部署与创建受邀账号 →](deploy/invite/README.md)　[上线前检查 →](docs/INVITE_WEB_BETA.md)

默认只开放聊天与历史；文档和记忆需要显式启用研究核心及上传权限。邀请模式**不开放完整图谱、语音或自动学习**，不能用离线演示冒充这些能力已上线。

下载的是邀请源码 ZIP？请按部署包指南操作；下方全仓库开发命令不适用于精简包。

<details>
<summary>开发者：启动完整图谱与桌面工作台</summary>

在仓库根目录使用 Python 3.11。完整解析依赖较大，先测试 TXT，再配置 PDF／图像处理。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all]"
cp -n env.example .env
```

按 `env.example` 配置实际可用的回答、嵌入与可选视觉模型，不要提交密钥。分别在两个终端运行：

```bash
# 终端 1
python -m uvicorn docthinker.server.app:app --host 127.0.0.1 --port 8000

# 终端 2
UI_HOST=127.0.0.1 UI_PORT=5001 python run_ui.py
```

[对话](http://127.0.0.1:5001/query) · [图谱与记忆](http://127.0.0.1:5001/knowledge-graph) · [API](http://127.0.0.1:8000/docs)

新建会话 → 上传 TXT → 等待处理 → 提问 → 查看回答依据。完整图谱、移动私人连接与邀请服务是不同运行模式，不能混用配置或权限。

</details>

## 骨架与智能，分工清楚

**Harness** 管执行、身份、范围、预算和恢复。**Insight Schema** 将任务相关的知识与证据义务注入检索、分析和交付；不是额外的聊天模式，也不意味着每一步都要再问一次模型。

程序负责局部图搜索、去重、选证、版本检查与调度；模型负责语义理解、候选解释和表达。新关联保留假设身份，证据不够就保持未知。记忆可以随存储扩容，单次检索、上下文和批准调用仍有明确上限。

[当前架构与算法 →](docs/ARCHITECTURE.md)　[长期任务机制 →](docs/LONG_PROJECT_RUNTIME.md)　[Insight Schema →](docs/INSIGHT_RUNTIME_DESIGN.md)

### 有哪些可复核的结果？

代码验收快照 [main `0cda681`](https://github.com/Yang-Jiashu/Doc-thinker/commit/0cda681) 通过 [完整 CI](https://github.com/Yang-Jiashu/Doc-thinker/actions/runs/39445691276) 与[产品契约检查](https://github.com/Yang-Jiashu/Doc-thinker/actions/runs/39445691569)：

- Python 3.10／3.11：**各 7,140 项通过、9 项可选跳过**；另有 13 个 subtest 通过。
- 客户端 **999 项通过**；三个真实浏览器视口通过；H5、微信游客及 App 资源构建通过。
- **30 阶段、3 次重启**，以及真实网关／双 worker 的续做、报告、来源撤回与租户隔离检查通过。模型为本地替身，不是回答质量测试。

真实 `step-5-preview` 的 8 次调用属于先前冻结快照，结果为 **accepted_with_caveats**，仍有转述措辞风险；后续兼容修复没有冒充新模型跑分。历史失败、token 与耗时都保留在[验收记录](docs/RELEASE_ACCEPTANCE_2026-10-20.md)。

**当前是可部署、可邀请测试的工程版本，不是已完成商用验收的成品。** 公开 HTTPS、手机真机、签名安装包、长期客户使用与生产容量需独立验收；来源链成立不等于因果判断已被证明。模型服务商和运营者可能处理所提交内容，不宣称端到端加密或无人批准的自主执行。

RSI 是研究方向：从失败中提出策略改动，经独立评测、审核采纳和可回滚验证后再改进。现在的记忆增长与候选关联**不是已实现 RSI 的证据**。[路线与验收条件 →](docs/RSI_ROADMAP.md)

<details>
<summary>原 Logo、完整 Demo 与框架图集</summary>

<img src="docs/assets/banner.png" width="900" alt="DocThinker 原横幅" />
<img src="docs/assets/agentic_memory_architecture.png" width="900" alt="原私人助手分层记忆框架" />
<img src="docs/assets/memory_kg_observability_demo.gif" width="900" alt="原记忆与图谱可观测性 Demo" />
<img src="docs/assets/gpu_knowledge_graph_demo.gif" width="900" alt="原 GPU 图谱 Demo" />
<img src="docs/assets/ui_memory_editor_screenshot.png" width="900" alt="原可编辑记忆界面" />
<img src="docs/assets/pipeline.png" width="900" alt="原文档处理流程" />
<img src="docs/assets/lifecycle.png" width="900" alt="原记忆生命周期" />
<img src="docs/assets/multi_agent_evolution.png" width="900" alt="原多智能体演进框架" />
<img src="docs/assets/sparql_cot.png" width="900" alt="原图查询辅助推理示意" />

手机端继续使用 <img src="clients/mobile/src/static/brand/logo-32.png" width="24" height="24" alt="证据之窗手机端标识" /> 证据之窗品牌；此处没有替换原素材。GitHub README 支持 GIF 播放，不运行交互脚本；实际图谱、手势与语音请进入对应应用测试。

</details>

## 深入了解

[记忆集成](docs/MEMORY_PLUGIN_GUIDE.md) · [图谱与手势](docs/GESTURE_INTERACTION.md) · [语音](docs/VOICE_MODELS.md) · [模型配置](docs/MODEL_CONFIGURATION.md) · [评测方法](docs/SELF_EVOLUTION_EVALUATION.md) · [贡献](CONTRIBUTING.md) · [许可](LICENSE)

```bash
python -m pip install -e ".[all,test]"
python -m pytest tests/ -q --ignore=tests/debug_db.py
python scripts/check_release_preservation.py
```
