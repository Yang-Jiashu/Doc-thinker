# Current UI showcase · 当前界面演示

These recordings show the **current 2.0 H5 client**, not redesigned mockups or the historical desktop UI. They are visible directly in both [English](../README.md) and [中文](../README.zh-CN.md) READMEs.

当前 2.0 的真实浏览器录屏。旧版桌面动图仍保留在首页的“历史 UI”区，不再用来代表当前界面。

| Recording | Actual interaction / 实际操作 | Runtime boundary / 运行边界 |
|---|---|---|
| [Workspace · 工作台](assets/current-ui-workspace.gif) | Prepare a customer-demo draft → choose “Continue project” → append the recipe without overwriting the draft. / 保留原问题、追加配方，不自动发送。 | Offline sample; no answer generation, task execution or model call. / 离线样例，无模型调用。 |
| [Memory · 确认记忆](assets/current-ui-memory.gif) | Save a response preference → confirm a correction → inspect versions 1 and 2. / 保存、修正并回看两个版本。 | Real isolated gateway and SQLite; synthetic user/data, **no model request**. This is explicit confirmation, not automatic preference learning. / 真实保存，不是自动学会偏好。 |
| [Graph · 联系与依据](assets/current-ui-graph.gif) | Ask how policy tools connect to domestic supply → locate the two-hop path → open a node and simulated passage → return. / 定位路径、查看节点与模拟原文、返回。 | Fixed offline sample: 6 nodes / 5 edges. Not live document ingestion, gesture recognition or proof of causation. / 固定样例，不代表现场建图或因果验证。 |
| [Tasks · 任务检查点](assets/current-ui-projects.gif) | Open a previously saved and approved personal-task plan → refresh progress → inspect completed steps and the explicit blocked state. / 打开事先保存并批准的计划、刷新进度、查看已完成与受阻步骤。 | Real gateway, worker and persistence; **local model double** cannot parse the independent review prompt. No final report is presented; an uncertain call is not automatically retried. This demonstrates failure handling, **not successful delivery** or a quality benchmark. / 本地替身不支持核对请求，如实展示异常保护，不冒充成功交付。 |

## What was recorded

- Source: [runtime commit `2970ef9`](https://github.com/Yang-Jiashu/Doc-thinker/commit/2970ef978f29137df73b1e5f03804cca4cc55c70); rebuilt with `npm run build:h5` on 2026-10-21 (Asia/Shanghai). PWA build ID: `cec9a1149a7e93be2b1b`.
- Chromium viewport: 1180 × 820. Native browser recordings: 12 fps; published GIFs: 1000 × 695, sampled at 8 fps with GIF timing rounding and duplicate-frame compression. No UI compositing, DOM substitution, edited screenshots or generated interface artwork. Clips have no audio; recording timing is **not** a latency benchmark.
- Workspace and graph use the labelled static demo. Memory and tasks use the existing isolated [runtime acceptance server](https://github.com/Yang-Jiashu/Doc-thinker/blob/2970ef978f29137df73b1e5f03804cca4cc55c70/tests/manual_project_task_runtime_check.py), with synthetic accounts, temporary storage and a loopback model double. No private documents, real credentials or external model calls are included.
- [Capture manifest](assets/current-ui-manifest.json): build identity, actual encoded GIF frame counts/durations, asset hashes, original recording hashes and recorded actions. Original WebM files are local capture artifacts, not bundled. Hashes establish consistency, not independent proof of product quality.
- Both READMEs display the four GIFs by default. Asset/link tests check the shipped files; invitation packaging verifies the same bytes inside the ZIP. Original logos, diagrams and recordings remain unchanged and visible as historical references.

## Try the interactions yourself · 自己试

### Workbench and graph, without a model key

```sh
cd clients/mobile
npm ci --ignore-scripts
npm run build:h5
npm run preview:h5
```

Open [localhost:5175](http://127.0.0.1:5175). In the assistant, enter your own draft and open **项目工具 → 继续项目 → 追加到草稿**. Nothing is sent until you choose to send it.

Open **图谱** and use the default energy example. Enter `政策工具和国内供给增加有什么联系？`, locate the path, open `可再生能源投资`, then inspect the simulated source. This exercises navigation, not genuine document understanding.

### Actual memory and task persistence, without a vendor key

In a development checkout with the backend test dependencies installed, start the isolated acceptance server from the repository root:

```sh
PYTHON_DOTENV_DISABLED=1 python -m tests.manual_project_task_runtime_check --serve
```

Use the temporary loopback URL and synthetic login printed by the helper; stop with Ctrl-C when finished. It uses private temporary storage and local model doubles, not the production service or real API configuration. It intentionally runs its offline acceptance checks before serving. Never expose this test helper publicly.

In **记忆**, create a confirmed response preference, correct it, then open **版本与来历**. In the assistant, create a new session, open **长任务**, choose **个人任务（无需上传文件）**, save a plan and approve only the displayed budget. Refresh to inspect its state; generated test text explicitly identifies itself as an offline response. The current helper understands composition requests but not the UI's independent review prompt: this branch stops with `provider_outcome_unknown`, retains completed steps and does not publish a final report. Do not retry an uncertain call as though it had never been sent. This fixture limitation was retained, not hidden by disabling review or changing production prompts.

For your own materials and real models, use the [private invitation deployment guide](../deploy/invite/README.md) and explicitly enable the bounded research core. These GIFs do not claim that full desktop graph growth, camera gestures, continuous voice, native-phone experience, autonomous RSI or general causal reliability have been accepted. See the separate [release acceptance record](RELEASE_ACCEPTANCE_2026-10-20.md).
