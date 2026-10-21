# Current UI showcase · 当前界面实录

[English homepage](../README.md) · [中文首页](../README.zh-CN.md) · [Historical recordings / 历史实录](HISTORY.md)

The current H5 workbench uses the **original DocThinker dog logo**, the memory observatory and an interactive source/event canvas in one client. The three recordings appear directly in both READMEs; the older four recordings remain unchanged in the history archive below.

新版统一使用**原狗狗 Logo**，记忆观测与可交互的来源／事件图在同一客户端衔接。首页直接展示下列三段新实录；原四段录屏移入历史区，不再冒充新版。

## Current recordings · 新版三段实录

### 01 · A request that stays yours / 保留你的思路

![Current workspace with the original dog logo: preserve a draft and append a project continuation recipe](assets/ui-v2-workspace.gif)

**Interaction:** keep an existing draft → open **Project tools → Continue project → Append to draft**. The original draft is retained; nothing is sent.

**实际操作：**保留草稿 → 打开**项目工具 → 继续项目 → 追加到草稿**。不自动提问、执行任务或生成答案。

### 02 · Inspect the memory / 看清记忆的层次与来历

![Current memory observatory: layers, source connections, explicit events and existing confirmed-memory versions](assets/ui-v2-memory-observatory.gif)

**Interaction:** switch between **layers, sources and events** → open an existing confirmed memory's version history. The clip inspects existing fixture records rather than silently changing them.

**实际操作：**切换**层级、来源、事件**面板 → 打开既存确认记忆的**版本与来历**。展示当前会话真实存储的合成记录，不是自动学会偏好或编造记忆增长。

Stored totals and the rechecked return window are separate. Unknown metrics are not zero; more versions or connections do not establish correctness. Explicit event relations retain their unverified status.

存储总数与本次重新核对的返回窗口分开；未知不填零，版本和连接变多也不等于知识更正确，显式事件关联不等于因果已验证。

### 03 · Navigate the sources / 沿图定位原文

![Current source and event canvas: pan and zoom, select a node, then open its synthetic source passage](assets/ui-v2-graph-canvas.gif)

**Interaction:** click a canvas node → focus → continuously pan and drag, with wheel zoom → open its accessible original passage. The canvas follows the current session's bounded projection; it does not manufacture missing nodes, causal links or source text.

**实际操作：**点选画布节点 → 聚焦 → 连续平移、滚轮缩放和拖动节点 → 查看可访问的合成原文。画布读取当前会话的有界投影，不自动补画缺失节点、因果链或原文。

## What these clips verify · 验证范围

- **Actual client and service, synthetic data.** All three clips use a real isolated local gateway/worker and temporary storage, with synthetic accounts and fixture materials, memories and events. The interface is not composited or replaced with a mockup. / **真实客户端与隔离服务、合成数据。** 不使用私人材料或真实凭据。
- **Zero external model calls.** These actions inspect, navigate and prepare a draft. A local test provider may exist in the acceptance runtime, but the recorded workflows do not ask it to generate answers. / **外部模型调用为零。** 这些操作不能作为模型质量、推理质量、token 或延迟的对比实验。
- **One original identity.** The current app uses the original dog artwork; installation icons only resize and letterbox it. The previous green-D recordings are historical, not the new brand. / **原狗狗品牌保持不变。** 历史素材原字节保留。
- **One canvas, two data views.** This recording shows the research source/event projection. Research-enabled invitations are limited to that view; private full-service users with research enabled can switch the **same H5 canvas** to **Knowledge relations · full service**, inspect the bounded GraphCore window and explicitly request a selected node's neighborhood. A legacy summary projection is not confirmed user memory; neither window is a whole-graph export. / **同一新版画布，两条数据入口。** 完整知识关系窗口需要私人完整服务，双视图切换还需开启研究能力；来源连接不等于引用支持结论，更不证明因果。
- **Browser interaction only.** These clips do not establish migrated camera-gesture recognition, continuous voice, native-phone experience, public production readiness, autonomous RSI or general causal reliability. / **不声称摄像头手势已迁移、连续语音或手机真机已验收，也不冒充生产与 RSI 验收。**

### Capture identity · 录制身份

- Source: [runtime commit `78f11ca`](https://github.com/Yang-Jiashu/Doc-thinker/commit/78f11ca3324cb0951109f37136b31f7e638c7734), built and recorded on 2026-10-21 (Asia/Shanghai). PWA build ID: `31e39111bf823dda1fcb`.
- Real isolated [memory/graph acceptance runtime](https://github.com/Yang-Jiashu/Doc-thinker/blob/78f11ca3324cb0951109f37136b31f7e638c7734/tests/manual_memory_graph_runtime_check.py), synthetic `preview` account, project `#00002`; strict loopback provider tripwire forbids every attempted model request. These workflows did not ask for an answer, alter the fixture records or call an external model.
- Headless Chromium, 1180 × 820 viewport, native 12 fps WebM. GIFs are 1000 × 695, sampled at 8 fps with 10 ms timing rounding and duplicate-frame compression; all frames were decoded after encoding. No interface compositing, DOM substitution or generated artwork. No audio; recording duration is not a response-speed benchmark.
- [New capture manifest](assets/ui-v2-manifest.json) binds source/build identity, original video hashes, shipped GIF hashes, encoded frame counts/durations and actual actions. Local original WebMs remain under `output/playwright/unified-ui-v2/`, outside the deployment bundle. Metadata checks establish byte consistency, not independent proof of reasoning quality.

## Historical recordings · 前版四段实录

The four files and their [original capture manifest](assets/current-ui-manifest.json) are retained unchanged. They show the previous green-D UI, **not** the new memory observatory or canvas. All four remain directly visible in [HISTORY](HISTORY.md).

原四段录屏与原 manifest 保持不变，在[历史页](HISTORY.md)直接可见；其旧绿色标识和界面不再代表新版。

| Recording | Actual interaction / 实际操作 | Runtime boundary / 运行边界 |
|---|---|---|
| [Workspace · 工作台](assets/current-ui-workspace.gif) | Prepare a customer-demo draft → choose “Continue project” → append the recipe without overwriting the draft. / 保留原问题、追加配方，不自动发送。 | Offline sample; no answer generation, task execution or model call. / 离线样例，无模型调用。 |
| [Memory · 确认记忆](assets/current-ui-memory.gif) | Save a response preference → confirm a correction → inspect versions 1 and 2. / 保存、修正并回看两个版本。 | Real isolated gateway and SQLite; synthetic user/data, **no model request**. This is explicit confirmation, not automatic preference learning. / 真实保存，不是自动学会偏好。 |
| [Graph · 联系与依据](assets/current-ui-graph.gif) | Ask how policy tools connect to domestic supply → locate the two-hop path → open a node and simulated passage → return. / 定位路径、查看节点与模拟原文、返回。 | Fixed offline sample: 6 nodes / 5 edges. Not live document ingestion, gesture recognition or proof of causation. / 固定样例，不代表现场建图或因果验证。 |
| [Tasks · 任务检查点](assets/current-ui-projects.gif) | Open a previously saved and approved personal-task plan → refresh progress → inspect completed steps and the explicit blocked state. / 打开事先保存并批准的计划、刷新进度、查看已完成与受阻步骤。 | Real gateway, worker and persistence; **local model double** cannot parse the independent review prompt. No final report is presented; an uncertain call is not automatically retried. This demonstrates failure handling, **not successful delivery** or a quality benchmark. / 本地替身不支持核对请求，如实展示异常保护，不冒充成功交付。 |

### Historical capture identity · 前版录制身份

- Source: [runtime commit `2970ef9`](https://github.com/Yang-Jiashu/Doc-thinker/commit/2970ef978f29137df73b1e5f03804cca4cc55c70); rebuilt with `npm run build:h5` on 2026-10-21 (Asia/Shanghai). PWA build ID: `cec9a1149a7e93be2b1b`.
- Chromium viewport: 1180 × 820. Native browser recordings: 12 fps; published GIFs: 1000 × 695, sampled at 8 fps with GIF timing rounding and duplicate-frame compression. No UI compositing, DOM substitution, edited screenshots or generated interface artwork. Clips have no audio; recording timing is **not** a latency benchmark.
- Workspace and graph use the labelled static demo. Memory and tasks use the existing isolated [runtime acceptance server](https://github.com/Yang-Jiashu/Doc-thinker/blob/2970ef978f29137df73b1e5f03804cca4cc55c70/tests/manual_project_task_runtime_check.py), with synthetic accounts, temporary storage and a loopback model double. No private documents, real credentials or external model calls are included.
- [Capture manifest](assets/current-ui-manifest.json): build identity, actual encoded GIF frame counts/durations, asset hashes, original recording hashes and recorded actions. Original WebM files are local capture artifacts, not bundled. Hashes establish consistency, not independent proof of product quality.
- The history archive displays these four GIFs directly; the current READMEs display the three newer clips above. Asset/link tests check the shipped files; invitation packaging verifies the same bytes inside the ZIP. Original logos, diagrams and recordings remain unchanged and visible as historical references.

## Try the interactions yourself · 自己试

### Workbench and graph, without a model key

```sh
cd clients/mobile
npm ci --ignore-scripts
npm run build:h5
npm run preview:h5
```

Open [localhost:5175](http://127.0.0.1:5175). In the assistant, enter your own draft and open **项目工具 → 继续项目 → 追加到草稿**. Nothing is sent until you choose to send it.

The default mode is explicitly labelled **offline sample data**. It can exercise drafting and the sample canvas, but it does not connect to the isolated storage shown in the new recordings, recreate their fixtures or evaluate answer quality.

默认离线模式能试草稿、导航和样例画布，**不会自动拥有新版录屏里的隔离资料与事件**。确认记忆的真实保存、版本与来源读取，需要连接已启用相应能力的服务。

Open **图谱** and use the default energy example. Enter `政策工具和国内供给增加有什么联系？`, locate the path, open `可再生能源投资`, then inspect the simulated source. This exercises navigation, not genuine document understanding.

### Reproduce the current memory and graph, without a vendor key

In a development checkout with the backend test dependencies installed, start the isolated acceptance server from the repository root:

```sh
PYTHON_DOTENV_DISABLED=1 python -m tests.manual_memory_graph_runtime_check --serve
```

Use the temporary loopback URL and synthetic login printed by the helper; stop with Ctrl-C when finished. It uses private temporary storage and a strict local provider tripwire, not production data or real API configuration. It runs its actual HTTP acceptance checks before serving. Never expose this test helper publicly.

Select **周五客户演示 · 合成项目** (`#00002`), the healthy fixture with one TXT, two confirmed memories, two explicit events and two recorded event relations. Inspect **Memory → layers / sources / events**, then open an existing memory's history. Use **Graph** to select a source-backed canvas node, pan, zoom and open its original passage. The separate first session intentionally exercises withdrawal and revision boundaries. Browser-generated IDs and timestamps may differ between runs.

这个 helper 会准备录屏所用的合成资料与事件。选择第二个“周五客户演示”会话测试新界面；第一个会话用于撤回和修订检查。源码全仓库中可运行，精简部署 ZIP 不含开发测试工具。它禁止模型调用；不要在这里发送真实问题，也不要把本机临时地址当成公开试用链接。

### Reproduce historical task recordings

The separate `python -m tests.manual_project_task_runtime_check --serve` helper reproduces the older task fixture. It uses local composition doubles but does not understand the UI's independent review prompt: that fixture stops with `provider_outcome_unknown`, retains completed steps and does not publish a final report. It is not the provider-forbidden current memory/graph fixture above. Do not retry an uncertain call as though it had never been sent; the historical limitation was not hidden by disabling review or changing production prompts.

For your own materials and real models, use the [private invitation deployment guide](../deploy/invite/README.md) and explicitly enable the bounded research core. Default invitations remain chat/history only. Neither the current nor historical clips claim that full desktop graph growth, camera gestures, continuous voice, native-phone experience, autonomous RSI or general causal reliability have been accepted. See the separate [release acceptance record](RELEASE_ACCEPTANCE_2026-10-20.md).
