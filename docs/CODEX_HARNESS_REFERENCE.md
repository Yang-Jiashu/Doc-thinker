# Codex Harness 源码参考与接入设计

DocThinker 的 Insight Schema 建立在 Harness 上。Harness 管理一次工作的状态、执行、预算和终止，Insight Runtime 在规定的阶段提出动作与上下文建议。具体检索、模型调用、工具执行和结果保存仍由所选 Harness 执行路径负责。

本次参考 [openai/codex](https://github.com/openai/codex/tree/a9b0815167c27ab13719dd12d13b63e02b70d221)，固定提交 `a9b0815167c27ab13719dd12d13b63e02b70d221`，提交日期为 2026-10-11。下面区分公开接入接口、内部实现借鉴及 DocThinker 的待实现方案。源码核对不代表已经集成或完成运行验证。

## 接入方式

[Codex 官方架构说明](https://developers.openai.com/blog/codex-as-a-platform) 将产品界面、业务数据和工具交给应用负责，由 Codex Harness 提供模型与工具循环、会话状态和执行事件。开放组件包括 CLI、SDK 和 App Server；模型访问和托管服务另行提供。

DocThinker 首先统一现有 Python 执行路径，在这一 Harness 中实现洞见策略接口，继续使用现有模型适配层。后续可以增加 Codex App Server 执行适配器，在同一产品界面中选择执行后端。一次运行只选择一条模型与工具循环，Insight Runtime 不再创建一条平行循环。

这一顺序保留模型选择空间。在本次固定源码中，[WireApi](https://github.com/openai/codex/blob/a9b0815167c27ab13719dd12d13b63e02b70d221/codex-rs/model-provider-info/src/lib.rs#L100) 只接受 Responses，明确拒绝 `wire_api = "chat"`。此前提供的 Step 接入信息是 Chat Completions 端点；直接交给 Codex 是否可用，需要验证其 Responses 兼容路径，不能仅更换 base URL 就宣称适配成功。

这里的“兼容”首先指使用同一任务、事件和能力契约。与 Codex App Server 的线协议兼容由专门适配器实现，不能因为名称相近就认为现有 HTTP 接口已经兼容。

## 值得借鉴的源码机制

| 机制 | 固定版本源码 | 对 DocThinker 的具体作用 |
|---|---|---|
| Thread、Turn、Item 与状态事件 | [协议事件](https://github.com/openai/codex/blob/a9b0815167c27ab13719dd12d13b63e02b70d221/codex-rs/app-server-protocol/src/protocol/common.rs#L1979)、[Turn 状态](https://github.com/openai/codex/blob/a9b0815167c27ab13719dd12d13b63e02b70d221/codex-rs/app-server-protocol/src/protocol/v2/turn.rs#L33) | 检索、补证、洞见应用、工具和产物成为可观察步骤；界面由真实执行事件更新 |
| 追加要求校验活跃轮次 | [TurnSteerParams](https://github.com/openai/codex/blob/a9b0815167c27ab13719dd12d13b63e02b70d221/codex-rs/app-server-protocol/src/protocol/v2/turn.rs#L311) | 手机和桌面同时操作时，以预期 run/turn ID 和目标版本防止串轮 |
| 每次模型请求固定上下文快照 | [StepContext](https://github.com/openai/codex/blob/a9b0815167c27ab13719dd12d13b63e02b70d221/codex-rs/core/src/session/step_context.rs#L23) | 一步内固定洞见、来源、模型设置和工具能力；新变化在下一执行边界生效 |
| 按稳定标识生成上下文差异 | [WorldState section 与 diff](https://github.com/openai/codex/blob/a9b0815167c27ab13719dd12d13b63e02b70d221/codex-rs/core/src/context/world_state/mod.rs#L312)、[历史留存检查](https://github.com/openai/codex/blob/a9b0815167c27ab13719dd12d13b63e02b70d221/codex-rs/core/src/context/world_state/mod.rs#L490) | 按 Schema ID、版本、阶段和内容指纹管理活跃洞见，只注入必要变化；被移出当前上下文后可重新装入 |
| 压缩后重新建立有效上下文 | [build_compaction_replacement_history](https://github.com/openai/codex/blob/a9b0815167c27ab13719dd12d13b63e02b70d221/codex-rs/core/src/compact.rs#L79) | 长任务压缩历史后，重新装入当前目标、约束、有效洞见和出处，建立新基线 |
| 工具前后存在独立生命周期处理 | [PreToolUse](https://github.com/openai/codex/blob/a9b0815167c27ab13719dd12d13b63e02b70d221/codex-rs/hooks/src/events/pre_tool_use.rs#L24)、[PostToolUse](https://github.com/openai/codex/blob/a9b0815167c27ab13719dd12d13b63e02b70d221/codex-rs/hooks/src/events/post_tool_use.rs#L24) | 将检索建议、结果核查、后续待办接在实际动作前后，记录请求与执行结果 |
| 工具并发与取消由核心调度 | [parallel.rs](https://github.com/openai/codex/blob/a9b0815167c27ab13719dd12d13b63e02b70d221/codex-rs/core/src/tools/parallel.rs#L204) | 可并行的读取任务共享预算；具有写入或顺序依赖的动作受控执行，取消有明确结果 |

WorldState 的差异更新是上下文管理机制。内容指纹表示是否变化，不证明知识正确；快照 patch 的删除也不表示此前已发送的旧文本消失。来源撤回后，DocThinker 必须重新构建后续可发送上下文，并重新检查派生洞见。

固定前缀或缓存命中的文本仍占用上下文窗口。Codex 的 [context window 计量](https://github.com/openai/codex/blob/a9b0815167c27ab13719dd12d13b63e02b70d221/codex-rs/core/src/session/context_window.rs#L58) 区分部分计量用途，但保留完整上下文上限。DocThinker 应同时约束证据、洞见、历史、生成输出和后续动作，不能只限制新增洞见的文本长度。

## Harness 内的洞见策略接口

以下是 DocThinker 拟议的内部契约，不是 Codex App Server 的现成 API：

```text
on_stage(stage, readonly_snapshot) -> Proposal

Proposal:
  schema_versions
  premise_checks
  proposed_actions
  context_changes
  estimated_cost
  expected_checks
```

宿主先验证 Proposal，再执行其中允许的操作，并记录实际成本和结果。插件的筛选和编译保持有界；需要补证、模型判断或写入时提出动作申请。预算、权限、取消、重试及提交由 Harness 决定。

阶段接口覆盖任务输入、检索前、证据就绪、工具结果、交付前和结果反馈。插件看到的是当前阶段的只读快照；输出附带任务版本和前提绑定，过时建议在执行前重新检查。

上下文分为固定规则、当前任务状态、本阶段洞见和有出处的材料。固定规则由应用提供；用户资料和模型提炼内容保持数据身份。当前证据的事实状态、候选解释和个人偏好分别保留。

## 公开接口和内部机制的边界

[App Server 文档](https://learn.chatgpt.com/docs/app-server) 提供 `thread/start`、`thread/resume`、`turn/start`、`turn/steer`、`turn/interrupt` 和执行事件。适配器负责 ID 映射、状态映射、错误和能力检查。`dynamicTools` 等实验字段需要单独确认版本支持。

[Hooks 文档](https://learn.chatgpt.com/docs/hooks) 提供 UserPromptSubmit、PreToolUse、PostToolUse、Stop 等事件；部分同步钩子可追加上下文或对支持的工具路径施加控制。后台钩子不能控制已经触发它的操作；托管工具等路径不保证经过本地工具钩子。

公开 Hooks 可作为 Codex 后端的部分接入点，MCP 或应用工具可提供材料、记忆、Graph 和洞见查询。对任何依赖 `additionalContext` 的接入，应由可信适配层输出受控内容：部分钩子的该字段会作为 developer context，不能将不可信原文直接升格为运行指令。

内部 StepContext、WorldState 及 extension contributor 适合借鉴实现。它们不意味着外部客户端可以任意注册“每次模型采样前”的 Rust 回调。通过公开接口不能保证的能力须明确标为不支持或降级，不能只添加 SKILL.md 就宣称全过程注入已经生效。

## 恢复语义

恢复历史、重新订阅正在执行的任务、继续模型推理、恢复外部工具进程分别处理。公开 `thread/resume` 不能单独证明任意任务都能无损断点续跑。

本次源码确有内部恢复机制：[RecoverTurnRequest](https://github.com/openai/codex/blob/a9b0815167c27ab13719dd12d13b63e02b70d221/codex-rs/protocol/src/turn_input.rs#L96) 用原 turn ID 重新开始采样；[turn_suspension](https://github.com/openai/codex/blob/a9b0815167c27ab13719dd12d13b63e02b70d221/codex-rs/core/src/session/turn_suspension.rs#L96) 说明部分 pending input 和交互等待状态尚未持久化。这些实现有适用条件，接入时要分别验证。

DocThinker 首先保留现有请求指纹、结果查询与 unknown 状态，之后补阶段 checkpoint。已知完成的动作不能重发；确定失败的动作按策略处理；结果未知的外部调用先核对状态。线程等待超时不等于线程停止，界面取消状态也需反映实际执行结果。

## 现有项目需要先收敛的部分

| 位置 | 当前职责 | 调整方向 |
|---|---|---|
| `docthinker/harness.py` | QueryHarness 的 prepare、图谱策略与回答后处理 | 保留策略能力，增加统一运行控制职责或对应适配接口 |
| `server/routers/query.py` | 普通与流式查询入口；研究路径直接交给 ResearchService | HTTP 层负责请求和事件传输，执行交给同一运行所有者 |
| `server/routers/mobile.py` | 单次任务、断线查询和请求账本衔接 | 接入统一运行控制器，保留旧端行为适配 |
| `server/research_service.py` | 研究检索、辅助、生成、溯源与回答保存 | 提供执行步骤；步骤边界调用 Insight 策略 |
| `server/request_journal.py` | running、completed、unknown 与结果回放 | 通过存储迁移扩展运行事件和 checkpoint；结果回放不冒充任务续跑 |
| `server/workbench_store.py` | 用户 brief 和答案书签 | 绑定目标版本、运行和产物，保持用户工作台职责 |
| `server/research_scheduler.py` | 有界本地辅助通道 | 复用并发和超时规则，所有工作归入运行预算 |

运行事件增加 `run_id`、稳定会话标识、`seq`、阶段、事件类型和内容引用。Web 使用流式传输，手机与小程序可按游标读取同一事件记录；旧客户端通过适配层保持现有 status/meta/chunk 接口。

统一执行所有者可以在现有 Harness 模块内完成职责迁移。Insight 仍只新增 Schema 与 Runtime 两个核心模块，避免为这次设计创建多套服务。

## 落地和验证顺序

1. 收敛运行所有者与事件接口，保持旧 API 和模型适配可用。
2. 加入阶段快照和 Insight Proposal，先让检索建议及证据核查可追踪。
3. 加入活跃洞见的差异编译、来源撤回后的上下文重建和整体预算。
4. 通过既定真实模型对照验证质量、成本与时延，再接候选经验修订。
5. 若采用 Codex 后端，按固定版本验证 App Server、工具、Hooks 和模型协议能力，并运行相同产品契约测试。

关键验证包括：重复提交只执行一次、用户纠正不会串轮、取消与未知结果正确呈现、超时工作不积压、压缩后目标和有效洞见仍可恢复、撤回来源不会在后续输入中复活，以及手机和桌面看到相同的运行状态。

源码对照后，第一版已实现本项目的共享运行核心和研究服务的三阶段洞见接入：普通/流式/移动请求共享生命周期，方法提案由宿主执行并记录，模型适配接口保持不变。旧 Graph 路径共用执行核心，但尚未适配三阶段 Insight 的证据契约。运行记录为有界进程内事件，尚无持久续跑或跨端事件游标。

未将 Codex 代码并入运行时，未配置用户的 Codex Hooks，也未执行付费模型调用。实现状态、离线测试边界和真实质量评测分组见 [Insight Schema 动态注入设计](INSIGHT_RUNTIME_DESIGN.md)。
