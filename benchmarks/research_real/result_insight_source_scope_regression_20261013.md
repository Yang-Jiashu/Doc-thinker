# 章节与回指归属复验 V7：旧错归未复现，范围与追溯仍有缺口

2026-10-13。两次真实 `step-5-preview` 请求均完整交付，构成一组有效配对。补查来源不再把 TriviaQA 续文与 FEVER 标题混进同一节选；本轮也没有再次出现这条错归属或明确的“FEVER SOTA”结论。但 AUTO 仍将仅说明任务 / 方法的材料说成 FEVER 上效果优秀、已被证明；OFF 的逗号组合引用没有被后端解析为来源引用。**本轮章节边界、冻结输入和交付回归通过；完整答案质量与端到端逐主张追溯仍未全部通过。**

## 协议、环境与限制

按新调用前登记的 `plan_insight_source_scope_regression_20261012.md` 执行，题目仍为未改写的 `smoke_assistant_report`，使用固定公开 RAG、MemGPT PDF。只比较动态 Insight OFF / AUTO，各一次；普通后台辅助和静态方法卡两臂都开启，不是整个后台辅助开 / 关。

这是已经看过 V5 / V6 结果的非盲 AI 开发回归，不是独立留出集、专家评审、跨模型试验或产品认证。本轮与 V6 的源码及抽样不同，不能将两轮答案的变化直接归因于本次修复，更不能选择好结果替换旧失败。

- 冻结准备：`dist/insight-scope-prepare-v7-20261013/`，两臂 `prepared_only`，0 provider。
- 正式运行：`dist/insight-scope-report-v7-20261013/`。
- 准备、开始、结束、最终检查的代码指纹均为 `bfbd5f34732e985f0452f081e82732fd3d61beaef7b8b5e25d50309c92e13a4c`。`protocol.json` 中 `fingerprints_match` 与 `frozen_run_and_actual_inputs_match` 均为 true。
- 实际 prompt digest 与冻结准备相同。另独立核对两份交付 `result.sources`，18 个节选的标签、文件、页、正文和字符范围与准备一致；本轮只读 SQLite 原文的 `text[start:end]` 全部逐字相等。
- 最多两次调用，30,000 上报 tokens 观察后软停止，不是供应商硬费用限制；不重试、不续写。两次均 HTTP 200、`finish_reason=stop`、status completed、交付门禁 passed；无未知 usage、无重试，总上报 16,640 tokens。
- 请求和返回模型均为 `step-5-preview`，Step Plan，生产输出上限 8192，显式温度 0.1、共享适配器 low reasoning、非流式。运行使用按哈希锁安装的新受限部署 Python 3.11 环境，没有安装可选 neural runtime；两臂 semantic 都关闭。

审查仅使用正式 `result.answer`、`result.sources` 和安全协议 / 用量 / 检查元数据，没有读取或保存供应商内部推理正文。交付门禁仅检查来源可用与正式答案，不证明语义正确。

## 输入修复：保留基础证据，隔开两条补查的章节

两臂基础 S1–S8 与 V6 完全相同，MemGPT recall / archival 定义、外部存储操作、RAG 双记忆架构、CuratedTrec 先行对象与 DMR 评价说明没有被挤掉。AUTO 两条新增节选发生以下变化：

| 来源 | 本轮实际范围 | 与 V6 的差别 |
|---|---|---|
| S9，RAG p5 | 字符 219–937，`3.4 Fact Verification`：三类 claim 标签、Wikipedia 检索与分类机制 | 不再接上下一节 QA Results / SOTA 句。旧 S9 的不使用检索证据监督、两类 / 三类对照、TQA-comparable split 结果本次未入选。 |
| S10，RAG p18 | 字符 312–1026，`E Further Details on FEVER`：再生成 claim、分类与跨文档边缘化；证据句抽取子任务未做及 Wikipedia dump 差异 | 不再把前面的 TriviaQA official Wiki / open-domain test-set 比较带入 FEVER 标签；保留了本节自身的未做范围。 |

这证明指定补查不再跨显式章节，**不等于自动识别了所有语义对象或所有来源都只含一种任务**：基础 S4 仍同时包含 CuratedTrec 预处理和随后 TriviaQA split，来源内的对象仍需要逐主张绑定。

最终本题来源没有需触发本次新规则的开头回指，因此没有 `source_antecedent_unverified` 提醒，符合实际输入；不能将“无提醒”解释成语义已验证。独立通用回归已另测有回指及最终预算丢弃 / 保留来源时的提示一致性。

| 臂 | 来源数 | 原文 UTF-8 字节 | 输入记录字节 | 静态卡 | 零调用准备毫秒 |
|---|---:|---:|---:|---:|---:|
| OFF | 8 | 9,181 | 20,772 | 2 | 548.04 |
| AUTO | 10 | 10,635 | 22,366 | 1 | 541.57 |

未放宽 12,000 原文字节、24,576 提示字节或来源数量。没有整条参考表来源，也没有严格整条的通用背景 / 应用展望；但 S1、S7 仍混有动机 / 背景，不是零噪声证明。DMR 的完整会话 / 基线细节等仍不在材料中。

## 正式回答：哪些真正用对了

- **RAG 架构归属正确。** OFF 用 S7 解释 seq2seq 参数记忆、Wikipedia dense index 以及 sequence / token 两种形式；AUTO 用 S2 / S7 支持双记忆架构。V5 中仅挂 S2 而缺 S7 的具体架构错引未复现。
- **OFF 使用了新恢复的记忆分工。** 它说明 recall 为完整事件历史、archival 为可读写溢出存储，列出时间 / 文本 / 向量检索和 LLM 自主函数调用；这些内容在 S5 / S8 中确有支持。两臂都列出 main context 的 instructions / FIFO / working context。
- **AUTO 对 MemGPT 解释不完整。** 它虽引用 S5 / S8，却主要列出 recall / archival 名称和自主调用，没解释两类存储分别保存什么、如何分工。两臂也没有完整解释外部内容并非直接可见、须搬入主上下文及分页约束。入选定义不能替代答案是否真正说明定义。
- **直接比较和私人部署范围较谨慎。** 两臂都承认没有共同基准上的 RAG-vs-MemGPT 直接比较，把私人研究助手迁移与已验证任务分开，没有声称隐私、隔离、合规或长时可靠性已验证。
- **旧错误的精确状态。** TriviaQA→FEVER 错归属、明确 FEVER SOTA、把摘要 open problems 当全文最终未解决，以及 static index 断言本轮均未复现。但两臂没有展开 CuratedTrec 预处理，不能把“未出现旧错绑”算成再次正确解释该预处理的主动验证。
- **没有再次用单系统消融冒充跨系统比较。** OFF 明确提出同一私有语料、多会话文档 QA 的 head-to-head；AUTO 只给 shared benchmark 等高层下一步，没有重现 V6 那张对照错位表格。不过这也不是已提供完整可复现实验：采样、模型与预算控制、阈值和统计计划仍未规定。

## 尚未通过的语义质量项

### AUTO 将“涉及 / 评测这个任务”升级成“效果优秀 / 已被证明”

AUTO 表格先写 `Also evaluated on FEVER claim classification [S9][S10]`，随后迁移段改成 `excelling at open-domain QA and FEVER [S2][S9]`，结尾又说 `RAG is proven for knowledge-intensive QA and FEVER`。

S2 支持具体 QA 结果；本轮 S9 / S10 支持 FEVER 的任务、方法、分类流程和没有完成证据句抽取的局限，并未给 FEVER 效果数字或优于基线的结果。因而它虽不再说 FEVER SOTA，却仍把“做了分类任务”扩大为“表现已证明优秀”。最后的缺隐私 / 长期助手声明不能抵消这个任务内的夸大。

AUTO 同时没有在结论中保留 S10 的重要范围：研究的是 claim classification，不是 FEVER 的 Wikipedia 支持句抽取子任务。只有泛称 FEVER，读者仍可能误解成完整事实核查能力。不能用一个正确章节标签包办所有任务范围。

### 其他需要收窄的表述

- OFF 把 S2 的 `recent approaches that use specialised pre-training objectives` 缩成 recent approaches，丢掉了所比较方法的限定；没有新增定量错误，但不应扩大对照范围。
- AUTO 的 MemGPT 表格同时提 document analysis / multi-session chat，并在共享指标格写 ROUGE-L / LLM-judge consistency。实际 S6 只给 DMR 的评价说明，应该明确这些是 DMR 指标，避免把它们读成两域通用评价。
- OFF 仍出现 unbounded context，但表述是在转述 S3 的架构目标，后文明确要求长期保留实验；本轮未再把无上限历史的质量或资源消耗写成实测保证。它不是对无限记忆正确性的验证。

开发判断：报告任务得到基本完成，旧的对象混排被收敛，但 AUTO 的性能泛化与定义解释不足仍需要返工。不能宣称更完整的来源一定产生更完整、更可信的答案。

## 实际追溯链出现的格式缺口

V7 OFF 有八处逗号组合引用，如 `[S5, S8]`、`[S2, S7]`。当前单标签解析没有把这些形式识别为来源引用：实际 `citation_validation` 仍 `valid=true`，`used_labels` 为 S7 / S2 / S3 / S1 / S5 / S6，漏掉仅在组合形式中出现的 S8；同一检查明确仅承诺 `citation_syntax_only`，不能当完整引用验收。

具体例子是 recall / archival 分工加三类搜索的那句话：原文 S5 / S8 支持其内容，读者也能看见 `[S5, S8]`，但实际 `claim_audit` 给 `source_refs=[]` 和 `uncited_statement`。这不是模型完全没有来源，而是**正确内容到机器可定位来源的连接在格式解析时断了**。本轮没有改写正式答案、自动重标引用或为其补造支持；它是需要单独修复的产品追溯问题。

本轮也暴露自动检查额度利用问题：AUTO 的十二段检查额度中，前十一段是标题碎片、范围声明、表头或分隔符，只剩一段实质 RAG 内容进入该窗口。OFF 则在十二段里出现被组合引用漏识别的记忆分工句。两份 audit 都 `review_needed`、`entailment_verified=false`，各有十二个 issue，另有 OFF 25 / AUTO 33 段未检查。这些边界被披露，不能据此称系统伪称全文核验；但它说明**有上限仍需要把检查用在实质主张上，issue 数也不是错误率**。

以上由实际保存的检查对象核实，不是新增浏览器点击、真实手机操作或新模型调用。

## 用量与速度

| 臂 | 输入 tokens | 输出 tokens | 总 tokens | 缓存 tokens | 模型秒 | 完整流程秒 | 模型前处理毫秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| OFF | 4,690 | 3,539 | 8,229 | 0 | 44.421 | 45.007 | 551.680 |
| AUTO | 5,042 | 3,369 | 8,411 | 768 | 47.295 | 47.910 | 593.189 |

AUTO 本次输入多 352、输出少 170、总 tokens 多 182（2.21%），完整流程慢 2.903 秒；总计 16,640 tokens。缓存属于输入子集，不重复相加。正式可见正文 OFF 4,247 字符 / 579 空白分词，AUTO 4,386 / 576；completion tokens 不是正文长度计数，reasoning_tokens 上报 0 不能证明没有内部推理。

两个普通辅助 scheduler 均 completed，每臂仍只有一次回答模型调用。完整流程不含共用的约 998 毫秒摄取，也不是首 token 或高并发延迟。AUTO 有缓存，供应商负载和采样不同，每臂只有一次；不能把这次差异解释成稳健算法成本，也不能拿 V6 的较慢 OFF 与 V7 的较快 OFF 当修复提速证明。两轮 OFF prompt digest 实际相同，仍产生不同长度和用量的回答，恰好说明单次采样的局限。

## 验收结论

**通过：** 本次补查的显式章节隔离、基础关键证据保留、冻结输入 / 实际来源一致、原文坐标、预算、真实完整交付和可核对的成本记录。指定旧跨任务错归没有复现。

**未通过：** 逐任务的性能范围约束、AUTO 对恢复定义的充分使用、组合引用的机器追溯，以及实质主张优先的有限自动检查。报告还不能交付为完全无需核查 / 返工的产品结果。

后续应分别修来源 / 结论范围和确定性的引用解析 / 检查窗口，不将二者混成“换一段 prompt 就全解决”。本报告只记录本轮事实，V6 及此前结果保持原样；新的修复或调用必须另有版本和记录。
