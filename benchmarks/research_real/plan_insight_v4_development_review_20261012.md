# Insight V4 开发复验：先检查输入，再审查真实交付

2026-10-12。本文在新一轮模型答案生成前固定审查口径。三题沿用开发 smoke 和相同公开 PDF，不是盲测、专家评审或未见题；不得据此宣布统计显著或普遍质量优势。

## 请求协议

- fixture：`insight_development_smoke_v2.json`；同一三题、同一 PDF SHA256，不修改题目适配新召回。
- 两臂：`--contrast insight`，静态知识卡与普通后台辅助均开启；只切换动态 Insight OFF/AUTO。方法卡可能因共享预算整卡省略，以真实输入为准。
- 模型 `step-5-preview`，官方 Step Plan 端点；温度显式 0.1，semantic 关闭，生产任务 completion 预算应为 8192。共享生产参数构造器，不增加评测专用模型提示。
- 最多六次真实请求、每题一组；失败、空正式回答、未完成生成或交付失败立即停，不自动重试，不补调模型修答案。
- 新付费目录计划为 `dist/insight-deploy-smoke-v4-20261012/`；旧 V1–V3 记录不改写。代码冻结前不发送。
- 正式执行期间不修改被 fingerprint 覆盖的生产或评测代码；请求前后复核版本指纹。若与准备时不同，重新检查输入，不拿旧 prepared 包代表新调用。
- 计划的 reported-token 额度为 100,000，但当前 runner 的 `--max-reported-tokens` **只是观察已发生 usage 后的软停止**，并非供应商硬费用上限。不得把它写成硬 cap。正式执行前另检查逐次剩余额度和保守预留；缺 usage 则停。fixture v2 中的 80,000 是上一轮协议，本轮实际 CLI 配置和这一差异必须随报告列出。

准备命令（无凭据、零 provider 请求）：

```sh
PYTHON_DOTENV_DISABLED=1 PYTHONPATH=. DOCTHINKER_RESEARCH_SEMANTIC=off \
  /private/tmp/docthinker-invite-package-test/bin/python scripts/benchmark_research_assist.py \
  --fixture benchmarks/research_real/insight_development_smoke_v2.json \
  --papers-dir dist/real-paper-ab-20261010/papers \
  --output dist/insight-deploy-smoke-v4-prepare-20261012 \
  --prepare-only --contrast insight --model step-5-preview \
  --base-url https://api.stepfun.com/step_plan/v1 \
  --repetitions 1 --max-requests 6 --max-reported-tokens 100000 \
  --temperature 0.1 --timeout-seconds 180
```

正式执行的额外协议收据步骤（不改变 runner）：

1. 在冻结后的全新准备目录检查 `summary.json` 为六份 `prepared_only` 且实际 dispatch 为零；每份 prepared 的真实请求参数都是 8192 / 0.1，来源与问题经过上述预审。
2. 比较准备 `run.json` 内的 `configuration.code_fingerprint` 和实时 `code_fingerprint()`；不同则重新准备。记录六个实际 `prompt_digest`，不得拿旧包代表新输入。
3. 在尚无付费记录的新目录，以现有 `private_json` 原子写入 `protocol.json`，记录 `phase=before`、准备目录与六个输入摘要、开始代码指纹、`max_requests=6`、`max_reported_tokens=100000`、`token_limit_basis=observed_usage_soft_stop_not_provider_hard_spend_limit`、`completion_cap=8192`、`temperature=0.1`、`semantic=off`、`automatic_retry=false`、`stop_on_failure=true`。不存密钥、环境变量快照或完整服务配置。
4. runner 实际参数必须和该协议相同，读取密钥只在进程内完成。协议收据仅记录和核验，不冒充供应商费用控制。
5. 结束或发生失败后，保留原 `before`，追加 `after` 字段：结束代码指纹、正式 `run.json` 的代码指纹、实际 dispatch 数、provider 回执数、已知 reported tokens、unknown usage 数和 outcomes。分别记录与准备/开始版本是否一致。失败也留痕，不能为了补齐六次而恢复重发。

## 第一次准备检查：尚未达到发送条件

代码指纹 `4cfb9e01d58f920ba3e1d955254a6dec8e8ae65d89abc6f709bac3dd6bbf09af`；六份 prepared-only，模型请求 0。此快照不是后续冻结版本；新准备必须使用新目录，不覆写它。

| 问题 | 臂 | 输入字节 | 原文字节 | 来源数 | 文档分布 |
|---|---|---:|---:|---:|---|
| 文献比较 | OFF | 18,803 | 7,966 | 8 | GraphRAG 4 / RAG 3 / Lost in the Middle 1 |
| 文献比较 | AUTO | 20,551 | 9,387 | 10 | GraphRAG 5 / RAG 3 / Lost in the Middle 2 |
| 可检验假设 | OFF | 20,065 | 7,308 | 8 | RAG 8 / Lost in the Middle 0 |
| 可检验假设 | AUTO | 21,938 | 8,871 | 10 | RAG 9 / Lost in the Middle 1 |
| 助手报告 | OFF | 19,836 | 8,403 | 8 | MemGPT 4 / RAG 4 |
| 助手报告 | AUTO | 21,613 | 9,893 | 10 | MemGPT 4 / RAG 6 |

六份均在 24,576 输入字节、12,000 原文字节预算内，实际 completion cap 都为 8192；OFF 未注入 Insight，AUTO 均注入。

输入已有改进：文献比较的 S3 现在含 RAG 的 seq2seq / dense index 和两种 formulation；S6 含 MIPS、top-K 与 latent-document marginalization。GraphRAG 的实体图、社区摘要与评估结果也进入材料。原 V3 的纯背景 S5 和参考文献表 S8 不再占据整条来源。

但还有实质缺口：

1. 假设题 OFF 的八条来源全部来自 RAG，完全没有 Lost in the Middle；AUTO 只补入其 p9 的 decoder-only / query-aware 讨论，未覆盖位置变化的核心实验。不能因为有八条“相关来源”就认为完成双论文问题覆盖。
2. 助手报告虽有两篇论文，但 RAG 原始架构仍未覆盖，来源偏评估和附录；MemGPT S1 整条是一般 preprompt 背景。其 S5 是拟评估标准而非已报告效果，后续答案不得把它写成实验胜出。
3. 低噪声不等于信息完整。文献比较 S5 同时包含方法和泛背景；假设题有若干与位置问题关系较弱的生成评估。它们不应因“不是纯背景”被称作零噪声。

## 答案出来后按固定口径逐项审查

1. **实际交付**：以产品 result.answer / sources / provenance 为准，不以 provider 的非空 content 代替。记录撤回、丢来源、保存或历史加载失败；失败样本保留，不筛掉以抬高质量。
2. **关键材料覆盖**：逐题检查任务要求所需方法、观察和实验条件是否存在于实际 source_quotes。特别检查 RAG 核心方法以及第二篇论文是否缺席；全文中有而 prompt 没有必须记为召回缺口。
3. **背景与参考表占比**：同时报告整条来源数比例与原文 UTF-8 字节比例，明确人工分类。只有整条均为无关通用背景或参考列表才计入该严格指标；混合片段、重复和低相关来源另列，不把严格指标冒充所有噪声。
4. **逐主张直接支持**：分别核对数值及定性判断与其实际引用节选，包含方法组合、优于谁、在哪个任务、用什么指标。不同分句单独看；同论文其他标签不能替代当前错误引用。标签存在只是结构检查，不是蕴含验证。
5. **条件与未知**：检查对象、任务、阶段、指标、对照、可改变项和固定项是否丢失；“未在本次节选看到”不能写成“论文没有”。区分示例、计划、已报告结果、推导与假设。
6. **任务完成**：综述应区分方法/评价/不可比点；假设应有原文观察、机制假设、最强替代解释和可证伪实验；助手报告应区分已有演示与私人助手场景迁移，不把类比当已验证能力，也不能把长期记忆机制等同于已验证的隐私安全、租户隔离或无上限准确性。
7. **真实成本和速度**：记录各臂 provider input/output/cache/total token、模型耗时、流程耗时；单独列预处理。不能把单次差值、缓存或供应商负载归因成算法普遍收益。

审查是开发 AI 的非盲逐来源复核，不是独立专家评分。可列具体通过项和错误，不编造参考标准答案、总分、置信区间或泛化结论。所有质量比较只使用完整有效的配对，但单臂失败必须单独报告。

## 第二次准备检查：文档覆盖改善，但角色覆盖仍不足

独立目录 `dist/insight-deploy-smoke-v4b-prepare-20261012/`，代码指纹 `066c9c50eb1ac04f890766e1807d71703d0c37e870c8ea8356837911984f0128`。六份 prepared-only，仍无 provider 请求，不覆盖第一次准备。

| 问题 | 臂 | 输入字节 | 原文字节 | 文档分布 |
|---|---|---:|---:|---|
| 文献比较 | OFF | 16,891 | 5,909 | GraphRAG 3 / RAG 3 / Lost in the Middle 2 |
| 文献比较 | AUTO | 18,653 | 7,330 | GraphRAG 4 / RAG 3 / Lost in the Middle 3 |
| 可检验假设 | OFF | 19,730 | 7,085 | RAG 4 / Lost in the Middle 4 |
| 可检验假设 | AUTO | 21,638 | 8,685 | RAG 6 / Lost in the Middle 4 |
| 助手报告 | OFF | 18,705 | 7,872 | MemGPT 4 / RAG 4 |
| 助手报告 | AUTO | 20,493 | 9,362 | MemGPT 4 / RAG 6 |

假设题恢复两篇论文：S3/S5 覆盖 RAG latent-document formulations 和 DPR；S2 覆盖 retrieval recall 与 reader 表现的不同饱和点；S7 包含 key-value 位置重排，S8 明示中间位置退化。S6 仍是长上下文技术通用背景，不能把文档平衡当成所有材料都有用。

助手报告仍没有 RAG 基本 seq2seq / dense-index 架构；MemGPT 主要是 preprompt 背景和总结，external memory、recall / archival 与主动 paging 的具体机制未入包。两论文各占四条并没有解决方法遗漏。

文献比较的 Lost in the Middle 两条基础节选变成 287 字节的截断结果和 416 字节的模型设置，原 U-shaped 位置核心观察丢失；RAG 还保留摘要架构，但第一次准备中的 MIPS 正文不再入包。下一次准备需要检查角色覆盖，不能仅以来源篇数、输入变短或字面相关性验收。

## 第三次准备检查：角色启发式没有稳定保住关键材料

独立目录 `dist/insight-deploy-smoke-v4c-prepare-20261012/`，代码指纹 `43b229bff7ceddfb0b9a52f7b3a823be69ddffb368734843e95e6f67e4fbf4ab`。六份 prepared-only，provider 请求仍为零。

| 问题 | 臂 | 输入字节 | 原文字节 | 来源数 |
|---|---|---:|---:|---:|
| 文献比较 | OFF | 17,768 | 6,452 | 8 |
| 文献比较 | AUTO | 18,585 | 7,105 | 9 |
| 可检验假设 | OFF | 15,665 | 3,985 | 5 |
| 可检验假设 | AUTO | 16,577 | 4,728 | 6 |
| 助手报告 | OFF | 16,225 | 5,218 | 7 |
| 助手报告 | AUTO | 18,012 | 6,708 | 9 |

文献比较两条 RAG 材料变为人评 UI / 工程配置和 Jeopardy 设置，核心架构缺失；Lost in the Middle 的三条材料为 QA、指令微调设置和相关工作，核心位置观察也缺失。假设题虽然更短，但只留下 150 字节的 key-value 实验规模说明，没有位置变化结果。报告题 MemGPT 的 main-context 三个组件有恢复，external-memory 两类存储和主动函数式调页仍缺；RAG 还选中一条社会影响应用设想，不能把它视为方法证据。

发现具体连续性风险：报告 RAG p7 的 index-hot-swapping 节选以“82 world leaders who had changed”及页码结束，句子和结果在 p8 的表格之后继续。下一页开头不是直接续句，不能简单拼接两个片段冒充连续原文。缺后续实验结论时，答案应限于当前可见设置或明确缺口，不能补出未提供的数值。完整后续材料确实存在于提取语料中，属于检索与结构定位问题，不是 PDF 没提取。

这轮减少输入的同时丢了关键内容，不能作为“更省 token 且质量不变”的证据。正式付费继续暂缓。

## 第四次准备检查：核心恢复，但仍是候选版本

独立目录 `dist/insight-deploy-smoke-v4d-prepare-20261012/`，代码指纹 `01bf9dcc2ec50f27df6c780b4eb44fe218951b7acd4af71d2698351ba54e4cad`。六份 prepared-only，零 dispatch；准备之后发现重新摄取文档产生的 UUID 会影响候选上限附近的顺序，尚需确定性回归，**此版本不用于正式付费**。

| 问题 | 臂 | 输入字节 | 原文字节 | 来源数 | 前台准备毫秒 |
|---|---|---:|---:|---:|---:|
| 文献比较 | OFF | 19,235 | 8,749 | 8 | 495.3 |
| 文献比较 | AUTO | 20,961 | 10,170 | 10 | 507.1 |
| 可检验假设 | OFF | 18,341 | 6,120 | 8 | 468.8 |
| 可检验假设 | AUTO | 20,256 | 7,683 | 10 | 488.9 |
| 助手报告 | OFF | 19,347 | 8,131 | 8 | 446.7 |
| 助手报告 | AUTO | 21,129 | 9,621 | 10 | 478.3 |

前台准备是到真实模型边界为止的本地流程耗时，不是生成速度；文档摄取和预热另计。prepared diagnostics 的 `scheduler_status` 全部为 `not_observed`，不能写成已执行、跳过或零调度成本。六份实际 cap 都是生产 8192，temperature=0.1，semantic 环境为 off；OFF 不注入 Insight，AUTO 注入。

逐内容检查：

- 文献比较 S1 包含 RAG 双记忆与 same-sequence / per-token formulations；S2 有社区 map-reduce、两类语料、指标及对照；S3/S6 有 Lost in the Middle 首尾优于中间的现象。不是只命中模型名字。
- 假设题 S1/S5 有 RAG 双记忆与两种 formulation，S2/S4 有中间位置退化的原文观察；但尚缺只重排信息位置、答案不变的具体受控过程。S7 只是 key-value 数量和样本规模，不能代替位置实验结果。
- 助手报告 S6 有 RAG 基本架构，S1 有 MemGPT 层级、函数和事件，S3 有主上下文三个组成部分，S5 有 DMR 问题生成和 ROUGE-L / LLM judge 评价方式。仍缺 external recall / archival 两类存储的明确定义及部分 DMR 基线/会话条件；S7 截在“page in”及页码处，不能声称核心连续上下文已完整。

人工噪声审计采用预先说明的严格整条口径：六臂整条参考文献表均为 0。助手报告 S8 整条为通用社会效益与应用展望，238 字节：OFF 为 1/8 条、238/8131 字节（2.93%），AUTO 为 1/10 条、238/9621 字节（2.47%）。其余四臂严格整条纯背景计数为 0。

这不表示所有剩余片段都高密度或无噪声。假设 AUTO S9 的 793 字节是相关工作混合少量本研究检索模块说明，不能算整条纯背景；报告 S7 的 1116 字节、综述 S4 的 1195 字节也混有背景和方法。应用展望被单列不意味着对迁移讨论毫无价值；这里的比例不是自动答案质量分。

## 冻结版输入检查（V4e）：可以进行有界真实比较，未宣布答案达标

独立目录 `dist/insight-deploy-smoke-v4e-prepare-20261012/`；指纹 `6e5e803cffbdee0dc60c43219aae4ee484e60601c358b61fb61dbfa9bbd89542`，准备结束时的实时指纹与之相等。六份 prepared-only、零 dispatch，每份实际 completion cap=8192、temperature=0.1，semantic 环境关闭。

重新摄取后的六份实际 `source_quote_payload` 与 V4d **逐字段相等**（标签、正文、文件名、页、节选坐标及截断标记）；内部 provenance ID 是新生成的，不能把 ID 差异当作原文漂移。上述核心覆盖、具体缺口、原文字节和人工噪声分类均沿用，不将“有层级概述”改写为“external recall / archival 细节齐全”。报告题包明确附有 `source_boundary_unverified`，提示末尾可能跨页或截断。

| 问题 | OFF 前台准备毫秒 | AUTO 前台准备毫秒 |
|---|---:|---:|
| 文献比较 | 503.463 | 513.488 |
| 可检验假设 | 468.308 | 491.209 |
| 助手报告 | 618.098 | 466.866 |

`scheduler_status` 仍为 `not_observed`，不能推断调度执行结果。本轮恢复了足以进行有界开发比较的核心材料，但具体条件和低词匹配机制仍有缺口。下一步需要审查真实生成及交付；这份准备检查自身没有生成答案，也不构成回答质量或稳定性的最终验收。
