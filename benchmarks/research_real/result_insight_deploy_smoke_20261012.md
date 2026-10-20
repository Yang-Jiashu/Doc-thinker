# Insight 接线与 Step 5 真实论文小试：未通过回答验收

2026-10-12。**离线接线已观察到生效；真实模型首个基线请求没有返回可用回答，因此没有形成任何有效 OFF／AUTO 配对，不能判断质量、token 或速度收益。** 未自动重试、未提高输出上限、未切换模型。

## 比较内容与适用范围

新增 `--contrast insight`：两臂均使用真实 `ResearchService`，静态知识卡和原有后台材料辅助都开启，只将 `insight_mode` 从 `off` 切换为 `auto`。静态卡与动态约束共享预算，最终保留的完整卡数可能不同；不是把静态卡或全部辅助一起关闭后作比较。默认 `--contrast context-assistance` 保留既有行为。

本次使用 `insight_development_smoke_v1.json` 的三个开发任务：文献比较、可检验假设、助手方案报告；材料为固定哈希版本的 RAG、Lost in the Middle、GraphRAG、MemGPT 原始公开 PDF。**题目有意使用已知路由词来覆盖三个现有 pack，不是未见题、未见论文或泛化证据。** 没有向回答模型注入参考答案，也没有编造标准事实作自动评分。

路由覆盖仍有明显缺口：对原有 `research_papers_ab_v1.json` 的 13 个问题作当前生产路由检查，全部没有匹配 pack，Insight schema 数为 0。这不等于它们没有普通检索或原有后台辅助，但说明当前 Insight 不会普遍自动激活；不能拿本次三个定向 smoke 问题掩盖这一限制。封存 holdout 未读取或修改。

## 零付费 prepare 检查

使用同一组真实 PDF，经真实提取、检索、预算及最终 provider 边界，完成六份输入；没有生成答案，也没有 provider 调用。语义模型明确设为 `DOCTHINKER_RESEARCH_SEMANTIC=off`，因此不能将其当作启用本地语义模型时的性能结论。

| 开发任务 | 最终来源数 OFF → AUTO | 实际输入字节 OFF → AUTO | Insight JSON 实际入模 |
|---|---:|---:|---|
| 文献比较 | 8 → 10 | 17,150 → 18,399 | OFF 无；AUTO 有 |
| 可检验假设 | 8 → 10 | 19,243 → 20,718 | OFF 无；AUTO 有 |
| 助手方案报告 | 8 → 10 | 17,614 → 19,084 | OFF 无；AUTO 有 |

这些数字证明实际输入发生了变化，不证明新增来源相关、推理正确或回答更好。字节数不是 token 计费值。prepare 记录位于本机临时目录 `/private/tmp/docthinker-insight-smoke-prepare-20261012/`。

## 唯一真实请求的原始事实

运行目录为 `dist/insight-deploy-smoke-20261012/`；原始 `run.json`、`calls.jsonl`、`summary.json` 保留未改。

| 字段 | 供应商回执／运行记录 |
|---|---|
| 调用 | `smoke_literature_comparison.r1.off` |
| 请求／实际返回模型 | `step-5-preview` / `step-5-preview` |
| 端点 | Step Plan Chat Completions |
| 参数 | temperature=0；reasoning_effort=low；max_tokens=2048；stream=false |
| HTTP / finish_reason | 200 / `length` |
| 适配器结果 | `empty_provider_answer`，未获得可用的正式回答内容 |
| 输入 / 输出 / 总 tokens | 4,112 / 2,048 / **6,160** |
| 缓存 / reasoning tokens | 供应商分别上报 0 / 0 |
| 模型请求耗时 | 31.670 秒 |
| 检索等模型前耗时 | 0.182 秒，不含预先提取／文档准备 |
| 已发请求 / 剩余未发 / 自动重试 | 1 / 5 / 0 |
| 有效生成配对 / 质量评分 | 0 / 未进行 |

HTTP 200 不是成功回答的证明。`length` 与输出 tokens 达上限共同说明此次响应到达输出限制，但没有可用正式回答；仅凭这些回执不能断言 token 用于隐藏推理，或认定增加上限就必然解决。失败发生在 **Insight OFF** 臂，更不能归因于 Insight AUTO 的增益或退步。

原计划最多六次请求、30,000 上报 tokens 软停止阈值、单次 90 秒超时。该阈值是在取得 usage 后检查，不是供应商硬消费上限。本次首次失败即停止，实际只消耗已上报的 6,160 tokens。没有后续付费试探。

## 元数据说明与修复

此次真实运行保存的生成代码指纹为 `184944a7ca859b3829c494d69b58e27293c08c810af66cd83fcdd87781f8a4c0`。运行时两臂 `arm_configuration.context_assistance_enabled=true`，请求参数与服务实例设置正确；但旧版 manifest 的部分描述沿用了另一种对照：

- `retrieval.excerpt_char_limits.off` 误标 1000；Insight 对照两臂基础相邻原文上限均为 1600 字符。
- `optional_evidence_selection.scope` 与 `query_translation.scope` 仍写 `ON only`；此次两臂均开启原有材料辅助。
- 原有 `source_budget_shared_by_both_arms.max_sources=8` 描述基础选择器，未明确 Insight 可在不增加总原文字节预算的前提下补至多两条来源。

这些是说明性元数据错误，不改变已执行请求和实际输入，也不使未完成的配对变得有效。**运行结束后**，runner 已按 contrast 修正说明、记录补充来源的数量／字节边界，并新增回归测试；没有回写或重新解释旧 manifest 为新版，也没有重新请求模型。

benchmark、分析器、回答契约、盲评汇总相关测试共 **101 passed**；包含真实服务＋合成 PDF 的无网络契约测试、prepare 零调用、实际 Insight 入模、静态卡保留、混合对照拒绝。合成模型测试只验证软件契约，不是生成质量证据。

## 当前结论

接线和预算回执可以复核，真实产品默认 2048 输出上限下的首次完整回答验收未通过。下一步需在明确成本边界下处理空回答／截断的交付体验，并另行完成可用回答的配对盲评；在此之前，不宣称 Insight 提高质量、节省 tokens、加快回答或已可稳定上线。
