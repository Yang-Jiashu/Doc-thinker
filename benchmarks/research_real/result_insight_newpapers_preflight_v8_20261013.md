# V8 新论文烟测：准备阶段发现无有效对比，未请求模型

记录日期：2026-10-13（Asia/Shanghai）。这是准备阶段的软件诊断结果，不是回答质量评测结果。

## 结论

两份登记的原始论文成功摄取，原文定位核验通过；但 OFF/AUTO 两臂最终 `messages` 逐项相同，`prompt_digest` 相同，没有实际辅助材料或 Insight 上下文。冻结代码的本地路由复核将本题判为 `not_applicable`。因此本轮没有形成有效的动态辅助对比，停在 prepare，未派发任何供应商请求。

不能据此说“辅助没有提升”“模型回答不好”或“已证明更省 token”。本轮没有模型回答、供应商 token 用量或回答时延，也没有可评分的质量配对。

## 记录与冻结边界

- 原计划：`benchmarks/research_real/plan_insight_newpapers_smoke_20261013.md`。
- 原 fixture：`benchmarks/research_real/insight_newpapers_smoke_20261013.json`，问题 ID 为 `newpapers_personal_assistant_roadmap`。
- 原登记：`benchmarks/research_real/insight_newpapers_smoke_20261013.registration.json`；组合 SHA256 为 `6ca9ec118754491c586834e3d89c59815042c5fa0ffdbd2bcbeb2cde00145a49`。
- 本轮目录：`dist/insight-newpapers-prepare-v8-20261013/`。`run.json` 的 `prepare_only=true`，`contrast=insight`。
- 准备时及本报告作者只读复核路由时的代码指纹均为 `0673da1a2e20509ac3f479199fbd4782c20c2c44fdf214254f0ae3540d30d83b`。
- 准备记录时间为 2026-10-12 17:12:55–17:12:57 UTC，即北京时间 2026-10-13 01:12:55–01:12:57。

原 fixture、计划和原 PDF 的登记 SHA 已重新核对，无改动。本报告另建文件，不回写原登记，不补造答案，不改题目以追求触发，也不修改这次准备记录。

## 摄取与原文验证

两篇固定版本均来自官方 arXiv，原始字节未压缩、裁页或改写，均低于 10 MiB 单文件上限：

| 原论文 | 字节 | 原 PDF SHA256 | 摄取状态 / 块数 |
|---|---:|---|---|
| [MemoryBank v1](https://arxiv.org/abs/2305.10250v1) | 518049 | `5bac0c9588b7999c9ae3821af027bee832964aaab787cf8fa8111b905e4b48fb` | ready / 28 |
| [LongMem v1](https://arxiv.org/abs/2306.07174v1) | 682858 | `6036d59b87763ef304b2bef4e6bc1ee295abf83604b88fd5b854d46ba2ca4f61` | ready / 32 |

`corpus.json` 记录总计 60 个块、共享一次摄取约 1033.17 ms，corpus digest 为 `38c63a47b7b38be0fff0d098e32b5bf7a790d08234516a9ce34096d092f3c5f2`，范围为 `same_frozen_ids_and_offline_indexes_no_history_or_memory`。只读查询对应 `research.db`，两篇文档与所选块均属于该测试库的 `#00001`；文档内容 SHA 与原 PDF 一致，答案表和记忆表均为 0 条。本项只证明这个隔离测试语料的来源与范围，不代表全产品的租户隔离认证。

两臂使用相同四个连续节选，总正文为 **4632 UTF-8 字节**，不是 token 数。逐项核验了文档映射、页码、块 ID、字符偏移和正文切片；并将片段与相应 PDF 页提取文字做空白归一后的连续匹配，四项均通过。偏移采用源块内字符坐标，左闭右开：

| 引用 | 文档 / PDF 页 | 源块 ID | 偏移 `[start,end)` | 正文字节 |
|---|---|---|---|---:|
| S1 | MemoryBank / 3 | `abefe0f4-961e-4173-a3da-b5439436efe0` | `[0,1188)` | 1202 |
| S2 | LongMem / 1 | `957972a3-b537-4287-b0c7-f7f2cc8ff68c` | `[595,1770)` | 1175 |
| S3 | MemoryBank / 2 | `20ea0b4e-9ace-430b-b850-f67e6e20f402` | `[826,1971)` | 1151 |
| S4 | LongMem / 6 | `5bb9f4ba-13c8-4a2d-a9a7-269ea6fd7d8e` | `[328,1432)` | 1104 |

四段实际正文 SHA256 依次为：

- S1：`b1245e49affc45bb584c4a79bcb6a59aea2150629ab92d9fb7ccde0e33fa6275`
- S2：`dc80ea6b55e68320c3d6cb7b6c1c22884d40f03b41771192bd3a9e3ae3b07dae`
- S3：`7062fe90de6e34b5bd0152edb8ec99c48674664fe021c3c95dd8139beceaff4f`
- S4：`734469cd267c74a035096f6b43bbd4a23f3492cd80cac92d7e8ec2300e31a9cf`

四段均标记 `truncated=true`，准备结果保留了 `excerpt_truncated` 提示。坐标正确、切片连续不代表材料完整，也不能证明限定条件、实验范围或关键证据全部入选。另有 `uncovered_query_terms` 词形提示；它不是语义遗漏金标，不能据此计算回答正确率。

## 两臂实际输入：动态辅助未产生差异

| 项目 | OFF | AUTO |
|---|---|---|
| 普通上下文辅助 | true | true |
| 静态知识卡模式 | auto | auto |
| 动态 Insight 模式 | off | auto |
| 最终来源数 / 正文字节 | 4 / 4632 | 4 / 4632 |
| 完整输入记录字节 `input_bytes` | 10636 | 10636 |
| 实际辅助 `has_actual_support` | false | false |
| 辅助块 / 材料条目 | 无 / 0 | 无 / 0 |
| Insight 上下文 | 无 | 无 |
| 状态 / 供应商调用 | prepared_only / 0 | prepared_only / 0 |

两臂最终 `messages` 直接比较一致；同一 `prompt_digest` 为：

`b4daaaadb234991505cc89ac73cf626c9af25127f578bb887f0cee9ee37c7779`

两臂准备参数也一致：`step-5-preview`、`max_tokens=8192`、`temperature=0.1`、`reasoning_effort=low`、`stream=false`；输出上限依据为生产任务预算，并非实验覆盖。检索语义路径固定关闭。这些只是待派发参数，不表示供应商实际执行过。

准备文件顶层旧字段 `assistance_enabled` 为 false/true，在本次 `contrast=insight` 中描述对比臂，不能误读为 OFF 臂关闭了所有普通辅助；真实臂配置以 `arm_configuration` 为准，普通辅助和静态卡配置两边一致。

### 路由诊断的证据范围

最终提示诊断记录的 `routing_status`、`scheduler_status` 和 `skipped_reason` 都是 `not_observed`；这些值本身不提供内部因果轨迹。为避免从“无上下文”臆测原因，本报告作者另在相同代码指纹下，只读调用 `route_knowledge_skill(原问题, "auto")`，实际得到：

```json
{"skill_id":null,"trace":{"algorithm":"local_knowledge_task_router_v2","status":"not_applicable","task_type":null,"model_calls":0,"network_calls":0,"memory_writes":0}}
```

这确认自然表达的“产品路线建议”没有命中本轮知识任务路由，且准备端确实没有形成动态辅助输入。它不是模型拒答、额度不足或已运行后质量变差的证据。后续应修通用决策交付物识别规则，而不是改这道题、添加特定论文关键词或为取得漂亮结果重复调用。

## 调用、成本和质量结论

`summary.json` 记录：`prepared_only=2`、`dispatched_requests=0`、`provider_receipts=0`、`automatic_retries=0`。`calls.jsonl` 仅含两条检索准备完成和两条提示准备完成事件，没有派发事件。两份准备文件均为 `provider_calls=0`。

`reported_total_tokens=null`、`known_usage_count=0`、`unknown_usage_count=0`；不能把“没有供应商用量记录”写成模型生成了零 token 的答案。准备耗时约 OFF 261.94 ms / AUTO 234.13 ms，只是本机准备耗时，不是回答速度，也不用于声称 AUTO 更快。

本轮判定：**资料摄取与所选原文定位通过；有效 A/B 处理差异不成立；停止付费派发；回答质量、token 成本、回答速度均未测。** 四个来源和更高层字节限制以内的输入准备通过，不等于回答证据充分。

## 后续版本与开发曝光

该题已因准备阶段诊断被开发人员看到；若据此修正路由并生成新代码版本，后续同题结果必须标为 **development / 已曝光开发烟测**，不能继续宣称独立未见题或原代码冻结下的验证。原登记、两份原始 PDF 和本轮失败准备记录保留。新的代码冻结、偏差说明和注册由运行者另建，不在本文件中冒认已完成，也不回填成本或质量结果。

即使后续产生有效两臂回答，仍只有一道题、一个配对，不是统计显著性检验、专家认证或整体产品质量证明。

## 本次准备记录的 SHA256

以下相对路径均基于 `dist/insight-newpapers-prepare-v8-20261013/`：

| 文件 | SHA256 |
|---|---|
| `run.json` | `b8a1b4e9675e771634a344d0c3dc41ec18d0ed7e2ae5cc2f961962b546ee6879` |
| `summary.json` | `35a91d8e8b14c1c4e859f3dafc6e0cd6190a1bf389fa55c67962791eefe4e714` |
| `calls.jsonl` | `879e4c94976c92200b6dd89162d3118fd0dbe4cf8e68331f254dca0283a48e2c` |
| `corpora/38c63a47b7b38be0fff0d098e32b5bf7a790d08234516a9ce34096d092f3c5f2/corpus.json` | `dfecfda9b7f505bc45d947f7a8a3e195a222dce083bc20e0029483584323606c` |
| `prepared/newpapers_personal_assistant_roadmap.r1.off.json` | `2f9f567e03db7229a1efbccd0afb113eae674a98f6edd2a89e25d678bb91eed4` |
| `prepared/newpapers_personal_assistant_roadmap.r1.on.json` | `279c87badeb4d71672f7f47e096cd03982a4563d37ea52d3c860b39915b15209` |

本报告仅检查公开论文、准备记录及该隔离测试库，未读取私密配置或密钥，未请求模型，也未读取任何模型隐藏推理。
