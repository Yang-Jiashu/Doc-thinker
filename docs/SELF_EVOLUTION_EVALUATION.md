# 自进化检索 A/B 评测标准

自进化补边不能只用“标准答案关键词覆盖率”判断。该指标会惩罚合理改写，
也无法识别回答虽然命中了关键词、但缺少原文依据或引入大量无关内容的情况。

建议在相同模型、低温度、关闭记忆和缓存的条件下，成对报告以下指标：

| 维度 | 指标 | 方向 |
|---|---|---|
| 完整性 | 参考要点软覆盖率、达到阈值的要点覆盖率 | 越高越好 |
| 忠实性 | 有原始 chunk 支持的回答主张比例 | 越高越好 |
| 幻觉风险 | 无原始 chunk 支持的主张比例 | 越低越好 |
| 聚焦度 | 与参考要点相关的回答主张比例 | 越高越好 |
| 成本 | 最终 chunk 数、上下文 token、生成耗时 | 越低越好 |
| 稳定性 | 同一配置重复运行的方差 | 越低越好 |

`docthinker.evaluation.compare_answers()` 提供不依赖在线模型的词面初筛评分。
历史字段 `grounded_claim_rate` 和 `balanced_score` **不是事实支持率或真实正确率**：
否定句与肯定句也可能获得相同高分。返回值标记 `scoring_method=lexical_proxy`、
`suitable_for_promotion=false`。不要把它的 `preferred` 当成自动晋升依据。
生产评测还应增加盲测人工评分或固定 judge model，对以下项目按 1—5 分评价：

1. 事实是否能由引用原文直接支持；
2. 因果链是否完整且没有越过证据；
3. 是否回答了问题，而非展示检索到的所有知识；
4. 是否存在错误的具体数字、型号、流程或因果关系；
5. 相同问题重复生成时结论是否稳定。

人工要点覆盖率可以辅助分析，但不能用多跳收益掩盖事实一致性退步。
新的离线验收工具采用下述明确门槛；它与 ECLRR 的关系审核是不同的检查。

可通过请求字段进行严格 A/B：

```json
{
  "use_memory": false,
  "use_conversation_context": false,
  "use_llm_cache": false,
  "remember_turn": false,
  "use_self_evolution": true,
  "evolution_mode": "explore",
  "enable_path_completion": false,
  "adaptive_context": false,
  "include_discovered_edges": true,
  "max_discovered_relations": 8,
  "min_discovered_edge_confidence": 0.8,
  "require_discovered_evidence": true,
  "top_k": 20,
  "chunk_top_k": 12,
  "max_relations": 32,
  "max_relation_tokens": 5000,
  "max_total_tokens": 24000
}
```

这份请求隔离的是“探索模式下已审核补边参与”的影响；对照组只将
`include_discovered_edges` 改为 `false`，其余参数必须完全一致。忠实 / 路径两组
使用各自固定模式另行评测；不要把不同模式、不同上下文预算的差异归因于补边。

## 离线候选验收：没有额外 LLM 调用

`assess_evolution_trial(baseline, candidate)` 消费外部已经完成的成对评测，不运行问答、
不生成标签、不改图谱、不自动部署。它负责问一句：这个改动是否值得进入人工复核？

```bash
python -m docthinker.evaluation --baseline baseline.json --candidate candidate.json
```

两个 JSON 文件结构相同。以下只展示一条记录，实际每类至少需要 5 条，共至少 15 条。

```json
{
  "metadata": {
    "dataset_id": "held-out-v1",
    "evidence_snapshot": "sha256:your-frozen-document-snapshot",
    "model_id": "your-fixed-model-version",
    "evaluator_id": "blind-review-panel-v1",
    "rubric_id": "quality-rubric-v1",
    "fixed_config_id": "same-conditions-except-the-candidate-change",
    "split": "held_out",
    "label_source": "human"
  },
  "cases": [
    {
      "case_id": "faithful-001",
      "mode": "faithful",
      "quality": 0.8,
      "unsupported_rate": 0.0,
      "total_tokens": 3200,
      "latency_ms": 2400,
      "success": true
    }
  ]
}
```

- `mode` 为 `faithful` / `path` / `explore`，两个版本必须有完全相同的 case ID 与类别。
- `quality` 为按固定 rubric 归一化到 0–1 的质量分；`unsupported_rate` 为人工或独立评审确认的无依据主张比例。失败请求保留，`success=false` 且质量记 0。
- `total_tokens` 使用完整请求的实际 provider 用量，包括额外生成调用；embedding/图像若计量体系不同，需提前固定统一成本口径并额外记录费用。`latency_ms` 包含本轮全部工作，不能只统计最后一次生成。
- `label_source` 只接受 `human` 或 `independent_evaluator`。字段声明本身不能证明评审独立，调用方负责隔离候选生成器、评审器与留出答案。
- 元数据必须一致；模型 / 原文 / rubric / 固定条件变化应另开实验。候选的唯一改动和版本另行归档，不能在试完后挑选有利题目。

### 默认试验门槛

| 检查 | 当前门槛 |
|---|---|
| 样本覆盖 | 每种题型至少 5 个成对案例；缺一类也不能通过 |
| 类内质量 | 均值下降不超过 0.01；单题下降不超过 0.15 |
| 无依据主张 | 每类平均不增加；单题增加不超过 0.05 |
| 运行失败 | 不能新增失败案例；修好另一题不能抵消这题变坏 |
| 成本 | 每类总 tokens 与耗时均不得增加超过 10% |
| 实质收益 | 三类等权平均质量提高至少 0.02，或总 tokens 减少至少 10% |
| 不确定性 | 配对质量差的近似下界低于 -0.01 时要求补测 |

下界计算为 `mean(delta) - 1.96 * stdev(delta) / sqrt(n)`，是小规模试验的粗略诊断，
不是小样本统计显著性的保证。每类 5 题只是最低输入要求；实际采纳需更多留出题、重复运行，
并报告稳定性和人审一致性。门槛应在实验前固定，不能看完结果再调。

结果 `recommend_review`（退出码 0）只表示建议复核；`reject`（1）表示未达门槛；
`insufficient_evidence` 或 `invalid_input`（2）要求补数据或修输入。所有结果都有
`automatic_promotion=false`。该工具不会将一次评测结果写回正式图谱或改变线上策略。
