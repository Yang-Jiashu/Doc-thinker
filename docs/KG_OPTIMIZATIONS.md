# KG_OPTIMIZATIONS

本文只针对新主线（`docthinker + graphcore + neuro_memory`）。

## 1. 已有能力

- KG 可视化页面：`docthinker/ui/templates/kg_viz_modern.html`
- KG 查询接口：`/api/v1/knowledge-graph/data`
- KG 扩展接口：`/api/v1/knowledge-graph/expand`
- 扩展节点调试接口：`/api/v1/knowledge-graph/debug-expanded`

## 2. 可继续优化项（优先级）

### P0（高优先级）

- 扩展去重策略从纯字符串去重升级为语义去重（当前阈值配置较保守）
- 扩展节点回写后的可解释性增强（记录来源提示词、上下文摘要、时间戳）
- 扩展节点评估闭环（人工确认/拒绝反馈）

### P1（中优先级）

- 图谱视图支持按来源过滤（原始节点 vs 扩展节点）
- 大图分页/按子图加载，避免前端一次性渲染过多节点
- 对高频扩展主题做缓存，降低重复调用

### P2（低优先级）

- 扩展质量报表（每次扩展新增率、去重率、采纳率）
- 扩展策略 A/B 实验（不同提示词模板对比）

## 3. 和记忆系统协同建议

- 将扩展节点关联到 `neuro_memory` 的显著性指标
- 根据 co-activation 热度提升节点排序权重
- 对“短期高热但长期无用”节点自动降权

## 4. 代码定位

- 扩展核心：`docthinker/kg_expansion/expander.py`
- 扩展路由：`docthinker/server/routers/graph.py`
- 前端触发：`docthinker/ui/templates/kg_viz_modern.html`
