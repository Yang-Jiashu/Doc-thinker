# neuro_memory 验证指南

## 1. 在完整服务里验证（推荐）

1. 启动后端：

```bash
python -m uvicorn docthinker.server.app:app --host 0.0.0.0 --port 8000
```

2. 检查记忆引擎状态：

```bash
curl http://127.0.0.1:8000/api/v1/graph/memory/stats
```

期望返回 `enabled: true`。

## 2. 生成记忆

通过 UI 或 API 连续发起多轮会话，内容要有连续主题（例如并购、季度复盘、项目规划）。

推荐接口：
- `POST /api/v1/query`
- `POST /api/v1/query/text`

## 3. 验证写入与联想

1. 再次查看统计：

```bash
curl http://127.0.0.1:8000/api/v1/graph/memory/stats
```

2. 检查是否出现 `episodes` 增长。

3. 在新问题中引用旧主题，观察回答是否出现跨轮联想内容。

## 4. 手动触发巩固（可选）

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/graph/memory/consolidate?recent_n=30&run_llm=true"
```

执行后再次查看 stats，确认图边数量可能增长。

## 5. 自检清单

- 能正常访问 `/api/v1/graph/memory/stats`
- `enabled` 为 `true`
- 多轮对话后 `episodes` 增加
- 巩固后 `edges` 有增长趋势
- 回答中能体现历史关联与类比
