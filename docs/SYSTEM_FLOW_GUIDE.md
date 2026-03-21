# SYSTEM_FLOW_GUIDE

## 1. 启动与入口

- UI: `python run_ui.py`（默认 5000）
- Backend: `python -m uvicorn docthinker.server.app:app --port 8000`

UI 侧页面通过 `docthinker/ui/app.py` 代理调用后端 `api/v1`。

## 2. 后端初始化流程（`docthinker/server/app.py`）

1. 加载 providers/settings
2. 初始化 `DocThinker` 与 GraphCore
3. 初始化 `MemoryEngine`（`neuro_memory`）
4. 初始化 `IngestionService`
5. 初始化 `HybridRAGOrchestrator` + `HyperGraphRAG`
6. 注册所有 `/api/v1/*` 路由

## 3. 查询流程

1. 前端调用 `/api/v1/query` 或 `/api/v1/query/text`
2. 查询路由进入编排器（复杂度分类、分解、检索、聚合）
3. 同步使用 `neuro_memory` 做：
   - observation 写入
   - analogy 检索
   - co-activation 记录
4. 返回最终答案与相关上下文

## 4. 入库流程

1. 前端上传文件到 UI 代理
2. UI 代理转发到 `/api/v1/ingest`
3. `IngestionService` 处理并写入 GraphCore
4. 会话/全局存储按配置同步

## 5. KG 扩展流程

1. 前端触发 `/api/v1/knowledge-graph/expand`
2. `docthinker/kg_expansion/expander.py` 基于现有图谱摘要生成候选节点
3. 执行去重与筛选
4. 将扩展节点写入图存储，并打标 `is_expanded=1`

## 6. 关键文件映射

- 查询：`docthinker/server/routers/query.py`
- 图谱：`docthinker/server/routers/graph.py`
- 记忆：`neuro_memory/engine.py`
- 扩展：`docthinker/kg_expansion/expander.py`
