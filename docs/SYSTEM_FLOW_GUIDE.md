# SYSTEM_FLOW_GUIDE

## 1. 启动与入口

- UI: `python run_ui.py`（默认 5000）
- Backend: `python -m uvicorn docthinker.server.app:app --port 8000`

UI 侧页面通过 `docthinker/ui/app.py` 代理调用后端 `api/v1`。

## 2. 后端初始化流程（`docthinker/server/app.py`）

1. 加载 providers/settings
2. 初始化 `DocThinker` 与 GraphCore，作为语义记忆 / KG recall backend
3. 初始化 `MemoryEngine`（`neuro_memory`），用于情节记忆、类比检索、扩散激活
4. 初始化 `IngestionService`
5. 初始化 Claw 三层对话记忆（working/core/archive）
6. 初始化 `HybridRAGOrchestrator` + `HyperGraphRAG`（实验性复杂查询路径）
7. 注册所有 `/api/v1/*` 路由

## 3. 查询流程

1. 前端调用 `/api/v1/query` 或 `/api/v1/query/text`
2. 查询路由调用 `AgentMemoryCore.recall()`；内部通过 `memory_core.protocols` 定义的 backend contracts 组装记忆：
   - Conversation backend 注入对话工作记忆、核心摘要和语义归档片段（当前 adapter：Claw）
   - Episodic backend 检索相似情节与类比 episode，形成 episodic analogy context（当前 adapter：Neuro Memory）
   - Expanded KG backend 执行 query-time match，并生成强制检索指令（当前 adapter：ExpandedNodeManager）
   - `MemoryPolicy` 控制启用层、召回宽度、expanded node 匹配阈值和回答实体抽取上限
   - 输出统一 `RecallBundle` 与 `MemoryTrace`
3. `DocThinker/GraphCore` 根据原始问题 + memory instruction 执行检索生成
4. 返回答案、sources、memory trace、expanded matches
5. 后台调用 `AgentMemoryCore.after_response()`：
   - 通过 conversation backend 更新对话记忆层
   - 通过 episodic backend 将本轮问答写入 episode store，供后续类比召回
   - 可选将对话 turn 写回 KG
   - 根据回答使用情况推进 expanded node candidate -> active -> promoted，并通过 graph promotion backend 写入正式图谱

## 4. 入库流程

1. 前端上传文件到 UI 代理
2. UI 代理转发到 `/api/v1/ingest`
3. `IngestionService` 处理并写入 session-scoped GraphCore
4. 本地 KG/KB/snapshot 更新
5. 后台触发密度聚类、潜在边发现、KG self-study

## 5. KG 扩展流程

1. 前端触发 `/api/v1/knowledge-graph/expand`
2. `docthinker/kg_expansion/expander.py` 基于现有图谱摘要生成候选节点
3. 执行去重与筛选
4. 将扩展节点写入图存储，并打标 `is_expanded=1`
5. `ExpandedNodeManager` 记录 candidate 生命周期；后续 query 使用和回答采纳会推动晋升

## 6. 关键文件映射

- 记忆门面：`docthinker/memory_core/core.py`
- 记忆协议：`docthinker/memory_core/protocols.py`
- 现有系统适配：`docthinker/memory_core/adapters.py`
- 轻量分发入口：`packages/docthinker-memory/`
- 查询：`docthinker/server/routers/query.py`
- 图谱：`docthinker/server/routers/graph.py`
- 对话记忆：`claw/memory_manager.py`
- 情节记忆：`neuro_memory/engine.py`
- 扩展：`docthinker/kg_expansion/expander.py`
