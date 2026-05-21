# PROJECT_STRUCTURE

## 1. 当前主线定位

当前仓库主线定位为 **Agentic Memory Framework**：将文档、对话和推理轨迹转化为可检索、可巩固、可自我演化的多层记忆系统。

当前保留以下运行入口：
- `run_ui.py` -> `docthinker/ui/app.py`（Flask UI）
- `uvicorn docthinker.server.app:app`（FastAPI 后端）

旧 NeuroAgent 主线已从仓库中移除。

## 2. 顶层结构

- `docthinker/`：应用主代码
- `graphcore/`：图检索底层能力
- `neuro_memory/`：类脑记忆模块
- `docs/`：维护文档
  - `MEMORY_PLUGIN_GUIDE.md`：第三方 agent / 插件作者接入 memory core 的指南
- `packages/docthinker-memory/`：轻量 agentic memory facade 分发入口，面向插件作者和第三方 agent 框架
- `tests/`：新主线测试
- `run_ui.py`：UI 启动入口

## 3. docthinker 子结构

- `docthinker/server/app.py`：后端应用入口与生命周期初始化
- `docthinker/memory_core/`：Agentic Memory 统一门面，对外提供 recall / after-response consolidation
  - `core.py`：`AgentMemoryCore` 与 `RecallBundle` / `MemoryTrace`
  - `protocols.py`：可插件化 backend contracts（conversation、episodic、expanded KG、graph promotion、chat-turn ingest）与 `MemoryPolicy`
  - `adapters.py`：将现有 Claw、Neuro Memory、ExpandedNodeManager、GraphCore 接入 protocols
- `docthinker/server/routers/`：API 路由
  - `query.py`：查询与回答
  - `ingest.py`：文档/文本入库
  - `graph.py`：图谱查询、记忆接口、KG扩展接口
  - `sessions.py`：会话管理
  - `health.py`：健康检查
- `docthinker/ui/app.py`：UI 路由与后端代理
- `docthinker/ui/templates/`：现代化模板（`*_modern.html`）
- `docthinker/auto_thinking/`：自动思考与多步推理
- `docthinker/hypergraph/`：超图 RAG
- `docthinker/kg_expansion/`：KG 扩展模块
- `claw/`：热/温/冷三层对话记忆
- `neuro_memory/`：情节记忆、扩散激活、类比检索
- `graphcore/`：语义记忆与实体关系检索底座

## 4. 关键数据流

1. UI 将请求转发到 `/api/v1/*`
2. 后端在 `lifespan` 初始化 session-scoped RAG、Claw、Neuro Memory、KG 扩展能力
3. `query` 路由通过 `memory_core.AgentMemoryCore.recall()` 组装 agent 记忆上下文
4. `DocThinker/GraphCore` 作为 recall backend 生成答案
5. 答案返回后通过 `AgentMemoryCore.after_response()` 触发记忆巩固、可选对话回写、扩展节点晋升
6. `graph` 路由负责图谱数据、记忆统计、KG 扩展调试与人工操作

## 5. 清理结果

- 仅保留 `docthinker + graphcore + claw + neuro_memory` 主线。
- 旧入口与旧模块已移除，不再维护兼容分支。
