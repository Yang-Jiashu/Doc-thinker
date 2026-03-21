# PROJECT_STRUCTURE

## 1. 当前唯一主线

当前仓库只维护以下主线：
- `run_ui.py` -> `docthinker/ui/app.py`（Flask UI）
- `uvicorn docthinker.server.app:app`（FastAPI 后端）

旧 NeuroAgent 主线已从仓库中移除。

## 2. 顶层结构

- `docthinker/`：应用主代码
- `graphcore/`：图检索底层能力
- `neuro_memory/`：类脑记忆模块
- `docs/`：维护文档
- `tests/`：新主线测试
- `run_ui.py`：UI 启动入口

## 3. docthinker 子结构

- `docthinker/server/app.py`：后端应用入口与生命周期初始化
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

## 4. 关键数据流

1. UI 将请求转发到 `/api/v1/*`
2. 后端在 `lifespan` 初始化 RAG + MemoryEngine
3. `query` 路由调用编排器与记忆引擎
4. `graph` 路由负责图谱数据、记忆统计、KG扩展

## 5. 清理结果

- 仅保留 `docthinker + graphcore + neuro_memory` 主线。
- 旧入口与旧模块已移除，不再维护兼容分支。
