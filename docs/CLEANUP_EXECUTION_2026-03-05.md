# CLEANUP_EXECUTION_2026-03-05

## 0. 目标

本次清理目标：
- 删除旧版主线（NeuroAgent 线）
- 只保留并维护新版主线：`docthinker + graphcore + neuro_memory`
- 删除冗余/不再使用的模板、测试与文档
- 统一入口与文档说明

## 1. 判定原则

以下文件判定为“可删除”：
- 不在当前主入口链路中（`run_ui.py` + `docthinker/server/app.py`）
- 仅被旧入口（`main.py`、`neuro_agent*.py`）引用
- 文档仅服务旧主线或历史 UI 迭代记录

## 2. 实际删除清单

### 2.1 删除旧主线代码目录

- `agent/`
- `cognition/`
- `perception/`
- `retrieval/`
- `neuro_core/`
- `api/`

### 2.2 删除旧入口与旧脚本

- `main.py`
- `neuro_agent.py`
- `neuro_agent_v2.py`
- `api_multi_document.py`
- `run_ui_v2.py`
- `launch_ui.py`
- `clean.py`
- `check_dim.py`
- `verify_cognitive.py`
- `verify_dual_write.py`
- `kg_viz_standalone.html`

### 2.3 删除旧 UI 路由与旧模板

- `docthinker/ui/routers/kg_visualization.py`
- `docthinker/ui/templates/base.html`
- `docthinker/ui/templates/config.html`
- `docthinker/ui/templates/index.html`
- `docthinker/ui/templates/knowledge_graph.html`
- `docthinker/ui/templates/kg_visualization.html`
- `docthinker/ui/templates/query.html`
- `docthinker/ui/templates/upload.html`

### 2.4 删除旧测试

- `tests/test_api_query.py`
- `tests/test_api_multi_query.py`
- `tests/test_api_ingest.py`
- `tests/test_session_flow.py`
- `tests/debug_db.py`
- `tests/test_embed_dim.py`

### 2.5 删除冗余/历史 MD 文档（根目录）

- `ARCHITECTURE.md`
- `CLEAN_UI_CHANGELOG.md`
- `CONTEXT_COMPOSER_DESIGN.md`
- `CORE_INNOVATIONS.md`
- `CYBERPUNK_UI_CHANGELOG.md`
- `FINAL_SUMMARY.md`
- `KG_VISUALIZATION_GUIDE.md`
- `NEURO_AGENT_README.md`
- `PROJECT_SUMMARY.md`
- `QUICK_START.md`
- `README_OLD.md`
- `README_STARTUP.md`
- `STARTUP_GUIDE.md`
- `START_BACKEND.md`
- `TOC_PRODUCT_ARCHITECTURE.md`
- `UI_CHANGELOG.md`
- `UI_README.md`

### 2.6 删除冗余 docs/* 文档

- `docs/CC_VS_DOCTHINKER_COMPARISON.md`
- `docs/CODE_AND_DOCS_OVERVIEW.md`
- `docs/FOLDERS.md`
- `docs/MULTIMODALRAG_VS_DOCTHINKER_WITH_PYRAMID_EXAMPLE.md`
- `docs/OPEN_SOURCE_READINESS.md`
- `docs/PUBLISH_TO_GITHUB.md`
- `docs/RAG_ANYTHING_VS_DOCTHINKER_MULTIMODAL_KG.md`

### 2.7 删除旧运行产物/临时目录

- `context_dumps/`
- `neuro_agent_data/`
- `token_stats.jsonl`
- `hypergraphrag.log`
- `test_ingest.txt`

## 3. 修改清单（保留主线）

### 3.1 路由与入口

- `run_ui.py`
  - 去除旧 `kg_visualization` 蓝图依赖
  - 保留新版主入口信息

- `docthinker/ui/app.py`
  - 移除旧蓝图注册逻辑
  - 新增 `/kg-viz` -> `/knowledge-graph` 兼容别名

- `docthinker/ui/templates/base_modern.html`
  - 侧栏链接改为 `/knowledge-graph`
  - active 判断改为 `knowledge_graph_page/kg_viz_page`

### 3.2 文档重写

- `README.md`
- `docs/PROJECT_STRUCTURE.md`
- `docs/SYSTEM_FLOW_GUIDE.md`
- `docs/KG_OPTIMIZATIONS.md`
- `neuro_memory/VERIFY.md`

## 4. 清理后主线定义

### 4.1 后端主线

- `docthinker/server/app.py`（唯一 FastAPI 主入口）
- 统一 API 前缀：`/api/v1`

### 4.2 前端主线

- `run_ui.py` -> `docthinker/ui/app.py`
- 主 KG 页面：`/knowledge-graph`
- 兼容别名：`/kg-viz`

### 4.3 关键能力位置

- 类脑记忆：`neuro_memory/engine.py`
- 自动思考：`docthinker/auto_thinking/orchestrator.py`
- KG 扩展：`docthinker/kg_expansion/expander.py`

## 5. 风险与注意事项

- 若外部脚本仍调用已删除旧入口（如 `main.py`），需要改为新入口。
- 老文档链接已大量移除，历史说明需回看 Git 历史。
- 这次清理不会删除 `graphcore` 或 `neuro_memory` 主模块。

## 6. 建议的后续动作

1. 运行一次最小回归：
   - 启动 `uvicorn docthinker.server.app:app`
   - 启动 `python run_ui.py`
   - 验证 `/query`、`/knowledge-graph`、`/api/v1/health`
2. 清理空目录与无效导入（可在后续提交中做）
3. 将 CI 的 `|| true` 去掉，恢复真实失败阻断
