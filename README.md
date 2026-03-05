# DocThinker

DocThinker 是一个以文档理解为核心的知识系统，当前主线为：
- `docthinker`：应用层、服务层、UI、自动思考编排
- `graphcore`：图检索与图存储能力
- `neuro_memory`：类脑记忆（扩散激活、情景记忆、联想检索）

本仓库已完成一次主线收敛清理：旧版 NeuroAgent 主线与其冗余文档/模块已移除。

## 1. 推荐启动方式

1. 启动后端 API（FastAPI）

```bash
python -m uvicorn docthinker.server.app:app --host 0.0.0.0 --port 8000
```

2. 启动前端 UI（Flask）

```bash
python run_ui.py
```

3. 访问页面
- 聊天：`http://127.0.0.1:5000/query`
- 知识图谱：`http://127.0.0.1:5000/knowledge-graph`
- 兼容路由：`http://127.0.0.1:5000/kg-viz`

## 2. 主线目录

- `docthinker/server/`：后端入口与路由
- `docthinker/ui/`：Web UI 与模板
- `docthinker/auto_thinking/`：自动思考与问题分解
- `docthinker/hypergraph/`：超图推理能力
- `docthinker/kg_expansion/`：KG 扩展模块
- `graphcore/`：图检索与图存储底座
- `neuro_memory/`：记忆引擎
- `docs/`：当前维护文档

## 3. 关键能力映射

- 扩散激活/联想：`neuro_memory/engine.py`
- 情景记忆：`neuro_memory/engine.py`
- KG 扩展：`docthinker/kg_expansion/expander.py`
- 类脑思考编排：`docthinker/auto_thinking/orchestrator.py`

## 4. 开发与测试

```bash
pip install -e "[all]"
pytest tests/ -v
```

当前保留的测试以新主线为准，旧主线测试已清理。

## 5. 文档

- 项目结构：[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)
- 系统流程：[docs/SYSTEM_FLOW_GUIDE.md](docs/SYSTEM_FLOW_GUIDE.md)
- KG 优化：[docs/KG_OPTIMIZATIONS.md](docs/KG_OPTIMIZATIONS.md)
- 清理执行记录：[docs/CLEANUP_EXECUTION_2026-03-05.md](docs/CLEANUP_EXECUTION_2026-03-05.md)
