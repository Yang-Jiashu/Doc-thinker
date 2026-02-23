# NeuroAgent 系统启动指南

## 概述

NeuroAgent 系统包含两个主要部分：
1. **Flask UI** (前端界面) - 端口 5000
2. **FastAPI** (后端 API) - 端口 8000 (可选)

## 快速启动

### 方式 1: 仅启动 Flask UI（推荐）

如果只使用 Web 界面，只需启动 Flask UI：

```bash
# 在项目根目录执行
python run_ui.py

# 或使用 V2 版本（包含 KG 可视化）
python run_ui_v2.py
```

访问地址：
- **聊天界面**: http://127.0.0.1:5000/query
- **知识图谱**: http://127.0.0.1:5000/knowledge-graph
- **KG 可视化 (新)**: http://127.0.0.1:5000/kg-viz

### 方式 2: 完整启动（UI + 后端 API）

如果需要使用完整功能，需要同时启动前后端：

**终端 1 - 启动 FastAPI 后端：**
```bash
python scripts/start_api.py
```

**终端 2 - 启动 Flask UI：**
```bash
python run_ui.py
```

## 启动脚本说明

### 1. `run_ui.py` - 原启动脚本

```bash
python run_ui.py
```

**功能：**
- 启动 Flask Web 服务
- 端口：5000
- 包含基础聊天界面
- 包含原知识图谱页面

**访问地址：**
- http://127.0.0.1:5000/query
- http://127.0.0.1:5000/knowledge-graph

### 2. `run_ui_v2.py` - V2 启动脚本（推荐）

```bash
# 基本启动
python run_ui_v2.py

# 指定端口
python run_ui_v2.py --port 8080

# 调试模式
python run_ui_v2.py --debug
```

**功能：**
- 启动 Flask Web 服务
- 包含新的 KG 可视化路由
- 支持命令行参数

**访问地址：**
- http://127.0.0.1:5000/query
- http://127.0.0.1:5000/knowledge-graph
- http://127.0.0.1:5000/kg-viz ⭐ 新的可视化

## 系统架构

```
用户浏览器
    │
    ├─► http://localhost:5000/query ──────► Flask UI (聊天界面)
    │
    ├─► http://localhost:5000/kg-viz ─────► Flask UI (KG可视化)
    │
    └─► http://localhost:5000/api/v1/* ───► Flask UI 代理 ──► FastAPI (可选)
                                                     │
                                                     └─► http://localhost:8000/api/v1/*
```

## 端口说明

| 服务 | 端口 | 用途 | 启动命令 |
|------|------|------|----------|
| Flask UI | 5000 | Web 界面 | `python run_ui.py` |
| FastAPI | 8000 | API 后端 | `python scripts/start_api.py` |

## 功能对比

### 仅启动 Flask UI (端口 5000)

✅ **可用功能：**
- 聊天界面 (/query)
- 知识图谱可视化 (/kg-viz, /knowledge-graph)
- 静态 KG 数据展示
- 基础会话管理

❌ **不可用功能：**
- 需要 FastAPI 处理的动态 API

### 启动 Flask + FastAPI (端口 5000 + 8000)

✅ **全部功能可用：**
- 聊天界面
- 知识图谱可视化
- 动态 API 接口
- 实时数据处理

## 目录结构

```
doc/
├── run_ui.py              # 原启动脚本
├── run_ui_v2.py           # V2 启动脚本（推荐）
├── 核心库目录/
│   └── ui/
│       ├── app.py         # Flask 主应用
│       ├── templates/
│       │   ├── query.html           # 聊天界面
│       │   ├── knowledge_graph.html # 原 KG 页面
│       │   └── kg_visualization.html ⭐ 新 KG 可视化
│       ├── routers/
│       │   └── kg_visualization.py  # KG API 路由
│       └── static/
└── neuro_core/            # 核心记忆系统
```

## 启动步骤详解

### 步骤 1: 确保依赖安装

```bash
# 安装依赖
pip install flask flask-cors requests

# 如果使用 FastAPI 后端
pip install fastapi uvicorn
```

### 步骤 2: 选择启动方式

**方式 A - 简单启动（仅 UI）：**
```bash
python run_ui_v2.py
```

**方式 B - 完整启动（UI + API）：**

终端 1:
```bash
python scripts/start_api.py
```

终端 2:
```bash
python run_ui_v2.py
```

### 步骤 3: 访问界面

打开浏览器访问：
- http://127.0.0.1:5000/query
- http://127.0.0.1:5000/kg-viz

## 常见问题

### 1. 端口冲突

如果 5000 端口被占用：

```bash
# 使用其他端口
python run_ui_v2.py --port 8080
```

### 2. 启动后页面空白

检查：
1. 是否正确安装了 Flask
2. 模板文件是否存在（核心库 ui/templates 目录）
3. 浏览器控制台是否有错误

### 3. KG 可视化不显示数据

检查：
1. 访问 http://127.0.0.1:5000/api/v1/graph/hierarchical 看是否返回数据
2. 检查浏览器控制台网络请求
3. 确认 `neuro_core` 模块可导入（或会使用模拟数据）

### 4. 如何停止服务

按 `Ctrl + C` 停止 Flask 服务。

## 开发模式

### 调试模式启动

```bash
python run_ui_v2.py --debug
```

### 代码修改自动重载

```bash
# Flask 调试模式会自动重载
export FLASK_ENV=development
python run_ui.py
```

## 生产部署

**注意：** Flask 内置服务器不适合生产环境。

### 使用 Gunicorn

```bash
# 安装
pip install gunicorn

# 启动
gunicorn -w 4 -b 0.0.0.0:5000 run_ui:app
```

### Docker 部署

```dockerfile
FROM python:3.10

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "run_ui_v2.py", "--host", "0.0.0.0"]
```

## 总结

| 场景 | 启动命令 | 访问地址 |
|------|---------|----------|
| 快速体验 | `python run_ui_v2.py` | http://localhost:5000/kg-viz |
| 开发调试 | `python run_ui_v2.py --debug` | http://localhost:5000/query |
| 完整功能 | 同时启动 Flask + FastAPI | http://localhost:5000 + http://localhost:8000 |
| 生产部署 | `gunicorn ...` | - |

**推荐**: 日常使用只需执行 `python run_ui_v2.py` 即可。
