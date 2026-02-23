# 启动后端服务

## 步骤 1: 启动 FastAPI 后端 (端口 8000)

```bash
python scripts/start_api.py
```

或者使用 hypercorn (如果安装了):

```bash
python -m hypercorn scripts.start_api:app --bind 0.0.0.0:8000
```

## 步骤 2: 启动 Flask 前端 (端口 5000)

在另一个终端窗口:

```bash
python run_ui.py
```

## 访问

- 前端: http://localhost:5000
- 后端 API: http://localhost:8000

## 后端连接说明

前端 Flask 会自动将以下请求转发到后端:

1. **聊天查询** → `POST http://127.0.0.1:8000/api/v1/query`
2. **文件上传** → `POST http://127.0.0.1:8000/api/v1/ingest/stream`
3. **其他 API** → 通过通用代理转发

如果后端未启动:
- 聊天会提示: "无法连接到后端服务"
- 文件上传会保存到本地 uploads 目录
