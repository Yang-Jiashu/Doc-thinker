# NeuroAgent 启动方式

## 1. 最简单的方式（推荐）

只启动 Flask UI，包含所有界面功能：

```bash
python run_ui_v2.py
```

然后浏览器打开：
- http://127.0.0.1:5000/query - 聊天界面
- http://127.0.0.1:5000/kg-viz - 知识图谱可视化（新）

## 2. 完整启动（UI + 后端 API）

如果需要完整的 API 功能，需要同时启动两个服务：

**终端 1 - 启动后端 API：**
```bash
python scripts/start_api.py
```

**终端 2 - 启动前端 UI：**
```bash
python run_ui_v2.py
```

## 3. 启动脚本对比

| 脚本 | 用途 | 端口 | 功能 |
|------|------|------|------|
| `run_ui.py` | 原启动脚本 | 5000 | 基础 UI |
| `run_ui_v2.py` | 新启动脚本 | 5000 | UI + KG可视化 |

## 4. 端口占用怎么办？

```bash
# 使用其他端口
python run_ui_v2.py --port 8080
```

## 5. 停止服务

按 `Ctrl + C` 即可停止。

---

**总结：日常使用只需执行 `python run_ui_v2.py`，然后访问 http://127.0.0.1:5000/kg-viz 即可。**
