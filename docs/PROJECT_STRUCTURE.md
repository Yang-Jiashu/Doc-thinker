# Doc Thinker / AutoThink 项目结构说明

> **从文档到检索的完整代码与文档总览**：见 [CODE_AND_DOCS_OVERVIEW.md](CODE_AND_DOCS_OVERVIEW.md)（文档、解析、入库、查询、各类 RAG 职责与数据流）。  
> **各文件夹用途与整理建议**：见 [FOLDERS.md](FOLDERS.md)（必留/可选/可删目录、.gitignore 建议）。

## 一、入口与启动方式

### 1. UI（Flask，端口 5000）

**推荐唯一启动方式（在项目根目录 `doc` 下）：**

```bash
python run_ui.py
```

或从项目根运行核心库的 UI 入口（见仓库结构）。

- 聊天页: http://127.0.0.1:5000/query  
- 知识图谱: http://127.0.0.1:5000/knowledge-graph  
- 模板目录为核心库下的 `ui/templates/`，由 Jinja 指定。

### 2. 后端 API（FastAPI，端口 8000）

```bash
python scripts/start_api.py
```

- 会话、检索、知识图谱/记忆图谱等接口挂载在 `/api/v1/` 下；Flask UI 会代理到该后端。

---

## 二、UI 相关目录（唯一一套）

| 路径 | 说明 |
|------|------|
| 核心库/ui/app.py | Flask 应用：路由、API 代理、HTML 注入「知识图谱」 |
| 核心库/ui/templates/ | 所有页面模板（唯一模板来源） |
| 核心库/ui/templates/base.html | 全站布局；侧栏 + 固定右上角「知识图谱」入口 |
| 核心库/ui/templates/query.html | 聊天页 |
| 核心库/ui/templates/knowledge_graph.html | 知识图谱 / 记忆图谱可视化 |
| 核心库/ui/static/ | 静态资源 |

- 已删除：项目根目录下重复的 `templates/`、`output.html`。  
- 已移除：`app.py` 内约 764 行“默认模板”死代码，不再写入或覆盖任何模板。

---

## 三、后端与核心模块

| 路径 | 说明 |
|------|------|
| 核心库/server/app.py | FastAPI 应用、生命周期、路由挂载 |
| 核心库/server/routers/ | 健康、会话、入库、查询、图谱等路由 |
| 核心库 | AutoThink 核心：检索、知识图谱、会话、认知处理、自动思考、超图等 |
| `neuro_memory/` | 类脑记忆（扩散激活、巩固、类比检索等） |

---

## 四、「知识图谱」入口（三重保障）

1. **base.html**：侧栏「新会话」下方有「知识图谱」链接；`<body>` 开头有固定右上角按钮。  
2. **app.py 注入**：对所有返回的 HTML 在 `</body>` 前注入同一固定按钮，不依赖模板内容。  
3. **直接访问**：任意时刻可打开 http://127.0.0.1:5000/knowledge-graph 。

---

## 五、启动顺序建议

1. 先启动后端：`python scripts/start_api.py`  
2. 再启动 UI：`python run_ui.py`  
3. 浏览器访问：http://127.0.0.1:5000/query ，右上角或侧栏应可见「知识图谱」。
