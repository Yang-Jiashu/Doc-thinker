# 如何验证 neuro_memory 是否好用

## 一、本地脚本验证（推荐先做）

在项目根目录执行：

```bash
# 1. 仅用内存 mock，不调 API，最快
python scripts/verify_neuro_memory.py

# 2. 使用真实 embedding（需配置 .env 里 OPENAI_API_KEY 或 LLM_BINDING_API_KEY）
python scripts/verify_neuro_memory.py --embed

# 3. embedding + 巩固时用 LLM 做跨事件推断
python scripts/verify_neuro_memory.py --embed --llm
```

**Windows 下若中文乱码**，可先执行：
```bash
chcp 65001
python scripts/verify_neuro_memory.py
```

**看什么算“好用”：**

- 能写入 4 条经历，并打印出 `episode_id`。
- “类比检索”能返回与问题相关的经历（例如问「公司并购后如何保留人才」应优先命中两条并购相关经历）。
- 第二次检索「季度目标会议」应优先命中两条会议相关经历。
- 使用 `--embed` 时，语义更准，相似经历得分更高；使用 `--llm` 时，巩固后边数可能增加（跨事件类比边）。

---

## 二、在完整服务里验证

### 1. 启动服务

```bash
# 在项目根目录
python scripts/start_api.py
# 或你当前的启动方式
```

### 2. 看记忆引擎是否启用

```bash
curl http://localhost:8000/graph/memory/stats
```

期望：`"enabled": true`，并返回 `episodes`、`edges`（初始可为 0）。

### 3. 通过对话产生记忆

- 用任意前端或 Postman 调用 `POST /query`（或 `/query/text`），带 `session_id`，多轮对话几段**有主题**的内容，例如：
  - 第一轮：问「我们公司最近收购了某团队，怎么整合比较好？」
  - 第二轮：问「上次说的整合，技术栈不一致怎么办？」
  - 第三轮：问「下周要开季度目标会，需要准备什么？」
- 每次问答后，后端会把本轮对话做认知分析并写入 `memory_engine`（若已启用）。

### 4. 查记忆状态

```bash
curl http://localhost:8000/graph/memory/stats
```

期望：`episodes` > 0，可能已有 `edges`。

### 5. 手动触发巩固（可选）

```bash
curl -X POST "http://localhost:8000/graph/memory/consolidate?recent_n=30&run_llm=true"
```

期望：返回 `success: true` 及 `edges_added`、`pairs_processed`。再次查 `/graph/memory/stats` 时边数可能增加。

### 6. 验证“类比检索”是否参与回答

- 再问一个**和之前几段经历都沾边**的问题，例如：「之前我们聊过并购整合和季度会议，现在要写一份总结，该包含哪些点？」
- 若记忆引擎工作正常，且启用了 Auto-Thinking，返回里应能看到 **「类比记忆」** 相关前缀或思路，把“并购”和“会议”两段经历都拉进来。

---

## 三、自检清单

| 项目 | 说明 |
|------|------|
| 脚本能跑通 | `python scripts/verify_neuro_memory.py` 无报错，有 4 条经历写入与两次类比检索结果 |
| 类比检索有结果 | 问并购 → 优先出现并购经历；问会议 → 优先出现会议经历 |
| 服务里 enabled | `GET /graph/memory/stats` 返回 `enabled: true` |
| 对话后 episodes 增加 | 多轮对话后再次查 stats，`episodes` 增大 |
| 巩固后边数变化 | 调用 `POST /graph/memory/consolidate` 后，stats 里 `edges` 可能增加 |
| 回答里出现类比 | 问跨多段经历的综合问题时，回答中能体现“历史经历/类比”内容 |

若以上都符合预期，可以认为 neuro_memory 的写入、联想、巩固与类比检索在你这套环境里是“好用”的。
