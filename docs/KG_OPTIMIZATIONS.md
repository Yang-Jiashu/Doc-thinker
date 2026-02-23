# 知识图谱：构建 / 变更 / 合并 — 可做优化清单

本文档列出当前实现中**已做**与**可进一步优化**的点，便于查漏补缺。涉及两类图：**主知识图谱（AutoThink 图引擎 + 核心库 KG）** 与 **记忆联想图（neuro_memory）**。

---

## 一、UI 与可视化

| 项目 | 状态 | 说明 |
|------|------|------|
| 主 KG 可视化 | ✅ 已有 | `knowledge_graph.html`：实体/关系、筛选、节点详情、编辑、删边 |
| 记忆图谱可视化 | ✅ 已接 | 同一页面数据源选「记忆图谱」→ 调用 `GET /graph/memory/graph-data` |
| 记忆图 API | ✅ 已有 | `GET /graph/memory/graph-data`、`/memory/stats`、`POST /graph/memory/consolidate` |
| 导出图谱为图片 | ⚠️ 占位 | 当前为占位，可接 html2canvas 或服务端生成 PNG/SVG |
| 布局算法 | ⚠️ 可增强 | 当前为环形布局，可改为力导向（d3-force / vis.js）便于大图浏览 |
| 大图分页/采样 | ❌ 未做 | 节点>500 时可只展示子图或按类型/会话采样，避免前端卡顿 |

---

## 二、KG 构建阶段（插入时）

### 主 KG（实体/关系抽取）

| 项目 | 状态 | 说明 |
|------|------|------|
| 实体消歧 | ✅ 已有 | `knowledge_graph.py` 中按 name/alias 查找已有实体并合并 |
| 关系去重 | ✅ 已有 | 同 (source, target, type) 时更新而非重复插入 |
| MinHash 名称相似 | ✅ 已有 | 用于实体名称相似度 |
| 批量写入存储 | ✅ 已有 | `bulk_save_entities/relationships`、FileKnowledgeGraphStorage |
| 抽取缓存（LLM） | ✅ 已有 | 图引擎/超图侧有 llm_response_cache |
| 多文档并行抽取 | ✅ 已有 | 插入流水线中可多 chunk 并行 |
| 实体类型归一化 | ⚠️ 可增强 | 类型为自由文本时可做枚举或 LLM 归一化，减少碎片化 |
| 关系类型归一化 | ⚠️ 可增强 | 同上，关系类型归一化 + 同义合并 |
| 低置信度过滤 | ⚠️ 可增强 | 对 confidence 低于阈值的实体/关系不入库或单独标记 |
| 增量抽取 | ✅ 已有 | 按 doc/chunk 增量，只处理新内容 |

### 记忆图（neuro_memory）

| 项目 | 状态 | 说明 |
|------|------|------|
| Episode 写入即联想 | ✅ 已有 | `add_observation` 中与已有 episode 建 EPISODE_SIMILARITY 边 |
| 与 entity/chunk 建边 | ✅ 已有 | entity_ids、raw_text_refs 写入时建 CONCEPT_LINK / SAME_DOCUMENT |
| content/graph embedding | ✅ 已有 | 用于相似度与结构类比 |
| 去重 | ✅ 已有 | episode_id 由内容 hash 生成，同内容同 id |

---

## 三、KG 变更（更新 / 删除）

| 项目 | 状态 | 说明 |
|------|------|------|
| 实体更新 API | ✅ 已有 | `PUT /knowledge-graph/entity/{entity_name}` |
| 关系删除 API | ✅ 已有 | `DELETE /knowledge-graph/relationship` |
| 实体删除 | ⚠️ 部分 | 可按实体删除关系，需确认是否级联删实体或保留孤立节点策略 |
| 记忆图节点/边删除 | ❌ 未做 | 当前无 API；可按需加「删除某 episode/边」接口 |
| 软删除 / 版本 | ❌ 未做 | 删除可改为标记 + 按版本查询，便于回溯 |
| 变更审计日志 | ❌ 未做 | 谁在何时改了什么，便于排查与回滚 |

---

## 四、KG 合并（多源、会话、巩固）

| 项目 | 状态 | 说明 |
|------|------|------|
| 全局 vs 会话 KG | ✅ 已有 | 数据源选 global / session_id，分别查图引擎与 session 图 |
| 会话级实体/关系写入 | ✅ 已有 | ingest 时 `_update_local_knowledge_graph` 等 |
| 跨文档实体合并 | ✅ 已有 | ingest 中基于 name/type 找已有实体并建 related_to/analogous_to |
| 记忆图巩固 | ✅ 已有 | `consolidate()`：重放、相似配对、LLM 跨事件推断、ANALOGOUS_TO/SAME_THEME 边 |
| 主 KG 与记忆图联动 | ⚠️ 部分 | 记忆图用 entity_ids 与主 KG 实体对应，未做双向同步（主 KG 改实体未反写记忆图） |
| 多会话合并为全局 | ⚠️ 需策略 | 会话图合并到全局时的冲突策略（以新为准 / 合并关系 / 人工审核） |
| 主题/图式节点 | ❌ 未做 | 巩固可产出「主题」虚拟节点并挂 episode，当前未持久化主题节点 |
| 边权衰减 | ❌ 未做 | 长期未激活的边可随时间或访问次数衰减，避免图过稠 |

---

## 五、查询与检索

| 项目 | 状态 | 说明 |
|------|------|------|
| 向量检索 chunk/entity | ✅ 已有 | 图引擎 / 超图的 top-k 检索 |
| 图扩散激活 | ✅ 已有 | neuro_memory `spreading_activation`，按边类型衰减 |
| 类比检索 | ✅ 已有 | content + structure + salience 综合打分 |
| 查询时类比上下文 | ✅ 已有 | 启用 thinking 时先 `retrieve_analogies` 再拼入 prompt |
| 主 KG 子图展开 | ✅ 已有 | 按实体/关系取邻接，用于 RAG 上下文 |
| 混合检索（BM25+向量） | ✅ 已有 | HyperGraph 侧 entity/relation/chunk 的 hybrid |
| 重排序 | ✅ 已有 | rerank_model_func |
| 查询结果写回记忆 | ✅ 已有 | 对话后 `add_observation`，检索到的 episode 更新 retrieval_count |

---

## 六、存储与性能

| 项目 | 状态 | 说明 |
|------|------|------|
| 主 KG 持久化 | ✅ 已有 | 文件 / 可选外部存储抽象 |
| 记忆图持久化 | ✅ 已有 | episodes.json、memory_graph.json、episode_vectors.json |
| 向量库 | ✅ 已有 | FAISS / Nano / 等，chunk/entity 向量 |
| 记忆图向量 | ✅ 已有 | 内存 EpisodeVectorStore，可换为外部向量库 |
| 图库（Neo4j 等） | ⚠️ 可选 | 主 KG / 记忆图可接 Neo4j 等，当前为内存/文件 |
| 大批量插入批处理 | ✅ 已有 | 批量 upsert、index_done_callback |
| 增量索引 | ✅ 已有 | 只处理新 doc/chunk |

---

## 七、可补充的优化（未实现）

1. **实体/关系类型归一化**：抽取时或后处理用枚举或 LLM 将类型映射到统一 schema，减少同义类型碎片。
2. **置信度过滤与复审**：低置信度实体/关系单独表或标记，支持人工复审后再入主图。
3. **记忆图边权衰减**：按时间或「从未被扩散激活」对边做衰减或剔除，控制图规模。
4. **主题节点持久化**：巩固时聚类得到的主题写成节点，episode–主题边写入图并参与可视化与检索。
5. **主 KG ↔ 记忆图双向同步**：主 KG 实体更名/合并时，同步更新记忆图中 entity_ids 或边。
6. **图谱导出**：导出为 PNG/SVG/GraphML，便于汇报与迁移。
7. **力导向布局**：前端换用 d3-force 或 vis.js 力导向，大图可读性更好。
8. **变更审计与软删除**：关键变更写日志；删除改为软删除 + 版本视图。
9. **多会话合并策略**：明确「会话 → 全局」合并时的冲突规则与可选的人工审核流程。

---

## 八、小结

- **已有**：主 KG 与记忆图的可视化入口、记忆图 API、巩固与类比检索、主 KG 的实体消歧/关系去重/增量写入、会话与全局双数据源。
- **建议优先做**：类型归一化、记忆图边权衰减、图谱导出、力导向布局、记忆图删除/编辑 API（若需运营）。
- **按需做**：主题节点、主 KG 与记忆图双向同步、审计与软删除、多会话合并策略。

如需对某一类（例如「仅记忆图」或「仅主 KG 构建」）做成分解与排期，可以指定范围再细化。
