# 类人脑联想与记忆重连算法设计

## 一、人脑思考的几条原则（作为算法约束）

1. **扩散激活 (Spreading Activation)**  
   想到一个概念时，与之相连的概念会被部分激活；激活沿边传播并随跳数衰减，多条路径可叠加。

2. **巩固与重放 (Consolidation & Replay)**  
   在“离线”（如后台任务）重放近期与高显著性记忆，重组联结：常用的联结加强，无关的弱化；跨经历的关系被推断出来。

3. **结构映射与类比 (Structure Mapping)**  
   类比不只靠“表面相似”，而是“关系结构相同”：A→B 的关系与 C→D 的关系同构时，才会被当作类比。

4. **显著性 (Salience)**  
   新近的、被多次检索的、或与目标高度相关的记忆更容易被激活；重要/意外事件权重要更高。

5. **图式与主题 (Schema / Theme)**  
   记忆会抽象成“图式”（如“一次会议”“一次并购”）；新经历会归入已有主题或形成新主题，便于类比与检索。

6. **区分与去混淆 (Differentiation)**  
   非常相似的记忆要显式记录“关键区分点”，避免检索时混淆。

---

## 二、算法模块

### 2.1 扩散激活 (Spreading Activation)

**输入**  
- 种子集合：节点 ID 列表（实体、chunk、或 episode）及初始激活值（可选）。  
- 可选：query 的 embedding，用于对“语义相关边”做加权。

**过程**  
- 初始：种子节点激活值 = 1.0（或给定值）。  
- 每轮传播：  
  - 对每条边 (u, v)，传递 `activation(u) * edge_weight * decay_per_hop(hop)`。  
  - 同一节点从多条路径得到的激活**相加**（叠加）。  
- 边类型与衰减：  
  - `semantic_similarity`：衰减慢（如 0.85^hop），表示“想到就易联想到”。  
  - `same_document`：衰减快（如 0.6^hop），表示“同一文档内邻近”。  
  - `concept_link` / `inferred_relation`：中等衰减。  
  - `episode_similarity`：跨事件类比，衰减同 semantic。  
- 最大跳数：2～3，避免过度扩散。  
- 可选：用 query 与节点/边的 embedding 相似度对边权做临时放大（检索时“与问题相关的路径”更容易被激活）。

**输出**  
- 按激活值排序的节点列表，以及（可选）每条边的贡献，用于可解释性。

---

### 2.2 记忆巩固 (Memory Consolidation)

**触发**  
- 每插入 N 条新 observation 后触发一次；或定时（如每小时）做轻量巩固。

**步骤**  

1. **重放采样**  
   - 从“近期 + 高显著性”的 episode 中抽样（如最近 50 条 + retrieval_count 高的 20 条）。  
   - 随机打乱顺序（模拟睡眠中的随机重放）。

2. **结构相似配对**  
   - 对 episode 两两（或与聚类中心）计算：  
     - 内容相似度（content_embedding）；  
     - 结构相似度（见下文的“结构描述 → embedding”）。  
   - 取 content_sim 与 structure_sim 都超过阈值的配对，进入下一步。

3. **跨事件关系推断**  
   - 对每对高相似 (A, B)，用 LLM：  
     - 输入：A 的 summary + 关键实体/关系，B 的 summary + 关键实体/关系。  
     - 输出：是否同一主题、是否可类比、以及“A 中 X 与 B 中 Y 在角色上对应”等。  
   - 根据 LLM 结果：在知识图中添加跨 episode 的关系边或“类比边”；在 episode 图上添加 `ANALOGOUS_TO` 边。

4. **权重更新**  
   - 在巩固窗口内被共同激活过的 (episode, entity) 或 (episode, episode) 边：weight += delta。  
   - 可选：对长期未被检索的边做轻微衰减（decay），避免图过于稠密。

5. **主题/图式更新**  
   - 对 episode 的 summary/concepts 做聚类（或用 LLM 归纳“主题标签”）。  
   - 形成“主题”虚拟节点，将 episode 连到主题；新 episode 可挂到已有主题或新建主题。

---

### 2.3 类比检索 (Analogical Retrieval)

**目标**  
- 给定 query（或新 episode），不仅用向量相似度，还用**关系结构**找“历史上类似的事件”。

**步骤**  

1. **Query 侧**  
   - 用 LLM 或已有 NER/关系抽取得到：实体集合 E_q、关系集合 R_q（(s, r, t) 三元组）。  
   - 生成“结构描述”字符串（如：`Entities: 2 org, 1 person. Relations: works_at(2), located_in(1).`）。  
   - 对 (query_text + 结构描述) 做 embedding → query_embedding；可选单独对结构描述做 embedding → query_structure_embedding。

2. **候选 Episode**  
   - 用 content_embedding 做向量检索，取 top-K（如 30）。  
   - 再用 structure_embedding 做一次检索或重排，得到“结构也像”的 subset。

3. **结构对齐打分**  
   - 对每个候选 episode，取其局部图的关系类型序列（或 (source_type, relation_type, target_type) 的集合）。  
   - 与 query 的 R_q 做“结构匹配”：  
     - 简化版：关系类型集合的 Jaccard，或  
     - 用“结构描述字符串”的 embedding 相似度作为 structure_score。  
   - 最终得分：  
     `score = α * content_sim + β * structure_sim + γ * salience`  
     （salience 可为 recency + retrieval_count 的归一化。）

4. **区分信息（可选）**  
   - 当 top-2 两个 episode 非常相似时，调用 LLM 生成一句“关键区分点”，附在返回结果中，便于展示与去混淆。

---

### 2.4 新记忆写入时的即时联想 (On-Insert Association)

**在 `add_observation` 内**  

1. **微观（chunk/实体级）**  
   - 用新内容的 embedding 在 chunks_vdb / entities_vdb 中检索 top-k。  
   - 对命中项与当前 episode 建边（或加强已有边），边类型 `semantic_similarity`，权重 = 相似度。

2. **中观（episode 级）**  
   - 当前 episode 的 content_embedding 写入 episodes_vdb。  
   - 在 episodes_vdb 中检索 top-k 相似旧 episode，建 `EPISODE_SIMILARITY` 边，权重 = 相似度。

3. **宏观（结构）**  
   - 为当前 episode 生成 graph_embedding（局部图的结构描述 → embed），存入 episode 记录。  
   - 巩固阶段再批量做 graph-graph 配对与跨事件推断。

---

## 三、数据结构约定

- **Episode**  
  - episode_id, timestamp, source_type, session_id；  
  - summary, key_points, concepts；  
  - entity_ids, relation_triples（或 relation_ids）；  
  - content_embedding, graph_embedding；  
  - raw_text_refs（chunk_id / doc_id 列表）；  
  - retrieval_count, last_retrieved_at（用于显著性）。

- **边类型**  
  - SEMANTIC_SIMILARITY, SAME_DOCUMENT, CONCEPT_LINK, INFERRED_RELATION, EPISODE_SIMILARITY, ANALOGOUS_TO, SAME_THEME。

- **存储**  
  - Episode 列表：KV（如 JSON 或 SQLite）；  
  - Episode 向量：episodes_vdb（content + 可选 graph）；  
  - 联想图：节点 = episode_id | entity_id | chunk_id；边 = 上述类型 + weight。可用现有 KnowledgeGraph / chunk_entity_relation_graph 扩展，或单独一张 memory_graph。

---

## 四、与现有 RAG 的衔接

- **写入**  
  - 文档/对话 ingest 完成后，用 CognitiveProcessor 的 insight + KG + chunk 信息调用 `MemoryEngine.add_observation(...)`。  
- **巩固**  
  - 在 ingest 批次结束或定时任务中调用 `MemoryEngine.consolidate()`。  
- **查询**  
  - 在 RAG 检索前调用 `MemoryEngine.retrieve_analogies(query)`，将返回的 episode/chunk 上下文拼进 prompt。

这样既保留现有 RAG 的 chunk/entity 级能力，又增加“事件级联想 + 结构类比 + 巩固重连”的人脑式记忆层。
