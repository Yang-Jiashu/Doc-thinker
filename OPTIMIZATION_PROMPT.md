# DocThinker System Optimization Prompt

> 你是一名顶级 RAG 系统架构师和 Prompt 工程师。
> 以下是 DocThinker 知识图谱系统的全面优化方案。
> 目标：**更快、更清晰、更深度、可观测、边自动建立且自我校验**。

---

## 〇、总体原则

```
速度 ← 并行化 + 批量化 + 缓存 + 精简 token
清晰 ← 结构化 JSON schema + 管道化阶段 + 单一职责
深度 ← Chain-of-Thought + 多角度 + 自一致性校验
校验 ← LLM 自检 + embedding 交叉验证 + 置信度打分
建边 ← 生成节点时强制生成边 + 回写验证
可观测 ← 每次 LLM 调用写 trace → JSON 文件 + 结构化日志
```

---

## 一、LLM 可观测性层 (Trace Layer)

### 问题
当前系统中**所有 LLM 调用都是黑箱**：expander、classifier、decomposer、consolidation 均无输入/输出日志。出错时无法定位是 prompt 问题还是模型问题。

### 方案：统一 Trace 中间件

在每个 LLM 调用点注入 `TracedLLM` 包装器，自动记录：

```python
import json, time, logging, hashlib
from pathlib import Path
from typing import Any, Callable, Optional

_trace_log = logging.getLogger("docthinker.llm_trace")

class LLMTrace:
    """每次 LLM 调用的完整记录。"""
    def __init__(self, *, stage: str, sub_stage: str = "",
                 session_id: str = "", trace_dir: Optional[Path] = None):
        self.stage = stage           # e.g. "expansion", "extraction", "query"
        self.sub_stage = sub_stage   # e.g. "cluster_expand", "top_node_expand"
        self.session_id = session_id
        self.trace_dir = trace_dir or Path("data/_traces")
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    async def __call__(self, llm_func: Callable, prompt: str, *,
                       system_prompt: str = "", metadata: dict = None) -> str:
        call_id = hashlib.md5(f"{time.time()}{prompt[:100]}".encode()).hexdigest()[:12]
        t0 = time.perf_counter()

        _trace_log.info("[%s/%s] call_id=%s | prompt_chars=%d",
                        self.stage, self.sub_stage, call_id, len(prompt))

        try:
            import asyncio
            if asyncio.iscoroutinefunction(llm_func):
                response = await llm_func(prompt)
            else:
                response = llm_func(prompt)
            elapsed = time.perf_counter() - t0
            status = "ok"
        except Exception as e:
            elapsed = time.perf_counter() - t0
            response = ""
            status = f"error: {e}"

        trace_record = {
            "call_id": call_id,
            "stage": self.stage,
            "sub_stage": self.sub_stage,
            "session_id": self.session_id,
            "timestamp": time.time(),
            "elapsed_s": round(elapsed, 3),
            "status": status,
            "prompt_chars": len(prompt),
            "response_chars": len(response),
            "prompt_preview": prompt[:500],   # 前 500 字符预览
            "response_preview": response[:500],
            "prompt_full": prompt,
            "response_full": response,
            "metadata": metadata or {},
        }

        # 写入 trace 文件 (每个 session 一个 JSONL)
        trace_file = self.trace_dir / f"{self.session_id or 'global'}_traces.jsonl"
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_record, ensure_ascii=False) + "\n")

        _trace_log.info("[%s/%s] call_id=%s | %.2fs | status=%s | resp_chars=%d",
                        self.stage, self.sub_stage, call_id, elapsed, status, len(response))

        if status != "ok":
            raise RuntimeError(f"LLM call failed: {status}")
        return response
```

### 使用方式
```python
# 在 expander.py 中
tracer = LLMTrace(stage="expansion", sub_stage="cluster_expand", session_id=sid)
response = await tracer(self.llm_func, prompt, metadata={"cluster_id": i, "node_count": len(nodes)})

# 在 classifier.py 中
tracer = LLMTrace(stage="auto_thinking", sub_stage="complexity_classify", session_id=sid)
response = await tracer(self.vlm.generate, prompt)
```

### Trace 查看
- 文件位置：`data/_traces/{session_id}_traces.jsonl`
- 每行一个 JSON，可直接 `jq` 查询或前端渲染
- 可加 API 端点 `GET /api/v1/traces?session_id=X&stage=expansion` 返回 trace 列表

---

## 二、KG 扩展 Prompt 重写 (核心改动)

### 当前问题

1. **扩展出的节点是空壳**：只有 `{"entity": "xxx", "reason": "xxx"}`，无描述、无边、无类型
2. **7 个角度串行调用**：每次扩展 7 次 LLM 串行 → 慢
3. **无自我校验**：LLM 生成什么就是什么，无一致性检查
4. **无自动建边**：节点没有边，孤岛节点对检索无意义

### 新 Prompt 体系

#### Prompt A：密度聚类摘要后的扩展 (Cluster-Based Expansion)

```
你是一名知识图谱构建专家。

## 背景
以下是从文档中提取的一组语义紧密的实体簇（通过向量密度聚类自动发现）：

### 簇摘要
{cluster_summary}

### 簇内实体
{cluster_entities_with_descriptions}

### 当前图谱中已有的所有实体（禁止重复）
{all_existing_entities}

## 任务
基于此簇的主题，推理出 **尚未存在于图谱中** 的相关知识。你需要：
1. 思考这个簇代表的核心主题是什么
2. 从以下角度联想新知识：
   - 该主题的前提条件或理论基础
   - 该主题的实际应用或案例
   - 该主题的相关对比概念
   - 该主题的演进历史或最新发展
3. 为每个新实体提供**具体的、有事实依据的描述**（不少于 20 字）
4. 为每个新实体指明它与簇内**哪个已有实体**存在关系，并说明关系类型

## 输出格式
严格输出 JSON 数组，不要其他文字：
```json
[
  {
    "entity": "新实体名称",
    "entity_type": "concept|person|technology|method|organization|event|other",
    "description": "该实体的具体描述，包含关键事实，不少于20字",
    "edges": [
      {
        "target": "簇内已有实体名",
        "relation": "关系类型关键词",
        "description": "为什么存在这个关系的简要说明"
      }
    ]
  }
]
```

## 质量要求
- 数量：不少于 {min_count} 个
- 描述必须具体：❌ "一种算法" → ✅ "一种基于密度的聚类算法，由Ester等人1996年提出，核心思想是..."
- 每个实体至少有 1 条边连接到簇内已有实体
- 不得与已有实体重复或高度相似
```

#### Prompt B：高权重核心节点扩展 (Top-Node Expansion)

```
你是一名知识图谱构建专家，擅长知识推理和关联发现。

## 背景
以下是知识图谱中连接最紧密的核心实体（按连接度排序的前 {top_n} 个）：

{top_nodes_with_descriptions}

### 当前图谱中已有的所有实体（禁止重复）
{all_existing_entities}

## 任务
围绕这些核心实体，从 **6 个认知维度** 推理新知识：

| 维度 | 说明 | 示例 |
|------|------|------|
| 层级关系 | 上位概念、下位概念、所属领域 | "机器学习" → 上位: "人工智能"; 下位: "强化学习" |
| 因果关联 | 前置条件、后续影响、因果链 | "过拟合" → 因: "训练数据不足"; 果: "泛化能力下降" |
| 类比迁移 | 其他领域的相似概念 | "注意力机制" ↔ "人类选择性注意" |
| 对立互补 | 相对立或互补的概念 | "监督学习" ↔ "无监督学习" |
| 时间演进 | 历史渊源、版本迭代、未来趋势 | "BERT" → "GPT" → "LLaMA" |
| 应用实践 | 具体场景、工具、案例 | "Transformer" → 应用: "机器翻译、文本摘要" |

## 输出格式
严格输出 JSON 数组：
```json
[
  {
    "entity": "新实体名称",
    "entity_type": "concept|person|technology|method|organization|event|other",
    "description": "具体描述（含关键事实，不少于20字）",
    "dimension": "层级关系|因果关联|类比迁移|对立互补|时间演进|应用实践",
    "edges": [
      {
        "target": "已有核心实体名",
        "relation": "关系关键词",
        "description": "关系说明"
      }
    ]
  }
]
```

## 质量要求
- 数量：不少于 {min_count} 个
- 每个维度至少覆盖 2 个新实体
- 描述必须具体且有事实依据
- 每个实体至少 1 条边
- 禁止生成已有实体的变体（如 "XX的应用"、"XX方法"）
```

#### Prompt C：扩展节点自我校验 (Self-Validation)

```
你是一名知识图谱质量审核员。

## 任务
以下是 LLM 自动扩展生成的候选实体列表。请逐一审核：

{candidate_entities_json}

## 审核标准
对每个候选实体，检查：
1. **事实性**：描述是否包含可验证的事实？（0-1分）
2. **非冗余性**：是否与以下已有实体语义重复？（0-1分）
   已有实体：{existing_entities}
3. **边有效性**：声称的关系是否合理？目标实体是否存在？（0-1分）
4. **具体性**：描述是否足够具体，还是空泛的废话？（0-1分）

## 输出格式
```json
[
  {
    "entity": "候选实体名",
    "factuality": 0.8,
    "non_redundancy": 0.9,
    "edge_validity": 1.0,
    "specificity": 0.7,
    "overall_score": 0.85,
    "verdict": "accept|revise|reject",
    "revision_note": "如果 verdict=revise，说明需要修改什么"
  }
]
```

只输出 JSON，不要其他文字。
```

### 执行流程

```
 ┌─────────────────────────────────────────────────────────┐
 │  Expansion Pipeline (手动触发)                           │
 │                                                         │
 │  Step 1: 加载聚类摘要 (ingest 时已生成)                   │
 │      ↓                                                  │
 │  Step 2: 并行调用 Prompt A (每个簇一个) ← asyncio.gather  │
 │      ↓                                                  │
 │  Step 3: 并行调用 Prompt B (top-50 节点)                  │
 │      ↓                                                  │
 │  Step 4: 合并 + 字面去重 + 语义去重 (embedding cosine)    │
 │      ↓                                                  │
 │  Step 5: 自我校验 Prompt C → 过滤 overall_score < 0.6    │
 │      ↓                                                  │
 │  Step 6: 写入图谱 (node + edge) + VDB + ExpandedManager  │
 │      ↓                                                  │
 │  Step 7: 写 trace → data/_traces/{session}_traces.jsonl  │
 └─────────────────────────────────────────────────────────┘
```

**速度优化关键**：
- Step 2/3 全部用 `asyncio.gather` 并行
- Step 5 校验可用 batch（一次性送所有候选给 LLM，而非逐个）
- 如果候选 > 50 个，分批校验（每批 25 个）

---

## 三、密度聚类 Prompt (Ingest 阶段)

### 聚类摘要生成 Prompt

```
你是一名知识分析专家。

## 任务
以下是通过密度聚类算法自动发现的一组语义紧密的实体（{node_count} 个）：

{entities_with_descriptions}

请用 2-4 句话总结这组实体共同代表的主题、核心概念和它们之间的关系模式。

## 输出格式
直接输出摘要文本，不要 JSON 包装，不要标题。
```

### 聚类参数建议
```python
HDBSCAN: min_cluster_size=4, min_samples=2, metric='cosine'
DBSCAN (fallback): eps=0.35, min_samples=3, metric='cosine'
```

---

## 四、Query 阶段 Deep Mode 优化

### 当前问题
- `match_nodes` 用 token 重叠匹配 → 准确率低
- `build_forced_instruction` 过于简单 → LLM 经常忽略
- 无 trace → 不知道哪些扩展节点被使用了

### 改进后的 Expanded Node 匹配

```python
async def match_nodes_v2(self, query: str, *, top_k=3, embedding_func=None):
    """用 embedding 相似度替代 token 重叠。"""
    if not embedding_func:
        return self.match_nodes(query, top_k=top_k)  # fallback

    q_emb = await embedding_func([query])
    candidates = []
    for key, item in self._records.items():
        if item.get("status") == "deprecated":
            continue
        corpus = f"{item['entity']} {item.get('description', '')} {item.get('reason', '')}"
        c_emb = await embedding_func([corpus])
        sim = cosine_similarity(q_emb[0], c_emb[0])
        if sim > 0.4:
            candidates.append({**item, "score": sim})

    candidates.sort(key=lambda x: -x["score"])
    return candidates[:top_k]
```

### 改进后的 Forced Instruction

```
## 扩展知识参考
系统通过知识图谱自我进化生成了以下扩展知识节点，它们与当前问题高度相关。
请在回答中优先核对这些知识，如果它们与你的分析一致，请自然地融入回答中并标注来源。

{for each matched node}
### 节点: {entity}
- 类型: {entity_type}
- 描述: {description}
- 关联实体: {edges → target (relation)}
- 匹配置信度: {score}
{end for}

注意：
1. 只采纳与问题确实相关的节点
2. 如果某个节点的信息与文档事实矛盾，请忽略该节点并在回答末尾注明
3. 被采纳的节点将获得更高权重，未被采纳的将逐渐衰减
```

---

## 五、Entity Extraction Prompt 优化

### 当前问题
- Prompt 过长（~100 行英文 system prompt）→ 消耗 token
- 中英混合场景下 entity_type 不统一
- 无 gleaning 触发条件说明

### 优化建议

**精简 System Prompt**：保留核心指令，删除冗余示例说明。当前 prompt 中有大量格式重复说明（entity 格式重复 3 次，relation 格式重复 2 次），应合并为一次清晰的说明。

**添加最小提取要求**：
```
## 最低提取标准
- 实体描述不少于 10 个字
- 关系描述不少于 8 个字
- 每个 chunk 至少提取 3 个实体和 2 个关系（除非文本确实无内容）
- 实体名称使用完整名称，避免缩写（除非缩写是通用术语）
```

**提取后增加 merge-aware 指令**：
```
## 命名一致性
当同一实体在文本中出现多种称呼时（如 "RT-DETR" 和 "Real-Time DEtection TRansformer"），
选择最常用的名称作为 entity_name，并在 description 中注明其他别名。
```

---

## 六、自动建边逻辑 (Auto Edge Building)

### 当前问题
- 扩展节点只在 **promotion** 时才建边（太晚了）
- 建边逻辑简单：只有 `expanded_from_root` 和 `co_mentioned`
- 无边类型丰富度

### 改进：扩展时即建边

```python
async def _apply_expansion_with_edges(self, candidates, graph, gc):
    """扩展时立即建立节点和边，并同步到 VDB。"""
    for c in candidates:
        entity = c["entity"]
        node_data = {
            "entity_id": entity,
            "entity_type": c.get("entity_type", "concept"),
            "description": c.get("description", ""),
            "source_id": "llm_expansion",
            "is_expanded": "1",
        }
        await graph.upsert_node(entity, node_data)

        for edge in c.get("edges", []):
            target = edge["target"]
            if await graph.has_node(target):
                edge_data = {
                    "keywords": edge.get("relation", "related_to"),
                    "description": edge.get("description", ""),
                    "source_id": "llm_expansion",
                    "weight": 0.6,  # 初始权重低于原始边
                }
                await graph.upsert_edge(entity, target, edge_data)
            else:
                _log.warning("[expand] edge target '%s' not in graph, skip", target)
```

### 边权重生命周期

```
初始权重 = 0.6 (低于原始边的 1.0)
每次被查询命中 → +0.15
每次被 LLM 回答引用 → +0.25
每次查询未命中 → -0.05
权重 < 0.1 → 标记为 deprecated
权重 > 1.2 → promote 为 permanent (等同原始边)
```

---

## 七、并行化改造

### 当前串行瓶颈

| 位置 | 当前行为 | 耗时 |
|------|----------|------|
| `expander.expand()` | 7 个角度**串行** `for i in indices` | 7 × LLM_time |
| `_run_density_clustering()` | 聚类后**串行**生成摘要 | N × LLM_time |
| `match_nodes()` | 逐个 token 匹配 | O(n × m) |
| `consolidation.consolidate()` | 逐对 episode 比较 | O(n²) |

### 改造方案

```python
# expander.py - 并行调用所有角度
async def expand(self, nodes_data, edges_data, **kwargs):
    # 构建所有 prompt
    prompts = []
    for i in indices:
        angle_name, angle_hint = self.angles[i]
        prompts.append((angle_name, self._build_prompt(angle_name, angle_hint, ...)))

    # 并行调用
    async def _call_one(angle_name, prompt):
        resp = await self._call_llm(prompt)
        parsed = self._parse_response(resp)
        for p in parsed:
            p["angle"] = angle_name
        return parsed

    results = await asyncio.gather(
        *[_call_one(name, prompt) for name, prompt in prompts],
        return_exceptions=True
    )
    all_suggestions = []
    for r in results:
        if isinstance(r, list):
            all_suggestions.extend(r)
```

```python
# clustering.py - 并行摘要生成
async def build_cluster_summaries(clusters, nodes_data, llm_func):
    async def _summarize_one(cluster_nodes):
        prompt = CLUSTER_SUMMARY_PROMPT.format(...)
        return await llm_func(prompt)

    summaries = await asyncio.gather(
        *[_summarize_one(c) for c in clusters],
        return_exceptions=True
    )
```

---

## 八、缓存策略

### LLM Response Cache

```python
# 对相同 prompt hash 的 LLM 调用进行缓存
# GraphCore 已有 llm_response_cache，扩展到 expansion 和 validation

CACHE_KEY_PREFIX = {
    "extraction": "extract",
    "expansion_cluster": "exp_cluster",
    "expansion_topnode": "exp_topnode",
    "validation": "validate",
    "cluster_summary": "cluster_sum",
}
```

### Embedding Cache

```python
# 对已计算过 embedding 的实体名进行缓存
# 避免 semantic_dedup 时重复计算
class EmbeddingCache:
    def __init__(self, max_size=10000):
        self._cache = {}

    async def get_or_compute(self, texts, embedding_func):
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            new_embs = await embedding_func(uncached)
            for t, e in zip(uncached, new_embs):
                self._cache[t] = e
        return [self._cache[t] for t in texts]
```

---

## 九、Trace 查看 API 端点

```python
@router.get("/traces")
async def list_traces(
    session_id: str,
    stage: Optional[str] = None,
    limit: int = 50,
):
    """返回 LLM 调用 trace 列表，支持按 stage 过滤。"""
    trace_file = Path(f"data/_traces/{session_id}_traces.jsonl")
    if not trace_file.exists():
        return {"traces": [], "count": 0}

    traces = []
    for line in trace_file.read_text(encoding="utf-8").strip().split("\n"):
        if not line.strip():
            continue
        record = json.loads(line)
        if stage and record.get("stage") != stage:
            continue
        # 返回时不包含完整 prompt/response（太大），只返回预览
        traces.append({
            "call_id": record["call_id"],
            "stage": record["stage"],
            "sub_stage": record["sub_stage"],
            "elapsed_s": record["elapsed_s"],
            "status": record["status"],
            "prompt_chars": record["prompt_chars"],
            "response_chars": record["response_chars"],
            "prompt_preview": record["prompt_preview"],
            "response_preview": record["response_preview"],
            "timestamp": record["timestamp"],
        })

    traces.sort(key=lambda x: -x["timestamp"])
    return {"traces": traces[:limit], "count": len(traces)}


@router.get("/traces/{call_id}")
async def get_trace_detail(call_id: str, session_id: str):
    """返回单个 trace 的完整内容（含完整 prompt 和 response）。"""
    trace_file = Path(f"data/_traces/{session_id}_traces.jsonl")
    if not trace_file.exists():
        raise HTTPException(404, "No traces found")

    for line in trace_file.read_text(encoding="utf-8").strip().split("\n"):
        record = json.loads(line)
        if record.get("call_id") == call_id:
            return record
    raise HTTPException(404, f"Trace {call_id} not found")
```

---

## 十、完整改造优先级

| 优先级 | 改造项 | 预期收益 | 工作量 |
|--------|--------|----------|--------|
| P0 | LLM Trace 中间件 | 可观测性从 0→100% | 1 天 |
| P0 | Prompt A/B 重写 (带边+描述) | 扩展质量从空壳→完整节点 | 1 天 |
| P0 | 并行化 expand (asyncio.gather) | 速度 7x → 1x | 0.5 天 |
| P1 | Prompt C 自我校验 | 过滤低质量节点 | 0.5 天 |
| P1 | 扩展时即建边 + VDB 同步 | 扩展节点可被检索 | 0.5 天 |
| P1 | Embedding-based match_nodes | 匹配准确率提升 | 0.5 天 |
| P2 | Trace 查看 API | 前端可查看中间过程 | 0.5 天 |
| P2 | Embedding 缓存 | 重复 embedding 计算减少 | 0.5 天 |
| P2 | Entity extraction prompt 精简 | token 消耗减少 ~30% | 0.5 天 |
| P3 | 边权重生命周期 | 长期进化更智能 | 1 天 |
| P3 | Forced instruction 重写 | 深度思考时扩展节点利用率提升 | 0.5 天 |

---

## 十一、代码改造 Checklist

### expander.py
- [ ] 替换 `PROMPT_TEMPLATE` → Prompt A (cluster) + Prompt B (top-node)
- [ ] `_parse_llm_json` → `_parse_rich_entities` (解析含 edges/description 的 JSON)
- [ ] `expand()` 内 `for i in indices` → `asyncio.gather`
- [ ] 新增 `_pick_top_nodes(nodes_data, edges_data, top_n=50)` 按度排序
- [ ] 写入图谱时同时写入边 (`graph.upsert_edge`)
- [ ] 每次 LLM 调用包装 `LLMTrace`

### clustering.py (新建)
- [ ] `cluster_nodes()` → HDBSCAN + DBSCAN fallback
- [ ] `build_cluster_summaries()` → 并行 LLM 摘要
- [ ] `save/load_cluster_summaries()` → JSON 持久化
- [ ] 每次 LLM 调用包装 `LLMTrace`

### manager.py
- [ ] `_normalize_record` → 增加 `entity_type`, `description`, `edges` 字段
- [ ] `match_nodes` → 增加 embedding 匹配模式
- [ ] `build_forced_instruction` → 重写为结构化指令（含描述和边信息）

### graph.py (expand endpoint)
- [ ] 加载 cluster_summaries
- [ ] 调用新的两部分扩展
- [ ] 扩展结果写入 VDB (entities_vdb + relationships_vdb)
- [ ] 返回 added_nodes + added_edges 计数

### ingest.py
- [ ] 文件处理完成后触发 `_run_density_clustering()`
- [ ] 聚类结果保存到 `knowledge/cluster_summaries.json`

### 新增 trace.py
- [ ] `LLMTrace` 类
- [ ] Trace API 端点 (list + detail)

---

## 十二、验证标准

改造完成后，应满足：

1. **速度**：扩展 50 节点图谱 < 30s（当前 > 120s）
2. **质量**：每个扩展节点有 description ≥ 20 字 + ≥ 1 条边
3. **校验**：自我校验过滤掉 > 20% 的低质量候选
4. **可观测**：每次扩展在 `data/_traces/` 下产生完整 trace
5. **检索**：扩展节点能在 Deep Mode query 中被 embedding 检索到
6. **不丢失深度思考**：Deep Mode 仍能激活 spreading activation + episodic memory + expanded node matching

---

*此优化方案由顶级 RAG 工程师视角设计，覆盖 Prompt 层、代码架构层、可观测性层三个维度。*
*核心理念：**LLM 调用不是黑箱，每一步都要可追踪、可校验、可回溯。***
