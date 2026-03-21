# 存储路径迁移操作文档（rag_storage_api -> data/#000xx）

## 1. 目标

将当前 `rag_storage_api` 目录中的会话存储彻底迁移到会话目录 `data/#000xx` 下，统一为“每个对话自包含”结构：

- 情景记忆（episodes / memory_graph / episode_vectors）放到：`data/#000xx/talk/`
- 会话知识图谱、向量库、KV、knowledge_bases 放到：`data/#000xx/knowledge/`
- 会话原始内容保持：`data/#000xx/content/`

并移除运行时对 `rag_storage_api` 的依赖。

## 2. 迁移后目录规范

以 `#00003` 为例：

```text
data/#00003/
  content/
  talk/
    talk.json
    episodes.json
    memory_graph.json
    episode_vectors.json
  knowledge/
    graph_chunk_entity_relation.graphml
    vdb_chunks.json
    vdb_entities.json
    vdb_relationships.json
    kv_store_*.json
    knowledge_graph.json
    knowledge_base.db
    knowledge_bases/
      doc_*.json
      session_*.json
```

系统级索引（会话列表、会话元信息）保留在：

```text
data/_system/knowledge_base.db
```

## 3. 代码改造范围

1. `SessionManager`  
- 会话工作目录 `path` 从 `rag_storage_api/sessions/#xxxx` 改为 `data/#xxxx/knowledge`
- 增加 `knowledge_dir` 元数据字段
- 迁移旧路径到新路径（保留兼容）

2. `MemoryEngine` 使用方式  
- 从全局单实例改为会话级实例
- `working_dir` 改为 `data/#xxxx/talk`
- `episodes.json/memory_graph.json/episode_vectors.json` 会话隔离

3. 后端路由
- 查询与记忆相关逻辑改为按 `session_id` 获取会话记忆引擎
- memory API 改为会话级（必须传 `session_id`）

4. 应用初始化
- 默认工作目录改为 `data/_system`
- 移除/替换 `rag_storage_api` 清理逻辑

5. 前端/代理层
- 所有 memory 相关调用补 `session_id`
- 处理新路径返回与提示信息

## 4. 迁移步骤

1. 备份
- 备份 `rag_storage_api/` 与 `data/`

2. 停服务
- 停止后端与前端

3. 部署改造代码

4. 首次启动自动迁移
- 将旧会话目录（`rag_storage_api/sessions/#xxxx`）迁移至 `data/#xxxx/knowledge`
- 会话元数据更新为新路径

5. 校验
- 新建会话并上传文档，确认仅在 `data/#xxxx` 生成数据
- 查询后确认 `data/#xxxx/talk/episodes.json` 有写入
- 图谱页确认 `data/#xxxx/knowledge` 内图、向量、KV 正常落盘

6. 清理
- 确认稳定后删除 `rag_storage_api/`

## 5. 回滚方案

若迁移后异常：

1. 停止服务
2. 回滚代码
3. 恢复备份目录（`rag_storage_api/`、`data/`）
4. 重启并验证

## 6. 验收标准

- 不再新增 `rag_storage_api/*` 运行时数据
- 每个会话仅在 `data/#xxxx/{content,talk,knowledge}` 写入
- 查询、图谱、上传、会话历史功能可用
- memory API、query API 不报路径错误

## 7. 本次实施记录（2026-03-06）

- 已完成代码改造：
  - 默认 `workdir` 改为 `data/_system`
  - 会话知识目录改为 `data/#xxxx/knowledge`
  - 情景记忆改为 `data/#xxxx/talk`（会话级）
  - memory 相关 API 改为必须传 `session_id`
- 已执行数据迁移：
  - 旧目录备份到 `data/_system/legacy_backup/rag_storage_api_20260306_201827`
  - `rag_storage_api/sessions/#xxxx` 已迁移到 `data/#xxxx/knowledge`
  - `rag_storage_api/episodes.json` 按 `session_id` 拆分到 `data/#xxxx/talk/episodes.json`
  - 无法归属会话的历史 episode 存放在 `data/_system/legacy_memory/episodes_orphan.json`
- 旧目录已删除：`rag_storage_api/`
