# Memory OS 项目说明

## 项目用途

Memory OS 是面向机器狗、IoT 设备和通用 Agent 的裸 Python 分层记忆模块。它保存对话上下文、用户长期偏好、时间记忆、事件、工具调用链和设备状态，不依赖 Web 框架作为核心运行时。

## 存储边界

- SQLite 是唯一持久化事实源：保存原始事件、滚动摘要、结构化用户偏好、偏好证据、后台任务、工具调用与步骤、设备状态历史。
- Redis 是实时和临时状态层：保存带 TTL 的最近对话、滚动摘要快照、用户记忆卡片、设备最新快照、运行中的工具链、临时结果、锁与幂等键。
- 生产环境 Redis 不可用时必须报错；只有显式开发/测试配置允许内存降级。
- 旧语义索引组件已删除，不再有向量库、embedding、索引 outbox 或语义索引后台任务。

## 核心行为

- 长期记忆归属 `user_id`。
- 短期对话和滚动摘要归属 `user_id + device_id`。
- 设备状态归属 `device_id`。
- 对外 API 不存在 Session，不需要开始或结束会话。
- 每次 `/api/query` 把 user 和 assistant 原文写入 SQLite，同时把最近消息写入 Redis。
- 每 10 轮对话触发后台滚动摘要，主请求不等待摘要完成。
- `anonymous` 允许当前对话和短期缓存，默认禁止生成长期偏好。
- 70B 偏好抽取只在后台 worker 中调用外部模型服务，不能阻塞 `/api/query`。
- 用户卡片由后台任务从 SQLite active preferences 重建；Redis 丢失时可以从 SQLite 恢复。
- 工具调用历史完整写入 SQLite；Redis 只保存正在运行的工具链状态。
- 设备最新状态写 Redis；首次出现、变化或达到心跳周期时追加 SQLite 历史。

## 主要入口

- `MemoryManager.create()`：组装 SQLite、Redis、摘要器和偏好抽取器。
- 对话：`get_conversation_context`、`add_conversation_turn`、`search`、`remember_at`。
- 偏好：`process_memory_jobs_once`、`rebuild_user_card`、`delete_user_memory`。
- 工具链：`begin_tool_run`、`record_tool_step`、`finish_tool_run`、`get_tool_run`。
- 设备：`update_device_state`、`get_device_state`、`get_device_history`。
- 旧数据迁移：`python3 -m memory.migrate`。

## 开发约束

- 所有持久时间使用带时区的 UTC ISO-8601；SQLite 启用 WAL、外键和 busy timeout。
- 不得把 Redis 中的数据当作唯一历史记录。
- 不得在主请求中调用 70B、生成 embedding、做全量历史扫描或执行长期偏好合并。
- 新增长期持久化行为时必须有 SQLite 测试；新增实时行为时必须覆盖 TTL/离线或 Redis 故障场景。
- 不提交 `.env`、API Key、`data/` 数据库或缓存文件。
- 修改逻辑后需要同步更新 `README.md`。

## 验证

```bash
python3 -m unittest discover -s tests -v
python3 example.py
```
## 补充说明
每次修改代码后，需要告诉用户具体运行命令。
