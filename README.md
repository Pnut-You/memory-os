# Memory OS

Memory OS 是面向机器狗、IoT 设备和通用 Agent 的轻量 Python 记忆模块。核心模块不依赖 Web 框架；本地调试界面使用 FastAPI 和原生 HTML/CSS/JavaScript。

## 架构边界

- 长期偏好归属 `user_id + device_id`；同一用户的不同机器狗不会共享偏好。
- 短期上下文归属 `user_id + device_id + session_id`。新请求到达时，如果距上次活动超过 15 秒，系统关闭旧 session、创建新 session，并为旧 session 异步排队长期偏好抽取；这不是后台心跳或定时关闭事件。
- 设备实时状态归属：`device_id`。
- 对外 API 不需要 start/end session 调用；session 边界由下一条消息与上次活动的时间差自动判定。
- Redis 是缓存和实时状态层；生产环境 Redis 不可用应报错。
- SQLite 是唯一持久化事实源。
- Qwen / DashScope OpenAI-compatible 模型用于回复、摘要和后台抽取；后台偏好模型不进入 `/api/query` 同步主链路。
- 主请求只读取紧凑用户卡片、滚动摘要、最近消息和必要事件上下文。
- 旧语义索引组件已删除，不使用向量库或 embedding。

## 目前使用的模型

所有外部模型调用默认固定为 `qwen3.5-flash-2026-02-23`，并显式关闭思考模式，以降低机器狗主链路延迟并避免滚动别名升级造成评测漂移。

| 用途 | 调用位置 | 执行方式 | 降级行为 |
| --- | --- | --- | --- |
| `/api/query` 普通回复及动作候选 | `ui/llm.py` | 同步主链路 | 请求失败则返回错误 |
| 短期滚动摘要 | `memory/summarizer.py` | 后台 | 模型失败使用本地摘要 |
| 日期 `time_memory` | `memory/summarizer.py` | 后台 | 模型失败使用本地摘要 |
| 日期 `action_memory` | `memory/summarizer.py` | 调试入口/后台 | 模型失败允许本地动作提取 |
| 结构化长期偏好 | `memory/preferences.py` | 后台 worker | 不使用本地规则补写偏好 |
| 七天 `action_preference_memory` | `memory/preferences.py` | 调试入口/后台 | 空结果不写入事件 |
      
## Redis Key 与作用

```text
memory-os:active-session:{user_id}:{device_id}
memory-os:session:{user_id}:{device_id}:{session_id}
memory-os:summary:{user_id}:{device_id}:{session_id}
memory-os:user-card:{user_id}:{device_id}
memory-os:user-preferences:{user_id}
memory-os:device-state:{device_id}
```

- `active-session`：某个 `user_id + device_id` 当前活跃 session 快照，TTL 一天。
- `session`：某个 `user_id + device_id + session_id` 的最近对话缓存，可从 SQLite 原始事件恢复。
- `summary`：某个 `user_id + device_id + session_id` 的滚动摘要快照，可从 `conversation_summaries` 恢复。
- `user-card`：某个 `user_id + device_id` 的 L0 记忆卡片，可从 SQLite active preferences 重建。
- `user-preferences`：可选偏好缓存，事实源仍是 SQLite。
- `device-state`：设备最新实时快照，TTL 过期即视为离线。

## SQLite 表结构与作用

- `events`
  - 保存原始事实事件，包括用户消息、assistant 回复、日期总结、原始动作序列、每日动作记忆、迁移遗留事件等。
  - 核心字段：`request_id`、`user_id`、`device_id`、`session_id`、`event_type`、`role`、`content`、`payload_json`、`created_at`。
  - `event_type='message'` 是对话原文；`time_memory` 是某一天做过事情的文本总结；`action_sequence` 是机器狗动作序列；`action_memory` 是按天聚合的文本动作事件记忆；`action_preference_memory` 是模型从最近 7 天动作事件记忆中抽取出的动作偏好事件。旧 `event_memory`、`event_preference_memory`、`action_chain_summary`、`event_summary`、`scheduled_task`、`recurring_task`、`conditional_task`、`pending_event` 只作为历史兼容数据保留，新请求不再写入。

- `conversation_sessions`
  - 保存内部自动切分的会话窗口。
  - `session_id` 由系统生成；`local_date`、`started_at`、`last_activity_at` 和 `expires_at` 用于页面按会话查看当天历史。
  - 15 秒空闲边界只在后续请求解析 session 时判断；没有后台定时器在第 15 秒主动关闭 session。

- `conversation_summaries`
  - 保存按 `user_id + device_id + session_id` 隔离的滚动摘要。
  - `summary_text` 是摘要正文，`compacted_through_event_id` 表示压缩到哪个事件，`version` 表示摘要版本。
  - `from_event_id`、`to_event_id`、`turn_count` 记录本次实际压缩的事件范围和轮次数，便于调试摘要边界。

- `user_preferences`
  - 保存 L1 结构化偏好，所有类型均按 `user_id + device_id` 隔离。
  - `preference_key` 必须来自固定偏好注册表或进入 candidate/other。
  - `status` 支持 `candidate`、`active`、`superseded`、`revoked`、`rejected`。
  - `supersedes_id` 记录偏好冲突替代关系。

- `preference_evidence`
  - 保存偏好证据，连接 `user_preferences` 和原始 `events`。
  - 用于调试“某条偏好是从哪句话来的”。

- `memory_jobs`
  - 后台任务队列。
  - 当前任务类型：`conversation_summary`、`preference_extraction`、`user_card_rebuild`、`daily_time_memory_extract`、`daily_action_memory_extract`、`weekly_action_preference_extract`。
  - 支持失败重试和可观察状态。

- `tool_runs`
  - 保存工具调用链的一次运行。
  - `context_id` 表示运行上下文，`input/output/status/error` 保存完整执行结果。

- `tool_steps`
  - 保存某个工具运行中的步骤。
  - 通过 `run_id` 关联 `tool_runs`。

- `device_state_history`
  - 保存设备状态历史。
  - Redis 保存最新状态；SQLite 只在首次出现、状态变化或达到心跳周期时追加历史。
  - 典型状态字段：`battery_percent`、`charging`、`network`、`location`、`motion_state`、`temperature_c`。

SQLite 启用 WAL、外键和 busy timeout。旧数据库迁移前会备份 `data/events.db`；旧设备维度记忆统一进入 `legacy-unassigned`，不会自动绑定到真实用户。

## 记忆系统链路

### `/api/query` 主请求链路

1. 校验 `user_id`、`device_id`、`query`。
2. 自动解析当前 session：新请求距上次活动不超过 15 秒时复用，超过 15 秒时为该请求创建新 session。
3. Redis pipeline 读取当前 `user_id + device_id + session_id` 的用户卡片、滚动摘要和最近消息。
4. 读取 SQLite 最近 `action_sequence` 作为可选上下文。
5. 组装精简 Prompt 并调用快速回复模型。
6. SQLite 单事务写入带 `session_id` 的 user 和 assistant 原文。
7. Redis 更新当前 session 最近消息。
8. 按当天日期排队日期总结任务；每日动作记忆由调试入口或未来定时任务触发。
9. 如果回复模型返回高置信度 `action_sequence` 候选，把动作候选写入 SQLite `action_sequence`；时间/条件任务候选会被忽略。
10. 达到偏好抽取阈值或出现明确偏好表达时，按事件 ID 范围调度后台偏好抽取任务。
11. 返回精简响应。

主请求不做模型偏好抽取、偏好合并、用户卡片重建、向量检索、embedding、全量历史扫描或同步摘要生成。

### 短期摘要链路

短期摘要按 `user_id + device_id + session_id + Asia/Shanghai 日期` 隔离，使用 SQLite 原始 conversation 作为事实源，不依赖 Redis 判断边界。

- 默认 `SHORT_MEMORY_SUMMARY_MIN_TURNS=20`、`SHORT_MEMORY_PROMPT_TRIGGER_TOKENS=5000`、`SHORT_MEMORY_RETAIN_RECENT_TURNS=5`。
- 只有当前 session 对话轮次达到 20，且本次发送给回复模型的完整 Prompt token 估算值达到 5000，才后台触发摘要。
- 完整 Prompt token 只用于触发判断；真正进入摘要的只有 conversation 中的 user/assistant 原文。
- 摘要不会压缩 System Prompt、长期记忆、工具说明、设备状态或当前用户问题。
- 最近 5 轮 conversation 必须保留原文；摘要只覆盖更早的 conversation。
- 摘要完成后后续请求会使用 `Conversation Summary + 最近 5 轮 Conversation 原文` 重建上下文；不会为了固定压到某个 token 数而继续过度摘要。
- `conversation_summaries.local_date`、`session_id`、`from_event_id` 和 `to_event_id` 可用于确认摘要日期、session 与压缩范围。

摘要生成在后台线程执行，失败不会阻塞 `/api/query`。

### 日期总结链路

时间记忆现在只表示“某一天做过事情的总结”，不是定时任务、提醒任务或条件任务。它由摘要模型从当天所有 session 原始会话生成，写入 SQLite `events` 的 `event_type='time_memory'`：

- `content` 保存摘要正文。
- `payload_json.memory_date` 保存归属日期，例如 `2026-07-02`。
- `payload_json.memory_at` 保存带时区时间戳，例如 `2026-07-02T21:00:00+08:00`。
- `payload_json.title` 保存可选标题。

`/api/query` 主链路不执行也不排队日期总结，只写入 SQLite 原文并返回“日期结束后处理”的调度信息。使用 `MemoryManager.create(..., start_scheduler=True)` 启动的后台 worker 默认在北京时间每天 `01:00`，从 SQLite 读取刚结束的前一自然日全部 session 原始会话，为每个 `user_id + device_id` 生成或覆盖当天唯一 `time_memory`。模型不可用时允许本地摘要降级，但降级结果仍是压缩摘要，不保存完整逐句 transcript。调试 UI 可以按日期重跑抽取，但不能手写摘要正文。

### 事件记忆库

事件库按日期展示机器狗行为事件记忆。`daily_action_memory_extract` 不在 `/api/query` 主链路执行；它与日期总结一起由后台 worker 在每天北京时间 `01:00` 自动处理前一自然日，也可以通过调试 UI 的“重跑日期事件抽取”按钮手动执行。任务从同一天所有 session 原始会话抽取当天唯一一条 `action_memory`。模型不可用时允许从已存在动作路由或明确动作词做本地降级：

- `content` 保存当天动作链路 text，例如“事件记忆（2026-07-03）：\n事件链路：\n1. 用户要求 坐下 -> 机器狗完成 坐下\n2. 用户要求 转圈 -> 机器狗完成 转圈 -> 用户反馈：用户表示肯定”。
- `payload_json.memory_date` 保存归属日期。
- `payload_json.metadata.session_ids` 保存来源会话列表。
- `payload_json.metadata.action_chain_count` 保存当天聚合的动作链路数量。
- `payload_json.source_event_ids` / `source_message_event_ids` 保存来源消息或动作事件。

日期事件记忆只保留站起、坐下、转圈、跳舞、前进、后退等机器狗动态行为动作，以及用户对这些动作执行结果的反馈。提醒、出行计划、车票协助、安抚、心情和普通偏好不进入事件库。

Agent 消费事件库时使用 text-only 接口 `GET /api/memories/events-text`。该接口只返回 `id`、`event_type`、`text`、`created_at`、`memory_date`、`session_id` 和 `device_id`，不暴露 `payload_json`。

### 七天事件偏好抽取

每周一北京时间 `01:00`，后台 worker 会等待上一自然周（周一至周日）的每日事件任务完成，再拼接该窗口内的 `action_memory.content`，交给动作偏好抽取模型提取重复动作链。模型返回的稳定动作偏好写成 `action_preference_memory` 文本事件，不写入普通长期偏好表，也不在 `/api/query` 主链路执行。事件库页的“七天事件偏好抽取”按钮继续作为同步调试入口。

自动调度使用 SQLite `memory_jobs` 保存日期/周窗口和唯一调度键。服务在准点停机时，启动后会补跑最近 30 个已结束自然日，以及完全落在该范围内的完整自然周；已经成功、跳过或正在处理的周期不会重复创建。可通过以下环境变量调整，生产环境必须保证至少一个 `start_scheduler=True` 的进程持续运行：

```bash
MEMORY_SCHEDULE_TIME=01:00
MEMORY_SCHEDULE_TIMEZONE=Asia/Shanghai
MEMORY_SCHEDULE_BACKFILL_DAYS=30
```

### 动作事件链路

系统不再用本地动作词表解析用户原文。只有回复模型返回高置信度 `type='action_sequence'` 候选且包含 `actions` 数组时，系统才写入 `event_type='action_sequence'`；回复模型返回高置信度 `type='action_feedback'` 时，系统优先使用模型提供的动作引用，没有引用则自动关联同一用户和设备下最近一次动作序列。没有可关联动作时，反馈不会写入事件库。最近动作序列会作为普通上下文提供给回复模型，由模型自行判断是否用于“重复上次操作”等请求。

### 偏好记忆链路

普通 `/api/query` 不同步调用偏好抽取模型。下一条消息到达且与上次活动间隔大于 15 秒时，系统关闭上一 session，并为该 session 创建唯一的 `preference_extraction` 后台任务。恰好 15 秒仍复用原 session；没有后续消息时不会依赖定时器主动关闭。

关键词和累计消息数量不再决定是否创建偏好任务。每个符合条件的已结束 session 都由 Flash 判断是否包含长期偏好，因此“这种安静环境让我很舒服”等没有固定关键词的表达也不会在模型调用前被规则漏掉。`anonymous` 不创建长期偏好任务。

偏好记忆页的“运行一次偏好抽取”按钮会调用 `POST /api/debug/users/{user_id}/preferences/extract`。这是本地调试入口，默认 `force=true`，会重跑当前 `user_id` 和可选 `device_id` 的当前偏好抽取上下文，即使上一轮抽取已经成功但没有抽出偏好，也不会被 `latest_processed_event_id` 卡住。它不是 `/api/query` 主链路的一部分。

Worker 的真实数据流：

1. 按任务的 `user_id + device_id + session_id` 从 SQLite 查询该已结束 session 的全部 `role='user'` 原始消息。
2. 构造 `source_user_events`，每项只包含真实 `event_id`、用户原文和时间。
3. 整个 session 的用户原文通过一次 Flash 请求完成抽取；不发送 assistant 文本、Redis 卡片、滚动摘要或旧 preferences。
4. 系统逐项校验 `event_id`、类型、置信度以及 value/evidence 的逐字来源，再执行确定性合并并写入 SQLite。
5. 任务结果记录输入消息数、输入/输出字符数、模型 usage、请求耗时和校验结果，用于定位性能问题。

例如一个 session 内有多条用户消息时，模型会在同一次请求中看到带真实 `event_id` 的用户原文数组：

```text
[{"event_id": 10, "text": "今天聊聊音乐"}, {"event_id": 12, "text": "我不喜欢吃香菜"}]
```

严格偏好抽取只收敛为三类：

- `profile.occupation`：职业或身份，例如“我是摄影师”。
- `preference.likes`：明确喜欢的事物，例如“我喜欢摄影”“我喜欢周杰伦”。
- `preference.dislikes`：明确不喜欢的事物，例如“我不喜欢吵闹”。

三类结构化偏好记忆的合并规则不同：

- `profile.occupation` 是每只狗的单值槽位，新职业 supersede 同一 `user_id + device_id` 下的旧职业。
- `preference.likes` 和 `preference.dislikes` 是每只狗的去重多值集合，重复表达同一对象只增加 `evidence_count`。
- `preference.likes` / `preference.dislikes` 不按模型原始 `value_json` 字面量去重，而是按规范化对象去重；例如 `{"code":"travel","label_zh":"旅游"}`、`{"code":"旅游"}` 和 `display_text_zh="喜欢旅游"` 会归为同一个“旅游”偏好。
- `preference.likes` 和 `preference.dislikes` 对同一个对象互斥。比如用户先说“我喜欢吃苹果”，后面又说“我不喜欢吃苹果了”，新的“不喜欢苹果”会 active，旧的“喜欢苹果”会变成 `revoked`，但“喜欢旅游”等其他 likes 不受影响。
- SQLite 不物理删除旧偏好和证据；Redis 用户卡片只从 active 偏好重建。

模型输出必须是完整 JSON，不能带 Markdown、解释或额外字段：

```json
{"preferences": [{"event_id": 12, "type": "occupation | likes | dislikes", "value": "当前用户原文中的内容", "evidence": "包含 value 的当前原句", "confidence": 0.0}]}
```

`event_id` 必须属于当前 session；`value` 必须逐字存在于对应原文，`evidence` 必须是对应原文中包含该值的语句；没有偏好时返回空数组。模型不生成数据库主键、scope、状态或合并动作，这些由程序确定。模型超时、非法 JSON、字段错误、空值或原文来源校验失败时不写入偏好；任务按配置的重试上限重试，不使用本地规则补写。

核心内部接口为 `get_user_card(user_id, device_id)`、`write_long_term_preference(...)` 和 `revoke_long_term_preference(...)`。读取 Redis 卡片失败时会从 SQLite active preferences 恢复；写入和撤销以 SQLite 为事实源，并使 Redis 用户卡片失效后排队重建。

偏好变化后，系统清理该用户的 Redis 用户卡片；后续读取可以从 SQLite active preferences 重建。读取只返回完全匹配 `user_id + device_id` 的偏好。

`/api/query` 遇到包含“长期记忆”“你记得”“记忆中”等明确长期信号，并询问职业、喜欢或不喜欢时，会按 `user_id + exact preference_key + active` 直接查询 SQLite；有值时直接返回，多值用顿号连接，无记录时返回“未记录”，不调用回复模型。包含“刚才”或“当前会话”的问题仍走短期上下文。

当前 session 的限制需要特别注意：15 秒只是下一次请求到达时的 session 复用判断，没有后台 session-end 事件。因此偏好抽取由明确表达或 10 条消息触发，而不是由 session 结束触发。

### 设备实时状态链路

设备最新状态写入 Redis `memory-os:device-state:{device_id}`。如果 Redis 快照过期，设备视为离线。状态首次出现、发生变化或达到心跳周期时，追加一条 SQLite `device_state_history`。

推荐核心状态字段：

```json
{
  "battery_percent": 80,
  "charging": false,
  "network": "wifi",
  "location": "客厅",
  "motion_state": "idle",
  "temperature_c": 36.5
}
```

## HTTP API

### POST `/api/query`

```json
{
  "user_id": "user-001",
  "device_id": "dog-001",
  "query": "带我去安静一点的地方"
}
```

响应：

```json
{
  "request_id": "2f1c...",
  "user_id": "user-001",
  "device_id": "dog-001",
  "assistant_reply": "好的，我会优先选择安静的路线。",
  "model": "qwen3.5-flash-2026-02-23"
}
```

`user_id` 和 `device_id` 只允许字母、数字、下划线和连字符，最长 128 字符。`anonymous` 可以完成当前对话和短期缓存，但不会生成长期偏好任务。

### 外部 Agent 记忆接口

外部 Agent 自己负责调用回复模型，Memory OS 只提供读上下文和写完整对话轮次两个接口。首版没有内置鉴权，只能绑定可信内网或放在已有鉴权网关后面。

推理前读取 `user-001 + dog-001` 的记忆：

```bash
curl 'http://127.0.0.1:8000/api/agent/memory-context?user_id=user-001&device_id=dog-001'
```

响应已经整理为可直接交给 Agent 的精简 JSON：

```json
{
  "user_id": "user-001",
  "device_id": "dog-001",
  "long_term_memory": "个人信息：\n用户的职业是程序员。\n\n偏好：\n用户的偏好包括：健身、游泳。\n\n不喜欢：\n用户不喜欢：香菜。",
  "short_term_memory": {
    "session_id": "sess-fde9898e9d294e2a96b654453fb78a48",
    "messages": [
      {"role": "user", "content": "你好"},
      {"role": "assistant", "content": "你好，我在。"}
    ]
  }
}
```

`long_term_memory` 是可直接放入 Agent system prompt 的字符串，固定分为“个人信息、偏好、不喜欢”三层，只使用完全匹配该 `user_id + device_id` 的 active 长期记忆。每层内容会去重；没有记录时显示“暂无已确认信息。”。`short_term_memory.messages` 严格来自当前内部 session，不会在空 session 时回退到历史 session。`session_id` 只用于诊断，外部不传回。Redis 缓存丢失时数据会从 SQLite 恢复；生产 Redis 故障直接报错。

Agent 得到最终回复后写入一轮短期对话：

```bash
curl -X POST 'http://127.0.0.1:8000/api/agent/conversation-turns' \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id":"agent-turn-001",
    "user_id":"user-001",
    "device_id":"dog-001",
    "user_text":"我喜欢飞盘",
    "assistant_text":"好的，我记住了",
    "prompt_token_count":5230
  }'
```

该接口只接受完整的 user/assistant 一轮，原文在 SQLite 单事务持久化并同步到 Redis。`prompt_token_count` 是外部 Agent 本次实际完整 Prompt token 数，用于判断是否达到 5000-token 摘要阈值；省略时正常写入，但本轮不会触发摘要。`request_id` 必须由 Agent 生成且保持唯一；相同内容重复提交返回 `idempotent_replay=true`，不会重复写入，不同内容复用同一 ID 返回 HTTP 409。接口不能直接写长期偏好，但内部 session 结束后仍会按既有后台流程抽取偏好。

### 调试 API

```text
GET /api/status
GET /api/debug/users/{user_id}?device_id=dog-001
GET /api/debug/users/{user_id}/sessions?device_id=dog-001
GET /api/debug/users/{user_id}/sessions/{session_id}
GET /api/debug/users/{user_id}/preferences?device_id=dog-001
POST /api/debug/users/{user_id}/preferences/extract
GET /api/debug/users/{user_id}/events?device_id=dog-001
GET /api/debug/users/{user_id}/time-memories?device_id=dog-001
POST /api/debug/users/{user_id}/time-memories
POST /api/debug/users/{user_id}/events/extract
GET /api/debug/events
GET /api/memories/events-text
GET /api/debug/users/{user_id}/actions
GET /api/debug/devices/{device_id}
POST /api/debug/devices/{device_id}/state
DELETE /api/debug/users/{user_id}/memory
```

`DELETE` 和设备状态写入接口只用于本地调试或受保护管理接口，不应默认暴露到公网。

`POST /api/debug/users/{user_id}/preferences/extract` 是本地调试接口，用于强制对当前用户执行一次后台偏好抽取：

```json
{
  "device_id": "dog-001",
  "force": true,
  "recent_user_messages": 20
}
```

返回里的 `mode`、`context_mode`、`summary_version`、`recent_turn_count`、`action_event_count`、`from_event_id`、`to_event_id`、`latest_processed_event_id`、`preference_context_preview`、`process.claimed`、`process.succeeded`、`process.failed`、`process.recovered_stale` 和 `process.errors` 用来观察本次 worker 执行结果；`memory.active_preferences` 和 `memory.candidate_preferences` 是刷新后的结构化偏好。

自动后台抽取和手动调试抽取的区别：

- 自动抽取：只处理上次成功抽取后的新用户消息，用于保持 `/api/query` 低延迟和后台任务轻量。
- 手动调试抽取：默认 `force=true`，重跑当前摘要 + 最近 5 轮上下文，适合排查模型空抽、漏抽、JSON 校验失败或选错 `user_id/device_id`。
- 如果返回“当前 user_id/device_id 下没有原始用户消息”，说明该调试范围内没有可送给模型的原始用户事件；先检查短期记忆页的 session 消息。

## 调试 UI

页面包含：

- 对话测试：输入 query 后按 Enter 发送，Shift+Enter 换行。
- 对话测试：显示类似 LangSmith 的“请求链路”，包括请求输入、上下文读取、滚动摘要、偏好记忆、最近对话、短期记忆 / 当前 Session、日期总结抽取、动作事件路由、SQLite 写入、回复模型输入和回复模型输出。
- 短期记忆：按 `user_id + device_id + 日期` 查看 session 列表，点击 session 后查看该 session 的摘要、会话消息和动作记忆。
- 长期记忆：按输入的 `user_id + device_id` 查看用户卡片、结构化偏好和证据；不展示会话列表或会话摘要。
- 偏好记忆页的抽取诊断会显示事件范围、摘要版本、最近轮次数、动作数、偏好抽取上下文预览、最新成功处理事件和 worker 错误详情。
- 日期总结：查看按天从会话历史自动抽取的文本摘要，也可按日期重跑抽取。
- 事件记忆库：通过 text-only 接口按日期查看“日期事件记忆”和“7 天事件偏好记忆”，也可手动触发日期事件抽取和七天事件偏好抽取。
- 设备实时状态：查看在线状态、最新快照、历史记录，并用少量核心字段写入调试状态。

## 配置

首次运行前先复制配置模板：

```bash
cp .env.example .env
```

然后编辑 `.env`，填入阿里云百炼 / Model Studio 的真实 `DASHSCOPE_API_KEY`。`LLM_API_KEY` 只作为兼容旧配置的 fallback；如果两者都配置，会优先使用 `DASHSCOPE_API_KEY`。如果没有配置 key，调试 UI 调用模型时会报错：

```text
LLM request failed: DASHSCOPE_API_KEY or LLM_API_KEY is not configured in .env
```

常用配置如下；未写入 `.env` 的短期摘要参数会使用这里列出的代码默认值：

```dotenv
DASHSCOPE_API_KEY=
LLM_API_KEY=
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3.5-flash-2026-02-23

PREFERENCE_EXTRACTOR_ENABLED=true
PREFERENCE_EXTRACTOR_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# 为空时默认复用 DASHSCOPE_API_KEY
PREFERENCE_EXTRACTOR_API_KEY=
PREFERENCE_EXTRACTOR_MODEL=qwen3.5-flash-2026-02-23
LONG_TERM_EXTRACTOR_MODE=small
LONG_TERM_SMALL_MODEL=qwen3.5-flash-2026-02-23
LONG_TERM_LARGE_MODEL=qwen3.5-flash-2026-02-23
PREFERENCE_EXTRACT_BATCH_SIZE=8
PREFERENCE_EXTRACT_MAX_ATTEMPTS=3

SHORT_MEMORY_SUMMARY_MIN_TURNS=20
SHORT_MEMORY_PROMPT_TRIGGER_TOKENS=5000
SHORT_MEMORY_RETAIN_RECENT_TURNS=5

REDIS_URL=redis://localhost:6379/0
REDIS_TTL_SECONDS=86400
REDIS_ALLOW_MEMORY_FALLBACK=true
REDIS_PREFIX=memory-os
```

不要把真实 API Key 写入 `.env.example` 或提交到仓库；真实 Key 只放在本机 `.env`。未单独填写 `PREFERENCE_EXTRACTOR_API_KEY` 时，偏好抽取器会优先复用 `DASHSCOPE_API_KEY`，再回退到 `LLM_API_KEY`；未单独填写 `PREFERENCE_EXTRACTOR_BASE_URL` 时会复用 `LLM_BASE_URL`。`small`、`hybrid` 和 `large` 模式当前都配置为同一个固定 Flash 快照，不会回退到旧 CodeQwen 或 Max 模型。如果 UI 返回 401 或 `invalid_api_key`，说明当前 key 无效、过期，或与 `LLM_BASE_URL` 不匹配；`/api/status` 会显示实际配置。

## 本地运行

本项目现在使用 `uv` 统一管理 Python 虚拟环境和依赖。旧的 `ui/.venv` 已删除，不再使用 `python -m venv ui/.venv`、`pip install -r requirements.txt` 或 `ui/.venv/bin/...`。依赖入口是根目录的 `pyproject.toml` 和 `uv.lock`，`uv sync` 会自动创建或更新根目录 `.venv`。

`uv sync` 的前提是电脑上已经安装过 `uv`。每台新电脑只需要安装一次 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version
```

第一次安装或同步依赖：

```bash
uv sync
```

确认当前 uv 环境：

```bash
uv run python --version
uv run python -c "import fastapi, redis, pydantic; print('dependencies ok')"
```

核心测试：

```bash
uv run python -m unittest discover -s tests -v
uv run python example.py
```

`example.py` 只读取并展示已有记忆，不会写入示例对话或长期偏好，也不会污染当前 SQLite 数据。

短期记忆在当前 session 未触发滚动摘要前不会裁剪原文对话：20 轮以内全部进入回复 prompt；达到 20 轮后，只有完整 prompt 估算达到 `SHORT_MEMORY_PROMPT_TRIGGER_TOKENS` 时才摘要较早对话，并保留最近 `SHORT_MEMORY_RETAIN_RECENT_TURNS` 轮原文。评测会先把每个 Case 的前置对话写入当前 session，再调用一次回复模型回答追问。默认数据集共 400 条，由 200 条 2-10 轮数据和 200 条 10-20 轮数据组成，Case ID 为 `short_001` 至 `short_400`。默认命令严格使用 Redis，Redis 不可用时会失败：

```bash
uv run python evaluation/run_short_term_eval.py
```

也可以直接使用 uv 创建的虚拟环境解释器：

```bash
.venv/bin/python evaluation/run_short_term_eval.py
```

不要直接运行 `python3 evaluation/run_short_term_eval.py`，因为系统 Python 通常没有 `redis`、`openai` 等项目依赖；`uv sync` 安装的是 `.venv` 环境。

合格判定会先执行 NFKC 规范化，移除空白、标点、符号以及“的/地/得”，再检查规范化后的目标事实是否完整包含；人称变化、对象变化或实质关键词缺失仍判失败。控制台会打印每条 Case 的期望、问题、回答、目标事实轮次和失败原因；JSONL 同时保存原始及规范化文本。每个 Case 使用独立 Redis 前缀，结束后清理对应测试 key。

先跑小样本可减少模型调用次数：

```bash
uv run python evaluation/run_short_term_eval.py --max-cases 5
```

保留的旧版 2-10 轮数据集：

```bash
uv run python evaluation/run_short_term_eval.py --dataset evaluation/datasets/short_term_memory_probe_2_10.jsonl
```

新版 10-20 轮高难度数据集：

```bash
uv run python evaluation/run_short_term_eval.py --dataset evaluation/datasets/short_term_memory_probe_10_20.jsonl
```

指定日志路径：

```bash
uv run python evaluation/run_short_term_eval.py --max-cases 5 --log-file evaluation/results/debug_short_eval.jsonl
```

只有本地无 Redis 调试时才显式允许内存 fallback：

```bash
uv run python evaluation/run_short_term_eval.py --allow-memory-redis-fallback
```

长期结构化偏好记忆评测检查 SQLite 中最终保存的 `profile.occupation`、`preference.likes`、`preference.dislikes` 以及确定性 Verification。默认数据集 60 条，职业、喜欢、不喜欢各 20 条。每个 Case 使用独立运行时 `user_id`、临时 SQLite 和随机 Redis 前缀，不共享长期偏好状态：

```bash
uv run python evaluation/run_preference_eval.py
```

先跑小样本：

```bash
uv run python evaluation/run_preference_eval.py --max-cases 5
```

Verification 按当前 `user_id + exact preference_key + active + scope=user` 查询 SQLite：有值时直接返回结构化值，多值用顿号连接；没有值时只返回“未记录”，不调用回复模型。只有 SQLite 抽取和 Verification 都通过，Case 才通过。JSONL 会记录模型原始输出、校验后输出、检索到的长期事实、验证问题、失败原因和是否发生模型回退。也可以指定日志路径：

```bash
uv run python evaluation/run_preference_eval.py --max-cases 5 --log-file evaluation/results/debug_preference_eval.jsonl
```

只跑某一类：

```bash
uv run python evaluation/run_preference_eval.py --type likes --max-cases 5
```

调试 UI：

```bash
REDIS_ALLOW_MEMORY_FALLBACK=true uv run uvicorn ui.app:app --reload --host 127.0.0.1 --port 8000
```

调试 UI 使用响应式布局，页面整体可滚动，卡片内部按内容区域滚动，适配笔记本、桌面和窄屏浏览器窗口。浏览器自动请求的 `/favicon.ico` 已由 UI 服务兜底返回，不会再产生 404 日志。

如果 `8000` 被占用：

```bash
REDIS_ALLOW_MEMORY_FALLBACK=true uv run uvicorn ui.app:app --reload --host 127.0.0.1 --port 8001
```

## 偏好抽取故障排查

- 点击“运行一次偏好抽取”没有偏好：先看页面上的 `process.errors`，常见原因是模型超时、Key 无效、模型返回非 JSON 或返回字段缺失。
- 页面不再把 `Internal Server Error` 当 JSON 解析；如果后端或代理返回纯文本错误，会直接显示文本摘要。
- 页面主行会直接显示 `错误:` 或模型诊断；严格偏好抽取失败时不会使用本地规则补写，完整错误在“错误详情”里。
- `process.recovered_stale > 0`：说明旧的 `running` job 卡住，系统已自动恢复并重新处理。
- `process.succeeded > 0` 但偏好为空：说明事件范围内没有候选偏好原文，或模型对候选原文返回了合法的 `none`。
- `/api/status` 的 `preference_extractor.configured` 必须为 `true`，否则不会调用 Qwen 抽取模型。
- 失败达到 `PREFERENCE_EXTRACT_MAX_ATTEMPTS` 后不会无限重试；修复配置后可再次点击手动抽取按钮创建或合并新任务。

## 迁移

运行：

```bash
uv run python -m memory.migrate
```

将 `user-001` 现有 `legacy-unassigned` 偏好显式绑定到 `dog-001`：

```bash
uv run python -m memory.migrate --preference-user-id user-001 --preference-device-id dog-001
```

该迁移幂等；不会自动迁移其他用户或其他狗的数据。

迁移会：

- 备份旧 `data/events.db`。
- 创建新 schema。
- 把旧 `session_id` 仅解释为旧设备标识。
- 将旧事件标记为 `legacy-unassigned`。
- 保留 legacy 表用于审计。
- 删除旧索引 outbox 表。

旧数据默认不会进入任何真实用户上下文。

### 启动命令

```bash
cd ~/pt/projects/i/memory-os

# 1. 每台电脑先安装一次 uv；如果 uv --version 已正常输出，可跳过
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装依赖并创建根目录 .venv，只需要第一次执行或依赖变化时执行
uv sync

# 3. 首次运行前复制 .env 并填写 DASHSCOPE_API_KEY
cp .env.example .env

# 4. 启动项目
REDIS_ALLOW_MEMORY_FALLBACK=true \
uv run uvicorn ui.app:app \
  --reload \
  --host 127.0.0.1 \
  --port 8000
```

### 测试命令

```bash
uv run python -m unittest discover -s tests -v
uv run python example.py
uv run python evaluation/run_short_term_eval.py
uv run python evaluation/run_preference_eval.py
```
