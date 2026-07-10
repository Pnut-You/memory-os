# Memory OS

Memory OS 是面向机器狗、IoT 设备和通用 Agent 的轻量 Python 记忆模块。核心模块不依赖 Web 框架；本地调试界面使用 FastAPI 和原生 HTML/CSS/JavaScript。

## 架构边界

- 偏好记忆归属：`user_id + device_id`，保留 `user_id` 字段用于未来多用户扩展。
- 短期上下文归属：`user_id + device_id + session_id`；15 秒无新响应/新输入即切到新 session，session 缓存保留一天。
- 设备实时状态归属：`device_id`。
- 对外 API 不需要 start/end session 调用，系统内部自动按 15 秒空闲窗口切分 session。
- Redis 是缓存和实时状态层；生产环境 Redis 不可用应报错。
- SQLite 是唯一持久化事实源。
- Qwen / DashScope OpenAI-compatible 模型用于后台偏好抽取，不进入 `/api/query` 主链路。
- 主请求只读取紧凑用户卡片、滚动摘要、最近消息和必要事件上下文。
- 旧语义索引组件已删除，不使用向量库或 embedding。

## 目前使用的模型
  1. 调试 UI /api/query 回复模型
      - ui/llm.py:36
      - 默认 qwen3.7-plus
      - 同步阻塞主请求
      - 还会顺带让模型返回 event_routes，用于动作事件候选

  2. 短期滚动摘要
      - memory/summarizer.py:25
      - 默认 qwen3.7-plus
      - 有本地 fallback，模型失败会退回本地摘要

  3. 日期总结 time_memory
      - memory/summarizer.py:40
      - 默认 qwen3.7-plus
      - 有本地 fallback

  4. 日期动作事件记忆 action_memory 抽取
      - memory/summarizer.py:55
      - 默认 qwen3.7-plus
      - 有本地 fallback
      - 只有 summarizer backend 是 llm 时优先调用模型

  5. 结构化长期偏好抽取 user_preferences
      - memory/preferences.py:275
      - 默认 qwen3.7-max
      - 后台 worker 调用，不在 /api/query 主链路里
      - 如果 preference_extractor.configured=false，直接返回空偏好，不做本地规则补偿

  6. 七天动作偏好记忆 action_preference_memory 抽取
      - memory/preferences.py:378
      - 默认同样是 qwen3.7-max
      - 从 7 天 action_memory 文本里抽稳定动作偏好
      
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
- `user-card`：某个 `user_id + device_id` 的 L0 记忆卡片，只放该设备下最重要的 active 偏好，可从 `user_preferences` 重建。
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

- `conversation_summaries`
  - 保存按 `user_id + device_id + session_id` 隔离的滚动摘要。
  - `summary_text` 是摘要正文，`compacted_through_event_id` 表示压缩到哪个事件，`version` 表示摘要版本。
  - `from_event_id`、`to_event_id`、`turn_count` 记录本次实际压缩的事件范围和轮次数，便于调试摘要边界。

- `user_preferences`
  - 保存 L1 结构化用户偏好，写入和查询都按 `user_id + device_id` 隔离。
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
2. 自动解析当前 session：同一 `user_id + device_id` 15 秒内复用，超过 15 秒新建。
3. Redis pipeline 读取当前 `user_id + device_id + session_id` 的用户卡片、滚动摘要和最近消息。
4. 读取 SQLite 最近 `action_sequence` 作为可选上下文。
5. 组装精简 Prompt 并调用快速回复模型。
6. SQLite 单事务写入带 `session_id` 的 user 和 assistant 原文。
7. Redis 更新当前 session 最近消息。
8. 按当天日期排队日期总结抽取和每日动作记忆抽取后台任务。
9. 如果回复模型返回高置信度 `action_sequence` 候选，把动作候选写入 SQLite `action_sequence`；时间/条件任务候选会被忽略。
10. 达到偏好抽取阈值或出现明确偏好表达时，按 `user_id + device_id` 调度后台偏好抽取任务。
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

`/api/query` 主链路只排队 `daily_time_memory_extract`，不等待摘要生成。后台 worker 从同一 `user_id + device_id + Asia/Shanghai 日期` 的所有 session 原始会话生成或覆盖当天唯一 `time_memory`。模型不可用时允许本地摘要降级，但降级结果仍是压缩摘要，不保存完整逐句 transcript。调试 UI 可以按日期重跑抽取，但不能手写摘要正文。

### 事件记忆库

事件库按日期展示机器狗行为事件记忆。`daily_action_memory_extract` 不在 `/api/query` 主链路自动执行；调试 UI 的“重跑日期事件抽取”按钮或未来定时任务会从同一天所有 session 原始会话抽取当天唯一一条 `action_memory`。模型不可用时允许从已存在动作路由或明确动作词做本地降级：

- `content` 保存当天动作链路 text，例如“事件记忆（2026-07-03）：\n事件链路：\n1. 用户要求 坐下 -> 机器狗完成 坐下\n2. 用户要求 转圈 -> 机器狗完成 转圈 -> 用户反馈：用户表示肯定”。
- `payload_json.memory_date` 保存归属日期。
- `payload_json.metadata.session_ids` 保存来源会话列表。
- `payload_json.metadata.action_chain_count` 保存当天聚合的动作链路数量。
- `payload_json.source_event_ids` / `source_message_event_ids` 保存来源消息或动作事件。

日期事件记忆只保留站起、坐下、转圈、跳舞、前进、后退等机器狗动态行为动作，以及用户对这些动作执行结果的反馈。提醒、出行计划、车票协助、安抚、心情和普通偏好不进入事件库。

Agent 消费事件库时使用 text-only 接口 `GET /api/memories/events-text`。该接口只返回 `id`、`event_type`、`text`、`created_at`、`memory_date`、`session_id` 和 `device_id`，不暴露 `payload_json`。

### 七天事件偏好抽取

事件库页的“七天事件偏好抽取”按钮会同步读取结束日期往前 7 天的 `action_memory.content`，拼接成 text 上下文后交给后台偏好抽取模型执行动作偏好抽取，方便调试页立即显示结果。模型返回的稳定动作偏好写成 `action_preference_memory` 文本事件；该入口不写入长期偏好表，也不在 `/api/query` 主链路执行。

### 动作事件链路

系统不再用本地动作词表解析用户原文。只有回复模型返回高置信度 `type='action_sequence'` 候选且包含 `actions` 数组时，系统才写入 `event_type='action_sequence'`；回复模型返回高置信度 `type='action_feedback'` 时，系统优先使用模型提供的动作引用，没有引用则自动关联同一用户和设备下最近一次动作序列。没有可关联动作时，反馈不会写入事件库。最近动作序列会作为普通上下文提供给回复模型，由模型自行判断是否用于“重复上次操作”等请求。

### 偏好记忆链路

普通 `/api/query` 不同步调用偏好抽取模型。系统在以下条件创建 `preference_extraction` 后台任务：

- 同一 `user_id + device_id` 新增 10 条 user message。
- 用户明确表达偏好或身份，例如“我喜欢摄影”“我不喜欢吵闹”“我是摄影师”“默认给我”“以后都”“记住”。

Worker 调用 Qwen / DashScope OpenAI-compatible 服务后，模型输出必须经过严格结构和原文来源校验，才能写入 `user_preferences` 和 `preference_evidence`。职业、喜欢、不喜欢归属 `user_id`，跨设备共享；偏好变化后清理该用户所有 Redis 用户卡片，卡片可从 SQLite 重建。

偏好记忆页的“运行一次偏好抽取”按钮会调用 `POST /api/debug/users/{user_id}/preferences/extract`。这是本地调试入口，默认 `force=true`，会重跑当前 `user_id` 和可选 `device_id` 的当前偏好抽取上下文，即使上一轮抽取已经成功但没有抽出偏好，也不会被 `latest_processed_event_id` 卡住。它不是 `/api/query` 主链路的一部分。

偏好任务仍由对话事件范围调度，但模型每次只接收其中一条当前用户原文，不接收 Redis 卡片、滚动摘要、旧 preferences、其他 Case 或 assistant 文本。SQLite `events` 是证据事实源，写入时由系统绑定当前原文的真实 `event_id`。结构化偏好只收敛为三类：

- `profile.occupation`：职业或身份，例如“我是摄影师”。
- `preference.likes`：明确喜欢的事物，例如“我喜欢摄影”“我喜欢周杰伦”。
- `preference.dislikes`：明确不喜欢的事物，例如“我不喜欢吵闹”。

三类结构化偏好记忆的合并规则不同：

- `profile.occupation` 是用户级单值槽位，新职业会 supersede 旧职业。
- `preference.likes` 和 `preference.dislikes` 是用户级去重多值集合，重复表达同一对象只增加 `evidence_count`。
- `preference.likes` / `preference.dislikes` 不按模型原始 `value_json` 字面量去重，而是按规范化对象去重；例如 `{"code":"travel","label_zh":"旅游"}`、`{"code":"旅游"}` 和 `display_text_zh="喜欢旅游"` 会归为同一个“旅游”偏好。
- `preference.likes` 和 `preference.dislikes` 对同一个对象互斥。比如用户先说“我喜欢吃苹果”，后面又说“我不喜欢吃苹果了”，新的“不喜欢苹果”会 active，旧的“喜欢苹果”会变成 `revoked`，但“喜欢旅游”等其他 likes 不受影响。
- SQLite 不物理删除旧偏好和证据；Redis 用户卡片只从 active 偏好重建。

明确偏好表达会立即创建后台任务，10 轮阈值仍是调度兜底。模型超时、非法 JSON、字段错误、空值或原文来源校验失败时不写入；job 会按现有重试上限重试，不使用本地规则生成答案。

内部注册表继续保留旧数据和其他模块使用的通用类型，但严格对话偏好抽取器只创建职业、喜欢、不喜欢三类，不允许模型自由生成其他 key。

每条用户原文的模型输出必须是完整 JSON，不能带 Markdown、解释或额外字段：

```json
{
  "type": "occupation | likes | dislikes | none",
  "value": "当前用户原文中的内容",
  "evidence": "包含 value 的当前原句",
  "confidence": 0.0
}
```

`value` 必须逐字存在于当前原文，`evidence` 必须是当前原文中包含该值的句子。`none` 必须使用空 `value/evidence`，属于合法无结果。`LONG_TERM_EXTRACTOR_MODE` 支持 `small`、`hybrid`、`large`：默认 `small` 使用 `codeqwen1.5-7b-chat`；`hybrid` 只在 small 调用或校验失败后使用 `qwen3.7-max`；`large` 直接使用 `qwen3.7-max`。两种模型都执行相同校验，temperature 固定为 0。

`/api/query` 遇到明确包含“长期记忆”“你记得”“记忆中”等信号，并询问职业、喜欢或不喜欢的问题时，会按 `user_id + exact preference_key` 查询 SQLite 并直接返回结构化值；无记录时返回“未记录”。包含“刚才”“当前会话”的问题仍走原短期链路。确定性长期问句会照常写入对话历史，但不会反向创建新的偏好抽取任务。

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
  "model": "qwen3.7-plus"
}
```

`user_id` 和 `device_id` 只允许字母、数字、下划线和连字符，最长 128 字符。`anonymous` 可以完成当前对话和短期缓存，但不会生成长期偏好任务。

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

`.env.example` 包含：

```dotenv
DASHSCOPE_API_KEY=
LLM_API_KEY=
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3.7-plus

PREFERENCE_EXTRACTOR_ENABLED=true
PREFERENCE_EXTRACTOR_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# 为空时默认复用 DASHSCOPE_API_KEY
PREFERENCE_EXTRACTOR_API_KEY=
PREFERENCE_EXTRACTOR_MODEL=qwen3.7-max
PREFERENCE_EXTRACT_BATCH_SIZE=8
PREFERENCE_EXTRACT_MIN_NEW_USER_MESSAGES=10
PREFERENCE_EXTRACT_MAX_ATTEMPTS=3

SHORT_MEMORY_SUMMARY_MIN_TURNS=20
SHORT_MEMORY_PROMPT_TRIGGER_TOKENS=5000
SHORT_MEMORY_RETAIN_RECENT_TURNS=5

REDIS_URL=redis://localhost:6379/0
REDIS_TTL_SECONDS=86400
REDIS_ALLOW_MEMORY_FALLBACK=true
REDIS_PREFIX=memory-os
```

不要把真实 API Key 写入 `.env.example` 或提交到仓库；真实 Key 只放在本机 `.env`。未单独填写 `PREFERENCE_EXTRACTOR_API_KEY` 时，偏好抽取器会优先复用 `DASHSCOPE_API_KEY`，再回退到 `LLM_API_KEY`；未单独填写 `PREFERENCE_EXTRACTOR_BASE_URL` 时会复用 `LLM_BASE_URL`。长期偏好默认使用 `small` 模式，普通回复和摘要仍使用 `LLM_MODEL`，两条链路互不降级。如果 UI 返回 401 或 `invalid_api_key`，说明当前 key 无效、过期，或与 `LLM_BASE_URL` 不匹配；`/api/status` 会显示实际配置。

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

短期记忆在当前 session 未触发滚动摘要前不会裁剪原文对话：20 轮以内全部进入回复 prompt；达到 20 轮后，只有完整 prompt 估算达到 `SHORT_MEMORY_PROMPT_TRIGGER_TOKENS` 时才摘要较早对话，并保留最近 `SHORT_MEMORY_RETAIN_RECENT_TURNS` 轮原文。短期记忆评测会先把每个 case 的前置对话写入当前 session，再调用一次回复模型回答追问，检查模型回复是否包含目标事实。默认数据集共 200 条，每条 10-20 轮，目标事实放在第 1-5 轮，后面追加机器狗短会话干扰内容。默认命令严格使用 Redis，Redis 不可用时会失败：

```bash
uv run python evaluation/run_short_term_eval.py
```

也可以直接使用 uv 创建的虚拟环境解释器：

```bash
.venv/bin/python evaluation/run_short_term_eval.py
```

不要直接运行 `python3 evaluation/run_short_term_eval.py`，因为系统 Python 通常没有 `redis`、`openai` 等项目依赖；`uv sync` 安装的是 `.venv` 环境。

评测会在控制台打印每条 case 的 `expected`、`question`、`answer`、目标事实所在轮次和距离追问的轮次数，并把整理后的完整输入输出写入 `evaluation/results/short_term_eval_<timestamp>.jsonl`。每个 case 使用独立 `memory-os-short-eval:*` Redis 前缀，case 结束后会清理对应测试 key。

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

长期结构化偏好记忆评测检查 SQLite 中最终保存的 `profile.occupation`、`preference.likes`、`preference.dislikes` 以及确定性 Verification。默认数据集 60 条，职业、喜欢、不喜欢各 20 条。每个 Case 在运行时使用独立 `user_id`、临时 SQLite 和随机 Redis prefix：

```bash
uv run python evaluation/run_preference_eval.py
```

先跑小样本：

```bash
uv run python evaluation/run_preference_eval.py --max-cases 5
```

评测会打印每条 Case 的输入、SQLite 抽取结果、验证问题和回答。Verification 按当前 `user_id + exact preference_key + active + scope=user` 查询 SQLite：有值时直接返回值，多值用顿号连接；没有值时只返回“未记录”，不调用回复模型。Case 只有 SQLite 和 Verification 都通过才计为通过。JSONL 会记录 `case_id`、运行时 `user_id`、`extractor_raw_output`、`extractor_validated_output`、`retrieved_long_term_facts`、`verification_prompt` 和 `fallback_used`：

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
- 页面主行会直接显示 `错误:` 或 `模型警告:`；`模型警告` 表示本地明确偏好已经保存，但外部抽取模型失败，完整错误在“错误详情”里。
- `process.recovered_stale > 0`：说明旧的 `running` job 卡住，系统已自动恢复并重新处理。
- `process.succeeded > 0` 但偏好为空：说明模型判断当前事件没有稳定长期偏好，或低置信度结果被写成 candidate/rejected。
- `/api/status` 的 `preference_extractor.configured` 必须为 `true`，否则不会调用 Qwen 抽取模型。
- 失败达到 `PREFERENCE_EXTRACT_MAX_ATTEMPTS` 后不会无限重试；修复配置后可再次点击手动抽取按钮创建或合并新任务。

## 迁移

运行：

```bash
uv run python -m memory.migrate
```

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
