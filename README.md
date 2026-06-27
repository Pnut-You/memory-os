# Memory OS

Memory OS 是面向机器狗、IoT 设备和通用 Agent 的轻量 Python 记忆模块。核心模块不依赖 Web 框架；本地调试界面使用 FastAPI 和原生 HTML/CSS/JavaScript。

## 架构边界

- 长期记忆归属：`user_id`。
- 短期上下文归属：`user_id + device_id`。
- 设备实时状态归属：`device_id`。
- 对外没有 Session，也不需要 start/end 调用。
- Redis 是缓存和实时状态层；生产环境 Redis 不可用应报错。
- SQLite 是唯一持久化事实源。
- 智谱 OpenAI-compatible 模型用于后台偏好抽取，不进入 `/api/query` 主链路。
- 主请求只读取紧凑用户卡片、滚动摘要、最近消息和必要事件上下文。
- 旧语义索引组件已删除，不使用向量库或 embedding。

## Redis Key 与作用

```text
memory-os:conversation:{device_id}:{user_id}
memory-os:summary:{device_id}:{user_id}
memory-os:user-card:{user_id}
memory-os:user-preferences:{user_id}
memory-os:device-state:{device_id}
memory-os:pending-event:{device_id}:{user_id}
memory-os:action-buffer:{device_id}:{user_id}
```

- `conversation`：最近 10 轮对话缓存，最多 20 条消息，可从 SQLite 原始事件恢复。
- `summary`：某个用户在某台设备上的滚动摘要快照，可从 `conversation_summaries` 恢复。
- `user-card`：L0 用户记忆卡片，只放最重要的 active 偏好，可从 `user_preferences` 重建。
- `user-preferences`：可选偏好缓存，事实源仍是 SQLite。
- `device-state`：设备最新实时快照，TTL 过期即视为离线。
- `pending-event`：缺少时间、地点或任务字段的待补全事件，下一轮用户补充后转成正式时间记忆。
- `action-buffer`：当前用户在当前设备上的最近单步动作缓存；用户说“重复刚才的动作”时优先使用它。

## SQLite 表结构与作用

- `events`
  - 保存原始事实事件，包括用户消息、assistant 回复、时间记忆、动作序列、迁移遗留事件等。
  - 核心字段：`request_id`、`user_id`、`device_id`、`event_type`、`role`、`content`、`payload_json`、`created_at`。
  - `event_type='message'` 是对话原文；`scheduled_task`、`recurring_task`、`conditional_task`、`pending_event` 表示时间/条件类记忆；`action_sequence` 是机器狗动作序列。

- `conversation_summaries`
  - 保存按 `user_id + device_id` 隔离的滚动摘要。
  - `summary_text` 是摘要正文，`compacted_through_event_id` 表示压缩到哪个事件，`version` 表示摘要版本。
  - `from_event_id`、`to_event_id`、`turn_count` 记录本次实际压缩的事件范围和轮次数，便于调试摘要边界。

- `user_preferences`
  - 保存 L1 结构化用户偏好。
  - `preference_key` 必须来自固定偏好注册表或进入 candidate/other。
  - `status` 支持 `candidate`、`active`、`superseded`、`revoked`、`rejected`。
  - `supersedes_id` 记录偏好冲突替代关系。

- `preference_evidence`
  - 保存偏好证据，连接 `user_preferences` 和原始 `events`。
  - 用于调试“某条偏好是从哪句话来的”。

- `memory_jobs`
  - 后台任务队列。
  - 当前任务类型：`conversation_summary`、`preference_extraction`、`user_card_rebuild`。
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
2. Redis pipeline 读取用户卡片、滚动摘要、最近消息。
3. 如用户说“重复上次操作”，优先读取 Redis `action-buffer`，没有缓存时再读取 SQLite 最近 `action_sequence` 作为上下文。
4. 组装精简 Prompt 并调用快速回复模型。
5. SQLite 单事务写入 user 和 assistant 原文。
6. Redis 更新最近消息。
7. 本地轻量规则识别时间记忆、待补全事件和动作事件，写入 SQLite 或短期 Redis buffer。
8. 达到 10 条 user message、出现明确偏好表达或动作习惯表达时，调度后台偏好抽取任务。
9. 返回精简响应。

主请求不做模型偏好抽取、长期偏好合并、用户卡片重建、向量检索、embedding、全量历史扫描或同步摘要生成。

### 滚动摘要链路

摘要按 `user_id + device_id` 隔离，使用 SQLite 原始对话作为事实源，不依赖 Redis 判断边界。

- 默认 `SUMMARY_EVERY_TURNS=10`、`SUMMARY_RETAIN_TURNS=5`。
- 第一次达到 10 轮完整对话时，只压缩第 1-5 轮，保留第 6-10 轮原文。
- 达到 15 轮时，只压缩第 6-10 轮，保留第 11-15 轮原文。
- 后续每新增 5 轮重复该流程。
- `conversation_summaries.from_event_id` 和 `to_event_id` 可用于确认本次压缩范围。
- 摘要正文不是无限拼接旧摘要；每次后台摘要会从 SQLite 重新取最近最多 20 轮已压缩会话生成有限窗口摘要，摘要正文硬限制为约 1600 字符，更早的稳定偏好应进入 `user_preferences` 和 Redis 用户卡片。

摘要生成在后台线程执行，失败不会阻塞 `/api/query`。

### 时间记忆链路

用户说：

```text
明天早上九点给我播放新闻
明天早上9点钟要叫我起床
后天晚上8点提醒我吃药
```

本地规则识别常见中文时间表达，写入 `events`：

- `event_type='scheduled_task'` 或 `event_type='recurring_task'`。
- `content` 保存用户原话。
- `payload_json.target_at` 保存目标时间。
- `payload_json.task` 保存任务文本。
- `created_at` 保存写入时间。

本地规则支持 `提醒`、`提醒我`、`播放`、`叫我`、`叫醒`、`唤醒`、`起床`、`闹钟`、`通知`、`新闻`、`音乐` 等轻量关键词。它会覆盖相对时间、明天/后天、下周几、每天/每周等常见表达。

如果用户只说“提醒我喝水”但没有时间，系统会写入 `pending_event`，并把待补全事件放入 Redis `pending-event`。下一轮用户补充“10分钟后”或“8点半”时，系统会把它转成正式 `scheduled_task`。如果用户说“如果电量低于20%就提醒我充电”，系统写入 `conditional_task`，不会当作普通时间提醒。

回复模型可以在本次 `/api/query` 的普通回复中附带候选 `event_routes`，但系统不会做第二次模型调用。模型候选只作为待验证输入，代码会校验类型、置信度和动作，不能绕过本地规则直接写长期记忆。调试模式会在请求链路中显示 `time_memory_routing`，用于确认本次是否入库。摘要完成后还会在后台扫描本次摘要窗口内的用户原话，补漏时间记忆，并用 `source_event_id` 去重，避免和单句实时路由重复。

### 动作事件链路

用户说：

```text
往前走往后走往左走然后坐下
```

系统写入 `event_type='action_sequence'`，`payload_json.actions` 保存动作顺序。之后用户说“重复上次操作”时，最近动作序列会进入回复模型上下文；真实硬件执行层后续可以直接消费这个结构。摘要完成后，后台会把同一窗口内连续多轮短动作消息合并为一个动作序列事件，例如“坐下”+“站起来”+“往前走”，并在 `payload_json.source_event_ids` 中记录来源消息。

如果用户只说“坐下”这种单步动作，系统先写入 Redis `action-buffer`，不立刻固化成 `action_sequence`。用户继续说“站起来”“往前走”时，buffer 会累积当前动作链；用户说“好的”“谢谢”等确认词不会清空 buffer。后续“重复刚才的动作”优先使用当前 buffer，其次使用最近已经固化的动作事件。后台摘要扫描只负责把连续多轮短动作补漏合并为事件库记录，并用 `source_event_ids` 避免重复写入。

### 长期偏好链路

普通 `/api/query` 不同步调用偏好抽取模型。系统在以下条件创建 `preference_extraction` 后台任务：

- 同一 `user_id` 新增 10 条 user message。
- 用户明确表达偏好或身份，例如“我喜欢摄影”“我不喜欢吵闹”“我是摄影师”“默认给我”“以后都”“记住”。
- 用户把动作序列表达为习惯，例如“以后默认往前走然后坐下”。

Worker 调用智谱 OpenAI-compatible 服务后，模型输出必须经过结构校验和确定性合并，才能写入 `user_preferences` 和 `preference_evidence`。偏好变化后创建 `user_card_rebuild` 任务，重建 Redis 用户卡片。

长期记忆页的“运行一次偏好抽取”按钮会调用 `POST /api/debug/users/{user_id}/preferences/extract`。这是本地调试入口，默认 `force=true`，会重跑当前 `user_id` 和可选 `device_id` 的当前偏好抽取上下文，即使上一轮抽取已经成功但没有抽出偏好，也不会被 `latest_processed_event_id` 卡住。它不是 `/api/query` 主链路的一部分。

偏好抽取上下文使用 `summary_plus_recent_turns`：

- Redis/SQLite 用户卡片。
- 当前 `user_id + device_id` 的滚动摘要。
- 摘要覆盖范围内的原始 user message 证据 `summary_evidence_events`。
- 最近 5 轮完整 user/assistant 会话。
- 最近动作事件。
- 现有 active/candidate preferences。

SQLite `events` 仍是事实源。滚动摘要用于补充历史上下文，最近 5 轮和动作事件提供可引用的真实 `event_id` 证据。长期结构化偏好优先收敛为三类：

- `profile.occupation`：职业或身份，例如“我是摄影师”。
- `preference.likes`：明确喜欢的事物，例如“我喜欢摄影”“我喜欢周杰伦”。
- `preference.dislikes`：明确不喜欢的事物，例如“我不喜欢吵闹”。

三类结构化长期记忆的合并规则不同：

- `profile.occupation` 是单值槽位。同一用户只保留一个 active 职业；如果用户说“我现在转行成医生”，旧的“程序员”会变成 `superseded`，新职业成为 active。
- `preference.likes` 是多值集合。同一用户可以同时 active “喜欢旅游”“喜欢摄影”“喜欢吃苹果”；重复表达同一对象只增加 `evidence_count`。
- `preference.dislikes` 是多值集合。同一用户可以同时 active “不喜欢香菜”“不喜欢吵闹”；重复表达同一对象只增加 `evidence_count`。
- `preference.likes` / `preference.dislikes` 不按模型原始 `value_json` 字面量去重，而是按规范化对象去重；例如 `{"code":"travel","label_zh":"旅游"}`、`{"code":"旅游"}` 和 `display_text_zh="喜欢旅游"` 会归为同一个“旅游”偏好。
- `preference.likes` 和 `preference.dislikes` 对同一个对象互斥。比如用户先说“我喜欢吃苹果”，后面又说“我不喜欢吃苹果了”，新的“不喜欢苹果”会 active，旧的“喜欢苹果”会变成 `revoked`，但“喜欢旅游”等其他 likes 不受影响。
- SQLite 不物理删除旧偏好和证据；Redis 用户卡片只从 active 偏好重建。

因此长期偏好不是只基于摘要抽取；即使 5 轮内还没有触发摘要，只要最近 5 轮上下文里有“我喜欢摄影”，后台抽取器就能看到这句话。摘要完成后也会自动创建覆盖该摘要窗口的偏好抽取任务，如果摘要里有“用户不喜欢吃香菜”，抽取器会结合 `summary_evidence_events` 回溯原始证据；本地规则还会兜底识别“喜欢/不喜欢/职业”三类结构化长期记忆。明确偏好表达会立即创建后台抽取任务，10 轮阈值只是兜底。本地明确规则会先写入 SQLite，随后模型结果再进入确定性合并；模型结果优先，规则只补模型漏掉的明确 likes/dislikes/occupation，例如模型抽到“喜欢旅游”但漏掉“我喜欢吃苹果”时，本地规则会补入“喜欢苹果”。如果外部模型失败但本地规则已经写入偏好，本次 job 记为部分成功并返回 `model_error`，不会丢掉已确定的本地长期记忆。

内部注册表也保留更通用的长期记忆类型：`profile`、`preference`、`habit`、`constraint`、`relationship`、`default_behavior`。现阶段 UI 优先展示职业、喜欢、不喜欢三类，其他类型会以 candidate 或 other 形式保留，避免模型自由生成未知 key 直接进入用户卡片。

偏好抽取输入包含：

- 滚动摘要：`summary_version`、摘要文本和压缩事件范围。
- 摘要证据：摘要覆盖范围内最多 20 轮用户原话。
- 最近 5 轮对话：每条 message 的 `event_id`、`role`、`text`、`created_at`。
- 最近动作事件：`event_id`、`device_id`、`content`、`payload_json.actions`、`created_at`。
- 现有 active/candidate preferences。
- 固定 preference registry。

模型输出字段：

```json
{
  "preference_key": "preference.likes",
  "category": "preference",
  "value": {"type": "string", "code": "photography", "label_zh": "摄影"},
  "display_text_zh": "喜欢摄影",
  "polarity": "prefer",
  "durability": "persistent",
  "strength": 0.85,
  "confidence": 0.96,
  "source_type": "explicit",
  "scope": "user",
  "reason_zh": "用户明确说喜欢摄影",
  "evidence": [{"event_id": 105, "text": "我喜欢摄影", "type": "explicit"}],
  "expires_at": null,
  "action": "upsert"
}
```

高置信明确偏好可进入 active；中等置信进入 candidate；未知 key 进入 other/candidate，不进入用户卡片。

如果模型要撤回多值偏好，必须返回 `action="revoke"` 并带上同一个 `value`；系统只撤回匹配 `preference_key + value` 的 active 记录，不会因为撤回“喜欢苹果”而删除“喜欢旅游”。

模型理想输出是包含 `schema_version`、`user_id`、`preferences` 的 JSON 对象。为了兼容调试模型偶尔返回数组或单条 preference，后台会用当前 `user_id` 做保守包装后再校验；仍无法解析时只记录 job error，不写入长期记忆。

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
  "model": "glm-4-flash"
}
```

`user_id` 和 `device_id` 只允许字母、数字、下划线和连字符，最长 128 字符。`anonymous` 可以完成当前对话和短期缓存，但不会生成长期偏好任务。

### 调试 API

```text
GET /api/status
GET /api/debug/users/{user_id}
GET /api/debug/users/{user_id}/preferences
POST /api/debug/users/{user_id}/preferences/extract
GET /api/debug/users/{user_id}/events
GET /api/debug/users/{user_id}/time-memories
POST /api/debug/users/{user_id}/time-memories
GET /api/debug/events
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
- 如果返回“当前 user_id/device_id 下没有原始用户消息”，说明该调试范围内没有可送给模型的原始用户事件；先检查长期记忆页的“最近原始用户消息”。

## 调试 UI

页面包含：

- 对话测试：输入 query 后按 Enter 发送，Shift+Enter 换行。
- 对话测试：显示类似 LangSmith 的“请求链路”，包括请求输入、上下文读取、滚动摘要、长期记忆、最近对话、时间记忆路由、动作事件路由、SQLite 写入、回复模型输入和回复模型输出。
- 长期记忆：查看用户卡片、偏好、滚动摘要和证据。
- 长期记忆页的抽取诊断会显示事件范围、摘要版本、最近轮次数、动作数、偏好抽取上下文预览、最新成功处理事件和 worker 错误详情。
- 时间记忆：查看自动识别出的时间任务，也可手动新增。
- 事件库：查看动作事件、对话消息、时间记忆等原始事件。
- 设备实时状态：查看在线状态、最新快照、历史记录，并用少量核心字段写入调试状态。

## 配置

`.env.example` 包含：

```dotenv
LLM_API_KEY=
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_MODEL=glm-4-flash

PREFERENCE_EXTRACTOR_ENABLED=true
PREFERENCE_EXTRACTOR_BASE_URL=https://open.bigmodel.cn/api/paas/v4
PREFERENCE_EXTRACTOR_API_KEY=
PREFERENCE_EXTRACTOR_MODEL=glm-4-flash
PREFERENCE_EXTRACT_BATCH_SIZE=8
PREFERENCE_EXTRACT_MIN_NEW_USER_MESSAGES=10
PREFERENCE_EXTRACT_MAX_ATTEMPTS=3

REDIS_URL=redis://localhost:6379/0
REDIS_TTL_SECONDS=86400
REDIS_ALLOW_MEMORY_FALLBACK=false
REDIS_PREFIX=memory-os
```

不要写入真实 API Key。未单独填写 `PREFERENCE_EXTRACTOR_API_KEY` 时，偏好抽取器会复用 `LLM_API_KEY`；未单独填写 `PREFERENCE_EXTRACTOR_BASE_URL` 和 `PREFERENCE_EXTRACTOR_MODEL` 时，会复用 `LLM_BASE_URL` 和 `LLM_MODEL`，默认适配智谱 OpenAI-compatible 接口。

## 本地运行

核心测试：

```bash
python3 -m unittest discover -s tests -v
python3 example.py
```

调试 UI：

```bash
ui/.venv/bin/python -m pip install -r ui/requirements.txt
REDIS_ALLOW_MEMORY_FALLBACK=true ui/.venv/bin/uvicorn ui.app:app --reload --host 127.0.0.1 --port 8000
```

调试 UI 使用响应式布局，页面整体可滚动，卡片内部按内容区域滚动，适配笔记本、桌面和窄屏浏览器窗口。浏览器自动请求的 `/favicon.ico` 已由 UI 服务兜底返回，不会再产生 404 日志。

如果 `8000` 被占用：

```bash
REDIS_ALLOW_MEMORY_FALLBACK=true ui/.venv/bin/uvicorn ui.app:app --reload --host 127.0.0.1 --port 8001
```

## 偏好抽取故障排查

- 点击“运行一次偏好抽取”没有偏好：先看页面上的 `process.errors`，常见原因是模型超时、Key 无效、模型返回非 JSON 或返回字段缺失。
- 页面不再把 `Internal Server Error` 当 JSON 解析；如果后端或代理返回纯文本错误，会直接显示文本摘要。
- 页面主行会直接显示 `错误:` 或 `模型警告:`；`模型警告` 表示本地明确偏好已经保存，但外部抽取模型失败，完整错误在“错误详情”里。
- `process.recovered_stale > 0`：说明旧的 `running` job 卡住，系统已自动恢复并重新处理。
- `process.succeeded > 0` 但偏好为空：说明模型判断当前事件没有稳定长期偏好，或低置信度结果被写成 candidate/rejected。
- `/api/status` 的 `preference_extractor.configured` 必须为 `true`，否则不会调用智谱抽取模型。
- 失败达到 `PREFERENCE_EXTRACT_MAX_ATTEMPTS` 后不会无限重试；修复配置后可再次点击手动抽取按钮创建或合并新任务。

## 迁移

运行：

```bash
python3 -m memory.migrate
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
REDIS_ALLOW_MEMORY_FALLBACK=true ui/.venv/bin/python -m uvicorn ui.app:app   --reload   --host 127.0.0.1   --port 8000
