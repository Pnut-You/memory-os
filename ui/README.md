# Memory OS 调试 UI

FastAPI + 原生 HTML/CSS/JavaScript 调试界面，不引入 React、Vue 或 Node 构建链。

页面使用响应式布局：桌面端多列显示，较窄窗口自动改为单列，长内容在卡片内部滚动。服务端提供 `/favicon.ico`，避免浏览器默认 favicon 请求产生 404。

## 页面

- 对话测试：输入 `user_id`、`device_id`、`query`，显示回复、总耗时、模型耗时、用户卡片版本和摘要版本。
- 对话测试：输入 query 后按 Enter 发送，Shift+Enter 换行。
- 长期记忆：查看 Redis 用户卡片、active/candidate 结构化偏好、滚动摘要和证据详情。
- 时间记忆：查看自动识别出的时间任务，也可手动新增。
- 事件库：查看动作事件、对话消息、时间记忆等事件。
- 设备实时状态：查看在线状态、最新快照、历史记录，并写入少量核心调试状态字段。

## API

```text
POST /api/query
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

`POST /api/query` 请求格式：

```json
{
  "user_id": "user-001",
  "device_id": "dog-001",
  "query": "带我去安静一点的地方"
}
```

## 运行

```bash
ui/.venv/bin/python -m pip install -r ui/requirements.txt
ui/.venv/bin/uvicorn ui.app:app --reload --host 127.0.0.1 --port 8000
```

生产环境需要 Redis；只有显式开发/测试配置允许内存降级。

## 查看位置

- 对话测试：发送后会显示“请求链路”，包括请求输入、上下文读取、滚动摘要、长期记忆、最近对话、时间记忆路由、动作事件路由、SQLite 写入、回复模型输入和回复模型输出。
- 滚动摘要：长期记忆页的“滚动摘要”面板会显示版本、压缩事件范围和本次压缩轮数；默认 10 轮触发，压缩较早 5 轮并保留最近 5 轮原文。摘要正文会从 SQLite 最近最多 20 轮已压缩会话重写，并限制在约 1600 字符内，不会无限拼接旧摘要。摘要完成后会触发后台补漏：偏好抽取、时间记忆扫描和跨轮动作合并。
- 结构化偏好：长期记忆页按“职业 / 喜欢 / 明确不喜欢”三类展示长期记忆，其他旧 key 或 candidate 放在“其他 / 历史偏好”里；按钮默认 `force=true`，会重跑当前滚动摘要 + 摘要证据原话 + 最近 5 轮完整会话 + 最近动作事件，并显示事件范围、摘要版本、最近轮次数、输入事件数、claimed、succeeded、failed、recovered 和错误详情。后端返回纯文本错误时页面会直接显示错误文本，不再显示 JSON parse 报错。
- 时间记忆：时间记忆页，`target_at` 是目标时间，`created_at` 是写入时间；对话里说“明天早上9点钟要叫我起床”“后天晚上8点提醒我吃药”会自动进入 `scheduled_task`，每天/每周类表达进入 `recurring_task`，缺时间的提醒先进入 `pending_event`，条件触发类提醒进入 `conditional_task`。结果也会出现在请求链路的 `time_memory_routing`；摘要后后台扫描会补漏但不会重复写入。
- 事件库：事件库页默认显示动作事件，也可以筛选时间/条件记忆聚合项或单独筛选 `scheduled_task`、`recurring_task`、`conditional_task`、`pending_event`。“重复上次操作”会优先使用 Redis action-buffer，再使用最近已固化动作序列作为上下文。摘要后后台扫描会把连续多轮“坐下 / 站起来 / 往前走”合并为一个动作序列，并在原始数据中显示 `source_event_ids`。
- 设备状态：设备实时状态页可查询最新快照、历史记录，也可本地调试写入 `battery_percent`、`charging`、`network`、`location`、`motion_state`、`temperature_c`。
