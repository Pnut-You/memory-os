# Memory OS 调试 UI

FastAPI + 原生 HTML/CSS/JavaScript 调试界面，不引入 React、Vue 或 Node 构建链。

页面使用响应式布局：桌面端多列显示，较窄窗口自动改为单列，长内容在卡片内部滚动。服务端提供 `/favicon.ico`，避免浏览器默认 favicon 请求产生 404。

## 页面

- 对话测试：输入 `user_id`、`device_id`、`query`，显示回复、总耗时、模型耗时、用户卡片版本和摘要版本。
- 对话测试：输入 query 后按 Enter 发送，Shift+Enter 换行。
- 偏好记忆：查看 Redis 用户卡片、active/candidate 结构化偏好、滚动摘要和证据详情。
- 日期总结：查看和手动新增某一天做过事情的文本摘要。
- 事件摘要库：查看和手动新增独立事件摘要，也可筛选动作事件、对话消息、日期总结等事件。
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
POST /api/debug/events/summaries
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
cd ~/pt/projects/i/memory-os
uv sync
REDIS_ALLOW_MEMORY_FALLBACK=true uv run uvicorn ui.app:app --reload --host 127.0.0.1 --port 8000
```

生产环境需要 Redis；只有显式开发/测试配置允许内存降级。

## 查看位置

- 对话测试：发送后会显示“请求链路”，包括请求输入、上下文读取、滚动摘要、偏好记忆、最近对话、文本记忆写入、动作事件路由、SQLite 写入、回复模型输入和回复模型输出。
- 滚动摘要：偏好记忆页的“滚动摘要”面板会显示版本、压缩事件范围和本次压缩轮数；默认 10 轮触发，压缩较早 5 轮并保留最近 5 轮原文。摘要正文会从 SQLite 最近最多 20 轮已压缩会话重写，并限制在约 1600 字符内，不会无限拼接旧摘要。摘要完成后会触发后台偏好抽取；日期总结不做历史补写。
- 结构化偏好：偏好记忆页按“职业 / 喜欢 / 明确不喜欢”三类展示偏好记忆，其他旧 key 或 candidate 放在“其他 / 历史偏好”里；按钮默认 `force=true`，会重跑当前滚动摘要 + 摘要证据原话 + 最近 5 轮完整会话 + 最近动作事件，并显示事件范围、摘要版本、最近轮次数、输入事件数、claimed、succeeded、failed、recovered 和错误详情。后端返回纯文本错误时页面会直接显示错误文本，不再显示 JSON parse 报错。
- 日期总结：日期总结页保存 `event_type='time_memory'` 的文本摘要，`content` 是当天做过事情的总结，`payload_json.memory_date` 是归属日期，`payload_json.memory_at` 是带时区时间戳；页面可手动新增，不再展示 `target_at` 或定时任务字段。
- 事件库：事件库页默认显示 `event_summary` 文本事件摘要，也可以筛选 `time_memory`、`message` 或 `action_sequence`。动作事件只来自回复模型返回的高置信度 `action_sequence` 候选；时间/条件任务候选不会从 `/api/query` 自动写入。
- 设备状态：设备实时状态页可查询最新快照、历史记录，也可本地调试写入 `battery_percent`、`charging`、`network`、`location`、`motion_state`、`temperature_c`。
