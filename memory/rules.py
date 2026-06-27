"""Lightweight Chinese rule parsers for local memory routing."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo


TIME_KEYWORDS = (
    "提醒",
    "提醒我",
    "播放",
    "放",
    "叫我",
    "叫醒",
    "唤醒",
    "起床",
    "闹钟",
    "通知",
    "新闻",
    "音乐",
)
REPEAT_ACTION_KEYWORDS = (
    "重复上次操作",
    "重复刚才操作",
    "重复上次动作",
    "重复刚才动作",
    "重复刚才的动作",
    "重复上一次动作",
    "再来一次",
    "照刚才做",
)
CONFIRMATION_KEYWORDS = ("好的", "好了", "可以了", "嗯", "谢谢", "好")
CONDITION_KEYWORDS = ("到", "后", "低于", "回到", "进入")
CANCEL_KEYWORDS = ("取消", "删除", "别提醒", "不用提醒")
UPDATE_KEYWORDS = ("改到", "修改到", "推迟", "提前")

ACTION_PATTERNS: list[tuple[str, str]] = [
    ("forward", "往前走"),
    ("backward", "往后走"),
    ("left", "往左走"),
    ("right", "往右走"),
    ("sit", "坐下"),
    ("stand", "站起来"),
    ("stop", "停止"),
    ("turn_around", "转身"),
]

CHINESE_HOURS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
}
WEEKDAYS = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}


def parse_time_memory(
    text: str,
    *,
    now: datetime | None = None,
    timezone_name: str = "Asia/Shanghai",
) -> dict[str, Any] | None:
    routed = parse_event_route(text, now=now, timezone_name=timezone_name)
    if routed and routed["type"] in {"scheduled_task", "recurring_task"} and routed["decision"] == "create":
        return {
            "target_at": routed.get("target_at"),
            "task": routed.get("task"),
            "parser": routed.get("parser", "local-rule-v2"),
            "event_type": routed["type"],
            "recurrence": routed.get("recurrence"),
        }
    return None


def parse_event_route(
    text: str,
    *,
    now: datetime | None = None,
    timezone_name: str = "Asia/Shanghai",
) -> dict[str, Any] | None:
    if _is_cancel(text):
        return {
            "type": "scheduled_task",
            "decision": "cancel",
            "confidence": 0.9,
            "missing_fields": [],
            "task": _task_text(text),
            "parser": "local-rule-v2",
        }
    if _is_update(text):
        return {
            "type": "scheduled_task",
            "decision": "update",
            "confidence": 0.8,
            "missing_fields": [],
            "task": _task_text(text),
            "target_at": _relative_or_absolute_time(text, now=now, timezone_name=timezone_name),
            "parser": "local-rule-v2",
        }
    if _conditional(text):
        return {
            "type": "conditional_task",
            "decision": "create",
            "confidence": 0.85,
            "missing_fields": [],
            "condition": _condition_text(text),
            "task": _task_text(text),
            "parser": "local-rule-v2",
        }
    if not any(keyword in text for keyword in TIME_KEYWORDS):
        return None
    tz = ZoneInfo(timezone_name)
    base = now.astimezone(tz) if now else datetime.now(tz)
    recurrence = _recurrence(text)
    target = _relative_or_absolute_time(text, now=base, timezone_name=timezone_name)
    if target is None:
        return {
            "type": "pending_event",
            "decision": "clarify",
            "confidence": 0.75,
            "missing_fields": ["time_or_condition"],
            "task": _task_text(text),
            "parser": "local-rule-v2",
        }
    return {
        "type": "recurring_task" if recurrence else "scheduled_task",
        "decision": "create",
        "confidence": 0.9,
        "missing_fields": [],
        "target_at": target,
        "task": _task_text(text),
        "recurrence": recurrence,
        "parser": "local-rule-v2",
    }


def _relative_or_absolute_time(text: str, *, now: datetime | None, timezone_name: str) -> str | None:
    tz = ZoneInfo(timezone_name)
    base = now.astimezone(tz) if now else datetime.now(tz)
    relative = re.search(r"(\d+|半)\s*(分钟|小时|个小时)后", text)
    if relative:
        amount_text, unit = relative.group(1), relative.group(2)
        amount = 0.5 if amount_text == "半" else int(amount_text)
        delta = timedelta(hours=amount) if "小时" in unit else timedelta(minutes=amount)
        return (base + delta).replace(microsecond=0).isoformat()
    day_offset = _day_offset(text)
    weekday_offset = _weekday_offset(text, base)
    if day_offset is None and weekday_offset is None and _recurrence(text) is None:
        return None
    hour = _hour(text)
    if hour is None:
        return None
    minute = 30 if "半" in text else 0
    if any(word in text for word in ("下午", "晚上", "傍晚")) and hour < 12:
        hour += 12
    days = weekday_offset if weekday_offset is not None else (day_offset or 0)
    target = (base + timedelta(days=days)).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )
    return target.isoformat()


def parse_action_sequence(text: str) -> dict[str, Any] | None:
    matches: list[tuple[int, dict[str, str]]] = []
    for code, label in ACTION_PATTERNS:
        for match in re.finditer(re.escape(label), text):
            matches.append((match.start(), {"code": code, "label_zh": label}))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0])
    return {
        "actions": [item for _, item in matches],
        "parser": "local-rule-v1",
    }


def is_repeat_action_request(text: str) -> bool:
    return any(keyword in text for keyword in REPEAT_ACTION_KEYWORDS)


def is_confirmation(text: str) -> bool:
    return text.strip() in CONFIRMATION_KEYWORDS


def _day_offset(text: str) -> int | None:
    if "后天" in text:
        return 2
    if "明天" in text:
        return 1
    if "今天" in text or "今晚" in text:
        return 0
    return None


def _weekday_offset(text: str, base: datetime) -> int | None:
    match = re.search(r"下周([一二三四五六日天])", text)
    if not match:
        return None
    target = WEEKDAYS[match.group(1)]
    return ((target - base.weekday()) % 7) + 7


def _recurrence(text: str) -> dict[str, Any] | None:
    if "每天" in text or "每日" in text:
        return {"type": "daily"}
    match = re.search(r"每周([一二三四五六日天])", text)
    if match:
        return {"type": "weekly", "weekday": WEEKDAYS[match.group(1)]}
    return None


def _conditional(text: str) -> bool:
    return any(pattern in text for pattern in ("到客厅后", "回到家", "电量低于")) or bool(
        re.search(r"(到.+后|回到.+|.+低于\d+%?)", text)
    )


def _condition_text(text: str) -> str:
    for sep in ("后", "提醒", "播放"):
        if sep in text:
            return text.split(sep, 1)[0] + (sep if sep == "后" else "")
    return text


def _is_cancel(text: str) -> bool:
    return any(keyword in text for keyword in CANCEL_KEYWORDS)


def _is_update(text: str) -> bool:
    return any(keyword in text for keyword in UPDATE_KEYWORDS)


def _hour(text: str) -> int | None:
    match = re.search(r"([0-2]?\d)\s*[点:时]", text)
    if match:
        hour = int(match.group(1))
        return hour if 0 <= hour <= 23 else None
    for word in sorted(CHINESE_HOURS, key=len, reverse=True):
        if f"{word}点" in text or f"{word}时" in text:
            return CHINESE_HOURS[word]
    return None


def _task_text(text: str) -> str:
    cleaned = re.sub(
        r"(今天|明天|后天|今晚)?(早上|上午|中午|下午|晚上|傍晚)?[零一二两三四五六七八九十\d]{1,3}\s*(点半|点钟|点|时|:)?",
        "",
        text,
    )
    cleaned = re.sub(r"^(要|帮我|请|到时候|记得|提醒我|给我)\s*", "", cleaned.strip())
    return cleaned.strip(" ，。,.") or text
