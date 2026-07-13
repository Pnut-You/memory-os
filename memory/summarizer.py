"""Conversation summarization with an OpenAI-compatible API and local fallback."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .config import QWEN_BASE_URL, QWEN_CHAT_MODEL


class Summarizer:
    MAX_SUMMARY_CHARS = 1600
    MAX_DAILY_SUMMARY_CHARS = 1200

    def __init__(self, api_key: str = "", base_url: str = QWEN_BASE_URL, model: str = QWEN_CHAT_MODEL) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    @property
    def backend(self) -> str:
        return "llm" if self.api_key else "local"

    def summarize(
        self,
        messages: list[dict[str, Any]],
        previous_summary: str = "",
    ) -> str:
        if not messages:
            return previous_summary
        if self.api_key:
            try:
                return self._call_llm(messages, previous_summary)
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError):
                pass
        return self._local_summary(messages, previous_summary)

    def summarize_daily(
        self,
        messages: list[dict[str, Any]],
        memory_date: str,
        *,
        max_chars: int | None = None,
    ) -> str:
        if not messages:
            return ""
        limit = max_chars or self.MAX_DAILY_SUMMARY_CHARS
        if self.api_key:
            try:
                return self._call_daily_llm(messages, memory_date, limit)
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError):
                pass
        return self._local_daily_summary(messages, memory_date, limit)

    def extract_daily_action_memories(
        self,
        messages: list[dict[str, Any]],
        memory_date: str,
    ) -> dict[str, Any]:
        if not messages:
            return {"backend": self.backend, "memories": []}
        if self.api_key:
            try:
                return {"backend": "llm", "memories": self._call_action_memory_llm(messages, memory_date)}
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError):
                pass
        return {"backend": "local", "memories": self._local_action_memories(messages, memory_date)}

    def _call_llm(self, messages: list[dict[str, Any]], previous_summary: str) -> str:
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        previous = previous_summary or "（暂无，这是第一次压缩）"
        payload = {
            "model": self.model,
            "enable_thinking": False,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你负责维护机器狗对话的有限窗口摘要。输入是最近最多20轮已压缩范围内的完整对话。"
                        "请重写一份新的窗口摘要，不要累计拼接旧摘要，不要保留输入之外更早的内容。"
                        "保留人物、偏好、事实、地点、任务、承诺和未完成事项；消除重复，不要编造。使用简洁中文。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"旧摘要仅供参考，不要直接拼接：\n{previous}\n\n最近窗口对话：\n{transcript}\n\n请输出不超过800字的中文摘要。",
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        return str(result["choices"][0]["message"]["content"]).strip()[: self.MAX_SUMMARY_CHARS]

    def _call_daily_llm(self, messages: list[dict[str, Any]], memory_date: str, limit: int) -> str:
        transcript = self._transcript(messages)
        payload = {
            "model": self.model,
            "enable_thinking": False,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你负责为机器狗 Memory OS 生成某一天的时间记忆摘要。"
                        "输入是当天所有 session 的完整原始对话。"
                        "输出必须是压缩后的中文摘要，不要逐句复述用户和助手说了什么。"
                        "保留当天用户做过/要求过的重要事情、稳定上下文、未完成事项和有用线索。"
                        "删除寒暄、重复表达和模型套话，不要编造。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"日期：{memory_date}\n当天对话：\n{transcript}\n\n请输出不超过{min(700, limit)}字的中文日期摘要。",
                },
            ],
        }
        request = self._request(payload)
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        return str(result["choices"][0]["message"]["content"]).strip()[:limit]

    def _call_action_memory_llm(self, messages: list[dict[str, Any]], memory_date: str) -> list[dict[str, Any]]:
        compact_messages = [
            {
                "event_id": int(item.get("id") or 0),
                "role": item.get("role"),
                "text": item.get("content"),
                "session_id": item.get("session_id"),
                "created_at": item.get("created_at"),
            }
            for item in messages
            if item.get("role") in {"user", "assistant"} and str(item.get("content") or "").strip()
        ]
        payload = {
            "model": self.model,
            "enable_thinking": False,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是机器狗 Memory OS 的动作事件库抽取器。只输出严格 JSON，不要 Markdown。"
                        "输入是某一天所有 session 的原始对话。"
                        "只抽取机器狗动态行为事件，例如站起、坐下、转圈、跳舞、前进、后退、巡检、跟随等。"
                        "不要抽取提醒、出行计划、普通聊天、心情安抚、音乐播放偏好或长期用户偏好。"
                        "输出日级 action_memory 文本，供后续模型做动作模式抽取。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "memory_date": memory_date,
                            "messages": compact_messages,
                            "output_schema": {
                                "memories": [
                                    {
                                        "content": "事件记忆（YYYY-MM-DD）：\\n事件链路：\\n1. 用户要求 坐下 -> 机器狗完成 坐下",
                                        "title": "YYYY-MM-DD 事件记忆",
                                        "source_message_event_ids": [1, 2],
                                        "confidence": 0.8,
                                    }
                                ]
                            },
                            "rules": [
                                "没有机器狗动态行为时返回 {\"memories\": []}。",
                                "content 必须是可直接保存的中文 text。",
                                "source_message_event_ids 只能引用输入 messages 中的 event_id。",
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        request = self._request(payload)
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
        raw = json.loads(_json_text(str(result["choices"][0]["message"]["content"])))
        memories = raw.get("memories") if isinstance(raw, dict) else []
        if not isinstance(memories, list):
            return []
        valid_ids = {int(item["event_id"]) for item in compact_messages if int(item.get("event_id") or 0)}
        parsed = []
        for item in memories:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            source_ids = []
            for event_id in item.get("source_message_event_ids") or []:
                try:
                    value = int(event_id)
                except (TypeError, ValueError):
                    continue
                if value in valid_ids:
                    source_ids.append(value)
            parsed.append(
                {
                    "content": content[: self.MAX_SUMMARY_CHARS],
                    "title": str(item.get("title") or f"{memory_date} 事件记忆"),
                    "source_message_event_ids": source_ids,
                    "confidence": max(0.0, min(1.0, float(item.get("confidence") or 0.8))),
                }
            )
        return parsed

    def _request(self, payload: dict[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )

    @staticmethod
    def _local_summary(messages: list[dict[str, Any]], previous_summary: str = "") -> str:
        lines = []
        for message in messages:
            content = " ".join(str(message.get("content", "")).split())
            if not content:
                continue
            label = "用户" if message.get("role") == "user" else "机器狗"
            lines.append(f"{label}: {content[:240]}")
        new_part = "\n".join(lines[-40:])
        if previous_summary:
            return (previous_summary + "\n" + new_part)[-Summarizer.MAX_SUMMARY_CHARS :]
        return ("对话摘要（本地摘要生成）\n" + new_part)[-Summarizer.MAX_SUMMARY_CHARS :]

    @classmethod
    def _local_daily_summary(cls, messages: list[dict[str, Any]], memory_date: str, limit: int) -> str:
        user_texts = [
            " ".join(str(item.get("content") or "").split())
            for item in messages
            if item.get("role") == "user" and str(item.get("content") or "").strip()
        ]
        if not user_texts:
            return ""
        seen: set[str] = set()
        points = []
        for text in user_texts:
            normalized = text[:80]
            if normalized in seen:
                continue
            seen.add(normalized)
            points.append(normalized)
            if len(points) >= 8:
                break
        summary = f"日期总结（本地摘要，{memory_date}）：当天共有 {len(user_texts)} 条用户输入。主要事项：" + "；".join(points)
        return summary[:limit]

    @classmethod
    def _local_action_memories(cls, messages: list[dict[str, Any]], memory_date: str) -> list[dict[str, Any]]:
        action_words = ("站起来", "站起", "坐下", "转圈", "跳舞", "前进", "往前", "后退", "往后", "巡检", "跟随", "趴下")
        lines = []
        source_ids = []
        for item in messages:
            if item.get("role") != "user":
                continue
            text = " ".join(str(item.get("content") or "").split())
            if not text or not any(word in text for word in action_words):
                continue
            action_text = cls._compact_action_text(text)
            lines.append(f"{len(lines) + 1}. 用户要求 {action_text} -> 机器狗完成 {action_text}")
            if item.get("id"):
                source_ids.append(int(item["id"]))
        if not lines:
            return []
        return [
            {
                "content": f"事件记忆（{memory_date}）：\n事件链路：\n" + "\n".join(lines),
                "title": f"{memory_date} 事件记忆",
                "source_message_event_ids": source_ids,
                "confidence": 0.55,
            }
        ]

    @staticmethod
    def _compact_action_text(text: str) -> str:
        text = text.strip("。！？!?. ")
        return text[:80]

    @staticmethod
    def _transcript(messages: list[dict[str, Any]]) -> str:
        lines = []
        for item in messages:
            role = item.get("role")
            if role not in {"user", "assistant"}:
                continue
            label = "用户" if role == "user" else "机器狗"
            content = " ".join(str(item.get("content") or "").split())
            if content:
                lines.append(f"[{item.get('id', '-')}] {label}: {content}")
        return "\n".join(lines)


def _json_text(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start >= 0 and object_end >= object_start:
        return text[object_start : object_end + 1]
    return text
