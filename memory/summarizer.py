"""Conversation summarization with an OpenAI-compatible API and local fallback."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .config import QWEN_BASE_URL, QWEN_CHAT_MODEL


class Summarizer:
    MAX_SUMMARY_CHARS = 1600

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

    def _call_llm(self, messages: list[dict[str, Any]], previous_summary: str) -> str:
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        previous = previous_summary or "（暂无，这是第一次压缩）"
        payload = {
            "model": self.model,
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
