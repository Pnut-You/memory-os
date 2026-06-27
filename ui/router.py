"""UI orchestration for query and debug inspection."""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from typing import Any

from memory import MemoryManager
from memory.rules import is_repeat_action_request

from .llm import DebugChatLLM


logger = logging.getLogger("memory_ui")


class MemoryDebugRouter:
    def __init__(self, manager: MemoryManager, llm: DebugChatLLM) -> None:
        self.manager = manager
        self.llm = llm
        self._latencies: deque[float] = deque(maxlen=200)

    def submit(self, user_id: str, device_id: str, query: str, debug: bool = False) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        context_started = time.perf_counter()
        context = self.manager.get_conversation_context(user_id, device_id)
        context_ms = (time.perf_counter() - context_started) * 1000

        llm_started = time.perf_counter()
        try:
            assistant_reply, model_info = self.llm.complete(
                query,
                context["recent_messages"][-10:],
                context["rolling_summary"],
                context["user_card"],
                context.get("latest_action_sequence") if is_repeat_action_request(query) else None,
            )
        except Exception as exc:
            logger.exception("chat.llm_failed request_id=%s user_id=%s device_id=%s", request_id, user_id, device_id)
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        llm_ms = (time.perf_counter() - llm_started) * 1000

        persist_started = time.perf_counter()
        try:
            persist_result = self.manager.add_conversation_turn(
                request_id,
                user_id,
                device_id,
                query,
                assistant_reply,
                model_event_routes=model_info.get("event_routes") if isinstance(model_info.get("event_routes"), list) else None,
            )
        except Exception as exc:
            logger.exception("chat.persistence_failed request_id=%s user_id=%s device_id=%s", request_id, user_id, device_id)
            raise RuntimeError(f"Conversation persistence failed: {exc}") from exc
        persist_ms = (time.perf_counter() - persist_started) * 1000

        total_ms = (time.perf_counter() - started) * 1000
        self._latencies.append(total_ms)
        result = {
            "request_id": request_id,
            "user_id": user_id,
            "device_id": device_id,
            "assistant_reply": assistant_reply,
            "model": str(model_info.get("model", self.llm.model)),
        }
        if debug:
            latest_action_sequence = context.get("latest_action_sequence") if is_repeat_action_request(query) else None
            if hasattr(self.llm, "build_messages"):
                prompt_messages = self.llm.build_messages(
                    query,
                    context["recent_messages"][-10:],
                    context["rolling_summary"],
                    context["user_card"],
                    latest_action_sequence,
                )
            else:
                prompt_messages = [
                    {"role": "system", "content": "debug llm does not expose build_messages"},
                    *[
                        {"role": str(item.get("role")), "content": str(item.get("content"))}
                        for item in context["recent_messages"][-10:]
                    ],
                    {"role": "user", "content": query},
                ]
            result["debug"] = {
                "context_ms": round(context_ms, 1),
                "llm_ms": round(llm_ms, 1),
                "persist_ms": round(persist_ms, 1),
                "total_ms": round(total_ms, 1),
                "user_card_version": (context.get("user_card") or {}).get("version"),
                "summary_version": context.get("summary_version", 0),
                "prompt_preview": self._prompt_preview(prompt_messages),
                "prompt_messages": prompt_messages,
                "trace_steps": self._trace_steps(
                    request_id,
                    user_id,
                    device_id,
                    query,
                    context,
                    prompt_messages,
                    assistant_reply,
                    persist_result,
                    {
                        "context_ms": round(context_ms, 1),
                        "llm_ms": round(llm_ms, 1),
                        "persist_ms": round(persist_ms, 1),
                        "total_ms": round(total_ms, 1),
                    },
                ),
            }
        return result

    @staticmethod
    def _prompt_preview(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return [
            {
                "role": item.get("role", ""),
                "content": str(item.get("content", ""))[:1200],
            }
            for item in messages
        ]

    def _trace_steps(
        self,
        request_id: str,
        user_id: str,
        device_id: str,
        query: str,
        context: dict[str, Any],
        prompt_messages: list[dict[str, str]],
        assistant_reply: str,
        persist_result: dict[str, Any],
        timings: dict[str, float],
    ) -> list[dict[str, Any]]:
        time_memory = persist_result.get("time_memory")
        action_event_id = persist_result.get("action_event_id")
        return [
            {
                "name": "request_input",
                "title_zh": "请求输入",
                "status": "ok",
                "data": {"request_id": request_id, "user_id": user_id, "device_id": device_id, "query": query},
            },
            {
                "name": "redis_context_read",
                "title_zh": "读取上下文缓存",
                "status": "ok",
                "duration_ms": timings["context_ms"],
                "data": {
                    "recent_message_count": len(context.get("recent_messages") or []),
                    "summary_version": context.get("summary_version", 0),
                    "user_card_version": (context.get("user_card") or {}).get("version"),
                },
            },
            {
                "name": "rolling_summary",
                "title_zh": "滚动摘要",
                "status": "ok" if context.get("rolling_summary") else "empty",
                "data": {
                    "version": context.get("summary_version", 0),
                    "summary_text": context.get("rolling_summary") or "",
                    "summary_pending": context.get("summary_pending", False),
                },
            },
            {
                "name": "long_term_memory",
                "title_zh": "长期记忆",
                "status": "ok" if context.get("user_card") else "empty",
                "data": context.get("user_card") or {},
            },
            {
                "name": "recent_messages",
                "title_zh": "最近对话",
                "status": "ok" if context.get("recent_messages") else "empty",
                "data": context.get("recent_messages") or [],
            },
            {
                "name": "time_memory_routing",
                "title_zh": "时间记忆路由",
                "status": "created" if time_memory else "skipped",
                "data": time_memory or {"reason": "no time memory pattern matched"},
            },
            {
                "name": "action_event_routing",
                "title_zh": "动作事件路由",
                "status": "created" if action_event_id else "skipped",
                "data": {"action_event_id": action_event_id} if action_event_id else {"reason": "no action sequence matched"},
            },
            {
                "name": "sqlite_persist",
                "title_zh": "SQLite 事实写入",
                "status": "ok",
                "duration_ms": timings["persist_ms"],
                "data": persist_result,
            },
            {
                "name": "llm_prompt_messages",
                "title_zh": "回复模型输入",
                "status": "ok",
                "data": self._prompt_preview(prompt_messages),
            },
            {
                "name": "llm_response",
                "title_zh": "回复模型输出",
                "status": "ok",
                "duration_ms": timings["llm_ms"],
                "data": {"assistant_reply": assistant_reply},
            },
            {
                "name": "background_jobs",
                "title_zh": "后台任务状态",
                "status": "ok",
                "data": self.manager.events.job_counts(),
            },
        ]

    def debug_user(self, user_id: str) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "user_card": self.manager.get_user_card(user_id) or self.manager.restore_user_card(user_id),
            "active_preferences": self.manager.events.list_preferences(user_id, status="active", limit=100),
            "candidate_preferences": self.manager.events.list_preferences(user_id, status="candidate", limit=100),
            "summaries": self.manager.events.list_summaries(user_id, limit=20),
            "time_memories": self.manager.events.list_time_memories(user_id, limit=100),
            "recent_messages": self.manager.events.list_events(user_id=user_id, event_type="message", limit=20),
            "evidence": self.manager.events.list_preference_evidence(user_id, limit=100),
        }

    def preferences(self, user_id: str) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "active": self.manager.events.list_preferences(user_id, status="active", limit=200),
            "candidate": self.manager.events.list_preferences(user_id, status="candidate", limit=200),
            "history": self.manager.events.list_preferences(user_id, status=None, limit=500),
            "evidence": self.manager.events.list_preference_evidence(user_id, limit=200),
        }

    def events(self, user_id: str | None, device_id: str | None, role: str | None) -> dict[str, Any]:
        return {
            "events": self.manager.events.list_events(
                user_id=user_id,
                device_id=device_id,
                role=role,
                limit=100,
            )
        }

    def event_library(
        self,
        user_id: str | None = None,
        device_id: str | None = None,
        event_type: str | None = None,
    ) -> dict[str, Any]:
        if event_type == "action_sequence":
            events = self.manager.events.list_action_events(user_id, device_id, limit=100)
        elif event_type == "time_memory":
            events = self.manager.events.list_time_memories(user_id, device_id, limit=100)
        else:
            events = self.manager.events.list_events(
                user_id=user_id,
                device_id=device_id,
                event_type=event_type or None,
                limit=100,
            )
        return {"events": events}

    def action_events(self, user_id: str, device_id: str | None = None) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "device_id": device_id,
            "actions": self.manager.events.list_action_events(user_id, device_id, limit=100),
        }

    def debug_device(self, device_id: str) -> dict[str, Any]:
        return {
            "device_id": device_id,
            "state": self.manager.get_device_state(device_id),
            "history": self.manager.get_device_history(device_id, limit=100),
            "recent_events": self.manager.events.list_events(device_id=device_id, limit=50),
        }

    def update_debug_device_state(
        self,
        device_id: str,
        state: dict[str, Any],
        observed_at: str | None = None,
    ) -> dict[str, Any]:
        result = self.manager.update_device_state(device_id, state, observed_at)
        return {"updated": result, **self.debug_device(device_id)}

    def create_time_memory(
        self,
        user_id: str,
        device_id: str,
        content: str,
        target_at: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event_id = self.manager.remember_at(
            user_id,
            device_id,
            content,
            target_at,
            {"target_at": target_at, "source": "debug", **(metadata or {})},
        )
        return {"event_id": event_id, "time_memories": self.manager.events.list_time_memories(user_id, device_id)}

    def time_memories(self, user_id: str, device_id: str | None = None) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "device_id": device_id,
            "time_memories": self.manager.events.list_time_memories(user_id, device_id),
        }

    def delete_user_memory(self, user_id: str) -> dict[str, Any]:
        return {"user_id": user_id, "deleted": self.manager.delete_user_memory(user_id)}

    def process_memory_jobs(self) -> dict[str, Any]:
        process = self.manager.process_memory_jobs_once()
        return {"process": process, "status": self.status()}

    def extract_user_preferences(
        self,
        user_id: str,
        device_id: str | None = None,
        *,
        force: bool = True,
        recent_user_messages: int = 20,
    ) -> dict[str, Any]:
        try:
            result = self.manager.trigger_preference_extraction(
                user_id,
                device_id,
                force_recent=force,
                recent_user_messages=recent_user_messages,
            )
            return {**result, "ok": not bool((result.get("process") or {}).get("errors")), "memory": self.debug_user(user_id), "status": self.status()}
        except Exception as exc:
            logger.exception("preference_extract.debug_failed user_id=%s device_id=%s", user_id, device_id)
            return {
                "ok": False,
                "user_id": user_id,
                "device_id": device_id,
                "created_job": False,
                "process": {
                    "claimed": 0,
                    "processed": 0,
                    "succeeded": 0,
                    "failed": 1,
                    "skipped": 0,
                    "recovered_stale": 0,
                    "errors": [{"error": str(exc), "final": False}],
                },
                "memory": self.debug_user(user_id),
                "status": self.status(),
            }

    def status(self) -> dict[str, Any]:
        manager_status = self.manager.status()
        latencies = list(self._latencies)
        avg = sum(latencies) / len(latencies) if latencies else 0.0
        p95 = sorted(latencies)[int(len(latencies) * 0.95) - 1] if len(latencies) >= 2 else avg
        return {
            "ready": True,
            "llm": self.llm.status(),
            **manager_status,
            "recent_jobs": self.manager.events.list_jobs(limit=10),
            "latency": {
                "avg_ms": round(avg, 1),
                "p95_ms": round(p95, 1),
                "samples": len(latencies),
            },
        }
