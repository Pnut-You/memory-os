"""Public entry point for the lightweight Memory OS runtime."""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .config import MemoryConfig
from .preferences import (
    PREFERENCE_REGISTRY,
    PreferenceExtractor,
    normalize_preference_key,
    should_schedule_preference,
)
from .redis_memory import ShortTermMemory
from .rules import is_confirmation, parse_action_sequence, parse_event_route, parse_time_memory
from .sqlite_event import LEGACY_USER_ID, SQLiteEventStore
from .summarizer import Summarizer
from .time_memory import TimeMemory
from .utils import iso_now, tokenize


logger = logging.getLogger(__name__)


@dataclass
class _ConversationState:
    summary_pending: bool = False


class MemoryManager:
    def __init__(
        self,
        short_term: ShortTermMemory,
        events: SQLiteEventStore,
        summarizer: Summarizer,
        time_memory: TimeMemory,
        preference_extractor: PreferenceExtractor,
        summary_every_turns: int = 10,
        summary_retain_turns: int = 5,
        device_state_ttl_seconds: int = 120,
        device_heartbeat_seconds: int = 300,
        tool_run_ttl_seconds: int = 3600,
        preference_extract_min_new_user_messages: int = 10,
        preference_extract_batch_size: int = 8,
        preference_extract_max_attempts: int = 3,
    ) -> None:
        self.redis = short_term
        self.events = events
        self.summarizer = summarizer
        self.time_memory = time_memory
        self.preference_extractor = preference_extractor
        self.summary_every_turns = max(1, summary_every_turns)
        if summary_retain_turns <= 0 or summary_retain_turns >= self.summary_every_turns:
            raise ValueError("summary_retain_turns must be > 0 and < summary_every_turns")
        self.summary_retain_turns = summary_retain_turns
        self.device_state_ttl_seconds = device_state_ttl_seconds
        self.device_heartbeat_seconds = device_heartbeat_seconds
        self.tool_run_ttl_seconds = tool_run_ttl_seconds
        self.preference_extract_min_new_user_messages = max(1, preference_extract_min_new_user_messages)
        self.preference_extract_batch_size = max(1, preference_extract_batch_size)
        self.preference_extract_max_attempts = max(1, preference_extract_max_attempts)
        self._conversation_states: dict[tuple[str, str], _ConversationState] = {}
        self._summary_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="memory-summary")
        self._summary_futures: set[Future[Any]] = set()
        self._job_stop = threading.Event()
        self._job_thread: threading.Thread | None = None
        self._lock = threading.RLock()

    @classmethod
    def create(
        cls,
        config: MemoryConfig | None = None,
        *,
        start_scheduler: bool = False,
        redis_client: Any | None = None,
    ) -> "MemoryManager":
        config = config or MemoryConfig.from_env()
        config.ensure_directories()
        events = SQLiteEventStore(config.sqlite_path)
        short = ShortTermMemory(
            config.redis_url,
            config.redis_ttl_seconds,
            config.redis_prefix,
            redis_client,
            config.redis_allow_memory_fallback,
        )
        summarizer = Summarizer(config.llm_api_key, config.llm_base_url, config.llm_model)
        timed = TimeMemory(events)
        extractor = PreferenceExtractor(
            enabled=config.preference_extractor_enabled,
            api_key=config.preference_extractor_api_key,
            base_url=config.preference_extractor_base_url,
            model=config.preference_extractor_model,
        )
        manager = cls(
            short,
            events,
            summarizer,
            timed,
            extractor,
            config.summary_every_turns,
            config.summary_retain_turns,
            config.device_state_ttl_seconds,
            config.device_heartbeat_seconds,
            config.tool_run_ttl_seconds,
            config.preference_extract_min_new_user_messages,
            config.preference_extract_batch_size,
            config.preference_extract_max_attempts,
        )
        if start_scheduler:
            manager.start_memory_worker()
        return manager

    def start_memory_worker(self, poll_seconds: float = 2.0) -> None:
        if self._job_thread and self._job_thread.is_alive():
            return
        self._job_stop.clear()

        def _loop() -> None:
            while not self._job_stop.wait(poll_seconds):
                try:
                    self.process_memory_jobs_once()
                except Exception:
                    logger.exception("memory_job.worker_failed")

        self._job_thread = threading.Thread(target=_loop, name="memory-jobs", daemon=True)
        self._job_thread.start()

    def get_conversation_context(self, user_id: str, device_id: str) -> dict[str, Any]:
        bundle = self.redis.get_context_bundle(device_id, user_id, recent_limit=10)
        summary = bundle["summary"] or self._restore_summary(user_id, device_id)
        user_card = bundle["user_card"] or self.restore_user_card(user_id)
        recent = bundle["recent_messages"]
        if not recent:
            latest_summary_id = int((summary or {}).get("compacted_through_event_id", 0) or 0)
            recent = self.events.message_range(user_id, device_id, latest_summary_id, 20)[-10:]
            if recent:
                self.redis.append_conversation(device_id, user_id, recent, max_items=20)
        action_buffer = self.redis.get_value("action-buffer", f"{device_id}:{user_id}") or {}
        latest_action = action_buffer if action_buffer.get("actions") else self.events.latest_action_sequence(user_id, device_id)
        return {
            "user_id": user_id,
            "device_id": device_id,
            "user_card": user_card,
            "rolling_summary": (summary or {}).get("summary_text", ""),
            "summary_version": int((summary or {}).get("version", 0) or 0),
            "summary_pending": self._conversation_states.get((user_id, device_id), _ConversationState()).summary_pending,
            "recent_messages": recent[-10:],
            "latest_action_sequence": latest_action,
            "action_buffer": action_buffer,
        }

    def add_conversation_turn(
        self,
        request_id: str,
        user_id: str,
        device_id: str,
        user_text: str,
        assistant_text: str,
        timestamp: str | None = None,
        model_event_routes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        timestamp = timestamp or iso_now()
        user_event_id, assistant_event_id = self.events.add_message_pair(
            request_id, user_id, device_id, user_text, assistant_text, timestamp
        )
        messages = [
            {"id": user_event_id, "role": "user", "content": user_text, "timestamp": timestamp},
            {"id": assistant_event_id, "role": "assistant", "content": assistant_text, "timestamp": timestamp},
        ]
        self.redis.append_conversation(device_id, user_id, messages, max_items=self.summary_every_turns * 2)
        routed = self._route_local_events(request_id, user_id, device_id, user_text, user_event_id)
        if model_event_routes and not any(key in routed for key in ("time_memory_event_id", "event_route_id")):
            model_routed = self._route_model_event_candidates(
                request_id, user_id, device_id, user_text, user_event_id, model_event_routes
            )
            routed.update(model_routed)
        self._schedule_summary(user_id, device_id)
        if user_id != "anonymous":
            self._maybe_schedule_preference(
                user_id,
                device_id,
                user_text,
                int(routed.get("action_event_id") or user_event_id),
            )
        return {"user_event_id": user_event_id, "assistant_event_id": assistant_event_id, **routed}

    def _route_model_event_candidates(
        self,
        request_id: str,
        user_id: str,
        device_id: str,
        user_text: str,
        source_event_id: int,
        routes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        accepted = []
        for route in routes:
            if not isinstance(route, dict):
                continue
            confidence = float(route.get("confidence") or 0)
            event_type = str(route.get("type") or "")
            decision = str(route.get("decision") or "")
            if confidence < 0.7 or event_type not in {"scheduled_task", "recurring_task", "conditional_task", "pending_event", "action_sequence"}:
                continue
            if decision not in {"create", "clarify", "cancel", "update"}:
                continue
            accepted.append({**route, "source": "reply_model_candidate", "source_event_id": source_event_id})
        if not accepted:
            return {}
        event_id = self.events.add_event(
            f"{request_id}-model-event-route",
            user_id,
            device_id,
            "pending_event",
            {"routes": accepted, "content": user_text},
            content=user_text,
        )
        self.redis.set_value(
            "pending-event",
            f"{device_id}:{user_id}",
            {"event_id": event_id, "routes": accepted, "content": user_text},
            900,
        )
        return {"model_event_route_id": event_id, "model_event_routes": accepted}

    def wait_for_summaries(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                pending = [future for future in self._summary_futures if not future.done()]
            if not pending:
                return True
            time.sleep(0.01)
        return False

    def remember_at(
        self,
        user_id: str,
        device_id: str,
        content: str,
        timestamp: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        return self.time_memory.remember(user_id, device_id, content, timestamp, metadata)

    def search(self, user_id: str, query: str, limit: int = 5) -> dict[str, Any]:
        card = self.get_user_card(user_id)
        events = self.events.search_user_text(user_id, query, limit)
        active = [
            pref
            for pref in self.events.list_preferences(user_id, status="active", limit=100)
            if tokenize(query) & tokenize(str(pref))
        ][:limit]
        return {"user_card": card, "events": events, "preferences": active}

    def get_user_card(self, user_id: str) -> dict[str, Any] | None:
        return self.redis.get_json(self.redis.user_card_key(user_id))

    def restore_user_card(self, user_id: str) -> dict[str, Any] | None:
        preferences = self.events.list_preferences(user_id, status="active", limit=15)
        if not preferences:
            return None
        card = self._build_user_card(user_id, preferences)
        self.redis.set_json(self.redis.user_card_key(user_id), card, self.redis.ttl_seconds)
        return card

    def rebuild_user_card(self, user_id: str) -> dict[str, Any] | None:
        card = self.restore_user_card(user_id)
        if card is None:
            self.redis.delete_key(self.redis.user_card_key(user_id))
        return card

    def process_memory_jobs_once(self, limit: int | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "claimed": 0,
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "skipped": 0,
            "recovered_stale": self.events.recover_stale_running_jobs(),
            "errors": [],
        }
        jobs = self.events.claim_jobs(
            limit or self.preference_extract_batch_size,
            max_attempts=self.preference_extract_max_attempts,
        )
        result["claimed"] = len(jobs)
        for job in jobs:
            try:
                job_result: dict[str, Any] | None = None
                if job["job_type"] == "preference_extraction":
                    job_result = self._process_preference_job(job)
                elif job["job_type"] == "user_card_rebuild":
                    self.rebuild_user_card(str(job["user_id"]))
                elif job["job_type"] == "conversation_summary":
                    self._run_summary(str(job["user_id"]), str(job["device_id"]))
                self.events.mark_job_succeeded(int(job["id"]))
                result["processed"] += 1
                result["succeeded"] += 1
                if job_result and job_result.get("skipped"):
                    result["skipped"] += 1
                if job_result:
                    result.setdefault("jobs", []).append({"id": job["id"], **job_result})
            except Exception as exc:
                logger.exception("memory_job.failed job_id=%s", job["id"])
                attempts = self.events.mark_job_failed(
                    int(job["id"]), str(exc), self.preference_extract_max_attempts
                )
                result["failed"] += 1
                result["errors"].append(
                    {
                        "job_id": int(job["id"]),
                        "job_type": str(job["job_type"]),
                        "attempts": attempts,
                        "final": attempts >= self.preference_extract_max_attempts,
                        "error": str(exc),
                    }
                )
        return result

    def trigger_preference_extraction(
        self,
        user_id: str,
        device_id: str | None = None,
        *,
        force_recent: bool = True,
        recent_user_messages: int = 20,
    ) -> dict[str, Any]:
        if user_id in {"anonymous", LEGACY_USER_ID}:
            return {
                "user_id": user_id,
                "device_id": device_id,
                "mode": "force_recent" if force_recent else "new_only",
                "created_job": False,
                "message": "anonymous 和 legacy-unassigned 不生成长期偏好",
                "process": self.process_memory_jobs_once(limit=1),
            }
        latest_event_id = self.events.latest_user_event_id(user_id, device_id if force_recent else None) or 0
        latest_global_event_id = self.events.latest_user_event_id(user_id) or 0
        latest_processed_id = self.events.latest_preference_extraction_event_id(user_id)
        mode = "force_recent" if force_recent else "new_only"
        input_user_events = 0
        input_action_events = 0
        if force_recent:
            recent_user_events = self.events.list_events(
                user_id=user_id,
                device_id=device_id,
                role="user",
                limit=max(1, recent_user_messages),
                ascending=False,
            )
            recent_action_events = self.events.list_action_events(user_id=user_id, device_id=device_id, limit=10)
            selected_ids = [int(event["id"]) for event in recent_user_events + recent_action_events]
            input_user_events = len(recent_user_events)
            input_action_events = len(recent_action_events)
            if not selected_ids:
                return {
                    "user_id": user_id,
                    "device_id": device_id,
                    "mode": mode,
                    "force": True,
                    "created_job": False,
                    "message": "当前 user_id/device_id 下没有原始用户消息",
                    "latest_event_id": latest_event_id,
                    "latest_global_event_id": latest_global_event_id,
                    "latest_processed_event_id": latest_processed_id,
                    "input_user_events": input_user_events,
                    "input_action_events": input_action_events,
                    "process": self.process_memory_jobs_once(limit=1),
                }
            from_event_id = max(0, min(selected_ids) - 1)
            to_event_id = max(selected_ids)
        else:
            from_event_id = latest_processed_id
            to_event_id = latest_global_event_id
        if not force_recent and to_event_id <= latest_processed_id:
            return {
                "user_id": user_id,
                "device_id": device_id,
                "mode": mode,
                "force": False,
                "created_job": False,
                "message": "没有可抽取的新用户消息",
                "latest_event_id": latest_event_id,
                "latest_global_event_id": latest_global_event_id,
                "latest_processed_event_id": latest_processed_id,
                "from_event_id": from_event_id,
                "to_event_id": to_event_id,
                "process": {
                    "claimed": 0,
                    "processed": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "skipped": 0,
                    "recovered_stale": 0,
                    "errors": [],
                },
            }
        try:
            job_id = self.events.restart_preference_extraction_job(
                user_id,
                device_id,
                from_event_id,
                to_event_id,
            )
        except Exception as exc:
            return {
                "user_id": user_id,
                "device_id": device_id,
                "mode": mode,
                "force": force_recent,
                "created_job": False,
                "latest_event_id": latest_event_id,
                "latest_global_event_id": latest_global_event_id,
                "latest_processed_event_id": latest_processed_id,
                "from_event_id": from_event_id,
                "to_event_id": to_event_id,
                "input_user_events": input_user_events,
                "input_action_events": input_action_events,
                "process": {
                    "claimed": 0,
                    "processed": 0,
                    "succeeded": 0,
                    "failed": 1,
                    "skipped": 0,
                    "recovered_stale": 0,
                    "errors": [{"error": f"failed to create preference extraction job: {exc}", "final": False}],
                },
            }
        process = self.process_memory_jobs_once(limit=2)
        if process.get("succeeded"):
            follow_up = self.process_memory_jobs_once(limit=2)
            if follow_up.get("claimed") or follow_up.get("recovered_stale"):
                process["follow_up"] = follow_up
        return {
            "user_id": user_id,
            "device_id": device_id,
            "mode": mode,
            "force": force_recent,
            "created_job": job_id is not None,
            "job_id": job_id,
            "latest_event_id": latest_event_id,
            "latest_global_event_id": latest_global_event_id,
            "latest_processed_event_id": latest_processed_id,
            "from_event_id": from_event_id,
            "to_event_id": to_event_id,
            "input_user_events": input_user_events,
            "input_action_events": input_action_events,
            "process": process,
        }

    def begin_tool_run(
        self,
        context_id: str,
        tool_name: str,
        input_data: Any,
        *,
        run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        run_id = run_id or uuid.uuid4().hex
        started_at = iso_now()
        actual_id, created = self.events.begin_tool_run(
            context_id, tool_name, input_data, run_id, idempotency_key, started_at
        )
        if created:
            self.redis.set_value(
                "tool-run",
                actual_id,
                {
                    "run_id": actual_id,
                    "context_id": context_id,
                    "tool_name": tool_name,
                    "status": "running",
                    "started_at": started_at,
                },
                self.tool_run_ttl_seconds,
            )
        return actual_id

    def record_tool_step(
        self, run_id: str, step_name: str, payload: Any, status: str = "completed"
    ) -> int:
        step_id = self.events.add_tool_step(run_id, step_name, status, payload)
        cached = self.redis.get_value("tool-run", run_id) or {"run_id": run_id}
        cached.update({"status": "running", "last_step": step_name, "updated_at": iso_now()})
        self.redis.set_value("tool-run", run_id, cached, self.tool_run_ttl_seconds)
        return step_id

    def finish_tool_run(
        self,
        run_id: str,
        output: Any = None,
        *,
        status: str = "completed",
        error: str | None = None,
    ) -> None:
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError("tool run status must be completed, failed, or cancelled")
        self.events.finish_tool_run(run_id, status, output, error)
        self.redis.delete_value("tool-run", run_id)

    def get_tool_run(self, run_id: str) -> dict[str, Any] | None:
        return self.events.get_tool_run(run_id)

    def update_device_state(
        self,
        device_id: str,
        state: dict[str, Any],
        observed_at: str | None = None,
    ) -> dict[str, Any]:
        observed_at = observed_at or iso_now()
        previous = self.redis.get_json(self.redis.device_state_key(device_id))
        previous_history = self.events.latest_device_history(device_id) if previous is None else None
        prior_state = previous.get("state") if previous else (
            previous_history.get("state") if previous_history else None
        )
        prior_time = previous.get("last_history_at") if previous else (
            previous_history.get("observed_at") if previous_history else None
        )
        reason: str | None = None
        if prior_state is None:
            reason = "initial"
        elif prior_state != state:
            reason = "change"
        elif prior_time and self._seconds_between(prior_time, observed_at) >= self.device_heartbeat_seconds:
            reason = "heartbeat"

        snapshot = {
            "device_id": device_id,
            "state": dict(state),
            "observed_at": observed_at,
            "last_history_at": observed_at if reason else prior_time,
        }
        self.redis.set_json(self.redis.device_state_key(device_id), snapshot, self.device_state_ttl_seconds)
        history_id = (
            self.events.add_device_state(device_id, state, reason, observed_at) if reason else None
        )
        return {
            "device_id": device_id,
            "state": dict(state),
            "observed_at": observed_at,
            "history_written": history_id is not None,
            "reason": reason,
        }

    def get_device_state(self, device_id: str) -> dict[str, Any]:
        snapshot = self.redis.get_json(self.redis.device_state_key(device_id))
        if snapshot is None:
            return {"device_id": device_id, "online": False, "state": None}
        return {
            "device_id": device_id,
            "state": snapshot["state"],
            "observed_at": snapshot["observed_at"],
            "online": True,
        }

    def get_device_history(self, device_id: str, limit: int = 100) -> list[dict[str, Any]]:
        return self.events.get_device_history(device_id, limit)

    def delete_user_memory(self, user_id: str) -> dict[str, int]:
        self.redis.delete_key(self.redis.user_card_key(user_id))
        self.redis.delete_key(self.redis.user_preferences_key(user_id))
        return self.events.delete_user_memory(user_id)

    def status(self) -> dict[str, Any]:
        return {
            "redis": {"backend": self.redis.backend},
            "sqlite": {"backend": "sqlite", "path": str(self.events.path)},
            "preference_extractor": self.preference_extractor.status(),
            "jobs": self.events.job_counts(),
            **self.events.stats(),
        }

    def _route_local_events(
        self,
        request_id: str,
        user_id: str,
        device_id: str,
        user_text: str,
        source_event_id: int,
    ) -> dict[str, Any]:
        routed: dict[str, Any] = {}
        completed = self._complete_pending_event(request_id, user_id, device_id, user_text, source_event_id)
        if completed:
            routed.update(completed)
            return routed
        event_route = parse_event_route(user_text)
        if event_route:
            routed["event_route"] = event_route
            if event_route["decision"] == "create" and event_route["type"] in {"scheduled_task", "recurring_task"}:
                event_id = self.time_memory.remember(
                    user_id,
                    device_id,
                    user_text,
                    event_route["target_at"],
                    {
                        "task": event_route["task"],
                        "source_event_id": source_event_id,
                        "parser": event_route["parser"],
                        "event_type": event_route["type"],
                        "recurrence": event_route.get("recurrence"),
                    },
                )
                routed["time_memory_event_id"] = event_id
                routed["time_memory"] = {
                    "event_id": event_id,
                    "target_at": event_route["target_at"],
                    "task": event_route["task"],
                    "parser": event_route["parser"],
                    "type": event_route["type"],
                    "recurrence": event_route.get("recurrence"),
                }
            elif event_route["type"] in {"conditional_task", "pending_event"} or event_route["decision"] in {"cancel", "update", "clarify"}:
                event_id = self.events.add_event(
                    f"{request_id}-{event_route['type']}",
                    user_id,
                    device_id,
                    event_route["type"],
                    {**event_route, "source_event_id": source_event_id},
                    content=user_text,
                )
                routed["event_route_id"] = event_id
                if event_route["type"] == "pending_event" or event_route["decision"] == "clarify":
                    self.redis.set_value(
                        "pending-event",
                        f"{device_id}:{user_id}",
                        {"event_id": event_id, **event_route, "content": user_text},
                        900,
                    )
        action = parse_action_sequence(user_text)
        if action and len(action.get("actions", [])) == 1:
            self._append_action_buffer(user_id, device_id, source_event_id, user_text, action)
            routed["action_buffered"] = True
            routed["action_buffer"] = self.redis.get_value("action-buffer", f"{device_id}:{user_id}")
        elif action:
            event_id = self.events.add_event(
                f"{request_id}-action",
                user_id,
                device_id,
                "action_sequence",
                {
                    **action,
                    "source_event_id": source_event_id,
                },
                content=user_text,
            )
            routed["action_event_id"] = event_id
            self.redis.set_value(
                "action-buffer",
                f"{device_id}:{user_id}",
                {
                    "event_type": "action_sequence",
                    "actions": action.get("actions", []),
                    "source_event_ids": [source_event_id],
                    "updated_at": iso_now(),
                },
                900,
            )
        elif is_confirmation(user_text):
            routed["action_buffer_preserved"] = bool(self.redis.get_value("action-buffer", f"{device_id}:{user_id}"))
        return routed

    def _complete_pending_event(
        self,
        request_id: str,
        user_id: str,
        device_id: str,
        user_text: str,
        source_event_id: int,
    ) -> dict[str, Any]:
        pending = self.redis.get_value("pending-event", f"{device_id}:{user_id}")
        if not pending:
            return {}
        target_at = self._parse_pending_time(user_text)
        if not target_at:
            return {}
        task = str(pending.get("task") or pending.get("content") or user_text)
        event_id = self.time_memory.remember(
            user_id,
            device_id,
            task,
            target_at,
            {
                "task": task,
                "source_event_id": source_event_id,
                "completed_pending_event_id": pending.get("event_id"),
                "parser": "pending-event-rule-v1",
                "event_type": "scheduled_task",
            },
        )
        self.redis.delete_value("pending-event", f"{device_id}:{user_id}")
        self.events.add_event(
            f"{request_id}-pending-event-completed",
            user_id,
            device_id,
            "pending_event",
            {
                "decision": "completed",
                "source_event_id": source_event_id,
                "completed_event_id": event_id,
                "target_at": target_at,
                "task": task,
            },
            content=user_text,
        )
        return {
            "time_memory_event_id": event_id,
            "time_memory": {
                "event_id": event_id,
                "target_at": target_at,
                "task": task,
                "parser": "pending-event-rule-v1",
                "type": "scheduled_task",
            },
            "pending_event_completed": True,
        }

    @staticmethod
    def _parse_pending_time(text: str) -> str | None:
        tz = ZoneInfo("Asia/Shanghai")
        now = datetime.now(tz)
        relative = re.search(r"(\d+|半)\s*(分钟|小时|个小时)后", text)
        if relative:
            amount = 0.5 if relative.group(1) == "半" else int(relative.group(1))
            delta = timedelta(hours=amount) if "小时" in relative.group(2) else timedelta(minutes=amount)
            return (now + delta).replace(microsecond=0).isoformat()
        match = re.search(r"([0-2]?\d)\s*[点:时](半)?", text)
        if not match:
            return None
        hour = int(match.group(1))
        minute = 30 if match.group(2) or "半" in text else 0
        if any(word in text for word in ("下午", "晚上", "傍晚")) and hour < 12:
            hour += 12
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target.isoformat()

    def _append_action_buffer(
        self,
        user_id: str,
        device_id: str,
        source_event_id: int,
        text: str,
        action: dict[str, Any],
    ) -> None:
        key = f"{device_id}:{user_id}"
        current = self.redis.get_value("action-buffer", key) or {"event_type": "action_sequence", "actions": [], "source_event_ids": [], "texts": []}
        actions = list(current.get("actions") or [])
        actions.extend(action.get("actions") or [])
        source_ids = list(current.get("source_event_ids") or [])
        source_ids.append(source_event_id)
        texts = list(current.get("texts") or [])
        texts.append(text)
        self.redis.set_value(
            "action-buffer",
            key,
            {
                "event_type": "action_sequence",
                "actions": actions[-10:],
                "source_event_ids": source_ids[-10:],
                "texts": texts[-10:],
                "updated_at": iso_now(),
            },
            900,
        )

    def close(self) -> None:
        self._job_stop.set()
        if self._job_thread:
            self._job_thread.join(timeout=2)
        self._summary_executor.shutdown(wait=True, cancel_futures=False)
        self.events.close()

    def _restore_summary(self, user_id: str, device_id: str) -> dict[str, Any] | None:
        summary = self.events.latest_summary(user_id, device_id)
        if summary:
            self.redis.set_json(self.redis.summary_key(device_id, user_id), summary, self.redis.ttl_seconds)
        return summary

    def _maybe_schedule_preference(
        self,
        user_id: str,
        device_id: str,
        user_text: str,
        user_event_id: int,
    ) -> None:
        if user_id in {"anonymous", LEGACY_USER_ID}:
            return
        latest_job_to = self.events.latest_preference_extraction_event_id(user_id)
        # Duplicate pending jobs are collapsed by a unique partial index.
        should = should_schedule_preference(user_text)
        action = parse_action_sequence(user_text)
        if action and any(word in user_text for word in ("以后", "默认", "每次", "总是", "习惯", "记住")):
            should = True
        if not should:
            should = self.events.count_user_messages_since(user_id, latest_job_to) >= self.preference_extract_min_new_user_messages
        if should:
            self.events.enqueue_job(
                user_id,
                "preference_extraction",
                device_id=device_id,
                from_event_id=latest_job_to,
                to_event_id=user_event_id,
            )

    def _schedule_summary(self, user_id: str, device_id: str) -> None:
        key = (user_id, device_id)
        with self._lock:
            state = self._conversation_states.setdefault(key, _ConversationState())
            if state.summary_pending:
                return
            latest = self.events.latest_summary(user_id, device_id)
            after_id = int(latest["compacted_through_event_id"]) if latest else 0
            turns = self.events.conversation_turns_after(
                user_id,
                device_id,
                after_id,
                self.summary_every_turns,
            )
            if len(turns) < self.summary_every_turns:
                return
            state.summary_pending = True
            future = self._summary_executor.submit(self._run_summary, user_id, device_id)
            self._summary_futures.add(future)
            future.add_done_callback(self._summary_futures.discard)

    def _run_summary(self, user_id: str, device_id: str) -> None:
        key = (user_id, device_id)
        try:
            latest = self.events.latest_summary(user_id, device_id)
            after_id = int(latest["compacted_through_event_id"]) if latest else 0
            version = int(latest["version"]) + 1 if latest else 1
            compact_turn_count = self.summary_every_turns - self.summary_retain_turns
            turns = self.events.conversation_turns_after(
                user_id,
                device_id,
                after_id,
                self.summary_every_turns,
            )
            if len(turns) < self.summary_every_turns:
                return
            compact_turns = turns[:compact_turn_count]
            compacted_through = int(compact_turns[-1]["assistant"]["id"])
            summary_turns = self.events.conversation_turns_until(
                user_id,
                device_id,
                compacted_through,
                20,
            )
            messages = []
            for turn in summary_turns:
                messages.extend([turn["user"], turn["assistant"]])
            previous_summary = ""
            summary = self.summarizer.summarize(messages, previous_summary)
            if not summary:
                raise RuntimeError("summarizer returned an empty result")
            from_event_id = int(compact_turns[0]["user"]["id"])
            self.events.add_summary(
                user_id,
                device_id,
                summary,
                compacted_through,
                version,
                from_event_id=from_event_id,
                to_event_id=compacted_through,
                turn_count=len(compact_turns),
            )
            summary_state = {
                "user_id": user_id,
                "device_id": device_id,
                "summary_text": summary,
                "compacted_through_event_id": compacted_through,
                "version": version,
                "from_event_id": from_event_id,
                "to_event_id": compacted_through,
                "turn_count": len(compact_turns),
                "created_at": iso_now(),
            }
            self.redis.set_json(self.redis.summary_key(device_id, user_id), summary_state, self.redis.ttl_seconds)
            if user_id not in {"anonymous", LEGACY_USER_ID}:
                self.events.enqueue_job(
                    user_id,
                    "preference_extraction",
                    device_id=device_id,
                    from_event_id=max(0, from_event_id - 1),
                    to_event_id=compacted_through,
                )
            self._route_summary_window_events(user_id, device_id, summary_turns)
        finally:
            with self._lock:
                state = self._conversation_states.setdefault(key, _ConversationState())
                state.summary_pending = False

    def _process_preference_job(self, job: dict[str, Any]) -> dict[str, Any]:
        user_id = str(job["user_id"])
        if user_id in {"anonymous", LEGACY_USER_ID}:
            return {"skipped": True, "reason": "user is not eligible for long-term preferences"}
        from_id = int(job.get("from_event_id") or 0)
        to_id = int(job.get("to_event_id") or self.events.latest_user_event_id(user_id) or 0)
        device_id = str(job.get("device_id") or "")
        message_events = [
            event
            for event in self.events.list_events(user_id=user_id, role="user", limit=200, ascending=True)
            if from_id < int(event["id"]) <= to_id
        ]
        if not device_id and message_events:
            device_id = str(message_events[-1]["device_id"])
        action_events = [
            event
            for event in self.events.list_action_events(user_id=user_id, device_id=device_id or job.get("device_id"), limit=50)
            if from_id < int(event["id"]) <= to_id
        ]
        preference_context = self._build_preference_context(
            user_id,
            device_id or None,
            from_id,
            to_id,
            message_events,
            action_events,
        )
        if not preference_context["recent_turns"] and not preference_context["action_events"] and not preference_context.get("rolling_summary"):
            return {"skipped": True, "reason": "no events in extraction range", "from_event_id": from_id, "to_event_id": to_id}
        input_user_events = sum(len(turn["messages"]) for turn in preference_context["recent_turns"])
        input_action_events = len(preference_context["action_events"])
        existing = self.events.list_preferences(user_id, status=None, limit=100)
        preference_context["existing_preference_count"] = len(existing)
        changed = False
        upserted = 0
        seen_preference_keys: set[tuple[str, str]] = set()
        for pref in self._rule_based_preference_candidates(user_id, preference_context):
            key = str(pref["preference_key"])
            marker = (
                key,
                self.events.normalized_preference_value_key(
                    key,
                    pref["value"],
                    str(pref["display_text_zh"]),
                ),
            )
            if marker in seen_preference_keys:
                continue
            seen_preference_keys.add(marker)
            pref_id = self.events.upsert_preference(
                user_id,
                pref["preference_key"],
                pref["category"],
                pref["value"],
                pref["display_text_zh"],
                pref["evidence"],
                polarity=pref["polarity"],
                durability="persistent",
                strength=pref["strength"],
                confidence=pref["confidence"],
                source_type=pref["source_type"],
                extractor_model="local-rule",
                prompt_version="rule-v1",
                reason_zh=pref["reason_zh"],
                status=pref["status"],
            )
            changed = changed or pref_id is not None
            if pref_id is not None:
                upserted += 1
        model_error: str | None = None
        try:
            result = self.preference_extractor.extract(user_id, preference_context, existing)
        except Exception as exc:
            if not upserted:
                raise
            result = None
            model_error = str(exc)
        model_preferences = list(result.preferences) if result else []
        model_preference_count = len(model_preferences)
        model_stored = 0
        for pref in model_preferences:
            key, category = normalize_preference_key(pref.preference_key)
            seen_preference_keys.add(
                (
                    key,
                    self.events.normalized_preference_value_key(key, pref.value, pref.display_text_zh),
                )
            )
            if pref.action == "revoke":
                self.events.revoke_preference(user_id, key, pref.value)
                changed = True
                upserted += 1
                continue
            status = None
            if key == "other":
                status = "candidate"
            pref_id = self.events.upsert_preference(
                user_id,
                key,
                category if key != "other" else pref.category or "other",
                pref.value,
                pref.display_text_zh,
                [item.model_dump() for item in pref.evidence],
                polarity=pref.polarity,
                durability=pref.durability,
                strength=pref.strength,
                confidence=pref.confidence,
                source_type=pref.source_type,
                expires_at=pref.expires_at,
                extractor_model=self.preference_extractor.model,
                prompt_version=self.preference_extractor.prompt_version,
                reason_zh=pref.reason_zh,
                scope=pref.scope,
                status=status,
            )
            changed = changed or pref_id is not None
            if pref_id is not None:
                upserted += 1
                model_stored += 1
        if changed:
            self.events.enqueue_job(user_id, "user_card_rebuild")
        job_result = {
            "skipped": False,
            "from_event_id": from_id,
            "to_event_id": to_id,
            "context_mode": "summary_plus_recent_turns",
            "summary_version": preference_context.get("summary_version", 0),
            "recent_turn_count": len(preference_context["recent_turns"]),
            "action_event_count": len(preference_context["action_events"]),
            "summary_evidence_event_count": len(preference_context.get("summary_evidence_events", [])),
            "existing_preference_count": len(existing),
            "input_events": input_user_events + input_action_events,
            "input_user_events": input_user_events,
            "input_action_events": input_action_events,
            "preference_context_preview": self._preference_context_preview(preference_context),
            "extracted_preferences": model_preference_count,
            "model_stored_preferences": model_stored,
            "stored_preferences": upserted,
            "changed": changed,
        }
        if model_error:
            job_result["model_error"] = model_error
            job_result["warning"] = "preference extractor failed; local explicit memory rules were saved"
        return job_result

    def _route_summary_window_events(self, user_id: str, device_id: str, turns: list[dict[str, Any]]) -> None:
        user_messages = [turn["user"] for turn in turns]
        existing_time_sources = {
            int(
                (event.get("payload_json") or {}).get("source_event_id")
                or (event.get("payload_json") or {}).get("metadata", {}).get("source_event_id")
                or 0
            )
            for event in self.events.list_time_memories(user_id, device_id, limit=200)
        }
        for message in user_messages:
            parsed = parse_time_memory(str(message.get("content") or ""))
            source_id = int(message["id"])
            if parsed and source_id not in existing_time_sources:
                self.time_memory.remember(
                    user_id,
                    device_id,
                    str(message.get("content") or ""),
                    parsed["target_at"],
                    {
                        "task": parsed["task"],
                        "source_event_id": source_id,
                        "parser": parsed["parser"],
                        "source": "summary_window",
                    },
                )
                existing_time_sources.add(source_id)

        existing_action_sources = {
            tuple((event.get("payload_json") or {}).get("source_event_ids") or [])
            for event in self.events.list_action_events(user_id, device_id, limit=200)
        }
        group: list[tuple[dict[str, Any], dict[str, Any]]] = []

        def flush() -> None:
            nonlocal group
            if len(group) < 2:
                group = []
                return
            source_ids = tuple(int(item[0]["id"]) for item in group)
            if source_ids in existing_action_sources:
                group = []
                return
            actions = []
            for _, parsed_action in group:
                actions.extend(parsed_action.get("actions", []))
            self.events.add_event(
                f"batch-action-{user_id}-{device_id}-{source_ids[0]}-{source_ids[-1]}",
                user_id,
                device_id,
                "action_sequence",
                {
                    "actions": actions[:10],
                    "parser": "summary-window-rule-v1",
                    "source_event_ids": list(source_ids),
                },
                content=" / ".join(str(item[0].get("content") or "") for item in group),
            )
            existing_action_sources.add(source_ids)
            group = []

        for message in user_messages:
            parsed_action = parse_action_sequence(str(message.get("content") or ""))
            if parsed_action:
                group.append((message, parsed_action))
                if len(group) >= 10:
                    flush()
            else:
                flush()
        flush()

    def _build_preference_context(
        self,
        user_id: str,
        device_id: str | None,
        from_event_id: int,
        to_event_id: int,
        message_events: list[dict[str, Any]],
        action_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        summary = self.events.latest_summary(user_id, device_id) if device_id else None
        recent_turns = self.events.latest_conversation_turns(user_id, device_id, 5) if device_id else []
        summary_to = int((summary or {}).get("to_event_id") or (summary or {}).get("compacted_through_event_id") or 0)
        evidence_turns = self.events.conversation_turns_until(user_id, device_id, summary_to, 20) if device_id and summary_to else []
        turns = []
        for turn in recent_turns:
            turns.append(
                {
                    "request_id": turn["request_id"],
                    "messages": [
                        {
                            "event_id": turn["user"]["id"],
                            "role": "user",
                            "text": turn["user"]["content"],
                            "created_at": turn["user"]["timestamp"],
                        },
                        {
                            "event_id": turn["assistant"]["id"],
                            "role": "assistant",
                            "text": turn["assistant"]["content"],
                            "created_at": turn["assistant"]["timestamp"],
                        },
                    ],
                }
            )
        context_action_events = [
            {
                "event_id": event["id"],
                "device_id": event["device_id"],
                "text": event["content"],
                "created_at": event["created_at"],
                "event_type": "action_sequence",
                "actions": (event.get("payload_json") or {}).get("actions", []),
            }
            for event in action_events[-10:]
        ]
        summary_evidence_events = [
            {
                "event_id": turn["user"]["id"],
                "device_id": device_id,
                "text": turn["user"]["content"],
                "created_at": turn["user"]["timestamp"],
                "event_type": "message",
            }
            for turn in evidence_turns
        ]
        return {
            "context_mode": "summary_plus_recent_turns",
            "user_id": user_id,
            "device_id": device_id,
            "from_event_id": from_event_id,
            "to_event_id": to_event_id,
            "user_card": self.get_user_card(user_id) or self.restore_user_card(user_id) or {},
            "rolling_summary": (summary or {}).get("summary_text", ""),
            "summary_version": int((summary or {}).get("version", 0) or 0),
            "summary_event_range": {
                "from_event_id": (summary or {}).get("from_event_id"),
                "to_event_id": (summary or {}).get("to_event_id") or (summary or {}).get("compacted_through_event_id"),
            },
            "summary_evidence_events": summary_evidence_events,
            "recent_turns": turns,
            "action_events": context_action_events,
            "range_user_event_count": len(message_events),
        }

    @staticmethod
    def _preference_context_preview(context: dict[str, Any]) -> dict[str, Any]:
        return {
            "context_mode": context.get("context_mode"),
            "summary_version": context.get("summary_version"),
            "summary": str(context.get("rolling_summary") or "")[:600],
            "summary_evidence_events": context.get("summary_evidence_events", [])[:10],
            "recent_turns": [
                {
                    "request_id": turn.get("request_id"),
                    "messages": [
                        {
                            "event_id": message.get("event_id"),
                            "role": message.get("role"),
                            "text": str(message.get("text") or "")[:240],
                        }
                        for message in turn.get("messages", [])
                    ],
                }
                for turn in context.get("recent_turns", [])
            ],
            "action_events": context.get("action_events", [])[:5],
            "existing_preference_count": context.get("existing_preference_count", 0),
        }

    def _rule_based_preference_candidates(self, user_id: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        del user_id
        candidates: list[dict[str, Any]] = []
        evidence_events = list(context.get("summary_evidence_events") or [])
        for turn in context.get("recent_turns", []):
            for message in turn.get("messages", []):
                if message.get("role") == "user":
                    evidence_events.append(
                        {
                            "event_id": message.get("event_id"),
                            "text": message.get("text"),
                            "created_at": message.get("created_at"),
                            "event_type": "message",
                        }
                    )
        seen: set[tuple[str, str]] = set()
        for event in evidence_events:
            text = str(event.get("text") or "")
            event_id = int(event.get("event_id") or 0)
            for key, pattern, polarity, label_prefix in (
                ("preference.dislikes", r"(?:我)?(?:不喜欢|讨厌|不爱吃|不吃)([^，。,.；;、\s]{1,24})", "avoid", "不喜欢"),
                ("preference.likes", r"(?<!不)(?<!没)(?:我)?(?:喜欢|爱|爱吃)([^，。,.；;、\s]{1,24})", "prefer", "喜欢"),
                ("profile.occupation", r"(?:我是|我的职业是|我从事|我做)([^，。,.；;、\s]{1,24})", "prefer", "职业是"),
            ):
                for match in re.finditer(pattern, text):
                    value = match.group(1).strip(" 的了呢啊呀")
                    if value.startswith(("吃", "喝", "看", "听")) and len(value) > 1:
                        value = value[1:]
                    if not value:
                        continue
                    marker = (key, value)
                    if marker in seen:
                        continue
                    seen.add(marker)
                    category = "profile" if key == "profile.occupation" else "preference"
                    confidence = 0.9 if event_id else 0.7
                    candidates.append(
                        {
                            "preference_key": key,
                            "category": category,
                            "value": {"type": "string", "code": value, "label_zh": value},
                            "display_text_zh": f"{label_prefix}{value}",
                            "polarity": polarity,
                            "strength": 0.75,
                            "confidence": confidence,
                            "source_type": "explicit" if event_id else "implicit",
                            "evidence": [
                                {
                                    "event_id": event_id,
                                    "text": text,
                                    "type": "explicit" if event_id else "summary",
                                    "confidence": confidence,
                                }
                            ]
                            if event_id
                            else [],
                            "reason_zh": "本地规则从摘要证据窗口或最近对话中识别到明确长期记忆",
                            "status": "active" if confidence >= 0.85 else "candidate",
                        }
                    )
        return candidates

    def _build_user_card(self, user_id: str, preferences: list[dict[str, Any]]) -> dict[str, Any]:
        primary_order = {
            "profile.occupation": 3,
            "preference.likes": 2,
            "preference.dislikes": 2,
        }
        selected = sorted(
            preferences,
            key=lambda item: (
                primary_order.get(str(item.get("preference_key")), 0),
                float(item.get("confidence", 0)),
                int(item.get("evidence_count", 0)),
            ),
            reverse=True,
        )[:15]
        labels = [str(item.get("display_text_zh", "")) for item in selected if item.get("display_text_zh")]
        profile = "，".join(labels)[:600]
        return {
            "user_id": user_id,
            "version": max([int(item.get("revision", 1) or 1) for item in selected] or [1]),
            "profile_text_zh": profile,
            "preferences": [
                {
                    "key": item["preference_key"],
                    "value": (item.get("value_json") or {}).get("code")
                    or (item.get("value_json") or {}).get("label_zh")
                    or item.get("value_json"),
                    "label_zh": item.get("display_text_zh", PREFERENCE_REGISTRY.get(item["preference_key"], "")),
                }
                for item in selected
            ],
            "updated_at": iso_now(),
        }

    @staticmethod
    def _seconds_between(start: str, end: str) -> float:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
