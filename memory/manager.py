"""Public entry point for the lightweight Memory OS runtime."""

from __future__ import annotations

import logging
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
        session_idle_seconds: int = 15,
        session_ttl_seconds: int = 86400,
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
        self.session_idle_seconds = max(1, session_idle_seconds)
        self.session_ttl_seconds = max(60, session_ttl_seconds)
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

    def resolve_session(
        self,
        user_id: str,
        device_id: str,
        *,
        timestamp: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        now = timestamp or iso_now()
        local_date = self._local_date(now)
        expires_at = (self._parse_datetime(now) + timedelta(seconds=self.session_ttl_seconds)).isoformat()
        if session_id:
            existing = self.events.get_session(session_id)
            if existing and existing.get("user_id") == user_id and existing.get("device_id") == device_id:
                session = {**existing, "last_activity_at": now, "expires_at": expires_at, "status": "active"}
                self.events.upsert_session(
                    session_id,
                    user_id,
                    device_id,
                    str(session.get("local_date") or local_date),
                    str(session.get("started_at") or now),
                    now,
                    expires_at,
                    "active",
                )
                self.redis.set_active_session(user_id, device_id, session, self.session_ttl_seconds)
                return session
        candidate = self.redis.get_active_session(user_id, device_id)
        if not candidate:
            candidate = self.events.latest_session(user_id, device_id, local_date)
        if candidate:
            last_activity = str(candidate.get("last_activity_at") or "")
            same_day = str(candidate.get("local_date") or local_date) == local_date
            gap_seconds = self._seconds_between(last_activity, now) if last_activity else self.session_idle_seconds + 1
            if same_day and 0 <= gap_seconds <= self.session_idle_seconds:
                session = {**candidate, "last_activity_at": now, "expires_at": expires_at, "status": "active"}
                self.events.upsert_session(
                    str(session["session_id"]),
                    user_id,
                    device_id,
                    local_date,
                    str(session.get("started_at") or now),
                    now,
                    expires_at,
                    "active",
                )
                self.redis.set_active_session(user_id, device_id, session, self.session_ttl_seconds)
                return session
        new_session = {
            "session_id": f"sess-{uuid.uuid4().hex}",
            "user_id": user_id,
            "device_id": device_id,
            "local_date": local_date,
            "started_at": now,
            "last_activity_at": now,
            "expires_at": expires_at,
            "status": "active",
        }
        self.events.upsert_session(**new_session)
        self.redis.set_active_session(user_id, device_id, new_session, self.session_ttl_seconds)
        return new_session

    def start_memory_worker(self, poll_seconds: float = 2.0) -> None:
        if self._job_thread and self._job_thread.is_alive():
            return
        self._job_stop.clear()

        def _loop() -> None:
            while not self._job_stop.wait(poll_seconds):
                try:
                    self.process_memory_jobs_once(include_daily=True)
                except Exception:
                    logger.exception("memory_job.worker_failed")

        self._job_thread = threading.Thread(target=_loop, name="memory-jobs", daemon=True)
        self._job_thread.start()

    def get_conversation_context(
        self,
        user_id: str,
        device_id: str,
        session_id: str | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        session = self.resolve_session(user_id, device_id, timestamp=timestamp, session_id=session_id)
        bundle = self.redis.get_context_bundle(device_id, user_id, recent_limit=10)
        summary = bundle["summary"] or self._restore_summary(user_id, device_id)
        user_card = bundle["user_card"] or self.restore_user_card(user_id)
        recent = self.redis.get_session_conversation(user_id, session["session_id"], limit=10)
        if not recent:
            latest_summary_id = int((summary or {}).get("compacted_through_event_id", 0) or 0)
            recent = self.events.message_range(
                user_id,
                device_id,
                latest_summary_id,
                20,
                session_id=session["session_id"],
            )[-10:]
            if recent:
                self.redis.append_session_conversation(
                    user_id,
                    session["session_id"],
                    recent,
                    ttl_seconds=self.session_ttl_seconds,
                    max_items=20,
                )
        latest_action = self.events.latest_action_sequence(user_id, device_id)
        return {
            "user_id": user_id,
            "device_id": device_id,
            "session_id": session["session_id"],
            "session": session,
            "user_card": user_card,
            "rolling_summary": (summary or {}).get("summary_text", ""),
            "summary_version": int((summary or {}).get("version", 0) or 0),
            "summary_pending": self._conversation_states.get((user_id, device_id), _ConversationState()).summary_pending,
            "recent_messages": recent[-10:],
            "latest_action_sequence": latest_action,
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
        session_id: str | None = None,
    ) -> dict[str, Any]:
        timestamp = timestamp or iso_now()
        session = self.resolve_session(user_id, device_id, timestamp=timestamp, session_id=session_id)
        user_event_id, assistant_event_id = self.events.add_message_pair(
            request_id,
            user_id,
            device_id,
            user_text,
            assistant_text,
            timestamp,
            session_id=session["session_id"],
        )
        messages = [
            {"id": user_event_id, "role": "user", "content": user_text, "timestamp": timestamp, "session_id": session["session_id"]},
            {"id": assistant_event_id, "role": "assistant", "content": assistant_text, "timestamp": timestamp, "session_id": session["session_id"]},
        ]
        self.redis.append_session_conversation(
            user_id,
            session["session_id"],
            messages,
            ttl_seconds=self.session_ttl_seconds,
            max_items=self.summary_every_turns * 2,
        )
        routed: dict[str, Any] = {}
        if model_event_routes:
            routed.update(
                self._route_model_event_candidates(
                    request_id,
                    user_id,
                    device_id,
                    user_text,
                    user_event_id,
                    model_event_routes,
                    timestamp,
                    session["session_id"],
                )
            )
        memory_date = self._local_date(timestamp)
        daily_jobs = self._schedule_daily_extraction(user_id, device_id, memory_date, user_event_id, assistant_event_id)
        self._schedule_summary(user_id, device_id)
        if user_id != "anonymous":
            self._maybe_schedule_preference(
                user_id,
                device_id,
                user_text,
                int(routed.get("action_event_id") or user_event_id),
            )
        return {
            "user_event_id": user_event_id,
            "assistant_event_id": assistant_event_id,
            "session_id": session["session_id"],
            "session": session,
            "daily_extraction": daily_jobs,
            **routed,
        }

    def _route_model_event_candidates(
        self,
        request_id: str,
        user_id: str,
        device_id: str,
        user_text: str,
        source_event_id: int,
        routes: list[dict[str, Any]],
        created_at: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        accepted: list[dict[str, Any]] = []
        for route in routes:
            if not isinstance(route, dict):
                continue
            confidence = float(route.get("confidence") or 0)
            event_type = str(route.get("type") or "")
            decision = str(route.get("decision") or "")
            if confidence < 0.7 or event_type not in {"action_sequence", "action_feedback"}:
                continue
            if decision != "create":
                continue
            if event_type == "action_feedback" and not self._has_action_reference(route):
                continue
            accepted.append({**route, "source": "reply_model_candidate", "source_event_id": source_event_id})
        if not accepted:
            return {}
        routed: dict[str, Any] = {"model_event_routes": accepted}
        for route in accepted:
            event_type = str(route.get("type") or "")
            decision = str(route.get("decision") or "")
            if event_type == "action_sequence" and decision == "create":
                actions = route.get("actions")
                if not isinstance(actions, list) or not actions:
                    continue
                event_id = self.events.add_event(
                    f"{request_id}-action",
                    user_id,
                    device_id,
                    "action_sequence",
                    {
                        "event_type": "action_sequence",
                        "actions": actions[:10],
                        "parser": "reply-model-candidate-v1",
                        "source_event_id": source_event_id,
                        "source_event_ids": [source_event_id],
                        "model_route": route,
                    },
                    content=user_text,
                    created_at=created_at,
                    session_id=session_id,
                )
                routed["action_event_id"] = event_id
                continue
            if event_type == "action_feedback" and decision == "create":
                feedback_text = str(route.get("feedback") or route.get("feedback_text") or user_text).strip()
                if not feedback_text:
                    continue
                event_id = self.events.add_event(
                    f"{request_id}-action-feedback",
                    user_id,
                    device_id,
                    "action_feedback",
                    {
                        "event_type": "action_feedback",
                        "feedback": feedback_text,
                        "action_id": route.get("action_id"),
                        "action_event_id": route.get("action_event_id"),
                        "action_memory_id": route.get("action_memory_id"),
                        "parser": "reply-model-candidate-v1",
                        "source_event_id": source_event_id,
                        "source_event_ids": [source_event_id],
                        "model_route": route,
                    },
                    content=feedback_text,
                    created_at=created_at,
                    session_id=session_id,
                )
                routed["action_feedback_event_id"] = event_id
                continue
        return routed

    @staticmethod
    def _has_action_reference(route: dict[str, Any]) -> bool:
        return bool(route.get("action_id") or route.get("action_event_id") or route.get("action_memory_id"))

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

    def process_memory_jobs_once(self, limit: int | None = None, include_daily: bool = False) -> dict[str, Any]:
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
            job_types={
                "conversation_summary",
                "preference_extraction",
                "user_card_rebuild",
                "daily_time_memory_extract",
                "daily_action_memory_extract",
                "weekly_action_preference_extract",
            }
            if include_daily
            else {"conversation_summary", "preference_extraction", "user_card_rebuild"},
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
                elif job["job_type"] == "daily_time_memory_extract":
                    job_result = self._process_daily_time_memory_job(job)
                elif job["job_type"] == "daily_action_memory_extract":
                    job_result = self._process_daily_action_memory_job(job)
                elif job["job_type"] == "weekly_action_preference_extract":
                    job_result = self._process_weekly_action_preference_job(job)
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

    def trigger_daily_extraction(
        self,
        user_id: str,
        device_id: str,
        memory_date: str,
        *,
        process_now: bool = True,
    ) -> dict[str, Any]:
        events = self.events.list_events_for_local_date(
            user_id,
            device_id,
            memory_date,
            event_type="message",
        )
        action_events = self.events.list_events_for_local_date(
            user_id,
            device_id,
            memory_date,
            event_type="action_sequence",
        )
        feedback_events = self.events.list_events_for_local_date(
            user_id,
            device_id,
            memory_date,
            event_type="action_feedback",
        )
        selected_ids = [int(event["id"]) for event in events + action_events + feedback_events]
        if not selected_ids:
            return {
                "user_id": user_id,
                "device_id": device_id,
                "memory_date": memory_date,
                "created_jobs": [],
                "message": "指定日期没有可抽取的会话或动作事件",
                "process": self.process_memory_jobs_once(limit=1, include_daily=True) if process_now else None,
            }
        from_event_id = max(0, min(selected_ids) - 1)
        to_event_id = max(selected_ids)
        created_jobs = []
        for job_type in ("daily_time_memory_extract", "daily_action_memory_extract"):
            job_id = self.events.upsert_pending_device_job(
                user_id,
                device_id,
                job_type,
                from_event_id=from_event_id,
                to_event_id=to_event_id,
            )
            created_jobs.append({"job_type": job_type, "job_id": job_id})
        return {
            "user_id": user_id,
            "device_id": device_id,
            "memory_date": memory_date,
            "from_event_id": from_event_id,
            "to_event_id": to_event_id,
            "created_jobs": created_jobs,
            "process": self.process_memory_jobs_once(limit=2, include_daily=True) if process_now else None,
        }

    def trigger_weekly_action_preference_extraction(
        self,
        user_id: str,
        device_id: str,
        end_date: str,
        *,
        process_now: bool = True,
    ) -> dict[str, Any]:
        if user_id in {"anonymous", LEGACY_USER_ID}:
            return {
                "user_id": user_id,
                "device_id": device_id,
                "end_date": end_date,
                "created_job": False,
                "message": "anonymous 和 legacy-unassigned 不生成长期偏好",
                "process": self.process_memory_jobs_once(limit=1, include_daily=True) if process_now else None,
            }
        dates = self._date_window(end_date, 7)
        memories = []
        for date in dates:
            memories.extend(self.events.list_action_memories(user_id, device_id, memory_date=date, limit=500))
        if not memories:
            return {
                "user_id": user_id,
                "device_id": device_id,
                "end_date": end_date,
                "dates": dates,
                "created_job": False,
                "message": "最近七天没有 action_memory 可用于偏好抽取",
                "process": self.process_memory_jobs_once(limit=1, include_daily=True) if process_now else None,
            }
        ids = [int(item["id"]) for item in memories]
        job_id = self.events.enqueue_job(
            user_id,
            "weekly_action_preference_extract",
            device_id=device_id,
            from_event_id=max(0, min(ids) - 1),
            to_event_id=max(ids),
        )
        return {
            "user_id": user_id,
            "device_id": device_id,
            "end_date": end_date,
            "dates": dates,
            "created_job": job_id is not None,
            "job_id": job_id,
            "input_action_memories": len(memories),
            "from_event_id": max(0, min(ids) - 1),
            "to_event_id": max(ids),
            "process": self.process_memory_jobs_once(limit=2, include_daily=True) if process_now else None,
        }

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
            "jobs": self.events.job_counts(include_daily=True),
            **self.events.stats(),
        }

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

    def _schedule_daily_extraction(
        self,
        user_id: str,
        device_id: str,
        memory_date: str,
        user_event_id: int,
        assistant_event_id: int,
    ) -> dict[str, Any]:
        if user_id in {"anonymous", LEGACY_USER_ID}:
            return {
                "memory_date": memory_date,
                "from_event_id": max(0, min(user_event_id, assistant_event_id) - 1),
                "to_event_id": max(user_event_id, assistant_event_id),
                "queued": [],
                "skipped": True,
                "reason": "user is not eligible for daily extraction",
            }
        from_event_id = max(0, min(user_event_id, assistant_event_id) - 1)
        to_event_id = max(user_event_id, assistant_event_id)
        queued = []
        for job_type in ("daily_time_memory_extract", "daily_action_memory_extract"):
            job_id = self.events.upsert_pending_device_job(
                user_id,
                device_id,
                job_type,
                from_event_id=from_event_id,
                to_event_id=to_event_id,
            )
            queued.append({"job_type": job_type, "job_id": job_id})
        return {
            "memory_date": memory_date,
            "from_event_id": from_event_id,
            "to_event_id": to_event_id,
            "queued": queued,
        }

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
        finally:
            with self._lock:
                state = self._conversation_states.setdefault(key, _ConversationState())
                state.summary_pending = False

    def _process_daily_time_memory_job(self, job: dict[str, Any]) -> dict[str, Any]:
        user_id = str(job["user_id"])
        device_id = str(job.get("device_id") or "")
        memory_date = self._job_memory_date(job)
        messages = self.events.list_events_for_local_date(
            user_id,
            device_id,
            memory_date,
            event_type="message",
        )
        if not messages:
            return {
                "skipped": True,
                "reason": "no messages for local date",
                "memory_date": memory_date,
                "event_type": "time_memory",
            }
        summary_input = [
            {
                "role": item.get("role"),
                "content": item.get("content"),
                "id": item.get("id"),
                "session_id": item.get("session_id"),
                "created_at": item.get("created_at"),
            }
            for item in messages
            if item.get("role") in {"user", "assistant"}
        ]
        summary = self.summarizer.summarize(summary_input, "")
        if not summary:
            return {
                "skipped": True,
                "reason": "empty daily summary",
                "memory_date": memory_date,
                "event_type": "time_memory",
            }
        content = f"日期总结（{memory_date}）\n{summary}"
        event_id = self.events.upsert_daily_time_memory(
            user_id,
            device_id,
            memory_date,
            content,
            source_event_ids=[int(item["id"]) for item in messages],
            metadata={
                "message_count": len(messages),
                "session_ids": sorted({str(item.get("session_id")) for item in messages if item.get("session_id")}),
                "summary_backend": getattr(self.summarizer, "backend", "unknown"),
            },
        )
        return {
            "skipped": False,
            "event_type": "time_memory",
            "event_id": event_id,
            "memory_date": memory_date,
            "message_count": len(messages),
        }

    def _process_daily_action_memory_job(self, job: dict[str, Any]) -> dict[str, Any]:
        user_id = str(job["user_id"])
        device_id = str(job.get("device_id") or "")
        memory_date = self._job_memory_date(job)
        action_events = self.events.list_events_for_local_date(
            user_id,
            device_id,
            memory_date,
            event_type="action_sequence",
        )
        feedback_events = self.events.list_events_for_local_date(
            user_id,
            device_id,
            memory_date,
            event_type="action_feedback",
        )
        if not action_events and not feedback_events:
            return {
                "skipped": True,
                "reason": "no action_sequence or action_feedback events for local date",
                "memory_date": memory_date,
                "event_type": "action_memory",
            }
        message_events = self.events.list_events_for_local_date(
            user_id,
            device_id,
            memory_date,
            event_type="message",
        )
        source_message_ids_by_session: dict[str, list[int]] = {}
        for message in message_events:
            sid = str(message.get("session_id") or "")
            if sid:
                source_message_ids_by_session.setdefault(sid, []).append(int(message["id"]))
        memories: list[dict[str, Any]] = []
        for event in action_events:
            actions = (event.get("payload_json") or {}).get("actions", [])
            if not isinstance(actions, list) or not actions:
                continue
            labels = [
                str(action.get("label_zh") or action.get("code") or "").strip()
                for action in actions
                if isinstance(action, dict) and str(action.get("label_zh") or action.get("code") or "").strip()
            ]
            action_text = str(event.get("content") or " -> ".join(labels)).strip()
            session_id = str(event.get("session_id") or (event.get("payload_json") or {}).get("session_id") or "")
            memories.append(
                {
                    "content": f"动作记忆（{memory_date}）：{action_text}",
                    "title": f"{memory_date} 动作记忆",
                    "session_id": session_id or None,
                    "event_at": event.get("created_at"),
                    "actions": [dict(action) for action in actions if isinstance(action, dict)],
                    "source_event_ids": [int(event["id"])],
                    "source_message_event_ids": source_message_ids_by_session.get(session_id, []),
                    "confidence": float(((event.get("payload_json") or {}).get("model_route") or {}).get("confidence") or 0.8),
                    "metadata": {
                        "source_event_type": "action_sequence",
                        "source_content": event.get("content"),
                    },
                }
            )
        for event in feedback_events:
            payload = event.get("payload_json") or {}
            feedback_text = str(payload.get("feedback") or event.get("content") or "").strip()
            if not feedback_text:
                continue
            session_id = str(event.get("session_id") or payload.get("session_id") or "")
            ref = payload.get("action_id") or payload.get("action_event_id") or payload.get("action_memory_id")
            memories.append(
                {
                    "content": f"动作反馈记忆（{memory_date}）：{feedback_text}",
                    "title": f"{memory_date} 动作反馈记忆",
                    "session_id": session_id or None,
                    "event_at": event.get("created_at"),
                    "actions": [],
                    "source_event_ids": [int(event["id"])],
                    "source_message_event_ids": source_message_ids_by_session.get(session_id, []),
                    "confidence": float((payload.get("model_route") or {}).get("confidence") or 0.8),
                    "metadata": {
                        "source_event_type": "action_feedback",
                        "action_reference": ref,
                    },
                }
            )
        if not memories:
            return {
                "skipped": True,
                "reason": "no structured action payloads",
                "memory_date": memory_date,
                "event_type": "action_memory",
            }
        event_ids = self.events.replace_daily_action_memories(user_id, device_id, memory_date, memories)
        return {
            "skipped": False,
            "event_type": "action_memory",
            "event_ids": event_ids,
            "memory_date": memory_date,
            "action_memory_count": len(event_ids),
            "source_event_ids": [int(event["id"]) for event in action_events + feedback_events],
        }

    def _process_weekly_action_preference_job(self, job: dict[str, Any]) -> dict[str, Any]:
        user_id = str(job["user_id"])
        if user_id in {"anonymous", LEGACY_USER_ID}:
            return {"skipped": True, "reason": "user is not eligible for weekly action preference memory"}
        from_id = int(job.get("from_event_id") or 0)
        to_id = int(job.get("to_event_id") or 0)
        device_id = str(job.get("device_id") or "")
        action_memories = [
            event
            for event in self.events.list_events(user_id=user_id, device_id=device_id or None, event_type="action_memory", limit=1000, ascending=True)
            if from_id < int(event["id"]) <= to_id
        ]
        if not action_memories:
            return {"skipped": True, "reason": "no action_memory events in extraction range", "from_event_id": from_id, "to_event_id": to_id}
        dates = sorted(
            {
                str((event.get("payload_json") or {}).get("memory_date") or self._local_date(str(event.get("created_at") or iso_now())))
                for event in action_memories
            }
        )
        start_date = dates[0]
        end_date = dates[-1]
        preference_context = {
            "context_mode": "weekly_action_memory_preferences",
            "user_id": user_id,
            "device_id": device_id or None,
            "from_event_id": from_id,
            "to_event_id": to_id,
            "user_card": self.get_user_card(user_id) or self.restore_user_card(user_id) or {},
            "action_memory_events": [
                {
                    "event_id": event["id"],
                    "device_id": event["device_id"],
                    "session_id": event.get("session_id"),
                    "memory_date": (event.get("payload_json") or {}).get("memory_date"),
                    "text": event.get("content"),
                    "created_at": event.get("created_at"),
                    "actions": (event.get("payload_json") or {}).get("actions", []),
                }
                for event in action_memories
            ],
            "action_memory_count": len(action_memories),
        }
        extraction = self.preference_extractor.extract(user_id, preference_context, [])
        memories = []
        for idx, pref in enumerate(extraction.preferences, start=1):
            text = str(pref.display_text_zh or "").strip()
            if not text:
                continue
            source_event_ids = [int(ev.event_id) for ev in pref.evidence if int(ev.event_id or 0) > 0]
            if not source_event_ids:
                source_event_ids = [int(event["id"]) for event in action_memories]
            memories.append(
                {
                    "content": f"七天动作偏好记忆（{start_date} 至 {end_date}）：{text}",
                    "title": f"{start_date} 至 {end_date} 动作偏好记忆 #{idx}",
                    "source_event_ids": source_event_ids,
                    "confidence": pref.confidence,
                    "metadata": {
                        "preference_key": pref.preference_key,
                        "category": pref.category,
                        "value": pref.value,
                        "reason_zh": pref.reason_zh,
                        "extractor_model": getattr(self.preference_extractor, "model", None),
                        "prompt_version": getattr(self.preference_extractor, "prompt_version", None),
                    },
                }
            )
        event_ids = self.events.replace_weekly_action_preference_memories(
            user_id,
            device_id,
            start_date,
            end_date,
            memories,
        )
        return {
            "skipped": False,
            "from_event_id": from_id,
            "to_event_id": to_id,
            "context_mode": "weekly_action_memory_preferences",
            "action_memory_count": len(action_memories),
            "input_events": len(action_memories),
            "input_action_events": len(action_memories),
            "stored_action_preference_memories": len(event_ids),
            "action_preference_memory_event_ids": event_ids,
            "preference_context_preview": self._preference_context_preview(preference_context),
        }

    def _job_memory_date(self, job: dict[str, Any]) -> str:
        event_id = int(job.get("to_event_id") or 0)
        event = self.events.get_event(event_id) if event_id else None
        return self._local_date(str((event or {}).get("created_at") or iso_now()))

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
        result = self.preference_extractor.extract(user_id, preference_context, existing)
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
        return job_result

    def _extract_and_store_preferences(
        self,
        user_id: str,
        preference_context: dict[str, Any],
        existing: list[dict[str, Any]],
    ) -> dict[str, Any]:
        changed = False
        upserted = 0
        result = self.preference_extractor.extract(user_id, preference_context, existing)
        model_preferences = list(result.preferences) if result else []
        model_preference_count = len(model_preferences)
        model_stored = 0
        for pref in model_preferences:
            key, category = normalize_preference_key(pref.preference_key)
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
        return {
            "existing_preference_count": len(existing),
            "extracted_preferences": model_preference_count,
            "model_stored_preferences": model_stored,
            "stored_preferences": upserted,
            "changed": changed,
        }

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
            "action_memory_events": context.get("action_memory_events", [])[:10],
            "existing_preference_count": context.get("existing_preference_count", 0),
        }


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
    def _local_date(value: str, timezone_name: str = "Asia/Shanghai") -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return value[:10]
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
        return parsed.astimezone(ZoneInfo(timezone_name)).date().isoformat()

    @classmethod
    def _date_window(cls, end_date: str, days: int) -> list[str]:
        parsed = cls._parse_datetime(f"{end_date}T00:00:00+08:00")
        return [
            (parsed - timedelta(days=offset)).date().isoformat()
            for offset in range(max(1, days) - 1, -1, -1)
        ]

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            parsed = datetime.fromisoformat(f"{value[:10]}T00:00:00+00:00")
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
        return parsed

    @classmethod
    def _seconds_between(cls, start: str, end: str) -> float:
        return (cls._parse_datetime(end) - cls._parse_datetime(start)).total_seconds()
