"""Timestamped memories stored durably in SQLite."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .sqlite_event import SQLiteEventStore
from .utils import iso_now


class TimeMemory:
    def __init__(self, events: SQLiteEventStore, index_wake: Any | None = None) -> None:
        self.events = events
        self.index_wake = index_wake

    def remember(
        self,
        user_id: str,
        device_id: str,
        content: str,
        timestamp: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        payload = dict(metadata or {})
        target_at = timestamp or payload.pop("target_at", None) or iso_now()
        event_type = str(payload.get("event_type") or "scheduled_task")
        payload.update({"target_at": target_at, "metadata": dict(metadata or {})})
        return self.events.add_event(
            f"time-{user_id}-{device_id}-{iso_now()}",
            user_id,
            device_id,
            event_type,
            payload,
            content=content,
            created_at=iso_now(),
        )

    def archive_due(self, before: datetime | None = None) -> int:
        """Compatibility no-op: time memories are already durable in SQLite."""
        del before
        return 0


class DailyArchiveScheduler:
    """Deprecated compatibility shim; no archive is needed for SQLite time memories."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def start(self) -> None:
        return None

    def stop(self, timeout: float = 2.0) -> None:
        del timeout
