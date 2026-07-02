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
        memory_at = timestamp or str(payload.get("memory_at") or "") or iso_now()
        memory_date = str(payload.get("memory_date") or memory_at[:10])
        title = str(payload.get("title") or "")
        payload.update(
            {
                "memory_at": memory_at,
                "memory_date": memory_date,
                "title": title,
                "source": payload.get("source", "manual"),
                "metadata": dict(metadata or {}),
            }
        )
        return self.events.add_event(
            f"time-{user_id}-{device_id}-{iso_now()}",
            user_id,
            device_id,
            "time_memory",
            payload,
            content=content,
            created_at=iso_now(),
        )

    def archive_due(self, before: datetime | None = None) -> int:
        """Compatibility no-op: time memories are already durable in SQLite."""
        del before
        return 0
