"""SQLite is the durable source of truth for Memory OS."""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .utils import iso_now, tokenize, utc_now


LEGACY_USER_ID = "legacy-unassigned"
LEGACY_INDEX_TABLE = "index" + "_" + "outbox"


class SQLiteEventStore:
    """Durable event, preference, job, tool and device-state store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._migration_log: list[str] = []
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._migrate_if_needed()
        self._create_schema()

    @property
    def migration_log(self) -> list[str]:
        return list(self._migration_log)

    def _migrate_if_needed(self) -> None:
        if not self._table_has_column("events", "session_id") or self._table_has_column("events", "user_id"):
            return
        backup = self.path.with_name(f"{self.path.name}.bak-{iso_now().replace(':', '-')}")
        self._conn.commit()
        shutil.copy2(self.path, backup)
        self._migration_log.append(f"backup={backup}")

        rows = self._conn.execute("SELECT * FROM events ORDER BY id").fetchall()
        memory_rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
        ).fetchone()
        old_tables = [
            name
            for (name,) in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        for name in old_tables:
            if name == LEGACY_INDEX_TABLE:
                self._conn.execute(f"DROP TABLE IF EXISTS {LEGACY_INDEX_TABLE}")
                self._migration_log.append(f"dropped={LEGACY_INDEX_TABLE}")
            else:
                self._conn.execute(f"ALTER TABLE {name} RENAME TO legacy_{name}")
                self._migration_log.append(f"renamed={name}->legacy_{name}")
        self._create_schema()

        imported = 0
        for row in rows:
            payload = self._loads(row["payload"], {})
            role = payload.get("role") if isinstance(payload, dict) else None
            content = payload.get("content") if isinstance(payload, dict) else None
            self._conn.execute(
                """INSERT INTO events
                (request_id,user_id,device_id,event_type,role,content,payload_json,created_at)
                VALUES(?,?,?,?,?,?,?,?)""",
                (
                    f"legacy-{row['id']}",
                    LEGACY_USER_ID,
                    str(row["session_id"]),
                    str(row["event_type"]),
                    role if role in {"user", "assistant", "system", "tool"} else None,
                    str(content) if content is not None else None,
                    row["payload"],
                    str(row["created_at"]),
                ),
            )
            imported += 1
        self._conn.commit()
        self._migration_log.append(f"legacy_events_imported={imported}")
        self._migration_log.append(f"legacy_memories_preserved={bool(memory_rows)}")

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                session_id TEXT,
                event_type TEXT NOT NULL,
                role TEXT,
                content TEXT,
                payload_json TEXT,
                created_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS uq_events_request_role
                ON events(request_id, role)
                WHERE role IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_events_user_device_time
                ON events(user_id, device_id, id);
            CREATE INDEX IF NOT EXISTS idx_events_filters
                ON events(user_id, device_id, role, id);
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                local_date TEXT NOT NULL,
                started_at TEXT NOT NULL,
                last_activity_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active'
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_user_device_date
                ON conversation_sessions(user_id, device_id, local_date, last_activity_at);

            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                compacted_through_event_id INTEGER NOT NULL,
                version INTEGER NOT NULL,
                from_event_id INTEGER,
                to_event_id INTEGER,
                turn_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_summaries_user_device
                ON conversation_summaries(user_id, device_id, version);

            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                preference_key TEXT NOT NULL,
                category TEXT NOT NULL,
                value_type TEXT NOT NULL,
                value_json TEXT NOT NULL,
                display_text_zh TEXT NOT NULL,
                polarity TEXT NOT NULL DEFAULT 'prefer',
                durability TEXT NOT NULL DEFAULT 'persistent',
                strength REAL NOT NULL DEFAULT 0.5,
                confidence REAL NOT NULL DEFAULT 0.5,
                source_type TEXT NOT NULL DEFAULT 'explicit',
                status TEXT NOT NULL DEFAULT 'candidate',
                evidence_count INTEGER NOT NULL DEFAULT 1,
                first_seen_at TEXT NOT NULL,
                last_confirmed_at TEXT NOT NULL,
                expires_at TEXT,
                revision INTEGER NOT NULL DEFAULT 1,
                supersedes_id INTEGER,
                extractor_model TEXT,
                prompt_version TEXT,
                reason_zh TEXT,
                scope TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (supersedes_id) REFERENCES user_preferences(id),
                CHECK (status IN ('candidate','active','superseded','revoked','rejected')),
                CHECK (confidence >= 0 AND confidence <= 1),
                CHECK (strength >= 0 AND strength <= 1),
                CHECK (json_valid(value_json))
            );
            CREATE INDEX IF NOT EXISTS idx_preferences_user_active
                ON user_preferences(user_id, category, preference_key)
                WHERE status = 'active';
            DROP INDEX IF EXISTS uq_active_user_preference;
            CREATE UNIQUE INDEX IF NOT EXISTS uq_active_single_value_preference
                ON user_preferences(user_id, preference_key)
                WHERE status = 'active'
                  AND preference_key NOT IN ('preference.likes','preference.dislikes');
            CREATE UNIQUE INDEX IF NOT EXISTS uq_active_multi_value_preference
                ON user_preferences(user_id, preference_key, value_json)
                WHERE status = 'active'
                  AND preference_key IN ('preference.likes','preference.dislikes');

            CREATE TABLE IF NOT EXISTS preference_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                preference_id INTEGER NOT NULL,
                event_id INTEGER NOT NULL,
                evidence_type TEXT NOT NULL,
                evidence_text TEXT NOT NULL,
                confidence REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (preference_id) REFERENCES user_preferences(id),
                FOREIGN KEY (event_id) REFERENCES events(id),
                UNIQUE (preference_id, event_id)
            );

            CREATE TABLE IF NOT EXISTS memory_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                device_id TEXT,
                job_type TEXT NOT NULL,
                from_event_id INTEGER,
                to_event_id INTEGER,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                available_at TEXT NOT NULL,
                locked_at TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK (status IN ('pending','running','succeeded','failed'))
            );
            CREATE UNIQUE INDEX IF NOT EXISTS uq_pending_preference_job
                ON memory_jobs(user_id, job_type)
                WHERE job_type='preference_extraction' AND status IN ('pending','running');
            CREATE INDEX IF NOT EXISTS idx_jobs_ready
                ON memory_jobs(status, available_at, id);

            CREATE TABLE IF NOT EXISTS tool_runs (
                run_id TEXT PRIMARY KEY,
                context_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                input TEXT NOT NULL,
                output TEXT,
                status TEXT NOT NULL,
                error TEXT,
                idempotency_key TEXT UNIQUE,
                started_at TEXT NOT NULL,
                finished_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_tool_runs_context
                ON tool_runs(context_id, started_at);

            CREATE TABLE IF NOT EXISTS tool_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES tool_runs(run_id) ON DELETE CASCADE,
                step_name TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tool_steps_run ON tool_steps(run_id, id);

            CREATE TABLE IF NOT EXISTS device_state_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                state TEXT NOT NULL,
                reason TEXT NOT NULL,
                observed_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_device_history
                ON device_state_history(device_id, observed_at);
            """
        )
        self._conn.commit()
        self._ensure_column("user_preferences", "reason_zh", "TEXT")
        self._ensure_column("user_preferences", "scope", "TEXT NOT NULL DEFAULT 'user'")
        self._ensure_column("events", "session_id", "TEXT")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_session ON events(user_id, session_id, id)"
        )
        self._conn.commit()
        self._ensure_column("conversation_summaries", "from_event_id", "INTEGER")
        self._ensure_column("conversation_summaries", "to_event_id", "INTEGER")
        self._ensure_column("conversation_summaries", "turn_count", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_memory_jobs_accepts_extraction_types()

    def upsert_session(
        self,
        session_id: str,
        user_id: str,
        device_id: str,
        local_date: str,
        started_at: str,
        last_activity_at: str,
        expires_at: str,
        status: str = "active",
    ) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO conversation_sessions
                (session_id,user_id,device_id,local_date,started_at,last_activity_at,expires_at,status)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_activity_at=excluded.last_activity_at,
                    expires_at=excluded.expires_at,
                    status=excluded.status""",
                (session_id, user_id, device_id, local_date, started_at, last_activity_at, expires_at, status),
            )
            self._conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM conversation_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def latest_session(
        self,
        user_id: str,
        device_id: str,
        local_date: str | None = None,
    ) -> dict[str, Any] | None:
        params: list[Any] = [user_id, device_id]
        where = "WHERE user_id=? AND device_id=?"
        if local_date:
            where += " AND local_date=?"
            params.append(local_date)
        with self._lock:
            row = self._conn.execute(
                f"""SELECT * FROM conversation_sessions {where}
                ORDER BY last_activity_at DESC LIMIT 1""",
                params,
            ).fetchone()
        return dict(row) if row else None

    def list_sessions(
        self,
        user_id: str,
        device_id: str | None = None,
        local_date: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        params: list[Any] = [user_id]
        where = "WHERE user_id=?"
        if device_id:
            where += " AND device_id=?"
            params.append(device_id)
        if local_date:
            where += " AND local_date=?"
            params.append(local_date)
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT * FROM conversation_sessions {where}
                ORDER BY last_activity_at DESC LIMIT ?""",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def add_message_pair(
        self,
        request_id: str,
        user_id: str,
        device_id: str,
        user_text: str,
        assistant_text: str,
        created_at: str | None = None,
        payload: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> tuple[int, int]:
        now = created_at or iso_now()
        event_payload = dict(payload or {})
        if session_id:
            event_payload.setdefault("session_id", session_id)
        with self._lock:
            if self._conn.in_transaction:
                self._conn.rollback()
            try:
                user_cursor = self._conn.execute(
                    """INSERT INTO events
                    (request_id,user_id,device_id,session_id,event_type,role,content,payload_json,created_at)
                    VALUES(?,?,?,?,?,?,?,?,?)""",
                    (request_id, user_id, device_id, session_id, "message", "user", user_text, self._json(event_payload), now),
                )
                assistant_cursor = self._conn.execute(
                    """INSERT INTO events
                    (request_id,user_id,device_id,session_id,event_type,role,content,payload_json,created_at)
                    VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        request_id,
                        user_id,
                        device_id,
                        session_id,
                        "message",
                        "assistant",
                        assistant_text,
                        self._json(event_payload),
                        now,
                    ),
                )
                self._conn.commit()
                return int(user_cursor.lastrowid), int(assistant_cursor.lastrowid)
            except Exception:
                self._conn.rollback()
                raise

    def add_event(
        self,
        request_id: str,
        user_id: str,
        device_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        role: str | None = None,
        content: str | None = None,
        created_at: str | None = None,
        session_id: str | None = None,
    ) -> int:
        event_payload = dict(payload or {})
        if session_id:
            event_payload.setdefault("session_id", session_id)
        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO events
                (request_id,user_id,device_id,session_id,event_type,role,content,payload_json,created_at)
                VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    request_id,
                    user_id,
                    device_id,
                    session_id,
                    event_type,
                    role,
                    content,
                    self._json(event_payload),
                    created_at or iso_now(),
                ),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def list_events(
        self,
        user_id: str | None = None,
        device_id: str | None = None,
        session_id: str | None = None,
        role: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
        ascending: bool = False,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if user_id:
            clauses.append("user_id=?")
            params.append(user_id)
        if device_id:
            clauses.append("device_id=?")
            params.append(device_id)
        if session_id:
            clauses.append("session_id=?")
            params.append(session_id)
        if role:
            clauses.append("role=?")
            params.append(role)
        if event_type:
            clauses.append("event_type=?")
            params.append(event_type)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        order = "ASC" if ascending else "DESC"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM events{where} ORDER BY id {order} LIMIT ?", params
            ).fetchall()
        return [self._decode_row(row, ("payload_json",)) for row in rows]

    def get_event(self, event_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return self._decode_row(row, ("payload_json",)) if row else None

    def list_events_for_local_date(
        self,
        user_id: str,
        device_id: str,
        memory_date: str,
        *,
        event_type: str | None = None,
        session_id: str | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        rows = self.list_events(
            user_id=user_id,
            device_id=device_id,
            session_id=session_id,
            event_type=event_type,
            limit=limit,
            ascending=True,
        )
        return [row for row in rows if self._local_date(row.get("created_at")) == memory_date]

    def count_user_messages_since(self, user_id: str, after_event_id: int | None) -> int:
        clause = "user_id=? AND role='user'"
        params: list[Any] = [user_id]
        if after_event_id:
            clause += " AND id>?"
            params.append(after_event_id)
        with self._lock:
            return int(self._conn.execute(f"SELECT COUNT(*) FROM events WHERE {clause}", params).fetchone()[0])

    def latest_user_event_id(self, user_id: str, device_id: str | None = None) -> int | None:
        where = "user_id=? AND role='user'"
        params: list[Any] = [user_id]
        if device_id:
            where += " AND device_id=?"
            params.append(device_id)
        with self._lock:
            row = self._conn.execute(
                f"SELECT MAX(id) AS id FROM events WHERE {where}", params
            ).fetchone()
        return int(row["id"]) if row and row["id"] is not None else None

    def latest_preference_extraction_event_id(self, user_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                """SELECT MAX(COALESCE(to_event_id,0)) AS event_id
                FROM memory_jobs
                WHERE user_id=? AND job_type='preference_extraction' AND status='succeeded'""",
                (user_id,),
            ).fetchone()
        return int(row["event_id"] or 0) if row else 0

    def message_range(
        self,
        user_id: str,
        device_id: str,
        after_event_id: int,
        limit: int,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        session_clause = " AND session_id=?" if session_id else ""
        params: list[Any] = [user_id, device_id, after_event_id]
        if session_id:
            params.append(session_id)
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT * FROM events
                WHERE user_id=? AND device_id=? AND event_type='message' AND id>?{session_clause}
                ORDER BY id ASC LIMIT ?""",
                params,
            ).fetchall()
        return [self._event_message(row) for row in rows]

    def conversation_turns_after(
        self,
        user_id: str,
        device_id: str,
        after_event_id: int,
        limit: int,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        session_clause = "AND u.session_id=?" if session_id else ""
        params: list[Any] = [user_id, device_id, after_event_id]
        if session_id:
            params.append(session_id)
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT
                    u.id AS user_id_event,
                    u.request_id AS request_id,
                    u.session_id AS session_id,
                    u.content AS user_content,
                    u.created_at AS user_created_at,
                    a.id AS assistant_id_event,
                    a.content AS assistant_content,
                    a.created_at AS assistant_created_at
                FROM events u
                JOIN events a
                    ON a.request_id=u.request_id
                    AND a.role='assistant'
                    AND a.event_type='message'
                WHERE u.user_id=? AND u.device_id=?
                    AND u.event_type='message'
                    AND u.role='user'
                    AND u.id>?
                    {session_clause}
                ORDER BY u.id ASC
                LIMIT ?""",
                params,
            ).fetchall()
        turns = []
        for row in rows:
            turns.append(
                {
                    "request_id": row["request_id"],
                    "session_id": row["session_id"],
                    "user": {
                        "id": int(row["user_id_event"]),
                        "role": "user",
                        "content": row["user_content"],
                        "timestamp": row["user_created_at"],
                    },
                    "assistant": {
                        "id": int(row["assistant_id_event"]),
                        "role": "assistant",
                        "content": row["assistant_content"],
                        "timestamp": row["assistant_created_at"],
                    },
                }
            )
        return turns

    def conversation_turns_until(
        self,
        user_id: str,
        device_id: str,
        through_event_id: int,
        limit: int,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        session_clause = "AND u.session_id=?" if session_id else ""
        params: list[Any] = [user_id, device_id, through_event_id]
        if session_id:
            params.append(session_id)
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT * FROM (
                    SELECT
                        u.id AS user_id_event,
                        u.request_id AS request_id,
                        u.session_id AS session_id,
                        u.content AS user_content,
                        u.created_at AS user_created_at,
                        a.id AS assistant_id_event,
                        a.content AS assistant_content,
                        a.created_at AS assistant_created_at
                    FROM events u
                    JOIN events a
                        ON a.request_id=u.request_id
                        AND a.role='assistant'
                        AND a.event_type='message'
                    WHERE u.user_id=? AND u.device_id=?
                        AND u.event_type='message'
                        AND u.role='user'
                        AND a.id<=?
                        {session_clause}
                    ORDER BY u.id DESC
                    LIMIT ?
                ) ORDER BY user_id_event ASC""",
                params,
            ).fetchall()
        turns = []
        for row in rows:
            turns.append(
                {
                    "request_id": row["request_id"],
                    "session_id": row["session_id"],
                    "user": {
                        "id": int(row["user_id_event"]),
                        "role": "user",
                        "content": row["user_content"],
                        "timestamp": row["user_created_at"],
                    },
                    "assistant": {
                        "id": int(row["assistant_id_event"]),
                        "role": "assistant",
                        "content": row["assistant_content"],
                        "timestamp": row["assistant_created_at"],
                    },
                }
            )
        return turns

    def latest_conversation_turns(
        self,
        user_id: str,
        device_id: str,
        limit: int,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        latest_id = self.latest_user_event_id(user_id, device_id) or 0
        if not latest_id:
            return []
        return self.conversation_turns_until(user_id, device_id, latest_id + 1_000_000_000, limit, session_id=session_id)

    def latest_summary(self, user_id: str, device_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """SELECT * FROM conversation_summaries
                WHERE user_id=? AND device_id=? ORDER BY version DESC,id DESC LIMIT 1""",
                (user_id, device_id),
            ).fetchone()
        return dict(row) if row else None

    def add_summary(
        self,
        user_id: str,
        device_id: str,
        summary_text: str,
        compacted_through_event_id: int,
        version: int,
        from_event_id: int | None = None,
        to_event_id: int | None = None,
        turn_count: int = 0,
    ) -> int:
        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO conversation_summaries
                (user_id,device_id,summary_text,compacted_through_event_id,version,
                 from_event_id,to_event_id,turn_count,created_at)
                VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    user_id,
                    device_id,
                    summary_text,
                    compacted_through_event_id,
                    version,
                    from_event_id,
                    to_event_id,
                    turn_count,
                    iso_now(),
                ),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def list_summaries(self, user_id: str, device_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        params: list[Any] = [user_id]
        where = "WHERE user_id=?"
        if device_id:
            where += " AND device_id=?"
            params.append(device_id)
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM conversation_summaries {where} ORDER BY id DESC LIMIT ?", params
            ).fetchall()
        return [dict(row) for row in rows]

    def list_time_memories(
        self,
        user_id: str,
        device_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        params: list[Any] = [user_id]
        where = "WHERE user_id=? AND event_type='time_memory'"
        if device_id:
            where += " AND device_id=?"
            params.append(device_id)
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM events {where} ORDER BY id DESC LIMIT ?", params
            ).fetchall()
        return [self._decode_row(row, ("payload_json",)) for row in rows]

    def add_event_summary(
        self,
        user_id: str,
        device_id: str,
        content: str,
        event_at: str | None = None,
        title: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        payload = {
            "event_at": event_at or iso_now(),
            "title": title,
            "source": (metadata or {}).get("source", "manual"),
            "metadata": dict(metadata or {}),
        }
        return self.add_event(
            f"event-summary-{user_id}-{device_id}-{iso_now()}",
            user_id,
            device_id,
            "event_summary",
            payload,
            content=content,
        )

    def list_event_summaries(
        self,
        user_id: str | None = None,
        device_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self.list_events(user_id=user_id, device_id=device_id, event_type="event_summary", limit=limit)

    def upsert_daily_time_memory(
        self,
        user_id: str,
        device_id: str,
        memory_date: str,
        content: str,
        *,
        source_event_ids: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        generated_at = iso_now()
        payload = {
            "memory_date": memory_date,
            "memory_at": generated_at,
            "title": f"{memory_date} 日期总结",
            "source": "daily_session_extraction",
            "source_event_ids": list(source_event_ids or []),
            "generated_at": generated_at,
            "metadata": dict(metadata or {}),
        }
        with self._lock:
            self._conn.execute(
                """DELETE FROM events
                WHERE user_id=? AND device_id=? AND event_type='time_memory'
                    AND json_extract(payload_json,'$.memory_date')=?""",
                (user_id, device_id, memory_date),
            )
            cursor = self._conn.execute(
                """INSERT INTO events
                (request_id,user_id,device_id,session_id,event_type,role,content,payload_json,created_at)
                VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    f"daily-time-memory-{user_id}-{device_id}-{memory_date}",
                    user_id,
                    device_id,
                    None,
                    "time_memory",
                    None,
                    content,
                    json.dumps(payload, ensure_ascii=False),
                    generated_at,
                ),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def replace_daily_action_memories(
        self,
        user_id: str,
        device_id: str,
        memory_date: str,
        memories: list[dict[str, Any]],
    ) -> list[int]:
        generated_at = iso_now()
        with self._lock:
            self._conn.execute(
                """DELETE FROM events
                WHERE user_id=? AND device_id=? AND event_type='action_memory'
                    AND json_extract(payload_json,'$.memory_date')=?""",
                (user_id, device_id, memory_date),
            )
            event_ids: list[int] = []
            for idx, memory in enumerate(memories, start=1):
                actions = memory.get("actions") if isinstance(memory.get("actions"), list) else []
                content = str(memory.get("content") or memory.get("action_text") or "").strip()
                if not content:
                    labels = [
                        str(action.get("label_zh") or action.get("code") or "").strip()
                        for action in actions
                        if isinstance(action, dict)
                    ]
                    content = " -> ".join(label for label in labels if label)
                if not content:
                    continue
                session_id = memory.get("session_id")
                payload = {
                    "memory_date": memory_date,
                    "event_at": memory.get("event_at") or generated_at,
                    "title": memory.get("title") or f"{memory_date} 动作记忆 #{idx}",
                    "source": memory.get("source") or "daily_action_memory_extraction",
                    "session_id": session_id,
                    "actions": actions,
                    "source_event_ids": list(memory.get("source_event_ids") or []),
                    "source_message_event_ids": list(memory.get("source_message_event_ids") or []),
                    "confidence": float(memory.get("confidence") or 0.8),
                    "generated_at": generated_at,
                    "metadata": dict(memory.get("metadata") or {}),
                }
                cursor = self._conn.execute(
                    """INSERT INTO events
                    (request_id,user_id,device_id,session_id,event_type,role,content,payload_json,created_at)
                    VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        f"action-memory-{user_id}-{device_id}-{memory_date}-{idx}",
                        user_id,
                        device_id,
                        str(session_id) if session_id else None,
                        "action_memory",
                        None,
                        content,
                        json.dumps(payload, ensure_ascii=False),
                        generated_at,
                    ),
                )
                event_ids.append(int(cursor.lastrowid))
            self._conn.commit()
            return event_ids

    def replace_daily_event_memories(
        self,
        user_id: str,
        device_id: str,
        memory_date: str,
        memories: list[dict[str, Any]],
    ) -> list[int]:
        generated_at = iso_now()
        with self._lock:
            self._conn.execute(
                """DELETE FROM events
                WHERE user_id=? AND device_id=? AND event_type='event_memory'
                    AND json_extract(payload_json,'$.memory_date')=?""",
                (user_id, device_id, memory_date),
            )
            event_ids: list[int] = []
            for idx, memory in enumerate(memories, start=1):
                content = str(memory.get("content") or "").strip()
                if not content:
                    continue
                payload = {
                    "memory_date": memory_date,
                    "event_at": memory.get("event_at") or generated_at,
                    "title": memory.get("title") or f"{memory_date} 事件记忆 #{idx}",
                    "source": memory.get("source") or "daily_time_memory_event_extraction",
                    "source_event_ids": list(memory.get("source_event_ids") or []),
                    "source_time_memory_id": memory.get("source_time_memory_id"),
                    "event_key": memory.get("event_key"),
                    "generated_at": generated_at,
                    "metadata": dict(memory.get("metadata") or {}),
                }
                cursor = self._conn.execute(
                    """INSERT INTO events
                    (request_id,user_id,device_id,session_id,event_type,role,content,payload_json,created_at)
                    VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        f"event-memory-{user_id}-{device_id}-{memory_date}-{idx}",
                        user_id,
                        device_id,
                        None,
                        "event_memory",
                        None,
                        content,
                        json.dumps(payload, ensure_ascii=False),
                        generated_at,
                    ),
                )
                event_ids.append(int(cursor.lastrowid))
            self._conn.commit()
            return event_ids

    def replace_weekly_action_preference_memories(
        self,
        user_id: str,
        device_id: str,
        start_date: str,
        end_date: str,
        memories: list[dict[str, Any]],
    ) -> list[int]:
        generated_at = iso_now()
        with self._lock:
            self._conn.execute(
                """DELETE FROM events
                WHERE user_id=? AND device_id=? AND event_type='action_preference_memory'
                    AND json_extract(payload_json,'$.start_date')=?
                    AND json_extract(payload_json,'$.end_date')=?""",
                (user_id, device_id, start_date, end_date),
            )
            event_ids: list[int] = []
            for idx, memory in enumerate(memories, start=1):
                content = str(memory.get("content") or "").strip()
                if not content:
                    continue
                payload = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "memory_date": end_date,
                    "title": memory.get("title") or f"{start_date} 至 {end_date} 动作偏好记忆 #{idx}",
                    "source": "weekly_action_preference_extraction",
                    "source_event_ids": list(memory.get("source_event_ids") or []),
                    "generated_at": generated_at,
                    "confidence": float(memory.get("confidence") or 0.8),
                    "metadata": dict(memory.get("metadata") or {}),
                }
                cursor = self._conn.execute(
                    """INSERT INTO events
                    (request_id,user_id,device_id,session_id,event_type,role,content,payload_json,created_at)
                    VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        f"action-preference-memory-{user_id}-{device_id}-{start_date}-{end_date}-{idx}",
                        user_id,
                        device_id,
                        None,
                        "action_preference_memory",
                        None,
                        content,
                        json.dumps(payload, ensure_ascii=False),
                        generated_at,
                    ),
                )
                event_ids.append(int(cursor.lastrowid))
            self._conn.commit()
            return event_ids

    def replace_weekly_event_preference_memories(
        self,
        user_id: str,
        device_id: str,
        start_date: str,
        end_date: str,
        memories: list[dict[str, Any]],
    ) -> list[int]:
        generated_at = iso_now()
        with self._lock:
            self._conn.execute(
                """DELETE FROM events
                WHERE user_id=? AND device_id=? AND event_type='event_preference_memory'
                    AND json_extract(payload_json,'$.start_date')=?
                    AND json_extract(payload_json,'$.end_date')=?""",
                (user_id, device_id, start_date, end_date),
            )
            event_ids: list[int] = []
            for idx, memory in enumerate(memories, start=1):
                content = str(memory.get("content") or "").strip()
                if not content:
                    continue
                payload = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "memory_date": end_date,
                    "title": memory.get("title") or f"{start_date} 至 {end_date} 事件偏好记忆 #{idx}",
                    "source": "weekly_event_memory_repetition",
                    "source_event_ids": list(memory.get("source_event_ids") or []),
                    "event_key": memory.get("event_key"),
                    "occurrence_count": int(memory.get("occurrence_count") or 0),
                    "evidence_dates": list(memory.get("evidence_dates") or []),
                    "generated_at": generated_at,
                    "metadata": dict(memory.get("metadata") or {}),
                }
                cursor = self._conn.execute(
                    """INSERT INTO events
                    (request_id,user_id,device_id,session_id,event_type,role,content,payload_json,created_at)
                    VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        f"event-preference-memory-{user_id}-{device_id}-{start_date}-{end_date}-{idx}",
                        user_id,
                        device_id,
                        None,
                        "event_preference_memory",
                        None,
                        content,
                        json.dumps(payload, ensure_ascii=False),
                        generated_at,
                    ),
                )
                event_ids.append(int(cursor.lastrowid))
            self._conn.commit()
            return event_ids

    def list_action_memories(
        self,
        user_id: str | None = None,
        device_id: str | None = None,
        memory_date: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        rows = self.list_events(
            user_id=user_id,
            device_id=device_id,
            session_id=session_id,
            event_type="action_memory",
            limit=limit,
            ascending=False,
        )
        if memory_date:
            rows = [row for row in rows if (row.get("payload_json") or {}).get("memory_date") == memory_date]
        return rows

    def list_event_memories(
        self,
        user_id: str | None = None,
        device_id: str | None = None,
        memory_date: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        rows = self.list_events(
            user_id=user_id,
            device_id=device_id,
            event_type="event_memory",
            limit=limit,
            ascending=False,
        )
        if memory_date:
            rows = [row for row in rows if (row.get("payload_json") or {}).get("memory_date") == memory_date]
        return rows

    def list_action_preference_memories(
        self,
        user_id: str | None = None,
        device_id: str | None = None,
        end_date: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        rows = self.list_events(
            user_id=user_id,
            device_id=device_id,
            event_type="action_preference_memory",
            limit=limit,
            ascending=False,
        )
        if end_date:
            rows = [row for row in rows if (row.get("payload_json") or {}).get("end_date") == end_date]
        return rows

    def list_event_preference_memories(
        self,
        user_id: str | None = None,
        device_id: str | None = None,
        end_date: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        rows = self.list_events(
            user_id=user_id,
            device_id=device_id,
            event_type="event_preference_memory",
            limit=limit,
            ascending=False,
        )
        if end_date:
            rows = [row for row in rows if (row.get("payload_json") or {}).get("end_date") == end_date]
        return rows

    def upsert_action_chain_summary(
        self,
        user_id: str,
        device_id: str,
        memory_date: str,
        content: str,
        *,
        actions: list[dict[str, Any]],
        frequency: int,
        source_event_ids: list[int] | None = None,
        source_message_event_ids: list[int] | None = None,
    ) -> int:
        generated_at = iso_now()
        payload = {
            "memory_date": memory_date,
            "event_at": generated_at,
            "title": f"{memory_date} 频繁动作链路",
            "source": "daily_action_chain_extraction",
            "actions": list(actions),
            "frequency": int(frequency),
            "source_event_ids": list(source_event_ids or []),
            "source_message_event_ids": list(source_message_event_ids or []),
            "generated_at": generated_at,
        }
        with self._lock:
            self._conn.execute(
                """DELETE FROM events
                WHERE user_id=? AND device_id=? AND event_type='action_chain_summary'
                    AND json_extract(payload_json,'$.memory_date')=?""",
                (user_id, device_id, memory_date),
            )
            cursor = self._conn.execute(
                """INSERT INTO events
                (request_id,user_id,device_id,session_id,event_type,role,content,payload_json,created_at)
                VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    f"action-chain-summary-{user_id}-{device_id}-{memory_date}",
                    user_id,
                    device_id,
                    None,
                    "action_chain_summary",
                    None,
                    content,
                    json.dumps(payload, ensure_ascii=False),
                    generated_at,
                ),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def list_action_chain_summaries(
        self,
        user_id: str | None = None,
        device_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self.list_events(user_id=user_id, device_id=device_id, event_type="action_chain_summary", limit=limit)

    def list_action_events(
        self,
        user_id: str | None = None,
        device_id: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        params: list[Any] = []
        where = "WHERE event_type='action_sequence'"
        if user_id:
            where += " AND user_id=?"
            params.append(user_id)
        if device_id:
            where += " AND device_id=?"
            params.append(device_id)
        if session_id:
            where += " AND session_id=?"
            params.append(session_id)
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM events {where} ORDER BY id DESC LIMIT ?", params
            ).fetchall()
        return [self._decode_row(row, ("payload_json",)) for row in rows]

    def latest_action_sequence(self, user_id: str, device_id: str) -> dict[str, Any] | None:
        rows = self.list_action_events(user_id, device_id, limit=1)
        return rows[0] if rows else None

    def upsert_preference(
        self,
        user_id: str,
        preference_key: str,
        category: str,
        value: dict[str, Any],
        display_text_zh: str,
        evidence: list[dict[str, Any]],
        *,
        polarity: str = "prefer",
        durability: str = "persistent",
        strength: float = 0.5,
        confidence: float = 0.5,
        source_type: str = "explicit",
        expires_at: str | None = None,
        extractor_model: str | None = None,
        prompt_version: str | None = None,
        reason_zh: str | None = None,
        scope: str = "user",
        status: str | None = None,
    ) -> int | None:
        now = iso_now()
        value_type = str(value.get("type") or "json")
        value_json = self._json(value)
        target_status = status or ("active" if source_type == "explicit" and confidence >= 0.85 else "candidate")
        if confidence < 0.65:
            target_status = "rejected"
        if durability == "temporary" and not expires_at:
            target_status = "candidate"

        with self._lock:
            active, duplicate_ids = self._active_preference_for_value(
                user_id,
                preference_key,
                value,
                display_text_zh,
            )
            if active and (
                active["value_json"] == value_json
                or preference_key in {"preference.likes", "preference.dislikes"}
            ) and target_status == "active":
                pref_id = int(active["id"])
                self._revoke_opposite_value_preference(
                    user_id,
                    preference_key,
                    value,
                    display_text_zh,
                    now,
                    enabled=True,
                )
                if duplicate_ids:
                    self._supersede_duplicate_preferences(pref_id, duplicate_ids, now)
                self._conn.execute(
                    """UPDATE user_preferences
                    SET value_type=?, value_json=?, display_text_zh=?,
                        evidence_count=evidence_count+?, last_confirmed_at=?, confidence=?,
                        strength=?, extractor_model=COALESCE(?, extractor_model),
                        prompt_version=COALESCE(?, prompt_version),
                        reason_zh=COALESCE(?, reason_zh),
                        scope=?, updated_at=?
                    WHERE id=?""",
                    (
                        value_type,
                        value_json,
                        display_text_zh,
                        max(1, len(evidence)),
                        now,
                        confidence,
                        strength,
                        extractor_model,
                        prompt_version,
                        reason_zh,
                        scope,
                        now,
                        pref_id,
                    ),
                )
            else:
                supersedes_id = int(active["id"]) if active and target_status == "active" else None
                if supersedes_id:
                    self._conn.execute(
                        "UPDATE user_preferences SET status='superseded',updated_at=? WHERE id=?",
                        (now, supersedes_id),
                    )
                self._revoke_opposite_value_preference(
                    user_id,
                    preference_key,
                    value,
                    display_text_zh,
                    now,
                    enabled=target_status == "active",
                )
                cursor = self._conn.execute(
                    """INSERT INTO user_preferences
                    (user_id,preference_key,category,value_type,value_json,display_text_zh,
                    polarity,durability,strength,confidence,source_type,status,evidence_count,
                    first_seen_at,last_confirmed_at,expires_at,revision,supersedes_id,
                    extractor_model,prompt_version,reason_zh,scope,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        user_id,
                        preference_key,
                        category,
                        value_type,
                        value_json,
                        display_text_zh,
                        polarity,
                        durability,
                        strength,
                        confidence,
                        source_type,
                        target_status,
                        max(1, len(evidence)),
                        now,
                        now,
                        expires_at,
                        (int(active["revision"]) + 1) if active else 1,
                        supersedes_id,
                        extractor_model,
                        prompt_version,
                        reason_zh,
                        scope,
                        now,
                        now,
                    ),
                )
                pref_id = int(cursor.lastrowid)
            for item in evidence:
                event_id = int(item.get("event_id") or 0)
                if not event_id:
                    continue
                self._conn.execute(
                    """INSERT OR IGNORE INTO preference_evidence
                    (preference_id,event_id,evidence_type,evidence_text,confidence,created_at)
                    VALUES(?,?,?,?,?,?)""",
                    (
                        pref_id,
                        event_id,
                        str(item.get("type") or item.get("evidence_type") or "explicit"),
                        str(item.get("text") or item.get("evidence_text") or ""),
                        item.get("confidence", confidence),
                        now,
                    ),
                )
            self._conn.commit()
        return pref_id if target_status != "rejected" else None

    def revoke_preference(
        self,
        user_id: str,
        preference_key: str,
        value: dict[str, Any] | None = None,
    ) -> None:
        value_json = self._json(value) if value is not None else None
        if preference_key in {"preference.likes", "preference.dislikes"} and value is None:
            return
        with self._lock:
            now = iso_now()
            if value_json is None:
                self._conn.execute(
                    """UPDATE user_preferences SET status='revoked',updated_at=?
                    WHERE user_id=? AND preference_key=? AND status='active'""",
                    (now, user_id, preference_key),
                )
            else:
                if preference_key in {"preference.likes", "preference.dislikes"}:
                    target_key = self.normalized_preference_value_key(preference_key, value)
                    rows = self._active_preference_rows(user_id, preference_key)
                    for row in rows:
                        if self.normalized_preference_value_key(
                            preference_key,
                            self._loads(row["value_json"], {}),
                            str(row["display_text_zh"] or ""),
                        ) == target_key:
                            self._conn.execute(
                                "UPDATE user_preferences SET status='revoked',updated_at=? WHERE id=?",
                                (now, int(row["id"])),
                            )
                else:
                    self._conn.execute(
                        """UPDATE user_preferences SET status='revoked',updated_at=?
                        WHERE user_id=? AND preference_key=? AND value_json=? AND status='active'""",
                        (now, user_id, preference_key, value_json),
                    )
            self._conn.commit()

    def _active_preference_for_value(
        self,
        user_id: str,
        preference_key: str,
        value: dict[str, Any],
        display_text_zh: str = "",
    ) -> tuple[sqlite3.Row | None, list[int]]:
        if preference_key in {"preference.likes", "preference.dislikes"}:
            target_key = self.normalized_preference_value_key(preference_key, value, display_text_zh)
            matches = [
                row
                for row in self._active_preference_rows(user_id, preference_key)
                if self.normalized_preference_value_key(
                    preference_key,
                    self._loads(row["value_json"], {}),
                    str(row["display_text_zh"] or ""),
                )
                == target_key
            ]
            if not matches:
                return None, []
            return matches[0], [int(row["id"]) for row in matches[1:]]
        row = self._conn.execute(
            """SELECT * FROM user_preferences
            WHERE user_id=? AND preference_key=? AND status='active'""",
            (user_id, preference_key),
        ).fetchone()
        return row, []

    def _revoke_opposite_value_preference(
        self,
        user_id: str,
        preference_key: str,
        value: dict[str, Any],
        display_text_zh: str,
        now: str,
        *,
        enabled: bool,
    ) -> None:
        if not enabled:
            return
        opposite = {
            "preference.likes": "preference.dislikes",
            "preference.dislikes": "preference.likes",
        }.get(preference_key)
        if not opposite:
            return
        target_key = self.normalized_preference_value_key(preference_key, value, display_text_zh)
        for row in self._active_preference_rows(user_id, opposite):
            if self.normalized_preference_value_key(
                opposite,
                self._loads(row["value_json"], {}),
                str(row["display_text_zh"] or ""),
            ) != target_key:
                continue
            self._conn.execute(
                "UPDATE user_preferences SET status='revoked',updated_at=? WHERE id=?",
                (now, int(row["id"])),
            )

    def _active_preference_rows(self, user_id: str, preference_key: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            """SELECT * FROM user_preferences
            WHERE user_id=? AND preference_key=? AND status='active'
            ORDER BY evidence_count DESC,id DESC""",
            (user_id, preference_key),
        ).fetchall()

    def _supersede_duplicate_preferences(self, pref_id: int, duplicate_ids: list[int], now: str) -> None:
        if not duplicate_ids:
            return
        placeholders = ",".join("?" for _ in duplicate_ids)
        duplicate_evidence = self._conn.execute(
            f"SELECT COALESCE(SUM(evidence_count),0) FROM user_preferences WHERE id IN ({placeholders})",
            duplicate_ids,
        ).fetchone()[0]
        self._conn.execute(
            f"UPDATE user_preferences SET status='superseded',updated_at=? WHERE id IN ({placeholders})",
            [now, *duplicate_ids],
        )
        self._conn.execute(
            "UPDATE user_preferences SET evidence_count=evidence_count+?,updated_at=? WHERE id=?",
            (int(duplicate_evidence or 0), now, pref_id),
        )

    @classmethod
    def normalized_preference_value_key(
        cls,
        preference_key: str,
        value: dict[str, Any],
        display_text_zh: str = "",
    ) -> str:
        if preference_key not in {"preference.likes", "preference.dislikes"}:
            return cls._json(value)
        candidates = [
            str(value.get("label_zh") or ""),
            display_text_zh,
            str(value.get("label") or ""),
            str(value.get("name") or ""),
            str(value.get("value") or ""),
            str(value.get("code") or ""),
        ]
        for candidate in candidates:
            normalized = cls._normalize_preference_object_text(candidate)
            if normalized:
                return normalized
        return cls._normalize_preference_object_text(cls._json(value))

    @staticmethod
    def _normalize_preference_object_text(text: str) -> str:
        text = str(text or "").strip().lower()
        if not text:
            return ""
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"^(用户)?(明确)?(偏好|喜欢|喜爱|爱吃|爱|不喜欢|讨厌|不爱吃|不吃)", "", text)
        text = re.sub(r"^(吃|喝|看|听|去|玩)", "", text)
        text = re.sub(r"(了|的|呢|啊|呀|吧)$", "", text)
        return text

    def list_preferences(
        self,
        user_id: str,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        params: list[Any] = [user_id]
        where = "WHERE user_id=?"
        if status:
            where += " AND status=?"
            params.append(status)
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM user_preferences {where} ORDER BY updated_at DESC,id DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._decode_row(row, ("value_json",)) for row in rows]

    def list_preference_evidence(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT e.*, p.preference_key, p.display_text_zh, ev.content AS source_content
                FROM preference_evidence e
                JOIN user_preferences p ON p.id=e.preference_id
                JOIN events ev ON ev.id=e.event_id
                WHERE p.user_id=?
                ORDER BY e.id DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def enqueue_job(
        self,
        user_id: str,
        job_type: str,
        device_id: str | None = None,
        from_event_id: int | None = None,
        to_event_id: int | None = None,
        available_at: str | None = None,
    ) -> int | None:
        now = iso_now()
        with self._lock:
            try:
                cursor = self._conn.execute(
                    """INSERT INTO memory_jobs
                    (user_id,device_id,job_type,from_event_id,to_event_id,status,available_at,created_at,updated_at)
                    VALUES(?,?,?,?,?,'pending',?,?,?)""",
                    (user_id, device_id, job_type, from_event_id, to_event_id, available_at or now, now, now),
                )
                self._conn.commit()
                return int(cursor.lastrowid)
            except sqlite3.IntegrityError:
                self._conn.rollback()
                if job_type == "preference_extraction":
                    self._conn.execute(
                        """UPDATE memory_jobs SET to_event_id=MAX(COALESCE(to_event_id,0),?),
                        updated_at=? WHERE user_id=? AND job_type=? AND status IN ('pending','running')""",
                        (to_event_id or 0, now, user_id, job_type),
                    )
                    self._conn.commit()
                    return None
                raise

    def upsert_pending_device_job(
        self,
        user_id: str,
        device_id: str,
        job_type: str,
        from_event_id: int | None = None,
        to_event_id: int | None = None,
        available_at: str | None = None,
    ) -> int | None:
        now = iso_now()
        with self._lock:
            row = self._conn.execute(
                """SELECT id,from_event_id,to_event_id FROM memory_jobs
                WHERE user_id=? AND device_id=? AND job_type=? AND status IN ('pending','running')
                ORDER BY id DESC LIMIT 1""",
                (user_id, device_id, job_type),
            ).fetchone()
            if row:
                job_id = int(row["id"])
                prior_from = int(row["from_event_id"] or from_event_id or 0)
                next_from = min(prior_from, int(from_event_id or prior_from or 0)) if prior_from else from_event_id
                next_to = max(int(row["to_event_id"] or 0), int(to_event_id or 0)) or to_event_id
                self._conn.execute(
                    """UPDATE memory_jobs
                    SET from_event_id=?, to_event_id=?, available_at=?, updated_at=?
                    WHERE id=?""",
                    (next_from, next_to, available_at or now, now, job_id),
                )
                self._conn.commit()
                return None
            return self.enqueue_job(
                user_id,
                job_type,
                device_id=device_id,
                from_event_id=from_event_id,
                to_event_id=to_event_id,
                available_at=available_at or now,
            )

    def restart_preference_extraction_job(
        self,
        user_id: str,
        device_id: str | None,
        from_event_id: int,
        to_event_id: int,
    ) -> int | None:
        now = iso_now()
        with self._lock:
            row = self._conn.execute(
                """SELECT id FROM memory_jobs
                WHERE user_id=? AND job_type='preference_extraction'
                    AND status IN ('pending','running','failed')
                ORDER BY id DESC LIMIT 1""",
                (user_id,),
            ).fetchone()
            if row:
                job_id = int(row["id"])
                self._conn.execute(
                    """UPDATE memory_jobs
                    SET device_id=?, from_event_id=?, to_event_id=?, status='pending',
                        attempts=0, available_at=?, locked_at=NULL,
                        last_error='manual preference extraction restart',
                        updated_at=?
                    WHERE id=?""",
                    (device_id, from_event_id, to_event_id, now, now, job_id),
                )
                self._conn.commit()
                return job_id
            return self.enqueue_job(
                user_id,
                "preference_extraction",
                device_id=device_id,
                from_event_id=from_event_id,
                to_event_id=to_event_id,
                available_at=now,
            )

    def recover_stale_running_jobs(self, stale_after_seconds: int = 900) -> int:
        cutoff = (utc_now() - timedelta(seconds=max(1, stale_after_seconds))).isoformat()
        now = iso_now()
        with self._lock:
            cursor = self._conn.execute(
                """UPDATE memory_jobs
                SET status='failed', locked_at=NULL, available_at=?, updated_at=?,
                    last_error=COALESCE(last_error, 'job recovered after stale running lock')
                WHERE status='running' AND locked_at IS NOT NULL AND locked_at<?""",
                (now, now, cutoff),
            )
            self._conn.commit()
            return int(cursor.rowcount)

    def claim_jobs(
        self,
        limit: int,
        max_attempts: int | None = None,
        job_types: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        now = iso_now()
        params: list[Any] = [now]
        attempts_clause = ""
        if max_attempts is not None:
            attempts_clause = " AND attempts<?"
            params.append(max(1, max_attempts))
        effective_job_types = job_types or {"conversation_summary", "preference_extraction", "user_card_rebuild"}
        placeholders = ",".join("?" for _ in effective_job_types)
        job_type_clause = f" AND job_type IN ({placeholders})"
        params.extend(sorted(effective_job_types))
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM memory_jobs
                WHERE id IN (
                    SELECT MIN(id) FROM memory_jobs
                    WHERE status IN ('pending','failed') AND available_at<=?
                    """ + attempts_clause + """
                    """ + job_type_clause + """
                    AND NOT EXISTS (
                        SELECT 1 FROM memory_jobs active
                        WHERE active.user_id=memory_jobs.user_id
                            AND active.job_type=memory_jobs.job_type
                            AND active.status IN ('pending','running')
                            AND active.id != memory_jobs.id
                    )
                    GROUP BY user_id, job_type
                )
                ORDER BY id LIMIT ?""",
                params,
            ).fetchall()
            ids = [int(row["id"]) for row in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                self._conn.execute(
                    f"UPDATE memory_jobs SET status='running',locked_at=?,updated_at=? WHERE id IN ({placeholders})",
                    [now, now, *ids],
                )
                self._conn.commit()
        return [dict(row) for row in rows]

    def mark_job_succeeded(self, job_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE memory_jobs SET status='succeeded',updated_at=?,last_error=NULL WHERE id=?",
                (iso_now(), job_id),
            )
            self._conn.commit()

    def mark_job_failed(self, job_id: int, error: str, max_attempts: int) -> int:
        now = iso_now()
        with self._lock:
            row = self._conn.execute("SELECT attempts FROM memory_jobs WHERE id=?", (job_id,)).fetchone()
            attempts = int(row["attempts"] or 0) + 1 if row else 1
            self._conn.execute(
                """UPDATE memory_jobs SET status='failed',attempts=?,last_error=?,available_at=?,
                locked_at=NULL,updated_at=? WHERE id=?""",
                (attempts, error[:2000], now, now, job_id),
            )
            self._conn.commit()
            return attempts

    def job_counts(self, include_daily: bool = False) -> dict[str, int]:
        where = ""
        params: list[Any] = []
        if not include_daily:
            where = "WHERE job_type NOT IN (?,?,?,?)"
            params.extend([
                "daily_time_memory_extract",
                "action_chain_extract",
                "daily_action_memory_extract",
                "weekly_action_preference_extract",
            ])
        with self._lock:
            rows = self._conn.execute(
                f"SELECT status,COUNT(*) AS count FROM memory_jobs {where} GROUP BY status",
                params,
            ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def list_jobs(
        self,
        user_id: str | None = None,
        job_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        params: list[Any] = []
        where = ""
        clauses: list[str] = []
        if user_id:
            clauses.append("user_id=?")
            params.append(user_id)
        if job_type:
            clauses.append("job_type=?")
            params.append(job_type)
        if clauses:
            where = "WHERE " + " AND ".join(clauses)
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT * FROM memory_jobs {where}
                ORDER BY id DESC LIMIT ?""",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_user_memory(self, user_id: str) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for table in ("preference_evidence",):
                cursor = self._conn.execute(
                    f"""DELETE FROM {table} WHERE preference_id IN
                    (SELECT id FROM user_preferences WHERE user_id=?)""",
                    (user_id,),
                )
                counts[table] = cursor.rowcount
            for table in ("user_preferences", "memory_jobs", "conversation_summaries"):
                cursor = self._conn.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
                counts[table] = cursor.rowcount
            cursor = self._conn.execute(
                """DELETE FROM events
                WHERE user_id=? AND event_type='time_memory'""",
                (user_id,),
            )
            counts["time_memories"] = cursor.rowcount
            cursor = self._conn.execute(
                """DELETE FROM events
                WHERE user_id=? AND event_type='event_summary'""",
                (user_id,),
            )
            counts["event_summaries"] = cursor.rowcount
            cursor = self._conn.execute(
                """DELETE FROM events
                WHERE user_id=? AND event_type='action_chain_summary'""",
                (user_id,),
            )
            counts["action_chain_summaries"] = cursor.rowcount
            cursor = self._conn.execute(
                """DELETE FROM events
                WHERE user_id=? AND event_type='action_memory'""",
                (user_id,),
            )
            counts["action_memories"] = cursor.rowcount
            cursor = self._conn.execute("DELETE FROM conversation_sessions WHERE user_id=?", (user_id,))
            counts["conversation_sessions"] = cursor.rowcount
            cursor = self._conn.execute(
                """DELETE FROM events
                WHERE user_id=? AND event_type IN (
                    'scheduled_task',
                    'recurring_task',
                    'conditional_task',
                    'pending_event'
                )""",
                (user_id,),
            )
            counts["legacy_time_events"] = cursor.rowcount
            self._conn.commit()
        return counts

    def search_user_text(self, user_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        query_tokens = tokenize(query)
        scored: list[tuple[int, dict[str, Any]]] = []
        for event in self.list_events(user_id=user_id, event_type="message", limit=1000, ascending=True):
            score = len(query_tokens & tokenize(str(event.get("content", ""))))
            if score:
                scored.append((score, event))
        scored.sort(key=lambda item: (item[0], item[1]["id"]), reverse=True)
        return [{**event, "score": score} for score, event in scored[:limit]]

    def begin_tool_run(
        self,
        context_id: str,
        tool_name: str,
        input_data: Any,
        run_id: str,
        idempotency_key: str | None,
        started_at: str,
    ) -> tuple[str, bool]:
        with self._lock:
            if idempotency_key:
                existing = self._conn.execute(
                    "SELECT run_id FROM tool_runs WHERE idempotency_key=?", (idempotency_key,)
                ).fetchone()
                if existing:
                    return str(existing["run_id"]), False
            self._conn.execute(
                """INSERT INTO tool_runs
                (run_id,context_id,tool_name,input,status,idempotency_key,started_at)
                VALUES(?,?,?,?,?,?,?)""",
                (run_id, context_id, tool_name, self._json(input_data), "running", idempotency_key, started_at),
            )
            self._conn.commit()
        return run_id, True

    def add_tool_step(self, run_id: str, step_name: str, status: str, payload: Any) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO tool_steps(run_id,step_name,status,payload,created_at) VALUES(?,?,?,?,?)",
                (run_id, step_name, status, self._json(payload), iso_now()),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def finish_tool_run(self, run_id: str, status: str, output: Any, error: str | None) -> None:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE tool_runs SET status=?,output=?,error=?,finished_at=? WHERE run_id=?",
                (status, self._json(output), error, iso_now(), run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown tool run: {run_id}")
            self._conn.commit()

    def get_tool_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM tool_runs WHERE run_id=?", (run_id,)).fetchone()
            steps = self._conn.execute(
                "SELECT * FROM tool_steps WHERE run_id=? ORDER BY id", (run_id,)
            ).fetchall()
        if not row:
            return None
        result = self._decode_row(row, ("input", "output"))
        result["steps"] = [self._decode_row(step, ("payload",)) for step in steps]
        return result

    def add_device_state(self, device_id: str, state: dict[str, Any], reason: str, observed_at: str) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO device_state_history(device_id,state,reason,observed_at) VALUES(?,?,?,?)",
                (device_id, self._json(state), reason, observed_at),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def latest_device_history(self, device_id: str) -> dict[str, Any] | None:
        rows = self.get_device_history(device_id, limit=1)
        return rows[0] if rows else None

    def get_device_history(self, device_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM device_state_history WHERE device_id=?
                ORDER BY observed_at DESC,id DESC LIMIT ?""",
                (device_id, limit),
            ).fetchall()
        return [self._decode_row(row, ("state",)) for row in rows]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            users = self._conn.execute(
                "SELECT COUNT(DISTINCT user_id) FROM events WHERE user_id<>?", (LEGACY_USER_ID,)
            ).fetchone()[0]
            active = self._conn.execute(
                "SELECT COUNT(*) FROM user_preferences WHERE status='active'"
            ).fetchone()[0]
        return {"user_count": int(users), "active_preference_count": int(active)}

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _table_has_column(self, table: str, column: str) -> bool:
        row = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not row:
            return False
        return any(info[1] == column for info in self._conn.execute(f"PRAGMA table_info({table})"))

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        if not self._table_has_column(table, column):
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
            self._conn.commit()

    def _ensure_memory_jobs_accepts_extraction_types(self) -> None:
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='memory_jobs'"
        ).fetchone()
        sql = str(row["sql"] if row else "")
        if "job_type IN" not in sql:
            return
        with self._lock:
            self._conn.execute("ALTER TABLE memory_jobs RENAME TO memory_jobs_old")
            self._conn.execute(
                """
                CREATE TABLE memory_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    device_id TEXT,
                    job_type TEXT NOT NULL,
                    from_event_id INTEGER,
                    to_event_id INTEGER,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    available_at TEXT NOT NULL,
                    locked_at TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    CHECK (status IN ('pending','running','succeeded','failed'))
                )
                """
            )
            self._conn.execute(
                """INSERT INTO memory_jobs
                (id,user_id,device_id,job_type,from_event_id,to_event_id,status,attempts,
                 available_at,locked_at,last_error,created_at,updated_at)
                SELECT id,user_id,device_id,job_type,from_event_id,to_event_id,status,attempts,
                    available_at,locked_at,last_error,created_at,updated_at
                FROM memory_jobs_old"""
            )
            self._conn.execute("DROP TABLE memory_jobs_old")
            self._conn.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS uq_pending_preference_job
                ON memory_jobs(user_id, job_type)
                WHERE job_type='preference_extraction' AND status IN ('pending','running')"""
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_ready ON memory_jobs(status, available_at, id)"
            )
            self._conn.commit()

    @staticmethod
    def _local_date(value: Any, timezone_name: str = "Asia/Shanghai") -> str:
        if not value:
            return ""
        text = str(value)
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return text[:10]
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
        return parsed.astimezone(ZoneInfo(timezone_name)).date().isoformat()

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)

    @staticmethod
    def _loads(value: str | None, default: Any) -> Any:
        if not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    @classmethod
    def _decode_row(cls, row: sqlite3.Row, json_fields: tuple[str, ...]) -> dict[str, Any]:
        value = dict(row)
        for field in json_fields:
            if value.get(field) is not None:
                value[field] = cls._loads(value[field], value[field])
        return value

    @staticmethod
    def _event_message(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "role": row["role"],
            "content": row["content"],
            "timestamp": row["created_at"],
            "session_id": row["session_id"],
        }


def new_request_id() -> str:
    return uuid.uuid4().hex
