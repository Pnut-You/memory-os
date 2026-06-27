"""Migration helpers for legacy Memory OS stores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .config import MemoryConfig
from .sqlite_event import LEGACY_USER_ID, SQLiteEventStore
from .utils import iso_now


def migrate_jsonl(store: SQLiteEventStore, jsonl_path: str | Path) -> dict[str, int]:
    path = Path(jsonl_path)
    result = {"read": 0, "imported": 0, "skipped": 0, "invalid": 0}
    if not path.exists():
        return result
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        result["read"] += 1
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            result["invalid"] += 1
            continue
        digest = record.get("id") or hashlib.sha256(raw_line.encode("utf-8")).hexdigest()
        request_id = f"legacy-jsonl-{digest}"
        existing = store.list_events(user_id=LEGACY_USER_ID, event_type="legacy_jsonl", limit=10)
        if any(item["request_id"] == request_id for item in existing):
            result["skipped"] += 1
            continue
        store.add_event(
            request_id,
            LEGACY_USER_ID,
            str(record.get("device_id") or record.get("session_id") or "legacy-device"),
            "legacy_jsonl",
            {
                "legacy_type": record.get("type", "legacy"),
                "metadata": record.get("metadata") if isinstance(record.get("metadata"), dict) else {},
            },
            content=json.dumps(record.get("content", ""), ensure_ascii=False, default=str),
            created_at=str(record.get("timestamp") or iso_now()),
        )
        result["imported"] += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", help="Legacy JSONL path; defaults to MEMORY_LOCAL_LONG_TERM_PATH")
    parser.add_argument("--sqlite", help="SQLite path; defaults to MEMORY_SQLITE_PATH")
    args = parser.parse_args()
    config = MemoryConfig.from_env()
    store = SQLiteEventStore(args.sqlite or config.sqlite_path)
    try:
        result = migrate_jsonl(store, args.jsonl or config.local_long_term_path)
        print(
            json.dumps(
                {
                    "sqlite_migration_log": store.migration_log,
                    "jsonl": result,
                    "legacy_user_id": LEGACY_USER_ID,
                },
                ensure_ascii=False,
            )
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
