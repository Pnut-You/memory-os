"""Shared helpers for the memory package."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_dotenv(path: str | Path = ".env") -> None:
    """Load a small, dependency-free subset of dotenv syntax."""
    file_path = Path(path)
    if not file_path.exists():
        return
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key:
            os.environ.setdefault(key, value)


def tokenize(text: str) -> set[str]:
    """Simple tokenizer that works for both whitespace text and CJK queries."""
    lowered = text.lower()
    words = {part for part in lowered.replace("\n", " ").split(" ") if part}
    chars = {char for char in lowered if "\u4e00" <= char <= "\u9fff"}
    return words | chars
