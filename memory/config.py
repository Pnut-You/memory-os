"""Environment-driven configuration for the plug-and-play memory system."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .utils import load_dotenv


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class MemoryConfig:
    data_dir: Path = Path("data")
    sqlite_path: Path = Path("data/events.db")
    local_long_term_path: Path = Path("data/long_term.jsonl")
    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_seconds: int = 86400
    redis_allow_memory_fallback: bool = False
    redis_prefix: str = "memory-os"
    device_state_ttl_seconds: int = 120
    device_heartbeat_seconds: int = 300
    tool_run_ttl_seconds: int = 3600
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4.1-mini"
    summary_every_turns: int = 10
    summary_retain_turns: int = 5
    preference_extractor_enabled: bool = True
    preference_extractor_base_url: str = ""
    preference_extractor_api_key: str = ""
    preference_extractor_model: str = ""
    preference_extract_batch_size: int = 8
    preference_extract_min_new_user_messages: int = 10
    preference_extract_max_attempts: int = 3

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "MemoryConfig":
        load_dotenv(env_file)
        data_dir = Path(os.getenv("MEMORY_DATA_DIR", "data"))
        return cls(
            data_dir=data_dir,
            sqlite_path=Path(os.getenv("MEMORY_SQLITE_PATH", str(data_dir / "events.db"))),
            local_long_term_path=Path(
                os.getenv("MEMORY_LOCAL_LONG_TERM_PATH", str(data_dir / "long_term.jsonl"))
            ),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            redis_ttl_seconds=int(os.getenv("REDIS_TTL_SECONDS", "86400")),
            redis_allow_memory_fallback=_bool("REDIS_ALLOW_MEMORY_FALLBACK", False),
            redis_prefix=os.getenv("REDIS_PREFIX", "memory-os"),
            device_state_ttl_seconds=max(1, int(os.getenv("DEVICE_STATE_TTL_SECONDS", "120"))),
            device_heartbeat_seconds=max(1, int(os.getenv("DEVICE_HEARTBEAT_SECONDS", "300"))),
            tool_run_ttl_seconds=max(1, int(os.getenv("TOOL_RUN_TTL_SECONDS", "3600"))),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
            summary_every_turns=max(1, int(os.getenv("SUMMARY_EVERY_TURNS", "10"))),
            summary_retain_turns=max(1, int(os.getenv("SUMMARY_RETAIN_TURNS", "5"))),
            preference_extractor_enabled=_bool("PREFERENCE_EXTRACTOR_ENABLED", True),
            preference_extractor_base_url=os.getenv("PREFERENCE_EXTRACTOR_BASE_URL")
            or os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            preference_extractor_api_key=os.getenv("PREFERENCE_EXTRACTOR_API_KEY")
            or os.getenv("LLM_API_KEY", ""),
            preference_extractor_model=os.getenv("PREFERENCE_EXTRACTOR_MODEL")
            or os.getenv("LLM_MODEL", "glm-4-flash"),
            preference_extract_batch_size=max(1, int(os.getenv("PREFERENCE_EXTRACT_BATCH_SIZE", "8"))),
            preference_extract_min_new_user_messages=max(
                1, int(os.getenv("PREFERENCE_EXTRACT_MIN_NEW_USER_MESSAGES", "10"))
            ),
            preference_extract_max_attempts=max(1, int(os.getenv("PREFERENCE_EXTRACT_MAX_ATTEMPTS", "3"))),
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.local_long_term_path.parent.mkdir(parents=True, exist_ok=True)
