"""Environment-driven configuration for the plug-and-play memory system."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .utils import load_dotenv


QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_CHAT_MODEL = "qwen3.7-plus"
QWEN_MEMORY_MODEL = "qwen3.7-max"


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


def _first_env(names: tuple[str, ...]) -> tuple[str, str]:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value, name
    return "", ""


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
    llm_api_key_source: str = ""
    llm_base_url: str = QWEN_BASE_URL
    llm_model: str = QWEN_CHAT_MODEL
    summary_every_turns: int = 10
    summary_retain_turns: int = 5
    short_memory_summary_min_turns: int = 20
    short_memory_prompt_trigger_tokens: int = 5000
    short_memory_retain_recent_turns: int = 5
    preference_extractor_enabled: bool = True
    preference_extractor_base_url: str = ""
    preference_extractor_api_key: str = ""
    preference_extractor_api_key_source: str = ""
    preference_extractor_model: str = ""
    preference_extract_batch_size: int = 8
    preference_extract_min_new_user_messages: int = 10
    preference_extract_max_attempts: int = 3

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "MemoryConfig":
        load_dotenv(env_file)
        data_dir = Path(os.getenv("MEMORY_DATA_DIR", "data"))
        llm_api_key, llm_api_key_source = _first_env(("DASHSCOPE_API_KEY", "LLM_API_KEY"))
        preference_api_key, preference_api_key_source = _first_env(
            ("PREFERENCE_EXTRACTOR_API_KEY", "DASHSCOPE_API_KEY", "LLM_API_KEY")
        )
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
            llm_api_key=llm_api_key,
            llm_api_key_source=llm_api_key_source,
            llm_base_url=os.getenv("LLM_BASE_URL", QWEN_BASE_URL),
            llm_model=os.getenv("LLM_MODEL", QWEN_CHAT_MODEL),
            summary_every_turns=max(1, int(os.getenv("SUMMARY_EVERY_TURNS", "10"))),
            summary_retain_turns=max(1, int(os.getenv("SUMMARY_RETAIN_TURNS", "5"))),
            short_memory_summary_min_turns=max(
                1,
                int(
                    os.getenv(
                        "SHORT_MEMORY_SUMMARY_MIN_TURNS",
                        os.getenv("SUMMARY_EVERY_TURNS", "20"),
                    )
                ),
            ),
            short_memory_prompt_trigger_tokens=max(
                1, int(os.getenv("SHORT_MEMORY_PROMPT_TRIGGER_TOKENS", "5000"))
            ),
            short_memory_retain_recent_turns=max(
                1,
                int(
                    os.getenv(
                        "SHORT_MEMORY_RETAIN_RECENT_TURNS",
                        os.getenv("SUMMARY_RETAIN_TURNS", "5"),
                    )
                ),
            ),
            preference_extractor_enabled=_bool("PREFERENCE_EXTRACTOR_ENABLED", True),
            preference_extractor_base_url=os.getenv("PREFERENCE_EXTRACTOR_BASE_URL")
            or os.getenv("LLM_BASE_URL", QWEN_BASE_URL),
            preference_extractor_api_key=preference_api_key,
            preference_extractor_api_key_source=preference_api_key_source,
            preference_extractor_model=os.getenv("PREFERENCE_EXTRACTOR_MODEL")
            or QWEN_MEMORY_MODEL,
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
