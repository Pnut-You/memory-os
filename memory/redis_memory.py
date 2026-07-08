"""Redis cache and realtime state helpers."""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from typing import Any


class ShortTermMemory:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 86400,
        prefix: str = "memory-os",
        redis_client: Any | None = None,
        allow_memory_fallback: bool = False,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.prefix = prefix
        self._redis = redis_client
        self._memory: dict[str, list[Any]] = defaultdict(list)
        self._expires: dict[str, float] = {}
        self._lock = threading.RLock()
        self.allow_memory_fallback = allow_memory_fallback
        if self._redis is None:
            try:
                import redis  # type: ignore

                candidate = redis.Redis.from_url(redis_url, decode_responses=True)
                candidate.ping()
                self._redis = candidate
            except Exception:
                self._redis = None
        if self._redis is None and not self.allow_memory_fallback:
            raise ConnectionError(
                "Redis is required. Set REDIS_ALLOW_MEMORY_FALLBACK=true only for development/tests."
            )

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"

    def conversation_key(self, device_id: str, user_id: str) -> str:
        return f"{self.prefix}:conversation:{device_id}:{user_id}"

    def session_conversation_key(self, user_id: str, device_id: str, session_id: str) -> str:
        return f"{self.prefix}:session:{user_id}:{device_id}:{session_id}"

    def active_session_key(self, user_id: str, device_id: str) -> str:
        return f"{self.prefix}:active-session:{user_id}:{device_id}"

    def summary_key(self, user_id: str, device_id: str, session_id: str | None = None) -> str:
        if session_id:
            return f"{self.prefix}:summary:{user_id}:{device_id}:{session_id}"
        return f"{self.prefix}:summary:{user_id}:{device_id}"

    def user_card_key(self, user_id: str, device_id: str | None = None) -> str:
        if device_id:
            return f"{self.prefix}:user-card:{user_id}:{device_id}"
        return f"{self.prefix}:user-card:{user_id}"

    def user_preferences_key(self, user_id: str) -> str:
        return f"{self.prefix}:user-preferences:{user_id}"

    def device_state_key(self, device_id: str) -> str:
        return f"{self.prefix}:device-state:{device_id}"

    def data_key(self, namespace: str, item_id: str) -> str:
        return f"{self.prefix}:{namespace}:{item_id}"

    def append_conversation(
        self,
        device_id: str,
        user_id: str,
        messages: list[dict[str, Any]],
        max_items: int = 20,
    ) -> None:
        key = self.conversation_key(device_id, user_id)
        if self._redis is not None:
            try:
                pipe = self._redis.pipeline()
                for message in messages:
                    pipe.rpush(key, json.dumps(message, ensure_ascii=False))
                pipe.ltrim(key, -max(1, max_items), -1)
                pipe.expire(key, self.ttl_seconds)
                pipe.execute()
                return
            except Exception as exc:
                if not self.allow_memory_fallback:
                    raise ConnectionError("Redis conversation append failed") from exc
                self._redis = None
        with self._lock:
            self._purge_expired(key)
            self._memory[key].extend(dict(message) for message in messages)
            self._memory[key] = self._memory[key][-max(1, max_items) :]
            self._expires[key] = time.time() + self.ttl_seconds

    def append_session_conversation(
        self,
        user_id: str,
        device_id: str,
        session_id: str,
        messages: list[dict[str, Any]],
        *,
        ttl_seconds: int | None = None,
        max_items: int = 20,
    ) -> None:
        key = self.session_conversation_key(user_id, device_id, session_id)
        ttl = ttl_seconds or self.ttl_seconds
        if self._redis is not None:
            try:
                pipe = self._redis.pipeline()
                for message in messages:
                    pipe.rpush(key, json.dumps(message, ensure_ascii=False))
                pipe.ltrim(key, -max(1, max_items), -1)
                pipe.expire(key, ttl)
                pipe.execute()
                return
            except Exception as exc:
                if not self.allow_memory_fallback:
                    raise ConnectionError("Redis session conversation append failed") from exc
                self._redis = None
        with self._lock:
            self._purge_expired(key)
            self._memory[key].extend(dict(message) for message in messages)
            self._memory[key] = self._memory[key][-max(1, max_items) :]
            self._expires[key] = time.time() + ttl

    def get_conversation(self, device_id: str, user_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        key = self.conversation_key(device_id, user_id)
        return self._get_list(key, limit)

    def get_session_conversation(
        self,
        user_id: str,
        device_id: str,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return self._get_list(self.session_conversation_key(user_id, device_id, session_id), limit)

    def _get_list(self, key: str, limit: int | None = None) -> list[dict[str, Any]]:
        if self._redis is not None:
            try:
                pipe = self._redis.pipeline()
                pipe.lrange(key, -limit if limit else 0, -1)
                result = pipe.execute()[0]
                return [json.loads(item) for item in result]
            except Exception as exc:
                if not self.allow_memory_fallback:
                    raise ConnectionError("Redis conversation read failed") from exc
                self._redis = None
        with self._lock:
            self._purge_expired(key)
            values = [dict(item) for item in self._memory.get(key, [])]
            return values[-limit:] if limit else values

    def clear_conversation(self, device_id: str, user_id: str) -> None:
        self.delete_key(self.conversation_key(device_id, user_id))
        self.delete_key(self.summary_key(user_id, device_id))

    def set_active_session(
        self,
        user_id: str,
        device_id: str,
        session: dict[str, Any],
        ttl_seconds: int | None = None,
    ) -> None:
        self.set_json(self.active_session_key(user_id, device_id), session, ttl_seconds or self.ttl_seconds)

    def get_active_session(self, user_id: str, device_id: str) -> dict[str, Any] | None:
        return self.get_json(self.active_session_key(user_id, device_id))

    def get_context_bundle(
        self,
        user_id: str,
        device_id: str,
        session_id: str,
        recent_limit: int,
    ) -> dict[str, Any]:
        keys = [
            self.user_card_key(user_id, device_id),
            self.summary_key(user_id, device_id, session_id),
            self.session_conversation_key(user_id, device_id, session_id),
        ]
        if self._redis is not None:
            try:
                pipe = self._redis.pipeline()
                pipe.get(keys[0])
                pipe.get(keys[1])
                pipe.lrange(keys[2], -recent_limit, -1)
                card_raw, summary_raw, conversation_raw = pipe.execute()
                return {
                    "user_card": json.loads(card_raw) if card_raw else None,
                    "summary": json.loads(summary_raw) if summary_raw else None,
                    "recent_messages": [json.loads(item) for item in conversation_raw],
                }
            except Exception as exc:
                if not self.allow_memory_fallback:
                    raise ConnectionError("Redis context read failed") from exc
                self._redis = None
        with self._lock:
            for key in keys:
                self._purge_expired(key)
            card = self._memory.get(keys[0], [None])[0]
            summary = self._memory.get(keys[1], [None])[0]
            conversation = self._memory.get(keys[2], [])[-recent_limit:]
            return {
                "user_card": dict(card) if isinstance(card, dict) else None,
                "summary": dict(summary) if isinstance(summary, dict) else None,
                "recent_messages": [dict(item) for item in conversation],
            }

    def set_json(self, key: str, value: dict[str, Any], ttl_seconds: int | None = None) -> None:
        ttl = ttl_seconds or self.ttl_seconds
        if self._redis is not None:
            try:
                pipe = self._redis.pipeline()
                pipe.set(key, json.dumps(value, ensure_ascii=False), ex=ttl)
                pipe.execute()
                return
            except Exception as exc:
                if not self.allow_memory_fallback:
                    raise ConnectionError("Redis write failed") from exc
                self._redis = None
        with self._lock:
            self._memory[key] = [dict(value)]
            self._expires[key] = time.time() + ttl

    def get_json(self, key: str) -> dict[str, Any] | None:
        if self._redis is not None:
            try:
                pipe = self._redis.pipeline()
                pipe.get(key)
                raw = pipe.execute()[0]
                return json.loads(raw) if raw else None
            except Exception as exc:
                if not self.allow_memory_fallback:
                    raise ConnectionError("Redis read failed") from exc
                self._redis = None
        with self._lock:
            self._purge_expired(key)
            values = self._memory.get(key, [])
            return dict(values[0]) if values and isinstance(values[0], dict) else None

    def set_value(self, namespace: str, item_id: str, value: dict[str, Any], ttl_seconds: int) -> None:
        self.set_json(self.data_key(namespace, item_id), value, ttl_seconds)

    def get_value(self, namespace: str, item_id: str) -> dict[str, Any] | None:
        return self.get_json(self.data_key(namespace, item_id))

    def delete_value(self, namespace: str, item_id: str) -> None:
        self.delete_key(self.data_key(namespace, item_id))

    def delete_key(self, key: str) -> None:
        if self._redis is not None:
            try:
                pipe = self._redis.pipeline()
                pipe.delete(key)
                pipe.execute()
                return
            except Exception as exc:
                if not self.allow_memory_fallback:
                    raise ConnectionError("Redis delete failed") from exc
                self._redis = None
        with self._lock:
            self._memory.pop(key, None)
            self._expires.pop(key, None)

    def _purge_expired(self, key: str) -> None:
        if self._expires.get(key, float("inf")) <= time.time():
            self._memory.pop(key, None)
            self._expires.pop(key, None)
