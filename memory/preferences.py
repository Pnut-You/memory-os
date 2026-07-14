"""Preference registry and extractor result validation."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


PREFERENCE_REGISTRY: dict[str, str] = {
    "profile.occupation": "用户职业或身份",
    "preference.likes": "用户明确喜欢的事物",
    "preference.dislikes": "用户明确不喜欢的事物",
    "habit.routine": "用户稳定习惯",
    "constraint.stable": "用户稳定约束或禁忌",
    "relationship.person": "用户关系人或宠物等事实",
    "default_behavior.preference": "用户要求的默认行为",
    "navigation.noise_level": "路线安静程度",
    "navigation.crowd_level": "路线人流程度",
    "navigation.route_type": "最短、安全、风景路线",
    "navigation.speed": "导航移动速度",
    "interaction.language": "交互语言",
    "interaction.reply_length": "回复长度",
    "interaction.style": "回复风格",
    "speech.volume": "播放音量",
    "speech.rate": "语速",
    "speech.voice_style": "音色风格",
    "content.music.genre": "音乐类型",
    "content.music.artist": "音乐人或歌手偏好",
    "content.story.type": "故事类型",
    "content.news.topic": "新闻主题",
    "motion.default_speed": "默认移动速度",
    "motion.turn_amplitude": "转弯幅度",
    "motion.follow_distance": "跟随距离",
}

class PreferenceExtractionError(RuntimeError):
    def __init__(self, message: str, attempts: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.fallback_used = len(attempts) > 1


@dataclass(slots=True)
class PreferenceEvidenceModel:
    event_id: int
    text: str
    type: str = "explicit"

    def model_dump(self) -> dict[str, Any]:
        return {"event_id": self.event_id, "text": self.text, "type": self.type}


@dataclass(slots=True)
class ExtractedPreferenceModel:
    preference_key: str
    category: str
    value: dict[str, Any]
    display_text_zh: str
    polarity: str = "prefer"
    durability: str = "persistent"
    strength: float = 0.5
    confidence: float = 0.5
    source_type: str = "explicit"
    evidence: list[PreferenceEvidenceModel] = field(default_factory=list)
    expires_at: str | None = None
    action: str = "upsert"
    reason_zh: str = ""
    scope: str = "user"


@dataclass(slots=True)
class PreferenceExtractionResult:
    schema_version: str
    user_id: str
    preferences: list[ExtractedPreferenceModel] = field(default_factory=list)
    raw_outputs: list[dict[str, Any]] = field(default_factory=list)
    validated_outputs: list[dict[str, Any]] = field(default_factory=list)
    fallback_used: bool = False
    request_metrics: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def model_validate(cls, value: dict[str, Any], default_user_id: str = "") -> "PreferenceExtractionResult":
        if isinstance(value, list):
            value = {"schema_version": "1.0", "user_id": default_user_id, "preferences": value}
        if isinstance(value, dict) and "preferences" not in value and "preference_key" in value:
            value = {"schema_version": "1.0", "user_id": default_user_id, "preferences": [value]}
        if not isinstance(value, dict):
            raise ValueError("extractor output must be an object")
        if default_user_id and not value.get("user_id"):
            value = {**value, "user_id": default_user_id}
        user_id = str(value.get("user_id") or "")
        if not user_id:
            raise ValueError("user_id is required")
        preferences = []
        for raw in value.get("preferences") or []:
            if not isinstance(raw, dict):
                raise ValueError("preference item must be an object")
            strength = float(raw.get("strength", 0.5))
            confidence = float(raw.get("confidence", 0.5))
            if not 0 <= strength <= 1 or not 0 <= confidence <= 1:
                raise ValueError("strength/confidence must be in [0,1]")
            display = str(raw.get("display_text_zh") or "")
            if not display:
                raise ValueError("display_text_zh is required")
            evidence = []
            for item in raw.get("evidence") or []:
                evidence.append(
                    PreferenceEvidenceModel(
                        event_id=int(item.get("event_id") or 0),
                        text=str(item.get("text") or ""),
                        type=str(item.get("type") or "explicit"),
                    )
                )
            action = str(raw.get("action") or "upsert")
            if action not in {"upsert", "revoke"}:
                raise ValueError("unsupported preference action")
            preferences.append(
                ExtractedPreferenceModel(
                    preference_key=str(raw.get("preference_key") or ""),
                    category=str(raw.get("category") or ""),
                    value=raw.get("value") if isinstance(raw.get("value"), dict) else {},
                    display_text_zh=display,
                    polarity=str(raw.get("polarity") or "prefer"),
                    durability=str(raw.get("durability") or "persistent"),
                    strength=strength,
                    confidence=confidence,
                    source_type=str(raw.get("source_type") or "explicit"),
                    evidence=evidence,
                    expires_at=raw.get("expires_at"),
                    action=action,
                    reason_zh=str(raw.get("reason_zh") or ""),
                    scope=str(raw.get("scope") or "user"),
                )
            )
        return cls(str(value.get("schema_version") or "1.0"), user_id, preferences)

    @classmethod
    def model_validate_json(cls, value: str, default_user_id: str = "") -> "PreferenceExtractionResult":
        return cls.model_validate(json.loads(_json_text(value)), default_user_id=default_user_id)


@dataclass(slots=True)
class ExtractedActionPreferenceMemory:
    content: str
    title: str = ""
    confidence: float = 0.5
    source_event_ids: list[int] = field(default_factory=list)
    reason_zh: str = ""

    def model_dump(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "title": self.title,
            "confidence": self.confidence,
            "source_event_ids": list(self.source_event_ids),
            "reason_zh": self.reason_zh,
        }


@dataclass(slots=True)
class ActionPreferenceExtractionResult:
    schema_version: str
    user_id: str
    memories: list[ExtractedActionPreferenceMemory] = field(default_factory=list)

    @classmethod
    def model_validate(cls, value: dict[str, Any], default_user_id: str = "") -> "ActionPreferenceExtractionResult":
        if isinstance(value, list):
            value = {"schema_version": "1.0", "user_id": default_user_id, "memories": value}
        if isinstance(value, dict) and "memories" not in value and "content" in value:
            value = {"schema_version": "1.0", "user_id": default_user_id, "memories": [value]}
        if not isinstance(value, dict):
            raise ValueError("action preference extractor output must be an object")
        if default_user_id and not value.get("user_id"):
            value = {**value, "user_id": default_user_id}
        user_id = str(value.get("user_id") or "")
        if not user_id:
            raise ValueError("user_id is required")
        memories = []
        for raw in value.get("memories") or []:
            if not isinstance(raw, dict):
                raise ValueError("memory item must be an object")
            content = str(raw.get("content") or "").strip()
            if not content:
                raise ValueError("content is required")
            confidence = float(raw.get("confidence", 0.5))
            if not 0 <= confidence <= 1:
                raise ValueError("confidence must be in [0,1]")
            source_event_ids = []
            for event_id in raw.get("source_event_ids") or []:
                try:
                    source_event_ids.append(int(event_id))
                except (TypeError, ValueError):
                    continue
            memories.append(
                ExtractedActionPreferenceMemory(
                    content=content,
                    title=str(raw.get("title") or ""),
                    confidence=confidence,
                    source_event_ids=source_event_ids,
                    reason_zh=str(raw.get("reason_zh") or ""),
                )
            )
        return cls(str(value.get("schema_version") or "1.0"), user_id, memories)

    @classmethod
    def model_validate_json(cls, value: str, default_user_id: str = "") -> "ActionPreferenceExtractionResult":
        return cls.model_validate(json.loads(_json_text(value)), default_user_id=default_user_id)


def _json_text(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    object_start = text.find("{")
    object_end = text.rfind("}")
    array_start = text.find("[")
    array_end = text.rfind("]")
    if object_start >= 0 and object_end >= object_start:
        if array_start < 0 or object_start < array_start:
            return text[object_start : object_end + 1]
    if array_start >= 0 and array_end >= array_start:
        return text[array_start : array_end + 1]
    return text


class PreferenceExtractor:
    def __init__(
        self,
        *,
        enabled: bool,
        api_key: str,
        base_url: str,
        model: str,
        mode: str = "small",
        small_model: str = "qwen3.5-flash-2026-02-23",
        large_model: str = "qwen3.5-flash-2026-02-23",
        prompt_version: str = "preferences-v1",
    ) -> None:
        self.enabled = enabled
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        if mode not in {"small", "hybrid", "large"}:
            raise ValueError("preference extractor mode must be small, hybrid, or large")
        self.mode = mode
        self.action_model = model
        self.small_model = small_model
        self.large_model = large_model
        self.model = large_model if mode == "large" else small_model
        self.prompt_version = "preferences-v2-strict"

    @property
    def configured(self) -> bool:
        return self.enabled and bool(self.api_key and self.base_url and self.model)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "configured": self.configured,
            "base_url": self.base_url,
            "model": self.model,
            "mode": self.mode,
            "small_model": self.small_model,
            "large_model": self.large_model,
        }

    def extract(
        self,
        user_id: str,
        events: dict[str, Any],
        existing_preferences: list[dict[str, Any]] | None = None,
    ) -> PreferenceExtractionResult:
        if not self.configured:
            return PreferenceExtractionResult(schema_version="2.0", user_id=user_id, preferences=[])
        source_events = []
        for event in self._source_user_events(events):
            event_id = int(event.get("event_id") or event.get("id") or 0)
            source_text = str(event.get("text") or event.get("content") or "").strip()
            if source_text and event_id:
                source_events.append({"event_id": event_id, "text": source_text})
        if not source_events:
            return PreferenceExtractionResult(schema_version="2.0", user_id=user_id, preferences=[])
        candidates, attempts, fallback_used, metrics = self._extract_session(source_events)
        preferences: list[ExtractedPreferenceModel] = []
        for candidate in candidates:
            event_id = int(candidate["event_id"])
            key = {
                "occupation": "profile.occupation",
                "likes": "preference.likes",
                "dislikes": "preference.dislikes",
            }[candidate["type"]]
            extracted_value = str(candidate["value"])
            preferences.append(
                ExtractedPreferenceModel(
                    preference_key=key,
                    category="profile" if candidate["type"] == "occupation" else "preference",
                    value={
                        "type": "string",
                        "value": extracted_value,
                        "code": extracted_value,
                        "label_zh": extracted_value,
                    },
                    display_text_zh=extracted_value,
                    polarity="avoid" if candidate["type"] == "dislikes" else "prefer",
                    durability="persistent",
                    strength=float(candidate["confidence"]),
                    confidence=float(candidate["confidence"]),
                    source_type="explicit",
                    evidence=[PreferenceEvidenceModel(event_id=event_id, text=str(candidate["evidence"]))],
                    scope="user",
                )
            )
        return PreferenceExtractionResult(
            schema_version="2.0",
            user_id=user_id,
            preferences=preferences,
            raw_outputs=attempts,
            validated_outputs=candidates,
            fallback_used=fallback_used,
            request_metrics=metrics,
        )

    def _extract_session(
        self, source_events: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, list[dict[str, Any]]]:
        models = [self.large_model] if self.mode == "large" else [self.small_model]
        if self.mode == "hybrid":
            models.append(self.large_model)
        attempts: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        last_error = "unknown validation error"
        for index, model in enumerate(models):
            raw = ""
            started = time.perf_counter()
            try:
                raw, usage = self._request_session_extraction(model, source_events)
                candidates = self._validate_session_output(raw, source_events)
                elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
                attempts.append({"model": model, "raw_output": raw, "validation_error": ""})
                metrics.append({
                    "model": model,
                    "input_events": len(source_events),
                    "input_chars": sum(len(item["text"]) for item in source_events),
                    "output_chars": len(raw),
                    "duration_ms": elapsed_ms,
                    "usage": usage,
                    "succeeded": True,
                })
                return candidates, attempts, index > 0, metrics
            except Exception as exc:
                elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
                last_error = f"{type(exc).__name__}: {exc}"
                attempts.append({"model": model, "raw_output": raw, "validation_error": last_error})
                metrics.append({
                    "model": model,
                    "input_events": len(source_events),
                    "input_chars": sum(len(item["text"]) for item in source_events),
                    "output_chars": len(raw),
                    "duration_ms": elapsed_ms,
                    "usage": {},
                    "succeeded": False,
                })
        raise PreferenceExtractionError(f"preference extraction validation failed: {last_error}", attempts)

    def _request_session_extraction(
        self, model: str, source_events: list[dict[str, Any]]
    ) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": model,
            "enable_thinking": False,
            "temperature": 0,
            "max_tokens": 800,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是长期偏好抽取器。输入是一个已结束会话中的全部用户原文。"
                        "只提取用户明确表达且适合长期保存的职业、喜欢和不喜欢。"
                        "不要从助手文本或常识推断。只输出严格 JSON 对象 {preferences: [...]}。"
                        "每项只能包含 event_id、type、value、evidence、confidence；"
                        "type 只能是 occupation、likes、dislikes。event_id 必须引用对应原文，"
                        "value 和 evidence 必须逐字来自该 event_id 的 text。没有结果返回空数组。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_messages": source_events,
                            "output_schema": {
                                "preferences": [{
                                    "event_id": 1,
                                    "type": "occupation | likes | dislikes",
                                    "value": "原文中的内容",
                                    "evidence": "原文中的对应语句",
                                    "confidence": 0.0,
                                }]
                            },
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty model output")
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            return content.strip(), usage
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"preference extraction http {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"preference extraction failed: {exc}") from exc

    @staticmethod
    def _validate_session_output(raw: str, source_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != {"preferences"}:
            raise ValueError("output must contain only preferences")
        items = value["preferences"]
        if not isinstance(items, list):
            raise ValueError("preferences must be an array")
        source_by_id = {int(item["event_id"]): str(item["text"]) for item in source_events}
        validated = []
        seen: set[tuple[int, str, str]] = set()
        required = {"event_id", "type", "value", "evidence", "confidence"}
        for item in items:
            if not isinstance(item, dict) or set(item) != required:
                raise ValueError("preference fields must exactly match event_id/type/value/evidence/confidence")
            event_id = item["event_id"]
            if isinstance(event_id, bool) or not isinstance(event_id, int) or event_id not in source_by_id:
                raise ValueError("event_id is not part of the current session")
            pref_type = item["type"]
            extracted_value = item["value"]
            evidence = item["evidence"]
            confidence = item["confidence"]
            if pref_type not in {"occupation", "likes", "dislikes"}:
                raise ValueError("invalid preference type")
            if not isinstance(extracted_value, str) or not isinstance(evidence, str):
                raise ValueError("value and evidence must be strings")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                raise ValueError("confidence must be a number in [0,1]")
            extracted_value = extracted_value.strip()
            evidence = evidence.strip()
            source_text = source_by_id[event_id]
            if not extracted_value or not evidence or extracted_value not in evidence or evidence not in source_text:
                raise ValueError("value and evidence must come from the referenced user text")
            identity = (event_id, pref_type, extracted_value)
            if identity in seen:
                continue
            seen.add(identity)
            validated.append({
                "event_id": event_id,
                "type": pref_type,
                "value": extracted_value,
                "evidence": evidence,
                "confidence": float(confidence),
            })
        return validated

    @staticmethod
    def _source_user_events(context: dict[str, Any]) -> list[dict[str, Any]]:
        source = context.get("source_user_events")
        if isinstance(source, list):
            return [item for item in source if isinstance(item, dict)]
        result: list[dict[str, Any]] = []
        for turn in context.get("recent_turns") or []:
            for message in turn.get("messages") or []:
                if isinstance(message, dict) and message.get("role") == "user":
                    result.append(message)
        return result

    def extract_action_preferences(
        self,
        user_id: str,
        action_memory_context: dict[str, Any],
    ) -> ActionPreferenceExtractionResult:
        if not self.configured:
            return ActionPreferenceExtractionResult(schema_version="1.0", user_id=user_id, memories=[])
        payload = {
            "model": self.action_model,
            "enable_thinking": False,
            "temperature": 0.1,
            "max_tokens": 1600,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是机器狗 Memory OS 的动作事件偏好抽取器。只输出严格 JSON，不要 Markdown。"
                        "输入是最近七天按天聚合的动作事件记忆 text。"
                        "只抽取反复出现、稳定、有未来复用价值的动作链路偏好。"
                        "单次出现、普通闲聊、时间总结、非机器狗动作不要输出。"
                        "输出写入事件偏好记忆，不是用户结构化长期偏好。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "schema_version": "1.0",
                            "user_id": user_id,
                            "action_memory_context": action_memory_context,
                            "output_schema": {
                                "schema_version": "1.0",
                                "user_id": user_id,
                                "memories": [
                                    {
                                        "content": "七天动作偏好记忆文本，说明稳定动作链路和证据",
                                        "title": "可选标题",
                                        "confidence": "0-1",
                                        "source_event_ids": [1, 2],
                                        "reason_zh": "为什么这是稳定动作偏好",
                                    }
                                ],
                            },
                            "rules": [
                                "顶层必须是对象，包含 schema_version、user_id、memories。",
                                "memories 必须是数组；没有稳定动作偏好时返回空数组。",
                                "source_event_ids 只能引用 action_memory_context.action_memories 中真实 event_id。",
                                "content 必须是中文自然文本，可直接作为事件库 text 展示。",
                                "不要输出 Markdown，不要解释。",
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            try:
                return ActionPreferenceExtractionResult.model_validate_json(content, default_user_id=user_id)
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(
                    f"action preference extraction validation failed: {exc}; response={content[:500]}"
                ) from exc
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"action preference extraction http {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"action preference extraction failed: {exc}") from exc


def normalize_preference_key(key: str) -> tuple[str, str]:
    if key in PREFERENCE_REGISTRY:
        return key, key.split(".", 1)[0]
    return "other", "other"
