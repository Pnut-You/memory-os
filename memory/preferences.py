"""Preference registry and extractor result validation."""

from __future__ import annotations

import json
import re
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

PREFERENCE_KEYWORDS = (
    "我是",
    "我的职业",
    "我从事",
    "我做",
    "我喜欢",
    "喜欢",
    "我爱",
    "爱好",
    "偏好",
    "我不喜欢",
    "不喜欢",
    "我习惯",
    "以后都",
    "以后不要",
    "默认给我",
    "尽量",
    "我更喜欢",
    "我讨厌",
    "记住",
)


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
        small_model: str = "codeqwen1.5-7b-chat",
        large_model: str = "qwen3.7-max",
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
        preferences: list[ExtractedPreferenceModel] = []
        raw_outputs: list[dict[str, Any]] = []
        validated_outputs: list[dict[str, Any]] = []
        fallback_used = False
        for event in self._source_user_events(events):
            event_id = int(event.get("event_id") or event.get("id") or 0)
            source_text = str(event.get("text") or event.get("content") or "").strip()
            if not source_text or not self._may_contain_target_preference(source_text):
                continue
            try:
                candidate, attempts, used_fallback = self._extract_one(source_text)
            except PreferenceExtractionError as exc:
                attempts = [{"event_id": event_id, **attempt} for attempt in exc.attempts]
                raise PreferenceExtractionError(str(exc), attempts) from exc
            raw_outputs.extend({"event_id": event_id, **attempt} for attempt in attempts)
            fallback_used = fallback_used or used_fallback
            validated_outputs.append({"event_id": event_id, **candidate})
            if candidate["type"] == "none":
                continue
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
            raw_outputs=raw_outputs,
            validated_outputs=validated_outputs,
            fallback_used=fallback_used,
        )

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

    @staticmethod
    def _may_contain_target_preference(text: str) -> bool:
        return bool(
            re.search(
                r"职业|工作|身份|从事|一名|我是|我做|喜欢|喜爱|爱好|偏爱|偏好|合胃口|"
                r"不喜欢|不爱|讨厌|受不了|不舒服|避开|不要有",
                text,
            )
        )

    def _extract_one(self, source_text: str) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
        models = [self.large_model] if self.mode == "large" else [self.small_model]
        if self.mode == "hybrid":
            models.append(self.large_model)
        attempts: list[dict[str, Any]] = []
        last_error = "unknown validation error"
        for index, model in enumerate(models):
            raw = ""
            try:
                raw = self._request_strict_extraction(model, source_text)
                candidate = self._validate_strict_output(raw, source_text)
                attempts.append({"model": model, "raw_output": raw, "validation_error": ""})
                return candidate, attempts, index > 0
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                attempts.append({"model": model, "raw_output": raw, "validation_error": last_error})
        raise PreferenceExtractionError(
            f"preference extraction validation failed: {last_error}",
            attempts,
        )

    def _request_strict_extraction(self, model: str, source_text: str) -> str:
        payload = {
            "model": model,
            "temperature": 0,
            "max_tokens": 160,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是长期偏好抽取器。只分析当前这一条用户原文。"
                        "只输出一个严格 JSON 对象，不要 Markdown、解释或额外字段。"
                        "type 只能是 occupation、likes、dislikes、none。"
                        "occupation 表示用户明确陈述的职业或工作身份；"
                        "likes 表示用户明确喜欢、偏好或正面接受的对象；"
                        "dislikes 表示用户明确不喜欢、讨厌、避开或感到不舒服的对象。"
                        "“不太喜欢”属于 dislikes，“偏好里加上”属于 likes，“偏好里不要有”属于 dislikes。"
                        "原文明示以上事实时不得返回 none。"
                        "value 必须逐字来自当前原文，evidence 必须是包含 value 的对应原句。"
                        "evidence 必须从当前原文逐字复制，不得改写人称、标点或任何字符。"
                        "四个字段始终都必须输出，confidence 始终必须是 0 到 1 的数字。"
                        "无法从原文确定时输出 type=none，value 和 evidence 为空字符串。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "current_user_text": source_text,
                            "output_schema": {
                                "type": "occupation | likes | dislikes | none",
                                "value": "当前原文中的内容",
                                "evidence": "当前原文中的对应原句",
                                "confidence": 0.0,
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
            return content.strip()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"preference extraction http {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"preference extraction failed: {exc}") from exc

    @staticmethod
    def _validate_strict_output(raw: str, source_text: str) -> dict[str, Any]:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("output must be a JSON object")
        required = {"type", "value", "evidence", "confidence"}
        if set(value) != required:
            raise ValueError("output fields must exactly match type/value/evidence/confidence")
        preference_type = value["type"]
        extracted_value = value["value"]
        evidence = value["evidence"]
        confidence = value["confidence"]
        if preference_type not in {"occupation", "likes", "dislikes", "none"}:
            raise ValueError("invalid preference type")
        if not isinstance(extracted_value, str) or not isinstance(evidence, str):
            raise ValueError("value and evidence must be strings")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be a number in [0,1]")
        extracted_value = extracted_value.strip()
        evidence = evidence.strip()
        if preference_type == "none":
            if extracted_value or evidence:
                raise ValueError("none output must have empty value and evidence")
            if PreferenceExtractor._has_explicit_preference_signal(source_text):
                raise ValueError("none is invalid for text with an explicit preference signal")
        else:
            if not extracted_value or not evidence:
                raise ValueError("value and evidence cannot be empty")
            if extracted_value not in source_text or evidence not in source_text or extracted_value not in evidence:
                raise ValueError("value and evidence must come from the current user text")
        return {
            "type": preference_type,
            "value": extracted_value,
            "evidence": evidence,
            "confidence": float(confidence),
        }

    @staticmethod
    def _has_explicit_preference_signal(text: str) -> bool:
        return bool(
            re.search(
                r"职业|工作身份|一名|我做.+工作|喜欢|喜爱|爱好|偏好|合胃口|"
                r"不.{0,2}喜欢|不爱|讨厌|受不了|不舒服|避开|不要有",
                text,
            )
        )

    def extract_action_preferences(
        self,
        user_id: str,
        action_memory_context: dict[str, Any],
    ) -> ActionPreferenceExtractionResult:
        if not self.configured:
            return ActionPreferenceExtractionResult(schema_version="1.0", user_id=user_id, memories=[])
        payload = {
            "model": self.action_model,
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


def should_schedule_preference(text: str) -> bool:
    return any(keyword in text for keyword in PREFERENCE_KEYWORDS)


def normalize_preference_key(key: str) -> tuple[str, str]:
    if key in PREFERENCE_REGISTRY:
        return key, key.split(".", 1)[0]
    return "other", "other"
