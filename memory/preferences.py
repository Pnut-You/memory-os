"""Preference registry and extractor result validation."""

from __future__ import annotations

import json
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
        prompt_version: str = "preferences-v1",
    ) -> None:
        self.enabled = enabled
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.prompt_version = prompt_version

    @property
    def configured(self) -> bool:
        return self.enabled and bool(self.api_key and self.base_url and self.model)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "configured": self.configured,
            "base_url": self.base_url,
            "model": self.model,
        }

    def extract(self, user_id: str, events: list[dict[str, Any]], existing_preferences: list[dict[str, Any]] | None = None) -> PreferenceExtractionResult:
        if not self.configured:
            return PreferenceExtractionResult(schema_version="1.0", user_id=user_id, preferences=[])
        compact_existing = [
            {
                "preference_key": item.get("preference_key"),
                "category": item.get("category"),
                "value_json": item.get("value_json"),
                "display_text_zh": item.get("display_text_zh"),
                "status": item.get("status"),
                "confidence": item.get("confidence"),
                "strength": item.get("strength"),
            }
            for item in (existing_preferences or [])[:30]
        ]
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": 1600,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是机器狗 Memory OS 的用户偏好抽取器。只输出严格 JSON，不要 Markdown。"
                        "结构化偏好记忆优先只写三类：profile.occupation、preference.likes、preference.dislikes。"
                        "同时支持 habit.routine、constraint.stable、relationship.person、default_behavior.preference。"
                        "职业只从明确身份/职业表述抽取，例如我是摄影师、我的职业是产品经理。"
                        "喜欢用于明确稳定喜欢的事物，例如我喜欢摄影、我喜欢周杰伦。"
                        "明确不喜欢用于明确负向偏好，例如我不喜欢吵闹、以后不要摇滚。"
                        "如果 rolling_summary 里有用户不喜欢/喜欢/职业信息，也必须结合 summary_evidence_events 回溯原始证据并抽取。"
                        "单次命令不要写入长期偏好；未知旧细分类不要优先使用，除非三类完全无法表达。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
	                            "schema_version": "1.0",
		                            "user_id": user_id,
	                            "allowed_keys": PREFERENCE_REGISTRY,
	                            "preference_context": events,
	                            "existing_preferences": compact_existing,
	                            "output_schema": {
	                                "schema_version": "1.0",
	                                "user_id": user_id,
                                "preferences": [
                                    {
                                        "preference_key": "profile.occupation/preference.likes/preference.dislikes 优先；未知用 other",
                                        "category": "profile/preference",
                                        "value": {"type": "enum/string/number/json", "code": "machine_readable", "label_zh": "中文值"},
                                        "display_text_zh": "中文偏好描述",
                                        "polarity": "prefer/avoid",
                                        "durability": "persistent/temporary",
                                        "strength": "0-1",
                                        "confidence": "0-1",
                                        "source_type": "explicit/implicit/action_pattern",
                                        "scope": "user/user_device",
                                        "reason_zh": "为什么这是偏好",
                                        "evidence": [{"event_id": 1, "text": "证据原文", "type": "explicit/action"}],
                                        "expires_at": None,
                                        "action": "upsert/revoke",
                                    }
                                ],
	                            },
                                "rules": [
                                    "顶层必须是对象，必须包含 schema_version、user_id、preferences。",
                                    "preferences 必须是数组；没有稳定偏好时返回空数组。",
                                    "优先使用三类结构化偏好记忆 key，不要把摄影、周杰伦等喜欢的事物写成 other。",
                                    "evidence 必须优先引用 preference_context.recent_turns、summary_evidence_events 或 action_events 中的真实 event_id。",
                                    "summary 可以提示应该抽取什么，但 evidence.text 要尽量使用 summary_evidence_events 里的用户原话。",
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
                return PreferenceExtractionResult.model_validate_json(content, default_user_id=user_id)
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(
                    f"preference extraction validation failed: {exc}; response={content[:500]}"
                ) from exc
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"preference extraction http {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"preference extraction failed: {exc}") from exc


def should_schedule_preference(text: str) -> bool:
    return any(keyword in text for keyword in PREFERENCE_KEYWORDS)


def normalize_preference_key(key: str) -> tuple[str, str]:
    if key in PREFERENCE_REGISTRY:
        return key, key.split(".", 1)[0]
    return "other", "other"
