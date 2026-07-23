"""Strict fact judge and metrics for Time Memory evaluation."""

from __future__ import annotations

import json
import re
import time
import unicodedata
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Any


LABELS = ("EXACT_MATCH", "SEMANTIC_MATCH", "PARTIAL_MATCH", "MISMATCH")
PASS_LABELS = {"EXACT_MATCH", "SEMANTIC_MATCH"}
PROMPT_VERSION = "time-memory-fact-judge-v3-strict"


class JudgeError(RuntimeError):
    def __init__(self, message: str, *, attempts: list[dict[str, str]] | None = None) -> None:
        super().__init__(message)
        self.attempts = attempts or []


@dataclass(frozen=True)
class JudgeConfig:
    api_key: str
    base_url: str
    model: str


class StrictTimeMemoryJudge:
    """Verify one expected fact against source messages and a generated summary."""

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config

    def judge(
        self,
        *,
        case_id: str,
        source_messages: list[dict[str, Any]],
        expected_fact: dict[str, Any],
        generated_summary: str,
    ) -> dict[str, Any]:
        request_body = {
            "case_id": case_id,
            "source_messages": source_messages,
            "expected_fact": expected_fact,
            "generated_summary": generated_summary,
        }
        payload = {
            "model": self.config.model,
            "enable_thinking": False,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": strict_judge_prompt()},
                {"role": "user", "content": json.dumps(request_body, ensure_ascii=False)},
            ],
        }
        attempts: list[dict[str, str]] = []
        for attempt_number in range(1, 3):
            raw_output = ""
            started = time.monotonic()
            try:
                request = urllib.request.Request(
                    f"{self.config.base_url.rstrip('/')}/chat/completions",
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=75) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                raw_output = str(response_payload["choices"][0]["message"]["content"]).strip()
                parsed = json.loads(extract_json_object(raw_output))
                validated = validate_judge_result(parsed, expected_fact)
                validated.update(
                    {
                        "raw_output": raw_output,
                        "attempt": attempt_number,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "model": self.config.model,
                        "prompt_version": PROMPT_VERSION,
                    }
                )
                return validated
            except Exception as exc:
                attempts.append(
                    {
                        "attempt": str(attempt_number),
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_output": raw_output,
                    }
                )
        raise JudgeError(f"strict judge failed after 2 attempts: {attempts[-1]['error']}", attempts=attempts)


def strict_judge_prompt() -> str:
    return (
        "你是严格的时间记忆事实评测器。你必须对比原始对话、Expected Fact、Critical Slots 和 "
        "Generated Summary，而不是做宽松的语义相似度判断。\n"
        "逐条核验当前 Expected Fact：主题相似不代表事实正确；所有 critical_slots 都必须在摘要中明确出现"
        "或能够无歧义地推断；不得用更宽泛的词替代关键人物、物品限定、位置、时间、数量、否定状态或完成状态。\n"
        "原始对话只用于确定正确答案，绝不能用来替摘要补全缺失字段。若时间、人物、位置等只出现在原始对话、"
        "没有出现在 Generated Summary，即使你知道正确值，也必须把该 slot 列入 missing_slots。\n"
        "工具箱交给邻居不等于红色工具箱交给周叔；钥匙已经收好不等于钥匙放在玄关第二层抽屉；"
        "准备修理不等于已经修好；门开着不等于门关闭。若一天中状态更新，以最后确认状态为准。\n"
        "EXACT_MATCH：完整事实和全部关键字段正确，措辞基本未改写。\n"
        "SEMANTIC_MATCH：事实和全部关键字段正确，仅有合理改写，例如十三度/13℃、晚上八点/20:00、"
        "修剪完毕/已完成修剪。\n"
        "PARTIAL_MATCH：核心事件仍存在且无错误/冲突/幻觉，但至少一个 critical slot 被省略。\n"
        "更宽泛的类别只能算保留大意：例如‘完成桂花树维护’没有明确‘修剪’，action=修剪应列入 missing_slots，"
        "返回 PARTIAL_MATCH；不要因为大类相近判 SEMANTIC_MATCH。\n"
        "MISMATCH：核心事实遗漏，任一关键字段错误或冲突，肯定/否定、计划/完成或旧/新状态混淆，"
        "或者摘要加入原始对话没有的事实。只要发现任何幻觉，本条必须为 MISMATCH。\n"
        "preserved_slots、missing_slots、conflicting_slots 只能填写 critical_slots 的键，每个键必须且只能"
        "出现在三者之一。hallucinated_facts 写摘要中无原始依据的具体事实，没有则为空数组。\n"
        "只输出严格 JSON："
        '{"label":"EXACT_MATCH | SEMANTIC_MATCH | PARTIAL_MATCH | MISMATCH",'
        '"preserved_slots":[],"missing_slots":[],"conflicting_slots":[],'
        '"hallucinated_facts":[],"reason":""}'
    )


def validate_judge_result(value: Any, expected_fact: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("judge output must be an object")
    label = str(value.get("label") or "").strip().upper()
    if label not in LABELS:
        raise ValueError(f"invalid label: {label!r}")
    critical_slots = expected_fact.get("critical_slots")
    if not isinstance(critical_slots, dict) or not critical_slots:
        raise ValueError("expected_fact.critical_slots must be a non-empty object")
    slot_names = set(critical_slots)
    result: dict[str, Any] = {"label": label}
    for key in ("preserved_slots", "missing_slots", "conflicting_slots", "hallucinated_facts"):
        items = value.get(key)
        if not isinstance(items, list) or not all(isinstance(item, str) and item.strip() for item in items):
            raise ValueError(f"{key} must be a string list")
        result[key] = list(dict.fromkeys(item.strip() for item in items))
    preserved = set(result["preserved_slots"])
    missing = set(result["missing_slots"])
    conflicting = set(result["conflicting_slots"])
    unknown = (preserved | missing | conflicting) - slot_names
    if unknown:
        raise ValueError(f"unknown critical slots: {sorted(unknown)}")
    overlaps = (preserved & missing) | (preserved & conflicting) | (missing & conflicting)
    if overlaps:
        # Conservative precedence: a conflict is never allowed to remain preserved, and a missing slot is not preserved.
        preserved -= missing | conflicting
        missing -= conflicting
        result["preserved_slots"] = [key for key in critical_slots if key in preserved]
        result["missing_slots"] = [key for key in critical_slots if key in missing]
        result["conflicting_slots"] = [key for key in critical_slots if key in conflicting]
        result["diagnostic_repairs"] = [f"resolved overlap for slot {key}" for key in sorted(overlaps)]
    unmentioned = slot_names - preserved - missing - conflicting
    if unmentioned:
        if label in PASS_LABELS:
            raise ValueError(f"{label} must explicitly preserve every critical slot")
        missing.update(unmentioned)
        result["missing_slots"] = [key for key in critical_slots if key in missing]
    hallucinations = result["hallucinated_facts"]
    if conflicting or hallucinations:
        derived_label = "MISMATCH"
    elif missing:
        derived_label = "PARTIAL_MATCH" if preserved else "MISMATCH"
    elif preserved == slot_names:
        derived_label = label if label in PASS_LABELS else "SEMANTIC_MATCH"
    else:
        derived_label = "MISMATCH"
    if derived_label != label:
        result["reported_label"] = label
        result["label"] = derived_label
        result.setdefault("diagnostic_repairs", []).append(
            f"normalized label from {label} to {derived_label} from slot/hallucination diagnosis"
        )
    reason = str(value.get("reason") or "").strip()
    if not reason:
        reason = "模型未提供文字理由；最终标签由 critical slot、冲突和幻觉诊断确定。"
        result.setdefault("diagnostic_repairs", []).append("filled empty reason with evaluator diagnostic")
    result["reason"] = reason
    return result


def flatten_source_messages(case: dict[str, Any]) -> list[dict[str, Any]]:
    flattened = []
    for session_index, session in enumerate(case["sessions"], 1):
        for message_index, message in enumerate(session["conversation"], 1):
            flattened.append(
                {
                    "session_index": session_index,
                    "message_index": message_index,
                    "started_at": str(session["started_at"]),
                    "role": str(message["role"]),
                    "content": str(message["content"]),
                }
            )
    return flattened


def normalize_text(value: str) -> str:
    return "".join(
        character.lower()
        for character in unicodedata.normalize("NFKC", str(value))
        if not character.isspace() and not unicodedata.category(character).startswith(("P", "S"))
    )


def rule_candidate(generated_summary: str, expected_fact: dict[str, Any]) -> str:
    fact = str(expected_fact["fact"])
    slots = [str(value) for value in expected_fact["critical_slots"].values()]
    if fact in generated_summary and all(value in generated_summary for value in slots):
        return "EXACT_CANDIDATE"
    normalized_summary = normalize_text(generated_summary)
    if all(normalize_text(value) in normalized_summary for value in slots):
        return "SEMANTIC_CANDIDATE"
    return "UNRESOLVED"


def fact_metrics(records: list[dict[str, Any]], *, fact_total: int | None = None) -> dict[str, Any]:
    counts = Counter(str(record.get("final_label") or "") for record in records)
    total = fact_total if fact_total is not None else len(records)
    exact = counts["EXACT_MATCH"]
    semantic = counts["SEMANTIC_MATCH"]
    partial = counts["PARTIAL_MATCH"]
    hallucinations = {
        (str(record.get("case_id") or ""), normalize_text(item))
        for record in records
        for item in record.get("hallucinated_facts") or []
        if normalize_text(item)
    }
    return {
        "exact_match_count": exact,
        "semantic_match_count": semantic,
        "partial_match_count": partial,
        "mismatch_count": counts["MISMATCH"],
        "fact_total": total,
        "strict_accuracy": round((exact + semantic) / total, 4) if total else 0.0,
        "coverage_score": round((exact + semantic + 0.5 * partial) / total, 4) if total else 0.0,
        "hallucination_count": len(hallucinations),
    }


def calibration_metrics(records: list[dict[str, Any]], total: int) -> dict[str, Any]:
    matrix = {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}
    for record in records:
        actual = str(record["expected_label"])
        predicted = str(record["predicted_label"])
        matrix[actual][predicted] += 1
    per_label = {}
    correct = 0
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in LABELS if actual != label)
        fn = sum(matrix[label][predicted] for predicted in LABELS if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(matrix[label].values()),
        }
        correct += tp
    accuracy = correct / total if total else 0.0
    return {
        "confusion_matrix": matrix,
        "per_label": per_label,
        "judge_accuracy": round(accuracy, 4),
        "passed": total == 40 and len(records) == total and accuracy >= 0.9,
    }


def extract_json_object(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model output does not contain a JSON object")
    return text[start : end + 1]


def normalized_hallucinations(records: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (str(record["case_id"]), normalize_text(item))
        for record in records
        for item in record.get("hallucinated_facts") or []
        if normalize_text(item)
    }
