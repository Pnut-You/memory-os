"""Run calibrated, strict Time Memory fact-retention evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import tempfile
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.run_time_memory_judge_calibration import (  # noqa: E402
    DEFAULT_DATASET as DEFAULT_CALIBRATION_DATASET,
    load_calibration_dataset,
    run_calibration,
)
from evaluation.time_memory_judge import (  # noqa: E402
    PASS_LABELS,
    JudgeConfig,
    JudgeError,
    StrictTimeMemoryJudge,
    fact_metrics,
    flatten_source_messages,
    normalize_text,
    rule_candidate,
)
from memory import MemoryConfig, MemoryManager  # noqa: E402
from memory.summarizer import Summarizer  # noqa: E402


BASELINE_COUNTS = {"activity": 20, "location_state": 20, "person_item": 20, "task": 20, "time_quantity": 20}
COMPLEX_REQUIRED_TAGS = {
    "状态更新", "跨Session", "相似位置", "否定", "计划完成", "多人物", "相似物品", "多数字",
}
EXPECTED_DEVICE_ID = "dog-006"
EXPECTED_SESSION_COUNTS = Counter({1: 22, 2: 22, 3: 22, 4: 22, 5: 21, 6: 21})


class StrictDailySummarizer(Summarizer):
    def summarize_daily(self, messages, memory_date, *, max_chars=None):
        if not messages:
            return ""
        if not self.api_key:
            raise RuntimeError("daily summary LLM is not configured")
        return self._call_daily_llm(messages, memory_date, max_chars or self.MAX_DAILY_SUMMARY_CHARS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run calibrated strict Time Memory evaluation.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "evaluation/datasets/time_memory_probe.jsonl")
    parser.add_argument("--calibration-dataset", type=Path, default=DEFAULT_CALIBRATION_DATASET)
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--allow-memory-redis-fallback", action="store_true")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--review-file", type=Path, default=None)
    parser.add_argument("--counterfactual-file", type=Path, default=None)
    parser.add_argument("--reuse-summaries-from", type=Path, default=None)
    parser.add_argument("--review-seed", type=int, default=20260722)
    return parser.parse_args()


def load_dataset(path: Path, *, require_full: bool = True) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            validate_case(row, line_number)
            case_id = str(row["case_id"])
            if case_id in seen:
                raise ValueError(f"{case_id}: duplicate case_id")
            seen.add(case_id)
            row["cohort"] = "formal"
            rows.append(row)
    if require_full:
        basic = [row for row in rows if not row.get("difficulty_tags")]
        complex_cases = [row for row in rows if row.get("difficulty_tags")]
        counts = Counter(str(row["category"]) for row in basic)
        tags = {str(tag) for row in complex_cases for tag in row.get("difficulty_tags") or []}
        if len(rows) != 130 or len(basic) != 100 or len(complex_cases) != 30 or counts != Counter(BASELINE_COUNTS):
            raise ValueError(f"formal dataset must contain 100 upgraded + 30 difficult cases, got {len(rows)}")
        if not COMPLEX_REQUIRED_TAGS <= tags or any(not 2 <= len(row["expected_facts"]) <= 3 for row in complex_cases):
            raise ValueError(f"difficult cases are incomplete; missing tags={sorted(COMPLEX_REQUIRED_TAGS-tags)}")
        session_counts = Counter(len(row["sessions"]) for row in rows)
        round_counts = Counter(
            len(session["conversation"]) // 2 for row in rows for session in row["sessions"]
        )
        if session_counts != EXPECTED_SESSION_COUNTS:
            raise ValueError(f"unexpected session distribution: {dict(session_counts)}")
        if set(round_counts) != set(range(1, 11)):
            raise ValueError(f"session rounds must cover 1-10, got {sorted(round_counts)}")
    return rows


def validate_case(case: dict[str, Any], line_number: int) -> None:
    for key in ("case_id", "category", "user_id", "device_id", "memory_date"):
        if not isinstance(case.get(key), str) or not case[key]:
            raise ValueError(f"line {line_number}: {key} must be a non-empty string")
    if case["device_id"] != EXPECTED_DEVICE_ID:
        raise ValueError(f"{case['case_id']}: device_id must be {EXPECTED_DEVICE_ID}")
    datetime.fromisoformat(str(case["memory_date"]))
    sessions = case.get("sessions")
    if not isinstance(sessions, list) or not 1 <= len(sessions) <= 6:
        raise ValueError(f"{case['case_id']}: sessions must contain 1-6 items")
    turns = 0
    previous = ""
    for session in sessions:
        started_at = str(session.get("started_at") or "")
        parsed = datetime.fromisoformat(started_at)
        if parsed.tzinfo is None or started_at[:10] != case["memory_date"] or (previous and started_at <= previous):
            raise ValueError(f"{case['case_id']}: sessions must be ordered, timezone-aware, and inside memory_date")
        previous = started_at
        conversation = session.get("conversation")
        if not isinstance(conversation, list) or not conversation or len(conversation) % 2:
            raise ValueError(f"{case['case_id']}: each conversation must contain complete pairs")
        session_rounds = len(conversation) // 2
        if not 1 <= session_rounds <= 10:
            raise ValueError(f"{case['case_id']}: each session must contain 1-10 turns")
        for index, message in enumerate(conversation):
            role = "user" if index % 2 == 0 else "assistant"
            if not isinstance(message, dict) or message.get("role") != role or not str(message.get("content") or "").strip():
                raise ValueError(f"{case['case_id']}: conversation roles/content are invalid")
        turns += len(conversation) // 2
    if not 1 <= turns <= 60:
        raise ValueError(f"{case['case_id']}: total turns must be 1-60")
    profile = case.get("session_profile")
    actual_rounds = [len(session["conversation"]) // 2 for session in sessions]
    if not isinstance(profile, dict) or profile.get("source") != "sqlite_full_local_day":
        raise ValueError(f"{case['case_id']}: session_profile must declare SQLite full-day source")
    if profile.get("session_count") != len(sessions) or profile.get("rounds_per_session") != actual_rounds:
        raise ValueError(f"{case['case_id']}: session_profile does not match sessions")
    expected_facts = case.get("expected_facts")
    if not isinstance(expected_facts, list) or not 1 <= len(expected_facts) <= 3:
        raise ValueError(f"{case['case_id']}: expected_facts must contain 1-3 facts")
    for expected_fact in expected_facts:
        if not isinstance(expected_fact, dict) or not str(expected_fact.get("fact") or "").strip():
            raise ValueError(f"{case['case_id']}: every expected fact needs full fact text")
        slots = expected_fact.get("critical_slots")
        if not isinstance(slots, dict) or not slots or not all(
            isinstance(key, str) and key and isinstance(value, str) and value.strip() for key, value in slots.items()
        ):
            raise ValueError(f"{case['case_id']}: critical_slots must be a non-empty string map")


def make_config(temp_root: Path, allow_fallback: bool, redis_prefix: str) -> MemoryConfig:
    config = MemoryConfig.from_env()
    config.data_dir = temp_root
    config.sqlite_path = temp_root / "events.db"
    config.local_long_term_path = temp_root / "long_term.jsonl"
    config.redis_prefix = redis_prefix
    config.redis_allow_memory_fallback = allow_fallback
    return config


def ensure_environment(allow_fallback: bool) -> MemoryConfig:
    config = MemoryConfig.from_env()
    if not config.llm_api_key:
        raise RuntimeError("DASHSCOPE_API_KEY or LLM_API_KEY is required")
    if not allow_fallback:
        import redis  # type: ignore
        client = redis.Redis.from_url(config.redis_url, decode_responses=True)
        client.ping()
    return config


def cleanup_eval_keys(config: MemoryConfig, redis_prefix: str, manager: MemoryManager | None) -> dict[str, Any]:
    client = getattr(getattr(manager, "redis", None), "_redis", None)
    deleted = 0
    errors = []
    if client is not None:
        try:
            keys = [str(key) for key in client.scan_iter(match=f"{redis_prefix}:*", count=200)]
            if keys:
                deleted = int(client.delete(*keys) or 0)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return {"deleted": deleted, "errors": errors, "redis_prefix": redis_prefix}


def write_case_sessions(manager: MemoryManager, case: dict[str, Any]) -> None:
    request_number = 0
    for session in case["sessions"]:
        session_id = None
        started_at = datetime.fromisoformat(str(session["started_at"]))
        for index in range(0, len(session["conversation"]), 2):
            request_number += 1
            result = manager.add_conversation_turn(
                request_id=f"{case['case_id']}-{request_number:02d}",
                user_id=str(case["user_id"]), device_id=str(case["device_id"]),
                user_text=str(session["conversation"][index]["content"]),
                assistant_text=str(session["conversation"][index + 1]["content"]),
                timestamp=started_at.replace(microsecond=min(999999, index * 1000)).isoformat(),
                session_id=session_id,
            )
            session_id = str(result["session_id"])


class suppress_job_tracebacks:
    def __enter__(self):
        self.logger = logging.getLogger("memory.manager")
        self.disabled = self.logger.disabled
        self.logger.disabled = True

    def __exit__(self, exc_type, exc, traceback):
        self.logger.disabled = self.disabled


def generate_summary(case: dict[str, Any], allow_fallback: bool) -> dict[str, Any]:
    prefix = f"memory-os-time-eval:{case['case_id']}:{uuid.uuid4().hex}"
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"time-eval-{case['case_id']}-") as temp_dir:
        config = make_config(Path(temp_dir), allow_fallback, prefix)
        manager = MemoryManager.create(config, start_scheduler=False)
        manager.summarizer = StrictDailySummarizer(config.llm_api_key, config.llm_base_url, config.llm_model)
        try:
            write_case_sessions(manager, case)
            with suppress_job_tracebacks():
                trigger = manager.trigger_daily_extraction(str(case["user_id"]), str(case["device_id"]), str(case["memory_date"]))
            process = trigger.get("process") or {}
            if process.get("failed") or process.get("errors"):
                raise RuntimeError(f"daily extraction failed: {process.get('errors')}")
            memories = [
                item for item in manager.events.list_time_memories(str(case["user_id"]), str(case["device_id"]), limit=10)
                if str((item.get("payload_json") or {}).get("memory_date") or "") == str(case["memory_date"])
            ]
            if len(memories) != 1:
                raise RuntimeError(f"expected one time_memory, got {len(memories)}")
            return {
                "generated_summary": str(memories[0].get("content") or ""),
                "memory_event_id": int(memories[0]["id"]),
                "trigger": trigger,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        finally:
            cleanup = cleanup_eval_keys(config, prefix, manager)
            manager.close()
            if cleanup["errors"]:
                print(f"cleanup warning for {case['case_id']}: {cleanup['errors']}", file=sys.stderr)


def evaluate_case(
    case: dict[str, Any], judge: StrictTimeMemoryJudge, allow_fallback: bool, generated_summary: str | None = None,
) -> dict[str, Any]:
    source_messages = flatten_source_messages(case)
    case_record = {
        "case_id": case["case_id"], "cohort": case["cohort"], "source_messages": source_messages,
        "expected_facts": case["expected_facts"], "generated_summary": "", "fact_results": [],
        "case_pass": False, "evaluation_errors": [],
    }
    if generated_summary is not None:
        case_record["generated_summary"] = generated_summary
        case_record["summary_reused"] = True
    else:
        try:
            generation = generate_summary(case, allow_fallback)
            case_record.update(generation)
        except Exception as exc:
            case_record["evaluation_errors"].append(f"summary generation: {type(exc).__name__}: {exc}")
            return case_record
    for fact_index, expected_fact in enumerate(case["expected_facts"], 1):
        base = {
            "case_id": case["case_id"], "cohort": case["cohort"], "fact_index": fact_index,
            "source_messages": source_messages, "expected_fact": expected_fact,
            "generated_summary": case_record["generated_summary"],
            "rule_result": rule_candidate(case_record["generated_summary"], expected_fact),
        }
        try:
            result = judge.judge(
                case_id=str(case["case_id"]), source_messages=source_messages,
                expected_fact=expected_fact, generated_summary=str(case_record["generated_summary"]),
            )
            base.update(result)
            base["judge_label"] = result["label"]
            base["final_label"] = result["label"]
        except JudgeError as exc:
            base["evaluation_error"] = str(exc)
            base["judge_attempts"] = exc.attempts
            case_record["evaluation_errors"].append(f"fact {fact_index}: {exc}")
        case_record["fact_results"].append(base)
    hallucinations = list(dict.fromkeys(
        item for result in case_record["fact_results"] for item in result.get("hallucinated_facts") or []
    ))
    if hallucinations:
        for result in case_record["fact_results"]:
            if "final_label" in result:
                result["final_label"] = "MISMATCH"
                result["case_hallucinations"] = hallucinations
                result["hallucinated_facts"] = hallucinations
                result["reason"] = f"{result.get('reason', '')}；同一摘要存在无来源事实：{'；'.join(hallucinations)}"
    case_record["case_pass"] = not case_record["evaluation_errors"] and all(
        result.get("final_label") in PASS_LABELS for result in case_record["fact_results"]
    )
    return case_record


def _console_text(value: Any, limit: int = 320) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def print_case_result(record: dict[str, Any], index: int, total: int) -> None:
    """Print concise per-fact output and expand only non-passing results."""
    summary_length = len(str(record.get("generated_summary") or ""))
    print(
        f"[CASE {index:03d}/{total:03d}] {record['case_id']}  "
        f"facts={len(record['expected_facts'])}  summary_chars={summary_length}",
        file=sys.stderr,
    )
    if not record["fact_results"]:
        for error in record.get("evaluation_errors") or ["no fact result was produced"]:
            print(f"  EVALUATION_ERROR: {_console_text(error)}", file=sys.stderr)
    for fact in record["fact_results"]:
        label = str(fact.get("final_label") or "EVALUATION_ERROR")
        status = "PASS" if label in PASS_LABELS else "FAIL"
        print(f"  [FACT {fact['fact_index']}] {label}  {status}", file=sys.stderr)
        print(f"    expected: {_console_text(fact['expected_fact']['fact'])}", file=sys.stderr)
        preserved = fact.get("preserved_slots") or []
        print(f"    preserved: {', '.join(preserved) if preserved else '-'}", file=sys.stderr)
        if label not in PASS_LABELS:
            missing = fact.get("missing_slots") or []
            conflicting = fact.get("conflicting_slots") or []
            hallucinations = fact.get("hallucinated_facts") or []
            print(f"    missing: {', '.join(missing) if missing else '-'}", file=sys.stderr)
            print(f"    conflicting: {', '.join(conflicting) if conflicting else '-'}", file=sys.stderr)
            if hallucinations:
                print(f"    hallucinations: {len(hallucinations)}", file=sys.stderr)
                for item in hallucinations:
                    print(f"      - {_console_text(item, 240)}", file=sys.stderr)
            detail = fact.get("evaluation_error") or fact.get("reason") or "no diagnostic reason"
            print(f"    reason: {_console_text(detail)}", file=sys.stderr)
    result = "PASS" if record["case_pass"] else "FAIL"
    print(f"  CASE RESULT: {result}\n", file=sys.stderr, flush=True)


def cohort_report(case_records: list[dict[str, Any]], cohort: str) -> dict[str, Any]:
    selected = [record for record in case_records if record["cohort"] == cohort]
    facts = [fact for record in selected for fact in record["fact_results"] if fact.get("final_label")]
    total = sum(len(record["expected_facts"]) for record in selected)
    report = fact_metrics(facts, fact_total=total)
    errors = [
        {"case_id": record["case_id"], "errors": record["evaluation_errors"]}
        for record in selected if record["evaluation_errors"]
    ]
    report.update({
        "case_total": len(selected), "case_pass_count": sum(bool(record["case_pass"]) for record in selected),
        "case_pass_rate": round(sum(bool(record["case_pass"]) for record in selected) / len(selected), 4) if selected else 0.0,
        "evaluation_errors": errors,
    })
    return report


def print_final_summary(report: dict[str, Any]) -> None:
    formal = report["formal"]
    counterfactual = report["counterfactual"]
    print("\n=== TIME MEMORY FINAL SUMMARY ===", file=sys.stderr)
    rows = (
        ("Exact Match", formal["exact_match_count"]),
        ("Semantic Match", formal["semantic_match_count"]),
        ("Partial Match", formal["partial_match_count"]),
        ("Mismatch", formal["mismatch_count"]),
        ("Evaluation Errors", len(formal["evaluation_errors"])),
        ("Fact Total", formal["fact_total"]),
        ("Strict Accuracy", f"{formal['strict_accuracy'] * 100:.2f}%"),
        ("Coverage Score", f"{formal['coverage_score'] * 100:.2f}%"),
        ("Hallucinations", formal["hallucination_count"]),
        ("Case PASS", f"{formal['case_pass_count']}/{formal['case_total']} ({formal['case_pass_rate'] * 100:.2f}%)"),
        (
            "Counterfactual Rejection",
            f"{counterfactual['rejected']}/{counterfactual['counterfactual_total']} "
            f"({counterfactual['counterfactual_rejection_rate'] * 100:.2f}%)",
        ),
    )
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{width}} : {value}", file=sys.stderr)
    print(
        f"Counterfactual Gate{' ' * (width - len('Counterfactual Gate'))} : "
        f"{'PASSED' if counterfactual['passed'] else 'FAILED'} (required >= 95.00%)",
        file=sys.stderr,
    )
    print(f"Review File{' ' * (width - len('Review File'))} : {report['review_file']}\n", file=sys.stderr, flush=True)


def write_review(path: Path, fact_records: list[dict[str, Any]], seed: int) -> int:
    rng = random.Random(seed)
    selected = []
    for label in ("PARTIAL_MATCH", "MISMATCH"):
        for record in fact_records:
            if record.get("final_label") == label:
                selected.append((record, f"all_{label.lower()}"))
    for label in ("SEMANTIC_MATCH", "EXACT_MATCH"):
        candidates = [record for record in fact_records if record.get("final_label") == label]
        for record in rng.sample(candidates, min(10, len(candidates))):
            selected.append((record, f"sample_{label.lower()}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record, selection_reason in selected:
            output = {
                "case_id": record["case_id"], "source_messages": record["source_messages"],
                "expected_fact": record["expected_fact"]["fact"],
                "critical_slots": record["expected_fact"]["critical_slots"],
                "generated_summary": record["generated_summary"], "rule_result": record["rule_result"],
                "judge_result": record.get("final_label", "EVALUATION_ERROR"),
                "preserved_slots": record.get("preserved_slots", []), "missing_slots": record.get("missing_slots", []),
                "conflicting_slots": record.get("conflicting_slots", []),
                "hallucinated_facts": record.get("hallucinated_facts", []), "reason": record.get("reason", ""),
                "selection_reason": selection_reason,
            }
            handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")
    return len(selected)


MUTATION_KEYS = {
    "person": {"person", "owner", "recipient", "beneficiary"},
    "location": {"location", "destination", "direction"},
    "quantity": {"quantity", "duration"},
    "time": {"time", "deadline"},
    "polarity": {"state", "observation_status", "event_status", "relation_status"},
    "completion": {"completion_status", "target_state"},
}
MUTATION_REPLACEMENTS = {
    "person": "另一位无关人员", "location": "另一个错误地点", "quantity": "错误数量",
    "time": "错误时间", "polarity": "相反状态", "completion": "相反完成状态",
}


def build_counterfactuals(facts: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    correct = [record for record in facts if record.get("final_label") in PASS_LABELS]
    rng.shuffle(correct)
    output = []
    used = set()
    for mutation_type, keys in MUTATION_KEYS.items():
        for record in correct:
            identity = (record["case_id"], record["fact_index"])
            if identity in used:
                continue
            slot = next(
                ((key, str(value)) for key, value in record["expected_fact"]["critical_slots"].items()
                 if key in keys and str(value) in record["generated_summary"]),
                None,
            )
            if not slot:
                continue
            key, value = slot
            replacement = MUTATION_REPLACEMENTS[mutation_type]
            mutated = record["generated_summary"].replace(value, replacement)
            mutated += (
                f"\n【反事实最终更正】Expected Fact 的 {key} 不是“{value}”，"
                f"而是“{replacement}”；此前与此冲突的表述全部作废。"
            )
            output.append({**record, "mutation_type": mutation_type, "mutated_slot": key, "original_value": value,
                           "counterfactual_summary": mutated})
            used.add(identity)
            if sum(item["mutation_type"] == mutation_type for item in output) == 3:
                break
    for record in correct:
        if len(output) >= 21:
            break
        identity = (record["case_id"], record["fact_index"])
        if identity in used:
            continue
        slot = next(((key, str(value)) for key, value in record["expected_fact"]["critical_slots"].items()
                     if str(value) in record["generated_summary"]), None)
        if slot:
            key, value = slot
            retained = [
                f"{slot_key}={slot_value}"
                for slot_key, slot_value in record["expected_fact"]["critical_slots"].items()
                if slot_key != key
            ]
            mutated = f"时间记忆仅保留：{'；'.join(retained)}。未记录 {key} 字段。"
            output.append({**record, "mutation_type": "slot_deletion", "mutated_slot": key, "original_value": value,
                           "counterfactual_summary": mutated})
            used.add(identity)
    return output[:21]


def run_counterfactuals(
    records: list[dict[str, Any]], judge: StrictTimeMemoryJudge, path: Path, seed: int,
) -> dict[str, Any]:
    candidates = build_counterfactuals(records, seed)
    errors = []
    completed = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(candidates, 1):
            prefix = f"[CF {index:02d}/{len(candidates):02d}] {item['case_id']}"
            print(
                f"{prefix}  mutation={item['mutation_type']}/{item['mutated_slot']}  RUNNING",
                file=sys.stderr,
                flush=True,
            )
            try:
                result = judge.judge(
                    case_id=f"{item['case_id']}-counterfactual-{index}", source_messages=item["source_messages"],
                    expected_fact=item["expected_fact"], generated_summary=item["counterfactual_summary"],
                )
                output = {"case_id": item["case_id"], "fact_index": item["fact_index"],
                          "mutation_type": item["mutation_type"], "mutated_slot": item["mutated_slot"],
                          "original_summary": item["generated_summary"],
                          "counterfactual_summary": item["counterfactual_summary"], "judge_result": result}
                completed.append(output)
                rejected = result["label"] in {"PARTIAL_MATCH", "MISMATCH"}
                print(
                    f"{prefix}  mutation={item['mutation_type']}/{item['mutated_slot']}  "
                    f"predicted={result['label']}  {'REJECTED' if rejected else 'ACCEPTED'}",
                    file=sys.stderr,
                    flush=True,
                )
            except JudgeError as exc:
                output = {"case_id": item["case_id"], "error": str(exc)}
                errors.append(output)
                print(
                    f"{prefix}  mutation={item['mutation_type']}/{item['mutated_slot']}  "
                    "predicted=EVALUATION_ERROR  FAIL",
                    file=sys.stderr,
                )
                print(f"  error: {_console_text(exc)}", file=sys.stderr, flush=True)
            handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
    rejected = sum(item["judge_result"]["label"] in {"PARTIAL_MATCH", "MISMATCH"} for item in completed)
    rate = rejected / len(candidates) if candidates else 0.0
    return {
        "counterfactual_total": len(candidates), "completed": len(completed), "rejected": rejected,
        "counterfactual_rejection_rate": round(rate, 4), "target": 0.95,
        "passed": len(candidates) >= 20 and not errors and rate >= 0.95,
        "evaluation_errors": errors + ([] if len(candidates) >= 20 else [{"error": "fewer than 20 counterfactuals"}]),
        "output_file": str(path),
    }


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    try:
        config = ensure_environment(args.allow_memory_redis_fallback)
        judge = StrictTimeMemoryJudge(JudgeConfig(config.llm_api_key, config.llm_base_url, config.llm_model))
        calibration_rows = load_calibration_dataset(args.calibration_dataset)
    except Exception as exc:
        print(f"preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = PROJECT_ROOT / "evaluation/results"
    calibration_path = results_dir / f"time_memory_calibration_{stamp}.jsonl"
    calibration = run_calibration(judge, calibration_rows, predictions_path=calibration_path)
    if args.calibration_only or not calibration["passed"]:
        print(json.dumps({"calibration": calibration}, ensure_ascii=False, indent=2))
        return 0 if calibration["passed"] else 3
    cases = load_dataset(args.dataset, require_full=not (args.case_id or args.max_cases))
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in requested]
        missing = requested - {str(case["case_id"]) for case in cases}
        if missing:
            raise ValueError(f"unknown case ids: {sorted(missing)}")
    if args.max_cases:
        cases = cases[: args.max_cases]
    summary_cache: dict[str, str] = {}
    if args.reuse_summaries_from:
        with args.reuse_summaries_from.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                cached = json.loads(line)
                summary = str(cached.get("generated_summary") or "")
                if summary:
                    cached_id = str(cached["case_id"])
                    summary_cache[cached_id] = summary
        missing_summaries = [str(case["case_id"]) for case in cases if str(case["case_id"]) not in summary_cache]
        if missing_summaries:
            raise ValueError(f"reuse summary log is missing cases: {missing_summaries[:10]}")
    log_path = args.log_file or results_dir / f"time_memory_eval_{stamp}.jsonl"
    review_path = args.review_file or results_dir / f"time_memory_review_{stamp}.jsonl"
    counter_path = args.counterfactual_file or results_dir / f"time_memory_counterfactual_{stamp}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    case_records = []
    with log_path.open("w", encoding="utf-8") as handle:
        for index, case in enumerate(cases, 1):
            print(f"[CASE {index:03d}/{len(cases):03d}] {case['case_id']}  RUNNING", file=sys.stderr, flush=True)
            record = evaluate_case(
                case, judge, args.allow_memory_redis_fallback,
                generated_summary=summary_cache.get(str(case["case_id"])) if summary_cache else None,
            )
            case_records.append(record)
            print_case_result(record, index, len(cases))
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
    facts = [fact for record in case_records for fact in record["fact_results"]]
    review_count = write_review(review_path, facts, args.review_seed)
    counterfactual = run_counterfactuals(facts, judge, counter_path, args.review_seed)
    report = {
        "calibration": calibration,
        "formal": cohort_report(case_records, "formal"),
        "counterfactual": counterfactual,
        "log_file": str(log_path), "review_file": str(review_path), "review_record_count": review_count,
        "elapsed_seconds": round(time.monotonic() - started, 3), "llm_model": config.llm_model,
    }
    print_final_summary(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    formal_errors = report["formal"]["evaluation_errors"]
    return 0 if not formal_errors and counterfactual["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
