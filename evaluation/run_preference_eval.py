"""Run long-term structured preference memory evaluation.

Usage:
    uv run python evaluation/run_preference_eval.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory import MemoryConfig, MemoryManager  # noqa: E402
from ui.llm import DebugChatLLM  # noqa: E402


EXPECTED_CASE_COUNT = 60
EXPECTED_USER_ID = "user-001"
EXPECTED_DEVICE_ID = "dog-005"
EXPECTED_COUNTS = {
    "occupation": 20,
    "likes": 20,
    "dislikes": 20,
}
PREFERENCE_KEYS = {
    "occupation": "profile.occupation",
    "likes": "preference.likes",
    "dislikes": "preference.dislikes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Memory OS long-term structured preference memory.",
        epilog=(
            "Recommended examples:\n"
            "  uv run python evaluation/run_preference_eval.py --max-cases 3\n"
            "  uv run python evaluation/run_preference_eval.py --type likes --max-cases 3\n"
            "  uv run python evaluation/run_preference_eval.py --case-id pref_001\n"
            "Progress is written to stderr; the final report is written to stdout as JSON."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "datasets" / "preference_memory_probe.jsonl",
        help="Path to the preference-memory JSONL dataset.",
    )
    parser.add_argument(
        "--allow-memory-redis-fallback",
        action="store_true",
        help="Allow in-memory Redis fallback for local development only.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Evaluate only the first N cases after validation. 0 means all cases.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Evaluate only the specified case_id. Can be passed multiple times.",
    )
    parser.add_argument(
        "--type",
        dest="case_type",
        choices=sorted(EXPECTED_COUNTS),
        help="Evaluate only one preference case type.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Write per-case input/output records to this JSONL file. Defaults to evaluation/results/preference_eval_<timestamp>.jsonl.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            validate_case(case, line_number)
            case_id = str(case["case_id"])
            if case_id in seen_case_ids:
                raise ValueError(f"{case_id}: duplicate case_id")
            seen_case_ids.add(case_id)
            cases.append(case)
    validate_distribution(cases)
    return cases


def validate_case(case: dict[str, Any], line_number: int) -> None:
    case_id = case.get("case_id")
    user_id = case.get("user_id")
    device_id = case.get("device_id")
    conversation = case.get("conversation")
    expected = case.get("expected")
    if not all(isinstance(value, str) and value for value in (case_id, user_id, device_id)):
        raise ValueError(f"line {line_number}: case_id/user_id/device_id must be non-empty strings")
    if user_id != EXPECTED_USER_ID:
        raise ValueError(f"{case_id}: user_id must be {EXPECTED_USER_ID}")
    if device_id != EXPECTED_DEVICE_ID:
        raise ValueError(f"{case_id}: device_id must be {EXPECTED_DEVICE_ID}")
    if not isinstance(conversation, list) or len(conversation) < 2 or len(conversation) % 2:
        raise ValueError(f"{case_id}: conversation must contain user/assistant pairs")
    for index in range(0, len(conversation), 2):
        user_msg = conversation[index]
        assistant_msg = conversation[index + 1]
        if not isinstance(user_msg, dict) or not isinstance(assistant_msg, dict):
            raise ValueError(f"{case_id}: conversation messages must be objects")
        if user_msg.get("role") != "user" or assistant_msg.get("role") != "assistant":
            raise ValueError(f"{case_id}: conversation must alternate user then assistant")
        if not str(user_msg.get("content") or "").strip():
            raise ValueError(f"{case_id}: user content cannot be empty")
        if not str(assistant_msg.get("content") or "").strip():
            raise ValueError(f"{case_id}: assistant content cannot be empty")
    if not isinstance(expected, dict):
        raise ValueError(f"{case_id}: expected must be an object")
    field = expected_field(case)
    if field == "occupation" and not isinstance(expected.get("occupation"), str):
        raise ValueError(f"{case_id}: expected.occupation must be a string")
    if field in {"likes", "dislikes"}:
        values = expected.get(field)
        if not isinstance(values, list) or not values or not all(isinstance(item, str) and item for item in values):
            raise ValueError(f"{case_id}: expected.{field} must be a non-empty string list")


def expected_field(case: dict[str, Any]) -> str:
    expected = case.get("expected") or {}
    present: list[str] = []
    if isinstance(expected.get("occupation"), str) and expected.get("occupation"):
        present.append("occupation")
    for field in ("likes", "dislikes"):
        value = expected.get(field)
        if isinstance(value, list) and value:
            present.append(field)
    if len(present) != 1:
        raise ValueError(f"{case.get('case_id')}: expected must contain exactly one target field")
    return present[0]


def expected_values(case: dict[str, Any]) -> list[str]:
    field = expected_field(case)
    expected = case["expected"]
    if field == "occupation":
        return [str(expected["occupation"])]
    return [str(item) for item in expected[field]]


def validate_distribution(cases: list[dict[str, Any]]) -> None:
    if len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError(f"dataset must contain {EXPECTED_CASE_COUNT} cases, got {len(cases)}")
    counts = Counter(expected_field(case) for case in cases)
    for field, expected_count in EXPECTED_COUNTS.items():
        if counts[field] != expected_count:
            raise ValueError(f"dataset must contain {expected_count} {field} cases, got {counts[field]}")


def filter_cases(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = cases
    if args.case_type:
        selected = [case for case in selected if expected_field(case) == args.case_type]
    if args.case_id:
        requested = set(args.case_id)
        selected = [case for case in selected if case["case_id"] in requested]
        found = {case["case_id"] for case in selected}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"unknown case_id: {', '.join(missing)}")
    if args.max_cases:
        selected = selected[: max(0, args.max_cases)]
    return selected


def make_config(temp_root: Path, allow_memory_redis_fallback: bool, redis_prefix: str) -> MemoryConfig:
    config = MemoryConfig.from_env()
    config.data_dir = temp_root
    config.sqlite_path = temp_root / "events.db"
    config.local_long_term_path = temp_root / "long_term.jsonl"
    config.redis_prefix = redis_prefix
    config.redis_allow_memory_fallback = allow_memory_redis_fallback
    return config


def make_llm(config: MemoryConfig) -> DebugChatLLM:
    return DebugChatLLM(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        model=config.llm_model,
        api_key_source=config.llm_api_key_source,
    )


def ensure_redis_available(config: MemoryConfig, allow_memory_redis_fallback: bool) -> None:
    if allow_memory_redis_fallback:
        return
    try:
        import redis  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Python package 'redis' is not installed in this interpreter. "
            f"Current interpreter: {sys.executable}. "
            "Use 'uv run python evaluation/run_preference_eval.py' or install the project dependencies."
        ) from exc
    try:
        client = redis.Redis.from_url(config.redis_url, decode_responses=True)
        client.ping()
    except Exception as exc:
        raise RuntimeError(
            f"Redis service is not reachable at {config.redis_url}. "
            "Start Redis or pass --allow-memory-redis-fallback for local development."
        ) from exc


def cleanup_eval_keys(config: MemoryConfig, redis_prefix: str, manager: MemoryManager | None = None) -> dict[str, Any]:
    pattern = f"{redis_prefix}:*"
    deleted = 0
    errors: list[str] = []
    redis_client = getattr(getattr(manager, "redis", None), "_redis", None)
    if redis_client is None:
        try:
            import redis  # type: ignore

            redis_client = redis.Redis.from_url(config.redis_url, decode_responses=True)
            redis_client.ping()
        except Exception as exc:
            redis_client = None
            errors.append(f"redis cleanup unavailable: {type(exc).__name__}: {exc}")
    if redis_client is not None:
        try:
            batch: list[str] = []
            for key in redis_client.scan_iter(match=pattern, count=200):
                batch.append(str(key))
                if len(batch) >= 200:
                    deleted += int(redis_client.delete(*batch) or 0)
                    batch = []
            if batch:
                deleted += int(redis_client.delete(*batch) or 0)
        except Exception as exc:
            errors.append(f"redis cleanup failed: {type(exc).__name__}: {exc}")
    return {"redis_prefix": redis_prefix, "deleted": deleted, "errors": errors}


def ensure_extractor_configured(manager: MemoryManager) -> None:
    status = manager.preference_extractor.status()
    if not status.get("configured"):
        raise RuntimeError(
            "Preference extractor is not configured. Set PREFERENCE_EXTRACTOR_API_KEY "
            "or DASHSCOPE_API_KEY before running the evaluation."
        )


def ensure_reply_llm_configured(config: MemoryConfig) -> None:
    llm = make_llm(config)
    status = llm.status()
    print(
        f"reply llm model: {status.get('model')} "
        f"base_url: {status.get('base_url')} key_source: {status.get('api_key_source') or '-'}",
        file=sys.stderr,
    )
    if not llm.configured:
        raise RuntimeError(
            "Reply LLM is not configured. Set DASHSCOPE_API_KEY or LLM_API_KEY "
            "before running preference verification."
        )
    config_error = llm._configuration_error()
    if config_error:
        raise RuntimeError(config_error)


def write_case_conversation(manager: MemoryManager, case: dict[str, Any]) -> None:
    session_id: str | None = None
    conversation = case["conversation"]
    for index in range(0, len(conversation), 2):
        request_number = index // 2 + 1
        user_msg = conversation[index]
        assistant_msg = conversation[index + 1]
        result = manager.add_conversation_turn(
            request_id=f"{case['case_id']}-{request_number:02d}",
            user_id=case["user_id"],
            device_id=case["device_id"],
            user_text=str(user_msg["content"]),
            assistant_text=str(assistant_msg["content"]),
            session_id=session_id,
        )
        session_id = str(result["session_id"])


def process_follow_up_jobs(manager: MemoryManager) -> None:
    for _ in range(3):
        result = manager.process_memory_jobs_once(limit=4)
        if not result.get("claimed") and not result.get("recovered_stale"):
            return


@contextmanager
def suppress_memory_job_tracebacks() -> Any:
    logger = logging.getLogger("memory.manager")
    previous_disabled = logger.disabled
    logger.disabled = True
    try:
        yield
    finally:
        logger.disabled = previous_disabled


def preference_blob(preferences: list[dict[str, Any]], field: str) -> str:
    key = PREFERENCE_KEYS[field]
    matching = [item for item in preferences if item.get("preference_key") == key]
    return "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in matching)


def actual_values(preferences: list[dict[str, Any]], field: str) -> list[str]:
    key = PREFERENCE_KEYS[field]
    values: list[str] = []
    for pref in preferences:
        if pref.get("preference_key") != key:
            continue
        text_parts = [str(pref.get("display_text_zh") or "")]
        value_json = pref.get("value_json")
        if isinstance(value_json, dict):
            text_parts.extend(str(value_json.get(name) or "") for name in ("label_zh", "label", "code", "value"))
        values.append(" / ".join(part for part in text_parts if part))
    return values


def compact_text(value: str, limit: int = 220) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit] + "..."


def user_input_text(case: dict[str, Any]) -> str:
    return " | ".join(
        str(item.get("content") or "")
        for item in case.get("conversation") or []
        if item.get("role") == "user"
    )


def verification_question(field: str) -> str:
    if field == "occupation":
        return "根据你的长期记忆，我的职业是什么？"
    if field == "likes":
        return "根据你的长期记忆，我喜欢什么？"
    return "根据你的长期记忆，我不喜欢什么？"


def run_preference_verification(
    manager: MemoryManager,
    config: MemoryConfig,
    case: dict[str, Any],
    field: str,
    expected: list[str],
) -> dict[str, Any]:
    user_card = manager.rebuild_user_card(case["user_id"], case["device_id"]) or {}
    llm = make_llm(config)
    question = verification_question(field)
    messages = llm.build_messages(
        question,
        short_term=[],
        rolling_summary="",
        user_card=user_card,
        latest_action_sequence=None,
    )
    answer, model_info = llm.complete(
        question,
        short_term=[],
        rolling_summary="",
        user_card=user_card,
        latest_action_sequence=None,
    )
    missing = [item for item in expected if str(item) not in answer]
    return {
        "verify_question": question,
        "verify_answer": answer,
        "verify_passed": not missing,
        "verification_reason": "" if not missing else f"answer missing expected: {', '.join(missing)}",
        "model_input_messages": messages,
        "user_card": user_card,
        "reply_llm_model": model_info.get("model") or llm.model,
        "reply_llm_usage": model_info.get("usage") or {},
    }


def evaluate_case(case: dict[str, Any], allow_memory_redis_fallback: bool) -> dict[str, Any]:
    redis_prefix = f"memory-os-pref-eval:{case['case_id']}:{uuid.uuid4().hex}"
    started = time.monotonic()
    record: dict[str, Any] = {
        "case_id": str(case["case_id"]),
        "user_id": str(case["user_id"]),
        "device_id": str(case["device_id"]),
        "field": expected_field(case),
        "expected": expected_values(case),
        "actual": [],
        "conversation": case["conversation"],
        "active_preferences": [],
        "preference_extractor_model": "",
        "reply_llm_model": "",
        "redis_prefix": redis_prefix,
        "sqlite_passed": False,
        "passed": False,
        "reason": "",
        "verify_question": "",
        "verify_answer": "",
        "verify_passed": False,
        "verification_reason": "",
        "model_input_messages": [],
        "user_card": {},
        "reply_llm_usage": {},
        "elapsed_seconds": None,
        "redis_cleanup": {"redis_prefix": redis_prefix, "deleted": 0, "errors": []},
    }
    with tempfile.TemporaryDirectory(prefix=f"memory-os-pref-eval-{case['case_id']}-") as temp_dir:
        config = make_config(Path(temp_dir), allow_memory_redis_fallback, redis_prefix)
        manager = MemoryManager.create(config, start_scheduler=False)
        try:
            ensure_extractor_configured(manager)
            record["preference_extractor_model"] = manager.preference_extractor.model
            manager.delete_user_memory(EXPECTED_USER_ID)
            write_case_conversation(manager, case)
            with suppress_memory_job_tracebacks():
                result = manager.trigger_preference_extraction(
                    case["user_id"],
                    case["device_id"],
                    force_recent=True,
                    recent_user_messages=20,
                )
            process = result.get("process") or {}
            if process.get("failed"):
                errors = process.get("errors") or []
                detail = errors[0].get("error") if errors and isinstance(errors[0], dict) else "unknown error"
                record["reason"] = f"preference extraction failed: {detail}"
                record["verification_reason"] = "skipped because preference extraction failed"
                return record
            with suppress_memory_job_tracebacks():
                process_follow_up_jobs(manager)
            preferences = manager.events.list_preferences(
                case["user_id"],
                status="active",
                limit=100,
                device_id=case["device_id"],
            )
            record["active_preferences"] = preferences
            field = str(record["field"])
            record["actual"] = actual_values(preferences, field)
            blob = preference_blob(preferences, field)
            for expected in record["expected"]:
                if str(expected) not in blob:
                    record["reason"] = f"missing expected {field}: {expected}"
                    record["verification_reason"] = "skipped because SQLite extraction failed"
                    return record
            record["sqlite_passed"] = True
            try:
                verification = run_preference_verification(
                    manager,
                    config,
                    case,
                    field,
                    [str(item) for item in record["expected"]],
                )
                record.update(verification)
            except Exception as exc:
                record["verification_reason"] = f"verification failed: {type(exc).__name__}: {exc}"
            record["passed"] = True
            return record
        except Exception as exc:
            record["reason"] = f"{type(exc).__name__}: {exc}"
            return record
        finally:
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            cleanup = cleanup_eval_keys(config, redis_prefix, manager)
            record["redis_cleanup"] = cleanup
            manager.close()


def preflight_environment(allow_memory_redis_fallback: bool) -> None:
    redis_prefix = f"memory-os-pref-eval:preflight:{uuid.uuid4().hex}"
    with tempfile.TemporaryDirectory(prefix="memory-os-pref-eval-preflight-") as temp_dir:
        config = make_config(Path(temp_dir), allow_memory_redis_fallback, redis_prefix)
        ensure_redis_available(config, allow_memory_redis_fallback)
        manager = MemoryManager.create(config, start_scheduler=False)
        try:
            backend = manager.redis.backend
            if backend != "redis" and not allow_memory_redis_fallback:
                raise RuntimeError(f"preference eval requires Redis backend, got {backend}")
            print(f"memory backend: {backend}", file=sys.stderr)
            status = manager.preference_extractor.status()
            print(
                f"preference extractor model: {status.get('model')} "
                f"base_url: {status.get('base_url')} key_source: {status.get('api_key_source') or '-'}",
                file=sys.stderr,
            )
            ensure_extractor_configured(manager)
            ensure_reply_llm_configured(config)
            manager.delete_user_memory(EXPECTED_USER_ID)
        finally:
            cleanup = cleanup_eval_keys(config, redis_prefix, manager)
            if cleanup["errors"]:
                print(f"preflight cleanup warnings: {cleanup['errors']}", file=sys.stderr)
            manager.close()


def accuracy(passed: int, total: int) -> float:
    return round(passed / total, 4) if total else 0.0


def default_log_file() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "evaluation" / "results" / f"preference_eval_{stamp}.jsonl"


def format_log_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": record.get("case_id"),
        "field": record.get("field"),
        "expected": record.get("expected") or [],
        "actual": record.get("actual") or [],
        "sqlite_passed": bool(record.get("sqlite_passed")),
        "passed": bool(record.get("passed")),
        "reason": record.get("reason") or "",
        "verify_question": record.get("verify_question") or "",
        "verify_answer": record.get("verify_answer") or "",
        "verify_passed": bool(record.get("verify_passed")),
        "verification_reason": record.get("verification_reason") or "",
        "elapsed_seconds": record.get("elapsed_seconds"),
        "redis_cleanup": record.get("redis_cleanup"),
        "debug": {
            "user_id": record.get("user_id"),
            "device_id": record.get("device_id"),
            "conversation": record.get("conversation") or [],
            "active_preferences": record.get("active_preferences") or [],
            "user_card": record.get("user_card") or {},
            "model_input_messages": record.get("model_input_messages") or [],
            "preference_extractor_model": record.get("preference_extractor_model"),
            "reply_llm_model": record.get("reply_llm_model"),
            "reply_llm_usage": record.get("reply_llm_usage") or {},
            "redis_prefix": record.get("redis_prefix"),
        },
    }


def print_case_record(index: int, total: int, record: dict[str, Any]) -> None:
    status = "PASS" if record.get("passed") else "FAIL"
    print(
        f"[{index}/{total}] {record['case_id']} {status} field={record.get('field')} "
        f"sqlite={bool(record.get('sqlite_passed'))} verify={bool(record.get('verify_passed'))} "
        f"elapsed={record.get('elapsed_seconds')}s",
        file=sys.stderr,
        flush=True,
    )
    print(f"  expected: {', '.join(str(item) for item in record.get('expected') or [])}", file=sys.stderr, flush=True)
    actual = record.get("actual") or []
    print(f"  input: {compact_text(user_input_text(record))}", file=sys.stderr, flush=True)
    print(
        f"  extracted: {compact_text(' | '.join(str(item) for item in actual) if actual else '<empty>')}",
        file=sys.stderr,
        flush=True,
    )
    if record.get("verify_question"):
        print(f"  verify_question: {record.get('verify_question')}", file=sys.stderr, flush=True)
        print(f"  verify_answer: {compact_text(str(record.get('verify_answer') or '<empty>'))}", file=sys.stderr, flush=True)
    if not record.get("passed"):
        print(f"  reason: {record.get('reason') or 'unknown failure'}", file=sys.stderr, flush=True)
    if record.get("verification_reason"):
        print(f"  verification_reason: {record.get('verification_reason')}", file=sys.stderr, flush=True)
    cleanup = record.get("redis_cleanup") or {}
    if cleanup.get("errors"):
        print(f"  cleanup_errors: {cleanup.get('errors')}", file=sys.stderr, flush=True)


def failed_case_output(record: dict[str, Any]) -> list[dict[str, Any]]:
    field = str(record["field"])
    actual = record.get("actual") or []
    return [
        {
            "case_id": str(record["case_id"]),
            "field": field,
            "expected": expected,
            "actual": actual,
        }
        for expected in record.get("expected") or []
    ]


def verification_failed_case_output(record: dict[str, Any]) -> list[dict[str, Any]]:
    if record.get("verify_passed") or not record.get("verify_question"):
        return []
    field = str(record["field"])
    answer = str(record.get("verify_answer") or "")
    return [
        {
            "case_id": str(record["case_id"]),
            "field": field,
            "expected": expected,
            "answer": answer,
            "reason": record.get("verification_reason") or "verification failed",
        }
        for expected in record.get("expected") or []
        if str(expected) not in answer
    ]


def run() -> int:
    args = parse_args()
    start_time = time.monotonic()
    try:
        cases = load_dataset(args.dataset)
    except Exception as exc:
        print(f"dataset validation failed: {exc}", file=sys.stderr)
        return 2
    try:
        cases = filter_cases(cases, args)
    except Exception as exc:
        print(f"dataset filter failed: {exc}", file=sys.stderr)
        return 2
    counts = Counter(expected_field(case) for case in cases)
    print(f"case distribution: {dict(sorted(counts.items()))}", file=sys.stderr)
    try:
        preflight_environment(args.allow_memory_redis_fallback)
    except Exception as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 2

    log_file = args.log_file or default_log_file()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"case log: {log_file}", file=sys.stderr)

    totals: dict[str, int] = defaultdict(int)
    passed: dict[str, int] = defaultdict(int)
    failed_cases: list[dict[str, Any]] = []
    verification_total = 0
    verification_passed = 0
    verification_failed_cases: list[dict[str, Any]] = []
    redis_cleanup_errors: list[dict[str, Any]] = []

    total_selected = len(cases)
    with log_file.open("w", encoding="utf-8") as log_handle:
        for index, case in enumerate(cases, 1):
            field = expected_field(case)
            totals[field] += 1
            print(
                f"[{index}/{total_selected}] {case['case_id']} [{field}] running...",
                file=sys.stderr,
                flush=True,
            )
            record = evaluate_case(case, args.allow_memory_redis_fallback)
            log_handle.write(json.dumps(format_log_record(record), ensure_ascii=False, sort_keys=True) + "\n")
            log_handle.flush()
            print_case_record(index, total_selected, record)
            if record.get("passed"):
                passed[field] += 1
            else:
                failed_cases.extend(failed_case_output(record))
            if record.get("verify_question"):
                verification_total += 1
                if record.get("verify_passed"):
                    verification_passed += 1
                else:
                    verification_failed_cases.extend(verification_failed_case_output(record))
            cleanup = record.get("redis_cleanup") or {}
            if cleanup.get("errors"):
                redis_cleanup_errors.append(
                    {
                        "case_id": str(record["case_id"]),
                        "redis_prefix": str(cleanup.get("redis_prefix") or ""),
                        "errors": cleanup.get("errors"),
                    }
                )

    total_cases = len(cases)
    total_passed = sum(passed.values())
    output = {
        "total_cases": total_cases,
        "overall_accuracy": accuracy(total_passed, total_cases),
        "occupation_accuracy": accuracy(passed["occupation"], totals["occupation"]),
        "likes_accuracy": accuracy(passed["likes"], totals["likes"]),
        "dislikes_accuracy": accuracy(passed["dislikes"], totals["dislikes"]),
        "verification_accuracy": accuracy(verification_passed, verification_total),
        "log_file": str(log_file),
        "failed_cases": failed_cases,
        "verification_failed_cases": verification_failed_cases,
    }
    if redis_cleanup_errors:
        output["redis_cleanup_errors"] = redis_cleanup_errors
    print(f"completed in {time.monotonic() - start_time:.2f}s", file=sys.stderr)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
