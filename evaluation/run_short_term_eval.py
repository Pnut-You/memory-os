"""Run short-term memory retention evaluation.

Usage:
    uv run python evaluation/run_short_term_eval.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import unicodedata
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory import MemoryConfig, MemoryManager  # noqa: E402
from ui.llm import DebugChatLLM  # noqa: E402
from ui.router import MemoryDebugRouter  # noqa: E402


DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "short_term_memory_probe.jsonl"
EXPECTED_DEFAULT_CASE_COUNT = 400
MIN_TURNS = 2
MAX_TURNS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Memory OS short-term session memory.",
        epilog=(
            "Recommended examples:\n"
            "  .venv/bin/python evaluation/run_short_term_eval.py --max-cases 3 --allow-memory-redis-fallback\n"
            "  .venv/bin/python evaluation/run_short_term_eval.py --case-id short_001 --allow-memory-redis-fallback\n"
            "  .venv/bin/python evaluation/run_short_term_eval.py --dataset evaluation/datasets/short_term_memory_probe_2_10.jsonl\n"
            "  .venv/bin/python evaluation/run_short_term_eval.py --dataset evaluation/datasets/short_term_memory_probe_10_20.jsonl\n"
            "Progress is written to stderr; the final report is written to stdout as JSON."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the short-term memory JSONL dataset.",
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
        "--log-file",
        type=Path,
        default=None,
        help="Write per-case input/output records to this JSONL file. Defaults to evaluation/results/short_term_eval_<timestamp>.jsonl.",
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
    if path.resolve() == DEFAULT_DATASET.resolve() and len(cases) != EXPECTED_DEFAULT_CASE_COUNT:
        raise ValueError(
            f"default dataset must contain {EXPECTED_DEFAULT_CASE_COUNT} cases, got {len(cases)}"
        )
    if not cases:
        raise ValueError("dataset must contain at least one case")
    validate_turn_distribution(cases)
    return cases


def validate_case(case: dict[str, Any], line_number: int) -> None:
    case_id = case.get("case_id")
    user_id = case.get("user_id")
    device_id = case.get("device_id")
    session_id = case.get("session_id")
    conversation = case.get("conversation")
    probe_question = case.get("probe_question")
    expected = case.get("expected")
    if not all(isinstance(value, str) and value for value in (case_id, user_id, device_id, session_id, probe_question)):
        raise ValueError(
            f"line {line_number}: case_id/user_id/device_id/session_id/probe_question must be non-empty strings"
        )
    if not isinstance(conversation, list) or len(conversation) < 2 or len(conversation) % 2:
        raise ValueError(f"{case_id}: conversation must contain user/assistant pairs")
    turn_count = len(conversation) // 2
    if turn_count < MIN_TURNS or turn_count > MAX_TURNS:
        raise ValueError(f"{case_id}: conversation must contain {MIN_TURNS}-{MAX_TURNS} turns, got {turn_count}")
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
    must_contain = expected.get("must_contain")
    if not isinstance(must_contain, list) or not must_contain:
        raise ValueError(f"{case_id}: expected.must_contain must be a non-empty list")
    if not all(isinstance(item, str) and item for item in must_contain):
        raise ValueError(f"{case_id}: expected.must_contain items must be non-empty strings")


def validate_turn_distribution(cases: list[dict[str, Any]]) -> None:
    counts = turn_distribution(cases)
    min_turns = min(counts)
    max_turns = max(counts)
    missing = [turns for turns in range(min_turns, max_turns + 1) if counts[turns] == 0]
    if missing:
        raise ValueError(f"dataset turn counts must be continuous from {min_turns} to {max_turns}; missing {missing}")


def turn_distribution(cases: list[dict[str, Any]]) -> Counter[int]:
    return Counter(len(case["conversation"]) // 2 for case in cases)


def fact_location(case: dict[str, Any]) -> dict[str, int | None]:
    conversation = case["conversation"]
    turn_count = len(conversation) // 2
    expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}
    keywords = expected.get("must_contain") if isinstance(expected, dict) else []
    fact_turn: int | None = None
    if isinstance(keywords, list):
        for turn_index in range(0, len(conversation), 2):
            content = str(conversation[turn_index].get("content") or "")
            if all(str(keyword) in content for keyword in keywords):
                fact_turn = turn_index // 2 + 1
                break
    return {
        "fact_turn": fact_turn,
        "turns_after_fact": turn_count - fact_turn if fact_turn is not None else None,
    }


def filter_cases(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = cases
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
    config.preference_extractor_enabled = False
    return config


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

    memory_backend = getattr(getattr(manager, "redis", None), "_memory", None)
    memory_expires = getattr(getattr(manager, "redis", None), "_expires", None)
    if isinstance(memory_backend, dict):
        for key in list(memory_backend):
            if str(key).startswith(f"{redis_prefix}:"):
                memory_backend.pop(key, None)
                if isinstance(memory_expires, dict):
                    memory_expires.pop(key, None)
                deleted += 1
    return {"redis_prefix": redis_prefix, "deleted": deleted, "errors": errors}


def ensure_redis_available(config: MemoryConfig, allow_memory_redis_fallback: bool) -> None:
    if allow_memory_redis_fallback:
        return
    try:
        import redis  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Python package 'redis' is not installed in this interpreter. "
            f"Current interpreter: {sys.executable}. "
            "Use '.venv/bin/python evaluation/run_short_term_eval.py' or install the project dependencies."
        ) from exc
    try:
        client = redis.Redis.from_url(config.redis_url, decode_responses=True)
        client.ping()
    except Exception as exc:
        raise RuntimeError(
            f"Redis service is not reachable at {config.redis_url}. "
            "Start Redis or pass --allow-memory-redis-fallback for local development."
        ) from exc


def make_llm(config: MemoryConfig) -> DebugChatLLM:
    return DebugChatLLM(
        config.llm_api_key,
        config.llm_base_url,
        config.llm_model,
        config.llm_api_key_source,
    )


def preflight_environment(allow_memory_redis_fallback: bool) -> dict[str, Any]:
    redis_prefix = f"memory-os-short-eval:preflight:{uuid.uuid4().hex}"
    config: MemoryConfig | None = None
    with tempfile.TemporaryDirectory(prefix="memory-os-short-eval-preflight-") as temp_dir:
        config = make_config(Path(temp_dir), allow_memory_redis_fallback, redis_prefix)
        ensure_redis_available(config, allow_memory_redis_fallback)
        manager = MemoryManager.create(config, start_scheduler=False)
        try:
            backend = manager.redis.backend
            if backend != "redis" and not allow_memory_redis_fallback:
                raise RuntimeError(f"short-term eval requires Redis backend, got {backend}")
            print(f"memory backend: {backend}", file=sys.stderr)
            llm = make_llm(config)
            status = llm.status()
            print(
                f"llm model: {status.get('model')} base_url: {status.get('base_url')} "
                f"key_source: {status.get('api_key_source') or '-'}",
                file=sys.stderr,
            )
            if not llm.configured:
                raise RuntimeError("short-term eval requires configured reply LLM")
            config_error = llm._configuration_error()
            if config_error:
                raise RuntimeError(config_error)
            return {"redis_backend": backend, "llm_model": status.get("model")}
        finally:
            cleanup = cleanup_eval_keys(config, redis_prefix, manager)
            if cleanup["errors"]:
                print(f"preflight cleanup warnings: {cleanup['errors']}", file=sys.stderr)
            manager.close()


def write_case_conversation(manager: MemoryManager, case: dict[str, Any]) -> str:
    conversation = case["conversation"]
    session_id: str | None = None
    for index in range(0, len(conversation), 2):
        user_msg = conversation[index]
        assistant_msg = conversation[index + 1]
        request_number = index // 2 + 1
        result = manager.add_conversation_turn(
            request_id=f"{case['case_id']}-{request_number:02d}",
            user_id=case["user_id"],
            device_id=case["device_id"],
            user_text=str(user_msg["content"]),
            assistant_text=str(assistant_msg["content"]),
            session_id=session_id,
        )
        session_id = str(result["session_id"])
    if not session_id:
        raise RuntimeError(f"{case['case_id']}: no session was created")
    return session_id


def evaluate_case(case: dict[str, Any], allow_memory_redis_fallback: bool) -> dict[str, Any]:
    redis_prefix = f"memory-os-short-eval:{case['case_id']}:{uuid.uuid4().hex}"
    started = time.monotonic()
    record: dict[str, Any] = {
        "case_id": str(case["case_id"]),
        "user_id": str(case["user_id"]),
        "device_id": str(case["device_id"]),
        "dataset_session_id": str(case["session_id"]),
        "turn_count": len(case["conversation"]) // 2,
        **fact_location(case),
        "conversation": case["conversation"],
        "probe_question": str(case["probe_question"]),
        "expected": case["expected"],
        "assistant_reply": "",
        "passed": False,
        "reason": "",
        "llm_called": False,
        "llm_model": "",
        "redis_prefix": redis_prefix,
        "redis_cleanup": {"redis_prefix": redis_prefix, "deleted": 0, "errors": []},
    }
    with tempfile.TemporaryDirectory(prefix=f"memory-os-short-eval-{case['case_id']}-") as temp_dir:
        config = make_config(Path(temp_dir), allow_memory_redis_fallback, redis_prefix)
        record["llm_model"] = config.llm_model
        manager = MemoryManager.create(config, start_scheduler=False)
        try:
            actual_session_id = write_case_conversation(manager, case)
            record["actual_session_id"] = actual_session_id
            pre_probe_context = manager.get_conversation_context(
                case["user_id"],
                case["device_id"],
                session_id=actual_session_id,
            )
            if pre_probe_context.get("session_id") != actual_session_id:
                record["reason"] = f"wrong session context: {pre_probe_context.get('session_id')}"
                return record
            router = MemoryDebugRouter(manager, make_llm(config))
            record["llm_called"] = True
            response = router.submit(
                case["user_id"],
                case["device_id"],
                str(case["probe_question"]),
                debug=True,
            )
            reply = str(response.get("assistant_reply") or "")
            record["assistant_reply"] = reply
            debug = response.get("debug") if isinstance(response.get("debug"), dict) else {}
            record["model_input_messages"] = debug.get("prompt_messages") or []
            record["request_id"] = response.get("request_id")
            normalized_reply = normalize_match_text(reply)
            record["normalized_reply"] = normalized_reply
            record["normalized_expected"] = [
                normalize_match_text(str(keyword)) for keyword in case["expected"]["must_contain"]
            ]
            for keyword, normalized_keyword in zip(
                case["expected"]["must_contain"], record["normalized_expected"]
            ):
                if normalized_keyword not in normalized_reply:
                    record["reason"] = (
                        f"missing normalized keyword: {keyword} "
                        f"(normalized={normalized_keyword!r}, reply={normalized_reply!r})"
                    )
                    return record
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


def accuracy(passed: int, total: int) -> float:
    return round(passed / total, 4) if total else 0.0


def normalize_match_text(value: str) -> str:
    """Normalize harmless Chinese surface differences for deterministic evaluation."""
    return "".join(
        character.lower()
        for character in unicodedata.normalize("NFKC", str(value))
        if character not in {"的", "地", "得"}
        and not character.isspace()
        and not unicodedata.category(character).startswith(("P", "S"))
    )


def default_log_file() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "evaluation" / "results" / f"short_term_eval_{stamp}.jsonl"


def cleanup_all_eval_keys(allow_memory_redis_fallback: bool) -> dict[str, Any]:
    config = MemoryConfig.from_env()
    config.redis_allow_memory_fallback = allow_memory_redis_fallback
    return cleanup_eval_keys(config, "memory-os-short-eval")


def compact_text(value: str, limit: int = 200) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit] + "..."


def expected_text(record: dict[str, Any]) -> str:
    return ",".join(str(item) for item in (record.get("expected") or {}).get("must_contain", []))


def format_log_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": record.get("case_id"),
        "turn_count": record.get("turn_count"),
        "fact_turn": record.get("fact_turn"),
        "turns_after_fact": record.get("turns_after_fact"),
        "expected_text": expected_text(record),
        "probe_question": record.get("probe_question"),
        "assistant_reply": record.get("assistant_reply"),
        "normalized_expected": record.get("normalized_expected") or [],
        "normalized_reply": record.get("normalized_reply") or "",
        "passed": bool(record.get("passed")),
        "reason": record.get("reason") or "",
        "elapsed_seconds": record.get("elapsed_seconds"),
        "redis_cleanup": record.get("redis_cleanup"),
        "debug": {
            "user_id": record.get("user_id"),
            "device_id": record.get("device_id"),
            "dataset_session_id": record.get("dataset_session_id"),
            "actual_session_id": record.get("actual_session_id"),
            "request_id": record.get("request_id"),
            "llm_called": bool(record.get("llm_called")),
            "llm_model": record.get("llm_model"),
            "redis_prefix": record.get("redis_prefix"),
            "conversation": record.get("conversation") or [],
            "model_input_messages": record.get("model_input_messages") or [],
        },
    }


def print_case_record(index: int, total: int, record: dict[str, Any]) -> None:
    status = "PASS" if record.get("passed") else "FAIL"
    print(
        f"[{index}/{total}] {record['case_id']} {status} "
        f"turns={record.get('turn_count')} fact_turn={record.get('fact_turn')} "
        f"after_fact={record.get('turns_after_fact')} elapsed={record.get('elapsed_seconds')}s",
        file=sys.stderr,
        flush=True,
    )
    print(f"  expected: {expected_text(record)}", file=sys.stderr, flush=True)
    print(f"  question: {record.get('probe_question')}", file=sys.stderr, flush=True)
    print(f"  answer: {compact_text(str(record.get('assistant_reply') or ''))}", file=sys.stderr, flush=True)
    if not record.get("passed"):
        print(f"  reason: {record.get('reason') or 'unknown failure'}", file=sys.stderr, flush=True)
    cleanup = record.get("redis_cleanup") or {}
    if cleanup.get("errors"):
        print(f"  cleanup_errors: {cleanup.get('errors')}", file=sys.stderr, flush=True)


def run() -> int:
    args = parse_args()
    start_time = time.monotonic()
    initial_cleanup = cleanup_all_eval_keys(args.allow_memory_redis_fallback)
    if initial_cleanup["deleted"] or initial_cleanup["errors"]:
        print(
            f"initial redis cleanup: deleted={initial_cleanup['deleted']} errors={initial_cleanup['errors']}",
            file=sys.stderr,
        )
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
    counts = turn_distribution(cases)
    print(f"turn distribution: {dict(sorted(counts.items()))}", file=sys.stderr)
    try:
        preflight = preflight_environment(args.allow_memory_redis_fallback)
    except Exception as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 2

    log_file = args.log_file or default_log_file()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"case log: {log_file}", file=sys.stderr)

    total_cases = len(cases)
    passed = 0
    failed_cases: list[dict[str, str]] = []
    llm_calls = 0
    redis_cleaned_cases = 0
    redis_cleanup_errors: list[dict[str, Any]] = []
    with log_file.open("w", encoding="utf-8") as log_handle:
        for index, case in enumerate(cases, 1):
            print(f"[{index}/{total_cases}] {case['case_id']} running...", file=sys.stderr, flush=True)
            record = evaluate_case(case, args.allow_memory_redis_fallback)
            log_handle.write(json.dumps(format_log_record(record), ensure_ascii=False, sort_keys=True) + "\n")
            log_handle.flush()
            print_case_record(index, total_cases, record)
            if record.get("passed"):
                passed += 1
            else:
                failed_cases.append(
                    {
                        "case_id": str(record["case_id"]),
                        "reason": str(record.get("reason") or "unknown failure"),
                        "reply": str(record.get("assistant_reply") or ""),
                    }
                )
            if record.get("llm_called"):
                llm_calls += 1
            cleanup = record.get("redis_cleanup") or {}
            if not cleanup.get("errors"):
                redis_cleaned_cases += 1
            else:
                redis_cleanup_errors.append(
                    {
                        "case_id": str(record["case_id"]),
                        "redis_prefix": str(cleanup.get("redis_prefix") or ""),
                        "errors": cleanup.get("errors"),
                    }
                )

    elapsed_seconds = time.monotonic() - start_time
    final_cleanup = cleanup_all_eval_keys(args.allow_memory_redis_fallback)
    if final_cleanup["deleted"] or final_cleanup["errors"]:
        print(
            f"final redis cleanup: deleted={final_cleanup['deleted']} errors={final_cleanup['errors']}",
            file=sys.stderr,
        )
    if final_cleanup["errors"]:
        redis_cleanup_errors.append(
            {
                "case_id": "__final_cleanup__",
                "redis_prefix": "memory-os-short-eval",
                "errors": final_cleanup["errors"],
            }
        )
    elif final_cleanup["deleted"]:
        redis_cleaned_cases += 1
    output = {
        "total_cases": total_cases,
        "accuracy": accuracy(passed, total_cases),
        "llm_model": preflight.get("llm_model"),
        "llm_calls": llm_calls,
        "log_file": str(log_file),
        "redis_cleaned_cases": redis_cleaned_cases,
        "redis_cleanup_errors": redis_cleanup_errors,
        "failed_cases": failed_cases,
    }
    print(f"completed in {elapsed_seconds:.2f}s", file=sys.stderr)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
