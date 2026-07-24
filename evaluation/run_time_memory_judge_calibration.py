"""Calibrate the strict Time Memory fact judge against fixed human labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.time_memory_judge import (  # noqa: E402
    LABELS,
    JudgeConfig,
    JudgeError,
    StrictTimeMemoryJudge,
    calibration_metrics,
)
from memory import MemoryConfig  # noqa: E402


DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "datasets" / "time_memory_judge_calibration.jsonl"


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_calibration_summary(metrics: dict[str, Any]) -> None:
    """Print a human-readable calibration report without changing stdout JSON."""
    print("\n=== JUDGE CALIBRATION SUMMARY ===", file=sys.stderr)
    print("human_label      EXACT  SEMANTIC  PARTIAL  MISMATCH", file=sys.stderr)
    matrix = metrics["confusion_matrix"]
    for label in LABELS:
        values = "  ".join(f"{matrix[label][predicted]:>8}" for predicted in LABELS)
        print(f"{label:<16}{values}", file=sys.stderr)
    print("\nlabel             precision  recall   f1       support", file=sys.stderr)
    for label in LABELS:
        row = metrics["per_label"][label]
        print(
            f"{label:<17} {_percent(row['precision']):>9}  {_percent(row['recall']):>7}  "
            f"{_percent(row['f1']):>7}  {row['support']:>7}",
            file=sys.stderr,
        )
    gate = "PASSED" if metrics["passed"] else "FAILED"
    print(
        f"Judge Accuracy: {_percent(metrics['judge_accuracy'])}  Gate: {gate} (required >= 90.00%)\n",
        file=sys.stderr,
        flush=True,
    )


def load_calibration_dataset(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row.get("case_id") or "")
            if not case_id or case_id in seen:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate case_id")
            seen.add(case_id)
            if row.get("expected_label") not in LABELS:
                raise ValueError(f"{case_id}: invalid expected_label")
            if not isinstance(row.get("source_messages"), list) or not row["source_messages"]:
                raise ValueError(f"{case_id}: source_messages must be non-empty")
            expected_fact = row.get("expected_fact")
            if not isinstance(expected_fact, dict) or not str(expected_fact.get("fact") or "").strip():
                raise ValueError(f"{case_id}: expected_fact.fact is required")
            if not isinstance(expected_fact.get("critical_slots"), dict) or not expected_fact["critical_slots"]:
                raise ValueError(f"{case_id}: critical_slots must be non-empty")
            if not str(row.get("generated_summary") or "").strip():
                raise ValueError(f"{case_id}: generated_summary is required")
            rows.append(row)
    counts = Counter(str(row["expected_label"]) for row in rows)
    if len(rows) != 40 or counts != Counter({label: 10 for label in LABELS}):
        raise ValueError(f"calibration dataset must have 10 rows per label, got {dict(counts)}")
    return rows


def run_calibration(
    judge: StrictTimeMemoryJudge,
    rows: list[dict[str, Any]],
    *,
    predictions_path: Path | None = None,
) -> dict[str, Any]:
    predictions = []
    errors = []
    handle = None
    if predictions_path:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        handle = predictions_path.open("w", encoding="utf-8")
    try:
        for index, row in enumerate(rows, 1):
            case_id = str(row["case_id"])
            prefix = f"[CAL {index:02d}/{len(rows):02d}] {case_id}"
            print(f"{prefix}  RUNNING", file=sys.stderr, flush=True)
            try:
                result = judge.judge(
                    case_id=case_id,
                    source_messages=row["source_messages"],
                    expected_fact=row["expected_fact"],
                    generated_summary=str(row["generated_summary"]),
                )
                prediction = {
                    "case_id": case_id,
                    "expected_label": row["expected_label"],
                    "predicted_label": result["label"],
                    "judge_result": result,
                }
                predictions.append(prediction)
                status = "PASS" if row["expected_label"] == result["label"] else "FAIL"
                print(
                    f"{prefix}  expected={row['expected_label']}  predicted={result['label']}  "
                    f"{status}  {result.get('elapsed_seconds', 0):.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                if status == "FAIL":
                    print(f"  reason: {str(result.get('reason') or '').strip()}", file=sys.stderr)
            except JudgeError as exc:
                prediction = {"case_id": case_id, "expected_label": row["expected_label"], "error": str(exc)}
                errors.append(prediction)
                print(
                    f"{prefix}  expected={row['expected_label']}  predicted=EVALUATION_ERROR  FAIL",
                    file=sys.stderr,
                )
                print(f"  error: {exc}", file=sys.stderr, flush=True)
            if handle:
                handle.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
    finally:
        if handle:
            handle.close()
    metrics = calibration_metrics(predictions, len(rows))
    metrics.update(
        {
            "total": len(rows),
            "completed": len(predictions),
            "evaluation_errors": errors,
            "passed": bool(metrics["passed"] and not errors),
            "predictions_path": str(predictions_path) if predictions_path else "",
        }
    )
    print_calibration_summary(metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the strict Time Memory Judge.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = MemoryConfig.from_env()
    if not config.llm_api_key:
        print("calibration failed: DASHSCOPE_API_KEY or LLM_API_KEY is required", file=sys.stderr)
        return 2
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or PROJECT_ROOT / "evaluation" / "results" / f"time_memory_calibration_{stamp}.jsonl"
    try:
        rows = load_calibration_dataset(args.dataset)
        judge = StrictTimeMemoryJudge(JudgeConfig(config.llm_api_key, config.llm_base_url, config.llm_model))
        report = run_calibration(judge, rows, predictions_path=output)
    except Exception as exc:
        print(f"calibration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
