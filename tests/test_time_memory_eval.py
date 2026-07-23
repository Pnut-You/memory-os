from __future__ import annotations

import json
import io
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from evaluation.run_time_memory_eval import (
    COMPLEX_REQUIRED_TAGS,
    EXPECTED_SESSION_COUNTS,
    build_counterfactuals,
    cohort_report,
    evaluate_case,
    load_dataset,
    print_case_result,
    print_final_summary,
    run_counterfactuals,
    validate_case,
    write_review,
)
from evaluation.run_time_memory_judge_calibration import load_calibration_dataset, run_calibration
from evaluation.time_memory_judge import (
    JudgeConfig,
    StrictTimeMemoryJudge,
    calibration_metrics,
    fact_metrics,
    flatten_source_messages,
    rule_candidate,
    validate_judge_result,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "evaluation/datasets/time_memory_probe.jsonl"
CALIBRATION = ROOT / "evaluation/datasets/time_memory_judge_calibration.jsonl"


def valid_result(label, slots):
    names = list(slots)
    if label in {"EXACT_MATCH", "SEMANTIC_MATCH"}:
        preserved, missing, conflicting = names, [], []
    elif label == "PARTIAL_MATCH":
        preserved, missing, conflicting = names[:1], names[1:] or names[:1], [],
        if len(names) == 1:
            preserved = [names[0]]
            missing = [names[0]]
    else:
        preserved, missing, conflicting = names[1:], [], names[:1]
    return {
        "label": label,
        "preserved_slots": preserved,
        "missing_slots": missing,
        "conflicting_slots": conflicting,
        "hallucinated_facts": [],
        "reason": "fixed test verdict",
    }


class CalibrationFakeJudge:
    def judge(self, *, case_id, source_messages, expected_fact, generated_summary):
        del source_messages, generated_summary
        label = next(label for label in ("EXACT_MATCH", "SEMANTIC_MATCH", "PARTIAL_MATCH", "MISMATCH")
                     if label.lower().split("_")[0] in case_id)
        result = valid_result(label, expected_fact["critical_slots"])
        # Calibration Partial rows always contain multiple slots.
        return validate_judge_result(result, expected_fact)


class CapturingJudge:
    def __init__(self):
        self.summaries = []

    def judge(self, *, case_id, source_messages, expected_fact, generated_summary):
        del case_id, source_messages
        self.summaries.append(generated_summary)
        return validate_judge_result(valid_result("SEMANTIC_MATCH", expected_fact["critical_slots"]), expected_fact)


class MismatchJudge:
    def judge(self, *, case_id, source_messages, expected_fact, generated_summary):
        del case_id, source_messages, generated_summary
        return validate_judge_result(valid_result("MISMATCH", expected_fact["critical_slots"]), expected_fact)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def read(self):
        return json.dumps(self.payload, ensure_ascii=False).encode()


class TimeMemoryEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.formal = load_dataset(BASELINE)
        cls.baseline = [row for row in cls.formal if not row.get("difficulty_tags")]
        cls.complex_cases = [row for row in cls.formal if row.get("difficulty_tags")]
        cls.calibration = load_calibration_dataset(CALIBRATION)

    def test_strict_datasets_have_fixed_sizes_and_shapes(self):
        self.assertEqual(len(self.baseline), 100)
        self.assertEqual(len(self.complex_cases), 30)
        self.assertEqual(len(self.formal), 130)
        self.assertEqual(sum(len(case["expected_facts"]) for case in self.baseline), 100)
        self.assertGreaterEqual(sum(len(case["expected_facts"]) for case in self.complex_cases), 60)
        tags = {tag for case in self.complex_cases for tag in case["difficulty_tags"]}
        self.assertTrue(COMPLEX_REQUIRED_TAGS <= tags)
        self.assertEqual(Counter(row["expected_label"] for row in self.calibration), Counter({
            "EXACT_MATCH": 10, "SEMANTIC_MATCH": 10, "PARTIAL_MATCH": 10, "MISMATCH": 10,
        }))
        self.assertEqual(Counter(len(case["sessions"]) for case in self.formal), EXPECTED_SESSION_COUNTS)
        rounds = Counter(
            len(session["conversation"]) // 2
            for case in self.formal
            for session in case["sessions"]
        )
        self.assertEqual(set(rounds), set(range(1, 11)))
        self.assertTrue(all(count > 0 for count in rounds.values()))
        for case in self.formal:
            self.assertEqual(case["session_profile"]["session_count"], len(case["sessions"]))
            self.assertEqual(
                case["session_profile"]["rounds_per_session"],
                [len(session["conversation"]) // 2 for session in case["sessions"]],
            )
            self.assertEqual(case["session_profile"]["source"], "sqlite_full_local_day")

    def test_parameterized_facts_keep_quantities_and_latest_locations_aligned(self):
        by_id = {row["case_id"]: row for row in self.formal}
        self.assertEqual(by_id["time_117"]["expected_facts"][0]["critical_slots"]["quantity"], "九份")
        self.assertEqual(by_id["time_127"]["expected_facts"][0]["critical_slots"]["quantity"], "十一份")
        self.assertEqual(by_id["time_128"]["expected_facts"][0]["critical_slots"]["quantity"], "十五份")
        self.assertEqual(by_id["time_129"]["expected_facts"][1]["critical_slots"]["object"], "西墙插座")
        self.assertEqual(
            by_id["time_130"]["expected_facts"][1]["critical_slots"]["location"],
            "客厅书柜下层",
        )

    def test_old_keyword_schema_is_rejected(self):
        case = dict(self.baseline[0])
        case.pop("expected_facts")
        case["expected"] = {"must_contain": ["备用钥匙"]}
        with self.assertRaisesRegex(ValueError, "expected_facts"):
            validate_case(case, 1)

    def test_source_messages_preserve_role_session_and_full_content(self):
        messages = flatten_source_messages(self.baseline[0])
        self.assertEqual(len(messages), 12)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[0]["session_index"], 1)
        self.assertIn("备用钥匙", messages[0]["content"])

    def test_evaluate_case_passes_generated_summary_to_judge(self):
        judge = CapturingJudge()
        with patch("evaluation.run_time_memory_eval.generate_summary", return_value={
            "generated_summary": "日期总结：备用钥匙在青瓷花瓶旁。", "memory_event_id": 1,
        }):
            record = evaluate_case(self.baseline[0], judge, allow_fallback=True)
        self.assertEqual(judge.summaries, ["日期总结：备用钥匙在青瓷花瓶旁。"])
        self.assertTrue(record["case_pass"])

    def test_judge_result_requires_complete_disjoint_slot_diagnosis(self):
        expected = {"fact": "钥匙在抽屉。", "critical_slots": {"object": "钥匙", "location": "抽屉"}}
        result = validate_judge_result({
            "label": "PARTIAL_MATCH", "preserved_slots": ["object"], "missing_slots": ["location"],
            "conflicting_slots": [], "hallucinated_facts": [], "reason": "位置缺失",
        }, expected)
        self.assertEqual(result["label"], "PARTIAL_MATCH")
        repaired = validate_judge_result({**result, "label": "SEMANTIC_MATCH"}, expected)
        self.assertEqual(repaired["reported_label"], "SEMANTIC_MATCH")
        self.assertEqual(repaired["label"], "PARTIAL_MATCH")

    def test_strict_judge_payload_does_not_include_human_label(self):
        captured = {}
        expected = {"fact": "温度是十三度。", "critical_slots": {"quantity": "十三度"}}
        response_result = valid_result("SEMANTIC_MATCH", expected["critical_slots"])

        def fake_urlopen(request, timeout):
            captured["request"] = json.loads(request.data.decode())
            captured["timeout"] = timeout
            return FakeResponse({"choices": [{"message": {"content": json.dumps(response_result, ensure_ascii=False)}}]})

        judge = StrictTimeMemoryJudge(JudgeConfig("key", "https://example.invalid/v1", "model"))
        with patch("evaluation.time_memory_judge.urllib.request.urlopen", side_effect=fake_urlopen):
            result = judge.judge(case_id="cal_semantic_001", source_messages=[{"role": "user", "content": "十三度"}],
                                 expected_fact=expected, generated_summary="13℃")
        user_payload = json.loads(captured["request"]["messages"][1]["content"])
        self.assertNotIn("expected_label", user_payload)
        self.assertEqual(captured["request"]["temperature"], 0)
        self.assertEqual(result["label"], "SEMANTIC_MATCH")

    def test_calibration_gate_and_metrics(self):
        console = io.StringIO()
        with redirect_stderr(console):
            report = run_calibration(CalibrationFakeJudge(), self.calibration)
        self.assertTrue(report["passed"])
        self.assertEqual(report["judge_accuracy"], 1.0)
        output = console.getvalue()
        self.assertIn("[CAL 02/40] cal_exact_002", output)
        self.assertIn("expected=EXACT_MATCH  predicted=EXACT_MATCH  PASS", output)
        self.assertIn("JUDGE CALIBRATION SUMMARY", output)
        self.assertIn("Judge Accuracy: 100.00%", output)
        for label in report["per_label"].values():
            self.assertEqual(label["precision"], 1.0)
            self.assertEqual(label["recall"], 1.0)
            self.assertEqual(label["f1"], 1.0)
        bad = [{"expected_label": "EXACT_MATCH", "predicted_label": "MISMATCH"} for _ in range(40)]
        self.assertFalse(calibration_metrics(bad, 40)["passed"])

    def test_fact_metrics_do_not_count_partial_as_strictly_correct(self):
        rows = [
            {"case_id": "a", "final_label": "EXACT_MATCH", "hallucinated_facts": []},
            {"case_id": "b", "final_label": "SEMANTIC_MATCH", "hallucinated_facts": []},
            {"case_id": "c", "final_label": "PARTIAL_MATCH", "hallucinated_facts": []},
            {"case_id": "d", "final_label": "MISMATCH", "hallucinated_facts": ["新增人物", "新增人物"]},
        ]
        report = fact_metrics(rows)
        self.assertEqual(report["strict_accuracy"], 0.5)
        self.assertEqual(report["coverage_score"], 0.625)
        self.assertEqual(report["hallucination_count"], 1)

    def test_case_pass_requires_every_fact_to_be_strictly_correct(self):
        records = [{
            "cohort": "formal", "case_id": "x", "expected_facts": [{}, {}], "case_pass": False,
            "evaluation_errors": [], "fact_results": [
                {"case_id": "x", "final_label": "EXACT_MATCH", "hallucinated_facts": []},
                {"case_id": "x", "final_label": "PARTIAL_MATCH", "hallucinated_facts": []},
            ],
        }]
        report = cohort_report(records, "formal")
        self.assertEqual(report["case_pass_count"], 0)
        self.assertEqual(report["partial_match_count"], 1)

    def test_rule_result_is_diagnostic_only(self):
        expected = {"fact": "温度是十三度。", "critical_slots": {"quantity": "十三度"}}
        self.assertEqual(rule_candidate("温度是十三度。", expected), "EXACT_CANDIDATE")
        self.assertEqual(rule_candidate("温度为13℃。", expected), "UNRESOLVED")

    def test_review_contains_all_failures_and_sampled_passes(self):
        base = {
            "case_id": "x", "source_messages": [], "expected_fact": {"fact": "事实", "critical_slots": {"x": "x"}},
            "generated_summary": "摘要", "rule_result": "UNRESOLVED", "preserved_slots": [], "missing_slots": ["x"],
            "conflicting_slots": [], "hallucinated_facts": [], "reason": "missing",
        }
        rows = [{**base, "case_id": f"p{i}", "final_label": "PARTIAL_MATCH"} for i in range(2)]
        rows += [{**base, "case_id": f"e{i}", "final_label": "EXACT_MATCH"} for i in range(12)]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "review.jsonl"
            count = write_review(path, rows, 1)
            output = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(count, 12)
        self.assertEqual(sum(item["judge_result"] == "PARTIAL_MATCH" for item in output), 2)
        self.assertEqual(sum(item["judge_result"] == "EXACT_MATCH" for item in output), 10)

    def test_console_case_output_expands_only_failure_diagnostics(self):
        record = {
            "case_id": "time_x", "expected_facts": [{}, {}], "generated_summary": "摘要", "case_pass": False,
            "evaluation_errors": [], "fact_results": [
                {
                    "fact_index": 1, "final_label": "SEMANTIC_MATCH",
                    "expected_fact": {"fact": "十三度", "critical_slots": {"quantity": "十三度"}},
                    "preserved_slots": ["quantity"], "missing_slots": [], "conflicting_slots": [],
                    "hallucinated_facts": [], "reason": "合理改写",
                },
                {
                    "fact_index": 2, "final_label": "PARTIAL_MATCH",
                    "expected_fact": {"fact": "钥匙在二层抽屉", "critical_slots": {"location": "二层抽屉"}},
                    "preserved_slots": [], "missing_slots": ["location"], "conflicting_slots": [],
                    "hallucinated_facts": [], "reason": "位置缺失",
                },
            ],
        }
        console = io.StringIO()
        with redirect_stderr(console):
            print_case_result(record, 1, 130)
        output = console.getvalue()
        self.assertIn("[CASE 001/130] time_x", output)
        self.assertIn("[FACT 1] SEMANTIC_MATCH  PASS", output)
        self.assertIn("[FACT 2] PARTIAL_MATCH  FAIL", output)
        self.assertIn("missing: location", output)
        self.assertIn("reason: 位置缺失", output)
        self.assertEqual(output.count("reason:"), 1)

    def test_counterfactual_and_final_console_output_stay_on_stderr(self):
        facts = []
        for index in range(30):
            facts.append({
                "case_id": f"c{index}", "fact_index": 1, "final_label": "EXACT_MATCH",
                "source_messages": [], "expected_fact": {"fact": "周叔在仓库。", "critical_slots": {
                    "person": f"周叔{index}", "location": f"仓库{index}", "quantity": f"{index + 1}份",
                }},
                "generated_summary": f"周叔{index}在仓库{index}处理了{index + 1}份材料。",
            })
        stderr = io.StringIO()
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir, redirect_stderr(stderr), redirect_stdout(stdout):
            counter = run_counterfactuals(facts, MismatchJudge(), Path(temp_dir) / "counter.jsonl", 1)
            print_final_summary({
                "formal": {
                    "exact_match_count": 1, "semantic_match_count": 1, "partial_match_count": 1,
                    "mismatch_count": 1, "evaluation_errors": [], "fact_total": 4,
                    "strict_accuracy": 0.5, "coverage_score": 0.625, "hallucination_count": 1,
                    "case_pass_count": 1, "case_total": 2, "case_pass_rate": 0.5,
                },
                "counterfactual": counter,
                "review_file": "review.jsonl",
            })
        output = stderr.getvalue()
        self.assertIn("[CF 01/21]", output)
        self.assertIn("predicted=MISMATCH  REJECTED", output)
        self.assertIn("TIME MEMORY FINAL SUMMARY", output)
        self.assertIn("Strict Accuracy", output)
        self.assertEqual(stdout.getvalue(), "")

    def test_counterfactual_builder_changes_critical_slots(self):
        facts = []
        for index in range(30):
            facts.append({
                "case_id": f"c{index}", "fact_index": 1, "final_label": "EXACT_MATCH",
                "source_messages": [], "expected_fact": {"fact": "周叔在仓库。", "critical_slots": {
                    "person": f"周叔{index}", "location": f"仓库{index}", "quantity": f"{index + 1}份",
                }},
                "generated_summary": f"周叔{index}在仓库{index}处理了{index + 1}份材料。",
            })
        counterfactuals = build_counterfactuals(facts, 1)
        self.assertGreaterEqual(len(counterfactuals), 20)
        self.assertTrue(all(item["counterfactual_summary"] != item["generated_summary"] for item in counterfactuals))
        for item in counterfactuals:
            if item["mutation_type"] == "slot_deletion":
                self.assertNotIn(item["original_value"], item["counterfactual_summary"])
            else:
                self.assertIn("反事实最终更正", item["counterfactual_summary"])


if __name__ == "__main__":
    unittest.main()
