from __future__ import annotations

import json
import unittest
from pathlib import Path

from evaluation.run_short_term_eval import DEFAULT_DATASET, load_dataset, normalize_match_text


DATASET_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "datasets"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ShortTermEvaluationTests(unittest.TestCase):
    def test_default_dataset_combines_both_200_case_sources(self):
        combined = load_dataset(DEFAULT_DATASET)
        short = load_jsonl(DATASET_DIR / "short_term_memory_probe_2_10.jsonl")
        long = load_jsonl(DATASET_DIR / "short_term_memory_probe_10_20.jsonl")

        self.assertEqual(len(combined), 400)
        self.assertEqual([case["case_id"] for case in combined], [f"short_{i:03d}" for i in range(1, 401)])
        self.assertEqual(
            [(case["conversation"], case["probe_question"], case["expected"]) for case in combined[:200]],
            [(case["conversation"], case["probe_question"], case["expected"]) for case in short],
        )
        self.assertEqual(
            [(case["conversation"], case["probe_question"], case["expected"]) for case in combined[200:]],
            [(case["conversation"], case["probe_question"], case["expected"]) for case in long],
        )

    def test_match_normalization_ignores_surface_only_differences(self):
        expected = normalize_match_text("帮我看一下地面的空间")
        actual = normalize_match_text("好的，帮我看一下地面空间！")
        self.assertIn(expected, actual)
        self.assertEqual(normalize_match_text("走得快"), normalize_match_text("走的快"))

    def test_match_normalization_keeps_substantive_differences(self):
        expected = normalize_match_text("提醒我拿雨伞")
        actual = normalize_match_text("提醒我拿快递")
        self.assertNotIn(expected, actual)


if __name__ == "__main__":
    unittest.main()
