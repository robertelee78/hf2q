"""Synthetic ordinary-check consumer edges; not real-model parity evidence."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d4_grammar_overlap as overlap
from test_d4_grammar_overlap import attempt, cycle, lane


class OrdinaryCheckTests(unittest.TestCase):
    def analyze(self, record):
        raw = (json.dumps(record) + "\n").encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.jsonl"
            path.write_bytes(raw)
            return overlap.analyze(path, expected_sha256=hashlib.sha256(raw).hexdigest(), scope="head")

    def rejected(self, record):
        with self.assertRaises(ValueError):
            self.analyze(record)

    def test_missing_counter_does_not_infer_ordinary_comparison_from_overlap(self):
        report = self.analyze(cycle())
        self.assertEqual(report["summary"]["active_consumed_overlap_cycles"], 1)
        self.assertEqual(report["summary"]["ordinary_planning_missing_counter_cycles"], 1)
        self.assertFalse(report["cycles"][0]["grammar_ordinary_check"]["comparison_recorded"])
        self.assertFalse(report["claims"]["ordinary_path_runtime_semantic_parity_proven"])

    def test_bonus_row_is_not_bounded_by_consumed_proposal_count(self):
        record = cycle([lane([attempt("divergence_discarded")]), None, None, None])
        record["grammar_ordinary_checked_rows"] = [2, None, None, None]
        report = self.analyze(record)
        self.assertEqual(report["summary"]["ordinary_planning_checked_rows"], 2)
        self.assertEqual(report["summary"]["ordinary_planning_checked_lanes"], 1)
        self.assertEqual(report["summary"]["ordinary_planning_recorded_cycles"], 1)
        self.assertEqual(report["summary"]["active_consumed_overlap_cycles"], 0)
        self.assertFalse(report["claims"]["ordinary_path_runtime_semantic_parity_proven"])

    def test_null_slots_are_observed_but_not_exercised_including_legacy(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                record = cycle(legacy=legacy)
                record["grammar_ordinary_checked_rows"] = [None] * 4
                report = self.analyze(record)
                check = report["cycles"][0]["grammar_ordinary_check"]
                self.assertEqual(check["counter_status"], "observed")
                self.assertEqual(check["checked_rows"], 0)
                self.assertFalse(check["comparison_recorded"])

    def test_malformed_counts_fail_closed(self):
        invalid = [None, "1", [], [1] * 3, [1] * 5]
        invalid += [[value, None, None, None] for value in (0, -1, 1.5, True, "1", 9)]
        for counts in invalid:
            with self.subTest(counts=counts):
                record = cycle()
                record["grammar_ordinary_checked_rows"] = counts
                self.rejected(record)

    def test_nonnull_count_requires_observed_work_even_for_legacy(self):
        for record in (cycle(legacy=True), cycle(no_callback=True)):
            record["grammar_ordinary_checked_rows"] = [1, None, None, None]
            self.rejected(record)

    def test_physical_slot_and_proposal_plus_anchor_budget_are_enforced(self):
        record = cycle([None, None, lane([attempt()]), None])
        record["grammar_ordinary_checked_rows"] = [None, None, 2, None]
        report = self.analyze(record)
        self.assertEqual(report["cycles"][0]["grammar_ordinary_check"]["rows_by_slot"],
                         [None, None, 2, None])
        record["grammar_ordinary_checked_rows"] = [2, None, None, None]
        self.rejected(record)
        record["grammar_ordinary_checked_rows"] = [None, None, 3, None]
        self.rejected(record)

    def test_anchor_only_comparison_is_not_useful_proposal_work(self):
        record = cycle([lane([], actual=0), None, None, None])
        record.update(active_lanes=1, selected_rows=1, accepted_rows=1)
        record["grammar_ordinary_checked_rows"] = [1, None, None, None]
        report = self.analyze(record)
        self.assertEqual(report["summary"]["ordinary_planning_checked_rows"], 1)
        self.assertTrue(report["cycles"][0]["grammar_ordinary_check"]["comparison_recorded"])
        self.assertFalse(report["cycles"][0]["grammar_work"]["useful_work_eligible"])
        self.assertEqual(report["summary"]["active_consumed_overlap_cycles"], 0)


if __name__ == "__main__":
    unittest.main()
