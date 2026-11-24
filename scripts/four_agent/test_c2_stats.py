"""Pure consumer-visible C2 uncertainty controls; these are not model passes."""
import copy
import math
import unittest

import c2_stats as c2
import q3_stats
from c2_metrics import APPEND, DIVERGENT, divergent_metrics
from test_c2_metrics import summary, turn


def evidence(rate=0.8):
    return {"scope": c2.ELIGIBILITY_SCOPE, "rate": rate,
            "not_eligible": {"anchor_capture": 0, "anchor_restore": 0}}


def blocks(ratios, kind="divergent"):
    names = c2.REQUIRED[kind]
    rows = []
    for i, ratio in enumerate(ratios, 1):
        identity = f"williams-{i}"
        rows.append({"block_id": identity,
                     "subject": {"block_id": identity, "outcome": "Pass",
                                 "metrics": dict.fromkeys(names, ratio), "eligibility": evidence(),
                                 "peak_footprint_bytes": 100, "calibration_peak_footprint_bytes": 100},
                     "reference": {"block_id": identity, "outcome": "Pass",
                                   "metrics": dict.fromkeys(names, 1.0), "eligibility": evidence()}})
    return rows


class PairedAcceptance(unittest.TestCase):
    def test_consumer_extracted_divergent_turns_drive_the_cohort_effect(self):
        rows = blocks([0.8] * 5)
        for row in rows:
            subject, reference = summary(), summary()
            for agent in subject["agents"].values():
                agent["turns"][3], agent["turns"][5] = turn(3, 0.8), turn(5, 0.7)
            row["subject"]["metrics"] = divergent_metrics(subject)
            row["reference"]["metrics"] = divergent_metrics(reference)
        result = c2.campaign("divergent", rows)
        self.assertEqual((result["status"], result["stop_reason"]), ("Pass", "precision"))
        self.assertEqual(result["effects"][DIVERGENT]["blocks"], 5)
        self.assertAlmostEqual(result["effects"][DIVERGENT]["ratio_ci95"][1], 0.8)
        self.assertEqual(result["runs"][0]["subject"]["metrics"]["slots"]["agent-0"]["turn_ttft_s"],
                         {"3": 0.8, "5": 0.7})

    def test_no_early_peek_pass_before_five_and_no_post_stop_extension(self):
        result = c2.campaign("divergent", blocks([0.8] * 4))
        self.assertEqual(result["status"], "Continue")
        self.assertIsNone(result["stop_reason"])
        self.assertIsNone(result["checks"][DIVERGENT])
        self.assertIsNone(result["checks"]["eligibility"])
        with self.assertRaisesRegex(ValueError, "beyond preregistered stop"):
            c2.campaign("divergent", blocks([0.8] * 6))

    def test_precision_continues_then_max_twenty_keeps_k1_when_bar_unmet(self):
        ratios = [0.5, 1.5] * 10
        self.assertEqual(c2.campaign("divergent", blocks(ratios[:5]))["status"], "Continue")
        result = c2.campaign("divergent", blocks(ratios))
        self.assertEqual((result["status"], result["stop_reason"]), ("Fail", "max_blocks"))
        self.assertGreater(result["effects"][DIVERGENT]["half_width_pp"], 2.5)
        self.assertFalse(result["checks"][DIVERGENT])
        with self.assertRaisesRegex(ValueError, "at most 20"):
            c2.campaign("divergent", blocks(ratios * 2))

    def test_divergent_interval_touching_one_fails_but_strict_gain_passes(self):
        self.assertEqual(c2.campaign("divergent", blocks([1] * 5))["status"], "Fail")
        self.assertEqual(c2.campaign("divergent", blocks([0.99] * 5))["status"], "Pass")

    def test_all_three_append_upper_bounds_not_q3_lower_regression_rule(self):
        uncertain = [1.01, 1.03, 1.01, 1.03, 1.01]
        for failing in APPEND:
            rows = blocks([1] * 5, "append")
            for row, ratio in zip(rows, uncertain):
                row["subject"]["metrics"][failing] = ratio
            result = c2.campaign("append", rows)
            effect = result["effects"][failing]
            with self.subTest(metric=failing):
                self.assertFalse(q3_stats.regression(effect))
                self.assertLess(effect["ratio_ci95"][0], 1.025)
                self.assertGreater(effect["ratio_ci95"][1], 1.025)
                self.assertEqual((result["status"], result["stop_reason"]), ("Fail", "precision"))
                self.assertFalse(result["checks"][failing])
                self.assertTrue(all(result["checks"][key] for key in APPEND if key != failing))
        self.assertEqual(c2.campaign("append", blocks([1.025] * 5, "append"))["status"], "Pass")

    def test_one_wide_append_contrast_prevents_precision_stop(self):
        rows = blocks([1] * 5, "append")
        for row, ratio in zip(rows, [0.5, 1.5, 0.5, 1.5, 0.5]):
            row["subject"]["metrics"][APPEND[-1]] = ratio
        self.assertEqual(c2.campaign("append", rows)["status"], "Continue")

    def test_eligibility_acceptance_is_separate_from_precision_stopping(self):
        rows = blocks([0.8] * 5)
        for row, rate in zip(rows, [0.2, 0.9, 0.2, 0.9, 0.2]):
            row["subject"]["eligibility"]["rate"] = rate
            row["reference"]["eligibility"]["rate"] = 0.4
        result = c2.campaign("divergent", rows)
        self.assertEqual((result["status"], result["stop_reason"]), ("Pass", "precision"))
        self.assertGreater(result["eligibility_effect"]["half_width_pp"], 2.5)
        self.assertEqual(result["eligibility_effect"]["direction"], "higher_is_better")
        for rate, passed in ((0.79, False), (0.8, True), (0.81, True)):
            rows = blocks([0.8] * 5)
            for row in rows:
                row["subject"]["eligibility"]["rate"] = rate
            result = c2.campaign("divergent", rows)
            self.assertEqual(result["checks"]["eligibility"], passed)

    def test_zero_in_either_arm_is_retained_not_epsilon_or_skipped(self):
        for arm in ("subject", "reference"):
            rows = blocks([0.8] * 5)
            rows[2][arm]["eligibility"]["rate"] = 0
            result = c2.campaign("divergent", rows)
            with self.subTest(arm=arm):
                self.assertEqual(result["status"], "Fail")
                self.assertEqual(result["blocks"], 5)
                self.assertEqual(result["zero_eligibility_blocks"], ["williams-3"])
                self.assertIsNone(result["eligibility_effect"])
                self.assertIn("pending owner review", result["reasons"][0])
                self.assertEqual(result["runs"][2][arm]["eligibility"]["rate"], 0)

    def test_hard_blockers_fail_before_minimum_without_statistical_peeking(self):
        for kind in c2.REQUIRED:
            clean = c2.campaign(kind, blocks([0.8], kind))
            self.assertEqual(clean["status"], "Continue")
            for blocker in ("zero_subject", "zero_reference", "anchor_capture", "anchor_future", "footprint"):
                rows = blocks([0.8], kind)
                if blocker.startswith("zero_"):
                    arm = blocker.removeprefix("zero_")
                    rows[0][arm]["eligibility"]["rate"] = 0
                elif blocker.startswith("anchor_"):
                    rows[0]["subject"]["eligibility"]["not_eligible"][blocker] = 1
                else:
                    rows[0]["subject"]["peak_footprint_bytes"] = 126
                result = c2.campaign(kind, rows)
                with self.subTest(kind=kind, blocker=blocker):
                    self.assertEqual(result["status"], "Fail")
                    self.assertIsNone(result["stop_reason"])
                    self.assertEqual(result["blocks"], 1)
                    self.assertTrue(result["reasons"])
                    self.assertTrue(all(result["checks"][name] is None for name in c2.REQUIRED[kind]))
                    self.assertEqual(result["checks"]["eligibility"],
                                     False if blocker.startswith("zero_") else None)

    def test_missing_reasons_are_not_zero_and_every_anchor_reason_must_be_zero(self):
        rows = blocks([0.8] * 5)
        del rows[0]["subject"]["eligibility"]["not_eligible"]["anchor_restore"]
        with self.assertRaisesRegex(ValueError, "explicit anchor_capture"):
            c2.campaign("divergent", rows)
        for reason in ("anchor_capture", "anchor_restore", "anchor_future_reason"):
            rows = blocks([0.8] * 5)
            rows[0]["reference"]["eligibility"]["not_eligible"][reason] = 1
            result = c2.campaign("divergent", rows)
            with self.subTest(reason=reason):
                self.assertEqual(result["status"], "Fail")
                self.assertFalse(result["checks"]["anchor_reasons_zero"])

    def test_one_footprint_outlier_fails_even_when_median_would_pass(self):
        rows = blocks([0.8] * 5)
        rows[3]["subject"]["peak_footprint_bytes"] = 126
        result = c2.campaign("divergent", rows)
        self.assertEqual(result["status"], "Fail")
        self.assertEqual(result["footprint_failures"], ["williams-4"])
        rows[3]["subject"]["peak_footprint_bytes"] = 125
        self.assertEqual(c2.campaign("divergent", rows)["status"], "Pass")

    def test_incomplete_duplicate_misaligned_and_failed_blocks_cannot_disappear(self):
        for mode in ("missing", "duplicate", "misaligned", "failed", "metric", "nonfinite", "calibration"):
            rows = blocks([0.8] * 5)
            if mode == "missing":
                del rows[1]["reference"]
            elif mode == "duplicate":
                rows[1] = copy.deepcopy(rows[0])
            elif mode == "misaligned":
                rows[1]["reference"]["block_id"] = "other-block"
            elif mode == "failed":
                rows[1]["reference"]["outcome"] = "Fail"
            elif mode == "metric":
                del rows[1]["subject"]["metrics"][DIVERGENT]
            elif mode == "nonfinite":
                rows[1]["subject"]["metrics"][DIVERGENT] = math.nan
            else:
                del rows[1]["subject"]["calibration_peak_footprint_bytes"]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                c2.campaign("divergent", rows)


class EligibilityInputs(unittest.TestCase):
    def test_scoped_actual_counters_and_supplied_rate_agree(self):
        record = evidence()
        record.update(joint_lane_steps=80, serial_lane_steps=20)
        self.assertEqual(c2.eligibility(record)["rate"], 0.8)
        del record["rate"]
        self.assertEqual(c2.eligibility(record)["rate"], 0.8)
        record["rate"] = 0.9
        with self.assertRaisesRegex(ValueError, "disagrees"):
            c2.eligibility(record)

    def test_no_width_proxy_missing_scope_invalid_rate_or_empty_denominator(self):
        invalid = [dict(evidence(), scope="all_decode"), dict(evidence(), rate=-0.1),
                   dict(evidence(), rate=1.1), dict(evidence(), rate=math.inf),
                   dict(evidence(), rate=True), dict(evidence(), joint_lane_steps=0, serial_lane_steps=0),
                   dict(evidence(), joint_lane_steps=1), dict(evidence(), joint_lane_steps=-1, serial_lane_steps=2)]
        proxy = evidence()
        del proxy["rate"]
        proxy["width_histogram"] = {"4": 20}
        invalid.append(proxy)
        for record in invalid:
            with self.subTest(record=record), self.assertRaises(ValueError):
                c2.eligibility(record)


if __name__ == "__main__":
    unittest.main()
