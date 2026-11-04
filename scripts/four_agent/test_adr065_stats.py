"""Pure controls for the ADR-065 two-arm analysis and B1 fairness ceilings on synthetic receipts."""
import copy
import math
import statistics
import unittest

import adr065_stats as s
import q3_stats


def metrics(makespan=100.0, ttft=(1.0, 1.0, 1.0, 1.0), sse=(0.2, 0.2, 0.2, 0.2), q5=True, thermal=True):
    """q3_metrics.run_metrics shape: per-slot values and their worst-slot run values."""
    worst = lambda values: max((v for v in values if v is not None), default=None)
    verdict = q5 if isinstance(q5, str) else ("CalibrationPass" if q5 else "ClientFailed")
    return {"makespan_s": makespan,
            "slots": {f"agent-{i}": {"ttft_p95_s": t, "sse_gap_max_s": g} for i, (t, g) in enumerate(zip(ttft, sse))},
            "worst_slot_ttft_p95_s": worst(ttft), "worst_slot_sse_gap_max_s": worst(sse),
            "memory": {"q5_verdict": verdict, "q5_pass": verdict in ("CalibrationPass", "AdmittedRunPass")},
            "thermal": {"valid": thermal}}


def entry(name, arm, kind="measured", block=None, outcome="Pass", **m):
    return {"name": name, "arm": arm, "kind": kind, "block": block, "attempts": [{"name": name + "-a0"}],
            "outcome": outcome, "failure_kind": None if outcome == "Pass" else "task",
            "reason": None if outcome == "Pass" else "tests failed", "metrics": metrics(**m)}


def ab_stage(workload, ratios, base=100.0, overrides=None, status="Complete"):
    """Blocks in ABBA order; B's makespan is ratio * A's. `overrides` maps (block, arm) -> entry kwargs."""
    overrides = overrides or {}
    blocks = []
    for b, ratio in enumerate(ratios, 1):
        order = q3_stats.block_order(["A", "B"], b)
        values = {"A": base, "B": base * ratio}
        blocks.append({"block": b, "order": order, "entries": [
            entry(f"{workload}-b{b:02d}-{arm}", arm, block=b,
                  **{"makespan": values[arm], **overrides.get((b, arm), {})}) for arm in order]})
    warmups = [entry(f"{workload}-w00-{arm}", arm, kind="warmup", makespan=base * 3, ttft=(99,) * 4) for arm in ("A", "B")]
    return {"workload": workload, "status": status, "warmups": warmups, "blocks": blocks}


def ab_manifest(stages, mode="ab", blocks=6, n1="required", status="Complete"):
    return {"status": status, "halted": None, "stages": stages,
            "plan": {"mode": mode, "family": "qwen38", "design": {"blocks": blocks}, "n1_control": n1}}


def baseline_manifest(per_stage, status="Complete"):
    stages = []
    for workload, rows in per_stage.items():
        blocks = [{"block": i, "order": ["A"], "entries": [entry(f"{workload}-b{i:02d}-A", "A", block=i, **row)]}
                  for i, row in enumerate(rows, 1)]
        stages.append({"workload": workload, "status": "Complete",
                       "warmups": [entry(f"{workload}-w00-A", "A", kind="warmup", ttft=(50,) * 4, sse=(9,) * 4)],
                       "blocks": blocks})
    return {"status": status, "halted": None, "stages": stages,
            "plan": {"mode": "baseline", "family": "qwen38", "baseline": {"label": "v0.1.29", "sha256": "b" * 64},
                     "arms": {"A": {"case": "/c.json", "case_sha256": "c" * 64}}},
            "identities": {"arms": {"A": {"binary": {"sha256": "b" * 64},
                                          "artifact": {"path": "/m.gguf", "bytes": 10, "sha256": "a" * 64}}},
                           "harness": {"git_commit": "d" * 40}}}


def effect(ratios):
    return q3_stats.paired_effect([(100 * r, 100) for r in ratios])


class Interval(unittest.TestCase):
    def test_paired_log_makespan_interval_by_hand(self):
        ratios = [0.90, 0.93, 0.88, 0.95, 0.91, 0.92]
        stage = ab_stage("n4", ratios)
        result = s.effect(stage)
        d = [math.log(r) for r in ratios]
        h = 2.5706 * statistics.stdev(d) / math.sqrt(6)
        m = statistics.fmean(d)
        self.assertEqual(result["blocks"], 6)
        self.assertAlmostEqual(result["mean_log_ratio"], m)
        low, high = result["improvement_ci95"]
        self.assertAlmostEqual(low, 100 * (1 - math.exp(m + h)))
        self.assertAlmostEqual(high, 100 * (1 - math.exp(m - h)))
        self.assertAlmostEqual(result["half_width_pp"], 100 * (math.exp(h) - 1))

    def test_blocks_alternate_ab_and_ba(self):
        orders = [b["order"] for b in ab_stage("n4", [1] * 4)["blocks"]]
        self.assertEqual(orders, [["A", "B"], ["B", "A"], ["A", "B"], ["B", "A"]])

    def test_a_block_with_a_failed_run_leaves_the_interval(self):
        stage = ab_stage("n4", [0.9] * 6, overrides={(3, "A"): {"outcome": "Fail"}})
        self.assertEqual(s.effect(stage)["blocks"], 5)


class Verdicts(unittest.TestCase):
    def test_order_and_mutual_exclusion(self):
        cases = {
            "Loss": [1.05, 1.06, 1.04, 1.05, 1.07, 1.05],
            "Win": [0.90, 0.91, 0.89, 0.90, 0.92, 0.90],
            "Equivalent": [1.000, 1.004, 0.996, 1.002, 0.998, 1.001],
        }
        for expected, ratios in cases.items():
            self.assertEqual(s.verdict(effect(ratios), 6)[0], expected, ratios)

    def test_wide_slower_interval_is_still_a_loss(self):
        result = effect([1.2, 1.1, 1.3, 1.15, 1.25, 1.1])
        self.assertGreater(result["half_width_pp"], s.MARGIN_PP)
        self.assertEqual(s.verdict(result, 6)[0], "Loss")

    def test_faster_but_imprecise_is_inconclusive_never_a_win(self):
        result = effect([0.70, 0.95, 0.80, 0.99, 0.75, 0.90])
        self.assertGreater(result["improvement_ci95"][0], 0)
        verdict, reasons = s.verdict(result, 6)
        self.assertEqual(verdict, "Inconclusive")
        self.assertIn("imprecise", reasons[0])

    def test_wide_interval_containing_zero_is_inconclusive_not_equivalent(self):
        verdict, reasons = s.verdict(effect([0.95, 1.05, 0.96, 1.04, 0.97, 1.03]), 6)
        self.assertEqual(verdict, "Inconclusive")
        self.assertIn("contains zero", reasons[0])

    def test_fewer_eligible_blocks_keep_a_loss_but_never_a_win_or_equivalence(self):
        self.assertEqual(s.verdict(effect([1.05] * 3 + [1.06, 1.04]), 6)[0], "Loss")
        self.assertEqual(s.verdict(effect([0.9, 0.91, 0.9, 0.89, 0.9]), 6)[0], "Inconclusive")
        self.assertEqual(s.verdict(effect([1.0, 1.001, 0.999, 1.0, 1.0]), 6)[0], "Inconclusive")
        self.assertEqual(s.verdict(effect([0.9]), 6), ("Inconclusive", ["no interval: 1 quality-eligible blocks of 6"]))

    def test_win_half_width_bound_is_inclusive(self):
        result = effect([0.9] * 6)
        result.update(improvement_ci95=[7.5, 12.5], half_width_pp=2.5)
        self.assertEqual(s.verdict(result, 6)[0], "Win")
        result.update(half_width_pp=2.5000001)
        self.assertEqual(s.verdict(result, 6)[0], "Inconclusive")

    def test_equivalence_bounds(self):
        result = effect([1.0] * 6)
        for ci, expected in (([-2.5, 2.5], "Equivalent"), ([-2.6, 1.0], "Inconclusive"), ([0.0, 2.4], "Equivalent"),
                             ([-0.1, -0.0001], "Loss")):
            result.update(improvement_ci95=ci, half_width_pp=1.0)
            self.assertEqual(s.verdict(result, 6)[0], expected, ci)
        self.assertTrue(s.within_margin({"improvement_ci95": [-2.5, 2.5]}))
        self.assertFalse(s.within_margin({"improvement_ci95": [-2.51, 0]}))
        self.assertFalse(s.within_margin({"improvement_ci95": None}))


class Ceilings(unittest.TestCase):
    def doc(self, manifest):
        return s.ceilings(manifest, {"campaign": {"sha256": "e" * 64}})

    def test_largest_passing_measured_value_per_workload(self):
        manifest = baseline_manifest({
            "n4": [{"ttft": (1.0, 2.0, 1.5, 1.2), "sse": (0.3, 0.2, 0.1, 0.4)},
                   {"ttft": (1.1, 1.9, 2.5, 1.0), "sse": (0.2, 0.9, 0.3, 0.2)},
                   {"outcome": "Fail", "ttft": (40, 40, 40, 40), "sse": (8, 8, 8, 8)}],
            "n1": [{"ttft": (0.5,), "sse": (None,)}, {"ttft": (0.6,), "sse": (None,)}]})
        doc = self.doc(manifest)
        n4, n1 = doc["ceilings"]["n4"], doc["ceilings"]["n1"]
        self.assertEqual(n4["ttft_p95_s"], {"value": 2.5, "run": "n4-b02-A", "slot": "agent-2"})
        self.assertEqual(n4["sse_gap_max_s"], {"value": 0.9, "run": "n4-b02-A", "slot": "agent-1"})
        self.assertEqual((n4["eligible_runs"], n4["measured_runs"]), (2, 3))
        self.assertEqual(n4["excluded_runs"], [{"run": "n4-b03-A", "reason": "outcome Fail"}])
        self.assertEqual(doc["cap_factor"], 1.5)
        self.assertEqual(n1["ttft_p95_s"]["value"], 0.6)
        self.assertIsNone(n1["sse_gap_max_s"]["value"])  # one semantic event per streamed turn: unmeasurable
        self.assertEqual([r["name"] for r in doc["retained_failures"]], ["n4-b03-A"])
        self.assertEqual(len(doc["runs"]), 7)  # warmups and the failed run are retained, never ceilings
        row = next(r for r in doc["runs"] if r["name"] == "n4-b01-A")
        self.assertEqual(row["slots"]["slots"]["agent-3"], {"ttft_p95_s": 1.2, "sse_gap_max_s": 0.4})
        self.assertEqual(row["slots"]["run_sse_gap_max_s"], 0.4)
        self.assertEqual(doc["baseline"]["binary_sha256"], "b" * 64)

    def test_only_complete_baseline_campaigns(self):
        manifest = baseline_manifest({"n4": [{}]})
        incomplete = copy.deepcopy(manifest)
        incomplete["status"] = "Halted"
        with self.assertRaisesRegex(ValueError, "Halted"):
            self.doc(incomplete)
        ab = copy.deepcopy(manifest)
        ab["plan"]["mode"] = "ab"
        with self.assertRaisesRegex(ValueError, "baseline"):
            self.doc(ab)
        failed = baseline_manifest({"n4": [{"outcome": "Fail"}]})
        with self.assertRaisesRegex(ValueError, "no eligible measured run"):
            self.doc(failed)

    def test_runs_without_calibration_pass_or_with_thermal_trouble_never_set_a_ceiling(self):
        manifest = baseline_manifest({"n4": [
            {"ttft": (1.0, 1.2, 1.1, 1.0), "sse": (0.3, 0.2, 0.2, 0.2)},
            {"q5": "CalibrationFailed", "ttft": (9, 9, 9, 9), "sse": (0.2,) * 4},
            {"q5": "AdmittedRunPass", "ttft": (8, 8, 8, 8), "sse": (0.2,) * 4},
            {"thermal": False, "ttft": (1, 1, 1, 1), "sse": (7, 7, 7, 7)}]})
        n4 = self.doc(manifest)["ceilings"]["n4"]
        self.assertEqual((n4["ttft_p95_s"]["value"], n4["sse_gap_max_s"]["value"], n4["eligible_runs"]), (1.2, 0.3, 1))
        self.assertEqual(n4["excluded_runs"], [{"run": "n4-b02-A", "reason": "Q5 verdict CalibrationFailed"},
                                               {"run": "n4-b03-A", "reason": "Q5 verdict AdmittedRunPass"},
                                               {"run": "n4-b04-A", "reason": "serious or critical thermal state"}])
        only_bad = baseline_manifest({"n4": [{"q5": "CalibrationFailed"}, {"thermal": False}]})
        with self.assertRaisesRegex(ValueError, "no eligible measured run .* for n4"):
            self.doc(only_bad)

    def test_safety_cap_per_run_and_slot_is_ceiling_plus_half(self):
        ceiling = {"ttft_p95_s": {"value": 2.0}, "sse_gap_max_s": {"value": 0.5}}
        check = lambda **m: s.fairness_check(entry("r", "B", **m), ceiling)
        self.assertEqual(check(ttft=(3.0, 2.5, 1.0, 1.0), sse=(0.75, 0.1, 0.1, 0.1)), [])  # at the cap complies
        problems = check(ttft=(1.0, 3.01, 1.0, 4.0), sse=(0.1, 0.76, 0.1, 0.1))
        self.assertEqual(len(problems), 3)
        self.assertEqual(problems[0], "r agent-1: p95 semantic TTFT 3.01 s > cap 3 s")
        self.assertIn("max semantic SSE gap 0.76 s > cap 0.75 s", problems[2])
        self.assertIn("Unverifiable", check(ttft=(None, 1, 1, 1))[0])
        self.assertIn("Unverifiable", check(sse=(None,) * 4)[0])
        unmeasurable = {"ttft_p95_s": {"value": 2.0}, "sse_gap_max_s": {"value": None}}
        self.assertEqual(s.fairness_check(entry("r", "B", sse=(None,) * 4), unmeasurable), [])
        self.assertIn("no SSE ceiling", s.fairness_check(entry("r", "B"), unmeasurable)[0])
        self.assertIn("no metrics", s.fairness_check({"name": "r", "metrics": None}, ceiling)[0])


class Analysis(unittest.TestCase):
    def setUp(self):
        self.ceilings = {"schema": s.CEILING_SCHEMA, "family": "qwen38", "ceilings": {
            key: {"ttft_p95_s": {"value": 2.0}, "sse_gap_max_s": {"value": 0.5}} for key in ("n4", "n1", "3+1", "2+2", "1+3")}}
        self.win = [0.90, 0.91, 0.89, 0.90, 0.92, 0.90]
        self.flat = [1.000, 1.004, 0.996, 1.002, 0.998, 1.001]

    def analyze(self, manifest):
        return s.analyze(manifest, self.ceilings, {})

    def test_win_with_controls_and_clean_gates(self):
        stages = [ab_stage("n4", self.win), ab_stage("n1", self.flat)] + [ab_stage(k, self.win) for k in s.MIXED]
        doc = self.analyze(ab_manifest(stages))
        self.assertEqual((doc["verdict"], doc["gate_failures"], doc["gates_pass"]), ("Win", [], True))
        self.assertTrue(doc["controls"]["n1"]["within_margin"])
        self.assertEqual(set(doc["controls"]), {"n1", "3+1", "2+2", "1+3"})
        self.assertEqual(len(doc["runs"]), 5 * (2 + 12))  # warmups retained (their high TTFT never gates)

    def test_failed_candidate_run_is_retained_and_gates(self):
        stage = ab_stage("n4", self.win, overrides={(4, "B"): {"outcome": "Fail"}})
        doc = self.analyze(ab_manifest([stage, ab_stage("n1", self.flat)]))
        self.assertEqual(doc["verdict"], "Inconclusive")  # 5 of 6 eligible blocks: never a Win
        self.assertEqual(doc["stages"]["n4"]["quality_ineligible_runs"], ["n4-b04-B"])
        self.assertTrue(any("quality-ineligible candidate run n4-b04-B (task: tests failed)" in g for g in doc["gate_failures"]))
        row = next(r for r in doc["runs"] if r["name"] == "n4-b04-B")
        self.assertEqual((row["outcome"], row["quality_eligible"], row["attempts"]), ("Fail", False, ["n4-b04-B-a0"]))

    def test_safety_cap_violation_in_any_run_blocks_acceptance_despite_a_win(self):
        stage = ab_stage("n4", self.win, overrides={(2, "B"): {"ttft": (1, 1, 3.2, 1)}, (5, "A"): {"ttft": (9, 9, 9, 9)}})
        doc = self.analyze(ab_manifest([stage, ab_stage("n1", self.flat)]))
        self.assertEqual(doc["verdict"], "Win")
        self.assertFalse(doc["gates_pass"])
        self.assertFalse(doc["stages"]["n4"]["paired_fairness"]["worst_slot_ttft_p95_s"]["regression"])
        self.assertEqual(doc["stages"]["n4"]["fairness_problems"],
                         ["n4-b02-B agent-2: p95 semantic TTFT 3.2 s > cap 3 s"])  # arm A is never checked

    def test_paired_fairness_regression_beyond_the_neutral_band(self):
        worse = [1.10, 1.12, 1.09, 1.11, 1.10, 1.13]
        stage = ab_stage("n4", self.win, overrides={(b, "B"): {"ttft": (t, 1, 1, 1)} for b, t in enumerate(worse, 1)})
        doc = self.analyze(ab_manifest([stage, ab_stage("n1", self.flat)]))
        record = doc["stages"]["n4"]["paired_fairness"]["worst_slot_ttft_p95_s"]
        self.assertTrue(record["regression"])
        self.assertEqual(record["effect"]["blocks"], 6)
        self.assertGreater(record["effect"]["ratio_ci95"][0], 1.025)
        self.assertEqual((doc["verdict"], doc["gates_pass"]), ("Win", False))
        self.assertTrue(doc["gate_failures"][0].startswith("fairness: n4: worst_slot_ttft_p95_s paired regression"))
        # Within the 1.025 band, or worse once in one block only: no regression (a single bad run is the cap's job).
        for overrides in ({(b, "B"): {"ttft": (1.02, 1, 1, 1)} for b in range(1, 7)}, {(3, "B"): {"ttft": (2.9, 1, 1, 1)}}):
            doc = self.analyze(ab_manifest([ab_stage("n4", self.win, overrides=overrides), ab_stage("n1", self.flat)]))
            self.assertEqual(doc["gate_failures"], [], overrides)

    def test_unmeasurable_sse_gap_does_not_gate_but_a_missing_ttft_interval_does(self):
        none = {(b, arm): {"sse": (None,) * 4} for b in range(1, 7) for arm in "AB"}
        self.ceilings["ceilings"]["n4"]["sse_gap_max_s"]["value"] = None
        doc = self.analyze(ab_manifest([ab_stage("n4", self.win, overrides=none), ab_stage("n1", self.flat)]))
        self.assertFalse(doc["stages"]["n4"]["paired_fairness"]["worst_slot_sse_gap_max_s"]["measurable"])
        self.assertEqual(doc["gate_failures"], [])
        one = [ab_stage("n4", self.win[:1]), ab_stage("n1", self.flat)]
        self.assertIn("fairness: n4: worst_slot_ttft_p95_s Unverifiable: 1 paired blocks, no interval",
                      self.analyze(ab_manifest(one))["gate_failures"])

    def test_halted_campaign_and_incomplete_controls_never_pass_the_gates(self):
        stages = [ab_stage("n4", self.win), ab_stage("n1", self.flat[:3], status="Halted")]
        stages += [{"workload": k, "status": "Pending", "warmups": [], "blocks": []} for k in s.MIXED]
        manifest = ab_manifest(stages, status="Halted")
        manifest["halted"] = {"reason": "run budget exhausted: 76 attempts (max 76)"}
        doc = self.analyze(manifest)
        self.assertEqual(doc["verdict"], "Win")  # the n4 stage is complete; acceptance still fails
        self.assertFalse(doc["gates_pass"])
        self.assertEqual(doc["gate_failures"][0], "campaign Halted, not Complete: run budget exhausted: 76 attempts (max 76)")
        self.assertIn("n1 control stage Halted, not Complete", doc["gate_failures"])
        self.assertFalse(doc["controls"]["n1"]["within_margin"])
        self.assertEqual(doc["controls"]["2+2"]["status"], "Pending")

    def test_n1_control_needs_every_fixed_block_quality_eligible(self):
        n1 = ab_stage("n1", self.flat, overrides={(2, "A"): {"outcome": "Fail"}})
        doc = self.analyze(ab_manifest([ab_stage("n4", self.win), n1]))
        self.assertIn("n1 control: 5 of 6 blocks quality-eligible", doc["gate_failures"])
        self.assertFalse(doc["controls"]["n1"]["within_margin"])

    def test_n1_control_required_or_reported(self):
        stages = [ab_stage("n4", self.win), ab_stage("n1", self.win)]
        self.assertIn("n1 control: interval does not lie within +-2.5 points", self.analyze(ab_manifest(stages))["gate_failures"])
        self.assertEqual(self.analyze(ab_manifest(stages, n1="reported"))["gate_failures"], [])
        self.assertIn("n1 control required but not run", self.analyze(ab_manifest([ab_stage("n4", self.win)]))["gate_failures"])

    def test_q5_thermal_and_missing_ceiling_gates(self):
        stage = ab_stage("n4", self.win, overrides={(1, "A"): {"q5": False}, (2, "B"): {"thermal": False}})
        del self.ceilings["ceilings"]["n1"]
        doc = self.analyze(ab_manifest([stage, ab_stage("n1", self.flat)]))
        self.assertEqual(doc["gate_failures"], ["n4: run n4-b01-A lacks a passing Q5 verdict",
                                                "n4: run n4-b02-B observed serious or critical thermal state",
                                                "fairness: n1: no frozen ceiling for this workload"])

    def test_pilot_incomplete_and_invalid_campaigns_have_no_verdict(self):
        stages = [ab_stage("n4", self.win), ab_stage("n1", self.flat)]
        self.assertEqual(self.analyze(ab_manifest(stages, mode="pilot"))["verdict"], "Pilot")
        self.assertIsNone(self.analyze(ab_manifest(stages, status="Invalid"))["verdict"])
        short = [ab_stage("n4", self.win[:4], status="Halted"), ab_stage("n1", self.flat)]
        doc = self.analyze(ab_manifest(short, status="Halted"))
        self.assertEqual((doc["verdict"], doc["reasons"]), ("Inconclusive", ["stage Halted: 4 of 6 blocks"]))

    def test_refuses_baselines_and_other_families(self):
        with self.assertRaisesRegex(ValueError, "A/B or pilot"):
            self.analyze(baseline_manifest({"n4": [{}]}))
        self.ceilings["family"] = "gemma4"
        with self.assertRaisesRegex(ValueError, "family qwen38"):
            self.analyze(ab_manifest([ab_stage("n4", self.win)]))


if __name__ == "__main__":
    unittest.main()
