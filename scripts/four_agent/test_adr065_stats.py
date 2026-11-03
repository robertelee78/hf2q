"""Pure controls for the ADR-065 two-arm analysis and B1 fairness ceilings on synthetic receipts."""
import copy
import math
import statistics
import unittest

import adr065_stats as s
import q3_stats


def metrics(makespan=100.0, ttft=(1.0, 1.0, 1.0, 1.0), sse=(0.2, 0.2, 0.2, 0.2), q5=True, thermal=True):
    return {"makespan_s": makespan,
            "slots": {f"agent-{i}": {"ttft_p95_s": t, "sse_gap_max_s": g} for i, (t, g) in enumerate(zip(ttft, sse))},
            "memory": {"q5_verdict": "CalibrationPass" if q5 else "ClientFailed", "q5_pass": q5},
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
        self.assertEqual((n4["passing_runs"], n4["measured_runs"]), (2, 3))
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
        with self.assertRaisesRegex(ValueError, "no passing"):
            self.doc(failed)

    def test_fairness_check_per_run_and_slot(self):
        ceiling = {"ttft_p95_s": {"value": 2.0}, "sse_gap_max_s": {"value": 0.5}}
        check = lambda **m: s.fairness_check(entry("r", "B", **m), ceiling)
        self.assertEqual(check(ttft=(2.0, 2.0, 1.0, 1.0), sse=(0.5, 0.1, 0.1, 0.1)), [])  # at the ceiling complies
        problems = check(ttft=(1.0, 2.01, 1.0, 3.0), sse=(0.1, 0.6, 0.1, 0.1))
        self.assertEqual(len(problems), 3)
        self.assertIn("r agent-1: p95 semantic TTFT", problems[0])
        self.assertIn("max semantic SSE gap 0.6", problems[2])
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

    def test_fairness_violation_in_any_run_blocks_acceptance_despite_a_win(self):
        stage = ab_stage("n4", self.win, overrides={(2, "B"): {"ttft": (1, 1, 2.2, 1)}, (5, "A"): {"ttft": (9, 9, 9, 9)}})
        doc = self.analyze(ab_manifest([stage, ab_stage("n1", self.flat)]))
        self.assertEqual(doc["verdict"], "Win")
        self.assertFalse(doc["gates_pass"])
        self.assertEqual(doc["stages"]["n4"]["fairness_problems"],
                         ["n4-b02-B agent-2: p95 semantic TTFT 2.2 s > ceiling 2 s"])  # arm A is never checked

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
