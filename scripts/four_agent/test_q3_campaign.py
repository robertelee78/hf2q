"""Pure controls for the ADR-063 Q3 campaign: design, estimator, verdicts, receipts and the driver.

The driver tests replace q5_calibration.py with a fake runner that writes receipt-shaped files;
they prove scheduling, stopping, identity freezing and classification, never model speed.
"""
import contextlib
import copy
import inspect
import io
import itertools
import json
import math
import os
import re
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import f2_scoreboard_render as f2
import q2_fake_service as fake
import q3_campaign as q3
import q3_identity
import q3_metrics
import q3_report
import q3_stats
import q5_case
import q5_calibration
from test_native_peer_ds4_campaign import manifest as native_manifest

HERE = Path(__file__).resolve().parent
EXP_SHA, STOCK_SHA, PEER_SHA, MODEL_SHA = "e" * 64, "5" * 64, "9" * 64, "a" * 64
SHAS = {"stock": STOCK_SHA, "experiment": EXP_SHA, "peer": PEER_SHA, "peer_matched": "8" * 64}


def effect_of(ratios):
    return q3_stats.paired_effect([(r * 100, 100) for r in ratios])


class Design(unittest.TestCase):
    def test_three_arm_williams_balances_position_and_carryover(self):
        rows = q3_stats.williams(3)
        self.assertEqual(sorted(map(tuple, rows)), sorted(itertools.permutations(range(3))))
        for position in range(3):
            self.assertEqual(sorted(r[position] for r in rows), [0, 0, 1, 1, 2, 2])
        pairs = [(r[i], r[i + 1]) for r in rows for i in range(2)]
        self.assertEqual({p: pairs.count(p) for p in pairs}, {p: 2 for p in itertools.permutations(range(3), 2)})

    def test_even_williams_square_is_single_latin_square_with_balanced_carryover(self):
        rows = q3_stats.williams(4)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(sorted(col) == [0, 1, 2, 3] for col in zip(*rows)))
        pairs = [(r[i], r[i + 1]) for r in rows for i in range(3)]
        self.assertEqual(sorted(pairs), sorted(itertools.permutations(range(4), 2)))
        self.assertEqual(q3_stats.williams(2), [[0, 1], [1, 0]])

    def test_block_order_cycles_sequences(self):
        arms = ["stock", "experiment", "peer"]
        orders = [q3_stats.block_order(arms, b) for b in range(1, 13)]
        self.assertEqual(orders[:6], orders[6:])
        self.assertEqual(len({tuple(o) for o in orders[:6]}), 6)


class Estimator(unittest.TestCase):
    def test_constant_ratio(self):
        e = effect_of([0.8] * 5)
        self.assertAlmostEqual(e["ratio"], 0.8)
        self.assertAlmostEqual(e["improvement_percent"], 20.0)
        self.assertEqual(e["half_width_pp"], 0.0)

    def test_interval_matches_hand_computation(self):
        ratios = [0.80, 0.85, 0.78, 0.82, 0.90, 0.84]
        d = [math.log(r) for r in ratios]
        mean = sum(d) / 6
        sd = math.sqrt(sum((x - mean) ** 2 for x in d) / 5)
        h = 2.5706 * sd / math.sqrt(6)
        e = effect_of(ratios)
        self.assertAlmostEqual(e["ratio_ci95"][0], math.exp(mean - h))
        self.assertAlmostEqual(e["ratio_ci95"][1], math.exp(mean + h))
        self.assertAlmostEqual(e["half_width_pp"], 100 * (math.exp(h) - 1))
        self.assertAlmostEqual(e["improvement_ci95"][0], 100 * (1 - math.exp(mean + h)))

    def test_higher_is_better_direction(self):
        e = q3_stats.paired_effect([(110, 100)] * 3, higher_is_better=True)
        self.assertAlmostEqual(e["improvement_percent"], 10.0)

    def test_single_block_has_no_interval_and_bad_values_refused(self):
        self.assertIsNone(effect_of([0.9])["ratio_ci95"])
        with self.assertRaises(ValueError):
            q3_stats.paired_effect([(0, 100), (1, 100)])

    def test_stopping_rule(self):
        precise = effect_of([0.9] * 5)
        wide = effect_of([0.7, 1.1, 0.8, 1.2, 0.9])
        self.assertIsNone(q3_stats.stop_decision(4, {"vs_stock": effect_of([0.9] * 4)}))
        self.assertEqual(q3_stats.stop_decision(5, {"vs_stock": precise, "vs_peer": precise}), "precision")
        self.assertIsNone(q3_stats.stop_decision(5, {"vs_stock": precise, "vs_peer": wide}))
        self.assertEqual(q3_stats.stop_decision(20, {"vs_stock": precise, "vs_peer": wide}), "max_blocks")
        self.assertIsNone(q3_stats.stop_decision(20, {"vs_stock": precise, "vs_peer": wide}, max_blocks=40))
        self.assertEqual(q3_stats.stop_decision(40, {"vs_stock": precise, "vs_peer": wide}, max_blocks=40),
                         "max_blocks")
        with self.assertRaisesRegex(ValueError, "predeclared 20 or 40"):
            q3_stats.stop_decision(20, {"vs_stock": wide}, max_blocks=30)
        self.assertEqual(effect_of([0.9] * 40)["blocks"], 40)
        # 2.5 points is the boundary: a 2.6-point half-width keeps running
        self.assertTrue(q3_stats.precise(effect_of([0.9, 0.9 * 1.021, 0.9, 0.9 * 1.021, 0.9, 0.9 * 1.021])))
        self.assertFalse(q3_stats.precise(effect_of([0.9, 0.96, 0.9, 0.96, 0.9])))


class FailurePolicy(unittest.TestCase):
    POLICY = {"version": 1, "reruns_per_position": 2, "rerun_failure_kinds": ["task"], "arms": ["stock", "peer", "splash"]}

    def test_clopper_pearson_matches_known_exact_intervals(self):
        for failures, attempts, low, high in ((0, 10, 0.0, 0.3085), (3, 30, 0.0211, 0.2653), (1, 12, 0.0021, 0.3848),
                                              (10, 10, 0.6915, 1.0), (5, 10, 0.1871, 0.8129)):
            interval = q3_stats.clopper_pearson(failures, attempts)
            self.assertAlmostEqual(interval[0], low, places=4)
            self.assertAlmostEqual(interval[1], high, places=4)
        self.assertIsNone(q3_stats.clopper_pearson(0, 0))
        self.assertEqual(q3_stats.failure_rate(0, 0)["failure_rate"], None)
        self.assertAlmostEqual(q3_stats.failure_rate(3, 30)["failure_rate"], 0.1)
        with self.assertRaises(ValueError):
            q3_stats.clopper_pearson(4, 3)

    def test_only_a_comparison_arm_task_failure_is_rerun_and_at_most_twice(self):
        decide = q3_stats.rerun_decision
        self.assertEqual([decide(self.POLICY, "peer", "Fail", "task", n) for n in range(4)],
                         ["rerun", "rerun", "exhausted", "exhausted"])
        for arm in ("stock", "splash"):
            self.assertEqual(decide(self.POLICY, arm, "Fail", "task", 0), "rerun")
        for arm in ("experiment", "peer_matched"):  # the candidate never; the diagnostic arm is not a contrast
            self.assertIsNone(decide(self.POLICY, arm, "Fail", "task", 0))
        for kind in ("server_exited", "client_timeout", "memory_stop", "out_of_memory", None):
            self.assertIsNone(decide(self.POLICY, "peer", "Fail", kind, 0))
        for outcome in ("Pass", "NotStarted", "Invalid"):  # a passing run is never rerun, whatever its speed
            self.assertIsNone(decide(self.POLICY, "peer", outcome, "task", 0))
        self.assertIsNone(decide(None, "peer", "Fail", "task", 0))  # a plan without the policy: the legacy rule
        # the decision cannot see a timing: these are all of its inputs
        self.assertEqual(list(inspect.signature(decide).parameters),
                         ["policy", "arm", "outcome", "failure_kind", "prior_failures"])

    def test_the_plan_field_is_versioned_and_absent_means_legacy(self):
        self.assertIsNone(q3_stats.failure_policy({}))
        self.assertEqual(q3_stats.failure_rule({}), q3_stats.LEGACY_FAILURE_RULE)
        self.assertEqual(q3_stats.failure_policy({"comparison_failure_policy": self.POLICY}), self.POLICY)
        self.assertIn("at most 2 reruns", q3_stats.failure_rule({"comparison_failure_policy": self.POLICY}))
        for change in ({"version": 2}, {"reruns_per_position": 3}, {"reruns_per_position": 1},
                       {"rerun_failure_kinds": ["task", "client_timeout"]}, {"arms": ["experiment"]}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "unsupported comparison_failure_policy"):
                q3_stats.failure_policy({"comparison_failure_policy": {**self.POLICY, **change}})


def state(**changes):
    base = {"experiment_failures": [], "peer_unavailable": None, "peer_failures": [],
            "quality": {"status": "parity", "reason": None}, "complete": True,
            "vs_stock": effect_of([0.8] * 5), "vs_peer": effect_of([0.8] * 5),
            "fairness": {"worst_slot_ttft_p95_s": effect_of([1.0] * 5)}, "thermal_valid": True, "memory_valid": True}
    base.update(changes)
    return base


class Verdicts(unittest.TestCase):
    def test_win_needs_five_percent_stock_gain_and_peer_parity(self):
        self.assertEqual(q3_stats.family_verdict(state())[0], "Win")
        self.assertEqual(q3_stats.family_verdict(state(vs_peer=effect_of([0.97] * 5)))[0], "Win")
        self.assertEqual(q3_stats.family_verdict(state(vs_stock=effect_of([0.97] * 5)))[0], "Inconclusive")

    def test_legacy_plan_margins_remain_recomputable(self):
        legacy = state(stock_margin_ratio=q3_stats.LEGACY_STOCK_MARGIN,
                       peer_margin_ratio=q3_stats.LEGACY_PEER_MARGIN,
                       vs_stock=effect_of([0.99] * 5), vs_peer=effect_of([0.97] * 5))
        verdict, reasons = q3_stats.family_verdict(legacy)
        self.assertEqual(verdict, "Inconclusive")
        self.assertIn("peer parity not demonstrated (95% CI upper bound of the ratio above 0.95)", reasons)

    def test_imprecise_faster_is_inconclusive_but_slower_than_peer_is_a_loss(self):
        wide = effect_of([0.5, 0.9, 0.6, 0.95, 0.7])
        verdict, reasons = q3_stats.family_verdict(state(vs_peer=wide))
        self.assertEqual(verdict, "Inconclusive")
        self.assertTrue(any("precision" in r for r in reasons))
        self.assertEqual(q3_stats.family_verdict(state(vs_peer=effect_of([1.4, 1.9, 1.6, 1.5, 1.8])))[0], "Loss")

    def test_failures_and_unavailability_take_precedence(self):
        self.assertEqual(q3_stats.family_verdict(state(experiment_failures=["x"]))[0], "Failed")
        self.assertEqual(q3_stats.family_verdict(state(peer_failures=["p"]))[0], "Failed")
        self.assertEqual(q3_stats.family_verdict(state(peer_unavailable={"reason": "oom"}))[0], "Unavailable")
        self.assertEqual(q3_stats.family_verdict(state(quality={"status": "loss", "reason": "q1"}))[0], "Failed")
        self.assertEqual(q3_stats.family_verdict(state(complete=False))[0], "Pending")
        # comparison failure policy: an exhausted stock arm is Unavailable, never Failed and never a Win
        verdict, reasons = q3_stats.family_verdict(state(stock_unavailable={"reason": "stock failed 3 attempts"}))
        self.assertEqual((verdict, reasons), ("Unavailable", ["no usable stock arm: stock failed 3 attempts"]))
        self.assertEqual(q3_stats.family_verdict(state(stock_unavailable={"reason": "s"}, experiment_failures=["x"]))[0],
                         "Failed")
        self.assertEqual(q3_stats.family_verdict(state(stock_unavailable=None))[0], "Win")

    def test_fairness_thermal_memory_and_quality_block_a_win(self):
        for change in ({"fairness": {"worst_slot_ttft_p95_s": effect_of([1.2] * 5)}}, {"thermal_valid": False},
                       {"memory_valid": False}, {"quality": {"status": "unresolved", "reason": "pending"}}):
            with self.subTest(change=change):
                self.assertEqual(q3_stats.family_verdict(state(**change))[0], "Inconclusive")
        self.assertEqual(q3_stats.family_verdict(state(thermal_valid=None))[0], "Win")  # unrecorded is not invalid


def turns(ttft=1.0, gaps_ms=(10, 20)):
    events = [0]
    for gap in gaps_ms:
        events.append(events[-1] + gap * 1_000_000)
    return [{"semantic_timing_v3": {"stream": False}},
            {"semantic_timing_v3": {"stream": True, "semantic_ttft_seconds": ttft}, "semantic_event_monotonic_ns": events}]


def write_run(out, sha, makespan=None, outcome="pass", thermal=0, artifact_sha=MODEL_SHA, build=None,
              phase_plan=None, engine_kind="hf2q", serialized=False):
    """Receipt-shaped files as q5_calibration.py writes them; a passing client directory comes from the
    model-free fake service (raw turns + coordinator summary) so the Q2 run audit can re-derive it."""
    out.mkdir(parents=True)
    if outcome == "preflight":
        receipt = {"verdict": {"verdict": "PreflightRefused", "reason": "competitor"}, "client": None, "stop": None}
        (out / "receipt.json").write_text(json.dumps(receipt))
        return
    ok = outcome == "pass"
    receipt = {"verdict": {"verdict": "CalibrationPass" if ok else "ClientFailed", "reason": None}, "stop": None,
               "client": {"exit_code": 0 if ok else 1, "elapsed_seconds": (makespan or 1) + 0.5},
               "samples": {"min_free_plus_cached_bytes": 6 << 30, "deltas": {"swap_used_bytes": 0},
                           "pressure_counts": {"normal": 3}, "peak_footprint_bytes": 1, "max_compressor_bytes": 1},
               "effective_floor": {"effective_floor_bytes": 2 << 30}, "preflight": {"thermal_state": thermal},
               "binary": {"sha256": sha}, "artifact": {"path": "/m.gguf", "bytes": 10, "sha256": artifact_sha},
               "readiness": {"documents": {}}, "cleanup": {"proven": ok}, "cleanup_proven": ok}
    if build is not None:
        (out / "readiness-1.json").write_text(json.dumps({"build_info": build}))
        receipt["readiness"]["documents"]["http://127.0.0.1:1/props"] = "readiness-1.json"
    (out / "receipt.json").write_text(json.dumps(receipt))
    if outcome == "oom":
        (out / "server.stderr").write_bytes(b"x" * 10 + b"error: Insufficient Memory (00000008:kIOGPUCommand"
                                            b"BufferCallbackErrorOutOfMemory)\n")
    if ok:
        fake.honest_run(out / "client", makespan, engine_kind, phase_plan, slots=1 if serialized else None)
        return
    names = [f"agent-{i}" for i in range(4)]
    summary = {"status": "Failed", "barrier": {"status": "Released"},
               **({"own_output_four_task_makespan_seconds": makespan} if makespan else {}),
               "agents": {a: {"status": "Failed", "final_source_sha256": fake.FINAL_SOURCE_SHA256, "turns": []}
                          for a in names}}
    (out / "client").mkdir()
    (out / "client" / "summary.json").write_text(json.dumps(summary))


PLANS = {slug: q3.WORKLOADS[key] for key, slug in q3.SLUGS.items()}


class FakeRunner:
    """Stands in for q5_calibration.py: decides each run from (arm, workload, call index)."""

    def __init__(self, decide):
        self.decide, self.calls = decide, []

    def __call__(self, argv, env, stdout, stderr):
        case, out = Path(argv[3]), Path(argv[5])
        workload, arm = case.stem.split("-", 1)
        self.calls.append((workload, arm, out.name, env))
        spec = {"sha": SHAS[arm], "build": "b1" if arm.startswith("peer") else None,
                "phase_plan": PLANS[workload], "engine_kind": "peer" if arm.startswith("peer") else "hf2q",
                **self.decide(arm, workload, len(self.calls))}
        write_run(out, **spec)
        return 0


def fake_identities(plan, changed=None):
    ids = {"arms": {arm: {"binary": {"path": f"/bin/{arm}", "sha256": SHAS[arm], "exists": True},
                          "artifact": {"path": "/m.gguf", "bytes": 10, "sha256": MODEL_SHA},
                          "peer_build_info": "b1" if arm.startswith("peer") else None, "missing_variables": [],
                          "file_checks": {}, "inputs": {}} for arm in plan["arms"]},
           "harness": {"git_commit": "c" * 40, "git_dirty": [], "sources": {}},
           "python": {"path": "/usr/bin/python3", "sha256": "7" * 64}, "host": {"chip": "test"},
           "negative_control_receipt": plan.get("negative_control_receipt")}
    if changed:
        ids["harness"]["sources"]["scripts/four_agent/q3_campaign.py"] = changed
    return ids


def noisy(base, call, spread):
    return base * (1 + spread * (1 if call % 2 else -1))


class Harness(unittest.TestCase):
    def setUp(self):
        # The generic driver, verdict and rows controls run qwen38 without the Splash arm (its fake service speaks the
        # hf2q and peer protocols only); test_native_peer_splash covers the Splash requirement and wiring.
        patcher = mock.patch.object(q3_report, "SPLASH_FAMILIES", ())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.nc = self.root / "nc-receipt.json"
        self.nc.write_text("{}")
        self.out = self.root / "evidence" / "q3-qwen38"
        self.out.parent.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, *extra, workloads="n4"):
        return q3.parse_args(["--family", "qwen38", "--stock-bin", "/bin/stock", "--stock-commit", "1" * 40,
                              "--experiment-bin", "/bin/experiment", "--experiment-commit", "2" * 40,
                              "--qe-identity", "qe-test", "--negative-control-receipt", str(self.nc),
                              "--workloads", workloads, "--output", str(self.out), *extra])

    def driver(self, decide, *extra, workloads="n4", identify=None, thermal=lambda: 0):
        plan, cases = q3.build_plan(self.args(*extra, workloads=workloads), {}, "qwen38")
        runner = FakeRunner(decide)
        identify = identify or (lambda: fake_identities(plan))
        return q3.Driver(plan, cases, self.out, {}, runner, thermal, identify), runner

    def verdict(self, manifest, **kwargs):
        return q3_report.compute_verdict(manifest, **kwargs)


class Driver(Harness):
    def win(self, arm, workload, call):
        return {"makespan": noisy({"experiment": 80, "stock": 100}.get(arm, 100), call, 0.002)}

    def test_win_stops_at_minimum_blocks_with_discarded_warmups(self):
        driver, runner = self.driver(self.win)
        manifest = driver.run()
        stage = manifest["stages"][0]
        self.assertEqual((manifest["status"], stage["stop"]), ("Complete", {"reason": "precision", "blocks": 5}))
        self.assertEqual([e["arm"] for e in stage["warmups"]], ["stock", "experiment", "peer"])
        self.assertEqual([b["order"] for b in stage["blocks"]],
                         [q3_stats.block_order(["stock", "experiment", "peer"], b) for b in range(1, 6)])
        self.assertEqual(len(runner.calls), 18)
        self.assertEqual({c[3]["HF2Q_Q5_HF2Q_BIN"] for c in runner.calls if c[1] == "experiment"}, {"/bin/experiment"})
        self.assertNotIn("HF2Q_Q5_HF2Q_BIN", next(c[3] for c in runner.calls if c[1] == "peer"))
        verdict = self.verdict(manifest)
        self.assertEqual(verdict["verdict"], "Win", verdict["reasons"])
        self.assertEqual(verdict["primary"]["effects"]["vs_stock"]["blocks"], 5)
        self.assertAlmostEqual(verdict["primary"]["effects"]["vs_peer"]["ratio"], 0.8, places=2)
        self.assertEqual(verdict["warmups_discarded"], 3)
        self.assertEqual(verdict["primary"]["arms"]["experiment"]["thermal"]["in_run"], "unrecorded")
        self.assertEqual(verdict["quality"]["patch_review"]["pending"], [fake.FINAL_SOURCE_SHA256])

    def test_noisy_campaign_runs_to_twenty_blocks_and_is_inconclusive(self):
        driver, runner = self.driver(lambda arm, w, call: {"makespan": noisy(100, call // 3, 0.2) * (0.9 if arm == "experiment" else 1)})
        manifest = driver.run()
        self.assertEqual(manifest["stages"][0]["stop"], {"reason": "max_blocks", "blocks": 20})
        self.assertEqual(len(runner.calls), 63)
        verdict = self.verdict(manifest)
        self.assertEqual(verdict["verdict"], "Inconclusive")
        self.assertTrue(any("precision" in r for r in verdict["reasons"]))

    def test_noisy_campaign_honors_predeclared_forty_block_cap(self):
        driver, runner = self.driver(
            lambda arm, w, call: {"makespan": noisy(100, call // 3, 0.4) * (0.9 if arm == "experiment" else 1)},
            "--max-blocks", "40")
        manifest = driver.run()
        self.assertEqual(manifest["plan"]["design"]["max_blocks"], 40)
        self.assertEqual(manifest["stages"][0]["stop"], {"reason": "max_blocks", "blocks": 40})
        self.assertEqual(len(runner.calls), 123)
        self.assertEqual(self.verdict(manifest)["verdict"], "Inconclusive")

    def test_slower_than_peer_is_a_loss(self):
        driver, _ = self.driver(lambda arm, w, call: {"makespan": noisy({"peer": 60}.get(arm, 100), call, 0.002)})
        self.assertEqual(self.verdict(driver.run())["verdict"], "Loss")

    def test_experiment_task_failure_is_terminal_and_skips_controls(self):
        def decide(arm, workload, call):
            return {"outcome": "fail"} if arm == "experiment" and call > 5 else self.win(arm, workload, call)
        driver, runner = self.driver(decide, workloads="n4,n1")
        manifest = driver.run()
        self.assertEqual(manifest["stages"][0]["stop"]["reason"], "experiment_failed")
        self.assertEqual(manifest["stages"][1]["status"], "NotRun")
        verdict = self.verdict(manifest)
        self.assertEqual(verdict["verdict"], "Failed")
        with self.assertRaises(q3.qc.CampaignError):  # Complete is terminal
            self.driver(self.win)[0].run(resume=True)

    def test_experiment_failure_in_the_primary_is_a_failure_row_of_record(self):
        def decide(arm, workload, call):
            return {"outcome": "fail"} if arm == "experiment" else self.win(arm, workload, call)
        driver, _ = self.driver(decide, workloads="n4,n1")
        manifest = driver.run()
        q3.write_verdict(self.out, manifest, [], None)
        failed = next(e for e in manifest["stages"][0]["warmups"] if e["arm"] == "experiment")
        summary = Path(failed["attempts"][-1]["output"]) / "client" / "summary.json"
        document = json.loads(summary.read_text())
        document["agents"]["agent-1"]["error"] = "KeyboardInterrupt: signal 15"  # stopped by the coordinator
        document["agents"]["agent-2"].update(error="Invalid: initial six-step workflow not followed", tools=[
            {"name": "run_tests", "result": json.dumps({"target": "ports", "passed": False,
                                                        "output": "error[E0282]: type annotations needed"})}])
        summary.write_text(json.dumps(document))
        rows = q3_report.rows(self.out, self.root, of_record=True)
        [row] = rows["rows"]
        self.assertEqual((row["status"], row["metric"], row["value"], row["q3_of_record"], row["receipt"]),
                         ("Failure", "task_pass", 0, True, "$E/evidence/q3-qwen38/campaign.json"))
        self.assertNotIn("control_of", row)
        self.assertIn(f"Q3 verdict of record Failed (rule 1): the n4 warmup experiment run {failed['name']} failed its "
                      "task (task): agent-2: run_tests ports failed with E0282, then Invalid: initial six-step workflow "
                      "not followed; the stage stopped", row["note"])
        self.assertNotIn("agent-1", row["note"])
        board = {"schema": f2.ROWS_SCHEMA, "evidence_root_symbol": "$E", "evidence_root_description": "test",
                 "neutral_band_ratio": 1.025, "neutral_band_basis": "test",
                 "families": {"qwen38": {"title": "Qwen3.8 27B", "artifact_sha256": MODEL_SHA, "note": ""}}, **rows}
        f2.validate_manifest(board)
        self.assertEqual(f2.verify_rows(board, self.root), [row["id"]])
        text = f2.render(board, [], "0" * 64)
        self.assertIn(f"0/4 tasks [FAIL]: {row['note']} (`{row['id']}`)", text)
        self.assertEqual(text.count(f"`{row['id']}`"), 2)  # the tier's stock cell and its rows-of-record list

    def test_peer_out_of_memory_makes_the_peer_unavailable_and_keeps_stock_contrast(self):
        def decide(arm, workload, call):
            return {"outcome": "oom"} if arm == "peer" else self.win(arm, workload, call)
        driver, runner = self.driver(decide)
        manifest = driver.run()
        self.assertEqual([c[1] for c in runner.calls].count("peer"), 1)  # only the warmup ran
        self.assertIn("out_of_memory", manifest["unavailable"]["peer"]["reason"])
        self.assertEqual(manifest["stages"][0]["blocks"][0]["order"], ["stock", "experiment"])
        verdict = self.verdict(manifest)
        self.assertEqual(verdict["verdict"], "Unavailable")
        self.assertIsNotNone(verdict["primary"]["effects"]["vs_stock"]["ratio"])

    def test_declared_peer_unavailability_runs_two_arms(self):
        evidence = self.root / "peer-oom-receipt.json"
        evidence.write_text("{}")
        driver, runner = self.driver(self.win, "--peer-unavailable-evidence", str(evidence))
        manifest = driver.run()
        self.assertEqual({c[1] for c in runner.calls}, {"stock", "experiment"})
        self.assertEqual(manifest["unavailable"]["peer"]["evidence"][0]["sha256"], q3.sha256(evidence))
        self.assertEqual(self.verdict(manifest)["verdict"], "Unavailable")

    def test_peer_task_failure_cannot_win(self):
        peer_runs = itertools.count()

        def decide(arm, workload, call):
            failed = arm == "peer" and next(peer_runs) == 2  # the peer's second measured run
            return {"outcome": "fail"} if failed else self.win(arm, workload, call)
        driver, _ = self.driver(decide, "--comparison-reruns", "0")  # the rule before the 2026-11-12 policy
        verdict = self.verdict(driver.run())
        self.assertEqual(verdict["verdict"], "Failed")
        self.assertTrue(verdict["peer"]["failures"])

    def test_unstarted_run_halts_and_resume_retries_after_fresh_warmups(self):
        def decide(arm, workload, call):
            return {"outcome": "preflight"} if call == 5 else self.win(arm, workload, call)
        driver, _ = self.driver(decide)
        manifest = driver.run()
        self.assertEqual(manifest["status"], "Halted")
        self.assertTrue(manifest["halted"]["resumable"])
        resumed, runner = self.driver(self.win)
        manifest = resumed.run(resume=True)
        self.assertEqual(manifest["status"], "Complete")
        stage = manifest["stages"][0]
        self.assertEqual([w["session"] for w in stage["warmups"]], [0, 0, 0, 1, 1, 1])
        retried = next(e for b in stage["blocks"] for e in b["entries"] if len(e["attempts"]) == 2)
        self.assertEqual([a["outcome"] for a in retried["attempts"]], ["NotStarted", "Pass"])
        self.assertEqual(runner.calls[0][2].split("-")[2], "w01")

    def test_identity_change_halts_for_good(self):
        calls = itertools.count()
        driver, _ = self.driver(self.win, identify=lambda: fake_identities(
            driver.plan, changed="x" if next(calls) > 6 else None))
        manifest = driver.run()
        self.assertEqual(manifest["status"], "IdentityDrift")
        self.assertIn("harness.sources", " ".join(manifest["identity_drift"]))
        with self.assertRaises(q3.qc.CampaignError):
            self.driver(self.win)[0].run(resume=True)

    def test_receipt_binary_that_contradicts_the_frozen_identity_halts(self):
        def decide(arm, workload, call):
            return {**self.win(arm, workload, call), **({"sha": "0" * 64} if call == 4 else {})}
        driver, _ = self.driver(decide)
        manifest = driver.run()
        self.assertEqual(manifest["status"], "IdentityDrift")
        self.assertIn("binary.sha256", manifest["halted"]["reason"])

    def test_serious_thermal_state_blocks_a_win(self):
        driver, _ = self.driver(self.win, thermal=lambda: 2)
        verdict = self.verdict(driver.run())
        self.assertEqual(verdict["verdict"], "Inconclusive")
        self.assertIn("a run observed serious or critical thermal state", verdict["reasons"])

    def test_controls_use_phase_plans_and_report_separately(self):
        driver, runner = self.driver(self.win, workloads="n4,n1,3+1")
        manifest = driver.run()
        self.assertEqual([s["status"] for s in manifest["stages"]], ["Complete"] * 3)
        n1 = json.loads((self.out / "cases" / "n1-stock.json").read_text())
        self.assertEqual(n1["client"]["argv"][-2:], ["--phase-plan", "1"])
        n4 = json.loads((self.out / "cases" / "n4-peer.json").read_text())
        self.assertEqual(n4, q5_case.loads((HERE / "q5_cases" / "peer-qwen38.json").read_bytes()))
        verdict = self.verdict(manifest)
        self.assertEqual(set(verdict["controls"]), {"n1", "3+1"})
        self.assertEqual(verdict["controls"]["3+1"]["phase_plan"], "3+1")

    def test_quality_and_patch_review_inputs(self):
        driver, _ = self.driver(self.win)
        manifest = driver.run()
        run = self.root / "quality-run"
        run.mkdir()
        (run / "receipt.json").write_text(json.dumps({"binary": {"sha256": EXP_SHA}}))
        quality = self.root / "quality.json"
        document = {"plan": {"arms": {"subject": {"family": "qwen38"}}, "reference_kind": "peer"},
                    "runs": [{"arm": "subject", "attempts": [{"output": str(run)}]}], "verdict": {"verdict": "Pass"}}
        quality.write_text(json.dumps(document))
        self.assertEqual(self.verdict(manifest, quality_paths=[quality])["verdict"], "Win")
        quality.write_text(json.dumps({**document, "verdict": {"verdict": "Loss"}}))
        self.assertEqual(self.verdict(manifest, quality_paths=[quality])["verdict"], "Failed")
        (run / "receipt.json").write_text(json.dumps({"binary": {"sha256": STOCK_SHA}}))
        quality.write_text(json.dumps(document))
        self.assertEqual(self.verdict(manifest, quality_paths=[quality])["verdict"], "Inconclusive")
        review = self.root / "review.json"
        review.write_text(json.dumps({"schema": q3_report.PATCH_REVIEW_SCHEMA, "verdicts": {fake.FINAL_SOURCE_SHA256: "Fail"}}))
        self.assertEqual(self.verdict(manifest, review_path=review)["verdict"], "Failed")

    def test_rows_validate_and_verify_against_the_f2_renderer(self):
        driver, _ = self.driver(self.win)
        manifest = driver.run()
        q3.write_verdict(self.out, manifest, [], None)
        rows = q3_report.rows(self.out, self.root)
        document = {"schema": f2.ROWS_SCHEMA, "evidence_root_symbol": "$E", "evidence_root_description": "test",
                    "neutral_band_ratio": 1.025, "neutral_band_basis": "test",
                    "families": {"qwen38": {"title": "Qwen3.8 27B", "artifact_sha256": MODEL_SHA, "note": ""}},
                    **rows}
        f2.validate_manifest(document)
        self.assertEqual(f2.verify_rows(document, self.root), [r["id"] for r in rows["rows"]])
        self.assertEqual({r["baseline_arm"] for r in rows["rows"]}, {"v0128_stock", "peer"})
        self.assertAlmostEqual(rows["rows"][0]["value"], 20.0, places=0)
        self.assertFalse(any("control_of" in r for r in rows["rows"]))  # default mode: primary rows only
        # the checked-in Qwen3.8 peer case runs --spec-type none: a matched comparison, never "best valid configuration"
        peer_row = next(r for r in rows["rows"] if r["baseline_arm"] == "peer")
        self.assertIn(f"(baseline {q3_report.f2_q3_tier.PEER_SPEC_OFF})", peer_row["note"])
        self.assertNotIn("best valid", peer_row["note"])
        record = q3_report.rows(self.out, self.root, of_record=True)
        board = {"schema": f2.ROWS_SCHEMA, "evidence_root_symbol": "$E", "evidence_root_description": "test",
                 "neutral_band_ratio": 1.025, "neutral_band_basis": "test",
                 "families": {"qwen38": {"title": "Qwen3.8 27B", "artifact_sha256": MODEL_SHA, "note": ""}},
                 "q3_verdict_notices": {"qwen38": "Pending v11"}, **record}
        text = f2.render(f2.validate_manifest(board), [], "0" * 64)
        line = next(l for l in text.splitlines() if l.startswith("| Qwen3.8 27B | "))
        self.assertIn(" — **Pending v11** |", line)
        self.assertIn("sooner than the peer, speculation off (matched settings) (95% CI", line)

    def test_a_superseded_record_stays_beside_its_replacement_and_never_wins(self):
        driver, _ = self.driver(self.win)
        q3.write_verdict(self.out, driver.run(), [], None)
        replacement = self.root / "evidence" / "q3-qwen38-v2"
        replacement.mkdir()
        for name in ("campaign.json", "verdict.json"):
            shutil.copyfile(self.out / name, replacement / name)
        verdict = json.loads((replacement / "verdict.json").read_text())
        verdict["verdict"] = "Loss"  # e.g. against a better peer configuration
        (replacement / "verdict.json").write_text(json.dumps(verdict))
        record = q3_report.rows(replacement, self.root, of_record=True)
        old = q3_report.rows(self.out, self.root, superseded_by=replacement / "verdict.json")
        self.assertEqual(list(old["groups"]), ["qwen38-q3-superseded"])
        self.assertEqual({(r["status"], r["superseded_by"]) for r in old["rows"]},
                         {("Descriptive", "$E/evidence/q3-qwen38-v2/verdict.json")})
        peer_row = next(r for r in old["rows"] if r["baseline_arm"] == "peer")
        self.assertRegex(peer_row["note"], r"^Superseded Q3 record, replaced as the verdict of record by \$E/evidence/"
                                           r"q3-qwen38-v2/verdict\.json; never a win or loss; .*finishes the same coding task")
        board = {"schema": f2.ROWS_SCHEMA, "evidence_root_symbol": "$E", "evidence_root_description": "test",
                 "neutral_band_ratio": 1.025, "neutral_band_basis": "test",
                 "families": {"qwen38": {"title": "Qwen3.8 27B", "artifact_sha256": MODEL_SHA, "note": ""}},
                 "groups": {**record["groups"], **old["groups"]}, "rows": record["rows"] + old["rows"]}
        f2.validate_manifest(board)
        self.assertEqual(f2.verify_rows(board, self.root), [r["id"] for r in board["rows"]])
        text = f2.render(board, [], "0" * 64)
        self.assertIn("### Descriptive wins (not F2 primary qualified results)\n\nNone.", text)
        section = text.split("## Superseded Q3 records")[1].split("## Open gaps")[0]
        for row in old["rows"]:
            self.assertEqual(text.count(f"`{row['id']}`"), 1)
            self.assertIn(f"`{row['id']}`", section)
        # a Loss family is never shown as a Win against stock, whatever its stock contrast
        line = next(l for l in text.splitlines() if l.startswith("| Qwen3.8 27B | Loss | "))
        self.assertIn("; not a Q3 Win: the family verdict is Loss vs the peer", line)
        self.assertNotIn("**Win**", line)
        # superseded_by must cite the family's current record
        stray = copy.deepcopy(board)
        stray["rows"][-1]["superseded_by"] = "$E/evidence/q3-qwen38/verdict.json"
        with self.assertRaisesRegex(ValueError, "superseded_by must cite"):
            f2.validate_manifest(stray)
        with self.assertRaisesRegex(ValueError, "another verdict.json"):
            q3_report.rows(self.out, self.root, superseded_by=self.out / "verdict.json")


class ComparisonFailures(Harness):
    """The comparison-arm failure policy (operator decision 2026-11-12): strict on the experiment, reruns for the
    comparison arms, speed on passing runs only."""

    def win(self, arm, workload, call):
        return {"makespan": noisy({"experiment": 80, "stock": 100}.get(arm, 100), call, 0.002)}

    def failing(self, arm, runs, makespan=None):
        """decide(): `arm` fails its task on the 0-based run indices `runs` of that arm (index 0 is its warmup)."""
        seen = itertools.count()

        def decide(a, workload, call):
            if a == arm and next(seen) in runs:
                return {"outcome": "fail", **({"makespan": makespan} if makespan else {})}
            return self.win(a, workload, call)
        return decide

    def measured(self, manifest, arm):
        return [e for b in manifest["stages"][0]["blocks"] for e in b["entries"] if e["arm"] == arm]

    def test_new_plans_freeze_the_policy_and_zero_reruns_plans_as_before(self):
        plan, _ = q3.build_plan(self.args(), {}, "qwen38")
        self.assertEqual(plan["comparison_failure_policy"], {"version": 1, "reruns_per_position": 2,
                                                             "rerun_failure_kinds": ["task"], "arms": ["stock", "peer"]})
        legacy, _ = q3.build_plan(self.args("--comparison-reruns", "0"), {}, "qwen38")
        self.assertNotIn("comparison_failure_policy", legacy)
        self.assertEqual({k: v for k, v in plan.items() if k != "comparison_failure_policy"}, legacy)
        matched, _ = q3.build_plan(self.args("--peer-matched-case", str(HERE / "q5_cases" / "peer-qwen38.json")), {}, "qwen38")
        self.assertEqual(matched["comparison_failure_policy"]["arms"], ["stock", "peer"])  # never the diagnostic arm
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.args("--comparison-reruns", "5")
        self.assertIn("at most 2 reruns", q3.schedule_preview(plan)["n4"]["failure_rule"])

    def test_candidate_failure_is_failed_at_once_and_never_rerun(self):
        driver, runner = self.driver(self.failing("experiment", {2}))
        manifest = driver.run()
        self.assertEqual(manifest["stages"][0]["stop"]["reason"], "experiment_failed")
        failed = self.measured(manifest, "experiment")[-1]
        self.assertEqual([a["outcome"] for a in failed["attempts"]], ["Fail"])
        self.assertNotIn("disposition", failed["attempts"][0])
        self.assertEqual(runner.calls[-1][1], "experiment")  # nothing ran after it
        verdict = self.verdict(manifest)
        self.assertEqual(verdict["verdict"], "Failed")
        self.assertEqual((verdict["arm_reliability"]["experiment"]["failures"], verdict["discarded_runs"]), (1, []))

    def test_one_comparison_failure_is_rerun_in_place_counted_and_kept_out_of_the_speed(self):
        clean, clean_runner = self.driver(self.win)
        reference = clean.run()
        shutil.rmtree(self.out)
        # the peer's second measured run fails after 1 s: a makespan that would move the ratio if it were paired
        driver, runner = self.driver(self.failing("peer", {2}, makespan=1.0))
        manifest = driver.run()
        stage = manifest["stages"][0]
        self.assertEqual((manifest["status"], stage["stop"]), ("Complete", {"reason": "precision", "blocks": 5}))
        self.assertEqual(manifest["unavailable"], {})
        # block order is the Williams order, and the rerun ran in the failed run's slot before the next arm
        self.assertEqual([b["order"] for b in stage["blocks"]], [b["order"] for b in reference["stages"][0]["blocks"]])
        arms = [c[1] for c in runner.calls]
        failed_at = next(i for i, c in enumerate(runner.calls) if c[2].endswith("-a0") and c[1] == "peer"
                         and "-b02-" in c[2])
        self.assertTrue(runner.calls[failed_at + 1][2].endswith("-b02-peer-a1"))
        self.assertEqual(arms[:failed_at + 1] + arms[failed_at + 2:], [c[1] for c in clean_runner.calls])
        rerun = self.measured(manifest, "peer")[1]
        self.assertEqual([(a["outcome"], a.get("disposition")) for a in rerun["attempts"]], [("Fail", "rerun"), ("Pass", None)])
        self.assertEqual(rerun["attempts"][0]["case"], rerun["attempts"][1]["case"])
        self.assertEqual((rerun["outcome"], rerun["metrics"]["makespan_s"] > 90), ("Pass", True))
        self.assertTrue(all(len(e["attempts"]) == 1 for b in stage["blocks"] for e in b["entries"] if e is not rerun))
        verdict = q3.write_verdict(self.out, manifest, [], None)
        self.assertEqual(verdict["verdict"], "Win", verdict["reasons"])
        self.assertEqual(verdict["peer"]["failures"], [])
        self.assertEqual(verdict["comparison_failure_policy"]["reruns_per_position"], 2)
        self.assertIn("at most 2 reruns", verdict["failure_rule"])
        peer = verdict["arm_reliability"]["peer"]
        self.assertEqual((peer["attempts"], peer["failures"], peer["reruns"], peer["exhausted_positions"]), (7, 1, 1, []))
        self.assertAlmostEqual(peer["failure_rate"], 1 / 7)
        self.assertEqual(peer["failure_rate_ci95"], q3_stats.clopper_pearson(1, 7))
        self.assertEqual((peer["measured"]["attempts"], peer["measured"]["failures"], peer["warmup"]["attempts"]), (6, 1, 1))
        self.assertEqual((verdict["arm_reliability"]["stock"]["failures"], verdict["arm_reliability"]["experiment"]["failures"]), (0, 0))
        self.assertEqual(verdict["primary"]["arms"]["peer"]["reliability"]["failures"], 1)
        self.assertEqual(peer["failures_by_kind"], {"task": 1})
        [discarded] = verdict["discarded_runs"]
        attempt = rerun["attempts"][0]
        self.assertEqual(discarded, {"stage": "n4", "kind": "measured", "block": 2,
                                     "position": stage["blocks"][1]["order"].index("peer") + 1, "arm": "peer",
                                     "entry": rerun["name"], "attempt": attempt["name"], "disposition": "rerun",
                                     "output": attempt["output"], "receipt_sha256": attempt["receipt_sha256"],
                                     "failure_kind": "task", "reason": "None", "detail": None, "makespan_s": 1.0,
                                     "client_elapsed_s": 1.5, "tasks_passed": 0})
        self.assertTrue((Path(discarded["output"]) / "receipt.json").is_file())  # the failed run stays on disk
        # speed: five pairs on passing runs; the 1 s failed attempt is in no effect and no median
        effect = verdict["primary"]["effects"]["vs_peer"]
        self.assertEqual(effect["blocks"], 5)
        self.assertAlmostEqual(effect["ratio"], 0.8, places=2)
        self.assertTrue(all(abs(d - math.log(0.8)) < 0.01 for d in effect["log_ratios"]))
        self.assertGreater(verdict["primary"]["arms"]["peer"]["makespan_median_s"], 90)
        self.assertEqual(verdict["primary"]["arms"]["peer"]["outcomes"], {"Pass": 5})
        note = next(r["note"] for r in q3_report.rows(self.out, self.root)["rows"] if r["baseline_arm"] == "peer")
        self.assertIn("The peer arm failed 1 of 6 measured attempts of this stage (1 task; 16.7%, Clopper-Pearson 95% CI "
                      "0.4–64.1%); speed is paired on passing runs only (comparison_failure_policy v1", note)
        # descriptive sensitivity: the effect without the block that had a discarded run
        sensitivity = verdict["primary"]["rerun_sensitivity"]
        self.assertEqual((sensitivity["blocks_with_discarded_runs"], sensitivity["effects"]["vs_peer"]["blocks"],
                          sensitivity["effects"]["vs_stock"]["blocks"]), ([2], 4, 4))
        self.assertEqual(self.verdict(reference)["primary"]["rerun_sensitivity"],
                         {"blocks_with_discarded_runs": [], "effects": None})
        # the failing agent's own error replaces the client's generic reason in the report
        summary = Path(discarded["output"]) / "client" / "summary.json"
        document = json.loads(summary.read_text())
        document["agents"]["agent-1"]["error"] = "KeyboardInterrupt: signal 15"  # stopped by the coordinator
        document["agents"]["agent-3"]["error"] = "TimeoutError: absolute request deadline"
        summary.write_text(json.dumps(document))
        self.assertEqual(self.verdict(manifest)["discarded_runs"][0]["detail"],
                         "agent-3: TimeoutError: absolute request deadline")

    def test_two_failures_then_a_pass_keeps_the_block_and_reports_both(self):
        driver, runner = self.driver(self.failing("stock", {3, 4}))
        manifest = driver.run()
        rerun = self.measured(manifest, "stock")[2]
        self.assertEqual([(a["outcome"], a.get("disposition")) for a in rerun["attempts"]],
                         [("Fail", "rerun"), ("Fail", "rerun"), ("Pass", None)])
        self.assertEqual([c[2] for c in runner.calls if rerun["name"] in c[2]],
                         [f"{rerun['name']}-a{i}" for i in range(3)])
        verdict = self.verdict(manifest)
        self.assertEqual(verdict["verdict"], "Win", verdict["reasons"])
        self.assertEqual(verdict["primary"]["effects"]["vs_stock"]["blocks"], 5)
        stock = verdict["arm_reliability"]["stock"]
        self.assertEqual((stock["attempts"], stock["failures"], stock["reruns"]), (8, 2, 2))
        self.assertEqual([d["disposition"] for d in verdict["discarded_runs"]], ["rerun", "rerun"])
        # as the real client does, a failed run records no makespan: the time to the abort is what the report has
        self.assertEqual([(d["makespan_s"], d["client_elapsed_s"]) for d in verdict["discarded_runs"]], [(None, 1.5)] * 2)
        self.assertEqual(verdict["stock_failures"], [])

    def test_a_failed_warmup_of_a_comparison_arm_is_rerun_too(self):
        driver, _ = self.driver(self.failing("peer", {0}))
        manifest = driver.run()
        warmup = next(e for e in manifest["stages"][0]["warmups"] if e["arm"] == "peer")
        self.assertEqual([a["outcome"] for a in warmup["attempts"]], ["Fail", "Pass"])
        verdict = self.verdict(manifest)
        self.assertEqual(verdict["verdict"], "Win", verdict["reasons"])
        self.assertEqual((verdict["arm_reliability"]["peer"]["warmup"]["failures"], verdict["discarded_runs"][0]["position"]),
                         (1, None))

    def test_three_failures_in_one_position_make_the_peer_unavailable_not_failed(self):
        driver, runner = self.driver(self.failing("peer", {2, 3, 4}))
        manifest = driver.run()
        stage = manifest["stages"][0]
        exhausted = self.measured(manifest, "peer")[1]
        self.assertEqual([(a["outcome"], a.get("disposition")) for a in exhausted["attempts"]],
                         [("Fail", "rerun"), ("Fail", "rerun"), ("Fail", "exhausted")])
        self.assertEqual(exhausted["outcome"], "Fail")
        gone = manifest["unavailable"]["peer"]
        self.assertEqual((gone["kind"], gone["evidence"]), ("task_failures_exhausted", [a["output"] for a in exhausted["attempts"]]))
        self.assertIn(f"peer failed its task in all 3 attempts of {exhausted['name']}", gone["reason"])
        self.assertEqual([c[1] for c in runner.calls].count("peer"), 5)  # warmup, block 1, three attempts of block 2
        self.assertEqual(stage["blocks"][2]["order"], [a for a in q3_stats.block_order(["stock", "experiment", "peer"], 3)
                                                       if a != "peer"])
        self.assertEqual(stage["stop"]["reason"], "precision")  # the stock contrast still completes
        verdict = self.verdict(manifest)
        self.assertEqual(verdict["verdict"], "Unavailable")
        self.assertIn("no runnable peer arm: peer failed its task in all 3 attempts", verdict["reasons"][0])
        peer = verdict["arm_reliability"]["peer"]
        self.assertEqual((peer["attempts"], peer["failures"], peer["reruns"], peer["exhausted_positions"]),
                         (5, 3, 2, [exhausted["name"]]))
        self.assertEqual([d["disposition"] for d in verdict["discarded_runs"]], ["rerun", "rerun", "exhausted"])
        self.assertIsNotNone(verdict["primary"]["effects"]["vs_stock"]["ratio"])

    def test_three_stock_failures_are_unavailable_and_the_peer_contrast_still_completes(self):
        driver, _ = self.driver(self.failing("stock", {1, 2, 3}))
        manifest = driver.run()
        self.assertEqual(manifest["unavailable"]["stock"]["kind"], "task_failures_exhausted")
        self.assertEqual(manifest["stages"][0]["stop"]["reason"], "precision")
        verdict = self.verdict(manifest)
        self.assertEqual(verdict["verdict"], "Unavailable")
        self.assertIn("no usable stock arm: stock failed its task in all 3 attempts", verdict["reasons"][0])
        self.assertEqual(verdict["primary"]["effects"]["vs_peer"]["blocks"], 5)
        # the exhaustion is read from the attempts too: a driver that died before recording the arm cannot hide it
        lost = json.loads(json.dumps(manifest))
        del lost["unavailable"]["stock"]
        self.assertEqual(q3_report.unavailable(lost)["stock"], manifest["unavailable"]["stock"])
        again = self.verdict(lost)
        self.assertEqual((again["verdict"], again["reasons"]), (verdict["verdict"], verdict["reasons"]))
        self.assertEqual(again["peer"]["unavailable"]["stock"]["kind"], "task_failures_exhausted")

    def test_rows_keep_the_measured_peer_contrast_when_only_the_stock_arm_is_exhausted(self):
        driver, _ = self.driver(self.failing("stock", {3, 4, 5}))  # two paired stock blocks, then three failures
        manifest = driver.run()
        verdict = q3.write_verdict(self.out, manifest, [], None)
        self.assertEqual((verdict["verdict"], verdict["primary"]["effects"]["vs_peer"]["blocks"],
                          verdict["primary"]["effects"]["vs_stock"]["blocks"]), ("Unavailable", 5, 2))
        rows = q3_report.rows(self.out, self.root)["rows"]
        self.assertEqual({r["baseline_arm"] for r in rows}, {"v0128_stock", "peer"})
        peer_row = next(r for r in rows if r["baseline_arm"] == "peer")
        self.assertIn("5 paired blocks", peer_row["note"])
        self.assertAlmostEqual(peer_row["value"], 20.0, places=0)
        stock_row = next(r for r in rows if r["baseline_arm"] == "v0128_stock")
        self.assertIn("The stock arm failed 3 of 5 measured attempts of this stage (3 task; 60.0%", stock_row["note"])
        # the verdict of record carries the comparison arms' failure counts and shows them in the tier
        record = q3_report.rows(self.out, self.root, of_record=True)
        self.assertEqual(record["groups"]["qwen38-q3-record"]["q3_record"]["comparison_failures"],
                         {"rule": verdict["failure_rule"], "arms": {"stock": {"attempts": 6, "failures": 3},
                                                                    "peer": {"attempts": 6, "failures": 0}}})
        board = {"schema": f2.ROWS_SCHEMA, "evidence_root_symbol": "$E", "evidence_root_description": "test",
                 "neutral_band_ratio": 1.025, "neutral_band_basis": "test",
                 "families": {"qwen38": {"title": "Qwen3.8 27B", "artifact_sha256": MODEL_SHA, "note": ""}}, **record}
        line = next(l for l in f2.render(f2.validate_manifest(board), [], "0" * 64).splitlines()
                    if l.startswith("| Qwen3.8 27B | "))
        self.assertIn("| Unavailable: no usable stock arm: stock failed its task in all 3 attempts", line)
        self.assertIn("sooner than the peer, speculation off (matched settings) (95% CI", line)
        self.assertIn("; comparison-arm failed runs: stock 3 of 6, peer 0 of 6 (comparison_failure_policy v1", line)
        self.assertEqual(f2.verify_rows(board, self.root), [r["id"] for r in record["rows"]])
        for fault in ("missing", "zero"):
            with self.subTest(fault=fault):
                tampered = json.loads(json.dumps(board))
                rec = tampered["groups"]["qwen38-q3-record"]["q3_record"]
                if fault == "missing":
                    del rec["comparison_failures"]
                else:
                    for arm in rec["comparison_failures"]["arms"].values():
                        arm["failures"] = 0
                f2.validate_manifest(tampered)
                with self.assertRaisesRegex(ValueError, "q3_record comparison_failures drift"):
                    f2.verify_rows(tampered, self.root)
        broken = json.loads(json.dumps(board))
        broken["groups"]["qwen38-q3-record"]["q3_record"]["comparison_failures"]["arms"]["stock"]["failures"] = 7
        with self.assertRaisesRegex(Exception, "comparison_failures"):
            f2.validate_manifest(broken)
        clean = self.root / "evidence" / "q3-qwen38-clean"  # a record without a comparison failure carries no field
        self.out = clean
        q3.write_verdict(clean, self.driver(self.win)[0].run(), [], None)
        self.assertNotIn("comparison_failures", q3_report.rows(clean, self.root, of_record=True)["groups"]["qwen38-q3-record"]["q3_record"])
        clean_rows = q3_report.rows(clean, self.root, of_record=True)
        forged = {**board, **clean_rows}
        forged["groups"]["qwen38-q3-record"]["q3_record"]["comparison_failures"] = (
            record["groups"]["qwen38-q3-record"]["q3_record"]["comparison_failures"])
        f2.validate_manifest(forged)
        with self.assertRaisesRegex(ValueError, "q3_record comparison_failures drift"):
            f2.verify_rows(forged, self.root)

    def test_a_failure_the_policy_does_not_rerun_is_still_counted_by_kind(self):
        seen = itertools.count()

        def decide(arm, workload, call):  # the stock arm's second measured run dies of a Metal out-of-memory
            return {"outcome": "oom"} if arm == "stock" and next(seen) == 2 else self.win(arm, workload, call)
        driver, _ = self.driver(decide)
        manifest = driver.run()
        entry = self.measured(manifest, "stock")[1]
        self.assertEqual([(a["outcome"], a["failure_kind"], a.get("disposition")) for a in entry["attempts"]],
                         [("Fail", "out_of_memory", None)])
        verdict = self.verdict(manifest)
        self.assertEqual((verdict["verdict"], len(verdict["stock_failures"]), verdict["discarded_runs"]), ("Win", 1, []))
        stock = verdict["arm_reliability"]["stock"]
        self.assertEqual((stock["attempts"], stock["failures"], stock["failures_by_kind"], stock["reruns"]),
                         (7, 1, {"out_of_memory": 1}, 0))
        self.assertEqual(stock["failure_rate_ci95"], q3_stats.clopper_pearson(1, 7))
        self.assertEqual(verdict["primary"]["arms"]["stock"]["reliability"]["failures_by_kind"], {"out_of_memory": 1})
        self.assertEqual((verdict["primary"]["effects"]["vs_stock"]["blocks"], verdict["primary"]["blocks"]), (5, 6))

    def test_a_discarded_attempt_in_a_serious_thermal_state_still_blocks_a_win(self):
        seen = itertools.count()

        def decide(arm, workload, call):
            return {"outcome": "fail", "thermal": 2} if arm == "peer" and next(seen) == 2 else self.win(arm, workload, call)
        driver, _ = self.driver(decide)
        manifest = driver.run()
        self.assertEqual([a["thermal_valid"] for a in self.measured(manifest, "peer")[1]["attempts"]], [False, True])
        verdict = self.verdict(manifest)
        self.assertEqual((verdict["thermal_valid"], verdict["verdict"]), (False, "Inconclusive"))
        self.assertIn("a run observed serious or critical thermal state", verdict["reasons"])

    def crash_on(self, decide, name_end):
        """Run until the attempt whose name ends with `name_end` has written its evidence, then kill the driver."""
        class Crash(Exception):
            pass
        driver, inner = self.driver(decide)

        def runner(argv, env, stdout, stderr):
            code = inner(argv, env, stdout, stderr)
            if Path(argv[5]).name.endswith(name_end):
                raise Crash(argv[5])
            return code
        driver.runner = runner
        with self.assertRaises(Crash):
            driver.run()
        return json.loads((self.out / "campaign.json").read_text())

    def test_a_driver_that_dies_during_a_rerun_keeps_the_count_on_resume(self):
        manifest = self.crash_on(self.failing("peer", {2, 3}), "-b02-peer-a1")
        entry = self.measured(manifest, "peer")[1]
        self.assertEqual([(a["outcome"], a.get("disposition")) for a in entry["attempts"]],
                         [("Fail", "rerun"), ("Launched", None)])
        # the resumed session scores the second failure from disk, runs fresh warmups, and the position fails again
        resumed, _ = self.driver(self.failing("peer", {1}))
        manifest = resumed.run(resume=True)
        entry = self.measured(manifest, "peer")[1]
        self.assertEqual([(a["outcome"], a.get("disposition")) for a in entry["attempts"]],
                         [("Fail", "rerun"), ("Fail", "rerun"), ("Fail", "exhausted")])
        self.assertEqual(manifest["unavailable"]["peer"]["kind"], "task_failures_exhausted")
        self.assertEqual(self.verdict(manifest)["verdict"], "Unavailable")

    def test_a_driver_that_dies_during_the_last_allowed_attempt_never_runs_a_fourth(self):
        self.crash_on(self.failing("peer", {2, 3, 4}), "-b02-peer-a2")
        resumed, runner = self.driver(self.win)
        manifest = resumed.run(resume=True)
        entry = self.measured(manifest, "peer")[1]
        self.assertEqual([(a["outcome"], a.get("disposition")) for a in entry["attempts"]],
                         [("Fail", "rerun"), ("Fail", "rerun"), ("Fail", "exhausted")])
        self.assertIn("peer", manifest["unavailable"])
        self.assertNotIn("peer", [c[1] for c in runner.calls])
        self.assertEqual(self.verdict(manifest)["verdict"], "Unavailable")

    def test_resume_records_an_exhausted_arm_the_dead_driver_did_not(self):
        peer_runs = itertools.count()

        def decide(arm, workload, call):  # stock exhausts in block 1; a later peer run does not start, so the driver halts
            if arm == "stock":
                return {"outcome": "fail"} if call > 1 else self.win(arm, workload, call)
            if arm == "peer" and next(peer_runs) == 2:
                return {"outcome": "preflight"}
            return self.win(arm, workload, call)
        driver, _ = self.driver(decide)
        manifest = driver.run()
        self.assertEqual((manifest["status"], set(manifest["unavailable"])), ("Halted", {"stock"}))
        path = self.out / "campaign.json"
        lost = json.loads(path.read_text())
        del lost["unavailable"]["stock"]  # as if the driver had died between saving the entry and recording the arm
        path.write_text(json.dumps(lost))
        resumed, runner = self.driver(self.win)
        manifest = resumed.run(resume=True)
        self.assertEqual(manifest["unavailable"]["stock"]["kind"], "task_failures_exhausted")
        self.assertNotIn("stock", [c[1] for c in runner.calls])
        self.assertEqual(self.verdict(manifest)["verdict"], "Unavailable")

    def test_with_no_contrast_left_the_stage_stops_instead_of_running_experiment_only_blocks(self):
        seen = {"stock": itertools.count(), "peer": itertools.count()}

        def decide(arm, workload, call):
            return {"outcome": "fail"} if arm in seen and next(seen[arm]) >= 1 else self.win(arm, workload, call)
        driver, runner = self.driver(decide)
        manifest = driver.run()
        self.assertEqual(set(manifest["unavailable"]), {"stock", "peer"})
        self.assertEqual(manifest["stages"][0]["stop"]["reason"], q3_stats.NO_CONTRAST)
        self.assertEqual(len(manifest["stages"][0]["blocks"]), 1)
        self.assertEqual(self.verdict(manifest)["verdict"], "Unavailable")

    def test_a_slow_passing_run_is_never_rerun(self):
        def decide(arm, workload, call):
            return {"makespan": 5000.0} if arm == "peer" and call % 2 else self.win(arm, workload, call)
        driver, _ = self.driver(decide)
        manifest = driver.run()
        every = [e for s in manifest["stages"] for e in s["warmups"] + [x for b in s["blocks"] for x in b["entries"]]]
        self.assertTrue(all(len(e["attempts"]) == 1 for e in every))
        verdict = self.verdict(manifest)
        self.assertEqual((verdict["discarded_runs"], verdict["arm_reliability"]["peer"]["reruns"]), ([], 0))

    def test_a_rerun_that_does_not_start_halts_and_the_resume_still_counts_the_earlier_failure(self):
        peer_runs = itertools.count()

        def decide(arm, workload, call):
            if arm == "peer":
                return {2: {"outcome": "fail"}, 3: {"outcome": "preflight"}}.get(next(peer_runs), self.win(arm, workload, call))
            return self.win(arm, workload, call)
        driver, _ = self.driver(decide)
        manifest = driver.run()
        self.assertEqual((manifest["status"], manifest["halted"]["resumable"]), ("Halted", True))
        again = itertools.count()

        def resumed_decide(arm, workload, call):  # after the fresh warmup the position fails once more, then passes
            if arm == "peer" and next(again) == 1:
                return {"outcome": "fail"}
            return self.win(arm, workload, call)
        resumed, _ = self.driver(resumed_decide)
        manifest = resumed.run(resume=True)
        self.assertEqual(manifest["status"], "Complete")
        entry = self.measured(manifest, "peer")[1]
        self.assertEqual([(a["outcome"], a.get("disposition")) for a in entry["attempts"]],
                         [("Fail", "rerun"), ("NotStarted", None), ("Fail", "rerun"), ("Pass", None)])
        verdict = self.verdict(manifest)
        self.assertEqual(verdict["verdict"], "Win", verdict["reasons"])
        self.assertEqual(verdict["arm_reliability"]["peer"]["failures"], 2)

    def test_a_plan_recorded_before_the_policy_keeps_its_rule(self):
        driver, runner = self.driver(self.failing("peer", {2}), "--comparison-reruns", "0")
        manifest = driver.run()
        self.assertNotIn("comparison_failure_policy", manifest["plan"])
        failed = self.measured(manifest, "peer")[1]
        self.assertEqual(([a["outcome"] for a in failed["attempts"]], failed["outcome"]), (["Fail"], "Fail"))
        old = json.loads(json.dumps(manifest))  # a campaign.json as written before this change
        for stage in old["stages"]:
            for entry in stage["warmups"] + [e for b in stage["blocks"] for e in b["entries"]]:
                for attempt in entry["attempts"]:
                    for key in ("makespan_s", "client_elapsed_s", "tasks_passed", "thermal_valid"):
                        del attempt[key]
        for document in (manifest, old):
            verdict = self.verdict(document)
            self.assertEqual(verdict["verdict"], "Failed")
            self.assertIn("peer arm failed its task", verdict["reasons"][0])
            self.assertEqual((verdict["comparison_failure_policy"], verdict["failure_rule"]),
                             (None, q3_stats.LEGACY_FAILURE_RULE))
            self.assertEqual((verdict["arm_reliability"]["peer"]["failures"], verdict["discarded_runs"]), (1, []))
        # the rule is part of the plan: neither kind of campaign resumes under the other
        with self.assertRaisesRegex(q3.qc.CampaignError, "identical plan"):
            self.driver(self.win)[0].run(resume=True)


class ControlsOnly(Harness):
    """--controls-of (#369 amendment): control stages only, bound to a primary verdict of record."""
    win = Driver.win

    def record(self, *extra, workloads="n4,n1", ids=fake_identities):
        """A completed default campaign (n4,n1 unless told otherwise) with its verdict.json: the primary of
        record. Later campaigns in this test write to a sibling directory."""
        driver, _ = self.driver(self.win, *extra, workloads=workloads, identify=lambda: ids(driver.plan))
        q3.write_verdict(self.out, driver.run(), [], None)
        record = self.out / "verdict.json"
        self.out = self.out.parent / "q3-qwen38-controls"
        return record

    def controls(self, record, *extra, workloads="3+1,2+2,1+3", **kwargs):
        return self.driver(self.win, "--controls-of", str(record), *extra, workloads=workloads, **kwargs)

    def test_default_mode_is_unchanged(self):
        self.assertEqual(q3.parse_args(["--dry-run"]).workloads, ",".join(q3.WORKLOADS))
        plan, _ = q3.build_plan(self.args(workloads=",".join(q3.WORKLOADS)), {}, "qwen38")
        self.assertEqual((plan["primary"], plan["workloads"]), ("n4", list(q3.WORKLOADS)))
        self.assertNotIn("controls_of", plan)

    def test_controls_follow_the_comparison_failure_rule_of_their_primary(self):
        record = self.record()
        plan, _ = q3.build_plan(self.args("--controls-of", str(record), workloads="3+1"), {}, "qwen38")
        primary = json.loads((record.parent / "campaign.json").read_text())["plan"]
        self.assertEqual(plan["comparison_failure_policy"], primary["comparison_failure_policy"])
        self.assertEqual(plan["controls_of"]["comparison_failure_policy"], primary["comparison_failure_policy"])
        with self.assertRaisesRegex(q3.qc.CampaignError, r"follows its primary's comparison failure rule \(2 reruns"):
            q3.build_plan(self.args("--controls-of", str(record), "--comparison-reruns", "0", workloads="3+1"), {}, "qwen38")
        shutil.rmtree(record.parent)  # a primary planned before the policy: its controls keep the earlier rule
        self.out = record.parent
        old = self.record("--comparison-reruns", "0")
        plan, _ = q3.build_plan(self.args("--controls-of", str(old), workloads="3+1"), {}, "qwen38")
        self.assertNotIn("comparison_failure_policy", plan)
        self.assertNotIn("comparison_failure_policy", plan["controls_of"])
        with self.assertRaisesRegex(q3.qc.CampaignError, r"follows its primary's comparison failure rule \(0 reruns"):
            q3.build_plan(self.args("--controls-of", str(old), "--comparison-reruns", "2", workloads="3+1"), {}, "qwen38")

    def test_controls_only_reasons_name_an_exhausted_stock_arm(self):
        record = self.record()
        seen = itertools.count()

        def decide(arm, workload, call):
            return {"outcome": "fail"} if arm == "stock" and next(seen) >= 1 else self.win(arm, workload, call)
        driver, _ = self.driver(decide, "--controls-of", str(record), workloads="3+1")
        verdict = self.verdict(driver.run())
        self.assertEqual(verdict["verdict"], q3_report.CONTROLS_ONLY)
        self.assertTrue(any(r.startswith("stock unavailable: stock failed its task in all 3 attempts") for r in verdict["reasons"]),
                        verdict["reasons"])

    def test_controls_only_needs_the_flag_and_never_reruns_a_recorded_stage(self):
        for workloads in ("3+1,2+2,1+3", "n1", "n1,n4"):  # without --controls-of n4 comes first
            with self.subTest(workloads=workloads), self.assertRaisesRegex(q3.qc.CampaignError, "must start with n4"):
                q3.build_plan(self.args(workloads=workloads), {}, "qwen38")
        record = self.record()
        for workloads in ("n4", "n4,3+1", "3+1,n4", "3+1,3+1", "3+1,n5", ","):
            with self.subTest(workloads=workloads), self.assertRaisesRegex(q3.qc.CampaignError, r"\(never n4\)"):
                q3.build_plan(self.args("--controls-of", str(record), workloads=workloads), {}, "qwen38")
        for workloads in ("n1", "n1,3+1", "3+1,n1"):  # n1 is Complete in the record: never re-run
            with self.subTest(workloads=workloads), self.assertRaisesRegex(
                    q3.qc.CampaignError, "n1: Complete in the campaign of record .*never re-run"):
                q3.build_plan(self.args("--controls-of", str(record), workloads=workloads), {}, "qwen38")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            q3.parse_args(["--dry-run", "--controls-of", str(record)])  # exactly one --family
        # only the current verdict.json, of a completed primary, is a record
        superseded = record.with_name("verdict.pre-patch-review.json")
        superseded.write_text(record.read_text())
        with self.assertRaisesRegex(q3.qc.CampaignError, "must be a Q3 verdict.json"):
            q3.build_plan(self.args("--controls-of", str(superseded), workloads="3+1"), {}, "qwen38")
        incomplete = self.root / "evidence" / "q3-qwen38-incomplete"
        incomplete.mkdir()
        (incomplete / "campaign.json").write_bytes((record.parent / "campaign.json").read_bytes())
        verdict = json.loads(record.read_text())
        verdict["primary"]["status"] = "Running"
        (incomplete / "verdict.json").write_text(json.dumps(verdict))
        with self.assertRaisesRegex(q3.qc.CampaignError, r"did not complete \(stage Running\)"):
            q3.build_plan(self.args("--controls-of", str(incomplete / "verdict.json"), workloads="3+1"), {}, "qwen38")

    def test_identity_mismatch_with_the_primary_of_record_is_refused(self):
        record = self.record()
        with self.assertRaisesRegex(q3.qc.CampaignError, "is family qwen38, not gemma4"):
            q3.build_plan(self.args("--controls-of", str(record), workloads="3+1"), {}, "gemma4")

        def changed(change):
            def identify(plan):
                ids = fake_identities(plan)
                change(ids)
                return ids
            return identify
        cases_dir = self.root / "cases"
        cases_dir.mkdir()
        for name in ("qwen38.json", "peer-qwen38.json"):  # same case, different bytes
            (cases_dir / name).write_text((HERE / "q5_cases" / name).read_text() + "\n")
        cases = {
            "arms.experiment.binary_sha256": ((), changed(lambda i: i["arms"]["experiment"]["binary"].update(sha256="0" * 64))),
            "arms.stock.model.sha256": ((), changed(lambda i: [a["artifact"].update(sha256="b" * 64) for a in i["arms"].values()])),
            "arms.peer.peer_build_info": ((), changed(lambda i: i["arms"]["peer"].update(peer_build_info="b2"))),
            "arms.peer.sibling_dylibs": ((), changed(lambda i: i["arms"]["peer"]["binary"].update(
                sibling_dylibs={"libggml.dylib": "0" * 64}))),
            "arms.peer.native_runtime": ((), changed(lambda i: i["arms"]["peer"].update(native_runtime={"a.metal": "0" * 64}))),
            "host.chip": ((), changed(lambda i: i["host"].update(chip="other"))),
            "python_sha256": ((), changed(lambda i: i["python"].update(sha256="0" * 64))),
            "arms.experiment.commit": (("--experiment-commit", "3" * 40), fake_identities),
            "arms.stock.case_sha256": (("--hf2q-case", str(cases_dir / "qwen38.json")), fake_identities),
            "arms.peer.case_sha256": (("--peer-case", str(cases_dir / "peer-qwen38.json")), fake_identities),
            "arms.peer.overrides.PORT": (("--peer-var", "PORT=18999"), fake_identities),
            "measured_arms": (("--peer-unavailable-evidence", str(self.nc)), fake_identities),
        }
        for drift, (extra, identify) in cases.items():
            with self.subTest(drift=drift):
                driver, runner = self.controls(record, *extra, identify=lambda: identify(driver.plan))
                with self.assertRaisesRegex(q3.qc.CampaignError, "differs from the primary of record") as refused:
                    driver.run()
                self.assertIn(drift, str(refused.exception))
                self.assertEqual(runner.calls, [])
                self.assertFalse(self.out.exists())
        campaign = record.parent / "campaign.json"
        campaign.write_text(campaign.read_text() + "\n")  # the verdict no longer binds its campaign
        with self.assertRaisesRegex(q3.qc.CampaignError, "does not bind"):
            self.controls(record)

    def test_controls_only_campaign_runs_controls_and_binds_the_primary_of_record(self):
        record = self.record()
        of_record = json.loads(record.read_text())
        self.assertNotIn("primary_of_record", of_record)
        # harness sources may differ from the primary of record's; binaries, commits, model and peer may not
        driver, runner = self.controls(record, identify=lambda: fake_identities(driver.plan, changed="x"))
        manifest = driver.run()
        self.assertEqual(manifest["status"], "Complete")
        self.assertEqual([(s["workload"], s["status"]) for s in manifest["stages"]],
                         [("3+1", "Complete"), ("2+2", "Complete"), ("1+3", "Complete")])
        self.assertEqual({c[0] for c in runner.calls}, {"p3p1", "p2p2", "p1p3"})
        self.assertEqual(sorted(p.name for p in (self.out / "cases").iterdir()),
                         sorted(f"{s}-{a}.json" for s in ("p3p1", "p2p2", "p1p3") for a in ("stock", "experiment", "peer")))
        plan = manifest["plan"]
        self.assertIsNone(plan["primary"])
        self.assertEqual(plan["controls_of"]["verdict"], {"path": str(record), "sha256": q3.sha256(record)})
        self.assertEqual(plan["controls_of"]["campaign"]["sha256"], q3.sha256(record.parent / "campaign.json"))
        verdict = q3.write_verdict(self.out, manifest, [], None)
        self.assertEqual(verdict["verdict"], "ControlsOnly")
        self.assertIsNone(verdict["primary"])
        self.assertIsNone(verdict["primary_workload"])
        self.assertIn("no primary verdict is computed", verdict["reasons"][0])
        self.assertEqual(set(verdict["controls"]), {"3+1", "2+2", "1+3"})
        bound = verdict["primary_of_record"]
        self.assertEqual((bound["path"], bound["sha256"], bound["verdict"]), (str(record), q3.sha256(record), "Win"))
        self.assertEqual(bound["effects"], of_record["primary"]["effects"])
        self.assertEqual(bound["harness"]["differs"], ["scripts/four_agent/q3_campaign.py"])
        self.assertEqual(bound["q2_run_audit_sha256"], {"of_record": None, "this_campaign": None})
        # F2 rows: control effects only, labeled controls of the primary of record, never a win or loss
        rows = q3_report.rows(self.out, self.root)
        self.assertEqual(list(rows["groups"]), ["qwen38-q3-controls"])
        self.assertEqual(len(rows["rows"]), 6)  # vs stock and vs peer for each control
        for row in rows["rows"]:
            self.assertEqual((row["status"], row["control_of"]), ("Descriptive", "$E/evidence/q3-qwen38/verdict.json"))
            self.assertTrue(row["verification"]["key"].startswith("verdict.json controls."))
            self.assertNotIn("Q3 verdict", row["note"])
            self.assertTrue(row["note"].endswith(" campaign Complete."), row["note"])
            if row["baseline_arm"] == "peer":  # the operator's peer wording: task makespan on the slower arm's base
                self.assertRegex(row["note"], r"hf2q 22222222 finishes the same coding task \d+\.\d% sooner than the peer "
                                              r"\(95% CI \d+\.\d–\d+\.\d%; median [\d.]+ s vs [\d.]+ s\)\.")
                self.assertIn("hf2q's worst-slot p95 TTFT was", row["note"])
        document = {"schema": f2.ROWS_SCHEMA, "evidence_root_symbol": "$E", "evidence_root_description": "test",
                    "neutral_band_ratio": 1.025, "neutral_band_basis": "test",
                    "families": {"qwen38": {"title": "Qwen3.8 27B", "artifact_sha256": MODEL_SHA, "note": ""}},
                    **rows}
        f2.validate_manifest(document)
        self.assertEqual(f2.verify_rows(document, self.root), [r["id"] for r in rows["rows"]])
        text = f2.render(document, [], "0" * 64)
        self.assertIn("### Descriptive wins (not F2 primary qualified results)\n\nNone.", text)
        self.assertIn("## Q3 controls of a primary verdict of record (never a win or loss)", text)
        for row in rows["rows"]:
            self.assertEqual(text.count(f"`{row['id']}`"), 1)  # only in the controls section
        # a rejected patch review is reported as a quality loss; the verdict stays ControlsOnly
        review = self.root / "review.json"
        review.write_text(json.dumps({"schema": q3_report.PATCH_REVIEW_SCHEMA, "verdicts": {fake.FINAL_SOURCE_SHA256: "Fail"}}))
        rejected = q3_report.compute_verdict(manifest, review_path=review)
        self.assertEqual(rejected["verdict"], "ControlsOnly")
        self.assertTrue(any(r.startswith("quality loss: independent patch review rejected") for r in rejected["reasons"]))
        # a controls-only verdict is no primary of record, and neither bound record file may change
        with self.assertRaisesRegex(q3.qc.CampaignError, "carries no primary verdict"):
            q3.build_plan(self.args("--controls-of", str(self.out / "verdict.json"), workloads="3+1"), {}, "qwen38")
        for bound in (record.parent / "campaign.json", record):
            with self.subTest(bound=bound.name):
                original = bound.read_bytes()
                bound.write_bytes(original + b"\n")
                with self.assertRaisesRegex(ValueError, f"{bound.name} changed since the campaign bound it"):
                    q3_report.compute_verdict(manifest)
                bound.write_bytes(original)

    def test_width_only_invalid_in_a_control_halts_and_stands(self):
        record = self.record()
        real, audits = q3.q2_run_audit.check_run, itertools.count()

        def width_only(*args, **kwargs):
            audit, failure = real(*args, **kwargs)
            return audit, ("slot-capped service: observed co-progress width 3 < 4" if next(audits) == 5 else failure)
        driver, runner = self.controls(record, workloads="3+1,2+2")
        with mock.patch.object(q3.q2_run_audit, "check_run", side_effect=width_only):
            manifest = driver.run()
        self.assertEqual(manifest["status"], "Invalid")
        self.assertFalse(manifest["halted"]["resumable"])
        self.assertIn("co-progress width 3 < 4", manifest["halted"]["reason"])
        self.assertEqual(len(runner.calls), 6)
        self.assertEqual([s["status"] for s in manifest["stages"]], ["Running", "Pending"])
        verdict = q3.write_verdict(self.out, manifest, [], None)
        self.assertEqual((verdict["verdict"], verdict["campaign_status"]), ("ControlsOnly", "Invalid"))
        self.assertTrue(any(r.startswith("campaign Invalid") for r in verdict["reasons"]))
        self.assertEqual(q3_report.rows(self.out, self.root)["rows"], [])
        with self.assertRaises(q3.qc.CampaignError):  # Invalid is terminal: no resume, no rerun
            self.controls(record, workloads="3+1,2+2")[0].run(resume=True)

    def test_override_contents_bind_but_approval_templates_and_harness_sources_do_not(self):
        template, approval = self.root / "chat-template.jinja", self.root / "client-approval.json"
        template.write_text("template")
        approval.write_text("approval")

        def ids(plan, chat="1" * 64, audit="a" * 64):
            identity = fake_identities(plan)
            identity["arms"]["peer"]["inputs"] = {str(template): chat, str(approval): "2" * 64}
            identity["harness"]["sources"][q3_report.Q2_RUN_AUDIT] = audit
            return identity
        record = self.record("--peer-var", f"CHAT_TEMPLATE={template}", workloads="n4", ids=ids)
        extra = ("--peer-var", f"CHAT_TEMPLATE={template}", "--peer-var", f"CLIENT_APPROVAL_TEMPLATE={approval}")
        plan, _ = q3.build_plan(self.args("--controls-of", str(record), *extra, workloads="3+1"), {}, "qwen38")
        self.assertEqual(plan["controls_of"]["identity"]["arms"]["peer"]["overrides"], {"CHAT_TEMPLATE": {"sha256": "1" * 64}})
        drift = " ".join(q3_identity.launch_problems(plan, ids(plan, chat="3" * 64)))
        self.assertIn("arms.peer.overrides.CHAT_TEMPLATE.sha256", drift)  # same path, other template content
        # a new approval template and a newer q2_run_audit.py launch, and both audit hashes are recorded
        driver, _ = self.controls(record, *extra, workloads="3+1", identify=lambda: ids(driver.plan, audit="b" * 64))
        manifest = driver.run()
        self.assertEqual(manifest["status"], "Complete")
        bound = q3.write_verdict(self.out, manifest, [], None)["primary_of_record"]
        self.assertEqual(bound["q2_run_audit_sha256"], {"of_record": "a" * 64, "this_campaign": "b" * 64})
        self.assertEqual(bound["harness"]["differs"], [q3_report.Q2_RUN_AUDIT])

    def test_peer_unavailable_evidence_must_be_the_recorded_evidence(self):
        first, second = self.root / "peer-oom-1.json", self.root / "peer-oom-2.json"
        first.write_text("{}")
        second.write_text("{ }")
        record = self.record("--peer-unavailable-evidence", str(first), workloads="n4")
        plan, _ = q3.build_plan(self.args("--controls-of", str(record), "--peer-unavailable-evidence", str(first),
                                          workloads="3+1"), {}, "qwen38")
        self.assertEqual(q3_identity.launch_problems(plan, fake_identities(plan)), [])
        plan, _ = q3.build_plan(self.args("--controls-of", str(record), "--peer-unavailable-evidence", str(second),
                                          workloads="3+1"), {}, "qwen38")
        drift = " ".join(q3_identity.launch_problems(plan, fake_identities(plan)))
        self.assertIn("peer_unavailable_evidence", drift)
        self.assertNotIn("measured_arms", drift)

    def test_resume_after_the_record_changed_is_refused(self):
        record = self.record()

        def decide(arm, workload, call):
            return {"outcome": "preflight"} if call == 5 else self.win(arm, workload, call)
        driver, _ = self.driver(decide, "--controls-of", str(record), workloads="3+1")
        manifest = driver.run()
        self.assertEqual((manifest["status"], manifest["halted"]["resumable"]), ("Halted", True))
        self.assertEqual(q3.qc.plan_sha256(self.controls(record, workloads="3+1")[0].plan), manifest["plan_sha256"])
        record.write_text(record.read_text() + "\n")  # e.g. the record's verdict was recomputed
        with self.assertRaisesRegex(q3.qc.CampaignError, "resume needs the identical plan"):
            self.controls(record, workloads="3+1")[0].run(resume=True)

    def test_experiment_failure_in_a_control_stage_is_a_failure_row(self):
        record = self.record()

        def decide(arm, workload, call):
            return {"outcome": "fail"} if arm == "experiment" and call > 5 else self.win(arm, workload, call)
        driver, _ = self.driver(decide, "--controls-of", str(record), workloads="3+1,2+2")
        manifest = driver.run()
        self.assertEqual(manifest["status"], "Complete")
        self.assertEqual([s["status"] for s in manifest["stages"]], ["Stopped", "NotRun"])
        verdict = q3.write_verdict(self.out, manifest, [], None)
        self.assertEqual(verdict["verdict"], "ControlsOnly")
        self.assertTrue(any(r.startswith("experiment failed its task") for r in verdict["reasons"]))
        rows = q3_report.rows(self.out, self.root)
        [row] = rows["rows"]
        self.assertEqual((row["status"], row["metric"], row["kind"], row["value"]), ("Failure", "task_pass", "measurement", 0))
        self.assertEqual(row["control_of"], "$E/evidence/q3-qwen38/verdict.json")
        self.assertTrue(row["verification"]["key"].startswith("campaign.json stages.0."))
        document = {"schema": f2.ROWS_SCHEMA, "evidence_root_symbol": "$E", "evidence_root_description": "test",
                    "neutral_band_ratio": 1.025, "neutral_band_basis": "test",
                    "families": {"qwen38": {"title": "Qwen3.8 27B", "artifact_sha256": MODEL_SHA, "note": ""}},
                    **rows}
        f2.validate_manifest(document)
        self.assertEqual(f2.verify_rows(document, self.root), [row["id"]])
        text = f2.render(document, [], "0" * 64)
        self.assertIn("### Descriptive wins (not F2 primary qualified results)\n\nNone.", text)
        controls = text.split("## Q3 controls of a primary verdict of record (never a win or loss)")[1].split("## Open gaps")[0]
        self.assertIn(f"0/4 tasks [FAIL] | {row['note']}", controls)
        self.assertEqual(text.count(f"`{row['id']}`"), 1)

    def test_native_ds4_controls_report_per_stage_and_label_rows(self):
        record_dir, out = self.root / "evidence" / "q3-deepseek4", self.root / "evidence" / "q3-deepseek4-controls"
        record_dir.mkdir()
        out.mkdir()
        for name in ("campaign.json", "verdict.json"):
            (record_dir / name).write_text("{}")
        document = native_manifest("ds4")
        document["stages"][0].update(workload="3+1", phase_plan="3+1")
        document["identities"]["harness"] = {"git_commit": "c" * 40, "sources": {}}
        document["plan"].update(primary=None, controls_of={
            **{name.split(".")[0]: {"path": str(record_dir / name), "sha256": q3.sha256(record_dir / name)}
               for name in ("campaign.json", "verdict.json")},
            "verdict_of_record": "Unavailable", "primary_workload": "n4", "effects": {}, "identities_sha256": "y",
            "identity": {}, "harness": {"git_commit": "c" * 40, "sources": {}}})
        (out / "campaign.json").write_text(json.dumps(document))
        verdict = q3_report.compute_verdict(document)
        (out / "verdict.json").write_text(json.dumps(verdict))
        self.assertEqual((verdict["verdict"], verdict["primary"]), ("ControlsOnly", None))
        self.assertEqual(set(verdict["native_peer"]), {"3+1"})
        self.assertTrue(verdict["native_peer"]["3+1"]["statement"].startswith(
            "hf2q 22222222 finishes the same coding task 20.0% sooner than ds4 c97964b"))
        rows = q3_report.rows(out, self.root)
        self.assertEqual({r["baseline_arm"] for r in rows["rows"]}, {"ds4_native"})  # native-peer campaign: native rows only
        self.assertEqual({(r["status"], r["control_of"]) for r in rows["rows"]},
                         {("Descriptive", "$E/evidence/q3-deepseek4/verdict.json")})
        native_row = next(r for r in rows["rows"] if r["baseline_arm"] == "ds4_native")
        self.assertTrue(native_row["verification"]["key"].startswith("verdict.json controls.3+1.effects.vs_peer"))
        self.assertIn("Median tokens per run", native_row["note"])
        board = {"schema": f2.ROWS_SCHEMA, "evidence_root_symbol": "$E", "evidence_root_description": "test",
                 "neutral_band_ratio": 1.025, "neutral_band_basis": "test",
                 "families": {"deepseek4": {"title": "DeepSeek-V4 Flash", "artifact_sha256": "a" * 64, "note": ""}},
                 **rows}
        f2.validate_manifest(board)
        self.assertEqual(f2.verify_rows(board, self.root), [r["id"] for r in rows["rows"]])
        text = f2.render(board, [], "0" * 64)
        self.assertNotIn("## Native-engine comparisons", text)  # controls render only in their own section
        self.assertEqual(text.count(f"`{native_row['id']}`"), 1)


class Plan(Harness):
    def test_controls_derive_from_the_canonical_case(self):
        case = q5_case.loads((HERE / "q5_cases" / "qwen38.json").read_bytes())
        self.assertEqual(q3.derive_case(case, "n4"), case)
        derived = q3.derive_case(case, "2+2")
        self.assertEqual(derived["client"]["argv"], case["client"]["argv"] + ["--phase-plan", "2+2"])
        self.assertTrue(derived["workload"]["id"].endswith("-q3-p2p2"))
        with self.assertRaises(q3.qc.CampaignError):
            q3.derive_case(derived, "1+3")

    def test_hf2q_binaries_come_only_from_explicit_options(self):
        with self.assertRaises(q3.qc.CampaignError):
            q3.build_plan(self.args(), {"HF2Q_Q5_HF2Q_BIN": "/bin/x"}, "qwen38")
        with self.assertRaises(q3.qc.CampaignError):
            q3.build_plan(self.args("--experiment-var", "HF2Q_BIN=/bin/x"), {}, "qwen38")
        plan, _ = q3.build_plan(q3.parse_args(["--dry-run", "--family", "qwen38"]), {}, "qwen38")
        self.assertIsNone(plan["arms"]["stock"]["overrides"]["HF2Q_BIN"])  # no silent case default
        self.assertIn("--stock-bin is required", q3_identity.launch_problems(plan, fake_identities(plan)))

    def test_experiment_vars_select_the_gemma_warm_attention_arm_only_on_the_experiment(self):
        # A non-default Gemma warm-attention experiment arm binds its resolved kernel at readiness; the
        # stock arm keeps the hf2q default and an empty (skipped) kernel check.
        argv = ["--family", "gemma4", "--stock-bin", "/bin/stock", "--stock-commit", "1" * 40,
                "--experiment-bin", "/bin/experiment", "--experiment-commit", "2" * 40,
                "--qe-identity", "qe-test", "--negative-control-receipt", str(self.nc),
                "--workloads", "n4", "--output", str(self.out),
                "--experiment-var", "GEMMA_WARM_TILED_ATTENTION=1",
                "--experiment-var", "GEMMA_WARM_ATTENTION_KERNEL=TiledF16"]
        plan, cases = q3.build_plan(q3.parse_args(argv), {}, "gemma4")
        experiment, stock = plan["arms"]["experiment"]["overrides"], plan["arms"]["stock"]["overrides"]
        self.assertEqual((experiment["GEMMA_WARM_TILED_ATTENTION"], experiment["GEMMA_WARM_ATTENTION_KERNEL"]),
                         ("1", "TiledF16"))
        self.assertNotIn("GEMMA_WARM_TILED_ATTENTION", stock)
        self.assertNotIn("GEMMA_WARM_ATTENTION_KERNEL", stock)
        field = "measurement_snapshot.active_controls.gemma_warm_attention_kernel"
        for arm, expected in (("experiment", "TiledF16"), ("stock", "")):
            values = q5_case.build_plan(cases[arm], {**q5_calibration.builtins(Path("/private/tmp/q3-dry") / arm),
                                                     "REPO": str(HERE.parent.parent)},
                                        q3.qc.arm_environ({}, plan["arms"][arm]["overrides"]))
            self.assertEqual([c["equals"] for c in values["readiness"]["checks"] if c["field"] == field],
                             [expected], arm)
        with self.assertRaises(q3.qc.CampaignError):
            q3.build_plan(q3.parse_args(argv + ["--experiment-var", "NOT_A_CASE_VARIABLE=1"]), {}, "gemma4")

    def test_arm_cases_must_match_role_and_family(self):
        with self.assertRaises(q3.qc.CampaignError):
            q3.build_plan(self.args("--peer-case", str(HERE / "q5_cases" / "qwen38.json")), {}, "qwen38")
        with self.assertRaises(q3.qc.CampaignError):
            q3.build_plan(self.args("--peer-case", str(HERE / "q5_cases" / "peer-gemma4.json")), {}, "qwen38")
        plan, _ = q3.build_plan(self.args("--peer-matched-case", str(HERE / "q5_cases" / "peer-qwen38.json")), {}, "qwen38")
        self.assertEqual(plan["measured_arms"], ["stock", "experiment", "peer", "peer_matched"])

    def test_launch_problems(self):
        plan, _ = q3.build_plan(self.args(), {}, "qwen38")
        self.assertEqual(q3_identity.launch_problems(plan, fake_identities(plan)), [])
        ids = fake_identities(plan)
        ids["arms"]["peer"]["artifact"] = {"path": "/other.gguf", "bytes": 10, "sha256": "b" * 64}
        ids["arms"]["experiment"]["binary"]["sha256"] = STOCK_SHA
        ids["harness"]["git_dirty"] = [" M scripts/four_agent/q3_campaign.py"]
        problems = " | ".join(q3_identity.launch_problems(plan, ids))
        for text in ("same model", "identical", "clean Git commit"):
            self.assertIn(text, problems)
        self.assertEqual(q3_identity.diff({"a": {"b": 1, "c": 2}}, {"a": {"b": 1, "c": 3}}), ["a.c"])

    def dry_run(self, argv, environ):
        """q3.main in a hermetic environment: only `environ` HF2Q_Q5_* overrides, a clean fake harness,
        a fixed host, and a runner that fails the test if anything launches."""
        clean = {k: v for k, v in os.environ.items() if not k.startswith(q5_case.OVERRIDE_PREFIX)}
        stdout = io.StringIO()
        with mock.patch.dict(os.environ, {**clean, **environ}, clear=True), \
                mock.patch.object(q3.qc, "subprocess_runner", side_effect=AssertionError("launched")), \
                mock.patch.object(q3.Driver, "run", side_effect=AssertionError("driver started")), \
                mock.patch("q5_host.host_identity", return_value={"chip": "test"}), \
                mock.patch.object(q3_identity, "harness",
                                  return_value={"git_commit": "c" * 40, "git_dirty": [], "sources": {}}), \
                contextlib.redirect_stdout(stdout):
            code = q3.main(argv)
        return code, json.loads(stdout.getvalue())

    def test_dry_run_prints_every_family_plan_without_launching(self):
        stock, experiment = self.root / "stock-hf2q", self.root / "experiment-hf2q"
        stock.write_bytes(b"stock")
        experiment.write_bytes(b"experiment")
        declared = ["--stock-bin", str(stock), "--stock-commit", "1" * 40, "--experiment-bin", str(experiment),
                    "--experiment-commit", "2" * 40, "--qe-identity", "qe-test",
                    "--negative-control-receipt", str(self.nc)]
        code, report = self.dry_run(["--dry-run", *declared], {})
        self.assertEqual(code, 0)
        self.assertEqual([f["family"] for f in report["families"]], list(q5_case.FAMILIES))
        for family in report["families"]:
            self.assertEqual(list(family["schedule"]), list(q3.WORKLOADS))
            self.assertEqual(len(family["schedule"]["n4"]["blocks"]), q3_stats.MAX_BLOCKS)
            arms = family["identities"]["arms"]
            self.assertEqual(set(arms), {"stock", "experiment", "peer"})
            self.assertEqual((arms["stock"]["binary"]["sha256"], arms["experiment"]["binary"]["sha256"]),
                             (q3.sha256(stock), q3.sha256(experiment)))
            self.assertEqual(arms["peer"]["peer_build_info"], "b128-e86f5fd")
            self.assertNotIn("--stock-bin is required", family["launch_problems"])
            # the dry run reports what still blocks a launch (here: the unset client variables)
            self.assertTrue(any("missing variables" in p for p in family["launch_problems"]))
        # without the hf2q binaries the plan still prints, and says why it cannot launch
        code, report = self.dry_run(["--dry-run", "--family", "gemma4"], {})
        self.assertEqual(code, 0)
        self.assertIn("--stock-bin is required", report["families"][0]["launch_problems"])
        # an ambient hf2q binary would apply to both hf2q arms: refused even in a dry run
        code, report = self.dry_run(["--dry-run", *declared], {"HF2Q_Q5_HF2Q_BIN": str(stock)})
        self.assertEqual((code, report["valid"]), (2, False))


class Metrics(unittest.TestCase):
    def receipt(self, **changes):
        return {"verdict": {"verdict": "CalibrationPass"}, "stop": None, "client": {"exit_code": 0}, **changes}

    def summary(self, status="FourAgentProtocolCompleteAwaitingIndependentReview", released=True):
        return {"status": status, "barrier": {"status": "Released" if released else "Pending"}, "error": "e"}

    def test_classification(self):
        c = q3_metrics.classify
        self.assertEqual(c(self.receipt(), self.summary(), False)[0], "Pass")
        self.assertEqual(c(None, None, False)[0], "Invalid")
        self.assertEqual(c(self.receipt(client=None, verdict={"verdict": "PreflightRefused"}), None, False)[:2],
                         ("NotStarted", "preflight"))
        self.assertEqual(c(self.receipt(client=None, stop={"reason": "server_exited"}), None, True)[:2],
                         ("NotStarted", "load_failure"))
        self.assertEqual(c(self.receipt(), self.summary(released=False), False)[:2], ("NotStarted", "before_release"))
        self.assertEqual(c(self.receipt(stop={"reason": "swap_growth"}), self.summary(), False)[:2], ("Fail", "memory_stop"))
        self.assertEqual(c(self.receipt(stop={"reason": "client_timeout"}), self.summary(), False)[:2], ("Fail", "client_timeout"))
        self.assertEqual(c(self.receipt(stop={"reason": "server_exited"}), self.summary(), True)[:2], ("Fail", "out_of_memory"))
        self.assertEqual(c(self.receipt(stop={"reason": "sampler_blind"}), self.summary(), False)[0], "Invalid")
        self.assertEqual(c(self.receipt(supervisor_error="boom"), self.summary(), False)[0], "Invalid")
        failed = self.receipt(client={"exit_code": 1})
        self.assertEqual(c(failed, self.summary("Failed"), False)[:2], ("Fail", "task"))
        self.assertEqual(c(failed, self.summary("Failed"), True)[:2], ("Fail", "out_of_memory"))

    def test_oom_marker_found_across_read_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.stderr"
            path.write_bytes(b"a" * 20 + b"Insufficient Memory" + b"b" * 5)
            self.assertTrue(q3_metrics.has_oom_marker(path, chunk=8))
            path.write_bytes(b"a" * 100)
            self.assertFalse(q3_metrics.has_oom_marker(path, chunk=8))
            self.assertFalse(q3_metrics.has_oom_marker(Path(tmp) / "missing"))

    def test_slot_metrics_use_streamed_turns_only(self):
        agent = {"turns": turns(2.0, (10, 30)) + [{"semantic_timing_v3": {"stream": True, "semantic_ttft_seconds": 4.0},
                                                   "semantic_event_monotonic_ns": [0, 5_000_000]}]}
        slot = q3_metrics.slot_metrics(agent)
        self.assertEqual((slot["streamed_turns"], slot["ttft_max_s"], slot["sse_gaps"]), (2, 4.0, 3))
        self.assertAlmostEqual(slot["sse_gap_max_s"], 0.03)
        self.assertAlmostEqual(slot["sse_gap_p95_s"], 0.03)

    def test_run_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            write_run(out, EXP_SHA, makespan=90.0, thermal=1, build="b1")
            receipt = json.loads((out / "receipt.json").read_text())
            summary = json.loads((out / "client" / "summary.json").read_text())
            m = q3_metrics.run_metrics(out, receipt, summary, 4, 3)
        self.assertAlmostEqual(m["goodput_tasks_per_hour"], 4 * 3600 / 90)
        self.assertEqual((m["tasks_passed"], m["memory"]["swap_growth_bytes"], m["peer_build_info"]), (4, 0, "b1"))
        self.assertFalse(m["thermal"]["valid"])  # critical after the run
        self.assertEqual(m["thermal"]["in_run"], "unrecorded")


DS4_ART = "d" * 64  # the campaign's own artifact (ds4's file), distinct from the stock record's hf2q artifact MODEL_SHA


class StockRecordHarness(Harness):
    """--no-stock-arm --stock-record (ADR-063 Q4-DS4-RACE decision D1): the verdict-of-record mode without a stock arm,
    its vs-stock terms read from a bound same-artifact stock record. The generic harness runs qwen38, so the stock-record
    families are widened to it here; the deepseek4-only rule has its own control."""
    win = Driver.win

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(q3, "STOCK_RECORD_FAMILIES", ("qwen38",))
        patcher.start()
        self.addCleanup(patcher.stop)

    def record(self, *extra, decide=None, workloads="n4", thermal=lambda: 0):
        """A completed default campaign with a stock arm (the record, on MODEL_SHA); later campaigns write beside it."""
        driver, _ = self.driver(decide or self.win, *extra, workloads=workloads, thermal=thermal)
        q3.write_verdict(self.out, driver.run(), [], None)
        record = self.out / "verdict.json"
        self.out = self.out.parent / "q3-qwen38-ds4file"
        return record

    def no_stock(self, *extra, workloads="n4"):
        return q3.parse_args(["--family", "qwen38", "--no-stock-arm", "--experiment-bin", "/bin/experiment",
                              "--experiment-commit", "2" * 40, "--qe-identity", "qe-test",
                              "--negative-control-receipt", str(self.nc), "--workloads", workloads,
                              "--output", str(self.out), *extra])

    def ids(self, plan):
        """Fake identities of a campaign on DS4_ART, with the bound record files as they are now on disk."""
        ids = fake_identities(plan)
        for arm in ("experiment", "peer"):
            ids["arms"][arm]["artifact"] = {"path": "/m.gguf", "bytes": 10, "sha256": DS4_ART}  # write_run's receipt
        ids["stock_record"] = q3_identity.stock_record_files(plan["stock_record"])
        return ids

    def campaign(self, record, decide=None, identify=None, workloads="n4"):
        plan, cases = q3.build_plan(self.no_stock("--stock-record", str(record), workloads=workloads), {}, "qwen38")
        runner = FakeRunner(lambda arm, w, call: {**(decide or self.win)(arm, w, call), "artifact_sha": DS4_ART})
        return q3.Driver(plan, cases, self.out, {}, runner, lambda: 0, identify or (lambda: self.ids(plan))), runner


class StockRecord(StockRecordHarness):
    """The refusal matrix, the binding, the mismatch and changed-file halts, the verdict state and the rows."""

    def test_argument_matrix(self):
        record = str(self.record())
        refusals = [(self.args("--stock-record", record), "--stock-record needs --no-stock-arm"),
                    (self.no_stock(), "--no-stock-arm needs exactly one of --stock-record"),
                    (self.no_stock("--stock-record", record, "--native-comparison"), "needs exactly one of"),
                    (self.no_stock("--stock-record", record, "--stock-bin", "/bin/stock"), "--stock-bin refused"),
                    (self.no_stock("--stock-record", record, "--stock-commit", "1" * 40), "--stock-commit refused"),
                    (self.no_stock("--stock-record", record, "--stock-label", "v0.1.28"), "--stock-label refused"),
                    (self.no_stock("--stock-record", record, "--stock-var", "PORT=1"), "--stock-var refused"),
                    (self.no_stock("--stock-record", record, "--controls-of", record), "no controls-only mode"),
                    (self.no_stock("--stock-record", record, "--peer-unavailable-evidence", str(self.nc)),
                     "--peer-unavailable-evidence would leave this campaign no contrast to measure")]
        for args, reason in refusals:
            with self.subTest(reason=reason), self.assertRaisesRegex(q3.qc.CampaignError, re.escape(reason)):
                q3.build_plan(args, {}, "qwen38")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            q3.parse_args(["--dry-run", "--no-stock-arm", "--stock-record", record])  # exactly one --family
        default, _ = q3.build_plan(self.args(), {}, "qwen38")
        self.assertEqual((default["measured_arms"], default["declared"]["stock"]["label"]),
                         (["stock", "experiment", "peer"], "v0.1.29 release"))
        self.assertEqual((default["design"]["stock_margin_ratio"], default["design"]["peer_margin_ratio"]),
                         (0.95, 1.0))
        self.assertNotIn("stock_record", default)

    def test_the_record_must_be_a_complete_same_artifact_stock_primary(self):
        record = self.record()
        with mock.patch.object(q3, "STOCK_RECORD_FAMILIES", ("deepseek4",)), \
                self.assertRaisesRegex(q3.qc.CampaignError, "--stock-record applies to deepseek4 only"):
            q3.build_plan(self.no_stock("--stock-record", str(record)), {}, "qwen38")
        with self.assertRaisesRegex(q3.qc.CampaignError, "is family qwen38, not gemma4"):
            with mock.patch.object(q3, "STOCK_RECORD_FAMILIES", ("gemma4",)):
                q3.build_plan(self.no_stock("--stock-record", str(record)), {}, "gemma4")
        incomplete = self.root / "evidence" / "q3-incomplete"
        incomplete.mkdir()
        shutil.copyfile(record.parent / "campaign.json", incomplete / "campaign.json")
        verdict = json.loads(record.read_text())
        verdict["primary"]["status"] = "Running"
        (incomplete / "verdict.json").write_text(json.dumps(verdict))
        with self.assertRaisesRegex(q3.qc.CampaignError, r"is not a complete qwen38 primary \(stage Running\)"):
            q3.build_plan(self.no_stock("--stock-record", str(incomplete / "verdict.json")), {}, "qwen38")
        (incomplete / "campaign.json").write_text((record.parent / "campaign.json").read_text() + " ")
        verdict["primary"]["status"] = "Complete"
        (incomplete / "verdict.json").write_text(json.dumps(verdict))
        with self.assertRaisesRegex(q3.qc.CampaignError, "does not bind"):
            q3.build_plan(self.no_stock("--stock-record", str(incomplete / "verdict.json")), {}, "qwen38")
        # a stock-record campaign is itself never a stock record
        driver, _ = self.campaign(record)
        q3.write_verdict(self.out, driver.run(), [], None)
        self.out = self.out.parent / "q3-qwen38-again"
        with self.assertRaisesRegex(q3.qc.CampaignError, "is a no-stock campaign itself"):
            q3.build_plan(self.no_stock("--stock-record", str(self.out.parent / "q3-qwen38-ds4file" / "verdict.json")),
                          {}, "qwen38")
        # nor is a native-peer campaign
        native = self.root / "evidence" / "q3-deepseek4-native"
        native.mkdir()
        document = native_manifest("ds4")
        document.update(schema=q3.SCHEMA, identities_sha256=q3.qc.plan_sha256(document["identities"]))
        (native / "campaign.json").write_text(json.dumps(document))
        q3.save(native / "verdict.json", q3_report.compute_verdict(document, campaign_sha256=q3.sha256(native / "campaign.json")))
        with mock.patch.object(q3, "STOCK_RECORD_FAMILIES", ("deepseek4",)), \
                self.assertRaisesRegex(q3.qc.CampaignError, "is a native campaign"):
            q3.build_plan(self.no_stock("--stock-record", str(native / "verdict.json")), {}, "deepseek4")

    def test_binding_launch_and_a_win_from_the_bound_record(self):
        record = self.record()
        recorded = json.loads(record.read_text())
        driver, runner = self.campaign(record)
        plan = driver.plan
        self.assertEqual((plan["measured_arms"], plan["declared"]["stock"], set(plan["arms"])),
                         (["experiment", "peer"], None, {"experiment", "peer"}))
        bound = plan["stock_record"]
        self.assertEqual((bound["verdict"], bound["campaign"]),
                         ({"path": str(record), "sha256": q3.sha256(record)},
                          {"path": str(record.parent / "campaign.json"), "sha256": q3.sha256(record.parent / "campaign.json")}))
        self.assertEqual((bound["family"], bound["primary_status"], bound["stock_binary_sha256"],
                          bound["experiment_binary_sha256"], bound["artifact_sha256"], bound["stock"]["label"]),
                         ("qwen38", "Complete", STOCK_SHA, EXP_SHA, MODEL_SHA, "v0.1.29 release"))
        self.assertEqual(q3_identity.launch_problems(plan, self.ids(plan)), [])  # no --stock-bin/--stock-commit needed
        manifest = driver.run()
        self.assertEqual((manifest["status"], manifest["stages"][0]["stop"]), ("Complete", {"reason": "precision", "blocks": 5}))
        self.assertEqual({c[1] for c in runner.calls}, {"experiment", "peer"})
        self.assertEqual(set(q3_report.required_effects(manifest["stages"][0], manifest)), {"vs_peer"})
        verdict = q3.write_verdict(self.out, manifest, [], None)
        self.assertEqual(verdict["verdict"], "Win", verdict["reasons"])
        self.assertNotIn("vs_stock", verdict["primary"]["effects"])
        self.assertEqual(verdict["stock_record"]["effects"]["vs_stock"], recorded["primary"]["effects"]["vs_stock"])
        self.assertEqual(verdict["stock_record"]["secondary"]["worst_slot_ttft_p95_s"]["vs_stock"],
                         recorded["primary"]["secondary"]["worst_slot_ttft_p95_s"]["vs_stock"])
        self.assertEqual((verdict["stock_record"]["status"], verdict["stock_record"]["path"],
                          verdict["stock_record"]["sha256"]), ("Complete", str(record), q3.sha256(record)))
        # the plan, binding included, is what a resume reproduces
        self.assertEqual(q3.qc.plan_sha256(self.campaign(record)[0].plan), manifest["plan_sha256"])

    def test_experiment_binary_mismatch_is_refused_at_launch_and_halts_before_a_run(self):
        record = self.record()
        driver, _ = self.campaign(record)
        ids = self.ids(driver.plan)
        ids["arms"]["experiment"]["binary"]["sha256"] = "0" * 64
        self.assertIn(f"the stock record {record} was measured on experiment binary {EXP_SHA}, not this campaign's "
                      f"--experiment-bin ({'0' * 64})", q3_identity.launch_problems(driver.plan, ids))
        with self.assertRaisesRegex(q3.qc.CampaignError, "was measured on experiment binary"):
            self.campaign(record, identify=lambda: ids)[0].run()
        calls = itertools.count()

        def identify():
            current = self.ids(plan)
            if next(calls) > 4:  # the experiment binary is rebuilt mid-campaign
                current["arms"]["experiment"]["binary"]["sha256"] = "0" * 64
            return current
        self.out = self.out.parent / "q3-qwen38-rebuilt"
        driver, _ = self.campaign(record, identify=identify)
        plan = driver.plan
        manifest = driver.run()
        self.assertEqual(manifest["status"], "IdentityDrift")
        self.assertIn("arms.experiment.binary.sha256", manifest["identity_drift"])

    def test_a_record_changed_after_binding_halts_the_campaign(self):
        record = self.record()
        original = record.read_bytes()
        driver, _ = self.campaign(record)
        record.write_bytes(original + b" ")
        self.assertIn(f"the stock record {record} changed since the campaign bound it (freeze the record read-only "
                      "before binding)", q3_identity.launch_problems(driver.plan, self.ids(driver.plan)))
        record.write_bytes(original)

        def decide(arm, workload, call):
            if call == 4:
                record.write_bytes(original + b"\n")  # the record is recomputed while the campaign runs
            return self.win(arm, workload, call)
        driver, _ = self.campaign(record, decide=decide)
        manifest = driver.run()
        self.assertEqual(manifest["status"], "IdentityDrift")
        self.assertEqual(manifest["identity_drift"], ["stock_record.verdict"])
        with self.assertRaises(q3.qc.CampaignError):
            self.campaign(record)[0].run(resume=True)
        verdict = q3_report.compute_verdict(manifest)  # a changed record never supplies a vs-stock effect
        self.assertEqual((verdict["verdict"], verdict["stock_record"]["status"]), ("Pending", "Changed"))
        self.assertEqual(verdict["reasons"], [f"the bound stock record {record} is Changed: {record} changed since the "
                                              "campaign bound it; no vs-stock effect"])
        record.unlink()
        self.assertEqual(q3_report.compute_verdict(manifest)["stock_record"]["status"], "Missing")

    def test_a_stage_without_a_remaining_contrast_stops_at_once(self):
        record = self.record()

        def peer_oom(arm, workload, call):
            return {"outcome": "oom"} if arm == "peer" else self.win(arm, workload, call)
        driver, runner = self.campaign(record, decide=peer_oom, workloads="n4,n1")
        manifest = driver.run()
        self.assertEqual(manifest["status"], "Complete")
        self.assertEqual([(s["workload"], s["status"], s["stop"]) for s in manifest["stages"]],
                         [("n4", "Complete", {"reason": q3_stats.NO_CONTRAST, "blocks": 0}),
                          ("n1", "Complete", {"reason": q3_stats.NO_CONTRAST, "blocks": 0})])
        self.assertEqual(len(runner.calls), 2)  # the two n4 warmups; never twenty blocks of experiment-only runs
        self.assertEqual(q3_report.required_effects(manifest["stages"][0], manifest), {})
        verdict = q3.write_verdict(self.out, manifest, [], None)
        self.assertEqual(verdict["verdict"], "Unavailable")
        # a plan with a stock arm keeps its contrast and its stopping rule
        self.out = self.out.parent / "q3-qwen38-with-stock"
        driver, _ = self.driver(peer_oom)
        self.assertEqual(driver.run()["stages"][0]["stop"], {"reason": "precision", "blocks": 5})

    def test_failures_thermal_and_memory_come_from_both_campaigns(self):
        def control_failure(arm, workload, call):  # the record's n1 control stops on an experiment failure
            return {"outcome": "fail"} if arm == "experiment" and workload == "n1" else self.win(arm, workload, call)
        record = self.record(decide=control_failure, workloads="n4,n1", thermal=lambda: 2)
        recorded = json.loads(record.read_text())
        self.assertTrue(recorded["experiment_failures"])
        self.assertIs(recorded["thermal_valid"], False)
        driver, _ = self.campaign(record)
        verdict = q3.write_verdict(self.out, driver.run(), [], None)
        self.assertEqual(verdict["experiment_failures"], [f"{f} (stock record {record})" for f in recorded["experiment_failures"]])
        self.assertIs(verdict["thermal_valid"], False)
        self.assertEqual(verdict["verdict"], "Failed")
        self.assertIn(f"(stock record {record})", verdict["reasons"][0])
        self.assertEqual(q3_report.tristate_and(True, None), None)
        self.assertEqual((q3_report.tristate_and(True, True), q3_report.tristate_and(None, False)), (True, False))

    def test_family_verdict_names_the_record_and_waits_for_it(self):
        source = {"record": "/e/q3/verdict.json", "sha256": "f" * 64, "status": "Complete", "reason": None}
        named = " (from the bound stock record /e/q3/verdict.json, sha256 " + "f" * 64 + ")"
        self.assertEqual(q3_stats.family_verdict(state(stock_source=source)), ("Win", []))
        verdict, reasons = q3_stats.family_verdict(state(stock_source=source, vs_stock=effect_of([1.0] * 5),
                                                         fairness={"worst_slot_ttft_p95_s": effect_of([1.2] * 5)}))
        self.assertEqual(verdict, "Inconclusive")
        self.assertIn("stock margin not demonstrated (95% CI upper bound of the ratio above 0.95)" + named, reasons)
        self.assertIn("fairness regression vs stock: worst_slot_ttft_p95_s" + named, reasons)
        wide = effect_of([0.5, 0.9, 0.6, 0.95, 0.7])
        self.assertTrue(q3_stats.family_verdict(state(stock_source=source, vs_stock=wide))[1][0].endswith(named))
        for status in ("Missing", "Changed", "Running"):
            with self.subTest(status=status):
                verdict, reasons = q3_stats.family_verdict(state(stock_source={**source, "status": status, "reason": "r"}))
                self.assertEqual(verdict, "Pending")
                self.assertEqual(reasons, [f"the bound stock record /e/q3/verdict.json is {status}: r; no vs-stock effect"])

    def test_rows_cite_the_record_for_the_vs_stock_effect(self):
        record = self.record()
        driver, _ = self.campaign(record)
        q3.write_verdict(self.out, driver.run(), [], None)
        record_ref = "$E/evidence/q3-qwen38/verdict.json"
        for of_record in (False, True):
            with self.subTest(of_record=of_record):
                rows = q3_report.rows(self.out, self.root, of_record)  # --of-record is accepted for this mode
                [group] = rows["groups"].values()
                self.assertIn(f"4 slots; stock v0.1.29 release on hf2q's artifact, from record {record_ref}; experiment "
                              "hf2q 22222222 vs the peer, Williams blocks.", group["workload"])
                stock = [r for r in rows["rows"] if r.get("baseline_arm") == "v0128_stock"]
                self.assertEqual({r["metric"] for r in stock},
                                 {"makespan_s", *(q3_report.f2_q3_tier.FAIRNESS if of_record else ())})
                ms = next(r for r in stock if r["metric"] == "makespan_s")
                self.assertEqual(ms["verification"]["key"], "verdict.json stock_record.effects.vs_stock.improvement_percent")
                self.assertEqual((ms["artifact_sha256"], ms["receipt"]),
                                 (MODEL_SHA, f"$E/evidence/q3-qwen38-ds4file/verdict.json"))
                self.assertIn(f"(baseline stock v0.1.29 release, from the same-artifact record {record_ref} on hf2q's "
                              "artifact)", ms["note"])
                peer = next(r for r in rows["rows"] if r.get("baseline_arm") == "peer" and r["metric"] == "makespan_s")
                self.assertEqual(peer["artifact_sha256"], DS4_ART)
                board = {"schema": f2.ROWS_SCHEMA, "evidence_root_symbol": "$E", "evidence_root_description": "test",
                         "neutral_band_ratio": 1.025, "neutral_band_basis": "test",
                         "families": {"qwen38": {"title": "Qwen3.8 27B", "artifact_sha256": DS4_ART, "note": ""}}, **rows}
                f2.validate_manifest(board)
                self.assertEqual(f2.verify_rows(board, self.root), [r["id"] for r in rows["rows"]])
                if of_record:
                    self.assertEqual(group["q3_record"]["stock_record"],
                                     {"verdict_ref": record_ref, "artifact_sha256": MODEL_SHA, "stock_label": "v0.1.29 release"})
                    line = next(l for l in f2.render(board, [], "0" * 64).splitlines() if l.startswith("| Qwen3.8 27B | "))
                    self.assertIn("sooner than v0.1.28 stock (hf2q artifact record) (95% CI", line)
                    self.assertIn("| vs stock (hf2q artifact record): TTFT p95 ", line)


if __name__ == "__main__":
    unittest.main()
