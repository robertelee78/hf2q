"""Pure controls for the ADR-063 Q3 campaign: design, estimator, verdicts, receipts and the driver.

The driver tests replace q5_calibration.py with a fake runner that writes receipt-shaped files;
they prove scheduling, stopping, identity freezing and classification, never model speed.
"""
import contextlib
import io
import itertools
import json
import math
import os
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
        # 2.5 points is the boundary: a 2.6-point half-width keeps running
        self.assertTrue(q3_stats.precise(effect_of([0.9, 0.9 * 1.021, 0.9, 0.9 * 1.021, 0.9, 0.9 * 1.021])))
        self.assertFalse(q3_stats.precise(effect_of([0.9, 0.96, 0.9, 0.96, 0.9])))


def state(**changes):
    base = {"experiment_failures": [], "peer_unavailable": None, "peer_failures": [],
            "quality": {"status": "parity", "reason": None}, "complete": True,
            "vs_stock": effect_of([0.8] * 5), "vs_peer": effect_of([0.8] * 5),
            "fairness": {"worst_slot_ttft_p95_s": effect_of([1.0] * 5)}, "thermal_valid": True, "memory_valid": True}
    base.update(changes)
    return base


class Verdicts(unittest.TestCase):
    def test_win_needs_stock_gain_and_peer_margin(self):
        self.assertEqual(q3_stats.family_verdict(state())[0], "Win")
        self.assertEqual(q3_stats.family_verdict(state(vs_peer=effect_of([0.97] * 5)))[0], "Inconclusive")
        self.assertEqual(q3_stats.family_verdict(state(vs_stock=effect_of([1.0] * 5)))[0], "Inconclusive")

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
        driver, _ = self.driver(decide)
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


if __name__ == "__main__":
    unittest.main()
