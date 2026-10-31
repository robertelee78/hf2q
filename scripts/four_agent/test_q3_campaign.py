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
           "harness": {"git_commit": "c" * 40, "git_dirty": [], "sources": {}}, "python": {}, "host": {},
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
        self.assertEqual({r["baseline_arm"] for r in rows["rows"]}, {"hf2q_current", "peer"})
        self.assertAlmostEqual(rows["rows"][0]["value"], 20.0, places=0)


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
