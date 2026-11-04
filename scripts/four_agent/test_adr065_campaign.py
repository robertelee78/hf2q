"""Controls for the ADR-065 campaign driver and the checked-in B1 campaign file.

The driver tests replace q5_calibration.py with a fake runner writing receipt-shaped files (as
test_q3_campaign.py does); they prove scheduling, fixed counts, cooldown, budget, identity checks,
ceilings and analysis plumbing, never model speed.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import adr065_campaign as ac
import adr065_stats
import q3_campaign as q3
import q5_artifacts
from test_q3_campaign import write_run

HERE = Path(__file__).resolve().parent
BIN_SHA, MODEL_SHA = "b" * 64, "a" * 64
SOURCE = {"path": "/campaign-b1.json", "sha256": "f" * 64}
PLANS = {slug: q3.WORKLOADS[key] for key, slug in q3.SLUGS.items()}


class CampaignFile(unittest.TestCase):
    def setUp(self):
        self.doc, _ = ac.load_campaign()

    def test_names_every_family_case_and_artifact(self):
        self.assertEqual(self.doc["run_order"], ["qwen38", "deepseek4", "gemma4"])
        self.assertEqual(set(self.doc["families"]), set(self.doc["run_order"]))
        for family, fam in self.doc["families"].items():
            case = ac.REPO / fam["q5_case"]["path"]
            self.assertEqual(q5_artifacts.sha256(case), fam["q5_case"]["sha256"], family)
            self.assertEqual(json.loads(case.read_text())["family"], family)
            self.assertRegex(fam["artifact"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertIn(fam["ab_blocks"], self.doc["design"]["ab_block_count"]["allowed"])
        self.assertEqual(self.doc["design"]["ab_block_count"]["default"], adr065_stats.DEFAULT_BLOCKS)
        self.assertEqual(tuple(self.doc["design"]["ab_block_count"]["allowed"]), adr065_stats.ALLOWED_BLOCKS)
        self.assertEqual(self.doc["design"]["verdict_order"], list(adr065_stats.VERDICTS))
        self.assertEqual(self.doc["design"]["margin_points"], adr065_stats.MARGIN_PP)
        fairness = self.doc["design"]["fairness"]
        self.assertEqual((fairness["neutral_ratio"], fairness["cap_factor"]),
                         (q3.q3_stats.NEUTRAL_RATIO, adr065_stats.CAP_FACTOR))
        self.assertEqual(self.doc["workloads"]["order"], list(q3.WORKLOADS))

    def test_declared_budget_matches_the_rule(self):
        execution, declared = self.doc["execution"], self.doc["execution"]["budget"]
        allowance, cooldown = execution["retry_allowance_attempts"], execution["cooldown_seconds"]
        base = ac.budget(self.doc["baseline_campaign"]["runs_per_stage"], ["A"], allowance)
        ab = ac.budget({k: 6 for k in q3.WORKLOADS}, ["A", "B"], allowance)
        self.assertEqual(base, {k: declared["baseline_per_family"][k] for k in base})
        self.assertEqual(ab, {k: declared["ab_per_family_at_6_blocks"][k] for k in ab})
        total = [0, 0]
        for family, fam in self.doc["families"].items():
            seconds = fam["run_seconds_estimate"]
            self.assertEqual(ac.hours(base["planned_runs"], seconds, cooldown), declared["baseline_per_family"]["hours"][family])
            self.assertEqual(ac.hours(ab["planned_runs"], seconds, cooldown), declared["ab_per_family_at_6_blocks"]["hours"][family])
            total = [t + base["planned_runs"] * (s + cooldown) for t, s in zip(total, seconds)]
        self.assertEqual([round(t / 3600, 1) for t in total], declared["baseline_total_hours"])


SOURCE_COMMIT, NATIVE_SUM, BYTES = "1" * 40, "d" * 64, 4096
LOCK = ('[[package]]\nname = "mlx-native"\nversion = "{version}"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\nchecksum = "{checksum}"\n')


def fake_lock(commit, version="0.15.18", checksum=NATIVE_SUM):
    if commit != SOURCE_COMMIT:
        raise ac.subprocess.CalledProcessError(128, ["git", "show"])
    return LOCK.format(version=version, checksum=checksum)


def fake_ids(plan, binary_sha=BIN_SHA, artifact_sha=MODEL_SHA, dirty=()):
    return {"arms": {arm: {"binary": {"path": plan["declared"]["binary"], "sha256": binary_sha, "bytes": BYTES,
                                      "exists": True},
                           "artifact": {"path": "/m.gguf", "bytes": 10, "sha256": artifact_sha},
                           "peer_build_info": None, "missing_variables": [], "file_checks": {}, "inputs": {}}
                     for arm in plan["arms"]},
            "harness": {"git_commit": "c" * 40, "git_dirty": list(dirty), "sources": {}},
            "python": {"path": "/usr/bin/python3", "sha256": "7" * 64}, "host": {"chip": "test"},
            "negative_control_receipt": plan["negative_control_receipt"]}


class Args:
    def __init__(self, mode, **kw):
        self.mode, self.family, self.qe_identity, self.workloads = mode, "qwen38", "qe-test", None
        self.baseline_bin = self.binary = self.b_case = self.blocks = self.pilot_of = None
        self.commit, self.label, self.n1_engages = "2" * 40, "candidate", False
        self.__dict__.update(kw)


class FakeRunner:
    def __init__(self, decide=None):
        self.decide, self.calls = decide or (lambda arm, workload, call: {}), []

    def __call__(self, argv, env, stdout, stderr):
        case, out = Path(argv[3]), Path(argv[5])
        workload, arm = case.stem.split("-", 1)
        self.calls.append((workload, arm, out.name, env))
        spec = {"sha": BIN_SHA, "makespan": 100.0, "phase_plan": PLANS[workload], **self.decide(arm, workload, len(self.calls))}
        write_run(out, **spec)
        return 0


class Clock:
    def __init__(self):
        self.now, self.slept = 1000.0, []

    def __call__(self):
        self.now += 5
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


class Harness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.campaign, _ = ac.load_campaign()
        nc = self.root / "nc.json"
        nc.write_text("{}")
        self.campaign["environment"]["negative_control_receipt"] = {"path": str(nc), "sha256": q5_artifacts.sha256(nc)}
        self.campaign["families"]["qwen38"]["artifact"].update(sha256=MODEL_SHA, bytes=10)
        self.receipt = self.root / "build-receipt.json"
        self.receipt.write_text(json.dumps({"schema": ac.BUILD_RECEIPT_SCHEMA, "source": SOURCE_COMMIT, "sha256": BIN_SHA,
                                            "bytes": BYTES, "native_version": "0.15.18", "native_checksum": NATIVE_SUM}))
        self.campaign["baseline"]["binary"].update(
            source_commit=SOURCE_COMMIT, sha256=BIN_SHA, bytes=BYTES,
            build_receipt={"path": str(self.receipt), "sha256": q5_artifacts.sha256(self.receipt)})
        self.campaign["baseline"]["native"]["checksum"] = NATIVE_SUM
        self.campaign["baseline_campaign"]["runs_per_stage"] = {"n4": 3, "n1": 2, "3+1": 1, "2+2": 1, "1+3": 1}
        self.bin = self.root / "hf2q"
        self.bin.write_bytes(b"\0mlx-native-0.15.18/src\0")
        self.case = HERE / "q5_cases" / "qwen38.json"

    def tearDown(self):
        self.tmp.cleanup()

    def b_case(self, change=lambda case: case["serve"]["env"].update(HF2Q_SWITCH="0")):
        case = json.loads(self.case.read_text())
        change(case)
        path = self.root / f"b-{len(list(self.root.glob('b-*.json')))}.json"
        path.write_text(json.dumps(case))
        return str(path)

    def plan(self, mode="baseline", **kw):
        if mode == "baseline":
            kw.setdefault("baseline_bin", str(self.bin))
        else:
            kw.setdefault("binary", str(self.bin))
            kw.setdefault("b_case", self.b_case())
        return ac.build_plan(Args(mode, **kw), self.campaign, SOURCE)

    def driver(self, plan, cases, out, decide=None, ids=None, versions=lambda path: ["0.15.18"]):
        runner, clock = FakeRunner(decide), Clock()
        driver = ac.Driver(plan, cases, out, {}, runner, lambda: 0, ids or (lambda: fake_ids(plan)),
                           clock=clock, sleep=clock.sleep, versions=versions, lock=fake_lock)
        return driver, runner, clock


class Plan(Harness):
    def test_baseline_plan_binds_the_campaign_file(self):
        plan, cases = self.plan()
        self.assertEqual((plan["mode"], plan["measured_arms"], plan["workloads"]), ("baseline", ["A"], list(q3.WORKLOADS)))
        self.assertEqual(plan["design"]["stage_blocks"], {"n4": 3, "n1": 2, "3+1": 1, "2+2": 1, "1+3": 1})
        self.assertEqual(plan["budget"]["planned_runs"], 5 + 8)
        self.assertEqual(plan["budget"]["max_attempts"], 13 + 6)
        overrides = plan["arms"]["A"]["overrides"]
        self.assertEqual(overrides["HF2Q_BIN"], str(self.bin))
        self.assertEqual(overrides["MODEL"], "/Users/robert/.local/share/hf2q/models/qwen38-stock/Qwen3.8-27B-vanilla.gguf")
        self.assertEqual(overrides["MODEL_SHA256"], MODEL_SHA)
        self.assertTrue(overrides["CLIENT_APPROVAL_TEMPLATE"].endswith("approval-templates-proof-8b286684/qwen38-client-approval.json"))
        self.assertEqual(plan["declared"]["commit"], SOURCE_COMMIT)
        self.assertEqual((plan["baseline"]["kind"], plan["baseline"]["native_version"], plan["baseline"]["native_checksum"]),
                         ("main-build", "0.15.18", NATIVE_SUM))
        self.assertEqual(plan["baseline"]["build_receipt"]["path"], str(self.receipt))
        default, _ = self.plan(baseline_bin=None)
        self.assertEqual(default["declared"]["binary"], "/Users/robert/.local/state/hf2q/v0130/b1-503/main-build/hf2q")
        self.assertEqual(plan["campaign_file"], SOURCE)
        self.assertEqual(ac.schedule(plan)["n4"]["blocks"], [["A"]] * 3)

    def test_changed_case_file_is_refused(self):
        self.campaign["families"]["qwen38"]["q5_case"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(q3.qc.CampaignError, "differs from the campaign file"):
            self.plan()

    def test_ab_kill_switch_block_count_and_pilot(self):
        plan, cases = self.plan("ab")
        self.assertEqual(plan["measured_arms"], ["A", "B"])
        self.assertEqual(plan["arms"]["B"]["kill_switch"], {"HF2Q_SWITCH": [None, "0"]})
        self.assertEqual((plan["design"]["blocks"], plan["n1_control"]), (6, "required"))
        self.assertEqual(ac.schedule(plan)["n4"]["blocks"][:4], [["A", "B"], ["B", "A"], ["A", "B"], ["B", "A"]])
        self.assertEqual(plan["budget"]["planned_runs"], 5 * (2 + 12))
        with self.assertRaisesRegex(q3.qc.CampaignError, "no kill switch"):
            self.plan("ab", b_case=str(self.case))
        with self.assertRaisesRegex(q3.qc.CampaignError, "except serve.env"):
            self.plan("ab", b_case=self.b_case(lambda c: c["serve"]["argv"].append("--x")))
        with self.assertRaisesRegex(q3.qc.CampaignError, "needs --pilot-of"):
            self.plan("ab", blocks=8)
        with self.assertRaisesRegex(q3.qc.CampaignError, "one of"):
            self.plan("ab", blocks=7)
        self.assertEqual(self.plan("ab", n1_engages=True)[0]["n1_control"], "reported")
        pilot_plan, pilot_cases = self.plan("pilot", blocks=8, workloads="n4")
        pilot_dir = self.root / "pilot"
        driver, _, _ = self.driver(pilot_plan, pilot_cases, pilot_dir)
        self.assertEqual(driver.run()["status"], "Complete")
        plan, _ = self.plan("ab", blocks=12, pilot_of=pilot_dir)
        self.assertEqual((plan["design"]["blocks"], plan["pilot_of"]["never_pooled"]), (12, True))
        with self.assertRaisesRegex(q3.qc.CampaignError, "not a qwen38 pilot"):
            self.plan("ab", blocks=12, pilot_of=self.root)

    def test_launch_problems(self):
        plan, _ = self.plan()
        self.assertEqual(ac.launch_problems(plan, fake_ids(plan), ["0.15.18"], fake_lock), [])
        problems = ac.launch_problems(plan, fake_ids(plan, binary_sha="9" * 64, artifact_sha="8" * 64, dirty=["M x"]),
                                      ["0.15.15", "0.15.18"], fake_lock)
        self.assertEqual(len(problems), 4, problems)
        self.assertIn("A: artifact /m.gguf is not the official", problems[0])
        self.assertIn(f"the baseline binary must be the main build {BIN_SHA} ({BYTES} bytes)", problems[1])
        self.assertIn("exactly one mlx-native version (0.15.18)", problems[2])
        ab, _ = self.plan("ab", commit="xyz")
        self.assertIn("--commit must be the 40-hex source commit of the binary",
                      ac.launch_problems(ab, fake_ids(ab), ["0.15.17"], fake_lock))
        mixed = fake_ids(ab)
        mixed["arms"]["B"]["binary"]["sha256"] = "9" * 64
        self.assertIn("every arm must run the same binary (arms differ only in a kill switch)",
                      ac.launch_problems(ab, mixed, ["0.15.18"], fake_lock))

    def test_placeholder_baseline_never_launches(self):
        self.campaign["baseline"]["binary"].update(source_commit=None, sha256=None, bytes=None)
        self.campaign["baseline"]["binary"]["build_receipt"]["sha256"] = None
        self.campaign["baseline"]["native"]["checksum"] = None
        plan, _ = self.plan()
        self.assertIsNone(plan["declared"]["commit"])
        problems = ac.launch_problems(plan, fake_ids(plan), ["0.15.18"], fake_lock)
        self.assertIn("the campaign file's baseline identity is a placeholder (source_commit, sha256, bytes, "
                      "native_checksum, build_receipt.sha256 unset): build the pinned main commit, write its build "
                      "receipt and finalize campaign-b1.json", problems)

    def test_build_receipt_and_native_pin_bind_the_main_build(self):
        plan, _ = self.plan()
        check = lambda **kw: ac.baseline_problems(plan["baseline"], {BIN_SHA}, {BYTES}, ["0.15.18"], **kw)
        self.assertEqual(check(lock=fake_lock), [])
        self.assertEqual(check(lock=lambda c: fake_lock(c, version="0.15.16")),
                         [f"Cargo.lock at {SOURCE_COMMIT} pins mlx-native 0.15.16 ({NATIVE_SUM}), not 0.15.18 ({NATIVE_SUM})"])
        self.assertIn("no mlx-native pin readable from Cargo.lock", check(lock=lambda c: fake_lock("0" * 40))[0])
        self.assertIn("exactly one mlx-native version (0.15.18); found ['0.15.16']",
                      ac.baseline_problems(plan["baseline"], {BIN_SHA}, {BYTES}, ["0.15.16"], fake_lock)[0])
        self.assertIn("must be the main build", ac.baseline_problems(plan["baseline"], {BIN_SHA}, {1}, ["0.15.18"], fake_lock)[0])
        document = json.loads(self.receipt.read_text())
        self.receipt.write_text(json.dumps({**document, "native_checksum": "e" * 64}))
        self.assertIn("missing or not", check(lock=fake_lock)[0])  # the receipt's bytes changed
        plan["baseline"]["build_receipt"]["sha256"] = q5_artifacts.sha256(self.receipt)
        self.assertEqual(check(lock=fake_lock), [f"build receipt {self.receipt} differs from the campaign file: native_checksum"])

    def test_build_receipt_command_reads_a_clean_checkout(self):
        repo = self.root / "src"
        repo.mkdir()
        git = lambda *argv: ac.subprocess.run(["git", "-C", str(repo), *argv], check=True, capture_output=True, text=True).stdout
        git("init", "-q")
        (repo / "Cargo.lock").write_text("lock\n")
        git("add", "Cargo.lock")
        git("-c", "user.email=t@t", "-c", "user.name=t", "-c", "core.hooksPath=/dev/null", "commit", "-qm", "c")
        commit = git("rev-parse", "HEAD").strip()
        lock = lambda c: LOCK.format(version="0.15.18", checksum=NATIVE_SUM) if c == commit else None
        receipt = ac.build_receipt(self.bin, repo, "cargo build --release --locked", lock=lock)
        self.assertEqual({k: receipt[k] for k in ("schema", "source", "sha256", "bytes", "native_version", "native_checksum")},
                         {"schema": ac.BUILD_RECEIPT_SCHEMA, "source": commit, "sha256": q5_artifacts.sha256(self.bin),
                          "bytes": self.bin.stat().st_size, "native_version": "0.15.18", "native_checksum": NATIVE_SUM})
        (repo / "Cargo.lock").write_text("dirty\n")
        with self.assertRaisesRegex(q3.qc.CampaignError, "uncommitted changes"):
            ac.build_receipt(self.bin, repo, "x", lock=lock)
        git("checkout", "-q", "Cargo.lock")
        with self.assertRaisesRegex(q3.qc.CampaignError, "embeds mlx-native"):
            ac.build_receipt(self.bin, repo, "x", lock=lambda c: LOCK.format(version="0.15.16", checksum=NATIVE_SUM))

    def test_native_versions_scan_spans_chunks(self):
        self.assertEqual(ac.native_versions(self.bin, chunk=4), ["0.15.18"])


class Runs(Harness):
    def test_baseline_runs_fixed_counts_with_cooldown_then_freezes_ceilings(self):
        plan, cases = self.plan()
        out = self.root / "baseline"
        driver, runner, clock = self.driver(plan, cases, out)
        manifest = driver.run()
        self.assertEqual(manifest["status"], "Complete", manifest.get("halted"))
        self.assertEqual(manifest["schema"], ac.SCHEMA)
        self.assertEqual([(s["workload"], s["stop"]) for s in manifest["stages"]],
                         [(k, {"reason": "fixed_count", "blocks": n}) for k, n in plan["design"]["stage_blocks"].items()])
        self.assertEqual(len(runner.calls), 13)
        self.assertEqual({c[3]["HF2Q_Q5_HF2Q_BIN"] for c in runner.calls}, {str(self.bin)})
        self.assertEqual(clock.slept[0], 60)  # a session's first load waits the full cooldown
        self.assertTrue(all(abs(w - 55) < 1e-9 for w in clock.slept[1:]))  # 5 s of fake clock elapse per run
        self.assertEqual(len(manifest["cooldowns"]), 13)
        (out / "campaign.json").write_text(json.dumps(manifest))
        doc = ac.write_ceilings(out, SOURCE)
        self.assertEqual(set(doc["ceilings"]), set(q3.WORKLOADS))
        self.assertTrue(all(c["ttft_p95_s"]["value"] > 0 for c in doc["ceilings"].values()))
        self.assertEqual(len(doc["runs"]), 13)
        self.assertEqual(json.loads((out / "ceilings.json").read_text())["schema"], adr065_stats.CEILING_SCHEMA)
        with self.assertRaisesRegex(q3.qc.CampaignError, "campaign file changed"):
            ac.write_ceilings(out, {**SOURCE, "sha256": "0" * 64})
        self.ceilings = out / "ceilings.json"

    def test_ab_retains_a_failed_candidate_run_and_keeps_its_fixed_count(self):
        self.test_baseline_runs_fixed_counts_with_cooldown_then_freezes_ceilings()
        plan, cases = self.plan("ab", workloads="n4,n1")

        def decide(arm, workload, call):
            if workload == "n4" and arm == "B" and call == 8:  # block 3 (A then B) after two warmups
                return {"outcome": "fail"}
            return {"makespan": 90.0 if arm == "B" else 100.0}
        out = self.root / "ab"
        driver, runner, _ = self.driver(plan, cases, out, decide)
        manifest = driver.run()
        self.assertEqual(manifest["status"], "Complete", manifest.get("halted"))
        n4 = manifest["stages"][0]
        self.assertEqual([b["order"] for b in n4["blocks"]], [q3.q3_stats.block_order(["A", "B"], b) for b in range(1, 7)])
        self.assertEqual(len(runner.calls), 2 * (2 + 12))
        doc = ac.write_analysis(out, self.ceilings, SOURCE)
        self.assertEqual(doc["stages"]["n4"]["effect"]["blocks"], 5)
        self.assertEqual(doc["verdict"], "Inconclusive")
        self.assertEqual(len(doc["stages"]["n4"]["quality_ineligible_runs"]), 1)
        self.assertTrue(any("quality-ineligible candidate run" in g for g in doc["gate_failures"]))
        self.assertEqual(len(doc["runs"]), 28)
        self.assertTrue((out / "analysis.json").exists())

    def test_budget_exhaustion_halts_for_good(self):
        plan, cases = self.plan()
        plan["budget"]["max_attempts"] = 3
        driver, runner, _ = self.driver(plan, cases, self.root / "budget")
        manifest = driver.run()
        self.assertEqual((manifest["status"], manifest["halted"]["resumable"]), ("Halted", False))
        self.assertIn("run budget exhausted: 3 attempts", manifest["halted"]["reason"])
        self.assertEqual(len(runner.calls), 3)

    def test_launch_problems_refuse_to_start(self):
        plan, cases = self.plan()
        driver, runner, _ = self.driver(plan, cases, self.root / "refused", versions=lambda path: ["0.15.15"])
        with self.assertRaisesRegex(q3.qc.CampaignError, "mlx-native"):
            driver.run()
        self.assertEqual(runner.calls, [])


if __name__ == "__main__":
    unittest.main()
