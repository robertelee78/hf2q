"""Crash-resume and halt-reporting controls of the actual Q3 and quality drivers.

The fake runners stand in for q5_calibration.py: they write receipt-shaped files and fake-service
client directories (raw turns + coordinator summary), then the driver "dies" by raising before it
settles the run. The resumed driver must score that Launched attempt from disk through the Q2 run
audit, never relaunch it, and keep an Invalid run terminal.
"""
import json
from pathlib import Path
import tempfile
import unittest

import q3_campaign as q3
import q3_report
import quality_campaign as qc
from test_q3_campaign import Harness, noisy
from test_quality_campaign import QWEN38, FakeRuns


class Crash(Exception):
    """The driver process ends while a run is in flight, after Q5 wrote its evidence."""


def win(arm, workload, call):
    return {"makespan": noisy({"experiment": 80, "stock": 100}.get(arm, 100), call, 0.002)}


def milliseconds(summary):
    summary["own_output_four_task_makespan_seconds"] /= 1000  # unit forgery: the audit recomputes it


def forge_client(client, forge):
    summary = json.loads((client / "summary.json").read_text())
    forge(summary)
    (client / "summary.json").write_text(json.dumps(summary))


class Q3LaunchedResume(Harness):
    def crash_at(self, call, forge=None):
        """Run a campaign whose driver dies during runner call `call`; returns the in-flight entry name."""
        driver, inner = self.driver(win)

        def runner(argv, env, stdout, stderr):
            code = inner(argv, env, stdout, stderr)
            if len(inner.calls) == call:
                if forge is not None:
                    forge_client(Path(argv[5]) / "client", forge)
                raise Crash(argv[5])
            return code
        driver.runner = runner
        with self.assertRaises(Crash):
            driver.run()
        manifest = json.loads((self.out / "campaign.json").read_text())
        self.assertEqual(manifest["status"], "Running")
        launched = [e for e in q3_report.entries(manifest) if e["attempts"] and e["attempts"][-1]["outcome"] == "Launched"]
        self.assertEqual(len(launched), 1)
        self.assertEqual(launched[0]["kind"], "measured")  # calls 1-3 are the three warmups
        return launched[0]["name"]

    @staticmethod
    def entry(manifest, name):
        return next(e for e in q3_report.entries(manifest) if e["name"] == name)

    def test_launched_honest_run_is_scored_from_disk_and_not_relaunched(self):
        name = self.crash_at(4)
        resumed, runner = self.driver(win)
        manifest = resumed.run(resume=True)
        self.assertEqual(manifest["status"], "Complete")
        entry = self.entry(manifest, name)
        self.assertEqual([(a["outcome"], a["q5_exit"]) for a in entry["attempts"]], [("Pass", None)])
        self.assertEqual(entry["outcome"], "Pass")
        self.assertIsNotNone(entry["metrics"]["audit"])  # the Q2 run audit re-derived the claimed Pass
        self.assertFalse(any(call[2].startswith(name) for call in runner.calls))

    def test_launched_forged_run_becomes_invalid_and_is_not_relaunched(self):
        name = self.crash_at(4, milliseconds)
        resumed, runner = self.driver(win)
        manifest = resumed.run(resume=True)
        self.assertEqual(manifest["status"], "Invalid")
        self.assertEqual(runner.calls, [])
        self.assertEqual((manifest["halted"]["entry"], manifest["halted"]["resumable"]), (name, False))
        self.assertIn("makespan seconds disagree with its clocks", manifest["halted"]["reason"])
        entry = self.entry(manifest, name)
        self.assertEqual([a["outcome"] for a in entry["attempts"]], ["Invalid"])
        self.assertEqual(entry["outcome"], "Invalid")
        with self.assertRaisesRegex(q3.qc.CampaignError, "^campaign is Invalid: "):
            self.driver(win)[0].run(resume=True)


class DefaultModeControlInvalid(Harness):
    def test_control_invalid_keeps_the_primary_verdict_and_states_the_campaign_status(self):
        driver, inner = self.driver(win, workloads="n4,n1")
        control = f"{q3.SLUGS['n1']}-experiment"

        def runner(argv, env, stdout, stderr):
            code = inner(argv, env, stdout, stderr)
            if Path(argv[3]).stem == control:
                forge_client(Path(argv[5]) / "client", milliseconds)
            return code
        driver.runner = runner
        manifest = driver.run()
        self.assertEqual(manifest["status"], "Invalid")
        self.assertEqual([(s["workload"], s["status"]) for s in manifest["stages"]], [("n4", "Complete"), ("n1", "Running")])
        self.assertIs(manifest["halted"]["resumable"], False)
        verdict = q3.write_verdict(self.out, manifest, [], None)
        self.assertEqual((verdict["verdict"], verdict["campaign_status"]), ("Win", "Invalid"))
        self.assertEqual(verdict["halted"], manifest["halted"])
        self.assertIn("makespan seconds disagree with its clocks", verdict["halted"]["reason"])
        self.assertEqual(verdict["primary"]["status"], "Complete")
        self.assertTrue(any(t["outcome"] == "Invalid" for t in verdict["rejected_trials"]))
        rows = q3_report.rows(self.out, self.root)["rows"]
        self.assertEqual(len(rows), 2)  # primary vs stock and vs peer
        for row in rows:
            self.assertTrue(row["note"].startswith("Q3 verdict Win, campaign Invalid; "), row["note"])


class CrashingRuns(FakeRuns):
    """FakeRuns whose first run is left Launched: the driver dies after the run's evidence is on disk."""

    def __init__(self, forge=None):
        super().__init__(lambda instance, arm: "pass")
        self.forge = forge

    def __call__(self, argv, env, stdout, stderr):
        super().__call__(argv, env, stdout, stderr)
        output = Path(argv[argv.index("--output") + 1])
        if self.forge is not None:
            forge_client(output / "client", self.forge)
        raise Crash(str(output))


class QualityLaunchedResume(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "nc.json").write_text("{}")
        self.out = root / "quality"
        args = qc.parse_args(["--subject-case", str(QWEN38), "--reference-case", str(QWEN38), "--reference-kind",
                              "stock-hf2q", "--instances", "1", "--negative-control-receipt", str(root / "nc.json"),
                              "--output", str(self.out)])
        self.plan, self.cases = qc.build_plan(args, {})

    def crash_first_run(self, forge=None):
        with self.assertRaises(Crash):
            qc.run(self.plan, self.cases, self.out, {}, CrashingRuns(forge))
        manifest = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual([a["outcome"] for a in manifest["runs"][0]["attempts"]], ["Launched"])

    def test_launched_honest_run_is_scored_from_disk_and_not_relaunched(self):
        self.crash_first_run()
        runs = FakeRuns(lambda instance, arm: "pass")
        manifest = qc.run(self.plan, self.cases, self.out, {}, runs, resume=True)
        self.assertEqual(manifest["status"], "Complete")
        first = manifest["runs"][0]
        self.assertEqual([(a["outcome"], a["q5_exit"]) for a in first["attempts"]], [("Pass", None)])
        self.assertEqual(first["outcome"], "Pass")
        self.assertIsNotNone(first["attempts"][0]["audit"])
        self.assertEqual(runs.calls, [(e["instance"], e["arm"]) for e in self.plan["schedule"][1:]])
        self.assertEqual(manifest["verdict"]["verdict"], "Insufficient")  # one instance, below the 24 minimum

    def test_launched_forged_run_becomes_invalid_and_is_not_relaunched(self):
        def unscored(summary):
            summary["agents"]["agent-1"]["tools"][-1]["passed"] = False  # status still claims success
        self.crash_first_run(unscored)
        runs = FakeRuns(lambda instance, arm: "pass")
        with self.assertRaisesRegex(qc.CampaignError, r"^run 0 is Invalid \(run audit: .*final tests/source.*\); no reruns"):
            qc.run(self.plan, self.cases, self.out, {}, runs, resume=True)
        self.assertEqual(runs.calls, [])
        manifest = json.loads((self.out / "manifest.json").read_text())
        first = manifest["runs"][0]
        self.assertEqual([a["outcome"] for a in first["attempts"]], ["Invalid"])
        self.assertIsNone(first["outcome"])
        self.assertNotIn("verdict", manifest)


if __name__ == "__main__":
    unittest.main()
