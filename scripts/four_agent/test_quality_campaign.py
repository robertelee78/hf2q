"""Pure controls for the paired quality campaign: verdict math, ABBA schedule, derived cases, scoring."""
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest

import coding_instances
import q2_fake_service as fake
import q5_calibration
import q5_case
import quality_campaign as qc
from qwen_n4_live_smoke import MODULES

HERE = Path(__file__).resolve().parent
QWEN38 = HERE / "q5_cases" / "qwen38.json"


def document(pairs, reference_kind="peer"):
    return {"schema": qc.OUTCOMES_SCHEMA, "reference_kind": reference_kind,
            "instances": [{"id": f"i{i:02d}", "subject": s, "reference": r} for i, (s, r) in enumerate(pairs)]}


def operating(subject, reference, n=24):
    """Exact verdict probabilities for independent instances (policy §3.5 table)."""
    cells = (subject * reference, (1 - subject) * reference, subject * (1 - reference),
             (1 - subject) * (1 - reference))
    totals = dict.fromkeys(("Pass", "Loss", "Inconclusive", "BelowFloor"), 0.0)
    for b in range(n + 1):
        for c in range(n + 1 - b):
            for d in range(n + 1 - b - c):
                a = n - b - c - d
                weight = (math.factorial(n) // (math.factorial(a) * math.factorial(b) * math.factorial(c)
                                                * math.factorial(d))) * cells[0] ** a * cells[1] ** b * \
                    cells[2] ** c * cells[3] ** d
                if weight:
                    pairs = [(True, True)] * a + [(False, True)] * b + [(True, False)] * c + [(False, False)] * d
                    totals[qc.paired(document(pairs))["verdict"]] += weight
    return totals


class Verdict(unittest.TestCase):
    def test_policy_examples(self):
        self.assertEqual(qc.one_sided_sign_p(2, 0), 0.25)  # the recorded Qwen3.6 cohort
        self.assertAlmostEqual(qc.one_sided_sign_p(5, 1), 7 / 64)
        self.assertEqual(qc.one_sided_sign_p(0, 0), 1.0)
        both = [(True, True)]
        cases = [(both * 23, "Insufficient"), (both * 24, "Pass"),
                 (both * 19 + [(False, True)] * 5, "Loss"),            # p = 1/32
                 (both * 20 + [(False, True)] * 4, "Inconclusive"),    # p = 1/16, b - c = 4 > 3
                 (both * 21 + [(False, True)] * 3, "Pass"),            # b - c = 3 is allowed
                 (both * 17 + [(False, False)] * 7, "BelowFloor"),     # 17/24 < 75%
                 (both * 18 + [(False, False)] * 6, "Pass")]
        for pairs, verdict in cases:
            with self.subTest(verdict=verdict, n=len(pairs)):
                self.assertEqual(qc.paired(document(pairs))["verdict"], verdict)
        loss = qc.paired(document(both * 19 + [(False, True)] * 5))
        self.assertEqual((loss["subject_only_fail"], loss["reference_only_fail"], loss["one_sided_p"]), (5, 0, 1 / 32))

    def test_floor_only_without_reference(self):
        self.assertEqual(qc.paired(document([(True, None)] * 18 + [(False, None)] * 6, None))["verdict"], "Pass")
        self.assertEqual(qc.paired(document([(True, None)] * 17 + [(False, None)] * 7, None))["verdict"], "BelowFloor")
        with self.assertRaises(qc.CampaignError):
            qc.paired(document([(True, None)] * 23 + [(True, True)]))
        with self.assertRaises(qc.CampaignError):
            qc.paired(document([(1, True)] * 24))

    def test_operating_characteristics_match_policy_table(self):
        for (subject, reference), expected in {
                (1.0, 1.0): (1.00, 0, 0, 0), (0.95, 1.0): (0.97, 0.006, 0.024, 0),
                (0.80, 1.0): (0.26, 0.54, 0.20, 0), (0.80, 0.90): (0.65, 0.12, 0.12, 0.11)}.items():
            totals = operating(subject, reference)
            observed = (totals["Pass"], totals["Loss"], totals["Inconclusive"], totals["BelowFloor"])
            with self.subTest(subject=subject, reference=reference):
                for got, want in zip(observed, expected):
                    self.assertAlmostEqual(got, want, delta=0.005 if want < 0.05 else 0.01)

    def test_port_matches_q1_policy_tool_when_present(self):
        try:
            import q1_decision_policy
        except ImportError:
            self.skipTest("q1_decision_policy.py is not on this branch")
        self.assertEqual(q1_decision_policy.POLICY["schema"], qc.POLICY["schema"])
        for counts in itertools.product(range(0, 25, 3), repeat=3):
            if sum(counts) <= 24:
                pairs = ([(False, True)] * counts[0] + [(True, False)] * counts[1] + [(False, False)] * counts[2] +
                         [(True, True)] * (24 - sum(counts)))
                self.assertEqual(q1_decision_policy.paired(document(pairs)), qc.paired(document(pairs)))


class Schedule(unittest.TestCase):
    def test_abba_order_and_one_run_per_arm(self):
        order = qc.schedule(4, ("subject", "reference"))
        self.assertEqual("".join(e["arm"][0] for e in order), "srrssrrs")
        self.assertEqual([e["index"] for e in order], list(range(8)))
        self.assertEqual(sorted((e["instance"], e["arm"]) for e in order),
                         sorted(itertools.product(range(4), ("reference", "subject"))))
        self.assertEqual([(e["instance"], e["arm"]) for e in qc.schedule(2, ("subject",))],
                         [(0, "subject"), (1, "subject")])


class DerivedCase(unittest.TestCase):
    def test_derived_case_freezes_the_sources_the_client_checks(self):
        case = q5_case.loads(QWEN38.read_bytes())
        derived = qc.derive_case(case, 7)
        self.assertEqual(derived["client"]["argv"][-2:], ["--instance", "7"])
        self.assertEqual(case["client"]["argv"], derived["client"]["argv"][:-2])
        values = {"REPO": str(HERE.parent.parent)}
        approval = derived["client"]["approval"]["set"]
        self.assertEqual(approval["instance"], 7)
        self.assertEqual(q5_calibration.resolve_set(approval["instance_table_sha256"], values),
                         coding_instances.load()[1])
        sources = q5_calibration.resolve_set(approval["sources"], values)
        self.assertEqual(set(sources), set(MODULES) | set(coding_instances.MODULES))
        self.assertNotEqual(derived["workload"]["id"], case["workload"]["id"])
        builtins = q5_calibration.builtins(Path("/private/tmp/quality-campaign-dry"))
        plan = q5_case.build_plan(derived, builtins, {})
        self.assertIn("--instance", plan["client"]["argv"])

    def test_refuses_pinned_or_foreign_clients(self):
        case = q5_case.loads(QWEN38.read_bytes())
        for mutate in (lambda c: c["client"]["argv"].extend(["--instance", "1"]),
                       lambda c: c["client"]["approval"]["set"].update(instance=1),
                       lambda c: c["client"]["argv"].__setitem__(4, "{REPO}/scripts/other_client.py"),
                       lambda c: c["client"]["approval"]["set"].pop("sources")):
            bad = json.loads(json.dumps(case))
            mutate(bad)
            with self.assertRaises((qc.CampaignError, q5_case.CaseError)):
                qc.derive_case(bad, 0)


def receipt(stop=None, client=True, verdict="CalibrationPass", exit_code=0):
    return {"verdict": {"verdict": verdict, "reason": None}, "stop": stop,
            **({"client": {"exit_code": exit_code}} if client else {})}


def summary(status=qc.SUCCESS, released=True, instance=0, table=None):
    return {"status": status, "barrier": {"status": "Released" if released else "Pending"},
            "instance": {"id": instance, "table_sha256": table or coding_instances.load()[1]},
            "agents": {"agent-0": {"status": "ProtocolCompleteAwaitingIndependentPatchReview"}}}


class Classify(unittest.TestCase):
    def outcome(self, r, s):
        return qc.classify(r, s, 0, coding_instances.load()[1])[0]

    def test_task_outcomes_only_after_the_four_agent_release(self):
        self.assertEqual(self.outcome(receipt(), summary()), "Pass")
        self.assertEqual(self.outcome(receipt(verdict="CalibrationFailed"), summary()), "Pass")
        self.assertEqual(self.outcome(receipt(verdict="ClientFailed", exit_code=1), summary("Failed")), "Fail")
        self.assertEqual(self.outcome(receipt({"reason": "server_exited"}), summary("Failed")), "Fail")
        self.assertEqual(self.outcome(receipt({"reason": "client_timeout"}), summary("Failed")), "Fail")
        self.assertEqual(self.outcome(receipt(verdict="PreflightRefused", client=False), None), "NotStarted")
        self.assertEqual(self.outcome(receipt(verdict="ClientFailed", exit_code=1),
                                      summary("Failed", released=False)), "NotStarted")
        self.assertEqual(self.outcome(receipt({"reason": "footprint_cap"}), summary("Failed")), "Invalid")
        self.assertEqual(self.outcome(receipt({"reason": "interrupted"}), summary("Failed")), "Invalid")
        self.assertEqual(self.outcome(receipt(), summary(instance=1)), "Invalid")
        self.assertEqual(self.outcome(receipt(), summary(table="0" * 64)), "Invalid")
        self.assertEqual(self.outcome(None, None), "Invalid")
        self.assertEqual(self.outcome(receipt(), None), "Invalid")


class FakeRuns:
    """Stands in for q5_calibration.py: writes the receipt and client directory a run would leave.

    A passing client directory comes from the model-free fake service (raw turns + coordinator
    summary), so the Q2 run audit re-derives the task success instead of trusting its status."""

    def __init__(self, result):
        self.result, self.calls = result, []

    def __call__(self, argv, env, stdout, stderr):
        case = json.loads(Path(argv[3]).read_text())
        output = Path(argv[argv.index("--output") + 1])
        instance = int(case["client"]["argv"][-1])
        arm = "reference" if "-reference.json" in argv[3] else "subject"
        self.calls.append((instance, arm))
        kind = self.result(instance, arm)
        if kind == "refused":
            output.mkdir()
            (output / "receipt.json").write_text(json.dumps(receipt(verdict="PreflightRefused", client=False)))
            return 2
        output.mkdir()
        stop = {"reason": "footprint_cap"} if kind == "stopped" else None
        value = receipt(stop, exit_code=0 if kind == "pass" else 1)
        value.update(cleanup={"proven": True}, cleanup_proven=True)
        (output / "receipt.json").write_text(json.dumps(value))
        if kind == "pass":
            fake.honest_run(output / "client", instance=instance)
            return 0
        (output / "client").mkdir()
        (output / "client" / "summary.json").write_text(json.dumps(summary("Failed", instance=instance)))
        return 1


class Campaign(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        (root / "nc.json").write_text("{}")
        self.out = root / "campaign"
        self.args = qc.parse_args(["--subject-case", str(QWEN38), "--reference-case", str(QWEN38),
                                   "--reference-kind", "stock-hf2q", "--instances", "24",
                                   "--negative-control-receipt", str(root / "nc.json"), "--output", str(self.out)])

    def test_complete_campaign_scores_paired_outcomes(self):
        plan, cases = qc.build_plan(self.args, {})
        runs = FakeRuns(lambda i, arm: "fail" if arm == "subject" and i < 5 else "pass")
        manifest = qc.run(plan, cases, self.out, {}, runs)
        self.assertEqual(manifest["status"], "Complete")
        self.assertEqual(runs.calls, [(e["instance"], e["arm"]) for e in plan["schedule"]])
        self.assertEqual(manifest["verdict"]["verdict"], "Loss")
        self.assertEqual(manifest["verdict"]["subject_only_fail"], 5)
        outcomes = json.loads((self.out / "outcomes.json").read_text())
        self.assertEqual(outcomes["reference_kind"], "stock-hf2q")
        self.assertEqual([i["subject"] for i in outcomes["instances"][:6]], [False] * 5 + [True])
        first = manifest["runs"][0]["attempts"][0]
        self.assertEqual((first["outcome"], first["client_status"]), ("Fail", "Failed"))

    def test_refusal_halts_and_resumes_but_started_tasks_never_rerun(self):
        plan, cases = qc.build_plan(self.args, {})
        refused = {3}
        runs = FakeRuns(lambda i, arm: "refused" if (i, arm) == (1, "subject") and refused else "pass")
        manifest = qc.run(plan, cases, self.out, {}, runs)
        self.assertEqual((manifest["status"], manifest["halted_at"]), ("Halted", 3))
        refused.clear()
        manifest = qc.run(plan, cases, self.out, {}, runs, resume=True)
        self.assertEqual((manifest["status"], manifest["verdict"]["verdict"]), ("Complete", "Pass"))
        self.assertEqual([a["outcome"] for a in manifest["runs"][3]["attempts"]], ["NotStarted", "Pass"])
        self.assertEqual(len(runs.calls), 49)
        with self.assertRaises(qc.CampaignError):
            qc.run(plan, cases, self.out, {}, runs, resume=True)

        other = Path(self.directory.name) / "other"
        args = qc.parse_args([*self.args_list(other)])
        plan, cases = qc.build_plan(args, {})
        manifest = qc.run(plan, cases, other, {}, FakeRuns(lambda i, arm: "stopped" if i == 0 else "pass"))
        self.assertEqual(manifest["status"], "Halted")
        with self.assertRaises(qc.CampaignError):
            qc.run(plan, cases, other, {}, FakeRuns(lambda i, arm: "pass"), resume=True)

    def test_resume_refuses_a_changed_plan(self):
        plan, cases = qc.build_plan(self.args, {})
        qc.run(plan, cases, self.out, {}, FakeRuns(lambda i, arm: "refused"))
        changed, _ = qc.build_plan(qc.parse_args([*self.args_list(self.out), "--subject-var", "PORT=18240"]), {})
        with self.assertRaises(qc.CampaignError):
            qc.run(changed, cases, self.out, {}, FakeRuns(lambda i, arm: "pass"), resume=True)

    def test_plan_refuses_mismatched_arms_and_unknown_overrides(self):
        with self.assertRaises(qc.CampaignError):
            qc.build_plan(self.args, {"HF2Q_Q5_NOT_A_VARIABLE": "x"})
        with self.assertRaises(qc.CampaignError):
            qc.build_plan(qc.parse_args([*self.args_list(self.out), "--subject-var", "NOPE=1"]), {})
        peer = qc.parse_args([*self.args_list(self.out)[:4], "--reference-kind", "peer",
                              *self.args_list(self.out)[6:]])
        with self.assertRaises(qc.CampaignError):  # an hf2q case cannot be the peer arm
            qc.build_plan(peer, {})
        plan, _ = qc.build_plan(self.args, {"HF2Q_Q5_MODEL": "/abs/model.gguf"})
        self.assertEqual(plan["arms"]["reference"]["overrides"], {"MODEL": "/abs/model.gguf"})
        env = qc.arm_environ({"HF2Q_Q5_MODEL": "/x", "PATH": "/bin"}, {"PORT": "1"})
        self.assertEqual(env, {"PATH": "/bin", "HF2Q_Q5_PORT": "1"})

    def args_list(self, out):
        root = Path(self.directory.name)
        return ["--subject-case", str(QWEN38), "--reference-case", str(QWEN38), "--reference-kind", "stock-hf2q",
                "--instances", "24", "--negative-control-receipt", str(root / "nc.json"), "--output", str(out)]


if __name__ == "__main__":
    unittest.main()
