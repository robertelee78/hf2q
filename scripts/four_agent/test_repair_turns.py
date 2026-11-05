"""Controls for the Qwen3.6 repair-turn workload (--repair-turns 1). No model, server or GPU; the RunAudit
cases use q2_fake_service's loopback HTTP stub."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import q5_case
import quality_campaign as qc
from q2_certify import replay_response
from qwen38_n1_tools import Tools
from qwen38_n1_verdict import EXPECTED, Invalid, Workflow, assistant, max_tools, qualify
from qwen_n4_live_smoke import AGENTS, agent_command, aggregate
from test_qwen_n4_live_smoke import PID, RUN, TOKEN, client_args, results

HERE = Path(__file__).resolve().parent
CASES = HERE / "q5_cases"
WRITE = ("write_file", None)
SUMMARY = {"role": "assistant", "content": "Done; all tests passed.", "reasoning_content": ""}


def call(name, args, cid):
    return {"role": "assistant", "content": "", "reasoning_content": "plan",
            "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def write_args():
    return {"path": "src/ports.rs", "content": "pub fn parse_port(s: &str) -> Result<u16, String> { todo!() }"}


def run(workflow, outcomes):
    """Drive a Workflow through its own expected calls; `outcomes` are the run_tests results in order.
    Returns the executed call names, ending with None for the summary."""
    names, outcomes, seen = [], list(outcomes), set()
    while True:
        expected = workflow.expected()
        if expected is None:
            workflow.policy(SUMMARY, "stop", len(names), seen)
            workflow.advance(None)
            return names + [None]
        name, args = expected
        cid, got, _ = workflow.policy(call(name, write_args() if args is None else args, f"c{len(names)}"),
                                      "tool_calls", len(names), seen)
        seen.add(cid)
        workflow.advance(got, outcomes.pop(0) if got == "run_tests" else None)
        names.append(got)


def receipts(names, outcomes):
    """Tool receipts as the client records them: run_tests carries `passed`."""
    outcomes, tools = list(outcomes), []
    for name in names:
        tool = {"name": name, "cleanup": True, "status": "Complete", "source_sha256": "final"}
        if name == "run_tests":
            tool["passed"] = outcomes.pop(0)
        tools.append(tool)
    return tools


def turn_receipts(count):
    return [{"turn": i, "stream": i % 2 == 1} for i in range(count)]


class WorkflowProtocol(unittest.TestCase):
    def test_no_failure_is_the_canonical_six_steps(self):
        self.assertEqual(run(Workflow(1), [True, True]), [e[0] for e in EXPECTED] + [None])

    def test_failed_ports_test_gets_one_write_and_rerun_then_resumes(self):
        names = run(Workflow(1), [False, True, True])
        self.assertEqual(names, ["read_file", "read_file", "write_file", "run_tests", "write_file", "run_tests",
                                 "read_file", "run_tests", None])

    def test_failed_final_test_gets_one_repair_on_the_same_target(self):
        workflow = Workflow(1)
        names = run(workflow, [True, False, True])
        self.assertEqual(names[-4:], ["run_tests", "write_file", "run_tests", None])

    def test_a_second_failure_fails_the_agent(self):
        for outcomes in ([False, False], [False, True, False], [True, False, False]):
            with self.subTest(outcomes=outcomes), self.assertRaisesRegex(Invalid, "no repair cycle left"):
                run(Workflow(1), outcomes)

    def test_repair_must_be_write_then_same_target(self):
        workflow = Workflow(1)
        for name, passed in (("read_file", None), ("read_file", None), ("write_file", None), ("run_tests", False)):
            workflow.advance(name, passed)
        self.assertEqual(workflow.expected(), WRITE)
        seen = set()
        with self.assertRaisesRegex(Invalid, "workflow not followed"):  # resuming the six steps skips the repair
            workflow.policy(call("read_file", {"path": "src/ports.rs"}, "x"), "tool_calls", 4, seen)
        workflow.advance("write_file")
        with self.assertRaisesRegex(Invalid, "workflow drift"):
            workflow.policy(call("run_tests", {"target": "all"}, "y"), "tool_calls", 5, seen)
        with self.assertRaisesRegex(Invalid, "exactly one live call"):  # no summary during a repair cycle
            workflow.policy(SUMMARY, "stop", 5, seen)

    def test_summary_only_after_the_workflow_and_nothing_after_it(self):
        workflow = Workflow(1)
        with self.assertRaisesRegex(Invalid, "summary before"):
            workflow.advance(None)
        run(workflow, [True, True])
        with self.assertRaises(Invalid):
            workflow.policy(SUMMARY, "stop", 7, set())
        with self.assertRaises(Invalid):
            workflow.advance(None)

    def test_budget_is_zero_or_one(self):
        self.assertEqual((max_tools(0), max_tools(1)), (6, 8))
        for value in (2, -1, True, "1", None):
            with self.subTest(value=value), self.assertRaises(Invalid):
                Workflow(value)

    def test_a_policy_serves_turns_past_the_canonical_seven(self):
        """Unary replay at turn 7 uses the workflow's expected step, not EXPECTED[7]."""
        workflow = Workflow(1)
        for name, passed in (("read_file", None), ("read_file", None), ("write_file", None), ("run_tests", False),
                             ("write_file", None), ("run_tests", True), ("read_file", None)):
            workflow.advance(name, passed)
        message = call("run_tests", {"target": "all"}, "c7")
        raw = json.dumps({"id": "r7", "object": "chat.completion", "model": "M", "choices": [
            {"index": 0, "message": message, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105,
                      "prompt_tokens_details": {"cached_tokens": 90}}}).encode()
        request = {"model": "M", "stream": False}
        _, got, _ = replay_response(raw, request, 7, set(), "hf2q", assistant_policy=workflow.policy)
        self.assertEqual(got[1:], ("run_tests", {"target": "all"}))
        self.assertIsNone(assistant(SUMMARY, "stop", 6, set()))  # canonical turn 6 is still the summary


class Qualification(unittest.TestCase):
    def test_a_repaired_agent_qualifies_only_with_the_repair_budget(self):
        names = run(Workflow(1), [False, True, True])[:-1]
        tools = receipts(names, [False, True, True])
        turns = turn_receipts(len(names) + 1)
        self.assertEqual(qualify(turns, tools, True, "final", "final", 1),
                         "ProtocolCompleteAwaitingIndependentPatchReview")
        with self.assertRaises(Invalid):
            qualify(turns, tools, True, "final", "final")

    def test_repair_qualification_refuses_bad_receipts(self):
        names = run(Workflow(1), [False, True, True])[:-1]
        tools = receipts(names, [False, True, True])
        cases = {"transport": (turn_receipts(9)[:8] + [{"turn": 8, "stream": True}], tools),
                 "turn count": (turn_receipts(8), tools),
                 "second failure": (turn_receipts(9), receipts(names, [False, False, True])),
                 "final failed": (turn_receipts(9), receipts(names, [False, True, False]))}
        for label, (turns, value) in cases.items():
            with self.subTest(label), self.assertRaises(Invalid):
                qualify(turns, value, True, "final", "final", 1)

    def test_canonical_qualification_is_unchanged(self):
        tools = receipts([e[0] for e in EXPECTED], [True, True])
        self.assertEqual(qualify(turn_receipts(7), tools, True, "final", "final"),
                         "ProtocolCompleteAwaitingIndependentPatchReview")
        self.assertEqual(qualify(turn_receipts(7), tools, True, "final", "final", 1),
                         "ProtocolCompleteAwaitingIndependentPatchReview")


class Coordinator(unittest.TestCase):
    def repaired(self):
        names = run(Workflow(1), [False, True, True])[:-1]
        values = results()
        for agent in AGENTS:
            values[agent].update(repair_turns=1, turns=turn_receipts(len(names) + 1),
                                 tools=receipts(names, [False, True, True]),
                                 final_source_sha256="final", observed_final_source_sha256="final")
        return values

    def test_aggregate_binds_the_repair_budget(self):
        ok = dict.fromkeys(AGENTS, 0), dict.fromkeys(AGENTS, True)
        values = self.repaired()
        self.assertEqual(aggregate(values, *ok, PID, RUN, repair_turns=1),
                         "FourAgentProtocolCompleteAwaitingIndependentReview")
        with self.assertRaises(Invalid):
            aggregate(values, *ok, PID, RUN)
        with self.assertRaises(Invalid):  # canonical agents in a repair run
            aggregate(results(), *ok, PID, RUN, repair_turns=1)
        values["agent-2"].pop("repair_turns")
        with self.assertRaises(Invalid):
            aggregate(values, *ok, PID, RUN, repair_turns=1)

    def test_agent_command_forwards_only_a_granted_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            approval = Path(tmp) / "agent-0.approval.json"
            approval.write_text("{}")
            args = client_args()
            args.repair_turns = 0
            canonical = agent_command(args, Path("/m"), "agent-0", Path("/r/a"), approval, Path("/r/b"), RUN, TOKEN)
            self.assertNotIn("--repair-turns", canonical)
            args.repair_turns = 1
            command = agent_command(args, Path("/m"), "agent-0", Path("/r/a"), approval, Path("/r/b"), RUN, TOKEN)
            self.assertEqual(command, canonical + ["--repair-turns", "1"])

    def test_tools_bound_follows_the_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(Invalid):
                Tools(Path(tmp) / "low", "/c", "/s", "/Library/Developer/SDKs/Mock.sdk", max_calls=5)
            tools = Tools(Path(tmp) / "t", "/c", "/s", "/Library/Developer/SDKs/Mock.sdk", max_calls=max_tools(1))
            try:
                self.assertEqual(tools.max_calls, 8)
            finally:
                tools.close()


class Cases(unittest.TestCase):
    def load(self, name):
        return q5_case.loads((CASES / name).read_bytes())

    def test_repair_cases_differ_from_their_bases_only_in_the_grant(self):
        for name, base in (("qwen36-repair1.json", "qwen36.json"), ("peer-qwen36-repair1.json", "peer-qwen36.json"),
                           ("splash-qwen36-repair1.json", "splash-qwen36.json")):
            case, original = self.load(name), self.load(base)
            self.assertEqual(q5_case.repair_turns(case), 1)
            self.assertEqual(q5_case.repair_turns(original), 0)
            self.assertEqual(case["client"]["argv"], original["client"]["argv"] + ["--repair-turns", "1"])
            self.assertEqual(case["client"]["approval"]["set"],
                             {**original["client"]["approval"]["set"], "repair_turns": 1})
            self.assertEqual(case["variables"], {**original["variables"], "MODEL_SHA256":
                             "815a22cab21649948352c0eea6e9be0d9d15606e2024d86e7bdb71fb9df08220"})
            self.assertEqual((case["serve"], case["readiness"], case["artifact"]),
                             (original["serve"], original["readiness"], original["artifact"]))
            self.assertNotEqual(case["workload"]["id"], original["workload"]["id"])
            derived = qc.derive_case(case, 4)
            self.assertEqual(derived["client"]["argv"][-4:], ["--repair-turns", "1", "--instance", "4"])
            self.assertEqual(derived["client"]["approval"]["set"]["repair_turns"], 1)

    def test_the_grant_is_qwen36_only_and_bound(self):
        for name in ("qwen36-repair1.json", "splash-qwen36-repair1.json"):
            with self.subTest(name):
                self.refuses_mutated_grants(self.load(name))

    def refuses_mutated_grants(self, case):
        mutations = {"other family": lambda c: c.update(family="qwen38"),
                     "argv without approval": lambda c: c["client"]["approval"]["set"].pop("repair_turns"),
                     "approval without argv": lambda c: c["client"]["argv"].__delitem__(slice(-2, None)),
                     "value 2": lambda c: c["client"]["argv"].__setitem__(-1, "2"),
                     "approval 2": lambda c: c["client"]["approval"]["set"].update(repair_turns=2),
                     "twice": lambda c: c["client"]["argv"].extend(["--repair-turns", "1"])}
        for label, mutate in mutations.items():
            bad = json.loads(json.dumps(case))
            mutate(bad)
            with self.subTest(label), self.assertRaises(q5_case.CaseError):
                q5_case.validate(bad)

    def test_the_client_admits_splash_for_qwen36_only(self):
        import coding_family
        import native_peer_splash as splash
        config = splash.config_text(splash.DEFAULT_KNOBS)
        qwen36, qwen38 = splash.controls("qwen36", config), splash.controls("qwen38", config)
        allowed = coding_family.repair_allowed
        self.assertTrue(allowed("qwen", "hf2q", None) and allowed("qwen", "peer", None))
        self.assertTrue(allowed("qwen", "splash", qwen36))
        self.assertFalse(allowed("qwen", "splash", qwen38))
        self.assertFalse(allowed("qwen", "splash", None))
        for kind in coding_family.NATIVE_KINDS:  # ds4, tensorfold, omlx
            self.assertFalse(allowed("qwen", kind, qwen36), kind)
        for family in ("deepseek4", "gemma4"):
            for kind in coding_family.ENGINE_KINDS:
                self.assertFalse(allowed(family, kind, qwen36), (family, kind))

    def test_a_splash_repair_campaign_needs_the_same_grant(self):
        mixed = qc.parse_args(["--subject-case", str(CASES / "qwen36-repair1.json"), "--reference-case",
                               str(CASES / "splash-qwen36.json"), "--reference-kind", "splash", "--dry-run"])
        with self.assertRaisesRegex(qc.CampaignError, "same repair turns"):
            qc.build_plan(mixed, {})
        same = qc.parse_args(["--subject-case", str(CASES / "qwen36-repair1.json"), "--reference-case",
                              str(CASES / "splash-qwen36-repair1.json"), "--reference-kind", "splash", "--dry-run"])
        plan, _ = qc.build_plan(same, {})
        self.assertEqual(plan["arms"]["subject"]["family"], "qwen36")

    def test_paired_campaign_refuses_arms_with_different_grants(self):
        mixed = qc.parse_args(["--subject-case", str(CASES / "qwen36-repair1.json"), "--reference-case",
                               str(CASES / "peer-qwen36.json"), "--reference-kind", "peer", "--dry-run"])
        with self.assertRaisesRegex(qc.CampaignError, "same repair turns"):
            qc.build_plan(mixed, {})
        same = qc.parse_args(["--subject-case", str(CASES / "qwen36-repair1.json"), "--reference-case",
                              str(CASES / "peer-qwen36-repair1.json"), "--reference-kind", "peer", "--dry-run"])
        plan, _ = qc.build_plan(same, {})
        self.assertEqual(plan["arms"]["subject"]["family"], "qwen36")


class RunAudit(unittest.TestCase):
    """Raw-evidence replay over real loopback HTTP: the actual exchange, Workflow, aggregate and Q2 audit."""

    def run_dir(self, engine_kind="hf2q", **options):
        import q2_fake_service as fake
        root = Path(tempfile.mkdtemp(prefix="q36-repair-audit-"))
        self.addCleanup(shutil.rmtree, root)
        client = root / "client"
        return client, fake.honest_run(client, 100.0, engine_kind, **options)

    def test_a_repaired_run_re_derives_its_pass(self):
        import q2_run_audit as audit
        for kind in ("hf2q", "splash"):
            with self.subTest(kind):
                client, summary = self.run_dir(kind, repair_turns=1, instance=3)
                self.assertEqual(summary["status"], "FourAgentProtocolCompleteAwaitingIndependentReview")
                self.assertEqual(summary["repair_turns"], 1)
                for agent in AGENTS:
                    result = summary["agents"][agent]
                    self.assertEqual([t["name"] for t in result["tools"]],
                                     ["read_file", "read_file", "write_file", "run_tests", "write_file", "run_tests",
                                      "read_file", "run_tests"])
                    self.assertEqual(len(result["turns"]), 9)
                report = audit.audit_run(client, summary, None, kind, True)
                self.assertEqual(report["expected_width"], 4)
                if kind == "splash":  # every repaired request bound to a Splash Done line and its render proof
                    self.assertEqual(report["native_trace"]["requests"], 36)

    def test_the_audit_refuses_a_repaired_run_relabelled_canonical(self):
        import q2_run_audit as audit
        for kind in ("hf2q", "splash"):
            with self.subTest(kind):
                client, summary = self.run_dir(kind, repair_turns=1)
                stripped = json.loads(json.dumps(summary))
                stripped.pop("repair_turns")
                for result in stripped["agents"].values():
                    result.pop("repair_turns")
                with self.assertRaises(Invalid):
                    audit.audit_run(client, stripped, None, kind, True)
                relabelled = json.loads(json.dumps(summary))
                relabelled.pop("repair_turns")
                with self.assertRaises(Invalid):
                    audit.audit_run(client, relabelled, None, kind, True)

    def test_canonical_runs_are_unchanged(self):
        import q2_run_audit as audit
        for kind in ("hf2q", "splash"):
            with self.subTest(kind):
                client, summary = self.run_dir(kind)
                self.assertNotIn("repair_turns", summary)
                self.assertTrue(all("repair_turns" not in r and len(r["turns"]) == 7
                                    for r in summary["agents"].values()))
                audit.audit_run(client, summary, None, kind, True)


if __name__ == "__main__":
    unittest.main()
