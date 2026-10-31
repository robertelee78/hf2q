"""ADR-063 Q2 adversarial controls: each false quality or speed win is rejected with its reason.

The model-free fake service (q2_fake_service.py) speaks OpenAI chat/SSE over real loopback HTTP.
The clients are the actual exchange/parser, the coordinator verdict is the actual aggregate, and
the campaign checks are q2_run_audit plus the actual Q3 and quality campaign drivers. Passing
these controls is harness evidence only: no inference, GPU, model quality or hardware speed.
Matrix and results: docs/research/adr063-q2-adversarial-controls.md.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import q2_fake_service as fake
import q2_run_audit as audit
import q3_metrics
import q3_report
import q3_stats
import quality_campaign as qc
from qwen38_n1_smoke import build_request, exchange
from qwen38_n1_verdict import Invalid
from test_q3_campaign import Harness
from test_quality_campaign import QWEN38, FakeRuns

PASSING_RECEIPT = {"verdict": {"verdict": "CalibrationPass"}, "cleanup": {"proven": True}, "cleanup_proven": True}
FOREIGN = " Independent coding agent agent-1, workload adr063-qwen-n4-parse-port-v1."


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def rewrite(client, summary, agent, turn, name, edit):
    """Change one archived file and re-bind its receipt hash/size, as a careful forger would."""
    path = Path(client) / agent / f"turn-{turn}" / name
    data = edit(path.read_bytes())
    path.write_bytes(data)
    receipt, kind = summary["agents"][agent]["turns"][turn], "response" if name == "response.bin" else "request"
    receipt[f"{kind}_sha256"], receipt[f"{kind}_bytes"] = sha256(data), len(data)


def unary_edit(change):
    def edit(raw):
        value = json.loads(raw)
        change(value)
        return json.dumps(value).encode()
    return edit


def request_edit(change):
    def edit(raw):
        value = json.loads(raw)
        change(value)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return edit


class WireControls(unittest.TestCase):
    """One agent's actual seven-turn exchange loop; the fault is injected by the server."""

    def drive(self, faults, engine_kind="hf2q"):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "agent-0"
        with fake.FakeService(faults=faults) as service:
            try:
                return fake.drive_agent(service.url, self.out, "agent-0", engine_kind=engine_kind)
            finally:
                self.served = list(service.requests)

    def rejects(self, fault, turn, reason, engine_kind="hf2q"):
        with self.assertRaisesRegex(ValueError, reason):
            self.drive({("agent-0", turn): fault}, engine_kind)
        # Fail closed with evidence: the bad response was received and its raw bytes retained.
        self.assertEqual(self.served[-1]["turn"], turn)
        self.assertTrue((self.out / f"turn-{turn}" / "response.bin").stat().st_size > 0)

    def test_positive_control_honest_service_completes(self):
        for engine in ("hf2q", "peer"):
            with self.subTest(engine=engine):
                summary = self.drive({}, engine)
                self.assertEqual(summary["status"], "ProtocolCompleteAwaitingIndependentPatchReview")
                self.assertEqual(len(self.served), 7)

    def test_role_only_sse_is_not_progress(self):
        self.rejects("role_only", 1, "no semantic SSE progress")

    def test_reasoning_without_semantic_output_is_not_progress(self):
        self.rejects("reasoning_only", 3, "no semantic SSE progress")

    def test_missing_final_summary(self):
        self.rejects("empty_final", 6, "final summary missing")

    def test_wrong_tool_arguments(self):
        self.rejects("wrong_args", 0, "tool arguments/workflow drift")
        self.rejects("wrong_args", 3, "tool arguments/workflow drift")

    def test_reused_tool_call_id_is_stale_lineage(self):
        self.rejects("reused_call_id", 2, "missing/duplicate/oversized tool id")

    def test_silent_truncation_mislabelled_as_tool_call_or_stop(self):
        for turn in (1, 2, 6):
            with self.subTest(turn=turn):
                self.rejects("truncated_stop", turn, "silent truncation")
        self.rejects("truncated_stop", 5, "silent truncation", engine_kind="peer")

    def test_truncated_tool_arguments(self):
        self.rejects("truncated_args", 4, "incomplete/truncated tool arguments JSON")

    def test_length_finish_is_not_a_tool_turn(self):
        self.rejects("length_finish", 5, "exactly one live call required")

    def test_output_beyond_the_allowance(self):
        self.rejects("inflated_budget", 0, "excess completion tokens")

    def test_usage_total_contradiction(self):
        self.rejects("usage_total_lie", 3, "usage total contradiction")

    def test_cache_accounting_lies(self):
        self.rejects("cached_over_prompt", 2, "excess cached tokens")
        self.rejects("cached_over_prompt", 1, "excess peer cached tokens", engine_kind="peer")
        self.rejects("cold_cached", 0, "prefix reuse not proven")
        self.rejects("no_reuse", 4, "prefix reuse not proven")

    def test_swapped_model_identity(self):
        self.rejects("model_swap", 0, "response identity")
        self.rejects("model_swap", 1, "SSE identity")

    def test_altered_output_allowance_never_reaches_the_wire(self):
        fixture = json.loads((fake.HERE / "fixtures" / "canonical-coding-requests.json").read_text())[0]["request"]
        for tool in fixture["tools"]:
            tool["function"]["strict"] = False
        request = build_request(fake.MODEL, fixture["messages"], fixture["tools"], 0)
        with tempfile.TemporaryDirectory() as tmp, fake.FakeService() as service:
            for allowance in (4096, 8193):
                request["max_completion_tokens"] = allowance
                out = Path(tmp) / str(allowance)
                out.mkdir()
                with self.subTest(allowance=allowance), self.assertRaisesRegex(Invalid, "output allowance drift"):
                    exchange(service.url, request, out, 0, set(), deadline=5)
            self.assertEqual(service.requests, [])


class AuditControls(unittest.TestCase):
    """Settled-run evidence: honest runs pass the audit; every forged variant is rejected."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.count = 0

    def run_dir(self, plan=None, engine="hf2q", serialized=False):
        self.count += 1
        client = Path(self.tmp.name) / f"run{self.count}" / "client"
        return client, fake.honest_run(client, 100.0, engine, plan, serialized=serialized)

    def reason(self, client, summary, plan=None, engine="hf2q", receipt=PASSING_RECEIPT, width=True):
        report, reason = audit.check_run(receipt, client, summary, plan, engine, width)
        self.assertIsNone(report)
        return reason

    def test_positive_controls_every_population(self):
        for plan, width in ((None, 4), ("1", 1), ("3+1", 3), ("2+2", 2), ("1+3", 3)):
            with self.subTest(plan=plan):
                client, summary = self.run_dir(plan)
                report, reason = audit.check_run(PASSING_RECEIPT, client, summary, plan, "hf2q", True)
                self.assertIsNone(reason)
                self.assertEqual((report["progress"]["width"], report["expected_width"]), (width, width))
                self.assertEqual(report["response_ids"], 7 * len(summary["agents"]))
        client, summary = self.run_dir(engine="peer")
        self.assertIsNone(audit.check_run(PASSING_RECEIPT, client, summary, None, "peer", True)[1])

    def test_serialized_service_is_not_four_wide(self):
        client, summary = self.run_dir(serialized=True)
        self.assertEqual(summary["status"], audit.SUCCESS)  # the coordinator alone would have accepted it
        self.assertRegex(self.reason(client, summary), r"serialized service: observed co-progress width 1 < 4")
        # Task quality is not a concurrency claim: the quality campaign audits without the width gate.
        self.assertIsNone(audit.check_run(PASSING_RECEIPT, client, summary, None, "hf2q", False)[1])

    def test_raw_hash_mismatch(self):
        client, summary = self.run_dir()
        path = client / "agent-2" / "turn-3" / "response.bin"
        path.write_bytes(path.read_bytes().replace(b"Step", b"Stop", 1))
        self.assertRegex(self.reason(client, summary), "archived response hash differs")

    def test_receipt_cache_accounting_lie(self):
        client, summary = self.run_dir()
        usage = summary["agents"]["agent-1"]["turns"][2]["usage"]
        usage["prompt_tokens_details"]["cached_tokens"] = usage["prompt_tokens"] + 1
        self.assertRegex(self.reason(client, summary), "receipt usage differs from the raw response")

    def test_archived_cache_lie_silent_truncation_and_allowance(self):
        def cached(value):
            value["usage"]["prompt_tokens_details"]["cached_tokens"] = value["usage"]["prompt_tokens"] + 1

        def truncated(value):
            value["usage"].update(completion_tokens=8192, total_tokens=value["usage"]["prompt_tokens"] + 8192)
        for name, edit, reason in (("response.bin", unary_edit(cached), "excess cached tokens"),
                                   ("response.bin", unary_edit(truncated), "silent truncation"),
                                   ("request.json", request_edit(lambda r: r.update(max_completion_tokens=4096)),
                                    "output allowance drift")):
            with self.subTest(reason=reason):
                client, summary = self.run_dir()
                rewrite(client, summary, "agent-0", 4, name, edit)
                self.assertRegex(self.reason(client, summary), reason)

    def test_metric_units_and_timer_direction(self):
        client, summary = self.run_dir()
        summary["own_output_four_task_makespan_seconds"] *= 1000  # milliseconds labelled seconds
        self.assertRegex(self.reason(client, summary), "makespan seconds disagree with its clocks")
        client, summary = self.run_dir()
        summary["agents"]["agent-0"]["turns"][1]["semantic_timing_v3"]["semantic_ttft_seconds"] *= 1000
        self.assertRegex(self.reason(client, summary), r"timer direction/unit corruption")
        client, summary = self.run_dir()
        turn = summary["agents"]["agent-3"]["turns"][3]
        turn["send_monotonic_ns"], turn["validated_terminal_monotonic_ns"] = \
            turn["validated_terminal_monotonic_ns"], turn["send_monotonic_ns"]
        self.assertRegex(self.reason(client, summary), "monotonic send/validated terminal required")

    def test_injected_speedups_in_receipts(self):
        client, summary = self.run_dir()
        agent = summary["agents"]["agent-2"]
        agent["task_completed_monotonic_ns"] = agent["turns"][-1]["validated_terminal_monotonic_ns"] - 1
        self.assertRegex(self.reason(client, summary), "task completion before its last validated response")
        client, summary = self.run_dir()
        summary["makespan_finished_ns"] -= 10 ** 9
        self.assertRegex(self.reason(client, summary), "makespan end is not the last task completion")
        client, summary = self.run_dir()
        turn = summary["agents"]["agent-1"]["turns"][5]
        turn["progress_event_monotonic_ns"][0] = turn["send_monotonic_ns"] - 1
        self.assertRegex(self.reason(client, summary), "progress events outside their request")

    def test_result_scoring_contradictions(self):
        client, summary = self.run_dir()
        summary["agents"]["agent-1"]["tools"][-1]["passed"] = False
        self.assertRegex(self.reason(client, summary), "final tests/source")
        client, summary = self.run_dir()
        summary["agents"]["agent-3"]["status"] = "Failed"
        self.assertRegex(self.reason(client, summary), "agent incomplete")

    def test_stale_lineage(self):
        client, summary = self.run_dir()
        reused = json.loads((client / "agent-0" / "turn-2" / "response.bin").read_text())["id"]
        rewrite(client, summary, "agent-0", 4, "response.bin", unary_edit(lambda v: v.update(id=reused)))
        self.assertRegex(self.reason(client, summary), "stale lineage: response id of agent-0 turn 4 repeats agent-0 turn 2")
        client, summary = self.run_dir()
        stale = (client / "agent-1" / "turn-3" / "request.json").read_bytes()
        rewrite(client, summary, "agent-0", 3, "request.json", lambda _raw: stale)
        self.assertRegex(self.reason(client, summary), "stale lineage: request history does not extend")
        client, summary = self.run_dir()
        summary["agents"]["agent-2"]["initial_messages_sha256"] = "0" * 64
        self.assertRegex(self.reason(client, summary), "stale lineage: first request")

    def test_cross_slot_leakage(self):
        client, summary = self.run_dir()
        other = json.loads((client / "agent-0" / "turn-6" / "response.bin").read_text())["id"]
        rewrite(client, summary, "agent-1", 6, "response.bin", unary_edit(lambda v: v.update(id=other)))
        self.assertRegex(self.reason(client, summary), "stale lineage: response id of agent-1 turn 6 repeats agent-0 turn 6")
        client, summary = self.run_dir()

        def leak(value):
            value["choices"][0]["message"]["content"] += FOREIGN
        rewrite(client, summary, "agent-2", 6, "response.bin", unary_edit(leak))
        self.assertRegex(self.reason(client, summary), "cross-slot leakage: agent-2 output carries another agent's identity")

    def test_cleanup_and_receipt_contradictions(self):
        client, summary = self.run_dir()
        summary["client_cleanup"]["agent-3"]["final_observation"]["live_owned_pids"] = [999]
        self.assertRegex(self.reason(client, summary), "actual owned cleanup unknown/live")
        client, summary = self.run_dir()
        summary["agents"]["agent-0"]["tools"][2]["cleanup"] = False
        self.assertRegex(self.reason(client, summary), "cleanup unknown")
        client, summary = self.run_dir()
        receipt = dict(PASSING_RECEIPT, cleanup_proven=False)
        self.assertRegex(self.reason(client, summary, receipt=receipt), "passing verdict without proven cleanup")

    def test_binary_and_model_identity(self):
        client, summary = self.run_dir(engine="peer")
        self.assertRegex(self.reason(client, summary, engine="hf2q"), "client summary engine differs from the arm")
        client, summary = self.run_dir()
        rewrite(client, summary, "agent-1", 4, "request.json", request_edit(lambda r: r.update(model="other-model")))
        rewrite(client, summary, "agent-1", 4, "response.bin", unary_edit(lambda v: v.update(model="other-model")))
        self.assertRegex(self.reason(client, summary), "model identity changed between turns")

    def test_population_and_cohort_membership(self):
        client, summary = self.run_dir()
        self.assertRegex(self.reason(client, summary, plan="1"), "agent set differs from the phase plan")
        client, summary = self.run_dir("1")
        self.assertRegex(self.reason(client, summary), "agent set differs from the phase plan")
        client, summary = self.run_dir("3+1")
        self.assertRegex(self.reason(client, summary), "aggregate phase cohort")
        late = max(summary["agents"][a]["task_completed_monotonic_ns"] for a in ("agent-0", "agent-1", "agent-2")) + 1
        summary["phase"]["releases"][1]["release_monotonic_ns"] = late
        self.assertRegex(self.reason(client, summary, plan="3+1"), "population never mixed")
        client, summary = self.run_dir("2+2")
        trigger = summary["phase"]["releases"][1]["trigger_first_decode_monotonic_ns"]
        trigger["agent-1"] = summary["agents"]["agent-1"]["turns"][1]["send_monotonic_ns"] - 1
        self.assertRegex(self.reason(client, summary, plan="2+2"), "first-decode trigger outside")
        client, summary = self.run_dir("1+3")
        release = summary["phase"]["releases"][1]
        release["release_monotonic_ns"] = max(release["trigger_first_decode_monotonic_ns"].values()) - 1
        self.assertRegex(self.reason(client, summary, plan="1+3"), "cohort released before its trigger")


class LiveConcurrency(unittest.TestCase):
    """Actual client processes against one fake service: concurrent versus one-request-at-a-time."""
    TIMING = fake.Timing(first=0.01, gap=0.01, unary=0.01)

    def live(self, serialized, plan=None, timing=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        client = Path(tmp.name) / "client"
        with fake.FakeService(timing or self.TIMING, serialized=serialized) as service:
            summary = fake.run_live(service, client, plan)
        self.assertEqual(summary["status"], audit.SUCCESS)  # the coordinator alone would accept both
        self.assertTrue(all(c["verified"] and c["reaped"] for c in summary["client_cleanup"].values()))
        return client, summary

    def test_concurrent_four_wide_service_is_accepted(self):
        client, summary = self.live(False)
        report = audit.audit_run(client, summary, None, "hf2q", True)
        self.assertEqual((report["progress"]["width"], report["http_overlap_max"]), (4, 4))

    def test_serialized_four_client_service_is_rejected_despite_overlapping_http(self):
        client, summary = self.live(True)
        turns = {a: v["turns"] for a, v in summary["agents"].items()}
        self.assertGreaterEqual(audit.http_overlap(turns), 3)  # requests were queued concurrently
        with self.assertRaisesRegex(ValueError, r"serialized service: observed co-progress width 1 < 4"):
            audit.audit_run(client, summary, None, "hf2q", True)

    def test_mixed_population_counts_cohorts_not_labels(self):
        client, summary = self.live(False, "3+1")
        report = audit.audit_run(client, summary, "3+1", "hf2q", True)
        self.assertGreaterEqual(report["progress"]["width"], 3)
        self.assertEqual(len(report["cohort_release_ns"]), 2)
        client, summary = self.live(True, "3+1")
        with self.assertRaisesRegex(ValueError, r"width 1 < 3"):
            audit.audit_run(client, summary, "3+1", "hf2q", True)

    def test_one_slot_control(self):
        client, summary = self.live(False, "1")
        report = audit.audit_run(client, summary, "1", "hf2q", True)
        self.assertEqual((report["progress"]["width"], report["expected_width"], report["http_overlap_max"]), (1, 1, 1))

    def test_known_speedup_has_the_right_direction(self):
        slow_timing = fake.Timing(first=0.04, gap=0.02, unary=0.04)
        slow = self.live(False, timing=slow_timing)[1]["own_output_four_task_makespan_seconds"]
        fast = self.live(False, timing=slow_timing.scaled(0.25))[1]["own_output_four_task_makespan_seconds"]
        # Per agent: 4 unary waits + 3 streams of (first + 5 gaps) of known server delay.
        self.assertGreaterEqual(slow, 4 * 0.04 + 3 * (0.04 + 5 * 0.02))
        self.assertLess(fast, slow)
        self.assertLess(q3_stats.paired_effect([(fast, slow)])["log_ratios"][0], 0)  # experiment faster: ratio < 1


class InjectedDelay(unittest.TestCase):
    """Known server delays appear, in seconds, as lower bounds of the client's v3 timing."""

    def receipts(self, timing):
        with tempfile.TemporaryDirectory() as tmp, fake.FakeService(timing) as service:
            return fake.drive_agent(service.url, Path(tmp) / "agent-0", "agent-0")

    def test_timer_direction_and_units(self):
        slow = self.receipts(fake.Timing(first=0.15, gap=0.01, unary=0.06))
        fast = self.receipts(fake.Timing(first=0.01, gap=0.01, unary=0.0))
        first_semantic = 0.15 + fake.REASONING_FRAMES * 0.01  # first reasoning delay + gaps up to the tool name
        for turn, other in zip(slow["turns"], fast["turns"]):
            v3 = turn["semantic_timing_v3"]
            if turn["stream"]:
                self.assertGreaterEqual(v3["semantic_ttft_seconds"], first_semantic)
                self.assertLess(v3["semantic_ttft_seconds"], first_semantic + 0.5)
                self.assertGreater(v3["semantic_ttft_seconds"], other["semantic_timing_v3"]["semantic_ttft_seconds"])
                self.assertEqual(v3["semantic_ttft_seconds"],
                                 (turn["semantic_event_monotonic_ns"][0] - turn["send_monotonic_ns"]) / 1e9)
                self.assertTrue(turn["send_monotonic_ns"] <= turn["progress_event_monotonic_ns"][0]
                                <= turn["semantic_event_monotonic_ns"][0] <= turn["validated_terminal_monotonic_ns"])
            else:
                self.assertGreaterEqual(v3["unary_completion_seconds"], 0.06)
        self.assertGreaterEqual(q3_metrics.slot_metrics(slow)["ttft_p95_s"], first_semantic)


class CampaignControls(Harness):
    """The actual Q3 and quality drivers refuse false wins written by the fake runner."""

    def test_q3_serialized_stock_arm_cannot_hand_the_experiment_a_win(self):
        def decide(arm, workload, call):
            return {"makespan": 80 if arm == "experiment" else 100, "serialized": arm == "stock"}
        driver, _ = self.driver(decide)
        manifest = driver.run()
        self.assertEqual(manifest["status"], "Invalid")
        self.assertIn("serialized service: observed co-progress width 1 < 4", manifest["halted"]["reason"])
        self.assertNotEqual(q3_report.compute_verdict(manifest)["verdict"], "Win")

    def forging(self, forge):
        def runner(argv, env, stdout, stderr):
            code = runner.inner(argv, env, stdout, stderr)
            if "experiment" in Path(argv[3]).stem:
                client = Path(argv[5]) / "client"
                summary = json.loads((client / "summary.json").read_text())
                forge(summary)
                (client / "summary.json").write_text(json.dumps(summary))
            return code
        driver, inner = self.driver(lambda arm, w, call: {"makespan": 100})
        runner.inner = inner
        driver.runner = runner
        return driver.run()

    def test_q3_forged_makespan_units_and_scoring_are_invalid(self):
        def milliseconds(summary):
            summary["own_output_four_task_makespan_seconds"] /= 1000
        for forge, reason in ((milliseconds, "makespan seconds disagree"),
                              (lambda s: s["agents"]["agent-2"]["tools"][-1].update(passed=False), "final tests/source")):
            with self.subTest(reason=reason):
                self.tearDown()
                self.setUp()
                manifest = self.forging(forge)
                self.assertEqual(manifest["status"], "Invalid")
                self.assertIn(reason, manifest["halted"]["reason"])

    def test_quality_forged_task_success_is_invalid(self):
        root = Path(self.tmp.name)
        args = qc.parse_args(["--subject-case", str(QWEN38), "--reference-case", str(QWEN38), "--reference-kind",
                              "stock-hf2q", "--instances", "24", "--negative-control-receipt", str(self.nc),
                              "--output", str(root / "quality")])
        plan, cases = qc.build_plan(args, {})

        class Forged(FakeRuns):
            def __call__(self, argv, env, stdout, stderr):
                code = super().__call__(argv, env, stdout, stderr)
                client = Path(argv[argv.index("--output") + 1]) / "client"
                summary = json.loads((client / "summary.json").read_text())
                summary["agents"]["agent-1"]["tools"][-1]["passed"] = False  # status still claims success
                (client / "summary.json").write_text(json.dumps(summary))
                return code
        manifest = qc.run(plan, cases, root / "quality", {}, Forged(lambda i, arm: "pass"))
        self.assertEqual((manifest["status"], manifest["halted_at"]), ("Halted", 0))
        attempt = manifest["runs"][0]["attempts"][0]
        self.assertEqual(attempt["outcome"], "Invalid")
        self.assertIn("final tests/source", attempt["reason"])
        self.assertNotIn("verdict", manifest)


if __name__ == "__main__":
    unittest.main()
