"""Model-free controls of the actual Q2 raw-cohort certifier (q2_certify) and manifest assembler.

The cohorts are synthetic: honest fake-service agents (the actual exchange client and parser),
stub compile/test receipts and synthetic native phase traces from test_f2_native_trace. The N4
cohort's four agents are re-timed to one shared release with fake.retime and bound by an actual
coordinator barrier release. Nothing here is hardware, model or QE evidence; the controls show
what certify_cohort accepts and the exact reasons it refuses.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import f2_trace
import q2_certify as certify
import q2_fake_service as fake
import q2_manifest
import qwen_n4_live_smoke as coordinator
import qwen38_n1_tools as tool_source
from qwen38_n1_verdict import Invalid
from test_f2_native_trace import fixture as trace_fixture, four_lanes, with_busy_owners

HERE = Path(__file__).resolve().parent
SOURCES = ("q2_certify.py", "qwen38_n1_verdict.py", "qwen38_n1_smoke.py", "qwen38_n1_tools.py",
           "qwen_n4_live_smoke.py", "coding_family.py", "f2_trace.py", "f2_semantic.py")
COHORT = "q2-certify-control-n1"
HARNESS = ("running 2 tests\ntest a ... ok\ntest b ... ok\n\ntest result: ok. 2 passed; 0 failed; 0 ignored; "
           "0 measured; 0 filtered out; finished in 0.00s\n\n")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def write_jsonl(path, records):
    path.write_bytes(b"".join(json.dumps(r, separators=(",", ":")).encode() + b"\n" for r in records))
    return path


def ref(path):
    return {"path": str(path), "sha256": certify.digest(path)}


class Cohort:
    """One N1 or N4 hf2q cohort laid out as q2_manifest.agent_refs reads it, plus its frozen plan inputs."""

    def __init__(self, root, trace=None, cardinality=1):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.serial, self.count = 0, cardinality
        template_root, results = fake.template()
        self.client = self.root / "client"
        names = [f"agent-{i}" for i in range(cardinality)]
        dirs = {"agent-0": self.client} if cardinality == 1 else {name: self.client / name for name in names}
        self.initial = {}
        for name, directory in dirs.items():
            shutil.copytree(template_root / name, directory)
            self.agent_files(directory, results[name]["tools"])
            self.initial[name] = self.root / "initial" / f"{name}.json"
            self.initial[name].parent.mkdir(exist_ok=True)
            shutil.copyfile(directory / "turn-0" / "request.json", self.initial[name])
        if cardinality == 4:
            self.barrier(dirs, copy.deepcopy({name: results[name] for name in names}))
        self.records, self.identity, self.scenario = trace_fixture() if trace is None else trace
        self.trace = write_jsonl(self.root / "native-phase-trace.jsonl", self.records)
        self.server_receipt = write_json(self.root / "receipt.json", {
            "protocol_complete": True, "final_verified": True, "cleanup_proved": True, "failure": None,
            "listener_cleanup_error": None, "cleanup": [{"reaped": True, "group_empty": True, "errors": []}]})
        self.readiness = write_json(self.root / "readiness.json", {
            "health": {"model": fake.MODEL, "context_length": 262144},
            "runtime": {"measurement_snapshot": {"process_pid": fake.SERVER_PID}, "serve": {"max_slots": cardinality}},
            "readyz": {"ready": True}, "pool_binding": None})
        self.spec = {"schema": "hf2q.q2.manifest-input.v1", "expected_plan": ref(self.file("plan", self.plan())),
                     "client_root": str(self.client), "server_receipt": ref(self.server_receipt),
                     "readiness": ref(self.readiness), "trace": ref(self.trace)}
        self.manifest = self.reassemble()

    def reassemble(self):
        """A new cohort manifest over the files as they are now."""
        return self.file("cohort-manifest", q2_manifest.assemble(self.spec))

    def agent_files(self, directory, tools):
        fixture = directory / "tools" / "fixture"
        for name, text in tool_source.IMMUTABLE.items():
            (fixture / name).parent.mkdir(parents=True, exist_ok=True)
            (fixture / name).write_text(text)
        for number, tool in enumerate(tools, 1):
            tool = dict(tool)
            if tool["name"] == "run_tests":
                tool["artifact_sha256"] = self.test_group(directory, number, tool["target"], fixture / "Cargo.toml")
            write_json(directory / "tools" / f"tool-{number:02d}" / "result.json", tool)

    def barrier(self, dirs, results):
        """One shared release: re-timed raw receipts and agent summaries, ready records, the actual
        coordinator release (bound by its ready hashes) and the coordinator summary."""
        releases, triggers = fake.retime(results, None, 100.0)
        self.release_ns = releases[0]
        for name, directory in dirs.items():
            for turn, receipt in enumerate(results[name]["turns"]):
                write_json(directory / f"turn-{turn}" / "receipt.json", receipt)
            write_json(directory / "summary.json", results[name])
        ready = {name: {"schema": "hf2q.n4-agent-ready.v1", "agent_id": name, "run_id": fake.RUN,
                        "server_pid": fake.SERVER_PID, "barrier_token": fake.TOKEN,
                        "identity_sha256": hashlib.sha256(coordinator.identity(name, fake.RUN).encode()).hexdigest(),
                        "initial_messages_sha256": results[name]["initial_messages_sha256"], "output": str(directory)}
                 for name, directory in dirs.items()}
        barrier = self.client / "barrier"
        hashes = {name: certify.digest(write_json(barrier / f"{name}.ready.json", record)) for name, record in ready.items()}
        release = dict(coordinator.barrier_release(ready, fake.RUN, fake.SERVER_PID, fake.TOKEN, hashes),
                       release_monotonic_ns=releases[0], f2_clock_id="CLOCK_MONOTONIC")
        release_path = write_json(barrier / "release.json", release)
        summary = fake.assemble(self.client, results, releases, triggers, None)
        summary["barrier"].update(ready=ready, release_sha256=certify.digest(release_path))
        write_json(self.client / "summary.json", summary)

    def test_group(self, directory, number, target, manifest):
        """Compile and test receipts in the actual Tools.tests layout; the binaries are never executed."""
        tool = directory / "tools" / f"tool-{number:02d}"
        names = ["ports"] if target == "ports" else ["qwen38_n1_fixture", "ports"]
        binaries = []
        for name in names:
            binary = tool / "target" / "debug" / "deps" / f"{name}-0123456789abcdef"
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_bytes(f"NOT EXECUTED {name} {number}".encode())
            binaries.append(binary)
        scope = ["--test", "ports"] if target == "ports" else ["--tests"]
        stdout = "".join(json.dumps({"reason": "compiler-artifact", "executable": str(binary), "profile": {"test": True},
                                     "target": {"name": name}}) + "\n" for name, binary in zip(names, binaries))
        stdout += json.dumps({"reason": "build-finished", "success": True}) + "\n"
        self.process(tool / "logs" / "compile", ["/toolchain/bin/cargo", "test", "--offline", "--locked", "--no-run",
                                                 "--message-format=json", "--manifest-path", str(manifest), *scope], stdout)
        for index, binary in enumerate(binaries):
            self.process(tool / "logs" / f"test-{index}", [str(binary), "--test-threads=1"], HARNESS)
        return [certify.digest(binary) for binary in binaries]

    @staticmethod
    def process(directory, argv, stdout):
        write_json(directory / "receipt.json", {"argv": argv, "exit_code": 0, "cleanup": True, "error": None,
                                                "cleanup_error": None, "cleanup_receipt": fake.CLEAN})
        (directory / "stdout.bin").write_text(stdout)
        (directory / "stderr.bin").write_bytes(b"")

    def file(self, stem, value):
        self.serial += 1
        return write_json(self.root / f"{stem}-{self.serial}.json", value)

    def plan(self, qe=None, **scenario):
        return {"schema": "hf2q.q2.expected-plan.v1", "cohort_id": COHORT, "identity": self.identity,
                "source_files": {name: certify.digest(HERE / name) for name in SOURCES}, "family": "qwen",
                "engine_kind": "hf2q", "model": fake.MODEL, "server_pid": fake.SERVER_PID,
                "run_id": self.scenario["run_id"], "required_cardinality": self.count,
                "initial_requests": {name: ref(path) for name, path in self.initial.items()},
                "barrier_token": fake.TOKEN if self.count == 4 else None,
                "trace_scenario": {**self.scenario, "composition": [0, 1] if self.count == 1 else [4, 0],
                                   "arrival_ns": 0, "population_policy": "ready_now_v1", **scenario},
                "qe_attestation": qe}

    def attestation(self, **changes):
        return ref(self.file("qe-verdict", {
            "schema": "hf2q.q2.qe-verdict.v1", "reviewer": "independent-qe-control", "independent": True,
            "verdict": "PASS", "manifest_sha256": certify.digest(self.manifest),
            "auditor_sha256": certify.digest(certify.__file__), "cohort_id": COHORT, "identity": self.identity,
            **changes}))

    def certify(self, plan, manifest=None):
        return certify.certify_cohort(str(manifest or self.manifest), plan)


class Certify(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cohort = Cohort(Path(cls.tmp.name) / "honest")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def refuses(self, plan, reason, error=Invalid):
        with self.assertRaisesRegex(error, "^" + re.escape(reason) + "$"):
            self.cohort.certify(plan)

    def test_positive_cohort_awaits_qe_then_a_bound_attestation_qualifies(self):
        report = self.cohort.certify(self.cohort.plan())
        self.assertEqual(report["status"], "PendingIndependentQE")
        self.assertIs(report["raw_protocol_verified"], True)
        self.assertIs(report["independently_qe_qualified"], False)
        self.assertIs(report["campaign_primary_eligible"], False)
        self.assertEqual(report["manifest_sha256"], certify.digest(self.cohort.manifest))
        self.assertEqual((report["trace"]["qualified"], report["trace"]["events"]), (True, len(self.cohort.records)))
        self.assertEqual((report["phase_window"]["status"], report["phase_window"]["population_policy"]),
                         ("Qualified", "ready_now_v1"))
        self.assertLess(report["makespan_started_ns"], report["makespan_finished_ns"])
        self.assertEqual(set(report["per_slot_metrics"]), {"agent-0"})
        qe = self.cohort.attestation()
        report = self.cohort.certify(self.cohort.plan(qe=qe))
        self.assertEqual((report["status"], report["independently_qe_qualified"], report["qe"]), ("Qualified", True, qe))
        self.assertIs(report["campaign_primary_eligible"], False)
        self.assertIn(qe, report["verified_refs"])
        self.assertIn(self.cohort.spec["trace"], report["verified_refs"])

    def test_plan_refusals_name_their_reason(self):
        cases = ((lambda p: p.pop("barrier_token"), "externally frozen expected-plan keys"),
                 (lambda p: p.update(extra=None), "externally frozen expected-plan keys"),
                 (lambda p: p.update(required_cardinality=2), "N1/N4 separate plan"),
                 (lambda p: p["identity"].pop("native_version"), "complete model/binary/native/template identity"),
                 (lambda p: p.update(engine_kind="ds4"), "closed engine family"),
                 (lambda p: p["source_files"].pop("f2_trace.py"), "all actual audit/production sources bound"),
                 (lambda p: p["source_files"].update({"f2_trace.py": "0" * 64}), "raw evidence hash changed"),
                 (lambda p: p["trace_scenario"].update(run_id="other_run_identity"),
                  "externally frozen actual trace population scenario"),
                 (lambda p: p["trace_scenario"].pop("population_policy"), "externally frozen trace population policy"),
                 (lambda p: p["trace_scenario"].update(population_policy="silent_reinterpretation"),
                  "externally frozen trace population policy"))
        for mutate, reason in cases:
            with self.subTest(reason=reason):
                plan = self.cohort.plan()
                plan["identity"] = dict(plan["identity"])
                mutate(plan)
                self.refuses(plan, reason)

    def test_peer_cohort_is_refused_explicitly_even_with_an_attestation(self):
        plan = self.cohort.plan(qe=self.cohort.attestation())
        plan["engine_kind"] = "peer"
        self.refuses(plan, "peer cohort has no hf2q server phase trace: never Q2-certifiable")

    def test_unqualified_window_and_trace_drift_are_refused(self):
        self.refuses(self.cohort.plan(composition=[1, 0]), "actual admitted-slot population/progress not qualified")
        self.refuses(self.cohort.plan(arrival_ns=10), "actual admitted-slot population/progress not qualified")
        self.refuses(self.cohort.plan(config_sha256="e" * 64), "trace configuration drift", ValueError)

    def test_attestation_must_bind_the_exact_manifest_and_auditor(self):
        for change in ({"manifest_sha256": "0" * 64}, {"auditor_sha256": "0" * 64}, {"independent": False},
                       {"verdict": "FAIL"}, {"cohort_id": "other-cohort"}):
            with self.subTest(change=change):
                self.refuses(self.cohort.plan(qe=self.cohort.attestation(**change)),
                             "independent QE verdict must bind exact raw cohort/source")

    def test_busy_owner_trace_needs_the_frozen_forward_demand_v2_policy(self):
        records, identity, scenario = trace_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            busy = Cohort(Path(tmp), (with_busy_owners(records, 10), identity, scenario))
            with self.assertRaisesRegex(Invalid, "^actual admitted-slot population/progress not qualified$"):
                busy.certify(busy.plan())  # ready_now_v1
            report = busy.certify(busy.plan(population_policy="forward_demand_v2"))
        self.assertEqual(report["status"], "PendingIndependentQE")
        window = report["phase_window"]
        self.assertEqual((window["status"], window["population_policy"]), ("Qualified", "forward_demand_v2"))
        self.assertGreater(window["target_fraction"], 0.99)
        self.assertLess(window["ready_now_fraction"], 0.001)

    def main(self, plan, output, plan_sha256=None):
        plan_path = self.cohort.file("main-plan", plan)
        argv = ["q2_certify.py", "--manifest", str(self.cohort.manifest), "--expected-plan", str(plan_path),
                "--expected-plan-sha256", plan_sha256 or certify.digest(plan_path), "--output", str(output)]
        with mock.patch.object(sys, "argv", argv):
            return certify.main()

    def test_main_writes_rejected_pending_and_qualified_certificates(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp).resolve()
            self.assertEqual(self.main(self.cohort.plan(composition=[1, 0]), out / "rejected.json"), 2)
            rejected = json.loads((out / "rejected.json").read_text())
            self.assertEqual((rejected["status"], rejected["failure"], rejected["manifest_sha256"]),
                             ("Rejected", "actual admitted-slot population/progress not qualified",
                              certify.digest(self.cohort.manifest)))
            self.assertIs(rejected["independently_qe_qualified"], False)
            self.assertEqual(self.main(self.cohort.plan(), out / "pending.json"), 2)
            self.assertEqual(json.loads((out / "pending.json").read_text())["status"], "PendingIndependentQE")
            self.assertEqual(self.main(self.cohort.plan(qe=self.cohort.attestation()), out / "qualified.json"), 0)
            self.assertEqual(json.loads((out / "qualified.json").read_text())["status"], "Qualified")
            with self.assertRaises(FileExistsError):  # create-new: a certificate is never overwritten
                self.main(self.cohort.plan(), out / "pending.json")
            with self.assertRaisesRegex(Invalid, "^frozen plan identity$"):
                self.main(self.cohort.plan(), out / "unbound.json", plan_sha256="0" * 64)
            self.assertFalse((out / "unbound.json").exists())

    def test_main_refuses_a_certificate_inside_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()
            (repo / ".git").mkdir()
            (repo / "evidence").mkdir()
            for output in (repo / "certificate.json", repo / "evidence" / "certificate.json"):
                with self.subTest(output=output), \
                        self.assertRaisesRegex(Invalid, "^canonical private certificate outside Git$"):
                    self.main(self.cohort.plan(), output)
                self.assertFalse(output.exists())


def n4_trace():
    """Four admitted prefill owners, transaction-busy after their submits: a (4,0) window only forward_demand_v2 credits."""
    records, identity, scenario = four_lanes((4, 0))
    with_busy_owners(records, 330)
    records[0]["payload"]["run_id"] = scenario["run_id"] = fake.RUN
    return records, identity, scenario


class CertifyN4(unittest.TestCase):
    """The four-agent branch: coordinator exits, release identity, ready join, barrier release, aggregate, clocks."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cohort = Cohort(Path(cls.tmp.name) / "n4", n4_trace(), cardinality=4)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def plan(self, **changes):
        return self.cohort.plan(population_policy="forward_demand_v2", **changes)

    def test_four_agents_released_together_await_qe_then_qualify(self):
        report = self.cohort.certify(self.plan())
        self.assertEqual((report["status"], report["required_cardinality"]), ("PendingIndependentQE", 4))
        self.assertEqual(report["makespan_started_ns"], self.cohort.release_ns)
        self.assertEqual(report["makespan_finished_ns"] - report["makespan_started_ns"], 100 * 10 ** 9)
        self.assertEqual(set(report["per_slot_metrics"]), {f"agent-{i}" for i in range(4)})
        window = report["phase_window"]
        self.assertEqual((window["status"], window["population_policy"], window["target_fraction"]),
                         ("Qualified", "forward_demand_v2", 1))
        self.assertEqual(window["slots_without_required_forward_witness"], [])
        with self.assertRaisesRegex(Invalid, "^actual admitted-slot population/progress not qualified$"):
            self.cohort.certify(self.cohort.plan())  # ready_now_v1: busy owners are not ready-now
        qe = self.cohort.attestation()
        report = self.cohort.certify(self.plan(qe=qe))
        self.assertEqual((report["status"], report["qe"]), ("Qualified", qe))

    def forged(self, name, mutate, reason):
        path = self.cohort.client / name
        original = path.read_bytes()
        value = json.loads(original)
        mutate(value)
        write_json(path, value)
        try:
            manifest = self.cohort.reassemble()
            with self.subTest(reason=reason), self.assertRaisesRegex(Invalid, "^" + re.escape(reason) + "$"):
                self.cohort.certify(self.plan(), manifest)
        finally:
            path.write_bytes(original)

    def test_release_identity_and_ready_join_forgeries(self):
        self.forged("summary.json", lambda s: s["barrier"].update(release_sha256="0" * 64), "actual release identity")
        self.forged("barrier/release.json", lambda r: r.update(release_monotonic_ns=r["release_monotonic_ns"] + 1),
                    "actual release identity")  # the coordinator summary still names the original release
        self.forged("summary.json", lambda s: s["barrier"]["ready"]["agent-1"].update(initial_messages_sha256="0" * 64),
                    "raw ready/aggregate join")
        self.forged("summary.json", lambda s: s["exits"].pop("agent-3"), "actual coordinator exits")
        self.forged("summary.json", lambda s: s.update(status="Failed"), "actual four-task aggregate")
        stale = self.plan()
        stale["barrier_token"] = "f" * 32
        with self.assertRaisesRegex(Invalid, "^stale/mismatched barrier release$"):
            self.cohort.certify(stale)

    def test_makespan_start_and_completion_clock_forgeries(self):
        self.forged("summary.json", lambda s: s.update(makespan_started_ns=s["makespan_started_ns"] + 10 ** 9),
                    "actual declared cohort release clock")
        self.forged("summary.json", lambda s: s.update(makespan_finished_ns=s["makespan_finished_ns"] - 10 ** 9),
                    "actual four-task completion clock")


class TraceFileControls(unittest.TestCase):
    """The server phase trace is streamed JSONL, bounded by its own header's metadata quota."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.records, _, _ = trace_fixture()

    def trace(self, records=None, name="trace.jsonl"):
        return write_jsonl(self.root / name, self.records if records is None else records)

    def open(self, path):
        return certify.TraceFile(certify.BoundReader(), ref(path))

    def test_multiline_jsonl_is_streamed_on_every_pass(self):
        trace = self.open(self.trace())
        self.assertEqual(trace.limit, f2_trace.TOTAL_LIMIT)
        self.assertEqual(list(trace), self.records)
        self.assertEqual(list(trace), self.records)  # re-iterable: qualify_window makes two passes

    def test_single_json_document_is_not_a_trace(self):
        path = self.root / "document.json"
        path.write_text(json.dumps(self.records) + "\n")
        with self.assertRaisesRegex(Invalid, "^bound trace header required$"):
            self.open(path)

    def test_size_bound_is_the_header_metadata_quota(self):
        path = self.trace()
        size = path.stat().st_size
        with mock.patch.object(f2_trace, "TOTAL_LIMIT", size - 1):
            with self.assertRaisesRegex(Invalid, "^phase trace exceeds its header metadata quota$"):
                self.open(path)
        with mock.patch.object(f2_trace, "TOTAL_LIMIT", size):
            self.assertEqual(self.open(path).limit, size)
        for profile, limit in (("native_command_capture_v2", f2_trace.NATIVE_TOTAL_LIMIT),
                               ("native_command_capture_v3", f2_trace.NATIVE_V3_TOTAL_LIMIT)):
            records, _, _ = trace_fixture()
            records[0]["payload"]["metadata_budget_profile"] = profile
            path = self.trace(records, f"{profile}.jsonl")
            with self.subTest(profile=profile), mock.patch.object(f2_trace, "TOTAL_LIMIT", 1):
                self.assertEqual(self.open(path).limit, limit)
        records, _, _ = trace_fixture()
        records[0]["payload"]["metadata_budget_profile"] = "native_command_capture_v9"
        with self.assertRaisesRegex(ValueError, "^unsupported metadata budget profile$"):
            self.open(self.trace(records, "unsupported.jsonl"))

    def test_assembler_uses_the_same_header_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            cohort = Cohort(Path(tmp))
            with mock.patch.object(f2_trace, "TOTAL_LIMIT", cohort.trace.stat().st_size - 1):
                with self.assertRaisesRegex(Invalid, "^phase trace exceeds its header metadata quota$"):
                    q2_manifest.assemble(cohort.spec)

    def test_over_quota_or_unterminated_lines_are_refused(self):
        path = self.trace()
        with path.open("ab") as stream:
            stream.write(b'{"padding":"' + b"a" * f2_trace.EVENT_LIMIT + b'"}\n')
        with self.assertRaisesRegex(Invalid, "^unterminated or over-quota trace line$"):
            list(self.open(path))
        path = self.trace(name="unterminated.jsonl")
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(Invalid, "^unterminated or over-quota trace line$"):
            list(self.open(path))

    def test_trace_changed_after_binding_fails_the_pass(self):
        path = self.trace()
        trace = self.open(path)
        self.records[-1]["at_ns"] += 1  # same length, still valid JSON
        self.trace()
        with self.assertRaisesRegex(Invalid, "^raw evidence hash changed$"):
            list(trace)

    def test_one_shot_iterators_are_refused_by_the_validators(self):
        _, identity, scenario = trace_fixture()
        with self.assertRaisesRegex(ValueError, "^header and final bound required$"):
            f2_trace.validate_trace(iter(self.records), identity, scenario)
        with self.assertRaisesRegex(ValueError, "^header and final bound required$"):
            f2_trace.qualify_window(iter(self.records), (0, 1), 0)


if __name__ == "__main__":
    unittest.main()
