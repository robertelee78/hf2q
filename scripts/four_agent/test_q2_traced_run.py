"""Model-free controls of the one-attempt traced-cohort launcher (q2_traced_run) and its freezer.

Nothing here launches a server, client or model: these are the argument, identity, readiness,
trace-readback, cleanup and run-id checks with their exact refusal reasons.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import q2_traced_freeze as freeze
import q2_traced_run as traced
from q5_case import loads as case_loads

HERE = Path(__file__).resolve().parent
SHA = {k: hashlib.sha256(k.encode()).hexdigest() for k in
       ("model", "binary", "native", "tokenizer", "template", "profile")}
COMMIT = "38485a2355dbabbb41782f7d01d02b8a8293af34"
IDENTITY = {"model_sha256": SHA["model"], "server_binary_sha256": SHA["binary"], "server_source": COMMIT,
            "native_version": "0.15.15", "native_checksum": SHA["native"], "tokenizer_sha256": SHA["tokenizer"],
            "template_sha256": SHA["template"], "profile_sha256": SHA["profile"]}
LOCK = ('version = 4\n[[package]]\nname = "mlx-native"\nversion = "0.15.15"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\nchecksum = "' + SHA["native"] + '"\n')
RUN = "q2t-q2-traced-n4-20261101T000000Z"
NONCE = "q2t_" + "a" * 48


def facts(**change):
    value = dict(binary_sha256=SHA["binary"], build={"source": COMMIT, "sha256": SHA["binary"]},
                 template_sha256=SHA["profile"], model_sha256=SHA["model"], native=("0.15.15", SHA["native"]))
    value.update(change)
    return value


def config(out=Path("/runs/x")):
    return traced.trace_config(out, RUN, IDENTITY, NONCE)


def record(seq, kind, payload, at=1000):
    return {"schema": "hf2q.f2.phase-trace.v1", "seq": seq, "at_ns": at + seq, "kind": kind, "payload": payload}


def header(cfg, **change):
    payload = {"identity": cfg["identity"], "run_id": cfg["run_id"], "metadata_budget_profile": cfg["metadata_budget_profile"],
               "native_observer": cfg["native_observer"], "origin_monotonic_ns": 5_000}
    payload.update(change)
    return record(0, "header", payload)


def ack(cfg, **change):
    payload = {"nonce": cfg["native_observer"]["control_nonce"], "worker_queue_checked": True, "quiescent": True,
               "baseline": {"pending_commands": 0, "invalid": False, "dropped_events": 0}}
    payload.update(change)
    return record(7, "native_arm_ack", payload)


def write_lines(path, records, tail=b""):
    path.write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in records) + tail)


class LauncherArgumentsTest(unittest.TestCase):
    BASE = ["--binary", "/b/hf2q", "--build-receipt", "/b/build-receipt.json", "--identity", "/i.json",
            "--negative-control-receipt", "/nc.json", "--output", "/runs/q2-traced-n4-1"]

    def test_every_population_maps_to_its_client_phase_plan(self):
        self.assertEqual({p: traced.phase_plan(p) for p in traced.POPULATIONS},
                         {"n4": None, "n1": "1", "3+1": "3+1", "2+2": "2+2", "1+3": "1+3"})
        with self.assertRaisesRegex(ValueError, "population must be one of n4, n1, 3\\+1, 2\\+2, 1\\+3"):
            traced.phase_plan("4+0")

    def test_parse_args_requires_known_population_and_absolute_paths(self):
        args = traced.parse_args(["--population", "2+2", *self.BASE])
        self.assertEqual((args.population, args.case), ("2+2", HERE / "q5_cases" / "qwen38.json"))
        for bad in (["--population", "n2", *self.BASE], ["--population", "n4", *self.BASE[:-2], "--output", "runs/x"],
                    ["--population", "n4", *self.BASE[2:]]):
            with self.subTest(argv=bad), self.assertRaises(SystemExit):
                traced.parse_args(bad)

    def test_prepare_refuses_reused_git_or_ambient_binary_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp).resolve()
            existing = parent / "q2-traced-n4-1"
            existing.mkdir()
            cases = ((["--output", str(existing)], {}, "output must be a new directory under a canonical parent"),
                     (["--output", str(HERE / "q2-traced-n4-new")], {}, "output must be outside any Git tree"),
                     (["--output", str(parent / "q2-traced-n4-2")], {"HF2Q_Q5_HF2Q_BIN": "/x"},
                      "set the binary with --binary, not HF2Q_Q5_HF2Q_BIN"))
            for extra, environ, reason in cases:
                with self.subTest(reason=reason):
                    args = traced.parse_args(["--population", "n4", *self.BASE[:-2], *extra])
                    with self.assertRaisesRegex(ValueError, reason):
                        traced.prepare(args, environ)


class RunIdentityTest(unittest.TestCase):
    def test_one_run_id_from_the_output_name(self):
        self.assertEqual(traced.run_id_for("q2-traced-n4-20261101T000000Z"), RUN)
        self.assertEqual(traced.run_id_for("q2-traced-3+1-xyz"), "q2t-q2-traced-3-1-xyz")
        for name in ("short", "x" * 61):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "16..64 character run id"):
                traced.run_id_for(name)

    def test_native_config_is_the_reviewed_v3_capture_with_the_shared_run_id(self):
        cfg = config(Path("/runs/q2-traced-n4-1"))
        self.assertEqual((cfg["output"], cfg["run_id"], cfg["identity"]),
                         ("/runs/q2-traced-n4-1/native-phase-trace.jsonl", RUN, IDENTITY))
        self.assertEqual((cfg["server_budget_bytes"], cfg["client_budget_bytes"], cfg["metadata_budget_profile"]),
                         (6 * 1024 ** 3, 128 * 1024 ** 2, "native_command_capture_v3"))
        self.assertEqual({k: v for k, v in cfg["native_observer"].items() if k != "witness_manifest"},
                         {"event_limit": 8_000_000, "metadata_limit_bytes": 16 * 1024 ** 2,
                          "diagnostic_case": True, "control_nonce": NONCE})
        self.assertEqual(cfg["native_observer"]["witness_manifest"]["input_sources"],
                         ["qwen35.greedy.actual_tokens", "qwen35.forward.actual_tokens"])
        self.assertNotIn("forward_demand_policy", cfg)
        with self.assertRaisesRegex(ValueError, "invalid control nonce"):
            traced.trace_config(Path("/r"), RUN, IDENTITY, "short")
        with self.assertRaisesRegex(ValueError, "invalid run id"):
            traced.trace_config(Path("/r"), "bad id", IDENTITY, NONCE)

    def test_identity_binds_binary_build_template_model_and_native_pin(self):
        self.assertEqual(traced.check_identity(dict(IDENTITY), **facts()), IDENTITY)
        refusals = [
            ({k: v for k, v in IDENTITY.items() if k != "profile_sha256"}, facts(),
             "exactly the eight q2_certify identity keys"),
            ({**IDENTITY, "tokenizer_sha256": "ABC"}, facts(), "identity tokenizer_sha256 must be 64 lowercase hex"),
            ({**IDENTITY, "server_source": "38485a23"}, facts(), "server_source must be a 40-hex commit"),
            (IDENTITY, facts(binary_sha256=SHA["model"]), "server_binary_sha256 is not the sha256 of --binary"),
            (IDENTITY, facts(build={"source": COMMIT, "sha256": SHA["model"]}), "build receipt sha256 differs"),
            (IDENTITY, facts(build={"source": "0" * 40, "sha256": SHA["binary"]}),
             "server_source differs from the build receipt source"),
            (IDENTITY, facts(template_sha256=SHA["template"]), "profile_sha256 is not the sha256 of the client approval"),
            (IDENTITY, facts(model_sha256=None), "model_sha256 differs from the artifact's conversion-receipt"),
            (IDENTITY, facts(native=("0.15.9", SHA["native"])), "native_version/native_checksum differ from mlx-native"),
        ]
        for identity, given, reason in refusals:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                traced.check_identity(dict(identity), **given)

    def test_native_pin_reads_the_one_published_package(self):
        self.assertEqual(traced.native_pin(LOCK), ("0.15.15", SHA["native"]))
        local = LOCK.replace('source = "registry+https://github.com/rust-lang/crates.io-index"\n', "")
        for text in (local, LOCK + LOCK.split("\n", 1)[1]):
            with self.assertRaisesRegex(ValueError, "exactly one published mlx-native"):
                traced.native_pin(text)


class CaseAndReadinessTest(unittest.TestCase):
    def setUp(self):
        self.case = case_loads((HERE / "q5_cases" / "qwen38.json").read_bytes())

    def test_traced_case_adds_only_the_trace_config_and_the_phase_plan(self):
        n4, mixed = traced.traced_case(self.case, "n4"), traced.traced_case(self.case, "1+3")
        self.assertEqual(n4["serve"]["env"], {**self.case["serve"]["env"],
                                              "HF2Q_PHASE_TRACE_CONFIG": "{OUT}/native-config.json"})
        self.assertEqual(n4["client"]["argv"], self.case["client"]["argv"])
        self.assertEqual(mixed["client"]["argv"], self.case["client"]["argv"] + ["--phase-plan", "1+3"])
        twice = traced.traced_case(self.case, "n4")
        with self.assertRaisesRegex(ValueError, "case already sets HF2Q_PHASE_TRACE_CONFIG"):
            traced.traced_case(twice, "n4")
        with self.assertRaisesRegex(ValueError, "Qwen3.8 four-slot case"):
            traced.traced_case({**self.case, "family": "gemma4"}, "n4")

    def ready(self, **change):
        runtime = {"serve": {"max_slots": 4}, "measurement_snapshot": {
            "process_pid": 41, "active_controls": {"vision_projector": False},
            "tokenizer_sha256": SHA["tokenizer"], "template_sha256": SHA["template"]}}
        values = dict(health={"model": "Qwen3.8 27B", "context_length": 262144}, runtime=runtime, readyz={"ready": True})
        values.update(change)
        return traced.readiness_envelope(values["health"], values["runtime"], values["readyz"],
                                         model_id="Qwen3.8 27B", server_pid=41, identity=IDENTITY)

    def test_readiness_envelope_is_checked_against_the_launched_process(self):
        self.assertEqual(set(self.ready()), {"health", "runtime", "readyz"})
        snap = lambda **c: {"serve": {"max_slots": 4}, "measurement_snapshot": {  # noqa: E731
            "process_pid": 41, "active_controls": {"vision_projector": False},
            "tokenizer_sha256": SHA["tokenizer"], "template_sha256": SHA["template"], **c}}
        refusals = [(dict(readyz={"ready": False}), "readyz is not ready"),
                    (dict(health={"model": "Qwen3.8 27B", "context_length": 131072}), "context_length 262144"),
                    (dict(runtime={**snap(), "serve": {"max_slots": 1}}), "serve.max_slots is not 4"),
                    (dict(runtime=snap(process_pid=40)), "process_pid is not the launched server pid"),
                    (dict(runtime=snap(active_controls={"vision_projector": True})), "vision projector loaded"),
                    (dict(runtime=snap(template_sha256=SHA["model"])), "tokenizer/template sha256 differ")]
        for change, reason in refusals:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                self.ready(**change)


class TraceReadbackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "native-phase-trace.jsonl"
        self.cfg = config()

    def tearDown(self):
        self.tmp.cleanup()

    def test_arm_readback_waits_for_the_ack_then_checks_it(self):
        self.assertIsNone(traced.arm_readback(self.path, self.cfg))
        write_lines(self.path, [header(self.cfg)], tail=b'{"partial"')
        self.assertIsNone(traced.arm_readback(self.path, self.cfg))
        write_lines(self.path, [header(self.cfg), ack(self.cfg)])
        self.assertEqual(traced.arm_readback(self.path, self.cfg)["ack"]["seq"], 7)
        refusals = [([header(self.cfg, run_id="q2t-another-run-identity")], "trace header differs from native-config"),
                    ([header(self.cfg), ack(self.cfg, nonce="q2t_" + "b" * 48)], "native arm ACK nonce/worker check"),
                    ([header(self.cfg), ack(self.cfg, quiescent=False)], "native arm ACK nonce/worker check"),
                    ([header(self.cfg), ack(self.cfg, baseline={"pending_commands": 1, "invalid": False,
                                                                 "dropped_events": 0})], "baseline is not clean"),
                    ([header(self.cfg), record(3, "slots", {"cause": "admit", "slots": [{"slot_id": 0}]}),
                      ack(self.cfg)], "a slot was admitted before the client launched")]
        for records, reason in refusals:
            with self.subTest(reason=reason):
                write_lines(self.path, records)
                with self.assertRaisesRegex(ValueError, reason):
                    traced.arm_readback(self.path, self.cfg)

    def test_trace_final_needs_the_bound_header_and_a_clean_end(self):
        end = record(9, "end", {"error": None, "dropped_events": 0})
        write_lines(self.path, [header(self.cfg), ack(self.cfg), end])
        final = traced.trace_final(self.path, self.cfg)
        self.assertEqual((final["end"]["seq"], final["origin_monotonic_ns"]), (9, 5_000))
        refusals = [([header(self.cfg), ack(self.cfg)], "no clean end record"),
                    ([header(self.cfg), record(9, "end", {"error": None, "dropped_events": 3})], "no clean end record"),
                    ([header(self.cfg, identity={**IDENTITY, "server_source": "0" * 40}), end], "run id/identity differ")]
        for records, reason in refusals:
            with self.subTest(reason=reason):
                write_lines(self.path, records)
                with self.assertRaisesRegex(ValueError, reason):
                    traced.trace_final(self.path, self.cfg)
        write_lines(self.path, [header(self.cfg), end], tail=b'{"kind"')
        with self.assertRaisesRegex(ValueError, "does not end with a complete record"):
            traced.trace_final(self.path, self.cfg)


class Proc:
    def __init__(self, pid, returncode):
        self.pid, self.returncode = pid, returncode


class CleanupTest(unittest.TestCase):
    def test_one_entry_per_owned_group_from_the_termination_proof(self):
        proven = {"remaining": [], "refused_pgids": [], "errors": []}
        entries = traced.cleanup_entries(proven, [("server", Proc(10, 0)), ("client", Proc(20, -15)), ("none", None)])
        self.assertEqual([(e["label"], e["reaped"], e["group_empty"], e["errors"]) for e in entries],
                         [("server", True, True, []), ("client", True, True, [])])
        held = {"remaining": [{"pid": 11, "pgid": 10}], "refused_pgids": [20], "errors": ["killpg"]}
        entries = traced.cleanup_entries(held, [("server", Proc(10, None)), ("client", Proc(20, 0))])
        self.assertEqual([(e["reaped"], e["group_empty"], e["errors"]) for e in entries],
                         [(False, False, ["killpg"]), (True, False, ["killpg"])])
        blind = traced.cleanup_entries({"remaining": None, "refused_pgids": [], "errors": []}, [("server", Proc(10, 0))])
        self.assertFalse(blind[0]["group_empty"])


class FreezeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name).resolve() / "q2-traced-3+1-20261101T000000Z"
        (self.run / "client" / "barrier").mkdir(parents=True)
        self.cfg = config(self.run)
        self.receipt = {"schema": traced.RECEIPT_SCHEMA, "status": traced.COMPLETE, "population": "3+1",
                        "phase_plan": "3+1", "run_id": RUN, "identity": IDENTITY, "server_pid": 41,
                        "protocol_complete": True, "final_verified": True, "cleanup_proved": True}
        self.approval = {"run_id": RUN, "engine_kind": "hf2q", "server_pid": 41, "model": "Qwen3.8 27B",
                         "family": "qwen", "barrier_token": "t" * 32}
        self.write()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self):
        for name, value in (("receipt.json", self.receipt), ("client-approval.json", self.approval),
                            ("native-config.json", self.cfg)):
            (self.run / name).write_text(json.dumps(value))
        write_lines(self.run / "native-phase-trace.jsonl", [header(self.cfg)])

    def test_one_run_id_across_receipt_approval_config_and_trace(self):
        self.assertEqual(freeze.load_run(self.run)["header"]["origin_monotonic_ns"], 5_000)
        self.approval["run_id"] = "q2t-another-run-identity"
        self.write()
        with self.assertRaisesRegex(ValueError, "must share one run id"):
            freeze.load_run(self.run)
        self.approval["run_id"], self.receipt["status"] = RUN, "FailedOrHeld"
        self.write()
        with self.assertRaisesRegex(ValueError, "not a complete traced cohort"):
            freeze.load_run(self.run)

    def test_windows_are_fixed_by_population_and_client_clocks(self):
        release = {"f2_clock_id": "CLOCK_MONOTONIC", "release_monotonic_ns": 9_000}
        (self.run / "client" / "barrier" / "release-2.json").write_text(json.dumps(release))
        info = freeze.load_run(self.run)
        self.assertEqual(freeze.WINDOWS["3+1"], (((1, 3), "release-2"),))
        self.assertEqual(freeze.arrival_ns(info, "release-2"), 4_000)
        for agent, first in zip(("agent-0", "agent-1", "agent-2", "agent-3"), (7_000, 8_000, 12_000, 6_000)):
            (self.run / "client" / agent).mkdir()
            turns = [{}, {"f2_clock_id": "CLOCK_MONOTONIC", "progress_event_monotonic_ns": [first, first + 5]}]
            (self.run / "client" / agent / "summary.json").write_text(json.dumps({"turns": turns}))
        self.assertEqual(freeze.arrival_ns(info, "decode"), 7_000)
        scenario = freeze.scenario(info, "forward_demand_v2", (1, 3), 4_000)
        self.assertEqual({k: scenario[k] for k in ("run_id", "composition", "arrival_ns", "population_policy")},
                         {"run_id": RUN, "composition": [1, 3], "arrival_ns": 4_000,
                          "population_policy": "forward_demand_v2"})
        with self.assertRaisesRegex(ValueError, "certifies the canonical N4 release only"):
            freeze.freeze_plan(info, Path(self.tmp.name), "forward_demand_v2")
        with self.assertRaisesRegex(ValueError, "population policy must be one of"):
            freeze.scenario(info, "ready_later", (1, 3), 4_000)


if __name__ == "__main__":
    unittest.main()
