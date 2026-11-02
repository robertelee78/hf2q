"""Host-only controls for the Q1 endpoint supervisor; no model, binary or child process."""
import hashlib
import json
import signal
import tempfile
import unittest
from pathlib import Path

import q1_endpoint_supervisor as s

IDENTITY = {"family": "Gemma4", "source_commit": "a" * 40}
FAMILY_BINDINGS = {
    "filter": "fam::endpoint", "packet_env": "HF2Q_Q1_FAM_PACKET",
    "completion_marker": "FAM_ENDPOINT_COMPLETE", "marker_stream": "stdout",
}


def at(agent, row, stage="Logits", layer=4294967295):
    return {"logical_step": 0, "layer": layer, "stage": stage, "agent": agent,
            "logical_row_start": row, "segment": 0}


def write_run(out, execution, captured, expected=None):
    (out / "output").mkdir(parents=True)
    (out / "output/execution.json").write_text(json.dumps(execution))
    trace = {"identity": IDENTITY, "acquisition": {"observed_model_opens": 3, "required_model_opens": 3},
             "coverage": {"captured": captured, "expected": captured if expected is None else expected, "errors": []}}
    (out / "output/trace.json").write_text(json.dumps(trace))


def bindings(cases, first="t0"):
    return {**FAMILY_BINDINGS, "family": "gemma", "first_case": first, "cases": cases}


LOCALIZER_CASE = {
    "kind": "localizer", "out": "/x/loc", "packet": {}, "operating_review": {},
    "filter": "adr063_invariance::gemma_localizer::q1_gemma_actual_stage_localizer",
    "packet_env": "HF2Q_Q1_GEMMA_LOCALIZER_PACKET",
    "completion_marker": "Q1_GEMMA_ACTUAL_STAGE_LOCALIZER_COMPLETE", "marker_stream": "stderr",
}


class Signals(unittest.TestCase):
    def test_hangup_and_quit_are_handled_like_term(self):
        self.assertEqual(set(s.HANDLED_SIGNALS), {signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT})
        s.INTERRUPTED = 0
        try:
            s.interrupt(signal.SIGHUP, None)
            self.assertEqual(s.INTERRUPTED, signal.SIGHUP)
        finally:
            s.INTERRUPTED = 0


class SingleRead(unittest.TestCase):
    def test_hashed_bytes_are_the_parsed_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bindings.json"
            data = json.dumps({"schema": s.SCHEMA}).encode()
            path.write_bytes(data)
            pinned = hashlib.sha256(data).hexdigest()
            self.assertEqual(s.checked_bytes(path, pinned, "pin"), data)
            self.assertEqual(s.load({"path": str(path), "sha256": pinned}, "pin"), {"schema": s.SCHEMA})
            path.write_bytes(data + b" ")
            with self.assertRaises(ValueError):
                s.checked_bytes(path, pinned, "pin")
            with self.assertRaises(ValueError):
                s.load({"path": "relative.json", "sha256": pinned}, "pin")


class CasePlan(unittest.TestCase):
    def test_endpoint_plan_uses_family_launch_and_case_filter_override(self):
        b = bindings({"t0": {"trajectory": "T0ColdSerial"},
                      "t2": {"trajectory": "T2ActualCohort", "filter": "fam::cohort"}})
        plan = s.case_plan(b, "t0")
        self.assertEqual(plan, dict(kind="endpoint", mode="T0ColdSerial", **FAMILY_BINDINGS))
        self.assertEqual(s.case_plan(b, "t2")["filter"], "fam::cohort")

    def test_first_case_must_be_the_t0_endpoint(self):
        for cases, first in (({"t0": {"trajectory": "T4ReverseOrder"}}, "t0"),
                             ({"t0": {"trajectory": "T0ColdSerial"}}, "t4"),
                             ({"loc": LOCALIZER_CASE}, "loc"),
                             ({"t0": {"trajectory": "T0ColdSerial", "kind": "localizer"}}, "t0")):
            with self.assertRaises(ValueError):
                s.case_plan(bindings(cases, first), next(iter(cases)))

    def test_localizer_plan_is_closed_and_has_no_trajectory(self):
        b = bindings({"t0": {"trajectory": "T0ColdSerial"}, "loc": LOCALIZER_CASE})
        plan = s.case_plan(b, "loc")
        self.assertEqual(plan["kind"], "localizer")
        self.assertEqual(plan["mode"], "localizer")
        self.assertEqual(plan["packet_env"], "HF2Q_Q1_GEMMA_LOCALIZER_PACKET")
        for bad in ({**LOCALIZER_CASE, "trajectory": "T3Chunk64"},
                    {**LOCALIZER_CASE, "marker_stream": "../receipt.json"},
                    {**LOCALIZER_CASE, "completion_marker": ""},
                    {k: v for k, v in LOCALIZER_CASE.items() if k != "filter"},
                    {**LOCALIZER_CASE, "kind": "other"}):
            with self.assertRaises(ValueError):
                s.case_plan(bindings({"t0": {"trajectory": "T0ColdSerial"}, "loc": bad}), "loc")


class Localizer(unittest.TestCase):
    LANES = [[1000 + 144 * lane + i for i in range(144)] for lane in range(4)]

    def run_case(self, packet_extra=None, execution_extra=None, agents=None, marker=None, packet_sha=None):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root)
            packet = {"agent": 2, "tokens": self.LANES[2], "workload": {"lanes": self.LANES},
                      "rerun_id": "loc-r64", "host_payload_limit": 1 << 20,
                      "expected": [{"at": at("agent-2", r, "CacheKey", 1)} for r in (0, 64)]}
            packet.update(packet_extra or {})
            (out / "packet.json").write_text(json.dumps(packet))
            captured = [{"at": at(agents or "agent-2", r, "CacheKey", 1), "logical_bytes": 1024} for r in (0, 64)]
            execution = {"schema": s.LOCALIZER_SCHEMA, "packet_sha256": packet_sha or s.sha(out / "packet.json"),
                         "rerun_id": "loc-r64", "records": 2, "whole_q1_qualified": False,
                         "prefill_chunks": [64, 64], "warm_live_kernel": "VecHybrid"}
            execution.update(execution_extra or {})
            write_run(out, execution, captured)
            (out / "stderr").write_text("noise\n" + (marker or "Q1_GEMMA_ACTUAL_STAGE_LOCALIZER_COMPLETE records=2 whole_q1_qualified=false") + "\n")
            b = bindings({"t0": {"trajectory": "T0ColdSerial"}, "loc": LOCALIZER_CASE})
            plan = s.case_plan(b, "loc")
            return s.validate_localizer(out, b, plan, packet, IDENTITY, {"phase_payload_upper_bytes": 1 << 20})

    def test_complete_single_lane_localizer_passes(self):
        evidence = self.run_case({"prefill_chunks": [64, 64]})
        self.assertEqual(evidence["captured_records"], 2)
        self.assertEqual(evidence["captured_bytes"], 2048)
        self.assertNotIn("logit_records", evidence)

    def test_refusals(self):
        for kwargs in (dict(packet_extra={"prefill_chunks": [128]}),                                  # route drift
                       dict(packet_extra={"prefill_chunks": [64, 64]}, agents="agent-0"),             # other lane
                       dict(packet_extra={"prefill_chunks": [64, 64]}, packet_sha="0" * 64),          # other packet
                       dict(packet_extra={"prefill_chunks": [64, 64]},
                            marker="Q1_GEMMA_ACTUAL_STAGE_LOCALIZER_COMPLETE records=3 whole_q1_qualified=false"),
                       dict(packet_extra={"prefill_chunks": [64, 64]}, execution_extra={"warm_live_kernel": "TiledF16"}),
                       dict(packet_extra={"prefill_chunks": [64, 64], "tokens": self.LANES[1]}),      # not lane 2
                       dict(packet_extra={"prefill_chunks": [64, 64], "host_payload_limit": 2047})):
            with self.assertRaises((ValueError, KeyError)):
                self.run_case(**kwargs)

    def test_declared_tiled_kernel_is_accepted_only_when_resolved(self):
        self.run_case({"prefill_chunks": [64, 64], "warm_live_kernel": "TiledF16"}, {"warm_live_kernel": "TiledF16"})
        with self.assertRaises(ValueError):
            self.run_case({"prefill_chunks": [64, 64], "warm_live_kernel": "TiledF16"})


class GemmaEndpointWarmKernel(unittest.TestCase):
    def run_case(self, resolved, declared=None):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root)
            lanes = [[lane * 1000 + i for i in range(144)] for lane in range(4)]
            captured = [{"at": at("agent-" + str(l), p), "logical_bytes": 8} for l in range(4) for p in range(144)]
            execution = {"whole_q1_qualified": False, "planned_tensor_payload": 8 * len(captured), "warm_live_kernel": resolved}
            write_run(out, execution, captured)
            packet = {"workload": {"lanes": lanes}}
            if declared:
                packet["warm_live_kernel"] = declared
            b = bindings({"t0": {"trajectory": "T0ColdSerial"}})
            return s.validate_endpoint(out, b, "t0", packet, IDENTITY, {"phase_payload_upper_bytes": 1 << 20})

    def test_shipped_vec_hybrid_by_default_and_declared_tiled_only(self):
        self.assertEqual(self.run_case("VecHybrid")["logit_records"], 576)
        self.run_case("TiledF16", "TiledF16")
        for resolved, declared in (("TiledF16", None), ("VecHybrid", "TiledF16")):
            with self.assertRaises(ValueError):
                self.run_case(resolved, declared)


if __name__ == "__main__":
    unittest.main()
