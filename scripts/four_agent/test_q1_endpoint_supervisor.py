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
    "kind": "localizer", "out": "/x/loc", "packet": {},
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


class RestoreField(unittest.TestCase):
    def test_every_qwen_family_requires_its_t1_restore64_field(self):
        # qwen_actual.rs writes `<family>_restore64_exact` for Qwen3.6 and Qwen3.8 T1Resume64.
        self.assertEqual(s.RESTORE64_FIELD["qwen36"], "qwen36_restore64_exact")
        self.assertEqual(s.RESTORE64_FIELD["qwen38"], "qwen38_restore64_exact")
        self.assertTrue(all(s.RESTORE64_FIELD.values()))


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
                    {**LOCALIZER_CASE, "kind": "other"},
                    # Pinning the operating review in the bindings is a hash cycle: it is a launch argument.
                    {**LOCALIZER_CASE, "operating_review": {"path": "/x/review.json", "sha256": "0" * 64}}):
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


MULTISTORE_CASE = {
    "kind": "multistore", "out": "/x/loc4", "packet": {},
    "filter": "adr063_invariance::gemma_actual::q1_gemma_actual_cohort_stage_localizer",
    "packet_env": "HF2Q_Q1_GEMMA_ENDPOINT_PACKET",
    "completion_marker": "Q1_GEMMA_COHORT_STAGE_LOCALIZER_COMPLETE", "marker_stream": "stdout",
    "stores": {"endpoint": ".", "stages": "stages"}, "checks": ["gemma_loc4"],
}


class MultistorePlan(unittest.TestCase):
    def plan(self, case):
        return s.case_plan(bindings({"t0": {"trajectory": "T0ColdSerial"}, "loc4": case}), "loc4")

    def test_closed_multistore_plan(self):
        plan = self.plan(MULTISTORE_CASE)
        self.assertEqual((plan["kind"], plan["mode"], plan["checks"], plan["completion_line"]), ("multistore", "multistore", ["gemma_loc4"], None))
        self.assertEqual(plan["stores"], {"endpoint": ".", "stages": "stages"})
        fixed = self.plan({**MULTISTORE_CASE, "completion_line": "Q1_GEMMA_COHORT_STAGE_LOCALIZER_COMPLETE mode=x whole_q1=false"})
        self.assertTrue(fixed["completion_line"].endswith("whole_q1=false"))
        self.plan({**MULTISTORE_CASE, "manifest": {"path": "/x/m.json", "sha256": "0" * 64}})
        for bad in ({**MULTISTORE_CASE, "trajectory": "T2ActualCohort"},
                    {**MULTISTORE_CASE, "checks": ["unknown_check"]},
                    {**MULTISTORE_CASE, "checks": []},
                    {**MULTISTORE_CASE, "checks": ["gemma_loc4", "gemma_loc4"]},
                    {**MULTISTORE_CASE, "stores": []},
                    {**MULTISTORE_CASE, "stores": {"endpoint": ".", "stages": "../escape"}},
                    {**MULTISTORE_CASE, "stores": {"endpoint": ".", "stages": "/abs"}},
                    {**MULTISTORE_CASE, "stores": {"endpoint": ".", "stages": "a//b"}},
                    {**MULTISTORE_CASE, "stores": {"endpoint": ".", "stages": "."}},
                    {**MULTISTORE_CASE, "completion_line": "OTHER_MARKER records=1"},
                    {**MULTISTORE_CASE, "manifest": {"path": "/x/m.json"}},
                    # refused at plan time, before any launch consumes the one-attempt review
                    {**MULTISTORE_CASE, "manifest": {"path": "m.json", "sha256": "0" * 64}},
                    {**MULTISTORE_CASE, "manifest": {"path": "./sub/m.json", "sha256": "0" * 64}},
                    {**MULTISTORE_CASE, "manifest": {"path": None, "sha256": "0" * 64}},
                    {**MULTISTORE_CASE, "operating_review": {"path": "/x/review.json", "sha256": "0" * 64}},
                    {k: v for k, v in MULTISTORE_CASE.items() if k != "stores"},
                    {**MULTISTORE_CASE, "marker_stream": "receipt.json"}):
            with self.assertRaises(ValueError):
                self.plan(bad)

    def test_storeless_case_and_fixed_line_exactly_once(self):
        line = "Q1_GEMMA_COHORT_STAGE_LOCALIZER_COMPLETE mode=prep whole_q1=false"
        case = {**MULTISTORE_CASE, "stores": {}, "checks": ["probe"], "completion_line": line}
        seen = []
        s.MULTISTORE_CHECKS["probe"] = lambda *a: seen.append(a[7]) or {"ok": True}
        try:
            b = bindings({"t0": {"trajectory": "T0ColdSerial"}, "loc4": case})
            plan = s.case_plan(b, "loc4")
            self.assertEqual(plan["stores"], {})
            for count, passes in ((1, True), (0, False), (2, False)):
                with tempfile.TemporaryDirectory() as root:
                    out = Path(root); (out / "output").mkdir()
                    (out / "packet.json").write_text("{}")
                    (out / "output/execution.json").write_text(json.dumps({"packet_sha256": s.sha(out / "packet.json"), "whole_q1_qualified": False}))
                    (out / "stdout").write_text("noise\n" + (line + "\n") * count)
                    extent = {"lanes": [0, 1, 2, 3], "host_payload_limit": 1, "expected_stores": {}}
                    run = lambda: s.validate_multistore(out, b, "loc4", plan, {}, extent, IDENTITY, {"phase_payload_upper_bytes": 1})
                    if passes:
                        evidence = run()
                        self.assertEqual((evidence["captured_records"], evidence["stores"], evidence["probe"]), (0, {}, {"ok": True}))
                    else:
                        with self.assertRaises(ValueError):
                            run()
            self.assertEqual(seen, [{}])  # the check owns a storeless receipt
        finally:
            del s.MULTISTORE_CHECKS["probe"]

    def test_multistore_cannot_be_the_first_case(self):
        with self.assertRaises(ValueError):
            s.case_plan(bindings({"loc4": {**MULTISTORE_CASE, "trajectory": "T0ColdSerial"}}, "loc4"), "loc4")

    def test_packet_scope_by_kind(self):
        endpoint = dict(kind="endpoint", mode="T2ActualCohort")
        # Every reviewed endpoint packet carries its own host_payload_limit; that stays admitted.
        self.assertTrue(s.packet_scope(endpoint, {"trajectory": "T2ActualCohort", "host_payload_limit": 962391936}))
        for extra in ({"expected_stores": {}}, {"lanes": [0, 1, 2, 3]}, {"stage_localizer": {}}):
            self.assertFalse(s.packet_scope(endpoint, {"trajectory": "T2ActualCohort", **extra}))
        self.assertFalse(s.packet_scope(endpoint, {"trajectory": "T0ColdSerial"}))
        self.assertFalse(s.packet_scope(dict(kind="localizer"), {"trajectory": "T0ColdSerial"}))
        self.assertTrue(s.packet_scope(dict(kind="multistore"), {"trajectory": "T2ActualCohort"}))

    EXTENT = {"lanes": [0, 1, 2, 3], "host_payload_limit": 10, "expected_stores": {"a": []}}
    MANIFEST_CASE = {**MULTISTORE_CASE, "manifest": {"path": "/x/m.json", "sha256": "0" * 64}}

    def manifest_extent(self, packet, manifest=None):
        return s.multistore_extent(self.MANIFEST_CASE, packet, json.dumps(manifest or self.EXTENT).encode())

    def test_packet_form_extent(self):
        self.assertEqual(s.multistore_extent(MULTISTORE_CASE, dict(self.EXTENT, other=1), None), self.EXTENT)
        extent = self.EXTENT
        for packet in ({},                                                     # no extent at all
                       {k: v for k, v in extent.items() if k != "expected_stores"},
                       {k: v for k, v in extent.items() if k != "lanes"},
                       {**extent, "lanes": [1, 0]}, {**extent, "lanes": [4]}, {**extent, "lanes": []},
                       {**extent, "host_payload_limit": 0}):
            with self.assertRaises((ValueError, KeyError)):
                s.multistore_extent(MULTISTORE_CASE, packet, None)

    def test_manifest_form_admits_equal_packet_lanes_and_host_limit(self):
        self.assertEqual(self.manifest_extent({"trajectory": "T5"}), self.EXTENT)  # plan-only packet
        packet = {"trajectory": "T2ActualCohort", "lanes": [0, 1, 2, 3], "host_payload_limit": 10}
        self.assertEqual(self.manifest_extent(packet), self.EXTENT)
        self.assertEqual(self.manifest_extent({"lanes": [0, 1, 2, 3]}), self.EXTENT)

    def test_manifest_form_refuses_unequal_packet_lanes_or_host_limit(self):
        for packet in ({"lanes": [0, 1, 2], "host_payload_limit": 10}, {"lanes": [0, 1, 2, 3], "host_payload_limit": 11},
                       {"host_payload_limit": 9}):
            with self.assertRaises(ValueError):
                self.manifest_extent(packet)

    def test_manifest_form_refuses_packet_expected_stores(self):
        for stores in ({"a": []}, {}):
            with self.assertRaises(ValueError):
                self.manifest_extent({"lanes": [0, 1, 2, 3], "host_payload_limit": 10, "expected_stores": stores})

    def test_manifest_binding_is_closed(self):
        for case, data in ((self.MANIFEST_CASE, None),                                      # case names an unread manifest
                           (MULTISTORE_CASE, json.dumps(self.EXTENT).encode()),             # bytes without a case binding
                           (self.MANIFEST_CASE, json.dumps({**self.EXTENT, "extra": 1}).encode()),
                           (self.MANIFEST_CASE, json.dumps({**self.EXTENT, "lanes": [3, 3]}).encode())):
            with self.assertRaises((ValueError, KeyError)):
                s.multistore_extent(case, {}, data)


class GemmaLoc4(unittest.TestCase):
    LANES = [[lane * 1000 + i for i in range(144)] for lane in range(4)]

    def run_case(self, trajectory="T2ActualCohort", mutate=None):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root)
            logits = [{"at": at("agent-" + str(l), p), "logical_bytes": 8} for l in range(4) for p in range(144)]
            stages = [{"at": at("agent-" + str(l), r, "Key", 1), "logical_bytes": 16} for l in range(4) for r in range(128)]
            case = dict(policy={"phase_payload_upper_bytes": 1 << 20, "lanes": 4, "stage_payload_upper_bytes": 1 << 16},
                        packet={"trajectory": trajectory, "workload": {"lanes": self.LANES},
                                "stage_localizer": {"rerun_id": "cohort-packed", "host_payload_limit": 16 * 512},
                                "lanes": [0, 1, 2, 3], "host_payload_limit": 8 * 576 + 16 * 512,
                                "routing_witness": {"payload_bytes": 1, "metadata_bytes": 1, "cpu_ledger_bytes": 1},
                                "expected_stores": {"endpoint": logits, "stages": stages}},
                        execution={"whole_q1_qualified": False, "planned_tensor_payload": 8 * 576, "warm_live_kernel": "VecHybrid",
                                   "schema": s.COHORT_SCHEMA, "trajectory": trajectory, "stage_route": s.COHORT_ROUTES.get(trajectory),
                                   "rerun_id": "cohort-packed", "stage_store": "stages", "lanes": [0, 1, 2, 3],
                                   "stage_records": 512, "stage_planned_payload": 16 * 512},
                        logits=logits, stages=stages, records=576 + 512, plan_case=dict(MULTISTORE_CASE),
                        witness={"layers": 1, "metadata_limit_bytes": 1 << 20}, witness_rows=[(l, p) for l in range(4) for p in range(144)])
            if mutate:
                mutate(case)
            packet = case["packet"]
            (out / "packet.json").write_text(json.dumps(packet))
            case["execution"]["packet_sha256"] = s.sha(out / "packet.json")
            write_run(out, case["execution"], case["logits"])
            (out / "output/stages").mkdir()
            rows = [{"kind": "routing_row", "native_boundary_valid": True, "selected_copy_valid": True, "native_decode_error": None,
                     "qualified": False, "scores_f32_bits": [0], "selected_weights_f32_bits": [0x3f800000],
                     "at": {"namespace": "Canonical", "row": {"lane": l, "position": p}, "layer": 0}} for l, p in case["witness_rows"]]
            (out / "output/routing-witness.jsonl").write_text("\n".join(json.dumps(x) for x in
                [{"kind": "header", "identity": IDENTITY, "qualified": False}] + rows +
                [{"kind": "end", "records": len(rows), "all_scored_rows_captured": True, "full_Q1_qualified": False}]) + "\n")
            (out / "output/stages/trace.json").write_text(json.dumps({"identity": IDENTITY,
                "acquisition": {"observed_model_opens": 3, "required_model_opens": 3},
                "coverage": {"captured": case["stages"], "expected": case["stages"], "errors": []}}))
            (out / "stdout").write_text("test result: ok.\nQ1_GEMMA_COHORT_STAGE_LOCALIZER_COMPLETE records=%d whole_q1_qualified=false\n" % case["records"])
            cases = {"t0": {"trajectory": "T0ColdSerial"}, "loc4": case["plan_case"]}
            b = {**bindings(cases), "routing_witness": case["witness"]} if case["witness"] is not None else bindings(cases)
            plan = s.case_plan(b, "loc4")
            extent = s.multistore_extent(case["plan_case"], packet, None)
            return s.validate_multistore(out, b, "loc4", plan, packet, extent, IDENTITY, case["policy"])

    def test_complete_four_lane_localizer_passes_for_both_routes(self):
        for trajectory in ("T2ActualCohort", "T0ColdSerial"):
            evidence = self.run_case(trajectory)
            self.assertEqual(evidence["captured_records"], 1088)
            self.assertEqual(evidence["stores"], {"endpoint": {"records": 576, "bytes": 4608}, "stages": {"records": 512, "bytes": 8192}})
            self.assertEqual(evidence["gemma_loc4"]["stage_route"], s.COHORT_ROUTES[trajectory])
            self.assertEqual(evidence["gemma_loc4"]["logit_records"], 576)

    def test_refusals(self):
        def drop_lane(c):  # stage store misses lane 3 (and its frozen manifest agrees)
            c["stages"] = [x for x in c["stages"] if x["at"]["agent"] != "agent-3"]
            c["packet"]["expected_stores"]["stages"] = c["stages"]; c["records"] -= 128
            c["execution"]["stage_records"] -= 128; c["execution"]["stage_planned_payload"] -= 2048
        def foreign_lane(c):
            c["packet"]["lanes"] = [0, 1, 2]
        def drift(c):  # captured store differs from the frozen packet extent
            c["packet"]["expected_stores"]["stages"] = c["stages"][:-1]
        def route(c):
            c["execution"]["stage_route"] = s.COHORT_ROUTES["T0ColdSerial"]
        def stage_bytes(c):
            c["execution"]["stage_planned_payload"] += 1
        def two_lane_policy(c):
            c["policy"]["lanes"] = 2
        def stage_policy(c):
            c["policy"]["stage_payload_upper_bytes"] = 1024
        def host(c):
            c["packet"]["host_payload_limit"] = 100
        def marker(c):
            c["records"] += 1
        def stores(c):
            c["plan_case"]["stores"] = {"endpoint": ".", "stages": "other"}
        def missing_logit(c):
            c["logits"] = c["logits"][:-1]; c["packet"]["expected_stores"]["endpoint"] = c["logits"]; c["records"] -= 1
        def ordinary_axis(c):
            c["packet"]["trajectory"] = "T3Chunk64"
        def no_plan(c):
            del c["packet"]["stage_localizer"]
        def no_bindings_witness(c):  # the bank would otherwise go unchecked
            c["witness"] = None
        def no_packet_witness(c):
            del c["packet"]["routing_witness"]
        def short_bank(c):
            c["witness_rows"] = c["witness_rows"][:-1]
        for mutate in (drop_lane, foreign_lane, drift, route, stage_bytes, two_lane_policy, stage_policy, host,
                       marker, stores, missing_logit, ordinary_axis, no_plan, no_bindings_witness, no_packet_witness, short_bank):
            with self.subTest(mutate.__name__), self.assertRaises((ValueError, KeyError, FileNotFoundError)):
                self.run_case(mutate=mutate)
class ResourcePolicyRules(unittest.TestCase):
    LIMITS = {"process_rss_limit_bytes": 10, "process_footprint_limit_bytes": 20}

    def test_runtime_pressure_defaults_to_normal_and_q5_admits_warning_never_critical(self):
        self.assertEqual(s.runtime_pressure({}), ("1",))
        self.assertEqual(s.runtime_pressure({"runtime_pressure_rule": "normal-every-sample"}), ("1",))
        q5 = s.runtime_pressure({"runtime_pressure_rule": "q5-critical-stop"})
        self.assertEqual(q5, ("1", "2"))
        self.assertNotIn("4", q5)
        for bad in ("warning", None, "q5"):
            with self.assertRaises(ValueError):
                s.runtime_pressure({"runtime_pressure_rule": bad})

    def test_process_caps_required_except_q5_footprint_calibration(self):
        self.assertEqual(s.process_limits(self.LIMITS), (False, 10, 20))
        calibration = {"footprint_calibration": True, "runtime_pressure_rule": "q5-critical-stop",
                       "process_rss_limit_bytes": None, "process_footprint_limit_bytes": None}
        self.assertEqual(s.process_limits(calibration), (True, None, None))
        for bad in ({**self.LIMITS, "process_rss_limit_bytes": None},
                    {**calibration, "runtime_pressure_rule": "normal-every-sample"},
                    {**calibration, "process_footprint_limit_bytes": 5},
                    {**self.LIMITS, "footprint_calibration": "yes"}):
            with self.assertRaises(ValueError):
                s.process_limits(bad)


class MemoryRecordSummary(unittest.TestCase):
    @staticmethod
    def sample(free, cached, wired, pressure="1", swap=100, compressor=50, start=0):
        return {"physical": {"free": free, "cached_files": cached, "wired": wired, "compressor": compressor},
                "pressure_raw": pressure, "swap_bytes": swap, "start_ns": start}

    def test_charge_is_baseline_min_minus_runtime_min_with_process_peaks(self):
        base = [self.sample(10, 90, 5), self.sample(12, 86, 6)]
        m = s.MemoryRecord(base)
        for t, (free, cached, wired, pressure) in enumerate([(5, 40, 50, "1"), (2, 6, 90, "2"), (3, 9, 88, "2")]):
            m.sample(self.sample(free, cached, wired, pressure, swap=100 + t, compressor=50 + 2 * t, start=1000 + t), 7 - t, 1000)
            m.ledger({"phys_footprint": 4 + t, "resident_size": 3, "wired_size": 9 - t, "lifetime_max_phys_footprint": 7})
        r = m.summary()
        self.assertEqual(r["baseline"]["min_free_plus_cached_bytes"], 98)
        self.assertEqual(r["min_free_plus_cached_bytes"], 8)
        self.assertEqual(r["charged_bytes"], 90)
        self.assertEqual(r["min_available_after_owned_outputs_bytes"], 5)
        self.assertEqual(r["pressure_counts"], {"1": 1, "2": 2})
        self.assertEqual((r["max_wired_growth_bytes"], r["max_swap_growth_bytes"], r["max_compressor_growth_bytes"]), (84, 2, 4))
        self.assertEqual(r["max_sample_drop_bytes"], 37)
        self.assertEqual((r["peak_phys_footprint_bytes"], r["peak_wired_bytes"], r["last_sample_t_ns"]), (6, 9, 2))
        self.assertEqual(m.series[1]["footprint"], 5)

    def test_max_sample_drop_is_the_largest_fall_not_the_largest_rise(self):
        m = s.MemoryRecord([self.sample(0, 100, 0)])
        for t, cached in enumerate([100, 70, 60, 61]):  # falls 30 then 10, one rise of 1
            m.sample(self.sample(0, cached, 0, start=t), cached, 0)
        self.assertEqual(m.summary()["max_sample_drop_bytes"], 30)
        rising = s.MemoryRecord([self.sample(0, 100, 0)])
        for t, cached in enumerate([50, 60, 90]):  # only rises after the first sample: the largest fall is negative
            rising.sample(self.sample(0, cached, 0, start=t), cached, 0)
        self.assertEqual(rising.summary()["max_sample_drop_bytes"], -10)


if __name__ == "__main__":
    unittest.main()
