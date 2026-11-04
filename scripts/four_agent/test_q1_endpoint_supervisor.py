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
                                   "stage_records": 512, "stage_planned_payload": 16 * 512,
                                   "localizer_mode": "Cohort", "route": s.LOCALIZER_ROUTES["Cohort"], "observer_effect_reference": None,
                                   "stage_extent_source": "PacketExpectedStores", "stage_manifest": None, "routing_witness_installed": True},
                        logits=logits, stages=stages, records=576 + 512, plan_case=dict(MULTISTORE_CASE), manifest=None,
                        witness={"layers": 1, "metadata_limit_bytes": 1 << 20}, witness_rows=[(l, p) for l in range(4) for p in range(144)])
            if mutate:
                mutate(case)
            packet = case["packet"]
            (out / "packet.json").write_text(json.dumps(packet))
            case["execution"]["packet_sha256"] = s.sha(out / "packet.json")
            write_run(out, case["execution"], case["logits"])
            (out / "output/stages").mkdir()
            if case["witness_rows"] is not None:
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
            extent = s.multistore_extent(case["plan_case"], packet, None if case["manifest"] is None else json.dumps(case["manifest"]).encode())
            return s.validate_multistore(out, b, "loc4", plan, packet, extent, IDENTITY, case["policy"])

    @staticmethod
    def plan_only(c):  # T2LocExt plan-only packet: stage extent from the bound case manifest
        binding = {"path": "/x/stage-manifest.json", "sha256": "1" * 64}
        c["plan_case"]["manifest"] = binding; c["packet"]["stage_manifest"] = binding
        c["manifest"] = {k: c["packet"][k] for k in ("lanes", "host_payload_limit", "expected_stores")}
        del c["packet"]["expected_stores"]
        c["execution"].update(stage_extent_source="BoundStageManifest", stage_manifest=binding)

    @staticmethod
    def drift(c):  # DriftAll128: All128 prefill-only, witness off everywhere
        c["packet"].update(localizer_mode="DriftAll128", routing_witness=None)
        c["packet"]["stage_localizer"]["decode_stages"] = []
        c["execution"].update(localizer_mode="DriftAll128", route=s.LOCALIZER_ROUTES["DriftAll128"], routing_witness_installed=False)
        c["witness"] = None; c["witness_rows"] = None

    def test_plan_only_form_observer_reference_and_drift_mode(self):
        evidence = self.run_case(mutate=self.plan_only)["gemma_loc4"]
        self.assertEqual((evidence["stage_extent_source"], evidence["routing_bank_asserted"]), ("BoundStageManifest", True))
        def named(c):
            c["packet"]["observer_effect_reference"] = "gemma-t0-b5g"; c["execution"]["observer_effect_reference"] = "gemma-t0-b5g"
        self.assertEqual(self.run_case("T0ColdSerial", lambda c: (self.plan_only(c), named(c)))["gemma_loc4"]["observer_effect_reference"], "gemma-t0-b5g")
        for form in (lambda c: None, self.plan_only):
            evidence = self.run_case("T0ColdSerial", lambda c: (form(c), self.drift(c)))["gemma_loc4"]
            self.assertEqual((evidence["route"], evidence["routing_bank_asserted"], evidence["observer_effect_reference"]),
                             ("gemma-all128-stage-drift-witness-off-v1", False, None))

    def test_new_form_refusals(self):
        def other_manifest(c):
            self.plan_only(c); c["packet"]["stage_manifest"] = {"path": "/x/other.json", "sha256": "2" * 64}
        def both_forms(c):
            self.plan_only(c); c["packet"]["expected_stores"] = c["manifest"]["expected_stores"]
        def source_label(c):
            c["execution"]["stage_extent_source"] = "BoundStageManifest"
        def unechoed_reference(c):
            c["packet"]["observer_effect_reference"] = "gemma-t0-b5g"
        def cohort_with_drift_route(c):  # a witness-off label on a cohort run
            c["execution"]["route"] = s.LOCALIZER_ROUTES["DriftAll128"]
        def cohort_witness_not_installed(c):
            c["execution"]["routing_witness_installed"] = False
        def cohort_without_witness(c):  # only the DriftAll128 route may omit the bank
            c["packet"]["routing_witness"] = None; c["witness"] = None; c["witness_rows"] = None
        def drift_route_label(c):
            self.drift(c); c["execution"]["route"] = s.LOCALIZER_ROUTES["Cohort"]
        def drift_packet_witness(c):
            self.drift(c); c["packet"]["routing_witness"] = {"payload_bytes": 1, "metadata_bytes": 1, "cpu_ledger_bytes": 1}
        def drift_absent_packet_witness(c):
            self.drift(c); del c["packet"]["routing_witness"]
        def drift_bindings_witness(c):
            self.drift(c); c["witness"] = {"layers": 1, "metadata_limit_bytes": 1 << 20}
        def drift_bank_file(c):
            self.drift(c); c["witness_rows"] = [(l, p) for l in range(4) for p in range(144)]
        def drift_reference(c):
            self.drift(c); c["packet"]["observer_effect_reference"] = "gemma-t0-b5g"; c["execution"]["observer_effect_reference"] = "gemma-t0-b5g"
        def drift_decode(c):
            self.drift(c); c["packet"]["stage_localizer"]["decode_stages"] = ["Norm"]
        def drift_installed(c):
            self.drift(c); c["execution"]["routing_witness_installed"] = True
        def unknown_mode(c):
            c["packet"]["localizer_mode"] = "Other"; c["execution"]["localizer_mode"] = "Other"
        for trajectory, mutate in (("T2ActualCohort", other_manifest), ("T2ActualCohort", both_forms), ("T2ActualCohort", source_label),
                                   ("T2ActualCohort", unechoed_reference), ("T2ActualCohort", cohort_with_drift_route),
                                   ("T2ActualCohort", cohort_witness_not_installed), ("T0ColdSerial", cohort_without_witness),
                                   ("T0ColdSerial", drift_route_label), ("T0ColdSerial", drift_packet_witness), ("T0ColdSerial", drift_absent_packet_witness),
                                   ("T0ColdSerial", drift_bindings_witness), ("T0ColdSerial", drift_bank_file), ("T0ColdSerial", drift_reference),
                                   ("T0ColdSerial", drift_decode), ("T0ColdSerial", drift_installed), ("T2ActualCohort", self.drift),
                                   ("T2ActualCohort", unknown_mode)):
            with self.subTest(mutate.__name__, trajectory=trajectory), self.assertRaises((ValueError, KeyError, FileNotFoundError)):
                self.run_case(trajectory, mutate)

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


MANIFEST = {"path": "/x/manifest.json", "sha256": "0" * 64}
REFUSED = (ValueError, KeyError, FileNotFoundError, TypeError, IndexError)


def rec(agent, row=0, stage="CacheKey", layer=0, step=0, segment=0, nbytes=1, shape=None):
    return {"at": {"logical_step": step, "layer": layer, "stage": stage, "agent": agent, "logical_row_start": row, "segment": segment},
            "logical_bytes": nbytes, "shape": shape or [1]}


def filler(n, lanes=4, step=128):
    return [rec("agent-%d" % (i % lanes), 0, "CacheKey", i, step) for i in range(n)]


def logit_records(rows, lanes=range(4)):
    return [rec("agent-%d" % l, p, "Logits", s.LOGIT_LAYER) for l in lanes for p in rows]


def multistore_case(check, stores, marker, line=None, **extra):
    case = {"kind": "multistore", "out": "/x/case", "packet": {}, "filter": "fam::" + check, "packet_env": "HF2Q_Q1_FAM_PACKET",
            "completion_marker": marker, "marker_stream": "stdout", "stores": {n: n for n in stores}, "checks": [check], "manifest": dict(MANIFEST)}
    if line is not None:
        case["completion_line"] = line
    case.update(extra)
    return case


def run_multistore(root, c):
    """Write one multistore case in manifest form (the manifest = the captured stores) and validate it."""
    out = Path(root) / "case"; (out / "output").mkdir(parents=True)
    (out / "packet.json").write_text(json.dumps(c["packet"]))
    execution = {**c["execution"], "packet_sha256": s.sha(out / "packet.json"), "whole_q1_qualified": False}
    (out / "output/execution.json").write_text(json.dumps(execution))
    for name, captured in c["stores"].items():
        d = out / "output" / name; d.mkdir(parents=True, exist_ok=True)
        (d / "trace.json").write_text(json.dumps({"identity": IDENTITY, "acquisition": {"observed_model_opens": 3, "required_model_opens": 3},
                                                  "coverage": {"captured": captured, "expected": c.get("expected", {}).get(name, captured), "errors": []}}))
    for hook in c.get("files", ()):
        hook(out)
    records = sum(len(v) for v in c["stores"].values())  # absent `stream`: the Σ completion line of what was captured
    (out / "stdout").write_text("test result: ok.\n" + c.get("stream", "%s records=%d whole_q1_qualified=false\n" % (c["case"]["completion_marker"], records)))
    b = {**bindings({"t0": {"trajectory": "T0ColdSerial"}, "x": c["case"], **c.get("cases", {})}), "family": c["family"], **c.get("bindings", {})}
    plan = s.case_plan(b, "x")
    total = sum(r["logical_bytes"] for v in c["stores"].values() for r in v)
    manifest = {"lanes": c["lanes"], "host_payload_limit": c["packet"].get("host_payload_limit", total),
                "expected_stores": c.get("expected", c["stores"])}
    extent = s.multistore_extent(c["case"], c["packet"], json.dumps(manifest).encode())
    return s.validate_multistore(out, b, "x", plan, c["packet"], extent, IDENTITY, {"phase_payload_upper_bytes": 1 << 40})


class CheckRegistry(unittest.TestCase):
    def test_closed_registry_names(self):
        self.assertEqual(set(s.MULTISTORE_CHECKS), {"gemma_loc4", "qwen_t6_continuation", "qwen_t5_phase", "qwen_t5_first_fork",
                                                    "gemma_t5_ordinary", "gemma_t5_ngram", "gemma_t5_localizer"})

    def test_storeless_manifest_admits_only_a_zero_limit_with_no_stores(self):
        case = multistore_case("qwen_t5_phase", {}, "Q1_QWEN_T5_PHASE_COMPLETE")
        empty = {"lanes": [0, 1, 2, 3], "host_payload_limit": 0, "expected_stores": {}}
        self.assertEqual(s.multistore_extent(case, {"mode": "Preparation"}, json.dumps(empty).encode()), empty)
        for bad in ({**empty, "expected_stores": {"committed": []}}, {**empty, "host_payload_limit": -1}):
            with self.assertRaises(ValueError):
                s.multistore_extent(case, {}, json.dumps(bad).encode())


def mutated(test, build, positives, negatives):
    for args in positives:
        with test.subTest("positive", args=args), tempfile.TemporaryDirectory() as root:
            c = build(root, *args); evidence = run_multistore(root, c)
            test.assertEqual(evidence["captured_records"], sum(len(v) for v in c["stores"].values()))
    for args, mutate in negatives:
        with test.subTest(mutate.__name__, args=args), tempfile.TemporaryDirectory() as root:
            c = build(root, *args); mutate(c)
            with test.assertRaises(REFUSED):
                run_multistore(root, c)


class QwenT6Continuation(unittest.TestCase):
    @staticmethod
    def build(root, mode):
        t6 = mode == "CancelledThenRecovered"; snaps = ("before", "recovered") if t6 else ("prefix64",)
        stores = {"continuation": logit_records(range(64, 144)) + filler(1120)}
        for n in snaps:
            stores[n] = filler(560, step=64)
        line = "Q1_QWEN_RECOVERY_CONTINUATION_COMPLETE mode=%s lanes=4 rows_per_lane=80 first64_logits=false whole_q1=false" % mode
        calls = [c for l in range(4) for c in [{"lane": l, "before": 64, "input_rows": 64, "after": 128, "output_rows": 64}] +
                 [{"lane": l, "before": p, "input_rows": 1, "after": p + 1, "output_rows": 1} for p in range(128, 144)]]
        execution = {"schema": "hf2q.q1.qwen-recovery-continuation.v1", "mode": mode, "calls": calls, "captured_rows_per_lane": 80,
                     "cancelled_unpublished_rows_per_lane": 64 if t6 else 0, "first64_logits_captured": False, "endpoint_scope": [64, 144],
                     "active_cache_boundaries": [128, 144], "trace_records": 1440, "planned_tensor_payload": sum(len(v) for v in stores.values()),
                     "recovery": [{"recovered_cursor": 64, "scheduler_in_flight_after": 0, "client_closed_after_actual_advance": True}] * 4 if t6 else [],
                     "recovery_proof": {"disposition": "ExactEqual", "compared_bytes": 560} if t6 else None}
        if not t6:
            execution["prefix_preparation"] = [{"cancellation_performed": False, "retained_cursor": 64, "scheduler_in_flight_after": 0}] * 4
        return dict(case=multistore_case("qwen_t6_continuation", stores, "Q1_QWEN_RECOVERY_CONTINUATION_COMPLETE", line),
                    packet={"mode": mode}, execution=execution, stores=stores, stream=line + "\n", family="qwen36", lanes=[0, 1, 2, 3])

    def test_t6_and_t3cm(self):
        T6, T3 = ("CancelledThenRecovered",), ("ColdChunkMatched64",)
        def trace_records(c): c["execution"]["trace_records"] = 1439
        def short_continuation(c):  # complete store, receipt agrees, but not the source count
            c["stores"]["continuation"].pop(); c["execution"]["trace_records"] = 1439; c["execution"]["planned_tensor_payload"] -= 1
        def short_snapshot(c):
            name = "before" if "before" in c["stores"] else "prefix64"
            c["stores"][name].pop(); c["execution"]["planned_tensor_payload"] -= 1
        def planned(c): c["execution"]["planned_tensor_payload"] += 1
        def packet_mode(c): c["packet"]["mode"] = "ColdBaseline"
        def receipt_mode(c): c["execution"]["mode"] = "ColdBaseline"
        def schema(c): c["execution"]["schema"] = "hf2q.q1.other.v1"
        def calls(c): c["execution"]["calls"] = c["execution"]["calls"][1:] + c["execution"]["calls"][:1]
        def line_twice(c): c["stream"] *= 2
        def line_absent(c): c["stream"] = "noise\n"
        def line_not_verbatim(c):
            c["case"]["completion_line"] = c["case"]["completion_line"].replace("rows_per_lane=80", "rows_per_lane=81"); c["stream"] = c["case"]["completion_line"] + "\n"
        def missing_logit_row(c):
            c["stores"]["continuation"][0]["at"]["logical_row_start"] = 63
        def cancelled_rows(c): c["execution"]["cancelled_unpublished_rows_per_lane"] = 0 if c["packet"]["mode"] == T6[0] else 64
        def family(c): c["family"] = "qwen38"
        def stores_by_mode(c): c["packet"]["mode"] = c["execution"]["mode"] = T3[0] if c["packet"]["mode"] == T6[0] else T6[0]
        def proof(c): c["execution"]["recovery_proof"]["disposition"] = "DivergedExactInvariant"
        def proof_bytes(c): c["execution"]["recovery_proof"]["compared_bytes"] = 559
        def not_closed(c): c["execution"]["recovery"] = [{**c["execution"]["recovery"][0], "client_closed_after_actual_advance": False}] * 4
        def three_owners(c): c["execution"]["recovery"] = c["execution"]["recovery"][:3]
        def cancelled_prefix(c): c["execution"]["prefix_preparation"] = [{**c["execution"]["prefix_preparation"][0], "cancellation_performed": True}] * 4
        def prefix_cursor(c): c["execution"]["prefix_preparation"] = [{**c["execution"]["prefix_preparation"][0], "retained_cursor": 63}] * 4
        common = (trace_records, short_continuation, short_snapshot, planned, packet_mode, receipt_mode, schema, calls, line_twice, line_absent,
                  line_not_verbatim, missing_logit_row, cancelled_rows, family, stores_by_mode)
        mutated(self, self.build, (T6, T3), [(m, f) for m in (T6, T3) for f in common] +
                [(T6, f) for f in (proof, proof_bytes, not_closed, three_owners)] + [(T3, f) for f in (cancelled_prefix, prefix_cursor)])


class QwenT5Phase(unittest.TestCase):
    @staticmethod
    def build(root, mode):
        forced = mode == "ForcedProposal"; prep = mode == "Preparation"
        stores = {} if prep else {"committed": logit_records(range(128, 144)) + filler(1776, step=129)}
        if forced:
            stores["raw-verifier"] = filler(20, step=129)
        line = "Q1_QWEN_T5_PHASE_COMPLETE mode=%s logical_lanes=4 explicit_MTP_selector=true forced_proposal=%s whole_q1=false" % (mode, str(forced).lower())
        rows = [128] + list(range(131, 144)) if forced else list(range(128, 144))
        execution = {"schema": s.QWEN_T5_SCHEMA, "mode": mode, "error": None, "forced_proposals": forced,
                     "status": "CompleteWorkloadPreparationNotComparable" if prep else "CompleteScopedAcquisition",
                     "committed_records": None if prep else 1840, "raw_records": 20 if forced else None,
                     "committed_payload_plan": 0 if prep else 1840, "raw_payload_plan": 20,
                     "actual_transitions": [{"lane": l, "matched": 1, "rejected": 1} for l in range(4)] if forced else [],
                     "ordinary_input_positions": [rows] * 4, "canonical_decisions": [[{"token": 5}] * 17] * 4}
        def prep_files(out):
            frozen = out / "output/frozen-workload.json"; frozen.write_text("{}")
            (out / "output/preparation-only.json").write_text(json.dumps({"frozen_file_sha256": s.sha(frozen), "comparable_acquisition": False,
                                                                          "requires_independent_review": True}))
        return dict(case=multistore_case("qwen_t5_phase", stores, "Q1_QWEN_T5_PHASE_COMPLETE", line), packet={"mode": mode},
                    execution=execution, stores=stores, stream=line + "\n", family="qwen36", lanes=[0, 1, 2, 3], files=[prep_files] if prep else [])

    def test_preparation_baseline_and_forced(self):
        P, O, F = ("Preparation",), ("OrdinaryBaseline",), ("ForcedProposal",)
        def status(c): c["execution"]["status"] = "FailedOrIncomplete"
        def error(c): c["execution"]["error"] = "refused"
        def receipt_mode(c): c["execution"]["mode"] = "OrdinaryBaseline" if c["packet"]["mode"] != "OrdinaryBaseline" else "ForcedProposal"
        def first_fork_packet(c): c["packet"]["first_fork"] = True
        def first_fork_receipt(c): c["execution"]["first_fork"] = {}
        def line_twice(c): c["stream"] *= 2
        def line_forced_flag(c):
            c["case"]["completion_line"] = c["case"]["completion_line"].replace("forced_proposal=", "forced_proposal=x"); c["stream"] = c["case"]["completion_line"] + "\n"
        def input_rows(c): c["execution"]["ordinary_input_positions"][0] = list(range(129, 144))
        def decisions(c): c["execution"]["canonical_decisions"] = c["execution"]["canonical_decisions"][:3]
        def committed_records(c): c["execution"]["committed_records"] = 1839
        def source_count(c):  # complete store, receipt agrees, but not the source-manifest count
            c["stores"]["committed"].pop(); c["execution"]["committed_records"] = 1839; c["execution"]["committed_payload_plan"] = 1839
        def committed_bytes(c): c["execution"]["committed_payload_plan"] = 1841
        def raw_records(c): c["execution"]["raw_records"] = 19
        def raw_on_baseline(c): c["execution"]["raw_records"] = 20
        def transitions(c): c["execution"]["actual_transitions"] = c["execution"]["actual_transitions"][:3]
        def baseline_transition(c): c["execution"]["actual_transitions"] = [{"lane": 0}]
        def no_raw_store(c): del c["stores"]["raw-verifier"]; c["case"]["stores"] = {"committed": "committed"}
        def prep_frozen_missing(c): c["files"] = []
        def prep_frozen_sha(c):
            c["files"].append(lambda out: (out / "output/frozen-workload.json").write_text("{} "))
        def prep_records(c): c["execution"]["committed_records"] = 0
        def prep_comparable(c):
            c["files"].append(lambda out: (out / "output/preparation-only.json").write_text(json.dumps({"frozen_file_sha256": s.sha(out / "output/frozen-workload.json"),
                                                                                                     "comparable_acquisition": True, "requires_independent_review": True})))
        common = (status, error, receipt_mode, first_fork_packet, first_fork_receipt, line_twice, line_forced_flag, input_rows, decisions)
        mutated(self, self.build, (P, O, F), [(m, f) for m in (P, O, F) for f in common] +
                [(m, f) for m in (O, F) for f in (committed_records, source_count, committed_bytes)] +
                [(F, f) for f in (raw_records, transitions, no_raw_store)] + [(O, f) for f in (raw_on_baseline, baseline_transition)] +
                [(P, f) for f in (prep_frozen_missing, prep_frozen_sha, prep_records, prep_comparable)])


class QwenT5FirstFork(unittest.TestCase):
    @staticmethod
    def build(root, mode):
        forced = mode == "ForcedProposal"
        stores = {"committed": logit_records(range(128, 131), [0]) + filler(10, 1, 129)}
        if forced:
            stores["raw-verifier"] = filler(4, 1, 129)
        captures = ["target-before-mtp", "target-after-mtp"] + (["post-draft-discard129"] if forced else [])
        for n in captures:
            stores[n] = filler(6 if n != "post-draft-discard129" else 148, 1, 129)
        line = "Q1_QWEN_T5_FIRST_FORK_COMPLETE mode=%s logical_lanes=1 agent=0 cursor=131 float_policy=Unspecified whole_q1=false" % mode
        execution = {"schema": s.QWEN_T5_SCHEMA, "mode": mode, "error": None, "forced_proposals": forced, "status": "CompleteFirstForkDiagnosticNotQ1Qualified",
                     "committed_records": 13, "raw_records": 4 if forced else None,
                     "first_fork": {"logical_lanes_executed": 1, "logical_lane": 0, "physical_slots": 1, "all4lane_execution": False,
                                    "capture_stores": {n: {"records": len(stores[n]), "errors": [], "target_and_mtp_separate_by_layer": True} for n in captures},
                                    "case_payload_plan": sum(len(v) for v in stores.values()),
                                    "reviewed_inputs": {"preparation_sha256": "a" * 64, "review_sha256": "b" * 64, "review_schema": s.QWEN_FROZEN_REVIEW_SCHEMA}}}
        inputs = {"preparation": {"path": "/x/preparation-only.json", "sha256": "a" * 64}, "review": {"path": "/x/review.json", "sha256": "b" * 64}}
        return dict(case=multistore_case("qwen_t5_first_fork", stores, "Q1_QWEN_T5_FIRST_FORK_COMPLETE", line),
                    packet={"mode": mode, "first_fork": True, "first_fork_inputs": inputs},
                    execution=execution, stores=stores, stream=line + "\n", family="qwen36", lanes=[0])

    def test_baseline_and_forced_arms(self):
        O, F = ("OrdinaryBaseline",), ("ForcedProposal",)
        def capture_count(c): c["execution"]["first_fork"]["capture_stores"]["target-after-mtp"]["records"] = 5
        def capture_error(c): c["execution"]["first_fork"]["capture_stores"]["target-before-mtp"]["errors"] = ["x"]
        def committed_records(c): c["execution"]["committed_records"] = 12
        def payload_plan(c): c["execution"]["first_fork"]["case_payload_plan"] += 1
        def phase_status(c): c["execution"]["status"] = "CompleteScopedAcquisition"
        def not_first_fork_packet(c): del c["packet"]["first_fork"]
        def four_lanes(c): c["lanes"] = [0, 1, 2, 3]
        def other_agent(c):
            c["stores"]["target-before-mtp"][0]["at"]["agent"] = "agent-1"; c["lanes"] = [0, 1]
        def lanes_executed(c): c["execution"]["first_fork"]["logical_lanes_executed"] = 4
        def other_review(c): c["packet"]["first_fork_inputs"]["review"]["sha256"] = "c" * 64
        def no_inputs(c): del c["packet"]["first_fork_inputs"]
        def line_twice(c): c["stream"] *= 2
        def preparation(c): c["packet"]["mode"] = c["execution"]["mode"] = "Preparation"
        def raw_count(c): c["execution"]["raw_records"] = 3
        def no_draft(c):
            del c["stores"]["post-draft-discard129"]; del c["case"]["stores"]["post-draft-discard129"]
            del c["execution"]["first_fork"]["capture_stores"]["post-draft-discard129"]; c["execution"]["first_fork"]["case_payload_plan"] -= 148
        common = (capture_count, capture_error, committed_records, payload_plan, phase_status, not_first_fork_packet, four_lanes, other_agent,
                  lanes_executed, other_review, no_inputs, line_twice, preparation)
        mutated(self, self.build, (O, F), [(m, f) for m in (O, F) for f in common] + [(F, f) for f in (raw_count, no_draft)])


def gemma_workload(root):
    lanes = [[2000 + 1000 * l + p for p in range(144)] for l in range(4)]
    path = Path(root) / "workload.json"; path.write_text(json.dumps({"family": "Gemma4", "vocab_size": 262144, "lanes": lanes}))
    return lanes, {"path": str(path), "sha256": s.sha(path)}


def gemma_decisions(lanes, arm, accepted, initial=None):
    result = []
    for l, w in enumerate(lanes):
        first = w[128] if initial is None else initial[l]
        d = {"lane": l, "initial_argmax": first, "frozen_128": w[128], "initial_equals_frozen": first == w[128], "transaction": None}
        start = 128
        if arm == "NgramTransaction":
            a = accepted[l]; injected = [w[129], w[130], w[131] + 100000]; v = [injected[i] if i < a else 7 + i for i in range(4)]
            d["transaction"] = {"proposer_ran": True, "proposed": [5], "injected": injected, "injected_origin": s.GEMMA_INJECTED_ORIGIN,
                                "verifier_argmax": v, "accepted": a, "rejected": 3 - a, "fallback": v[a], "rollback_cursor": 129 + a}
            start = 129 + a
        d["decode_inputs"] = [{"position": p, "token": w[p]} for p in range(start, 144)]
        d["decode_argmax"] = [{"position": p, "token": 9} for p in range(start, 144)]
        result.append(d)
    return result


def gemma_witness_keys(arm, accepted):
    if arm == "OrdinaryGlobal":
        return [("Canonical", l, p, 0) for l in range(4) for p in range(144)]
    return [(ns, l, p, 0) for l in range(4) for ns, rows in (("T5Initial", range(128)), ("T5Verify", range(132)),
                                                              ("T5Continuation", range(129 + accepted[l], 144))) for p in rows]


def write_witness(out, keys, mode, header=None, end=None, bits=0):
    rows = [{"kind": "routing_row", "native_boundary_valid": True, "selected_copy_valid": True, "native_decode_error": None, "qualified": False,
             "scores_f32_bits": [bits], "selected_weights_f32_bits": [0x3f800000],
             "at": {"namespace": ns, "row": {"lane": l, "position": p}, "layer": k}} for ns, l, p, k in keys]
    lines = [{"kind": "header", "identity": IDENTITY, "mode": mode, "qualified": False, **(header or {})}] + rows + \
            [{"kind": "end", "records": len(rows), "all_scored_rows_captured": True, "full_Q1_qualified": False, **(end or {})}]
    (out / "output/routing-witness.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")


ACCEPTED = [2, 1, 0, 2]


class GemmaT5(unittest.TestCase):
    def setUp(self):
        self.layers = s.GEMMA_ROUTING_LAYERS; s.GEMMA_ROUTING_LAYERS = 1  # one routing layer keeps the fixture small

    def tearDown(self):
        s.GEMMA_ROUTING_LAYERS = self.layers

    @staticmethod
    def arm_execution(lanes, arm, schema=s.GEMMA_T5_PHASE_SCHEMA, initial=None):
        names = s.GEMMA_T5_STORES[arm]
        return {"schema": schema, "arm": arm, "status": "CompleteScopedAcquisition", "error": None, "lanes": [0, 1, 2, 3],
                "decisions": gemma_decisions(lanes, arm, ACCEPTED, initial), "class_b_refusal": None,
                "stores": {n: s.GEMMA_T5_RECORDS[n] for n in names}, "routing_witness": "routing-witness.jsonl", "whole_q1_qualified": False}

    @classmethod
    def build(cls, root, check, arm, branch=None):
        lanes, workload = gemma_workload(root)
        stores = {"prefork": logit_records([127]) + filler(4200, step=128), "committed": logit_records(range(128, 144)) + filler(4200, step=144)}
        if arm == "NgramTransaction":
            stores["raw-verifier"] = [rec("agent-%d" % l, 0, "SpecVerifierLogits", s.LOGIT_LAYER, 128, shape=[4, 262144]) for l in range(4)]
        schema = s.GEMMA_T5_LOCALIZER_SCHEMA if check == "gemma_t5_localizer" else s.GEMMA_T5_PHASE_SCHEMA
        execution = cls.arm_execution(lanes, arm, schema)
        packet = {"arm": arm, "lanes": [0, 1, 2, 3], "host_payload_limit": 1 << 30, "greedy_workload": workload,
                  "routing_witness": {"payload_bytes": 1, "metadata_bytes": 1, "cpu_ledger_bytes": 1}}
        c = dict(packet=packet, execution=execution, stores=stores, family="gemma", lanes=[0, 1, 2, 3], cases={},
                 bindings={"routing_witness": {"layers": 1, "metadata_limit_bytes": 1 << 30}}, witness=gemma_witness_keys(arm, ACCEPTED),
                 witness_mode="Ordinary" if arm == "OrdinaryGlobal" else "Transaction", witness_header={}, witness_end={}, witness_bits=0)
        if check == "gemma_t5_ngram":
            ordinary = cls.arm_execution(lanes, "OrdinaryGlobal")
            sibling = Path(root) / "ord"; (sibling / "output").mkdir(parents=True)
            (sibling / "output/execution.json").write_text(json.dumps(ordinary))
            (sibling / "receipt.json").write_text(json.dumps({"case_complete": True, "helper_evidence": {"execution": ordinary}}))
            c["cases"] = {"t5ord": multistore_case("gemma_t5_ordinary", ("prefork", "committed"), "Q1_GEMMA_T5_COMPLETE", out=str(sibling))}
            c["sibling"] = sibling
        if check == "gemma_t5_localizer":
            plan = {"rerun_id": "t5loc", "layers": [0], "prefill_stages": ["Key"], "decode_stages": ["Norm"], "host_payload_limit": 1 << 20,
                    "accepted": None if branch == "P" else ACCEPTED}
            if arm == "OrdinaryGlobal":
                stages = [rec("agent-%d" % l, r, "Key") for l in range(4) for r in range(128)] + [rec("agent-%d" % l, p, "Norm") for l in range(4) for p in range(128, 144)]
            else:
                stages = [rec("agent-%d" % l, r, "Key", segment=1) for l in range(4) for r in range(132)] + \
                         [rec("agent-%d" % l, p, "Norm") for l in range(4) for p in range(129 + ACCEPTED[l], 144)]
            stores["stages"] = stages
            ngram = cls.arm_execution(lanes, "NgramTransaction") if branch == "N" else \
                {"schema": s.GEMMA_T5_PHASE_SCHEMA, "arm": "NgramTransaction", "status": "FailedOrIncomplete", "error": "refused", "decisions": [],
                 "class_b_refusal": {"row": 127, "reason": s.GEMMA_ROW127_REFUSAL, "lane": 1}}
            receipts = {}
            for name, value in (("ordinary", cls.arm_execution(lanes, "OrdinaryGlobal")), ("ngram", ngram)):
                path = Path(root) / (name + "-t5.json"); path.write_text(json.dumps(value)); receipts[name] = {"path": str(path), "sha256": s.sha(path)}
            packet.update(stage_localizer=plan, branch=branch, **{"pass": 1}, t5_receipts=receipts, manifest=dict(MANIFEST))
            execution.update(branch=branch, **{"pass": 1}, plan=plan, stage_records=len(stages))
        c["case"] = multistore_case(check, stores, "Q1_GEMMA_T5_LOCALIZER_COMPLETE" if check == "gemma_t5_localizer" else "Q1_GEMMA_T5_COMPLETE")
        c["files"] = [lambda out: write_witness(out, c["witness"], c["witness_mode"], c["witness_header"], c["witness_end"], c["witness_bits"])]
        c["marker"] = c["case"]["completion_marker"]
        return c

    # Shared arm negatives (every Gemma T5 check).
    @staticmethod
    def common():
        def status(c): c["execution"]["status"] = "FailedOrIncomplete"
        def error(c): c["execution"]["error"] = "x"
        def schema(c): c["execution"]["schema"] = "hf2q.q1.other.v1"
        def receipt_arm(c): c["execution"]["arm"] = "Other"
        def receipt_count(c): c["execution"]["stores"]["prefork"] = 4203
        def source_count(c):  # complete store and receipt agree, but not the source manifest count
            c["stores"]["committed"].pop(); c["execution"]["stores"]["committed"] = 4263
        def marker_count(c): c["stream"] = "%s records=1 whole_q1_qualified=false\n" % c["marker"]
        def refusal(c): c["execution"]["class_b_refusal"] = {"row": 127}
        def decode_input(c): c["execution"]["decisions"][1]["decode_inputs"][0]["token"] += 1
        def decode_argmax(c): c["execution"]["decisions"][2]["decode_argmax"].pop()
        def frozen128(c): c["execution"]["decisions"][0]["frozen_128"] += 1
        def equals_lie(c): c["execution"]["decisions"][3]["initial_equals_frozen"] = False
        def lane_order(c): c["execution"]["decisions"].reverse()
        def logit_row(c): c["stores"]["prefork"][0]["at"]["logical_row_start"] = 126
        def workload_sha(c): c["packet"]["greedy_workload"]["sha256"] = "0" * 64
        def missing_row(c): c["witness"] = c["witness"][:-1]
        def extra_namespace(c): c["witness"] = c["witness"] + [("UnpublishedCancelled", 0, 64, 0)]
        def header_mode(c): c["witness_mode"] = "Cancelled"
        def header_identity(c): c["witness_header"] = {"identity": {"family": "Other"}}
        def qualified(c): c["witness_header"] = {"qualified": True}
        def end_records(c): c["witness_end"] = {"records": 1}
        def not_all_scored(c): c["witness_end"] = {"all_scored_rows_captured": False}
        def nonfinite(c): c["witness_bits"] = 0x7f800000
        def quota(c): c["bindings"]["routing_witness"]["metadata_limit_bytes"] = 1000
        def thirty_layers(c): c["bindings"]["routing_witness"]["layers"] = 30
        def no_witness(c): c["bindings"] = {}
        def packet_witness_off(c): c["packet"]["routing_witness"] = None
        def packet_lanes(c): c["packet"]["lanes"] = [0, 1, 2]
        return (status, error, schema, receipt_arm, receipt_count, source_count, marker_count, refusal, decode_input, decode_argmax, frozen128,
                equals_lie, lane_order, logit_row, workload_sha, missing_row, extra_namespace, header_mode, header_identity, qualified, end_records,
                not_all_scored, nonfinite, quota, thirty_layers, no_witness, packet_witness_off, packet_lanes)

    def test_ordinary_arm(self):
        O = ("gemma_t5_ordinary", "OrdinaryGlobal")
        with tempfile.TemporaryDirectory() as root:  # a cross-route initial difference is recorded, never refused
            c = self.build(root, *O); w = gemma_workload(root)[0]
            c["execution"]["decisions"] = gemma_decisions(w, "OrdinaryGlobal", ACCEPTED, [w[l][128] + (l == 2) for l in range(4)])
            evidence = run_multistore(root, c)["gemma_t5_ordinary"]
            self.assertEqual((evidence["initial_equals_frozen"], evidence["stores"]), ([True, True, False, True], {"prefork": 4204, "committed": 4264}))
        def ngram_arm(c): c["packet"]["arm"] = "NgramTransaction"
        def transaction(c): c["execution"]["decisions"][0]["transaction"] = {}
        def ngram_namespaces(c): c["witness"] = gemma_witness_keys("NgramTransaction", ACCEPTED)
        mutated(self, self.build, (O,), [(O, f) for f in self.common() + (ngram_arm, transaction, ngram_namespaces)])

    def test_ngram_arm(self):
        N = ("gemma_t5_ngram", "NgramTransaction")
        with tempfile.TemporaryDirectory() as root:
            evidence = run_multistore(root, self.build(root, *N))["gemma_t5_ngram"]
            self.assertEqual((evidence["accepted"], evidence["ordinary_case"]), (ACCEPTED, "t5ord"))
        def t(c, lane=0): return c["execution"]["decisions"][lane]["transaction"]
        def accepted(c): t(c)["accepted"] = 1; t(c)["rejected"] = 2; t(c)["fallback"] = t(c)["verifier_argmax"][1]; t(c)["rollback_cursor"] = 130
        def fallback(c): t(c)["fallback"] += 1
        def cursor(c): t(c)["rollback_cursor"] = 132
        def rejected(c): t(c)["rejected"] = 0
        def all_accepted(c):  # verifier agrees with mut131: a = 3 is outside {0,1,2}
            x = t(c); x["verifier_argmax"][:3] = x["injected"]; x.update(accepted=3, rejected=0, fallback=x["verifier_argmax"][3], rollback_cursor=132)
        def unmutated(c): t(c)["injected"][2] = c["execution"]["decisions"][0]["decode_inputs"][0]["token"]  # lane 0 a=2: first input is f131
        def eos(c): t(c, 2)["injected"][2] = 106
        def origin(c): t(c)["injected_origin"] = "ProductionProposal"
        def proposer(c): t(c)["proposer_ran"] = False
        def initial_not_frozen(c): c["execution"]["decisions"][1]["initial_equals_frozen"] = False
        def ordinary_initial(c):
            ordinary = json.loads((c["sibling"] / "output/execution.json").read_text()); ordinary["decisions"][0]["initial_argmax"] += 1
            (c["sibling"] / "output/execution.json").write_text(json.dumps(ordinary))
            (c["sibling"] / "receipt.json").write_text(json.dumps({"case_complete": True, "helper_evidence": {"execution": ordinary}}))
        def sibling_incomplete(c): (c["sibling"] / "receipt.json").write_text(json.dumps({"case_complete": False, "helper_evidence": {}}))
        def sibling_unbound(c): (c["sibling"] / "output/execution.json").write_text(json.dumps({"schema": "other"}))
        def no_sibling(c): c["cases"] = {}
        def raw_shape(c): c["stores"]["raw-verifier"][0]["shape"] = [3, 262144]
        def raw_step(c): c["stores"]["raw-verifier"][1]["at"]["logical_step"] = 129
        def continuation_rows(c): c["witness"] = [k for k in c["witness"] if not (k[0] == "T5Continuation" and k[1] == 0 and k[2] == 131)]
        def verify_rows(c): c["witness"] = [k for k in c["witness"] if not (k[0] == "T5Verify" and k[2] == 131)]
        def ordinary_mode(c): c["witness_mode"] = "Ordinary"
        mutated(self, self.build, (N,), [(N, f) for f in self.common() + (accepted, fallback, cursor, rejected, all_accepted, unmutated, eos, origin,
                                                                        proposer, initial_not_frozen, ordinary_initial, sibling_incomplete,
                                                                        sibling_unbound, no_sibling, raw_shape, raw_step, continuation_rows,
                                                                        verify_rows, ordinary_mode)])

    def test_localizer_arms_and_branches(self):
        ON, NN, OP = (("gemma_t5_localizer", "OrdinaryGlobal", "N"), ("gemma_t5_localizer", "NgramTransaction", "N"),
                      ("gemma_t5_localizer", "OrdinaryGlobal", "P"))
        with tempfile.TemporaryDirectory() as root:
            evidence = run_multistore(root, self.build(root, *OP))["gemma_t5_localizer"]
            self.assertEqual((evidence["branch"], evidence["ngram_class_b_refusal"]["row"], evidence["stage_records"]), ("P", 127, 576))
        def plan_echo(c): c["execution"]["plan"] = {**c["execution"]["plan"], "layers": [1]}
        def branch_echo(c): c["execution"]["branch"] = "P" if c["packet"]["branch"] == "N" else "N"
        def pass_echo(c): c["execution"]["pass"] = 2
        def bad_pass(c): c["packet"]["pass"] = c["execution"]["pass"] = 3
        def bad_branch(c): c["packet"]["branch"] = c["execution"]["branch"] = "X"
        def stage_records(c): c["execution"]["stage_records"] += 1
        def packet_manifest(c): c["packet"]["manifest"] = {"path": "/x/other.json", "sha256": "0" * 64}
        def receipt_sha(c): c["packet"]["t5_receipts"]["ordinary"]["sha256"] = "0" * 64
        def one_receipt(c): del c["packet"]["t5_receipts"]["ngram"]
        def stage_layer(c): c["stores"]["stages"][0]["at"]["layer"] = 5
        def stage_role(c): c["stores"]["stages"][0]["at"]["stage"] = "Norm"
        def stage_step(c): c["stores"]["stages"][0]["at"]["logical_step"] = 128
        def stage_segment(c): c["stores"]["stages"][0]["at"]["segment"] = 1 - c["stores"]["stages"][0]["at"]["segment"]
        def missing_stage_row(c):
            c["stores"]["stages"].pop(); c["execution"]["stage_records"] -= 1
        def stage_bytes(c): c["packet"]["stage_localizer"]["host_payload_limit"] = c["execution"]["plan"]["host_payload_limit"] = 1
        def accepted_mismatch(c): c["packet"]["stage_localizer"]["accepted"] = c["execution"]["plan"]["accepted"] = [1, 1, 0, 2]
        def accepted_null(c): c["packet"]["stage_localizer"]["accepted"] = c["execution"]["plan"]["accepted"] = None
        def accepted_present(c): c["packet"]["stage_localizer"]["accepted"] = c["execution"]["plan"]["accepted"] = ACCEPTED
        def no_row127(c):
            path = Path(c["packet"]["t5_receipts"]["ngram"]["path"]); v = json.loads(path.read_text()); v["class_b_refusal"]["row"] = 128
            path.write_text(json.dumps(v)); c["packet"]["t5_receipts"]["ngram"]["sha256"] = s.sha(path)
        def phase_schema(c): c["execution"]["schema"] = s.GEMMA_T5_PHASE_SCHEMA
        common = self.common() + (plan_echo, branch_echo, pass_echo, bad_pass, bad_branch, stage_records, packet_manifest, receipt_sha, one_receipt,
                                  stage_layer, stage_role, stage_step, stage_segment, missing_stage_row, stage_bytes, phase_schema)
        def ngram_branch_p(c):  # branch P runs the ordinary-global arm only
            c["packet"]["branch"] = c["execution"]["branch"] = "P"
        negatives = [(m, f) for m in (ON, NN, OP) for f in common] + [(m, f) for m in (ON, NN) for f in (accepted_mismatch, accepted_null, ngram_branch_p)] + \
                    [(OP, f) for f in (accepted_present, no_row127)]
        mutated(self, self.build, (ON, NN, OP), negatives)


if __name__ == "__main__":
    unittest.main()
