"""ADR-063 Q4-TRACE synthetic controls (P1-P5): Qwen joint ordinary decode forward witnesses.

The fixtures follow the real record order of one owning-worker joint transaction: `transaction`, the
transaction `slots` snapshot, the joint `forward_input`, the first encoder's auxiliary scope, the Known
witness scope and `forward_witness`, the forward's native commands, `forward_input_end`, then the
per-lane settle scopes and commands of the same epoch. They are not hardware evidence.
"""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import f2_native_trace as native
import f2_trace as trace
import q4_trace_verdict as verdict
from test_f2_native_trace import four_lanes, record, with_busy_owners

JOINT = native.JOINT_SOURCE
ORIGIN = 1000
GEOMETRY = {"pipeline_identity": 70, "dispatch_kind": "thread_groups", "grid": [4, 1, 1],
            "threads_per_threadgroup": [32, 1, 1]}


class Joint:
    """Appends joint forwards to the four serial-lane fixture. Forward k owns ids 100 + 20 k ...:
    epoch E, two ids kept for pre-input auxiliary scopes, forward id E+3, first-encoder scope E+4,
    Known witness scope E+5, post-witness rotation scope E+6 (the watermark), settle scope E+7."""

    def __init__(self, pre_aux=False):
        self.records, self.identity, self.scenario = four_lanes()
        for policy in (self.records[0]["payload"]["native_observer"]["witness_manifest"],
                       self.scenario["forward_witness_manifest"]):
            policy["input_sources"] = ["qwen35.greedy.actual_tokens", JOINT]
        self.owners = record(self.records, "slots")["payload"]["slots"]
        self.cut = next(i for i, r in enumerate(self.records) if r["kind"] == "native_event"
                        and r["payload"]["kind"] == "CheckedDrain")
        self.body, self.tail = self.records[:self.cut], self.records[self.cut:]
        self.at, self.pre_aux, self.forwards = 1000, pre_aux, 0

    def add(self, kind, payload, advance=True):
        self.at += 10 if advance else 0
        self.body.append({"schema": trace.SCHEMA, "seq": 0, "kind": kind, "at_ns": self.at, "payload": payload})
        return self.body[-1]

    def event(self, command, epoch, kind, begin=None, **change):
        payload = {"clock_id": "CLOCK_MONOTONIC", "command_id": command, "invocation_epoch": epoch,
                   "queue_identity": 50, "device_registry_id": 60, "kind": kind,
                   "begin_monotonic_ns": 0, "end_monotonic_ns": 0, "dispatch_ordinal": 0,
                   "noncompute_operations": 0, "scope": None, "captured_scope": None, "capture_command_id": None,
                   "capture_generation": None, "replay_binding": "Direct", "tainted": False, "geometry": None,
                   "submitted": False, "checked_status": None, **change}
        item = self.add("native_event", payload)
        payload["begin_monotonic_ns"] = begin if begin is not None else ORIGIN + item["at_ns"]
        payload["end_monotonic_ns"] = payload["begin_monotonic_ns"] if begin is not None else ORIGIN + item["at_ns"]

    def command(self, command, epoch, scopes=(), work=True):
        """Created, one dispatch per scope id, then a checked commit; `work=False` is an empty command."""
        self.event(command, epoch, "Created")
        for ordinal, scope in enumerate(scopes, 1):
            self.event(command, epoch, "DispatchEncoded", dispatch_ordinal=ordinal,
                       scope={"epoch": epoch, "id": scope}, geometry=copy.deepcopy(GEOMETRY))
        done = {"dispatch_ordinal": len(scopes)}
        if work:
            self.event(command, epoch, "Submitted", submitted=True, **done)
            self.event(command, epoch, "CheckedDrain", submitted=True, checked_status=4, **done)
            self.event(command, epoch, "Dropped", submitted=True, checked_status=4, **done)
        else:
            self.event(command, epoch, "Dropped", **done)

    def scope(self, epoch, identity, rows=None):
        self.add("scope_definition", {"scope": {"epoch": epoch, "id": identity},
                 "operation": "target.input_rms_norm" if rows else "opaque_auxiliary",
                 "membership": "Known" if rows else "OpaqueAuxiliary", "rows": rows or []})

    def forward(self, lanes, witnessed=True, boundary=False, unmapped=0):
        k = self.forwards
        self.forwards += 1
        epoch = 100 + 20 * k
        forward_id, aux, known, rotation, settle = epoch + 3, epoch + 4, epoch + 5, epoch + 6, epoch + 7
        command = 1000 + 10 * k
        owners = [self.owners[slot] for slot in lanes]
        self.add("transaction", {"family": "qwen35", "operation": native.JOINT_OPERATION,
                 "logical_lanes": [{name: o[name] for name in ("request_id", "slot_id", "generation")}
                                   for o in owners],
                 "physical_membership": "Unknown",
                 "reason": "exact native graph submit/drain and token-row hook required"})
        busy = [dict(o, runnable_forward=False, runnable_emit=False, inflight=None,
                     blocked="OwningWorkerTransaction") for o in self.owners]
        self.add("slots", {"cause": "owning_worker_enter_transaction", "slots": busy})
        if self.pre_aux:  # same records and clock as the base fixture: only scopes, at the same instant
            for identity in (epoch + 1, epoch + 2):
                self.add("scope_definition", {"scope": {"epoch": epoch, "id": identity},
                         "operation": "opaque_auxiliary", "membership": "OpaqueAuxiliary", "rows": []},
                         advance=False)
        rows = [{"request_id": o["request_id"], "slot_id": o["slot_id"], "generation": o["generation"],
                 "token_position": 200 + 10 * k + o["slot_id"], "token_count": 1, "phase": "ordinary"}
                for o in owners]
        if not witnessed:
            self.scope(epoch, aux)
            for _ in range(unmapped):
                self.add("forward_witness_unmapped", {"operator": "target_input_rms_norm",
                         "reason": "NoActualInputManifest", "progress_eligible": False})
            self.command(command, epoch, (aux, aux))
            return
        self.add("forward_input", {"forward_id": forward_id, "epoch": epoch, "source": JOINT, "role": "target",
                 "matrix_rows": 4, "padding_rows": 4 - len(rows),
                 "packed_rows": [{"matrix_row_start": i, "matrix_row_end": i + 1, "logical": row}
                                 for i, row in enumerate(rows)],
                 "operator_site": "pre_attention_input", "execution_proven": False})
        self.scope(epoch, aux)
        self.scope(epoch, known, copy.deepcopy(rows))
        self.add("forward_witness", {"forward_id": forward_id, "scope": {"epoch": epoch, "id": known},
                 "source": JOINT, "operator": "target_input_rms_norm", "layer": None, "role": "target",
                 "rows": 4, "columns": 8, "dtype": "F32", "input_parent_identity": 4096,
                 "input_byte_offset": 0, "input_read_bytes": 128, "input_data_bytes": 128,
                 "input_parent_bytes": 128, "operator_site": "pre_attention_input", "dispatches_expected": 1,
                 "execution_proven": False, "full_body_batching_proven": False})
        self.scope(epoch, rotation)
        self.command(command, epoch, (aux, known, rotation))
        if boundary:
            self.command(command + 1, epoch, work=False)  # dispatch-free, created before the marker
        end = self.add("forward_input_end", {"forward_id": forward_id, "epoch": epoch,
                       "scope_watermark": rotation, "end_monotonic_ns": 0})
        end["payload"]["end_monotonic_ns"] = ORIGIN + end["at_ns"]
        if boundary:  # dispatch-free, created exactly at the marker's clock: ambiguous
            self.event(command + 2, epoch, "Created", begin=end["payload"]["end_monotonic_ns"])
            self.event(command + 2, epoch, "Dropped")
        self.scope(epoch, settle)
        self.command(command + 3, epoch, (settle, settle))
        if boundary:
            self.command(command + 4, epoch, work=False)  # dispatch-free, created after the marker

    def build(self):
        records = self.body + self.tail
        finalize(records)
        return records, self.identity, self.scenario


def finalize(records):
    for seq, item in enumerate(records):
        item["seq"] = seq
    final = record(records, "native_final")
    final["payload"]["events"] = sum(r["kind"] == "native_event" for r in records)
    final["at_ns"] = max(final["at_ns"], records[-3]["at_ns"])
    return records


def widths(pre_aux=False):
    joint = Joint(pre_aux)
    for index, lanes in enumerate(((0, 1, 2, 3), (0, 1, 2), (1, 3), (2,))):
        joint.forward(lanes, boundary=index == 0)
    return joint.build()


def full_width_only():
    joint = Joint(False)
    for index in range(2):
        joint.forward((0, 1, 2, 3), boundary=index == 0)
    return joint.build()


def joint_records(records, kind, forward=0):
    """Records of `kind` belonging to the `forward`-th joint transaction."""
    epoch = 100 + 20 * forward
    def owned(r):
        p = r["payload"]
        if kind == "forward_input":
            return p.get("source") == JOINT and p["epoch"] == epoch
        if kind in ("forward_input_end",):
            return p["epoch"] == epoch
        if kind in ("forward_witness", "scope_definition"):
            return p["scope"]["epoch"] == epoch
        if kind == "transaction":
            return r is transactions(records)[forward]
        raise AssertionError(kind)
    return [r for r in records if r["kind"] == kind and owned(r)]


def transactions(records):
    return [r for r in records if r["kind"] == "transaction"]


def report(records, identity, scenario):
    return trace.validate_trace(records, identity, scenario)


class JointWitnessP1(unittest.TestCase):
    def test_every_width_is_witnessed_bound_and_split_at_its_end_marker(self):
        result = report(*widths())
        joint = result[native.JOINT_OPERATION]
        self.assertEqual(result["unmapped_operator_attempts"], 0)
        self.assertEqual(result["unmapped_by_operation"], {native.JOINT_OPERATION: 0})
        self.assertEqual((joint["transactions"], joint["witnessed"], joint["unwitnessed"]), (4, 4, 0))
        self.assertEqual(joint["width_histogram"], {"logical4_pad0": 1, "logical3_pad1": 1,
                                                    "logical2_pad2": 1, "logical1_pad3": 1})
        self.assertEqual((joint["boundary_ambiguous_commands"], joint["pre_forward_commands"]), (1, 0))
        first, *others = joint["forwards"]
        self.assertEqual((first["epoch"], first["forward_id"], first["witness_command_id"]), (100, 103, 1000))
        self.assertEqual(first["lanes"], [[5, 0, 7, 200], [6, 1, 7, 201], [7, 2, 7, 202], [8, 3, 7, 203]])
        # forward: the witness command (3 dispatches) and the dispatch-free command before the marker;
        # settle: the settle command (2 dispatches) and the dispatch-free command after it.
        self.assertEqual((first["forward"]["commands"], first["forward"]["dispatches"]), (2, 3))
        self.assertEqual((first["settle"]["commands"], first["settle"]["dispatches"]), (2, 2))
        self.assertLess(first["forward"]["last_checked_return_ns"], first["settle"]["first_submit_ns"])
        self.assertEqual(first["epoch_span_ns"][1], first["settle"]["last_checked_return_ns"])
        for forward in others:
            self.assertEqual((forward["forward"]["commands"], forward["settle"]["commands"]), (1, 1))
        self.assertEqual([(f["logical_rows"], f["padding_rows"]) for f in joint["forwards"]],
                         [(4, 0), (3, 1), (2, 2), (1, 3)])
        # Joint witnesses never enter the Q2 v2 counting input.
        self.assertEqual([item["command_id"] for item in result["submissions"]], [1, 2, 3, 4])
        self.assertFalse(result["operator_shared_batch4_observed"])

    def test_histogram_keys_do_not_depend_on_observed_widths(self):
        joint = report(*full_width_only())[native.JOINT_OPERATION]
        self.assertEqual(joint["width_histogram"], {"logical4_pad0": 2, "logical3_pad1": 0,
                                                    "logical2_pad2": 0, "logical1_pad3": 0})

    def test_pre_input_auxiliary_scopes_reduce_identically(self):
        base, variant = report(*widths()), report(*widths(pre_aux=True))
        for result in (base, variant):
            del result["events"], result["bytes"]
        self.assertEqual(base, variant)

    def test_qualify_window_parser_binds_the_same_transactions(self):
        records, identity, scenario = widths()
        report(records, identity, scenario)
        result = trace.qualify_window(records, (0, 4), 0, population_policy="forward_demand_v2")
        self.assertEqual(result["status"], "Qualified")
        drop(records, joint_records(records, "transaction", 0))
        finalize(records)
        with self.assertRaisesRegex(ValueError, "not bound to its open joint transaction lanes"):
            trace.qualify_window(records, (0, 4), 0, population_policy="forward_demand_v2")


def fails(test, mutate, reason, build=widths):
    records, identity, scenario = build()
    mutate(records)
    finalize(records)
    with test.assertRaisesRegex(ValueError, reason):
        report(records, identity, scenario)


def drop(records, items):
    for item in items:
        records.remove(item)


def move_after(records, item, anchor):
    records.remove(item)
    records.insert(records.index(anchor) + 1, item)
    item["at_ns"] = anchor["at_ns"]


class JointWitnessP2(unittest.TestCase):
    def input(self, records, forward=0):
        return joint_records(records, "forward_input", forward)[0]["payload"]

    def test_padding_and_shape_rules(self):
        def serial_padding(records):
            record(records, "forward_input")["payload"]["padding_rows"] = 0
        fails(self, serial_padding, "actual source-bound forward input required")
        def wide_padding(records):
            self.input(records, 1).update(matrix_rows=5, padding_rows=2)
        fails(self, wide_padding, "joint physical rows disagree")
        def unpadded(records):
            self.input(records, 1)["padding_rows"] = 0
        fails(self, unpadded, "joint physical rows disagree")
        def temporal(records):
            p = self.input(records, 3)
            p["packed_rows"][0]["matrix_row_end"] = 2
            p["packed_rows"][0]["logical"]["token_count"] = 2
            p["padding_rows"] = 2
        fails(self, temporal, "temporal rows")

    def test_role_phase_and_lane_identity_rules(self):
        def verify(records):
            p = self.input(records, 1)
            p["role"] = "target_verify"
            for span in p["packed_rows"]:
                span["logical"]["phase"] = "verify"
        fails(self, verify, "actual verifier source/role required")
        def prefill(records):
            self.input(records, 1)["packed_rows"][1]["logical"]["phase"] = "prefill"
        fails(self, prefill, "ordinary one-token target decode only")
        def duplicate(records):
            rows = self.input(records, 2)["packed_rows"]
            rows[1]["logical"] = dict(rows[0]["logical"], token_position=500)
        fails(self, duplicate, "repeats one logical lane")
        def reordered(records):
            lanes = joint_records(records, "transaction", 1)[0]["payload"]["logical_lanes"]
            lanes[0], lanes[1] = lanes[1], lanes[0]
        fails(self, reordered, "not bound to its open joint transaction lanes")
        def other_lane(records):
            joint_records(records, "transaction", 2)[0]["payload"]["logical_lanes"][1] = dict(
                joint_records(records, "transaction", 2)[0]["payload"]["logical_lanes"][1], request_id=7, slot_id=2)
        fails(self, other_lane, "not bound to its open joint transaction lanes")

    def test_recycled_generation_after_the_transaction_is_not_its_lane(self):
        def recycled(records):
            p = self.input(records, 2)  # lanes (1, 3); slot 1 recycles to generation 8 after the transaction
            snapshot = copy.deepcopy(next(r for r in records if r["kind"] == "slots" and
                                          r["seq"] < joint_records(records, "forward_input", 2)[0]["seq"]
                                          and r["seq"] > joint_records(records, "transaction", 2)[0]["seq"]))
            snapshot["payload"]["slots"][1].update(generation=8, request_id=99)
            records.insert(records.index(joint_records(records, "forward_input", 2)[0]), snapshot)
            p["packed_rows"][0]["logical"].update(generation=8, request_id=99)
        fails(self, recycled, "not bound to its open joint transaction lanes")

    def test_one_input_per_open_joint_transaction(self):
        def second_input(records):
            drop(records, joint_records(records, "transaction", 1))
            drop(records, [r for r in records if r["kind"] == "slots" and r["payload"]["cause"] ==
                           "owning_worker_enter_transaction"][1:2])
        fails(self, second_input, "not bound to its open joint transaction lanes")
        def not_joint(records):
            joint_records(records, "transaction", 0)[0]["payload"]["operation"] = "ordinary_or_temporal_mtp_decode"
        fails(self, not_joint, "not bound to its open joint transaction lanes")

    def test_witness_binds_physical_rows(self):
        def logical(records):
            joint_records(records, "forward_witness", 1)[0]["payload"].update(
                rows=3, input_read_bytes=96, input_data_bytes=96, input_parent_bytes=96)
        fails(self, logical, "required witness replaced/stale/missing input")

    def test_epoch_freshness_and_consistency(self):
        def reused(records):
            self.input(records, 2)["epoch"] = 120
        fails(self, reused, "stale joint input epoch")
        def serial_epoch(records):
            self.input(records, 0)["epoch"] = 10
        fails(self, serial_epoch, "stale joint input epoch")
        def first_scope(records):
            joint_records(records, "scope_definition", 0)[0]["payload"]["scope"]["epoch"] = 999
        fails(self, first_scope, "joint input epoch differs from its first following scope")
        def end_before_scope(records):
            move_after(records, joint_records(records, "forward_input_end", 0)[0],
                       joint_records(records, "forward_input", 0)[0])
        fails(self, end_before_scope, "unconfirmed at forward_input_end")
        def transaction_before_scope(records):
            start = records.index(joint_records(records, "forward_input", 0)[0])
            stop = records.index(joint_records(records, "transaction", 1)[0])
            del records[start + 1:stop]
        fails(self, transaction_before_scope, "unconfirmed before the next transaction")

    def test_end_marker_is_present_unique_placed_matched_and_above_the_witness(self):
        def missing(records):
            drop(records, joint_records(records, "forward_input_end", 0))
        fails(self, missing, "without forward_input_end before the next transaction")
        def missing_last(records):
            drop(records, joint_records(records, "forward_input_end", 3))
        fails(self, missing_last, "without forward_input_end at the end of the trace")
        def duplicate(records):
            end = joint_records(records, "forward_input_end", 0)[0]
            records.insert(records.index(end) + 1, copy.deepcopy(end))
        fails(self, duplicate, "missing/duplicate/misplaced/mismatched")
        def misplaced(records):
            move_after(records, joint_records(records, "forward_input_end", 0)[0],
                       joint_records(records, "transaction", 1)[0])
        fails(self, misplaced, "without forward_input_end before the next transaction")
        for field, value in (("forward_id", 104), ("epoch", 101)):
            def mismatched(records, field=field, value=value):
                joint_records(records, "forward_input_end", 0)[0]["payload"][field] = value
            fails(self, mismatched, "missing/duplicate/misplaced/mismatched")
        def low_watermark(records):
            joint_records(records, "forward_input_end", 0)[0]["payload"]["scope_watermark"] = 104
        fails(self, low_watermark, "watermark below its witness scope")

    def test_end_marker_requires_the_joint_policy(self):
        def stray_marker(records):
            end = record(records, "forward_witness")
            records.insert(records.index(end) + 1, dict(copy.deepcopy(end), kind="forward_input_end", payload={
                "forward_id": 30, "epoch": 10, "scope_watermark": 20, "end_monotonic_ns": ORIGIN + end["at_ns"]}))
        fails(self, stray_marker, "requires the joint source policy", build=four_lanes)


class JointWitnessP3(unittest.TestCase):
    def test_unwitnessed_joint_transaction_reduces_but_fails_acceptance(self):
        joint = Joint()
        joint.forward((0, 1, 2, 3))
        joint.forward((0, 1), witnessed=False, unmapped=2)
        result = report(*joint.build())
        section = result[native.JOINT_OPERATION]
        self.assertEqual((section["transactions"], section["witnessed"], section["unwitnessed"]), (2, 1, 1))
        self.assertEqual(result["unmapped_by_operation"], {native.JOINT_OPERATION: 2})
        self.assertEqual(section["unmapped_attempts"], 2)
        outcome = verdict.acceptance(result)
        self.assertFalse(outcome["accepted"])
        self.assertEqual(set(outcome["failed"]), {"unmapped_operator_attempts == 0",
                                                  "unmapped_by_operation.joint_ordinary_decode == 0",
                                                  "joint.unwitnessed == 0",
                                                  "joint.witnessed == joint.transactions > 0"})
        self.assertTrue(verdict.acceptance({"validate_trace": report(*widths())})["accepted"])

    def test_trailing_unwitnessed_transaction_counts_at_finish(self):
        joint = Joint()
        joint.forward((0, 1, 2, 3))
        joint.forward((2, 3), witnessed=False)
        self.assertEqual(report(*joint.build())[native.JOINT_OPERATION]["unwitnessed"], 1)


class JointWitnessP4(unittest.TestCase):
    def test_old_policy_report_keys_are_unchanged_with_unmapped_joint_transactions(self):
        legacy = report(*four_lanes())
        joint = Joint()
        joint.forward((0, 1, 2, 3), witnessed=False, unmapped=48)
        records, identity, scenario = joint.build()
        for policy in (records[0]["payload"]["native_observer"]["witness_manifest"],
                       scenario["forward_witness_manifest"]):
            policy["input_sources"] = ["qwen35.greedy.actual_tokens"]
        result = report(records, identity, scenario)
        self.assertEqual(set(result), set(legacy))
        self.assertEqual(result["unmapped_operator_attempts"], 48)


class JointWitnessP5(unittest.TestCase):
    def cohort(self, serial_lanes=4):
        """Serial witnesses for the first `serial_lanes` owners, joint forwards over all four."""
        records, identity, scenario = widths()
        drop(records, [r for r in records if r["kind"] in ("forward_input", "forward_witness")
                       and r["payload"]["forward_id"] in range(30 + serial_lanes, 34) or
                       r["kind"] == "scope_definition" and r["payload"]["scope"]["id"] in range(20 + serial_lanes, 24)
                       or r["kind"] == "native_event" and r["payload"]["command_id"] in range(1 + serial_lanes, 5)])
        with_busy_owners(records, 330)
        finalize(records)
        report(records, identity, scenario)
        return records

    def windows(self, records):
        return {(composition, policy): trace.qualify_window(records, composition, 0, population_policy=policy)
                for composition in verdict.COMPOSITIONS for policy in verdict.POLICIES}

    def test_joint_witnesses_change_no_q2_v2_window(self):
        for serial_lanes in (4, 3):
            with self.subTest(serial_lanes=serial_lanes):
                self.isolated(self.cohort(serial_lanes))

    def isolated(self, records):
        hooked = self.windows(records)
        stripped = verdict.PreHook(records)
        self.assertEqual(stripped.epochs, {100, 120, 140, 160})
        plain_records = list(stripped)
        self.assertFalse(any(r["kind"] == "forward_input_end" or r["payload"].get("source") == JOINT
                             for r in plain_records))
        self.assertEqual(hooked, self.windows(plain_records))
        self.assertTrue(any(result["status"] == "Qualified" for result in hooked.values()) or
                        hooked[((0, 4), "forward_demand_v2")]["slots_without_required_forward_witness"])
        result = verdict.isolation(records, [(c, 0) for c in verdict.COMPOSITIONS])
        self.assertTrue(result["identical"])
        self.assertEqual(len(result["windows"]), 14)
        native_hooked = native.NativeTrace(records[0]["payload"], {"forward_witness_manifest":
            records[0]["payload"]["native_observer"]["witness_manifest"]})
        self.assertFalse(trace.validate_trace(records, records[0]["payload"]["identity"], {
            "schema": "hf2q.f2.trace-expectation.v1", "run_id": records[0]["payload"]["run_id"],
            "config_sha256": records[0]["payload"]["config_sha256"],
            "forward_witness_manifest": native_hooked.policy})["operator_shared_batch4_observed"])

    def test_the_isolation_check_sees_joint_witnesses_that_leak_into_submissions(self):
        # Slot 3 has no serial witness: only a leaked joint witness could credit it.
        records = self.cohort(serial_lanes=3)
        finish = native.NativeTrace.finish
        def leaky(self):
            joint, self.joint_forwards = self.joint_forwards, {}
            try:
                return finish(self)
            finally:
                self.joint_forwards = joint
        with mock.patch.object(native.NativeTrace, "finish", leaky):
            result = verdict.isolation(records, [(c, 0) for c in verdict.COMPOSITIONS])
            leaked = trace.validate_trace(records, records[0]["payload"]["identity"], {
                "schema": "hf2q.f2.trace-expectation.v1", "run_id": records[0]["payload"]["run_id"],
                "config_sha256": records[0]["payload"]["config_sha256"],
                "forward_witness_manifest": records[0]["payload"]["native_observer"]["witness_manifest"]})
        self.assertFalse(result["identical"])
        self.assertTrue(leaked["operator_shared_batch4_observed"])


class ExactnessRule(unittest.TestCase):
    @staticmethod
    def transcripts(exact):
        return {"schema": verdict.TRANSCRIPTS, "runs": 2, "classification":
                "ExactTranscripts" if exact else "DivergedTranscripts", "divergent": []}

    def test_rows_are_evaluated_top_to_bottom(self):
        x, d = self.transcripts(True), self.transcripts(False)
        cases = [((x, "fail", x, None), "HookAttributable"), ((d, "fail", d, None), "HookAttributable"),
                 ((x, "pass", d, None), "Exact"), ((d, "pass", d, None), "ObserverAttributable"),
                 ((d, "pass", x, None), "Unresolved"), ((d, "pass", x, x), "ScheduleNondeterministic"),
                 ((d, "pass", x, d), "HookAttributable")]
        for args, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(verdict.classify(*args)["classification"], expected)

    def test_repeat_only_for_unresolved_and_inputs_are_closed(self):
        x, d = self.transcripts(True), self.transcripts(False)
        with self.assertRaisesRegex(ValueError, "only for the Unresolved row"):
            verdict.classify(x, "pass", x, x)
        with self.assertRaisesRegex(ValueError, "two-run"):
            verdict.classify(dict(x, runs=3), "pass", x)
        with self.assertRaisesRegex(ValueError, "pass or fail"):
            verdict.classify(x, "skipped", x)


class VerdictCli(unittest.TestCase):
    def run_cli(self, *args):
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            code = verdict.main([str(a) for a in args])
        return code, out.getvalue()

    def test_acceptance_isolation_and_classify_from_files(self):
        records, identity, scenario = widths()
        reduced = report(records, identity, scenario)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            (root / "windows.json").write_text(json.dumps({
                "schema": "hf2q.q2.traced-windows.v1", "validate_trace": reduced,
                "windows": [{"composition": [0, 4], "arrival_ns": 0}]}))
            code, out = self.run_cli("acceptance", root / "windows.json")
            self.assertEqual((code, json.loads(out)["accepted"]), (0, True))
            code, out = self.run_cli("isolation", root / "trace.jsonl", "--windows", root / "windows.json")
            self.assertEqual((code, len(json.loads(out)["windows"])), (0, 2))
            code, out = self.run_cli("isolation", root / "trace.jsonl", "--windows", root / "windows.json",
                                     "--all-compositions")
            self.assertEqual((code, len(json.loads(out)["windows"])), (0, 14))
            for name, exact in (("exact", True), ("diverged", False)):
                (root / f"{name}.json").write_text(json.dumps(ExactnessRule.transcripts(exact)))
            code, out = self.run_cli("classify", "--t1-o1", root / "diverged.json", "--g1", "pass",
                                     "--t0-o0", root / "diverged.json")
            self.assertEqual((code, json.loads(out)["classification"]), (0, "ObserverAttributable"))
            reduced["unmapped_operator_attempts"] = 1
            (root / "bad.json").write_text(json.dumps(reduced))
            self.assertEqual(self.run_cli("acceptance", root / "bad.json")[0], 1)


if __name__ == "__main__":
    unittest.main()
